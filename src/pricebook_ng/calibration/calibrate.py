"""Sequential bootstrap over heterogeneous calibration instruments (L3).

`calibrate(spec) → (DiscountingModel, CalibrationResult)`: the model is the OUTPUT (doc 22
Q2). The spec describes a curve SET (doc 22 Q4) built in DEPENDENCY ORDER — discount (OIS)
first, self-discounting; then the projection (IBOR) curve discounted on it. Each curve is
bootstrapped pillar by pillar: an ordered run of `CalibrationInstrument`s, each Brent-solved
so its `residual` is zero. Dispatch on quote type is a registry lookup (no isinstance); each
instrument composes the SAME L1 building blocks the L4 engine composes (§3d) — the calibrator
never imports L4. Still one SEQUENTIAL orchestration (no method/numerics/Jacobian, rule of two).

The `residual(discount, projection)` convention: `projection` is always the curve being
solved (the trial); `discount` is the discounting curve — the trial itself in the discount
build (self-discounting), the already-built discount curve in the projection build.

Provenance:
  quarry: python/pricebook/curves/bootstrap.py; curves/ncurve_solver.py (InstrumentPricer pattern)
  source: redesign/22 (calibrate contract); §3d (shared atoms); doc 18 §1-§3 (pillars, placement)
  oracle: every calibrating instrument reprices to par; EURIBOR swap → zero dual-curve
  slice:  cash-instruments (T1 slice 3)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Protocol

from pricebook_ng.foundation import (
    Accrual,
    Currency,
    DayCountConvention,
    Frequency,
    Interpolation,
    RateIndex,
    RollRule,
    ScheduleTerms,
    Tenor,
    TimeMeasure,
    brent,
    build_schedule,
    root_nd,
)
from pricebook_ng.market.building_blocks import deposit_df, float_leg_pv, forward, rpv01
from pricebook_ng.market.curve import CurveHandle, DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.quotes import DepositQuote, FRAQuote, FutureQuote, ParSwapQuote
from pricebook_ng.market.snapshot import MarketSnapshot, ScalarKey
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.cash import FRA, Deposit, Future
from pricebook_ng.products.swap import FixedLeg, FloatLeg, VanillaSwap, XccyBasisSwap

__all__ = [
    "CalibrationSpec",
    "CurrencyCurves",
    "CurveBuild",
    "XccyBasisQuote",
    "XccyBuild",
    "CalibrationResult",
    "CalibrationMethod",
    "SolveConfig",
    "Jacobian",
    "CalibrationFailure",
    "ParSwapQuote",
    "calibrate",
    "par_swap",
    "xccy_swaps",
    "product",
    "curve_set_residuals",
]

Quote = ParSwapQuote | DepositQuote | FRAQuote | FutureQuote


class CalibrationMethod(Enum):
    """The two orchestrations (doc 22 Q4). Same signature, same residual definition; only
    the solve differs — sequential pillar-by-pillar, or the whole `CurveSet` at once."""

    SEQUENTIAL = "sequential"
    SIMULTANEOUS = "simultaneous"


@dataclass(frozen=True)
class SolveConfig:
    """The reproducibility knobs for the solve (invariant 5): the `method` and the solver
    tolerance / iteration cap. Bundled so `CalibrationSpec` stays ≤5 fields (§3b). The knobs
    are shared across both methods (Brent xtol / LM xtol·ftol), so no per-family sub-config
    is needed yet."""

    method: CalibrationMethod = CalibrationMethod.SEQUENTIAL
    tolerance: float = 1e-12
    max_iterations: int = 1000


@dataclass(frozen=True)
class Jacobian:
    """The par→zero Jacobian an artifact of the SIMULTANEOUS solve: `matrix[i][j] =
    ∂residualᵢ/∂dfⱼ`, with `instruments` (row) and `pillars` (col) labels. Risk transforms
    (par delta, key-rate) are C3, not here."""

    matrix: tuple[tuple[float, ...], ...]
    instruments: tuple[str, ...]
    pillars: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationFailure:
    """A calibration that did not converge — carried as a value, never raised (invariant 4)."""

    reason: str


@dataclass(frozen=True)
class CurveBuild:
    """How to build one curve: the `index` it projects (and is keyed by), both legs'
    `frequency` and `day_count`, its calibrating `quotes` (any mix of deposit/FRA/future/
    swap), and the `interpolation`."""

    index: RateIndex
    frequency: Frequency
    day_count: DayCountConvention
    quotes: tuple[Quote, ...]
    interpolation: Interpolation = Interpolation.LOG_LINEAR


@dataclass(frozen=True)
class XccyBasisQuote:
    """A cross-currency basis spread quote: the `spread` (on the foreign leg) at `tenor`.
    The pair is carried by the `XccyBuild` that owns it (a new asset adds keys, not fields)."""

    tenor: Tenor
    spread: float


@dataclass(frozen=True)
class XccyBuild:
    """How to build the foreign-collateral (xccy-basis) curve: the foreign `currency`, the
    `collateral` currency it is discounted under, and the basis `quotes`. The foreign leg's
    conventions (index, frequency, day count) are reused from that currency's OIS build in
    the same spec — no duplication (doc 18 §1)."""

    currency: Currency
    collateral: Currency
    quotes: tuple[XccyBasisQuote, ...]


@dataclass(frozen=True)
class CurrencyCurves:
    """One currency's curve builds: the `discount` (OIS) build and the `projection` (IBOR)
    build. A `CalibrationSpec` holds one per currency (doc 22 Q4 — the spec is data)."""

    currency: Currency
    discount: CurveBuild
    projection: CurveBuild


@dataclass(frozen=True)
class CalibrationSpec:
    """The curve set to build, all data (doc 22 Q4): one `CurrencyCurves` per currency
    (single-currency = a 1-tuple, the degenerate case), the `solve` config, an optional
    `xccy` basis build, and the `fx` spots the snapshot carries. The unified multi-currency
    front — ONE `calibrate`, no per-topic entry point (§1). Convenience `.currency`/`.discount`/
    `.projection` accessors resolve the single-currency case (slices 1–5) and raise on multi."""

    valuation_date: date
    curves: tuple[CurrencyCurves, ...]
    solve: SolveConfig = field(default_factory=SolveConfig)
    xccy: XccyBuild | None = None
    fx: Mapping[ScalarKey, float] = field(default_factory=dict)

    @classmethod
    def single_currency(
        cls,
        *,
        valuation_date: date,
        currency: Currency,
        discount: CurveBuild,
        projection: CurveBuild,
        solve: SolveConfig | None = None,
    ) -> CalibrationSpec:
        """The degenerate single-currency spec (slices 1–5): one `CurrencyCurves` element."""
        return cls(
            valuation_date,
            (CurrencyCurves(currency, discount, projection),),
            solve or SolveConfig(),
        )

    def _single(self) -> CurrencyCurves:
        if len(self.curves) != 1:
            raise ValueError("multi-currency spec: use .curves, not .currency/.discount/.projection")
        return self.curves[0]

    @property
    def currency(self) -> Currency:
        return self._single().currency

    @property
    def discount(self) -> CurveBuild:
        return self._single().discount

    @property
    def projection(self) -> CurveBuild:
        return self._single().projection


@dataclass(frozen=True)
class CalibrationResult:
    """Sits BESIDE the model, never on it (F1 §5): the per-instrument `residuals` across
    both curves, whether the solve `converged`, and the `jacobian` (the SIMULTANEOUS solve
    populates it; SEQUENTIAL leaves it `None` — its post-hoc Jacobian arrives with the C3
    risk consumer, doc 22 Q4)."""

    residuals: tuple[float, ...]
    converged: bool
    jacobian: Jacobian | None = None


# ── calibration instruments — each composes the L1 atoms into a par residual ────────────
class CalibrationInstrument(Protocol):
    """A calibrating instrument that reprices against the curves to a `residual` (0 at
    solution) and knows which `pillar_date` it fixes."""

    @property
    def pillar_date(self) -> date: ...

    def residual(self, discount: CurveHandle, projection: CurveHandle) -> float: ...


@dataclass(frozen=True)
class SwapInstrument:
    """A par swap: `par_rate(discount, projection) − quoted rate`."""

    swap: VanillaSwap
    rate: float

    @property
    def pillar_date(self) -> date:
        return self.swap.fixed_leg.schedule.periods[-1].payment_date

    def residual(self, discount: CurveHandle, projection: CurveHandle) -> float:
        return _par_rate(self.swap, discount, projection) - self.rate


@dataclass(frozen=True)
class DepositInstrument:
    """A deposit pins the curve being solved (`projection` is the trial): `df(end) =
    df(start)·deposit_df`. Composes the `deposit_df` atom (§3d)."""

    deposit: Deposit

    @property
    def pillar_date(self) -> date:
        return self.deposit.accrual.end

    def residual(self, discount: CurveHandle, projection: CurveHandle) -> float:
        a = self.deposit.accrual
        return projection.df(a.end) - projection.df(a.start) * deposit_df(self.deposit.rate, a)


@dataclass(frozen=True)
class FRAInstrument:
    """An FRA pins the projection forward over its accrual to the quoted rate. Composes
    the `forward` atom (§3d); its `df(start)` must be pinned by an earlier pillar."""

    fra: FRA

    @property
    def pillar_date(self) -> date:
        return self.fra.accrual.end

    def residual(self, discount: CurveHandle, projection: CurveHandle) -> float:
        return forward(projection, self.fra.accrual) - self.fra.rate


@dataclass(frozen=True)
class FutureInstrument:
    """A future pins the projection forward over its IMM accrual to `1 − price` (the forward
    approximation; convexity deferred to the models topic). Composes the `forward` atom."""

    future: Future

    @property
    def pillar_date(self) -> date:
        return self.future.accrual.end

    def residual(self, discount: CurveHandle, projection: CurveHandle) -> float:
        return forward(projection, self.future.accrual) - (1.0 - self.future.price)


@dataclass(frozen=True)
class XccyBasisInstrument:
    """A par cross-currency basis swap. Solves the foreign-collateral curve (the trial =
    `projection` per the convention); `discount` here carries the foreign OIS curve that
    PROJECTS the foreign floating. Composes the SAME `float_leg_pv`/`rpv01` atoms the engine
    composes (§3d): the foreign leg discounted on the trial, projected on the foreign OIS, plus
    the basis annuity, plus the foreign notional exchange — zero at par. The domestic leg and
    FX spot cancel (see `XccyBasisSwap`), so they do not enter the residual."""

    swap: XccyBasisSwap

    @property
    def pillar_date(self) -> date:
        return self.swap.foreign_leg.schedule.periods[-1].payment_date

    def residual(self, discount: CurveHandle, projection: CurveHandle) -> float:
        leg = self.swap.foreign_leg
        trial, foreign_ois = projection, discount
        floating = float_leg_pv(leg.schedule, leg.day_count, trial, foreign_ois)
        annuity = rpv01(leg.schedule, leg.day_count, trial)
        return floating + self.swap.basis * annuity + trial.df(self.pillar_date) - 1.0


def par_swap(
    spec: CalibrationSpec, build: CurveBuild, tenor: Tenor, rate: float, notional: float = 1.0
) -> VanillaSwap:
    """Build the par swap the bootstrap calibrates to (and the engine reprices) — ONE
    construction, so the calibrator and the pricing oracle share the exact schedule and
    index (§3d argument identity). Its float leg fixes on `build.index`."""
    terms = ScheduleTerms(frequency=build.frequency, roll=RollRule(calendar=None))
    schedule = build_schedule(spec.valuation_date, spec.valuation_date + tenor, terms)
    return VanillaSwap(
        notional=notional,
        currency=spec.currency,
        fixed_leg=FixedLeg(schedule, build.day_count, rate),
        float_leg=FloatLeg(schedule, build.day_count, build.index),
    )


def _par_rate(swap: VanillaSwap, discount: CurveHandle, projection: CurveHandle) -> float:
    return float_leg_pv(
        swap.float_leg.schedule, swap.float_leg.day_count, discount, projection
    ) / rpv01(swap.fixed_leg.schedule, swap.fixed_leg.day_count, discount)


def _currency_curves(spec: CalibrationSpec, currency: Currency) -> CurrencyCurves:
    for cc in spec.curves:
        if cc.currency == currency:
            return cc
    raise KeyError(f"no curve build for {currency.code} in the spec")


def _xccy_swap(
    spec: CalibrationSpec, ois: CurveBuild, collateral: Currency, quote: XccyBasisQuote
) -> XccyBasisSwap:
    """Build the xccy basis swap the bootstrap calibrates to (and the engine reprices) — ONE
    construction, shared so calibrator and pricer use the same schedule/index (§3d). The
    foreign leg reuses the foreign currency's OIS conventions (`ois`)."""
    terms = ScheduleTerms(frequency=ois.frequency, roll=RollRule(calendar=None))
    schedule = build_schedule(spec.valuation_date, spec.valuation_date + quote.tenor, terms)
    return XccyBasisSwap(1.0, FloatLeg(schedule, ois.day_count, ois.index), collateral, quote.spread)


def xccy_swaps(spec: CalibrationSpec) -> tuple[XccyBasisSwap, ...]:
    """The calibrating xccy basis swaps — the SAME products the L4 engine reprices to zero (§3d).
    Empty when the spec has no xccy build."""
    if spec.xccy is None:
        return ()
    ois = _currency_curves(spec, spec.xccy.currency).discount
    return tuple(_xccy_swap(spec, ois, spec.xccy.collateral, q) for q in spec.xccy.quotes)


# ── quote → product (L2), and quote → calibration instrument (L3), both by type ─────────
def _deposit(spec: CalibrationSpec, build: CurveBuild, q: DepositQuote) -> Deposit:
    accrual = Accrual(spec.valuation_date, spec.valuation_date + q.tenor, build.day_count)
    return Deposit(1.0, spec.currency, accrual, q.rate)


def _fra(spec: CalibrationSpec, build: CurveBuild, q: FRAQuote) -> FRA:
    start = spec.valuation_date + q.start if q.start is not None else spec.valuation_date
    accrual = Accrual(start, spec.valuation_date + q.end, build.day_count)
    return FRA(1.0, spec.currency, accrual, build.index, q.rate)


def _future(spec: CalibrationSpec, build: CurveBuild, q: FutureQuote) -> Future:
    accrual = Accrual(q.imm_start, q.imm_start + build.index.id.tenor, build.day_count)
    return Future(1.0, spec.currency, accrual, build.index, q.price)


_PRODUCT_FACTORY: dict[type, Callable[[CalibrationSpec, CurveBuild, Any], object]] = {
    ParSwapQuote: lambda spec, build, q: par_swap(spec, build, q.tenor, q.rate),
    DepositQuote: _deposit,
    FRAQuote: _fra,
    FutureQuote: _future,
}


def product(spec: CalibrationSpec, build: CurveBuild, quote: Quote) -> object:
    """The tradeable L2 product a quote calibrates to — the SAME product the engine reprices
    (the calibration instrument wraps it), so calibrator and engine share it (§3d)."""
    return _PRODUCT_FACTORY[type(quote)](spec, build, quote)


# Quote type → the calibration instrument it becomes (structural dispatch, no isinstance).
_INSTRUMENT_FACTORY: dict[type, Callable[[CalibrationSpec, CurveBuild, Any], CalibrationInstrument]] = {
    ParSwapQuote: lambda spec, build, q: SwapInstrument(par_swap(spec, build, q.tenor, q.rate), q.rate),
    DepositQuote: lambda spec, build, q: DepositInstrument(_deposit(spec, build, q)),
    FRAQuote: lambda spec, build, q: FRAInstrument(_fra(spec, build, q)),
    FutureQuote: lambda spec, build, q: FutureInstrument(_future(spec, build, q)),
}


def _instrument(spec: CalibrationSpec, build: CurveBuild, quote: object) -> CalibrationInstrument:
    return _INSTRUMENT_FACTORY[type(quote)](spec, build, quote)


def _instruments(spec: CalibrationSpec, build: CurveBuild) -> list[CalibrationInstrument]:
    """The build's calibration instruments, ordered by pillar date — the order both
    orchestrations solve in and the Jacobian labels follow."""
    return sorted(
        (_instrument(spec, build, q) for q in build.quotes), key=lambda i: i.pillar_date
    )


def _curve_set(spec: CalibrationSpec, discount: DiscountCurve, projection: DiscountCurve) -> CurveSet:
    """Assemble the keyed `CurveSet` (the OIS/discount index projects off the discount curve
    — the degenerate case; the IBOR index off the projection curve)."""
    return CurveSet(
        {
            CurveKey(CurveRole.DISCOUNT, spec.currency): discount,
            CurveKey(CurveRole.PROJECTION, spec.discount.index): discount,
            CurveKey(CurveRole.PROJECTION, spec.projection.index): projection,
        }
    )


def curve_set_residuals(spec: CalibrationSpec, curves: CurveSet) -> tuple[float, ...]:
    """Every calibrating instrument's residual against `curves` — the vector both
    orchestrations drive to zero (discount build first, then projection; each build's
    instruments ordered by pillar date, matching the Jacobian rows). ≈0 at a solved set."""
    discount = curves.discount(spec.currency)
    return tuple(
        inst.residual(discount, curves.projection(build.index))
        for build in (spec.discount, spec.projection)
        for inst in _instruments(spec, build)
    )


_DF_BRACKET_CAP = 1e6  # a pillar DF above this is unphysical — treat as unbracketable


def _solve_pillar_df(residual: Callable[[float], float]) -> float:
    """Solve one pillar's discount factor, bracketing ROBUSTLY first. A DF is positive but
    EXCEEDS 1 for a negative rate (`deposit_df(-0.5%, 1y) = 1.005`), so a fixed `[1e-6, 1.0]`
    bracket bakes in a positive-rates assumption; instead expand the upper bound until the
    residual changes sign (capped). Raises `ValueError` if no bracket exists — the caller turns
    that into a `CalibrationFailure` (invariant 4)."""
    lo, hi = 1e-9, 1.0
    f_lo, f_hi = residual(lo), residual(hi)
    while f_lo * f_hi > 0.0 and hi < _DF_BRACKET_CAP:
        hi *= 2.0
        f_hi = residual(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(f"could not bracket a positive discount factor below {_DF_BRACKET_CAP}")
    return brent(residual, lo, hi)


def _bootstrap(
    spec: CalibrationSpec, build: CurveBuild, discount: CurveHandle | None
) -> DiscountCurve | CalibrationFailure:
    """Sequential per-pillar bootstrap over the build's instruments. `discount is None` ⇒
    self-discounting (the discount build); otherwise the instruments discount on the
    supplied (already-built) discount curve. A pillar that cannot be bracketed/solved is
    returned as a `CalibrationFailure`, never raised (invariant 4)."""
    time_measure = TimeMeasure(spec.valuation_date, build.day_count)
    times: list[float] = [0.0]
    dfs: list[float] = [1.0]
    for inst in _instruments(spec, build):
        t_k = time_measure.year_fraction(inst.pillar_date)

        def residual(df_k: float, inst: CalibrationInstrument = inst, t_k: float = t_k) -> float:
            trial = DiscountCurve(time_measure, (*times, t_k), (*dfs, df_k), build.interpolation)
            return inst.residual(trial if discount is None else discount, trial)

        try:
            df_k = _solve_pillar_df(residual)
        except (ValueError, FloatingPointError, ZeroDivisionError) as exc:
            return CalibrationFailure(f"pillar {inst.pillar_date.isoformat()}: {exc}")
        times.append(t_k)
        dfs.append(df_k)
    return DiscountCurve(time_measure, tuple(times), tuple(dfs), build.interpolation)


def _bootstrap_xccy(
    spec: CalibrationSpec, curves: Mapping[CurveKey, CurveHandle]
) -> tuple[DiscountCurve, tuple[float, ...]] | CalibrationFailure:
    """Sequential post-bootstrap of the foreign-collateral (xccy-basis) curve, on top of the
    already-built per-currency curves (dependency order: domestic OIS → foreign OIS → foreign-in-
    collateral LAST, doc 18 §1). Each pillar's DF is Brent-solved so its xccy basis swap reprices
    to zero, projecting the foreign floating on the foreign OIS. Folding these pillars into the
    SIMULTANEOUS global state is deferred (2nd xccy consumer) — so sequential==simultaneous does
    NOT cover xccy."""
    xccy = spec.xccy
    assert xccy is not None  # guarded by the caller
    ois = _currency_curves(spec, xccy.currency).discount
    foreign_ois = curves[CurveKey(CurveRole.DISCOUNT, xccy.currency)]
    time_measure = TimeMeasure(spec.valuation_date, ois.day_count)
    instruments = sorted(
        (XccyBasisInstrument(_xccy_swap(spec, ois, xccy.collateral, q)) for q in xccy.quotes),
        key=lambda i: i.pillar_date,
    )
    times: list[float] = [0.0]
    dfs: list[float] = [1.0]
    for inst in instruments:
        t_k = time_measure.year_fraction(inst.pillar_date)

        def residual(df_k: float, inst: XccyBasisInstrument = inst, t_k: float = t_k) -> float:
            trial = DiscountCurve(time_measure, (*times, t_k), (*dfs, df_k), ois.interpolation)
            return inst.residual(foreign_ois, trial)

        try:
            df_k = _solve_pillar_df(residual)
        except (ValueError, FloatingPointError, ZeroDivisionError) as exc:
            return CalibrationFailure(f"xccy pillar {inst.pillar_date.isoformat()}: {exc}")
        times.append(t_k)
        dfs.append(df_k)
    curve = DiscountCurve(time_measure, tuple(times), tuple(dfs), ois.interpolation)
    residuals = tuple(inst.residual(foreign_ois, curve) for inst in instruments)
    return curve, residuals


def _calibrate_sequential(
    spec: CalibrationSpec,
) -> tuple[DiscountingModel, CalibrationResult] | CalibrationFailure:
    """Dependency-ordered bootstrap: discount (self-discounting) then projection. A pillar that
    cannot be solved propagates as a `CalibrationFailure` (invariant 4). No Jacobian (its
    post-hoc form arrives with the C3 risk consumer, doc 22 Q4)."""
    discount = _bootstrap(spec, spec.discount, discount=None)
    if isinstance(discount, CalibrationFailure):
        return discount
    projection = _bootstrap(spec, spec.projection, discount=discount)
    if isinstance(projection, CalibrationFailure):
        return projection
    curves = _curve_set(spec, discount, projection)
    model = DiscountingModel(MarketSnapshot(spec.valuation_date, curves))
    residuals = curve_set_residuals(spec, curves)
    return model, CalibrationResult(
        residuals, converged=all(abs(r) < 1e-10 for r in residuals), jacobian=None
    )


def _calibrate_simultaneous(
    spec: CalibrationSpec,
) -> tuple[DiscountingModel, CalibrationResult] | CalibrationFailure:
    """Global N-D solve: ONE flat state vector of all pillar DFs across BOTH curves; the
    residual vector is every calibrating instrument (discount instruments self-discount;
    projection instruments discount on the discount curve being solved and project on the
    projection curve being solved), solved JOINTLY via the scipy LM adapter (§7bb — no
    hand-rolled Newton, no scipy in this module). Seed: flat 3% DFs (the `ncurve_solver`
    default). The Jacobian is the solve's artifact; non-convergence is a value (invariant 4)."""
    tm_d = TimeMeasure(spec.valuation_date, spec.discount.day_count)
    tm_p = TimeMeasure(spec.valuation_date, spec.projection.day_count)
    d_insts = _instruments(spec, spec.discount)
    p_insts = _instruments(spec, spec.projection)
    d_times = [tm_d.year_fraction(i.pillar_date) for i in d_insts]
    p_times = [tm_p.year_fraction(i.pillar_date) for i in p_insts]
    n_d = len(d_insts)

    def _curves(x: Sequence[float]) -> tuple[DiscountCurve, DiscountCurve]:
        disc = DiscountCurve(
            tm_d, (0.0, *d_times), (1.0, *(float(v) for v in x[:n_d])), spec.discount.interpolation
        )
        proj = DiscountCurve(
            tm_p, (0.0, *p_times), (1.0, *(float(v) for v in x[n_d:])), spec.projection.interpolation
        )
        return disc, proj

    def _residual(x: Sequence[float]) -> list[float]:
        disc, proj = _curves(x)
        return [i.residual(disc, disc) for i in d_insts] + [i.residual(disc, proj) for i in p_insts]

    x0 = [math.exp(-0.03 * t) for t in (*d_times, *p_times)]
    solution, jac, converged = root_nd(
        _residual, x0, spec.solve.tolerance, spec.solve.max_iterations
    )
    if not converged:
        return CalibrationFailure(
            f"global curve solve did not converge "
            f"(tolerance={spec.solve.tolerance}, max_iterations={spec.solve.max_iterations})"
        )
    disc, proj = _curves(solution)
    curves = _curve_set(spec, disc, proj)
    model = DiscountingModel(MarketSnapshot(spec.valuation_date, curves))
    rows = tuple(f"discount@{i.pillar_date.isoformat()}" for i in d_insts) + tuple(
        f"projection@{i.pillar_date.isoformat()}" for i in p_insts
    )
    cols = tuple(f"discount@{t:.6f}" for t in d_times) + tuple(f"projection@{t:.6f}" for t in p_times)
    jacobian = Jacobian(tuple(tuple(row) for row in jac), rows, cols)
    return model, CalibrationResult(curve_set_residuals(spec, curves), converged=True, jacobian=jacobian)


def _sub(spec: CalibrationSpec, cc: CurrencyCurves) -> CalibrationSpec:
    """A single-currency view of one currency's builds — lets the per-currency solve reuse
    the existing (single-currency) machinery unchanged through the convenience accessors."""
    return CalibrationSpec(spec.valuation_date, (cc,), spec.solve)


def _solve_currency(
    sub: CalibrationSpec,
) -> tuple[DiscountingModel, CalibrationResult] | CalibrationFailure:
    """One currency's curves via the chosen orchestration (doc 22 Q4) — both compose the SAME
    `CalibrationInstrument` residuals; only the solve differs."""
    if sub.solve.method is CalibrationMethod.SIMULTANEOUS:
        return _calibrate_simultaneous(sub)
    return _calibrate_sequential(sub)


def calibrate(
    spec: CalibrationSpec,
) -> tuple[DiscountingModel, CalibrationResult] | CalibrationFailure:
    """Calibrate the whole curve SET (doc 22 Q4): each currency's curves in turn (independent
    OIS+projection systems), all keys assembled into ONE `CurveSet` carried by ONE model. The
    single-currency 1-tuple is the degenerate case — identical numbers to slices 1–5. Any
    currency's failure to converge is returned as a value (invariant 4)."""
    merged: dict[CurveKey, CurveHandle] = {}
    residuals: list[float] = []
    converged = True
    jacobian: Jacobian | None = None
    for cc in spec.curves:
        out = _solve_currency(_sub(spec, cc))
        if isinstance(out, CalibrationFailure):
            return out
        model, result = out
        merged.update(model.market.curves.curves)
        residuals.extend(result.residuals)
        converged = converged and result.converged
        if len(spec.curves) == 1:  # preserve the single-currency Jacobian (SIMULTANEOUS)
            jacobian = result.jacobian
    if spec.xccy is not None:  # foreign-collateral curve, always a sequential post-bootstrap
        xccy_out = _bootstrap_xccy(spec, merged)
        if isinstance(xccy_out, CalibrationFailure):
            return xccy_out
        xccy_curve, xccy_residuals = xccy_out
        merged[CurveKey(CurveRole.DISCOUNT, spec.xccy.currency, spec.xccy.collateral)] = xccy_curve
        residuals.extend(xccy_residuals)
        converged = converged and all(abs(r) < 1e-10 for r in xccy_residuals)
    snapshot = MarketSnapshot(spec.valuation_date, CurveSet(merged), scalars=dict(spec.fx))
    return DiscountingModel(snapshot), CalibrationResult(
        tuple(residuals), converged=converged, jacobian=jacobian
    )
