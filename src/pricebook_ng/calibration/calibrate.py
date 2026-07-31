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

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from pricebook_ng.foundation import (
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
)
from pricebook_ng.market.building_blocks import float_leg_pv, rpv01
from pricebook_ng.market.curve import CurveHandle, DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.swap import FixedLeg, FloatLeg, VanillaSwap

__all__ = [
    "CalibrationSpec",
    "CurveBuild",
    "CalibrationResult",
    "ParSwapQuote",
    "calibrate",
    "par_swap",
]


@dataclass(frozen=True)
class ParSwapQuote:
    """A par-swap calibrating quote: the swap `tenor` and its market par `rate`. (Moves to
    L1 `market/quotes.py` with the other quote types in Step B.)"""

    tenor: Tenor
    rate: float


@dataclass(frozen=True)
class CurveBuild:
    """How to build one curve: the `index` it projects (and is keyed by), both legs'
    `frequency` and `day_count`, its calibrating `quotes`, and the `interpolation`."""

    index: RateIndex
    frequency: Frequency
    day_count: DayCountConvention
    quotes: tuple[ParSwapQuote, ...]
    interpolation: Interpolation = Interpolation.LOG_LINEAR


@dataclass(frozen=True)
class CalibrationSpec:
    """The curve set to build, all data (doc 22 Q4): the `discount` (OIS) build and the
    `projection` (IBOR) build, on one `valuation_date`/`currency`."""

    valuation_date: date
    currency: Currency
    discount: CurveBuild
    projection: CurveBuild


@dataclass(frozen=True)
class CalibrationResult:
    """Sits BESIDE the model, never on it (F1 §5): the per-instrument `residuals` across
    both curves and whether the solve `converged`."""

    residuals: tuple[float, ...]
    converged: bool


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


# Quote type → the calibration instrument it becomes (structural dispatch, no isinstance).
_INSTRUMENT_FACTORY: dict[type, Callable[[CalibrationSpec, CurveBuild, Any], CalibrationInstrument]] = {
    ParSwapQuote: lambda spec, build, q: SwapInstrument(
        par_swap(spec, build, q.tenor, q.rate), q.rate
    ),
}


def _instrument(spec: CalibrationSpec, build: CurveBuild, quote: object) -> CalibrationInstrument:
    return _INSTRUMENT_FACTORY[type(quote)](spec, build, quote)


def _bootstrap(
    spec: CalibrationSpec, build: CurveBuild, discount: CurveHandle | None
) -> DiscountCurve:
    """Sequential per-pillar bootstrap over the build's instruments (ordered by pillar
    date). `discount is None` ⇒ self-discounting (the discount build); otherwise the
    instruments discount on the supplied (already-built) discount curve."""
    time_measure = TimeMeasure(spec.valuation_date, build.day_count)
    instruments = sorted(
        (_instrument(spec, build, q) for q in build.quotes), key=lambda i: i.pillar_date
    )
    times: list[float] = [0.0]
    dfs: list[float] = [1.0]
    for inst in instruments:
        t_k = time_measure.year_fraction(inst.pillar_date)

        def residual(df_k: float, inst: CalibrationInstrument = inst, t_k: float = t_k) -> float:
            trial = DiscountCurve(time_measure, (*times, t_k), (*dfs, df_k), build.interpolation)
            return inst.residual(trial if discount is None else discount, trial)

        df_k = brent(residual, 1e-6, 1.0)
        times.append(t_k)
        dfs.append(df_k)
    return DiscountCurve(time_measure, tuple(times), tuple(dfs), build.interpolation)


def calibrate(spec: CalibrationSpec) -> tuple[DiscountingModel, CalibrationResult]:
    """Build the curve set in dependency order (discount → projection), assemble the
    `CurveSet`, and wrap it in a `DiscountingModel`."""
    discount_curve = _bootstrap(spec, spec.discount, discount=None)
    projection_curve = _bootstrap(spec, spec.projection, discount=discount_curve)
    curves = CurveSet(
        {
            CurveKey(CurveRole.DISCOUNT, spec.currency): discount_curve,
            CurveKey(CurveRole.PROJECTION, spec.discount.index): discount_curve,
            CurveKey(CurveRole.PROJECTION, spec.projection.index): projection_curve,
        }
    )
    model = DiscountingModel(MarketSnapshot(spec.valuation_date, curves))
    residuals = tuple(
        _instrument(spec, build, q).residual(discount_curve, curves.projection(build.index))
        for build in (spec.discount, spec.projection)
        for q in build.quotes
    )
    return model, CalibrationResult(residuals, converged=all(abs(r) < 1e-10 for r in residuals))
