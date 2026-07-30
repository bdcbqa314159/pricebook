"""Dual-curve bootstrap — one `calibrate(spec)`, sequential, dependency-ordered (L3).

`calibrate(spec) → (DiscountingModel, CalibrationResult)`: the model is the OUTPUT (doc
22 Q2). The spec describes a curve SET (doc 22 Q4): a discount build + one projection
build. The curves are built in DEPENDENCY ORDER — the discount (OIS) curve first,
self-discounting; then the projection (IBOR) curve, its swaps discounted on the built
discount curve. Each pillar's df is solved (Brent) so its par swap reprices to the quote,
through the SAME `rpv01`/`float_leg_pv` building blocks the L4 engine composes (§3d) — the
calibrator never imports L4. Still one SEQUENTIAL orchestration: no `method`/`numerics`/
Jacobian (rule of two). A second projection (→ an `{index: build}` map), the global solve,
and CSA/xccy discounting arrive with their own slices.

Provenance:
  quarry: python/pricebook/curves/bootstrap.py; curves/ncurve_solver.py (concept only)
  source: redesign/22 (calibrate contract; trial model = whole CurveSet); §3d (shared atoms)
  oracle: every OIS + EURIBOR calibrating swap reprices to par; EURIBOR swap → zero dual-curve
  slice:  dual-curve-euribor-estr (T1 slice 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

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


@dataclass(frozen=True)
class ParSwapQuote:
    """A par-swap calibrating quote: the swap `tenor` and its market par `rate`."""

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
    """Sits BESIDE the model, never on it (F1 §5): the per-quote `residuals` (par_rate −
    quote) across both curves and whether the solve `converged`."""

    residuals: tuple[float, ...]
    converged: bool


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


def _bootstrap(
    spec: CalibrationSpec, build: CurveBuild, discount: CurveHandle | None
) -> DiscountCurve:
    """Sequential per-pillar bootstrap. `discount is None` ⇒ self-discounting (the OIS
    curve is its own discount and projection); otherwise the swaps discount on the
    supplied (already-built) discount curve and project off the curve being built."""
    time_measure = TimeMeasure(spec.valuation_date, build.day_count)
    times: list[float] = [0.0]
    dfs: list[float] = [1.0]
    for quote in sorted(build.quotes, key=lambda q: q.tenor.months()):
        swap = par_swap(spec, build, quote.tenor, quote.rate)
        t_k = time_measure.year_fraction(spec.valuation_date + quote.tenor)

        def residual(df_k: float, swap: VanillaSwap = swap, t_k: float = t_k, rate: float = quote.rate) -> float:
            trial = DiscountCurve(time_measure, (*times, t_k), (*dfs, df_k), build.interpolation)
            return _par_rate(swap, trial if discount is None else discount, trial) - rate

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
        _par_rate(par_swap(spec, build, q.tenor, q.rate), discount_curve, curves.projection(build.index))
        - q.rate
        for build in (spec.discount, spec.projection)
        for q in build.quotes
    )
    return model, CalibrationResult(residuals, converged=all(abs(r) < 1e-10 for r in residuals))
