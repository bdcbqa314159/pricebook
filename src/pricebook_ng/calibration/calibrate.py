"""Single-curve bootstrap — the first calibrator (L3, a free function).

`calibrate(spec) → (DiscountingModel, CalibrationResult)`: the model is the OUTPUT — a
model never calibrates itself (doc 22 Q2). This is the SEQUENTIAL orchestration (doc 22
Q4): the discount curve is built pillar by pillar, each pillar's df solved (Brent) so
its par swap reprices to the quoted rate. The residual is formed through the SAME
`rpv01`/`float_leg_pv` building blocks the L4 engine composes (§3d) — the calibrator
never imports L4. Reprice-to-par is blind to mis-resolution (F1 §6); the real safety is
the shared atoms, backstopped by the engine repricing each swap to zero NPV (doc 22 Q2).
`method`/`numerics` arrive with the second orchestration / the first tuned knob (rule of
two); a numerically-priced calibration target is out of scope until §0(B) (doc 22).

Provenance:
  quarry: python/pricebook/curves/bootstrap.py
  source: redesign/22 (calibrate contract, sequential orchestration); §3d (shared atoms)
  oracle: every calibrating swap reprices to par; par swap → zero NPV via the engine
  slice:  swap-to-zero-npv (T1 slice 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    Interpolation,
    RollRule,
    ScheduleTerms,
    Tenor,
    TimeMeasure,
    brent,
    build_schedule,
)
from pricebook_ng.market.building_blocks import float_leg_pv, rpv01
from pricebook_ng.market.curve import DiscountCurve
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.swap import FixedLeg, FloatLeg, VanillaSwap


@dataclass(frozen=True)
class ParSwapQuote:
    """A par-swap calibrating quote: the swap `tenor` and its market par `rate`."""

    tenor: Tenor
    rate: float


@dataclass(frozen=True)
class SingleCurveSpec:
    """How to build the curve and its calibrating swaps: the `valuation_date` (the
    curve anchor), the `currency`, both legs' `frequency` and `day_count`, and the
    curve `interpolation`. Single-curve, unadjusted dates (no calendar) — the minimal
    T1 world."""

    valuation_date: date
    currency: Currency
    frequency: Frequency
    day_count: DayCountConvention
    interpolation: Interpolation = Interpolation.LOG_LINEAR


@dataclass(frozen=True)
class CalibrationSpec:
    """The calibration inputs, all data (doc 22 Q2): the curve/instrument `target` and
    the `quotes` that pin it."""

    target: SingleCurveSpec
    quotes: tuple[ParSwapQuote, ...]


@dataclass(frozen=True)
class CalibrationResult:
    """Sits BESIDE the model, never on it (F1 §5): the final per-quote `residuals`
    (par_rate − quote) and whether the solve `converged`."""

    residuals: tuple[float, ...]
    converged: bool


def single_curve_swap(
    spec: SingleCurveSpec, tenor: Tenor, rate: float, notional: float = 1.0
) -> VanillaSwap:
    """Build the par swap the bootstrap calibrates to (and the engine reprices) — ONE
    construction, so the calibrator and the pricing oracle share the exact schedule
    (§3d argument identity)."""
    terms = ScheduleTerms(frequency=spec.frequency, roll=RollRule(calendar=None))
    schedule = build_schedule(spec.valuation_date, spec.valuation_date + tenor, terms)
    return VanillaSwap(
        notional=notional,
        currency=spec.currency,
        fixed_leg=FixedLeg(schedule, spec.day_count, rate),
        float_leg=FloatLeg(schedule, spec.day_count),
    )


def _par_rate(swap: VanillaSwap, curve: DiscountCurve) -> float:
    return float_leg_pv(swap.float_leg.schedule, curve) / rpv01(
        swap.fixed_leg.schedule, swap.fixed_leg.day_count, curve
    )


def calibrate(spec: CalibrationSpec) -> tuple[DiscountingModel, CalibrationResult]:
    """Sequential single-curve bootstrap: solve each pillar's discount factor so its par
    swap reprices to the quoted rate, then wrap the built curve in a `DiscountingModel`."""
    target = spec.target
    time_measure = TimeMeasure(target.valuation_date, target.day_count)
    quotes = sorted(spec.quotes, key=lambda q: q.tenor.months())
    times: list[float] = [0.0]
    dfs: list[float] = [1.0]
    for quote in quotes:
        swap = single_curve_swap(target, quote.tenor, quote.rate)
        t_k = time_measure.year_fraction(target.valuation_date + quote.tenor)

        def residual(df_k: float, swap: VanillaSwap = swap, t_k: float = t_k, rate: float = quote.rate) -> float:
            trial = DiscountCurve(time_measure, (*times, t_k), (*dfs, df_k), target.interpolation)
            return _par_rate(swap, trial) - rate

        df_k = brent(residual, 1e-6, 1.0)
        times.append(t_k)
        dfs.append(df_k)

    curve = DiscountCurve(time_measure, tuple(times), tuple(dfs), target.interpolation)
    model = DiscountingModel(MarketSnapshot(target.valuation_date, curve))
    residuals = tuple(
        _par_rate(single_curve_swap(target, q.tenor, q.rate), curve) - q.rate for q in quotes
    )
    return model, CalibrationResult(residuals, converged=all(abs(r) < 1e-10 for r in residuals))
