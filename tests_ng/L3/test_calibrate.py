"""L3 oracle — the sequential single-curve bootstrap.

The calibrator builds the discount curve pillar by pillar; every calibrating par
swap must reprice to its quoted rate through the SAME building blocks the engine
composes (§3d). `calibrate` returns the model as its OUTPUT and a result BESIDE it.
"""

from datetime import date

from pricebook_ng.calibration.calibrate import (
    CalibrationSpec,
    ParSwapQuote,
    SingleCurveSpec,
    calibrate,
    single_curve_swap,
)
from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    Interpolation,
    Tenor,
    TenorUnit,
)
from pricebook_ng.market.building_blocks import float_leg_pv, rpv01

VAL = date(2026, 1, 15)
TARGET = SingleCurveSpec(
    valuation_date=VAL,
    currency=Currency.USD,
    frequency=Frequency.ANNUAL,
    day_count=DayCountConvention.ACT_365_FIXED,
    interpolation=Interpolation.LOG_LINEAR,
)
QUOTES = (
    ParSwapQuote(Tenor(1, TenorUnit.YEAR), 0.030),
    ParSwapQuote(Tenor(2, TenorUnit.YEAR), 0.032),
    ParSwapQuote(Tenor(3, TenorUnit.YEAR), 0.034),
    ParSwapQuote(Tenor(5, TenorUnit.YEAR), 0.036),
)
SPEC = CalibrationSpec(target=TARGET, quotes=QUOTES)


def test_bootstrap_reprices_every_calibrating_swap_to_par() -> None:
    model, result = calibrate(SPEC)
    assert result.converged
    curve = model.market.discount_curve
    for q in QUOTES:
        swap = single_curve_swap(TARGET, q.tenor, q.rate)
        par_rate = float_leg_pv(swap.float_leg.schedule, curve) / rpv01(
            swap.fixed_leg.schedule, swap.fixed_leg.day_count, curve
        )
        assert abs(par_rate - q.rate) < 1e-10


def test_calibration_result_residuals_are_small() -> None:
    _, result = calibrate(SPEC)
    assert len(result.residuals) == len(QUOTES)
    assert max(abs(r) for r in result.residuals) < 1e-10


def test_model_carries_its_market() -> None:
    model, _ = calibrate(SPEC)
    assert model.market.valuation_date == VAL
    assert model.market.discount_curve.df(VAL) == 1.0
