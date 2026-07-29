"""L4 headline oracle — a par swap prices to ZERO NPV through the full engine.

This is doc 18's C2 close condition and doc 22 §Q2's backstop: not merely that the
calibrator residual is zero, but that the L4 engine — composing the same §3d atoms,
reaching the market only through the model — reprices each par swap to zero. Plus an
analytic-vs-finite-difference DV01, statelessness, and failure-as-a-value.
"""

from datetime import date

from pricebook_ng.calibration.calibrate import (
    CalibrationSpec,
    ParSwapQuote,
    SingleCurveSpec,
    calibrate,
    single_curve_swap,
)
from pricebook_ng.engine.linear import price
from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    Interpolation,
    PricingFailure,
    PricingResult,
    Tenor,
    TenorUnit,
)
from pricebook_ng.market.building_blocks import rpv01

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


def test_par_swap_prices_to_zero_npv_through_the_engine() -> None:
    model, _ = calibrate(SPEC)
    for q in QUOTES:
        swap = single_curve_swap(TARGET, q.tenor, q.rate)
        result = price(swap, model)
        assert isinstance(result, PricingResult)
        assert result.pv.currency == Currency.USD
        assert abs(result.pv.amount) < 1e-9  # zero NPV


def test_dv01_analytic_matches_finite_difference() -> None:
    model, _ = calibrate(SPEC)
    curve = model.market.discount_curve
    swap = single_curve_swap(TARGET, Tenor(5, TenorUnit.YEAR), 0.036)
    # PV = N·(float − rate·annuity) is exactly linear in rate ⇒ dPV/drate = −N·annuity.
    analytic = -swap.notional * rpv01(
        swap.fixed_leg.schedule, swap.fixed_leg.day_count, curve
    )
    h = 1e-6
    up = price(single_curve_swap(TARGET, Tenor(5, TenorUnit.YEAR), 0.036 + h), model)
    dn = price(single_curve_swap(TARGET, Tenor(5, TenorUnit.YEAR), 0.036 - h), model)
    assert isinstance(up, PricingResult) and isinstance(dn, PricingResult)
    finite_diff = (up.pv.amount - dn.pv.amount) / (2 * h)
    assert abs(analytic - finite_diff) < 1e-6 * abs(analytic)


def test_repricing_is_byte_identical() -> None:
    model, _ = calibrate(SPEC)
    swap = single_curve_swap(TARGET, Tenor(3, TenorUnit.YEAR), 0.034)
    assert price(swap, model) == price(swap, model)


def test_cashflow_beyond_the_curve_is_a_failure_value() -> None:
    # curve reaches 5y; a 10y swap has payments past the last pillar ⇒ a value, not a raise.
    model, _ = calibrate(SPEC)
    swap = single_curve_swap(TARGET, Tenor(10, TenorUnit.YEAR), 0.036)
    assert isinstance(price(swap, model), PricingFailure)
