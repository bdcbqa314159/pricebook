"""L4 headline oracle — a EURIBOR swap prices to ZERO off ESTR discounting.

The genuine dual-curve result (project off EURIBOR_3M, discount off ESTR), plus the
degeneracy regression guard (an OIS swap — projection == discount — still prices to
zero through the generalised path, i.e. slice 1 survives), the DV01, statelessness,
and failure-as-a-value.
"""

from datetime import date

from pricebook_ng.calibration.calibrate import (
    CalibrationSpec,
    CurveBuild,
    ParSwapQuote,
    calibrate,
    par_swap,
)
from pricebook_ng.engine.linear import price
from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    PricingFailure,
    PricingResult,
    Tenor,
    TenorUnit,
    get_rate_index,
)
from pricebook_ng.market.building_blocks import rpv01

VAL = date(2026, 1, 15)
ESTR = get_rate_index("ESTR")
EURIBOR_3M = get_rate_index("EURIBOR_3M")
DC = DayCountConvention.ACT_365_FIXED
OIS = tuple(
    ParSwapQuote(Tenor(y, TenorUnit.YEAR), r) for y, r in [(1, 0.030), (2, 0.032), (3, 0.034), (5, 0.036)]
)
EURIBOR = tuple(
    ParSwapQuote(Tenor(y, TenorUnit.YEAR), r) for y, r in [(1, 0.0312), (2, 0.0332), (3, 0.0352), (5, 0.0372)]
)
DISCOUNT = CurveBuild(index=ESTR, frequency=Frequency.ANNUAL, day_count=DC, quotes=OIS)
PROJECTION = CurveBuild(index=EURIBOR_3M, frequency=Frequency.ANNUAL, day_count=DC, quotes=EURIBOR)
SPEC = CalibrationSpec.single_currency(valuation_date=VAL, currency=Currency.EUR, discount=DISCOUNT, projection=PROJECTION)


def test_euribor_dual_curve_swap_prices_to_zero_npv() -> None:
    model, _ = calibrate(SPEC)
    for q in EURIBOR:
        result = price(par_swap(SPEC, PROJECTION, q.tenor, q.rate), model)
        assert isinstance(result, PricingResult)
        assert result.pv.currency == Currency.EUR
        assert abs(result.pv.amount) < 1e-9


def test_ois_swap_still_prices_to_zero_through_the_generalised_path() -> None:
    # slice-1 regression guard: projection == discount, priced via the general float leg.
    model, _ = calibrate(SPEC)
    for q in OIS:
        result = price(par_swap(SPEC, DISCOUNT, q.tenor, q.rate), model)
        assert isinstance(result, PricingResult)
        assert abs(result.pv.amount) < 1e-9


def test_dv01_analytic_matches_finite_difference() -> None:
    model, _ = calibrate(SPEC)
    discount = model.market.curves.discount(Currency.EUR)
    swap = par_swap(SPEC, PROJECTION, Tenor(5, TenorUnit.YEAR), 0.0372)
    # PV = N·(float − rate·annuity) is exactly linear in rate ⇒ dPV/drate = −N·annuity.
    analytic = -swap.notional * rpv01(swap.fixed_leg.schedule, swap.fixed_leg.day_count, discount)
    h = 1e-6
    up = price(par_swap(SPEC, PROJECTION, Tenor(5, TenorUnit.YEAR), 0.0372 + h), model)
    dn = price(par_swap(SPEC, PROJECTION, Tenor(5, TenorUnit.YEAR), 0.0372 - h), model)
    assert isinstance(up, PricingResult) and isinstance(dn, PricingResult)
    finite_diff = (up.pv.amount - dn.pv.amount) / (2 * h)
    assert abs(analytic - finite_diff) < 1e-6 * abs(analytic)


def test_repricing_is_byte_identical() -> None:
    model, _ = calibrate(SPEC)
    swap = par_swap(SPEC, PROJECTION, Tenor(3, TenorUnit.YEAR), 0.0352)
    assert price(swap, model) == price(swap, model)


def test_cashflow_beyond_the_curve_is_a_failure_value() -> None:
    model, _ = calibrate(SPEC)
    swap = par_swap(SPEC, PROJECTION, Tenor(10, TenorUnit.YEAR), 0.0372)
    assert isinstance(price(swap, model), PricingFailure)
