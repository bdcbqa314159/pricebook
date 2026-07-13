"""Seasoned swap oracle — float-leg fixings (L4).

A float coupon whose reset is strictly before valuation was already fixed: its
rate comes from the realized `FixingHistory` (finally consuming the type A1 added
to the snapshot), not a forward off the curve. Periods that already paid are
settled (A2 segment-and-settle); future periods project forwards.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.discounting import DiscountingEngine
from pricebook_ng.engine.swap import SwapEngine
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.market.snapshot import FixingHistory, FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel

CCY = Currency.USD
NOTIONAL = 1_000_000.0
FIXED_TERMS = ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360)
FLOAT_TERMS = ScheduleTerms(Frequency.ANNUAL, DC.ACT_360)
FLOAT_DC = DC.ACT_360
NUM = NumericalConfig()


def _market(valuation, fixings=None, rate=0.03):
    return MarketSnapshot(
        valuation_date=valuation,
        discount_curve=FlatDiscountCurve(rate, valuation, DC.ACT_365_FIXED),
        fixings=FixingHistory(fixings or {}),
    )


def _swap(start, maturity, pay_fixed=True):
    return vanilla_swap(
        face=Money(NOTIONAL, CCY), fixed_rate=0.03, start=start, maturity=maturity,
        terms=SwapTerms(FIXED_TERMS, FLOAT_TERMS, pay_fixed),
    )


def _independent_float_pv(dates, df, valuation, fixings):
    pv = 0.0
    for a, b in zip(dates[:-1], dates[1:]):
        if b <= valuation:
            continue                                   # settled
        if a < valuation:
            accrual = fixings[a] * year_fraction(a, b, FLOAT_DC)   # realized fixing
        else:
            accrual = df(a) / df(b) - 1.0                          # forward
        pv += NOTIONAL * accrual * df(b)
    return pv


def test_seasoned_swap_uses_fixing_for_current_period():
    start, maturity = date(2025, 1, 15), date(2028, 1, 15)
    valuation = date(2025, 7, 15)             # mid the first float period
    reset = date(2025, 1, 15)
    market = _market(valuation, {reset: 0.028})
    swap = _swap(start, maturity)

    result = SwapEngine().price(swap, DiscountingModel(market), NUM)
    assert isinstance(result, PricingResult)

    df = market.discount_curve.df
    float_pv = _independent_float_pv(swap.float_leg.schedule, df, valuation, {reset: 0.028})
    fixed_pv = DiscountingEngine().price(swap.fixed_leg, DiscountingModel(market), NUM).pv.amount
    assert result.pv.amount == pytest.approx(float_pv - fixed_pv, abs=1e-6)  # payer = float - fixed


def test_missing_fixing_is_a_failure():
    market = _market(date(2025, 7, 15), fixings={})   # reset not recorded
    swap = _swap(date(2025, 1, 15), date(2028, 1, 15))
    result = SwapEngine().price(swap, DiscountingModel(market), NUM)
    assert isinstance(result, PricingFailure)


def test_already_paid_float_period_is_excluded():
    start, maturity = date(2024, 1, 15), date(2028, 1, 15)
    valuation = date(2025, 7, 15)             # first period (ends 2025-01-15) already paid
    fixings = {date(2024, 1, 15): 0.027, date(2025, 1, 15): 0.028}
    market = _market(valuation, fixings)
    swap = _swap(start, maturity)

    df = market.discount_curve.df
    float_pv = _independent_float_pv(swap.float_leg.schedule, df, valuation, fixings)
    fixed_pv = DiscountingEngine().price(swap.fixed_leg, DiscountingModel(market), NUM).pv.amount
    result = SwapEngine().price(swap, DiscountingModel(market), NUM)
    assert result.pv.amount == pytest.approx(float_pv - fixed_pv, abs=1e-6)
    # the 2024 period paid on 2025-01-15 (<= valuation) is settled, not in PV
    assert swap.float_leg.schedule[1] <= valuation


def test_spot_swap_needs_no_fixings():
    # start == valuation: the first reset is today -> forward, no FixingHistory needed
    valuation = date(2026, 1, 15)
    market = _market(valuation)               # empty fixings
    swap = _swap(valuation, date(2029, 1, 15))
    result = SwapEngine().price(swap, DiscountingModel(market), NUM)
    assert isinstance(result, PricingResult)


def test_float_leg_carries_day_count():
    swap = _swap(date(2026, 1, 15), date(2029, 1, 15))
    assert swap.float_leg.day_count is FLOAT_DC
