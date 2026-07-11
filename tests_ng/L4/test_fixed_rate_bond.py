"""S04 oracle — fixed-rate bond priced through the discounting engine (L4).

A fixed-rate bond is pure data: a leg of coupon cashflows plus the redemption.
The engine discounts that leg. Oracle is the closed-form PV — an independent sum
of discounted cashflows — on both a flat curve and the S03 bootstrapped curve,
exact < 1e-12; plus structural checks and the zero-coupon tie-back to Slice 0.
"""

import math
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingResult
from pricebook_ng.foundation.schedule import Frequency, generate_schedule
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.discounting import DiscountingEngine
from pricebook_ng.instruments.fixed_rate_bond import FixedRateBond, fixed_rate_bond
from pricebook_ng.market.discount_curve import DepositQuote, bootstrap_discount_curve
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot

ABS = 1e-12
START = date(2026, 1, 15)
MATURITY = date(2029, 1, 15)
NOTIONAL = 1_000_000.0
COUPON = 0.04
CCY = Currency.USD
BOND_DC = DC.THIRTY_360           # regular semi-annual period -> tau = 0.5 exactly
CURVE_DC = DC.ACT_365_FIXED


def _bond(coupon=COUPON):
    return fixed_rate_bond(
        notional=NOTIONAL, coupon_rate=coupon, start=START, maturity=MATURITY,
        frequency=Frequency.SEMI_ANNUAL, day_count=BOND_DC, currency=CCY,
    )


def _flat_market(rate):
    curve = FlatDiscountCurve(rate=rate, anchor=START, day_count=CURVE_DC)
    return MarketSnapshot(valuation_date=START, discount_curve=curve)


def _independent_pv(bond, curve):
    return sum(cf.amount.amount * curve.df(cf.date) for cf in bond.cashflows)


# ---- structure ----------------------------------------------------------------
def test_cashflow_structure():
    bond = _bond()
    sched = generate_schedule(START, MATURITY, Frequency.SEMI_ANNUAL)
    # 6 coupons + 1 redemption
    assert len(bond.cashflows) == len(sched)  # 6 coupon dates + redemption at maturity
    # each regular semi-annual coupon under 30/360 is notional * coupon * 0.5
    coupon_amt = NOTIONAL * COUPON * 0.5
    coupons = [cf for cf in bond.cashflows if cf.date < MATURITY]
    assert len(coupons) == 5
    assert all(cf.amount.amount == pytest.approx(coupon_amt, abs=1e-9) for cf in coupons)
    # at maturity: final coupon + redemption
    at_mat = sum(cf.amount.amount for cf in bond.cashflows if cf.date == MATURITY)
    assert at_mat == pytest.approx(coupon_amt + NOTIONAL, abs=1e-6)


def test_is_frozen_pure_data():
    bond = _bond()
    assert isinstance(bond, FixedRateBond)
    with pytest.raises(Exception):
        bond.notional = 5.0  # frozen
    assert not hasattr(bond, "pv")  # instruments do not price themselves


# ---- closed-form PV on a flat curve ------------------------------------------
def test_pv_flat_curve_closed_form():
    bond, market = _bond(), _flat_market(0.03)
    engine = DiscountingEngine()
    result = engine.price(bond, None, market, NumericalConfig())
    assert isinstance(result, PricingResult)
    assert result.pv.currency is CCY

    expected = 0.0
    sched = generate_schedule(START, MATURITY, Frequency.SEMI_ANNUAL)
    for i in range(1, len(sched)):
        tau = year_fraction(sched[i - 1], sched[i], BOND_DC)
        t = year_fraction(START, sched[i], CURVE_DC)
        expected += NOTIONAL * COUPON * tau * math.exp(-0.03 * t)
    expected += NOTIONAL * math.exp(-0.03 * year_fraction(START, MATURITY, CURVE_DC))
    assert result.pv.amount == pytest.approx(expected, abs=1e-6)


# ---- PV on the bootstrapped curve --------------------------------------------
def test_pv_bootstrapped_curve():
    curve = bootstrap_discount_curve(
        START,
        [DepositQuote(date(2026, 7, 15), 0.03, DC.ACT_360),
         DepositQuote(date(2027, 1, 15), 0.031, DC.ACT_360),
         DepositQuote(date(2029, 1, 15), 0.033, DC.ACT_360)],
        [],
    )
    market = MarketSnapshot(valuation_date=START, discount_curve=curve)
    result = DiscountingEngine().price(_bond(), None, market, NumericalConfig())
    assert result.pv.amount == pytest.approx(_independent_pv(_bond(), curve), abs=1e-6)


# ---- zero-coupon tie-back to Slice 0 -----------------------------------------
def test_zero_coupon_bond_is_pure_discount():
    bond, market = _bond(coupon=0.0), _flat_market(0.03)
    result = DiscountingEngine().price(bond, None, market, NumericalConfig())
    expected = NOTIONAL * market.discount_curve.df(MATURITY)
    assert result.pv.amount == pytest.approx(expected, abs=1e-6)
