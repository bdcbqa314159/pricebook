"""Amendment A2 oracle — valuation is temporality-aware (L4).

The engine partitions a product's cashflows by the valuation date
(`model.market.valuation_date`): cashflows on or before it are historical
(excluded from PV — the shell settles them), future cashflows discount from
valuation. The current coupon period accrues; PricingResult decomposes into
dirty PV, accrued, and clean.
"""

import math
from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.cashflow import Cashflow
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.discounting import DiscountingEngine
from pricebook_ng.instruments.fixed_cashflow import FixedCashflowTrade
from pricebook_ng.instruments.fixed_rate_bond import fixed_rate_bond
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel

ABS = 1e-6
CCY = Currency.USD
NOTIONAL = 1_000_000.0
COUPON = 0.04
DC30 = DC.THIRTY_360

START = date(2024, 1, 15)
MATURITY = date(2029, 1, 15)
VALUATION = date(2026, 4, 15)          # mid the 2026-01-15 .. 2026-07-15 period
CURVE_DC = DC.ACT_365_FIXED
RATE = 0.03


def _model(valuation=VALUATION, rate=RATE):
    curve = FlatDiscountCurve(rate=rate, anchor=valuation, day_count=CURVE_DC)
    return DiscountingModel(MarketSnapshot(valuation_date=valuation, discount_curve=curve))


def _seasoned_bond():
    return fixed_rate_bond(
        face=Money(NOTIONAL, CCY), coupon_rate=COUPON, start=START, maturity=MATURITY,
        terms=ScheduleTerms(Frequency.SEMI_ANNUAL, DC30),
    )


def test_seasoned_bond_excludes_paid_coupons():
    bond, model = _seasoned_bond(), _model()
    df = model.market.discount_curve.df
    result = DiscountingEngine().price(bond, model, NumericalConfig())
    # independent dirty PV: only cashflows strictly after valuation
    expected = sum(
        cf.amount.amount * df(cf.date) for cf in bond.cashflows if cf.date > VALUATION
    )
    assert result.pv.amount == pytest.approx(expected, abs=ABS)
    # there ARE excluded past coupons (so this is a real seasoning test)
    assert any(cf.date <= VALUATION for cf in bond.cashflows)


def test_accrued_and_clean_dirty_decomposition():
    bond, model = _seasoned_bond(), _model()
    result = DiscountingEngine().price(bond, model, NumericalConfig())
    # current period 2026-01-15 .. 2026-07-15, 30/360 -> accrued = coupon * 0.25/0.5
    coupon = NOTIONAL * COUPON * year_fraction(date(2026, 1, 15), date(2026, 7, 15), DC30)
    frac = (year_fraction(date(2026, 1, 15), VALUATION, DC30)
            / year_fraction(date(2026, 1, 15), date(2026, 7, 15), DC30))
    assert result.accrued.amount == pytest.approx(coupon * frac, abs=ABS)
    assert result.accrued.amount == pytest.approx(10_000.0, abs=ABS)  # 20000 * 0.5
    # dirty = clean + accrued
    assert result.clean.amount == pytest.approx(result.pv.amount - result.accrued.amount, abs=ABS)


def test_at_issue_bond_has_zero_accrued():
    # valuation == start: no period has accrued yet
    bond = _seasoned_bond()
    model = _model(valuation=START)
    result = DiscountingEngine().price(bond, model, NumericalConfig())
    assert result.accrued.amount == pytest.approx(0.0, abs=ABS)
    assert result.clean.amount == pytest.approx(result.pv.amount, abs=ABS)


def test_forward_starting_prices_only_future():
    future = date(2030, 1, 1)
    trade = FixedCashflowTrade(Cashflow(date=future, amount=Money(NOTIONAL, CCY)))
    model = _model()
    result = DiscountingEngine().price(trade, model, NumericalConfig())
    t = year_fraction(VALUATION, future, CURVE_DC)
    assert result.pv.amount == pytest.approx(NOTIONAL * math.exp(-RATE * t), abs=ABS)


def test_cashflow_on_valuation_date_is_historical():
    trade = FixedCashflowTrade(Cashflow(date=VALUATION, amount=Money(NOTIONAL, CCY)))
    result = DiscountingEngine().price(trade, _model(), NumericalConfig())
    assert result.pv.amount == pytest.approx(0.0, abs=ABS)  # excluded, not discounted
