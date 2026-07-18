"""A6.2 oracle — a trade's life: realized (benefit table) + mark reconcile (L6 shell).

Extends the A3 benefit-table oracle to the full A3 split: the shell **remembers**
realized cash (undiscounted, past) and the core computes the **mark** (future PV +
accrued). Over a bond's life:
  - at issue: realized = 0, so total economics = mark = the full discounted PV;
  - mid-life: realized = Σ paid (undiscounted); the mark prices ONLY future flows
    (the engine excludes the paid ones, A2); dirty = clean + accrued;
  - at maturity: mark = 0, realized = the total nominal.
And a Book's mark is the sum of its trades' marks (linearity), matching its realized.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.engine.discounting import DiscountingEngine
from pricebook_ng.products.fixed_rate_bond import fixed_rate_bond
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.shell.booking import Book, Trade

CCY = Currency.USD
NOTIONAL = 1_000_000.0
START = date(2024, 1, 15)
MATURITY = date(2029, 1, 15)
RATE = 0.03
NUM = NumericalConfig()
ENGINE = DiscountingEngine()


def _bond():
    return fixed_rate_bond(
        face=Money(NOTIONAL, CCY), coupon_rate=0.04, start=START, maturity=MATURITY,
        terms=ScheduleTerms(Frequency.SEMI_ANNUAL, DC.THIRTY_360),
    )


def _trade():
    return Trade(products=(_bond(),), start_date=START)


def _market(valuation, rate=RATE):
    return MarketSnapshot(
        valuation_date=valuation, discount_curve=FlatDiscountCurve(rate, valuation, DC.ACT_365_FIXED)
    )


def test_mark_at_issue_equals_full_discounted_pv():
    market = _market(START)
    curve = market.discount_curve
    expected = sum(cf.amount.amount * curve.df(cf.date) for cf in _bond().cashflows)  # all future
    mark = _trade().mark(market, NUM, ENGINE)
    assert mark.pv.amount == pytest.approx(expected, abs=1e-6)
    assert _trade().realized(START).amount == 0.0  # nothing paid -> total economics = mark


def test_mid_life_mark_prices_only_future_flows():
    v = date(2026, 1, 15)  # a coupon date (accrued 0), mid-life
    market = _market(v)
    curve = market.discount_curve
    future = sum(cf.amount.amount * curve.df(cf.date) for cf in _bond().cashflows if cf.date > v)
    paid = sum(cf.amount.amount for cf in _bond().cashflows if cf.date <= v)
    mark = _trade().mark(market, NUM, ENGINE)
    assert mark.pv.amount == pytest.approx(future, abs=1e-4)      # engine excluded the paid flows
    assert _trade().realized(v).amount == pytest.approx(paid, abs=1e-6)  # …captured by the shell


def test_dirty_equals_clean_plus_accrued_mid_period():
    v = date(2026, 4, 15)  # 3 months into a 6-month period
    mark = _trade().mark(_market(v), NUM, ENGINE)
    assert mark.accrued is not None and mark.accrued.amount > 0.0
    assert mark.pv.amount == pytest.approx(mark.clean.amount + mark.accrued.amount, abs=1e-9)


def test_end_of_life_mark_is_zero_and_realized_is_total():
    after = date(2029, 6, 1)
    total_nominal = sum(cf.amount.amount for cf in _bond().cashflows)
    assert _trade().mark(_market(after), NUM, ENGINE).pv.amount == pytest.approx(0.0, abs=1e-6)
    assert _trade().realized(after).amount == pytest.approx(total_nominal, abs=1e-6)


def test_book_mark_is_sum_of_trade_marks():
    v = date(2026, 1, 15)
    market = _market(v)
    single = _trade().mark(market, NUM, ENGINE).pv.amount
    b = Book(trades=(_trade(), _trade()))
    assert b.value(market, NUM, ENGINE).pv.amount == pytest.approx(2.0 * single, abs=1e-6)
    assert b.realized(v).amount == pytest.approx(2.0 * _trade().realized(v).amount, abs=1e-6)
