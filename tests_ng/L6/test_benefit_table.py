"""Amendment A3 oracle — Trade / Book + the benefit table (L6 shell).

A `Trade` holds a collection of products + a start date; a `Book` collects
trades. The shell remembers **realized P&L** (the benefit table): cashflows that
have already paid, as actual cash — never discounted. The engine computes the
mark; total economics = realized + mark, reconciling over the trade's life.
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
from pricebook_ng.shell.booking import Book, Trade, book

CCY = Currency.USD
NOTIONAL = 1_000_000.0
START = date(2024, 1, 15)
MATURITY = date(2029, 1, 15)


def _bond():
    return fixed_rate_bond(
        face=Money(NOTIONAL, CCY), coupon_rate=0.04, start=START, maturity=MATURITY,
        terms=ScheduleTerms(Frequency.SEMI_ANNUAL, DC.THIRTY_360),
    )


def _trade():
    return Trade(products=(_bond(),), start_date=START)


def _market(valuation, rate=0.03):
    curve = FlatDiscountCurve(rate=rate, anchor=valuation, day_count=DC.ACT_365_FIXED)
    return MarketSnapshot(valuation_date=valuation, discount_curve=curve)


def test_realized_at_issue_is_zero():
    assert book(_trade()).realized(START).amount == pytest.approx(0.0, abs=1e-9)


def test_realized_sums_paid_cashflows_undiscounted():
    v = date(2026, 4, 15)
    expected = sum(cf.amount.amount for cf in _bond().cashflows if cf.date <= v)
    realized = book(_trade()).realized(v)
    assert realized.amount == pytest.approx(expected, abs=1e-6)
    assert realized.currency is CCY


def test_realized_plus_remaining_is_total_nominal():
    v = date(2026, 4, 15)
    cfs = _bond().cashflows
    remaining = sum(cf.amount.amount for cf in cfs if cf.date > v)
    total = sum(cf.amount.amount for cf in cfs)
    assert book(_trade()).realized(v).amount + remaining == pytest.approx(total, abs=1e-6)


def test_end_of_life_realized_is_total_and_mark_is_zero():
    booked = book(_trade())
    after = date(2029, 6, 1)  # past maturity
    total = sum(cf.amount.amount for cf in _bond().cashflows)
    assert booked.realized(after).amount == pytest.approx(total, abs=1e-6)
    mark = booked.value(_market(after), NumericalConfig(), DiscountingEngine())
    assert mark.pv.amount == pytest.approx(0.0, abs=1e-6)  # nothing left to price


def test_book_aggregates_realized_over_trades():
    v = date(2026, 4, 15)
    single = book(_trade()).realized(v).amount
    b = Book(trades=(_trade(), _trade()))
    assert b.realized(v).amount == pytest.approx(2 * single, abs=1e-6)
