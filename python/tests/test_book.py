"""Tests for position management: Book, Desk, positions, and limits."""

import pytest
from datetime import date

from pricebook.book import (
    Book, Desk, BookLimits, Position, LimitBreach,
    tenor_bucket,
)
from pricebook.trade import Trade
from pricebook.pricing_context import PricingContext
from pricebook.swaption import Swaption


REF = date(2024, 1, 15)


def _ctx(rate=0.05):
    return PricingContext.simple(REF, rate=rate, vol=0.20)


def _swn(expiry_year=2025, end_year=2030, strike=0.05, notional=1_000_000):
    return Swaption(
        date(expiry_year, 1, 15), date(end_year, 1, 15),
        strike=strike, notional=notional,
    )


# ---- Tenor bucketing ----

class TestTenorBucket:
    def test_short_tenor(self):
        assert tenor_bucket(REF, date(2024, 3, 15)) == "≤3M"

    def test_six_month(self):
        assert tenor_bucket(REF, date(2024, 7, 1)) == "3M-6M"

    def test_two_year(self):
        assert tenor_bucket(REF, date(2025, 12, 15)) == "1Y-2Y"

    def test_five_year(self):
        assert tenor_bucket(REF, date(2028, 6, 15)) == "3Y-5Y"

    def test_ten_year(self):
        assert tenor_bucket(REF, date(2033, 6, 15)) == "7Y-10Y"

    def test_thirty_plus(self):
        assert tenor_bucket(REF, date(2060, 1, 15)) == "30Y+"


# ---- Book basics ----

class TestBook:
    def test_create_empty(self):
        book = Book("USD_Swaps")
        assert book.name == "USD_Swaps"
        assert len(book) == 0

    def test_add_trades(self):
        book = Book("USD_Vol")
        book.add(Trade(_swn(), trade_id="t1"))
        book.add(Trade(_swn(), trade_id="t2"))
        assert len(book) == 2

    def test_trades_property(self):
        book = Book("USD_Vol")
        t = Trade(_swn(), trade_id="t1")
        book.add(t)
        assert len(book.trades) == 1

    def test_pv(self):
        ctx = _ctx()
        book = Book("USD_Vol")
        book.add(Trade(_swn(), trade_id="t1"))
        pv = book.pv(ctx)
        assert isinstance(pv, float)
        assert pv > 0

    def test_pv_sum_of_trades(self):
        ctx = _ctx()
        t1 = Trade(_swn(end_year=2030), trade_id="t1")
        t2 = Trade(_swn(end_year=2035), trade_id="t2")
        book = Book("USD_Vol")
        book.add(t1)
        book.add(t2)
        assert book.pv(ctx) == pytest.approx(t1.pv(ctx) + t2.pv(ctx))

    def test_pv_by_trade(self):
        ctx = _ctx()
        book = Book("USD_Vol")
        book.add(Trade(_swn(), trade_id="swn1"))
        pvs = book.pv_by_trade(ctx)
        assert pvs[0][0] == "swn1"

    def test_dv01(self):
        ctx = _ctx()
        book = Book("USD_Vol")
        book.add(Trade(_swn(), trade_id="t1"))
        dv01 = book.dv01(ctx)
        assert dv01 != 0.0


# ---- Positions ----

class TestPositions:
    def test_single_trade(self):
        ctx = _ctx()
        book = Book("test")
        book.add(Trade(_swn(notional=10_000_000), trade_id="t1"))
        positions = book.positions(ctx)
        assert len(positions) == 1
        assert positions[0].instrument_type == "Swaption"
        assert positions[0].net_notional == 10_000_000
        assert positions[0].trade_count == 1

    def test_long_short_net(self):
        ctx = _ctx()
        book = Book("test")
        book.add(Trade(_swn(notional=10_000_000), direction=1, trade_id="long"))
        book.add(Trade(_swn(notional=4_000_000), direction=-1, trade_id="short"))
        positions = book.positions(ctx)
        assert len(positions) == 1
        assert positions[0].net_notional == pytest.approx(6_000_000)
        assert positions[0].trade_count == 2

    def test_different_tenors(self):
        ctx = _ctx()
        book = Book("test")
        book.add(Trade(_swn(end_year=2026, notional=5_000_000), trade_id="2y"))
        book.add(Trade(_swn(end_year=2040, notional=5_000_000), trade_id="16y"))
        positions = book.positions(ctx)
        assert len(positions) == 2
        buckets = {p.tenor_bucket for p in positions}
        assert len(buckets) == 2

    def test_counterparty_notional(self):
        book = Book("test")
        book.add(Trade(_swn(notional=10_000_000), counterparty="ACME", trade_id="t1"))
        book.add(Trade(_swn(notional=5_000_000), counterparty="ACME", trade_id="t2"))
        book.add(Trade(_swn(notional=3_000_000), counterparty="BETA", trade_id="t3"))
        cp = book.notional_by_counterparty()
        assert cp["ACME"] == pytest.approx(15_000_000)
        assert cp["BETA"] == pytest.approx(3_000_000)


# ---- Limits ----

class TestLimits:
    def test_no_breach(self):
        ctx = _ctx()
        limits = BookLimits(max_dv01=1_000_000)
        book = Book("test", limits=limits)
        book.add(Trade(_swn(notional=1_000_000), trade_id="t1"))
        breaches = book.check_limits(ctx)
        assert len(breaches) == 0

    def test_dv01_breach(self):
        ctx = _ctx()
        limits = BookLimits(max_dv01=0.001)  # tiny limit
        book = Book("test", limits=limits)
        book.add(Trade(_swn(notional=100_000_000), trade_id="t1"))
        breaches = book.check_limits(ctx)
        assert any(b.limit_type == "dv01" for b in breaches)

    def test_counterparty_breach(self):
        ctx = _ctx()
        limits = BookLimits(max_notional_per_counterparty={"ACME": 5_000_000})
        book = Book("test", limits=limits)
        book.add(Trade(_swn(notional=10_000_000), counterparty="ACME", trade_id="t1"))
        breaches = book.check_limits(ctx)
        assert any(b.limit_type == "counterparty_notional" for b in breaches)
        assert breaches[0].actual_value == pytest.approx(10_000_000)

    def test_counterparty_no_breach(self):
        ctx = _ctx()
        limits = BookLimits(max_notional_per_counterparty={"ACME": 20_000_000})
        book = Book("test", limits=limits)
        book.add(Trade(_swn(notional=10_000_000), counterparty="ACME", trade_id="t1"))
        breaches = book.check_limits(ctx)
        assert len(breaches) == 0

    def test_tenor_dv01_breach(self):
        ctx = _ctx()
        swn = _swn(end_year=2030, notional=100_000_000)
        bucket = tenor_bucket(REF, date(2030, 1, 15))
        limits = BookLimits(tenor_dv01_limits={bucket: 0.001})  # tiny limit
        book = Book("test", limits=limits)
        book.add(Trade(swn, trade_id="t1"))
        breaches = book.check_limits(ctx)
        assert any(b.limit_type == "tenor_dv01" for b in breaches)

    def test_breach_details(self):
        ctx = _ctx()
        limits = BookLimits(max_dv01=0.001)
        book = Book("breach_test", limits=limits)
        book.add(Trade(_swn(notional=50_000_000), trade_id="t1"))
        breaches = book.check_limits(ctx)
        assert len(breaches) >= 1
        b = breaches[0]
        assert b.limit_name == "book:breach_test"
        assert b.actual_value > b.limit_value


# ---- Desk ----

class TestDesk:
    def test_create(self):
        desk = Desk("Rates")
        assert desk.name == "Rates"
        assert len(desk.books) == 0

    def test_add_book(self):
        desk = Desk("Rates")
        desk.add_book(Book("USD_Vol"))
        desk.add_book(Book("EUR_Vol"))
        assert len(desk.books) == 2

    def test_duplicate_book_raises(self):
        desk = Desk("Rates")
        desk.add_book(Book("USD_Vol"))
        with pytest.raises(ValueError, match="already in desk"):
            desk.add_book(Book("USD_Vol"))

    def test_get_book(self):
        desk = Desk("Rates")
        desk.add_book(Book("USD_Vol"))
        book = desk.get_book("USD_Vol")
        assert book.name == "USD_Vol"

    def test_get_missing_raises(self):
        desk = Desk("Rates")
        with pytest.raises(KeyError):
            desk.get_book("missing")

    def test_pv_across_books(self):
        ctx = _ctx()
        b1 = Book("b1")
        b1.add(Trade(_swn(end_year=2030), trade_id="t1"))
        b2 = Book("b2")
        b2.add(Trade(_swn(end_year=2035), trade_id="t2"))
        desk = Desk("Rates")
        desk.add_book(b1)
        desk.add_book(b2)
        assert desk.pv(ctx) == pytest.approx(b1.pv(ctx) + b2.pv(ctx))

    def test_pv_by_book(self):
        ctx = _ctx()
        b1 = Book("USD")
        b1.add(Trade(_swn(), trade_id="t1"))
        desk = Desk("Rates")
        desk.add_book(b1)
        pvs = desk.pv_by_book(ctx)
        assert "USD" in pvs

    def test_dv01(self):
        ctx = _ctx()
        b1 = Book("b1")
        b1.add(Trade(_swn(), trade_id="t1"))
        desk = Desk("Rates")
        desk.add_book(b1)
        assert desk.dv01(ctx) != 0.0

    def test_positions_across_books(self):
        ctx = _ctx()
        b1 = Book("b1")
        b1.add(Trade(_swn(notional=10_000_000), trade_id="t1"))
        b2 = Book("b2")
        b2.add(Trade(_swn(notional=5_000_000), direction=-1, trade_id="t2"))
        desk = Desk("Rates")
        desk.add_book(b1)
        desk.add_book(b2)
        positions = desk.positions(ctx)
        assert len(positions) == 1
        assert positions[0].net_notional == pytest.approx(5_000_000)
        assert positions[0].trade_count == 2

    def test_desk_dv01_breach(self):
        ctx = _ctx()
        b1 = Book("b1")
        b1.add(Trade(_swn(notional=100_000_000), trade_id="t1"))
        desk = Desk("Rates", max_dv01=0.001)
        desk.add_book(b1)
        breaches = desk.check_limits(ctx)
        assert any(b.limit_type == "dv01" and "desk" in b.limit_name for b in breaches)

    def test_desk_propagates_book_breaches(self):
        ctx = _ctx()
        limits = BookLimits(max_dv01=0.001)
        b1 = Book("b1", limits=limits)
        b1.add(Trade(_swn(notional=50_000_000), trade_id="t1"))
        desk = Desk("Rates")
        desk.add_book(b1)
        breaches = desk.check_limits(ctx)
        assert any("book:" in b.limit_name for b in breaches)

    def test_risk_summary(self):
        ctx = _ctx()
        b1 = Book("USD_Vol")
        b1.add(Trade(_swn(), trade_id="t1"))
        desk = Desk("Rates")
        desk.add_book(b1)
        summary = desk.risk_summary(ctx)
        assert summary["desk"] == "Rates"
        assert summary["n_books"] == 1
        assert "total_pv" in summary
        assert "total_dv01" in summary
        assert "positions" in summary
        assert "breaches" in summary
