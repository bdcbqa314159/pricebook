"""Tests for daily P&L workflow and attribution."""

import pytest
from datetime import date

from pricebook.daily_pnl import (
    DailyPnL, compute_daily_pnl, compute_desk_pnl,
    TradeAttribution, BookAttribution, attribute_pnl,
)
from pricebook.book import Book, Desk
from pricebook.trade import Trade
from pricebook.swaption import Swaption
from pricebook.pricing_context import PricingContext


REF = date(2024, 1, 15)
NEXT = date(2024, 1, 16)


def _ctx(val_date=REF, rate=0.05, vol=0.20):
    return PricingContext.simple(val_date, rate=rate, vol=vol)


def _swn(expiry_year=2025, end_year=2030, strike=0.05, notional=1_000_000):
    return Swaption(
        date(expiry_year, 1, 15), date(end_year, 1, 15),
        strike=strike, notional=notional,
    )


def _book_with_trades():
    book = Book("USD_Vol")
    book.add(Trade(_swn(strike=0.05), trade_id="swn1"))
    book.add(Trade(_swn(strike=0.04), trade_id="swn2"))
    return book


# ---- Official P&L ----

class TestDailyPnL:
    def test_decomposition_sums(self):
        book = _book_with_trades()
        prior = _ctx(REF, rate=0.05)
        current = _ctx(NEXT, rate=0.05)
        result = compute_daily_pnl(book, prior, current)
        assert result.total_pnl == pytest.approx(
            result.market_move_pnl + result.new_trade_pnl + result.amendment_pnl
        )

    def test_market_move_pnl(self):
        book = _book_with_trades()
        prior = _ctx(REF, rate=0.05)
        current = _ctx(NEXT, rate=0.06)  # rates up 100bp
        result = compute_daily_pnl(book, prior, current)
        assert result.market_move_pnl == pytest.approx(
            book.pv(current) - book.pv(prior)
        )

    def test_new_trade_pnl(self):
        book = _book_with_trades()
        prior = _ctx(REF)
        current = _ctx(NEXT)
        new_trade = Trade(_swn(strike=0.03), trade_id="new1")
        result = compute_daily_pnl(book, prior, current, new_trades=[new_trade])
        assert result.new_trade_pnl == pytest.approx(new_trade.pv(current))

    def test_amendment_pnl(self):
        book = _book_with_trades()
        prior = _ctx(REF)
        current = _ctx(NEXT)
        amendments = {"swn1": 500.0, "swn2": -200.0}
        result = compute_daily_pnl(book, prior, current, amendments=amendments)
        assert result.amendment_pnl == pytest.approx(300.0)

    def test_no_market_move(self):
        """Same curves → market move P&L is only from date change."""
        book = _book_with_trades()
        ctx = _ctx(REF)
        result = compute_daily_pnl(book, ctx, ctx)
        assert result.market_move_pnl == pytest.approx(0.0)
        assert result.total_pnl == pytest.approx(0.0)

    def test_metadata(self):
        book = _book_with_trades()
        prior = _ctx(REF)
        current = _ctx(NEXT)
        result = compute_daily_pnl(book, prior, current)
        assert result.book_name == "USD_Vol"
        assert result.prior_date == REF
        assert result.current_date == NEXT

    def test_prior_pv(self):
        book = _book_with_trades()
        prior = _ctx(REF)
        current = _ctx(NEXT)
        result = compute_daily_pnl(book, prior, current)
        assert result.prior_pv == pytest.approx(book.pv(prior))


# ---- Desk P&L ----

class TestDeskPnL:
    def test_desk_pnl(self):
        b1 = Book("b1")
        b1.add(Trade(_swn(strike=0.05), trade_id="t1"))
        b2 = Book("b2")
        b2.add(Trade(_swn(strike=0.04), trade_id="t2"))
        desk = Desk("Rates")
        desk.add_book(b1)
        desk.add_book(b2)

        prior = _ctx(REF)
        current = _ctx(NEXT, rate=0.06)
        results = compute_desk_pnl(desk, prior, current)
        assert len(results) == 2
        total = sum(r.total_pnl for r in results)
        assert total == pytest.approx(desk.pv(current) - desk.pv(prior))


# ---- P&L Attribution ----

class TestAttribution:
    def test_components_sum_to_total(self):
        book = _book_with_trades()
        prior = _ctx(REF, rate=0.05)
        current = _ctx(NEXT, rate=0.06, vol=0.22)
        attrib = attribute_pnl(book, prior, current)
        assert attrib.total_pnl == pytest.approx(
            attrib.rate_pnl + attrib.vol_pnl + attrib.theta_pnl + attrib.unexplained
        )

    def test_per_trade_sum(self):
        book = _book_with_trades()
        prior = _ctx(REF, rate=0.05)
        current = _ctx(NEXT, rate=0.06)
        attrib = attribute_pnl(book, prior, current)
        assert len(attrib.by_trade) == 2
        assert sum(a.total_pnl for a in attrib.by_trade) == pytest.approx(attrib.total_pnl)

    def test_rate_move_dominates(self):
        """A big rate move should produce nonzero rate P&L."""
        book = Book("test")
        book.add(Trade(_swn(notional=10_000_000), trade_id="big"))
        prior = _ctx(REF, rate=0.05)
        current = _ctx(REF, rate=0.08)  # +300bp, same date/vol
        attrib = attribute_pnl(book, prior, current)
        assert attrib.rate_pnl != 0.0
        # Rate should be the dominant component
        assert abs(attrib.rate_pnl) > abs(attrib.vol_pnl)

    def test_vol_move_detected(self):
        """A vol move should show up in vol_pnl."""
        book = Book("test")
        book.add(Trade(_swn(notional=10_000_000), trade_id="vega"))
        prior = _ctx(REF, rate=0.05, vol=0.20)
        current = _ctx(REF, rate=0.05, vol=0.30)  # +10 vol pts, same rates/date
        attrib = attribute_pnl(book, prior, current)
        assert attrib.vol_pnl != 0.0
        assert abs(attrib.vol_pnl) > abs(attrib.rate_pnl)

    def test_no_move_zero_pnl(self):
        """Same context → zero P&L."""
        book = _book_with_trades()
        ctx = _ctx(REF)
        attrib = attribute_pnl(book, ctx, ctx)
        assert attrib.total_pnl == pytest.approx(0.0)
        assert attrib.rate_pnl == pytest.approx(0.0)
        assert attrib.vol_pnl == pytest.approx(0.0)

    def test_bucket_attribution(self):
        book = _book_with_trades()
        prior = _ctx(REF, rate=0.05)
        current = _ctx(NEXT, rate=0.06)
        attrib = attribute_pnl(book, prior, current)
        assert len(attrib.by_bucket) >= 1
        # Bucket totals should sum to book total
        bucket_total = sum(b["total_pnl"] for b in attrib.by_bucket.values())
        assert bucket_total == pytest.approx(attrib.total_pnl)

    def test_per_trade_explained(self):
        book = Book("test")
        book.add(Trade(_swn(), trade_id="t1"))
        prior = _ctx(REF, rate=0.05, vol=0.20)
        current = _ctx(NEXT, rate=0.06, vol=0.22)
        attrib = attribute_pnl(book, prior, current)
        ta = attrib.by_trade[0]
        assert ta.total_pnl == pytest.approx(ta.explained + ta.unexplained)

    def test_trade_ids(self):
        book = _book_with_trades()
        prior = _ctx(REF)
        current = _ctx(NEXT, rate=0.06)
        attrib = attribute_pnl(book, prior, current)
        ids = {a.trade_id for a in attrib.by_trade}
        assert "swn1" in ids
        assert "swn2" in ids
