"""Tests for FX daily P&L."""

import pytest
from datetime import date

from pricebook.fx_book import FXBook
from pricebook.fx_daily_pnl import (
    FXBookAttribution,
    FXDailyPnL,
    attribute_fx_pnl,
    compute_fx_daily_pnl,
)
from pricebook.swaption import Swaption
from pricebook.trade import Trade


PRIOR = date(2024, 1, 15)
CURRENT = date(2024, 1, 16)


def _trade(direction=1, trade_id="t"):
    instr = Swaption(date(2025, 1, 15), date(2030, 1, 15),
                     strike=0.05, notional=1_000_000)
    return Trade(instr, direction=direction, trade_id=trade_id)


def _eur_usd_book():
    book = FXBook("test")
    book.add(_trade(), pair="EUR/USD", notional=10_000_000, spot_rate=1.085)
    return book


# ---- Step 1: official P&L ----

class TestComputeFXDailyPnL:
    def test_empty_book(self):
        book = FXBook("test")
        pnl = compute_fx_daily_pnl(book, {}, {}, PRIOR, CURRENT)
        assert pnl.total_pnl == 0.0

    def test_spot_pnl(self):
        book = _eur_usd_book()
        pnl = compute_fx_daily_pnl(
            book, {"EUR/USD": 1.085}, {"EUR/USD": 1.095},
            PRIOR, CURRENT,
        )
        # 10M × (1.095 − 1.085) = 100,000
        assert pnl.spot_pnl == pytest.approx(100_000)

    def test_spot_negative_when_rate_falls(self):
        book = _eur_usd_book()
        pnl = compute_fx_daily_pnl(
            book, {"EUR/USD": 1.085}, {"EUR/USD": 1.075},
            PRIOR, CURRENT,
        )
        assert pnl.spot_pnl == pytest.approx(-100_000)

    def test_short_flips_sign(self):
        book = FXBook("test")
        book.add(_trade(direction=-1), pair="EUR/USD",
                 notional=10_000_000, spot_rate=1.085)
        pnl = compute_fx_daily_pnl(
            book, {"EUR/USD": 1.085}, {"EUR/USD": 1.095},
            PRIOR, CURRENT,
        )
        assert pnl.spot_pnl == pytest.approx(-100_000)

    def test_carry_pnl(self):
        book = _eur_usd_book()
        pnl = compute_fx_daily_pnl(
            book, {"EUR/USD": 1.085}, {"EUR/USD": 1.085},
            PRIOR, CURRENT,
            carry_points={"EUR/USD": 0.00005},
        )
        # 10M × 0.00005 = 500
        assert pnl.carry_pnl == pytest.approx(500)

    def test_basis_pnl(self):
        book = _eur_usd_book()
        pnl = compute_fx_daily_pnl(
            book, {"EUR/USD": 1.085}, {"EUR/USD": 1.085},
            PRIOR, CURRENT,
            basis_changes={"EUR/USD": -0.0001},
        )
        assert pnl.basis_pnl == pytest.approx(-1_000)

    def test_decomposition_sum(self):
        """Step 1 test: P&L = spot + carry + basis + new + amendments."""
        book = _eur_usd_book()
        pnl = compute_fx_daily_pnl(
            book,
            {"EUR/USD": 1.085}, {"EUR/USD": 1.095},
            PRIOR, CURRENT,
            carry_points={"EUR/USD": 0.00005},
            basis_changes={"EUR/USD": -0.0001},
            new_trade_pnls={"t_new": 5_000},
            amendments={"t1": 1_000},
        )
        assert pnl.total_pnl == pytest.approx(
            pnl.spot_pnl + pnl.carry_pnl + pnl.basis_pnl
            + pnl.new_trade_pnl + pnl.amendment_pnl
        )

    def test_dates_recorded(self):
        pnl = compute_fx_daily_pnl(FXBook("t"), {}, {}, PRIOR, CURRENT)
        assert pnl.prior_date == PRIOR
        assert pnl.current_date == CURRENT


# ---- Step 2: attribution ----

class TestAttributeFXPnL:
    def test_empty_book(self):
        book = FXBook("test")
        attrib = attribute_fx_pnl(book, {}, {}, PRIOR, CURRENT)
        assert attrib.total_pnl == 0.0
        assert attrib.by_pair == []
        assert attrib.by_currency == []

    def test_per_pair_breakdown(self):
        book = FXBook("test")
        book.add(_trade(), pair="EUR/USD", notional=10_000_000, spot_rate=1.085)
        book.add(_trade(), pair="GBP/USD", notional=5_000_000, spot_rate=1.27)
        attrib = attribute_fx_pnl(
            book,
            {"EUR/USD": 1.085, "GBP/USD": 1.27},
            {"EUR/USD": 1.095, "GBP/USD": 1.28},
            PRIOR, CURRENT,
        )
        assert len(attrib.by_pair) == 2
        eur = next(p for p in attrib.by_pair if p.pair == "EUR/USD")
        gbp = next(p for p in attrib.by_pair if p.pair == "GBP/USD")
        assert eur.spot_pnl == pytest.approx(100_000)
        assert gbp.spot_pnl == pytest.approx(50_000)

    def test_attribution_sums_to_total(self):
        """Step 2 test: attribution sums to total."""
        book = FXBook("test")
        book.add(_trade(), pair="EUR/USD", notional=10_000_000, spot_rate=1.085)
        book.add(_trade(), pair="GBP/USD", notional=5_000_000, spot_rate=1.27)
        attrib = attribute_fx_pnl(
            book,
            {"EUR/USD": 1.085, "GBP/USD": 1.27},
            {"EUR/USD": 1.095, "GBP/USD": 1.28},
            PRIOR, CURRENT,
            carry_points={"EUR/USD": 0.00005, "GBP/USD": 0.00003},
        )
        sum_pairs = sum(p.total_pnl for p in attrib.by_pair)
        assert sum_pairs == pytest.approx(attrib.total_pnl)
        sum_ccy = sum(c.pnl for c in attrib.by_currency)
        assert sum_ccy == pytest.approx(attrib.total_pnl)

    def test_per_currency_breakdown(self):
        book = FXBook("test")
        book.add(_trade(), pair="EUR/USD", notional=10_000_000, spot_rate=1.085)
        attrib = attribute_fx_pnl(
            book,
            {"EUR/USD": 1.085}, {"EUR/USD": 1.095},
            PRIOR, CURRENT,
        )
        ccy_map = {c.currency: c.pnl for c in attrib.by_currency}
        assert "EUR" in ccy_map
        assert "USD" in ccy_map
        # Total across currencies = total P&L
        assert sum(ccy_map.values()) == pytest.approx(attrib.total_pnl)
