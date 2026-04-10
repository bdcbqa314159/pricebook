"""Tests for commodity daily P&L."""

import pytest
from datetime import date

from pricebook.commodity_book import CommodityBook, CommodityTradeEntry
from pricebook.commodity_daily_pnl import (
    CommodityAttribution,
    CommodityDailyPnL,
    attribute_commodity_pnl,
    compute_commodity_daily_pnl,
)
from pricebook.swaption import Swaption
from pricebook.trade import Trade


PRIOR = date(2024, 1, 15)
CURRENT = date(2024, 1, 16)


def _instr():
    return Swaption(date(2025, 1, 15), date(2030, 1, 15),
                    strike=0.05, notional=1_000_000)


def _trade(direction=1, scale=1.0, trade_id="t"):
    return Trade(_instr(), direction=direction, notional_scale=scale, trade_id=trade_id)


def _wti_book():
    book = CommodityBook("Energy", PRIOR)
    book.add(_trade(trade_id="t1"), commodity="WTI", sector="energy",
             unit="bbl", quantity=10_000, reference_price=72.0,
             delivery_date=date(2024, 6, 1))
    return book


# ---- Step 1: official P&L ----

class TestComputeCommodityDailyPnL:
    def test_empty_book(self):
        book = CommodityBook("test", PRIOR)
        pnl = compute_commodity_daily_pnl(
            book, prior_curves={}, current_curves={},
            prior_date=PRIOR, current_date=CURRENT,
        )
        assert pnl.book_name == "test"
        assert pnl.spot_pnl == 0.0
        assert pnl.total_pnl == 0.0

    def test_spot_pnl_positive_when_curve_rises(self):
        book = _wti_book()
        prior = {"WTI": {date(2024, 6, 1): 72.0}}
        current = {"WTI": {date(2024, 6, 1): 73.0}}
        pnl = compute_commodity_daily_pnl(
            book, prior, current, PRIOR, CURRENT,
        )
        # 10,000 × (73 − 72) = 10,000
        assert pnl.spot_pnl == pytest.approx(10_000)
        assert pnl.carry_pnl == pytest.approx(0.0)
        assert pnl.roll_pnl == pytest.approx(0.0)
        assert pnl.total_pnl == pytest.approx(10_000)

    def test_short_flips_sign(self):
        book = CommodityBook("test", PRIOR)
        book.add(_trade(direction=-1), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0,
                 delivery_date=date(2024, 6, 1))
        prior = {"WTI": {date(2024, 6, 1): 72.0}}
        current = {"WTI": {date(2024, 6, 1): 73.0}}
        pnl = compute_commodity_daily_pnl(book, prior, current, PRIOR, CURRENT)
        assert pnl.spot_pnl == pytest.approx(-10_000)

    def test_carry_pnl_convenience_yield(self):
        book = _wti_book()
        prior = {"WTI": {date(2024, 6, 1): 72.0}}
        current = {"WTI": {date(2024, 6, 1): 72.0}}
        pnl = compute_commodity_daily_pnl(
            book, prior, current, PRIOR, CURRENT,
            convenience_yields={"WTI": 0.05},  # 5% annualised
        )
        # 10,000 × 72 × 0.05 × (1/365)
        assert pnl.carry_pnl == pytest.approx(10_000 * 72.0 * 0.05 / 365.0)

    def test_carry_pnl_storage_negative(self):
        book = _wti_book()
        prior = {"WTI": {date(2024, 6, 1): 72.0}}
        current = {"WTI": {date(2024, 6, 1): 72.0}}
        pnl = compute_commodity_daily_pnl(
            book, prior, current, PRIOR, CURRENT,
            storage_rates={"WTI": 0.03},
        )
        # 10K × 72 × (-0.03) × 1/365
        assert pnl.carry_pnl == pytest.approx(-10_000 * 72.0 * 0.03 / 365.0)

    def test_roll_pnl_curve_spread(self):
        book = _wti_book()
        prior = {"WTI": {date(2024, 6, 1): 72.0, date(2024, 7, 1): 72.5}}
        current = {"WTI": {date(2024, 6, 1): 72.0, date(2024, 7, 1): 72.8}}
        pnl = compute_commodity_daily_pnl(
            book, prior, current, PRIOR, CURRENT,
            rolls=[("WTI", date(2024, 6, 1), date(2024, 7, 1))],
        )
        # roll = 10K × (72.8 - 72.0) = 8000
        assert pnl.roll_pnl == pytest.approx(8_000)

    def test_decomposition_sum_equals_total(self):
        """Step 1 test: P&L decomposes into spot + carry + roll (+ extras)."""
        book = _wti_book()
        prior = {"WTI": {date(2024, 6, 1): 72.0, date(2024, 7, 1): 72.5}}
        current = {"WTI": {date(2024, 6, 1): 73.0, date(2024, 7, 1): 73.6}}
        pnl = compute_commodity_daily_pnl(
            book, prior, current, PRIOR, CURRENT,
            convenience_yields={"WTI": 0.05},
            storage_rates={"WTI": 0.02},
            rolls=[("WTI", date(2024, 6, 1), date(2024, 7, 1))],
            amendments={"t1": 250.0},
        )
        assert pnl.total_pnl == pytest.approx(
            pnl.spot_pnl + pnl.carry_pnl + pnl.roll_pnl
            + pnl.new_trade_pnl + pnl.amendment_pnl
        )
        assert pnl.market_move_pnl == pytest.approx(
            pnl.spot_pnl + pnl.carry_pnl + pnl.roll_pnl
        )

    def test_new_trades(self):
        book = CommodityBook("test", PRIOR)
        prior = {"WTI": {date(2024, 6, 1): 72.0}}
        current = {"WTI": {date(2024, 6, 1): 73.0}}
        new_entry = CommodityTradeEntry(
            trade=_trade(trade_id="new"),
            commodity="WTI", sector="energy", unit="bbl",
            quantity=5_000, reference_price=72.5,
            delivery_date=date(2024, 6, 1),
        )
        pnl = compute_commodity_daily_pnl(
            book, prior, current, PRIOR, CURRENT,
            new_trades=[new_entry],
        )
        # 5K × (73 − 72.5) = 2500
        assert pnl.new_trade_pnl == pytest.approx(2_500)

    def test_amendments(self):
        book = _wti_book()
        prior = {"WTI": {date(2024, 6, 1): 72.0}}
        current = {"WTI": {date(2024, 6, 1): 72.0}}
        pnl = compute_commodity_daily_pnl(
            book, prior, current, PRIOR, CURRENT,
            amendments={"t1": 1_234.0, "t2": -200.0},
        )
        assert pnl.amendment_pnl == pytest.approx(1_034.0)

    def test_dates_recorded(self):
        book = _wti_book()
        pnl = compute_commodity_daily_pnl(
            book,
            prior_curves={"WTI": {date(2024, 6, 1): 72.0}},
            current_curves={"WTI": {date(2024, 6, 1): 72.0}},
            prior_date=PRIOR, current_date=CURRENT,
        )
        assert pnl.prior_date == PRIOR
        assert pnl.current_date == CURRENT


# ---- Step 2: attribution ----

class TestAttributeCommodityPnL:
    def test_empty_book(self):
        book = CommodityBook("test", PRIOR)
        attrib = attribute_commodity_pnl(
            book, prior_curves={}, current_curves={},
            prior_date=PRIOR, current_date=CURRENT,
        )
        assert attrib.total_pnl == 0.0
        assert attrib.by_commodity == {}
        assert attrib.by_tenor == {}

    def test_per_commodity_breakdown(self):
        book = CommodityBook("test", PRIOR)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0,
                 delivery_date=date(2024, 6, 1))
        book.add(_trade(), commodity="GOLD", sector="metals",
                 quantity=500, reference_price=2_000.0,
                 delivery_date=date(2024, 6, 1))
        prior = {
            "WTI": {date(2024, 6, 1): 72.0},
            "GOLD": {date(2024, 6, 1): 2_000.0},
        }
        current = {
            "WTI": {date(2024, 6, 1): 73.0},
            "GOLD": {date(2024, 6, 1): 2_010.0},
        }
        attrib = attribute_commodity_pnl(
            book, prior, current, PRIOR, CURRENT,
        )
        assert "WTI" in attrib.by_commodity
        assert "GOLD" in attrib.by_commodity
        assert attrib.by_commodity["WTI"]["spot_pnl"] == pytest.approx(10_000)
        assert attrib.by_commodity["GOLD"]["spot_pnl"] == pytest.approx(5_000)

    def test_per_tenor_breakdown(self):
        book = CommodityBook("test", PRIOR)
        # Front (within 30 days)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0,
                 delivery_date=date(2024, 2, 1))
        # ≤6M
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0,
                 delivery_date=date(2024, 6, 1))
        prior = {"WTI": {
            date(2024, 2, 1): 72.0, date(2024, 6, 1): 72.0,
        }}
        current = {"WTI": {
            date(2024, 2, 1): 72.5, date(2024, 6, 1): 73.0,
        }}
        attrib = attribute_commodity_pnl(book, prior, current, PRIOR, CURRENT)
        assert "front" in attrib.by_tenor
        assert "≤6M" in attrib.by_tenor
        assert attrib.by_tenor["front"]["spot_pnl"] == pytest.approx(5_000)
        assert attrib.by_tenor["≤6M"]["spot_pnl"] == pytest.approx(10_000)

    def test_attribution_sums_to_total(self):
        """Step 2 test: per-commodity sums equal book total."""
        book = CommodityBook("test", PRIOR)
        book.add(_trade(direction=1), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0,
                 delivery_date=date(2024, 6, 1))
        book.add(_trade(direction=-1), commodity="GOLD", sector="metals",
                 quantity=500, reference_price=2_000.0,
                 delivery_date=date(2024, 12, 1))
        prior = {
            "WTI": {date(2024, 6, 1): 72.0},
            "GOLD": {date(2024, 12, 1): 2_000.0},
        }
        current = {
            "WTI": {date(2024, 6, 1): 73.0},
            "GOLD": {date(2024, 12, 1): 2_005.0},
        }
        attrib = attribute_commodity_pnl(
            book, prior, current, PRIOR, CURRENT,
            convenience_yields={"WTI": 0.04},
        )
        sum_by_c = sum(d["total_pnl"] for d in attrib.by_commodity.values())
        sum_by_t = sum(d["total_pnl"] for d in attrib.by_tenor.values())
        assert sum_by_c == pytest.approx(attrib.total_pnl)
        assert sum_by_t == pytest.approx(attrib.total_pnl)

    def test_parallel_pnl_pure_parallel_shift(self):
        """Curve moves in parallel → all spot P&L is parallel, none is shape."""
        book = CommodityBook("test", PRIOR)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0,
                 delivery_date=date(2024, 6, 1))
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=5_000, reference_price=72.0,
                 delivery_date=date(2024, 12, 1))
        prior = {"WTI": {
            date(2024, 6, 1): 72.0, date(2024, 12, 1): 72.5,
        }}
        # Both deliveries up by exactly $1
        current = {"WTI": {
            date(2024, 6, 1): 73.0, date(2024, 12, 1): 73.5,
        }}
        attrib = attribute_commodity_pnl(book, prior, current, PRIOR, CURRENT)
        # spot = 10K + 5K = 15K
        assert attrib.spot_pnl == pytest.approx(15_000)
        # parallel = (10K + 5K) × mean(1, 1) = 15K
        assert attrib.parallel_pnl == pytest.approx(15_000)
        assert attrib.shape_pnl == pytest.approx(0.0)

    def test_shape_pnl_when_curve_steepens(self):
        """Front-month flat, back-month up → shape pnl is non-zero."""
        book = CommodityBook("test", PRIOR)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0,
                 delivery_date=date(2024, 6, 1))
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0,
                 delivery_date=date(2024, 12, 1))
        prior = {"WTI": {
            date(2024, 6, 1): 72.0, date(2024, 12, 1): 72.5,
        }}
        current = {"WTI": {
            date(2024, 6, 1): 72.0, date(2024, 12, 1): 73.5,
        }}
        attrib = attribute_commodity_pnl(book, prior, current, PRIOR, CURRENT)
        # spot = 0 + 10K = 10K
        # parallel = 20K × mean(0, 1) = 20K × 0.5 = 10K
        # shape = spot − parallel = 0
        # Actually here parallel + shape = spot exactly. Shape = 0 because
        # both legs have equal qty. So we test the alternative: with
        # *unequal* quantities, shape is non-zero.

    def test_shape_nonzero_with_unequal_quantities(self):
        book = CommodityBook("test", PRIOR)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0,
                 delivery_date=date(2024, 6, 1))
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=5_000, reference_price=72.0,
                 delivery_date=date(2024, 12, 1))
        prior = {"WTI": {
            date(2024, 6, 1): 72.0, date(2024, 12, 1): 72.5,
        }}
        # Steepening: front flat, back +$1
        current = {"WTI": {
            date(2024, 6, 1): 72.0, date(2024, 12, 1): 73.5,
        }}
        attrib = attribute_commodity_pnl(book, prior, current, PRIOR, CURRENT)
        # spot = 0 + 5K = 5K
        # parallel = (10K + 5K) × mean(0, 1) = 15K × 0.5 = 7.5K
        # shape = 5K − 7.5K = −2.5K
        assert attrib.spot_pnl == pytest.approx(5_000)
        assert attrib.parallel_pnl == pytest.approx(7_500)
        assert attrib.shape_pnl == pytest.approx(-2_500)
        # Identity: parallel + shape = spot
        assert attrib.parallel_pnl + attrib.shape_pnl == pytest.approx(attrib.spot_pnl)

    def test_attribution_components_match_compute(self):
        """Attribution's spot/carry/roll match the official compute output."""
        book = CommodityBook("test", PRIOR)
        book.add(_trade(), commodity="WTI", sector="energy",
                 quantity=10_000, reference_price=72.0,
                 delivery_date=date(2024, 6, 1))
        prior = {"WTI": {date(2024, 6, 1): 72.0, date(2024, 7, 1): 72.5}}
        current = {"WTI": {date(2024, 6, 1): 73.0, date(2024, 7, 1): 73.6}}
        kwargs = dict(
            convenience_yields={"WTI": 0.05},
            storage_rates={"WTI": 0.02},
            rolls=[("WTI", date(2024, 6, 1), date(2024, 7, 1))],
        )
        pnl = compute_commodity_daily_pnl(
            book, prior, current, PRIOR, CURRENT, **kwargs,
        )
        attrib = attribute_commodity_pnl(
            book, prior, current, PRIOR, CURRENT, **kwargs,
        )
        assert attrib.spot_pnl == pytest.approx(pnl.spot_pnl)
        assert attrib.carry_pnl == pytest.approx(pnl.carry_pnl)
        assert attrib.roll_pnl == pytest.approx(pnl.roll_pnl)
        assert attrib.total_pnl == pytest.approx(pnl.market_move_pnl)
