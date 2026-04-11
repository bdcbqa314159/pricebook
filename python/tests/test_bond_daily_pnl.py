"""Tests for bond daily P&L and attribution."""

import pytest
from datetime import date

from pricebook.bond_book import BondBook, BondTradeEntry
from pricebook.bond_daily_pnl import (
    BondBookAttribution,
    BondDailyPnL,
    BondTradeAttribution,
    attribute_bond_pnl,
    compute_bond_daily_pnl,
)
from pricebook.swaption import Swaption
from pricebook.trade import Trade


PRIOR = date(2024, 1, 15)
CURRENT = date(2024, 1, 16)


def _instr():
    return Swaption(date(2025, 1, 15), date(2030, 1, 15),
                    strike=0.05, notional=1_000_000)


def _trade(direction=1, trade_id="t"):
    return Trade(_instr(), direction=direction, trade_id=trade_id)


def _book():
    book = BondBook("test", PRIOR)
    book.add(_trade(trade_id="t1"), issuer="UST", sector="govt",
             face_amount=10_000_000, dirty_price=98.5, coupon_rate=0.04,
             maturity=date(2034, 1, 15), dv01_per_million=85.0,
             duration=8.2)
    return book


# ---- Step 1: official P&L ----

class TestComputeBondDailyPnL:
    def test_empty_book(self):
        book = BondBook("test", PRIOR)
        pnl = compute_bond_daily_pnl(book, {}, {}, PRIOR, CURRENT)
        assert pnl.total_pnl == 0.0
        assert pnl.mtm_pnl == 0.0

    def test_mtm_from_price_change(self):
        book = _book()
        prior = {"UST": 98.5}
        current = {"UST": 99.0}
        pnl = compute_bond_daily_pnl(book, prior, current, PRIOR, CURRENT)
        # 10M × (99.0 - 98.5) / 100 = 50,000
        assert pnl.mtm_pnl == pytest.approx(50_000)

    def test_mtm_negative_when_price_falls(self):
        book = _book()
        prior = {"UST": 98.5}
        current = {"UST": 97.5}
        pnl = compute_bond_daily_pnl(book, prior, current, PRIOR, CURRENT)
        assert pnl.mtm_pnl == pytest.approx(-100_000)

    def test_short_flips_sign(self):
        book = BondBook("test", PRIOR)
        book.add(_trade(direction=-1), issuer="UST", face_amount=10_000_000,
                 dirty_price=98.5, coupon_rate=0.04)
        prior = {"UST": 98.5}
        current = {"UST": 99.0}
        pnl = compute_bond_daily_pnl(book, prior, current, PRIOR, CURRENT)
        assert pnl.mtm_pnl == pytest.approx(-50_000)

    def test_accrual_pnl(self):
        book = _book()
        pnl = compute_bond_daily_pnl(
            book, {"UST": 98.5}, {"UST": 98.5}, PRIOR, CURRENT,
        )
        # 10M × 0.04 × 1/365
        assert pnl.accrual_pnl == pytest.approx(10_000_000 * 0.04 / 365.0)

    def test_new_trades(self):
        book = BondBook("test", PRIOR)
        new = BondTradeEntry(
            trade=_trade(trade_id="new"), issuer="UST",
            face_amount=5_000_000, dirty_price=98.0,
        )
        pnl = compute_bond_daily_pnl(
            book, {}, {"UST": 99.0}, PRIOR, CURRENT,
            new_trades=[new],
        )
        # 5M × (99.0 - 98.0) / 100 = 50,000
        assert pnl.new_trade_pnl == pytest.approx(50_000)

    def test_amendments(self):
        book = _book()
        pnl = compute_bond_daily_pnl(
            book, {"UST": 98.5}, {"UST": 98.5}, PRIOR, CURRENT,
            amendments={"t1": 1_234.0},
        )
        assert pnl.amendment_pnl == pytest.approx(1_234.0)

    def test_decomposition_sum(self):
        """Step 1 test: P&L = MTM + accrual + new + amendments."""
        book = _book()
        new = BondTradeEntry(
            trade=_trade(trade_id="new"), issuer="UST",
            face_amount=5_000_000, dirty_price=98.0,
        )
        pnl = compute_bond_daily_pnl(
            book, {"UST": 98.5}, {"UST": 99.0}, PRIOR, CURRENT,
            new_trades=[new], amendments={"t1": 500.0},
        )
        assert pnl.total_pnl == pytest.approx(
            pnl.mtm_pnl + pnl.accrual_pnl + pnl.new_trade_pnl + pnl.amendment_pnl
        )

    def test_dates_recorded(self):
        pnl = compute_bond_daily_pnl(BondBook("t", PRIOR), {}, {}, PRIOR, CURRENT)
        assert pnl.prior_date == PRIOR
        assert pnl.current_date == CURRENT


# ---- Step 2: attribution ----

class TestAttributeBondPnL:
    def test_empty_book(self):
        book = BondBook("test", PRIOR)
        attrib = attribute_bond_pnl(book, {}, {}, PRIOR, CURRENT)
        assert attrib.total_pnl == 0.0
        assert attrib.by_trade == []

    def test_carry_coupon_only(self):
        book = _book()
        attrib = attribute_bond_pnl(
            book, {"UST": 98.5}, {"UST": 98.5}, PRIOR, CURRENT,
        )
        # No price change → total ≈ 0, carry = coupon income
        expected_carry = 10_000_000 * 0.04 / 365.0
        assert attrib.carry_pnl == pytest.approx(expected_carry)

    def test_carry_net_of_financing(self):
        book = _book()
        attrib = attribute_bond_pnl(
            book, {"UST": 98.5}, {"UST": 98.5}, PRIOR, CURRENT,
            financing_rates={"UST": 0.05},
        )
        coupon = 10_000_000 * 0.04 / 365.0
        financing = 10_000_000 * (98.5 / 100.0) * 0.05 / 365.0
        assert attrib.carry_pnl == pytest.approx(coupon - financing)

    def test_rolldown(self):
        book = _book()
        attrib = attribute_bond_pnl(
            book, {"UST": 98.5}, {"UST": 98.5}, PRIOR, CURRENT,
            rolldown_prices={"UST": 98.6},
        )
        # Roll-down = 10M × (98.6 - 98.5) / 100 = 10,000
        assert attrib.rolldown_pnl == pytest.approx(10_000)

    def test_curve_move(self):
        book = _book()
        attrib = attribute_bond_pnl(
            book, {"UST": 98.5}, {"UST": 99.0}, PRIOR, CURRENT,
            parallel_shift=5.0,
        )
        # DV01 = 10M × 85/1M = 850
        # curve_pnl = 850 × 5 = 4250
        assert attrib.curve_pnl == pytest.approx(4_250)

    def test_spread_move(self):
        book = _book()
        attrib = attribute_bond_pnl(
            book, {"UST": 98.5}, {"UST": 98.5}, PRIOR, CURRENT,
            spread_changes={"UST": -3.0},
        )
        # spread_pnl = DV01 × spread = 850 × (-3) = -2550
        assert attrib.spread_pnl == pytest.approx(-2_550)

    def test_attribution_sums_to_total(self):
        """Step 2 test: carry + rolldown + curve + spread + unexplained = total."""
        book = _book()
        attrib = attribute_bond_pnl(
            book, {"UST": 98.5}, {"UST": 99.5}, PRIOR, CURRENT,
            parallel_shift=10.0,
            spread_changes={"UST": 2.0},
            rolldown_prices={"UST": 98.55},
            financing_rates={"UST": 0.045},
        )
        assert attrib.total_pnl == pytest.approx(
            attrib.carry_pnl + attrib.rolldown_pnl
            + attrib.curve_pnl + attrib.spread_pnl + attrib.unexplained
        )

    def test_per_issuer_breakdown(self):
        book = BondBook("test", PRIOR)
        book.add(_trade(trade_id="t1"), issuer="UST", sector="govt",
                 face_amount=10_000_000, dirty_price=98.5,
                 dv01_per_million=85.0, coupon_rate=0.04)
        book.add(_trade(trade_id="t2"), issuer="DBR", sector="govt",
                 face_amount=5_000_000, dirty_price=101.0,
                 dv01_per_million=70.0, coupon_rate=0.02)
        attrib = attribute_bond_pnl(
            book, {"UST": 98.5, "DBR": 101.0},
            {"UST": 99.0, "DBR": 101.5}, PRIOR, CURRENT,
        )
        assert "UST" in attrib.by_issuer
        assert "DBR" in attrib.by_issuer
        sum_issuer = sum(d["total_pnl"] for d in attrib.by_issuer.values())
        assert sum_issuer == pytest.approx(attrib.total_pnl)

    def test_per_tenor_breakdown(self):
        book = BondBook("test", PRIOR)
        book.add(_trade(), issuer="UST_2Y", face_amount=10_000_000,
                 dirty_price=99.0, maturity=date(2025, 7, 15))  # ≤2Y → 1-2Y
        book.add(_trade(), issuer="UST_10Y", face_amount=10_000_000,
                 dirty_price=98.0, maturity=date(2032, 1, 15))  # ~8Y → 7-10Y
        attrib = attribute_bond_pnl(
            book, {"UST_2Y": 99.0, "UST_10Y": 98.0},
            {"UST_2Y": 99.2, "UST_10Y": 98.5}, PRIOR, CURRENT,
        )
        assert len(attrib.by_tenor) == 2
        sum_tenor = sum(d["total_pnl"] for d in attrib.by_tenor.values())
        assert sum_tenor == pytest.approx(attrib.total_pnl)

    def test_explained_property(self):
        book = _book()
        attrib = attribute_bond_pnl(
            book, {"UST": 98.5}, {"UST": 99.0}, PRIOR, CURRENT,
            parallel_shift=5.0,
        )
        a = attrib.by_trade[0]
        assert a.explained + a.unexplained == pytest.approx(a.total_pnl)
