"""Tests for repo trading gaps: unified RepoTrade, lifecycle, margin, collateral, P&L."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.repo_desk import (
    RepoTrade, RepoBook, RepoTradeEntry,
    CollateralPool, CollateralPosition,
    repo_daily_pnl, RepoDailyPnL,
)
from pricebook.serialisable import from_dict
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)


def _make_trade(**kwargs):
    defaults = dict(
        counterparty="BankA", collateral_issuer="UST10Y",
        face_amount=50_000_000, bond_price=102.0, repo_rate=0.045,
        term_days=30, coupon_rate=0.04125, direction="repo",
        start_date=REF, haircut=0.02,
    )
    defaults.update(kwargs)
    return RepoTrade(**defaults)


# ── Gap 1 + 8: Unified object with PV ──

class TestUnifiedRepoTrade:

    def test_pv_repo(self):
        """Repo direction: PV = df × repurchase - cash_lent."""
        trade = _make_trade()
        curve = make_flat_curve(REF, 0.04)
        pv = trade.pv(curve, REF)
        assert math.isfinite(pv)

    def test_pv_reverse(self):
        trade = _make_trade(direction="reverse")
        curve = make_flat_curve(REF, 0.04)
        pv = trade.pv(curve, REF)
        assert math.isfinite(pv)

    def test_pv_ctx(self):
        """pv_ctx works for Trade/Portfolio integration."""
        trade = _make_trade()
        curve = make_flat_curve(REF, 0.04)
        ctx = type("Ctx", (), {"discount_curve": curve})()
        pv = trade.pv_ctx(ctx)
        assert pv == trade.pv(curve, REF)

    def test_cash_with_haircut(self):
        trade = _make_trade(haircut=0.02)
        # cash = face × price / 100 × (1 - haircut)
        expected = 50_000_000 * 102.0 / 100.0 * 0.98
        assert trade.cash_amount == pytest.approx(expected)

    def test_effective_rate_higher(self):
        trade = _make_trade(haircut=0.02, repo_rate=0.045)
        assert trade.effective_rate > trade.repo_rate

    def test_carry(self):
        trade = _make_trade()
        assert math.isfinite(trade.carry)

    def test_from_entry_compat(self):
        """Convert old RepoTradeEntry to new RepoTrade."""
        entry = RepoTradeEntry(
            counterparty="X", collateral_issuer="UST10Y",
            face_amount=10_000_000, bond_price=100.0, repo_rate=0.04,
            term_days=30, direction="repo",
        )
        trade = RepoTrade.from_entry(entry)
        assert trade.counterparty == "X"
        assert trade.repo_rate == 0.04


# ── Gap 2: Roll mechanics ──

class TestRollMechanics:

    def test_roll_creates_new(self):
        trade = _make_trade()
        new_trade = trade.roll(new_rate=0.048, new_term_days=30)
        assert new_trade.status == "live"
        assert new_trade.repo_rate == 0.048
        assert trade.status == "rolled"

    def test_roll_preserves_collateral(self):
        trade = _make_trade()
        new_trade = trade.roll(0.05, 60)
        assert new_trade.collateral_issuer == trade.collateral_issuer
        assert new_trade.face_amount == trade.face_amount
        assert new_trade.counterparty == trade.counterparty

    def test_roll_cost(self):
        trade = _make_trade(repo_rate=0.045, term_days=30)
        cost = trade.roll_cost(new_rate=0.050, new_term_days=30)
        assert cost > 0  # higher rate → more expensive

    def test_roll_cost_cheaper(self):
        trade = _make_trade(repo_rate=0.045, term_days=30)
        cost = trade.roll_cost(new_rate=0.040, new_term_days=30)
        assert cost < 0  # cheaper to roll


# ── Gap 3: Variation margin ──

class TestVariationMargin:

    def test_margin_required(self):
        trade = _make_trade(haircut=0.02)
        margin = trade.margin_required(102.0)
        expected = 50_000_000 * 102.0 / 100 * 0.02
        assert margin == pytest.approx(expected)

    def test_margin_call_price_drop(self):
        """Bond drops → more margin needed."""
        trade = _make_trade(bond_price=102.0, haircut=0.02)
        call = trade.margin_call(100.0)
        # Lower price → lower collateral value → margin call negative
        # (counterparty needs less margin, you get some back)
        assert call < 0

    def test_margin_call_price_rise(self):
        trade = _make_trade(bond_price=102.0, haircut=0.02)
        call = trade.margin_call(104.0)
        assert call > 0  # more collateral value → more margin posted

    def test_variation_margin_repo_direction(self):
        trade = _make_trade(direction="repo", bond_price=102.0)
        vm = trade.variation_margin(100.0)
        # Bond drops: repo seller (you) benefits, VM positive for you
        assert vm > 0


# ── Gap 4: Collateral pool ──

class TestCollateralPool:

    def test_add_inventory(self):
        pool = CollateralPool()
        pool.add_inventory("UST10Y", 100_000_000)
        assert pool.available("UST10Y") == 100_000_000

    def test_pledge_reduces_available(self):
        pool = CollateralPool()
        pool.add_inventory("UST10Y", 100_000_000)
        pool.pledge("UST10Y", "BankA", 30_000_000)
        assert pool.available("UST10Y") == 70_000_000

    def test_pledge_exceeds_raises(self):
        pool = CollateralPool()
        pool.add_inventory("UST10Y", 50_000_000)
        with pytest.raises(ValueError, match="Cannot pledge"):
            pool.pledge("UST10Y", "BankA", 60_000_000)

    def test_release(self):
        pool = CollateralPool()
        pool.add_inventory("UST10Y", 100_000_000)
        pool.pledge("UST10Y", "BankA", 50_000_000)
        pool.release("UST10Y", "BankA", 20_000_000)
        assert pool.available("UST10Y") == 70_000_000

    def test_can_pledge(self):
        pool = CollateralPool()
        pool.add_inventory("UST10Y", 50_000_000)
        assert pool.can_pledge("UST10Y", 30_000_000) is True
        assert pool.can_pledge("UST10Y", 60_000_000) is False

    def test_summary(self):
        pool = CollateralPool()
        pool.add_inventory("UST10Y", 100_000_000)
        pool.add_inventory("UST5Y", 50_000_000)
        pool.pledge("UST10Y", "BankA", 40_000_000)
        summary = pool.summary()
        assert len(summary) == 2
        ust10 = [s for s in summary if s["issuer"] == "UST10Y"][0]
        assert ust10["available"] == 60_000_000


# ── Gap 5: SOFR-linked repo ──

class TestSOFRRepo:

    def test_sofr_compound_interest(self):
        trade = _make_trade(rate_type="sofr_compound", term_days=5)
        daily_rates = [0.045, 0.046, 0.044, 0.047, 0.045]
        interest = trade.sofr_interest(daily_rates)
        assert interest > 0
        # Should be close to simple: cash × avg_rate × 5/360
        avg = sum(daily_rates) / len(daily_rates)
        simple = trade.cash_amount * avg * 5 / 360
        assert interest == pytest.approx(simple, rel=0.01)

    def test_fixed_ignores_sofr(self):
        trade = _make_trade(rate_type="fixed")
        interest = trade.sofr_interest([0.05, 0.05, 0.05])
        assert interest == pytest.approx(trade.interest)


# ── Gap 6: Daily P&L ──

class TestDailyPnL:

    def test_unchanged_near_zero(self):
        book = RepoBook("Test")
        entry = RepoTradeEntry(
            counterparty="A", collateral_issuer="UST10Y",
            face_amount=50_000_000, bond_price=100.0, repo_rate=0.04,
            term_days=30, direction="repo", start_date=REF,
        )
        book.add(entry)
        curve = make_flat_curve(REF, 0.04)
        pnl = repo_daily_pnl(book, curve, curve, REF, REF)
        assert abs(pnl.total_pnl) < 1  # same curve → ~0

    def test_rate_shift_nonzero(self):
        book = RepoBook("Test")
        book.add(RepoTradeEntry(
            counterparty="A", collateral_issuer="UST10Y",
            face_amount=50_000_000, bond_price=100.0, repo_rate=0.04,
            term_days=30, direction="repo", start_date=REF,
        ))
        c0 = make_flat_curve(REF, 0.04)
        c1 = make_flat_curve(REF, 0.045)
        pnl = repo_daily_pnl(book, c0, c1, REF, REF + timedelta(days=1))
        assert pnl.total_pnl != 0

    def test_to_dict(self):
        book = RepoBook("Test")
        book.add(RepoTradeEntry(
            counterparty="A", collateral_issuer="UST10Y",
            face_amount=50_000_000, bond_price=100.0, repo_rate=0.04,
            term_days=30, direction="repo", start_date=REF,
        ))
        curve = make_flat_curve(REF, 0.04)
        pnl = repo_daily_pnl(book, curve, curve, REF, REF)
        d = pnl.to_dict()
        assert "carry" in d
        assert "rate" in d


# ── Gap 7: Lifecycle ──

class TestLifecycle:

    def test_initial_status(self):
        trade = _make_trade()
        assert trade.status == "live"

    def test_mature(self):
        trade = _make_trade()
        trade.mature()
        assert trade.status == "matured"

    def test_terminate(self):
        trade = _make_trade()
        trade.terminate_early()
        assert trade.status == "terminated"

    def test_remaining_days(self):
        trade = _make_trade(start_date=REF, term_days=30)
        assert trade.remaining_days(REF) == 30
        assert trade.remaining_days(REF + timedelta(days=10)) == 20
        assert trade.remaining_days(REF + timedelta(days=40)) == 0

    def test_maturity_date(self):
        trade = _make_trade(start_date=REF, term_days=30)
        assert trade.maturity_date == REF + timedelta(days=30)

    def test_open_repo(self):
        trade = _make_trade(term_days=0)
        assert trade.is_open
        assert trade.maturity_date is None


# ── Serialisation ──

class TestRepoTradeSerialisation:

    def test_round_trip(self):
        trade = _make_trade(trade_id="T001")
        d = trade.to_dict()
        trade2 = from_dict(d)
        assert trade2.counterparty == trade.counterparty
        assert trade2.repo_rate == trade.repo_rate
        assert trade2.haircut == trade.haircut
        assert trade2.status == trade.status
        assert trade2.trade_id == "T001"

    def test_carry_preserved(self):
        trade = _make_trade()
        d = trade.to_dict()
        trade2 = from_dict(d)
        assert trade2.carry == pytest.approx(trade.carry)
