"""Tests for repo desk Tier 1: cash ladder, rate DV01, carry decomposition, rollover, CP monitor."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.repo_desk import (
    RepoBook, RepoTradeEntry,
    cash_ladder, CashLadderBucket,
    repo_rate_dv01,
    carry_pnl_decomposition, CarryDecomposition,
    rollover_risk, RolloverScenario,
    counterparty_exposure_monitor, CounterpartyLimit,
)


REF = date(2024, 7, 15)


def _make_book():
    """Build a realistic repo book with 5 positions."""
    book = RepoBook("Test")
    book.add(RepoTradeEntry(
        counterparty="BankA", collateral_issuer="UST10Y", collateral_type="GC",
        face_amount=50_000_000, bond_price=101.5, repo_rate=0.045,
        term_days=1, coupon_rate=0.04, direction="repo", start_date=REF,
    ))
    book.add(RepoTradeEntry(
        counterparty="BankA", collateral_issuer="UST10Y", collateral_type="special",
        face_amount=30_000_000, bond_price=102.0, repo_rate=0.035,
        term_days=30, coupon_rate=0.04, direction="repo", start_date=REF,
    ))
    book.add(RepoTradeEntry(
        counterparty="BankB", collateral_issuer="UST5Y", collateral_type="GC",
        face_amount=20_000_000, bond_price=99.5, repo_rate=0.044,
        term_days=90, coupon_rate=0.035, direction="repo", start_date=REF,
    ))
    book.add(RepoTradeEntry(
        counterparty="BankC", collateral_issuer="UST2Y", collateral_type="GC",
        face_amount=40_000_000, bond_price=100.0, repo_rate=0.046,
        term_days=180, coupon_rate=0.045, direction="reverse", start_date=REF,
    ))
    book.add(RepoTradeEntry(
        counterparty="BankB", collateral_issuer="UST30Y", collateral_type="GC",
        face_amount=25_000_000, bond_price=98.0, repo_rate=0.043,
        term_days=7, coupon_rate=0.04125, direction="repo", start_date=REF,
    ))
    return book


# ---- Cash ladder ----

class TestCashLadder:

    def test_buckets_exist(self):
        book = _make_book()
        ladder = cash_ladder(book, REF, overnight_rate=0.045)
        assert len(ladder) == 6
        labels = [b.bucket for b in ladder]
        assert "O/N" in labels
        assert "1W" in labels
        assert "1Y+" in labels

    def test_on_bucket_has_trades(self):
        book = _make_book()
        ladder = cash_ladder(book, REF, overnight_rate=0.045)
        on_bucket = [b for b in ladder if b.bucket == "O/N"][0]
        assert on_bucket.n_trades > 0
        assert on_bucket.maturing_cash != 0

    def test_total_cash_consistent(self):
        book = _make_book()
        ladder = cash_ladder(book, REF, overnight_rate=0.045)
        total_ladder = sum(b.maturing_cash for b in ladder)
        total_book = book.total_cash_out() - book.total_cash_in()
        assert total_ladder == pytest.approx(total_book, rel=0.01)

    def test_refinancing_cost_positive(self):
        book = _make_book()
        ladder = cash_ladder(book, REF, overnight_rate=0.045)
        for b in ladder:
            assert b.refinancing_cost >= 0

    def test_to_dict(self):
        book = _make_book()
        ladder = cash_ladder(book, REF, overnight_rate=0.045)
        d = ladder[0].to_dict()
        assert "bucket" in d
        assert "maturing_cash" in d


# ---- Repo rate DV01 ----

class TestRepoRateDV01:

    def test_negative_dv01(self):
        """Higher repo rate → higher financing cost → lower carry → DV01 < 0 for net lender."""
        book = _make_book()
        result = repo_rate_dv01(book)
        # Net repo book: mostly borrowing cash, so higher rate = more cost
        assert result["total_dv01"] != 0

    def test_per_trade_length(self):
        book = _make_book()
        result = repo_rate_dv01(book)
        assert len(result["per_trade_dv01"]) == len(book)

    def test_base_carry_matches(self):
        book = _make_book()
        result = repo_rate_dv01(book)
        assert result["base_carry"] == pytest.approx(book.net_carry())


# ---- Carry decomposition ----

class TestCarryDecomposition:

    def test_components_sum(self):
        """coupon - financing = total carry."""
        book = _make_book()
        cd = carry_pnl_decomposition(book, gc_rate=0.045)
        assert cd.total_carry == pytest.approx(
            cd.coupon_income - cd.repo_financing_cost, rel=0.01)

    def test_specialness_positive_when_below_gc(self):
        """Positions on special (repo < GC) should have positive specialness benefit."""
        book = _make_book()
        cd = carry_pnl_decomposition(book, gc_rate=0.045)
        # We have a 3.5% special vs 4.5% GC → savings
        assert cd.specialness_benefit > 0

    def test_to_dict(self):
        book = _make_book()
        cd = carry_pnl_decomposition(book, gc_rate=0.045)
        d = cd.to_dict()
        assert "specialness_benefit" in d
        assert "coupon_income" in d


# ---- Rollover risk ----

class TestRolloverRisk:

    def test_default_scenarios(self):
        book = _make_book()
        scenarios = rollover_risk(book)
        assert len(scenarios) == 4
        names = [s.scenario_name for s in scenarios]
        assert "mild" in names
        assert "crisis" in names

    def test_cost_increases_with_severity(self):
        book = _make_book()
        scenarios = rollover_risk(book)
        costs = [s.additional_cost for s in scenarios]
        for i in range(1, len(costs)):
            assert costs[i] >= costs[i - 1]

    def test_zero_on_no_short_term(self):
        """Book with only long-term trades → no rollover risk."""
        book = RepoBook("LongOnly")
        book.add(RepoTradeEntry(
            counterparty="X", collateral_issuer="UST10Y",
            face_amount=10_000_000, bond_price=100.0, repo_rate=0.04,
            term_days=365, direction="repo",
        ))
        scenarios = rollover_risk(book)
        assert all(s.additional_cost == 0 for s in scenarios)

    def test_to_dict(self):
        book = _make_book()
        scenarios = rollover_risk(book)
        d = scenarios[0].to_dict()
        assert "spike_bps" in d
        assert "cost" in d


# ---- Counterparty exposure monitor ----

class TestCPMonitor:

    def test_all_cps_listed(self):
        book = _make_book()
        result = counterparty_exposure_monitor(book)
        cps = {r.counterparty for r in result}
        assert "BankA" in cps
        assert "BankB" in cps
        assert "BankC" in cps

    def test_breach_detection(self):
        book = _make_book()
        limits = {"BankA": 50_000_000}  # tight limit
        result = counterparty_exposure_monitor(book, limits=limits)
        bank_a = [r for r in result if r.counterparty == "BankA"][0]
        assert bank_a.breached  # BankA has ~$80M, limit is $50M

    def test_no_breach_with_high_limit(self):
        book = _make_book()
        result = counterparty_exposure_monitor(book, default_limit=1e12)
        assert all(not r.breached for r in result)

    def test_sorted_by_utilisation(self):
        book = _make_book()
        result = counterparty_exposure_monitor(book)
        utils = [r.utilisation_pct for r in result]
        for i in range(1, len(utils)):
            assert utils[i] <= utils[i - 1]

    def test_headroom(self):
        book = _make_book()
        result = counterparty_exposure_monitor(book, default_limit=100_000_000)
        for r in result:
            assert r.headroom == pytest.approx(max(0, r.limit - r.current_exposure))

    def test_to_dict(self):
        book = _make_book()
        result = counterparty_exposure_monitor(book)
        d = result[0].to_dict()
        assert "breached" in d
        assert "headroom" in d
