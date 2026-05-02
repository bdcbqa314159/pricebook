"""Tests for repo desk Tier 3: fail workflow, substitution, balance sheet, stress testing."""

from __future__ import annotations

from datetime import date

import pytest

from pricebook.repo_desk import (
    RepoBook, RepoTradeEntry,
    FailsTracker, SettlementFail,
    fail_workflow, FailResolution,
    find_substitutes, SubstitutionCandidate,
    balance_sheet_efficiency, BalanceSheetMetrics,
    stress_test_suite, StressTestResult,
)

REF = date(2024, 7, 15)


def _make_book():
    book = RepoBook("Test")
    book.add(RepoTradeEntry(
        counterparty="A", collateral_issuer="UST10Y", face_amount=50_000_000,
        bond_price=101.5, repo_rate=0.045, term_days=1, coupon_rate=0.04,
        direction="repo", start_date=REF,
    ))
    book.add(RepoTradeEntry(
        counterparty="B", collateral_issuer="UST5Y", collateral_type="special",
        face_amount=30_000_000, bond_price=100.0, repo_rate=0.035,
        term_days=90, coupon_rate=0.035, direction="repo", start_date=REF,
    ))
    return book


# ---- Fail workflow ----

class TestFailWorkflow:

    def test_categorisation(self):
        tracker = FailsTracker()
        tracker.add(SettlementFail("A", "UST10Y", 10_000_000, REF, 1, 300))
        tracker.add(SettlementFail("B", "UST5Y", 20_000_000, REF, 5, 300))
        tracker.add(SettlementFail("C", "UST2Y", 5_000_000, REF, 10, 300))

        results = fail_workflow(tracker)
        cats = [r.category for r in results]
        assert "system" in cats      # 1 day
        assert "counterparty" in cats  # 5+ days

    def test_escalation(self):
        tracker = FailsTracker()
        tracker.add(SettlementFail("A", "UST10Y", 10_000_000, REF, 3, 300))
        tracker.add(SettlementFail("B", "UST5Y", 20_000_000, REF, 7, 300))

        results = fail_workflow(tracker, escalation_days=5)
        assert not results[0].escalated  # 3 days
        assert results[1].escalated      # 7 days

    def test_buy_in_cost(self):
        tracker = FailsTracker()
        tracker.add(SettlementFail("A", "UST10Y", 10_000_000, REF, 5, 300))

        results = fail_workflow(
            tracker,
            current_prices={"UST10Y": 102.0},
            contract_prices={"UST10Y": 100.0},
        )
        # Buy-in = (102 - 100) / 100 × 10M = $200K
        assert results[0].buy_in_cost == pytest.approx(200_000)

    def test_to_dict(self):
        tracker = FailsTracker()
        tracker.add(SettlementFail("A", "UST10Y", 10_000_000, REF, 3, 300))
        results = fail_workflow(tracker)
        d = results[0].to_dict()
        assert "category" in d
        assert "buy_in_cost" in d


# ---- Collateral substitution ----

class TestSubstitution:

    def test_sorted_by_cost(self):
        subs = find_substitutes(
            failed_repo_rate=0.035,
            alternatives={
                "UST5Y": (0.040, 2.0, True),
                "UST2Y": (0.036, 1.5, True),
                "UST30Y": (0.045, 3.0, False),
            },
        )
        costs = [s.cost_vs_original_bps for s in subs]
        for i in range(1, len(costs)):
            assert costs[i] >= costs[i-1]

    def test_cheapest_first(self):
        subs = find_substitutes(
            0.035,
            {"A": (0.036, 2.0, True), "B": (0.050, 3.0, True)},
        )
        assert subs[0].bond_id == "A"

    def test_availability_tracked(self):
        subs = find_substitutes(
            0.035,
            {"A": (0.036, 2.0, True), "B": (0.037, 2.5, False)},
        )
        avail = {s.bond_id: s.available for s in subs}
        assert avail["A"] is True
        assert avail["B"] is False

    def test_to_dict(self):
        subs = find_substitutes(0.035, {"A": (0.040, 2.0, True)})
        d = subs[0].to_dict()
        assert "cost_bp" in d


# ---- Balance sheet efficiency ----

class TestBalanceSheet:

    def test_positive_roc(self):
        book = _make_book()
        bs = balance_sheet_efficiency(book)
        assert bs.total_assets > 0
        assert bs.total_capital_used > 0

    def test_leverage(self):
        book = _make_book()
        bs = balance_sheet_efficiency(book, haircut_pct=2.0)
        assert bs.leverage_ratio == pytest.approx(50.0)  # 100/2 = 50x

    def test_to_dict(self):
        book = _make_book()
        bs = balance_sheet_efficiency(book)
        d = bs.to_dict()
        assert "roc_pct" in d
        assert "leverage" in d


# ---- Stress testing ----

class TestStressTest:

    def test_five_scenarios(self):
        book = _make_book()
        results = stress_test_suite(book)
        assert len(results) == 5
        names = [r.scenario_name for r in results]
        assert "2008_crisis" in names
        assert "covid_mar2020" in names
        assert "cb_tightening" in names

    def test_2008_worst(self):
        """2008 should be the worst scenario."""
        book = _make_book()
        results = stress_test_suite(book)
        totals = {r.scenario_name: abs(r.total_impact) for r in results}
        # 2008 should have large impact
        assert totals["2008_crisis"] > 0

    def test_components_sum(self):
        """total = carry + margin + fails."""
        book = _make_book()
        results = stress_test_suite(book)
        for r in results:
            expected = r.carry_impact + r.margin_call + r.fails_impact
            assert r.total_impact == pytest.approx(expected, abs=1.0)

    def test_to_dict(self):
        book = _make_book()
        results = stress_test_suite(book)
        d = results[0].to_dict()
        assert "carry" in d
        assert "margin_call" in d
        assert "fails" in d
