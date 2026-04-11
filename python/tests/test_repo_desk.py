"""Tests for repo desk."""

import pytest
from datetime import date

from pricebook.repo_desk import (
    CTDRepoCandidate,
    FailsTracker,
    RepoBook,
    RepoTradeEntry,
    SettlementFail,
    TermVsOvernightResult,
    cheapest_to_deliver_repo,
    repo_rate_monitor,
    term_vs_overnight,
)


# ---- Step 1: repo book ----

class TestRepoTradeEntry:
    def test_cash_amount(self):
        e = RepoTradeEntry("CP1", "UST", face_amount=10_000_000,
                           bond_price=98.5, repo_rate=0.05, term_days=30)
        assert e.cash_amount == pytest.approx(9_850_000)

    def test_carry_repo_direction(self):
        e = RepoTradeEntry("CP1", "UST", face_amount=10_000_000,
                           bond_price=98.5, repo_rate=0.05, term_days=365,
                           coupon_rate=0.04, direction="repo")
        # coupon = 10M × 0.04 = 400K; financing = 9.85M × 0.05 = 492.5K
        # carry = 400K - 492.5K = -92.5K
        assert e.carry == pytest.approx(400_000 - 492_500)

    def test_carry_reverse_flips_sign(self):
        e = RepoTradeEntry("CP1", "UST", face_amount=10_000_000,
                           bond_price=98.5, repo_rate=0.05, term_days=365,
                           coupon_rate=0.04, direction="reverse")
        repo_carry = RepoTradeEntry("CP1", "UST", face_amount=10_000_000,
                                     bond_price=98.5, repo_rate=0.05, term_days=365,
                                     coupon_rate=0.04, direction="repo").carry
        assert e.carry == pytest.approx(-repo_carry)

    def test_financing_cost(self):
        e = RepoTradeEntry("CP1", "UST", face_amount=10_000_000,
                           bond_price=100.0, repo_rate=0.05, term_days=365)
        assert e.financing_cost == pytest.approx(500_000)


class TestRepoBook:
    def test_empty(self):
        book = RepoBook("test")
        assert len(book) == 0
        assert book.net_carry() == 0.0

    def test_add_and_carry(self):
        book = RepoBook("test")
        book.add(RepoTradeEntry("CP1", "UST", face_amount=10_000_000,
                                bond_price=100.0, repo_rate=0.04,
                                term_days=365, coupon_rate=0.05,
                                direction="repo"))
        # coupon = 500K, financing = 400K, carry = +100K
        assert book.net_carry() == pytest.approx(100_000)

    def test_total_cash_out(self):
        book = RepoBook("test")
        book.add(RepoTradeEntry("CP1", "UST", face_amount=10_000_000,
                                bond_price=100.0, direction="repo"))
        assert book.total_cash_out() == pytest.approx(10_000_000)
        assert book.total_cash_in() == 0.0

    def test_total_cash_in(self):
        book = RepoBook("test")
        book.add(RepoTradeEntry("CP1", "UST", face_amount=5_000_000,
                                bond_price=100.0, direction="reverse"))
        assert book.total_cash_in() == pytest.approx(5_000_000)

    def test_by_counterparty(self):
        book = RepoBook("test")
        book.add(RepoTradeEntry("BankA", "UST", face_amount=10_000_000,
                                bond_price=100.0, repo_rate=0.04))
        book.add(RepoTradeEntry("BankB", "UST", face_amount=5_000_000,
                                bond_price=100.0, repo_rate=0.05))
        by_cp = book.by_counterparty()
        assert len(by_cp) == 2
        assert {c.counterparty for c in by_cp} == {"BankA", "BankB"}

    def test_by_collateral_type(self):
        book = RepoBook("test")
        book.add(RepoTradeEntry("CP1", "UST", collateral_type="GC",
                                face_amount=10_000_000, bond_price=100.0,
                                repo_rate=0.04))
        book.add(RepoTradeEntry("CP1", "UST_10Y", collateral_type="special",
                                face_amount=5_000_000, bond_price=100.0,
                                repo_rate=0.02))
        by_ct = book.by_collateral_type()
        assert len(by_ct) == 2
        assert {c.collateral_type for c in by_ct} == {"GC", "special"}

    def test_gc_rate(self):
        book = RepoBook("test")
        book.add(RepoTradeEntry("CP1", "UST", collateral_type="GC",
                                face_amount=10_000_000, bond_price=100.0,
                                repo_rate=0.04))
        book.add(RepoTradeEntry("CP1", "UST2", collateral_type="GC",
                                face_amount=10_000_000, bond_price=100.0,
                                repo_rate=0.05))
        # Weighted avg: (10M×0.04 + 10M×0.05) / 20M = 0.045
        assert book.gc_rate() == pytest.approx(0.045)

    def test_gc_rate_none_when_empty(self):
        book = RepoBook("test")
        assert book.gc_rate() is None

    def test_special_rate(self):
        book = RepoBook("test")
        book.add(RepoTradeEntry("CP1", "UST_10Y", collateral_type="special",
                                face_amount=10_000_000, bond_price=100.0,
                                repo_rate=0.02))
        assert book.special_rate("UST_10Y") == pytest.approx(0.02)
        assert book.special_rate("OTHER") is None


class TestRepoRateMonitor:
    def test_rich_signal(self):
        history = [0.04, 0.042, 0.038, 0.041, 0.039] * 4
        sig = repo_rate_monitor(0.06, history, threshold=2.0)
        assert sig.signal == "rich"

    def test_fair_signal(self):
        history = [0.04, 0.042, 0.038, 0.041, 0.039] * 4
        sig = repo_rate_monitor(0.04, history)
        assert sig.signal == "fair"


# ---- Step 2: financing optimisation ----

class TestCheapestToDeliverRepo:
    def test_picks_lowest_financing(self):
        """Step 2 test: optimal collateral minimises financing cost."""
        candidates = [
            CTDRepoCandidate("UST_2Y", 99.5, 0.045, 0.04, 30),
            CTDRepoCandidate("UST_5Y", 98.0, 0.043, 0.035, 30),
            CTDRepoCandidate("UST_10Y", 95.0, 0.040, 0.03, 30),
        ]
        best = cheapest_to_deliver_repo(candidates)
        assert best is not None
        # Lowest price × lowest rate → UST_10Y has lowest financing cost
        assert best.issuer == "UST_10Y"
        # Verify it really is the cheapest
        for c in candidates:
            assert best.financing_cost <= c.financing_cost

    def test_empty_returns_none(self):
        assert cheapest_to_deliver_repo([]) is None

    def test_carry_positive_means_profitable(self):
        c = CTDRepoCandidate("UST", 100.0, 0.03, 0.05, 365)
        # coupon = 5, financing = 3 → carry = 2
        assert c.carry > 0


class TestTermVsOvernight:
    def test_term_cheaper(self):
        result = term_vs_overnight(
            face_amount=10_000_000, bond_price=100.0,
            term_rate=0.04, overnight_rate=0.05, term_days=30,
        )
        assert result.recommendation == "term"
        assert result.savings > 0

    def test_overnight_cheaper(self):
        result = term_vs_overnight(
            face_amount=10_000_000, bond_price=100.0,
            term_rate=0.05, overnight_rate=0.04, term_days=30,
        )
        assert result.recommendation == "overnight"
        assert result.savings < 0

    def test_equal_rates(self):
        result = term_vs_overnight(
            face_amount=10_000_000, bond_price=100.0,
            term_rate=0.04, overnight_rate=0.04, term_days=30,
        )
        assert result.recommendation == "indifferent"
        assert result.savings == pytest.approx(0.0)

    def test_cost_formula(self):
        result = term_vs_overnight(
            face_amount=10_000_000, bond_price=100.0,
            term_rate=0.04, overnight_rate=0.05, term_days=365,
        )
        assert result.term_cost == pytest.approx(10_000_000 * 0.04)
        assert result.overnight_cost == pytest.approx(10_000_000 * 0.05)


# ---- Fails tracking ----

class TestFailsTracker:
    def test_empty(self):
        tracker = FailsTracker()
        assert len(tracker) == 0
        assert tracker.total_penalty() == 0.0

    def test_penalty_cost(self):
        tracker = FailsTracker()
        tracker.add(SettlementFail(
            "BankA", "UST_10Y", face_amount=10_000_000,
            fail_date=date(2024, 1, 15), days_outstanding=3,
            penalty_rate_bps=300,
        ))
        # 10M × 3% × 3/365
        expected = 10_000_000 * 0.03 * 3 / 365.0
        assert tracker.total_penalty() == pytest.approx(expected)

    def test_total_face(self):
        tracker = FailsTracker()
        tracker.add(SettlementFail("A", "X", 10_000_000, date(2024, 1, 15)))
        tracker.add(SettlementFail("B", "Y", 5_000_000, date(2024, 1, 15)))
        assert tracker.total_face_outstanding() == pytest.approx(15_000_000)

    def test_by_counterparty(self):
        tracker = FailsTracker()
        tracker.add(SettlementFail("A", "X", 10_000_000, date(2024, 1, 15)))
        tracker.add(SettlementFail("A", "Y", 5_000_000, date(2024, 1, 15)))
        tracker.add(SettlementFail("B", "Z", 3_000_000, date(2024, 1, 15)))
        by_cp = tracker.by_counterparty()
        assert by_cp["A"] == pytest.approx(15_000_000)
        assert by_cp["B"] == pytest.approx(3_000_000)
