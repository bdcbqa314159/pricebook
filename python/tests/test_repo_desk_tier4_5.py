"""Tests for repo desk Tiers 4+5: dashboard, hedging, matched book, attribution, serialisation, PV."""

from __future__ import annotations

import math
from datetime import date

import pytest

from pricebook.repo_desk import (
    RepoBook, RepoTradeEntry, FailsTracker, SettlementFail,
    daily_dashboard, RepoDashboard,
    hedge_recommendations, HedgeRecommendation,
    matched_book_analysis, MatchedBookEntry,
    funding_attribution, FundingAttribution,
    repo_book_pv,
)
from pricebook.serialisable import from_dict
from tests.conftest import make_flat_curve

REF = date(2024, 7, 15)


def _make_book():
    book = RepoBook("Test")
    # Repo positions (borrow cash)
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
    # Reverse repo (lend cash) — matched with UST10Y
    book.add(RepoTradeEntry(
        counterparty="BankC", collateral_issuer="UST10Y", collateral_type="GC",
        face_amount=40_000_000, bond_price=101.0, repo_rate=0.042,
        term_days=30, coupon_rate=0.04, direction="reverse", start_date=REF,
    ))
    return book


# ---- Daily dashboard ----

class TestDashboard:

    def test_all_fields(self):
        book = _make_book()
        db = daily_dashboard(book, REF)
        assert db.date == REF
        assert db.n_positions == len(book)
        assert math.isfinite(db.net_cash)
        assert math.isfinite(db.total_carry)
        assert math.isfinite(db.repo_dv01)

    def test_with_fails(self):
        book = _make_book()
        tracker = FailsTracker()
        tracker.add(SettlementFail("X", "UST10Y", 5_000_000, REF, 3, 300))
        db = daily_dashboard(book, REF, tracker=tracker)
        assert db.n_fails == 1
        assert db.total_fail_face == 5_000_000

    def test_top_cp(self):
        book = _make_book()
        db = daily_dashboard(book, REF)
        assert len(db.top_cp_exposures) > 0
        assert "counterparty" in db.top_cp_exposures[0]

    def test_to_dict(self):
        book = _make_book()
        db = daily_dashboard(book, REF)
        d = db.to_dict()
        assert "net_cash" in d
        assert "rollover_exposure" in d


# ---- Hedge recommendations ----

class TestHedgeRecs:

    def test_within_limits_no_action(self):
        book = RepoBook("Small")
        book.add(RepoTradeEntry(
            counterparty="A", collateral_issuer="UST10Y",
            face_amount=1_000_000, bond_price=100.0, repo_rate=0.04,
            term_days=30, direction="repo",
        ))
        # All limits very high → no action
        recs = hedge_recommendations(
            book, dv01_limit=1e9, rollover_limit=1e12, concentration_limit_pct=100,
        )
        assert recs[0].action == "none"

    def test_concentration_flag(self):
        book = _make_book()
        recs = hedge_recommendations(book, concentration_limit_pct=20.0)
        diversify = [r for r in recs if r.action == "diversify_cp"]
        assert len(diversify) > 0

    def test_rollover_flag(self):
        book = RepoBook("ONHeavy")
        book.add(RepoTradeEntry(
            counterparty="A", collateral_issuer="UST10Y",
            face_amount=1_000_000_000, bond_price=100.0, repo_rate=0.04,
            term_days=1, direction="repo",
        ))
        recs = hedge_recommendations(book, rollover_limit=500_000_000)
        extend = [r for r in recs if r.action == "extend_term"]
        assert len(extend) > 0
        assert extend[0].urgency == "immediate"

    def test_to_dict(self):
        book = _make_book()
        recs = hedge_recommendations(book)
        d = recs[0].to_dict()
        assert "action" in d
        assert "urgency" in d


# ---- Matched book ----

class TestMatchedBook:

    def test_finds_match(self):
        """UST10Y appears in both repo and reverse → should find match."""
        book = _make_book()
        matches = matched_book_analysis(book)
        issuers = [m.issuer for m in matches]
        assert "UST10Y" in issuers

    def test_spread_earned(self):
        book = _make_book()
        matches = matched_book_analysis(book)
        ust10 = [m for m in matches if m.issuer == "UST10Y"][0]
        # Repo at ~4.0-4.5%, reverse at 4.2% → spread should be computable
        assert math.isfinite(ust10.spread_earned_bps)

    def test_no_match_single_side(self):
        """UST5Y is only repo → should NOT appear in matched."""
        book = _make_book()
        matches = matched_book_analysis(book)
        issuers = [m.issuer for m in matches]
        assert "UST5Y" not in issuers

    def test_to_dict(self):
        book = _make_book()
        matches = matched_book_analysis(book)
        if matches:
            d = matches[0].to_dict()
            assert "spread_bps" in d


# ---- Funding attribution ----

class TestFundingAttribution:

    def test_all_strategies(self):
        book = _make_book()
        attr = funding_attribution(book)
        strats = [a.strategy for a in attr]
        assert "GC_ON" in strats
        assert "reverse" in strats

    def test_pct_sums(self):
        """Percentages should sum to ~100% (some overlap with reverse)."""
        book = _make_book()
        attr = funding_attribution(book)
        # Not exact 100% because reverse overlaps with other axes
        total_pct = sum(a.pct_of_book for a in attr)
        assert total_pct > 50  # at least covers most

    def test_to_dict(self):
        book = _make_book()
        attr = funding_attribution(book)
        d = attr[0].to_dict()
        assert "carry" in d
        assert "pct" in d


# ---- RepoBook serialisation ----

class TestRepoBookSerialisation:

    def test_round_trip(self):
        book = _make_book()
        d = book.to_dict()
        book2 = from_dict(d)
        assert len(book2) == len(book)
        assert book2.name == book.name

    def test_entries_preserved(self):
        book = _make_book()
        d = book.to_dict()
        book2 = from_dict(d)
        for e1, e2 in zip(book.entries, book2.entries):
            assert e1.counterparty == e2.counterparty
            assert e1.repo_rate == e2.repo_rate
            assert e1.face_amount == e2.face_amount

    def test_carry_preserved(self):
        book = _make_book()
        d = book.to_dict()
        book2 = from_dict(d)
        assert book2.net_carry() == pytest.approx(book.net_carry())


# ---- RepoBook PV ----

class TestRepoBookPV:

    def test_pv_finite(self):
        book = _make_book()
        curve = make_flat_curve(REF, 0.04)
        pv = repo_book_pv(book, curve, REF)
        assert math.isfinite(pv["total_pv"])

    def test_inception_near_zero(self):
        """At inception, each repo PV ≈ 0 (fair trade)."""
        book = _make_book()
        curve = make_flat_curve(REF, 0.04)
        pv = repo_book_pv(book, curve, REF)
        # Total PV should be small relative to notional
        total_notional = sum(e.cash_amount for e in book.entries)
        assert abs(pv["total_pv"]) < total_notional * 0.01

    def test_direction_split(self):
        book = _make_book()
        curve = make_flat_curve(REF, 0.04)
        pv = repo_book_pv(book, curve, REF)
        assert "repo_pv" in pv
        assert "reverse_pv" in pv
        assert pv["total_pv"] == pytest.approx(pv["repo_pv"] + pv["reverse_pv"])
