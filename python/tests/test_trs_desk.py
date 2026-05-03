"""Tests for TRS desk: risk metrics, carry, P&L, book, dashboard, XVA, stress."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.trs import TotalReturnSwap, FundingLegSpec
from pricebook.trs_desk import (
    trs_risk_metrics, TRSRiskMetrics,
    trs_carry_decomposition, TRSCarryDecomposition,
    trs_daily_pnl, TRSDailyPnL,
    TRSBook, TRSBookEntry,
    trs_dashboard, TRSDashboard,
    trs_all_in_cost, TRSAllInCost,
    trs_stress_suite, TRSStressResult,
    trs_capital_summary, TRSCapitalSummary,
    trs_hedge_recommendations, TRSHedgeRecommendation,
)
from pricebook.bond import FixedRateBond
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)
END = REF + relativedelta(months=6)


def _equity_trs():
    return TotalReturnSwap(
        underlying=100.0, notional=1_000_000,
        start=REF, end=END,
        funding=FundingLegSpec(spread=0.005),
        repo_spread=0.01, haircut=0.05,
        initial_price=100.0, sigma=0.20,
    )


def _bond_trs():
    bond = FixedRateBond.treasury_note(date(2024, 2, 15), date(2034, 2, 15), 0.04125)
    return TotalReturnSwap(
        underlying=bond, notional=10_000_000,
        start=REF, end=END,
        funding=FundingLegSpec(spread=0.005),
        repo_spread=0.005, initial_price=102.0,
    )


# ── Risk metrics ──

class TestRiskMetrics:

    def test_equity_delta_nonzero(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        rm = trs_risk_metrics(trs, curve)
        assert rm.delta != 0  # equity TRS has delta

    def test_equity_gamma(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        rm = trs_risk_metrics(trs, curve)
        assert math.isfinite(rm.gamma)

    def test_dv01_nonzero(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        rm = trs_risk_metrics(trs, curve)
        assert rm.dv01 != 0

    def test_funding_dv01_nonzero(self):
        """Funding spread affects PV."""
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        rm = trs_risk_metrics(trs, curve)
        assert rm.funding_dv01 != 0

    def test_to_dict(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        rm = trs_risk_metrics(trs, curve)
        d = rm.to_dict()
        assert "delta" in d
        assert "funding_dv01" in d


# ── Carry decomposition ──

class TestCarry:

    def test_components(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        cd = trs_carry_decomposition(trs, curve)
        assert math.isfinite(cd.income)
        assert math.isfinite(cd.funding_cost)
        assert math.isfinite(cd.repo_cost)

    def test_to_dict(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        cd = trs_carry_decomposition(trs, curve)
        d = cd.to_dict()
        assert "income" in d
        assert "net" in d


# ── Daily P&L ──

class TestDailyPnL:

    def test_unchanged_small(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        pnl = trs_daily_pnl(trs, curve, curve, REF)
        assert abs(pnl.total) < 1  # same curve → ~0

    def test_rate_shift(self):
        trs = _equity_trs()
        c0 = make_flat_curve(REF, 0.04)
        c1 = make_flat_curve(REF, 0.045)
        pnl = trs_daily_pnl(trs, c0, c1, REF + relativedelta(days=1))
        assert pnl.total != 0

    def test_to_dict(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        pnl = trs_daily_pnl(trs, curve, curve, REF)
        d = pnl.to_dict()
        assert "delta" in d


# ── Book ──

class TestBook:

    def test_add_and_count(self):
        book = TRSBook("TestBook")
        book.add(TRSBookEntry("T1", _equity_trs(), "JPM"))
        book.add(TRSBookEntry("T2", _bond_trs(), "GS"))
        assert len(book) == 2

    def test_total_notional(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        book.add(TRSBookEntry("T2", _bond_trs()))
        assert book.total_notional() == 1_000_000 + 10_000_000

    def test_by_type(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        book.add(TRSBookEntry("T2", _bond_trs()))
        bt = book.by_type()
        assert "equity" in bt
        assert "bond" in bt

    def test_independent_amount(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs(), independent_amount=500_000))
        book.add(TRSBookEntry("T2", _bond_trs(), independent_amount=200_000))
        assert book.total_independent_amount() == 700_000

    def test_aggregate_risk(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        curve = make_flat_curve(REF, 0.04)
        risk = book.aggregate_risk(curve)
        assert "total_pv" in risk
        assert "total_delta" in risk
        assert risk["n_positions"] == 1


# ── Dashboard ──

class TestDashboard:

    def test_dashboard(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs(), "JPM"))
        curve = make_flat_curve(REF, 0.04)
        db = trs_dashboard(book, REF, curve)
        assert db.n_positions == 1
        assert db.total_notional == 1_000_000
        assert math.isfinite(db.total_pv)

    def test_to_dict(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        curve = make_flat_curve(REF, 0.04)
        db = trs_dashboard(book, REF, curve)
        d = db.to_dict()
        assert "pv" in d
        assert "by_type" in d


# ── XVA all-in cost ──

class TestAllInCost:

    def test_all_in_greater_than_headline(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        cost = trs_all_in_cost(trs, curve, capital_charge=50_000)
        assert cost.all_in_spread_bps >= cost.headline_spread_bps

    def test_hidden_cost(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        cost = trs_all_in_cost(trs, curve, capital_charge=50_000, initial_margin=100_000)
        assert cost.hidden_cost_bps >= 0

    def test_to_dict(self):
        trs = _equity_trs()
        curve = make_flat_curve(REF, 0.04)
        cost = trs_all_in_cost(trs, curve)
        d = cost.to_dict()
        assert "hidden_bps" in d


# ── Stress testing ──

class TestStress:

    def test_five_scenarios(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        curve = make_flat_curve(REF, 0.04)
        results = trs_stress_suite(book, curve)
        assert len(results) == 5

    def test_equity_crash_has_impact(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        curve = make_flat_curve(REF, 0.04)
        results = trs_stress_suite(book, curve)
        crash = [r for r in results if r.scenario == "equity_crash"][0]
        assert crash.total_pnl != 0  # crash has material impact

    def test_to_dict(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        curve = make_flat_curve(REF, 0.04)
        results = trs_stress_suite(book, curve)
        d = results[0].to_dict()
        assert "delta" in d
        assert "total" in d


# ── Regulatory capital ──

class TestCapital:

    def test_capital_summary(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        book.add(TRSBookEntry("T2", _bond_trs()))
        curve = make_flat_curve(REF, 0.04)
        cap = trs_capital_summary(book, curve)
        assert len(cap.entries) == 2
        assert cap.total_ead > 0
        assert cap.total_rwa > 0
        assert cap.total_capital > 0

    def test_equity_higher_ead_per_notional(self):
        """Equity SF=0.32 >> bond SF=0.005."""
        book_eq = TRSBook()
        book_eq.add(TRSBookEntry("T1", _equity_trs()))
        book_bd = TRSBook()
        book_bd.add(TRSBookEntry("T2", _bond_trs()))
        curve = make_flat_curve(REF, 0.04)
        cap_eq = trs_capital_summary(book_eq, curve)
        cap_bd = trs_capital_summary(book_bd, curve)
        # Per-notional EAD: equity >> bond
        assert (cap_eq.total_ead / 1_000_000) > (cap_bd.total_ead / 10_000_000)

    def test_rw_affects_rwa(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        curve = make_flat_curve(REF, 0.04)
        cap_corp = trs_capital_summary(book, curve, counterparty_type="corporate")
        cap_bank = trs_capital_summary(book, curve, counterparty_type="bank")
        assert cap_corp.total_rwa > cap_bank.total_rwa  # RW 100% > 20%

    def test_capital_is_8pct_of_rwa(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        curve = make_flat_curve(REF, 0.04)
        cap = trs_capital_summary(book, curve)
        assert abs(cap.total_capital - cap.total_rwa * 0.08) < 0.01

    def test_to_dict(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        curve = make_flat_curve(REF, 0.04)
        cap = trs_capital_summary(book, curve)
        d = cap.to_dict()
        assert "total_ead" in d
        assert "by_trade" in d


# ── Hedge recommendations ──

class TestHedgeRecommendations:

    def test_no_recs_within_limits(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        curve = make_flat_curve(REF, 0.04)
        recs = trs_hedge_recommendations(
            book, curve,
            delta_limit=1e12, dv01_limit=1e12,
            vega_limit=1e12, funding_dv01_limit=1e12,
        )
        assert len(recs) == 0

    def test_recs_when_breached(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        curve = make_flat_curve(REF, 0.04)
        recs = trs_hedge_recommendations(
            book, curve,
            delta_limit=0.001,  # very tight limit
            dv01_limit=0.001,
        )
        assert len(recs) >= 1

    def test_rec_has_action(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        curve = make_flat_curve(REF, 0.04)
        recs = trs_hedge_recommendations(book, curve, delta_limit=0.001)
        if recs:
            assert recs[0].action != ""
            assert recs[0].breach_pct > 0

    def test_to_dict(self):
        book = TRSBook()
        book.add(TRSBookEntry("T1", _equity_trs()))
        curve = make_flat_curve(REF, 0.04)
        recs = trs_hedge_recommendations(book, curve, delta_limit=0.001)
        if recs:
            d = recs[0].to_dict()
            assert "risk" in d
            assert "action" in d
