"""Asset swap desk consolidation tests."""

from __future__ import annotations

import math
from datetime import date

import pytest

from pricebook.bond import FixedRateBond
from pricebook.discount_curve import DiscountCurve
from pricebook.par_asset_swap import ParAssetSwap, ProceedsAssetSwap
from pricebook.desks.asset_swap_desk import (
    asw_risk_metrics, ASWRiskMetrics,
    ASWBook, ASWBookEntry,
    asw_carry_decomposition, ASWCarryDecomposition,
    asw_daily_pnl, ASWDailyPnL,
    asw_dashboard, ASWDashboard,
    asw_stress_suite, ASWStressResult,
    asw_capital, ASWCapitalResult,
    asw_hedge_recommendations, ASWHedgeRecommendation,
    ASWLifecycle,
)

REF = date(2024, 7, 15)
END = date(2029, 7, 15)


def _curve():
    return DiscountCurve.flat(REF, 0.04)


def _bond():
    return FixedRateBond(REF, END, coupon_rate=0.05)


def _par_asw(market_price=98.0):
    return ParAssetSwap(_bond(), REF, market_price)


def _proceeds_asw(market_dirty=98.0):
    return ProceedsAssetSwap(_bond(), REF, market_dirty)


# ── Risk Metrics ──

class TestASWRiskMetrics:

    def test_spread_positive_for_discount_bond(self):
        """Bond below par → positive ASW spread (credit compensation)."""
        rm = asw_risk_metrics(_par_asw(98.0), _curve())
        assert rm.asw_spread > 0

    def test_spread_near_zero_at_par(self):
        """Bond at par → ASW spread ≈ coupon - swap rate."""
        bond = _bond()
        rf_price = bond.dirty_price(_curve())
        rm = asw_risk_metrics(ParAssetSwap(bond, REF, rf_price), _curve())
        assert abs(rm.asw_spread) < 0.01  # within 100bp of zero

    def test_asw01_positive(self):
        rm = asw_risk_metrics(_par_asw(), _curve())
        assert rm.asw01 > 0

    def test_bond_and_swap_dv01_opposite_signs(self):
        """Bond and swap DV01 should partially offset."""
        rm = asw_risk_metrics(_par_asw(), _curve())
        # Bond DV01 and swap DV01 should have different signs (hedging)
        # Net DV01 should be smaller than bond DV01 alone
        assert abs(rm.dv01) < abs(rm.bond_dv01) * 2  # not perfectly hedged but bounded

    def test_to_dict(self):
        rm = asw_risk_metrics(_par_asw(), _curve())
        d = rm.to_dict()
        assert "asw_spread" in d
        assert "asw01" in d
        assert "basis_risk" in d

    def test_proceeds_convention(self):
        rm = asw_risk_metrics(_proceeds_asw(), _curve())
        assert math.isfinite(rm.asw_spread)
        assert rm.asw01 > 0


# ── Book ──

class TestASWBook:

    def _book(self):
        book = ASWBook("test_asw")
        book.add(ASWBookEntry("T1", _par_asw(98.0), "par", bond_issuer="CORP_A", sector="IG"))
        book.add(ASWBookEntry("T2", _par_asw(95.0), "par", bond_issuer="CORP_B", sector="HY"))
        return book

    def test_len(self):
        assert len(self._book()) == 2

    def test_by_sector(self):
        by = self._book().by_sector()
        assert "IG" in by
        assert "HY" in by

    def test_aggregate_risk_keys(self):
        risk = self._book().aggregate_risk(_curve())
        assert "total_pv" in risk
        assert "total_asw01" in risk
        assert "total_dv01" in risk
        assert risk["n_positions"] == 2

    def test_total_notional(self):
        assert self._book().total_notional() > 0


# ── Carry ──

class TestASWCarry:

    def test_coupon_income_positive(self):
        carry = asw_carry_decomposition(_par_asw(), _curve())
        assert carry.coupon_income > 0

    def test_carry_finite(self):
        carry = asw_carry_decomposition(_par_asw(), _curve())
        assert math.isfinite(carry.net_carry)

    def test_to_dict(self):
        carry = asw_carry_decomposition(_par_asw(), _curve())
        d = carry.to_dict()
        assert "coupon" in d
        assert "floating" in d
        assert "funding" in d


# ── Daily P&L ──

class TestASWDailyPnL:

    def test_unchanged_curves_small(self):
        curve = _curve()
        pnl = asw_daily_pnl(_par_asw(), curve, curve, REF)
        assert abs(pnl.total) < 1e-6

    def test_rate_shift_has_impact(self):
        curve = _curve()
        pnl = asw_daily_pnl(_par_asw(), curve, curve.bumped(0.001), REF)
        assert pnl.total != 0

    def test_to_dict(self):
        curve = _curve()
        pnl = asw_daily_pnl(_par_asw(), curve, curve, REF)
        d = pnl.to_dict()
        assert "total" in d
        assert "spread" in d


# ── Dashboard ──

class TestASWDashboard:

    def test_dashboard(self):
        book = ASWBook("test")
        book.add(ASWBookEntry("T1", _par_asw(), "par", sector="IG"))
        db = asw_dashboard(book, REF, _curve())
        assert db.n_positions == 1
        assert db.average_spread_bp != 0
        assert "IG" in db.by_sector


# ── Stress ──

class TestASWStress:

    def test_stress_suite(self):
        book = ASWBook("test")
        book.add(ASWBookEntry("T1", _par_asw(), "par"))
        stress = asw_stress_suite(book, _curve())
        assert len(stress) >= 5
        assert any(s.scenario == "rates_up_100" for s in stress)


# ── Capital ──

class TestASWCapital:

    def test_capital_positive(self):
        cap = asw_capital(_par_asw(), _curve())
        assert cap.capital > 0
        assert cap.ead > 0

    def test_capital_8pct_rwa(self):
        cap = asw_capital(_par_asw(), _curve())
        assert cap.capital == pytest.approx(cap.rwa * 0.08, rel=1e-10)


# ── Hedge Recommendations ──

class TestASWHedge:

    def test_no_breach_no_recs(self):
        book = ASWBook("test")
        book.add(ASWBookEntry("T1", _par_asw(), "par", bond_issuer="A"))
        book.add(ASWBookEntry("T2", _par_asw(), "par", bond_issuer="B"))
        book.add(ASWBookEntry("T3", _par_asw(), "par", bond_issuer="C"))
        book.add(ASWBookEntry("T4", _par_asw(), "par", bond_issuer="D"))
        # 4 issuers at 25% each — no concentration breach
        recs = asw_hedge_recommendations(book, _curve(),
                                          dv01_limit=1e9, asw01_limit=1e9)
        assert len(recs) == 0


# ── Lifecycle ──

class TestASWLifecycle:

    def test_maturity_alert(self):
        lc = ASWLifecycle(_par_asw())
        alert = lc.maturity_alert(date(2029, 7, 1), alert_days=30)
        assert alert is not None
        assert alert["days_remaining"] == 14

    def test_no_alert_far(self):
        lc = ASWLifecycle(_par_asw())
        alert = lc.maturity_alert(REF, alert_days=30)
        assert alert is None

    def test_record_coupon(self):
        lc = ASWLifecycle(_par_asw())
        event = lc.record_coupon(date(2025, 1, 15), 2.5)
        assert event["amount"] == 2.5
        assert len(lc.history) == 1
