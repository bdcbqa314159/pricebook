"""Tests for bond trading desk: risk metrics, carry, dashboard, stress, lifecycle."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.bond import FixedRateBond
from pricebook.bond_trading_desk import (
    bond_risk_metrics, BondRiskMetrics,
    bond_carry_roll, BondCarryRollDecomposition,
    bond_dashboard, BondDashboard,
    bond_stress_suite, BondStressResult,
    bond_scenario_stress,
    bond_hedge_recommendations, BondHedgeRecommendation,
    bond_funding_cost, BondFundingCost,
    BondLifecycle,
)
from pricebook.pricing_context import PricingContext
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)


def _5y_bond():
    return FixedRateBond.treasury_note(REF, REF + relativedelta(years=5), 0.04)


def _10y_bond():
    return FixedRateBond.treasury_note(REF, REF + relativedelta(years=10), 0.04125)


# ── Risk metrics + L11 ──

class TestBondRiskMetrics:

    def test_l11_5y_at_par(self):
        """L11: 5Y 4% semi-annual at ~par on 4% flat curve.
        Modified duration ≈ 4.49, DV01 ≈ 0.045."""
        bond = _5y_bond()
        curve = make_flat_curve(REF, 0.04)
        rm = bond_risk_metrics(bond, curve, REF)
        assert abs(rm.modified_duration - 4.49) < 0.05
        assert abs(rm.dv01 - 0.045) < 0.005

    def test_effective_duration_matches_modified(self):
        """For a bullet bond, effective duration ≈ modified duration."""
        bond = _5y_bond()
        curve = make_flat_curve(REF, 0.04)
        rm = bond_risk_metrics(bond, curve, REF)
        assert abs(rm.effective_duration - rm.modified_duration) < 0.1

    def test_convexity_positive(self):
        bond = _5y_bond()
        curve = make_flat_curve(REF, 0.04)
        rm = bond_risk_metrics(bond, curve, REF)
        assert rm.convexity > 0

    def test_dv01_negative(self):
        """Curve-based DV01: rates up → price down."""
        bond = _5y_bond()
        curve = make_flat_curve(REF, 0.04)
        rm = bond_risk_metrics(bond, curve, REF)
        assert rm.dv01_curve < 0

    def test_key_rate_dv01_sums_to_parallel(self):
        """Sum of key-rate DV01s ≈ parallel DV01."""
        bond = _5y_bond()
        curve = make_flat_curve(REF, 0.04)
        rm = bond_risk_metrics(bond, curve, REF)
        kr_sum = sum(rm.key_rate_dv01.values())
        assert abs(kr_sum - rm.dv01_curve) / abs(rm.dv01_curve) < 0.10

    def test_key_rate_has_multiple_tenors(self):
        bond = _5y_bond()
        curve = make_flat_curve(REF, 0.04)
        rm = bond_risk_metrics(bond, curve, REF)
        assert len(rm.key_rate_dv01) >= 3

    def test_10y_longer_duration(self):
        """10Y bond has longer duration than 5Y."""
        curve = make_flat_curve(REF, 0.04)
        rm5 = bond_risk_metrics(_5y_bond(), curve, REF)
        rm10 = bond_risk_metrics(_10y_bond(), curve, REF)
        assert rm10.modified_duration > rm5.modified_duration

    def test_ytm_near_coupon_at_par(self):
        """At par, YTM ≈ coupon rate."""
        bond = _5y_bond()
        curve = make_flat_curve(REF, 0.04)
        rm = bond_risk_metrics(bond, curve, REF)
        assert abs(rm.ytm - 0.04) < 0.005

    def test_to_dict(self):
        bond = _5y_bond()
        curve = make_flat_curve(REF, 0.04)
        rm = bond_risk_metrics(bond, curve, REF)
        d = rm.to_dict()
        assert "mod_dur" in d
        assert "key_rate_dv01" in d
        assert "eff_dur" in d
        assert "convexity" in d


# ── Carry-and-roll ──

class TestCarryRoll:

    def test_coupon_carry_positive(self):
        bond = _5y_bond()
        curve = make_flat_curve(REF, 0.04)
        cr = bond_carry_roll(bond, curve, repo_rate=0.04, horizon_days=30)
        assert cr.coupon_carry > 0

    def test_net_carry_sign(self):
        """At par with repo = coupon rate, net carry ≈ 0."""
        bond = _5y_bond()
        curve = make_flat_curve(REF, 0.04)
        cr = bond_carry_roll(bond, curve, repo_rate=0.04, horizon_days=30)
        assert abs(cr.net_carry) < 500  # small for 100k face

    def test_roll_down_finite(self):
        bond = _5y_bond()
        curve = make_flat_curve(REF, 0.04)
        cr = bond_carry_roll(bond, curve, horizon_days=30)
        assert math.isfinite(cr.roll_down_return)

    def test_to_dict(self):
        bond = _5y_bond()
        curve = make_flat_curve(REF, 0.04)
        cr = bond_carry_roll(bond, curve, horizon_days=30)
        d = cr.to_dict()
        assert "coupon" in d
        assert "roll_down" in d
        assert "total" in d


# ── Dashboard ──

class TestBondDashboard:

    def _positions(self):
        return [
            ("T1", "UST", _5y_bond(), 10_000_000),
            ("T2", "UST", _10y_bond(), 5_000_000),
        ]

    def test_dashboard_fields(self):
        curve = make_flat_curve(REF, 0.04)
        db = bond_dashboard(self._positions(), REF, curve)
        assert db.n_positions == 2
        assert db.total_face == 15_000_000
        assert math.isfinite(db.weighted_duration)
        assert db.weighted_duration > 0

    def test_top_positions(self):
        curve = make_flat_curve(REF, 0.04)
        db = bond_dashboard(self._positions(), REF, curve)
        assert len(db.top_positions) <= 5

    def test_to_dict(self):
        curve = make_flat_curve(REF, 0.04)
        db = bond_dashboard(self._positions(), REF, curve)
        d = db.to_dict()
        assert "duration" in d
        assert "by_tenor" in d


# ── Stress testing ──

class TestBondStress:

    def _pos(self):
        return [("T1", _5y_bond(), 10_000_000)]

    def test_five_scenarios(self):
        curve = make_flat_curve(REF, 0.04)
        results = bond_stress_suite(self._pos(), curve)
        assert len(results) == 5

    def test_rates_up_negative_pnl(self):
        """Rates up → long bond loses."""
        curve = make_flat_curve(REF, 0.04)
        results = bond_stress_suite(self._pos(), curve)
        up = [r for r in results if r.scenario == "rates_up_100"][0]
        assert up.total_pnl < 0

    def test_scenario_full_reprice(self):
        ctx = PricingContext(valuation_date=REF, discount_curve=make_flat_curve(REF, 0.04))
        results = bond_scenario_stress(self._pos(), ctx)
        assert len(results) == 3

    def test_to_dict(self):
        curve = make_flat_curve(REF, 0.04)
        results = bond_stress_suite(self._pos(), curve)
        d = results[0].to_dict()
        assert "rate" in d
        assert "total" in d


# ── Hedge recommendations ──

class TestBondHedge:

    def _pos(self):
        return [("T1", _5y_bond(), 10_000_000)]

    def test_no_recs_within_limits(self):
        curve = make_flat_curve(REF, 0.04)
        recs = bond_hedge_recommendations(self._pos(), curve,
            dv01_limit=1e12, duration_limit=100)
        assert len(recs) == 0

    def test_recs_when_breached(self):
        curve = make_flat_curve(REF, 0.04)
        recs = bond_hedge_recommendations(self._pos(), curve, dv01_limit=0.001)
        assert len(recs) >= 1


# ── Funding cost ──

class TestBondFundingCost:

    def test_breakeven_repo(self):
        """Breakeven repo ≈ coupon / price for par bond."""
        bond = _5y_bond()
        fc = bond_funding_cost(bond, 100.0, 0.04, settlement=REF)
        # breakeven = coupon / (price/100 * face) = 0.04*100 / (1.0*100) = 0.04
        assert abs(fc.breakeven_repo - 0.04) < 0.001

    def test_net_income_sign(self):
        """Funding above coupon rate → negative net income."""
        bond = _5y_bond()
        fc = bond_funding_cost(bond, 100.0, 0.06)
        assert fc.net_income < 0

    def test_to_dict(self):
        bond = _5y_bond()
        fc = bond_funding_cost(bond, 100.0, 0.04)
        d = fc.to_dict()
        assert "breakeven_repo" in d
        assert "all_in_yield" in d


# ── Lifecycle ──

class TestBondLifecycle:

    def test_upcoming_coupons(self):
        bond = _5y_bond()
        lc = BondLifecycle(bond, "T1", REF)
        events = lc.upcoming_events(REF, horizon_days=200)
        assert len(events) >= 1
        assert events[0]["type"] == "coupon"

    def test_maturity_alert(self):
        # Bond maturing in 20 days
        bond = FixedRateBond.treasury_note(REF - relativedelta(years=5),
            REF + relativedelta(days=20), 0.04)
        lc = BondLifecycle(bond, "T1", REF)
        alert = lc.maturity_alert(REF, alert_days=30)
        assert alert is not None
        assert alert["days_remaining"] == 20

    def test_no_maturity_alert_far(self):
        bond = _5y_bond()
        lc = BondLifecycle(bond, "T1", REF)
        alert = lc.maturity_alert(REF, alert_days=30)
        assert alert is None

    def test_process_coupon_records(self):
        bond = _5y_bond()
        lc = BondLifecycle(bond, "T1", REF)
        ev = lc.process_coupon(REF + relativedelta(months=6), 2000.0)
        assert ev["amount"] == 2000.0
        assert len(lc.history) >= 2  # created + coupon

    def test_history_ordered(self):
        bond = _5y_bond()
        lc = BondLifecycle(bond, "T1", REF)
        lc.process_coupon(REF + relativedelta(months=6), 2000.0)
        hist = lc.history
        dates = [h["date"] for h in hist]
        assert dates == sorted(dates)
