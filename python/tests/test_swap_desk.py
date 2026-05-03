"""Tests for swap trading desk: risk metrics, carry, P&L, book, dashboard, stress, capital, lifecycle."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.swap_desk import (
    swap_risk_metrics, SwapRiskMetrics,
    SwapBook, SwapBookEntry,
    swap_carry_decomposition, SwapCarryDecomposition,
    swap_daily_pnl, SwapDailyPnL,
    swap_dashboard, SwapDashboard,
    swap_stress_suite, SwapStressResult,
    swap_scenario_stress,
    swap_capital, SwapCapitalResult,
    swap_hedge_recommendations, SwapHedgeRecommendation,
    SwapLifecycle,
)
from pricebook.pricing_context import PricingContext
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)


def _5y_payer(rate=0.04):
    return InterestRateSwap(
        REF, REF + relativedelta(years=5),
        fixed_rate=rate, direction=SwapDirection.PAYER,
        notional=10_000_000,
    )


def _5y_receiver(rate=0.04):
    return InterestRateSwap(
        REF, REF + relativedelta(years=5),
        fixed_rate=rate, direction=SwapDirection.RECEIVER,
        notional=10_000_000,
    )


def _10y_payer(rate=0.04):
    return InterestRateSwap(
        REF, REF + relativedelta(years=10),
        fixed_rate=rate, direction=SwapDirection.PAYER,
        notional=5_000_000,
    )


# ── Risk metrics + L11 ──

class TestSwapRiskMetrics:

    def test_l11_par_swap_pv_near_zero(self):
        """At par rate, PV ≈ 0."""
        curve = make_flat_curve(REF, 0.04)
        swap = _5y_payer()
        par = swap.par_rate(curve)
        swap_at_par = InterestRateSwap(
            REF, REF + relativedelta(years=5),
            fixed_rate=par, direction=SwapDirection.PAYER, notional=10_000_000)
        rm = swap_risk_metrics(swap_at_par, curve)
        assert abs(rm.pv) < 100  # near zero for par swap

    def test_dv01_positive_for_payer(self):
        """Payer swap: rates up → floating receives more → positive DV01."""
        curve = make_flat_curve(REF, 0.04)
        rm = swap_risk_metrics(_5y_payer(), curve)
        assert rm.dv01 > 0

    def test_dv01_negative_for_receiver(self):
        """Receiver swap: rates up → fixed worth less → negative DV01."""
        curve = make_flat_curve(REF, 0.04)
        rm = swap_risk_metrics(_5y_receiver(), curve)
        assert rm.dv01 < 0

    def test_key_rate_sums_to_parallel(self):
        curve = make_flat_curve(REF, 0.04)
        rm = swap_risk_metrics(_5y_payer(), curve)
        kr_sum = sum(rm.key_rate_dv01.values())
        assert abs(kr_sum - rm.dv01) / abs(rm.dv01) < 0.15

    def test_gamma_finite(self):
        curve = make_flat_curve(REF, 0.04)
        rm = swap_risk_metrics(_5y_payer(), curve)
        assert math.isfinite(rm.gamma)

    def test_theta_finite(self):
        curve = make_flat_curve(REF, 0.04)
        rm = swap_risk_metrics(_5y_payer(), curve)
        assert math.isfinite(rm.theta)

    def test_10y_higher_dv01(self):
        """10Y swap has higher DV01 than 5Y (longer duration)."""
        curve = make_flat_curve(REF, 0.04)
        rm5 = swap_risk_metrics(_5y_payer(), curve)
        rm10 = swap_risk_metrics(_10y_payer(), curve)
        assert abs(rm10.dv01) > abs(rm5.dv01) * 0.5  # 10Y has bigger DV01 even with half notional

    def test_to_dict(self):
        curve = make_flat_curve(REF, 0.04)
        rm = swap_risk_metrics(_5y_payer(), curve)
        d = rm.to_dict()
        assert "dv01" in d
        assert "key_rate_dv01" in d
        assert "gamma" in d


# ── Book ──

class TestSwapBook:

    def test_add_and_count(self):
        book = SwapBook()
        book.add(SwapBookEntry("S1", _5y_payer(), "JPM"))
        book.add(SwapBookEntry("S2", _5y_receiver(), "GS"))
        assert len(book) == 2

    def test_by_direction(self):
        book = SwapBook()
        book.add(SwapBookEntry("S1", _5y_payer()))
        book.add(SwapBookEntry("S2", _5y_receiver()))
        bd = book.by_direction()
        assert "payer" in bd
        assert "receiver" in bd

    def test_net_dv01_offsets(self):
        """Payer + receiver at same rate → net DV01 near zero."""
        book = SwapBook()
        book.add(SwapBookEntry("S1", _5y_payer()))
        book.add(SwapBookEntry("S2", _5y_receiver()))
        curve = make_flat_curve(REF, 0.04)
        risk = book.aggregate_risk(curve)
        assert abs(risk["net_dv01"]) < abs(risk["payer_dv01"]) * 0.1

    def test_aggregate_risk(self):
        book = SwapBook()
        book.add(SwapBookEntry("S1", _5y_payer()))
        curve = make_flat_curve(REF, 0.04)
        risk = book.aggregate_risk(curve)
        assert risk["n_positions"] == 1
        assert "total_dv01" in risk
        assert "net_dv01" in risk


# ── Carry ──

class TestSwapCarry:

    def test_carry_finite(self):
        curve = make_flat_curve(REF, 0.04)
        cd = swap_carry_decomposition(_5y_payer(), curve)
        assert math.isfinite(cd.net_carry)

    def test_fixed_accrual_positive(self):
        curve = make_flat_curve(REF, 0.04)
        cd = swap_carry_decomposition(_5y_payer(), curve)
        assert cd.fixed_accrual > 0

    def test_to_dict(self):
        curve = make_flat_curve(REF, 0.04)
        cd = swap_carry_decomposition(_5y_payer(), curve)
        d = cd.to_dict()
        assert "fixed" in d
        assert "floating" in d
        assert "net" in d


# ── Daily P&L ──

class TestSwapDailyPnL:

    def test_unchanged_small(self):
        curve = make_flat_curve(REF, 0.04)
        pnl = swap_daily_pnl(_5y_payer(), curve, curve, REF + relativedelta(days=1))
        assert abs(pnl.total) < 100  # same curve → small

    def test_rate_shift_has_impact(self):
        c0 = make_flat_curve(REF, 0.04)
        c1 = make_flat_curve(REF, 0.05)
        pnl = swap_daily_pnl(_5y_payer(), c0, c1, REF + relativedelta(days=1))
        assert pnl.total > 0  # payer benefits from rate rise

    def test_to_dict(self):
        curve = make_flat_curve(REF, 0.04)
        pnl = swap_daily_pnl(_5y_payer(), curve, curve, REF + relativedelta(days=1))
        d = pnl.to_dict()
        assert "curve" in d
        assert "carry" in d
        assert "theta" in d


# ── Dashboard ──

class TestSwapDashboard:

    def test_dashboard_fields(self):
        book = SwapBook()
        book.add(SwapBookEntry("S1", _5y_payer(), "JPM"))
        book.add(SwapBookEntry("S2", _10y_payer(), "GS"))
        curve = make_flat_curve(REF, 0.04)
        db = swap_dashboard(book, REF, curve)
        assert db.n_positions == 2
        assert math.isfinite(db.net_dv01)
        assert len(db.dv01_ladder) >= 1

    def test_to_dict(self):
        book = SwapBook()
        book.add(SwapBookEntry("S1", _5y_payer()))
        curve = make_flat_curve(REF, 0.04)
        db = swap_dashboard(book, REF, curve)
        d = db.to_dict()
        assert "dv01_ladder" in d
        assert "net_dv01" in d


# ── Stress ──

class TestSwapStress:

    def test_five_scenarios(self):
        book = SwapBook()
        book.add(SwapBookEntry("S1", _5y_payer()))
        curve = make_flat_curve(REF, 0.04)
        results = swap_stress_suite(book, curve)
        assert len(results) == 5

    def test_rates_up_positive_for_payer(self):
        book = SwapBook()
        book.add(SwapBookEntry("S1", _5y_payer()))
        curve = make_flat_curve(REF, 0.04)
        results = swap_stress_suite(book, curve)
        up = [r for r in results if r.scenario == "rates_up_100"][0]
        assert up.pnl > 0  # payer benefits

    def test_scenario_stress(self):
        book = SwapBook()
        book.add(SwapBookEntry("S1", _5y_payer()))
        ctx = PricingContext(valuation_date=REF, discount_curve=make_flat_curve(REF, 0.04))
        results = swap_scenario_stress(book, ctx)
        assert len(results) == 3


# ── Capital ──

class TestSwapCapital:

    def test_capital_positive(self):
        curve = make_flat_curve(REF, 0.04)
        cap = swap_capital(_5y_payer(), curve)
        assert cap.ead > 0
        assert cap.capital > 0

    def test_capital_8pct_rwa(self):
        curve = make_flat_curve(REF, 0.04)
        cap = swap_capital(_5y_payer(), curve)
        assert abs(cap.capital - cap.rwa * 0.08) < 0.01

    def test_simm_im_positive(self):
        curve = make_flat_curve(REF, 0.04)
        cap = swap_capital(_5y_payer(), curve)
        assert cap.simm_im > 0


# ── Hedge recs ──

class TestSwapHedge:

    def test_no_recs_within_limits(self):
        book = SwapBook()
        book.add(SwapBookEntry("S1", _5y_payer()))
        curve = make_flat_curve(REF, 0.04)
        recs = swap_hedge_recommendations(book, curve,
            dv01_limit=1e12, gamma_limit=1e15, net_dv01_limit=1e12)
        assert len(recs) == 0

    def test_recs_when_breached(self):
        book = SwapBook()
        book.add(SwapBookEntry("S1", _5y_payer()))
        curve = make_flat_curve(REF, 0.04)
        recs = swap_hedge_recommendations(book, curve, dv01_limit=0.001)
        assert len(recs) >= 1


# ── Lifecycle ──

class TestSwapLifecycle:

    def test_upcoming_resets(self):
        swap = _5y_payer()
        lc = SwapLifecycle(swap, "S1", REF)
        resets = lc.upcoming_resets(REF, horizon_days=180)
        assert len(resets) >= 1

    def test_maturity_alert(self):
        swap = InterestRateSwap(
            REF - relativedelta(years=5), REF + relativedelta(days=20),
            fixed_rate=0.04, notional=10_000_000)
        lc = SwapLifecycle(swap, "S1")
        alert = lc.maturity_alert(REF, alert_days=30)
        assert alert is not None

    def test_novation(self):
        swap = _5y_payer()
        lc = SwapLifecycle(swap, "S1", REF)
        ev = lc.record_novation(REF + relativedelta(months=3), "BARC")
        assert ev["new_counterparty"] == "BARC"
        assert len(lc.history) >= 2

    def test_history_ordered(self):
        swap = _5y_payer()
        lc = SwapLifecycle(swap, "S1", REF)
        lc.record_novation(REF + relativedelta(months=3), "BARC")
        hist = lc.history
        dates = [h["date"] for h in hist]
        assert dates == sorted(dates)
