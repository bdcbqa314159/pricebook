"""Tests for inflation desk consolidation: dashboard, stress, hedge, lifecycle."""

from __future__ import annotations

import math
from datetime import date

from dateutil.relativedelta import relativedelta

from pricebook.inflation_desk import (
    InflationRiskMetrics,
    InflationDashboard,
    inflation_stress_suite, InflationStressResult,
    inflation_hedge_recommendations, InflationHedgeRecommendation,
    InflationLifecycle,
)

REF = date(2024, 7, 15)


# ── Risk metrics ──

class TestInflationRiskMetrics:

    def test_to_dict(self):
        rm = InflationRiskMetrics(pv=10_000_000, ie01=5000, real_dv01=3000,
                                   nominal_dv01=8000, notional=50_000_000)
        d = rm.to_dict()
        assert "ie01" in d
        assert "real_dv01" in d


# ── Stress ──

class TestInflationStress:

    def test_five_scenarios(self):
        results = inflation_stress_suite(total_ie01=5_000, total_real_dv01=3_000)
        assert len(results) == 5

    def test_breakeven_up_positive_for_long_inflation(self):
        """Long breakeven: breakeven up → gain."""
        results = inflation_stress_suite(total_ie01=5_000)
        up = [r for r in results if r.scenario == "breakeven_up_50"][0]
        assert up.pnl > 0

    def test_breakeven_down_negative(self):
        results = inflation_stress_suite(total_ie01=5_000)
        dn = [r for r in results if r.scenario == "breakeven_dn_50"][0]
        assert dn.pnl < 0

    def test_to_dict(self):
        results = inflation_stress_suite(total_ie01=5_000)
        d = results[0].to_dict()
        assert "scenario" in d
        assert "pnl" in d


# ── Hedge recs ──

class TestInflationHedge:

    def test_no_recs_within_limits(self):
        recs = inflation_hedge_recommendations(total_ie01=100, ie01_limit=1e12)
        assert len(recs) == 0

    def test_recs_when_breached(self):
        recs = inflation_hedge_recommendations(total_ie01=100_000, ie01_limit=1_000)
        assert len(recs) >= 1
        assert recs[0].action != ""

    def test_rec_format(self):
        rec = InflationHedgeRecommendation("ie01", 100_000, 50_000, 2.0, "Reduce IE01")
        d = rec.to_dict()
        assert "risk" in d
        assert "action" in d


# ── Lifecycle ──

class TestInflationLifecycle:

    def test_maturity_alert(self):
        class FakeLinker:
            maturity = REF + relativedelta(days=20)
        lc = InflationLifecycle(FakeLinker(), "TIPS1")
        alert = lc.maturity_alert(REF, alert_days=30)
        assert alert is not None
        assert alert["days_remaining"] == 20

    def test_no_maturity_alert_far(self):
        class FakeLinker:
            maturity = REF + relativedelta(years=5)
        lc = InflationLifecycle(FakeLinker(), "TIPS1")
        assert lc.maturity_alert(REF, alert_days=30) is None

    def test_record_cpi_fixing(self):
        class FakeSwap:
            end = REF + relativedelta(years=5)
        lc = InflationLifecycle(FakeSwap(), "ZC1")
        ev = lc.record_cpi_fixing(REF + relativedelta(months=2), 310.5)
        assert ev["cpi"] == 310.5

    def test_history_ordered(self):
        class FakeSwap:
            end = REF + relativedelta(years=5)
        lc = InflationLifecycle(FakeSwap(), "ZC1")
        lc.record_cpi_fixing(REF + relativedelta(months=2), 310.5)
        lc.record_cpi_fixing(REF + relativedelta(months=5), 312.0)
        dates = [h["date"] for h in lc.history]
        assert dates == sorted(dates)
