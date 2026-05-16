"""Tests for commodity desk consolidation: dashboard, stress, hedge, lifecycle."""

from __future__ import annotations

import math
from datetime import date

from dateutil.relativedelta import relativedelta

from pricebook.desks.commodity_desk import (
    CommodityRiskMetrics,
    commodity_dashboard, CommodityDashboard,
    commodity_stress_suite, CommodityStressResult,
    commodity_hedge_recommendations, CommodityHedgeRecommendation,
    CommodityLifecycle,
)
from pricebook.desks.commodity_book import CommodityBook
from pricebook.core.trade import Trade
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)


# ── Dashboard ──

class TestCommodityDashboard:

    def test_dashboard_fields(self):
        book = CommodityBook("Energy", REF)
        book.add(Trade(instrument=None, trade_id="C1"),
                commodity="WTI", sector="energy", quantity=1_000, reference_price=75.0)
        db = commodity_dashboard(book, REF)
        assert db.n_positions >= 1
        assert "WTI" in db.by_commodity

    def test_to_dict(self):
        book = CommodityBook("Energy", REF)
        book.add(Trade(instrument=None, trade_id="C1"),
                commodity="WTI", sector="energy", quantity=1_000, reference_price=75.0)
        d = commodity_dashboard(book, REF).to_dict()
        assert "by_commodity" in d
        assert "delta" in d


# ── Stress ──

class TestCommodityStress:

    def test_five_scenarios(self):
        results = commodity_stress_suite(total_delta=10_000_000)
        assert len(results) == 5

    def test_price_down_negative_for_long(self):
        results = commodity_stress_suite(total_delta=10_000_000)
        dn = [r for r in results if r.scenario == "price_dn_10"][0]
        assert dn.pnl < 0

    def test_price_up_positive_for_long(self):
        results = commodity_stress_suite(total_delta=10_000_000)
        up = [r for r in results if r.scenario == "price_up_10"][0]
        assert up.pnl > 0


# ── Hedge recs ──

class TestCommodityHedge:

    def test_no_recs_within_limits(self):
        book = CommodityBook("Energy", REF)
        book.add(Trade(instrument=None, trade_id="C1"),
                commodity="WTI", sector="energy", quantity=10, reference_price=75.0)
        recs = commodity_hedge_recommendations(book, delta_limit=1e12, concentration_pct=1.1)
        assert len(recs) == 0

    def test_rec_format(self):
        rec = CommodityHedgeRecommendation("delta", 100_000, 50_000, 2.0, "Reduce via futures")
        d = rec.to_dict()
        assert "risk" in d
        assert "action" in d


# ── Lifecycle ──

class TestCommodityLifecycle:

    def test_delivery_alert(self):
        class FakeSwap:
            end = REF + relativedelta(days=3)
        lc = CommodityLifecycle(FakeSwap(), "C1")
        alert = lc.delivery_alert(REF, alert_days=5)
        assert alert is not None
        assert alert["days_remaining"] == 3

    def test_no_alert_far(self):
        class FakeSwap:
            end = REF + relativedelta(months=6)
        lc = CommodityLifecycle(FakeSwap(), "C1")
        assert lc.delivery_alert(REF, alert_days=5) is None

    def test_record_roll(self):
        class FakeSwap:
            end = REF + relativedelta(months=3)
        lc = CommodityLifecycle(FakeSwap(), "C1")
        ev = lc.record_roll(REF, REF + relativedelta(months=6), roll_cost=500)
        assert ev["roll_cost"] == 500

    def test_record_nomination(self):
        class FakeSwap:
            end = REF + relativedelta(months=3)
        lc = CommodityLifecycle(FakeSwap(), "C1")
        ev = lc.record_nomination(REF, volume=10_000)
        assert ev["volume"] == 10_000

    def test_history_ordered(self):
        class FakeSwap:
            end = REF + relativedelta(months=6)
        lc = CommodityLifecycle(FakeSwap(), "C1")
        lc.record_nomination(REF + relativedelta(months=1), 5_000)
        lc.record_roll(REF + relativedelta(months=3), REF + relativedelta(months=9))
        dates = [h["date"] for h in lc.history]
        assert dates == sorted(dates)
