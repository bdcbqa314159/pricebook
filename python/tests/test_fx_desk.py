"""Tests for FX desk consolidation: dashboard, stress, XVA, hedge, lifecycle."""

from __future__ import annotations

import math
from datetime import date

from dateutil.relativedelta import relativedelta

from pricebook.desks.fx_desk import (
    fx_risk_metrics, FXRiskMetrics,
    fx_dashboard, FXDashboard,
    fx_stress_suite, FXStressResult,
    fx_hedge_recommendations, FXHedgeRecommendation,
    FXLifecycle,
)
from pricebook.fx.fx_forward import FXForward
from pricebook.core.currency import Currency, CurrencyPair
from pricebook.desks.fx_book import FXBook
from pricebook.core.trade import Trade
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)
EUR_USD = CurrencyPair(Currency("EUR"), Currency("USD"))


def _fwd():
    return FXForward(
        pair=EUR_USD,
        maturity=REF + relativedelta(months=3),
        strike=1.08,
        notional=10_000_000,
    )


# ── Risk metrics ──

class TestFXRiskMetrics:

    def test_pv_finite(self):
        fwd = _fwd()
        usd = make_flat_curve(REF, 0.05)
        eur = make_flat_curve(REF, 0.03)
        rm = fx_risk_metrics(fwd, eur, usd, pair="EUR/USD")
        assert math.isfinite(rm.pv)

    def test_delta_nonzero(self):
        fwd = _fwd()
        usd = make_flat_curve(REF, 0.05)
        eur = make_flat_curve(REF, 0.03)
        rm = fx_risk_metrics(fwd, eur, usd, pair="EUR/USD")
        assert rm.spot_delta != 0 or rm.notional > 0

    def test_to_dict(self):
        fwd = _fwd()
        usd = make_flat_curve(REF, 0.05)
        eur = make_flat_curve(REF, 0.03)
        rm = fx_risk_metrics(fwd, eur, usd, pair="EUR/USD")
        d = rm.to_dict()
        assert "delta" in d
        assert "pair" in d


# ── Dashboard ──

class TestFXDashboard:

    def test_dashboard_fields(self):
        book = FXBook("G10", REF)
        book.add(Trade(instrument=_fwd()), pair="EUR/USD",
                notional=10_000_000, spot_rate=1.08)
        db = fx_dashboard(book, REF)
        assert db.n_positions >= 1
        assert "EUR/USD" in db.by_pair

    def test_to_dict(self):
        book = FXBook("G10", REF)
        book.add(Trade(instrument=_fwd()), pair="EUR/USD",
                notional=10_000_000, spot_rate=1.08)
        d = fx_dashboard(book, REF).to_dict()
        assert "by_pair" in d
        assert "by_currency" in d


# ── Stress ──

class TestFXStress:

    def test_five_scenarios(self):
        """Spot stress suite: 4 pure-spot scenarios (the previously-included
        'combined' scenario was a silent no-op on rates and was removed in
        v1.031). Use fx_scenario_stress() for full rate+spot reprice."""
        positions = [("EUR/USD", 10_000_000, 1.08)]
        results = fx_stress_suite(positions)
        assert len(results) == 4

    def test_spot_down_negative(self):
        """Long EUR: spot down → loss."""
        positions = [("EUR/USD", 10_000_000, 1.08)]
        results = fx_stress_suite(positions)
        dn = [r for r in results if r.scenario == "spot_dn_5"][0]
        assert dn.pnl < 0


# ── Hedge recs ──

class TestFXHedge:

    def test_no_recs_within_limits(self):
        book = FXBook("G10", REF)
        book.add(Trade(instrument=_fwd()), pair="EUR/USD",
                notional=1_000, spot_rate=1.08)
        recs = fx_hedge_recommendations(book, delta_limit=1e12, ccy_limit=1e12)
        assert len(recs) == 0

    def test_recs_when_breached(self):
        book = FXBook("G10", REF)
        book.add(Trade(instrument=_fwd()), pair="EUR/USD",
                notional=100_000_000, spot_rate=1.08)
        recs = fx_hedge_recommendations(book, delta_limit=1_000)
        assert len(recs) >= 1


# ── Lifecycle ──

class TestFXLifecycle:

    def test_settlement_alert(self):
        fwd = FXForward(EUR_USD, REF + relativedelta(days=1), 1.08, 10_000_000)
        lc = FXLifecycle(fwd, "FX1")
        alert = lc.settlement_alert(REF, alert_days=2)
        assert alert is not None
        assert alert["days_remaining"] == 1

    def test_no_alert_far(self):
        fwd = _fwd()
        lc = FXLifecycle(fwd, "FX1")
        alert = lc.settlement_alert(REF, alert_days=2)
        assert alert is None

    def test_record_settlement(self):
        lc = FXLifecycle(_fwd(), "FX1")
        ev = lc.record_settlement(REF + relativedelta(months=3), 1.09)
        assert ev["rate"] == 1.09

    def test_record_fixing(self):
        lc = FXLifecycle(_fwd(), "FX1")
        ev = lc.record_fixing(REF + relativedelta(months=3), 1.085)
        assert ev["rate"] == 1.085

    def test_record_roll(self):
        lc = FXLifecycle(_fwd(), "FX1")
        ev = lc.record_roll(REF, REF + relativedelta(months=6), roll_cost=150)
        assert ev["roll_cost"] == 150
        assert len(lc.history) == 1

    def test_history_ordered(self):
        lc = FXLifecycle(_fwd(), "FX1")
        lc.record_fixing(REF + relativedelta(months=1), 1.07)
        lc.record_settlement(REF + relativedelta(months=3), 1.09)
        dates = [h["date"] for h in lc.history]
        assert dates == sorted(dates)
