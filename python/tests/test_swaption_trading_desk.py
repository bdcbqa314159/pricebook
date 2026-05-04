"""Tests for swaption trading desk: risk, book, dashboard, stress, capital, lifecycle."""

from __future__ import annotations

import math
from datetime import date

from dateutil.relativedelta import relativedelta

from pricebook.swaption import Swaption, SwaptionType
from pricebook.swaption_trading_desk import (
    swaption_risk_metrics, SwaptionRiskMetrics,
    SwaptionBook, SwaptionBookEntry,
    swaption_dashboard, SwaptionDashboard,
    swaption_stress_suite, SwaptionStressResult,
    swaption_capital, SwaptionCapitalResult,
    swaption_hedge_recommendations, SwaptionHedgeRecommendation,
    SwaptionLifecycle,
)
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)


def _payer_1y5y(strike=0.04):
    return Swaption(
        expiry=REF + relativedelta(years=1),
        swap_end=REF + relativedelta(years=6),
        strike=strike,
        swaption_type=SwaptionType.PAYER,
        notional=50_000_000,
    )


def _receiver_2y10y(strike=0.035):
    return Swaption(
        expiry=REF + relativedelta(years=2),
        swap_end=REF + relativedelta(years=12),
        strike=strike,
        swaption_type=SwaptionType.RECEIVER,
        notional=25_000_000,
    )


# ── Risk metrics ──

class TestSwaptionRiskMetrics:

    def test_pv_finite(self):
        curve = make_flat_curve(REF, 0.04)
        rm = swaption_risk_metrics(_payer_1y5y(), curve, vol=0.20)
        assert math.isfinite(rm.pv)
        assert rm.pv > 0  # ATM payer has positive value

    def test_delta_positive_for_payer(self):
        """Payer swaption: rates up → more valuable → positive delta."""
        curve = make_flat_curve(REF, 0.04)
        rm = swaption_risk_metrics(_payer_1y5y(), curve, vol=0.20)
        assert rm.delta > 0

    def test_vega_positive(self):
        curve = make_flat_curve(REF, 0.04)
        rm = swaption_risk_metrics(_payer_1y5y(), curve, vol=0.20)
        assert rm.vega > 0

    def test_to_dict(self):
        curve = make_flat_curve(REF, 0.04)
        rm = swaption_risk_metrics(_payer_1y5y(), curve, vol=0.20)
        d = rm.to_dict()
        assert "delta" in d
        assert "gamma" in d
        assert "vega" in d
        assert "tenor" in d


# ── Book ──

class TestSwaptionBook:

    def test_add_and_count(self):
        book = SwaptionBook()
        book.add(SwaptionBookEntry("SW1", _payer_1y5y(), 0.20, "JPM"))
        book.add(SwaptionBookEntry("SW2", _receiver_2y10y(), 0.18, "GS"))
        assert len(book) == 2

    def test_by_type(self):
        book = SwaptionBook()
        book.add(SwaptionBookEntry("SW1", _payer_1y5y(), 0.20))
        book.add(SwaptionBookEntry("SW2", _receiver_2y10y(), 0.18))
        bt = book.by_type()
        assert "payer" in bt
        assert "receiver" in bt

    def test_aggregate_risk(self):
        book = SwaptionBook()
        book.add(SwaptionBookEntry("SW1", _payer_1y5y(), 0.20))
        curve = make_flat_curve(REF, 0.04)
        risk = book.aggregate_risk(curve)
        assert risk["n_positions"] == 1
        assert risk["total_vega"] > 0


# ── Dashboard ──

class TestSwaptionDashboard:

    def test_dashboard_fields(self):
        book = SwaptionBook()
        book.add(SwaptionBookEntry("SW1", _payer_1y5y(), 0.20))
        book.add(SwaptionBookEntry("SW2", _receiver_2y10y(), 0.18))
        curve = make_flat_curve(REF, 0.04)
        db = swaption_dashboard(book, REF, curve)
        assert db.n_positions == 2
        assert len(db.vega_ladder) >= 1

    def test_to_dict(self):
        book = SwaptionBook()
        book.add(SwaptionBookEntry("SW1", _payer_1y5y(), 0.20))
        curve = make_flat_curve(REF, 0.04)
        d = swaption_dashboard(book, REF, curve).to_dict()
        assert "vega_ladder" in d
        assert "vega" in d


# ── Stress ──

class TestSwaptionStress:

    def test_five_scenarios(self):
        book = SwaptionBook()
        book.add(SwaptionBookEntry("SW1", _payer_1y5y(), 0.20))
        curve = make_flat_curve(REF, 0.04)
        results = swaption_stress_suite(book, curve)
        assert len(results) == 5

    def test_vol_up_positive_for_long(self):
        """Long swaption: vol up → more valuable."""
        book = SwaptionBook()
        book.add(SwaptionBookEntry("SW1", _payer_1y5y(), 0.20))
        curve = make_flat_curve(REF, 0.04)
        results = swaption_stress_suite(book, curve)
        vol_up = [r for r in results if r.scenario == "vol_up_5"][0]
        assert vol_up.pnl > 0


# ── Capital ──

class TestSwaptionCapital:

    def test_capital_positive(self):
        curve = make_flat_curve(REF, 0.04)
        cap = swaption_capital(_payer_1y5y(), curve, 0.20)
        assert cap.ead > 0
        assert cap.capital > 0

    def test_capital_8pct_rwa(self):
        curve = make_flat_curve(REF, 0.04)
        cap = swaption_capital(_payer_1y5y(), curve, 0.20)
        assert abs(cap.capital - cap.rwa * 0.08) < 0.01

    def test_simm_positive(self):
        curve = make_flat_curve(REF, 0.04)
        cap = swaption_capital(_payer_1y5y(), curve, 0.20)
        assert cap.simm_im >= 0


# ── Hedge recs ──

class TestSwaptionHedge:

    def test_no_recs_within_limits(self):
        book = SwaptionBook()
        book.add(SwaptionBookEntry("SW1", _payer_1y5y(), 0.20))
        curve = make_flat_curve(REF, 0.04)
        recs = swaption_hedge_recommendations(book, curve,
            delta_limit=1e12, gamma_limit=1e15, vega_limit=1e12)
        assert len(recs) == 0

    def test_recs_when_vega_breached(self):
        book = SwaptionBook()
        book.add(SwaptionBookEntry("SW1", _payer_1y5y(), 0.20))
        curve = make_flat_curve(REF, 0.04)
        recs = swaption_hedge_recommendations(book, curve, vega_limit=0.001)
        assert len(recs) >= 1


# ── Lifecycle ──

class TestSwaptionLifecycle:

    def test_expiry_alert(self):
        swn = Swaption(
            expiry=REF + relativedelta(days=3),
            swap_end=REF + relativedelta(years=5, days=3),
            strike=0.04, swaption_type=SwaptionType.PAYER, notional=50_000_000)
        lc = SwaptionLifecycle(swn, "SW1")
        alert = lc.expiry_alert(REF, alert_days=5)
        assert alert is not None
        assert alert["days_remaining"] == 3

    def test_no_alert_far(self):
        lc = SwaptionLifecycle(_payer_1y5y(), "SW1")
        assert lc.expiry_alert(REF, alert_days=5) is None

    def test_record_exercise(self):
        lc = SwaptionLifecycle(_payer_1y5y(), "SW1")
        ev = lc.record_exercise(REF + relativedelta(years=1), 0.042)
        assert ev["swap_rate"] == 0.042

    def test_record_lapse(self):
        lc = SwaptionLifecycle(_payer_1y5y(), "SW1")
        ev = lc.record_lapse(REF + relativedelta(years=1))
        assert ev["type"] == "lapse"

    def test_history_ordered(self):
        lc = SwaptionLifecycle(_payer_1y5y(), "SW1")
        lc.record_exercise(REF + relativedelta(years=1), 0.042)
        dates = [h["date"] for h in lc.history]
        assert dates == sorted(dates)
