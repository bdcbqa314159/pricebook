"""Tests for CDS trading desk: risk, book, carry, P&L, dashboard, stress, capital, lifecycle."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.cds import CDS
from pricebook.cds_desk import (
    cds_risk_metrics, CDSRiskMetrics,
    CDSBook, CDSBookEntry, CDSProductType,
    cds_carry_decomposition, CDSCarryDecomposition,
    cds_daily_pnl, CDSDailyPnL,
    cds_dashboard, CDSDashboard,
    cds_stress_suite, CDSStressResult,
    cds_scenario_stress,
    cds_capital, CDSCapitalResult,
    cds_hedge_recommendations, CDSHedgeRecommendation,
    CDSLifecycle,
)
from pricebook.pricing_context import PricingContext
from pricebook.survival_curve import SurvivalCurve
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 7, 15)
END = REF + relativedelta(years=5)


def _cds_buy(spread=0.01):
    return CDS(REF, END, spread=spread, notional=10_000_000, recovery=0.4)


def _cds_sell(spread=0.01):
    return CDS(REF, END, spread=spread, notional=-10_000_000, recovery=0.4)


def _surv(h=0.02):
    return make_flat_survival(REF, h)


# ── Risk metrics + L11 ──

class TestCDSRiskMetrics:

    def test_l11_par_spread(self):
        """Par spread for 2% hazard, 40% recovery ≈ 120bp."""
        cds = _cds_buy()
        curve = make_flat_curve(REF, 0.04)
        surv = _surv(0.02)
        rm = cds_risk_metrics(cds, curve, surv)
        # par ≈ h × (1-R) = 0.02 × 0.6 = 0.012
        assert abs(rm.par_spread - 0.012) < 0.002

    def test_cs01_positive_for_protection_buyer(self):
        """Buying protection: spreads widen → position gains → CS01 > 0."""
        cds = _cds_buy()
        curve = make_flat_curve(REF, 0.04)
        rm = cds_risk_metrics(cds, curve, _surv())
        assert rm.cs01 > 0

    def test_rec01_negative_for_protection_buyer(self):
        """Higher recovery → less protection value → rec01 < 0."""
        cds = _cds_buy()
        curve = make_flat_curve(REF, 0.04)
        rm = cds_risk_metrics(cds, curve, _surv())
        assert rm.rec01 < 0

    def test_carry_positive(self):
        """Below par spread → protection is cheap → positive carry."""
        cds = _cds_buy(spread=0.005)  # paying 50bp on 120bp par
        curve = make_flat_curve(REF, 0.04)
        rm = cds_risk_metrics(cds, curve, _surv())
        assert rm.carry > 0  # cheap protection earns carry

    def test_jtd_positive_for_buyer(self):
        """Protection buyer gains on default: (1-R)×N - PV > 0 when PV > 0."""
        cds = _cds_buy(spread=0.005)
        curve = make_flat_curve(REF, 0.04)
        rm = cds_risk_metrics(cds, curve, _surv())
        assert rm.jump_to_default > 0

    def test_bucket_cs01_is_dict(self):
        cds = _cds_buy()
        curve = make_flat_curve(REF, 0.04)
        rm = cds_risk_metrics(cds, curve, _surv())
        assert isinstance(rm.bucket_cs01, dict)

    def test_to_dict(self):
        cds = _cds_buy()
        curve = make_flat_curve(REF, 0.04)
        rm = cds_risk_metrics(cds, curve, _surv())
        d = rm.to_dict()
        assert "cs01" in d
        assert "jtd" in d
        assert "par_spread" in d


# ── Book ──

class TestCDSBook:

    def test_add_and_count(self):
        book = CDSBook()
        surv = _surv()
        book.add(CDSBookEntry("C1", _cds_buy(), surv, reference_name="AAPL", sector="tech"))
        book.add(CDSBookEntry("C2", _cds_buy(), surv, reference_name="MSFT", sector="tech"))
        assert len(book) == 2

    def test_by_name(self):
        book = CDSBook()
        surv = _surv()
        book.add(CDSBookEntry("C1", _cds_buy(), surv, reference_name="AAPL"))
        book.add(CDSBookEntry("C2", _cds_buy(), surv, reference_name="MSFT"))
        bn = book.by_name()
        assert "AAPL" in bn
        assert "MSFT" in bn

    def test_by_sector(self):
        book = CDSBook()
        surv = _surv()
        book.add(CDSBookEntry("C1", _cds_buy(), surv, sector="tech"))
        book.add(CDSBookEntry("C2", _cds_buy(), surv, sector="financials"))
        bs = book.by_sector()
        assert "tech" in bs
        assert "financials" in bs

    def test_aggregate_risk(self):
        book = CDSBook()
        surv = _surv()
        book.add(CDSBookEntry("C1", _cds_buy(), surv, reference_name="AAPL"))
        curve = make_flat_curve(REF, 0.04)
        risk = book.aggregate_risk(curve)
        assert risk["n_positions"] == 1
        assert risk["total_cs01"] > 0
        assert "total_jtd" in risk


# ── Carry ──

class TestCDSCarry:

    def test_premium_positive(self):
        cds = _cds_buy()
        curve = make_flat_curve(REF, 0.04)
        cd = cds_carry_decomposition(cds, curve, _surv())
        assert cd.premium_income > 0

    def test_default_risk_negative(self):
        cds = _cds_buy()
        curve = make_flat_curve(REF, 0.04)
        cd = cds_carry_decomposition(cds, curve, _surv())
        assert cd.default_risk < 0

    def test_to_dict(self):
        cds = _cds_buy()
        curve = make_flat_curve(REF, 0.04)
        d = cds_carry_decomposition(cds, curve, _surv()).to_dict()
        assert "premium" in d
        assert "net" in d


# ── Daily P&L ──

class TestCDSDailyPnL:

    def test_unchanged_small(self):
        cds = _cds_buy()
        curve = make_flat_curve(REF, 0.04)
        surv = _surv()
        pnl = cds_daily_pnl(cds, curve, curve, surv, surv, REF + relativedelta(days=1))
        assert abs(pnl.total) < 100

    def test_spread_widening_positive_for_buyer(self):
        cds = _cds_buy()
        curve = make_flat_curve(REF, 0.04)
        surv_t0 = _surv(0.02)
        surv_t1 = _surv(0.04)  # spreads widen
        pnl = cds_daily_pnl(cds, curve, curve, surv_t0, surv_t1, REF + relativedelta(days=1))
        assert pnl.total > 0  # protection buyer gains

    def test_to_dict(self):
        cds = _cds_buy()
        curve = make_flat_curve(REF, 0.04)
        surv = _surv()
        d = cds_daily_pnl(cds, curve, curve, surv, surv, REF + relativedelta(days=1)).to_dict()
        assert "spread" in d
        assert "carry" in d


# ── Dashboard ──

class TestCDSDashboard:

    def test_dashboard_fields(self):
        book = CDSBook()
        surv = _surv()
        book.add(CDSBookEntry("C1", _cds_buy(), surv, reference_name="AAPL", sector="tech"))
        curve = make_flat_curve(REF, 0.04)
        db = cds_dashboard(book, REF, curve)
        assert db.n_positions == 1
        assert math.isfinite(db.total_cs01)
        assert "AAPL" in db.by_name

    def test_to_dict(self):
        book = CDSBook()
        surv = _surv()
        book.add(CDSBookEntry("C1", _cds_buy(), surv, reference_name="AAPL"))
        curve = make_flat_curve(REF, 0.04)
        d = cds_dashboard(book, REF, curve).to_dict()
        assert "cs01" in d
        assert "jtd" in d
        assert "by_name" in d


# ── Stress ──

class TestCDSStress:

    def test_five_scenarios(self):
        book = CDSBook()
        book.add(CDSBookEntry("C1", _cds_buy(), _surv()))
        curve = make_flat_curve(REF, 0.04)
        results = cds_stress_suite(book, curve)
        assert len(results) == 5

    def test_spread_wide_positive_for_buyer(self):
        book = CDSBook()
        book.add(CDSBookEntry("C1", _cds_buy(), _surv()))
        curve = make_flat_curve(REF, 0.04)
        results = cds_stress_suite(book, curve)
        wide = [r for r in results if r.scenario == "spread_wide_200"][0]
        assert wide.spread_pnl > 0  # buyer gains on wider spreads

    def test_scenario_stress(self):
        book = CDSBook()
        surv = _surv()
        book.add(CDSBookEntry("C1", _cds_buy(), surv))
        ctx = PricingContext(
            valuation_date=REF, discount_curve=make_flat_curve(REF, 0.04),
            credit_curves={"default": surv})
        results = cds_scenario_stress(book, ctx)
        assert len(results) == 4


# ── Capital ──

class TestCDSCapital:

    def test_capital_positive(self):
        curve = make_flat_curve(REF, 0.04)
        cap = cds_capital(_cds_buy(), curve, _surv())
        assert cap.ead > 0
        assert cap.capital > 0

    def test_capital_8pct_rwa(self):
        curve = make_flat_curve(REF, 0.04)
        cap = cds_capital(_cds_buy(), curve, _surv())
        assert abs(cap.capital - cap.rwa * 0.08) < 0.01

    def test_simm_positive(self):
        curve = make_flat_curve(REF, 0.04)
        cap = cds_capital(_cds_buy(), curve, _surv())
        assert cap.simm_im > 0


# ── Hedge recs ──

class TestCDSHedge:

    def test_no_recs_within_limits(self):
        book = CDSBook()
        book.add(CDSBookEntry("C1", _cds_buy(), _surv(), reference_name="AAPL"))
        curve = make_flat_curve(REF, 0.04)
        recs = cds_hedge_recommendations(book, curve,
            cs01_limit=1e12, jtd_limit=1e12, concentration_limit_pct=1.1)
        assert len(recs) == 0

    def test_recs_when_breached(self):
        book = CDSBook()
        book.add(CDSBookEntry("C1", _cds_buy(), _surv(), reference_name="AAPL"))
        curve = make_flat_curve(REF, 0.04)
        recs = cds_hedge_recommendations(book, curve, cs01_limit=0.001)
        assert len(recs) >= 1

    def test_concentration_breach(self):
        book = CDSBook()
        surv = _surv()
        book.add(CDSBookEntry("C1", _cds_buy(), surv, reference_name="AAPL"))
        curve = make_flat_curve(REF, 0.04)
        recs = cds_hedge_recommendations(book, curve,
            cs01_limit=1e12, jtd_limit=1e12, concentration_limit_pct=0.01)
        conc = [r for r in recs if r.risk_type == "concentration"]
        assert len(conc) >= 1


# ── Lifecycle ──

class TestCDSLifecycle:

    def test_credit_event_payout(self):
        cds = _cds_buy()
        surv = _surv()
        lc = CDSLifecycle(cds, surv, "C1", REF)
        curve = make_flat_curve(REF, 0.04)
        payout = lc.credit_event(REF + relativedelta(years=1), curve)
        assert payout == (1 - 0.4) * 10_000_000  # 6M

    def test_maturity_alert(self):
        cds = CDS(REF - relativedelta(years=5), REF + relativedelta(days=20),
                  spread=0.01, notional=10_000_000)
        lc = CDSLifecycle(cds, _surv(), "C1")
        alert = lc.maturity_alert(REF, alert_days=30)
        assert alert is not None

    def test_succession(self):
        cds = _cds_buy()
        lc = CDSLifecycle(cds, _surv(), "C1", REF)
        ev = lc.record_succession(REF + relativedelta(months=3), "AAPL_NEW")
        assert ev["new_reference"] == "AAPL_NEW"

    def test_history_ordered(self):
        cds = _cds_buy()
        lc = CDSLifecycle(cds, _surv(), "C1", REF)
        lc.record_succession(REF + relativedelta(months=3), "AAPL_NEW")
        hist = lc.history
        dates = [h["date"] for h in hist]
        assert dates == sorted(dates)
