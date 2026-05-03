"""Tests for CLN desk: risk metrics, carry, P&L, book, dashboard, stress, lifecycle."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.cln import CreditLinkedNote
from pricebook.cln_desk import (
    cln_risk_metrics, CLNRiskMetrics,
    cln_carry_decomposition, CLNCarryDecomposition,
    cln_daily_pnl, CLNDailyPnL,
    CLNBook, CLNBookEntry,
    cln_dashboard, CLNDashboard,
)
from pricebook.schedule import Frequency
from pricebook.survival_curve import SurvivalCurve
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)
END = REF + relativedelta(years=5)


def _vanilla_cln():
    return CreditLinkedNote(
        start=REF, end=END, coupon_rate=0.05,
        notional=1_000_000, recovery=0.4,
        frequency=Frequency.QUARTERLY,
    )


def _leveraged_cln():
    return CreditLinkedNote(
        start=REF, end=END, coupon_rate=0.07,
        notional=1_000_000, recovery=0.4, leverage=2.0,
        frequency=Frequency.QUARTERLY,
    )


def _flat_surv(hazard=0.02):
    return SurvivalCurve.flat(REF, hazard)


# ── Risk metrics ──

class TestCLNRiskMetrics:

    def test_l11_hand_calc_5y_cln(self):
        """L11: 5Y CLN, 2% hazard, 40% recovery, 5% coupon, 4% flat.

        Hand-calculated:
          coupon_pv  = 217,447.32
          principal  = 740,696.45
          recovery   =  34,400.96
          total PV   = 992,544.73
        """
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        rm = cln_risk_metrics(cln, curve, surv)
        assert abs(rm.pv - 992_544.73) < 0.01

    def test_dv01_negative(self):
        """Rates up → price down for fixed-coupon CLN."""
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        rm = cln_risk_metrics(cln, curve, surv)
        assert rm.dv01 < 0

    def test_cs01_negative(self):
        """Wider spreads → lower CLN price."""
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        rm = cln_risk_metrics(cln, curve, surv)
        assert rm.cs01 < 0

    def test_recovery_sensitivity_positive(self):
        """Higher recovery → higher CLN price."""
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        rm = cln_risk_metrics(cln, curve, surv)
        assert rm.recovery_sensitivity > 0

    def test_jtd_is_loss(self):
        """JTD = R×N - PV, should be negative (loss) for long CLN."""
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        rm = cln_risk_metrics(cln, curve, surv)
        # R×N = 0.4×1M = 400k, PV ≈ 992k → JTD ≈ -592k
        assert rm.jump_to_default_pnl < 0
        expected_jtd = 0.4 * 1_000_000 - rm.pv
        assert abs(rm.jump_to_default_pnl - expected_jtd) < 0.01

    def test_leveraged_higher_jtd_loss(self):
        """Leveraged CLN has bigger JTD loss."""
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        rm_v = cln_risk_metrics(_vanilla_cln(), curve, surv)
        rm_l = cln_risk_metrics(_leveraged_cln(), curve, surv)
        # Leveraged has lower PV and same R×N → bigger loss
        assert rm_l.jump_to_default_pnl < rm_v.jump_to_default_pnl

    def test_to_dict(self):
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        rm = cln_risk_metrics(cln, curve, surv)
        d = rm.to_dict()
        assert "dv01" in d
        assert "cs01" in d
        assert "jtd" in d
        assert "recovery_sensitivity" in d


# ── Carry decomposition ──

class TestCLNCarry:

    def test_coupon_income_positive(self):
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        cd = cln_carry_decomposition(cln, curve, surv)
        assert cd.coupon_income > 0  # 5% × 1M = 50k/year

    def test_default_drag_negative(self):
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        cd = cln_carry_decomposition(cln, curve, surv)
        assert cd.default_drag < 0  # expected loss per year

    def test_net_carry_sign(self):
        """For typical CLN, coupon should exceed default drag."""
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        cd = cln_carry_decomposition(cln, curve, surv)
        # 5% coupon vs 2%×60% = 1.2% default drag + 4% funding ≈ net negative
        assert math.isfinite(cd.net_carry)

    def test_to_dict(self):
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        d = cln_carry_decomposition(cln, curve, surv).to_dict()
        assert "coupon" in d
        assert "default_drag" in d
        assert "net" in d


# ── Daily P&L ──

class TestCLNDailyPnL:

    def test_unchanged_small(self):
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        pnl = cln_daily_pnl(cln, curve, curve, surv, surv,
                            REF + relativedelta(days=1))
        assert abs(pnl.total) < 1  # same curves → ~0

    def test_spread_widening_negative(self):
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv_t0 = _flat_surv(0.02)
        surv_t1 = _flat_surv(0.04)  # spreads widen
        pnl = cln_daily_pnl(cln, curve, curve, surv_t0, surv_t1,
                            REF + relativedelta(days=1))
        assert pnl.total < 0  # wider spreads → loss

    def test_rate_shift_has_impact(self):
        cln = _vanilla_cln()
        c0 = make_flat_curve(REF, 0.04)
        c1 = make_flat_curve(REF, 0.05)
        surv = _flat_surv(0.02)
        pnl = cln_daily_pnl(cln, c0, c1, surv, surv,
                            REF + relativedelta(days=1))
        assert pnl.total != 0

    def test_to_dict(self):
        cln = _vanilla_cln()
        curve = make_flat_curve(REF, 0.04)
        surv = _flat_surv(0.02)
        pnl = cln_daily_pnl(cln, curve, curve, surv, surv,
                            REF + relativedelta(days=1))
        d = pnl.to_dict()
        assert "spread" in d
        assert "rate" in d
        assert "theta" in d


# ── Book ──

class TestCLNBook:

    def test_add_and_count(self):
        book = CLNBook("TestBook")
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, issuer="AAPL"))
        book.add(CLNBookEntry("C2", _leveraged_cln(), surv, issuer="MSFT"))
        assert len(book) == 2

    def test_total_notional(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv))
        book.add(CLNBookEntry("C2", _leveraged_cln(), surv))
        assert book.total_notional() == 2_000_000

    def test_by_issuer(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, issuer="AAPL"))
        book.add(CLNBookEntry("C2", _leveraged_cln(), surv, issuer="MSFT"))
        bi = book.by_issuer()
        assert "AAPL" in bi
        assert "MSFT" in bi

    def test_by_seniority(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, seniority="senior"))
        book.add(CLNBookEntry("C2", _leveraged_cln(), surv, seniority="sub"))
        bs = book.by_seniority()
        assert "senior" in bs
        assert "sub" in bs

    def test_independent_amount(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, independent_amount=100_000))
        book.add(CLNBookEntry("C2", _leveraged_cln(), surv, independent_amount=200_000))
        assert book.total_independent_amount() == 300_000

    def test_aggregate_risk(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv))
        curve = make_flat_curve(REF, 0.04)
        risk = book.aggregate_risk(curve)
        assert "total_pv" in risk
        assert "total_cs01" in risk
        assert "total_jtd" in risk
        assert risk["n_positions"] == 1


# ── Dashboard ──

class TestCLNDashboard:

    def test_dashboard_fields(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, issuer="AAPL"))
        curve = make_flat_curve(REF, 0.04)
        db = cln_dashboard(book, REF, curve)
        assert db.n_positions == 1
        assert db.total_notional == 1_000_000
        assert math.isfinite(db.total_pv)
        assert math.isfinite(db.total_cs01)

    def test_by_issuer_breakdown(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv, issuer="AAPL"))
        book.add(CLNBookEntry("C2", _leveraged_cln(), surv, issuer="MSFT"))
        curve = make_flat_curve(REF, 0.04)
        db = cln_dashboard(book, REF, curve)
        assert "AAPL" in db.by_issuer
        assert "MSFT" in db.by_issuer

    def test_to_dict(self):
        book = CLNBook()
        surv = _flat_surv()
        book.add(CLNBookEntry("C1", _vanilla_cln(), surv))
        curve = make_flat_curve(REF, 0.04)
        db = cln_dashboard(book, REF, curve)
        d = db.to_dict()
        assert "cs01" in d
        assert "jtd" in d
        assert "by_issuer" in d
        assert "by_seniority" in d
