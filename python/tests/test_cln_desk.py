"""Tests for CLN desk: risk metrics, carry, P&L, book, dashboard, stress, lifecycle."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.cln import CreditLinkedNote
from pricebook.cln_desk import (
    cln_risk_metrics, CLNRiskMetrics,
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
