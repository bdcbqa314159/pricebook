"""Tests for illiquid credit: distressed, recovery sensitivity, EM conventions."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.cds import CDS
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.illiquid_credit import (
    distressed_recovery_sensitivity, is_distressed,
    implied_default_prob, recovery_breakeven,
    RESTRUCTURING_CLAUSES, EM_CONVENTIONS,
)

REF = date(2026, 4, 28)
END = REF + timedelta(days=1825)


def _disc():
    return DiscountCurve.flat(REF, 0.03)


def _surv(h=0.05):
    return SurvivalCurve.flat(REF, h)


class TestDistressed:

    def test_recovery_sensitivity(self):
        cds = CDS(REF, END, spread=0.05, notional=10_000_000, recovery=0.4)
        sens = distressed_recovery_sensitivity(cds, _disc(), _surv())
        assert math.isfinite(sens.base_pv)
        assert math.isfinite(sens.recovery_dv01)
        assert len(sens.pv_at_recoveries) == 16

    def test_recovery_dv01_sign(self):
        """Higher recovery → lower protection value → negative dv01 for buyer."""
        cds = CDS(REF, END, spread=0.05, notional=10_000_000, recovery=0.4)
        sens = distressed_recovery_sensitivity(cds, _disc(), _surv())
        # Protection buyer: higher recovery means less payout on default
        assert sens.recovery_dv01 < 0

    def test_pv_varies_with_recovery(self):
        cds = CDS(REF, END, spread=0.05, notional=10_000_000, recovery=0.4)
        sens = distressed_recovery_sensitivity(cds, _disc(), _surv())
        pvs = list(sens.pv_at_recoveries.values())
        assert max(pvs) > min(pvs)  # PV should vary

    def test_is_distressed(self):
        assert is_distressed(600)
        assert not is_distressed(200)

    def test_implied_default_prob(self):
        p = implied_default_prob(0.01, recovery=0.4, maturity_years=5)
        assert 0 < p < 1
        # Higher spread → higher default prob
        p_high = implied_default_prob(0.10, recovery=0.4, maturity_years=5)
        assert p_high > p

    def test_recovery_breakeven(self):
        cds = CDS(REF, END, spread=0.02, notional=10_000_000, recovery=0.4)
        r_be = recovery_breakeven(cds, _disc(), _surv(0.03))
        assert 0 < r_be < 1
        # At breakeven recovery, PV should be near zero
        cds_be = CDS(REF, END, spread=0.02, notional=10_000_000, recovery=r_be)
        pv = cds_be.pv(_disc(), _surv(0.03))
        assert abs(pv) < 100  # near zero for 10M notional


class TestConventions:

    def test_restructuring_clauses(self):
        assert "CR" in RESTRUCTURING_CLAUSES
        assert "XR" in RESTRUCTURING_CLAUSES

    def test_em_conventions(self):
        assert EM_CONVENTIONS["standard"]["coupon_bps"] == 100
        assert EM_CONVENTIONS["hy_sovereign"]["recovery"] == 0.25

    def test_result_dict(self):
        cds = CDS(REF, END, spread=0.05, notional=10_000_000, recovery=0.4)
        sens = distressed_recovery_sensitivity(cds, _disc(), _surv())
        d = sens.to_dict()
        assert "recovery_dv01" in d
        assert "pv_at_recoveries" in d
