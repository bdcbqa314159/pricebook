"""Tests for risky floating payments: L0-L3."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from pricebook.discount_curve import DiscountCurve
from pricebook.floating_leg import FloatingLeg
from pricebook.schedule import Frequency
from pricebook.survival_curve import SurvivalCurve
from pricebook.risky_floating import risky_floating_pv, CreditRiskyFRN
from pricebook.cra_curve import CRADiscountCurve
from pricebook.risky_floating_mc import price_risky_frn_mc
from pricebook.serialisable import from_dict

REF = date(2026, 4, 28)
END = REF + timedelta(days=1825)


def _disc():
    return DiscountCurve.flat(REF, 0.03)


def _surv(h=0.02):
    return SurvivalCurve.flat(REF, h)


# ---- Layer 0 ----

class TestRiskyFloatingPV:

    def test_zero_hazard_equals_riskfree(self):
        disc = _disc()
        sc = _surv(0.0001)
        leg = FloatingLeg(REF, END, Frequency.QUARTERLY, notional=1_000_000, spread=0.005)
        rf_pv = leg.pv(disc) + 1_000_000 * disc.df(END)
        risky = risky_floating_pv(leg, disc, None, sc, 1_000_000, 0.4)
        assert risky.total_pv == pytest.approx(rf_pv, rel=0.01)

    def test_high_hazard_lower_pv(self):
        disc = _disc()
        leg = FloatingLeg(REF, END, Frequency.QUARTERLY, notional=1_000_000, spread=0.005)
        r_low = risky_floating_pv(leg, disc, None, _surv(0.01), 1_000_000, 0.4)
        r_high = risky_floating_pv(leg, disc, None, _surv(0.10), 1_000_000, 0.4)
        assert r_high.total_pv < r_low.total_pv

    def test_recovery_increases_pv(self):
        disc = _disc()
        sc = _surv(0.05)
        leg = FloatingLeg(REF, END, Frequency.QUARTERLY, notional=1_000_000, spread=0.005)
        r_low = risky_floating_pv(leg, disc, None, sc, 1_000_000, 0.2)
        r_high = risky_floating_pv(leg, disc, None, sc, 1_000_000, 0.6)
        assert r_high.total_pv > r_low.total_pv

    def test_decomposition(self):
        disc = _disc()
        sc = _surv(0.03)
        leg = FloatingLeg(REF, END, Frequency.QUARTERLY, notional=1_000_000, spread=0.005)
        r = risky_floating_pv(leg, disc, None, sc, 1_000_000, 0.4)
        recon = r.coupon_pv + r.accrued_on_default + r.principal_pv + r.recovery_pv
        assert r.total_pv == pytest.approx(recon)


# ---- Layer 1 ----

class TestCreditRiskyFRN:

    def test_no_credit_risk(self):
        frn = CreditRiskyFRN(REF, END, spread=0.005)
        assert frn.dirty_price(_disc()) > 0

    def test_credit_lowers_price(self):
        frn = CreditRiskyFRN(REF, END, spread=0.005)
        rf = frn.dirty_price(_disc())
        risky = frn.dirty_price(_disc(), survival_curve=_surv(0.05))
        assert risky < rf

    def test_higher_recovery(self):
        frn_low = CreditRiskyFRN(REF, END, spread=0.005, recovery=0.2)
        frn_high = CreditRiskyFRN(REF, END, spread=0.005, recovery=0.6)
        sc = _surv(0.05)
        assert frn_high.dirty_price(_disc(), survival_curve=sc) > \
               frn_low.dirty_price(_disc(), survival_curve=sc)

    def test_cs01(self):
        frn = CreditRiskyFRN(REF, END, spread=0.005)
        cs01 = frn.credit_spread_sensitivity(_disc(), _surv(0.03))
        assert cs01 < 0

    def test_serialisation(self):
        frn = CreditRiskyFRN(REF, END, spread=0.005, notional=5_000_000, recovery=0.35)
        d = frn.to_dict()
        assert d["type"] == "credit_risky_frn"
        frn2 = from_dict(d)
        assert frn2.spread == 0.005
        assert frn2.recovery == 0.35

    def test_pv_round_trip(self):
        frn = CreditRiskyFRN(REF, END, spread=0.005)
        p1 = frn.dirty_price(_disc(), survival_curve=_surv(0.03))
        frn2 = from_dict(frn.to_dict())
        p2 = frn2.dirty_price(_disc(), survival_curve=_surv(0.03))
        assert p1 == pytest.approx(p2, abs=1e-8)


# ---- Layer 2 ----

class TestCRACurve:

    def test_cra_df_less(self):
        cra = CRADiscountCurve(_disc(), _surv(0.02))
        assert cra.df(END) < _disc().df(END)

    def test_zero_hazard(self):
        cra = CRADiscountCurve(_disc(), _surv(0.0001))
        assert cra.df(END) == pytest.approx(_disc().df(END), rel=0.01)

    def test_recovery_increases_df(self):
        cra0 = CRADiscountCurve(_disc(), _surv(0.05), recovery=0.0)
        cra4 = CRADiscountCurve(_disc(), _surv(0.05), recovery=0.4)
        assert cra4.df(END) > cra0.df(END)

    def test_forward_differs(self):
        cra = CRADiscountCurve(_disc(), _surv(0.05))
        d1, d2 = REF + timedelta(days=365), REF + timedelta(days=730)
        assert cra.forward_rate(d1, d2) != pytest.approx(_disc().forward_rate(d1, d2), abs=0.001)

    def test_credit_adjustment_positive(self):
        cra = CRADiscountCurve(_disc(), _surv(0.03))
        assert cra.credit_adjustment(END) > 0

    def test_serialisation(self):
        cra = CRADiscountCurve(_disc(), _surv(0.02), recovery=0.4)
        d = cra.to_dict()
        assert d["type"] == "cra_curve"
        cra2 = from_dict(d)
        assert cra2.df(END) == pytest.approx(cra.df(END), rel=1e-6)


# ---- Layer 3 ----

class TestRiskyFloatingMC:

    def test_basic(self):
        frn = CreditRiskyFRN(REF, END, spread=0.005)
        r = price_risky_frn_mc(frn, 0.03, 0.02, n_paths=10_000)
        assert math.isfinite(r.price) and r.price > 0

    def test_zero_corr_near_independent(self):
        frn = CreditRiskyFRN(REF, END, spread=0.005)
        r = price_risky_frn_mc(frn, 0.03, 0.02, correlation=0.0, n_paths=50_000)
        assert abs(r.wrong_way_adjustment) < 3.0

    def test_wrong_way_sign(self):
        """Negative correlation: rates and credit move together → WWR."""
        frn = CreditRiskyFRN(REF, END, spread=0.005)
        r_neg = price_risky_frn_mc(frn, 0.03, 0.03, rate_vol=0.02,
                                    hazard_vol=0.30, correlation=-0.7, n_paths=100_000)
        r_pos = price_risky_frn_mc(frn, 0.03, 0.03, rate_vol=0.02,
                                    hazard_vol=0.30, correlation=0.7, n_paths=100_000)
        # Negative corr should give lower price than positive
        assert r_neg.price < r_pos.price

    def test_invalid_corr(self):
        frn = CreditRiskyFRN(REF, END, spread=0.005)
        with pytest.raises(ValueError, match="correlation"):
            price_risky_frn_mc(frn, 0.03, 0.02, correlation=1.5)

    def test_result_dict(self):
        frn = CreditRiskyFRN(REF, END, spread=0.005)
        r = price_risky_frn_mc(frn, 0.03, 0.02, n_paths=5_000)
        d = r.to_dict()
        assert "wrong_way_adjustment" in d
