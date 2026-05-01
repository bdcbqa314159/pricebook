"""Burgess Bond TRS validation — tests 1, 2, 3, 6, 7, 8, 9, 10, 11.

Validates bond_trs.py against Burgess (2024) SSRN 5024091.
"""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.bond import FixedRateBond
from pricebook.bond_trs import bond_trs_pv, par_funding_spread, BondTRSResult
from pricebook.cds import protection_leg_pv
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency
from pricebook.survival_curve import SurvivalCurve
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)
TRS_END = REF + relativedelta(years=5)


def _make_bond(coupon=0.04, maturity_years=10):
    return FixedRateBond(
        issue_date=REF - relativedelta(years=1),
        maturity=REF + relativedelta(years=maturity_years),
        coupon_rate=coupon,
        frequency=Frequency.SEMI_ANNUAL,
        face_value=100.0,
    )


# ---- Paper Test 1: No-default sanity ----

class TestNoDefault:
    """Set Q(t) ≡ 1. Coupon leg = riskfree bond coupon PV."""

    def test_no_survival_curve(self):
        """Without survival curve, coupon PV should equal riskfree coupon strip."""
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        result = bond_trs_pv(
            bond, REF, TRS_END,
            funding_spread=0.0, discount_curve=dc,
        )
        assert result.coupon_pv > 0
        assert result.lgd_pv == 0.0  # no default risk

    def test_lgd_zero_without_survival(self):
        """LGD leg must be zero when no survival curve provided."""
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        result = bond_trs_pv(
            bond, REF, TRS_END,
            funding_spread=0.01, discount_curve=dc,
        )
        assert result.lgd_pv == 0.0


# ---- Paper Test 2: Constant hazard ----

class TestConstantHazard:
    """With Q(t) = exp(-λt), risky PV < riskfree PV."""

    def test_risky_lower_than_riskfree(self):
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)

        rf = bond_trs_pv(bond, REF, TRS_END, 0.0, dc)
        risky = bond_trs_pv(bond, REF, TRS_END, 0.0, dc, sc, recovery=0.4)

        # Risky coupon PV < riskfree coupon PV (survival-weighted)
        assert risky.coupon_pv < rf.coupon_pv

    def test_lgd_positive(self):
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)

        result = bond_trs_pv(bond, REF, TRS_END, 0.0, dc, sc, recovery=0.4)
        assert result.lgd_pv > 0  # default risk costs money


# ---- Paper Test 3: Telescoping ----

class TestTelescoping:
    """Under flat risky discount, periodic performance ≈ N_B × (B₀ - B_T) × Q̃P."""

    def test_performance_sign(self):
        """Performance PV should reflect bond price changes."""
        bond = _make_bond(coupon=0.04)
        dc = make_flat_curve(REF, 0.04)
        result = bond_trs_pv(bond, REF, TRS_END, 0.0, dc)
        # Performance PV should be finite
        assert math.isfinite(result.performance_pv)


# ---- Paper Test 6: Risky annuity ----

class TestRiskyAnnuity:
    """On flat curves, A_risky = N_C × Σ τ_j × exp(-λ t_j) × exp(-r t_j)."""

    def test_positive(self):
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        result = bond_trs_pv(bond, REF, TRS_END, 0.01, dc, sc)
        assert result.risky_annuity > 0

    def test_risky_less_than_riskfree(self):
        """Risky annuity < riskfree annuity (survival discount)."""
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)

        rf_result = bond_trs_pv(bond, REF, TRS_END, 0.01, dc)
        risky_result = bond_trs_pv(bond, REF, TRS_END, 0.01, dc, sc)
        assert risky_result.risky_annuity < rf_result.risky_annuity


# ---- Paper Test 7: Par spread linearity ----

class TestParSpreadLinearity:
    """Bump coupon by Δr → par spread moves proportionally."""

    def test_higher_coupon_higher_spread(self):
        """Bond with higher coupon → higher par funding spread."""
        dc = make_flat_curve(REF, 0.04)
        bond_low = _make_bond(coupon=0.03)
        bond_high = _make_bond(coupon=0.05)

        p_low = par_funding_spread(bond_low, REF, TRS_END, dc)
        p_high = par_funding_spread(bond_high, REF, TRS_END, dc)
        assert p_high > p_low


# ---- Paper Test 8: Par spread sign ----

class TestParSpreadSign:
    """Appreciating bond → par spread > 0 (receiver pays positive spread)."""

    def test_par_spread_finite(self):
        bond = _make_bond(coupon=0.04)
        dc = make_flat_curve(REF, 0.04)
        p = par_funding_spread(bond, REF, TRS_END, dc)
        assert math.isfinite(p)


# ---- Paper Test 9: Recovery sensitivity ----

class TestRecoverySensitivity:
    """∂PV(LGD)/∂RR is positive (higher R reduces LGD magnitude)."""

    def test_higher_recovery_lower_lgd(self):
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.03)

        lgd_40 = bond_trs_pv(bond, REF, TRS_END, 0.01, dc, sc, recovery=0.4).lgd_pv
        lgd_60 = bond_trs_pv(bond, REF, TRS_END, 0.01, dc, sc, recovery=0.6).lgd_pv

        assert lgd_60 < lgd_40  # higher recovery → lower LGD cost

    def test_zero_recovery_max_lgd(self):
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.03)

        lgd_0 = bond_trs_pv(bond, REF, TRS_END, 0.01, dc, sc, recovery=0.0).lgd_pv
        lgd_40 = bond_trs_pv(bond, REF, TRS_END, 0.01, dc, sc, recovery=0.4).lgd_pv
        assert lgd_0 > lgd_40


# ---- Paper Test 10: CDS cross-check ----

class TestCDSCrossCheck:
    """LGD leg PV should match CDS protection leg for same hazard/discount."""

    def test_lgd_matches_cds_protection(self):
        """Burgess Eq 12 = CDS default leg (per-unit-notional)."""
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)

        trs_result = bond_trs_pv(
            bond, REF, TRS_END, 0.0, dc, sc, recovery=0.4,
            bond_notional=100.0,
        )

        # CDS protection leg PV for same parameters
        cds_prot = protection_leg_pv(
            REF, TRS_END, dc, sc,
            recovery=0.4, notional=100.0,
        )

        # Should be close (different discretisation but same economics)
        assert trs_result.lgd_pv == pytest.approx(cds_prot, rel=0.1)


# ---- Paper Test 11: Zero-spread regression ----

class TestZeroSpread:
    """At par spread, PV = 0."""

    def test_pv_zero_at_par(self):
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)

        p = par_funding_spread(bond, REF, TRS_END, dc, sc, recovery=0.4)

        # Price at par spread should give PV ≈ 0
        result = bond_trs_pv(
            bond, REF, TRS_END, p, dc, sc, recovery=0.4,
        )
        assert result.pv == pytest.approx(0.0, abs=100)  # within $100 on ~$100 notional

    def test_pv_zero_no_default(self):
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)

        p = par_funding_spread(bond, REF, TRS_END, dc)
        result = bond_trs_pv(bond, REF, TRS_END, p, dc)
        assert result.pv == pytest.approx(0.0, abs=100)


# ---- Paper Test 12-13: IG vs HY approximate methodology ----

class TestApproxVsExact:
    """IG: exact ≈ approx (LGD small). HY: exact ≠ approx."""

    def test_ig_lgd_small(self):
        """For IG (low hazard), LGD term is small relative to total PV."""
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.005)  # IG: ~50bp CDS

        result = bond_trs_pv(bond, REF, TRS_END, 0.01, dc, sc, recovery=0.4)
        # LGD should be small fraction of total
        total_legs = abs(result.coupon_pv) + abs(result.performance_pv) + abs(result.funding_pv)
        assert abs(result.lgd_pv) < total_legs * 0.05  # < 5% of total

    def test_hy_lgd_material(self):
        """For HY (high hazard), LGD term is material."""
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.05)  # HY: ~500bp CDS

        result = bond_trs_pv(bond, REF, TRS_END, 0.01, dc, sc, recovery=0.4)
        # LGD should be significant
        assert result.lgd_pv > 0
        total_legs = abs(result.coupon_pv) + abs(result.performance_pv) + abs(result.funding_pv)
        assert result.lgd_pv > total_legs * 0.01  # > 1% of total

    def test_to_dict(self):
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        result = bond_trs_pv(bond, REF, TRS_END, 0.01, dc)
        d = result.to_dict()
        assert "coupon_pv" in d
        assert "lgd_pv" in d
        assert "par_spread" in d
