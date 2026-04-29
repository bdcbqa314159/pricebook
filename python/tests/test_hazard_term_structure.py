"""Tests for Layer 2: hazard term structure, forward rates, proxy curves."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.cds import bootstrap_credit_curve
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.hazard_term_structure import (
    proxy_survival_curve, liquidity_spread, spread_from_survival, compare_curves,
)

REF = date(2026, 4, 28)
END_5Y = REF + timedelta(days=1825)


def _disc():
    return DiscountCurve.flat(REF, 0.03)


def _surv(hazard=0.02):
    return SurvivalCurve.flat(REF, hazard)


def _bootstrapped():
    disc = _disc()
    spreads = [(REF + timedelta(days=365*i), 0.005 + 0.001*i) for i in range(1, 6)]
    return bootstrap_credit_curve(REF, spreads, disc)


# ---- Forward hazard rates ----

class TestForwardHazard:

    def test_forward_hazard_positive(self):
        sc = _surv(0.02)
        d1 = REF + timedelta(days=365)
        d2 = REF + timedelta(days=730)
        fh = sc.forward_hazard(d1, d2)
        assert fh > 0
        assert fh == pytest.approx(0.02, abs=0.001)  # flat curve

    def test_forward_survival(self):
        sc = _surv(0.02)
        d1 = REF + timedelta(days=365)
        d2 = REF + timedelta(days=730)
        fs = sc.forward_survival(d1, d2)
        # Q(d2|d1) = Q(d2)/Q(d1) ≈ exp(-h × 1Y)
        assert fs == pytest.approx(math.exp(-0.02), abs=0.001)

    def test_forward_hazard_vs_spot(self):
        """Forward hazard from ref to d should equal spot hazard at d."""
        sc = _surv(0.02)
        d = REF + timedelta(days=1825)
        fh = sc.forward_hazard(REF, d)
        sh = sc.hazard_rate(d)
        assert fh == pytest.approx(sh, abs=0.001)

    def test_forward_hazard_term_structure(self):
        """Non-flat curve: forward hazard rates increase."""
        sc = _bootstrapped()
        d1 = REF + timedelta(days=365)
        d2 = REF + timedelta(days=730)
        d3 = REF + timedelta(days=1460)
        d4 = REF + timedelta(days=1825)
        fh_short = sc.forward_hazard(d1, d2)
        fh_long = sc.forward_hazard(d3, d4)
        # Spreads increase with tenor → forward hazard increases
        assert fh_long > fh_short

    def test_marginal_default_density(self):
        sc = _surv(0.02)
        d = REF + timedelta(days=365)
        mdd = sc.marginal_default_density(d)
        assert mdd > 0
        # f(t) = h × Q(t) ≈ 0.02 × exp(-0.02) ≈ 0.0196
        assert mdd == pytest.approx(0.02 * math.exp(-0.02), abs=0.001)

    def test_d1_after_d2_raises(self):
        sc = _surv()
        with pytest.raises(ValueError):
            sc.forward_hazard(REF + timedelta(days=730), REF + timedelta(days=365))


# ---- Term structure ----

class TestTermStructure:

    def test_term_structure_output(self):
        sc = _bootstrapped()
        ts = sc.term_structure()
        assert len(ts) == 5
        assert "date" in ts[0]
        assert "survival" in ts[0]
        assert "hazard_rate" in ts[0]

    def test_survival_decreasing(self):
        sc = _bootstrapped()
        ts = sc.term_structure()
        survivals = [t["survival"] for t in ts]
        for i in range(1, len(survivals)):
            assert survivals[i] < survivals[i-1]


# ---- Bumping ----

class TestBumping:

    def test_bumped_parallel(self):
        sc = _surv(0.02)
        bumped = sc.bumped(0.001)
        d = REF + timedelta(days=1825)
        assert bumped.survival(d) < sc.survival(d)  # higher hazard → lower survival

    def test_bumped_at_single_pillar(self):
        sc = _bootstrapped()
        bumped = sc.bumped_at(2, 0.001)  # bump 3rd pillar
        # Only pillar 2 and beyond should change
        d_before = sc._pillar_dates[1]
        d_at = sc._pillar_dates[2]
        assert bumped.survival(d_before) == pytest.approx(sc.survival(d_before), abs=1e-8)
        assert bumped.survival(d_at) < sc.survival(d_at)


# ---- Proxy curves ----

class TestProxyCurve:

    def test_additive_shift_widens(self):
        sc = _surv(0.02)
        proxy = proxy_survival_curve(sc, additive_shift=0.01)
        d = REF + timedelta(days=1825)
        assert proxy.survival(d) < sc.survival(d)

    def test_multiplicative_scale(self):
        sc = _surv(0.02)
        proxy = proxy_survival_curve(sc, multiplicative_scale=2.0)
        d = REF + timedelta(days=1825)
        # Double hazard → survival ≈ exp(-0.04 × 5) vs exp(-0.02 × 5)
        assert proxy.survival(d) < sc.survival(d)

    def test_no_shift_identical(self):
        sc = _surv(0.02)
        proxy = proxy_survival_curve(sc)
        d = REF + timedelta(days=1825)
        assert proxy.survival(d) == pytest.approx(sc.survival(d), abs=1e-6)

    def test_preserves_shape(self):
        """Additive shift preserves the relative shape of the curve."""
        sc = _bootstrapped()
        proxy = proxy_survival_curve(sc, additive_shift=0.005)
        # Both should have same pillar count
        assert len(proxy._pillar_dates) == len(sc._pillar_dates)


# ---- Liquidity ----

class TestLiquidity:

    def test_liquidity_positive(self):
        assert liquidity_spread(0.015, 0.010) == pytest.approx(0.005)

    def test_zero_liquidity(self):
        assert liquidity_spread(0.010, 0.010) == pytest.approx(0.0)

    def test_spread_from_survival(self):
        sc = _surv(0.02)
        s = spread_from_survival(sc, REF + timedelta(days=1825))
        # spread ≈ (1-R) × h = 0.6 × 0.02 = 0.012
        assert s == pytest.approx(0.012, abs=0.002)


# ---- Compare curves ----

class TestCompareCurves:

    def test_compare(self):
        a = _surv(0.02)
        b = _surv(0.03)
        result = compare_curves(a, b, "IG", "HY")
        assert len(result) > 0
        assert "survival_IG" in result[0]
        assert "survival_HY" in result[0]
        assert result[0]["spread_diff_bp"] < 0  # a has lower hazard
