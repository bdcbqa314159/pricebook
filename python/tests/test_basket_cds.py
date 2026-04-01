"""Tests for basket CDS, Gaussian copula, and leveraged CLN."""

import pytest
from datetime import date

from pricebook.basket_cds import (
    simulate_defaults_copula,
    count_defaults,
    ftd_spread,
    ntd_spread,
    LeveragedCLN,
)
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)
T = 5.0


def _survs(n=5, hazard=0.02):
    return [make_flat_survival(REF, hazard) for _ in range(n)]


class TestGaussianCopula:
    def test_shape(self):
        defaults = simulate_defaults_copula(_survs(5), T, rho=0.3, n_sims=1000)
        assert defaults.shape == (1000, 5)

    def test_zero_correlation_independent(self):
        """Zero correlation: defaults are independent."""
        defaults = simulate_defaults_copula(_survs(5), T, rho=0.0, n_sims=50_000)
        # Each name defaults with probability ≈ 1 - exp(-0.02*5) ≈ 0.095
        per_name = defaults.mean(axis=0)
        # P(default) ≈ 1 - exp(-0.02*5) ≈ 0.095
        for p in per_name:
            assert 0.05 < p < 0.15

    def test_high_correlation_clusters(self):
        """High correlation: defaults cluster (more all-or-nothing)."""
        defaults_low = simulate_defaults_copula(_survs(5), T, rho=0.1, n_sims=50_000)
        defaults_high = simulate_defaults_copula(_survs(5), T, rho=0.9, n_sims=50_000)
        # With high correlation: more scenarios with 0 or all defaults
        n_low = count_defaults(defaults_low)
        n_high = count_defaults(defaults_high)
        # Variance of number of defaults should be higher with correlation
        assert n_high.var() > n_low.var()

    def test_higher_hazard_more_defaults(self):
        d1 = simulate_defaults_copula(_survs(5, 0.01), T, rho=0.3, n_sims=10_000)
        d2 = simulate_defaults_copula(_survs(5, 0.05), T, rho=0.3, n_sims=10_000)
        assert count_defaults(d2).mean() > count_defaults(d1).mean()


class TestFTDSpread:
    def test_positive(self):
        disc = make_flat_curve(REF, 0.04)
        s = ftd_spread(_survs(5), disc, rho=0.3, T=T)
        assert s > 0

    def test_ftd_less_than_sum(self):
        """FTD spread < sum of individual spreads."""
        disc = make_flat_curve(REF, 0.04)
        s_ftd = ftd_spread(_survs(5), disc, rho=0.3, T=T)
        # Individual par spread ≈ hazard * (1-R) ≈ 0.02 * 0.6 = 0.012
        assert s_ftd < 5 * 0.02

    def test_higher_correlation_lower_ftd(self):
        """Higher correlation → defaults cluster → less likely to see exactly 1 first → lower FTD spread."""
        disc = make_flat_curve(REF, 0.04)
        s_low = ftd_spread(_survs(5), disc, rho=0.1, T=T)
        s_high = ftd_spread(_survs(5), disc, rho=0.9, T=T)
        assert s_high < s_low


class TestNTDSpread:
    def test_ntd_decreasing_in_n(self):
        """Higher N → less likely to trigger → lower spread."""
        disc = make_flat_curve(REF, 0.04)
        s1 = ntd_spread(_survs(5), disc, rho=0.3, T=T, n=1)
        s2 = ntd_spread(_survs(5), disc, rho=0.3, T=T, n=2)
        s3 = ntd_spread(_survs(5), disc, rho=0.3, T=T, n=3)
        assert s1 > s2
        assert s2 > s3

    def test_first_equals_ftd(self):
        disc = make_flat_curve(REF, 0.04)
        s_ftd = ftd_spread(_survs(5), disc, rho=0.3, T=T, seed=42)
        s_1td = ntd_spread(_survs(5), disc, rho=0.3, T=T, n=1, seed=42)
        assert s_ftd == pytest.approx(s_1td, rel=0.01)

    def test_ntd_positive(self):
        disc = make_flat_curve(REF, 0.04)
        s = ntd_spread(_survs(5), disc, rho=0.3, T=T, n=2)
        assert s > 0


class TestLeveragedCLN:
    def test_pv_positive(self):
        disc = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.02)
        cln = LeveragedCLN(leverage=1.0, coupon_rate=0.06, T=5.0)
        assert cln.pv(disc, surv) > 0

    def test_higher_leverage_lower_pv(self):
        """More leverage → more default loss → lower PV."""
        disc = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.03)
        cln1 = LeveragedCLN(leverage=1.0, coupon_rate=0.06, T=5.0)
        cln3 = LeveragedCLN(leverage=3.0, coupon_rate=0.06, T=5.0)
        assert cln3.pv(disc, surv) < cln1.pv(disc, surv)

    def test_zero_hazard_no_leverage_effect(self):
        """No default → leverage doesn't matter."""
        disc = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.0001)
        cln1 = LeveragedCLN(leverage=1.0, coupon_rate=0.06, T=5.0)
        cln5 = LeveragedCLN(leverage=5.0, coupon_rate=0.06, T=5.0)
        assert cln1.pv(disc, surv) == pytest.approx(cln5.pv(disc, surv), rel=0.01)

    def test_higher_coupon_higher_pv(self):
        disc = make_flat_curve(REF, 0.04)
        surv = make_flat_survival(REF, 0.02)
        cln_low = LeveragedCLN(coupon_rate=0.04, T=5.0)
        cln_high = LeveragedCLN(coupon_rate=0.10, T=5.0)
        assert cln_high.pv(disc, surv) > cln_low.pv(disc, surv)
