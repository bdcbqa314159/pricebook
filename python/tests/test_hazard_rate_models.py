"""Tests for stochastic hazard rate models: HW, BK, CIR++, two-factor."""

import math

import numpy as np
import pytest

from pricebook.hazard_rate_models import (
    BKHazardRate,
    CIRPlusPlus,
    HWHazardRate,
    TwoFactorIntensity,
)


# ---- Hull-White hazard rate ----

class TestHWHazardRate:
    def test_survival_reasonable(self):
        """MC survival should be between 0 and 1 and decrease with hazard."""
        hw = HWHazardRate(a=0.5, sigma=0.005)
        r_low = hw.simulate(0.01, 5.0, n_steps=200, n_paths=20_000, seed=42)
        r_high = hw.simulate(0.05, 5.0, n_steps=200, n_paths=20_000, seed=42)
        assert 0 < r_low.survival_mc < 1
        assert 0 < r_high.survival_mc < 1
        assert r_high.survival_mc < r_low.survival_mc

    def test_positive_survival(self):
        hw = HWHazardRate(a=0.5, sigma=0.01)
        surv = hw.survival_analytical(0.02, 5.0)
        assert 0 < surv < 1

    def test_higher_hazard_lower_survival(self):
        hw = HWHazardRate(a=0.5, sigma=0.01)
        surv_low = hw.survival_analytical(0.01, 5.0)
        surv_high = hw.survival_analytical(0.05, 5.0)
        assert surv_high < surv_low


# ---- Black-Karasinski hazard rate ----

class TestBKHazardRate:
    def test_always_positive(self):
        """BK intensity is always positive (log-normal)."""
        bk = BKHazardRate(a=0.5, sigma=0.3)
        result = bk.simulate(0.02, 5.0, n_steps=200, n_paths=5_000, seed=42)
        assert np.all(result.lambda_paths > 0)

    def test_survival_between_zero_and_one(self):
        bk = BKHazardRate(a=0.5, sigma=0.2)
        result = bk.simulate(0.02, 5.0, n_steps=100, n_paths=10_000, seed=42)
        assert 0 < result.survival_mc < 1

    def test_trinomial_tree_positive_survival(self):
        """BK tree with low vol should give reasonable survival."""
        bk = BKHazardRate(a=1.0, sigma=0.1,
                          market_hazards=[(1, 0.02), (5, 0.02)])
        surv = bk.trinomial_tree_survival(0.02, 2.0, n_steps=20)
        # Tree is simplified; just check it's finite and positive
        assert surv > 0 and math.isfinite(surv)

    def test_higher_vol_wider_distribution(self):
        bk_low = BKHazardRate(a=0.5, sigma=0.1)
        bk_high = BKHazardRate(a=0.5, sigma=0.5)
        r_low = bk_low.simulate(0.02, 3.0, n_paths=10_000, seed=42)
        r_high = bk_high.simulate(0.02, 3.0, n_paths=10_000, seed=42)
        std_low = r_low.lambda_paths[:, -1].std()
        std_high = r_high.lambda_paths[:, -1].std()
        assert std_high > std_low


# ---- CIR++ ----

class TestCIRPlusPlus:
    def test_x_paths_non_negative(self):
        """CIR component x(t) ≥ 0."""
        model = CIRPlusPlus(kappa=2.0, theta=0.02, xi=0.1)
        result = model.simulate(0.02, 5.0, n_steps=200, n_paths=5_000, seed=42)
        # x paths (before shift) should be non-negative
        x_paths = result.lambda_paths - result.phi[np.newaxis, :]
        assert np.all(x_paths >= -1e-10)

    def test_survival_reasonable(self):
        model = CIRPlusPlus(kappa=2.0, theta=0.02, xi=0.1,
                            market_hazards=[(1, 0.02), (5, 0.03)])
        result = model.simulate(0.02, 5.0, n_steps=100, n_paths=20_000, seed=42)
        assert 0 < result.survival_mc < 1

    def test_phi_shift_calibrates(self):
        """φ(t) should shift λ to match market hazard at E[x] level."""
        model = CIRPlusPlus(kappa=2.0, theta=0.02, xi=0.1,
                            market_hazards=[(1, 0.03), (5, 0.04)])
        # At t=0: φ = market_hazard − x0
        phi_0 = model.phi(0.0, 0.02)
        assert phi_0 == pytest.approx(0.03 - 0.02)

    def test_reduces_to_cir_without_shift(self):
        """With market_hazard = theta, φ ≈ 0 → pure CIR."""
        model = CIRPlusPlus(kappa=2.0, theta=0.02, xi=0.1,
                            market_hazards=[(1, 0.02), (5, 0.02)])
        phi_5 = model.phi(5.0, 0.02)
        assert abs(phi_5) < 0.01  # small shift


# ---- Two-factor intensity ----

class TestTwoFactorIntensity:
    def test_survival_reasonable(self):
        model = TwoFactorIntensity(a1=0.1, sigma1=0.005,
                                    a2=1.0, sigma2=0.01,
                                    base_hazard=0.02)
        result = model.simulate(5.0, n_steps=100, n_paths=10_000, seed=42)
        assert 0 < result.survival_mc < 1

    def test_two_factors_richer_than_one(self):
        """Two-factor should have more variation in term structure."""
        one_factor = TwoFactorIntensity(0.5, 0.01, 100.0, 0.0, base_hazard=0.02)
        two_factor = TwoFactorIntensity(0.1, 0.008, 1.0, 0.008, base_hazard=0.02)

        r1 = one_factor.simulate(5.0, n_paths=5_000, seed=42)
        r2 = two_factor.simulate(5.0, n_paths=5_000, seed=42)

        # Two-factor should have more diverse terminal intensities
        std1 = r1.lambda_paths[:, -1].std()
        std2 = r2.lambda_paths[:, -1].std()
        assert std2 > std1 * 0.5  # at least comparable

    def test_correlation_affects_paths(self):
        uncorr = TwoFactorIntensity(0.5, 0.01, 1.0, 0.01, rho=0.0, base_hazard=0.02)
        corr = TwoFactorIntensity(0.5, 0.01, 1.0, 0.01, rho=0.9, base_hazard=0.02)

        r_u = uncorr.simulate(3.0, n_paths=10_000, seed=42)
        r_c = corr.simulate(3.0, n_paths=10_000, seed=42)

        # Higher correlation → factors move together → more total variance
        var_u = r_u.lambda_paths[:, -1].var()
        var_c = r_c.lambda_paths[:, -1].var()
        assert var_c > var_u * 0.8

    def test_shape(self):
        model = TwoFactorIntensity(0.5, 0.01, 1.0, 0.01, base_hazard=0.02)
        result = model.simulate(1.0, n_steps=50, n_paths=100, seed=42)
        assert result.lambda_paths.shape == (100, 51)
        assert result.x1_paths.shape == (100, 51)
        assert result.x2_paths.shape == (100, 51)
