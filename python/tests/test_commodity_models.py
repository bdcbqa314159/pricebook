"""Tests for commodity stochastic models."""

import math

import numpy as np
import pytest

from pricebook.commodity_models import (
    CommodityJumpDiffusion,
    CommodityJumpResult,
    GibsonSchwartz,
    GibsonSchwartzResult,
    SchwartzOneFactor,
    SchwartzOneFactorResult,
    SchwartzSmith,
    SchwartzSmithResult,
)


# ---- Schwartz one-factor ----

class TestSchwartzOneFactor:
    def test_basic_forward(self):
        model = SchwartzOneFactor(kappa=1.0, mu=math.log(100), sigma=0.25)
        F = model.forward_price(100, 1.0)
        assert F > 0

    def test_forward_at_zero_equals_spot(self):
        model = SchwartzOneFactor(kappa=1.0, mu=math.log(100), sigma=0.25)
        assert model.forward_price(100, 0.0) == 100

    def test_forward_reverts_to_long_run(self):
        """For very long T, forward approaches e^μ (up to vol adjustment)."""
        model = SchwartzOneFactor(kappa=2.0, mu=math.log(80), sigma=0.20)
        F_long = model.forward_price(100, 20.0)
        long_run = math.exp(math.log(80) + 0.20**2 / (4 * 2.0))
        assert F_long == pytest.approx(long_run, rel=0.05)

    def test_simulate_basic(self):
        model = SchwartzOneFactor(1.0, math.log(100), 0.25)
        result = model.simulate(100, 1.0, n_paths=1000, n_steps=50, seed=42)
        assert isinstance(result, SchwartzOneFactorResult)
        assert result.spot_paths.shape == (1000, 51)

    def test_simulate_mean_reversion(self):
        """High initial value → mean terminal should be lower (reverts down)."""
        model = SchwartzOneFactor(kappa=2.0, mu=math.log(50), sigma=0.10)
        result = model.simulate(200, 2.0, n_paths=2000, n_steps=100, seed=42)
        # After 2 years with κ=2, reversion is strong
        assert result.mean_terminal < 200

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            SchwartzOneFactor(kappa=0, mu=1.0, sigma=0.2)
        with pytest.raises(ValueError):
            SchwartzOneFactor(kappa=1.0, mu=1.0, sigma=0.0)

    def test_long_run_level(self):
        model = SchwartzOneFactor(1.0, math.log(80), 0.25)
        result = model.simulate(100, 1.0, n_paths=100, seed=42)
        assert result.long_run_level == pytest.approx(80, rel=1e-6)


# ---- Gibson-Schwartz ----

class TestGibsonSchwartz:
    def test_basic_forward(self):
        model = GibsonSchwartz(r=0.03, kappa=1.5, alpha=0.05,
                                sigma_s=0.30, sigma_delta=0.15, rho=-0.3)
        F = model.forward_price(spot=100, convenience_yield=0.04, T=1.0)
        assert F > 0

    def test_forward_at_zero_equals_spot(self):
        model = GibsonSchwartz(0.03, 1.5, 0.05, 0.30, 0.15, -0.3)
        assert model.forward_price(100, 0.04, 0.0) == 100

    def test_higher_convenience_lower_forward(self):
        """Higher convenience yield → lower forward (backwardation)."""
        model = GibsonSchwartz(0.03, 1.5, 0.05, 0.30, 0.15, -0.3)
        F_low = model.forward_price(100, 0.02, 1.0)
        F_high = model.forward_price(100, 0.08, 1.0)
        assert F_high < F_low

    def test_simulate_basic(self):
        model = GibsonSchwartz(0.03, 1.5, 0.05, 0.30, 0.15, -0.3)
        result = model.simulate(100, 0.04, 1.0, n_paths=1000, n_steps=50, seed=42)
        assert isinstance(result, GibsonSchwartzResult)
        assert result.spot_paths.shape == (1000, 51)
        assert result.convenience_yield_paths.shape == (1000, 51)

    def test_convenience_reverts_to_alpha(self):
        """Mean terminal convenience should be near α (with strong κ)."""
        model = GibsonSchwartz(0.03, 3.0, 0.05, 0.20, 0.10, 0.0)
        result = model.simulate(100, 0.10, 5.0, n_paths=3000, n_steps=100, seed=42)
        assert result.mean_terminal_delta == pytest.approx(0.05, abs=0.01)

    def test_invalid_rho(self):
        with pytest.raises(ValueError):
            GibsonSchwartz(0.03, 1.5, 0.05, 0.30, 0.15, rho=1.5)

    def test_invalid_kappa(self):
        with pytest.raises(ValueError):
            GibsonSchwartz(0.03, 0.0, 0.05, 0.30, 0.15, 0.0)


# ---- Schwartz-Smith ----

class TestSchwartzSmith:
    def test_basic_forward(self):
        model = SchwartzSmith(kappa=1.0, sigma_chi=0.20, mu_xi=0.02,
                               sigma_xi=0.15, rho=0.3)
        F = model.forward_price(chi0=0.05, xi0=math.log(100), T=1.0)
        assert F > 0

    def test_forward_at_zero(self):
        model = SchwartzSmith(1.0, 0.20, 0.02, 0.15, 0.3)
        # At T=0, forward = S₀ = exp(χ₀ + ξ₀)
        assert model.forward_price(0.0, math.log(100), 0.0) == pytest.approx(100)

    def test_simulate_basic(self):
        model = SchwartzSmith(1.0, 0.20, 0.02, 0.15, 0.3)
        result = model.simulate(chi0=0.0, xi0=math.log(100), T=1.0,
                                 n_paths=1000, n_steps=50, seed=42)
        assert isinstance(result, SchwartzSmithResult)
        assert result.spot_paths.shape == (1000, 51)

    def test_short_term_mean_reverts(self):
        """χ should revert to 0 (its long-run mean)."""
        model = SchwartzSmith(kappa=3.0, sigma_chi=0.05, mu_xi=0.0,
                               sigma_xi=0.01, rho=0.0)
        result = model.simulate(chi0=0.20, xi0=math.log(100), T=5.0,
                                 n_paths=2000, n_steps=100, seed=42)
        # Mean of chi at end should be near 0
        assert abs(result.short_term_paths[:, -1].mean()) < 0.05

    def test_long_term_drifts(self):
        """ξ should drift with μ_ξ × T."""
        model = SchwartzSmith(kappa=1.0, sigma_chi=0.01, mu_xi=0.10,
                               sigma_xi=0.01, rho=0.0)
        result = model.simulate(0.0, math.log(100), 2.0,
                                 n_paths=2000, n_steps=100, seed=42)
        # ξ at T=2 should be near log(100) + 0.10 × 2 = log(100) + 0.20
        expected = math.log(100) + 0.10 * 2.0
        assert result.long_term_paths[:, -1].mean() == pytest.approx(expected, abs=0.02)

    def test_decomposition(self):
        """spot = exp(chi + xi)."""
        model = SchwartzSmith(1.0, 0.15, 0.02, 0.10, 0.0)
        result = model.simulate(0.1, math.log(100), 1.0, n_paths=100, n_steps=30, seed=42)
        log_spot = result.short_term_paths + result.long_term_paths
        np.testing.assert_allclose(np.exp(log_spot), result.spot_paths, rtol=1e-10)


# ---- Jump-diffusion ----

class TestCommodityJumpDiffusion:
    def test_basic(self):
        model = CommodityJumpDiffusion(
            mu=0.03, sigma=0.25, lambda_jump=0.5,
            mu_jump=0.0, sigma_jump=0.10,
        )
        result = model.simulate(100, 1.0, n_paths=500, n_steps=50, seed=42)
        assert isinstance(result, CommodityJumpResult)

    def test_no_jumps(self):
        model = CommodityJumpDiffusion(0.03, 0.25, lambda_jump=0.0,
                                         mu_jump=0.0, sigma_jump=0.01)
        result = model.simulate(100, 1.0, n_paths=500, n_steps=50, seed=42)
        assert result.mean_jumps_per_path == 0.0

    def test_high_intensity_more_jumps(self):
        low_lambda = CommodityJumpDiffusion(0.03, 0.25, 0.5, 0.0, 0.05)
        high_lambda = CommodityJumpDiffusion(0.03, 0.25, 10.0, 0.0, 0.05)
        res_low = low_lambda.simulate(100, 1.0, n_paths=500, seed=42)
        res_high = high_lambda.simulate(100, 1.0, n_paths=500, seed=42)
        assert res_high.mean_jumps_per_path > res_low.mean_jumps_per_path

    def test_spike_model_mean_reversion(self):
        """With κ_mr > 0, price reverts toward μ_lr."""
        model = CommodityJumpDiffusion(
            mu=0.0, sigma=0.3, lambda_jump=2.0,
            mu_jump=0.0, sigma_jump=0.2,
            kappa_mr=2.0, mu_lr=math.log(50),
        )
        result = model.simulate(100, 2.0, n_paths=2000, n_steps=100, seed=42)
        # Terminal mean should be below starting 100 (reverts to 50)
        mean_term = result.spot_paths[:, -1].mean()
        assert mean_term < 100

    def test_paths_positive(self):
        model = CommodityJumpDiffusion(0.03, 0.25, 1.0, -0.05, 0.10)
        result = model.simulate(100, 1.0, n_paths=500, seed=42)
        assert np.all(result.spot_paths > 0)
