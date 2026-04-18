"""Tests for stochastic correlation."""

import math
import numpy as np
import pytest

from pricebook.stochastic_correlation import (
    CIRCorrelation,
    CIRCorrelationResult,
    DispersionCalibrationResult,
    StochCorrPricingResult,
    WishartCovariance,
    WishartResult,
    calibrate_stoch_corr_to_dispersion,
    simulate_two_asset_stoch_corr,
)


# ---- CIR correlation ----

class TestCIRCorrelation:
    def test_basic(self):
        model = CIRCorrelation(rho0=0.5, kappa=2.0, theta=0.3, sigma=0.2)
        result = model.simulate(1.0, n_paths=500, n_steps=50, seed=42)
        assert isinstance(result, CIRCorrelationResult)

    def test_rho_bounded(self):
        """ρ must stay in (-1, 1)."""
        model = CIRCorrelation(0.5, 2.0, 0.3, 0.5)
        result = model.simulate(2.0, n_paths=1000, n_steps=100, seed=42)
        assert result.min_rho > -1
        assert result.max_rho < 1

    def test_mean_reversion(self):
        """Terminal ρ should revert toward θ."""
        model = CIRCorrelation(rho0=0.9, kappa=5.0, theta=0.3, sigma=0.1)
        result = model.simulate(3.0, n_paths=2000, n_steps=100, seed=42)
        assert result.mean_terminal_rho < 0.9
        assert result.mean_terminal_rho == pytest.approx(0.3, abs=0.15)

    def test_negative_initial(self):
        model = CIRCorrelation(rho0=-0.5, kappa=2.0, theta=-0.3, sigma=0.2)
        result = model.simulate(1.0, n_paths=500, seed=42)
        assert result.min_rho > -1

    def test_paths_shape(self):
        model = CIRCorrelation(0.5, 2.0, 0.3, 0.2)
        result = model.simulate(1.0, n_paths=100, n_steps=30, seed=42)
        assert result.rho_paths.shape == (100, 31)


# ---- Two-asset with stochastic ρ ----

class TestSimulateTwoAssetStochCorr:
    def test_basic(self):
        corr = CIRCorrelation(0.5, 2.0, 0.3, 0.2)
        result = simulate_two_asset_stoch_corr(
            100, 100, 0.03, 0.02, 0.02, 0.20, 0.25,
            corr, T=1.0, n_paths=500, seed=42,
        )
        assert isinstance(result, StochCorrPricingResult)
        assert result.spot1_paths.shape == (500, 101)
        assert result.spot2_paths.shape == (500, 101)

    def test_spots_positive(self):
        corr = CIRCorrelation(0.5, 2.0, 0.3, 0.2)
        result = simulate_two_asset_stoch_corr(
            100, 100, 0.03, 0.02, 0.02, 0.20, 0.25,
            corr, 1.0, n_paths=200, seed=42,
        )
        assert np.all(result.spot1_paths > 0)
        assert np.all(result.spot2_paths > 0)

    def test_mean_correlation_near_theta(self):
        """Average ρ across time should be between ρ₀ and θ."""
        corr = CIRCorrelation(0.8, 3.0, 0.3, 0.1)
        result = simulate_two_asset_stoch_corr(
            100, 100, 0.03, 0.02, 0.02, 0.20, 0.25,
            corr, 2.0, n_paths=500, seed=42,
        )
        assert 0.2 < result.mean_correlation < 0.9


# ---- Wishart ----

class TestWishartCovariance:
    def test_basic(self):
        Sigma0 = np.array([[0.04, 0.01], [0.01, 0.0625]])
        theta = np.array([[0.04, 0.005], [0.005, 0.0625]])
        model = WishartCovariance(Sigma0, kappa=2.0, theta=theta, sigma=0.1)
        result = model.simulate(1.0, n_paths=200, n_steps=30, seed=42)
        assert isinstance(result, WishartResult)

    def test_pd(self):
        """Covariance should remain positive definite."""
        Sigma0 = np.array([[0.04, 0.01], [0.01, 0.0625]])
        theta = np.array([[0.04, 0.01], [0.01, 0.0625]])
        model = WishartCovariance(Sigma0, 2.0, theta, 0.05)
        result = model.simulate(1.0, n_paths=100, n_steps=30, seed=42)
        assert result.is_pd

    def test_correlation_bounded(self):
        Sigma0 = np.array([[0.04, 0.01], [0.01, 0.0625]])
        theta = np.array([[0.04, 0.01], [0.01, 0.0625]])
        model = WishartCovariance(Sigma0, 2.0, theta, 0.1)
        result = model.simulate(1.0, n_paths=200, n_steps=30, seed=42)
        assert np.all(np.abs(result.correlation_paths) < 1)

    def test_variance_positive(self):
        Sigma0 = np.array([[0.04, 0.01], [0.01, 0.0625]])
        theta = np.array([[0.04, 0.01], [0.01, 0.0625]])
        model = WishartCovariance(Sigma0, 2.0, theta, 0.05)
        result = model.simulate(1.0, n_paths=100, n_steps=30, seed=42)
        assert np.all(result.covariance_paths[:, :, 0, 0] > 0)
        assert np.all(result.covariance_paths[:, :, 1, 1] > 0)

    def test_mean_terminal_corr(self):
        Sigma0 = np.array([[0.04, 0.01], [0.01, 0.0625]])
        theta = np.array([[0.04, 0.005], [0.005, 0.0625]])
        model = WishartCovariance(Sigma0, 2.0, theta, 0.05)
        result = model.simulate(1.0, n_paths=200, n_steps=50, seed=42)
        # Initial ρ ≈ 0.01 / sqrt(0.04 × 0.0625) ≈ 0.2
        assert -1 < result.mean_terminal_corr < 1


# ---- Calibration ----

class TestCalibrateStochCorrToDispersion:
    def test_basic(self):
        result = calibrate_stoch_corr_to_dispersion(
            component_vols=[0.20, 0.25],
            weights=[0.5, 0.5],
            index_variance_target=0.04,
        )
        assert isinstance(result, DispersionCalibrationResult)
        assert -1 < result.theta < 1

    def test_residual_small(self):
        """Should match target index variance."""
        result = calibrate_stoch_corr_to_dispersion(
            [0.20, 0.25], [0.5, 0.5], 0.04,
        )
        assert result.residual < 0.01

    def test_higher_target_higher_theta(self):
        """Higher index variance → higher implied correlation."""
        low = calibrate_stoch_corr_to_dispersion([0.20, 0.25], [0.5, 0.5], 0.03)
        high = calibrate_stoch_corr_to_dispersion([0.20, 0.25], [0.5, 0.5], 0.05)
        assert high.theta > low.theta

    def test_three_assets(self):
        result = calibrate_stoch_corr_to_dispersion(
            [0.20, 0.25, 0.30], [1/3, 1/3, 1/3], 0.05,
        )
        assert -1 < result.theta < 1
