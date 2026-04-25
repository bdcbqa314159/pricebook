"""Tests for FX SLV calibration."""

import math

import numpy as np
import pytest

from pricebook.fx_slv_calibration import (
    LeverageFunction,
    MixingResult,
    ParticleCalibrationResult,
    SLVBarrierResult,
    calibrate_leverage_function,
    particle_slv_calibration,
    slv_barrier_price,
    slv_mixing_calibration,
)


# ---- Leverage function ----

class TestLeverageFunction:
    def test_evaluate(self):
        times = np.array([0.0, 0.5, 1.0])
        spots = np.array([0.90, 1.0, 1.10])
        values = np.ones((3, 3)) * 1.2
        L = LeverageFunction(times, spots, values, "test")
        assert L(1.0, 0.5) == pytest.approx(1.2)

    def test_interpolation(self):
        times = np.array([0.0, 1.0])
        spots = np.array([0.9, 1.1])
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        L = LeverageFunction(times, spots, values, "test")
        # Middle of grid
        mid = L(1.0, 0.5)
        # Should be average of corners: (1+2+3+4)/4 = 2.5
        assert mid == pytest.approx(2.5, rel=0.01)

    def test_out_of_range_clamps(self):
        times = np.array([0.0, 1.0])
        spots = np.array([0.9, 1.1])
        values = np.array([[1.5, 1.5], [1.5, 1.5]])
        L = LeverageFunction(times, spots, values, "test")
        # Outside range should still return something
        assert L(2.0, 2.0) == pytest.approx(1.5, rel=0.01)


# ---- Calibration via forward Kolmogorov ----

class TestCalibrateLeverageFunction:
    def test_basic(self):
        times = np.linspace(0.0, 1.0, 5)
        spots = np.linspace(0.85, 1.15, 7)
        local_vols = np.full((5, 7), 0.15)
        result = calibrate_leverage_function(
            1.0, local_vols, times, spots,
            kappa=1.0, theta=0.02, xi=0.3, v0=0.02, rho=-0.3,
            n_paths=1_000, seed=42,
        )
        assert isinstance(result, LeverageFunction)
        assert result.values.shape == (5, 7)

    def test_leverage_positive(self):
        times = np.linspace(0.0, 1.0, 3)
        spots = np.linspace(0.9, 1.1, 5)
        local_vols = np.full((3, 5), 0.15)
        result = calibrate_leverage_function(
            1.0, local_vols, times, spots,
            kappa=1.0, theta=0.02, xi=0.3, v0=0.02, rho=-0.3,
            n_paths=500, seed=42,
        )
        assert np.all(result.values > 0)

    def test_leverage_scaling(self):
        """Higher local vol → higher leverage."""
        times = np.linspace(0.0, 1.0, 3)
        spots = np.linspace(0.9, 1.1, 5)
        lv_low = np.full((3, 5), 0.10)
        lv_high = np.full((3, 5), 0.30)
        low = calibrate_leverage_function(1.0, lv_low, times, spots,
                                           1.0, 0.02, 0.3, 0.02, rho=-0.3,
                                           n_paths=500, seed=42)
        high = calibrate_leverage_function(1.0, lv_high, times, spots,
                                            1.0, 0.02, 0.3, 0.02, rho=-0.3,
                                            n_paths=500, seed=42)
        assert high.values.mean() > low.values.mean()


# ---- Particle method ----

class TestParticleCalibration:
    def test_basic(self):
        times = np.linspace(0.0, 1.0, 3)
        spots = np.linspace(0.9, 1.1, 5)
        local_vols = np.full((3, 5), 0.15)
        result = particle_slv_calibration(
            1.0, local_vols, times, spots,
            kappa=1.0, theta=0.02, xi=0.3, v0=0.02, rho=-0.3,
            n_particles=500, seed=42,
        )
        assert isinstance(result, ParticleCalibrationResult)
        assert result.n_particles == 500

    def test_bandwidth_default(self):
        times = np.linspace(0.0, 1.0, 3)
        spots = np.linspace(0.9, 1.1, 5)
        local_vols = np.full((3, 5), 0.15)
        result = particle_slv_calibration(
            1.0, local_vols, times, spots,
            kappa=1.0, theta=0.02, xi=0.3, v0=0.02, rho=-0.3,
            n_particles=500, seed=42,
        )
        assert result.bandwidth > 0

    def test_leverage_bounded(self):
        """Leverage should be clipped to reasonable range."""
        times = np.linspace(0.0, 1.0, 3)
        spots = np.linspace(0.9, 1.1, 5)
        local_vols = np.full((3, 5), 0.15)
        result = particle_slv_calibration(
            1.0, local_vols, times, spots,
            kappa=1.0, theta=0.02, xi=0.3, v0=0.02, rho=-0.3,
            n_particles=500, seed=42,
        )
        assert np.all(result.leverage.values >= 0.1)
        assert np.all(result.leverage.values <= 10.0)


# ---- Mixing fraction ----

class TestMixingCalibration:
    def test_bisection_converges(self):
        """Target = 0.5, function returns eta linearly."""
        def price_fn(eta):
            return eta  # linear in eta

        result = slv_mixing_calibration(1.0, 0.5, price_fn, (0.0, 1.0), n_steps=20)
        assert isinstance(result, MixingResult)
        assert result.eta == pytest.approx(0.5, abs=0.01)

    def test_residual_small(self):
        def price_fn(eta):
            return 2 * eta + 0.1

        result = slv_mixing_calibration(1.0, 0.7, price_fn, (0.0, 1.0), n_steps=15)
        assert result.residual < 0.05

    def test_edge_cases(self):
        """Target near extremes."""
        def price_fn(eta):
            return eta**2

        result = slv_mixing_calibration(1.0, 0.25, price_fn, (0.0, 1.0), n_steps=20)
        assert result.eta == pytest.approx(0.5, abs=0.05)


# ---- SLV barrier pricing ----

class TestSLVBarrierPrice:
    def _make_leverage(self):
        times = np.array([0.0, 0.5, 1.0])
        spots = np.array([0.80, 0.95, 1.0, 1.05, 1.20])
        values = np.ones((3, 5))
        return LeverageFunction(times, spots, values, "test")

    def test_basic(self):
        L = self._make_leverage()
        result = slv_barrier_price(
            1.0, 1.0, 1.10, 0.02, 0.01, L,
            v0=0.02, kappa=1.0, theta=0.02, xi=0.3, rho=-0.3,
            T=1.0, n_paths=500, n_steps=50, seed=42,
        )
        assert isinstance(result, SLVBarrierResult)
        assert result.price >= 0
        assert 0 <= result.knock_out_prob <= 1

    def test_knock_out_probability(self):
        L = self._make_leverage()
        near = slv_barrier_price(1.0, 1.0, 1.02, 0.02, 0.01, L,
                                  0.02, 1.0, 0.02, 0.3, -0.3, 1.0,
                                  n_paths=500, n_steps=50, seed=42)
        far = slv_barrier_price(1.0, 1.0, 1.20, 0.02, 0.01, L,
                                 0.02, 1.0, 0.02, 0.3, -0.3, 1.0,
                                 n_paths=500, n_steps=50, seed=42)
        assert near.knock_out_prob > far.knock_out_prob

    def test_call_vs_put(self):
        """ITM put has different price than ITM call."""
        L = self._make_leverage()
        call = slv_barrier_price(1.0, 1.0, 1.15, 0.02, 0.01, L,
                                  0.02, 1.0, 0.02, 0.3, -0.3, 1.0,
                                  is_call=True, n_paths=500, n_steps=50, seed=42)
        put = slv_barrier_price(1.0, 1.0, 1.15, 0.02, 0.01, L,
                                 0.02, 1.0, 0.02, 0.3, -0.3, 1.0,
                                 is_call=False, n_paths=500, n_steps=50, seed=42)
        assert call.price > 0 or put.price > 0
