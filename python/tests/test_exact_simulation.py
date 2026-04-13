"""Tests for exact CIR simulation and implicit Euler."""

import math

import numpy as np
import pytest

from pricebook.exact_simulation import (
    ExactCIRResult,
    exact_cir,
    exact_cir_zcb,
    implicit_euler_paths,
    implicit_euler_step,
)
from pricebook.numerical_safety import martingale_test


# ---- Exact CIR ----

class TestExactCIR:
    def test_mean_reverts_to_theta(self):
        kappa, theta, xi = 2.0, 0.04, 0.3
        result = exact_cir(0.10, kappa, theta, xi, T=5.0,
                           n_steps=50, n_paths=20_000, seed=42)
        terminal_mean = result.paths[:, -1].mean()
        assert terminal_mean == pytest.approx(theta, rel=0.10)

    def test_non_negative(self):
        """Exact CIR paths are always non-negative."""
        result = exact_cir(0.04, 1.0, 0.04, 0.5, T=1.0,
                           n_steps=50, n_paths=5_000, seed=42)
        assert np.all(result.paths >= 0)

    def test_shape(self):
        result = exact_cir(0.04, 2.0, 0.04, 0.3, T=1.0,
                           n_steps=20, n_paths=100, seed=42)
        assert result.paths.shape == (100, 21)
        assert len(result.times) == 21

    def test_variance_formula(self):
        """Terminal variance should match analytical: Var = θξ²/(2κ) (steady state)."""
        kappa, theta, xi = 4.0, 0.04, 0.2
        result = exact_cir(theta, kappa, theta, xi, T=10.0,
                           n_steps=100, n_paths=50_000, seed=42)
        # Steady-state variance = θξ²/(2κ) = 0.04×0.04/(8) = 0.0002
        expected_var = theta * xi * xi / (2 * kappa)
        actual_var = result.paths[:, -1].var()
        assert actual_var == pytest.approx(expected_var, rel=0.20)


# ---- Exact CIR ZCB price ----

class TestExactCIRZCB:
    def test_zero_maturity(self):
        assert exact_cir_zcb(0.05, 2.0, 0.04, 0.3, T=0.0) == 1.0

    def test_positive_rate_discounts(self):
        """Bond price < 1 for positive rate and T > 0."""
        price = exact_cir_zcb(0.05, 2.0, 0.04, 0.3, T=1.0)
        assert 0 < price < 1

    def test_matches_mc(self):
        """Exact ZCB matches MC estimate from exact CIR paths."""
        kappa, theta, xi, r0, T = 2.0, 0.04, 0.3, 0.05, 1.0
        analytical = exact_cir_zcb(r0, kappa, theta, xi, T)

        # MC: P(0,T) = E[exp(-∫r dt)] ≈ E[exp(-Σ r_i Δt)]
        result = exact_cir(r0, kappa, theta, xi, T,
                           n_steps=100, n_paths=50_000, seed=42)
        dt = T / 100
        integral = np.sum(result.paths[:, :-1], axis=1) * dt
        mc_price = np.exp(-integral).mean()

        assert mc_price == pytest.approx(analytical, rel=0.02)

    def test_higher_rate_lower_price(self):
        p_low = exact_cir_zcb(0.03, 2.0, 0.04, 0.3, 1.0)
        p_high = exact_cir_zcb(0.08, 2.0, 0.04, 0.3, 1.0)
        assert p_high < p_low


# ---- Implicit Euler ----

class TestImplicitEuler:
    def test_ou_mean_reverts(self):
        """Implicit Euler on OU process should mean-revert."""
        kappa, theta = 5.0, 1.0  # stiff mean-reversion

        def drift(x):
            return kappa * (theta - x)

        def diffusion(x):
            return 0.2 * np.ones_like(x)

        paths = implicit_euler_paths(
            x0=5.0, drift=drift, diffusion=diffusion,
            T=2.0, n_steps=200, n_paths=5_000, seed=42,
        )
        terminal_mean = paths[:, -1].mean()
        assert terminal_mean == pytest.approx(theta, rel=0.10)

    def test_stiff_doesnt_blow_up(self):
        """With large κ, implicit Euler stays stable (need κΔt < 1 for fixed-point)."""
        kappa = 20.0  # stiff but κΔt = 0.4 with 50 steps

        def drift(x):
            return kappa * (1.0 - x)

        def diffusion(x):
            return 0.1 * np.ones_like(x)

        paths = implicit_euler_paths(
            x0=10.0, drift=drift, diffusion=diffusion,
            T=1.0, n_steps=50, n_paths=1_000, seed=42,
        )
        # Should not blow up — all values finite
        assert np.all(np.isfinite(paths))
        # Should have mean-reverted close to 1.0
        assert paths[:, -1].mean() == pytest.approx(1.0, rel=0.15)

    def test_shape(self):
        def drift(x): return -x
        def diffusion(x): return 0.1 * np.ones_like(x)
        paths = implicit_euler_paths(0.0, drift, diffusion,
                                      T=1.0, n_steps=10, n_paths=100, seed=42)
        assert paths.shape == (100, 11)
