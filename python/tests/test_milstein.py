"""Tests for Milstein scheme."""

import math

import numpy as np
import pytest

from pricebook.milstein import (
    MilsteinResult,
    milstein_cev,
    milstein_cir,
    milstein_gbm,
    milstein_paths,
)
from pricebook.numerical_safety import martingale_test, convergence_rate


SPOT, RATE, VOL, T = 100.0, 0.05, 0.20, 1.0


class TestMilsteinGBM:
    def test_passes_martingale_test(self):
        """GBM Milstein should pass the martingale test."""
        result = milstein_gbm(SPOT, RATE, VOL, T, n_steps=100,
                              n_paths=50_000, seed=42)
        mt = martingale_test(result.paths[:, -1], SPOT, RATE, T)
        assert mt.passed

    def test_terminal_mean_near_forward(self):
        result = milstein_gbm(SPOT, RATE, VOL, T, n_steps=100,
                              n_paths=50_000, seed=42)
        forward = SPOT * math.exp(RATE * T)
        mean = result.paths[:, -1].mean()
        assert mean == pytest.approx(forward, rel=0.02)

    def test_shape(self):
        result = milstein_gbm(SPOT, RATE, VOL, T, n_steps=50,
                              n_paths=1000, seed=42)
        assert result.paths.shape == (1000, 51)
        assert len(result.times) == 51
        assert result.dt == pytest.approx(T / 50)

    def test_strong_order_one(self):
        """Milstein GBM should have strong order ~1.0 vs exact."""
        errors = []
        steps_list = [10, 20, 40, 80]
        n_paths = 10_000

        for n_steps in steps_list:
            # Milstein paths
            mil = milstein_gbm(SPOT, RATE, VOL, T, n_steps, n_paths, seed=42)
            # Exact GBM terminal
            rng = np.random.default_rng(42)
            z_total = np.zeros(n_paths)
            dt = T / n_steps
            for _ in range(n_steps):
                z_total += rng.standard_normal(n_paths) * np.sqrt(dt)
            exact = SPOT * np.exp((RATE - 0.5 * VOL**2) * T + VOL * z_total)
            err = np.mean(np.abs(mil.paths[:, -1] - exact))
            errors.append(err)

        dts = [T / s for s in steps_list]
        cr = convergence_rate(dts, errors, expected_order=1.0, order_tol=0.4)
        assert cr.estimated_order > 0.6  # should be ~1.0


class TestMilsteinCEV:
    def test_beta_one_matches_gbm(self):
        """CEV with β=1 should match GBM."""
        gbm = milstein_gbm(SPOT, RATE, VOL, T, 100, 10_000, seed=42)
        cev = milstein_cev(SPOT, RATE, VOL, beta=1.0, T=T,
                           n_steps=100, n_paths=10_000, seed=42)
        # Terminal means should be similar
        assert cev.paths[:, -1].mean() == pytest.approx(
            gbm.paths[:, -1].mean(), rel=0.05)

    def test_positive_paths(self):
        result = milstein_cev(SPOT, RATE, VOL, beta=0.5, T=T,
                              n_steps=100, n_paths=1000, seed=42)
        # CEV with absorption should stay non-negative mostly
        assert result.paths[:, -1].mean() > 0


class TestMilsteinCIR:
    def test_mean_reverts(self):
        """CIR should mean-revert toward θ."""
        kappa, theta, xi = 2.0, 0.04, 0.3
        result = milstein_cir(0.10, kappa, theta, xi, T=5.0,
                              n_steps=500, n_paths=10_000, seed=42)
        terminal_mean = result.paths[:, -1].mean()
        assert terminal_mean == pytest.approx(theta, rel=0.15)

    def test_non_negative(self):
        """CIR paths should be non-negative (absorption)."""
        result = milstein_cir(0.04, 1.0, 0.04, 0.5, T=1.0,
                              n_steps=100, n_paths=1000, seed=42)
        assert np.all(result.paths >= 0)

    def test_feller_satisfied_stays_positive(self):
        """Under Feller condition, variance should stay strictly positive."""
        kappa, theta, xi = 4.0, 0.04, 0.2  # 2κθ=0.32 > ξ²=0.04
        result = milstein_cir(0.04, kappa, theta, xi, T=1.0,
                              n_steps=200, n_paths=5000, seed=42)
        # Very few paths should hit zero when Feller is satisfied
        hit_zero = np.sum(np.any(result.paths == 0, axis=1))
        assert hit_zero / 5000 < 0.05
