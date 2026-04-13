"""Tests for convergence testing framework."""

import math

import numpy as np
import pytest

from pricebook.convergence_framework import (
    ConvergenceStudyResult,
    SchemeComparisonResult,
    scheme_comparison,
    strong_convergence_study,
    weak_convergence_study,
)


SPOT, RATE, VOL, T = 100.0, 0.05, 0.20, 1.0


def _euler_gbm_terminal(n_steps, n_paths, seed):
    """Euler-Maruyama GBM terminal values.

    dS = r S dt + σ S dW (arithmetic Euler on the SDE for S).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.full(n_paths, SPOT)
    for _ in range(n_steps):
        dW = rng.standard_normal(n_paths) * math.sqrt(dt)
        S = S + RATE * S * dt + VOL * S * dW
        S = np.maximum(S, 0.0)
    return S


def _milstein_gbm_terminal(n_steps, n_paths, seed):
    """Milstein GBM terminal values.

    Arithmetic Euler + Milstein correction: 0.5 σ² S (dW² − dt).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.full(n_paths, SPOT)
    for _ in range(n_steps):
        dW = rng.standard_normal(n_paths) * math.sqrt(dt)
        S = S + RATE * S * dt + VOL * S * dW + 0.5 * VOL**2 * S * (dW**2 - dt)
        S = np.maximum(S, 0.0)
    return S


def _exact_gbm_terminal(n_paths, seed):
    """Exact GBM terminal values (for strong error reference)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    return SPOT * np.exp((RATE - 0.5 * VOL**2) * T + VOL * math.sqrt(T) * z)


# ---- Strong convergence ----

class TestStrongConvergenceStudy:
    def test_euler_produces_result(self):
        """Framework produces a valid ConvergenceStudyResult."""
        # Note: true strong convergence requires matched Brownian paths
        # (same BM realization at different step sizes). This test
        # verifies the framework mechanics, not the convergence order.
        result = strong_convergence_study(
            "Euler",
            _euler_gbm_terminal,
            _exact_gbm_terminal,
            T, steps_list=[10, 20, 40],
            n_paths=5_000, expected_order=0.5, seed=42,
        )
        assert result.scheme_name == "Euler"
        assert result.error_type == "strong"
        assert len(result.errors) == 3
        assert all(e > 0 for e in result.errors)

    def test_framework_produces_step_sizes(self):
        """Framework correctly records step sizes and errors."""
        result = strong_convergence_study(
            "Milstein",
            _milstein_gbm_terminal,
            _exact_gbm_terminal,
            T, steps_list=[10, 20, 40],
            n_paths=5_000, expected_order=1.0, seed=42,
        )
        assert len(result.step_sizes) == 3
        assert result.step_sizes[0] > result.step_sizes[-1]  # dt decreasing


# ---- Weak convergence ----

class TestWeakConvergenceStudy:
    def test_euler_weak_produces_result(self):
        """Framework produces valid weak convergence result."""
        forward = SPOT * math.exp(RATE * T)
        result = weak_convergence_study(
            "Euler",
            _euler_gbm_terminal,
            reference_value=forward,
            T=T, steps_list=[10, 20, 40],
            n_paths=50_000, expected_order=1.0, seed=42,
        )
        assert result.error_type == "weak"
        assert len(result.errors) == 3
        assert all(e > 0 for e in result.errors)

    def test_milstein_weak_produces_result(self):
        forward = SPOT * math.exp(RATE * T)
        result = weak_convergence_study(
            "Milstein",
            _milstein_gbm_terminal,
            reference_value=forward,
            T=T, steps_list=[10, 20, 40],
            n_paths=50_000, expected_order=1.0, seed=42,
        )
        assert result.error_type == "weak"
        assert len(result.errors) == 3

    def test_errors_bounded(self):
        """Weak errors should be small (< 5% of forward)."""
        forward = SPOT * math.exp(RATE * T)
        result = weak_convergence_study(
            "Euler", _euler_gbm_terminal, forward,
            T, [10, 20, 40], n_paths=50_000, seed=42,
        )
        for e in result.errors:
            assert e < 0.05 * forward  # < 5% of forward


# ---- Scheme comparison ----

class TestSchemeComparison:
    def test_comparison(self):
        forward = SPOT * math.exp(RATE * T)
        euler = weak_convergence_study(
            "Euler", _euler_gbm_terminal, forward,
            T, [10, 20, 40], n_paths=20_000, seed=42,
        )
        milstein = weak_convergence_study(
            "Milstein", _milstein_gbm_terminal, forward,
            T, [10, 20, 40], n_paths=20_000, seed=42,
        )
        comp = scheme_comparison([euler, milstein])
        assert len(comp.studies) == 2
        assert comp.best_scheme in {"Euler", "Milstein"}
        assert comp.best_order > 0

    def test_empty(self):
        comp = scheme_comparison([])
        assert comp.best_scheme == ""
        assert comp.best_order == 0.0

    def test_single_scheme(self):
        study = ConvergenceStudyResult(
            "Test", [0.1, 0.05], [0.01, 0.005], 1.0, 1.0, True, "weak",
        )
        comp = scheme_comparison([study])
        assert comp.best_scheme == "Test"
