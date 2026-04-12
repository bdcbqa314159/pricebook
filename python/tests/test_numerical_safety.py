"""Tests for numerical safety: CFL, Feller, martingale, convergence."""

import math

import numpy as np
import pytest

from pricebook.numerical_safety import (
    CFLResult,
    ConvergenceResult,
    FellerResult,
    MartingaleTestResult,
    check_cfl,
    check_feller,
    convergence_rate,
    martingale_test,
    strong_convergence_test,
    weak_convergence_test,
)


# ---- CFL condition ----

class TestCheckCFL:
    def test_stable(self):
        # vol=0.20, dt=0.001, dx=0.05 → dt_max ≈ 0.05²/0.04 = 0.0625
        result = check_cfl(vol=0.20, rate=0.05, dt=0.001, dx=0.05)
        assert result.is_stable
        assert result.ratio < 1.0

    def test_unstable(self):
        # dt too large for the grid
        result = check_cfl(vol=0.20, rate=0.05, dt=0.1, dx=0.01)
        assert not result.is_stable
        assert "UNSTABLE" in result.recommendation

    def test_dt_max_formula(self):
        vol, rate, dx = 0.30, 0.05, 0.02
        mu = abs(rate - 0.5 * vol * vol)
        expected_max = dx * dx / (vol * vol + mu * dx)
        result = check_cfl(vol, rate, dt=0.001, dx=dx)
        assert result.dt_max == pytest.approx(expected_max)

    def test_zero_vol(self):
        result = check_cfl(vol=0.0, rate=0.05, dt=0.1, dx=0.01)
        assert result.is_stable  # no diffusion

    def test_borderline(self):
        vol, rate, dx = 0.20, 0.0, 0.05
        # Compute dt_max, then check that 99% of it is stable
        probe = check_cfl(vol, rate, dt=0.001, dx=dx)
        result = check_cfl(vol, rate, dt=probe.dt_max * 0.99, dx=dx)
        assert result.is_stable


# ---- Feller condition ----

class TestCheckFeller:
    def test_satisfied(self):
        # 2×2×0.04 = 0.16 > 0.09 = 0.3²
        result = check_feller(kappa=2.0, theta=0.04, xi=0.3)
        assert result.is_satisfied

    def test_violated(self):
        # 2×1×0.04 = 0.08 < 0.25 = 0.5²
        result = check_feller(kappa=1.0, theta=0.04, xi=0.5)
        assert not result.is_satisfied
        assert "VIOLATED" in result.recommendation

    def test_boundary(self):
        # 2κθ = ξ² exactly
        result = check_feller(kappa=1.0, theta=0.04, xi=math.sqrt(0.08))
        assert result.is_satisfied


# ---- Martingale test ----

class TestMartingaleTest:
    def test_gbm_passes(self):
        """GBM exact simulation should pass the martingale test."""
        np.random.seed(42)
        spot, rate, T = 100.0, 0.05, 1.0
        n = 100_000
        z = np.random.randn(n)
        S_T = spot * np.exp((rate - 0.5 * 0.2**2) * T + 0.2 * math.sqrt(T) * z)
        result = martingale_test(S_T, spot, rate, T, tol=0.01)
        assert result.passed
        assert result.relative_error < 0.01

    def test_biased_drift_fails(self):
        """Wrong drift should fail the martingale test."""
        np.random.seed(42)
        spot, rate, T = 100.0, 0.05, 1.0
        n = 100_000
        z = np.random.randn(n)
        # Wrong drift: forgot the -0.5σ² Itô correction
        S_T = spot * np.exp(rate * T + 0.2 * math.sqrt(T) * z)
        result = martingale_test(S_T, spot, rate, T, tol=0.01)
        assert not result.passed

    def test_reports_stats(self):
        np.random.seed(42)
        S_T = np.full(1000, 100.0)
        result = martingale_test(S_T, 100.0, 0.0, 1.0)
        assert result.n_paths == 1000
        assert result.std_error >= 0


# ---- Convergence rate ----

class TestConvergenceRate:
    def test_order_2(self):
        """Method with O(h²) should give order ≈ 2."""
        h = [0.1, 0.05, 0.025, 0.0125]
        errors = [h_i**2 for h_i in h]
        result = convergence_rate(h, errors, expected_order=2.0)
        assert result.estimated_order == pytest.approx(2.0, abs=0.1)
        assert result.is_consistent

    def test_order_1(self):
        h = [0.1, 0.05, 0.025, 0.0125]
        errors = [h_i**1 for h_i in h]
        result = convergence_rate(h, errors, expected_order=1.0)
        assert result.estimated_order == pytest.approx(1.0, abs=0.1)

    def test_order_half(self):
        """Euler strong convergence O(h^0.5)."""
        h = [0.1, 0.05, 0.025, 0.0125]
        errors = [h_i**0.5 for h_i in h]
        result = convergence_rate(h, errors, expected_order=0.5)
        assert result.estimated_order == pytest.approx(0.5, abs=0.1)

    def test_inconsistent_order(self):
        h = [0.1, 0.05, 0.025]
        errors = [0.01, 0.009, 0.0088]  # barely decreasing → order ≈ 0
        result = convergence_rate(h, errors, expected_order=2.0)
        assert not result.is_consistent

    def test_single_point(self):
        result = convergence_rate([0.1], [0.01])
        assert result.estimated_order == 0.0


# ---- Strong / weak convergence ----

class TestStrongWeakConvergence:
    def test_strong_error_zero_same_paths(self):
        paths = np.array([1.0, 2.0, 3.0])
        assert strong_convergence_test(paths, paths) == pytest.approx(0.0)

    def test_strong_error_positive(self):
        fine = np.array([1.0, 2.0, 3.0])
        coarse = np.array([1.1, 1.9, 3.2])
        err = strong_convergence_test(fine, coarse)
        assert err > 0
        assert err == pytest.approx(np.mean(np.abs(fine - coarse)))

    def test_weak_error_zero(self):
        vals = np.array([99.0, 101.0])
        assert weak_convergence_test(vals, 100.0) == pytest.approx(0.0)

    def test_weak_error_nonzero(self):
        vals = np.array([105.0, 105.0])
        assert weak_convergence_test(vals, 100.0) == pytest.approx(5.0)
