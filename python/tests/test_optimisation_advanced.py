"""Tests for advanced optimisation."""

import math

import numpy as np
import pytest

from pricebook.optimisation_advanced import (
    ADMMResult,
    CMAESResult,
    ConstrainedResult,
    QPResult,
    admm_lasso,
    cma_es,
    constrained_minimize,
    markowitz_portfolio,
    quadratic_program,
)


# ---- QP ----

class TestQuadraticProgram:
    def test_unconstrained(self):
        H = [[2, 0], [0, 2]]
        c = [-4, -6]
        result = quadratic_program(H, c)
        assert result.converged
        np.testing.assert_allclose(result.x, [2, 3], atol=1e-10)

    def test_equality_constrained(self):
        """min x² + y² s.t. x + y = 1 → x = y = 0.5."""
        H = [[2, 0], [0, 2]]
        c = [0, 0]
        A = [[1, 1]]
        b = [1]
        result = quadratic_program(H, c, np.array(A), np.array(b))
        assert result.converged
        np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-10)


class TestMarkowitzPortfolio:
    def test_equal_assets(self):
        """Two identical assets → equal weights."""
        mu = [0.10, 0.10]
        cov = [[0.04, 0.01], [0.01, 0.04]]
        result = markowitz_portfolio(mu, cov)
        assert result.converged
        np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-8)

    def test_weights_sum_to_one(self):
        mu = [0.10, 0.12, 0.08]
        cov = [[0.04, 0.01, 0.005],
               [0.01, 0.09, 0.02],
               [0.005, 0.02, 0.025]]
        result = markowitz_portfolio(mu, cov)
        assert result.converged
        assert np.sum(result.x) == pytest.approx(1.0, abs=1e-10)

    def test_target_return(self):
        mu = [0.05, 0.15]
        cov = [[0.01, 0.002], [0.002, 0.04]]
        result = markowitz_portfolio(mu, cov, target_return=0.10)
        assert result.converged
        assert mu[0] * result.x[0] + mu[1] * result.x[1] == pytest.approx(0.10, abs=1e-8)


# ---- Constrained minimisation ----

class TestConstrainedMinimize:
    def test_equality_constraint(self):
        """min x² + y² s.t. x + y = 1."""
        result = constrained_minimize(
            f=lambda x: x[0]**2 + x[1]**2,
            x0=[0.0, 0.0],
            constraints_eq=[lambda x: x[0] + x[1] - 1],
        )
        assert result.converged
        np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-3)

    def test_inequality_constraint(self):
        """min (x-2)² s.t. x ≤ 1 → x* = 1."""
        result = constrained_minimize(
            f=lambda x: (x[0] - 2)**2,
            x0=[0.0],
            constraints_ineq=[lambda x: x[0] - 1],  # x ≤ 1 → x - 1 ≤ 0
        )
        assert result.x[0] == pytest.approx(1.0, abs=0.1)


# ---- ADMM LASSO ----

class TestADMMLasso:
    def test_recovers_sparse_signal(self):
        """LASSO should recover a sparse vector from noisy observations."""
        np.random.seed(42)
        n, m = 50, 100
        x_true = np.zeros(n)
        x_true[:5] = [3, -2, 1, 4, -1]
        A = np.random.randn(m, n)
        b = A @ x_true + 0.1 * np.random.randn(m)

        result = admm_lasso(A, b, lam=0.5, rho=1.0, max_iter=2000, tol=1e-4)
        # Should recover the non-zero pattern
        assert np.sum(np.abs(result.x) > 0.1) <= 10  # sparse

    def test_zero_lambda_is_ols(self):
        """λ=0 → ordinary least squares."""
        np.random.seed(42)
        A = np.array([[1, 0], [0, 1], [1, 1]])
        b = np.array([1, 2, 3])
        result = admm_lasso(A, b, lam=0.0)
        ols = np.linalg.lstsq(A, b, rcond=None)[0]
        np.testing.assert_allclose(result.x, ols, atol=1e-3)


# ---- CMA-ES ----

class TestCMAES:
    def test_sphere(self):
        """CMA-ES finds minimum of sphere function."""
        result = cma_es(lambda x: np.sum(x**2), [5.0, 5.0],
                        sigma0=2.0, max_iter=500, seed=42)
        np.testing.assert_allclose(result.x, [0, 0], atol=0.1)
        assert result.objective < 0.01

    def test_rosenbrock(self):
        """CMA-ES on Rosenbrock (harder)."""
        def rosenbrock(x):
            return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
        result = cma_es(rosenbrock, [0.0, 0.0], sigma0=1.0,
                        max_iter=2000, seed=42)
        np.testing.assert_allclose(result.x, [1, 1], atol=0.5)

    def test_3d(self):
        result = cma_es(lambda x: np.sum((x - 1)**2), [5, 5, 5],
                        sigma0=2.0, max_iter=500, seed=42)
        np.testing.assert_allclose(result.x, [1, 1, 1], atol=0.2)
