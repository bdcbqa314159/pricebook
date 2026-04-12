"""Tests for multi-dimensional root finding."""

import numpy as np
import pytest

from pricebook.nd_solvers import (
    NDSolverResult,
    broyden,
    damped_newton,
    finite_difference_jacobian,
    newton_nd,
)


# ---- Test systems ----

def _linear_system(x):
    """2x2 linear: A·x - b = 0 → x = [1, 2]."""
    A = np.array([[3, 1], [1, 2]])
    b = np.array([5, 5])
    return A @ x - b


def _linear_jacobian(x):
    return np.array([[3, 1], [1, 2]])


def _nonlinear_system(x):
    """x² + y² = 5, x·y = 2 → solutions include (1, 2) and (2, 1)."""
    return np.array([x[0]**2 + x[1]**2 - 5, x[0] * x[1] - 2])


def _nonlinear_jacobian(x):
    return np.array([[2*x[0], 2*x[1]], [x[1], x[0]]])


def _rosenbrock(x):
    """Rosenbrock system: hard test for nonlinear solvers."""
    return np.array([
        10 * (x[1] - x[0]**2),
        1 - x[0],
    ])


# ---- Finite-difference Jacobian ----

class TestFiniteDifferenceJacobian:
    def test_linear_system(self):
        J = finite_difference_jacobian(_linear_system, np.array([0.0, 0.0]))
        expected = np.array([[3, 1], [1, 2]])
        np.testing.assert_allclose(J, expected, atol=1e-6)

    def test_nonlinear_at_point(self):
        x = np.array([1.0, 2.0])
        J = finite_difference_jacobian(_nonlinear_system, x)
        expected = _nonlinear_jacobian(x)
        np.testing.assert_allclose(J, expected, atol=1e-5)


# ---- Newton-Raphson ----

class TestNewtonND:
    def test_linear_with_analytical_jacobian(self):
        result = newton_nd(_linear_system, [0.0, 0.0], jacobian=_linear_jacobian)
        assert result.converged
        np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-10)

    def test_linear_with_fd_jacobian(self):
        result = newton_nd(_linear_system, [0.0, 0.0])
        assert result.converged
        np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-8)

    def test_nonlinear(self):
        result = newton_nd(_nonlinear_system, [1.5, 2.5],
                           jacobian=_nonlinear_jacobian)
        assert result.converged
        # Should find (1, 2) or (2, 1)
        assert (
            np.allclose(result.x, [1, 2], atol=1e-8) or
            np.allclose(result.x, [2, 1], atol=1e-8)
        )

    def test_residual_below_tol(self):
        result = newton_nd(_nonlinear_system, [1.5, 2.5])
        assert result.residual_norm < 1e-10

    def test_3d_system(self):
        def f(x):
            return np.array([x[0] + x[1] + x[2] - 6,
                             x[0]*x[1] - x[2] - 2,
                             x[0]**2 - x[1] - x[2] - 1])
        result = newton_nd(f, [2.0, 2.0, 2.0])
        assert result.converged
        assert result.residual_norm < 1e-8


# ---- Broyden ----

class TestBroyden:
    def test_linear(self):
        result = broyden(_linear_system, [0.0, 0.0])
        assert result.converged
        np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-8)

    def test_nonlinear(self):
        result = broyden(_nonlinear_system, [1.5, 2.5])
        assert result.converged
        assert result.residual_norm < 1e-8

    def test_fewer_jacobian_evaluations(self):
        """Broyden should converge without recomputing full Jacobian."""
        result = broyden(_nonlinear_system, [1.5, 2.5])
        assert result.converged
        # Broyden typically needs more iterations than Newton but each is cheaper
        assert result.iterations < 50

    def test_rosenbrock(self):
        result = broyden(_rosenbrock, [0.5, 0.5], max_iter=200)
        assert result.converged
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-6)


# ---- Damped Newton ----

class TestDampedNewton:
    def test_linear(self):
        result = damped_newton(_linear_system, [0.0, 0.0])
        assert result.converged
        np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-8)

    def test_nonlinear(self):
        result = damped_newton(_nonlinear_system, [1.5, 2.5])
        assert result.converged
        assert result.residual_norm < 1e-8

    def test_robustness_bad_initial(self):
        """Damped Newton should handle a poor initial guess better than plain Newton."""
        # Start far from solution — damping prevents overshoot
        result = damped_newton(_nonlinear_system, [10.0, 5.0], max_iter=100)
        assert result.converged
        assert result.residual_norm < 1e-8

    def test_rosenbrock(self):
        result = damped_newton(_rosenbrock, [0.0, 0.0])
        assert result.converged
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-6)

    def test_with_analytical_jacobian(self):
        result = damped_newton(_nonlinear_system, [0.5, 2.5],
                               jacobian=_nonlinear_jacobian)
        assert result.converged

    def test_calibration_style(self):
        """Simulate a calibration-like problem: match 3 targets with 3 params."""
        target = np.array([0.25, 0.20, 0.23])
        def model(params):
            a, b, c = params[0], params[1], params[2]
            strikes = np.array([90, 100, 110])
            vols = a + b * (strikes - 100) / 100 + c * ((strikes - 100) / 100)**2
            return vols - target
        result = damped_newton(model, [0.22, 0.01, 0.01], max_iter=100)
        assert result.residual_norm < 1e-8
