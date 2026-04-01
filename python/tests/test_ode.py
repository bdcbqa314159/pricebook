"""Tests for ODE solvers."""

import pytest
import math
import numpy as np

from pricebook.ode import rk4, rk45, bdf, ODEResult


class TestRK4:
    def test_exponential_decay(self):
        """dy/dt = -y, y(0) = 1 → y(t) = exp(-t)."""
        result = rk4(lambda t, y: -y, (0, 2), [1.0], dt=0.01)
        assert isinstance(result, ODEResult)
        assert result.method == "rk4"
        # Final value should be exp(-2) ≈ 0.1353
        assert result.y[-1, 0] == pytest.approx(math.exp(-2), rel=1e-6)

    def test_harmonic_oscillator(self):
        """y'' + y = 0, y(0)=0, y'(0)=1 → y = sin(t)."""
        def f(t, y):
            return [y[1], -y[0]]

        result = rk4(f, (0, 2 * math.pi), [0.0, 1.0], dt=0.01)
        # After one period: y ≈ 0, y' ≈ 1
        assert result.y[-1, 0] == pytest.approx(0.0, abs=1e-4)
        assert result.y[-1, 1] == pytest.approx(1.0, abs=1e-4)

    def test_linear_growth(self):
        """dy/dt = 1, y(0) = 0 → y(t) = t."""
        result = rk4(lambda t, y: [1.0], (0, 5), [0.0], dt=0.1)
        assert result.y[-1, 0] == pytest.approx(5.0, abs=1e-10)

    def test_returns_all_points(self):
        result = rk4(lambda t, y: -y, (0, 1), [1.0], dt=0.1)
        assert len(result.t) == 11  # 0, 0.1, ..., 1.0
        assert result.t[0] == 0.0
        assert result.t[-1] == pytest.approx(1.0)


class TestRK45:
    def test_exponential_decay(self):
        result = rk45(lambda t, y: -y, (0, 2), [1.0], tol=1e-8)
        assert result.method == "rk45"
        assert result.y[-1, 0] == pytest.approx(math.exp(-2), rel=1e-6)

    def test_harmonic_oscillator(self):
        def f(t, y):
            return [y[1], -y[0]]

        result = rk45(f, (0, 2 * math.pi), [0.0, 1.0], tol=1e-8)
        assert result.y[-1, 0] == pytest.approx(0.0, abs=1e-4)
        assert result.y[-1, 1] == pytest.approx(1.0, abs=1e-4)

    def test_fewer_evals_than_rk4(self):
        """Adaptive should use fewer evaluations for smooth problems."""
        f = lambda t, y: -y
        r4 = rk4(f, (0, 5), [1.0], dt=0.001)
        r45 = rk45(f, (0, 5), [1.0], tol=1e-8)
        # RK45 should use fewer total evaluations
        assert r45.n_evaluations < r4.n_evaluations

    def test_accuracy_matches_rk4(self):
        f = lambda t, y: -y
        r4 = rk4(f, (0, 2), [1.0], dt=0.001)
        r45 = rk45(f, (0, 2), [1.0], tol=1e-10)
        exact = math.exp(-2)
        assert abs(r45.y[-1, 0] - exact) <= abs(r4.y[-1, 0] - exact) + 1e-8


class TestBDF:
    def test_exponential_decay(self):
        result = bdf(lambda t, y: -y, (0, 2), [1.0], tol=1e-8)
        assert result.method == "bdf"
        assert result.y[-1, 0] == pytest.approx(math.exp(-2), rel=1e-4)

    def test_stiff_system(self):
        """Stiff problem: dy/dt = -1000*y, y(0)=1. BDF should handle this."""
        result = bdf(lambda t, y: -1000 * y, (0, 0.01), [1.0], tol=1e-6)
        exact = math.exp(-1000 * 0.01)
        assert result.y[-1, 0] == pytest.approx(exact, rel=0.01)


class TestAllSolversAgree:
    def test_exponential(self):
        f = lambda t, y: -y
        exact = math.exp(-3)
        r4 = rk4(f, (0, 3), [1.0], dt=0.01)
        r45 = rk45(f, (0, 3), [1.0], tol=1e-8)
        r_bdf = bdf(f, (0, 3), [1.0], tol=1e-8)

        assert r4.y[-1, 0] == pytest.approx(exact, rel=1e-4)
        assert r45.y[-1, 0] == pytest.approx(exact, rel=1e-6)
        assert r_bdf.y[-1, 0] == pytest.approx(exact, rel=1e-4)
