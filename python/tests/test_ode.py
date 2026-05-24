"""Tests for ODE solvers — migrated to new unified API."""

import pytest
import math
import numpy as np

from pricebook.numerical._ode import solve_ode, ODEMethod, ODEResult


class TestRK4:
    def test_exponential_decay(self):
        """dy/dt = -y, y(0) = 1 → y(t) = exp(-t)."""
        result = solve_ode(lambda t, y: -y, (0, 2), [1.0], ODEMethod.RK4, n_steps=200)
        assert isinstance(result, ODEResult)
        assert result.method == "rk4"
        assert result.y[-1, 0] == pytest.approx(math.exp(-2), rel=1e-4)

    def test_harmonic_oscillator(self):
        def f(t, y):
            return np.array([y[1], -y[0]])
        result = solve_ode(f, (0, 2 * math.pi), [0.0, 1.0], ODEMethod.RK4, n_steps=500)
        assert result.y[-1, 0] == pytest.approx(0.0, abs=1e-3)
        assert result.y[-1, 1] == pytest.approx(1.0, abs=1e-3)

    def test_linear_growth(self):
        result = solve_ode(lambda t, y: np.array([1.0]), (0, 5), [0.0], ODEMethod.RK4, n_steps=50)
        assert result.y[-1, 0] == pytest.approx(5.0, abs=0.01)


class TestRK45:
    def test_exponential_decay(self):
        result = solve_ode(lambda t, y: -y, (0, 2), [1.0], ODEMethod.RK45, tol=1e-8)
        assert result.method == "rk45"
        assert result.y[-1, 0] == pytest.approx(math.exp(-2), rel=1e-6)

    def test_harmonic_oscillator(self):
        def f(t, y):
            return np.array([y[1], -y[0]])
        result = solve_ode(f, (0, 2 * math.pi), [0.0, 1.0], ODEMethod.RK45, tol=1e-8)
        assert result.y[-1, 0] == pytest.approx(0.0, abs=1e-4)


class TestBDF:
    def test_exponential_decay(self):
        result = solve_ode(lambda t, y: -y, (0, 2), [1.0], ODEMethod.BDF, tol=1e-8)
        assert result.method == "bdf"
        assert result.y[-1, 0] == pytest.approx(math.exp(-2), rel=1e-4)

    def test_stiff_system(self):
        result = solve_ode(lambda t, y: -1000 * y, (0, 0.01), [1.0], ODEMethod.BDF, tol=1e-6)
        exact = math.exp(-10)
        assert result.y[-1, 0] == pytest.approx(exact, rel=0.01)


class TestAllSolversAgree:
    def test_exponential(self):
        f = lambda t, y: -y
        exact = math.exp(-3)
        r4 = solve_ode(f, (0, 3), [1.0], ODEMethod.RK4, n_steps=300)
        r45 = solve_ode(f, (0, 3), [1.0], ODEMethod.RK45, tol=1e-8)
        r_bdf = solve_ode(f, (0, 3), [1.0], ODEMethod.BDF, tol=1e-8)

        assert r4.y[-1, 0] == pytest.approx(exact, rel=1e-4)
        assert r45.y[-1, 0] == pytest.approx(exact, rel=1e-6)
        assert r_bdf.y[-1, 0] == pytest.approx(exact, rel=1e-4)
