"""Tests for numerical._ode — comprehensive ODE solver suite."""

import math
import pytest
import numpy as np

from pricebook.numerical._ode import (
    ODESolver, ODEMethod, ODEResult,
    solve_ode, solve_backward, solve_riccati, solve_system,
    euler, rk4, rk45, bdf, adams,
)


# ═══════════════════════════════════════════════════════════════
# Backward compatibility (old API still works)
# ═══════════════════════════════════════════════════════════════

class TestBackwardCompat:
    def test_euler(self):
        result = euler(lambda t, y: y, (0, 1), [1.0], n_steps=1000)
        assert abs(result.y[-1][0] - math.e) < 0.01

    def test_rk4(self):
        result = rk4(lambda t, y: y, (0, 1), [1.0], n_steps=100)
        assert abs(result.y[-1][0] - math.e) < 1e-6

    def test_rk45(self):
        result = rk45(lambda t, y: y, (0, 1), [1.0])
        assert abs(result.y[-1][0] - math.e) < 1e-4

    def test_bdf(self):
        result = bdf(lambda t, y: -100 * np.array(y), (0, 0.1), [1.0])
        assert abs(result.y[-1][0] - math.exp(-10)) < 0.05

    def test_adams(self):
        result = adams(lambda t, y: y, (0, 1), [1.0])
        assert abs(result.y[-1][0] - math.e) < 1e-4


# ═══════════════════════════════════════════════════════════════
# Method selection
# ═══════════════════════════════════════════════════════════════

class TestMethodSelection:
    @pytest.mark.parametrize("method", [
        ODEMethod.EULER, ODEMethod.RK4, ODEMethod.RK45,
        ODEMethod.BDF, ODEMethod.LSODA, ODEMethod.RADAU,
    ])
    def test_all_methods_solve_exponential(self, method):
        """Every method should solve dy/dt = y, y(0)=1 → y(1)=e."""
        result = solve_ode(lambda t, y: y, (0, 1), [1.0], method=method,
                            n_steps=500, tol=1e-6)
        assert result.success or method in (ODEMethod.EULER, ODEMethod.RK4, ODEMethod.IMPLICIT_EULER)
        assert abs(result.y[-1][0] - math.e) < 0.1

    def test_dop853_high_order(self):
        result = solve_ode(lambda t, y: y, (0, 1), [1.0], method=ODEMethod.DOP853)
        assert abs(result.y[-1][0] - math.e) < 1e-6

    def test_rk23_low_order(self):
        result = solve_ode(lambda t, y: y, (0, 1), [1.0], method=ODEMethod.RK23)
        assert abs(result.y[-1][0] - math.e) < 1e-3


# ═══════════════════════════════════════════════════════════════
# Class-based solver
# ═══════════════════════════════════════════════════════════════

class TestODESolver:
    def test_class_api(self):
        solver = ODESolver(method=ODEMethod.RK45, tol=1e-8)
        result = solver.solve(lambda t, y: y, (0, 1), [1.0])
        assert abs(result.y[-1][0] - math.e) < 1e-6

    def test_reuse_solver(self):
        """Same solver instance can be reused."""
        solver = ODESolver(ODEMethod.RK4, n_steps=200)
        r1 = solver.solve(lambda t, y: y, (0, 1), [1.0])
        r2 = solver.solve(lambda t, y: -y, (0, 1), [1.0])
        assert abs(r1.y[-1][0] - math.e) < 1e-4
        assert abs(r2.y[-1][0] - math.exp(-1)) < 1e-4


# ═══════════════════════════════════════════════════════════════
# Stiff systems
# ═══════════════════════════════════════════════════════════════

class TestStiff:
    def test_bdf_stiff(self):
        """Stiff system: dy/dt = -1000y, y(0)=1."""
        result = solve_ode(lambda t, y: -1000 * y, (0, 0.01), [1.0],
                            method=ODEMethod.BDF)
        assert abs(result.y[-1][0] - math.exp(-10)) < 0.01

    def test_radau_stiff(self):
        result = solve_ode(lambda t, y: -1000 * y, (0, 0.01), [1.0],
                            method=ODEMethod.RADAU)
        assert abs(result.y[-1][0] - math.exp(-10)) < 0.01

    def test_bdf_with_jacobian(self):
        """BDF with analytical Jacobian should be more efficient."""
        def f(t, y):
            return np.array([-100 * y[0], -0.1 * y[1]])

        def jac(t, y):
            return np.array([[-100, 0], [0, -0.1]])

        r = solve_ode(f, (0, 1), [1.0, 1.0], method=ODEMethod.BDF, jac=jac)
        assert abs(r.y[-1][0] - math.exp(-100)) < 1e-3
        assert abs(r.y[-1][1] - math.exp(-0.1)) < 1e-3

    def test_lsoda_auto_detection(self):
        """LSODA should auto-detect stiffness."""
        result = solve_system(lambda t, y: -500 * y, (0, 0.01), np.array([1.0]))
        assert result.method == "lsoda"


# ═══════════════════════════════════════════════════════════════
# Implicit Euler
# ═══════════════════════════════════════════════════════════════

class TestImplicitEuler:
    def test_basic(self):
        result = solve_ode(lambda t, y: -y, (0, 1), [1.0],
                            method=ODEMethod.IMPLICIT_EULER, n_steps=200)
        assert abs(result.y[-1][0] - math.exp(-1)) < 0.05

    def test_stiff_stable(self):
        """Implicit Euler should handle stiff without blowing up."""
        result = solve_ode(lambda t, y: -100 * y, (0, 0.1), [1.0],
                            method=ODEMethod.IMPLICIT_EULER, n_steps=50)
        assert abs(result.y[-1][0]) < 0.1  # decayed, not exploded


# ═══════════════════════════════════════════════════════════════
# Dense output
# ═══════════════════════════════════════════════════════════════

class TestDenseOutput:
    def test_interpolation(self):
        result = solve_ode(lambda t, y: y, (0, 1), [1.0],
                            method=ODEMethod.RK45, dense_output=True)
        # Query at t=0.5
        y_half = result(0.5)
        val = float(np.atleast_1d(y_half).flatten()[0])
        assert abs(val - math.exp(0.5)) < 1e-3

    def test_fallback_interp(self):
        """Fixed-step methods use linear interpolation fallback."""
        result = solve_ode(lambda t, y: y, (0, 1), [1.0],
                            method=ODEMethod.RK4, n_steps=100)
        y_half = result(0.5)
        assert abs(float(y_half) - math.exp(0.5)) < 0.01


# ═══════════════════════════════════════════════════════════════
# Backward integration
# ═══════════════════════════════════════════════════════════════

class TestBackward:
    def test_backward(self):
        """Solve backward: dy/dt = y from t=1 to t=0, y(1)=e → y(0)=1."""
        result = solve_backward(lambda t, y: y, (0, 1), [math.e])
        assert abs(result.y[0][0] - 1.0) < 0.01

    def test_backward_pde_style(self):
        """PDE backward in time: terminal condition → initial value."""
        result = solve_backward(lambda t, y: -2 * y, (0, 1), [math.exp(-2)])
        assert abs(result.y[0][0] - 1.0) < 0.1


# ═══════════════════════════════════════════════════════════════
# Riccati ODE
# ═══════════════════════════════════════════════════════════════

class TestRiccati:
    def test_linear_riccati(self):
        """Riccati with c=0 → linear ODE: dy/dt = a + by."""
        # dy/dt = 1 + 0×y + 0×y² → y = t + y0
        result = solve_riccati(a=1.0, b=0.0, c=0.0, t_span=(0, 1), y0=0.0)
        assert abs(result.y[-1][0] - 1.0) < 0.01

    def test_quadratic_riccati(self):
        """dy/dt = -y², y(0)=1 → y(t) = 1/(1+t)."""
        result = solve_riccati(a=0.0, b=0.0, c=-1.0, t_span=(0, 1), y0=1.0)
        expected = 1.0 / 2.0  # y(1) = 1/(1+1)
        assert abs(result.y[-1][0] - expected) < 0.01

    def test_affine_riccati(self):
        """dy/dt = 1 - y², y(0)=0 → y(t) = tanh(t)."""
        result = solve_riccati(a=1.0, b=0.0, c=-1.0, t_span=(0, 1), y0=0.0)
        expected = math.tanh(1.0)
        assert abs(result.y[-1][0] - expected) < 0.01


# ═══════════════════════════════════════════════════════════════
# Systems
# ═══════════════════════════════════════════════════════════════

class TestSystems:
    def test_2d_system(self):
        """Coupled system: dx/dt = -y, dy/dt = x → rotation."""
        def f(t, y):
            return np.array([-y[1], y[0]])
        result = solve_ode(f, (0, 2 * math.pi), [1.0, 0.0], method=ODEMethod.RK45)
        # After full rotation: back to (1, 0)
        assert abs(result.y[-1][0] - 1.0) < 0.01
        assert abs(result.y[-1][1]) < 0.01

    def test_3d_lorenz(self):
        """Lorenz system (just verify it runs without error)."""
        sigma, rho, beta = 10, 28, 8/3
        def f(t, y):
            return np.array([
                sigma * (y[1] - y[0]),
                y[0] * (rho - y[2]) - y[1],
                y[0] * y[1] - beta * y[2],
            ])
        result = solve_ode(f, (0, 1), [1.0, 1.0, 1.0], method=ODEMethod.RK45)
        assert result.success


# ═══════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════

class TestSerialization:
    def test_to_dict(self):
        result = solve_ode(lambda t, y: y, (0, 1), [1.0])
        d = result.to_dict()
        assert "method" in d
        assert "success" in d
        assert "n_evaluations" in d
