"""Tests for numerical integration and differentiation frameworks."""

import math
import pytest
import numpy as np

from pricebook.numerical._integrate import (
    integrate, IntegrationMethod, IntegrationResult,
    integrate_2d, integrate_semi_infinite, integrate_complex_contour,
)
from pricebook.numerical._differentiate import (
    derivative, gradient, jacobian, hessian,
    DiffMethod, DiffResult,
)


# ═══════════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════════

class TestIntegrationMethods:
    @pytest.mark.parametrize("method", [
        IntegrationMethod.ADAPTIVE, IntegrationMethod.GAUSS_LEGENDRE,
        IntegrationMethod.SIMPSON, IntegrationMethod.TRAPEZOID,
        IntegrationMethod.CLENSHAW_CURTIS, IntegrationMethod.TANH_SINH,
    ])
    def test_all_methods_integrate_x2(self, method):
        """∫₀¹ x² dx = 1/3."""
        result = integrate(lambda x: x**2, 0, 1, method=method)
        assert abs(result.value - 1/3) < 0.01

    def test_adaptive_sin(self):
        """∫₀^π sin(x) dx = 2."""
        r = integrate(math.sin, 0, math.pi)
        assert abs(r.value - 2.0) < 1e-10

    def test_gauss_legendre_polynomial(self):
        """Gauss-Legendre exact for polynomials."""
        r = integrate(lambda x: x**5 - 3*x**3 + 2, -1, 1,
                       method=IntegrationMethod.GAUSS_LEGENDRE, n=4)
        exact = 2 * 2  # integral of even terms only
        assert abs(r.value - exact) < 1e-8

    def test_tanh_sinh_singular(self):
        """∫₀¹ 1/√x dx = 2 (endpoint singularity)."""
        r = integrate(lambda x: 1/math.sqrt(max(x, 1e-15)), 0.001, 1,
                       method=IntegrationMethod.TANH_SINH, n=100)
        assert abs(r.value - (2 - 2*math.sqrt(0.001))) < 0.1


class TestSemiInfinite:
    def test_exp_decay(self):
        """∫₀^∞ e^{-x} dx = 1."""
        r = integrate(lambda x: math.exp(-x), 0, np.inf, method=IntegrationMethod.ADAPTIVE)
        assert abs(r.value - 1.0) < 1e-6

    def test_gaussian(self):
        """∫_{-∞}^{∞} e^{-x²} dx = √π."""
        r = integrate(lambda x: math.exp(-x**2), -np.inf, np.inf)
        assert abs(r.value - math.sqrt(math.pi)) < 1e-6


class TestIntegration2D:
    def test_unit_square(self):
        """∫₀¹∫₀¹ (x+y) dy dx = 1."""
        r = integrate_2d(lambda y, x: x + y, (0, 1), (0, 1))
        assert abs(r.value - 1.0) < 1e-6


class TestComplexContour:
    def test_unit_circle(self):
        """∮ 1/z dz = 2πi around unit circle."""
        result = integrate_complex_contour(
            f=lambda z: 1/z,
            t_range=(0, 2 * math.pi),
            contour=lambda t: np.exp(1j * t),
            contour_derivative=lambda t: 1j * np.exp(1j * t),
            n=50,
        )
        assert abs(result - 2j * math.pi) < 0.01


class TestIntegrationResult:
    def test_to_dict(self):
        r = integrate(lambda x: x, 0, 1)
        d = r.to_dict()
        assert "value" in d
        assert "method" in d


# ═══════════════════════════════════════════════════════════════
# Differentiation
# ═══════════════════════════════════════════════════════════════

class TestDerivativeMethods:
    @pytest.mark.parametrize("method", [
        DiffMethod.FORWARD, DiffMethod.CENTRAL,
        DiffMethod.RICHARDSON, DiffMethod.FIVE_POINT,
    ])
    def test_all_methods_derivative_x2(self, method):
        """d/dx(x²) at x=3 = 6."""
        r = derivative(lambda x: x**2, 3.0, method=method)
        assert abs(r.value - 6.0) < 0.01

    def test_complex_step(self):
        """Complex-step derivative: machine precision."""
        r = derivative(lambda x: x**3, 2.0, method=DiffMethod.COMPLEX_STEP)
        assert abs(r.value - 12.0) < 1e-12

    def test_complex_step_trig(self):
        """d/dx(sin(x)) at x=π/4 = cos(π/4)."""
        import cmath
        r = derivative(lambda x: cmath.sin(x), math.pi/4, method=DiffMethod.COMPLEX_STEP)
        assert abs(r.value - math.cos(math.pi/4)) < 1e-12

    def test_richardson_high_accuracy(self):
        r = derivative(lambda x: math.exp(x), 1.0, method=DiffMethod.RICHARDSON)
        assert abs(r.value - math.e) < 1e-8

    def test_second_derivative(self):
        """d²/dx²(x³) at x=2 = 12."""
        r = derivative(lambda x: x**3, 2.0, order=2)
        assert abs(r.value - 12.0) < 0.1


class TestGradient:
    def test_2d(self):
        """∇(x² + y²) at (3, 4) = (6, 8)."""
        r = gradient(lambda x: x[0]**2 + x[1]**2, np.array([3.0, 4.0]))
        np.testing.assert_allclose(r.value, [6.0, 8.0], atol=1e-6)

    def test_complex_step_gradient(self):
        r = gradient(lambda x: x[0]**2 + x[1]**3, np.array([2.0, 3.0]),
                      method=DiffMethod.COMPLEX_STEP)
        np.testing.assert_allclose(r.value, [4.0, 27.0], atol=1e-10)


class TestJacobian:
    def test_2x2(self):
        """J of f(x,y) = (x²y, 5x+sin(y)) at (1, π/6)."""
        def f(x):
            return np.array([x[0]**2 * x[1], 5 * x[0] + math.sin(x[1])])
        r = jacobian(f, np.array([1.0, math.pi/6]))
        # J = [[2xy, x²], [5, cos(y)]]
        assert abs(r.value[0, 0] - 2 * 1 * math.pi/6) < 0.01
        assert abs(r.value[1, 0] - 5.0) < 0.01
        assert abs(r.value[1, 1] - math.cos(math.pi/6)) < 0.01

    def test_shape(self):
        def f(x):
            return np.array([x[0] + x[1], x[0] * x[1], x[0]**2])
        r = jacobian(f, np.array([1.0, 2.0]))
        assert r.value.shape == (3, 2)


class TestHessian:
    def test_quadratic(self):
        """H of f(x,y) = x² + 3xy + y² is [[2, 3], [3, 2]]."""
        def f(x):
            return x[0]**2 + 3 * x[0] * x[1] + x[1]**2
        r = hessian(f, np.array([1.0, 1.0]))
        np.testing.assert_allclose(r.value, [[2, 3], [3, 2]], atol=0.1)

    def test_symmetric(self):
        def f(x):
            return math.sin(x[0]) * math.cos(x[1])
        r = hessian(f, np.array([1.0, 1.0]))
        assert abs(r.value[0, 1] - r.value[1, 0]) < 1e-6


class TestAutoStep:
    def test_central_step(self):
        """Auto step for central should give good accuracy."""
        r = derivative(lambda x: math.exp(x), 0.0)
        assert abs(r.value - 1.0) < 1e-8

    def test_to_dict(self):
        r = derivative(lambda x: x**2, 1.0)
        d = r.to_dict()
        assert "method" in d
        assert "n_evaluations" in d
