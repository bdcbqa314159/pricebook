"""Tests for approximation theory: Chebyshev, Padé, Richardson, B-splines."""

import math

import numpy as np
import pytest

from pricebook.approximation import (
    ChebyshevInterpolant,
    PadeApproximant,
    RichardsonTable,
    bspline_basis,
    chebyshev_interpolate,
    pade_approximant,
    richardson_table,
)


# ---- Chebyshev interpolation ----

class TestChebyshevInterpolation:
    def test_polynomial_exact(self):
        """Chebyshev of degree n reproduces polynomials of degree ≤ n exactly."""
        f = lambda x: x**3 - 2*x + 1
        interp = chebyshev_interpolate(f, -1, 1, n=5)
        x_test = np.linspace(-1, 1, 50)
        np.testing.assert_allclose(interp.evaluate(x_test), f(x_test), atol=1e-10)

    def test_exp_accurate(self):
        """exp(x) on [-1,1] with n=20 should be accurate to ~1e-12."""
        interp = chebyshev_interpolate(np.exp, -1, 1, n=20)
        x_test = np.linspace(-0.9, 0.9, 100)
        err = np.max(np.abs(interp.evaluate(x_test) - np.exp(x_test)))
        assert err < 1e-10

    def test_arbitrary_interval(self):
        """Interpolation on [50, 150] for BS pricing."""
        f = lambda x: np.maximum(x - 100, 0.0)  # call payoff
        interp = chebyshev_interpolate(f, 50, 150, n=50)
        # At S=120: payoff = 20
        assert interp.evaluate(120.0) == pytest.approx(20.0, abs=0.5)

    def test_convergence_diagnostic(self):
        interp = chebyshev_interpolate(np.exp, -1, 1, n=20)
        # Last coefficients should be tiny for smooth functions
        assert interp.max_coeff_magnitude() < 1e-8

    def test_scalar_evaluation(self):
        interp = chebyshev_interpolate(np.sin, 0, math.pi, n=15)
        assert interp.evaluate(math.pi / 2) == pytest.approx(1.0, abs=1e-8)


# ---- Padé approximant ----

class TestPadeApproximant:
    def test_exp_pade_22(self):
        """Padé [2/2] of exp(x) should be accurate near x=0."""
        # Taylor of exp: 1, 1, 1/2, 1/6, 1/24
        coeffs = [1, 1, 0.5, 1/6, 1/24]
        pade = pade_approximant(coeffs, L=2, M=2)
        # exp(0.5) ≈ 1.6487
        assert pade.evaluate(0.5) == pytest.approx(math.exp(0.5), rel=1e-3)

    def test_pade_10_is_taylor(self):
        """Padé [L/0] = Taylor polynomial of degree L."""
        coeffs = [1, 1, 0.5, 1/6]
        pade = pade_approximant(coeffs, L=3, M=0)
        x = 0.3
        taylor = 1 + x + 0.5*x**2 + x**3/6
        assert pade.evaluate(x) == pytest.approx(taylor, rel=1e-10)

    def test_pade_reproduces_rational(self):
        """Padé [1/1] of 1/(1+x): Taylor = 1, -1, 1, -1 → exact."""
        coeffs = [1, -1, 1, -1]
        pade = pade_approximant(coeffs, L=1, M=1)
        # Should be exactly 1/(1+x)
        assert pade.evaluate(0.5) == pytest.approx(1 / 1.5, rel=1e-8)
        assert pade.evaluate(2.0) == pytest.approx(1 / 3.0, rel=1e-8)


# ---- Richardson table ----

class TestRichardsonTable:
    def test_exact_extrapolation(self):
        """O(h²) estimates should give exact result after one Richardson step."""
        # f(h) = 1 + h², f(h/2) = 1 + h²/4 → Richardson = 1
        values = [1 + 0.1**2, 1 + 0.05**2, 1 + 0.025**2]
        result = richardson_table(values, order=2)
        assert result.best_estimate == pytest.approx(1.0, abs=1e-8)

    def test_table_shape(self):
        values = [1.1, 1.025, 1.006]
        result = richardson_table(values)
        assert result.table.shape == (3, 3)
        assert len(result.estimates) == 3

    def test_diagonal_improves(self):
        """Each diagonal entry should be closer to the true value."""
        true = 1.0
        values = [true + 0.1, true + 0.025, true + 0.00625]
        result = richardson_table(values, order=2)
        errors = [abs(e - true) for e in result.estimates]
        # Each should be better (or at least not much worse)
        assert errors[-1] < errors[0]

    def test_single_value(self):
        result = richardson_table([1.5])
        assert result.best_estimate == pytest.approx(1.5)


# ---- B-spline basis ----

class TestBSplineBasis:
    def test_degree_zero(self):
        """Degree 0: piecewise constant, 1 in the interval."""
        knots = [0, 1, 2, 3, 4]
        assert bspline_basis(0.5, knots, 0, 0) == 1.0
        assert bspline_basis(1.5, knots, 0, 0) == 0.0
        assert bspline_basis(1.5, knots, 0, 1) == 1.0

    def test_partition_of_unity(self):
        """B-splines of degree d on a uniform knot vector sum to 1."""
        knots = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
        degree = 3
        x = 2.5
        total = sum(bspline_basis(x, knots, degree, i) for i in range(len(knots) - degree - 1))
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_non_negative(self):
        """B-splines are non-negative."""
        knots = [0, 0, 0, 1, 2, 3, 3, 3]
        degree = 2
        for x in np.linspace(0.01, 2.99, 50):
            for i in range(len(knots) - degree - 1):
                assert bspline_basis(x, knots, degree, i) >= -1e-15

    def test_linear_spline(self):
        """Degree 1 B-spline is a hat function."""
        knots = [0, 1, 2, 3]
        # B_{1,1}(x): peaks at x=2, zero at x=1 and x=3
        assert bspline_basis(2.0, knots, 1, 1) == pytest.approx(1.0)
        assert bspline_basis(1.0, knots, 1, 1) == pytest.approx(0.0)
        assert bspline_basis(1.5, knots, 1, 1) == pytest.approx(0.5)
