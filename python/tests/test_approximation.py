"""Tests for approximation theory: Chebyshev, Padé, Richardson, B-splines."""

import math

import numpy as np
import pytest

from pricebook.core.approximation import (
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

    def test_degree_zero_constant(self):
        """n=0 yields a constant interpolant (no ZeroDivisionError)."""
        interp = chebyshev_interpolate(lambda x: 7.0, 0, 2, n=0)
        assert interp.evaluate(0.0) == pytest.approx(7.0)
        assert interp.evaluate(2.0) == pytest.approx(7.0)

    def test_degenerate_interval_raises(self):
        with pytest.raises(ValueError, match="degenerate"):
            chebyshev_interpolate(np.exp, 1.0, 1.0, n=10)


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

    def test_singular_denominator_raises(self):
        """A singular denominator system must error, not silently truncate."""
        # c = [1, 0, 0, 0, 0] makes the [2/2] denominator system singular.
        with pytest.raises(ValueError, match="singular or"):
            pade_approximant([1, 0, 0, 0, 0], L=2, M=2)

    def test_near_singular_threshold(self):
        """Singular raises; a small perturbation that is well-conditioned must
        succeed — pins the guard direction (not over-eager)."""
        with pytest.raises(ValueError, match="singular or"):
            pade_approximant([1, 0, 0, 0, 0], L=2, M=2)
        pade = pade_approximant([1, 0.0, 0.5, 0.1, 0.2], L=2, M=2)  # well-conditioned
        assert pade.denominator[0] == 1.0
        assert np.all(np.isfinite(pade.numerator))

    def test_exp_golden_coefficients(self):
        """Pin the recovered coefficients of exp's diagonal Padés against the
        textbook closed forms — a value check (test_exp_pade_22) can't catch a
        wrong-but-close coefficient set."""
        # [1/1] of exp = (1 + x/2) / (1 - x/2)
        p11 = pade_approximant([1, 1, 0.5], L=1, M=1)
        np.testing.assert_allclose(p11.numerator, [1.0, 0.5], atol=1e-12)
        np.testing.assert_allclose(p11.denominator, [1.0, -0.5], atol=1e-12)
        # [2/2] of exp = (1 + x/2 + x²/12) / (1 - x/2 + x²/12)
        p22 = pade_approximant([1, 1, 0.5, 1 / 6, 1 / 24], L=2, M=2)
        np.testing.assert_allclose(p22.numerator, [1.0, 0.5, 1 / 12], atol=1e-12)
        np.testing.assert_allclose(p22.denominator, [1.0, -0.5, 1 / 12], atol=1e-12)


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

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            richardson_table([])

    def test_order_drives_cancellation_both_directions(self):
        """The `order` must actually drive the cancellation — checked in BOTH
        directions. A pure h^3 error series is exact with order=3 but the
        order=2 ladder {2,4,6} can never hit h^3, so it must NOT be exact.
        A no-op or wrong-exponent bug fails the negative direction."""
        h = 0.2
        values = [1 + (h / 2**i) ** 3 for i in range(4)]  # pure h^3 error
        assert abs(richardson_table(values, order=3).best_estimate - 1) < 1e-12
        assert abs(richardson_table(values, order=2).best_estimate - 1) > 1e-7

    def test_best_estimate_is_table_corner(self):
        values = [1.1, 1.025, 1.006, 1.0015]
        r = richardson_table(values, order=2)
        assert r.best_estimate == float(r.table[-1, -1])
        assert r.best_estimate == r.estimates[-1]


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

    def test_index_out_of_range_raises(self):
        """Negative or too-large i must error, not wrap to a wrong value."""
        knots = [0, 1, 2, 3, 4]
        with pytest.raises(IndexError):
            bspline_basis(1.5, knots, 0, -1)
        with pytest.raises(IndexError):
            bspline_basis(1.5, knots, 0, 99)

    def test_partition_of_unity_swept(self):
        """Partition of unity must hold across the WHOLE interior span, not at
        one cherry-picked point (the single-point test could pass by luck)."""
        knots = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
        degree = 3
        nbasis = len(knots) - degree - 1
        for x in np.linspace(0.0, 4.0, 101)[:-1]:  # exclude the right endpoint
            total = sum(bspline_basis(x, knots, degree, i) for i in range(nbasis))
            assert total == pytest.approx(1.0, abs=1e-12)

    def test_right_endpoint_gap_is_documented(self):
        """KNOWN convention: B_{i,0} uses the half-open [t[i], t[i+1]), so at
        x == t[-1] every basis is 0 and partition-of-unity drops to 0. Pinned
        so a future change to the convention is visible, not silent."""
        knots = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
        degree = 3
        nbasis = len(knots) - degree - 1
        total = sum(bspline_basis(4.0, knots, degree, i) for i in range(nbasis))
        assert total == 0.0  # not 1.0 — the right-endpoint half-open gap

    def test_cubic_matches_scipy_oracle(self):
        """Cubic Cox-de Boor pinned against scipy.interpolate as an independent
        oracle (no elegant closed form for cubic basis at off-knot points)."""
        from scipy.interpolate import BSpline

        knots = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
        degree = 3
        for i in (2, 3, 4):
            local = np.asarray(knots[i : i + degree + 2], dtype=float)
            ref = BSpline.basis_element(local, extrapolate=False)
            for x in (1.3, 2.6, 3.4):
                expected = float(ref(x))
                expected = 0.0 if np.isnan(expected) else expected
                assert bspline_basis(x, knots, degree, i) == pytest.approx(
                    expected, abs=1e-12
                )
