"""Tests for root-finding solvers."""

import pytest
import math

from pricebook.solvers import newton, secant, halley, itp, brentq, SolverResult


# Test functions
def f_cubic(x):
    return x**3 - 2*x - 5  # root near 2.0946

def f_cubic_prime(x):
    return 3*x**2 - 2

def f_cubic_prime2(x):
    return 6*x

def f_sin(x):
    return math.sin(x)  # root at 0, pi, 2pi, ...

def f_sin_prime(x):
    return math.cos(x)

def f_sin_prime2(x):
    return -math.sin(x)

CUBIC_ROOT = 2.0945514815423265  # known root of x^3 - 2x - 5


class TestNewton:
    def test_cubic(self):
        r = newton(f_cubic, f_cubic_prime, x0=2.0)
        assert r.converged
        assert r.root == pytest.approx(CUBIC_ROOT, abs=1e-10)

    def test_sin(self):
        r = newton(f_sin, f_sin_prime, x0=3.0)
        assert r.converged
        assert r.root == pytest.approx(math.pi, abs=1e-10)

    def test_fewer_iterations_than_brent(self):
        """Newton should converge in fewer iterations than Brent."""
        r = newton(f_cubic, f_cubic_prime, x0=2.0)
        assert r.iterations < 10  # Newton is quadratic

    def test_returns_solver_result(self):
        r = newton(f_cubic, f_cubic_prime, x0=2.0)
        assert isinstance(r, SolverResult)
        assert abs(r.function_value) < 1e-10


class TestSecant:
    def test_cubic(self):
        r = secant(f_cubic, x0=2.0, x1=3.0)
        assert r.converged
        assert r.root == pytest.approx(CUBIC_ROOT, abs=1e-10)

    def test_sin(self):
        r = secant(f_sin, x0=3.0, x1=3.5)
        assert r.converged
        assert r.root == pytest.approx(math.pi, abs=1e-10)

    def test_no_derivative_needed(self):
        """Secant works without fprime."""
        r = secant(f_cubic, x0=1.0, x1=3.0)
        assert r.converged


class TestHalley:
    def test_cubic(self):
        r = halley(f_cubic, f_cubic_prime, f_cubic_prime2, x0=2.0)
        assert r.converged
        assert r.root == pytest.approx(CUBIC_ROOT, abs=1e-12)

    def test_fewer_iterations_than_newton(self):
        """Halley (cubic convergence) should need fewer iterations than Newton."""
        r_h = halley(f_cubic, f_cubic_prime, f_cubic_prime2, x0=2.0)
        r_n = newton(f_cubic, f_cubic_prime, x0=2.0)
        assert r_h.iterations <= r_n.iterations

    def test_sin(self):
        r = halley(f_sin, f_sin_prime, f_sin_prime2, x0=3.0)
        assert r.converged
        assert r.root == pytest.approx(math.pi, abs=1e-12)


class TestITP:
    def test_cubic(self):
        r = itp(f_cubic, a=2.0, b=3.0)
        assert r.converged
        assert r.root == pytest.approx(CUBIC_ROOT, abs=1e-10)

    def test_sin(self):
        r = itp(f_sin, a=3.0, b=3.5)
        assert r.converged
        assert r.root == pytest.approx(math.pi, abs=1e-10)

    def test_opposite_signs_required(self):
        with pytest.raises(ValueError, match="opposite signs"):
            itp(f_cubic, a=3.0, b=4.0)

    def test_fewer_iterations_than_bisection(self):
        """ITP should beat pure bisection."""
        # Pure bisection on [2, 3] needs ~40 iterations for 1e-12 tol
        r = itp(f_cubic, a=2.0, b=3.0)
        assert r.iterations < 40


class TestBrentq:
    def test_cubic(self):
        root = brentq(f_cubic, 2.0, 3.0)
        assert root == pytest.approx(CUBIC_ROOT, abs=1e-10)

    def test_returns_float(self):
        """brentq returns float for backward compatibility."""
        root = brentq(f_cubic, 2.0, 3.0)
        assert isinstance(root, float)


class TestAllSolversAgree:
    """All solvers find the same root."""

    def test_cubic_root(self):
        r_newton = newton(f_cubic, f_cubic_prime, x0=2.0)
        r_secant = secant(f_cubic, x0=2.0, x1=3.0)
        r_halley = halley(f_cubic, f_cubic_prime, f_cubic_prime2, x0=2.0)
        r_itp = itp(f_cubic, a=2.0, b=3.0)
        r_brent = brentq(f_cubic, 2.0, 3.0)

        assert r_newton.root == pytest.approx(CUBIC_ROOT, abs=1e-10)
        assert r_secant.root == pytest.approx(CUBIC_ROOT, abs=1e-10)
        assert r_halley.root == pytest.approx(CUBIC_ROOT, abs=1e-10)
        assert r_itp.root == pytest.approx(CUBIC_ROOT, abs=1e-10)
        assert r_brent == pytest.approx(CUBIC_ROOT, abs=1e-10)

    def test_implied_vol_style(self):
        """All solvers find the same root for a pricing-style problem."""
        target = 0.25
        def f(x):
            return x**2 - target  # root at sqrt(0.25) = 0.5
        def fp(x):
            return 2*x
        def fpp(x):
            return 2.0

        expected = 0.5
        assert newton(f, fp, x0=0.3).root == pytest.approx(expected, abs=1e-10)
        assert secant(f, x0=0.3, x1=0.7).root == pytest.approx(expected, abs=1e-10)
        assert halley(f, fp, fpp, x0=0.3).root == pytest.approx(expected, abs=1e-10)
        assert itp(f, a=0.1, b=0.9).root == pytest.approx(expected, abs=1e-10)
        assert brentq(f, 0.1, 0.9) == pytest.approx(expected, abs=1e-10)
