"""Tests for numerical integration."""

import pytest
import math

from pricebook.quadrature import (
    gauss_legendre,
    gauss_laguerre,
    gauss_hermite,
    adaptive_simpson,
    QuadratureResult,
)


class TestGaussLegendre:
    def test_constant(self):
        """Integral of 1 over [0, 1] = 1."""
        r = gauss_legendre(lambda x: 1.0, 0, 1, n=4)
        assert r.value == pytest.approx(1.0)

    def test_polynomial_exact(self):
        """n=4 is exact for degree ≤ 7 polynomials."""
        # x^3 on [0, 1] = 1/4
        r = gauss_legendre(lambda x: x**3, 0, 1, n=4)
        assert r.value == pytest.approx(0.25, abs=1e-14)

    def test_high_degree_exact(self):
        """n=8 exact for degree ≤ 15."""
        # x^10 on [0, 1] = 1/11
        r = gauss_legendre(lambda x: x**10, 0, 1, n=8)
        assert r.value == pytest.approx(1.0 / 11, abs=1e-12)

    def test_sin(self):
        """Integral of sin(x) on [0, pi] = 2."""
        r = gauss_legendre(math.sin, 0, math.pi, n=16)
        assert r.value == pytest.approx(2.0, abs=1e-12)

    def test_exp(self):
        """Integral of exp(x) on [0, 1] = e - 1."""
        r = gauss_legendre(math.exp, 0, 1, n=16)
        assert r.value == pytest.approx(math.e - 1, abs=1e-12)

    def test_returns_result(self):
        r = gauss_legendre(lambda x: x, 0, 1, n=4)
        assert isinstance(r, QuadratureResult)
        assert r.n_evaluations == 4


class TestGaussLaguerre:
    def test_exponential_integral(self):
        """∫ exp(-x) dx from 0 to ∞ = 1. With Laguerre weight, f(x) = 1."""
        r = gauss_laguerre(lambda x: 1.0, n=8)
        assert r.value == pytest.approx(1.0, abs=1e-10)

    def test_x_times_exp(self):
        """∫ x * exp(-x) dx from 0 to ∞ = 1. With weight, f(x) = x."""
        r = gauss_laguerre(lambda x: x, n=8)
        assert r.value == pytest.approx(1.0, abs=1e-10)

    def test_x_squared(self):
        """∫ x^2 * exp(-x) dx = 2 (= Gamma(3))."""
        r = gauss_laguerre(lambda x: x**2, n=8)
        assert r.value == pytest.approx(2.0, abs=1e-10)


class TestGaussHermite:
    def test_gaussian_integral(self):
        """∫ exp(-x^2) dx from -∞ to ∞ = sqrt(pi). With weight, f(x) = 1."""
        r = gauss_hermite(lambda x: 1.0, n=8)
        assert r.value == pytest.approx(math.sqrt(math.pi), abs=1e-10)

    def test_x_squared(self):
        """∫ x^2 * exp(-x^2) dx = sqrt(pi)/2."""
        r = gauss_hermite(lambda x: x**2, n=8)
        assert r.value == pytest.approx(math.sqrt(math.pi) / 2, abs=1e-10)

    def test_odd_function_zero(self):
        """∫ x * exp(-x^2) dx = 0 (odd integrand)."""
        r = gauss_hermite(lambda x: x, n=8)
        assert r.value == pytest.approx(0.0, abs=1e-10)


class TestAdaptiveSimpson:
    def test_sin(self):
        r = adaptive_simpson(math.sin, 0, math.pi)
        assert r.value == pytest.approx(2.0, abs=1e-10)

    def test_exp(self):
        r = adaptive_simpson(math.exp, 0, 1)
        assert r.value == pytest.approx(math.e - 1, abs=1e-10)

    def test_sqrt(self):
        """∫ 1/sqrt(x) on [0.001, 1] ≈ 2*(1 - sqrt(0.001))."""
        expected = 2.0 * (1 - math.sqrt(0.001))
        r = adaptive_simpson(lambda x: 1.0 / math.sqrt(x), 0.001, 1.0, tol=1e-8)
        assert r.value == pytest.approx(expected, abs=1e-6)

    def test_polynomial(self):
        """∫ x^4 on [0, 1] = 1/5."""
        r = adaptive_simpson(lambda x: x**4, 0, 1)
        assert r.value == pytest.approx(0.2, abs=1e-10)

    def test_returns_result(self):
        r = adaptive_simpson(lambda x: x, 0, 1)
        assert isinstance(r, QuadratureResult)
        assert r.n_evaluations > 0


class TestBlackScholesIntegral:
    """Reproduce Black-Scholes call price via risk-neutral integration."""

    def test_bs_via_quadrature(self):
        S, K, r, vol, T = 100, 100, 0.05, 0.20, 1.0
        from pricebook.equity_option import equity_option_price
        from pricebook.black76 import OptionType

        bs_price = equity_option_price(S, K, r, vol, T, OptionType.CALL)

        # C = exp(-rT) * ∫ max(S_T - K, 0) * p(S_T) dS_T
        # where p is the lognormal density
        def integrand(s_t):
            if s_t <= K:
                return 0.0
            mu = math.log(S) + (r - 0.5 * vol**2) * T
            sigma = vol * math.sqrt(T)
            log_pdf = -0.5 * ((math.log(s_t) - mu) / sigma)**2 - math.log(s_t * sigma * math.sqrt(2 * math.pi))
            return (s_t - K) * math.exp(log_pdf)

        result = adaptive_simpson(integrand, K * 0.01, K * 5, tol=1e-6)
        quad_price = math.exp(-r * T) * result.value

        assert quad_price == pytest.approx(bs_price, rel=0.001)
