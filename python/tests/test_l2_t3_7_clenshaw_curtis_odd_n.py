"""Regression for L2 Tier-3 T3.7 — Clenshaw-Curtis weights for odd n.

Pre-fix `_clenshaw_curtis` used the n-EVEN weight formula for all n:
* endpoint weight = 1/(n²−1), but for n odd it should be 1/n²;
* the boundary term `b_k = 1 at k = n/2` is only correct for n even (n
  odd has no k = n/2 term).

Pre-fix ∫₀¹ 1 dx with n=3 returned ≈ 0.9 instead of 1.  Polynomial
integration was correspondingly biased.
"""

from __future__ import annotations

import math

import pytest

from pricebook.numerical._integrate import (
    IntegrationMethod,
    integrate,
)


class TestClenshawCurtisOddN:
    @pytest.mark.parametrize("n", [3, 5, 7, 9, 15])
    def test_constant_integrates_exactly(self, n):
        """∫₀¹ 1 dx = 1 — must be exact for any n."""
        result = integrate(lambda x: 1.0, 0.0, 1.0,
                           method=IntegrationMethod.CLENSHAW_CURTIS, n=n)
        assert math.isclose(result.value, 1.0, abs_tol=1e-12)

    @pytest.mark.parametrize("n", [3, 5, 7, 9, 15])
    def test_polynomial_x_squared(self, n):
        """∫₀¹ x² dx = 1/3 — exact for n ≥ 2."""
        result = integrate(lambda x: x * x, 0.0, 1.0,
                           method=IntegrationMethod.CLENSHAW_CURTIS, n=n)
        assert math.isclose(result.value, 1/3, abs_tol=1e-10)

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_polynomial_cube(self, n):
        """∫₀¹ x³ dx = 1/4 — exact for n ≥ 3."""
        result = integrate(lambda x: x ** 3, 0.0, 1.0,
                           method=IntegrationMethod.CLENSHAW_CURTIS, n=n)
        assert math.isclose(result.value, 0.25, abs_tol=1e-10)

    def test_sin_converges_with_n(self):
        """∫₀^π sin(x) dx = 2.  Error should DECREASE monotonically with n
        for both odd and even sequences."""
        errors = []
        for n in [3, 5, 9, 17, 33]:
            result = integrate(math.sin, 0.0, math.pi,
                               method=IntegrationMethod.CLENSHAW_CURTIS, n=n)
            errors.append(abs(result.value - 2.0))
        # Strictly decreasing UNTIL we hit machine precision.
        for i in range(1, len(errors)):
            if errors[i - 1] < 1e-13:
                break  # both at machine precision; floor differences are noise
            assert errors[i] < errors[i - 1], (
                f"CC error not decreasing: {errors}"
            )
        # By n=33, must be at or near machine precision.
        assert errors[-1] < 1e-10

    def test_weights_sum_to_interval_length(self):
        """Weights × half should sum to (b − a) for any n.  Smoke-test
        odd n directly."""
        from pricebook.numerical._integrate import _clenshaw_curtis
        for n in [3, 5, 7, 9, 11, 13]:
            result = _clenshaw_curtis(lambda x: 1.0, 0.0, 1.0, n)
            assert math.isclose(result.value, 1.0, abs_tol=1e-12), (
                f"n={n}: weights don't integrate 1 correctly (= {result.value})"
            )
