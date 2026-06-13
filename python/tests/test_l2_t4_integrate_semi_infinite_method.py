"""Regression for L2 Wave-2 audit — `integrate_semi_infinite` silently
downgraded any non-Laguerre method to ADAPTIVE.

Pre-fix:

    if method == IntegrationMethod.GAUSS_LAGUERRE:
        ... Gauss-Laguerre code ...
    else:
        return _adaptive(f, a, np.inf)      # discards user's `method`

A user passing ``method=IntegrationMethod.GAUSS_HERMITE`` or
``method=IntegrationMethod.SIMPSON`` got the SciPy-adaptive result with
no warning that their METHOD argument was discarded.  A method-comparison
study would see identical numbers for every "method" choice, masking the
fact that only one method was actually running.

Post-fix only the two methods that have defined semi-infinite behaviour
are accepted; everything else raises ``ValueError`` with a pointer to
the documented choices.
"""

from __future__ import annotations

import math

import pytest

from pricebook.numerical._integrate import (
    IntegrationMethod,
    integrate_semi_infinite,
)


class TestSupportedMethodsWork:
    def test_gauss_laguerre_works(self):
        # ∫_0^∞ e^{-x} dx = 1.
        r = integrate_semi_infinite(
            lambda x: math.exp(-x), a=0.0,
            method=IntegrationMethod.GAUSS_LAGUERRE, n=30,
        )
        assert r.value == pytest.approx(1.0, abs=1e-9)

    def test_adaptive_works(self):
        r = integrate_semi_infinite(
            lambda x: math.exp(-x), a=0.0,
            method=IntegrationMethod.ADAPTIVE,
        )
        assert r.value == pytest.approx(1.0, abs=1e-6)


class TestUnsupportedMethodsRaise:
    @pytest.mark.parametrize("method", [
        IntegrationMethod.GAUSS_LEGENDRE,
        IntegrationMethod.GAUSS_HERMITE,
        IntegrationMethod.SIMPSON,
        IntegrationMethod.TRAPEZOID,
        IntegrationMethod.TANH_SINH,
        IntegrationMethod.CLENSHAW_CURTIS,
        IntegrationMethod.ROMBERG,
    ])
    def test_other_methods_raise_value_error(self, method):
        with pytest.raises(ValueError, match="is not supported"):
            integrate_semi_infinite(
                lambda x: math.exp(-x), a=0.0, method=method,
            )
