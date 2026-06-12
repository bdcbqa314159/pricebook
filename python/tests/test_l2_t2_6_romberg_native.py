"""Regression for L2 Tier-2 T2.6 — `_romberg` reimplemented natively.

Pre-fix `numerical/_integrate.py::_romberg` wrapped
`scipy.integrate.romberg`, which was REMOVED in SciPy 1.15.  Every call to
`integrate(..., method=IntegrationMethod.ROMBERG)` raised ImportError.

Post-fix uses a native Romberg-on-trapezoid implementation (Richardson
extrapolation of repeated trapezoidal estimates).
"""

from __future__ import annotations

import math

import pytest

from pricebook.numerical._integrate import (
    IntegrationMethod,
    integrate,
)


class TestRombergNative:
    def test_constant_function(self):
        """∫_0^1 5 dx = 5."""
        result = integrate(lambda x: 5.0, 0.0, 1.0, method=IntegrationMethod.ROMBERG)
        assert math.isclose(result.value, 5.0, abs_tol=1e-12)

    def test_polynomial(self):
        """∫_0^1 x³ dx = 1/4."""
        result = integrate(lambda x: x ** 3, 0.0, 1.0, method=IntegrationMethod.ROMBERG)
        assert math.isclose(result.value, 0.25, abs_tol=1e-10)

    def test_sin(self):
        """∫_0^π sin(x) dx = 2."""
        result = integrate(math.sin, 0.0, math.pi, method=IntegrationMethod.ROMBERG)
        assert math.isclose(result.value, 2.0, abs_tol=1e-10)

    def test_gaussian(self):
        """∫_{-5}^{5} exp(-x²/2)/sqrt(2π) dx ≈ 1 (5σ truncation of standard normal)."""
        result = integrate(
            lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi),
            -5.0, 5.0, method=IntegrationMethod.ROMBERG,
        )
        # 5σ truncation captures > 1 − 5.7e-7 of the mass.
        assert math.isclose(result.value, 1.0, abs_tol=1e-6)

    def test_no_import_error(self):
        """Smoke: pre-fix this raised ImportError on SciPy 1.15+."""
        # If we get this far without ImportError, the native implementation
        # didn't hit the dead scipy.integrate.romberg path.
        result = integrate(lambda x: 1.0, 0.0, 1.0, method=IntegrationMethod.ROMBERG)
        assert math.isclose(result.value, 1.0, abs_tol=1e-12)
        assert result.method == "romberg"
        assert result.converged

    def test_oscillatory_converges(self):
        """∫_0^{2π} cos(x) dx = 0."""
        result = integrate(math.cos, 0.0, 2 * math.pi, method=IntegrationMethod.ROMBERG)
        assert abs(result.value) < 1e-8
