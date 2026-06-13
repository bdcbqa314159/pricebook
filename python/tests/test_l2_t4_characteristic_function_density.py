"""Regression for L2 Wave-2 audit — `CharacteristicFunction.density`
had two robustness gaps.

Pre-fix:
- ``density(x_grid, n_quad=1)`` raised ``IndexError`` at ``du = u[1] - u[0]``
  because the linspace had only one point.
- ``density(scalar)`` (a single x point as a scalar) raised ``TypeError``
  at ``len(x)`` because ``np.asarray(scalar)`` produces a 0-d array
  which has no ``len()``.

Post-fix:
- ``n_quad < 2`` raises ``ValueError`` upfront with a clear message.
- Scalar inputs are promoted to a 1-d array via ``np.atleast_1d``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.numerical._fourier import CharacteristicFunction


def _gaussian_cf(u: complex) -> complex:
    """CF of N(0, 1): φ(u) = exp(-u²/2)."""
    return np.exp(-0.5 * u ** 2)


class TestDensityValidation:
    def test_n_quad_one_raises(self):
        cf = CharacteristicFunction(_gaussian_cf, T=1.0)
        with pytest.raises(ValueError, match="n_quad must be >= 2"):
            cf.density(np.array([0.0, 1.0]), n_quad=1)

    def test_n_quad_zero_raises(self):
        cf = CharacteristicFunction(_gaussian_cf, T=1.0)
        with pytest.raises(ValueError, match="n_quad must be >= 2"):
            cf.density(np.array([0.0, 1.0]), n_quad=0)


class TestDensityScalarInput:
    def test_scalar_x_accepted(self):
        """Pre-fix: TypeError on len(0-d array).  Post-fix: scalar → 1-d."""
        cf = CharacteristicFunction(_gaussian_cf, T=1.0)
        result = cf.density(0.0, n_quad=200)
        # Result is a 1-element array carrying the density at x=0.
        assert result.shape == (1,)
        # N(0,1) density at 0 is 1/√(2π) ≈ 0.3989.
        assert result[0] == pytest.approx(1.0 / math.sqrt(2 * math.pi), abs=0.01)

    def test_python_float_accepted(self):
        cf = CharacteristicFunction(_gaussian_cf, T=1.0)
        result = cf.density(1.5, n_quad=200)
        assert result.shape == (1,)


class TestDensityHealthyPath:
    def test_gaussian_density_matches_closed_form(self):
        """Sanity: density of N(0,1) recovered via Fourier inversion."""
        cf = CharacteristicFunction(_gaussian_cf, T=1.0)
        x = np.array([-1.0, 0.0, 1.0])
        d = cf.density(x, n_quad=200)
        # N(0,1) values
        expected = np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
        np.testing.assert_allclose(d, expected, atol=1e-3)
