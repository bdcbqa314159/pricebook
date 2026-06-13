"""Regression for L2 Wave-2 audit — `Dual` math operations silently
returned 0 derivative at known singularities instead of raising.

Pre-fix the following degenerate inputs produced silent wrong-results:

- ``sqrt(Dual(0, x))`` returned ``Dual(0, 0)``.  Mathematically the
  derivative ``1/(2·√x) → +∞`` at x = 0.  Silent 0 derivative hides
  failures in MC paths that touch zero (e.g. Heston QE-discretisation
  variance hitting the zero-boundary).
- ``Dual(0, x) ** Dual(y, z)`` returned ``Dual(0, 0)`` regardless of
  the exponent.  Derivative is singular when base is 0 and exponent
  involves a log term.
- ``Dual(negative, x) ** Dual(y, z)`` used ``log(abs(self.val))`` —
  log of a negative number is not real.  The pre-fix formula ran but
  the derivative had no meaningful real-valued interpretation.
- ``Dual(0, x) ** n`` with ``n < 1`` was a singularity disguised as
  ``n · 0^(n-1) · der`` (Python evaluating ``0^(n-1)`` for n < 1 gives
  ``0^negative`` → ``ZeroDivisionError``, or for fractional n gives
  ``0^fraction = 0`` — neither correctly captures the singularity).
- ``b ** Dual(x, y)`` with ``b <= 0`` returned ``der = 0`` silently.

Post-fix all five paths raise ``ValueError`` with diagnostic context.
The valid-input paths are unchanged.
"""

from __future__ import annotations

import math

import pytest

from pricebook.numerical.auto_diff import Dual, sqrt


class TestSqrtAtZeroRaises:
    def test_sqrt_dual_zero_raises(self):
        with pytest.raises(ValueError, match="singular"):
            sqrt(Dual(0.0, 1.0))

    def test_sqrt_dual_positive_works(self):
        r = sqrt(Dual(4.0, 1.0))
        assert r.val == pytest.approx(2.0)
        # d/dx sqrt(x) at x=4 is 1/(2·2) = 0.25
        assert r.der == pytest.approx(0.25)

    def test_sqrt_plain_float_works(self):
        # Plain floats unaffected — math.sqrt is called directly.
        assert sqrt(4.0) == pytest.approx(2.0)


class TestPowAtZeroBaseRaises:
    def test_dual_zero_to_dual_exponent_raises(self):
        with pytest.raises(ValueError, match="Dual base 0 with Dual exponent"):
            Dual(0.0, 1.0) ** Dual(2.0, 1.0)

    def test_dual_zero_to_fractional_exponent_raises(self):
        with pytest.raises(ValueError, match="Dual base 0 with exponent"):
            Dual(0.0, 1.0) ** 0.5

    def test_dual_zero_to_negative_exponent_raises(self):
        with pytest.raises(ValueError, match="Dual base 0 with exponent"):
            Dual(0.0, 1.0) ** -1.0

    def test_dual_zero_to_integer_exponent_n_geq_1_works(self):
        """For n >= 1 integer, the derivative at x=0 is well-defined
        (0 for n > 1, 1 for n = 1)."""
        # 0^1 → val=0, der=1·0^0·1 = 1
        r = Dual(0.0, 1.0) ** 1.0
        assert r.val == 0.0
        assert r.der == pytest.approx(1.0)
        # 0^2 → val=0, der=2·0·1 = 0
        r = Dual(0.0, 1.0) ** 2.0
        assert r.val == 0.0
        assert r.der == pytest.approx(0.0)


class TestPowAtNegativeBaseRaises:
    def test_dual_negative_to_dual_exponent_raises(self):
        with pytest.raises(ValueError, match="negative"):
            Dual(-2.0, 1.0) ** Dual(2.0, 1.0)

    def test_dual_negative_to_plain_integer_works(self):
        """Integer exponent against negative base is fine: derivative is
        real."""
        r = Dual(-2.0, 1.0) ** 3.0
        # (-2)^3 = -8; derivative = 3·(-2)^2·1 = 12
        assert r.val == pytest.approx(-8.0)
        assert r.der == pytest.approx(12.0)


class TestRpowAtNonPositiveBaseRaises:
    def test_zero_base_dual_exponent_raises(self):
        with pytest.raises(ValueError, match="Non-positive base"):
            0.0 ** Dual(2.0, 1.0)

    def test_negative_base_dual_exponent_raises(self):
        with pytest.raises(ValueError, match="Non-positive base"):
            (-1.0) ** Dual(2.0, 1.0)

    def test_positive_base_works(self):
        # e^x at x=1 with derivative 1 — but here use base=2.
        r = 2.0 ** Dual(3.0, 1.0)
        assert r.val == pytest.approx(8.0)
        # d/dx 2^x = 2^x · ln(2) → 8 · ln(2)
        assert r.der == pytest.approx(8.0 * math.log(2.0))


class TestHealthyPathsUnchanged:
    def test_basic_power_dual_base_positive(self):
        r = Dual(2.0, 1.0) ** 3.0
        assert r.val == pytest.approx(8.0)
        assert r.der == pytest.approx(12.0)  # 3·2²·1

    def test_basic_sqrt_dual_positive(self):
        r = sqrt(Dual(9.0, 1.0))
        assert r.val == pytest.approx(3.0)
        assert r.der == pytest.approx(1.0 / 6.0)
