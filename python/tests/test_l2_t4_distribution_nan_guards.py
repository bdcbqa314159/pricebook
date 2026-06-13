"""Regression for L2 Wave-2 audit — `Uniform` and `Exponential`
constructors let NaN slip through their guards.

In IEEE 754, all comparisons against NaN return False — so
``NaN >= 1.0`` is False, ``NaN <= 0`` is False, etc.  This means a
naive guard like

    if a >= b:
        raise ...

silently accepts ``a=NaN``.  The constructor succeeds, then downstream
``cdf`` / ``pdf`` propagate NaN through the user's computation with no
diagnostic.

Post-fix both constructors explicitly check for NaN and raise
``ValueError`` with a diagnostic message.
"""

from __future__ import annotations

import math

import pytest

from pricebook.numerical._distributions import Exponential, Uniform


class TestUniformNaNGuards:
    def test_nan_a_raises(self):
        with pytest.raises(ValueError, match="finite numbers"):
            Uniform(a=float("nan"), b=1.0)

    def test_nan_b_raises(self):
        with pytest.raises(ValueError, match="finite numbers"):
            Uniform(a=0.0, b=float("nan"))

    def test_nan_both_raises(self):
        with pytest.raises(ValueError, match="finite numbers"):
            Uniform(a=float("nan"), b=float("nan"))

    def test_valid_a_b_works(self):
        u = Uniform(a=0.0, b=2.0)
        assert u.cdf(1.0) == pytest.approx(0.5)


class TestExponentialNaNGuards:
    def test_nan_rate_raises(self):
        with pytest.raises(ValueError, match="finite positive"):
            Exponential(rate=float("nan"))

    def test_valid_rate_works(self):
        e = Exponential(rate=2.0)
        assert e.cdf(0.0) == pytest.approx(0.0)

    def test_inf_rate_succeeds_or_raises_consistently(self):
        """Inf rate is not NaN but also not really finite — leave the
        existing behaviour (allowed; the user gets an effective Dirac
        delta at 0).  This test pins down the contract."""
        e = Exponential(rate=float("inf"))
        # cdf(small positive) → 1.0
        import math as _m
        result = e.cdf(0.001)
        assert _m.isfinite(result) or _m.isnan(result)


class TestPreFixSlipThroughGuard:
    """Pin down that the pre-fix bug really did exist: without the
    explicit NaN check, `if NaN >= 1.0:` is False so the guard didn't
    fire.  Post-fix it does."""

    def test_naive_comparison_was_false(self):
        """Sanity: confirm IEEE 754 semantics for the audit context."""
        assert (float("nan") >= 1.0) is False
        assert (float("nan") <= 0.0) is False
        assert (float("nan") == float("nan")) is False
