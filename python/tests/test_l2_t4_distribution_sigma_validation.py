"""Regression for L2 Wave-2 audit — `Normal` and `LogNormal` constructors
accepted any ``sigma``, including 0 and negative.

Pre-fix:
- ``Normal(sigma=0)``: ``cdf/pdf`` divided by zero, emitting
  ``RuntimeWarning`` and returning 1.0 (or NaN) silently.
- ``Normal(sigma=-1)``: produced a "flipped" distribution (the math
  works, but σ < 0 is mathematically meaningless — only σ ≥ 0
  parameterises a normal).
- ``LogNormal(sigma=0)``: same divide-by-zero pattern.

Post-fix both classes raise ``ValueError`` at construction with a
diagnostic message.
"""

from __future__ import annotations

import pytest

from pricebook.numerical._distributions import LogNormal, Normal


class TestNormalSigmaValidation:
    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            Normal(mu=0.0, sigma=0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            Normal(mu=0.0, sigma=-1.0)

    def test_positive_sigma_works(self):
        n = Normal(mu=0.0, sigma=1.0)
        assert n.cdf(0.0) == pytest.approx(0.5)


class TestLogNormalSigmaValidation:
    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            LogNormal(mu=0.0, sigma=0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            LogNormal(mu=0.0, sigma=-0.5)

    def test_positive_sigma_works(self):
        ln = LogNormal(mu=0.0, sigma=1.0)
        # E[LogNormal(0, 1)] = exp(0.5) ≈ 1.6487
        import math as _m
        assert ln.cdf(_m.exp(0.5)) > 0
