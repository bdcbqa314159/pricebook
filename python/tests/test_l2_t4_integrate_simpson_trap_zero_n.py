"""Regression for L2 Wave-2 audit — Simpson and trapezoid quadrature
crashed with bare ``ZeroDivisionError`` when ``n=0``.

Pre-fix both ``_simpson`` and ``_trapezoid`` computed ``h = (b - a) / n``
with no validation, so a caller routing through the public
``integrate(method=SIMPSON, n=0)`` API got an opaque
``ZeroDivisionError`` deep in the implementation.

Post-fix both raise ``ValueError`` upfront with a clear message.
"""

from __future__ import annotations

import pytest

from pricebook.numerical._integrate import _simpson, _trapezoid


class TestSimpsonZeroN:
    def test_zero_n_raises_value_error(self):
        with pytest.raises(ValueError, match="_simpson: n must be >= 1"):
            _simpson(lambda x: x, 0.0, 1.0, 0)

    def test_negative_n_raises_value_error(self):
        with pytest.raises(ValueError, match="_simpson: n must be >= 1"):
            _simpson(lambda x: x, 0.0, 1.0, -3)

    def test_positive_n_still_works(self):
        r = _simpson(lambda x: x * x, 0.0, 1.0, 100)
        assert r.value == pytest.approx(1.0 / 3.0, abs=1e-8)


class TestTrapezoidZeroN:
    def test_zero_n_raises_value_error(self):
        with pytest.raises(ValueError, match="_trapezoid: n must be >= 1"):
            _trapezoid(lambda x: x, 0.0, 1.0, 0)

    def test_negative_n_raises_value_error(self):
        with pytest.raises(ValueError, match="_trapezoid: n must be >= 1"):
            _trapezoid(lambda x: x, 0.0, 1.0, -1)

    def test_positive_n_still_works(self):
        r = _trapezoid(lambda x: x, 0.0, 1.0, 100)
        assert r.value == pytest.approx(0.5, abs=1e-8)
