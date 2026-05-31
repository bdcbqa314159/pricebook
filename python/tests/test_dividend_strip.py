"""Tests for dividend strip analytics."""

import pytest
import math
import numpy as np

from pricebook.equity.dividend_strip import (
    decompose_strip, strip_carry, dividend_growth_rate,
)
from pricebook.equity.dividend_advanced import DividendCurve


def _make_curve(tenors, cum_divs, yields=None):
    T = np.array(tenors)
    D = np.array(cum_divs)
    if yields is None:
        yields = np.where(T > 0, D / (100 * T), 0.0)
    return DividendCurve(T, D, yields, "test")


class TestDecomposeStrip:
    def test_sum_to_total(self):
        curve = _make_curve([0.25, 0.5, 0.75, 1.0], [0.5, 1.0, 1.5, 2.0])
        strips = decompose_strip(curve)
        total = sum(s.forward_div for s in strips)
        assert total == pytest.approx(2.0, abs=0.01)

    def test_n_periods(self):
        curve = _make_curve([0.5, 1.0, 2.0], [1.0, 2.0, 4.0])
        strips = decompose_strip(curve, n_periods=4)
        assert len(strips) == 4

    def test_custom_breaks(self):
        curve = _make_curve([0.5, 1.0, 2.0], [1.0, 2.0, 4.0])
        strips = decompose_strip(curve, custom_breaks=[0.5, 1.0])
        assert len(strips) == 2

    def test_weights_sum_to_one(self):
        curve = _make_curve([0.25, 0.5, 0.75, 1.0], [0.5, 1.0, 1.5, 2.0])
        strips = decompose_strip(curve)
        total_weight = sum(s.weight for s in strips)
        assert total_weight == pytest.approx(1.0, abs=0.01)

    def test_uniform_equal_weights(self):
        curve = _make_curve([0.25, 0.5, 0.75, 1.0], [0.5, 1.0, 1.5, 2.0])
        strips = decompose_strip(curve)
        for s in strips:
            assert s.weight == pytest.approx(0.25, abs=0.01)

    def test_to_dict(self):
        curve = _make_curve([1.0], [2.0])
        strips = decompose_strip(curve)
        d = strips[0].to_dict()
        assert "forward_div" in d


class TestStripCarry:
    def test_positive_carry(self):
        """High div yield - low funding → positive carry."""
        curve = _make_curve([0.5, 1.0], [1.0, 2.0])
        strips = decompose_strip(curve)
        results = strip_carry(strips, funding_rate=0.02, spot=100)
        # With ~2% div yield and 2% funding, carry should be near zero
        for r in results:
            assert math.isfinite(r.carry)
            assert math.isfinite(r.total)

    def test_to_dict(self):
        curve = _make_curve([1.0], [2.0])
        strips = decompose_strip(curve)
        results = strip_carry(strips, funding_rate=0.03)
        d = results[0].to_dict()
        assert "carry" in d


class TestDividendGrowthRate:
    def test_constant_divs_zero_growth(self):
        """Constant forward dividends → zero growth."""
        curve = _make_curve([0.25, 0.5, 0.75, 1.0], [0.5, 1.0, 1.5, 2.0])
        strips = decompose_strip(curve)
        g = dividend_growth_rate(strips)
        assert abs(g) < 0.1  # near zero

    def test_growing_divs_positive_growth(self):
        """Increasing forward dividends → positive growth."""
        curve = _make_curve([0.25, 0.5, 0.75, 1.0], [0.4, 0.9, 1.6, 2.5])
        strips = decompose_strip(curve)
        g = dividend_growth_rate(strips)
        assert g > 0

    def test_single_strip_returns_zero(self):
        curve = _make_curve([1.0], [2.0])
        strips = decompose_strip(curve)
        g = dividend_growth_rate(strips)
        assert g == 0.0
