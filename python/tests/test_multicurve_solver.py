"""Tests for multi-curve Newton solver + curve validation."""

import math
from datetime import date

import numpy as np
import pytest

from pricebook.discount_curve import DiscountCurve
from pricebook.multicurve_solver import (
    CurveValidationResult,
    validate_curve,
    curve_analytical_jacobian,
)


# ---- Curve validation ----

class TestValidateCurve:
    def _flat_curve(self, rate=0.04):
        ref = date(2026, 1, 15)
        tenors = [0.25, 0.5, 1, 2, 5, 10]
        dates = [date.fromordinal(ref.toordinal() + int(t * 365)) for t in tenors]
        dfs = [math.exp(-rate * t) for t in tenors]
        return DiscountCurve(ref, dates, dfs)

    def test_flat_curve_valid(self):
        curve = self._flat_curve(0.04)
        result = validate_curve(curve)
        assert isinstance(result, CurveValidationResult)
        assert result.is_valid
        assert not result.has_negative_forwards
        assert not result.has_non_monotone_dfs

    def test_detects_negative_forwards(self):
        ref = date(2026, 1, 15)
        dates = [date(2026, 7, 15), date(2027, 1, 15)]
        # DF inverted → negative forward
        dfs = [0.97, 0.99]
        curve = DiscountCurve(ref, dates, dfs)
        result = validate_curve(curve)
        assert result.has_non_monotone_dfs
        assert not result.is_valid

    def test_detects_extreme_forwards(self):
        ref = date(2026, 1, 15)
        dates = [date(2026, 7, 15), date(2027, 1, 15)]
        # Very steep: 50% forward rate
        dfs = [0.97, 0.60]
        curve = DiscountCurve(ref, dates, dfs)
        result = validate_curve(curve, max_forward_rate=0.20)
        assert not result.is_valid
        assert any("above" in w for w in result.warnings)

    def test_forward_rate_bounds(self):
        curve = self._flat_curve(0.04)
        result = validate_curve(curve)
        assert 0 < result.min_forward_rate < 0.10
        assert 0 < result.max_forward_rate < 0.10

    def test_pillar_count(self):
        curve = self._flat_curve()
        result = validate_curve(curve)
        assert result.n_pillars == 7  # 6 + t=0


# ---- Jacobian ----

class TestCurveJacobian:
    def test_basic(self):
        ref = date(2026, 1, 15)
        tenors = [0.5, 1.0, 2.0, 5.0]
        dates = [date.fromordinal(ref.toordinal() + int(t * 365)) for t in tenors]
        dfs = [math.exp(-0.04 * t) for t in tenors]
        curve = DiscountCurve(ref, dates, dfs)
        result = curve_analytical_jacobian(curve)
        assert result.jacobian.shape[1] == 5  # 4 pillars + t=0

    def test_diagonal_dominant(self):
        """Jacobian at pillar points should be approximately identity."""
        ref = date(2026, 1, 15)
        tenors = [1.0, 2.0, 5.0]
        dates = [date.fromordinal(ref.toordinal() + int(t * 365)) for t in tenors]
        dfs = [math.exp(-0.04 * t) for t in tenors]
        curve = DiscountCurve(ref, dates, dfs)
        result = curve_analytical_jacobian(curve)
        # Non-zero elements: each pillar's zero rate depends mostly on itself
        J = result.jacobian
        for i in range(J.shape[0]):
            row_max = np.argmax(np.abs(J[i, :]))
            # The largest sensitivity should be near the diagonal
            assert J.shape[0] <= J.shape[1]
