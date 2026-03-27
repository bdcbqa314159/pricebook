"""Tests for interpolation methods."""

import math
import pytest
import numpy as np

from pricebook.interpolation import (
    InterpolationMethod,
    LinearInterpolator,
    LogLinearInterpolator,
    CubicSplineInterpolator,
    MonotoneCubicInterpolator,
    create_interpolator,
)


@pytest.fixture
def knots():
    """Simple test data: x = [0, 1, 2, 3], y = [1.0, 0.98, 0.95, 0.90]."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 0.98, 0.95, 0.90])
    return x, y


class TestLinear:
    """Piecewise linear interpolation."""

    def test_at_knots(self, knots):
        interp = LinearInterpolator(*knots)
        for x, y in zip(*knots):
            assert interp(x) == pytest.approx(y)

    def test_midpoint(self, knots):
        interp = LinearInterpolator(*knots)
        # Midpoint of [0, 1]: (1.0 + 0.98) / 2 = 0.99
        assert interp(0.5) == pytest.approx(0.99)

    def test_flat_extrapolation_left(self, knots):
        interp = LinearInterpolator(*knots)
        assert interp(-1.0) == pytest.approx(1.0)

    def test_flat_extrapolation_right(self, knots):
        interp = LinearInterpolator(*knots)
        assert interp(5.0) == pytest.approx(0.90)


class TestLogLinear:
    """Log-linear interpolation (linear in log space)."""

    def test_at_knots(self, knots):
        interp = LogLinearInterpolator(*knots)
        for x, y in zip(*knots):
            assert interp(x) == pytest.approx(y)

    def test_midpoint_differs_from_linear(self, knots):
        lin = LinearInterpolator(*knots)
        loglin = LogLinearInterpolator(*knots)
        # Log-linear midpoint != linear midpoint
        assert loglin(0.5) != pytest.approx(lin(0.5), abs=1e-6)

    def test_midpoint_value(self, knots):
        interp = LogLinearInterpolator(*knots)
        # At x=0.5: exp(0.5 * log(1.0) + 0.5 * log(0.98))
        expected = math.exp(0.5 * math.log(1.0) + 0.5 * math.log(0.98))
        assert interp(0.5) == pytest.approx(expected)

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            LogLinearInterpolator(np.array([0.0, 1.0]), np.array([1.0, -0.5]))


class TestCubicSpline:
    """Natural cubic spline interpolation."""

    def test_at_knots(self, knots):
        interp = CubicSplineInterpolator(*knots)
        for x, y in zip(*knots):
            assert interp(x) == pytest.approx(y)

    def test_smooth_between_knots(self, knots):
        interp = CubicSplineInterpolator(*knots)
        # Should produce a smooth curve — just check it's in range
        val = interp(1.5)
        assert 0.90 <= val <= 1.0


class TestMonotoneCubic:
    """Monotone cubic Hermite (Hyman filter)."""

    def test_at_knots(self, knots):
        interp = MonotoneCubicInterpolator(*knots)
        for x, y in zip(*knots):
            assert interp(x) == pytest.approx(y)

    def test_preserves_monotonicity(self, knots):
        interp = MonotoneCubicInterpolator(*knots)
        # Sample many points and check monotone decreasing
        xs = np.linspace(0.0, 3.0, 100)
        ys = [interp(x) for x in xs]
        for i in range(1, len(ys)):
            assert ys[i] <= ys[i - 1] + 1e-12

    def test_no_overshoot(self):
        """Monotone cubic should not overshoot beyond data range."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 1.0, 0.5, 0.5, 0.5])
        interp = MonotoneCubicInterpolator(x, y)
        xs = np.linspace(0.0, 4.0, 200)
        ys = [interp(xi) for xi in xs]
        assert min(ys) >= 0.5 - 1e-12
        assert max(ys) <= 1.0 + 1e-12


class TestFactory:
    """create_interpolator factory."""

    def test_all_methods(self, knots):
        for method in InterpolationMethod:
            interp = create_interpolator(method, *knots)
            # Should recover knot values
            for x, y in zip(*knots):
                assert interp(x) == pytest.approx(y, abs=1e-10)


class TestValidation:
    """Input validation."""

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            LinearInterpolator(np.array([1.0, 2.0]), np.array([1.0]))

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            LinearInterpolator(np.array([1.0]), np.array([1.0]))

    def test_non_increasing_x(self):
        with pytest.raises(ValueError):
            LinearInterpolator(np.array([1.0, 0.5, 2.0]), np.array([1.0, 2.0, 3.0]))
