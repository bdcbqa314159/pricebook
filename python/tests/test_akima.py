"""Tests for Akima spline interpolation."""

import math
import pytest
import numpy as np

from pricebook.interpolation import (
    AkimaInterpolator,
    InterpolationMethod,
    LinearInterpolator,
    LogLinearInterpolator,
    CubicSplineInterpolator,
    MonotoneCubicInterpolator,
    create_interpolator,
)
from pricebook.discount_curve import DiscountCurve
from datetime import date


XS = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
YS = np.array([1.0, 0.98, 0.95, 0.90, 0.84, 0.77])


class TestAkimaSlopes:
    def test_slopes_computed(self):
        interp = AkimaInterpolator(XS, YS)
        assert len(interp._slopes) == len(XS)

    def test_slopes_finite(self):
        interp = AkimaInterpolator(XS, YS)
        assert np.all(np.isfinite(interp._slopes))

    def test_two_points(self):
        """Minimum: 2 points should work."""
        interp = AkimaInterpolator(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
        assert interp(0.5) == pytest.approx(1.5, rel=0.01)


class TestAkimaValues:
    def test_exact_at_knots(self):
        interp = AkimaInterpolator(XS, YS)
        for x, y in zip(XS, YS):
            assert interp(float(x)) == pytest.approx(float(y))

    def test_between_knots(self):
        interp = AkimaInterpolator(XS, YS)
        for x in [0.5, 1.5, 2.5, 3.5, 4.5]:
            v = interp(x)
            # Should be between adjacent knot values (roughly)
            assert 0.5 < v < 1.1

    def test_flat_extrapolation_left(self):
        interp = AkimaInterpolator(XS, YS)
        assert interp(-1.0) == pytest.approx(1.0)

    def test_flat_extrapolation_right(self):
        interp = AkimaInterpolator(XS, YS)
        assert interp(10.0) == pytest.approx(0.77)

    def test_monotone_data_stays_in_range(self):
        """Akima should not overshoot for monotone data."""
        interp = AkimaInterpolator(XS, YS)
        for x in np.linspace(0.0, 5.0, 100):
            v = interp(float(x))
            assert 0.7 <= v <= 1.05


class TestAkimaComparison:
    def test_continuous(self):
        """Akima should produce continuous values (no jumps at knots)."""
        aki = AkimaInterpolator(XS, YS)
        for x0 in [1.0, 2.0, 3.0, 4.0]:
            left = aki(x0 - 1e-8)
            right = aki(x0 + 1e-8)
            assert left == pytest.approx(right, abs=1e-6)

    def test_no_wild_oscillation(self):
        """Akima should avoid the overshoots that cubic splines can produce."""
        # Data with a sharp jump
        xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        ys = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])

        aki = AkimaInterpolator(xs, ys)
        cubic = CubicSplineInterpolator(xs, ys)

        # Check range between knots
        aki_range = max(aki(float(x)) for x in np.linspace(0.0, 5.0, 200))
        cubic_range = max(cubic(float(x)) for x in np.linspace(0.0, 5.0, 200))

        # Akima should stay closer to data range
        assert aki_range <= cubic_range + 0.1

    def test_matches_linear_for_linear_data(self):
        """On perfectly linear data, Akima should give linear results."""
        xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        aki = AkimaInterpolator(xs, ys)
        for x in [0.5, 1.5, 2.5, 3.5]:
            assert aki(x) == pytest.approx(x, abs=1e-10)


class TestAkimaIntegration:
    def test_factory(self):
        interp = create_interpolator(InterpolationMethod.AKIMA, XS, YS)
        assert isinstance(interp, AkimaInterpolator)
        assert interp(2.5) == pytest.approx(AkimaInterpolator(XS, YS)(2.5))

    def test_discount_curve(self):
        """DiscountCurve with Akima interpolation gives valid DFs."""
        ref = date(2024, 1, 15)
        tenors = [1, 2, 3, 5, 7, 10]
        dates = [date.fromordinal(ref.toordinal() + int(t * 365)) for t in tenors]
        dfs = [math.exp(-0.05 * t) for t in tenors]

        curve = DiscountCurve(ref, dates, dfs, interpolation=InterpolationMethod.AKIMA)

        for t in [0.5, 1.5, 4.0, 8.0]:
            d = date.fromordinal(ref.toordinal() + int(t * 365))
            df = curve.df(d)
            assert 0.0 < df <= 1.0

    def test_enum_value(self):
        assert InterpolationMethod.AKIMA.value == "akima"
