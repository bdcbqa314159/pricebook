"""Tests for forward rate interpolation."""

import math
import pytest
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.forward_interpolation import (
    ForwardInterpolationMethod, build_forward_curve,
    monotone_convex_forwards, extract_forwards,
)


REF = date(2024, 1, 15)


@pytest.fixture
def pillar_dates():
    return [date(2024 + y, 1, 15) for y in [1, 2, 3, 5, 7, 10]]


@pytest.fixture
def pillar_dfs():
    """DFs consistent with ~4% flat rate."""
    return [math.exp(-0.04 * y) for y in [1, 2, 3, 5, 7, 10]]


@pytest.fixture
def upward_dfs():
    """DFs consistent with upward-sloping forwards (2% → 6%)."""
    rates = [0.02, 0.025, 0.03, 0.04, 0.05, 0.06]
    years = [1, 2, 3, 5, 7, 10]
    return [math.exp(-r * t) for r, t in zip(rates, years)]


class TestBuildForwardCurve:
    @pytest.mark.parametrize("method", list(ForwardInterpolationMethod))
    def test_all_methods_produce_curve(self, pillar_dates, pillar_dfs, method):
        curve = build_forward_curve(REF, pillar_dates, pillar_dfs, method)
        assert isinstance(curve, DiscountCurve)
        df_5y = curve.df(date(2029, 1, 15))
        assert 0.5 < df_5y < 1.0

    def test_flat_forwards_constant(self, pillar_dates, pillar_dfs):
        """Flat input → piecewise constant forwards should be ~constant."""
        curve = build_forward_curve(REF, pillar_dates, pillar_dfs,
                                     ForwardInterpolationMethod.PIECEWISE_CONSTANT)
        df_3y = curve.df(date(2027, 1, 15))
        expected = math.exp(-0.04 * 3)
        assert abs(df_3y - expected) < 0.01

    def test_monotone_convex_smooth(self, pillar_dates, upward_dfs):
        """Monotone convex should produce smooth forwards."""
        curve = build_forward_curve(REF, pillar_dates, upward_dfs,
                                     ForwardInterpolationMethod.MONOTONE_CONVEX)
        # Extract forwards at several points
        fwds = extract_forwards(curve, [1.0, 2.0, 3.0, 5.0, 7.0, 9.0])
        # Should be roughly increasing (upward sloping)
        assert fwds[-1] > fwds[0]

    def test_pillar_dfs_preserved(self, pillar_dates, pillar_dfs):
        """Pillar DFs should be approximately preserved."""
        curve = build_forward_curve(REF, pillar_dates, pillar_dfs,
                                     ForwardInterpolationMethod.MONOTONE_CONVEX)
        for d, expected_df in zip(pillar_dates, pillar_dfs):
            actual_df = curve.df(d)
            assert abs(actual_df - expected_df) < 0.01, \
                f"At {d}: expected {expected_df:.6f}, got {actual_df:.6f}"


class TestMonotoneConvexForwards:
    def test_basic(self):
        times = [1.0, 2.0, 5.0, 10.0]
        zeros = [0.03, 0.035, 0.04, 0.045]
        f = monotone_convex_forwards(times, zeros)
        # Forward at time 0 should be near first zero
        assert abs(f(0.5) - zeros[0]) < 0.02
        # Forward should be positive everywhere
        for t in [0.1, 1.0, 3.0, 5.0, 8.0, 10.0]:
            assert f(t) > 0

    def test_flat_input(self):
        times = [1.0, 5.0, 10.0]
        zeros = [0.04, 0.04, 0.04]
        f = monotone_convex_forwards(times, zeros)
        # Flat zeros → flat forwards
        for t in [0.5, 2.0, 7.0]:
            assert abs(f(t) - 0.04) < 0.005

    def test_single_point(self):
        f = monotone_convex_forwards([5.0], [0.04])
        assert abs(f(3.0) - 0.04) < 0.001


class TestExtractForwards:
    def test_from_flat_curve(self):
        curve = DiscountCurve.flat(REF, 0.04)
        fwds = extract_forwards(curve, [1.0, 5.0, 10.0])
        for f in fwds:
            assert abs(f - 0.04) < 0.005
