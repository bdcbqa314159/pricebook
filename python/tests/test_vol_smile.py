"""Tests for VolSmile."""

import pytest
import numpy as np

from pricebook.vol_smile import VolSmile


STRIKES = [90.0, 95.0, 100.0, 105.0, 110.0]
VOLS = [0.25, 0.22, 0.20, 0.22, 0.25]


class TestConstruction:
    def test_basic(self):
        smile = VolSmile(STRIKES, VOLS)
        assert len(smile.strikes) == 5

    def test_sorts_by_strike(self):
        smile = VolSmile([110, 90, 100], [0.25, 0.25, 0.20])
        np.testing.assert_array_equal(smile.strikes, [90, 100, 110])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            VolSmile([90, 100], [0.20])

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            VolSmile([100], [0.20])

    def test_negative_vol_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            VolSmile([90, 100], [-0.01, 0.20])


class TestAtPillars:
    def test_exact_strike(self):
        smile = VolSmile(STRIKES, VOLS)
        assert smile.vol(100.0) == pytest.approx(0.20, abs=1e-10)

    def test_first_strike(self):
        smile = VolSmile(STRIKES, VOLS)
        assert smile.vol(90.0) == pytest.approx(0.25, abs=1e-10)

    def test_last_strike(self):
        smile = VolSmile(STRIKES, VOLS)
        assert smile.vol(110.0) == pytest.approx(0.25, abs=1e-10)


class TestInterpolation:
    def test_between_pillars(self):
        smile = VolSmile(STRIKES, VOLS)
        v = smile.vol(97.5)
        # Between 95 (0.22) and 100 (0.20), should be around 0.21
        assert 0.19 < v < 0.23

    def test_smooth(self):
        """Cubic spline should be smooth between points."""
        smile = VolSmile(STRIKES, VOLS)
        vols = [smile.vol(k) for k in np.linspace(91, 109, 50)]
        # No jumps: consecutive differences should be small
        diffs = np.abs(np.diff(vols))
        assert np.all(diffs < 0.01)

    def test_symmetric_smile(self):
        """Symmetric strikes/vols should give symmetric interpolation."""
        smile = VolSmile(STRIKES, VOLS)
        v_low = smile.vol(92.5)
        v_high = smile.vol(107.5)
        assert v_low == pytest.approx(v_high, abs=0.005)


class TestExtrapolation:
    def test_flat_below(self):
        smile = VolSmile(STRIKES, VOLS)
        assert smile.vol(50.0) == pytest.approx(0.25)
        assert smile.vol(80.0) == pytest.approx(0.25)

    def test_flat_above(self):
        smile = VolSmile(STRIKES, VOLS)
        assert smile.vol(150.0) == pytest.approx(0.25)
        assert smile.vol(120.0) == pytest.approx(0.25)


class TestMinimalSmile:
    def test_two_points(self):
        """Minimum: 2 points should still work."""
        smile = VolSmile([90, 110], [0.25, 0.20])
        v = smile.vol(100.0)
        assert 0.19 < v < 0.26
