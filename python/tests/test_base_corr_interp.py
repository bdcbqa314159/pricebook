"""Tests for base correlation surface with interpolation."""

import pytest
import numpy as np

from pricebook.credit.tranche_pricing import BaseCorrelationSurface


class TestBaseCorrelationSurface:
    def test_construction(self):
        bcs = BaseCorrelationSurface([0.03, 0.06, 0.09, 0.12, 0.22],
                                      [0.15, 0.25, 0.35, 0.45, 0.60])
        assert len(bcs.detachments) == 5

    def test_linear_interpolation(self):
        bcs = BaseCorrelationSurface([0.03, 0.06, 0.12],
                                      [0.20, 0.35, 0.50])
        # Midpoint of 0.03 and 0.06
        r = bcs.interpolate(0.045, method="linear")
        assert r == pytest.approx(0.275, abs=0.01)

    def test_cubic_interpolation(self):
        bcs = BaseCorrelationSurface([0.03, 0.06, 0.09, 0.12, 0.22],
                                      [0.15, 0.25, 0.35, 0.45, 0.60])
        r = bcs.interpolate(0.075, method="cubic")
        assert 0.25 < r < 0.40  # between neighbours

    def test_callable(self):
        bcs = BaseCorrelationSurface([0.03, 0.12], [0.20, 0.50])
        assert bcs(0.03) == pytest.approx(0.20, abs=0.01)
        assert bcs(0.12) == pytest.approx(0.50, abs=0.01)

    def test_at_calibration_points(self):
        """Interpolation should reproduce calibration points."""
        bcs = BaseCorrelationSurface([0.03, 0.06, 0.09, 0.12],
                                      [0.18, 0.30, 0.42, 0.55])
        for d, r in zip(bcs.detachments, bcs.base_correlations):
            assert bcs.interpolate(d, "linear") == pytest.approx(r, abs=0.01)

    def test_monotonicity_enforced(self):
        """Cubic should enforce monotonicity via clamping."""
        bcs = BaseCorrelationSurface([0.03, 0.06, 0.09, 0.12, 0.22],
                                      [0.15, 0.25, 0.35, 0.45, 0.60])
        # All interpolated values should be within [0.15, 0.60]
        for d in np.linspace(0.03, 0.22, 20):
            r = bcs.interpolate(d, "cubic")
            assert 0.14 <= r <= 0.61


class TestArbitrageCheck:
    def test_valid_surface(self):
        bcs = BaseCorrelationSurface([0.03, 0.06, 0.12], [0.20, 0.35, 0.50])
        result = bcs.check_arbitrage()
        assert result["valid"]

    def test_non_monotonic_detected(self):
        bcs = BaseCorrelationSurface([0.03, 0.06, 0.12], [0.40, 0.30, 0.50])
        result = bcs.check_arbitrage()
        assert not result["valid"]
        assert any("Non-monotonic" in i for i in result["issues"])

    def test_out_of_bounds(self):
        bcs = BaseCorrelationSurface([0.03, 0.12], [1.5, 0.50])
        result = bcs.check_arbitrage()
        assert not result["valid"]


class TestBumpAndFactory:
    def test_bump(self):
        bcs = BaseCorrelationSurface([0.03, 0.12], [0.20, 0.50])
        bumped = bcs.bump(0.05)
        assert bumped.base_correlations[0] == pytest.approx(0.25)
        assert bumped.base_correlations[1] == pytest.approx(0.55)

    def test_bump_clamped(self):
        bcs = BaseCorrelationSurface([0.03, 0.12], [0.95, 0.98])
        bumped = bcs.bump(0.10)
        assert all(r <= 0.999 for r in bumped.base_correlations)

    def test_from_calibration(self):
        cal = {0.03: 0.18, 0.06: 0.30, 0.12: 0.50}
        bcs = BaseCorrelationSurface.from_calibration(cal)
        assert bcs.detachments == [0.03, 0.06, 0.12]
        assert bcs.base_correlations == [0.18, 0.30, 0.50]

    def test_to_dict(self):
        bcs = BaseCorrelationSurface([0.03, 0.12], [0.20, 0.50])
        d = bcs.to_dict()
        assert "detachments" in d
        assert "base_correlations" in d
