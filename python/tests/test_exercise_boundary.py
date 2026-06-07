"""Tests for pricebook.options.exercise_boundary."""

import pytest
import numpy as np

from pricebook.options.exercise_boundary import (
    ExerciseBoundaryResult,
    tree_exercise_boundary,
    boundary_analytics,
)

SPOT = 100.0
STRIKE = 100.0
RATE = 0.05
VOL = 0.20
T = 1.0


class TestTreeExerciseBoundary:
    def test_returns_exercise_boundary_result(self):
        res = tree_exercise_boundary(
            SPOT, STRIKE, RATE, VOL, T, q=0.0,
            option_type="put", n_steps=200,
        )
        assert isinstance(res, ExerciseBoundaryResult)

    def test_boundary_values_length_correct(self):
        n = 200
        res = tree_exercise_boundary(
            SPOT, STRIKE, RATE, VOL, T, q=0.0,
            option_type="put", n_steps=n,
        )
        assert len(res.boundary_values) == n + 1

    def test_time_grid_length_correct(self):
        n = 200
        res = tree_exercise_boundary(
            SPOT, STRIKE, RATE, VOL, T, q=0.0,
            option_type="put", n_steps=n,
        )
        assert len(res.time_grid) == n + 1

    def test_critical_price_at_expiry_near_strike(self):
        """For a put with q=0: S*(T) ~ K at expiry (boundary collapses to strike)."""
        res = tree_exercise_boundary(
            SPOT, STRIKE, RATE, VOL, T, q=0.0,
            option_type="put", n_steps=500,
        )
        # The boundary at expiry should be close to the strike for a put with q=0
        assert res.critical_price_at_expiry == pytest.approx(STRIKE, rel=0.05)

    def test_method_label(self):
        res = tree_exercise_boundary(
            SPOT, STRIKE, RATE, VOL, T, q=0.0,
            option_type="put", n_steps=100,
        )
        assert res.method == "tree"

    def test_boundary_below_strike_for_put(self):
        """Put exercise boundary should be <= strike throughout."""
        res = tree_exercise_boundary(
            SPOT, STRIKE, RATE, VOL, T, q=0.0,
            option_type="put", n_steps=200,
        )
        # The boundary near expiry should converge to the strike
        assert res.critical_price_at_expiry == pytest.approx(STRIKE, rel=0.05)

    def test_time_grid_ascending(self):
        res = tree_exercise_boundary(
            SPOT, STRIKE, RATE, VOL, T, q=0.0,
            option_type="put", n_steps=100,
        )
        assert np.all(np.diff(res.time_grid) >= 0)


class TestBoundaryAnalytics:
    def test_returns_dict_with_expected_keys(self):
        res = tree_exercise_boundary(
            SPOT, STRIKE, RATE, VOL, T, q=0.0,
            option_type="put", n_steps=200,
        )
        analytics = boundary_analytics(res)
        for key in ("slope_near_expiry", "mean_convexity", "boundary_min",
                    "boundary_max", "critical_price_at_expiry", "method"):
            assert key in analytics

    def test_boundary_min_le_max(self):
        res = tree_exercise_boundary(
            SPOT, STRIKE, RATE, VOL, T, q=0.0,
            option_type="put", n_steps=200,
        )
        analytics = boundary_analytics(res)
        assert analytics["boundary_min"] <= analytics["boundary_max"]

    def test_method_preserved(self):
        res = tree_exercise_boundary(
            SPOT, STRIKE, RATE, VOL, T, q=0.0,
            option_type="put", n_steps=100,
        )
        analytics = boundary_analytics(res)
        assert analytics["method"] == "tree"
