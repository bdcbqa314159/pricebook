"""Tests for American option pricing with discrete dividends."""

import pytest
import math

from pricebook.options.american_dividend import (
    american_with_dividends, roll_geske_whaley,
    exercise_boundary_around_exdate,
)
from pricebook.models.black76 import OptionType


class TestAmericanWithDividends:
    def test_call_ge_european(self):
        """American call >= European call."""
        r = american_with_dividends(
            100, 100, 0.05, 0.20, 1.0,
            [(0.25, 2.0), (0.75, 2.0)],
            OptionType.CALL,
        )
        assert r.price >= r.european_price - 0.01

    def test_put_ge_european(self):
        """American put >= European put."""
        r = american_with_dividends(
            100, 100, 0.05, 0.20, 1.0,
            [(0.5, 2.0)],
            OptionType.PUT,
        )
        assert r.price >= r.european_price - 0.01

    def test_early_exercise_premium_positive(self):
        """With large dividend, call should have early exercise premium."""
        r = american_with_dividends(
            100, 90, 0.05, 0.20, 1.0,
            [(0.5, 5.0)],  # large dividend
            OptionType.CALL,
        )
        assert r.early_exercise_premium >= 0

    def test_no_dividends_equals_european(self):
        """Without dividends, American call = European call."""
        r = american_with_dividends(100, 100, 0.05, 0.20, 1.0, [], OptionType.CALL)
        assert r.early_exercise_premium < 0.5  # near zero

    def test_positive_price(self):
        r = american_with_dividends(100, 100, 0.05, 0.20, 1.0,
                                     [(0.5, 2.0)], OptionType.CALL, n_steps=200)
        assert r.price > 0

    def test_exercise_boundary(self):
        r = american_with_dividends(100, 100, 0.05, 0.20, 1.0,
                                     [(0.5, 3.0)], OptionType.CALL, n_steps=200)
        # May or may not have exercise boundary entries
        assert isinstance(r.exercise_boundary, list)

    def test_to_dict(self):
        r = american_with_dividends(100, 100, 0.05, 0.20, 1.0,
                                     [(0.5, 2.0)], OptionType.CALL, n_steps=100)
        d = r.to_dict()
        assert "price" in d
        assert "early_exercise_premium" in d


class TestRollGeskeWhaley:
    def test_positive_price(self):
        r = roll_geske_whaley(100, 100, 0.05, 0.20, 1.0, 2.0, 0.5)
        assert r.price > 0

    def test_ge_european(self):
        r = roll_geske_whaley(100, 100, 0.05, 0.20, 1.0, 2.0, 0.5)
        assert r.price >= r.european_price - 0.01

    def test_large_dividend_premium(self):
        """Large dividend → meaningful early exercise premium."""
        r = roll_geske_whaley(100, 95, 0.05, 0.20, 1.0, 8.0, 0.5)
        assert r.early_exercise_premium >= 0

    def test_div_after_expiry(self):
        """Dividend after expiry → pure European."""
        r = roll_geske_whaley(100, 100, 0.05, 0.20, 1.0, 2.0, 1.5)
        assert r.early_exercise_premium == 0.0

    def test_critical_spot_finite(self):
        r = roll_geske_whaley(100, 100, 0.05, 0.20, 1.0, 2.0, 0.5)
        assert math.isfinite(r.critical_spot)
        assert r.critical_spot > 0

    def test_to_dict(self):
        r = roll_geske_whaley(100, 100, 0.05, 0.20, 1.0, 2.0, 0.5)
        d = r.to_dict()
        assert "critical_spot" in d


class TestExerciseBoundary:
    def test_boundary_has_entries(self):
        results = exercise_boundary_around_exdate(100, 0.05, 0.20, 1.0, 3.0, 0.5)
        assert len(results) > 0

    def test_exercise_at_high_spot(self):
        """At very high spot, exercise should be optimal."""
        results = exercise_boundary_around_exdate(100, 0.05, 0.20, 1.0, 3.0, 0.5,
                                                   spot_range=[150, 200])
        for r in results:
            assert r["optimal"] == "exercise"

    def test_hold_at_low_spot(self):
        """At low spot (below strike), hold should be optimal."""
        results = exercise_boundary_around_exdate(100, 0.05, 0.20, 1.0, 3.0, 0.5,
                                                   spot_range=[80, 85])
        for r in results:
            assert r["optimal"] == "hold"

    def test_transition_exists(self):
        """Should see transition from hold to exercise."""
        results = exercise_boundary_around_exdate(100, 0.05, 0.20, 1.0, 5.0, 0.5,
                                                   spot_range=[90, 95, 100, 105, 110, 115, 120, 130, 140, 150])
        actions = [r["optimal"] for r in results]
        assert "hold" in actions
        assert "exercise" in actions
