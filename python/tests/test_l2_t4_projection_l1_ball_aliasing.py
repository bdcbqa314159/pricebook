"""Regression for L2 Wave-2 audit — `projection_l1_ball` returned the
INPUT alias when ``x`` was already inside the L1 ball.

Pre-fix:

    if np.sum(np.abs(x)) <= radius:
        return x         # ← aliasing bug

Mutating the result mutated the caller's input array.  The other branch
(``np.sign(x) * proj``) always returned a fresh array via broadcast
multiplication, so the inconsistency was: fast path aliased, slow path
did not.  Calling code expecting a defensive copy would silently get a
shared buffer in the fast path.

Post-fix: both branches return a fresh array (``x.copy()`` in the fast
path, broadcast multiplication in the slow path).
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.numerical._optimize import projection_l1_ball


class TestNoAliasingFastPath:
    def test_result_is_fresh_copy_when_inside_ball(self):
        x = np.array([0.3, 0.4])  # ||x||_1 = 0.7 < 1.0
        result = projection_l1_ball(x, radius=1.0)
        assert result is not x

    def test_mutating_result_does_not_mutate_input(self):
        """Most important: writes to result must NOT propagate to input."""
        x = np.array([0.3, 0.4])
        x_orig = x.copy()
        result = projection_l1_ball(x, radius=1.0)
        result[0] = 99.0
        np.testing.assert_array_equal(x, x_orig)


class TestSlowPathStillCopies:
    def test_result_is_not_input_when_outside_ball(self):
        x = np.array([2.0, 3.0])  # ||x||_1 = 5 > 1.0
        result = projection_l1_ball(x, radius=1.0)
        assert result is not x

    def test_projected_value_norm_equals_radius(self):
        """Sanity: projection onto a ball of radius r has ||result||_1 = r
        (when input was outside)."""
        x = np.array([2.0, 3.0])
        result = projection_l1_ball(x, radius=1.0)
        assert np.sum(np.abs(result)) == pytest.approx(1.0)


class TestMathematicalCorrectness:
    def test_projection_inside_ball_is_identity(self):
        """When x is inside the ball, the projection's value should equal
        x (even though it's a fresh array)."""
        x = np.array([0.2, 0.3, -0.1])
        result = projection_l1_ball(x, radius=2.0)
        np.testing.assert_array_equal(result, x)
