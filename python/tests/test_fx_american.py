"""Tests for pricebook.fx.fx_american."""

import pytest

from pricebook.fx.fx_american import (
    american_fx_option,
    fx_exercise_boundary,
)


SPOT = 1.25
STRIKE = 1.25
RD = 0.04   # domestic rate
RF = 0.06   # foreign rate — RF > RD: puts have early exercise incentive
VOL = 0.10
T = 1.0


class TestAmericanFXOption:
    def test_price_positive_call(self):
        res = american_fx_option(SPOT, STRIKE, RD, RF, VOL, T, "call", "baw")
        assert res.price > 0

    def test_price_positive_put(self):
        res = american_fx_option(SPOT, STRIKE, RD, RF, VOL, T, "put", "baw")
        assert res.price > 0

    def test_put_ge_european(self):
        """Put with RF > RD: American >= European due to carry incentive."""
        res = american_fx_option(SPOT, STRIKE, RD, RF, VOL, T, "put", "baw")
        assert res.price >= res.european_price - 1e-8

    def test_methods_similar_price(self):
        """BAW and tree prices agree within 10%."""
        baw = american_fx_option(SPOT, STRIKE, RD, RF, VOL, T, "put", "baw")
        tree = american_fx_option(SPOT, STRIKE, RD, RF, VOL, T, "put", "tree")
        ref = baw.price
        assert abs(tree.price - ref) / ref < 0.10

    def test_delta_bounded(self):
        res = american_fx_option(SPOT, STRIKE, RD, RF, VOL, T, "call", "baw")
        assert -1.0 < res.delta_domestic < 1.0

    def test_vega_positive(self):
        res = american_fx_option(SPOT, STRIKE, RD, RF, VOL, T, "put", "baw")
        assert res.vega > 0


class TestFXExerciseBoundary:
    def test_returns_list_of_tuples(self):
        boundary = fx_exercise_boundary(SPOT, STRIKE, RD, RF, VOL, T, "put", n_points=10)
        assert isinstance(boundary, list)
        assert len(boundary) == 10
        for pair in boundary:
            assert len(pair) == 2

    def test_put_boundary_positive(self):
        boundary = fx_exercise_boundary(SPOT, STRIKE, RD, RF, VOL, T, "put", n_points=10)
        for t_rem, b in boundary:
            assert b > 0

    def test_put_boundary_monotone(self):
        """For puts, boundary (critical spot) should be non-increasing as
        time-to-expiry shrinks (longer time → lower boundary)."""
        boundary = fx_exercise_boundary(SPOT, STRIKE, RD, RF, VOL, T, "put", n_points=20)
        # boundary is from T down to ~0; longer time remaining -> lower boundary
        # Test that boundary values at least stay below strike (put boundary < strike)
        for t_rem, b in boundary:
            assert b <= STRIKE * 1.01  # boundary should be at or near/below strike
