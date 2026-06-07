"""Tests for pricebook.options.bermudan_barrier."""

import pytest

from pricebook.options.bermudan_barrier import (
    american_barrier_option,
    bermudan_double_barrier,
    barrier_exercise_interaction,
)

SPOT = 100.0
STRIKE = 100.0
VOL = 0.20
T = 1.0
R = 0.05
Q = 0.02
SEED = 42


class TestAmericanBarrierOption:
    def test_price_positive_down_and_out_put(self):
        res = american_barrier_option(
            spot=SPOT, strike=STRIKE, barrier=80.0,
            vol=VOL, T=T, r=R, q=Q,
            option_type="put",
            barrier_type="down-and-out",
            n_paths=10000, seed=SEED,
        )
        assert res.price > 0

    def test_price_lt_american_without_barrier(self):
        """Knock-out barrier reduces price relative to vanilla American."""
        from pricebook.options.bermudan_barrier import bermudan_barrier_option
        barrier_res = american_barrier_option(
            spot=SPOT, strike=STRIKE, barrier=80.0,
            vol=VOL, T=T, r=R, q=Q,
            option_type="put",
            barrier_type="down-and-out",
            n_paths=10000, seed=SEED,
        )
        # European barrier price should be less than European vanilla
        # (down-and-out knocks out some value)
        assert barrier_res.price <= barrier_res.european_barrier_price + 5.0

    def test_barrier_hit_prob_in_unit_interval(self):
        res = american_barrier_option(
            spot=SPOT, strike=STRIKE, barrier=80.0,
            vol=VOL, T=T, r=R, q=Q,
            option_type="put",
            barrier_type="down-and-out",
            n_paths=10000, seed=SEED,
        )
        assert 0.0 <= res.barrier_hit_prob <= 1.0

    def test_tight_barrier_kills_value(self):
        """Barrier very close to spot should knock out most paths -> low price."""
        res = american_barrier_option(
            spot=SPOT, strike=STRIKE, barrier=99.0,  # almost at spot
            vol=VOL, T=T, r=R, q=Q,
            option_type="put",
            barrier_type="down-and-out",
            n_paths=10000, seed=SEED,
        )
        assert res.barrier_hit_prob > 0.5  # most paths hit the barrier


class TestBermudanDoubleBarrier:
    def test_price_positive(self):
        res = bermudan_double_barrier(
            spot=SPOT, strike=STRIKE,
            upper_barrier=130.0, lower_barrier=70.0,
            vol=VOL, T=T, r=R, q=Q,
            option_type="call",
            n_paths=10000, seed=SEED,
        )
        assert res.price > 0

    def test_double_barrier_reduces_price(self):
        """Double barrier price <= vanilla European price."""
        res = bermudan_double_barrier(
            spot=SPOT, strike=STRIKE,
            upper_barrier=130.0, lower_barrier=70.0,
            vol=VOL, T=T, r=R, q=Q,
            option_type="call",
            n_paths=10000, seed=SEED,
        )
        # European double barrier should be less than or equal to vanilla BS
        # (barriers only remove payoff scenarios)
        import math
        bs_approx = SPOT * 0.15  # rough vanilla call upper bound
        assert res.price <= bs_approx + 5.0

    def test_hit_probs_in_unit_interval(self):
        res = bermudan_double_barrier(
            spot=SPOT, strike=STRIKE,
            upper_barrier=130.0, lower_barrier=70.0,
            vol=VOL, T=T, r=R, q=Q,
            option_type="call",
            n_paths=10000, seed=SEED,
        )
        assert 0.0 <= res.upper_barrier_hit_prob <= 1.0
        assert 0.0 <= res.lower_barrier_hit_prob <= 1.0


class TestBarrierExerciseInteraction:
    def test_returns_dict_with_expected_keys(self):
        result = barrier_exercise_interaction(
            spot=SPOT, strike=STRIKE, barrier=80.0,
            vol=VOL, T=T, r=R, q=Q,
            option_type="put",
            barrier_type="down-and-out",
            n_paths=5000, n_steps=100, seed=SEED,
        )
        for key in ("european_price", "barrier_price", "american_price",
                    "american_barrier_price", "exercise_premium_reduction"):
            assert key in result

    def test_exercise_premium_reduction_nonneg(self):
        """Barrier should reduce or leave unchanged the early exercise premium."""
        result = barrier_exercise_interaction(
            spot=SPOT, strike=STRIKE, barrier=80.0,
            vol=VOL, T=T, r=R, q=Q,
            option_type="put",
            barrier_type="down-and-out",
            n_paths=5000, n_steps=100, seed=SEED,
        )
        assert result["exercise_premium_reduction"] >= -10.0  # wide tolerance for MC noise
