"""Regression for L2 Tier-3 T3.8 / T3.9 / T3.10 — three small bugs in
`numerical/_trees.py`'s convenience functions.

* T3.8 — ``solve_tree_2d`` returned ``delta=gamma=theta=0`` unconditionally.
  Greeks for 2-asset trees were just missing.
* T3.9 — ``solve_tree_2d`` American-exercise projection only handled the
  ``"spread_call"`` payoff_type.  Other payoffs (spread_put, best_of_call,
  worst_of_call, and custom callables) silently priced as European even when
  ``is_american=True`` was passed.
* T3.10 — ``solve_tree`` convenience wrapper didn't accept
  ``exercise_dates``, so a caller passing ``exercise=BERMUDAN`` got an
  empty exercise set and the option silently degraded to European.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.numerical._trees import (
    ExerciseType,
    TreeMethod,
    solve_tree,
    solve_tree_2d,
)


# ============================================================
# T3.10 — solve_tree(exercise_dates=...)
# ============================================================


class TestSolveTreeBermudan:
    def test_bermudan_with_exercise_dates_dominates_european(self):
        """Bermudan put with mid-life exercise points should be ≥ European
        on a put (early exercise is potentially valuable for puts).  Pre-fix
        the wrapper dropped exercise_dates → Bermudan ≡ European."""
        res_berm = solve_tree(
            spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0,
            method=TreeMethod.CRR, n_steps=200,
            exercise=ExerciseType.BERMUDAN,
            exercise_dates=[50, 100, 150, 199],
            is_call=False,
        )
        res_eur = solve_tree(
            spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0,
            method=TreeMethod.CRR, n_steps=200,
            exercise=ExerciseType.EUROPEAN,
            is_call=False,
        )
        # Bermudan must strictly exceed European for a put (early exercise
        # value).  Pre-fix they were identical (Bermudan silently degraded).
        assert res_berm.price > res_eur.price + 1e-4, (
            f"Bermudan put ({res_berm.price:.4f}) should exceed European "
            f"({res_eur.price:.4f}) — T3.10 fix not applied?"
        )

    def test_bermudan_with_no_exercise_dates_equals_european(self):
        """No exercise dates supplied → BERMUDAN behaves like European
        (since `_should_exercise` returns False for every step)."""
        res_berm = solve_tree(
            spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0,
            n_steps=100, exercise=ExerciseType.BERMUDAN, is_call=False,
        )
        res_eur = solve_tree(
            spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0,
            n_steps=100, exercise=ExerciseType.EUROPEAN, is_call=False,
        )
        assert abs(res_berm.price - res_eur.price) < 1e-10


# ============================================================
# T3.8 — solve_tree_2d Greeks
# ============================================================


class TestSolveTree2DGreeks:
    def test_spread_call_delta_nonzero(self):
        """A 2-asset spread call has positive delta on S1, negative on S2.
        Pre-fix both were 0."""
        res = solve_tree_2d(
            S1=100.0, S2=95.0, strike=5.0,
            rate=0.05, vol1=0.20, vol2=0.25, rho=0.5, T=1.0,
            n_steps=80, payoff_type="spread_call",
        )
        # delta1 (the conventional `delta` field) must be positive — increase
        # S1 ⇒ increase spread payoff.
        assert res.delta > 0.05, f"delta1 = {res.delta:.4f} not > 0"
        # delta2 is in node_prices[1] by convention (2-element array).
        assert res.node_prices is not None
        assert res.node_prices.shape == (2,)
        assert res.node_prices[1] < -0.05, (
            f"delta2 = {res.node_prices[1]:.4f} not < 0"
        )

    def test_delta_sum_at_high_spread_approaches_one(self):
        """For a deep-ITM spread call with S1 >> S2 + K, the spread call
        approaches (S1 - S2 - K) intrinsic and delta1 ≈ +1, delta2 ≈ -1.
        Pre-fix both were 0."""
        res = solve_tree_2d(
            S1=200.0, S2=50.0, strike=10.0,
            rate=0.05, vol1=0.10, vol2=0.10, rho=0.5, T=0.25,
            n_steps=60, payoff_type="spread_call",
        )
        assert 0.85 < res.delta < 1.05, f"delta1 = {res.delta:.4f}"
        assert -1.05 < res.node_prices[1] < -0.85, (
            f"delta2 = {res.node_prices[1]:.4f}"
        )


# ============================================================
# T3.9 — American projection works for all payoff_types
# ============================================================


class TestSolveTree2DAmerican:
    def test_american_spread_put_premium(self):
        """spread_put is one of the cases pre-fix silently dropped from the
        American projection.  Post-fix American ≥ European for an ITM
        spread put."""
        am = solve_tree_2d(
            S1=80.0, S2=100.0, strike=5.0,
            rate=0.05, vol1=0.30, vol2=0.25, rho=0.3, T=1.0,
            n_steps=80, payoff_type="spread_put", is_american=True,
        )
        eu = solve_tree_2d(
            S1=80.0, S2=100.0, strike=5.0,
            rate=0.05, vol1=0.30, vol2=0.25, rho=0.3, T=1.0,
            n_steps=80, payoff_type="spread_put", is_american=False,
        )
        # American >= European, and strictly > for sufficiently ITM put.
        assert am.price >= eu.price - 1e-10, (
            f"American {am.price:.4f} < European {eu.price:.4f}"
        )
        assert am.price > eu.price + 1e-4, (
            f"American spread put no premium over European: "
            f"am={am.price:.4f}, eu={eu.price:.4f}"
        )

    def test_american_worst_of_call_with_custom_payoff(self):
        """Custom payoff callable case (pre-fix not in the American branch).
        Smoke check: the American result must be ≥ European."""
        def payoff(s1, s2):
            return max(min(s1, s2) - 90.0, 0.0)

        am = solve_tree_2d(
            S1=100.0, S2=110.0, strike=0.0,
            rate=0.05, vol1=0.25, vol2=0.30, rho=0.4, T=1.0,
            n_steps=60, payoff=payoff, is_american=True,
        )
        eu = solve_tree_2d(
            S1=100.0, S2=110.0, strike=0.0,
            rate=0.05, vol1=0.25, vol2=0.30, rho=0.4, T=1.0,
            n_steps=60, payoff=payoff, is_american=False,
        )
        assert am.price >= eu.price - 1e-10
