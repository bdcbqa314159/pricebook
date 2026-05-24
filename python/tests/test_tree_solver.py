"""Tests for unified tree framework."""

import math
import pytest
import numpy as np
from scipy.stats import norm

from pricebook.numerical._trees import (
    TreeSolver, TreeMethod, ExerciseType, BarrierType, TreeResult,
    solve_tree, solve_tree_2d,
    tree_greeks, binomial_2d, TreeGreeks, Binomial2DResult,
)


def _bs_call(S, K, T, vol, r):
    d1 = (math.log(S/K) + (r + 0.5*vol**2)*T) / (vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)


# ═══════════════════════════════════════════════════════════════
# Method selection
# ═══════════════════════════════════════════════════════════════

class TestMethodSelection:
    @pytest.mark.parametrize("method", [TreeMethod.CRR, TreeMethod.JR, TreeMethod.TRINOMIAL])
    def test_core_methods_price_call(self, method):
        r = solve_tree(100, 100, 0.04, 0.25, 1.0, method=method, n_steps=200)
        bs = _bs_call(100, 100, 1.0, 0.25, 0.04)
        assert abs(r.price - bs) < 1.0

    def test_all_methods_produce_result(self):
        """Every method should produce a valid result."""
        for method in TreeMethod:
            r = solve_tree(100, 100, 0.04, 0.25, 1.0, method=method, n_steps=100)
            assert r.price > 0


# ═══════════════════════════════════════════════════════════════
# Exercise types
# ═══════════════════════════════════════════════════════════════

class TestExercise:
    def test_american_put_geq_european(self):
        eu = solve_tree(100, 100, 0.04, 0.25, 1.0, exercise=ExerciseType.EUROPEAN, is_call=False)
        am = solve_tree(100, 100, 0.04, 0.25, 1.0, exercise=ExerciseType.AMERICAN, is_call=False)
        assert am.price >= eu.price - 0.01

    def test_american_call_eq_european(self):
        eu = solve_tree(100, 100, 0.04, 0.25, 1.0, exercise=ExerciseType.EUROPEAN, is_call=True)
        am = solve_tree(100, 100, 0.04, 0.25, 1.0, exercise=ExerciseType.AMERICAN, is_call=True)
        assert abs(am.price - eu.price) < 0.5

    def test_bermudan_between(self):
        eu = solve_tree(100, 100, 0.04, 0.25, 1.0, exercise=ExerciseType.EUROPEAN, is_call=False, n_steps=100)
        am = solve_tree(100, 100, 0.04, 0.25, 1.0, exercise=ExerciseType.AMERICAN, is_call=False, n_steps=100)
        # Bermudan: exercise allowed at steps 25, 50, 75
        solver = TreeSolver(TreeMethod.CRR, 100, ExerciseType.BERMUDAN, exercise_dates=[25, 50, 75])
        bm = solver.solve(100, 100, 0.04, 0.25, 1.0, is_call=False)
        assert eu.price <= bm.price <= am.price + 0.1


# ═══════════════════════════════════════════════════════════════
# Barriers
# ═══════════════════════════════════════════════════════════════

class TestBarrier:
    def test_down_out_less_than_vanilla(self):
        vanilla = solve_tree(100, 100, 0.04, 0.25, 1.0)
        ko = solve_tree(100, 100, 0.04, 0.25, 1.0,
                         barrier_type=BarrierType.DOWN_OUT, barrier_level=80)
        assert ko.price < vanilla.price

    def test_up_out_put(self):
        vanilla = solve_tree(100, 100, 0.04, 0.25, 1.0, is_call=False)
        ko = solve_tree(100, 100, 0.04, 0.25, 1.0, is_call=False,
                         barrier_type=BarrierType.UP_OUT, barrier_level=120)
        assert ko.price < vanilla.price


# ═══════════════════════════════════════════════════════════════
# Greeks
# ═══════════════════════════════════════════════════════════════

class TestGreeks:
    def test_delta_call_range(self):
        r = solve_tree(100, 100, 0.04, 0.25, 1.0)
        assert 0 < r.delta < 1

    def test_gamma_positive(self):
        r = solve_tree(100, 100, 0.04, 0.25, 1.0)
        assert r.gamma > 0

    def test_vega_positive(self):
        r = solve_tree(100, 100, 0.04, 0.25, 1.0)
        assert r.vega > 0

    def test_put_delta_negative(self):
        r = solve_tree(100, 100, 0.04, 0.25, 1.0, is_call=False)
        assert r.delta < 0


# ═══════════════════════════════════════════════════════════════
# Payoff flexibility
# ═══════════════════════════════════════════════════════════════

class TestPayoff:
    def test_custom_payoff(self):
        """Digital call: payoff = 1 if S > K else 0."""
        r = solve_tree(100, 100, 0.04, 0.25, 1.0,
                         payoff=lambda S: 1.0 if S > 100 else 0.0)
        assert 0 < r.price < 1

    def test_straddle_payoff(self):
        r = solve_tree(100, 100, 0.04, 0.25, 1.0,
                         payoff=lambda S: abs(S - 100))
        assert r.price > 0


# ═══════════════════════════════════════════════════════════════
# Convergence
# ═══════════════════════════════════════════════════════════════

class TestConvergence:
    def test_convergence_analysis(self):
        solver = TreeSolver(TreeMethod.CRR)
        conv = solver.convergence_analysis(100, 100, 0.04, 0.25, 1.0)
        assert "richardson" in conv
        assert conv["richardson"] is not None
        bs = _bs_call(100, 100, 1.0, 0.25, 0.04)
        assert abs(conv["richardson"] - bs) < abs(conv["prices"][-1] - bs)


# ═══════════════════════════════════════════════════════════════
# 2D Tree
# ═══════════════════════════════════════════════════════════════

class TestTree2D:
    def test_spread_call(self):
        r = solve_tree_2d(100, 90, 5, 0.04, 0.20, 0.25, 0.5, 1.0)
        assert r.price > 0

    def test_callable_payoff(self):
        r = solve_tree_2d(100, 100, 0, 0.04, 0.20, 0.25, 0.3, 1.0,
                           payoff=lambda s1, s2: max(s1 + s2 - 200, 0))
        assert r.price >= 0


# ═══════════════════════════════════════════════════════════════
# Class API
# ═══════════════════════════════════════════════════════════════

class TestClassAPI:
    def test_reusable(self):
        solver = TreeSolver(TreeMethod.CRR, 100)
        r1 = solver.solve(100, 100, 0.04, 0.25, 1.0)
        r2 = solver.solve(100, 110, 0.03, 0.30, 0.5)
        assert r1.price != r2.price

    def test_to_dict(self):
        r = solve_tree(100, 100, 0.04, 0.25, 1.0)
        d = r.to_dict()
        assert "price" in d and "delta" in d and "method" in d


# ═══════════════════════════════════════════════════════════════
# Backward compatibility
# ═══════════════════════════════════════════════════════════════

class TestBackwardCompat:
    def test_tree_greeks_old_api(self):
        def pricer(spot, strike, rate, vol, T, n, is_call, is_am, q):
            return solve_tree(spot, strike, rate, vol, T, n_steps=n,
                               exercise=ExerciseType.AMERICAN if is_am else ExerciseType.EUROPEAN,
                               is_call=is_call, div_yield=q).price
        g = tree_greeks(pricer, 100, 0.04, 0.25, 1.0, 100, n_steps=100)
        assert isinstance(g, TreeGreeks)
        assert g.price > 0

    def test_binomial_2d_old_api(self):
        r = binomial_2d(100, 90, 5, 0.04, 0.20, 0.25, 0.5, 1.0)
        assert isinstance(r, Binomial2DResult)
        assert r.price > 0
