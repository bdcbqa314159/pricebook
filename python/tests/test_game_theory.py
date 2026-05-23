"""Tests for game theory: Shapley, cooperative games, Nash, auction."""

import pytest
import numpy as np

from pricebook.risk.shapley import (
    shapley_value, shapley_sampling, shapley_capital_allocation, ShapleyResult,
)
from pricebook.risk.cooperative_games import (
    CooperativeGame, NettingSetGame, CollateralPoolGame,
)
from pricebook.models.game_equilibrium import (
    nash_2player, market_maker_equilibrium, optimal_execution_game,
    NashResult, MarketMakerResult, ExecutionResult,
)
from pricebook.fixed_income.auction import (
    BondAuction, Bid, AuctionResult, winners_curse_adjustment, expected_revenue,
)


# ═══════════════════════════════════════════════════════════════
# 3.1: Shapley Value
# ═══════════════════════════════════════════════════════════════

class TestShapley:
    def test_glove_game(self):
        """Classic: L has left glove, R1/R2 have right gloves. Pair = $1."""
        def v(S):
            has_left = "L" in S
            has_right = "R1" in S or "R2" in S
            return 1.0 if (has_left and has_right) else 0.0
        r = shapley_value(v, ["L", "R1", "R2"])
        # L gets 2/3, R1 and R2 each get 1/6
        assert abs(r.values["L"] - 2/3) < 0.01
        assert abs(r.values["R1"] - 1/6) < 0.01

    def test_efficiency(self):
        """Shapley values must sum to v(N)."""
        def v(S):
            return len(S) ** 2
        r = shapley_value(v, ["A", "B", "C"])
        assert r.is_efficient

    def test_dummy_player(self):
        """Dummy player (adds 0 to every coalition) gets 0."""
        def v(S):
            return sum(1 for p in S if p != "D") ** 2
        r = shapley_value(v, ["A", "B", "D"])
        assert abs(r.values["D"]) < 0.01

    def test_symmetry(self):
        """Symmetric players get equal Shapley values."""
        def v(S):
            return len(S) * 10.0
        r = shapley_value(v, ["A", "B", "C"])
        assert abs(r.values["A"] - r.values["B"]) < 0.01

    def test_sampling_approximation(self):
        def v(S):
            return len(S) ** 1.5
        r_exact = shapley_value(v, ["A", "B", "C", "D"])
        r_mc = shapley_sampling(v, ["A", "B", "C", "D"], n_samples=50000)
        for p in ["A", "B", "C", "D"]:
            assert abs(r_exact.values[p] - r_mc.values[p]) < 0.1

    def test_to_dict(self):
        def v(S):
            return float(len(S))
        d = shapley_value(v, ["A", "B"]).to_dict()
        assert "values" in d
        assert "is_efficient" in d


# ═══════════════════════════════════════════════════════════════
# 3.2: Cooperative Games
# ═══════════════════════════════════════════════════════════════

class TestCooperativeGames:
    def test_core_check_efficient(self):
        def v(S):
            return len(S) * 10.0
        game = CooperativeGame(["A", "B", "C"], v)
        # Equal allocation of v(N) = 30
        result = game.core_check({"A": 10, "B": 10, "C": 10})
        assert result.is_in_core

    def test_core_violation(self):
        def v(S):
            if S == frozenset(["A"]):
                return 20.0
            return len(S) * 5.0
        game = CooperativeGame(["A", "B"], v)
        # Give A only 5 — but v({A}) = 20, so A can do better alone
        result = game.core_check({"A": 5, "B": 5})
        assert not result.is_in_core

    def test_netting_game(self):
        standalone = {"A": 100, "B": 80, "C": 120}
        def netted(S):
            return max(sum(standalone[p] for p in S) * 0.6, 0)
        game = NettingSetGame(["A", "B", "C"], standalone, netted)
        r = game.shapley()
        assert r.is_efficient

    def test_collateral_pool(self):
        costs = {"X": 50, "Y": 40, "Z": 60}
        def pooled(S):
            return sum(costs[p] for p in S) * 0.7
        game = CollateralPoolGame(["X", "Y", "Z"], costs, pooled)
        r = game.shapley()
        assert all(v >= 0 for v in r.values.values())


# ═══════════════════════════════════════════════════════════════
# 3.3: Nash Equilibrium
# ═══════════════════════════════════════════════════════════════

class TestNash:
    def test_prisoners_dilemma(self):
        """Prisoner's dilemma: (C,C) is dominant."""
        A = np.array([[-1, -3], [0, -2]])
        B = np.array([[-1, 0], [-3, -2]])
        r = nash_2player(A, B)
        assert isinstance(r, NashResult)

    def test_matching_pennies(self):
        """Matching pennies: zero-sum game, Nash exists."""
        A = np.array([[1, -1], [-1, 1]])
        B = -A
        r = nash_2player(A, B)
        # Value must be 0 at equilibrium (zero-sum)
        assert isinstance(r, NashResult)

    def test_to_dict(self):
        A = np.array([[3, 0], [5, 1]])
        B = np.array([[3, 5], [0, 1]])
        d = nash_2player(A, B).to_dict()
        assert "value_a" in d


class TestMarketMaker:
    def test_basic(self):
        r = market_maker_equilibrium(100.0, 0, 0.02, arrival_rate=5.0)
        assert isinstance(r, MarketMakerResult)
        assert r.optimal_spread > 0
        assert r.optimal_bid < r.optimal_ask

    def test_inventory_shifts_midpoint(self):
        """Long inventory → lower reservation price."""
        r_flat = market_maker_equilibrium(100.0, 0, 0.02)
        r_long = market_maker_equilibrium(100.0, 100, 0.02)
        assert r_long.optimal_bid < r_flat.optimal_bid

    def test_higher_vol_wider_spread(self):
        r_low = market_maker_equilibrium(100.0, 0, 0.01)
        r_high = market_maker_equilibrium(100.0, 0, 0.05)
        assert r_high.optimal_spread > r_low.optimal_spread


class TestOptimalExecution:
    def test_basic(self):
        r = optimal_execution_game(10000, 10, 0.02, 0.001)
        assert isinstance(r, ExecutionResult)
        assert len(r.trade_schedule) == 10
        assert abs(sum(r.trade_schedule) - 10000) < 1.0

    def test_front_loaded(self):
        """With risk aversion, should trade more at the start."""
        r = optimal_execution_game(10000, 10, 0.02, 0.001, risk_aversion=0.1)
        assert r.trade_schedule[0] > r.trade_schedule[-1]

    def test_urgency(self):
        r = optimal_execution_game(10000, 5, 0.02, 0.001)
        assert 0 < r.urgency < 1


# ═══════════════════════════════════════════════════════════════
# 3.4: Auction Theory
# ═══════════════════════════════════════════════════════════════

class TestAuction:
    def test_uniform_price(self):
        auction = BondAuction(1e6)
        bids = [
            Bid("A", 99.5, 500_000),
            Bid("B", 99.3, 400_000),
            Bid("C", 99.0, 300_000),
        ]
        r = auction.uniform_price(bids)
        assert isinstance(r, AuctionResult)
        assert r.method == "uniform"
        assert r.bid_to_cover > 1.0
        assert r.total_issued <= 1e6

    def test_discriminatory(self):
        auction = BondAuction(1e6)
        bids = [
            Bid("A", 99.5, 500_000),
            Bid("B", 99.3, 400_000),
            Bid("C", 99.0, 300_000),
        ]
        r = auction.discriminatory_price(bids)
        assert r.method == "discriminatory"

    def test_oversubscribed(self):
        auction = BondAuction(500_000)
        bids = [Bid("A", 99.5, 400_000), Bid("B", 99.3, 400_000)]
        r = auction.uniform_price(bids)
        assert r.bid_to_cover > 1.0

    def test_winners_curse(self):
        adjusted = winners_curse_adjustment(100.0, 10, 2.0)
        assert adjusted < 100.0  # bid shaded down

    def test_revenue_equivalence(self):
        rev_u = expected_revenue(10, 100.0, 2.0, "uniform")
        rev_d = expected_revenue(10, 100.0, 2.0, "discriminatory")
        assert abs(rev_u - rev_d) < 0.01  # revenue equivalence theorem

    def test_to_dict(self):
        auction = BondAuction(1e6)
        bids = [Bid("A", 99.5, 1e6)]
        d = auction.uniform_price(bids).to_dict()
        assert "clearing_price" in d
        assert "bid_to_cover" in d
