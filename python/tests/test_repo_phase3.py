"""Tests for repo Phase 3: leverage optimization, collateral transformation."""

import pytest

from pricebook.risk.leverage_optimisation import (
    optimise_leverage, leverage_frontier, LeverageOptResult,
)
from pricebook.risk.collateral_transformation import (
    transformation_cost, optimise_transformation, funding_arbitrage,
    TransformationResult, TransformationTrade,
)


# ═══════════════════════════════════════════════════════════════
# 3.1: Leverage Optimization
# ═══════════════════════════════════════════════════════════════

class TestLeverageOptimisation:
    def test_basic(self):
        carries = [0.005, 0.008, 0.003]   # 50bp, 80bp, 30bp carry
        haircuts = [0.02, 0.06, 0.01]      # 2%, 6%, 1%
        rwa = [0.20, 1.0, 0.04]            # sovereign, corporate, agency
        r = optimise_leverage(carries, haircuts, rwa, capital=1e6)
        assert isinstance(r, LeverageOptResult)
        assert r.optimal_carry > 0
        assert r.leverage_ratio > 0
        assert r.n_trades == 3

    def test_highest_carry_preferred(self):
        """Trade with best carry should get at least as much allocation."""
        carries = [0.01, 0.001, 0.001]
        haircuts = [0.02, 0.02, 0.02]
        rwa = [0.20, 0.20, 0.20]
        r = optimise_leverage(carries, haircuts, rwa, capital=1e6)
        assert r.optimal_weights[0] >= r.optimal_weights[1]

    def test_haircut_constraint_binds(self):
        """High haircuts should limit leverage."""
        carries = [0.01, 0.01]
        haircuts = [0.50, 0.50]  # 50% haircuts → max 2× leverage
        rwa = [0.20, 0.20]
        r = optimise_leverage(carries, haircuts, rwa, capital=1e6, max_leverage=10)
        assert r.leverage_ratio <= 2.1  # haircut constrains before leverage cap

    def test_concentration_limit(self):
        """No single trade should exceed max_single_trade_pct."""
        carries = [0.01]
        haircuts = [0.02]
        rwa = [0.20]
        r = optimise_leverage(carries, haircuts, rwa, capital=1e6,
                               max_leverage=10, max_single_trade_pct=0.20)
        max_trade = max(r.optimal_weights)
        max_allowed = 0.20 * 1e6 * 10
        assert max_trade <= max_allowed * 1.01

    def test_empty(self):
        r = optimise_leverage([], [], [], 1e6)
        assert r.n_trades == 0
        assert r.optimal_carry == 0.0

    def test_to_dict(self):
        r = optimise_leverage([0.01], [0.02], [0.20], 1e6)
        d = r.to_dict()
        assert "leverage_ratio" in d


class TestLeverageFrontier:
    def test_frontier(self):
        carries = [0.005, 0.008]
        haircuts = [0.02, 0.06]
        rwa = [0.20, 1.0]
        frontier = leverage_frontier(carries, haircuts, rwa, 1e6)
        assert len(frontier) == 8  # default 8 leverage levels
        # Higher leverage should give equal or higher carry
        for i in range(1, len(frontier)):
            assert frontier[i]["optimal_carry"] >= frontier[i-1]["optimal_carry"] - 1e-6


# ═══════════════════════════════════════════════════════════════
# 3.2: Collateral Transformation
# ═══════════════════════════════════════════════════════════════

class TestTransformationCost:
    def test_upgrade_cost(self):
        """Upgrading HY (15% haircut) to sovereign (2%) has a cost but benefit."""
        r = transformation_cost(0.15, 0.02, repo_spread_bp=50)
        assert r["repo_cost_bp"] > 0
        assert r["haircut_benefit_bp"] > 0

    def test_zero_spread_free(self):
        r = transformation_cost(0.10, 0.02, repo_spread_bp=0)
        assert r["total_cost_bp"] == 0


class TestOptimiseTransformation:
    def test_basic(self):
        available = [
            {"asset": "IG_corp", "haircut": 0.06, "notional": 5e6, "repo_spread_bp": 30},
            {"asset": "HY_corp", "haircut": 0.15, "notional": 3e6, "repo_spread_bp": 80},
        ]
        r = optimise_transformation(available, "HQLA_L1", 0.02, max_notional=5e6)
        assert isinstance(r, TransformationResult)
        assert len(r.trades) >= 1
        assert r.haircut_improvement > 0

    def test_empty(self):
        r = optimise_transformation([], "HQLA_L1", 0.02, 1e6)
        assert len(r.trades) == 0

    def test_to_dict(self):
        available = [{"asset": "IG", "haircut": 0.06, "notional": 1e6, "repo_spread_bp": 20}]
        d = optimise_transformation(available, "HQLA_L1", 0.02, 1e6).to_dict()
        assert "total_cost_bp" in d


class TestFundingArbitrage:
    def test_identifies_arbitrage(self):
        sec = {"UST": 0.05, "IG": 0.055, "HY": 0.07}
        unsec = {"UST": 0.06, "IG": 0.06, "HY": 0.06}
        haircuts = {"UST": 0.02, "IG": 0.06, "HY": 0.15}
        results = funding_arbitrage(sec, unsec, haircuts)
        assert len(results) == 3
        # UST should have best arbitrage (lowest secured, lowest haircut)
        assert results[0]["asset"] == "UST"
        assert results[0]["arbitrage_bp"] > 0
