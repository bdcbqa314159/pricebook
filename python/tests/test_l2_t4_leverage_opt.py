"""Regression for L2 phase-2 audit of `risk.leverage_optimisation`:

Pre-fix `optimise_leverage` used the ABSOLUTE concentration constraint
``w_i ≤ max_single_pct × capital × max_leverage`` instead of the
docstring-promised RELATIVE form ``w_i ≤ max_single_pct × Σw``.

When actual leverage falls below max_leverage, the absolute form is
looser.  Example: capital=$100M, max_lev=10x, max_single=30%,
actual_lev=5x ($500M notional).
- Absolute (pre-fix): single trade allowed up to $300M (60% of portfolio).
- Relative (docstring): single trade capped at $150M (30%).

Fix: implement the relative constraint linearly via
``(1 - max_pct)·w_i - max_pct·Σ_{j≠i} w_j ≤ 0``.
"""

from __future__ import annotations

import pytest

from pricebook.risk.leverage_optimisation import optimise_leverage


class TestRelativeConcentration:
    def test_no_single_trade_above_pct_of_portfolio(self):
        """Every w_i / Σw should be ≤ max_single_trade_pct.

        With N=4 trades and max_pct=0.30, the LP is feasible: each trade
        can be 30% of portfolio, plus a 10% slack distributed elsewhere.
        """
        result = optimise_leverage(
            trade_carries=[0.05, 0.03, 0.02, 0.01],   # trade 0 is best
            trade_haircuts=[0.05, 0.05, 0.05, 0.05],
            trade_rwa_weights=[0.20, 0.50, 0.50, 0.50],
            capital=100.0,
            max_leverage=10.0,
            max_single_trade_pct=0.30,
        )
        weights = result.optimal_weights
        total = sum(weights)
        assert total > 0  # LP must be feasible.
        for w in weights:
            assert w / total <= 0.30 + 1e-6  # within tolerance

    def test_no_concentration_when_leverage_constrained(self):
        """When leverage cap is the binding constraint, single trade still bounded.

        Feasibility: N=3 trades, max_pct=0.40 ≥ 1/3 = 0.333 → feasible.
        """
        result = optimise_leverage(
            trade_carries=[0.10, 0.005, 0.005],  # one dominant trade
            trade_haircuts=[0.01, 0.01, 0.01],
            trade_rwa_weights=[0.10, 0.10, 0.10],
            capital=100.0,
            max_leverage=5.0,
            max_single_trade_pct=0.40,
        )
        total = sum(result.optimal_weights)
        assert total > 0
        assert max(result.optimal_weights) / total <= 0.40 + 1e-6


class TestSingleTradeRecovery:
    def test_uniform_carries_uniform_weights_under_concentration(self):
        """With equal carries and tight concentration cap, weights are uniform."""
        n = 5
        max_pct = 1.0 / n  # exactly equal weight is the only feasible single-trade-cap
        result = optimise_leverage(
            trade_carries=[0.05] * n,
            trade_haircuts=[0.05] * n,
            trade_rwa_weights=[0.20] * n,
            capital=100.0,
            max_leverage=10.0,
            max_single_trade_pct=max_pct,
        )
        weights = result.optimal_weights
        total = sum(weights)
        # Each weight = total/n.
        for w in weights:
            assert w == pytest.approx(total / n, rel=1e-6)


class TestOptimiserHappyPath:
    def test_solves_and_returns_carry(self):
        result = optimise_leverage(
            trade_carries=[0.05, 0.03],
            trade_haircuts=[0.10, 0.05],
            trade_rwa_weights=[0.20, 0.30],
            capital=100.0,
            max_leverage=5.0,
            max_single_trade_pct=0.50,
        )
        assert result.optimal_carry > 0
        assert result.leverage_ratio > 0
        assert result.n_trades == 2
