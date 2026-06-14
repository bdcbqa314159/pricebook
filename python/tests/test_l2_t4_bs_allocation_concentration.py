"""Regression for L2 T4 audit of `regulatory.balance_sheet_allocation.optimise_allocation`:

Pre-fix the ``max_single_trade_pct`` parameter was declared in the
signature but the body never referenced it — every call ignored the
concentration limit and any single trade could absorb the entire
capital budget.

Fix: collapse the concentration limit into the per-trade upper bound:
    rwa_i × w_i × 0.08 ≤ max_single_trade_pct × total_capital
"""

from __future__ import annotations

import pytest

from pricebook.regulatory.balance_sheet_allocation import optimise_allocation


class TestConcentrationLimit:
    def _trades(self):
        # Three trades with identical RWA per dollar (10% RW); identical
        # carry minus xva so the LP is indifferent across them — pre-fix
        # the LP would pile into a single trade up to max_notional.
        return [
            {"trade_id": "A", "carry": 100.0, "xva_cost": 0.0,
             "rwa": 100.0, "max_notional": 1_000_000.0},
            {"trade_id": "B", "carry": 100.0, "xva_cost": 0.0,
             "rwa": 100.0, "max_notional": 1_000_000.0},
            {"trade_id": "C", "carry": 100.0, "xva_cost": 0.0,
             "rwa": 100.0, "max_notional": 1_000_000.0},
        ]

    def test_no_trade_exceeds_concentration_cap(self):
        """With max_single_trade_pct=0.25, no trade's capital usage may
        exceed 25% of total_capital."""
        trades = self._trades()
        cap = 1000.0
        pct = 0.25
        r = optimise_allocation(trades, total_capital=cap, max_single_trade_pct=pct)
        for tid, w in r.allocations.items():
            trade = next(t for t in trades if t["trade_id"] == tid)
            capital_used = trade["rwa"] * w * 0.08
            assert capital_used <= pct * cap + 1e-6, (
                f"trade {tid} uses {capital_used} > {pct * cap}"
            )

    def test_concentration_forces_diversification(self):
        """With tight concentration, must select multiple trades."""
        trades = self._trades()
        # Total capital allows ~125 RWA worth (1000 / 0.08), but each
        # trade is capped to 0.20 × 1000 = 200 of capital → 2500 notional
        # at 10% RW.  So one trade can absorb up to 200 cap; full capital
        # budget requires at least 5 trades.  We only have 3 → all 3
        # selected at their max.
        r = optimise_allocation(trades, total_capital=1000.0, max_single_trade_pct=0.20)
        assert r.n_trades_selected >= 2

    def test_lower_pct_more_diversification(self):
        """Tighter concentration → at least as many trades."""
        trades = self._trades()
        r_loose = optimise_allocation(trades, total_capital=1000.0,
                                       max_single_trade_pct=1.0)
        r_tight = optimise_allocation(trades, total_capital=1000.0,
                                       max_single_trade_pct=0.20)
        assert r_tight.n_trades_selected >= r_loose.n_trades_selected

    def test_default_pct_does_not_break_existing_calls(self):
        """Default max_single_trade_pct=0.25 — must still produce a
        well-formed result for the standard test_optimise_allocation
        regression."""
        trades = [
            {"trade_id": f"T{i}", "carry": 100.0 * (i + 1), "xva_cost": 10.0,
             "rwa": 1000.0, "max_notional": 100_000.0}
            for i in range(5)
        ]
        r = optimise_allocation(trades, total_capital=50_000.0)
        assert r.total_capital_used <= 50_000.0 + 1e-3
        assert r.n_trades_selected > 0
