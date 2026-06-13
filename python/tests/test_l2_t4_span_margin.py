"""Regression for L2 phase-2 audit of `risk.portfolio_margin.span_margin`:

Pre-fix applied the SPAN 35% extreme-scenario cap to "the last 2
scenarios" regardless of whether those were the auto-built extreme
moves or user-supplied scenarios.  User-supplied custom scenario lists
silently had their last two entries mis-scaled by 0.35.

Fix: only apply the cap when scenarios were auto-built from the
price_scan_range / vol_scan_range parameters.
"""

from __future__ import annotations

import pytest

from pricebook.risk.portfolio_margin import (
    Position, span_margin,
)


class TestUserSuppliedScenariosNoExtremeCap:
    def test_user_scenarios_full_loss(self):
        """With user-supplied scenarios, every loss should be counted at full magnitude."""
        # Single long call position with positive delta.  Down-move = loss.
        positions = [
            Position(instrument_type="equity_option", quantity=1,
                     delta=0.5, gamma=0.0, vega=0.0, notional=100_000),
        ]
        # User passes three scenarios; all of them should be counted in full.
        scenarios = [(-0.10, 0.0), (-0.05, 0.0), (0.10, 0.0)]
        result = span_margin(positions, scenarios=scenarios,
                              price_scan_range=0.10, vol_scan_range=0.30)
        # Worst loss = -PnL of -10% move = +1 · 0.5 · -10000 = -5000 (PnL),
        # loss = +5000.  Pre-fix would have *0.35 = $1750.
        assert result.initial_margin == pytest.approx(5000.0, abs=1e-6)


class TestAutoBuiltGridAppliesCap:
    def test_auto_built_extreme_cap_active(self):
        """Auto-built grid: the last 2 (extreme) scenarios still get the 35% cap."""
        # Single long position; the worst loss in the auto-built grid will be
        # the -1.0 × PSR scenario (full PSR), NOT the -2.0 × PSR extreme
        # which is 35%-capped.
        positions = [
            Position(instrument_type="equity_option", quantity=1,
                     delta=0.5, gamma=0.0, vega=0.0, notional=100_000),
        ]
        result = span_margin(positions, scenarios=None,
                              price_scan_range=0.10, vol_scan_range=0.30)
        # -1.0 × PSR scenario: loss = 1 × 0.5 × -10000 = -5000 → loss=5000.
        # -2.0 × PSR scenario (capped): loss = 1 × 0.5 × -20000 = -10000 → loss=10000 × 0.35 = 3500.
        # Worst = 5000 (regular scenario beats capped extreme).
        assert result.initial_margin == pytest.approx(5000.0, abs=1e-6)


class TestVegaScenariosNotCapped:
    """User scenarios with vega-only moves shouldn't be 35%-capped at position N-1 or N-2."""

    def test_vega_scenarios_full_count(self):
        positions = [
            Position(instrument_type="equity_option", quantity=1,
                     delta=0.0, gamma=0.0, vega=100.0, notional=100_000),
        ]
        # 4 scenarios; last 2 (idx 2, 3) are big vol moves; pre-fix would cap them.
        scenarios = [(0.0, -0.10), (0.0, -0.20), (0.0, -0.30), (0.0, -0.40)]
        result = span_margin(positions, scenarios=scenarios,
                              price_scan_range=0.10, vol_scan_range=0.30)
        # Worst loss = vega · (-0.40) × -1 = +40 (PnL), loss = -40 (profit)?
        # Vol going DOWN with long vega: -40 PnL → loss = 40.
        # Last scenario: vega_pnl = 1 × 100 × -0.40 = -40.  loss = 40.
        # Pre-fix: 40 × 0.35 = 14.  Post-fix: 40.
        assert result.initial_margin == pytest.approx(40.0, abs=1e-6)
