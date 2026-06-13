"""Regression for L2 phase-2 audit of `risk.prudent_valuation`:

(a) ``market_price_uncertainty_ava`` had non-monotonic behaviour in
    ``n_quotes``.  Pre-fix: n_quotes=0 hit a special-case fallback
    returning ``half_spread``; n_quotes=1 gave reliability=0.2 →
    AVA = 4.5·half_spread.  Adding the first quote *increased* AVA,
    contradicting the "more quotes = more reliable = smaller AVA" intent.

(b) ``close_out_cost_ava`` silently ignored the caller's ``position_days``
    parameter — overwritten when daily_volume > 0, dropped entirely otherwise.

Fix: floor reliability at 0.1 (drops special-case); honour caller's
``position_days`` even when daily_volume is unknown.
"""

from __future__ import annotations

import math

import pytest

from pricebook.risk.prudent_valuation import (
    close_out_cost_ava, market_price_uncertainty_ava,
)


class TestMPUMonotonicInQuotes:
    def test_more_quotes_means_smaller_ava(self):
        """Increasing n_quotes from 1 to 5 should monotonically decrease AVA."""
        args = dict(mid_price=100.0, bid_price=99.0, ask_price=101.0)
        avas = [
            market_price_uncertainty_ava(**args, n_quotes=k).ava
            for k in range(1, 6)
        ]
        # Monotone strictly non-increasing.
        for i in range(len(avas) - 1):
            assert avas[i] >= avas[i + 1] - 1e-12

    def test_zero_quotes_is_max_ava(self):
        """n_quotes=0 (no quotes) should give the MAXIMUM AVA, not less than n_quotes=1."""
        args = dict(mid_price=100.0, bid_price=99.0, ask_price=101.0)
        zero_q = market_price_uncertainty_ava(**args, n_quotes=0).ava
        one_q = market_price_uncertainty_ava(**args, n_quotes=1).ava
        # Pre-fix: zero_q = 1.0 (half_spread), one_q = 4.5 → bug.
        # Post-fix: zero_q ≥ one_q (monotone).
        assert zero_q >= one_q

    def test_five_quotes_gives_confidence_half_spread(self):
        """At full reliability (5+ quotes), AVA = half_spread × confidence."""
        result = market_price_uncertainty_ava(
            mid_price=100.0, bid_price=99.0, ask_price=101.0,
            n_quotes=5, confidence=0.90,
        )
        # half_spread = 1, reliability = 1, ava = 1·0.9/1 = 0.9.
        assert result.ava == pytest.approx(0.9, abs=1e-12)


class TestCloseOutCostHonorsPositionDays:
    def test_position_days_used_when_no_volume(self):
        """When daily_volume=0, caller's position_days should drive size adjustment."""
        # With position_days=10 (large) → big size adjustment via log(10).
        r_long = close_out_cost_ava(notional=1e6, asset_class="bond_ig",
                                     daily_volume=0.0, position_days=10.0)
        # With position_days=1.0 → no/default size adjustment.
        r_short = close_out_cost_ava(notional=1e6, asset_class="bond_ig",
                                      daily_volume=0.0, position_days=1.0)
        # Larger position_days → larger size_adjustment → larger total AVA.
        assert r_long.ava > r_short.ava

    def test_position_days_overrides_default_premium(self):
        """A large explicit position_days produces a log-based premium > default 50%."""
        r = close_out_cost_ava(notional=1e6, asset_class="bond_ig",
                                 daily_volume=0.0, position_days=20.0)
        # log(20) ≈ 3.0; size_adj_bp ≈ 3.0 × base; > 0.5 × base default.
        # base for bond_ig = 3.0 bp.
        # Pre-fix would have used 0.5 × 3.0 = 1.5 bp; post-fix uses ~9.0 bp.
        expected_size_adj_bp = 3.0 * math.log(20)
        assert r.size_adjustment_bp == pytest.approx(expected_size_adj_bp, rel=1e-9)


class TestRegressionsExistingTests:
    """Ensure pre-fix non-regressed cases still work."""

    def test_daily_volume_branch_unchanged_when_position_days_default(self):
        """With daily_volume>0 and default position_days=1, behaviour matches pre-fix."""
        r = close_out_cost_ava(notional=1e6, asset_class="bond_ig",
                                 daily_volume=5e5, position_days=1.0)
        # notional / daily_volume = 2; max(2, 1, 1) = 2; log(2) ≈ 0.693.
        expected_size_adj_bp = 3.0 * math.log(2)
        assert r.size_adjustment_bp == pytest.approx(expected_size_adj_bp, rel=1e-9)
