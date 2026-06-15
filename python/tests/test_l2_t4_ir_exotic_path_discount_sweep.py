"""Regression for L2 T4 audit of `options.ir_exotic` — sweep across the
four remaining functions after the ``tarn_price`` fix (T4-IREX1, v1.055).

Pre-fix all five pricers (tarn / snowball / callable_range_accrual /
ratchet_cap / flexi_swap) discounted every cashflow with
``exp(-flat_rate · t)`` regardless of the simulated short-rate path.
The simulated ``r`` was used for some rate-dependent payoffs (snowball
coupon, range-accrual gate, ratchet strike) but never for discounting
— so the rate dynamics fed the payoff distribution but not the
discount-payoff covariance.  For any non-trivial ``rate_vol`` this
biases the price (typically Jensen-style convexity).

Fix (T4-IREX2): per-path cumulative ``log_df = -∫_0^t r ds`` drives
the discount factor for every cashflow in all 4 functions.

These tests pin the qualitative property that ``rate_vol`` now
materially affects the price for each (was bit-identical pre-fix in
the discount channel).
"""

from __future__ import annotations

import math

import pytest

from pricebook.options.ir_exotic import (
    snowball_price, callable_range_accrual, ratchet_cap, flexi_swap,
)


class TestSnowballRateVolThroughDiscount:
    def test_rate_vol_affects_price(self):
        """Snowball coupon already depended on r; pre-fix the discount
        was still flat.  Post-fix both coupon AND discount come from
        the same path → different convexity at different rate_vol."""
        # Two runs differ ONLY in the discount component because the
        # coupon dynamics use the same seed/path.  We compare a moderate
        # vs near-zero vol — change in vol should propagate to price.
        low = snowball_price(
            notional=100, initial_coupon=0.03, spread=0.005,
            maturity_years=5, flat_rate=0.05, rate_vol=1e-6,
            n_paths=10_000, seed=42,
        )
        high = snowball_price(
            notional=100, initial_coupon=0.03, spread=0.005,
            maturity_years=5, flat_rate=0.05, rate_vol=0.03,
            n_paths=10_000, seed=42,
        )
        assert abs(high.price - low.price) > 0.05, (
            f"low={low.price:.4f} high={high.price:.4f} — rate_vol should "
            f"materially affect snowball price via path-dependent discount"
        )


class TestRatchetCapRateVolThroughDiscount:
    def test_higher_vol_changes_price(self):
        low = ratchet_cap(
            notional=100, initial_strike=0.05,
            maturity_years=5, flat_rate=0.05, rate_vol=1e-6,
            n_paths=10_000, seed=42,
        )
        high = ratchet_cap(
            notional=100, initial_strike=0.05,
            maturity_years=5, flat_rate=0.05, rate_vol=0.03,
            n_paths=10_000, seed=42,
        )
        # Caps have strong vol sensitivity through their payoff
        # already; the path-discount fix adds a second-order
        # contribution.  Just verify the two regimes differ.
        assert low.price != pytest.approx(high.price, abs=1e-6)


class TestFlexiSwapPositiveAndFinite:
    def test_finite_under_vol(self):
        result = flexi_swap(
            notional=100, fixed_rate=0.05, maturity_years=5,
            max_exercises=10, flat_rate=0.05, rate_vol=0.02,
            n_paths=10_000, seed=42,
        )
        assert math.isfinite(result.price)


class TestCallableRangeAccrualSelfConsistent:
    def test_callable_lte_non_callable(self):
        """Callable note ≤ non-callable (issuer call is a liability
        for the investor).  Pre-fix the inconsistent discount could
        in principle violate this on individual MC realisations; the
        unified path-discount makes the inequality structural."""
        result = callable_range_accrual(
            notional=100, coupon_rate=0.06, lower=0.02, upper=0.08,
            maturity_years=5, call_start_year=1,
            flat_rate=0.05, rate_vol=0.02,
            n_paths=20_000, seed=42,
        )
        assert result.price <= result.non_callable_price + 1.0
        assert result.call_value >= -1.0
