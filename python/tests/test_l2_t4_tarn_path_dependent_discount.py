"""Regression for L2 T4 audit of `options.ir_exotic.tarn_price`:

Pre-fix the function simulated a short-rate path via OU dynamics but
the simulated ``r`` was never referenced in the payoff OR the
discount factor — every cashflow was discounted by
``exp(-flat_rate · t)`` regardless of the path.  Consequence:
``rate_vol`` was a silent-no-op API param (changing it gave
bit-identical prices, because the simulated path was dead code).

Fix: introduce a per-path cumulative ``log_df`` tracking
``-∫_0^t r_s ds`` (left-Riemann integration of the simulated short
rate) and use ``exp(log_df)`` as the discount factor for every
cashflow.  ``rate_vol`` now drives the convexity correction.
"""

from __future__ import annotations

import math

import pytest

from pricebook.options.ir_exotic import tarn_price


class TestRateVolMatters:
    def test_higher_rate_vol_changes_price(self):
        """Pre-fix: identical price at any rate_vol (silent no-op).
        Post-fix: higher rate_vol → different price (Jensen / convexity
        through the stochastic discount).
        """
        low_vol = tarn_price(
            notional=100, coupon_rate=0.05, target=0.10,
            maturity_years=5, flat_rate=0.05,
            rate_vol=0.001, n_paths=20_000, seed=42,
        )
        high_vol = tarn_price(
            notional=100, coupon_rate=0.05, target=0.10,
            maturity_years=5, flat_rate=0.05,
            rate_vol=0.05, n_paths=20_000, seed=42,
        )
        # Pre-fix these were exactly equal.  Post-fix they differ
        # materially (convexity through stochastic discount).
        assert abs(high_vol.price - low_vol.price) > 0.01, (
            f"low_vol={low_vol.price:.6f}, high_vol={high_vol.price:.6f} "
            f"— rate_vol must affect price under path-dependent discount"
        )

    def test_zero_rate_vol_matches_deterministic(self):
        """With rate_vol = 0 and OU drift ≡ 0 (start at flat_rate), the
        path is deterministic at ``flat_rate`` and the price should
        match the pre-fix deterministic baseline closely."""
        result = tarn_price(
            notional=100, coupon_rate=0.05, target=0.10,
            maturity_years=5, flat_rate=0.05,
            rate_vol=0.0, n_paths=5_000, seed=42,
        )
        # Deterministic case: target hit at t = 2y, then pay par.
        # Approx PV = sum_{i=1}^{8} 0.05/4 × exp(-0.05 × i/4) + exp(-0.05 × 2) × 1
        # × 100 to normalize.
        coupon_per = 0.05 / 4
        pv = sum(coupon_per * math.exp(-0.05 * (i / 4))
                 for i in range(1, 9))  # 2y = 8 quarterly periods
        pv += math.exp(-0.05 * 2.0) * 1.0
        expected = pv * 100
        assert result.price == pytest.approx(expected, rel=2e-2)


class TestPositiveAndFinite:
    def test_finite_under_nonzero_vol(self):
        result = tarn_price(
            notional=100, coupon_rate=0.05, target=0.20,
            maturity_years=5, flat_rate=0.04,
            rate_vol=0.02, n_paths=20_000, seed=42,
        )
        assert math.isfinite(result.price)
        assert result.price > 0
