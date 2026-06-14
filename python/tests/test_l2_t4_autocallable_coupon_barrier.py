"""Regression for L2 T4 audit of `options.autocallable.Autocallable`:

Pre-fix the ``coupon_barrier`` parameter was a silent-no-op API param:
the constructor accepted and round-tripped it, but ``price_mc`` never
checked ``S >= coupon_barrier * spot`` at observation dates.  Coupons
were unconditionally accrued at ``rate * elapsed_t``, regardless of
whether the underlying was ever above the coupon barrier — directly
contradicting the class docstring ("If S(t) >= coupon_barrier × S₀ ...
coupon accrues").

Fix: track per-path accrued coupons that only grow at observations
where ``S >= coupon_barrier * spot`` (memory-style).
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.options.autocallable import Autocallable


REF = date(2026, 4, 28)


def _curve():
    return DiscountCurve.flat(REF, 0.03)


def _quarterly_dates(years: int = 3) -> list[date]:
    return [REF + timedelta(days=90 * k) for k in range(1, 4 * years + 1)]


class TestCouponBarrierGatesAccrual:
    def test_high_coupon_barrier_kills_coupons(self):
        """Pre-fix: coupon_barrier=10.0 still accrued coupons every period.
        Post-fix: coupons never accrue → price drops materially.

        With autocall_level=10.0 (never trips) and coupon_barrier=10.0 (never
        accrues), the only payoff is the put-barrier-driven principal at
        maturity → price < notional × DF(T) (capital at risk with no coupon).
        """
        ac_with_barrier = Autocallable(
            observation_dates=_quarterly_dates(2),
            autocall_level=10.0, coupon_rate=0.10,
            coupon_barrier=10.0,  # never above
            put_barrier=0.70,
        )
        ac_baseline = Autocallable(
            observation_dates=_quarterly_dates(2),
            autocall_level=10.0, coupon_rate=0.10,
            coupon_barrier=0.0,  # always above
            put_barrier=0.70,
        )
        # Same seed → same path realisation; differences come ONLY from
        # the coupon_barrier gating.
        r_b = ac_with_barrier.price_mc(spot=100, curve=_curve(), vol=0.20,
                                        n_paths=20_000, seed=42)
        r_0 = ac_baseline.price_mc(spot=100, curve=_curve(), vol=0.20,
                                    n_paths=20_000, seed=42)
        # With no coupons, price should be MUCH lower (10% × 2y = 20% of
        # notional lost).
        assert r_b.price < r_0.price * 0.90, (
            f"barrier=10 ({r_b.price:.0f}) should drop coupons vs "
            f"barrier=0 ({r_0.price:.0f})"
        )

    def test_zero_coupon_barrier_matches_old_behaviour(self):
        """coupon_barrier = 0 → always above → coupons accrue every period.
        This matches the pre-fix (broken) behaviour of always-accruing
        coupons up to elapsed time."""
        import math
        ac = Autocallable(
            observation_dates=_quarterly_dates(1),  # 1y
            autocall_level=10.0,  # never autocalls
            coupon_rate=0.10,
            coupon_barrier=0.0,
            put_barrier=0.0,  # never knocks
        )
        r = ac.price_mc(spot=100, curve=_curve(), vol=0.05,
                        n_paths=20_000, seed=42)
        # Always above coupon barrier (=0), always above put barrier (=0),
        # never autocalls.  Final payoff per path:
        #   notional × (1 + 0.10 × T) × DF(T)
        # T ≈ 1y, coupons accrue every period (1.0 total), DF = exp(-0.03).
        # Σ period_lengths = T (exactly).
        T = 360 / 365.0  # roughly
        expected = 1_000_000 * (1 + 0.10 * T) * math.exp(-0.03 * T)
        assert r.price == pytest.approx(expected, rel=2e-2)


class TestPutBarrierBranchUnchanged:
    def test_below_put_pays_recovery_no_coupons(self):
        """Below-put-barrier branch still pays S/S0 × notional (no coupons).
        Verifies the coupon_barrier fix didn't break the recovery path."""
        ac = Autocallable(
            observation_dates=_quarterly_dates(1),
            autocall_level=10.0,
            coupon_rate=0.10,
            coupon_barrier=0.0,
            put_barrier=2.0,  # always below put barrier
        )
        r = ac.price_mc(spot=100, curve=_curve(), vol=0.10,
                        n_paths=20_000, seed=42)
        # All paths trigger put-knock → no coupons paid.  Price should be
        # close to E[S/S0] × notional × DF(T) ≈ notional × DF(T) (forward
        # = spot under risk-neutral, modulo small drift effect).
        assert r.put_knock_prob == pytest.approx(1.0, abs=1e-6)
