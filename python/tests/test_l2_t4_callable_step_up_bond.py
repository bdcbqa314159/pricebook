"""Regression for L2 T4 audit of `structured.rates_structured.callable_step_up_bond`:

Pre-fix the bond_value used in the issuer call decision was over-
discounted by one period.  At call iteration ``i`` (call time
``(i+1)*dt``), remaining coupon ``j`` is paid at ``(j+1)*dt`` so the
discount horizon is ``(j-i)*dt`` — the pre-fix code used
``(j+1-i)*dt``.  Principal had the same off-by-one
(``(n-i)*dt`` vs correct ``(n-i-1)*dt``).

Effect: bond_value is biased LOW by `exp(-r·dt)`, so the issuer
calls LESS often than they should.  The callable bond was therefore
slightly OVER-priced (less call benefit captured).

Fix: shift the discount horizons by one period to match the actual
call date.
"""

from __future__ import annotations

import pytest

from pricebook.structured.rates_structured import callable_step_up_bond


class TestCallableBondCallsCorrectly:
    def test_call_value_increases_after_fix(self):
        """The call value (non_callable - callable) should be POSITIVE for an
        ITM call.  Use a deep-ITM step-up bond: rates very low → bond worth
        much more than par → issuer SHOULD call almost immediately."""
        coupons = [10.0] * 5  # 10% coupons (percent form) in a low-rate world
        r = callable_step_up_bond(
            face=100.0, coupon_schedule=coupons,
            rate=0.01, vol=0.005, T=5.0,
            n_paths=5_000, seed=42,
        )
        # Non-callable value should be high (high coupon vs low rate).
        assert r.non_callable_price > 130.0
        # Callable value should be lower (call benefit accrues to issuer).
        assert r.price < r.non_callable_price
        # Call value should be substantial: at least 5% of face.
        assert r.call_value > 5.0

    def test_no_explosion_at_short_T(self):
        """At T close to a single dt, the fix shouldn't blow up."""
        coupons = [5.0, 6.0, 7.0]
        r = callable_step_up_bond(
            face=100.0, coupon_schedule=coupons,
            rate=0.04, vol=0.005, T=3.0,
            n_paths=3_000, seed=42,
        )
        assert r.price > 0
        assert r.non_callable_price > 0
        # Average call time should be in [0, T].
        assert 0 <= r.expected_call_time <= 3.0
