"""Regression for L2 T4 audit of `structured.cat_bond.cat_bond_price`:

Pre-fix the coupon-PV loop used ``range(1, int(T) + 1)`` which silently
dropped any non-integer remainder of ``T``:

- T = 0.5 (6-month bond): loop is empty → zero coupon PV.
- T = 3.5: 3 full annual coupons but the 0.5-year final accrual was missed.

Fix: add a fractional final accrual at time T when T has a non-integer
remainder.
"""

from __future__ import annotations

import math

import pytest

from pricebook.structured.cat_bond import cat_bond_price


class TestFractionalTerm:
    def test_six_month_bond_has_nonzero_coupons(self):
        """T=0.5: pre-fix returned zero coupon PV; post-fix gives 0.5y of accrual."""
        result = cat_bond_price(
            notional=1000.0, coupon_spread=0.05,
            risk_free_rate=0.03, expected_loss=0.02, T=0.5,
        )
        # Coupon = 0.03 + 0.05 = 0.08. Half-year accrual = 0.04 per 1 notional.
        # PV of coupon ≈ 0.04 × df(0.5) × survival(0.5) × 1000 ≈ 0.04 × ~0.985 × ~0.99 × 1000 ≈ 39.
        # Pre-fix gave 0 from the loop, so price was based only on PV(principal) ≈ 985.
        # Post-fix price > pre-fix price by ~39.
        assert result.price > 100.0  # must include some coupon

    def test_fractional_term_continuous_in_T(self):
        """Price should be continuous in T at integer boundaries."""
        # Compare T just below 3 to T just above 3.
        r_below = cat_bond_price(1000.0, 0.05, 0.03, 0.02, T=2.99)
        r_at = cat_bond_price(1000.0, 0.05, 0.03, 0.02, T=3.00)
        r_above = cat_bond_price(1000.0, 0.05, 0.03, 0.02, T=3.01)
        # Prices should be close (within ~0.5 per 100).
        assert abs(r_at.price - r_below.price) < 0.5
        assert abs(r_above.price - r_at.price) < 0.5

    def test_integer_T_unchanged(self):
        """Pre-fix integer-T behaviour preserved (no remainder, no new coupon)."""
        # Recompute the integer-T case manually using the same formula but
        # excluding the new fractional branch (T - int(T) = 0 → no extra).
        # For T=3, the new code adds 0 extra coupons → identical to pre-fix.
        result = cat_bond_price(1000.0, 0.05, 0.03, 0.02, T=3.0)
        # Expected coupons: t=1, 2, 3.  Verify price is in a reasonable range
        # (mostly the integer-T value).
        coupon = 0.03 + 0.05
        expected_coupon_pv = sum(
            coupon * 1000.0 * math.exp(-0.03 * t) * math.exp(-0.02 * t)
            for t in (1, 2, 3)
        )
        prob_loss = 1.0 - math.exp(-0.02 * 3.0)
        df_T = math.exp(-0.03 * 3.0)
        expected_principal = 1000.0 * df_T * (1.0 - prob_loss)
        expected_price = 100.0 * (expected_coupon_pv + expected_principal) / 1000.0
        assert result.price == pytest.approx(expected_price, rel=1e-9)
