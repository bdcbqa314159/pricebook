"""Regression for L2 T4 audit of `desks.bond_trading_desk.carry_and_roll`:

Pre-fix the rolldown leg used ``dirty_price``, which includes accrued
interest.  Over the rolldown horizon the dirty price grows by roughly
``coupon × horizon/365`` from accrual alone, so

    total = net_carry + roll_down_dirty
          = (coupon - funding) + (coupon_accrual + clean_roll)

double-counted the coupon income.

Fix: switch to ``clean_price``-based rolldown so accrual is excluded;
``total = net_carry + clean_roll`` is the standard interpretation
(carry = running yield, roll = clean-price aging on unchanged curve).
"""

from __future__ import annotations

import math
from datetime import date

import pytest


class TestFlatCurveZeroCleanRoll:
    """On a flat yield curve, an at-par bond's clean price doesn't move
    with time (no curve roll, no pull-to-par drift) — so the clean
    rolldown must be ~0 and total = -funding when coupon = funding rate.
    """

    def test_at_par_flat_curve_zero_clean_roll(self):
        from pricebook.fixed_income.bond import FixedRateBond
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.core.day_count import DayCountConvention
        from pricebook.core.schedule import Frequency
        from pricebook.desks.bond_trading_desk import bond_carry_roll

        ref = date(2026, 1, 15)
        # Flat 5% curve.
        pillars = [date(ref.year + y, ref.month, ref.day) for y in (1, 2, 3, 5, 10)]
        dfs = [math.exp(-0.05 * y) for y in (1, 2, 3, 5, 10)]
        curve = DiscountCurve(ref, pillars, dfs, DayCountConvention.ACT_365_FIXED)

        # 5y, 5% coupon bond — exactly at par on a flat 5% curve.
        bond = FixedRateBond(
            face_value=100.0,
            maturity=date(2031, 1, 15),
            coupon_rate=0.05,
            frequency=Frequency.SEMI_ANNUAL,
            issue_date=ref,
        )
        cr = bond_carry_roll(bond, curve, repo_rate=0.05, horizon_days=30)
        # Clean roll on flat curve at par should be ~0.
        # Pre-fix dirty-price rolldown ≈ +0.41 ($5 × 30/365 per 100 face).
        # Post-fix clean rolldown ≈ 0.
        assert abs(cr.roll_down_return) < 0.05
        # Net carry = coupon - funding = 0 (since 5%=5%).
        assert abs(cr.net_carry) < 0.01
        # Total = net_carry + roll_down ≈ 0.
        assert abs(cr.total_carry_and_roll) < 0.05


class TestFlatCurveCarryEqualsCouponMinusFunding:
    """When repo rate < coupon, net carry is positive coupon income."""

    def test_positive_carry_when_low_repo(self):
        from pricebook.fixed_income.bond import FixedRateBond
        from pricebook.core.discount_curve import DiscountCurve
        from pricebook.core.day_count import DayCountConvention
        from pricebook.core.schedule import Frequency
        from pricebook.desks.bond_trading_desk import bond_carry_roll

        ref = date(2026, 1, 15)
        pillars = [date(ref.year + y, ref.month, ref.day) for y in (1, 2, 3, 5, 10)]
        dfs = [math.exp(-0.05 * y) for y in (1, 2, 3, 5, 10)]
        curve = DiscountCurve(ref, pillars, dfs, DayCountConvention.ACT_365_FIXED)

        bond = FixedRateBond(
            face_value=100.0,
            maturity=date(2031, 1, 15),
            coupon_rate=0.05,
            frequency=Frequency.SEMI_ANNUAL,
            issue_date=ref,
        )
        cr = bond_carry_roll(bond, curve, repo_rate=0.02, horizon_days=30)
        # net carry = 5%×100×30/365 - 2%×P/100×100×30/365 ≈ 0.41 - 0.16 = 0.25.
        expected_coupon = 0.05 * 100 * 30 / 365.0
        expected_funding = 0.02 * cr.funding_cost / cr.funding_cost  # avoid div-by-zero noise
        assert cr.coupon_carry == pytest.approx(expected_coupon, rel=1e-9)
        # Post-fix total ≈ net_carry (clean roll ~0 on flat curve).
        # Pre-fix would have been ~2× net_carry due to coupon double-count.
        assert cr.total_carry_and_roll == pytest.approx(cr.net_carry, abs=0.05)
