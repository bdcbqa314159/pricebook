"""Regression for L2 T4 audit of `desks.swap_desk` theta computation.

Pre-fix the theta lambda for rolldown used a no-op ternary
``proj if proj is curve else proj``: both branches returned ``proj``.
Under single-curve setup this meant the rolled discount curve was
used for discounting but the original-t=0 curve was used for the
floating leg's projection — so the floating forward rates were stale
(1 day behind the discount) and theta picked up an inconsistent
discount/projection split.

Fix: under single-curve, pass ``None`` to ``swap.pv`` so the floating
leg uses the rolled discount curve for projection too.  Under
dual-curve, pre-roll the projection by the same day count.

We verify by comparing theta for two equivalent pricings of an at-par
swap: the corrected theta should match an explicit consistent-roll
pricing exactly.
"""

from __future__ import annotations

from datetime import date

import pytest

import math

from pricebook.core.day_count import DayCountConvention
from pricebook.core.discount_curve import DiscountCurve
from pricebook.fixed_income.swap import InterestRateSwap, SwapDirection
from pricebook.desks.swap_desk import swap_risk_metrics, swap_daily_pnl


def _build_curve(ref: date) -> DiscountCurve:
    """Flat 5% USD curve out to 10 years."""
    pillars = [date(ref.year + y, ref.month, ref.day) for y in (1, 2, 3, 5, 7, 10)]
    rate = 0.05
    dfs = [math.exp(-rate * y) for y in (1, 2, 3, 5, 7, 10)]
    return DiscountCurve(ref, pillars, dfs, DayCountConvention.ACT_365_FIXED)


def _build_swap(start: date) -> InterestRateSwap:
    """Standard 5y receive-fixed swap."""
    end = date(start.year + 5, start.month, start.day)
    return InterestRateSwap(
        start=start, end=end,
        fixed_rate=0.05,
        direction=SwapDirection.RECEIVER,
        notional=10_000_000.0,
    )


class TestThetaConsistentRoll:
    """Single-curve theta must equal the consistent roll PV diff."""

    def test_single_curve_theta_matches_consistent_roll(self):
        ref = date(2026, 1, 15)
        curve = _build_curve(ref)
        swap = _build_swap(date(2026, 1, 15))

        rm = swap_risk_metrics(swap, curve)
        # Independently compute consistent-roll theta: roll curve, use it
        # for BOTH discount and projection.
        rolled = curve.roll_down(1)
        pv_rolled = swap.pv(rolled, None)
        pv_base = swap.pv(curve, None)
        consistent_theta = pv_rolled - pv_base

        assert rm.theta == pytest.approx(consistent_theta, rel=1e-9, abs=1e-6)

    def test_daily_pnl_carry_plus_theta_well_defined(self):
        """swap_daily_pnl should produce finite theta under single-curve."""
        ref0 = date(2026, 1, 15)
        ref1 = date(2026, 1, 16)
        curve_t0 = _build_curve(ref0)
        curve_t1 = _build_curve(ref1)
        swap = _build_swap(date(2026, 1, 15))

        r = swap_daily_pnl(swap, curve_t0, curve_t1, ref1)
        # Theta should be finite (not NaN/inf), and the consistent-roll
        # value matches.
        assert r.theta == r.theta  # not NaN
        rolled = curve_t0.roll_down(1)
        expected = swap.pv(rolled, None) - swap.pv(curve_t0, None)
        assert r.theta == pytest.approx(expected, rel=1e-9, abs=1e-6)
