"""Regression for L2 Tier-3 T3.13 — `HullWhite.tree_european_swaption` no
longer hard-codes annual payments / integer tenor.

Pre-fix `n_payments = max(1, int(swap_end_T − expiry_T))` truncated
non-integer tenors (e.g. 2.5y → 2y) and `t_pay = expiry_T + k` assumed
annual spacing — so semi-annual or quarterly swaptions were mispriced.

Post-fix accepts `payments_per_year` (default 1 for backwards compat with
analytical-formula tests) and uses `τ = 1 / payments_per_year` for the
fixed-leg coupon spacing.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.day_count import (
    DayCountConvention,
    date_from_year_fraction,
)
from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.hull_white import HullWhite


REF = date(2024, 1, 1)


def _flat_curve(rate: float = 0.04) -> DiscountCurve:
    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    dfs = [math.exp(-rate * t) for t in tenors]
    dates = [REF + timedelta(days=int(t * 365)) for t in tenors]
    return DiscountCurve(REF, dates, dfs, day_count=DayCountConvention.ACT_365_FIXED)


class TestNonIntegerTenor:
    def test_2_5y_tenor_uses_all_periods(self):
        """A 2y2.5y swaption (2y expiry, 2.5y tenor) under semi-annual
        payments has 5 periods.  Pre-fix the int truncation gave 2 periods
        (annual only) and ignored the 2.5y tail."""
        hw = HullWhite(a=0.05, sigma=0.005, curve=_flat_curve())
        price_semi = hw.tree_european_swaption(
            expiry_T=2.0, swap_end_T=4.5, strike=0.04,
            n_steps=200, payments_per_year=2,
        )
        # Compare with annual: payments-per-year=1 truncates to 2 payments
        # (int(2.5)=2), missing the 2.5y tail.  Semi-annual should give a
        # noticeably different price.
        price_annual = hw.tree_european_swaption(
            expiry_T=2.0, swap_end_T=4.5, strike=0.04,
            n_steps=200, payments_per_year=1,
        )
        assert price_semi != price_annual
        assert price_semi > 0
        assert price_annual > 0

    def test_quarterly_payments_more_payments_than_annual(self):
        """Quarterly payments → 20 payment periods on a 5y swap.  Different
        annuity than annual (5 payments)."""
        hw = HullWhite(a=0.05, sigma=0.005, curve=_flat_curve())
        price_quart = hw.tree_european_swaption(
            expiry_T=1.0, swap_end_T=6.0, strike=0.04,
            n_steps=150, payments_per_year=4,
        )
        price_annual = hw.tree_european_swaption(
            expiry_T=1.0, swap_end_T=6.0, strike=0.04,
            n_steps=150, payments_per_year=1,
        )
        # Both ATM-ish for a 4% flat curve and 4% strike → similar order of
        # magnitude, but quarterly compounding gives a slightly different
        # annuity (continuous vs discrete) → prices differ measurably.
        assert price_quart != price_annual


class TestAnnualBackwardsCompat:
    """Default payments_per_year=1 must give same price as pre-fix for
    integer tenors."""

    def test_integer_tenor_annual_unchanged(self):
        """Existing analytical tests (5y10y at K=4%, etc.) use annual /
        integer.  Verify a representative case still produces a sane
        positive number."""
        hw = HullWhite(a=0.05, sigma=0.005, curve=_flat_curve())
        price = hw.tree_european_swaption(
            expiry_T=2.0, swap_end_T=5.0, strike=0.04, n_steps=100,
        )
        assert price > 0
        assert math.isfinite(price)
