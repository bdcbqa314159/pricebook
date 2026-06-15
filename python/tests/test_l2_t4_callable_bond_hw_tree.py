"""Regression for L2 T4 audit of `fixed_income.callable_bond._trinomial_backward`:

Same defect class as ``bermudan_swaption`` (T4-BERM1) rolled forward
to the callable / puttable bond HW trinomial tree (T4-CB1):

1. **Wrong trinomial probabilities** — drift term used ``/6`` instead
   of textbook ``/2``.
2. **Missing α(t) shift** — node short rate was ``r0 + j·dr`` for
   every step; tree didn't reprice the input discount curve.
3. **Coupon & option applied after discount** — coupon added without
   the one-step discount factor, option compared discounted
   continuation to undiscounted exercise price.
4. **Terminal coupon double-counted** — init had it, loop added it
   again at step+1 = n_steps.

Fix: ``HullWhite.build_tree_alphas``; textbook ``/2`` probabilities;
apply coupon and option BEFORE the backward discount; skip the
+coupon at the terminal step.

Sanity:
- Straight bond (no calls) priced via the option-tree machinery
  should match ``DiscountCurve``-based PV (no option ⇒ deterministic).
- Callable bond ≤ straight bond (issuer's call is a liability for
  the investor).
- Puttable bond ≥ straight bond.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention
from pricebook.models.hull_white import HullWhite
from pricebook.fixed_income.callable_bond import (
    callable_bond_price, puttable_bond_price, _straight_bond_hw,
)


REF = date(2026, 1, 15)


@pytest.fixture
def flat_curve():
    return DiscountCurve.flat(REF, 0.05)


@pytest.fixture
def hw_flat(flat_curve):
    return HullWhite(a=0.1, sigma=0.01, curve=flat_curve)


def _sloped_curve():
    """Upward-sloping curve."""
    dates = [REF + timedelta(days=d) for d in [30, 365, 1825, 3650]]
    rates = [0.02, 0.03, 0.045, 0.06]
    dfs = [math.exp(-r * ((d - REF).days / 365.0))
           for r, d in zip(rates, dates)]
    return DiscountCurve(
        reference_date=REF, dates=dates, dfs=dfs,
        day_count=DayCountConvention.ACT_365_FIXED,
    )


class TestCallableLeqStraight:
    def test_callable_below_straight_flat(self, hw_flat):
        """Callable bond must price at or below the straight bond
        (issuer's call right is a liability for the investor)."""
        straight = _straight_bond_hw(
            hw_flat, coupon_rate=0.06, maturity_years=10.0,
            n_steps=80, notional=100.0, coupon_frequency=1.0,
        )
        callable_p = callable_bond_price(
            hw_flat, coupon_rate=0.06, maturity_years=10.0,
            call_dates_years=[5, 6, 7, 8, 9],
            call_price=100.0,
            n_steps=80, notional=100.0, coupon_frequency=1.0,
        )
        # Allow MC noise / tree discretisation slack.
        assert callable_p <= straight + 0.5, (
            f"callable={callable_p:.4f}, straight={straight:.4f}"
        )


class TestPuttableGeqStraight:
    def test_puttable_above_straight_flat(self, hw_flat):
        """Puttable bond must price at or above the straight bond."""
        straight = _straight_bond_hw(
            hw_flat, coupon_rate=0.04, maturity_years=10.0,
            n_steps=80, notional=100.0, coupon_frequency=1.0,
        )
        puttable_p = puttable_bond_price(
            hw_flat, coupon_rate=0.04, maturity_years=10.0,
            put_dates_years=[5, 6, 7, 8, 9],
            put_price=100.0,
            n_steps=80, notional=100.0, coupon_frequency=1.0,
        )
        assert puttable_p >= straight - 0.5, (
            f"puttable={puttable_p:.4f}, straight={straight:.4f}"
        )


class TestSlopedCurveProducesFinitePrice:
    def test_callable_finite_on_sloped_curve(self):
        """On a non-flat curve the previous α(t) = r0 mismatch would
        have made the tree's drift inconsistent with the curve.
        Post-fix the tree calibrates to the curve via
        ``build_tree_alphas`` and produces a finite, positive price."""
        curve = _sloped_curve()
        hw = HullWhite(a=0.1, sigma=0.01, curve=curve)
        p = callable_bond_price(
            hw, coupon_rate=0.05, maturity_years=10.0,
            call_dates_years=[3, 5, 7, 9],
            call_price=100.0,
            n_steps=80, notional=100.0, coupon_frequency=1.0,
        )
        assert math.isfinite(p)
        assert 50 < p < 150   # sane bond price


class TestCouponDiscountedProperly:
    def test_high_coupon_short_maturity_matches_curve_pv(self, hw_flat):
        """Zero call dates (no optionality) → the tree price equals the
        straight bond PV from the curve, to discretization.  Pre-fix
        the post-discount coupon-add over-counted coupons by exp(+r·dt)
        per period, and the terminal coupon was added twice — both
        would inflate the tree price relative to the curve PV.
        """
        straight_curve = _straight_bond_hw(
            hw_flat, coupon_rate=0.08, maturity_years=5.0,
            n_steps=60, notional=100.0, coupon_frequency=1.0,
        )
        # Use a put with put_price=0 → put never exercised → same as
        # straight bond.
        bond = puttable_bond_price(
            hw_flat, coupon_rate=0.08, maturity_years=5.0,
            put_dates_years=[],   # no put dates
            put_price=0.0,
            n_steps=60, notional=100.0, coupon_frequency=1.0,
        )
        # Tree price should match curve-PV straight bond to within
        # tree discretization (~2%).
        assert bond == pytest.approx(straight_curve, rel=2e-2), (
            f"tree (no-option) = {bond:.4f}, curve PV = {straight_curve:.4f}"
        )
