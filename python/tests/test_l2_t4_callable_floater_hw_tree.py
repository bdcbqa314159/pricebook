"""Regression for L2 T4 audit of `fixed_income.callable_floater`:

Same defect class as ``callable_bond`` (T4-CB1) and
``bermudan_capfloor`` (T4-BCF1), applied here to the callable / puttable
floating-rate-note (FRN) HW tree (T4-CF1):

1. **Wrong trinomial probabilities** — pre-fix used ``/6`` instead of
   textbook Hull §32.4 eq. 32.10 ``/2``.

2. **Coupon applied AFTER backward discount** — the coupon at step+1
   was added to ``new_values`` (already discounted to step), so the
   coupon's own one-step discount factor was missing.

3. (``_frn_tree_with_option`` only) **Option applied AFTER backward
   discount** — ``min(v, call_price)`` / ``max(v, put_price)`` compared
   the discounted continuation against the undiscounted exercise
   price, biasing the exercise decision by ~exp(r·dt) per step.

Note: α(t) is implicit ``r0`` here (the module takes a raw ``r0``
rather than a ``DiscountCurve``), so the missing-α(t) issue in
``callable_bond`` doesn't apply — flat-curve interface is what's
intended.

Fix (T4-CF1): textbook ``/2`` probabilities; apply coupon (and option,
where present) BEFORE the backward discount.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from pricebook.fixed_income.callable_floater import callable_frn, puttable_frn


REF = date(2026, 1, 15)


class TestStraightFRNApproxPar:
    def test_straight_frn_near_par(self):
        """For a flat-curve FRN with zero spread, the straight-FRN
        price should be very close to par.  Pre-fix the missing
        coupon discount factor inflated coupons by ~exp(r·dt) per
        period — for n_periods=10, r=0.04, dt=0.5 that's a ~2%
        over-shoot per period × 10 = ~20% in total.  Post-fix the
        flat-curve FRN should sit within ~1% of par."""
        r = callable_frn(
            reference_date=REF, maturity_years=5.0, spread=0.0,
            hw_a=0.05, hw_sigma=0.01, r0=0.04,
            call_dates_years=[],  # no calls → straight FRN
            frequency=2, n_steps=80, notional=100.0,
        )
        # straight_frn_price should be near par.
        assert r.straight_frn_price == pytest.approx(100.0, rel=2e-2), (
            f"straight_frn_price = {r.straight_frn_price:.4f}, expected ≈ 100.0"
        )


class TestCallableLEStraight:
    def test_callable_le_straight(self):
        """Callable FRN ≤ straight FRN (issuer's call costs the investor)."""
        r = callable_frn(
            reference_date=REF, maturity_years=5.0, spread=0.005,
            hw_a=0.05, hw_sigma=0.01, r0=0.04,
            call_dates_years=[2.0, 3.0, 4.0],
            frequency=2, n_steps=80, notional=100.0,
        )
        assert r.price <= r.straight_frn_price + 1e-6


class TestPuttableGEStraight:
    def test_puttable_ge_straight(self):
        """Puttable FRN ≥ straight FRN (investor's put is valuable)."""
        r = puttable_frn(
            reference_date=REF, maturity_years=5.0, spread=0.005,
            hw_a=0.05, hw_sigma=0.01, r0=0.04,
            put_dates_years=[2.0, 3.0, 4.0],
            frequency=2, n_steps=80, notional=100.0,
        )
        assert r.price >= r.straight_frn_price - 1e-6
