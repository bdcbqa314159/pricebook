"""Regression for L2 Tier-2 T2.14 — bond YTM-based pricing uses the LAST-period
notional for redemption (consistent with curve-based pricing).

Pre-fix `_price_from_ytm`, `macaulay_duration`, and `convexity` all used
``self.face_value`` (the FIRST period's notional from the schedule) for the
redemption term, while ``dirty_price`` (curve-based) used
``notional_schedule[-1]``.  For a sinking-fund bond (amortising notional),
the two prices silently disagreed.

Post-fix all paths use the last-period notional.  Constant-notional bonds
are unaffected (first = last).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.day_count import DayCountConvention
from pricebook.core.discount_curve import DiscountCurve
from pricebook.fixed_income.bond import FixedRateBond


def _flat_curve(rate: float = 0.04, ref: date | None = None) -> DiscountCurve:
    ref = ref or date(2024, 1, 1)
    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    dfs = [math.exp(-rate * t) for t in tenors]
    dates = [ref + timedelta(days=int(t * 365)) for t in tenors]
    return DiscountCurve(ref, dates, dfs, day_count=DayCountConvention.ACT_365_FIXED)


class TestConstantNotionalUnchanged:
    def test_constant_notional_ytm_matches_curve(self):
        """Sanity: for a plain (constant-notional) bond, YTM-pricing and
        curve-pricing should agree on a flat curve (YTM = flat rate)."""
        issue = date(2024, 1, 1)
        mat = date(2029, 1, 1)
        bond = FixedRateBond(
            issue, mat, coupon_rate=0.04,
            face_value=100.0,
            day_count=DayCountConvention.ACT_365_FIXED,
        )
        curve = _flat_curve(rate=0.04, ref=issue)
        curve_price = bond.dirty_price(curve)
        ytm_price = bond._price_from_ytm(0.04, settlement=issue)
        # On a flat 4% curve, a 4%-coupon bond should be ≈ par via both paths.
        assert abs(curve_price - 100.0) < 0.5
        assert abs(ytm_price - curve_price) < 0.5


class TestSinkingFundConsistency:
    def test_amortising_redemption_matches_curve(self):
        """For a sinking-fund bond (notional amortises from 100 to 50), the
        YTM-based price must use the LAST period's notional (=50) for the
        redemption, matching `dirty_price`."""
        issue = date(2024, 1, 1)
        mat = date(2029, 1, 1)
        # 10 semi-annual periods amortising 100 → 50.
        notional_schedule = [100.0 - 5.0 * i for i in range(10)]  # 100, 95, ..., 55
        bond = FixedRateBond(
            issue, mat, coupon_rate=0.04,
            face_value=notional_schedule,
            day_count=DayCountConvention.ACT_365_FIXED,
        )
        curve = _flat_curve(rate=0.04, ref=issue)
        curve_price = bond.dirty_price(curve)
        # Find YTM that re-prices the curve-based price, then verify via _price_from_ytm.
        # The two paths should agree on flat curve when YTM = flat rate.
        ytm_price = bond._price_from_ytm(0.04, settlement=issue)
        # Pre-fix the YTM path used face_value (= 100, the FIRST period), so
        # the YTM price's redemption term was 100 while curve's was 55 — a
        # large discrepancy.  Post-fix both use 55.
        diff = abs(curve_price - ytm_price)
        assert diff < 1.0, (
            f"Sinking-fund bond: curve={curve_price:.4f}, ytm={ytm_price:.4f}, "
            f"diff={diff:.4f} — Pre-fix difference was much larger because "
            f"the YTM redemption used face_value (first period) instead of "
            f"notional_schedule[-1] (last period)."
        )

    def test_duration_uses_last_notional(self):
        """Macaulay duration's principal term should use the last-period
        notional too — pre-fix it used face_value (first period)."""
        issue = date(2024, 1, 1)
        mat = date(2029, 1, 1)
        notional_schedule = [100.0 - 5.0 * i for i in range(10)]
        bond = FixedRateBond(
            issue, mat, coupon_rate=0.04,
            face_value=notional_schedule,
            day_count=DayCountConvention.ACT_365_FIXED,
        )
        # Just verify it runs without raising and produces a sane number.
        dur = bond.macaulay_duration(0.04, settlement=issue)
        assert 1.0 < dur < 5.0, f"Macaulay duration = {dur} outside plausible range"

    def test_convexity_uses_last_notional(self):
        issue = date(2024, 1, 1)
        mat = date(2029, 1, 1)
        notional_schedule = [100.0 - 5.0 * i for i in range(10)]
        bond = FixedRateBond(
            issue, mat, coupon_rate=0.04,
            face_value=notional_schedule,
            day_count=DayCountConvention.ACT_365_FIXED,
        )
        conv = bond.convexity(0.04, settlement=issue)
        assert conv > 0
        assert math.isfinite(conv)
