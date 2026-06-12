"""Regression for L2 Wave-2 audit — bond ex-dividend dirty-price bug.

Pre-fix `dirty_price` summed every cashflow with ``payment_date > settlement``
including any upcoming coupon whose record date had already passed.  But
`accrued_interest` correctly returned a NEGATIVE value during the ex-dividend
window (buyer is being compensated for the coupon that goes to the
record-date holder).  The combination

    clean_price = dirty_price - accrued_interest
                = (PV including the unreceivable coupon) - (negative)
                = dirty + |accrued|

was approximately ONE FULL COUPON too high right after the ex-div boundary,
producing a discontinuity of size ≈ coupon at the ex-div date.

By market convention the clean price should be barely perceptible (only a
few days of curve-time discount drift) across that boundary — clean is the
"quoted" price exactly to make this transition smooth.

Post-fix `dirty_price` excludes the upcoming coupon's cashflow when
settlement falls in its ex-div window.  ``clean = dirty - accrued`` then
moves continuously across ex-div.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.day_count import DayCountConvention
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.schedule import Frequency
from pricebook.fixed_income.bond import FixedRateBond


def _bond(ex_div_days: int = 7) -> FixedRateBond:
    return FixedRateBond(
        issue_date=date(2024, 1, 1),
        maturity=date(2029, 1, 1),
        coupon_rate=0.05,
        frequency=Frequency.ANNUAL,
        face_value=100.0,
        day_count=DayCountConvention.ACT_365_FIXED,
        ex_div_days=ex_div_days,
    )


def _flat_curve(ref: date, rate: float = 0.04) -> DiscountCurve:
    tenors = [0.25, 1, 2, 5, 10]
    dates = [ref + timedelta(days=int(t * 365)) for t in tenors]
    return DiscountCurve(ref, dates, [math.exp(-rate * t) for t in tenors])


class TestExDivCleanPriceContinuity:
    def test_clean_price_continuous_across_ex_div_boundary(self):
        """Crossing the ex-div boundary by one day should NOT cause a
        coupon-sized jump in clean price.  Pre-fix it did (~$5 on a 5%
        coupon)."""
        bond = _bond(ex_div_days=7)
        # 2025-01-01 is the next coupon date.  ex_div_days=7 means
        # ex-div period is [2024-12-25, 2025-01-01).
        cum_div = date(2024, 12, 24)   # NOT ex-div (15 days before coupon... wait 8 days)
        ex_div = date(2024, 12, 25)    # first day of ex-div window

        clean_cum = bond.clean_price(_flat_curve(cum_div))
        clean_ex = bond.clean_price(_flat_curve(ex_div))
        jump = abs(clean_ex - clean_cum)
        assert jump < 0.10, \
            f"clean price jumped {jump:.4f} across ex-div boundary (cum={clean_cum:.4f}, ex={clean_ex:.4f}) — pre-fix would have been ~5.0"

    def test_dirty_price_excludes_unreceivable_coupon(self):
        """In ex-div, the buyer does NOT receive the upcoming coupon.
        Dirty price should reflect only the cashflows actually received."""
        bond = _bond(ex_div_days=7)
        ex_settle = date(2024, 12, 28)
        cum_settle = date(2024, 12, 20)
        dirty_ex = bond.dirty_price(_flat_curve(ex_settle))
        dirty_cum = bond.dirty_price(_flat_curve(cum_settle))
        # Dirty price drops by ~1 coupon at the ex-div boundary
        # (the upcoming 5% coupon = 5.0 per 100 face).
        drop = dirty_cum - dirty_ex
        assert 4.5 < drop < 5.0, \
            f"expected ~5.0 drop in dirty at ex-div boundary, got {drop:.4f}"


class TestExDivAccruedSign:
    def test_accrued_is_negative_in_ex_div_window(self):
        """Sanity: accrued sign in ex-div is unchanged by the dirty-price fix."""
        bond = _bond(ex_div_days=7)
        ex_settle = date(2024, 12, 28)
        assert bond.accrued_interest(ex_settle) < 0


class TestNoExDivBehaviourUnchanged:
    def test_ex_div_days_zero_disables_logic(self):
        """When ex_div_days=0 (the default), no coupon is excluded.
        Behaviour must match pre-fix for the entire interior of the bond's
        life."""
        bond = _bond(ex_div_days=0)
        settle = date(2024, 12, 28)
        c = _flat_curve(settle)
        # Just check it runs and returns a sensible value.
        d = bond.dirty_price(c)
        assert 100.0 < d < 115.0  # 5% coupon at 4% rate → premium bond
