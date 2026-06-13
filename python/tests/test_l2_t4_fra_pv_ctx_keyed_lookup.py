"""Regression for L2 Wave-2 audit — `FRA.pv_ctx` silently picked the
first projection curve when the day-count-keyed lookup missed.

Pre-fix:

    if dc_key and dc_key in ctx.projection_curves:
        proj = ctx.projection_curves[dc_key]
    else:
        proj = next(iter(ctx.projection_curves.values()))

In a multi-curve setup (e.g. ACT/360 USD-LIBOR vs ACT/365 GBP), an
ACT/360 FRA could silently get priced against an ACT/365 projection
curve — wrong-curve forward rates with no diagnostic.

Post-fix the function raises ``KeyError`` listing the offending
day-count and the available keys.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.day_count import DayCountConvention
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.pricing_context import PricingContext
from pricebook.fixed_income.fra import FRA


REF = date(2024, 1, 1)


def _flat_curve(rate: float = 0.04) -> DiscountCurve:
    tenors = [0.25, 0.5, 1.0, 2.0, 5.0]
    dates = [REF + timedelta(days=int(t * 365)) for t in tenors]
    return DiscountCurve(REF, dates, [math.exp(-rate * t) for t in tenors])


class TestKeyedLookupMissRaises:
    def test_act_360_fra_no_act_360_curve_raises(self):
        """An ACT/360 FRA against a context with only ACT/365 projection
        curves must raise, not silently pick the wrong one."""
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=_flat_curve(0.04),
            projection_curves={
                # Only ACT/365 Fixed available.
                DayCountConvention.ACT_365_FIXED.value: _flat_curve(0.04),
            },
        )
        fra = FRA(
            start=REF + timedelta(days=180),
            end=REF + timedelta(days=270),
            strike=0.04,
            notional=1_000_000.0,
            day_count=DayCountConvention.ACT_360,  # different from ctx keys
        )
        with pytest.raises(KeyError, match="ACT/360"):
            fra.pv_ctx(ctx)


class TestKeyedLookupHitWorks:
    def test_act_360_fra_with_matching_curve_prices(self):
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=_flat_curve(0.04),
            projection_curves={
                DayCountConvention.ACT_360.value: _flat_curve(0.04),
            },
        )
        fra = FRA(
            start=REF + timedelta(days=180),
            end=REF + timedelta(days=270),
            strike=0.04,
            notional=1_000_000.0,
            day_count=DayCountConvention.ACT_360,
        )
        pv = fra.pv_ctx(ctx)
        # ATM-ish on a flat curve via log-linear interpolation between
        # pillars: PV won't be exactly zero (year-fraction convention
        # in the strike vs the curve's discounting), but small.
        assert abs(pv) < 1_000.0
        # The KEY assertion: it priced without raising.
        import math as _m
        assert _m.isfinite(pv)


class TestNoProjectionCurvesFallsBackToDiscount:
    """If the context has NO projection curves at all, the legacy
    behaviour of falling back to the discount curve is preserved."""

    def test_empty_projection_falls_back(self):
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=_flat_curve(0.04),
            projection_curves={},   # empty
        )
        fra = FRA(
            start=REF + timedelta(days=180),
            end=REF + timedelta(days=270),
            strike=0.04,
            notional=1_000_000.0,
            day_count=DayCountConvention.ACT_360,
        )
        pv = fra.pv_ctx(ctx)
        # Discount-curve fallback succeeds without raising.
        import math as _m
        assert _m.isfinite(pv)
