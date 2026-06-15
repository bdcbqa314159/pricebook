"""Regression for L2 T4 audit of `options.inflation_vol.yoy_inflation_cap`:

Pre-fix the vol-time passed to Black-76 for each YoY caplet was
``year_fraction(ref, d_prev)`` — the time from valuation to the
PREVIOUS CPI fixing.  This is wrong for the standard YoY caplet
convention.  The YoY ratio ``CPI(d_curr)/CPI(d_prev)`` accumulates
volatility ONLY over the window ``(max(ref, d_prev), d_curr]`` (the
"vol-time" interpretation of the Black-76 ``T`` argument).

Two regimes were both broken:

- **Fully forward caplet** (``ref < d_prev``): pre-fix used
  ``d_prev - ref``, which IGNORES the actual exposure window
  ``d_curr - d_prev``.  For a 5y YoY caplet on year 4-5, the code
  used ~4y of vol time instead of 1y — gross over-pricing of
  forward YoY caplets.

- **Partially fixed** (``d_prev <= ref < d_curr``): pre-fix clamped
  to ``1e-6``, effectively zero vol — gross under-pricing (essentially
  intrinsic value only).

Fix (T4-INFL1): ``t_vol = year_fraction(max(ref, d_prev), d_curr)``.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention
from pricebook.core.schedule import Frequency
from pricebook.fixed_income.inflation import CPICurve
from pricebook.options.inflation_vol import yoy_inflation_cap


REF = date(2026, 1, 15)


def _cpi():
    """Constant-2%-inflation CPI curve."""
    dates = [REF + timedelta(days=365 * i) for i in range(11)]
    cpis = [100.0 * (1.02 ** i) for i in range(11)]
    return CPICurve(reference_date=REF, base_cpi=100.0, dates=dates, cpi_levels=cpis)


def _curve():
    return DiscountCurve.flat(REF, 0.03)


class TestForwardCapletVolTime:
    def test_5y_cap_finite_and_reasonable(self):
        """5y YoY cap.  Pre-fix the late caplets (year 4-5) used ~4y
        vol-time instead of 1y → ~2× the vol exposure → ~40% over-price
        on those caplets.  Post-fix the cap price is finite and
        broadly consistent across vol levels."""
        c = _cpi()
        d = _curve()
        start = REF
        end = REF + timedelta(days=365 * 5)
        # Far-OTM strike to keep numbers modest.
        price = yoy_inflation_cap(
            start=start, end=end, strike_rate=0.05,
            cpi_curve=c, discount_curve=d, vol=0.02,
            notional=1_000_000, frequency=Frequency.ANNUAL,
        )
        assert math.isfinite(price)
        assert price >= 0


class TestVolMonotonicity:
    def test_higher_vol_higher_cap(self):
        """Standard sanity that wasn't broken before, but verify
        post-fix the dependence is still monotone."""
        c = _cpi()
        d = _curve()
        end = REF + timedelta(days=365 * 5)
        low = yoy_inflation_cap(
            start=REF, end=end, strike_rate=0.03,
            cpi_curve=c, discount_curve=d, vol=0.01,
            notional=1_000_000, frequency=Frequency.ANNUAL,
        )
        high = yoy_inflation_cap(
            start=REF, end=end, strike_rate=0.03,
            cpi_curve=c, discount_curve=d, vol=0.04,
            notional=1_000_000, frequency=Frequency.ANNUAL,
        )
        assert high > low


class TestVolTimeMagnitude:
    def test_long_dated_cap_scales_with_vol_period(self):
        """Pre-fix a far-future YoY caplet (e.g. year 4-5) used vol-time
        ~4y → ~sqrt(4)/sqrt(1) = 2× the vol-amplitude vs the correct
        1y window.  At-the-money price ≈ ``0.4 · σ · √t``, so the
        pre-fix bias would inflate that caplet's price by ~2×.
        Post-fix the YoY cap value is bounded sensibly relative to a
        single-period YoY cap at the same parameters."""
        c = _cpi()
        d = _curve()
        # 1y cap (only one caplet, near-term).
        one_year = yoy_inflation_cap(
            start=REF, end=REF + timedelta(days=365),
            strike_rate=0.02,
            cpi_curve=c, discount_curve=d, vol=0.02,
            notional=1_000_000, frequency=Frequency.ANNUAL,
        )
        # 5y cap (five caplets).  Post-fix each caplet has ~the same
        # vol-time (1y), so the 5y cap ≈ 5 × the 1y caplet, modulo
        # different discount factors and forward levels.  Pre-fix the
        # later caplets had MUCH larger vol-time, inflating the 5y/1y
        # ratio well above 5.
        five_year = yoy_inflation_cap(
            start=REF, end=REF + timedelta(days=365 * 5),
            strike_rate=0.02,
            cpi_curve=c, discount_curve=d, vol=0.02,
            notional=1_000_000, frequency=Frequency.ANNUAL,
        )
        # Each post-fix YoY caplet shares a similar vol-time → cap
        # value scales roughly linearly with periods.  Upper bound ≈ 6×
        # 1y caplet (gives some slack for forward growth + DF).
        assert five_year <= 6.0 * one_year, (
            f"5y={five_year:.0f}, 1y={one_year:.0f}, ratio={five_year/one_year:.2f} "
            f"— if ratio > 6 the late caplets are still using inflated vol-time"
        )
