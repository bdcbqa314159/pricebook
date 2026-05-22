"""Turn-of-year / seasonal term structure.

Models year-end, quarter-end, and month-end funding premia as
a deterministic overlay on the base curve.

    from pricebook.curves.seasonal_curve import (
        SeasonalPattern, SeasonalCurve, extract_seasonal_pattern,
    )

References:
    Munnik & Westerman (2005). Turn-of-Year Effects in Money Markets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention, year_fraction


@dataclass
class SeasonalPattern:
    """Seasonal funding premium pattern."""
    year_end_spread_bp: float = 10.0     # Dec 31 premium
    quarter_end_spread_bp: float = 3.0   # Mar/Jun/Sep 30 premium
    month_end_spread_bp: float = 1.0     # other month-ends
    spread_decay_days: int = 5           # how many days the premium persists

    def to_dict(self) -> dict:
        return vars(self)


# Pre-built patterns per currency
USD_SEASONAL = SeasonalPattern(12.0, 4.0, 1.5, 5)
EUR_SEASONAL = SeasonalPattern(8.0, 3.0, 1.0, 5)
GBP_SEASONAL = SeasonalPattern(10.0, 3.0, 1.0, 5)


class SeasonalCurve:
    """Discount curve with seasonal overlay.

    Forward rates have additional spread near period-ends.
    The df is adjusted: df_seasonal(T) = df_base(T) × exp(-seasonal_integral(0,T)).
    """

    def __init__(self, base_curve: DiscountCurve, pattern: SeasonalPattern):
        self.base_curve = base_curve
        self.pattern = pattern
        self.reference_date = base_curve.reference_date

    def df(self, d: date) -> float:
        """Discount factor with seasonal adjustment."""
        base_df = self.base_curve.df(d)
        seasonal_integral = self._seasonal_integral(self.reference_date, d)
        return base_df * math.exp(-seasonal_integral)

    def forward_rate(self, d1: date, d2: date) -> float:
        """Forward rate including seasonal premium."""
        df1 = self.df(d1)
        df2 = self.df(d2)
        dc = DayCountConvention.ACT_365_FIXED
        t = year_fraction(d1, d2, dc)
        if t <= 0 or df2 <= 0:
            return 0.0
        return (df1 / df2 - 1.0) / t

    def seasonal_spread(self, d: date) -> float:
        """Instantaneous seasonal spread at date d (annualised, decimal)."""
        return self._point_spread(d) / 10_000

    def _point_spread(self, d: date) -> float:
        """Seasonal spread in bp at a given date."""
        p = self.pattern

        # Year end: Dec 31
        year_end = date(d.year, 12, 31)
        days_to_ye = abs((d - year_end).days)
        if days_to_ye <= p.spread_decay_days:
            return p.year_end_spread_bp * (1 - days_to_ye / (p.spread_decay_days + 1))

        # Check previous year end too
        prev_ye = date(d.year - 1, 12, 31)
        days_from_prev = (d - prev_ye).days
        if 0 < days_from_prev <= p.spread_decay_days:
            return p.year_end_spread_bp * (1 - days_from_prev / (p.spread_decay_days + 1))

        # Quarter end: Mar 31, Jun 30, Sep 30
        for m, day in [(3, 31), (6, 30), (9, 30)]:
            qe = date(d.year, m, day)
            days_to_qe = abs((d - qe).days)
            if days_to_qe <= p.spread_decay_days:
                return p.quarter_end_spread_bp * (1 - days_to_qe / (p.spread_decay_days + 1))

        # Month end (other months)
        for m in [1, 2, 4, 5, 7, 8, 10, 11]:
            if m == 2:
                me_day = 29 if _is_leap(d.year) else 28
            elif m in (4, 6, 9, 11):
                me_day = 30
            else:
                me_day = 31
            me = date(d.year, m, me_day)
            days_to_me = abs((d - me).days)
            if days_to_me <= p.spread_decay_days:
                return p.month_end_spread_bp * (1 - days_to_me / (p.spread_decay_days + 1))

        return 0.0

    def _seasonal_integral(self, d1: date, d2: date) -> float:
        """Integral of seasonal spread from d1 to d2 (in year-fraction units)."""
        if d2 <= d1:
            return 0.0
        n_days = (d2 - d1).days
        # Daily summation
        total = 0.0
        for i in range(n_days):
            d = d1 + timedelta(days=i)
            total += self._point_spread(d) / 10_000 / 365.0
        return total


def extract_seasonal_pattern(
    on_fixings: dict[date, float],
    smooth_window: int = 21,
) -> SeasonalPattern:
    """Extract seasonal pattern from historical overnight fixings.

    Computes the average excess spread near year-end, quarter-end,
    and month-end relative to a smoothed baseline.

    Args:
        on_fixings: {date: overnight_rate} historical data.
        smooth_window: rolling window for baseline (business days).
    """
    if len(on_fixings) < smooth_window * 2:
        return SeasonalPattern()

    dates = sorted(on_fixings.keys())
    rates = np.array([on_fixings[d] for d in dates])

    # Simple moving average as baseline
    baseline = np.convolve(rates, np.ones(smooth_window) / smooth_window, mode="same")
    excess = rates - baseline

    # Classify each day
    ye_excess = []
    qe_excess = []
    me_excess = []

    for i, d in enumerate(dates):
        e = excess[i] * 10_000  # convert to bp
        if d.month == 12 and d.day >= 27:
            ye_excess.append(e)
        elif d.month == 1 and d.day <= 5:
            ye_excess.append(e)
        elif d.day >= 28 and d.month in (3, 6, 9):
            qe_excess.append(e)
        elif d.day >= 28:
            me_excess.append(e)

    return SeasonalPattern(
        year_end_spread_bp=max(float(np.mean(ye_excess)), 0) if ye_excess else 10.0,
        quarter_end_spread_bp=max(float(np.mean(qe_excess)), 0) if qe_excess else 3.0,
        month_end_spread_bp=max(float(np.mean(me_excess)), 0) if me_excess else 1.0,
    )


def strip_seasonal(
    curve: DiscountCurve,
    pattern: SeasonalPattern,
) -> DiscountCurve:
    """Remove seasonal from a curve (for smooth analysis)."""
    sc = SeasonalCurve(curve, pattern)
    ref = curve.reference_date
    new_dfs = []
    for d in curve.pillar_dates:
        seasonal = sc._seasonal_integral(ref, d)
        new_dfs.append(curve.df(d) * math.exp(seasonal))  # remove = add back
    return DiscountCurve(ref, curve.pillar_dates, new_dfs)


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
