"""Inflation caps/floors and inflation vol surface.

Zero-coupon inflation cap: payoff max(CPI(T)/CPI(0) - (1+K)^T, 0).
Year-on-year inflation cap: strip of caplets on annual inflation rate.
Inflation vol surface for smile calibration.

    from pricebook.inflation_vol import (
        zc_inflation_cap, yoy_inflation_cap, InflationVolSurface,
    )
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from dateutil.relativedelta import relativedelta

from pricebook.black76 import black76_price, OptionType
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.inflation import CPICurve
from pricebook.schedule import Frequency, generate_schedule
from pricebook.sabr import sabr_implied_vol, sabr_calibrate


# ---- Zero-coupon inflation cap/floor ----

def zc_inflation_cap(
    reference_date: date,
    maturity: date,
    strike_rate: float,
    cpi_curve: CPICurve,
    discount_curve: DiscountCurve,
    vol: float,
    notional: float = 1_000_000.0,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Price a zero-coupon inflation cap (or floor).

    Cap payoff at T: notional × max(CPI(T)/CPI(0) - (1+K)^T, 0).
    Modelled as a call on the inflation index ratio using Black-76.

    Args:
        strike_rate: annual inflation strike (e.g. 0.02 for 2%).
        cpi_curve: forward CPI curve.
        vol: inflation vol (lognormal on the index ratio).
    """
    T = year_fraction(reference_date, maturity, DayCountConvention.ACT_365_FIXED)
    if T <= 0:
        return 0.0

    # Forward inflation ratio
    fwd_cpi = cpi_curve.cpi(maturity)
    fwd_ratio = fwd_cpi / cpi_curve.base_cpi  # CPI(T) / CPI(0)

    # Strike as ratio
    strike_ratio = (1 + strike_rate) ** T

    df = discount_curve.df(maturity)

    price = black76_price(fwd_ratio, strike_ratio, vol, T, df, option_type)
    return notional * price


# ---- Year-on-year inflation cap/floor ----

def yoy_inflation_cap(
    start: date,
    end: date,
    strike_rate: float,
    cpi_curve: CPICurve,
    discount_curve: DiscountCurve,
    vol: float,
    notional: float = 1_000_000.0,
    frequency: Frequency = Frequency.ANNUAL,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Price a year-on-year inflation cap (or floor).

    Strip of caplets, each paying max(CPI(t_i)/CPI(t_{i-1}) - 1 - K, 0).

    Args:
        strike_rate: annual inflation strike per period.
        vol: inflation vol per caplet.
    """
    ref = cpi_curve.reference_date
    schedule = generate_schedule(start, end, frequency)
    total = 0.0

    for i in range(1, len(schedule)):
        d_prev = schedule[i - 1]
        d_curr = schedule[i]
        if d_curr <= ref:
            continue

        # Forward YoY inflation rate
        cpi_prev = cpi_curve.cpi(d_prev)
        cpi_curr = cpi_curve.cpi(d_curr)
        fwd_yoy = cpi_curr / cpi_prev  # ratio, not rate

        strike_yoy = 1 + strike_rate  # ratio

        t_fix = max(year_fraction(ref, d_prev, DayCountConvention.ACT_365_FIXED), 1e-6)
        df = discount_curve.df(d_curr)

        caplet = black76_price(fwd_yoy, strike_yoy, vol, t_fix, df, option_type)
        total += notional * caplet

    return total


# ---- Inflation vol surface ----

class InflationVolSurface:
    """ATM inflation vol surface (expiry-based).

    Args:
        reference_date: valuation date.
        expiries: expiry dates.
        vols: ATM vols at each expiry.
    """

    def __init__(
        self,
        reference_date: date,
        expiries: list[date],
        vols: list[float],
    ):
        if len(expiries) != len(vols):
            raise ValueError("expiries and vols must have same length")
        self.reference_date = reference_date
        self._times = np.array([
            year_fraction(reference_date, d, DayCountConvention.ACT_365_FIXED)
            for d in expiries
        ])
        self._vols = np.array(vols)

    def vol(self, expiry: date, strike: float | None = None) -> float:
        """Interpolated ATM vol at the given expiry."""
        t = year_fraction(self.reference_date, expiry, DayCountConvention.ACT_365_FIXED)
        t = max(t, self._times[0])

        if len(self._times) == 1:
            return float(self._vols[0])

        # Flat extrapolation
        if t <= self._times[0]:
            return float(self._vols[0])
        if t >= self._times[-1]:
            return float(self._vols[-1])

        # Linear interpolation
        idx = int(np.searchsorted(self._times, t)) - 1
        idx = max(0, min(idx, len(self._times) - 2))
        frac = (t - self._times[idx]) / (self._times[idx + 1] - self._times[idx])
        return float(self._vols[idx] + frac * (self._vols[idx + 1] - self._vols[idx]))

    def bump(self, shift: float) -> InflationVolSurface:
        """Return new surface with all vols shifted."""
        new_vols = (self._vols + shift).tolist()
        expiry_dates = [
            date.fromordinal(self.reference_date.toordinal() + int(t * 365))
            for t in self._times
        ]
        return InflationVolSurface(self.reference_date, expiry_dates, new_vols)
