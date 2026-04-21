"""Day count conventions for year fraction calculations."""

from datetime import date
from enum import Enum


class DayCountConvention(Enum):
    ACT_360 = "ACT/360"
    ACT_365_FIXED = "ACT/365F"
    THIRTY_360 = "30/360"
    THIRTY_E_360 = "30E/360"
    ACT_ACT_ISDA = "ACT/ACT ISDA"
    ACT_ACT_ICMA = "ACT/ACT ICMA"


def year_fraction(
    start: date,
    end: date,
    convention: DayCountConvention,
    ref_start: date | None = None,
    ref_end: date | None = None,
    frequency: int | None = None,
) -> float:
    """Compute the year fraction between two dates under a given convention.

    Args:
        start: accrual start date.
        end: accrual end date.
        convention: day count convention.
        ref_start: coupon period start (needed for ACT/ACT ICMA).
        ref_end: coupon period end (needed for ACT/ACT ICMA).
        frequency: coupons per year (needed for ACT/ACT ICMA).
    """
    if start == end:
        return 0.0
    if start > end:
        raise ValueError(f"start ({start}) must be before end ({end})")

    if convention == DayCountConvention.ACT_360:
        return _act_360(start, end)
    elif convention == DayCountConvention.ACT_365_FIXED:
        return _act_365_fixed(start, end)
    elif convention == DayCountConvention.THIRTY_360:
        return _thirty_360(start, end)
    elif convention == DayCountConvention.THIRTY_E_360:
        return _thirty_e_360(start, end)
    elif convention == DayCountConvention.ACT_ACT_ISDA:
        return _act_act_isda(start, end)
    elif convention == DayCountConvention.ACT_ACT_ICMA:
        return _act_act_icma(start, end, ref_start, ref_end, frequency)
    else:
        raise ValueError(f"Unsupported convention: {convention}")


def _act_360(start: date, end: date) -> float:
    """ACT/360: actual days divided by 360. Standard for USD money markets."""
    return (end - start).days / 360.0


def _act_365_fixed(start: date, end: date) -> float:
    """ACT/365 Fixed: actual days divided by 365. Standard for GBP markets."""
    return (end - start).days / 365.0


def _thirty_360(start: date, end: date) -> float:
    """
    30/360 US (Bond Basis): assumes 30-day months and 360-day years.

    ISDA 2006 rules:
    1. If d1 = 31, change to 30
    2. If d2 = 31 and d1 >= 30 (after adjustment), change d2 to 30
    """
    d1 = start.day
    d2 = end.day

    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30

    days = 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)
    return days / 360.0


def _thirty_e_360(start: date, end: date) -> float:
    """
    30E/360 (Eurobond Basis / ISDA 2006 / German):
    1. If d1 = 31, change to 30
    2. If d2 = 31, change to 30 (unconditionally, unlike US 30/360)

    Used for Eurobonds, Bunds, and EUR corporate bonds.
    """
    d1 = min(start.day, 30)
    d2 = min(end.day, 30)
    days = 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)
    return days / 360.0


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _act_act_isda(start: date, end: date) -> float:
    """ACT/ACT ISDA: days in each year divided by that year's length (365 or 366).

    Handles periods spanning multiple years by splitting at year boundaries.
    """
    if start.year == end.year:
        days_in_year = 366 if _is_leap(start.year) else 365
        return (end - start).days / days_in_year

    total = 0.0
    # Fraction in start year
    year_end = date(start.year + 1, 1, 1)
    days_in_year = 366 if _is_leap(start.year) else 365
    total += (year_end - start).days / days_in_year

    # Full years in between
    for y in range(start.year + 1, end.year):
        total += 1.0

    # Fraction in end year
    year_start = date(end.year, 1, 1)
    days_in_year = 366 if _is_leap(end.year) else 365
    total += (end - year_start).days / days_in_year

    return total


def _act_act_icma(
    start: date,
    end: date,
    ref_start: date | None = None,
    ref_end: date | None = None,
    frequency: int | None = None,
) -> float:
    """ACT/ACT ICMA (Rule 251.1): actual days / (frequency × period length).

    Used for government bonds (UST, Bunds, Gilts, JGBs).
    The denominator is the actual length of the coupon period × frequency.

    year_frac = (end - start) / ((ref_end - ref_start) × frequency)

    If ref_start/ref_end not provided, falls back to ACT/365 Fixed.
    """
    if ref_start is None or ref_end is None or frequency is None:
        return (end - start).days / 365.0  # fallback

    period_days = (ref_end - ref_start).days
    if period_days <= 0:
        return (end - start).days / 365.0  # fallback

    return (end - start).days / (period_days * frequency)
