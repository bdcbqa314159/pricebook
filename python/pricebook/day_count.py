"""Day count conventions for year fraction calculations."""

from datetime import date
from enum import Enum


class DayCountConvention(Enum):
    ACT_360 = "ACT/360"
    ACT_365_FIXED = "ACT/365F"
    THIRTY_360 = "30/360"


def year_fraction(start: date, end: date, convention: DayCountConvention) -> float:
    """Compute the year fraction between two dates under a given convention."""
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
