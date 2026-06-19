"""Day count conventions for year fraction calculations."""

from __future__ import annotations

from datetime import date, timedelta
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pricebook.core.calendar import Calendar


class DayCountConvention(Enum):
    ACT_360 = "ACT/360"
    ACT_365_FIXED = "ACT/365F"
    THIRTY_360 = "30/360"
    THIRTY_E_360 = "30E/360"
    ACT_ACT_ISDA = "ACT/ACT ISDA"
    ACT_ACT_ICMA = "ACT/ACT ICMA"
    BUS_252 = "BUS/252"


def year_fraction(
    start: date,
    end: date,
    convention: DayCountConvention,
    ref_start: date | None = None,
    ref_end: date | None = None,
    frequency: int | None = None,
    calendar: Calendar | None = None,
    *,
    strict_icma: bool = True,
) -> float:
    """Compute the year fraction between two dates under a given convention.

    Args:
        start: accrual start date.
        end: accrual end date.
        convention: day count convention.
        ref_start: coupon period start (needed for ACT/ACT ICMA).
        ref_end: coupon period end (needed for ACT/ACT ICMA).
        frequency: coupons per year (needed for ACT/ACT ICMA).
        calendar: business day calendar (required for BUS/252).
        strict_icma: when True (default since T-ICMA-SLICE3, 2026-06-19),
            ACT/ACT ICMA raises `ValueError` if any of `ref_start`,
            `ref_end`, `frequency` is missing or invalid (e.g.
            `period_days <= 0`, `frequency <= 0`). When False, ACT/ACT
            ICMA silently falls back to ACT/365F — the pre-A.1-B1
            historical behaviour, retained as an opt-in for callers that
            genuinely want the degradation (no production caller does
            post-T-ICMA-SLICE2). New code should always pass coupon
            anchors; the opt-out exists only for legacy test fixtures.
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
        return _act_act_icma(start, end, ref_start, ref_end, frequency,
                             strict=strict_icma)
    elif convention == DayCountConvention.BUS_252:
        return _bus_252(start, end, calendar)
    else:
        raise ValueError(f"Unsupported convention: {convention}")


def _act_360(start: date, end: date) -> float:
    """ACT/360: actual days divided by 360. Standard for USD money markets."""
    return (end - start).days / 360.0


def _act_365_fixed(start: date, end: date) -> float:
    """ACT/365 Fixed: actual days divided by 365. Standard for GBP markets."""
    return (end - start).days / 365.0


def _is_last_day_of_feb(d: date) -> bool:
    """Check if date is the last day of February."""
    return d.month == 2 and d.day == (29 if _is_leap(d.year) else 28)


def _thirty_360(start: date, end: date) -> float:
    """
    30/360 US (Bond Basis): assumes 30-day months and 360-day years.

    ISDA 2006 rules:
    1. If d1 is the last day of February, change d1 to 30
    2. If d1 = 31, change d1 to 30
    3. If d2 = 31 and d1 >= 30 (after adjustment), change d2 to 30
    """
    d1 = start.day
    d2 = end.day

    if _is_last_day_of_feb(start):
        d1 = 30
    if d1 == 31:
        d1 = 30
    if _is_last_day_of_feb(end) and _is_last_day_of_feb(start):
        d2 = 30
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
    *,
    strict: bool = False,
) -> float:
    """ACT/ACT ICMA (Rule 251.1): actual days / (frequency × period length).

    Used for government bonds (UST, Bunds, Gilts, JGBs).
    The denominator is the actual length of the coupon period × frequency.

    year_frac = (end - start) / ((ref_end - ref_start) × frequency)

    When `strict=False` (back-compat default), silently falls back to
    ACT/365 Fixed if `ref_start`, `ref_end`, or `frequency` are missing or
    invalid. This was the historical behaviour and the cause of audit
    finding A.1 B1 (UST coupons silently priced at 1.9836 / 2.0164 instead
    of exactly 2.0000).

    When `strict=True`, raises `ValueError` with a clear message instead.
    Callers migrated to strict mode are responsible for supplying the
    coupon-period anchors. New code should always pass `strict=True`.
    """
    def _missing_args() -> list[str]:
        missing = []
        if ref_start is None:
            missing.append("ref_start")
        if ref_end is None:
            missing.append("ref_end")
        if frequency is None:
            missing.append("frequency")
        return missing

    missing = _missing_args()
    if missing:
        if strict:
            raise ValueError(
                "ACT/ACT ICMA requires coupon-period anchors. Missing: "
                f"{', '.join(missing)}. Pass `ref_start`, `ref_end`, and "
                f"`frequency` to `year_fraction(...)`."
            )
        return (end - start).days / 365.0  # silent ACT/365F fallback (legacy)

    if frequency is not None and frequency <= 0:
        if strict:
            raise ValueError(
                f"ACT/ACT ICMA `frequency` must be > 0; got {frequency}."
            )
        return (end - start).days / 365.0  # legacy

    period_days = (ref_end - ref_start).days  # type: ignore[operator]
    if period_days <= 0:
        if strict:
            raise ValueError(
                f"ACT/ACT ICMA requires `ref_end > ref_start`; got "
                f"ref_start={ref_start}, ref_end={ref_end} "
                f"(period_days={period_days})."
            )
        return (end - start).days / 365.0  # legacy

    return (end - start).days / (period_days * frequency)


def _bus_252(start: date, end: date, calendar: Calendar | None = None) -> float:
    """BUS/252 (Brazilian convention): business days between dates / 252.

    Used by all BRL-denominated instruments: NTN-F, NTN-B, LTN, DI futures.
    The denominator is always 252 (the conventional number of business days
    per year in Brazil).

    If no calendar is provided, defaults to São Paulo calendar.
    """
    if calendar is None:
        from pricebook.core.calendar import SaoPauloCalendar
        calendar = SaoPauloCalendar()
    bd = business_days_between(start, end, calendar)
    return bd / 252.0


def business_days_between(start: date, end: date, calendar: Calendar) -> int:
    """Count business days between start (exclusive) and end (inclusive).

    This matches the market convention: the settlement date counts,
    the trade date does not.
    """
    count = 0
    current = start + timedelta(days=1)
    while current <= end:
        if calendar.is_business_day(current):
            count += 1
        current += timedelta(days=1)
    return count


def date_from_year_fraction(reference_date: date, t: float) -> date:
    """Convert a year fraction to a date, avoiding int(t*365) drift.

    Uses timedelta with round() instead of int() to minimise rounding error.
    For t <= 0 returns the reference_date.

        d = date_from_year_fraction(ref, 5.0)  # 5 years from ref
    """
    if t <= 0:
        return reference_date
    return reference_date + timedelta(days=round(t * 365.25))
