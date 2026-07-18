"""Day-count conventions and `year_fraction` (L0).

Finance-free: how a span of time is counted, not what it is worth. Ten conventions —
the seven market-standard ones plus the ACT/365L, 30E/360 ISDA and NL/365 gaps.

Two conventions need context, passed explicitly (no hidden defaults):
  - ACT/ACT ICMA takes coupon-period anchors via `CouponPeriod` — **strict**: missing
    anchors raise (the quarry's `strict_icma=False` silently fell back to ACT/365F and
    priced a UST coupon at 1.9836 instead of 2.0000). `strict_icma` is deleted.
  - BUS/252 takes the business-day `Calendar` — **required**: no calendar raises (the
    quarry silently defaulted to São Paulo).

Provenance:
  quarry: python/pricebook/core/day_count.py
  source: ISDA 2006 §4.16; ICMA Rule 251; Brazilian BUS/252 (ANBIMA)
  oracle: ISDA §4.16 worked examples; ICMA Rule 251 (UST coupon = exactly 2.0000);
          30U/360 February edge cases; ACT/365L / NL/365 leap-day handling
  slice:  daycounts (Topic 0 S2)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pricebook_ng.foundation.calendar import Calendar


class DayCountConvention(Enum):
    ACT_360 = "ACT/360"
    ACT_365_FIXED = "ACT/365F"
    THIRTY_360 = "30U/360"            # US SIA, with the February rules
    THIRTY_E_360 = "30E/360"         # Eurobond basis
    ACT_ACT_ISDA = "ACT/ACT ISDA"
    ACT_ACT_ICMA = "ACT/ACT ICMA"
    BUS_252 = "BUS/252"
    ACT_365L = "ACT/365L"            # gap: leap-aware denominator
    THIRTY_E_360_ISDA = "30E/360 ISDA"  # gap: last-day-of-month + termination rule
    NL_365 = "NL/365"                # gap: no-leap (exclude 29 Feb)


@dataclass(frozen=True)
class CouponPeriod:
    """The coupon period's convention context, serving three conventions: ICMA
    reference anchors + `frequency` (Rule 251, and the ACT/365L annual-vs-frequent
    denominator, ISDA §4.16(i)) and `is_final` (the 30E/360-ISDA termination rule).
    Bundles what they need so `year_fraction` stays ≤5 args."""

    reference_start: date
    reference_end: date
    frequency: int
    is_final: bool = False


def year_fraction(
    start: date,
    end: date,
    convention: DayCountConvention,
    *,
    coupon_period: CouponPeriod | None = None,
    calendar: "Calendar | None" = None,
) -> float:
    """Year fraction of ``[start, end)`` under `convention`. `coupon_period` is
    required for ACT/ACT ICMA (and carries the 30E/360-ISDA final flag); `calendar`
    is required for BUS/252."""
    C = DayCountConvention
    if convention is C.ACT_360:
        return (end - start).days / 360.0
    if convention is C.ACT_365_FIXED:
        return (end - start).days / 365.0
    if convention is C.THIRTY_360:
        return _thirty_360(start, end)
    if convention is C.THIRTY_E_360:
        return _thirty_e_360(start, end)
    if convention is C.THIRTY_E_360_ISDA:
        is_final = coupon_period.is_final if coupon_period is not None else False
        return _thirty_e_360_isda(start, end, is_final)
    if convention is C.ACT_ACT_ISDA:
        return _act_act_isda(start, end)
    if convention is C.ACT_ACT_ICMA:
        return _act_act_icma(start, end, coupon_period)
    if convention is C.ACT_365L:
        return _act_365l(start, end, coupon_period)
    if convention is C.NL_365:
        return _nl_365(start, end)
    if convention is C.BUS_252:
        if calendar is None:
            raise ValueError("BUS/252 requires a calendar (no default).")
        return business_days_between(start, end, calendar) / 252.0
    raise ValueError(f"unsupported convention: {convention}")


# ── helpers ──────────────────────────────────────────────────────────────────────
def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _last_day_of_month(d: date) -> int:
    if d.month == 12:
        return 31
    return (date(d.year, d.month + 1, 1) - timedelta(days=1)).day


def _is_last_of_feb(d: date) -> bool:
    return d.month == 2 and d.day == (29 if _is_leap(d.year) else 28)


def _30_360(d1: int, d2: int, start: date, end: date) -> float:
    days = 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)
    return days / 360.0


def _thirty_360(start: date, end: date) -> float:
    """30U/360 (US SIA) with the end-of-February adjustments (ISDA §4.16 30/360 US)."""
    d1, d2 = start.day, end.day
    if _is_last_of_feb(start):
        d1 = 30
    if d1 == 31:
        d1 = 30
    if _is_last_of_feb(end) and _is_last_of_feb(start):
        d2 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return _30_360(d1, d2, start, end)


def _thirty_e_360(start: date, end: date) -> float:
    """30E/360 (Eurobond basis): d=31 → 30 unconditionally for both ends; no Feb rule."""
    return _30_360(min(start.day, 30), min(end.day, 30), start, end)


def _thirty_e_360_isda(start: date, end: date, is_final: bool) -> float:
    """30E/360 ISDA (§4.16): last day of month → 30 for both ends, EXCEPT d2 is not
    adjusted when `end` is the termination date and falls in February."""
    d1 = 30 if start.day == _last_day_of_month(start) else start.day
    d2 = end.day
    if end.day == _last_day_of_month(end) and not (is_final and end.month == 2):
        d2 = 30
    return _30_360(d1, d2, start, end)


def _act_act_isda(start: date, end: date) -> float:
    """ACT/ACT ISDA: days in each calendar year over that year's length, summed."""
    if start.year == end.year:
        return (end - start).days / (366.0 if _is_leap(start.year) else 365.0)
    total = (date(start.year + 1, 1, 1) - start).days / (366.0 if _is_leap(start.year) else 365.0)
    total += float(end.year - start.year - 1)
    total += (end - date(end.year, 1, 1)).days / (366.0 if _is_leap(end.year) else 365.0)
    return total


def _act_act_icma(start: date, end: date, cp: CouponPeriod | None) -> float:
    """ACT/ACT ICMA (Rule 251.1): days / (period length × frequency). Strict — the
    anchors are required, never silently replaced by ACT/365F."""
    if cp is None:
        raise ValueError(
            "ACT/ACT ICMA requires coupon-period anchors (pass `coupon_period=`)."
        )
    period_days = (cp.reference_end - cp.reference_start).days
    if cp.frequency <= 0 or period_days <= 0:
        raise ValueError(
            f"ACT/ACT ICMA needs frequency>0 and reference_end>reference_start; "
            f"got frequency={cp.frequency}, period_days={period_days}."
        )
    return (end - start).days / (period_days * cp.frequency)


def _leap_days_in(start: date, end: date) -> int:
    """Number of 29-Februaries in ``[start, end)``."""
    count = 0
    for year in range(start.year, end.year + 1):
        if _is_leap(year):
            feb29 = date(year, 2, 29)
            if start <= feb29 < end:
                count += 1
    return count


def _act_365l(start: date, end: date, cp: CouponPeriod | None) -> float:
    """ACT/365L (ISDA §4.16(i)) — frequency-dependent denominator:
    annual (or no frequency context) → 366 iff a 29 Feb lies in the period, else 365;
    more frequent than annual → 366 iff the period END date is in a leap year, else 365."""
    frequency = cp.frequency if cp is not None else 1
    if frequency <= 1:
        denom = 366.0 if _leap_days_in(start, end) else 365.0
    else:
        denom = 366.0 if _is_leap(end.year) else 365.0
    return (end - start).days / denom


def _nl_365(start: date, end: date) -> float:
    """NL/365 (no-leap): actual days minus any 29-Februaries, over 365."""
    return ((end - start).days - _leap_days_in(start, end)) / 365.0


def business_days_between(start: date, end: date, calendar: "Calendar") -> int:
    """Business days in ``(start, end]`` — settlement counts, trade date does not."""
    count, cur = 0, start + timedelta(days=1)
    while cur <= end:
        if calendar.is_business_day(cur):
            count += 1
        cur += timedelta(days=1)
    return count
