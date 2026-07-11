"""Day-count conventions and year fractions.

Conventions: ACT/360, ACT/365F, 30/360 (US bond basis), 30E/360 (Eurobond
basis), ACT/ACT ISDA, ACT/ACT ICMA, and BUS/252 (business-days/252 — needs a
`Calendar`, added in S02 once calendars existed).

Debt shed in the crossing (CLAUDE.md 5): the quarry's ACT/ACT ICMA carried a
`strict_icma` flag that, when off, silently degraded to ACT/365F on missing
coupon anchors — the cause of audit finding A.1 B1 (UST coupons mispriced). That
is hidden wrongness, not deferred scope, so it does not cross: ICMA here always
requires `ref_start`, `ref_end`, `frequency` and raises otherwise.

Provenance:
  quarry: python/pricebook/core/day_count.py
  source: ISDA 2006 Definitions s.4.16; ICMA Rule 251.1
  oracle: published ISDA/ICMA year-fraction vectors, exact < 1e-12
  slice:  S00 (ACT/365F); S01 (calendar-free conventions); S02 (BUS/252)
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pricebook_ng.foundation.calendar import Calendar


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
    *,
    ref_start: date | None = None,
    ref_end: date | None = None,
    frequency: int | None = None,
    calendar: Calendar | None = None,
) -> float:
    """Year fraction between two dates under `convention`.

    `start` must be on or before `end`. ACT/ACT ICMA additionally requires the
    coupon-period anchors `ref_start`, `ref_end`, and `frequency` (coupons/year);
    BUS/252 requires a `calendar`.
    """
    if start > end:
        raise ValueError(f"start ({start}) must be on or before end ({end})")
    if start == end:
        return 0.0

    if convention is DayCountConvention.BUS_252:
        if calendar is None:
            raise ValueError("BUS/252 requires a calendar")
        return calendar.business_days_between(start, end) / 252.0
    if convention is DayCountConvention.ACT_360:
        return (end - start).days / 360.0
    if convention is DayCountConvention.ACT_365_FIXED:
        return (end - start).days / 365.0
    if convention is DayCountConvention.THIRTY_360:
        return _thirty_360(start, end)
    if convention is DayCountConvention.THIRTY_E_360:
        return _thirty_e_360(start, end)
    if convention is DayCountConvention.ACT_ACT_ISDA:
        return _act_act_isda(start, end)
    if convention is DayCountConvention.ACT_ACT_ICMA:
        return _act_act_icma(start, end, ref_start, ref_end, frequency)
    raise ValueError(f"Unsupported convention: {convention}")


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _days_in_year(year: int) -> int:
    return 366 if _is_leap(year) else 365


def _is_last_day_of_feb(d: date) -> bool:
    return d.month == 2 and d.day == (29 if _is_leap(d.year) else 28)


def _thirty_360(start: date, end: date) -> float:
    """30U/360 (US SIA bond basis) with end-of-February and day-31 rules."""
    d1, d2 = start.day, end.day
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
    """30E/360 (Eurobond basis): d1 and d2 capped at 30 unconditionally."""
    d1 = min(start.day, 30)
    d2 = min(end.day, 30)
    days = 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)
    return days / 360.0


def _act_act_isda(start: date, end: date) -> float:
    """ACT/ACT ISDA: days in each calendar year over that year's length."""
    if start.year == end.year:
        return (end - start).days / _days_in_year(start.year)
    total = (date(start.year + 1, 1, 1) - start).days / _days_in_year(start.year)
    total += float(end.year - start.year - 1)  # whole years in between
    total += (end - date(end.year, 1, 1)).days / _days_in_year(end.year)
    return total


def _act_act_icma(
    start: date,
    end: date,
    ref_start: date | None,
    ref_end: date | None,
    frequency: int | None,
) -> float:
    """ACT/ACT ICMA (Rule 251.1): days / (frequency * coupon-period length).

    Anchors are mandatory — no silent fallback (see module docstring).
    """
    missing = [n for n, v in (("ref_start", ref_start), ("ref_end", ref_end),
                              ("frequency", frequency)) if v is None]
    if missing:
        raise ValueError(
            f"ACT/ACT ICMA requires coupon-period anchors; missing: {', '.join(missing)}"
        )
    assert ref_start is not None and ref_end is not None and frequency is not None
    if frequency <= 0:
        raise ValueError(f"ACT/ACT ICMA frequency must be > 0, got {frequency}")
    period_days = (ref_end - ref_start).days
    if period_days <= 0:
        raise ValueError(
            f"ACT/ACT ICMA requires ref_end > ref_start; got {ref_start}..{ref_end}"
        )
    return (end - start).days / (period_days * frequency)
