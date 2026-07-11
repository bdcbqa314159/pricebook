"""Payment schedule generation.

Regular periodic dates with a short front stub (backward generation from the
end date) and the EOM roll, optionally business-day adjusted. Month arithmetic
uses the stdlib `calendar` module — no third-party dependency for the new tree.

Long/back stubs are deferred: the quarry's variants carried an approximate
`months*30` gap heuristic, and no current consumer needs them (CLAUDE.md 6b).
They land, cleanly oracled, when an instrument slice first requires them.

Provenance:
  quarry: python/pricebook/core/schedule.py
  source: ISDA 2006 s.4.10 (period end dates, EOM rule)
  oracle: hand-computed coupon schedules incl. EOM + short front stub (exact)
  slice:  S02
"""

from __future__ import annotations

import calendar as _stdcal
from datetime import date
from enum import Enum

from pricebook_ng.foundation.calendar import BusinessDayConvention, Calendar


class Frequency(Enum):
    MONTHLY = 1
    QUARTERLY = 3
    SEMI_ANNUAL = 6
    ANNUAL = 12


def _end_of_month(d: date) -> date:
    return d.replace(day=_stdcal.monthrange(d.year, d.month)[1])


def _add_months(d: date, months: int, snap_to_eom: bool) -> date:
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    last = _stdcal.monthrange(year, month)[1]
    day = last if snap_to_eom else min(d.day, last)
    return date(year, month, day)


def generate_schedule(
    start: date,
    end: date,
    frequency: Frequency,
    calendar: Calendar | None = None,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    eom: bool = True,
) -> list[date]:
    """Dates from `start` to `end` at `frequency`, inclusive of both ends.

    Rolls backward from `end` (short front stub if the period doesn't divide
    evenly). EOM is decided once, anchored on `start` (ISDA 2006 s.4.10). When a
    `calendar` is given, every date is business-day adjusted under `convention`.
    """
    if start >= end:
        raise ValueError(f"start ({start}) must be before end ({end})")

    snap_to_eom = eom and start == _end_of_month(start)
    months = frequency.value

    unadjusted = [end]
    current = end
    while True:
        current = _add_months(current, -months, snap_to_eom)
        if current <= start:
            break
        unadjusted.append(current)
    unadjusted.append(start)
    unadjusted.reverse()

    if calendar is not None:
        return [calendar.adjust(d, convention) for d in unadjusted]
    return unadjusted
