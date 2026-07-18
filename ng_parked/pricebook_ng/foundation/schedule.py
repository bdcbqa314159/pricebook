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
  slice:  S02; S05 (RollRule/ScheduleTerms bundles)
"""

from __future__ import annotations

import calendar as _stdcal
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from pricebook_ng.foundation.calendar import RollRule
from pricebook_ng.foundation.time import DayCountConvention


class Frequency(Enum):
    MONTHLY = 1
    QUARTERLY = 3
    SEMI_ANNUAL = 6
    ANNUAL = 12


@dataclass(frozen=True)
class ScheduleTerms:
    """A leg's period conventions: coupon frequency, accrual day-count, and the
    roll rule. Bundles what a fixed leg needs so builders stay under the arg
    ceiling (CLAUDE.md 3b)."""

    frequency: Frequency
    day_count: DayCountConvention
    roll: RollRule = field(default_factory=RollRule)


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
    roll: RollRule | None = None,
) -> list[date]:
    """Dates from `start` to `end` at `frequency`, inclusive of both ends.

    Rolls backward from `end` (short front stub if the period doesn't divide
    evenly). EOM is decided once, anchored on `start` (ISDA 2006 s.4.10). When
    `roll.calendar` is set, every date is business-day adjusted under
    `roll.business_day`.
    """
    if start >= end:
        raise ValueError(f"start ({start}) must be before end ({end})")

    roll = roll or RollRule()
    snap_to_eom = roll.eom and start == _end_of_month(start)
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

    if roll.calendar is not None:
        return [roll.calendar.adjust(d, roll.business_day) for d in unadjusted]
    return unadjusted
