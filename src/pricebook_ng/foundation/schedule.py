"""Payment schedules, roll rules, and IMM/CDS roll dates (L0).

Finance-free: it lays out *when* periods start and end, not what they pay. A
`Schedule` carries **both** the unadjusted dates (accrual boundaries) and the
business-day-adjusted dates (payment dates) — C2, they are not the same list.
EOM is anchored **once from `start`** (ISDA §4.10). Long stubs are explicit (the
stub period merges with its neighbour by construction) — the quarry's
`first_gap < months*30*0.5` merge heuristic is shed.

IMM (3rd Wednesday) and CDS (20th of Mar/Jun/Sep/Dec) roll dates are new here —
neither existed in the quarry; futures and credit both need them.

Provenance:
  quarry: python/pricebook/core/schedule.py
  source: ISDA 2006 §4.10 (EOM, period end dates); CME IMM; ISDA CDS roll (20th)
  oracle: EOM anchoring, four stubs, adjusted≠unadjusted, published IMM/CDS tables
  slice:  schedules (Topic 0 S3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

from pricebook_ng.foundation.calendar import BusinessDayConvention, Calendar
from pricebook_ng.foundation.tenor import Tenor, TenorUnit

_IMM_MONTHS = (3, 6, 9, 12)


@dataclass(frozen=True)
class Frequency:
    """A schedule period, as a `Tenor` step — or `BULLET` (`step=None`), a single period
    to maturity. A month-int enum could not express 28-day (TIIE), daily, or bullet, so
    the step is a `Tenor` (gate audit S3). The named frequencies are class constants."""

    step: Tenor | None

    def __str__(self) -> str:
        return "BULLET" if self.step is None else str(self.step)


Frequency.DAILY = Frequency(Tenor(1, TenorUnit.DAY))
Frequency.WEEKLY = Frequency(Tenor(1, TenorUnit.WEEK))
Frequency.MONTHLY = Frequency(Tenor(1, TenorUnit.MONTH))
Frequency.QUARTERLY = Frequency(Tenor(3, TenorUnit.MONTH))
Frequency.SEMI_ANNUAL = Frequency(Tenor(6, TenorUnit.MONTH))
Frequency.ANNUAL = Frequency(Tenor(1, TenorUnit.YEAR))
Frequency.BULLET = Frequency(None)


class StubType(Enum):
    SHORT_FRONT = "short_front"
    LONG_FRONT = "long_front"
    SHORT_BACK = "short_back"
    LONG_BACK = "long_back"


@dataclass(frozen=True)
class RollRule:
    """How unadjusted dates become business days: the calendar (None → no
    adjustment), the business-day convention, and whether month-end rolls stay at
    month-end (EOM, anchored from the schedule start — ISDA §4.10)."""

    calendar: Calendar | None = None
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    eom: bool = True


@dataclass(frozen=True)
class ScheduleTerms:
    """The recurring terms of a schedule: frequency, the roll rule, and the stub."""

    frequency: Frequency
    roll: RollRule = field(default_factory=RollRule)
    stub: StubType = StubType.SHORT_FRONT


@dataclass(frozen=True)
class Schedule:
    """A generated schedule: unadjusted period boundaries (for accrual) and their
    business-day-adjusted counterparts (for payment). Same length; equal when the
    roll rule has no calendar."""

    unadjusted: tuple[date, ...]
    adjusted: tuple[date, ...]


# ── date arithmetic (stdlib only) ────────────────────────────────────────────────
def _last_day_of_month(year: int, month: int) -> int:
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return (nxt - timedelta(days=1)).day


def _end_of_month(d: date) -> date:
    return date(d.year, d.month, _last_day_of_month(d.year, d.month))


def _add_months(d: date, months: int, snap_eom: bool) -> date:
    """Add `months`, clamping the day to the target month's length. `snap_eom`
    forces the result to month-end (the schedule-level EOM decision, made once)."""
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    result = date(year, month, min(d.day, _last_day_of_month(year, month)))
    return _end_of_month(result) if snap_eom else result


def _step(d: date, tenor: Tenor, forward: bool, snap_eom: bool) -> date:
    """Advance `d` by one `tenor` step (backward if not `forward`). EOM snapping applies
    to month/year steps only."""
    sign = 1 if forward else -1
    if tenor.unit is TenorUnit.DAY:
        return d + sign * timedelta(days=tenor.count)
    if tenor.unit is TenorUnit.WEEK:
        return d + sign * timedelta(weeks=tenor.count)
    return _add_months(d, sign * tenor.months(), snap_eom)


def _unadjusted(start: date, end: date, frequency: Frequency, stub: StubType, eom: bool) -> list[date]:
    if frequency.step is None:                       # BULLET — a single period
        return [start, end]
    tenor = frequency.step
    snap = eom and tenor.unit in (TenorUnit.MONTH, TenorUnit.YEAR) and start == _end_of_month(start)

    if stub in (StubType.SHORT_FRONT, StubType.LONG_FRONT):
        dates, cur = [end], end
        while (cur := _step(cur, tenor, False, snap)) > start:
            dates.append(cur)
        dates.append(start)
        dates.reverse()
        # a genuine front stub exists iff `start` is not a regular period before dates[1]
        if stub is StubType.LONG_FRONT and len(dates) > 2 and _step(dates[1], tenor, False, snap) != start:
            dates = [dates[0], *dates[2:]]
        return dates

    dates, cur = [start], start
    while (cur := _step(cur, tenor, True, snap)) < end:
        dates.append(cur)
    dates.append(end)
    if stub is StubType.LONG_BACK and len(dates) > 2 and _step(dates[-2], tenor, True, snap) != end:
        dates = [*dates[:-2], dates[-1]]
    return dates


def build_schedule(start: date, end: date, terms: ScheduleTerms) -> Schedule:
    """Generate the schedule from `start` to `end` under `terms`."""
    if start >= end:
        raise ValueError(f"start ({start}) must be before end ({end}).")
    unadj = _unadjusted(start, end, terms.frequency, terms.stub, terms.roll.eom)
    cal = terms.roll.calendar
    adj = ([cal.adjust(d, terms.roll.convention) for d in unadj] if cal is not None else unadj)
    return Schedule(unadjusted=tuple(unadj), adjusted=tuple(adj))


# ── IMM and CDS roll dates (new) ─────────────────────────────────────────────────
def imm_date(year: int, month: int) -> date:
    """The IMM date: the 3rd Wednesday of `month` (CME futures roll)."""
    first = date(year, month, 1)
    first_wed = first + timedelta(days=(2 - first.weekday()) % 7)  # 2 = Wednesday
    return first_wed + timedelta(weeks=2)


def next_imm(on_or_after: date) -> date:
    """The next IMM date (3rd Wednesday of Mar/Jun/Sep/Dec) on or after `on_or_after`."""
    for year in (on_or_after.year, on_or_after.year + 1):
        for m in _IMM_MONTHS:
            d = imm_date(year, m)
            if d >= on_or_after:
                return d
    raise AssertionError("unreachable")


def cds_roll_date(year: int, month: int) -> date:
    """The CDS roll date: the 20th of `month` (ISDA standard CDS maturities)."""
    return date(year, month, 20)


def next_cds_roll(on_or_after: date) -> date:
    """The next CDS roll (20th of Mar/Jun/Sep/Dec) on or after `on_or_after`."""
    for year in (on_or_after.year, on_or_after.year + 1):
        for m in _IMM_MONTHS:
            d = cds_roll_date(year, m)
            if d >= on_or_after:
                return d
    raise AssertionError("unreachable")
