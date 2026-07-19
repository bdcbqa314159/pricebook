"""Payment schedules, roll rules, and IMM/CDS roll dates (L0).

Finance-free: it lays out *when* periods start and end, not what they pay. A
`Schedule` carries **both** the unadjusted dates (accrual boundaries) and the
business-day-adjusted dates (payment dates) — C2, they are not the same list.
EOM is anchored **once, on the generation seed** — the maturity for backward (front-stub)
generation, the effective date for forward (ISDA §4.10; audit 1.4). Long stubs are explicit (the
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
from typing import ClassVar

from pricebook_ng.foundation.calendars import BusinessDayConvention, Calendar
from pricebook_ng.foundation.tenor import Tenor, TenorUnit

_IMM_MONTHS = (3, 6, 9, 12)


@dataclass(frozen=True)
class Frequency:
    """A schedule period, as a `Tenor` step — or `BULLET` (`step=None`), a single period
    to maturity. A month-int enum could not express 28-day (TIIE), daily, or bullet, so
    the step is a `Tenor` (gate audit S3). The named frequencies are class constants."""

    step: Tenor | None

    # The named frequencies (assigned below). Declared so `Frequency.MONTHLY` type-checks.
    DAILY: ClassVar[Frequency]
    WEEKLY: ClassVar[Frequency]
    MONTHLY: ClassVar[Frequency]
    QUARTERLY: ClassVar[Frequency]
    SEMI_ANNUAL: ClassVar[Frequency]
    ANNUAL: ClassVar[Frequency]
    BULLET: ClassVar[Frequency]

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


class RollConvention(Enum):
    """A rule-based roll anchor for `ScheduleTerms.roll_day` (S8): interior periods land on
    the IMM date (3rd Wednesday) or the CDS date (20th) of their month, regardless of the
    effective date — the normal case for IMM-dated FRAs/futures and standard CDS."""

    IMM = "imm"
    CDS = "cds"


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
    """The recurring terms of a schedule: frequency, the roll rule, the stub, and an
    optional `roll_day` — the interior-period roll anchor: a day-of-month (`int`, a bond the
    15th), a rule (`RollConvention.IMM`/`CDS`, landing on the 3rd Wednesday / 20th), or
    `None` (anchor on `start`). Rule-based anchors are what IMM-dated FRAs/futures need,
    where trade date ≠ effective date ≠ roll day (S8)."""

    frequency: Frequency
    roll: RollRule = field(default_factory=RollRule)
    stub: StubType = StubType.SHORT_FRONT
    roll_day: int | RollConvention | None = None


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


def _step_k(
    anchor: date,
    tenor: Tenor,
    k: int,
    snap_eom: bool,
    roll_day: int | RollConvention | None,
) -> date:
    """`anchor` shifted by `k` tenor-steps (`k < 0` = backward), computed FROM the anchor
    (`anchor + k·tenor`), not by accumulating single steps — so a month/year roll day never
    drifts through a short-month clamp (audit 1.4: May 31 − 6M is Nov 30, not Nov 29 via Feb).
    Month/year steps then snap to the `roll_day` anchor: a `RollConvention` rule (IMM 3rd-Wed /
    CDS 20th), an `int` day-of-month, else month-end if `snap_eom`."""
    if tenor.unit is TenorUnit.DAY:
        return anchor + timedelta(days=k * tenor.count)
    if tenor.unit is TenorUnit.WEEK:
        return anchor + timedelta(weeks=k * tenor.count)
    stepped = _add_months(anchor, k * tenor.months(), snap_eom)
    if roll_day is RollConvention.IMM:
        return imm_date(stepped.year, stepped.month)
    if roll_day is RollConvention.CDS:
        return cds_roll_date(stepped.year, stepped.month)
    if roll_day is not None:
        return date(
            stepped.year,
            stepped.month,
            min(roll_day, _last_day_of_month(stepped.year, stepped.month)),
        )
    return stepped


def _unadjusted(start: date, end: date, terms: ScheduleTerms) -> list[date]:
    if terms.frequency.step is None:  # BULLET — a single period
        return [start, end]
    tenor, stub, roll_day = terms.frequency.step, terms.stub, terms.roll_day
    is_month = tenor.unit in (TenorUnit.MONTH, TenorUnit.YEAR)

    def _snap_on(anchor: date) -> bool:
        # EOM anchors on the GENERATION seed (ISDA §4.10): maturity for backward (front)
        # generation, effective date for forward (back) generation — audit 1.4.
        return (
            roll_day is None
            and terms.roll.eom
            and is_month
            and anchor == _end_of_month(anchor)
        )

    if stub in (StubType.SHORT_FRONT, StubType.LONG_FRONT):
        snap, dates, k = _snap_on(end), [end], 1
        while (cur := _step_k(end, tenor, -k, snap, roll_day)) > start:
            dates.append(cur)
            k += 1
        dates.append(start)
        dates.reverse()
        # `cur` is the first regular boundary at/below `start`; a genuine front stub exists
        # iff it is not `start` itself.
        if stub is StubType.LONG_FRONT and len(dates) > 2 and cur != start:
            dates = [dates[0], *dates[2:]]
        return dates

    snap, dates, k = _snap_on(start), [start], 1
    while (cur := _step_k(start, tenor, k, snap, roll_day)) < end:
        dates.append(cur)
        k += 1
    dates.append(end)
    if stub is StubType.LONG_BACK and len(dates) > 2 and cur != end:
        dates = [*dates[:-2], dates[-1]]
    return dates


def build_schedule(start: date, end: date, terms: ScheduleTerms) -> Schedule:
    """Generate the schedule from `start` to `end` under `terms`."""
    if start >= end:
        raise ValueError(f"start ({start}) must be before end ({end}).")
    unadj = _unadjusted(start, end, terms)
    cal = terms.roll.calendar
    adj = (
        [cal.adjust(d, terms.roll.convention) for d in unadj]
        if cal is not None
        else unadj
    )
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
