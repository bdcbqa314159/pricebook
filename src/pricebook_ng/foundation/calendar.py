"""Business-day calendar and date-adjustment conventions.

A minimal, data-driven calendar: a weekend rule (Sat/Sun) plus an explicit set
of holidays. Concrete national calendars (TARGET, US, London, Sao Paulo, ...)
migrate when a downstream instrument slice first needs them — no speculative
zoo of 30 calendars (CLAUDE.md 6b, rule of two).

Provenance:
  quarry: python/pricebook/core/calendar.py
  source: ISDA 2006 s.4.12 (business day conventions)
  oracle: hand-computed adjustments over weekends/holidays (exact dates)
  slice:  S02; S05 (RollRule bundle)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class BusinessDayConvention(Enum):
    UNADJUSTED = "unadjusted"
    FOLLOWING = "following"
    MODIFIED_FOLLOWING = "modified_following"
    PRECEDING = "preceding"
    MODIFIED_PRECEDING = "modified_preceding"


@dataclass(frozen=True)
class RollRule:
    """How schedule dates roll: which calendar, business-day convention, and the
    end-of-month rule. `calendar=None` means dates are left unadjusted."""

    calendar: "Calendar | None" = None
    business_day: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    eom: bool = True


@dataclass(frozen=True)
class Calendar:
    """Weekend (Sat/Sun) + an explicit holiday set. Empty set = weekend-only."""

    holidays: frozenset[date] = field(default_factory=frozenset)

    def is_business_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.holidays

    def adjust(self, d: date, convention: BusinessDayConvention) -> date:
        if convention is BusinessDayConvention.UNADJUSTED or self.is_business_day(d):
            return d
        if convention is BusinessDayConvention.FOLLOWING:
            return self._following(d)
        if convention is BusinessDayConvention.PRECEDING:
            return self._preceding(d)
        if convention is BusinessDayConvention.MODIFIED_FOLLOWING:
            adjusted = self._following(d)
            return self._preceding(d) if adjusted.month != d.month else adjusted
        if convention is BusinessDayConvention.MODIFIED_PRECEDING:
            adjusted = self._preceding(d)
            return self._following(d) if adjusted.month != d.month else adjusted
        raise ValueError(f"Unknown convention: {convention}")

    def business_days_between(self, start: date, end: date) -> int:
        """Business days in (start, end] — start exclusive, end inclusive."""
        count = 0
        current = start + timedelta(days=1)
        while current <= end:
            if self.is_business_day(current):
                count += 1
            current += timedelta(days=1)
        return count

    def _following(self, d: date) -> date:
        while not self.is_business_day(d):
            d += timedelta(days=1)
        return d

    def _preceding(self, d: date) -> date:
        while not self.is_business_day(d):
            d -= timedelta(days=1)
        return d
