"""Business day calendars and date adjustment conventions."""

from datetime import date, timedelta
from enum import Enum
from abc import ABC, abstractmethod


class BusinessDayConvention(Enum):
    UNADJUSTED = "unadjusted"
    FOLLOWING = "following"
    MODIFIED_FOLLOWING = "modified_following"
    PRECEDING = "preceding"
    MODIFIED_PRECEDING = "modified_preceding"


class Calendar(ABC):
    """Base class for business day calendars."""

    def __init__(self):
        self._holiday_cache: dict[int, set[date]] = {}

    @abstractmethod
    def _compute_holidays(self, year: int) -> set[date]:
        """Compute the set of holidays for a given year."""

    def is_holiday(self, d: date) -> bool:
        """Check if a date is a holiday (excluding weekends)."""
        # Ensure both this year and next are computed, since observed
        # holidays can spill across year boundaries (e.g. Jan 1 on Saturday
        # is observed Dec 31 of the previous year).
        for y in (d.year, d.year + 1):
            if y not in self._holiday_cache:
                self._holiday_cache[y] = self._compute_holidays(y)
        return d in self._holiday_cache[d.year] or d in self._holiday_cache[d.year + 1]

    def is_weekend(self, d: date) -> bool:
        return d.weekday() >= 5

    def is_business_day(self, d: date) -> bool:
        return not self.is_weekend(d) and not self.is_holiday(d)

    def adjust(self, d: date, convention: BusinessDayConvention) -> date:
        """Adjust a date according to a business day convention."""
        if convention == BusinessDayConvention.UNADJUSTED:
            return d

        if self.is_business_day(d):
            return d

        if convention == BusinessDayConvention.FOLLOWING:
            return self._following(d)

        if convention == BusinessDayConvention.MODIFIED_FOLLOWING:
            adjusted = self._following(d)
            if adjusted.month != d.month:
                return self._preceding(d)
            return adjusted

        if convention == BusinessDayConvention.PRECEDING:
            return self._preceding(d)

        if convention == BusinessDayConvention.MODIFIED_PRECEDING:
            adjusted = self._preceding(d)
            if adjusted.month != d.month:
                return self._following(d)
            return adjusted

        raise ValueError(f"Unknown convention: {convention}")

    def _following(self, d: date) -> date:
        current = d
        while not self.is_business_day(current):
            current += timedelta(days=1)
        return current

    def _preceding(self, d: date) -> date:
        current = d
        while not self.is_business_day(current):
            current -= timedelta(days=1)
        return current

    @staticmethod
    def _observe(d: date) -> date:
        """Weekend observation: Saturday -> Friday, Sunday -> Monday."""
        if d.weekday() == 5:
            return d - timedelta(days=1)
        if d.weekday() == 6:
            return d + timedelta(days=1)
        return d

    @staticmethod
    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
        """Find the nth occurrence of a weekday in a month (weekday: 0=Mon)."""
        first = date(year, month, 1)
        days_to_add = (weekday - first.weekday()) % 7
        first_occurrence = first + timedelta(days=days_to_add)
        return first_occurrence + timedelta(weeks=n - 1)

    @staticmethod
    def _last_weekday(year: int, month: int, weekday: int) -> date:
        """Find the last occurrence of a weekday in a month."""
        if month == 12:
            last = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last = date(year, month + 1, 1) - timedelta(days=1)
        days_back = (last.weekday() - weekday) % 7
        return last - timedelta(days=days_back)


class USSettlementCalendar(Calendar):
    """
    US Settlement calendar (SIFMA/Federal Reserve).

    Holidays: New Year's, MLK Day, Presidents' Day, Memorial Day,
    Juneteenth, Independence Day, Labor Day, Columbus Day,
    Veterans Day, Thanksgiving, Christmas.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()

        holidays.add(self._observe(date(year, 1, 1)))                # New Year's
        holidays.add(self._nth_weekday(year, 1, 0, 3))               # MLK Day
        holidays.add(self._nth_weekday(year, 2, 0, 3))               # Presidents' Day
        holidays.add(self._last_weekday(year, 5, 0))                 # Memorial Day
        if year >= 2021:
            holidays.add(self._observe(date(year, 6, 19)))           # Juneteenth
        holidays.add(self._observe(date(year, 7, 4)))                # Independence Day
        holidays.add(self._nth_weekday(year, 9, 0, 1))               # Labor Day
        holidays.add(self._nth_weekday(year, 10, 0, 2))              # Columbus Day
        holidays.add(self._observe(date(year, 11, 11)))              # Veterans Day
        holidays.add(self._nth_weekday(year, 11, 3, 4))              # Thanksgiving
        holidays.add(self._observe(date(year, 12, 25)))              # Christmas

        return holidays
