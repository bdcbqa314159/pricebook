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


class TARGETCalendar(Calendar):
    """TARGET calendar (Trans-European Automated Real-time Gross settlement).

    Used for EUR-denominated products. Holidays: New Year's, Good Friday,
    Easter Monday, Labour Day (1 May), Christmas Day, 26 Dec.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))   # New Year's
        holidays.add(date(year, 5, 1))   # Labour Day
        holidays.add(date(year, 12, 25)) # Christmas
        holidays.add(date(year, 12, 26)) # St Stephen's

        # Easter (anonymous Gregorian algorithm)
        easter = self._easter(year)
        holidays.add(easter - timedelta(days=2))  # Good Friday
        holidays.add(easter + timedelta(days=1))   # Easter Monday

        return holidays

    @staticmethod
    def _easter(year: int) -> date:
        """Compute Easter Sunday (anonymous Gregorian algorithm)."""
        a = year % 19
        b, c = divmod(year, 100)
        d, e = divmod(b, 4)
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i, k = divmod(c, 4)
        L = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * L) // 451
        month, day = divmod(h + L - 7 * m + 114, 31)
        return date(year, month, day + 1)


class LondonCalendar(Calendar):
    """London (UK) banking calendar.

    Holidays: New Year's, Good Friday, Easter Monday, Early May,
    Spring Bank Holiday, Summer Bank Holiday, Christmas, Boxing Day.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(self._observe(date(year, 1, 1)))   # New Year's

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))  # Good Friday
        holidays.add(easter + timedelta(days=1))   # Easter Monday

        holidays.add(self._nth_weekday(year, 5, 0, 1))   # Early May
        holidays.add(self._last_weekday(year, 5, 0))      # Spring Bank Holiday
        holidays.add(self._last_weekday(year, 8, 0))      # Summer Bank Holiday
        holidays.add(self._observe(date(year, 12, 25)))    # Christmas
        holidays.add(self._observe(date(year, 12, 26)))    # Boxing Day

        return holidays


class TokyoCalendar(Calendar):
    """Tokyo (Japan) banking calendar.

    Major holidays: New Year's (1-3 Jan), Coming of Age Day, National Foundation,
    Vernal Equinox, Showa Day, Constitution Day, Greenery Day, Children's Day,
    Marine Day, Mountain Day, Respect for Aged, Autumnal Equinox,
    Sports Day, Culture Day, Labour Thanksgiving, Emperor's Birthday.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))   # New Year's
        holidays.add(date(year, 1, 2))
        holidays.add(date(year, 1, 3))
        holidays.add(self._nth_weekday(year, 1, 0, 2))   # Coming of Age (2nd Mon Jan)
        holidays.add(date(year, 2, 11))                    # National Foundation
        holidays.add(date(year, 2, 23))                    # Emperor's Birthday
        holidays.add(date(year, 3, 21))                    # Vernal Equinox (approx)
        holidays.add(date(year, 4, 29))                    # Showa Day
        holidays.add(date(year, 5, 3))                     # Constitution
        holidays.add(date(year, 5, 4))                     # Greenery
        holidays.add(date(year, 5, 5))                     # Children's
        holidays.add(self._nth_weekday(year, 7, 0, 3))    # Marine Day (3rd Mon Jul)
        holidays.add(date(year, 8, 11))                    # Mountain Day
        holidays.add(self._nth_weekday(year, 9, 0, 3))    # Respect for Aged (3rd Mon Sep)
        holidays.add(date(year, 9, 23))                    # Autumnal Equinox (approx)
        holidays.add(self._nth_weekday(year, 10, 0, 2))   # Sports Day (2nd Mon Oct)
        holidays.add(date(year, 11, 3))                    # Culture Day
        holidays.add(date(year, 11, 23))                   # Labour Thanksgiving

        return holidays


class CHFCalendar(Calendar):
    """Swiss banking calendar (Zurich).

    Holidays: New Year's, Berchtoldstag (2 Jan), Good Friday, Easter Monday,
    Labour Day (1 May), Ascension, Whit Monday, Swiss National Day (1 Aug),
    Christmas Day, St Stephen's (26 Dec).
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))    # New Year's
        holidays.add(date(year, 1, 2))    # Berchtoldstag
        holidays.add(date(year, 5, 1))    # Labour Day
        holidays.add(date(year, 8, 1))    # Swiss National Day
        holidays.add(date(year, 12, 25))  # Christmas
        holidays.add(date(year, 12, 26))  # St Stephen's

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter + timedelta(days=1))    # Easter Monday
        holidays.add(easter + timedelta(days=39))   # Ascension
        holidays.add(easter + timedelta(days=50))   # Whit Monday

        return holidays


class AUDCalendar(Calendar):
    """Australian banking calendar (Sydney).

    Holidays: New Year's, Australia Day (26 Jan), Good Friday, Easter Saturday,
    Easter Monday, Anzac Day (25 Apr), Queen's Birthday (2nd Mon Jun),
    Bank Holiday (1st Mon Aug), Christmas, Boxing Day.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(self._observe(date(year, 1, 1)))    # New Year's
        holidays.add(self._observe(date(year, 1, 26)))   # Australia Day
        holidays.add(date(year, 4, 25))                   # Anzac Day
        holidays.add(self._nth_weekday(year, 6, 0, 2))   # Queen's Birthday (2nd Mon Jun)
        holidays.add(self._nth_weekday(year, 8, 0, 1))   # Bank Holiday (1st Mon Aug)
        holidays.add(self._observe(date(year, 12, 25)))   # Christmas
        holidays.add(self._observe(date(year, 12, 26)))   # Boxing Day

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter - timedelta(days=1))   # Easter Saturday
        holidays.add(easter + timedelta(days=1))    # Easter Monday

        return holidays


class CADCalendar(Calendar):
    """Canadian banking calendar (Toronto).

    Holidays: New Year's, Family Day (3rd Mon Feb), Good Friday,
    Victoria Day (Mon before 25 May), Canada Day (1 Jul),
    Civic Holiday (1st Mon Aug), Labour Day (1st Mon Sep),
    Thanksgiving (2nd Mon Oct), Remembrance Day (11 Nov),
    Christmas, Boxing Day.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(self._observe(date(year, 1, 1)))     # New Year's
        holidays.add(self._nth_weekday(year, 2, 0, 3))    # Family Day (3rd Mon Feb)

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))   # Good Friday

        # Victoria Day: Monday before May 25
        may25 = date(year, 5, 25)
        days_since_mon = (may25.weekday() - 0) % 7
        if days_since_mon == 0:
            days_since_mon = 7
        holidays.add(may25 - timedelta(days=days_since_mon))

        holidays.add(self._observe(date(year, 7, 1)))     # Canada Day
        holidays.add(self._nth_weekday(year, 8, 0, 1))    # Civic Holiday
        holidays.add(self._nth_weekday(year, 9, 0, 1))    # Labour Day
        holidays.add(self._nth_weekday(year, 10, 0, 2))   # Thanksgiving
        holidays.add(self._observe(date(year, 11, 11)))    # Remembrance Day
        holidays.add(self._observe(date(year, 12, 25)))    # Christmas
        holidays.add(self._observe(date(year, 12, 26)))    # Boxing Day

        return holidays


class JointCalendar(Calendar):
    """Joint calendar: a date is a holiday if it's a holiday in ANY component."""

    def __init__(self, *calendars: Calendar):
        super().__init__()
        self._calendars = calendars

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        for cal in self._calendars:
            if year not in cal._holiday_cache:
                cal._holiday_cache[year] = cal._compute_holidays(year)
            holidays |= cal._holiday_cache[year]
        return holidays
