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

    def add_business_days(self, d: date, n: int) -> date:
        """Move forward (n > 0) or backward (n < 0) by n business days."""
        step = 1 if n >= 0 else -1
        remaining = abs(n)
        current = d
        while remaining > 0:
            current += timedelta(days=step)
            if self.is_business_day(current):
                remaining -= 1
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
        obs_xmas = self._observe(date(year, 12, 25))    # Christmas
        obs_boxing = self._observe(date(year, 12, 26))    # Boxing Day
        holidays.add(obs_xmas)
        # When Dec 25 is Sunday, both observe to Dec 26 — shift Boxing Day to 27
        if obs_boxing == obs_xmas:
            holidays.add(obs_boxing + timedelta(days=1))
        else:
            holidays.add(obs_boxing)

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
        obs_xmas = self._observe(date(year, 12, 25))    # Christmas
        obs_boxing = self._observe(date(year, 12, 26))   # Boxing Day
        holidays.add(obs_xmas)
        # When Dec 25 is Sunday, both observe to Dec 26 — shift Boxing Day to 27
        if obs_boxing == obs_xmas:
            holidays.add(obs_boxing + timedelta(days=1))
        else:
            holidays.add(obs_boxing)

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
        obs_xmas = self._observe(date(year, 12, 25))    # Christmas
        obs_boxing = self._observe(date(year, 12, 26))    # Boxing Day
        holidays.add(obs_xmas)
        # When Dec 25 is Sunday, both observe to Dec 26 — shift Boxing Day to 27
        if obs_boxing == obs_xmas:
            holidays.add(obs_boxing + timedelta(days=1))
        else:
            holidays.add(obs_boxing)

        return holidays


class SEKCalendar(Calendar):
    """Swedish banking calendar (Stockholm).

    Holidays: New Year's, Epiphany (6 Jan), Good Friday, Easter Monday,
    May Day (1 May), Ascension, National Day (6 Jun), Midsummer Eve
    (Fri before Midsummer Day = Sat between 20-26 Jun),
    Christmas Eve, Christmas, Boxing Day, New Year's Eve.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 1, 6))     # Epiphany
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 6, 6))     # National Day

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter + timedelta(days=1))    # Easter Monday
        holidays.add(easter + timedelta(days=39))   # Ascension Day

        # Midsummer Eve: Friday before the Saturday between Jun 20-26
        for d in range(20, 27):
            candidate = date(year, 6, d)
            if candidate.weekday() == 5:  # Saturday
                holidays.add(candidate - timedelta(days=1))  # Friday
                break

        holidays.add(date(year, 12, 24))   # Christmas Eve
        holidays.add(date(year, 12, 25))   # Christmas
        holidays.add(date(year, 12, 26))   # Boxing Day
        holidays.add(date(year, 12, 31))   # New Year's Eve

        return holidays


class NOKCalendar(Calendar):
    """Norwegian banking calendar (Oslo).

    Holidays: New Year's, Maundy Thursday, Good Friday, Easter Monday,
    May Day (1 May), Constitution Day (17 May), Ascension,
    Whit Monday, Christmas, Boxing Day.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 5, 17))    # Constitution Day

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=3))   # Maundy Thursday
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter + timedelta(days=1))    # Easter Monday
        holidays.add(easter + timedelta(days=39))   # Ascension
        holidays.add(easter + timedelta(days=50))   # Whit Monday

        holidays.add(date(year, 12, 25))   # Christmas
        holidays.add(date(year, 12, 26))   # Boxing Day

        return holidays


class NZDCalendar(Calendar):
    """New Zealand banking calendar (Wellington).

    Holidays: New Year's (1+2 Jan), Waitangi Day (6 Feb),
    Good Friday, Easter Monday, Anzac Day (25 Apr),
    Queen's Birthday (1st Mon Jun), Matariki (variable from 2022),
    Labour Day (4th Mon Oct), Christmas, Boxing Day.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(self._observe(date(year, 1, 1)))    # New Year's Day
        holidays.add(self._observe(date(year, 1, 2)))    # Day after New Year's
        holidays.add(self._observe(date(year, 2, 6)))    # Waitangi Day
        holidays.add(date(year, 4, 25))                   # Anzac Day

        holidays.add(self._nth_weekday(year, 6, 0, 1))   # Queen's Birthday (1st Mon Jun)
        holidays.add(self._nth_weekday(year, 10, 0, 4))  # Labour Day (4th Mon Oct)

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter + timedelta(days=1))    # Easter Monday

        obs_xmas = self._observe(date(year, 12, 25))
        obs_boxing = self._observe(date(year, 12, 26))
        holidays.add(obs_xmas)
        if obs_boxing == obs_xmas:
            holidays.add(obs_boxing + timedelta(days=1))
        else:
            holidays.add(obs_boxing)

        return holidays


# ═══════════════════════════════════════════════════════════════
# EM Calendars — CEE
# ═══════════════════════════════════════════════════════════════


class WarsawCalendar(Calendar):
    """Polish banking calendar (Warsaw / PLN).

    Holidays: New Year's, Epiphany, Easter Monday, May Day, Constitution Day
    (3 May), Corpus Christi, Assumption (15 Aug), All Saints' (1 Nov),
    Independence Day (11 Nov), Christmas (25-26 Dec).
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 1, 6))     # Epiphany
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 5, 3))     # Constitution Day
        holidays.add(date(year, 8, 15))    # Assumption
        holidays.add(date(year, 11, 1))    # All Saints'
        holidays.add(date(year, 11, 11))   # Independence Day
        holidays.add(date(year, 12, 25))   # Christmas
        holidays.add(date(year, 12, 26))   # St Stephen's

        easter = TARGETCalendar._easter(year)
        holidays.add(easter + timedelta(days=1))    # Easter Monday
        holidays.add(easter + timedelta(days=60))   # Corpus Christi

        return holidays


class PragueCalendar(Calendar):
    """Czech banking calendar (Prague / CZK).

    Holidays: New Year's/Restoration Day, Good Friday, Easter Monday,
    May Day, Liberation Day (8 May), Cyril & Methodius (5 Jul),
    Jan Hus Day (6 Jul), Statehood Day (28 Sep), Independence Day (28 Oct),
    Freedom & Democracy Day (17 Nov), Christmas Eve, Christmas (25-26 Dec).
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's / Restoration Day
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 5, 8))     # Liberation Day
        holidays.add(date(year, 7, 5))     # Cyril & Methodius
        holidays.add(date(year, 7, 6))     # Jan Hus Day
        holidays.add(date(year, 9, 28))    # Statehood Day
        holidays.add(date(year, 10, 28))   # Independence Day
        holidays.add(date(year, 11, 17))   # Freedom & Democracy
        holidays.add(date(year, 12, 24))   # Christmas Eve
        holidays.add(date(year, 12, 25))   # Christmas
        holidays.add(date(year, 12, 26))   # St Stephen's

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter + timedelta(days=1))    # Easter Monday

        return holidays


class BudapestCalendar(Calendar):
    """Hungarian banking calendar (Budapest / HUF).

    Holidays: New Year's, 1848 Revolution (15 Mar), Good Friday, Easter Monday,
    May Day, Whit Monday, St Stephen's (20 Aug), Republic Day (23 Oct),
    All Saints' (1 Nov), Christmas (25-26 Dec).
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 3, 15))    # 1848 Revolution
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 8, 20))    # St Stephen's / State Foundation
        holidays.add(date(year, 10, 23))   # Republic Day
        holidays.add(date(year, 11, 1))    # All Saints'
        holidays.add(date(year, 12, 25))   # Christmas
        holidays.add(date(year, 12, 26))   # Boxing Day

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter + timedelta(days=1))    # Easter Monday
        holidays.add(easter + timedelta(days=50))   # Whit Monday

        return holidays


class BucharestCalendar(Calendar):
    """Romanian banking calendar (Bucharest / RON).

    Holidays: New Year's (1-2 Jan), Unification Day (24 Jan),
    Orthodox Easter (Friday + Monday), May Day, Children's Day (1 Jun),
    Orthodox Whit Monday, Dormition (15 Aug), St Andrew (30 Nov),
    National Day (1 Dec), Christmas (25-26 Dec).

    Note: Romania uses Orthodox Easter which differs from Western Easter.
    We use the Julian calendar algorithm with Gregorian offset.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 1, 2))     # Day after New Year's
        holidays.add(date(year, 1, 24))    # Unification Day
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 6, 1))     # Children's Day
        holidays.add(date(year, 8, 15))    # Dormition of the Theotokos
        holidays.add(date(year, 11, 30))   # St Andrew
        holidays.add(date(year, 12, 1))    # National Day
        holidays.add(date(year, 12, 25))   # Christmas
        holidays.add(date(year, 12, 26))   # Boxing Day

        oe = _orthodox_easter(year)
        holidays.add(oe - timedelta(days=2))   # Orthodox Good Friday
        holidays.add(oe + timedelta(days=1))    # Orthodox Easter Monday
        holidays.add(oe + timedelta(days=50))   # Orthodox Whit Monday

        return holidays


# ═══════════════════════════════════════════════════════════════
# EM Calendars — Turkey & MENA
# ═══════════════════════════════════════════════════════════════


class IstanbulCalendar(Calendar):
    """Turkish banking calendar (Istanbul / TRY).

    Holidays: New Year's, National Sovereignty (23 Apr), May Day,
    Youth Day (19 May), Democracy Day (15 Jul), Victory Day (30 Aug),
    Republic Day (29 Oct).

    Note: Islamic holidays (Ramadan Bayram, Sacrifice Bayram) are variable
    and follow the lunar calendar. We include fixed secular holidays only.
    For production use, Islamic holiday dates should be loaded externally.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 4, 23))    # National Sovereignty & Children's Day
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 5, 19))    # Youth & Sports Day
        holidays.add(date(year, 7, 15))    # Democracy & National Unity Day
        holidays.add(date(year, 8, 30))    # Victory Day
        holidays.add(date(year, 10, 29))   # Republic Day
        return holidays


class RiyadhCalendar(Calendar):
    """Saudi banking calendar (Riyadh / SAR).

    Holidays: National Day (23 Sep), Founding Day (22 Feb, from 2022).

    Note: Eid al-Fitr (~6 days) and Eid al-Adha (~5 days) follow the
    Hijri lunar calendar. We include fixed secular holidays only.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 9, 23))    # National Day
        if year >= 2022:
            holidays.add(date(year, 2, 22))  # Founding Day
        return holidays


class TelAvivCalendar(Calendar):
    """Israeli banking calendar (Tel Aviv / ILS).

    Holidays based on Hebrew calendar — dates shift each Gregorian year.
    We include approximate fixed-date secular holidays. Production use
    should load precise Hebrew calendar dates externally.

    Fixed: Independence Day vicinity, Yom Kippur vicinity.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        # Israeli weekend: Friday-Saturday. Sunday is a business day.
        # We keep standard Sat-Sun weekend in the base class for simplicity.
        # The approximate holidays are placeholders.
        holidays.add(date(year, 4, 14))    # Passover vicinity
        holidays.add(date(year, 4, 20))    # Passover end vicinity
        holidays.add(date(year, 5, 2))     # Independence Day vicinity
        holidays.add(date(year, 9, 25))    # Rosh Hashanah vicinity
        holidays.add(date(year, 9, 26))    # Rosh Hashanah 2
        holidays.add(date(year, 10, 4))    # Yom Kippur vicinity
        holidays.add(date(year, 10, 9))    # Sukkot vicinity
        return holidays

    def is_weekend(self, d: date) -> bool:
        """Israel: Friday-Saturday weekend."""
        return d.weekday() in (4, 5)


class CairoCalendar(Calendar):
    """Egyptian banking calendar (Cairo / EGP).

    Fixed holidays: Revolution Day (25 Jan), Sinai Liberation (25 Apr),
    May Day, Revolution Day (23 Jul), Armed Forces Day (6 Oct).
    Islamic holidays are lunar and not included.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 7))     # Coptic Christmas
        holidays.add(date(year, 1, 25))    # Revolution Day
        holidays.add(date(year, 4, 25))    # Sinai Liberation
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 7, 23))    # Revolution Day
        holidays.add(date(year, 10, 6))    # Armed Forces Day
        return holidays


# ═══════════════════════════════════════════════════════════════
# EM Calendars — Africa
# ═══════════════════════════════════════════════════════════════


class JohannesburgCalendar(Calendar):
    """South African banking calendar (Johannesburg / ZAR).

    Holidays: New Year's, Human Rights Day (21 Mar), Good Friday, Family Day
    (Easter Monday), Freedom Day (27 Apr), Workers' Day (1 May),
    Youth Day (16 Jun), Women's Day (9 Aug), Heritage Day (24 Sep),
    Day of Reconciliation (16 Dec), Christmas, Day of Goodwill (26 Dec).
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(self._observe(date(year, 1, 1)))     # New Year's
        holidays.add(self._observe(date(year, 3, 21)))    # Human Rights Day
        holidays.add(self._observe(date(year, 4, 27)))    # Freedom Day
        holidays.add(self._observe(date(year, 5, 1)))     # Workers' Day
        holidays.add(self._observe(date(year, 6, 16)))    # Youth Day
        holidays.add(self._observe(date(year, 8, 9)))     # Women's Day
        holidays.add(self._observe(date(year, 9, 24)))    # Heritage Day
        holidays.add(self._observe(date(year, 12, 16)))   # Day of Reconciliation
        holidays.add(self._observe(date(year, 12, 25)))   # Christmas
        holidays.add(self._observe(date(year, 12, 26)))   # Day of Goodwill

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter + timedelta(days=1))    # Family Day (Easter Monday)

        return holidays

    @staticmethod
    def _observe(d: date) -> date:
        """South Africa: Sunday holidays move to Monday."""
        if d.weekday() == 6:
            return d + timedelta(days=1)
        return d


class NairobiCalendar(Calendar):
    """Kenyan banking calendar (Nairobi / KES).

    Holidays: New Year's, Good Friday, Easter Monday, May Day,
    Madaraka Day (1 Jun), Mashujaa Day (20 Oct), Jamhuri Day (12 Dec),
    Christmas, Boxing Day.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 6, 1))     # Madaraka Day
        holidays.add(date(year, 10, 20))   # Mashujaa Day
        holidays.add(date(year, 12, 12))   # Jamhuri Day
        holidays.add(date(year, 12, 25))   # Christmas
        holidays.add(date(year, 12, 26))   # Boxing Day

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter + timedelta(days=1))    # Easter Monday

        return holidays


class LagosCalendar(Calendar):
    """Nigerian banking calendar (Lagos / NGN).

    Fixed holidays: New Year's, May Day, Democracy Day (12 Jun),
    Independence Day (1 Oct), Christmas, Boxing Day.
    Good Friday, Easter Monday. Islamic holidays not included.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 6, 12))    # Democracy Day
        holidays.add(date(year, 10, 1))    # Independence Day
        holidays.add(date(year, 12, 25))   # Christmas
        holidays.add(date(year, 12, 26))   # Boxing Day

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter + timedelta(days=1))    # Easter Monday

        return holidays


# ═══════════════════════════════════════════════════════════════
# EM Calendars — Latin America
# ═══════════════════════════════════════════════════════════════


class SaoPauloCalendar(Calendar):
    """Brazilian banking calendar (São Paulo / BRL).

    Holidays: New Year's, Carnival (Mon-Tue before Ash Wednesday),
    Good Friday, Tiradentes (21 Apr), May Day, Corpus Christi,
    Independence Day (7 Sep), Our Lady Aparecida (12 Oct),
    All Souls' (2 Nov), Republic Day (15 Nov), Christmas.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 4, 21))    # Tiradentes
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 9, 7))     # Independence Day
        holidays.add(date(year, 10, 12))   # Our Lady Aparecida
        holidays.add(date(year, 11, 2))    # All Souls'
        holidays.add(date(year, 11, 15))   # Republic Day
        holidays.add(date(year, 12, 25))   # Christmas

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=48))  # Carnival Monday
        holidays.add(easter - timedelta(days=47))  # Carnival Tuesday
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter + timedelta(days=60))  # Corpus Christi

        return holidays


class MexicoCityCalendar(Calendar):
    """Mexican banking calendar (Mexico City / MXN).

    Holidays: New Year's, Constitution Day (1st Mon Feb), Benito Juárez
    (3rd Mon Mar), Maundy Thursday, Good Friday, May Day,
    Independence Day (16 Sep), Revolution Day (3rd Mon Nov),
    Presidential Inauguration (1 Dec every 6 years), Christmas.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(self._nth_weekday(year, 2, 0, 1))   # Constitution Day (1st Mon Feb)
        holidays.add(self._nth_weekday(year, 3, 0, 3))   # Benito Juárez (3rd Mon Mar)
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 9, 16))    # Independence Day
        holidays.add(self._nth_weekday(year, 11, 0, 3))  # Revolution Day (3rd Mon Nov)
        holidays.add(date(year, 12, 25))   # Christmas

        # Presidential inauguration every 6 years (2024, 2030, ...)
        if year >= 2024 and (year - 2024) % 6 == 0:
            holidays.add(date(year, 10, 1))  # Inauguration (moved to Oct 1 from 2024)

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=3))   # Maundy Thursday
        holidays.add(easter - timedelta(days=2))   # Good Friday

        return holidays


class SantiagoCalendar(Calendar):
    """Chilean banking calendar (Santiago / CLP).

    Holidays: New Year's, Good Friday, Easter Saturday, May Day,
    Navy Day (21 May), San Pedro & San Pablo (29 Jun),
    Our Lady of Carmen (16 Jul), Assumption (15 Aug),
    Independence Day (18 Sep), Army Day (19 Sep), Columbus Day (12 Oct),
    Reformation Day (31 Oct), All Saints' (1 Nov), Immaculate Conception (8 Dec),
    Christmas.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 5, 21))    # Navy Day
        holidays.add(date(year, 6, 29))    # San Pedro & San Pablo
        holidays.add(date(year, 7, 16))    # Our Lady of Carmen
        holidays.add(date(year, 8, 15))    # Assumption
        holidays.add(date(year, 9, 18))    # Independence Day
        holidays.add(date(year, 9, 19))    # Army Day
        holidays.add(date(year, 10, 12))   # Columbus Day
        holidays.add(date(year, 10, 31))   # Reformation Day
        holidays.add(date(year, 11, 1))    # All Saints'
        holidays.add(date(year, 12, 8))    # Immaculate Conception
        holidays.add(date(year, 12, 25))   # Christmas

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter - timedelta(days=1))   # Easter Saturday

        return holidays


class BogotaCalendar(Calendar):
    """Colombian banking calendar (Bogotá / COP).

    Holidays: New Year's, Epiphany (Mon), St Joseph (Mon),
    Maundy Thursday, Good Friday, May Day, Ascension (Mon),
    Corpus Christi (Mon), Sacred Heart (Mon),
    St Peter & St Paul (Mon), Independence Day (20 Jul),
    Battle of Boyacá (7 Aug), Assumption (Mon),
    Columbus Day (Mon), All Saints' (Mon),
    Independence of Cartagena (Mon), Immaculate Conception (8 Dec),
    Christmas.

    Colombia's "emiliani" law moves many holidays to Monday.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(self._next_monday(date(year, 1, 6)))    # Epiphany
        holidays.add(self._next_monday(date(year, 3, 19)))   # St Joseph
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(self._next_monday(date(year, 6, 29)))   # St Peter & St Paul
        holidays.add(date(year, 7, 20))    # Independence Day
        holidays.add(date(year, 8, 7))     # Battle of Boyacá
        holidays.add(self._next_monday(date(year, 8, 15)))   # Assumption
        holidays.add(self._next_monday(date(year, 10, 12)))  # Columbus Day
        holidays.add(self._next_monday(date(year, 11, 1)))   # All Saints'
        holidays.add(self._next_monday(date(year, 11, 11)))  # Cartagena Independence
        holidays.add(date(year, 12, 8))    # Immaculate Conception
        holidays.add(date(year, 12, 25))   # Christmas

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=3))   # Maundy Thursday
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(self._next_monday(easter + timedelta(days=43)))   # Ascension
        holidays.add(self._next_monday(easter + timedelta(days=64)))   # Corpus Christi
        holidays.add(self._next_monday(easter + timedelta(days=71)))   # Sacred Heart

        return holidays

    @staticmethod
    def _next_monday(d: date) -> date:
        """Move to next Monday if not already Monday (emiliani law)."""
        if d.weekday() == 0:
            return d
        return d + timedelta(days=(7 - d.weekday()))


# ═══════════════════════════════════════════════════════════════
# EM Calendars — Asia
# ═══════════════════════════════════════════════════════════════


class BeijingCalendar(Calendar):
    """Chinese banking calendar (Beijing / CNY).

    Fixed holidays: New Year's, Qingming (5 Apr approx), May Day,
    National Day (1-3 Oct).

    Note: Chinese New Year (~Jan/Feb) and Mid-Autumn Festival follow
    the lunar calendar. Exact dates shift yearly and should be loaded
    externally for production. We include fixed secular holidays only.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 4, 5))     # Qingming (Tomb Sweeping)
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 10, 1))    # National Day
        holidays.add(date(year, 10, 2))    # National Day
        holidays.add(date(year, 10, 3))    # National Day
        return holidays


class SeoulCalendar(Calendar):
    """South Korean banking calendar (Seoul / KRW).

    Fixed holidays: New Year's, Independence Movement (1 Mar), Children's Day
    (5 May), Memorial Day (6 Jun), Liberation Day (15 Aug),
    National Foundation (3 Oct), Hangeul Day (9 Oct), Christmas.

    Note: Seollal (Lunar New Year) and Chuseok (Mid-Autumn) follow the
    lunar calendar and are not included. Load externally for production.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 3, 1))     # Independence Movement Day
        holidays.add(date(year, 5, 5))     # Children's Day
        holidays.add(date(year, 6, 6))     # Memorial Day
        holidays.add(date(year, 8, 15))    # Liberation Day
        holidays.add(date(year, 10, 3))    # National Foundation Day
        holidays.add(date(year, 10, 9))    # Hangeul Day
        holidays.add(date(year, 12, 25))   # Christmas
        return holidays


class MumbaiCalendar(Calendar):
    """Indian banking calendar (Mumbai / INR).

    Fixed holidays: Republic Day (26 Jan), Independence Day (15 Aug),
    Gandhi Jayanti (2 Oct), Christmas.

    Note: Holi, Diwali, Eid, Guru Nanak Jayanti, etc. follow lunar/religious
    calendars and shift yearly. Not included — load externally for production.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 26))    # Republic Day
        holidays.add(date(year, 8, 15))    # Independence Day
        holidays.add(date(year, 10, 2))    # Gandhi Jayanti
        holidays.add(date(year, 12, 25))   # Christmas
        return holidays


class SingaporeCalendar(Calendar):
    """Singapore banking calendar (SGD).

    Fixed holidays: New Year's, May Day, National Day (9 Aug), Christmas.
    Good Friday.

    Note: Chinese New Year, Deepavali, Hari Raya Puasa/Haji follow
    lunar/Islamic calendars. Not included.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 8, 9))     # National Day
        holidays.add(date(year, 12, 25))   # Christmas

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))   # Good Friday

        return holidays


class HongKongCalendar(Calendar):
    """Hong Kong banking calendar (HKD).

    Fixed holidays: New Year's, Good Friday, Easter Saturday, Easter Monday,
    May Day, Tuen Ng vicinity (Jun), HKSAR Day (1 Jul), National Day (1 Oct),
    Christmas, Boxing Day.

    Note: Chinese New Year, Ching Ming, Mid-Autumn, Chung Yeung follow
    lunar calendar. Not included.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 7, 1))     # HKSAR Day
        holidays.add(date(year, 10, 1))    # National Day
        holidays.add(date(year, 12, 25))   # Christmas
        holidays.add(date(year, 12, 26))   # Boxing Day

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter - timedelta(days=1))   # Easter Saturday
        holidays.add(easter + timedelta(days=1))    # Easter Monday

        return holidays


class JakartaCalendar(Calendar):
    """Indonesian banking calendar (Jakarta / IDR).

    Fixed holidays: New Year's, May Day, Pancasila Day (1 Jun),
    Independence Day (17 Aug), Christmas.

    Note: Nyepi, Waisak, Isra Mi'raj, Eid al-Fitr, Eid al-Adha, Mawlid,
    Chinese New Year follow religious/lunar calendars. Not included.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 6, 1))     # Pancasila Day
        holidays.add(date(year, 8, 17))    # Independence Day
        holidays.add(date(year, 12, 25))   # Christmas

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter + timedelta(days=39))  # Ascension

        return holidays


class KualaLumpurCalendar(Calendar):
    """Malaysian banking calendar (Kuala Lumpur / MYR).

    Fixed holidays: New Year's, Federal Territory Day (1 Feb),
    May Day, Yang di-Pertuan Agong Birthday (1st Mon Jun),
    Malaysia Day (16 Sep), Christmas.

    Note: Chinese New Year, Thaipusam, Nuzul Quran, Hari Raya Aidilfitri/Haji,
    Deepavali, Mawlid follow lunar/religious calendars. Not included.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 2, 1))     # Federal Territory Day
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(self._nth_weekday(year, 6, 0, 1))   # Agong Birthday (1st Mon Jun)
        holidays.add(date(year, 8, 31))    # Merdeka Day
        holidays.add(date(year, 9, 16))    # Malaysia Day
        holidays.add(date(year, 12, 25))   # Christmas
        return holidays


class BangkokCalendar(Calendar):
    """Thai banking calendar (Bangkok / THB).

    Fixed holidays: New Year's, Chakri Day (6 Apr), Songkran (13-15 Apr),
    May Day, King's Birthday (28 Jul), Queen's Birthday (12 Aug),
    Chulalongkorn Day (23 Oct), King Bhumibol Day (5 Dec),
    Constitution Day (10 Dec), New Year's Eve.

    Note: Makha Bucha, Visakha Bucha, Asanha Bucha follow lunar calendar.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 4, 6))     # Chakri Memorial Day
        holidays.add(date(year, 4, 13))    # Songkran
        holidays.add(date(year, 4, 14))    # Songkran
        holidays.add(date(year, 4, 15))    # Songkran
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 7, 28))    # King's Birthday
        holidays.add(date(year, 8, 12))    # Queen's Birthday
        holidays.add(date(year, 10, 23))   # Chulalongkorn Day
        holidays.add(date(year, 12, 5))    # King Bhumibol Birthday
        holidays.add(date(year, 12, 10))   # Constitution Day
        holidays.add(date(year, 12, 31))   # New Year's Eve
        return holidays


class ManilaCalendar(Calendar):
    """Philippine banking calendar (Manila / PHP).

    Fixed holidays: New Year's, Maundy Thursday, Good Friday,
    Araw ng Kagitingan (9 Apr), May Day, Independence Day (12 Jun),
    National Heroes Day (last Mon Aug), Bonifacio Day (30 Nov),
    Christmas Eve, Christmas, Rizal Day (30 Dec), New Year's Eve.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 4, 9))     # Araw ng Kagitingan
        holidays.add(date(year, 5, 1))     # May Day
        holidays.add(date(year, 6, 12))    # Independence Day
        holidays.add(self._last_weekday(year, 8, 0))  # National Heroes Day (last Mon Aug)
        holidays.add(date(year, 11, 30))   # Bonifacio Day
        holidays.add(date(year, 12, 24))   # Christmas Eve
        holidays.add(date(year, 12, 25))   # Christmas
        holidays.add(date(year, 12, 30))   # Rizal Day
        holidays.add(date(year, 12, 31))   # New Year's Eve

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=3))   # Maundy Thursday
        holidays.add(easter - timedelta(days=2))   # Good Friday

        return holidays


class DenmarkCalendar(Calendar):
    """Danish banking calendar (Copenhagen / DKK).

    Holidays: New Year's, Maundy Thursday, Good Friday, Easter Monday,
    Great Prayer Day (4th Fri after Easter, until 2023), Ascension,
    Whit Monday, Constitution Day (5 Jun), Christmas (24-26 Dec).
    """

    def _compute_holidays(self, year: int) -> set[date]:
        holidays = set()
        holidays.add(date(year, 1, 1))     # New Year's
        holidays.add(date(year, 6, 5))     # Constitution Day
        holidays.add(date(year, 12, 24))   # Christmas Eve
        holidays.add(date(year, 12, 25))   # Christmas
        holidays.add(date(year, 12, 26))   # Boxing Day
        holidays.add(date(year, 12, 31))   # New Year's Eve

        easter = TARGETCalendar._easter(year)
        holidays.add(easter - timedelta(days=3))   # Maundy Thursday
        holidays.add(easter - timedelta(days=2))   # Good Friday
        holidays.add(easter + timedelta(days=1))    # Easter Monday
        if year <= 2023:
            holidays.add(easter + timedelta(days=26))  # Great Prayer Day (Store Bededag)
        holidays.add(easter + timedelta(days=39))   # Ascension
        holidays.add(easter + timedelta(days=50))   # Whit Monday

        return holidays


# ═══════════════════════════════════════════════════════════════
# Orthodox Easter (used by Romania, etc.)
# ═══════════════════════════════════════════════════════════════


def _orthodox_easter(year: int) -> date:
    """Compute Orthodox Easter Sunday (Julian algorithm + Gregorian offset).

    The Julian Easter is computed, then converted to Gregorian by adding
    the century offset (13 days for 1900-2099).
    """
    a = year % 4
    b = year % 7
    c = year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month = (d + e + 114) // 31
    day = ((d + e + 114) % 31) + 1
    julian = date(year, month, day)
    # Gregorian offset: 13 days for 1900-2099
    return julian + timedelta(days=13)


# ═══════════════════════════════════════════════════════════════
# Joint Calendar
# ═══════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════
# Calendar Registry
# ═══════════════════════════════════════════════════════════════


_CALENDAR_REGISTRY: dict[str, type[Calendar]] = {
    # G10
    "USD": USSettlementCalendar,
    "EUR": TARGETCalendar,
    "GBP": LondonCalendar,
    "JPY": TokyoCalendar,
    "CHF": CHFCalendar,
    "AUD": AUDCalendar,
    "CAD": CADCalendar,
    "SEK": SEKCalendar,
    "NOK": NOKCalendar,
    "NZD": NZDCalendar,
    "DKK": DenmarkCalendar,
    # CEE
    "PLN": WarsawCalendar,
    "CZK": PragueCalendar,
    "HUF": BudapestCalendar,
    "RON": BucharestCalendar,
    # Turkey & MENA
    "TRY": IstanbulCalendar,
    "SAR": RiyadhCalendar,
    "ILS": TelAvivCalendar,
    "EGP": CairoCalendar,
    # Africa
    "ZAR": JohannesburgCalendar,
    "KES": NairobiCalendar,
    "NGN": LagosCalendar,
    # LatAm
    "BRL": SaoPauloCalendar,
    "MXN": MexicoCityCalendar,
    "CLP": SantiagoCalendar,
    "COP": BogotaCalendar,
    # Asia
    "CNY": BeijingCalendar,
    "KRW": SeoulCalendar,
    "INR": MumbaiCalendar,
    "SGD": SingaporeCalendar,
    "HKD": HongKongCalendar,
    "IDR": JakartaCalendar,
    "MYR": KualaLumpurCalendar,
    "THB": BangkokCalendar,
    "PHP": ManilaCalendar,
}


def get_calendar(currency_code: str) -> Calendar:
    """Get a calendar instance by ISO currency code.

    Args:
        currency_code: 3-letter ISO currency code (e.g. "USD", "BRL", "INR").

    Returns:
        A new Calendar instance for that currency.

    Raises:
        ValueError: if no calendar is registered for the given code.
    """
    code = currency_code.upper()
    cls = _CALENDAR_REGISTRY.get(code)
    if cls is None:
        available = sorted(_CALENDAR_REGISTRY.keys())
        raise ValueError(f"No calendar for {code!r}. Available: {available}")
    return cls()


def list_calendars() -> list[str]:
    """Return sorted list of available currency codes with calendars."""
    return sorted(_CALENDAR_REGISTRY.keys())
