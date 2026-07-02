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
        """US-style observation: Saturday → previous Friday, Sunday → next Monday.

        Matches 5 U.S.C. § 6103 for US federal holidays. Subclasses that
        follow a different national rule should override this — see
        `_observe_next_working_day` for the UK / AU / NZ / CA convention.
        """
        if d.weekday() == 5:
            return d - timedelta(days=1)
        if d.weekday() == 6:
            return d + timedelta(days=1)
        return d

    @staticmethod
    def _observe_next_working_day(d: date) -> date:
        """UK / AU / NZ / CA convention: Saturday OR Sunday → next Monday.

        Codified in:
        - UK Banking and Financial Dealings Act 1971
        - Australian Public Holidays Acts (state-by-state, but uniform Sat→Mon)
        - New Zealand Holidays Act 2003
        - Canadian federal/provincial bank holiday acts

        Differs from the US-style `_observe` for Saturday holidays only —
        US substitutes a Saturday holiday to the *previous* Friday, while
        the Commonwealth rule substitutes to the *next* Monday. Sunday
        substitution is the same (next Monday) in both.

        Skipping consecutive holidays (e.g. when Christmas Sat → Mon and
        Boxing Sun → also Mon, they collide on Monday and Boxing must
        bump to Tuesday) is handled at the per-holiday level by callers,
        not here.
        """
        if d.weekday() == 5:
            return d + timedelta(days=2)
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


# ═══════════════════════════════════════════════════════════════
# Easter algorithms (Western Gregorian + Orthodox)
# ═══════════════════════════════════════════════════════════════


def _gregorian_easter(year: int) -> date:
    """Western Easter Sunday (anonymous Gregorian algorithm)."""
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


def _orthodox_easter(year: int) -> date:
    """Orthodox Easter Sunday (Julian algorithm + Gregorian offset).

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
# Holiday rule DSL — a rule is a callable (cal, year) -> Iterable[date]
# consumed by SpecCalendar. This replaces ~35 near-identical
# _compute_holidays methods with declarative HOLIDAYS lists.
# ═══════════════════════════════════════════════════════════════


def _in_range(year: int, since: int | None, until: int | None) -> bool:
    return (since is None or year >= since) and (until is None or year <= until)


def fixed(month: int, day: int, *, observe: bool = False, since=None, until=None):
    """A fixed (month, day) holiday; optionally observed / year-gated."""

    def rule(cal, year):
        if not _in_range(year, since, until):
            return ()
        d = date(year, month, day)
        return (cal._observe(d) if observe else d,)

    return rule


def easter(offset: int, *, since=None, until=None):
    """A Western-Easter-relative holiday: Easter Sunday + `offset` days."""

    def rule(cal, year):
        if not _in_range(year, since, until):
            return ()
        return (_gregorian_easter(year) + timedelta(days=offset),)

    return rule


def orthodox(offset: int):
    """An Orthodox-Easter-relative holiday: Orthodox Easter + `offset` days."""

    def rule(cal, year):
        return (_orthodox_easter(year) + timedelta(days=offset),)

    return rule


def nth(month: int, weekday: int, n: int):
    """The nth (n>0) or last (n=-1) `weekday` of `month` (weekday: 0=Mon)."""

    def rule(cal, year):
        if n == -1:
            return (cal._last_weekday(year, month, weekday),)
        return (cal._nth_weekday(year, month, weekday, n),)

    return rule


def _to_next_monday(d: date) -> date:
    """Move to the next Monday unless already Monday (Colombia emiliani law)."""
    if d.weekday() == 0:
        return d
    return d + timedelta(days=(7 - d.weekday()))


def monday(rule):
    """Modifier: shift every date produced by `rule` to the next Monday."""

    def wrapped(cal, year):
        return tuple(_to_next_monday(d) for d in rule(cal, year))

    return wrapped


def christmas_boxing(cal, year):
    """Christmas + Boxing Day under the observe rule, resolving the collision:
    when Dec 25 is Sunday both observe to Dec 26, so Boxing bumps to Dec 27."""
    obs_xmas = cal._observe(date(year, 12, 25))
    obs_boxing = cal._observe(date(year, 12, 26))
    if obs_boxing == obs_xmas:
        return (obs_xmas, obs_boxing + timedelta(days=1))
    return (obs_xmas, obs_boxing)


def victoria_day(cal, year):
    """Canadian Victoria Day: the Monday before May 25."""
    may25 = date(year, 5, 25)
    days_since_mon = (may25.weekday() - 0) % 7
    if days_since_mon == 0:
        days_since_mon = 7
    return (may25 - timedelta(days=days_since_mon),)


def midsummer_eve(cal, year):
    """Swedish Midsummer Eve: the Friday before the Saturday in Jun 20-26."""
    for d in range(20, 27):
        candidate = date(year, 6, d)
        if candidate.weekday() == 5:  # Saturday
            return (candidate - timedelta(days=1),)
    return ()


def mexico_inauguration(cal, year):
    """Mexican Presidential Inauguration: Oct 1 every 6 years from 2024."""
    if year >= 2024 and (year - 2024) % 6 == 0:
        return (date(year, 10, 1),)
    return ()


class SpecCalendar(Calendar):
    """A calendar defined declaratively by a list of holiday rules.

    Subclasses set ``HOLIDAYS`` (a list of rules from the DSL above) and, if the
    national substitution rule is not US-style, override ``_observe``.
    """

    HOLIDAYS: list = []

    def _compute_holidays(self, year: int) -> set[date]:
        out: set[date] = set()
        for rule in self.HOLIDAYS:
            out.update(rule(self, year))
        return out


# ═══════════════════════════════════════════════════════════════
# G10
# ═══════════════════════════════════════════════════════════════


class USSettlementCalendar(SpecCalendar):
    """US Settlement calendar (SIFMA/Federal Reserve)."""

    HOLIDAYS = [
        fixed(1, 1, observe=True),      # New Year's
        nth(1, 0, 3),                   # MLK Day
        nth(2, 0, 3),                   # Presidents' Day
        nth(5, 0, -1),                  # Memorial Day (last Mon May)
        fixed(6, 19, observe=True, since=2021),  # Juneteenth
        fixed(7, 4, observe=True),      # Independence Day
        nth(9, 0, 1),                   # Labor Day
        nth(10, 0, 2),                  # Columbus Day
        fixed(11, 11, observe=True),    # Veterans Day
        nth(11, 3, 4),                  # Thanksgiving (4th Thu)
        fixed(12, 25, observe=True),    # Christmas
    ]


class TARGETCalendar(SpecCalendar):
    """TARGET calendar (EUR): New Year's, Good Friday, Easter Monday, 1 May, 25-26 Dec."""

    HOLIDAYS = [fixed(1, 1), fixed(5, 1), fixed(12, 25), fixed(12, 26), easter(-2), easter(1)]


class LondonCalendar(SpecCalendar):
    """London (UK) banking calendar. Commonwealth Sat/Sun → next-working-day rule."""

    _observe = staticmethod(Calendar._observe_next_working_day)
    HOLIDAYS = [
        fixed(1, 1, observe=True),
        easter(-2), easter(1),
        nth(5, 0, 1),    # Early May
        nth(5, 0, -1),   # Spring Bank Holiday
        nth(8, 0, -1),   # Summer Bank Holiday
        christmas_boxing,
    ]


class TokyoCalendar(Calendar):
    """Tokyo (Japan) banking calendar.

    Kept bespoke: the *furikae kyūjitsu* substitution (a fixed holiday on Sunday
    walks forward past consecutive holidays) is genuinely special and doesn't
    reduce to the rule DSL.
    """

    def _compute_holidays(self, year: int) -> set[date]:
        # Fixed-date holidays.
        fixed_days = {
            date(year, 1, 1),     # New Year's
            date(year, 1, 2),     # bridging
            date(year, 1, 3),     # bridging
            date(year, 2, 11),    # National Foundation
            date(year, 2, 23),    # Emperor's Birthday (from 2020)
            date(year, 3, 21),    # Vernal Equinox (approx)
            date(year, 4, 29),    # Showa Day
            date(year, 5, 3),     # Constitution
            date(year, 5, 4),     # Greenery
            date(year, 5, 5),     # Children's
            date(year, 8, 11),    # Mountain Day
            date(year, 9, 23),    # Autumnal Equinox (approx)
            date(year, 11, 3),    # Culture Day
            date(year, 11, 23),   # Labour Thanksgiving
        }
        # Variable-Monday holidays (cannot fall on Sunday by construction).
        monday_holidays = {
            self._nth_weekday(year, 1, 0, 2),    # Coming of Age (2nd Mon Jan)
            self._nth_weekday(year, 7, 0, 3),    # Marine Day (3rd Mon Jul)
            self._nth_weekday(year, 9, 0, 3),    # Respect for Aged (3rd Mon Sep)
            self._nth_weekday(year, 10, 0, 2),   # Sports Day (2nd Mon Oct)
        }
        holidays = fixed_days | monday_holidays

        # Furikae kyūjitsu: for every fixed holiday on Sunday, walk forward
        # to the first non-holiday day.
        substitutes: set[date] = set()
        for d in fixed_days:
            if d.weekday() == 6:  # Sunday
                candidate = d + timedelta(days=1)
                while candidate in holidays or candidate in substitutes:
                    candidate = candidate + timedelta(days=1)
                substitutes.add(candidate)

        return holidays | substitutes


class CHFCalendar(SpecCalendar):
    """Swiss banking calendar (Zurich)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(1, 2), fixed(5, 1), fixed(8, 1), fixed(12, 25), fixed(12, 26),
        easter(-2), easter(1), easter(39), easter(50),
    ]


class AUDCalendar(SpecCalendar):
    """Australian banking calendar (Sydney). Sat/Sun → next working day."""

    _observe = staticmethod(Calendar._observe_next_working_day)
    HOLIDAYS = [
        fixed(1, 1, observe=True), fixed(1, 26, observe=True), fixed(4, 25),
        nth(6, 0, 2),    # Queen's Birthday (2nd Mon Jun)
        nth(8, 0, 1),    # Bank Holiday (1st Mon Aug)
        christmas_boxing,
        easter(-2), easter(-1), easter(1),
    ]


class CADCalendar(SpecCalendar):
    """Canadian banking calendar (Toronto). Sat/Sun → next working day."""

    _observe = staticmethod(Calendar._observe_next_working_day)
    HOLIDAYS = [
        fixed(1, 1, observe=True),
        nth(2, 0, 3),    # Family Day (3rd Mon Feb)
        easter(-2),
        victoria_day,
        fixed(7, 1, observe=True),
        nth(8, 0, 1),    # Civic Holiday
        nth(9, 0, 1),    # Labour Day
        nth(10, 0, 2),   # Thanksgiving
        fixed(11, 11, observe=True),   # Remembrance Day
        christmas_boxing,
    ]


class SEKCalendar(SpecCalendar):
    """Swedish banking calendar (Stockholm)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(1, 6), fixed(5, 1), fixed(6, 6),
        easter(-2), easter(1), easter(39),
        midsummer_eve,
        fixed(12, 24), fixed(12, 25), fixed(12, 26), fixed(12, 31),
    ]


class NOKCalendar(SpecCalendar):
    """Norwegian banking calendar (Oslo)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(5, 1), fixed(5, 17),
        easter(-3), easter(-2), easter(1), easter(39), easter(50),
        fixed(12, 25), fixed(12, 26),
    ]


class NZDCalendar(SpecCalendar):
    """New Zealand banking calendar (Wellington). Sat/Sun → next working day."""

    _observe = staticmethod(Calendar._observe_next_working_day)
    HOLIDAYS = [
        fixed(1, 1, observe=True), fixed(1, 2, observe=True), fixed(2, 6, observe=True),
        fixed(4, 25),
        nth(6, 0, 1),    # Queen's Birthday (1st Mon Jun)
        nth(10, 0, 4),   # Labour Day (4th Mon Oct)
        easter(-2), easter(1),
        christmas_boxing,
    ]


class DenmarkCalendar(SpecCalendar):
    """Danish banking calendar (Copenhagen)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(6, 5), fixed(12, 24), fixed(12, 25), fixed(12, 26), fixed(12, 31),
        easter(-3), easter(-2), easter(1),
        easter(26, until=2023),   # Great Prayer Day (Store Bededag), abolished 2024
        easter(39), easter(50),
    ]


# ═══════════════════════════════════════════════════════════════
# EM Calendars — CEE
# ═══════════════════════════════════════════════════════════════


class WarsawCalendar(SpecCalendar):
    """Polish banking calendar (Warsaw / PLN)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(1, 6), fixed(5, 1), fixed(5, 3), fixed(8, 15), fixed(11, 1),
        fixed(11, 11), fixed(12, 25), fixed(12, 26),
        easter(1), easter(60),
    ]


class PragueCalendar(SpecCalendar):
    """Czech banking calendar (Prague / CZK)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(5, 1), fixed(5, 8), fixed(7, 5), fixed(7, 6), fixed(9, 28),
        fixed(10, 28), fixed(11, 17), fixed(12, 24), fixed(12, 25), fixed(12, 26),
        easter(-2), easter(1),
    ]


class BudapestCalendar(SpecCalendar):
    """Hungarian banking calendar (Budapest / HUF)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(3, 15), fixed(5, 1), fixed(8, 20), fixed(10, 23), fixed(11, 1),
        fixed(12, 25), fixed(12, 26),
        easter(-2), easter(1), easter(50),
    ]


class BucharestCalendar(SpecCalendar):
    """Romanian banking calendar (Bucharest / RON). Uses Orthodox Easter."""

    HOLIDAYS = [
        fixed(1, 1), fixed(1, 2), fixed(1, 24), fixed(5, 1), fixed(6, 1), fixed(8, 15),
        fixed(11, 30), fixed(12, 1), fixed(12, 25), fixed(12, 26),
        orthodox(-2), orthodox(1), orthodox(50),
    ]


# ═══════════════════════════════════════════════════════════════
# EM Calendars — Turkey & MENA
# ═══════════════════════════════════════════════════════════════


class IstanbulCalendar(SpecCalendar):
    """Turkish banking calendar (Istanbul / TRY). Fixed secular holidays only."""

    HOLIDAYS = [
        fixed(1, 1), fixed(4, 23), fixed(5, 1), fixed(5, 19), fixed(7, 15), fixed(8, 30),
        fixed(10, 29),
    ]


class RiyadhCalendar(SpecCalendar):
    """Saudi banking calendar (Riyadh / SAR). Fixed secular holidays only."""

    HOLIDAYS = [fixed(9, 23), fixed(2, 22, since=2022)]


class TelAvivCalendar(SpecCalendar):
    """Israeli banking calendar (Tel Aviv / ILS). Approximate fixed placeholders."""

    HOLIDAYS = [
        fixed(4, 14), fixed(4, 20), fixed(5, 2), fixed(9, 25), fixed(9, 26), fixed(10, 4),
        fixed(10, 9),
    ]

    def is_weekend(self, d: date) -> bool:
        """Israel: Friday-Saturday weekend."""
        return d.weekday() in (4, 5)


class CairoCalendar(SpecCalendar):
    """Egyptian banking calendar (Cairo / EGP). Fixed secular holidays only."""

    HOLIDAYS = [
        fixed(1, 7), fixed(1, 25), fixed(4, 25), fixed(5, 1), fixed(7, 23), fixed(10, 6),
    ]


# ═══════════════════════════════════════════════════════════════
# EM Calendars — Africa
# ═══════════════════════════════════════════════════════════════


class JohannesburgCalendar(SpecCalendar):
    """South African banking calendar (Johannesburg / ZAR). Sunday → Monday only."""

    @staticmethod
    def _observe(d: date) -> date:
        if d.weekday() == 6:
            return d + timedelta(days=1)
        return d

    HOLIDAYS = [
        fixed(1, 1, observe=True), fixed(3, 21, observe=True), fixed(4, 27, observe=True),
        fixed(5, 1, observe=True), fixed(6, 16, observe=True), fixed(8, 9, observe=True),
        fixed(9, 24, observe=True), fixed(12, 16, observe=True), fixed(12, 25, observe=True),
        fixed(12, 26, observe=True),
        easter(-2), easter(1),   # Good Friday, Family Day (Easter Monday) — unobserved
    ]


class NairobiCalendar(SpecCalendar):
    """Kenyan banking calendar (Nairobi / KES)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(5, 1), fixed(6, 1), fixed(10, 20), fixed(12, 12), fixed(12, 25),
        fixed(12, 26),
        easter(-2), easter(1),
    ]


class LagosCalendar(SpecCalendar):
    """Nigerian banking calendar (Lagos / NGN)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(5, 1), fixed(6, 12), fixed(10, 1), fixed(12, 25), fixed(12, 26),
        easter(-2), easter(1),
    ]


# ═══════════════════════════════════════════════════════════════
# EM Calendars — Latin America
# ═══════════════════════════════════════════════════════════════


class SaoPauloCalendar(SpecCalendar):
    """Brazilian banking calendar (São Paulo / BRL)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(4, 21), fixed(5, 1), fixed(9, 7), fixed(10, 12), fixed(11, 2),
        fixed(11, 15), fixed(12, 25),
        easter(-48), easter(-47),   # Carnival Monday, Tuesday
        easter(-2), easter(60),     # Good Friday, Corpus Christi
    ]


class MexicoCityCalendar(SpecCalendar):
    """Mexican banking calendar (Mexico City / MXN)."""

    HOLIDAYS = [
        fixed(1, 1),
        nth(2, 0, 1),    # Constitution Day (1st Mon Feb)
        nth(3, 0, 3),    # Benito Juárez (3rd Mon Mar)
        fixed(5, 1), fixed(9, 16),
        nth(11, 0, 3),   # Revolution Day (3rd Mon Nov)
        fixed(12, 25),
        mexico_inauguration,
        easter(-3), easter(-2),   # Maundy Thursday, Good Friday
    ]


class SantiagoCalendar(SpecCalendar):
    """Chilean banking calendar (Santiago / CLP)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(5, 1), fixed(5, 21), fixed(6, 29), fixed(7, 16), fixed(8, 15),
        fixed(9, 18), fixed(9, 19), fixed(10, 12), fixed(10, 31), fixed(11, 1), fixed(12, 8),
        fixed(12, 25),
        easter(-2), easter(-1),   # Good Friday, Easter Saturday
    ]


class BogotaCalendar(SpecCalendar):
    """Colombian banking calendar (Bogotá / COP). Emiliani law → many holidays to Monday."""

    HOLIDAYS = [
        fixed(1, 1),
        monday(fixed(1, 6)),    # Epiphany
        monday(fixed(3, 19)),   # St Joseph
        fixed(5, 1),
        monday(fixed(6, 29)),   # St Peter & St Paul
        fixed(7, 20), fixed(8, 7),
        monday(fixed(8, 15)),   # Assumption
        monday(fixed(10, 12)),  # Columbus Day
        monday(fixed(11, 1)),   # All Saints'
        monday(fixed(11, 11)),  # Cartagena Independence
        fixed(12, 8), fixed(12, 25),
        easter(-3), easter(-2),          # Maundy Thursday, Good Friday
        monday(easter(43)),              # Ascension
        monday(easter(64)),              # Corpus Christi
        monday(easter(71)),              # Sacred Heart
    ]


class LimaCalendar(SpecCalendar):
    """Peruvian banking calendar (Lima / PEN)."""

    HOLIDAYS = [
        fixed(1, 1), easter(-3), easter(-2), fixed(5, 1), fixed(6, 29), fixed(7, 28),
        fixed(7, 29), fixed(8, 30), fixed(10, 8), fixed(11, 1), fixed(12, 8), fixed(12, 25),
    ]


class BuenosAiresCalendar(SpecCalendar):
    """Argentine banking calendar (Buenos Aires / ARS)."""

    HOLIDAYS = [
        fixed(1, 1),
        easter(-48), easter(-47), easter(-2),   # Carnival Mon/Tue, Good Friday
        fixed(3, 24), fixed(4, 2), fixed(5, 1), fixed(5, 25), fixed(6, 20), fixed(7, 9),
        fixed(8, 17), fixed(10, 12), fixed(11, 20), fixed(12, 8), fixed(12, 25),
    ]


# ═══════════════════════════════════════════════════════════════
# EM Calendars — Asia
# ═══════════════════════════════════════════════════════════════


class BeijingCalendar(SpecCalendar):
    """Chinese banking calendar (Beijing / CNY). Fixed secular holidays only."""

    HOLIDAYS = [fixed(1, 1), fixed(4, 5), fixed(5, 1), fixed(10, 1), fixed(10, 2), fixed(10, 3)]


class SeoulCalendar(SpecCalendar):
    """South Korean banking calendar (Seoul / KRW). Fixed secular holidays only."""

    HOLIDAYS = [
        fixed(1, 1), fixed(3, 1), fixed(5, 5), fixed(6, 6), fixed(8, 15), fixed(10, 3),
        fixed(10, 9), fixed(12, 25),
    ]


class MumbaiCalendar(SpecCalendar):
    """Indian banking calendar (Mumbai / INR). Fixed secular holidays only."""

    HOLIDAYS = [fixed(1, 26), fixed(8, 15), fixed(10, 2), fixed(12, 25)]


class SingaporeCalendar(SpecCalendar):
    """Singapore banking calendar (SGD). Fixed + Good Friday."""

    HOLIDAYS = [fixed(1, 1), fixed(5, 1), fixed(8, 9), fixed(12, 25), easter(-2)]


class HongKongCalendar(SpecCalendar):
    """Hong Kong banking calendar (HKD)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(5, 1), fixed(7, 1), fixed(10, 1), fixed(12, 25), fixed(12, 26),
        easter(-2), easter(-1), easter(1),   # Good Friday, Easter Saturday, Easter Monday
    ]


class JakartaCalendar(SpecCalendar):
    """Indonesian banking calendar (Jakarta / IDR)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(5, 1), fixed(6, 1), fixed(8, 17), fixed(12, 25),
        easter(-2), easter(39),   # Good Friday, Ascension
    ]


class KualaLumpurCalendar(SpecCalendar):
    """Malaysian banking calendar (Kuala Lumpur / MYR)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(2, 1), fixed(5, 1),
        nth(6, 0, 1),    # Agong Birthday (1st Mon Jun)
        fixed(8, 31), fixed(9, 16), fixed(12, 25),
    ]


class BangkokCalendar(SpecCalendar):
    """Thai banking calendar (Bangkok / THB)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(4, 6), fixed(4, 13), fixed(4, 14), fixed(4, 15), fixed(5, 1),
        fixed(7, 28), fixed(8, 12), fixed(10, 23), fixed(12, 5), fixed(12, 10), fixed(12, 31),
    ]


class ManilaCalendar(SpecCalendar):
    """Philippine banking calendar (Manila / PHP)."""

    HOLIDAYS = [
        fixed(1, 1), fixed(4, 9), fixed(5, 1), fixed(6, 12),
        nth(8, 0, -1),   # National Heroes Day (last Mon Aug)
        fixed(11, 30), fixed(12, 24), fixed(12, 25), fixed(12, 30), fixed(12, 31),
        easter(-3), easter(-2),   # Maundy Thursday, Good Friday
    ]


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
    "PEN": LimaCalendar,
    "ARS": BuenosAiresCalendar,
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
