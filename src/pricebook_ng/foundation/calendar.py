"""Business-day calendars — the holiday-rule DSL and one `Calendar` type (L0).

Finance-free: a calendar counts *which days are business days*; it prices nothing.

Recorded invariant (S9): **time-of-day never enters `Calendar`** — a calendar is date-only.
Fixing time (WM/R 4pm) is *index* metadata (on `FxFixing`); an option's expiry cut is
*product* data (L2). Putting a `time` on `Calendar` is the drift this record prevents.

Structure (clean) vs content (mined from the quarry `core/calendar.py`):
the quarry had ~38 `Calendar` subclasses keyed by currency. Here there is **one**
frozen `Calendar` value — an identity, a weekend rule, an `Observance` regime, and a
tuple of declarative holiday `Rule`s — and each market is a *declaration* (see
`market_calendars.py`), **keyed by identity** (C1: currency → calendar is a lookup,
not the key). The per-rule `observe` flag is lifted to the calendar's `Observance`.

Provenance:
  quarry: python/pricebook/core/calendar.py
  source: 5 U.S.C. §6103 (US) · UK Banking and Financial Dealings Act 1971 ·
          NZ Holidays Act 2003 · Japanese furikae kyūjitsu · Colombian Ley Emiliani
  oracle: published holiday & observance dates per market (test_calendars)
  slice:  calendars (Topic 0 S1)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from functools import lru_cache

# A holiday rule: given the calendar (for its observance) and a year, the dates it adds.
Rule = Callable[["Calendar", int], Iterable[date]]


class BusinessDayConvention(Enum):
    UNADJUSTED = "unadjusted"
    FOLLOWING = "following"
    MODIFIED_FOLLOWING = "modified_following"
    PRECEDING = "preceding"
    MODIFIED_PRECEDING = "modified_preceding"
    NEAREST = "nearest"


class Weekend(Enum):
    SAT_SUN = (5, 6)      # most of the world
    FRI_SAT = (4, 5)      # Israel and much of MENA


class Observance(Enum):
    """How a fixed-date holiday landing on a weekend is substituted."""
    NONE = "none"                      # not shifted (TARGET, most of the EU/EM)
    US = "us"                          # Sat → prev Fri, Sun → next Mon (5 U.S.C. §6103)
    NEXT_WORKING_DAY = "next_working"  # Sat or Sun → next Mon (UK/AU/NZ/CA)
    SUNDAY_ONLY = "sunday_only"        # Sun → next Mon (South Africa)
    FURIKAE = "furikae"                # Japan: a Sunday holiday walks forward past holidays


class Coverage(Enum):
    """Whether a calendar's holiday set is complete or omits lunar/religious dates."""
    COMPLETE = "complete"
    SECULAR_ONLY = "secular_only"      # omits Islamic/Hebrew/lunisolar holidays (marked, not silent)


# ── Easter ──────────────────────────────────────────────────────────────────────
def gregorian_easter(year: int) -> date:
    """Western Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day + 1)


def orthodox_easter(year: int) -> date:
    """Orthodox Easter Sunday (Julian algorithm + 13-day Gregorian offset, 1900–2099)."""
    a, b, c = year % 4, year % 7, year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month = (d + e + 114) // 31
    day = ((d + e + 114) % 31) + 1
    return date(year, month, day) + timedelta(days=13)


# ── the rule DSL (a rule is `(cal, year) -> dates`) ──────────────────────────────
def _in_range(year: int, since: int | None, until: int | None) -> bool:
    return (since is None or year >= since) and (until is None or year <= until)


def fixed(
    month: int, day: int, *,
    observed: bool = True, since: int | None = None, until: int | None = None,
) -> Rule:
    """A fixed (month, day) holiday. `observed` defaults to the calendar's `Observance`
    regime; `observed=False` pins it to the actual date regardless of the regime — the
    documented exception being AU/NZ ANZAC Day (25 Apr), commemorated, never mondayised."""
    def rule(cal: Calendar, year: int) -> tuple[date, ...]:
        if not _in_range(year, since, until):
            return ()
        d = date(year, month, day)
        return (cal.observe(d) if observed else d,)
    return rule


def easter(offset: int, *, since: int | None = None, until: int | None = None) -> Rule:
    """A Western-Easter-relative holiday (never a weekend, so not substituted)."""
    def rule(cal: Calendar, year: int) -> tuple[date, ...]:
        if not _in_range(year, since, until):
            return ()
        return (gregorian_easter(year) + timedelta(days=offset),)
    return rule


def orthodox(offset: int) -> Rule:
    """An Orthodox-Easter-relative holiday."""
    def rule(cal: Calendar, year: int) -> tuple[date, ...]:
        return (orthodox_easter(year) + timedelta(days=offset),)
    return rule


def nth(month: int, weekday: int, n: int) -> Rule:
    """The nth (n>0) or last (n=-1) `weekday` (0=Mon) of `month` — never a weekend."""
    def rule(cal: Calendar, year: int) -> tuple[date, ...]:
        return (_last_weekday(year, month, weekday) if n == -1
                else _nth_weekday(year, month, weekday, n),)
    return rule


def monday(inner: Rule) -> Rule:
    """Colombian Ley Emiliani: shift each date `inner` produces to the next Monday."""
    def rule(cal: Calendar, year: int) -> tuple[date, ...]:
        return tuple(_to_next_monday(d) for d in inner(cal, year))
    return rule


def christmas_boxing(cal: Calendar, year: int) -> tuple[date, ...]:
    """Christmas + Boxing Day, resolving the collision when both observe to the same
    day (Dec 25 Sun both → Dec 26, so Boxing bumps to Dec 27)."""
    xmas = cal.observe(date(year, 12, 25))
    boxing = cal.observe(date(year, 12, 26))
    return (xmas, boxing + timedelta(days=1)) if boxing == xmas else (xmas, boxing)


def victoria_day(cal: Calendar, year: int) -> tuple[date, ...]:
    """Canadian Victoria Day: the Monday before May 25."""
    may25 = date(year, 5, 25)
    back = (may25.weekday() - 0) % 7 or 7
    return (may25 - timedelta(days=back),)


def midsummer_eve(cal: Calendar, year: int) -> tuple[date, ...]:
    """Swedish Midsummer Eve: the Friday before the Saturday in Jun 20–26."""
    for d in range(20, 27):
        if date(year, 6, d).weekday() == 5:
            return (date(year, 6, d) - timedelta(days=1),)
    return ()


def mexico_inauguration(cal: Calendar, year: int) -> tuple[date, ...]:
    """Mexican Presidential Inauguration: Oct 1 every 6 years from 2024."""
    return (date(year, 10, 1),) if year >= 2024 and (year - 2024) % 6 == 0 else ()


# ── date helpers ─────────────────────────────────────────────────────────────────
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7) + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _to_next_monday(d: date) -> date:
    return d if d.weekday() == 0 else d + timedelta(days=7 - d.weekday())


# ── the one Calendar value ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class Calendar:
    """A business-day calendar: an identity, a weekend rule, a substitution regime,
    and a tuple of declarative holiday rules. Frozen and hashable — holiday sets are
    computed on demand and cached."""

    identity: str
    rules: tuple[Rule, ...]
    weekend: Weekend = Weekend.SAT_SUN
    observance: Observance = Observance.US
    coverage: Coverage = Coverage.COMPLETE

    def observe(self, d: date) -> date:
        """Substitute a weekend holiday under this calendar's regime (identity for
        NONE/FURIKAE — furikae is resolved over the whole set in `_holidays`)."""
        wd = d.weekday()
        obs = self.observance
        if obs is Observance.US:
            if wd == 5:
                return d - timedelta(days=1)
            if wd == 6:
                return d + timedelta(days=1)
        elif obs is Observance.NEXT_WORKING_DAY:
            if wd == 5:
                return d + timedelta(days=2)
            if wd == 6:
                return d + timedelta(days=1)
        elif obs is Observance.SUNDAY_ONLY and wd == 6:
            return d + timedelta(days=1)
        return d

    def is_weekend(self, d: date) -> bool:
        return d.weekday() in self.weekend.value

    def is_holiday(self, d: date) -> bool:
        # observed holidays can spill across a year boundary (Jan 1 Sat → Dec 31)
        return d in self._holidays(d.year) or d in self._holidays(d.year + 1)

    def is_business_day(self, d: date) -> bool:
        return not self.is_weekend(d) and not self.is_holiday(d)

    def adjust(self, d: date, convention: BusinessDayConvention) -> date:
        C = BusinessDayConvention
        if convention is C.UNADJUSTED or self.is_business_day(d):
            return d
        if convention is C.FOLLOWING:
            return self._following(d)
        if convention is C.PRECEDING:
            return self._preceding(d)
        if convention is C.MODIFIED_FOLLOWING:
            nxt = self._following(d)
            return self._preceding(d) if nxt.month != d.month else nxt
        if convention is C.MODIFIED_PRECEDING:
            prv = self._preceding(d)
            return self._following(d) if prv.month != d.month else prv
        if convention is C.NEAREST:
            nxt, prv = self._following(d), self._preceding(d)
            return prv if (d - prv) <= (nxt - d) else nxt
        raise ValueError(f"unknown convention: {convention}")

    def add_business_days(self, d: date, n: int) -> date:
        step = 1 if n >= 0 else -1
        remaining, cur = abs(n), d
        while remaining:
            cur += timedelta(days=step)
            if self.is_business_day(cur):
                remaining -= 1
        return cur

    def _following(self, d: date) -> date:
        while not self.is_business_day(d):
            d += timedelta(days=1)
        return d

    def _preceding(self, d: date) -> date:
        while not self.is_business_day(d):
            d -= timedelta(days=1)
        return d

    def _holidays(self, year: int) -> frozenset[date]:
        return _holidays_of(self, year)  # module-level cache (Calendar is frozen/hashable)


@lru_cache(maxsize=None)
def _holidays_of(cal: Calendar, year: int) -> frozenset[date]:
    raw: set[date] = set()
    for rule in cal.rules:
        raw.update(rule(cal, year))
    if cal.observance is Observance.FURIKAE:
        raw |= _furikae_substitutes(raw)
    return frozenset(raw)


def _furikae_substitutes(holidays: set[date]) -> set[date]:
    """For each holiday on a Sunday, the first following non-holiday day."""
    subs: set[date] = set()
    for d in holidays:
        if d.weekday() == 6:
            cand = d + timedelta(days=1)
            while cand in holidays or cand in subs:
                cand += timedelta(days=1)
            subs.add(cand)
    return subs


@dataclass(frozen=True)
class JointCalendar:
    """A date is a holiday if it is a holiday in ANY component; a business day only if
    it is a business day in ALL of them."""

    calendars: tuple[Calendar, ...]

    def __init__(self, *calendars: Calendar) -> None:
        object.__setattr__(self, "calendars", calendars)

    def is_weekend(self, d: date) -> bool:
        return any(c.is_weekend(d) for c in self.calendars)

    def is_holiday(self, d: date) -> bool:
        return any(c.is_holiday(d) for c in self.calendars)

    def is_business_day(self, d: date) -> bool:
        return all(c.is_business_day(d) for c in self.calendars)
