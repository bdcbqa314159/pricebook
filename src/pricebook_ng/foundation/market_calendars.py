"""The 37 market calendars as declarations, keyed by identity (L0).

A market is *data*, not a class (the quarry had one subclass each). Adding a market is
a declaration here — never a code change. Currency → calendar is a lookup
(`calendar_for_currency`), not the registry key (C1).

Provenance:
  quarry: python/pricebook/core/calendar.py (the currency-keyed registry + specs)
  source: national holiday statutes (see calendar.py); market conventions per centre
  oracle: published holiday & observance dates per market (test_calendars)
  slice:  calendars (Topic 0 S1)
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType

from pricebook_ng.foundation.calendars import (
    Calendar,
    CalendarProtocol,
    Coverage,
    HolidaySet,
    JointCalendar,
    Observance,
    Weekend,
    christmas_boxing,
    dates,
    easter,
    equinox,
    fixed,
    mexico_inauguration,
    midsummer_eve,
    monday,
    nth,
    orthodox,
    victoria_day,
)

_US = Observance.US
_NWD = Observance.NEXT_WORKING_DAY
# secular-only: omits lunar/religious holidays; add on crossing EM rates (Cowork §4)
_SEC = Coverage.SECULAR_ONLY

# ── G10 ──────────────────────────────────────────────────────────────────────────
_calendars: dict[
    str, Calendar
] = {}  # mutable backing; frozen into `_CALENDARS` below (A2)


def register_calendar(cal: Calendar) -> Calendar:
    """Register a market calendar (open registry — the most open-ended domain; audit 3.3).
    A conflicting identity **raises** (no silent overwrite)."""
    if cal.identity in _calendars:
        raise ValueError(f"calendar {cal.identity!r} is already registered")
    _calendars[cal.identity] = cal
    return cal


_reg = register_calendar  # the 37 declarations below


@contextmanager
def temporary_calendar(cal: Calendar) -> Iterator[Calendar]:
    """Register a calendar for a `with` block, then remove it — registry isolation for tests."""
    register_calendar(cal)
    try:
        yield cal
    finally:
        del _calendars[cal.identity]


# US government-securities (SIFMA/Treasury) calendar. Good Friday is a close (no SOFR fixing).
# Observance is SUNDAY_ONLY, not federal Sat→Fri: SIFMA does NOT shift Saturday holidays to the
# preceding Friday (2021-12-31 was open) — audit 2.2. (A federal/Fed-bank calendar with Sat→Fri,
# and NYSE, are separate calendars added with their first equity/EFFR consumer, not speculatively.)
US_GOVERNMENT_SECURITIES = _reg(
    Calendar(
        "US_GOVERNMENT_SECURITIES",
        HolidaySet(
            (
                fixed(1, 1),
                nth(1, 0, 3),
                nth(2, 0, 3),
                easter(-2),  # Good Friday — bond market close, no SOFR publication
                nth(5, 0, -1),
                fixed(6, 19, since=2021),
                fixed(7, 4),
                nth(9, 0, 1),
                nth(10, 0, 2),
                fixed(11, 11),
                nth(11, 3, 4),
                fixed(12, 25),
            ),
        ),
        observance=Observance.SUNDAY_ONLY,
    )
)

TARGET = _reg(
    Calendar(
        "TARGET",
        (
            fixed(1, 1),
            fixed(5, 1),
            fixed(12, 25),
            fixed(12, 26),
            easter(-2),
            easter(1),
        ),
        observance=Observance.NONE,
    )
)

LONDON = _reg(
    Calendar(
        "LONDON",
        (
            fixed(1, 1),
            easter(-2),
            easter(1),
            # Early May BH: moved to Fri 8 May in 2020 (VE Day 75th)
            nth(5, 0, 1, until=2019),
            nth(5, 0, 1, since=2021),
            # Spring BH: moved to Thu 2 Jun in 2022 (Platinum Jubilee)
            nth(5, 0, -1, until=2021),
            nth(5, 0, -1, since=2023),
            nth(8, 0, -1),
            christmas_boxing,
            # one-off bank holidays (moves + additions) — historical SONIA accuracy (audit 2.3)
            dates(
                (2020, 5, 8),  # VE Day 75th (moved Early May BH)
                (2022, 6, 2),  # Platinum Jubilee (moved Spring BH)
                (2022, 6, 3),  # Platinum Jubilee extra day
                (2022, 9, 19),  # State funeral of Elizabeth II
                (2023, 5, 8),  # Coronation of Charles III
            ),
        ),
        observance=_NWD,
    )
)

TOKYO = _reg(
    Calendar(
        "TOKYO",
        (
            fixed(1, 1),
            fixed(1, 2),
            fixed(1, 3),
            fixed(2, 11),
            fixed(2, 23, since=2020),  # Emperor's Birthday (Naruhito) from 2020
            fixed(12, 23, until=2018),  # Emperor's Birthday (Akihito) through 2018
            equinox(3),  # Vernal Equinox — astronomical (audit 2.4)
            fixed(4, 29),
            fixed(5, 3),
            fixed(5, 4),
            fixed(5, 5),
            fixed(8, 11),
            equinox(9),  # Autumnal Equinox — astronomical
            fixed(11, 3),
            fixed(11, 23),
            nth(1, 0, 2),
            nth(7, 0, 3),
            nth(9, 0, 3),
            nth(10, 0, 2),
        ),
        observance=Observance.FURIKAE,
    )
)

ZURICH = _reg(
    Calendar(
        "ZURICH",
        (
            fixed(1, 1),
            fixed(1, 2),
            fixed(5, 1),
            fixed(8, 1),
            fixed(12, 25),
            fixed(12, 26),
            easter(-2),
            easter(1),
            easter(39),
            easter(50),
        ),
        observance=Observance.NONE,
    )
)

SYDNEY = _reg(
    Calendar(
        "SYDNEY",
        (
            fixed(1, 1),
            fixed(1, 26),
            fixed(4, 25, observed=False),
            nth(6, 0, 2),
            nth(8, 0, 1),
            christmas_boxing,
            easter(-2),
            easter(-1),
            easter(1),
        ),
        observance=_NWD,
    )
)

TORONTO = _reg(
    Calendar(
        "TORONTO",
        (
            fixed(1, 1),
            nth(2, 0, 3),
            easter(-2),
            victoria_day,
            fixed(7, 1),
            nth(8, 0, 1),
            nth(9, 0, 1),
            nth(10, 0, 2),
            fixed(11, 11),
            christmas_boxing,
        ),
        observance=_NWD,
    )
)

STOCKHOLM = _reg(
    Calendar(
        "STOCKHOLM",
        (
            fixed(1, 1),
            fixed(1, 6),
            fixed(5, 1),
            fixed(6, 6),
            easter(-2),
            easter(1),
            easter(39),
            midsummer_eve,
            fixed(12, 24),
            fixed(12, 25),
            fixed(12, 26),
            fixed(12, 31),
        ),
        observance=Observance.NONE,
    )
)

OSLO = _reg(
    Calendar(
        "OSLO",
        (
            fixed(1, 1),
            fixed(5, 1),
            fixed(5, 17),
            easter(-3),
            easter(-2),
            easter(1),
            easter(39),
            easter(50),
            fixed(12, 25),
            fixed(12, 26),
        ),
        observance=Observance.NONE,
    )
)

WELLINGTON = _reg(
    Calendar(
        "WELLINGTON",
        (
            fixed(1, 1),
            fixed(1, 2),
            fixed(2, 6),
            fixed(4, 25, observed=False),
            nth(6, 0, 1),
            nth(10, 0, 4),
            easter(-2),
            easter(1),
            christmas_boxing,
        ),
        observance=_NWD,
    )
)

COPENHAGEN = _reg(
    Calendar(
        "COPENHAGEN",
        (
            fixed(1, 1),
            fixed(6, 5),
            fixed(12, 24),
            fixed(12, 25),
            fixed(12, 26),
            fixed(12, 31),
            easter(-3),
            easter(-2),
            easter(1),
            easter(26, until=2023),
            easter(39),
            easter(50),
        ),
        observance=Observance.NONE,
    )
)

# ── CEE ──────────────────────────────────────────────────────────────────────────
WARSAW = _reg(
    Calendar(
        "WARSAW",
        (
            fixed(1, 1),
            fixed(1, 6),
            fixed(5, 1),
            fixed(5, 3),
            fixed(8, 15),
            fixed(11, 1),
            fixed(11, 11),
            fixed(12, 25),
            fixed(12, 26),
            easter(1),
            easter(60),
        ),
        observance=Observance.NONE,
    )
)

PRAGUE = _reg(
    Calendar(
        "PRAGUE",
        (
            fixed(1, 1),
            fixed(5, 1),
            fixed(5, 8),
            fixed(7, 5),
            fixed(7, 6),
            fixed(9, 28),
            fixed(10, 28),
            fixed(11, 17),
            fixed(12, 24),
            fixed(12, 25),
            fixed(12, 26),
            easter(-2),
            easter(1),
        ),
        observance=Observance.NONE,
    )
)

BUDAPEST = _reg(
    Calendar(
        "BUDAPEST",
        (
            fixed(1, 1),
            fixed(3, 15),
            fixed(5, 1),
            fixed(8, 20),
            fixed(10, 23),
            fixed(11, 1),
            fixed(12, 25),
            fixed(12, 26),
            easter(-2),
            easter(1),
            easter(50),
        ),
        observance=Observance.NONE,
    )
)

BUCHAREST = _reg(
    Calendar(
        "BUCHAREST",
        (
            fixed(1, 1),
            fixed(1, 2),
            fixed(1, 24),
            fixed(5, 1),
            fixed(6, 1),
            fixed(8, 15),
            fixed(11, 30),
            fixed(12, 1),
            fixed(12, 25),
            fixed(12, 26),
            orthodox(-2),
            orthodox(1),
            orthodox(50),
        ),
        observance=Observance.NONE,
    )
)

# ── Turkey & MENA ────────────────────────────────────────────────────────────────
ISTANBUL = _reg(
    Calendar(
        "ISTANBUL",
        (
            fixed(1, 1),
            fixed(4, 23),
            fixed(5, 1),
            fixed(5, 19),
            fixed(7, 15),
            fixed(8, 30),
            fixed(10, 29),
        ),
        observance=Observance.NONE,
        coverage=_SEC,
    )
)

RIYADH = _reg(
    Calendar(
        "RIYADH",
        (
            fixed(9, 23),
            fixed(2, 22, since=2022),
        ),
        weekend=Weekend.FRI_SAT,
        observance=Observance.NONE,
        coverage=_SEC,
    )
)

TEL_AVIV = _reg(
    Calendar(
        "TEL_AVIV",
        (
            fixed(4, 14),
            fixed(4, 20),
            fixed(5, 2),
            fixed(9, 25),
            fixed(9, 26),
            fixed(10, 4),
            fixed(10, 9),
        ),
        weekend=Weekend.FRI_SAT,
        observance=Observance.NONE,
        coverage=_SEC,
    )
)

CAIRO = _reg(
    Calendar(
        "CAIRO",
        (
            fixed(1, 7),
            fixed(1, 25),
            fixed(4, 25),
            fixed(5, 1),
            fixed(7, 23),
            fixed(10, 6),
        ),
        weekend=Weekend.FRI_SAT,
        observance=Observance.NONE,
        coverage=_SEC,
    )
)

# ── Africa ───────────────────────────────────────────────────────────────────────
JOHANNESBURG = _reg(
    Calendar(
        "JOHANNESBURG",
        (
            fixed(1, 1),
            fixed(3, 21),
            fixed(4, 27),
            fixed(5, 1),
            fixed(6, 16),
            fixed(8, 9),
            fixed(9, 24),
            fixed(12, 16),
            fixed(12, 25),
            fixed(12, 26),
            easter(-2),
            easter(1),
        ),
        observance=Observance.SUNDAY_ONLY,
    )
)

NAIROBI = _reg(
    Calendar(
        "NAIROBI",
        (
            fixed(1, 1),
            fixed(5, 1),
            fixed(6, 1),
            fixed(10, 20),
            fixed(12, 12),
            fixed(12, 25),
            fixed(12, 26),
            easter(-2),
            easter(1),
        ),
        observance=Observance.NONE,
    )
)

LAGOS = _reg(
    Calendar(
        "LAGOS",
        (
            fixed(1, 1),
            fixed(5, 1),
            fixed(6, 12),
            fixed(10, 1),
            fixed(12, 25),
            fixed(12, 26),
            easter(-2),
            easter(1),
        ),
        observance=Observance.NONE,
    )
)

# ── LatAm ────────────────────────────────────────────────────────────────────────
SAO_PAULO = _reg(
    Calendar(
        "SAO_PAULO",
        (
            fixed(1, 1),
            fixed(4, 21),
            fixed(5, 1),
            fixed(9, 7),
            fixed(10, 12),
            fixed(11, 2),
            fixed(11, 15),
            fixed(12, 25),
            easter(-48),
            easter(-47),
            easter(-2),
            easter(60),
        ),
        observance=Observance.NONE,
    )
)

MEXICO_CITY = _reg(
    Calendar(
        "MEXICO_CITY",
        (
            fixed(1, 1),
            nth(2, 0, 1),
            nth(3, 0, 3),
            fixed(5, 1),
            fixed(9, 16),
            nth(11, 0, 3),
            fixed(12, 25),
            mexico_inauguration,
            easter(-3),
            easter(-2),
        ),
        observance=Observance.NONE,
    )
)

SANTIAGO = _reg(
    Calendar(
        "SANTIAGO",
        (
            fixed(1, 1),
            fixed(5, 1),
            fixed(5, 21),
            fixed(6, 29),
            fixed(7, 16),
            fixed(8, 15),
            fixed(9, 18),
            fixed(9, 19),
            fixed(10, 12),
            fixed(10, 31),
            fixed(11, 1),
            fixed(12, 8),
            fixed(12, 25),
            easter(-2),
            easter(-1),
        ),
        observance=Observance.NONE,
    )
)

BOGOTA = _reg(
    Calendar(
        "BOGOTA",
        (
            fixed(1, 1),
            monday(fixed(1, 6)),
            monday(fixed(3, 19)),
            fixed(5, 1),
            monday(fixed(6, 29)),
            fixed(7, 20),
            fixed(8, 7),
            monday(fixed(8, 15)),
            monday(fixed(10, 12)),
            monday(fixed(11, 1)),
            monday(fixed(11, 11)),
            fixed(12, 8),
            fixed(12, 25),
            easter(-3),
            easter(-2),
            monday(easter(43)),
            monday(easter(64)),
            monday(easter(71)),
        ),
        observance=Observance.NONE,
    )
)

LIMA = _reg(
    Calendar(
        "LIMA",
        (
            fixed(1, 1),
            easter(-3),
            easter(-2),
            fixed(5, 1),
            fixed(6, 29),
            fixed(7, 28),
            fixed(7, 29),
            fixed(8, 30),
            fixed(10, 8),
            fixed(11, 1),
            fixed(12, 8),
            fixed(12, 25),
        ),
        observance=Observance.NONE,
    )
)

BUENOS_AIRES = _reg(
    Calendar(
        "BUENOS_AIRES",
        (
            fixed(1, 1),
            easter(-48),
            easter(-47),
            easter(-2),
            fixed(3, 24),
            fixed(4, 2),
            fixed(5, 1),
            fixed(5, 25),
            fixed(6, 20),
            fixed(7, 9),
            fixed(8, 17),
            fixed(10, 12),
            fixed(11, 20),
            fixed(12, 8),
            fixed(12, 25),
        ),
        observance=Observance.NONE,
    )
)

# ── Asia ─────────────────────────────────────────────────────────────────────────
BEIJING = _reg(
    Calendar(
        "BEIJING",
        (
            fixed(1, 1),
            fixed(4, 5),
            fixed(5, 1),
            fixed(10, 1),
            fixed(10, 2),
            fixed(10, 3),
        ),
        observance=Observance.NONE,
        coverage=_SEC,
    )
)

SEOUL = _reg(
    Calendar(
        "SEOUL",
        (
            fixed(1, 1),
            fixed(3, 1),
            fixed(5, 5),
            fixed(6, 6),
            fixed(8, 15),
            fixed(10, 3),
            fixed(10, 9),
            fixed(12, 25),
        ),
        observance=Observance.NONE,
        coverage=_SEC,
    )
)

MUMBAI = _reg(
    Calendar(
        "MUMBAI",
        (
            fixed(1, 26),
            fixed(8, 15),
            fixed(10, 2),
            fixed(12, 25),
        ),
        observance=Observance.NONE,
        coverage=_SEC,
    )
)

SINGAPORE = _reg(
    Calendar(
        "SINGAPORE",
        (
            fixed(1, 1),
            fixed(5, 1),
            fixed(8, 9),
            fixed(12, 25),
            easter(-2),
        ),
        observance=Observance.NONE,
    )
)

HONG_KONG = _reg(
    Calendar(
        "HONG_KONG",
        (
            fixed(1, 1),
            fixed(5, 1),
            fixed(7, 1),
            fixed(10, 1),
            fixed(12, 25),
            fixed(12, 26),
            easter(-2),
            easter(-1),
            easter(1),
        ),
        observance=Observance.NONE,
    )
)

JAKARTA = _reg(
    Calendar(
        "JAKARTA",
        (
            fixed(1, 1),
            fixed(5, 1),
            fixed(6, 1),
            fixed(8, 17),
            fixed(12, 25),
            easter(-2),
            easter(39),
        ),
        observance=Observance.NONE,
    )
)

KUALA_LUMPUR = _reg(
    Calendar(
        "KUALA_LUMPUR",
        (
            fixed(1, 1),
            fixed(2, 1),
            fixed(5, 1),
            nth(6, 0, 1),
            fixed(8, 31),
            fixed(9, 16),
            fixed(12, 25),
        ),
        observance=Observance.NONE,
    )
)

BANGKOK = _reg(
    Calendar(
        "BANGKOK",
        (
            fixed(1, 1),
            fixed(4, 6),
            fixed(4, 13),
            fixed(4, 14),
            fixed(4, 15),
            fixed(5, 1),
            fixed(7, 28),
            fixed(8, 12),
            fixed(10, 23),
            fixed(12, 5),
            fixed(12, 10),
            fixed(12, 31),
        ),
        observance=Observance.NONE,
        coverage=_SEC,
    )
)

MANILA = _reg(
    Calendar(
        "MANILA",
        (
            fixed(1, 1),
            fixed(4, 9),
            fixed(5, 1),
            fixed(6, 12),
            nth(8, 0, -1),
            fixed(11, 30),
            fixed(12, 24),
            fixed(12, 25),
            fixed(12, 30),
            fixed(12, 31),
            easter(-3),
            easter(-2),
        ),
        observance=Observance.NONE,
    )
)

# Closed registry — the 37 markets are all declared above. The frozen view cannot be rebound
# or mutated (the quarry bug was registry mutation at import); `get_calendar` reads it (A2).
_CALENDARS: Mapping[str, Calendar] = MappingProxyType(_calendars)


# ── currency → calendar identity (a lookup, C1) ──────────────────────────────────
_CURRENCY: dict[str, str] = {
    "USD": "US_GOVERNMENT_SECURITIES",
    "EUR": "TARGET",
    "GBP": "LONDON",
    "JPY": "TOKYO",
    "CHF": "ZURICH",
    "AUD": "SYDNEY",
    "CAD": "TORONTO",
    "SEK": "STOCKHOLM",
    "NOK": "OSLO",
    "NZD": "WELLINGTON",
    "DKK": "COPENHAGEN",
    "PLN": "WARSAW",
    "CZK": "PRAGUE",
    "HUF": "BUDAPEST",
    "RON": "BUCHAREST",
    "TRY": "ISTANBUL",
    "SAR": "RIYADH",
    "ILS": "TEL_AVIV",
    "EGP": "CAIRO",
    "ZAR": "JOHANNESBURG",
    "KES": "NAIROBI",
    "NGN": "LAGOS",
    "BRL": "SAO_PAULO",
    "MXN": "MEXICO_CITY",
    "CLP": "SANTIAGO",
    "COP": "BOGOTA",
    "PEN": "LIMA",
    "ARS": "BUENOS_AIRES",
    "CNY": "BEIJING",
    "KRW": "SEOUL",
    "INR": "MUMBAI",
    "SGD": "SINGAPORE",
    "HKD": "HONG_KONG",
    "IDR": "JAKARTA",
    "MYR": "KUALA_LUMPUR",
    "THB": "BANGKOK",
    "PHP": "MANILA",
}


def get_calendar(identity: str) -> CalendarProtocol:
    """The calendar for a market identity (e.g. `US_GOVERNMENT_SECURITIES`, `TARGET`), or a
    `JointCalendar` for a composite `"A+B"` identity — the rehydration target of
    `JointCalendar.to_dict` (A3), so an XCCY calendar round-trips by name like any other."""
    if "+" in identity:
        parts = []
        for part in identity.split("+"):
            cal = _CALENDARS.get(part)
            if cal is None:
                raise ValueError(
                    f"no calendar {part!r} in composite {identity!r}. Known: {sorted(_CALENDARS)}"
                )
            parts.append(cal)
        return JointCalendar(*parts)
    cal = _CALENDARS.get(identity)
    if cal is None:
        raise ValueError(f"no calendar {identity!r}. Known: {sorted(_CALENDARS)}")
    return cal


def calendar_for_currency(currency: str) -> Calendar:
    """The market calendar conventionally used for a currency (a lookup, not the key)."""
    identity = _CURRENCY.get(currency.upper())
    if identity is None:
        raise ValueError(
            f"no calendar for currency {currency!r}. Known: {sorted(_CURRENCY)}"
        )
    return _CALENDARS[identity]


def list_calendars() -> list[str]:
    """The declared market identities, sorted."""
    return sorted(_CALENDARS)
