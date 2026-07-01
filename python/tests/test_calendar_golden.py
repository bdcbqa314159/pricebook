"""Byte-exact characterization net for every registered calendar.

Pins ``_compute_holidays`` for all currency calendars across 2000-2050. Any
change to any holiday set (e.g. from the boilerplate/SpecCalendar refactor)
trips the matching code's fingerprint. This must stay green through the refactor
— it is behaviour-preserving by construction, and this net proves it.
"""

import hashlib

import pytest

from pricebook.core.calendar import get_calendar, list_calendars

_YEARS = range(2000, 2051)

# Fingerprints captured from the pre-refactor implementation (v1.214).
_GOLDEN = {
    "ARS": "4f0d980847eea8ec",
    "AUD": "94a9fa5eede7357f",
    "BRL": "3356fca86392c6a9",
    "CAD": "14137d9f0f0b653b",
    "CHF": "319e84c3b74ef4ef",
    "CLP": "1c47d60980f31566",
    "CNY": "d29bfa0418a3e2ef",
    "COP": "432850c706c2d89b",
    "CZK": "efead87188c794d7",
    "DKK": "d6c4ec1d5a6eb752",
    "EGP": "44f57d80b082ea1f",
    "EUR": "0fa1453bac5c4971",
    "GBP": "a5442129c21bde96",
    "HKD": "98456414eded0295",
    "HUF": "9085f7b001912d99",
    "IDR": "acb138fbc7a24ba7",
    "ILS": "33e62173d309a1d8",
    "INR": "9a5243fe843d80e8",
    "JPY": "c124720f586fe905",
    "KES": "3093b4d760d6493a",
    "KRW": "2a6bc31b29a1994c",
    "MXN": "cdd721276c4e78ac",
    "MYR": "7b2bbd83476c7efc",
    "NGN": "476a5f90fe87b88b",
    "NOK": "e418d791f6461ce0",
    "NZD": "abbc4a773b386926",
    "PEN": "ce9c3ba3fe7cdd4c",
    "PHP": "773a5e9d3660dd3c",
    "PLN": "3878123859109a83",
    "RON": "30a7a8d666ab187d",
    "SAR": "76b0412d4f537c78",
    "SEK": "8ce005026005670b",
    "SGD": "f524e005c97d95d3",
    "THB": "de8643d33577e106",
    "TRY": "d8d452317d5c9e59",
    "USD": "12203de4866deb07",
    "ZAR": "f743acaa75a72397",
}


def _fingerprint(code: str) -> str:
    cal = get_calendar(code)
    parts = []
    for y in _YEARS:
        hols = sorted(cal._compute_holidays(y))
        parts.append(f"{y}:" + ",".join(d.isoformat() for d in hols))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def test_calendar_registry_unchanged():
    """No calendar silently added or dropped by the refactor."""
    assert set(list_calendars()) == set(_GOLDEN)


@pytest.mark.parametrize("code", sorted(_GOLDEN))
def test_calendar_holidays_byte_exact(code):
    assert _fingerprint(code) == _GOLDEN[code], f"{code} holiday set changed"
