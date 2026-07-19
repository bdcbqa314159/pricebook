"""Money, Quantity, and open currency/unit registries (L0).

Gate audit S1/S4 + the meta-rule: **open-ended domains get a registry** (a new member is a
*declaration*, never an L0 enum edit). `Currency` (an ISO-4217 code + `minor_units`) and
`Unit` (a symbol) are open registries — a new market/commodity is `register_currency(...)`
/ `register_unit(...)`, not a source change. The standard members are class constants
(`Currency.USD`, `Unit.BARREL`), interned so `is`/`==` both hold. `Money`/`Quantity` are
closed under same-currency / same-unit arithmetic; mixing is a type error.

**Why `Currency` is an open registry, not an `Enum` (A3 — do not "fix" this back).**
`setattr(Currency, "USD", ...)` gives the ergonomics of `Currency.USD` but the members are
*invisible to type checkers and autocomplete* — a real cost. It is paid deliberately: the
ratified scope contract (`redesign/01_scope_contract.md`) puts **BRL and the whole LatAm /
EM set in scope**, and markets are open-ended (a new currency is a market event, not a code
release). Reverting to an `Enum` would regain autocomplete but **silently drop every currency
not hard-coded — BRL included — breaking the scope contract.** The registry is the correct
shape for an open-ended domain (meta-rule); the lost autocomplete is the accepted price.
The registry *view* is read-only (`CURRENCIES`/`UNITS` are `MappingProxyType`); additions go
through `register_*` only (A2).

Provenance:
  quarry: python/pricebook/core/currency.py
  source: ISO 4217 (codes + minor units); ACI Model Code
  oracle: currency-mixing rejected; same-unit-only quantity; open registration; JPY 0 minor units
  slice:  open-currency-unit (Topic 0 gate rework, S1/S4)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Mapping


@dataclass(frozen=True)
class Currency:
    """An ISO-4217 currency: its `code` and `minor_units` (decimal places — USD 2, JPY 0)."""

    code: str
    minor_units: int = 2

    # The 37 standard members (populated by the declaration loop below). Declared so
    # `Currency.USD` type-checks and autocompletes — the static half of the open registry.
    # Keep in sync with the loop; a new market is `register_currency(...)`, no annotation needed.
    USD: ClassVar[Currency]
    EUR: ClassVar[Currency]
    GBP: ClassVar[Currency]
    JPY: ClassVar[Currency]
    CHF: ClassVar[Currency]
    AUD: ClassVar[Currency]
    CAD: ClassVar[Currency]
    SEK: ClassVar[Currency]
    NOK: ClassVar[Currency]
    NZD: ClassVar[Currency]
    DKK: ClassVar[Currency]
    PLN: ClassVar[Currency]
    CZK: ClassVar[Currency]
    HUF: ClassVar[Currency]
    RON: ClassVar[Currency]
    TRY: ClassVar[Currency]
    SAR: ClassVar[Currency]
    ILS: ClassVar[Currency]
    EGP: ClassVar[Currency]
    ZAR: ClassVar[Currency]
    KES: ClassVar[Currency]
    NGN: ClassVar[Currency]
    BRL: ClassVar[Currency]
    MXN: ClassVar[Currency]
    CLP: ClassVar[Currency]
    COP: ClassVar[Currency]
    PEN: ClassVar[Currency]
    ARS: ClassVar[Currency]
    CNY: ClassVar[Currency]
    KRW: ClassVar[Currency]
    INR: ClassVar[Currency]
    SGD: ClassVar[Currency]
    HKD: ClassVar[Currency]
    IDR: ClassVar[Currency]
    MYR: ClassVar[Currency]
    THB: ClassVar[Currency]
    PHP: ClassVar[Currency]

    @property
    def value(self) -> str:  # the ISO code (kept for call sites that read `.value`)
        return self.code


_CURRENCIES: dict[str, Currency] = {}


def register_currency(code: str, minor_units: int = 2) -> Currency:
    """Declare a currency (open registry — a new market is a declaration, not an enum edit).
    Re-registering an existing code **raises** (audit 3.3): a silent overwrite would break the
    interning that makes `Currency.USD is currency("USD")` hold."""
    if code in _CURRENCIES:
        raise ValueError(
            f"currency {code!r} is already registered (re-registration not allowed)"
        )
    c = Currency(code, minor_units)
    _CURRENCIES[code] = c
    return c


@contextmanager
def temporary_currency(code: str, minor_units: int = 2) -> Iterator[Currency]:
    """Register a currency for a `with` block, then remove it — registry isolation for tests."""
    c = register_currency(code, minor_units)
    try:
        yield c
    finally:
        del _CURRENCIES[code]


def currency(code: str) -> Currency:
    c = _CURRENCIES.get(code.upper())
    if c is None:
        raise ValueError(f"unknown currency {code!r}. Known: {sorted(_CURRENCIES)}")
    return c


def list_currencies() -> list[str]:
    return sorted(_CURRENCIES)


# The 37 standard declarations (matching the market calendars). Zero-decimal: JPY/KRW/CLP.
_ZERO = {"JPY", "KRW", "CLP"}
for _code in (
    "USD EUR GBP JPY CHF AUD CAD SEK NOK NZD DKK PLN CZK HUF RON TRY SAR ILS EGP "
    "ZAR KES NGN BRL MXN CLP COP PEN ARS CNY KRW INR SGD HKD IDR MYR THB PHP"
).split():
    setattr(Currency, _code, register_currency(_code, 0 if _code in _ZERO else 2))

# Read-only public view. The registry stays OPEN (`register_currency` writes the backing at
# runtime — S1), but the view refuses direct mutation, so nothing can rebind or corrupt it (A2).
CURRENCIES: Mapping[str, Currency] = MappingProxyType(_CURRENCIES)


@dataclass(frozen=True)
class Unit:
    """A physical settlement unit for commodities, by `symbol` (open registry — S4)."""

    symbol: str

    # Standard members (populated by the loop below); declared so `Unit.BARREL` type-checks.
    BARREL: ClassVar[Unit]
    GALLON: ClassVar[Unit]
    MMBTU: ClassVar[Unit]
    THERM: ClassVar[Unit]
    MWH: ClassVar[Unit]
    TONNE: ClassVar[Unit]
    TROY_OUNCE: ClassVar[Unit]
    BUSHEL: ClassVar[Unit]
    POUND: ClassVar[Unit]

    @property
    def value(self) -> str:
        return self.symbol


_UNITS: dict[str, Unit] = {}


def register_unit(name: str, symbol: str) -> Unit:
    if symbol in _UNITS:
        raise ValueError(
            f"unit {symbol!r} is already registered (re-registration not allowed)"
        )
    u = Unit(symbol)
    _UNITS[symbol] = u
    return u


def unit(symbol: str) -> Unit:
    u = _UNITS.get(symbol)
    if u is None:
        raise ValueError(f"unknown unit {symbol!r}. Known: {sorted(_UNITS)}")
    return u


def list_units() -> list[str]:
    return sorted(_UNITS)


for _name, _sym in [
    ("BARREL", "bbl"),
    ("GALLON", "gal"),
    ("MMBTU", "MMBtu"),
    ("THERM", "thm"),
    ("MWH", "MWh"),
    ("TONNE", "t"),
    ("TROY_OUNCE", "ozt"),
    ("BUSHEL", "bu"),
    ("POUND", "lb"),
]:
    setattr(Unit, _name, register_unit(_name, _sym))

# Read-only public view of the open unit registry (A2, as for currencies above).
UNITS: Mapping[str, Unit] = MappingProxyType(_UNITS)


@dataclass(frozen=True)
class Money:
    """An amount in a single currency. Adding/subtracting different currencies is a
    type error; scale by a plain number with `*`."""

    amount: float
    currency: Currency

    def __post_init__(self) -> None:
        if not isinstance(self.currency, Currency):
            raise TypeError(f"currency must be a Currency, got {self.currency!r}")

    def _guard(self, other: Money) -> None:
        if self.currency != other.currency:
            raise TypeError(
                f"cannot mix {self.currency.code} and {other.currency.code}"
            )

    def __add__(self, other: Money) -> Money:
        self._guard(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._guard(other)
        return Money(self.amount - other.amount, self.currency)

    def __neg__(self) -> Money:
        return Money(-self.amount, self.currency)

    def __mul__(self, scale: float) -> Money:
        return Money(self.amount * scale, self.currency)

    __rmul__ = __mul__

    def to_dict(self) -> dict:
        return {"amount": self.amount, "currency": self.currency.code}

    @classmethod
    def from_dict(cls, d: dict) -> Money:
        return cls(d["amount"], currency(d["currency"]))


@dataclass(frozen=True)
class Quantity:
    """A physical amount in a single unit. Closed under same-unit arithmetic only —
    barrels and MWh do not add."""

    amount: float
    unit: Unit

    def __post_init__(self) -> None:
        if not isinstance(self.unit, Unit):
            raise TypeError(f"unit must be a Unit, got {self.unit!r}")

    def _guard(self, other: Quantity) -> None:
        if self.unit != other.unit:
            raise TypeError(f"cannot mix {self.unit.symbol} and {other.unit.symbol}")

    def __add__(self, other: Quantity) -> Quantity:
        self._guard(other)
        return Quantity(self.amount + other.amount, self.unit)

    def __sub__(self, other: Quantity) -> Quantity:
        self._guard(other)
        return Quantity(self.amount - other.amount, self.unit)

    def __neg__(self) -> Quantity:
        return Quantity(-self.amount, self.unit)

    def __mul__(self, scale: float) -> Quantity:
        return Quantity(self.amount * scale, self.unit)

    __rmul__ = __mul__

    def to_dict(self) -> dict:
        return {"amount": self.amount, "unit": self.unit.symbol}

    @classmethod
    def from_dict(cls, d: dict) -> Quantity:
        return cls(d["amount"], unit(d["unit"]))


@dataclass(frozen=True)
class CurrencyPair:
    """An FX pair `base`/`quote` (one unit of base costs `price` of quote). Its IDENTITY is the
    two currencies only — the spot settlement lag is a *convention*, looked up by
    `settlement.spot_lag(pair)`, not a field (audit 3.6: a lag on the identity fragments dict
    keys, so `CurrencyPair(USD, JPY)` would differ from the same pair with a lag)."""

    base: Currency
    quote: Currency

    @property
    def name(self) -> str:
        return f"{self.base.code}{self.quote.code}"
