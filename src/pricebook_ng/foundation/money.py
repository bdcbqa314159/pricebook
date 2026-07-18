"""Money, Quantity, and currency identity (L0).

Finance-free value types carried at boundaries. `Money` makes currency-mixing a type
error; `Quantity` does the same for physical units (commodities settle in barrels /
MWh / tonnes, which `Money` cannot express). Both are closed under arithmetic **only
within the same currency / unit**.

Provenance:
  quarry: python/pricebook/core/currency.py
  source: ISO 4217 currency codes; ACI Model Code (FX spot conventions)
  oracle: currency-mixing rejected at type level; same-unit-only quantity arithmetic
  slice:  money-quantity (Topic 0 S4)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Currency(Enum):
    # G10
    USD = "USD"; EUR = "EUR"; GBP = "GBP"; JPY = "JPY"; CHF = "CHF"; AUD = "AUD"
    CAD = "CAD"; SEK = "SEK"; NOK = "NOK"; NZD = "NZD"; DKK = "DKK"
    # CEE
    PLN = "PLN"; CZK = "CZK"; HUF = "HUF"; RON = "RON"
    # Turkey & MENA
    TRY = "TRY"; SAR = "SAR"; ILS = "ILS"; EGP = "EGP"
    # Africa
    ZAR = "ZAR"; KES = "KES"; NGN = "NGN"
    # LatAm
    BRL = "BRL"; MXN = "MXN"; CLP = "CLP"; COP = "COP"; PEN = "PEN"; ARS = "ARS"
    # Asia
    CNY = "CNY"; KRW = "KRW"; INR = "INR"; SGD = "SGD"; HKD = "HKD"; IDR = "IDR"
    MYR = "MYR"; THB = "THB"; PHP = "PHP"


class Unit(Enum):
    """Physical settlement units for commodities."""
    BARREL = "bbl"          # crude oil
    GALLON = "gal"          # refined products
    MMBTU = "MMBtu"         # natural gas
    THERM = "thm"           # natural gas (retail)
    MWH = "MWh"             # power
    TONNE = "t"             # metals, softs, freight
    TROY_OUNCE = "ozt"      # precious metals
    BUSHEL = "bu"           # grains
    POUND = "lb"            # softs (coffee, sugar, cotton)


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
        if self.currency is not other.currency:
            raise TypeError(f"cannot mix {self.currency.value} and {other.currency.value}")

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
        if self.unit is not other.unit:
            raise TypeError(f"cannot mix {self.unit.value} and {other.unit.value}")

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


@dataclass(frozen=True)
class CurrencyPair:
    """An FX pair `base`/`quote` (one unit of base costs `price` of quote) with its
    spot settlement lag (T+`spot_lag`, market default 2; USD/CAD is 1)."""

    base: Currency
    quote: Currency
    spot_lag: int = 2

    @property
    def name(self) -> str:
        return f"{self.base.value}{self.quote.value}"
