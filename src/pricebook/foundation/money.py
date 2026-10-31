"""Money and currency (L0).

**`Currency` is an open frozen value, not an `Enum`** — and that is a decision, not an omission.
Markets are open-ended: a new currency is a market event, not a code release, and an enum would
silently drop every code not hard-coded. The stock makes the same argument at length and answers
it with a 289-line registry (interned class constants, read-only `MappingProxyType` views,
`register_currency`). The *insight* is kept; the machinery is not, because nothing yet needs named
constants — a frozen dataclass is open by construction, so `Currency("BRL")` already works. The
registry returns if and when a consumer wants `Currency.USD` with validation behind it.

**Currency mixing is an error, never a silent addition.** That is the only reason this type exists
rather than passing floats: a PV that quietly adds USD to EUR is wrong in a way no downstream
check recovers from.

Provenance:
  stock:  _stock/src/pricebook/foundation/money.py (purified: Money + Currency + the mixing
          guard; the registry, Quantity, Unit and CurrencyPair stay in the stock until pulled)
  source: ISO 4217 (codes, minor units)
  oracle: currency mixing refused (test_adding_money_in_different_currencies_is_refused);
          PV arithmetic in one currency closes (tranche-1 closed form)
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Currency:
    """An ISO-4217 currency: its `code`, and how many minor units it subdivides into.

    `minor_units` defaults to 2 because most currencies have it; JPY (0) and the three-decimal
    dinars are the cases that make it a field rather than a constant. Frozen and hashable, so it
    keys a curve by VALUE — two `Currency("USD")` are one key, which is what a snapshot needs."""

    code: str
    minor_units: int = 2

    def __post_init__(self) -> None:
        if not self.code or not self.code.strip():
            raise ValueError("Currency needs a non-empty code")
        if self.minor_units < 0:
            raise ValueError(f"minor_units cannot be negative: {self.minor_units}")

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True)
class Money:
    """An `amount` in a `currency`. Closed under same-currency arithmetic; mixing raises.

    Used at boundaries — cashflows, PVs, results — and **not** inside numerical loops, where
    plain floats are the right representation and the currency is an invariant of the loop."""

    amount: float
    currency: Currency

    def _same(self, other: Money) -> None:
        if self.currency != other.currency:
            raise ValueError(
                f"cannot combine different currencies: {self.currency} and {other.currency}"
            )

    def __add__(self, other: Money) -> Money:
        self._same(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._same(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: float) -> Money:
        return Money(self.amount * factor, self.currency)

    __rmul__ = __mul__

    def __neg__(self) -> Money:
        return Money(-self.amount, self.currency)

    def to_dict(self) -> dict:
        return {"amount": self.amount, "currency": self.currency.code,
                "minor_units": self.currency.minor_units}

    @classmethod
    def from_dict(cls, d: dict) -> Money:
        return cls(float(d["amount"]), Currency(d["currency"], int(d.get("minor_units", 2))))
