"""Money and Currency — the value type carried at cashflow/PV boundaries.

`Money(amount, Currency)` makes currency-mixing a type error where it matters
(CLAUDE.md 3, vocabulary ratification 1). Plain floats stay inside hot loops.

Provenance:
  quarry: python/pricebook/core/currency.py
  source: ACI Model Code 2015; ISDA FX definitions — G10 currency set
  oracle: N/A (pure value type; exercised by the Slice 0 closed-form oracle)
  slice:  S00
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Currency(Enum):
    EUR = "EUR"
    GBP = "GBP"
    AUD = "AUD"
    NZD = "NZD"
    USD = "USD"
    CAD = "CAD"
    CHF = "CHF"
    NOK = "NOK"
    SEK = "SEK"
    JPY = "JPY"


@dataclass(frozen=True)
class Money:
    """An amount in a single currency. Frozen value object, boundary-only.

    ponytail: no arithmetic operators yet — Slice 0 has no consumer that adds or
    subtracts Money. Add currency-guarded __add__/__sub__ when the second real
    consumer arrives (rule of two, CLAUDE.md 6b).
    """

    amount: float
    currency: Currency

    def __post_init__(self) -> None:
        if not isinstance(self.currency, Currency):
            raise TypeError(f"currency must be a Currency, got {self.currency!r}")
