"""Currency and currency pair conventions."""

from enum import Enum


class Currency(Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


# Market convention: which currency is base in each pair.
# Base currency = 1 unit, quote currency = price.
# E.g. EUR/USD = 1.10 means 1 EUR costs 1.10 USD.
_BASE_PRIORITY = [Currency.EUR, Currency.GBP, Currency.USD]


class CurrencyPair:
    """
    A currency pair with market quoting convention.

    The base currency is the one you hold 1 unit of.
    The quote currency is what you pay.
    Spot rate: how many units of quote to buy 1 unit of base.
    """

    def __init__(self, base: Currency, quote: Currency):
        if base == quote:
            raise ValueError(f"base and quote must differ, got {base}")
        self.base = base
        self.quote = quote

    @classmethod
    def from_currencies(cls, ccy1: Currency, ccy2: Currency) -> "CurrencyPair":
        """Create a pair using market convention for base/quote ordering."""
        if ccy1 == ccy2:
            raise ValueError(f"currencies must differ, got {ccy1}")
        for ccy in _BASE_PRIORITY:
            if ccy == ccy1:
                return cls(ccy1, ccy2)
            if ccy == ccy2:
                return cls(ccy2, ccy1)
        return cls(ccy1, ccy2)

    def __repr__(self) -> str:
        return f"{self.base.value}/{self.quote.value}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CurrencyPair):
            return NotImplemented
        return self.base == other.base and self.quote == other.quote

    def __hash__(self) -> int:
        return hash((self.base, self.quote))

    def invert(self) -> "CurrencyPair":
        """Return the inverse pair (swap base and quote)."""
        return CurrencyPair(self.quote, self.base)
