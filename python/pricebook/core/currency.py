"""Currency and currency pair conventions for G10 FX.

Covers all 10 G10 currencies with correct market quoting conventions,
settlement lags, and base/quote priority.

References:
    ACI Model Code, 2015.
    ISDA FX definitions, 2024.
"""

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


# Market convention base priority (ACI standard):
# EUR > GBP > AUD > NZD > USD > CAD > CHF > NOK > SEK > JPY
# The higher-priority currency is always the base.
# E.g. EUR/USD (not USD/EUR), GBP/USD (not USD/GBP), USD/JPY (not JPY/USD).
_BASE_PRIORITY = [
    Currency.EUR, Currency.GBP, Currency.AUD, Currency.NZD,
    Currency.USD, Currency.CAD, Currency.CHF,
    Currency.NOK, Currency.SEK, Currency.JPY,
]

# Settlement lags per pair type
# Most G10 pairs: T+2. Exception: USD/CAD is T+1.
_SETTLEMENT_LAGS: dict[tuple[str, str], int] = {
    ("USD", "CAD"): 1,
}
_DEFAULT_SETTLEMENT_LAG = 2


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

    @property
    def settlement_lag(self) -> int:
        """Settlement lag in business days (T+1 for USD/CAD, T+2 for others)."""
        key = (self.base.value, self.quote.value)
        key_inv = (self.quote.value, self.base.value)
        return _SETTLEMENT_LAGS.get(key, _SETTLEMENT_LAGS.get(key_inv, _DEFAULT_SETTLEMENT_LAG))

    @property
    def is_ndf(self) -> bool:
        """True if this is a non-deliverable forward pair (EM currencies).
        All G10 pairs are deliverable."""
        g10 = {c.value for c in Currency}
        return self.base.value not in g10 or self.quote.value not in g10

    def forward_rate(self, spot: float, rate_base: float, rate_quote: float,
                     T: float) -> float:
        """FX forward via covered interest parity (continuous rates).
        F = S × exp((r_quote − r_base) × T).
        """
        import math
        return spot * math.exp((rate_quote - rate_base) * T)

    def forward_rate_from_curves(
        self, spot: float, maturity: "date",
        base_curve: "DiscountCurve", quote_curve: "DiscountCurve",
    ) -> float:
        """FX forward via CIP with discount curves (exact).
        F = S × df_base(T) / df_quote(T).
        """
        return spot * base_curve.df(maturity) / quote_curve.df(maturity)

    def forward_points(self, spot: float, rate_base: float, rate_quote: float,
                       T: float) -> float:
        """Forward points = F − S."""
        return self.forward_rate(spot, rate_base, rate_quote, T) - spot


def all_g10_pairs() -> list["CurrencyPair"]:
    """Generate all 45 unique G10 cross pairs in market convention."""
    currencies = list(Currency)
    pairs = []
    for i in range(len(currencies)):
        for j in range(i + 1, len(currencies)):
            pairs.append(CurrencyPair.from_currencies(currencies[i], currencies[j]))
    return pairs
