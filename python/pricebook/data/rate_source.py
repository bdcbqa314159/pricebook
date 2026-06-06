"""Rate source protocol: abstract interface for any rate data provider.

Separates *how to fetch* from *how to build curves*. Any data provider
(Euribor website, FRED API, ECB SDW, BOE, file upload) implements
this protocol. The curve builder doesn't care where the data comes from.

    class MySource(RateSource):
        def fetch(self, d: date) -> RateFixing: ...
        def fetch_range(self, start, end) -> list[RateFixing]: ...

Architecture:
    RateSource (data)  →  MarketCurve (conventions + bootstrap)  →  DiscountCurve (pricing)
       ↑                        ↑                                         ↑
    provider-specific     currency-specific                         universal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Protocol, runtime_checkable

from dateutil.relativedelta import relativedelta


class RateType(Enum):
    """Type of rate fixing."""
    DEPOSIT = "deposit"           # term deposit (Euribor, LIBOR)
    OVERNIGHT = "overnight"       # overnight rate (ESTR, SOFR, SONIA)
    SWAP = "swap"                 # IRS par rate
    FUTURES = "futures"           # IR futures implied rate
    BOND_YIELD = "bond_yield"     # sovereign bond yield


@dataclass
class TenorDefinition:
    """A single tenor with its offset from reference date."""
    label: str                    # "1W", "1M", "3M", "6M", "12M"
    key: str                      # "1w", "1m", "3m", "6m", "12m"
    offset: relativedelta         # time offset from reference date
    years: float                  # approximate year fraction

    def maturity(self, ref: date) -> date:
        return ref + self.offset


# Standard tenor sets
DEPOSIT_TENORS = [
    TenorDefinition("1W",  "1w",  relativedelta(weeks=1),   7 / 365),
    TenorDefinition("1M",  "1m",  relativedelta(months=1),  1 / 12),
    TenorDefinition("3M",  "3m",  relativedelta(months=3),  0.25),
    TenorDefinition("6M",  "6m",  relativedelta(months=6),  0.5),
    TenorDefinition("12M", "12m", relativedelta(months=12), 1.0),
]

OVERNIGHT_TENOR = TenorDefinition("ON", "on", relativedelta(days=1), 1 / 365)

SWAP_TENORS = [
    TenorDefinition(f"{y}Y", f"{y}y", relativedelta(years=y), float(y))
    for y in [2, 3, 5, 7, 10, 15, 20, 30]
]


@dataclass
class RateFixing:
    """One date's rate fixings from a source.

    Can hold deposits, overnight, swaps, or any combination.
    """
    date: date
    rates: dict[str, float]       # tenor_key → rate (decimal)
    rate_type: RateType = RateType.DEPOSIT
    source: str = ""
    currency: str = ""

    def rate(self, tenor_key: str) -> float | None:
        return self.rates.get(tenor_key)

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "rates": self.rates,
            "type": self.rate_type.value,
            "source": self.source,
            "currency": self.currency,
        }


@runtime_checkable
class RateSource(Protocol):
    """Protocol for rate data providers.

    Any data source (website, API, file) implements this.
    The MarketCurve builder works with any RateSource.
    """

    @property
    def currency(self) -> str:
        """ISO currency code."""
        ...

    @property
    def rate_type(self) -> RateType:
        """What kind of rates this source provides."""
        ...

    @property
    def tenors(self) -> list[TenorDefinition]:
        """Available tenors."""
        ...

    @property
    def source_name(self) -> str:
        """Human-readable name of the data source."""
        ...

    @property
    def attribution(self) -> str:
        """Attribution string — must be displayed when using data."""
        ...

    def fetch(self, d: date) -> RateFixing | None:
        """Fetch fixings for a specific date."""
        ...

    def fetch_year(self, year: int) -> list[RateFixing]:
        """Fetch all daily fixings for a year."""
        ...
