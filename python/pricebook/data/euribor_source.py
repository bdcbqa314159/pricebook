"""Euribor rate source: implements RateSource protocol.

Fetches from euriborrates.com and returns RateFixing objects.
This module knows about HTML parsing and the website structure.
The MarketCurve builder doesn't.

DATA SOURCE: https://euriborrates.com/
    An independent, non-commercial, non-profit information resource.
"""

from __future__ import annotations

from datetime import date

from pricebook.data.rate_source import (
    RateSource, RateFixing, RateType, TenorDefinition, DEPOSIT_TENORS,
)
from pricebook.data.euribor_loader import (
    fetch_date as _fetch_date,
    fetch_year_all_tenors as _fetch_year_all,
    TENORS as _TENORS,
    attribution as _attribution,
)

_SOURCE = "euriborrates.com"


class EuriborSource:
    """Euribor deposit rates from euriborrates.com.

    Implements the RateSource protocol.

    Data source: https://euriborrates.com/
    """

    @property
    def currency(self) -> str:
        return "EUR"

    @property
    def rate_type(self) -> RateType:
        return RateType.DEPOSIT

    @property
    def tenors(self) -> list[TenorDefinition]:
        return DEPOSIT_TENORS

    @property
    def source_name(self) -> str:
        return _SOURCE

    @property
    def attribution(self) -> str:
        return _attribution()

    def fetch(self, d: date) -> RateFixing | None:
        """Fetch Euribor fixings for a specific date.

        Returns None if no data (weekend/holiday).
        """
        raw = _fetch_date(d)
        if raw is None:
            return None
        return RateFixing(
            date=raw.date,
            rates=dict(raw.rates),
            rate_type=RateType.DEPOSIT,
            source=_SOURCE,
            currency="EUR",
        )

    def fetch_year(self, year: int) -> list[RateFixing]:
        """Fetch all business days for a year (all 5 tenors)."""
        raw_list = _fetch_year_all(year)
        return [
            RateFixing(
                date=raw.date,
                rates=dict(raw.rates),
                rate_type=RateType.DEPOSIT,
                source=_SOURCE,
                currency="EUR",
            )
            for raw in raw_list
        ]
