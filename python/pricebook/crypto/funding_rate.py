"""Funding rate analytics: term structure, carry, prediction.

* :class:`FundingCurve` — term structure of funding rates.
* :func:`funding_carry` — annualised carry from funding.
* :func:`predicted_funding` — next funding rate prediction.
* :func:`historical_funding_stats` — statistics on historical funding.

References:
    BitMEX, *Funding Rate Mechanism*.
    Deribit, *Funding Rate Calculation*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FundingCurve:
    """Term structure of funding rates.

    Built from futures basis at different expiries.
    """
    dates: list                 # settlement dates or labels
    rates: list[float]          # annualised funding rate per tenor
    spot: float
    source: str = ""

    def rate_at(self, idx: int) -> float:
        return self.rates[idx] if idx < len(self.rates) else 0.0

    @property
    def front(self) -> float:
        """Front (nearest) funding rate."""
        return self.rates[0] if self.rates else 0.0

    @property
    def back(self) -> float:
        """Back (farthest) funding rate."""
        return self.rates[-1] if self.rates else 0.0

    @property
    def slope(self) -> float:
        """Curve slope: back − front."""
        return self.back - self.front

    def to_dict(self) -> dict:
        return {
            "n_tenors": len(self.rates),
            "front_rate": self.front,
            "back_rate": self.back,
            "slope": self.slope,
        }


def funding_from_futures_basis(
    spot: float,
    futures_prices: list[float],
    days_to_expiry: list[float],
    interval_hours: float = 8.0,
) -> FundingCurve:
    """Derive funding rate term structure from futures basis.

    Each futures contract implies an annualised basis:
    basis_ann = (F/S − 1) × (365 / days_to_expiry)

    Convert to per-interval funding rate:
    funding = basis_ann × interval_hours / (365 × 24)

    Args:
        spot: current spot price.
        futures_prices: futures prices at different expiries.
        days_to_expiry: days to expiry per contract.
        interval_hours: funding interval (8h default).
    """
    rates = []
    for F, days in zip(futures_prices, days_to_expiry):
        if spot > 0 and days > 0:
            basis_ann = (F / spot - 1) * (365 / days)
            funding = basis_ann * interval_hours / (365 * 24)
            rates.append(funding)
        else:
            rates.append(0.0)

    return FundingCurve(
        dates=[f"{int(d)}d" for d in days_to_expiry],
        rates=rates,
        spot=spot,
    )


@dataclass
class FundingCarryResult:
    """Funding carry analysis."""
    annualised_carry_pct: float     # annualised carry in %
    daily_carry: float              # daily carry per unit notional
    monthly_carry: float
    position_side: str
    funding_rate: float

    def to_dict(self) -> dict:
        return vars(self)


def funding_carry(
    funding_rate: float,
    notional: float = 100_000.0,
    interval_hours: float = 8.0,
    side: str = "short",
) -> FundingCarryResult:
    """Annualised carry from funding rate.

    When funding > 0: shorts earn carry (longs pay).
    When funding < 0: longs earn carry (shorts pay).

    Args:
        funding_rate: per-interval funding rate.
        notional: position notional.
        interval_hours: funding interval.
        side: "long" or "short".
    """
    intervals_per_day = 24 / interval_hours
    intervals_per_year = 365 * intervals_per_day

    # Positive funding: shorts collect
    if side == "short":
        carry_per_interval = funding_rate * notional
    else:
        carry_per_interval = -funding_rate * notional

    daily = carry_per_interval * intervals_per_day
    annualised = funding_rate * intervals_per_year * 100

    return FundingCarryResult(
        annualised_carry_pct=annualised if side == "short" else -annualised,
        daily_carry=daily,
        monthly_carry=daily * 30,
        position_side=side,
        funding_rate=funding_rate,
    )


def predicted_funding(
    recent_rates: list[float],
    method: str = "ewma",
    halflife: int = 12,
) -> float:
    """Predict next funding rate.

    Methods:
    - "last": use last observed rate.
    - "mean": average of recent rates.
    - "ewma": exponentially weighted moving average.

    Args:
        recent_rates: list of recent funding rates (newest last).
        method: prediction method.
        halflife: EWMA halflife in periods.
    """
    if not recent_rates:
        return 0.0

    if method == "last":
        return recent_rates[-1]
    elif method == "mean":
        return sum(recent_rates) / len(recent_rates)
    elif method == "ewma":
        alpha = 1 - math.exp(-math.log(2) / halflife)
        ewma = recent_rates[0]
        for r in recent_rates[1:]:
            ewma = alpha * r + (1 - alpha) * ewma
        return ewma
    return recent_rates[-1]


@dataclass
class FundingStatsResult:
    """Historical funding rate statistics."""
    mean: float
    median: float
    std: float
    min_rate: float
    max_rate: float
    pct_positive: float         # % of periods with positive funding
    annualised_mean: float      # mean × intervals_per_year × 100
    n_periods: int

    def to_dict(self) -> dict:
        return vars(self)


def historical_funding_stats(
    rates: list[float],
    interval_hours: float = 8.0,
) -> FundingStatsResult:
    """Statistics on historical funding rates.

    Args:
        rates: list of historical funding rates.
        interval_hours: funding interval.
    """
    if not rates:
        return FundingStatsResult(0, 0, 0, 0, 0, 0, 0, 0)

    arr = np.array(rates)
    intervals_per_year = 365 * 24 / interval_hours

    return FundingStatsResult(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        min_rate=float(np.min(arr)),
        max_rate=float(np.max(arr)),
        pct_positive=float(np.mean(arr > 0) * 100),
        annualised_mean=float(np.mean(arr) * intervals_per_year * 100),
        n_periods=len(rates),
    )
