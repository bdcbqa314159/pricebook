"""Repo term structure: term repo pricing, forward repo, specials.

Builds a repo rate term structure from observable GC (general collateral)
rates and derives forward repo rates for carry analysis.

    from pricebook.repo_term import RepoCurve, forward_repo_rate

References:
    Choudhry, *The Repo Handbook*, Butterworth-Heinemann, 2010.
    Tuckman & Serrat, *Fixed Income Securities*, Wiley, 2012, Ch. 15.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction


@dataclass
class RepoRate:
    """A single repo rate observation."""
    tenor_days: int
    rate: float      # simple rate, annualised


class RepoCurve:
    """Term structure of repo rates.

    Built from GC repo rates at standard tenors (O/N, 1W, 2W, 1M, 3M, 6M).
    Interpolates linearly in rate space for intermediate tenors.

    Args:
        reference_date: curve date.
        tenors: list of RepoRate observations, sorted by tenor.
    """

    def __init__(self, reference_date: date, tenors: list[RepoRate]):
        if not tenors:
            raise ValueError("need at least 1 repo rate")
        self.reference_date = reference_date
        self._tenors = sorted(tenors, key=lambda r: r.tenor_days)
        self._days = [r.tenor_days for r in self._tenors]
        self._rates = [r.rate for r in self._tenors]

    def rate(self, days: int) -> float:
        """Interpolated repo rate for a given tenor in days."""
        if days <= self._days[0]:
            return self._rates[0]
        if days >= self._days[-1]:
            return self._rates[-1]
        # Linear interpolation
        for i in range(len(self._days) - 1):
            if self._days[i] <= days <= self._days[i + 1]:
                frac = (days - self._days[i]) / (self._days[i + 1] - self._days[i])
                return self._rates[i] + frac * (self._rates[i + 1] - self._rates[i])
        return self._rates[-1]

    def discount_factor(self, days: int) -> float:
        """Discount factor from simple repo rate: df = 1 / (1 + r × T)."""
        r = self.rate(days)
        return 1.0 / (1.0 + r * days / 360.0)


def forward_repo_rate(
    repo_curve: RepoCurve,
    start_days: int,
    end_days: int,
) -> float:
    """Forward repo rate between two future dates.

    Derived from no-arbitrage: df(0,T2) = df(0,T1) × df(T1,T2).
    Forward rate = (df1/df2 - 1) / (T2-T1) × 360.
    """
    if end_days <= start_days:
        return repo_curve.rate(start_days)
    df1 = repo_curve.discount_factor(start_days)
    df2 = repo_curve.discount_factor(end_days)
    dt = (end_days - start_days) / 360.0
    return (df1 / df2 - 1.0) / dt


@dataclass
class SpecialRepoSpread:
    """Special repo spread analysis."""
    collateral: str
    gc_rate: float
    special_rate: float
    spread: float           # GC - special (positive = on special)
    is_special: bool        # spread > threshold


def identify_specials(
    gc_rate: float,
    collateral_rates: dict[str, float],
    threshold: float = 0.0025,
) -> list[SpecialRepoSpread]:
    """Identify bonds trading special in the repo market.

    A bond is "on special" when its repo rate is significantly below GC.

    Args:
        gc_rate: general collateral repo rate.
        collateral_rates: {bond_id: repo_rate} for specific bonds.
        threshold: minimum spread to flag as special (default 25bp).
    """
    results = []
    for name, rate in sorted(collateral_rates.items()):
        spread = gc_rate - rate
        results.append(SpecialRepoSpread(
            collateral=name,
            gc_rate=gc_rate,
            special_rate=rate,
            spread=spread,
            is_special=spread > threshold,
        ))
    return results
