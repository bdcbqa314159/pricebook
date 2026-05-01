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
                denom = self._days[i + 1] - self._days[i]
                if denom <= 0:
                    return self._rates[i]
                frac = (days - self._days[i]) / denom
                return self._rates[i] + frac * (self._rates[i + 1] - self._rates[i])
        return self._rates[-1]

    def discount_factor(self, days: int) -> float:
        """Discount factor from simple repo rate: df = 1 / (1 + r × T)."""
        r = self.rate(days)
        return 1.0 / (1.0 + r * days / 360.0)


    def as_discount_curve(self) -> "DiscountCurve":
        """Convert to DiscountCurve for use with standard pricing infrastructure.

        Uses simple-rate discount factors: df = 1 / (1 + r × T).
        This bridges RepoCurve into any function expecting DiscountCurve.
        """
        from pricebook.discount_curve import DiscountCurve
        from pricebook.day_count import date_from_year_fraction
        from datetime import timedelta

        dates = [self.reference_date + timedelta(days=d) for d in self._days]
        dfs = [self.discount_factor(d) for d in self._days]
        return DiscountCurve(self.reference_date, dates, dfs)


def repo_ois_basis(
    repo_curve: RepoCurve,
    ois_curve: "DiscountCurve",
    tenors_days: list[int] | None = None,
) -> list[dict[str, float]]:
    """Repo-OIS basis at each tenor in basis points.

    basis = repo_rate − OIS_zero_rate (in bp).
    Positive: repo > OIS (typical for GC).
    Negative: repo < OIS (on special).
    """
    if tenors_days is None:
        tenors_days = [1, 7, 30, 90, 180, 360]

    results = []
    for days in tenors_days:
        repo_r = repo_curve.rate(days)
        d = repo_curve.reference_date + __import__("datetime").timedelta(days=days)
        ois_r = ois_curve.zero_rate(d)
        basis_bp = (repo_r - ois_r) * 10_000
        results.append({
            "tenor_days": days,
            "repo_rate": repo_r,
            "ois_rate": ois_r,
            "basis_bp": basis_bp,
        })
    return results


def term_repo_carry(
    bond_dirty_price: float,
    coupon_income: float,
    repo_rate: float,
    ois_rate: float,
    holding_days: int,
    face_value: float = 100.0,
) -> dict[str, float]:
    """Carry decomposition: repo cost vs OIS cost vs coupon income.

    net_carry = coupon_income − repo_cost
    carry_advantage = ois_cost − repo_cost (positive when repo < OIS, on special)
    """
    yf = holding_days / 360.0
    repo_cost = bond_dirty_price * repo_rate * yf
    ois_cost = bond_dirty_price * ois_rate * yf
    net_carry = coupon_income - repo_cost
    carry_advantage = ois_cost - repo_cost

    return {
        "coupon_income": coupon_income,
        "repo_cost": repo_cost,
        "ois_cost": ois_cost,
        "net_carry": net_carry,
        "carry_advantage_bp": carry_advantage / bond_dirty_price * 10_000,
    }


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
    return (df1 - df2) / (dt * df2)


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
