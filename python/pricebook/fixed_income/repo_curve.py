"""Repo term structure: spot rates, forward rates, special vs GC spread.

Builds a repo rate curve from market data and derives forward repo rates
via no-arbitrage. Tracks special vs GC spread by issue.

    from pricebook.fixed_income.repo_curve import (
        RepoCurve, build_repo_curve, forward_repo_rate,
        special_gc_spread, repo_carry_from_curve,
    )

References:
    Tuckman & Serrat (2012). Fixed Income Securities, Ch. 12 (Repo).
    Choudhry (2010). The Repo Handbook, Ch. 3-4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
from dateutil.relativedelta import relativedelta

from pricebook.core.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# Repo curve
# ---------------------------------------------------------------------------

@dataclass
class RepoCurve:
    """Term structure of repo rates.

    Stores repo rates at standard tenors and interpolates.
    Supports GC (general collateral) and special repo curves.
    """
    reference_date: date
    tenors_days: list[int]
    rates: list[float]
    curve_type: str = "GC"  # "GC" or "special"

    def rate_at(self, days: int) -> float:
        """Interpolated repo rate at given tenor (days)."""
        return float(np.interp(days, self.tenors_days, self.rates))

    def rate_at_date(self, d: date) -> float:
        """Repo rate to a specific date."""
        days = (d - self.reference_date).days
        return self.rate_at(max(days, 1))

    def discount_factor(self, d: date) -> float:
        """Repo discount factor: df = 1 / (1 + r × days/360)."""
        days = (d - self.reference_date).days
        r = self.rate_at(max(days, 1))
        return 1.0 / (1.0 + r * days / 360.0)

    def to_dict(self) -> dict:
        return {"date": self.reference_date.isoformat(),
                "tenors_days": self.tenors_days, "rates": self.rates,
                "type": self.curve_type}


def build_repo_curve(
    reference_date: date,
    rates: dict[str, float],
    curve_type: str = "GC",
) -> RepoCurve:
    """Build a repo curve from market rate quotes.

    Args:
        rates: {"ON": 0.053, "1W": 0.0528, "1M": 0.0525, "3M": 0.052, ...}
        curve_type: "GC" or "special".
    """
    tenor_map = {
        "ON": 1, "TN": 2, "1W": 7, "2W": 14, "3W": 21,
        "1M": 30, "2M": 60, "3M": 91, "6M": 182, "9M": 273, "1Y": 365,
    }
    tenors = []
    rate_vals = []
    for label, r in sorted(rates.items(), key=lambda x: tenor_map.get(x[0], 999)):
        days = tenor_map.get(label)
        if days is None:
            continue
        tenors.append(days)
        rate_vals.append(r)

    return RepoCurve(reference_date, tenors, rate_vals, curve_type)


# ---------------------------------------------------------------------------
# Forward repo rate
# ---------------------------------------------------------------------------

@dataclass
class ForwardRepoResult:
    """Forward repo rate between two dates."""
    start_date: date
    end_date: date
    forward_rate: float
    spot_rate_start: float
    spot_rate_end: float

    def to_dict(self) -> dict:
        return {"start": self.start_date.isoformat(), "end": self.end_date.isoformat(),
                "forward_rate": self.forward_rate}


def forward_repo_rate(
    curve: RepoCurve,
    start_date: date,
    end_date: date,
) -> ForwardRepoResult:
    """No-arbitrage forward repo rate.

    Forward = [(1 + r2 × d2/360) / (1 + r1 × d1/360) - 1] × 360 / (d2 - d1)

    Args:
        start_date: forward start.
        end_date: forward end.
    """
    ref = curve.reference_date
    d1 = (start_date - ref).days
    d2 = (end_date - ref).days
    if d2 <= d1:
        raise ValueError(f"end_date must be after start_date")

    r1 = curve.rate_at(max(d1, 1))
    r2 = curve.rate_at(d2)

    df1 = 1.0 / (1.0 + r1 * d1 / 360.0)
    df2 = 1.0 / (1.0 + r2 * d2 / 360.0)

    fwd_days = d2 - d1
    fwd_rate = (df1 / df2 - 1.0) * 360.0 / fwd_days

    return ForwardRepoResult(start_date, end_date, fwd_rate, r1, r2)


# ---------------------------------------------------------------------------
# Special vs GC spread
# ---------------------------------------------------------------------------

@dataclass
class SpecialGCSpread:
    """Special repo rate vs GC rate for a specific bond."""
    bond_label: str
    special_rate: float
    gc_rate: float
    spread_bps: float
    is_special: bool
    tenor_days: int

    def to_dict(self) -> dict:
        return dict(vars(self))


def special_gc_spread(
    bond_label: str,
    special_rate: float,
    gc_curve: RepoCurve,
    tenor_days: int = 1,
    threshold_bps: float = 25.0,
) -> SpecialGCSpread:
    """Compute special-GC spread for a bond.

    A bond is "on special" when its repo rate is significantly below GC.
    Threshold: typically 25bp below GC = on special.
    """
    gc_rate = gc_curve.rate_at(tenor_days)
    spread = (gc_rate - special_rate) * 1e4

    return SpecialGCSpread(
        bond_label=bond_label, special_rate=special_rate,
        gc_rate=gc_rate, spread_bps=spread,
        is_special=spread > threshold_bps, tenor_days=tenor_days,
    )


# ---------------------------------------------------------------------------
# Repo-financed carry from curve
# ---------------------------------------------------------------------------

@dataclass
class RepoCarryResult:
    """Carry computation using repo term structure."""
    coupon_income: float
    repo_cost: float
    net_carry: float
    annualised_carry_bps: float
    repo_rate_used: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def repo_carry_from_curve(
    bond_price: float,
    coupon_rate: float,
    repo_curve: RepoCurve,
    hold_days: int = 30,
    face: float = 100.0,
) -> RepoCarryResult:
    """Compute carry using the repo term structure (not a flat rate).

    Carry = coupon income - repo financing cost over hold period.
    Uses term-matched repo rate from the curve.
    """
    repo_rate = repo_curve.rate_at(hold_days)

    coupon_income = coupon_rate * face * hold_days / 365.0
    repo_cost = bond_price * repo_rate * hold_days / 360.0
    net_carry = coupon_income - repo_cost
    ann_carry_bps = net_carry / bond_price * 365.0 / hold_days * 1e4

    return RepoCarryResult(coupon_income, repo_cost, net_carry, ann_carry_bps, repo_rate)
