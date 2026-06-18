"""Repo rate sensitivities — trade and portfolio level.

    from pricebook.fixed_income.repo_greeks import (
        repo_dv01, carry_sensitivity_ladder, repo_portfolio_greeks,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.core.day_count import DayCountConvention


@dataclass
class RepoGreeksResult:
    """Repo rate sensitivities for a single trade."""
    repo_dv01: float             # PV change per 1bp repo rate shift
    carry_dv01: float            # carry change per 1bp repo rate shift
    roll_theta: float            # daily carry (P&L from 1 day passing)
    duration_mismatch: float     # bond duration - repo tenor (gap risk)

    def to_dict(self) -> dict:
        return dict(vars(self))


def repo_dv01(
    notional: float,
    repo_rate: float,
    repo_days: int,
    bond_price: float = 100.0,
    day_count_denom: float = 360.0,
) -> RepoGreeksResult:
    """Compute repo rate DV01 for a single repo trade.

    repo_dv01 = notional × t / denom  (interest sensitivity to 1bp)
    carry_dv01 = -bond_price × t / denom  (carry change per 1bp rate move)
    roll_theta = daily carry = (coupon_yield - repo_rate) × notional / denom

    Args:
        notional: repo cash amount.
        repo_rate: agreed repo rate.
        repo_days: term in days.
        bond_price: collateral price (for carry).
        day_count_denom: 360 for ACT/360, 365 for ACT/365.
    """
    t = repo_days / day_count_denom
    bp = 0.0001

    # Interest DV01: change in interest payment per 1bp
    interest_base = notional * repo_rate * t
    interest_up = notional * (repo_rate + bp) * t
    rdv01 = interest_up - interest_base

    # Carry DV01: change in carry per 1bp repo rate
    # carry = coupon - financing, financing = price × rate × t
    carry_dv01_val = -bond_price * bp * t * notional / 100.0

    # Roll theta: daily carry
    roll = notional * repo_rate / day_count_denom

    return RepoGreeksResult(
        repo_dv01=rdv01,
        carry_dv01=carry_dv01_val,
        roll_theta=roll,
        duration_mismatch=0.0,
    )


@dataclass
class CarrySensitivityBucket:
    """Carry sensitivity for a tenor bucket."""
    bucket: str
    dv01: float
    pct_of_total: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def carry_sensitivity_ladder(
    trades: list[dict],
    buckets: list[tuple[int, int]] | None = None,
) -> list[CarrySensitivityBucket]:
    """Carry sensitivity by tenor bucket.

    Args:
        trades: list of dicts with keys: notional, repo_rate, repo_days, bond_price.
        buckets: [(min_days, max_days), ...]. Default: O/N, 1W, 1M, 3M, 6M, 1Y+.
    """
    if buckets is None:
        buckets = [(0, 1), (2, 7), (8, 30), (31, 90), (91, 180), (181, 9999)]
        labels = ["O/N", "1W", "1M", "3M", "6M", "1Y+"]
    else:
        labels = [f"{lo}-{hi}d" for lo, hi in buckets]

    bucket_dv01 = [0.0] * len(buckets)
    for trade in trades:
        days = trade["repo_days"]
        greeks = repo_dv01(
            trade["notional"], trade["repo_rate"], days,
            trade.get("bond_price", 100.0),
            trade.get("day_count_denom", 360.0),
        )
        for i, (lo, hi) in enumerate(buckets):
            if lo <= days <= hi:
                bucket_dv01[i] += greeks.repo_dv01
                break

    total = sum(abs(d) for d in bucket_dv01) or 1.0
    return [
        CarrySensitivityBucket(
            bucket=labels[i],
            dv01=bucket_dv01[i],
            pct_of_total=abs(bucket_dv01[i]) / total * 100,
        )
        for i in range(len(buckets))
    ]


@dataclass
class RepoPortfolioGreeks:
    """Aggregated repo Greeks for a portfolio."""
    total_repo_dv01: float
    total_carry_dv01: float
    total_roll_theta: float
    n_trades: int
    weighted_avg_tenor_days: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def repo_portfolio_greeks(trades: list[dict]) -> RepoPortfolioGreeks:
    """Aggregate repo Greeks across a portfolio.

    Args:
        trades: list of dicts with: notional, repo_rate, repo_days, bond_price.
    """
    total_rdv01 = 0.0
    total_cdv01 = 0.0
    total_theta = 0.0
    total_notional = 0.0
    weighted_tenor = 0.0

    for t in trades:
        g = repo_dv01(
            t["notional"], t["repo_rate"], t["repo_days"],
            t.get("bond_price", 100.0),
            t.get("day_count_denom", 360.0),
        )
        total_rdv01 += g.repo_dv01
        total_cdv01 += g.carry_dv01
        total_theta += g.roll_theta
        total_notional += abs(t["notional"])
        weighted_tenor += abs(t["notional"]) * t["repo_days"]

    avg_tenor = weighted_tenor / total_notional if total_notional > 0 else 0.0

    return RepoPortfolioGreeks(
        total_repo_dv01=total_rdv01,
        total_carry_dv01=total_cdv01,
        total_roll_theta=total_theta,
        n_trades=len(trades),
        weighted_avg_tenor_days=avg_tenor,
    )
