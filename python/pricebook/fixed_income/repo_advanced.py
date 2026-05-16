"""Advanced repo: curve construction, specials, multi-counterparty financing.

* :class:`RepoCurve` — term structure of repo rates.
* :func:`build_repo_curve` — bootstrap from tenor quotes.
* :func:`repo_spread_to_ois` — repo spread vs OIS (GC spread).
* :func:`identify_specials` — detect bonds trading special.
* :class:`RepoCounterparty` — counterparty for financing optimisation.
* :func:`optimise_financing` — multi-counterparty financing optimisation.
* :func:`repo_haircut_curve` — haircut term structure by asset type.

References:
    Choudhry, *The Repo Handbook*, Butterworth-Heinemann, 2010.
    Duffie, *Special Repo Rates*, J. Finance, 1996.
    Krishnamurthy, *The Bond/Old-Bond Spread*, J. Financial Econ., 2002.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---- Repo curve ----

@dataclass
class RepoCurve:
    """Repo curve: tenor → rate (simple)."""
    tenors_days: np.ndarray
    rates_pct: np.ndarray
    collateral_type: str            # "GC", "specials", or ticker

    def rate_at(self, days: int) -> float:
        """Rate at a given tenor (days)."""
        if days <= self.tenors_days[0]:
            return float(self.rates_pct[0])
        if days >= self.tenors_days[-1]:
            return float(self.rates_pct[-1])
        return float(np.interp(days, self.tenors_days, self.rates_pct))

    def financing_cost(self, notional: float, days: int) -> float:
        """Financing cost for borrowing at this repo rate."""
        rate = self.rate_at(days)
        return notional * rate / 100 * days / 360


def build_repo_curve(
    tenors_days: list[int],
    rates_pct: list[float],
    collateral_type: str = "GC",
) -> RepoCurve:
    """Construct a repo curve from observed tenor points."""
    return RepoCurve(
        tenors_days=np.array(tenors_days),
        rates_pct=np.array(rates_pct),
        collateral_type=collateral_type,
    )


# ---- Repo-OIS spread ----

@dataclass
class RepoOISSpread:
    """Repo spread vs OIS."""
    tenor_days: int
    repo_rate_pct: float
    ois_rate_pct: float
    spread_bps: float
    regime: str                     # "normal", "tightening", "stressed"


def repo_spread_to_ois(
    repo_curve: RepoCurve,
    ois_rate_pct: float,
    tenor_days: int = 1,
) -> RepoOISSpread:
    """GC repo vs OIS spread.

    Normal: -5 to +5 bps.
    Tightening: >10 bps (repo > OIS, funding stress).
    Stressed: >25 bps or extremes.
    """
    repo_rate = repo_curve.rate_at(tenor_days)
    spread_bps = (repo_rate - ois_rate_pct) * 100

    if abs(spread_bps) < 5:
        regime = "normal"
    elif abs(spread_bps) < 25:
        regime = "tightening"
    else:
        regime = "stressed"

    return RepoOISSpread(
        tenor_days=tenor_days,
        repo_rate_pct=repo_rate,
        ois_rate_pct=ois_rate_pct,
        spread_bps=float(spread_bps),
        regime=regime,
    )


# ---- Specials identification ----

@dataclass
class SpecialBond:
    """Bond trading special in repo."""
    bond_id: str
    gc_rate_pct: float
    special_rate_pct: float
    specialness_bps: float          # GC - special
    is_ultra_special: bool


def identify_specials(
    gc_rate_pct: float,
    bond_repo_rates: dict[str, float],
    specialness_threshold_bps: float = 10.0,
    ultra_threshold_bps: float = 50.0,
) -> list[SpecialBond]:
    """Identify bonds trading special (lower repo rate than GC).

    Args:
        gc_rate_pct: general collateral repo rate.
        bond_repo_rates: {bond_id: repo_rate_pct}.
        specialness_threshold_bps: flag if GC - special > this.
        ultra_threshold_bps: flag ultra-special (very tight).
    """
    specials = []
    for bond_id, rate in bond_repo_rates.items():
        specialness = (gc_rate_pct - rate) * 100
        if specialness > specialness_threshold_bps:
            specials.append(SpecialBond(
                bond_id=bond_id,
                gc_rate_pct=gc_rate_pct,
                special_rate_pct=rate,
                specialness_bps=float(specialness),
                is_ultra_special=specialness > ultra_threshold_bps,
            ))
    return sorted(specials, key=lambda s: -s.specialness_bps)


# ---- Counterparty financing ----

@dataclass
class RepoCounterparty:
    """A counterparty for repo financing."""
    name: str
    rate_pct: float                 # offered rate
    max_capacity: float              # max notional they'll lend
    haircut_pct: float              # required haircut
    credit_quality: str             # "AAA", "AA", etc.


@dataclass
class FinancingPlan:
    """Result of multi-counterparty financing optimisation."""
    total_notional: float
    total_cost: float
    counterparty_allocations: dict[str, float]     # name → allocated notional
    avg_weighted_rate_pct: float
    unmet_demand: float             # amount we couldn't finance


def optimise_financing(
    notional_needed: float,
    counterparties: list[RepoCounterparty],
    days: int = 1,
) -> FinancingPlan:
    """Greedy optimisation: allocate to cheapest counterparties first.

    Respects capacity constraints. Simple sorting — more sophisticated
    approaches could use LP.
    """
    # Sort by rate ascending
    sorted_cps = sorted(counterparties, key=lambda c: c.rate_pct)

    remaining = notional_needed
    allocations = {}
    total_cost = 0.0

    for cp in sorted_cps:
        if remaining <= 0:
            break
        alloc = min(remaining, cp.max_capacity)
        allocations[cp.name] = alloc
        total_cost += alloc * cp.rate_pct / 100 * days / 360
        remaining -= alloc

    total_alloc = notional_needed - remaining
    weighted_rate = (total_cost / max(total_alloc, 1e-10) * 360 / days) * 100 if total_alloc > 0 else 0.0

    return FinancingPlan(
        total_notional=float(total_alloc),
        total_cost=float(total_cost),
        counterparty_allocations=allocations,
        avg_weighted_rate_pct=float(weighted_rate),
        unmet_demand=float(max(remaining, 0.0)),
    )


# ---- Haircut curve ----

@dataclass
class HaircutCurve:
    """Repo haircut term structure by asset type."""
    asset_type: str
    tenor_days: np.ndarray
    haircuts_pct: np.ndarray

    def haircut_at(self, days: int) -> float:
        if days <= self.tenor_days[0]:
            return float(self.haircuts_pct[0])
        if days >= self.tenor_days[-1]:
            return float(self.haircuts_pct[-1])
        return float(np.interp(days, self.tenor_days, self.haircuts_pct))


def repo_haircut_curve(
    asset_type: str = "treasury",
) -> HaircutCurve:
    """Typical haircut curve by asset type.

    Treasuries: 1-3%.
    IG corporates: 3-8%.
    HY corporates: 10-20%.
    Equities: 15-30%.
    """
    if asset_type == "treasury":
        haircuts = [1.0, 1.5, 2.0, 3.0]
    elif asset_type == "ig_corp":
        haircuts = [3.0, 4.0, 5.5, 8.0]
    elif asset_type == "hy_corp":
        haircuts = [10.0, 12.0, 15.0, 20.0]
    elif asset_type == "equity":
        haircuts = [15.0, 20.0, 25.0, 30.0]
    else:
        haircuts = [5.0, 7.0, 10.0, 15.0]

    tenors = [1, 30, 90, 365]
    return HaircutCurve(
        asset_type=asset_type,
        tenor_days=np.array(tenors),
        haircuts_pct=np.array(haircuts),
    )
