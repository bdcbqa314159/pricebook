"""Commodity structured products: autocallables, linked bonds, TARFs, range notes.

* :func:`commodity_autocallable` — autocall on single commodity.
* :func:`commodity_linked_bond` — coupon tied to commodity index.
* :func:`commodity_tarf` — Target Redemption Forward on commodity.
* :func:`commodity_range_accrual` — daily in-range coupon.
* :func:`dual_commodity_note` — long one, short another.

References:
    Clewlow & Strickland, *Energy Derivatives*, Wiley, 2000, Ch. 9.
    Geman, *Commodities and Commodity Derivatives*, Wiley, 2005.
    Bouzoubaa & Osseiran, *Exotic Options and Hybrids*, Wiley, 2010.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Commodity autocallable ----

@dataclass
class CommodityAutocallResult:
    """Commodity autocallable result."""
    price: float
    autocall_probability: float
    mean_autocall_time: float
    coupon_rate: float
    notional: float


def commodity_autocallable(
    spot_paths: np.ndarray,         # (n_paths, n_obs+1)
    autocall_barrier: float,
    coupon: float,
    discount_factors: np.ndarray,   # (n_obs+1,)
    observation_dates: list[int],   # indices into paths time axis
    notional: float = 1.0,
    protection_barrier: float | None = None,
    has_memory: bool = True,
) -> CommodityAutocallResult:
    """Commodity autocallable (gold/oil/wheat).

    At each observation date, if S >= autocall_barrier → pay notional + coupons.
    At maturity if not autocalled: soft protection if S >= protection_barrier.

    Args:
        autocall_barrier: level triggering early redemption.
        coupon: per-period coupon (if autocalled or at maturity).
        observation_dates: indices in spot_paths (excluding t=0).
        protection_barrier: soft protection level at expiry.
        has_memory: if True, unpaid coupons accrue and pay on autocall.
    """
    n_paths = spot_paths.shape[0]
    obs_sorted = sorted(observation_dates)

    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)
    missed = np.zeros(n_paths)
    autocall_time_idx = np.full(n_paths, obs_sorted[-1])

    for obs_idx in obs_sorted:
        S = spot_paths[:, obs_idx]
        df = discount_factors[obs_idx]

        triggered = alive & (S >= autocall_barrier)

        if has_memory:
            payout = notional + missed + coupon
            pv += np.where(triggered, payout * df, 0.0)
            missed = np.where(alive & ~triggered, missed + coupon, missed)
        else:
            pv += np.where(triggered, (notional + coupon) * df, 0.0)
            pv += np.where(alive & ~triggered, coupon * df, 0.0)

        autocall_time_idx = np.where(triggered & alive, obs_idx, autocall_time_idx)
        alive &= ~triggered

    # Terminal
    T_idx = obs_sorted[-1]
    df_T = discount_factors[T_idx]
    S_T = spot_paths[:, T_idx]

    if protection_barrier is None:
        pv += np.where(alive, notional * df_T, 0.0)
    else:
        # If below protection, lose proportional to drop
        terminal = np.where(S_T >= protection_barrier, notional,
                             notional * S_T / spot_paths[:, 0])
        pv += np.where(alive, terminal * df_T, 0.0)

    ac_prob = float(1 - alive.mean())
    price = float(pv.mean())

    if ac_prob > 1e-6:
        # Mean autocall time index → approximate time
        mask = autocall_time_idx < obs_sorted[-1]
        mean_ac_idx = float(autocall_time_idx[mask].mean()) if mask.sum() > 0 else float(T_idx)
    else:
        mean_ac_idx = float(T_idx)

    return CommodityAutocallResult(
        price=price,
        autocall_probability=ac_prob,
        mean_autocall_time=mean_ac_idx,
        coupon_rate=coupon / notional,
        notional=notional,
    )


# ---- Commodity-linked bond ----

@dataclass
class CommodityLinkedBondResult:
    """Commodity-linked bond result."""
    price: float
    bond_floor: float
    commodity_upside: float
    participation: float
    notional: float


def commodity_linked_bond(
    spot_paths: np.ndarray,         # (n_paths, n_times+1)
    base_coupon: float,             # fixed base coupon
    participation: float,           # % of commodity return added
    discount_factors: np.ndarray,
    notional: float = 1.0,
    coupon_dates: list[int] | None = None,
) -> CommodityLinkedBondResult:
    """Bond with coupons tied to commodity performance.

    Coupon at each date = base_coupon + participation × max(S/S₀ − 1, 0).
    Principal returned at maturity.

    Args:
        spot_paths: commodity paths.
        base_coupon: guaranteed coupon rate.
        participation: fraction of commodity performance.
        coupon_dates: indices for coupon payments (default: every year).
    """
    n_paths, n_times = spot_paths.shape
    if coupon_dates is None:
        # Assume annual coupons, one per year
        n_years = n_times - 1
        coupon_dates = list(range(1, n_years + 1))

    S0 = spot_paths[:, 0]
    pv = np.zeros(n_paths)

    for idx in coupon_dates:
        if idx >= n_times:
            continue
        S = spot_paths[:, idx]
        commodity_return = np.maximum(S / S0 - 1, 0)
        coupon_paid = notional * (base_coupon + participation * commodity_return)
        pv += coupon_paid * discount_factors[idx]

    # Terminal principal
    pv += notional * discount_factors[-1]

    price = float(pv.mean())

    # Decomposition
    # Bond floor: base coupons + principal
    bond_floor = sum(notional * base_coupon * discount_factors[i]
                      for i in coupon_dates if i < n_times) + \
                 notional * discount_factors[-1]
    commodity_upside = price - bond_floor

    return CommodityLinkedBondResult(
        price=price,
        bond_floor=float(bond_floor),
        commodity_upside=float(commodity_upside),
        participation=participation,
        notional=notional,
    )


# ---- Commodity TARF ----

@dataclass
class CommodityTARFResult:
    """Commodity TARF result."""
    price: float
    target: float
    prob_target_hit: float
    mean_termination_time: float


def commodity_tarf(
    spot_paths: np.ndarray,         # (n_paths, n_obs+1)
    strike: float,
    target: float,
    discount_factors: np.ndarray,
    is_buyer_long: bool = True,
    notional: float = 1.0,
) -> CommodityTARFResult:
    """Commodity TARF: accumulates P&L until target reached, then terminates.

    Each observation: P&L = (S - K) × notional (or (K - S) for short).
    Cumulative P&L tracked; when ≥ target, contract terminates and pays target.

    Args:
        spot_paths: (n_paths, n_obs+1) commodity paths.
        strike: contract strike.
        target: cumulative target profit.
        is_buyer_long: True = buyer profits when S > K.
    """
    n_paths, n_obs = spot_paths.shape
    cum_profit = np.zeros(n_paths)
    pv = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    term_idx = np.full(n_paths, n_obs - 1)

    for i in range(1, n_obs):
        S = spot_paths[:, i]
        if is_buyer_long:
            pnl = (S - strike) * notional
        else:
            pnl = (strike - S) * notional

        df = discount_factors[i]

        new_cum = cum_profit + np.where(alive & (pnl > 0), pnl, 0.0)
        hit = alive & (new_cum >= target)

        # Cap profit at target; pay loss regardless if negative
        cap_pnl = np.where(hit, target - cum_profit, pnl)
        pv += np.where(alive, cap_pnl * df, 0.0)

        cum_profit = np.minimum(new_cum, target)
        term_idx = np.where(hit & alive, i, term_idx)
        alive &= ~hit

    prob_hit = float(1 - alive.mean())
    mean_term = float(term_idx.mean())

    return CommodityTARFResult(
        price=float(pv.mean()),
        target=target,
        prob_target_hit=prob_hit,
        mean_termination_time=mean_term,
    )


# ---- Commodity range accrual ----

@dataclass
class CommodityRangeAccrualResult:
    """Commodity range accrual result."""
    price: float
    accrual_rate: float
    range_low: float
    range_high: float
    n_observations: int


def commodity_range_accrual(
    spot_paths: np.ndarray,         # (n_paths, n_obs+1)
    range_low: float,
    range_high: float,
    coupon_per_day: float,
    discount_factor_T: float,
    notional: float = 1.0,
) -> CommodityRangeAccrualResult:
    """Commodity range accrual: pays coupon × fraction_of_days_in_range.

    Simplified with single terminal discounting.
    """
    n_paths, n_obs = spot_paths.shape
    # Count observations in range (exclude initial point)
    in_range = ((spot_paths[:, 1:] >= range_low)
                & (spot_paths[:, 1:] <= range_high))
    n_in_range = in_range.sum(axis=1)
    n_days = n_obs - 1

    accrual_rate = float(n_in_range.mean() / max(n_days, 1))
    payoff = notional * coupon_per_day * n_in_range
    price = float(discount_factor_T * payoff.mean())

    return CommodityRangeAccrualResult(
        price=price,
        accrual_rate=accrual_rate,
        range_low=range_low,
        range_high=range_high,
        n_observations=n_days,
    )


# ---- Dual-commodity note ----

@dataclass
class DualCommodityResult:
    """Dual commodity note result."""
    price: float
    commodity_long: str
    commodity_short: str
    participation: float
    floor_return: float


def dual_commodity_note(
    long_paths: np.ndarray,
    short_paths: np.ndarray,
    participation: float,
    floor_return: float,
    discount_factor_T: float,
    notional: float = 1.0,
    commodity_long_name: str = "oil",
    commodity_short_name: str = "gas",
) -> DualCommodityResult:
    """Dual commodity note: long one, short another.

    Payoff at T: notional × (1 + max(floor, participation × (S_long_return − S_short_return)))

    Args:
        long_paths: (n_paths, n_times+1) long leg prices.
        short_paths: (n_paths, n_times+1) short leg prices.
        participation: fraction of spread return applied.
        floor_return: minimum return (protection).
    """
    long_return = long_paths[:, -1] / long_paths[:, 0] - 1
    short_return = short_paths[:, -1] / short_paths[:, 0] - 1
    spread = long_return - short_return

    payoff = notional * (1 + np.maximum(floor_return, participation * spread))
    price = float(discount_factor_T * payoff.mean())

    return DualCommodityResult(
        price=price,
        commodity_long=commodity_long_name,
        commodity_short=commodity_short_name,
        participation=participation,
        floor_return=floor_return,
    )
