"""Equity structured products: autocallables, reverse convertibles, shark-fin.

* :func:`equity_autocallable` — phoenix autocall with memory coupon.
* :func:`worst_of_autocallable` — autocall on min(S₁, ..., S_n).
* :func:`reverse_convertible` — bond + short put.
* :func:`shark_fin_note` — capped call with knock-out.
* :func:`airbag_note` — capped-floored structure.

References:
    Bouzoubaa & Osseiran, *Exotic Options and Hybrids*, Wiley, 2010.
    Jaeckel, *By Implication*, Wilmott, 2006 (for structured products).
    Allen, Granzia & Otsuki, *Equity Structured Products*, RBC Capital, 2011.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Autocallable ----

@dataclass
class EquityAutocallableResult:
    """Equity autocallable pricing result."""
    price: float
    autocall_probability: float
    mean_autocall_time: float
    loss_probability: float
    notional: float


def equity_autocallable(
    spot: float,
    autocall_barrier: float,
    coupon_barrier: float,
    protection_barrier: float,
    coupon: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    T: float,
    observation_dates: list[float],
    notional: float = 1.0,
    has_memory: bool = True,
    n_paths: int = 10_000,
    seed: int | None = 42,
) -> EquityAutocallableResult:
    """Phoenix autocallable with memory coupon.

    At each observation t_i:
    - If S(t_i) ≥ autocall_barrier: pay notional + accumulated coupons, terminate.
    - Else if S(t_i) ≥ coupon_barrier: pay coupon (and release missed if memory).
    - Else: no payment (coupon missed; accumulates if memory).

    At expiry (if not autocalled):
    - If S_T ≥ protection_barrier: pay notional + any missed coupons.
    - Else: pay notional × S_T / spot (loss proportional to spot drop).

    Args:
        autocall_barrier: ≥ this level → early redemption.
        coupon_barrier: ≥ this level → coupon paid.
        protection_barrier: terminal soft protection level.
    """
    rng = np.random.default_rng(seed)
    obs_sorted = sorted(observation_dates)

    S = np.full(n_paths, spot)
    t_prev = 0.0
    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)
    missed_coupons = np.zeros(n_paths)
    autocall_time = np.full(n_paths, T)

    for t_obs in obs_sorted:
        dt = t_obs - t_prev
        drift = (rate - dividend_yield - 0.5 * vol**2) * dt
        diff = vol * math.sqrt(dt)

        dW = rng.standard_normal(n_paths)
        S = S * np.exp(drift + diff * dW)

        df_t = math.exp(-rate * t_obs)

        # Autocall trigger
        triggered = alive & (S >= autocall_barrier)
        # Coupon eligibility
        coupon_paid = alive & (S >= coupon_barrier) & ~triggered

        if has_memory:
            # On autocall: pay notional + missed + current
            autocall_payout = missed_coupons + coupon
            pv += np.where(triggered, (notional + autocall_payout) * df_t, 0.0)
            # On coupon-only: pay all accumulated + current
            coupon_payout_with_memory = missed_coupons + coupon
            pv += np.where(coupon_paid, coupon_payout_with_memory * df_t, 0.0)
            # Update missed coupons
            missed_coupons = np.where(coupon_paid, 0.0, missed_coupons)
            missed_coupons = np.where(alive & ~triggered & ~coupon_paid,
                                       missed_coupons + coupon,
                                       missed_coupons)
        else:
            pv += np.where(triggered, (notional + coupon) * df_t, 0.0)
            pv += np.where(coupon_paid, coupon * df_t, 0.0)

        autocall_time = np.where(triggered & alive, t_obs, autocall_time)
        alive &= ~triggered
        t_prev = t_obs

    # At T: for paths still alive
    df_T = math.exp(-rate * T)
    # Protected: S_T ≥ protection
    protected = alive & (S >= protection_barrier)
    loss = alive & ~protected

    # Protected paths: pay notional + any missed coupons (memory)
    if has_memory:
        pv += np.where(protected, (notional + missed_coupons) * df_T, 0.0)
    else:
        pv += np.where(protected, notional * df_T, 0.0)
    # Loss paths: proportional loss
    pv += np.where(loss, notional * (S / spot) * df_T, 0.0)

    price = float(pv.mean())
    ac_prob = float(1 - alive.mean())
    loss_prob = float(loss.mean())

    if ac_prob > 1e-6:
        ac_mask = (autocall_time < T)
        mean_ac = float(autocall_time[ac_mask].mean()) if ac_mask.sum() > 0 else T
    else:
        mean_ac = T

    return EquityAutocallableResult(price, ac_prob, mean_ac, loss_prob, notional)


# ---- Worst-of autocallable ----

@dataclass
class WorstOfAutocallResult:
    """Worst-of autocallable result."""
    price: float
    autocall_probability: float
    mean_autocall_time: float
    notional: float
    n_assets: int


def worst_of_autocallable(
    spots: list[float],
    autocall_barrier_pct: float,     # as fraction of initial spots
    coupon: float,
    rate: float,
    dividend_yields: list[float],
    vols: list[float],
    correlations: np.ndarray,
    T: float,
    observation_dates: list[float],
    notional: float = 1.0,
    has_memory: bool = True,
    n_paths: int = 10_000,
    seed: int | None = 42,
) -> WorstOfAutocallResult:
    """Autocallable on worst-of basket: autocall when min(S_i / S_i^0) ≥ barrier.

    Common retail structured product: pays higher coupons than single-name
    but autocall requires ALL assets to rise above barrier.

    Args:
        spots: initial asset spots.
        autocall_barrier_pct: e.g. 1.0 = barrier at initial spot (autocall if all ≥).
        coupon: per-period coupon.
        correlations: n×n correlation matrix.
    """
    n = len(spots)
    spots_arr = np.array(spots)
    vols_arr = np.array(vols)
    div_arr = np.array(dividend_yields)

    try:
        L = np.linalg.cholesky(correlations)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(correlations + 1e-8 * np.eye(n))

    rng = np.random.default_rng(seed)
    obs_sorted = sorted(observation_dates)

    S = np.tile(spots_arr, (n_paths, 1))
    t_prev = 0.0
    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)
    missed = np.zeros(n_paths)
    ac_time = np.full(n_paths, T)

    for t_obs in obs_sorted:
        dt = t_obs - t_prev
        drift = (rate - div_arr - 0.5 * vols_arr**2) * dt
        Z = rng.standard_normal((n_paths, n)) @ L.T
        S = S * np.exp(drift + vols_arr * math.sqrt(dt) * Z)

        df_t = math.exp(-rate * t_obs)

        # Worst-of ratio
        ratios = S / spots_arr
        worst = ratios.min(axis=1)

        triggered = alive & (worst >= autocall_barrier_pct)

        if has_memory:
            payout = notional + missed + coupon
            pv += np.where(triggered, payout * df_t, 0.0)
            missed = np.where(alive & ~triggered, missed + coupon, missed)
        else:
            pv += np.where(triggered, (notional + coupon) * df_t, 0.0)
            pv += np.where(alive & ~triggered, coupon * df_t, 0.0)

        ac_time = np.where(triggered & alive, t_obs, ac_time)
        alive &= ~triggered
        t_prev = t_obs

    # At T: simple return of notional × worst performance
    ratios = S / spots_arr
    worst = ratios.min(axis=1)
    df_T = math.exp(-rate * T)
    pv += np.where(alive, notional * np.minimum(worst, 1.0) * df_T, 0.0)

    price = float(pv.mean())
    ac_prob = float(1 - alive.mean())

    if ac_prob > 1e-6:
        mask = (ac_time < T)
        mean_ac = float(ac_time[mask].mean()) if mask.sum() > 0 else T
    else:
        mean_ac = T

    return WorstOfAutocallResult(price, ac_prob, mean_ac, notional, n)


# ---- Reverse convertible ----

@dataclass
class ReverseConvertibleResult:
    """Reverse convertible note result."""
    price: float
    bond_pv: float
    short_put_value: float
    effective_yield: float
    strike: float
    notional: float


def reverse_convertible(
    spot: float,
    strike: float,
    coupon_rate: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    T: float,
    notional: float = 1.0,
    n_coupons: int = 2,
) -> ReverseConvertibleResult:
    """Reverse convertible: bond with enhanced coupon + embedded short put.

    At maturity:
    - If S_T ≥ strike: pay notional.
    - If S_T < strike: pay notional × S_T / strike (converted into stock).

    Cash flows: coupons × notional × coupon_rate / n + terminal payoff.

    Structure = zero-coupon bond + coupons − ATM put / at-strike put.

    Args:
        strike: conversion strike.
        coupon_rate: annualised coupon rate (higher than risk-free).
        n_coupons: number of coupon payments over T.
    """
    from pricebook.black76 import black76_price, OptionType

    # Bond leg: notional at T + coupons
    bond_pv = 0.0
    for i in range(1, n_coupons + 1):
        t = i * T / n_coupons
        bond_pv += notional * (coupon_rate / n_coupons) * math.exp(-rate * t)
    bond_pv += notional * math.exp(-rate * T)

    # Short put: if S_T < K, we give stock worth S_T instead of notional
    # Put value (per unit of strike): notional / strike units of puts with strike K
    F = spot * math.exp((rate - dividend_yield) * T)
    df = math.exp(-rate * T)
    put_per_unit = black76_price(F, strike, vol, T, df, OptionType.PUT)
    short_put = (notional / strike) * put_per_unit

    # RC PV = bond PV − short put cost
    price = bond_pv - short_put

    # Effective yield (comparing to risk-free)
    if price > 0 and T > 0:
        eff_yield = (notional / price) ** (1 / T) - 1
    else:
        eff_yield = rate

    return ReverseConvertibleResult(
        price=float(price),
        bond_pv=float(bond_pv),
        short_put_value=float(short_put),
        effective_yield=float(eff_yield),
        strike=strike,
        notional=notional,
    )


# ---- Shark-fin ----

@dataclass
class SharkFinResult:
    """Shark-fin note result."""
    price: float
    knock_out_probability: float
    max_payout: float
    notional: float


def shark_fin_note(
    spot: float,
    strike: float,
    knock_out_barrier: float,
    rebate: float,
    participation: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    T: float,
    notional: float = 1.0,
    n_paths: int = 10_000,
    n_steps: int = 100,
    seed: int | None = 42,
) -> SharkFinResult:
    """Shark-fin: capped call with knock-out + rebate.

    Payoff at T (if not knocked out):
        notional × (1 + participation × max(S_T/K − 1, 0))
    capped at (notional × (1 + max_return))

    If knocked out at any time (S ≥ barrier):
        notional × (1 + rebate)

    Args:
        strike: participation strike.
        knock_out_barrier: upper barrier.
        rebate: return if knocked out (typically small).
        participation: participation rate (e.g. 1.0 = 100%).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (rate - dividend_yield - 0.5 * vol**2) * dt
    diff = vol * math.sqrt(dt)
    df = math.exp(-rate * T)

    S = np.full(n_paths, spot)
    alive = np.ones(n_paths, dtype=bool)

    for step in range(n_steps):
        dW = rng.standard_normal(n_paths)
        S = S * np.exp(drift + diff * dW)
        # Check knock-out
        alive &= (S < knock_out_barrier)

    # Payoff
    # For alive paths: participation × max(S_T/K - 1, 0)
    terminal_return = np.where(alive,
                                participation * np.maximum(S / strike - 1, 0),
                                rebate)

    payoff = notional * (1 + terminal_return)
    price = df * float(payoff.mean())
    ko_prob = float(1 - alive.mean())

    # Max payout = notional × (1 + participation × (barrier/strike - 1))
    max_payout = notional * (1 + participation * max(knock_out_barrier / strike - 1, 0))

    return SharkFinResult(price, ko_prob, max_payout, notional)


# ---- Airbag note ----

@dataclass
class AirbagResult:
    """Airbag note (capped-floored) result."""
    price: float
    notional: float
    cap: float
    floor: float
    upside: float


def airbag_note(
    spot: float,
    strike: float,
    cap: float,                # max return
    floor: float,              # min return (typically negative)
    rate: float,
    dividend_yield: float,
    vol: float,
    T: float,
    notional: float = 1.0,
) -> AirbagResult:
    """Airbag / capped-floored note.

    Payoff = notional × (1 + max(floor, min(cap, S_T/S_0 − 1)))

    Decomposition:
        = notional × (1 + floor) + notional × (call(floor_K) − call(cap_K)) / spot

    where floor_K = spot × (1 + floor), cap_K = spot × (1 + cap).
    """
    from pricebook.black76 import black76_price, OptionType

    F = spot * math.exp((rate - dividend_yield) * T)
    df = math.exp(-rate * T)

    floor_K = spot * (1 + floor)
    cap_K = spot * (1 + cap)

    # Long call at floor_K, short call at cap_K (bull call spread)
    c_floor = black76_price(F, floor_K, vol, T, df, OptionType.CALL)
    c_cap = black76_price(F, cap_K, vol, T, df, OptionType.CALL)
    spread = c_floor - c_cap

    # Participation × (spread) + guaranteed (1 + floor)
    base = notional * (1 + floor) * df
    upside = notional * spread / spot

    price = base + upside

    return AirbagResult(float(price), notional, cap, floor, float(upside))
