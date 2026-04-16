"""FX structured products: TARFs, autocallables, dual-currency, pivots.

Extends :mod:`pricebook.fx_exotic` with structured payoff bundles:

* :func:`fx_tarf_price` — Target Redemption Forward.
* :func:`fx_autocallable_price` — autocallable with memory coupon.
* :func:`fx_dual_currency_deposit` — DCD yield + conversion risk.
* :func:`fx_pivot_option` — digital range / pivot structure.

References:
    Wystup, *FX Options and Structured Products*, 2nd ed., Wiley, 2017, Ch. 5.
    Clark, *FX Option Pricing*, Wiley, 2011.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- TARF ----

@dataclass
class TARFResult:
    """Target Redemption Forward pricing result."""
    price: float
    expected_profit: float
    prob_target_hit: float
    mean_termination_time: float
    target: float
    strike: float


def fx_tarf_price(
    spot: float,
    strike: float,
    target: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    n_observations: int = 12,
    notional: float = 1.0,
    is_buyer_long_usd: bool = True,
    n_paths: int = 10_000,
    seed: int | None = 42,
) -> TARFResult:
    """FX TARF: accumulate profit until target hit, then terminate.

    Standard TARF structure:
    - At each observation, if S > K (for long USD buyer), accumulate
      profit (S - K) × notional.
    - Terminate when cumulative profit ≥ target.
    - Otherwise continue to expiry.

    The buyer pays to enter the TARF because the target caps their
    upside while they retain full downside risk.

    Args:
        spot: current FX spot (units of ccy2 per ccy1).
        strike: TARF strike.
        target: cumulative profit trigger.
        T: total maturity.
        n_observations: fixing dates.
        notional: per-period notional.
        is_buyer_long_usd: True if buyer profits when S > K.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_observations
    drift = (rate_dom - rate_for - 0.5 * vol**2) * dt
    diff = vol * math.sqrt(dt)

    S = np.full(n_paths, spot)
    cum_profit = np.zeros(n_paths)
    pv = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    termination_step = np.full(n_paths, n_observations)

    for step in range(n_observations):
        dW = rng.standard_normal(n_paths)
        S = S * np.exp(drift + diff * dW)

        t = (step + 1) * dt
        df_t = math.exp(-rate_dom * t)

        # Period P&L
        if is_buyer_long_usd:
            pnl = (S - strike) * notional
        else:
            pnl = (strike - S) * notional

        # Accumulate only where alive
        cum_profit_new = cum_profit + np.where(alive & (pnl > 0), pnl, 0.0)

        # Check if target would be breached
        hit_this_step = alive & (cum_profit_new >= target)

        # If hit, cap profit at target; pay target and terminate
        cap_profit = np.where(hit_this_step, target - cum_profit, pnl)
        cap_profit = np.where(alive, cap_profit, 0.0)

        # Add period P&L (profit or loss) to PV
        pv += cap_profit * df_t

        # Update cumulative profit (capped at target)
        cum_profit = np.minimum(cum_profit_new, target)

        # Terminate paths that hit the target
        termination_step = np.where(hit_this_step & alive, step + 1, termination_step)
        alive &= ~hit_this_step

    expected_profit = float(pv.mean())
    prob_hit = float(1 - alive.mean())
    mean_term = float(termination_step.mean()) * dt

    return TARFResult(expected_profit, expected_profit, prob_hit,
                      mean_term, target, strike)


# ---- Autocallable ----

@dataclass
class AutocallableResult:
    """FX autocallable result."""
    price: float
    autocall_probability: float
    mean_autocall_time: float
    coupon_rate: float


def fx_autocallable_price(
    spot: float,
    autocall_barrier: float,
    coupon: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    observation_dates: list[float],
    notional: float = 1.0,
    protection_barrier: float | None = None,
    has_memory: bool = True,
    n_paths: int = 10_000,
    seed: int | None = 42,
) -> AutocallableResult:
    """FX autocallable: autocalls at barrier, with optional memory coupon.

    Structure:
    - At each observation date, if S ≥ autocall_barrier: pay
      cumulative coupons + notional, terminate.
    - If never autocalled: pay notional at T, minus loss if S < protection_barrier.
    - Memory: missed coupons paid on autocall.

    Args:
        spot: current FX spot.
        autocall_barrier: level triggering early redemption.
        coupon: periodic coupon (paid if autocalled or at maturity).
        observation_dates: early redemption dates.
        protection_barrier: soft protection at expiry.
        has_memory: if True, unpaid coupons accumulate and pay on autocall.
    """
    rng = np.random.default_rng(seed)
    df_T = math.exp(-rate_dom * T)

    # Sort observations
    obs_dates = sorted(observation_dates)
    n_obs = len(obs_dates)

    # Simulate paths to each observation date
    S = np.full(n_paths, spot)
    t_prev = 0.0
    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)
    autocall_time = np.full(n_paths, T)
    missed_coupons = np.zeros(n_paths)

    for i, t_obs in enumerate(obs_dates):
        dt = t_obs - t_prev
        drift = (rate_dom - rate_for - 0.5 * vol**2) * dt
        diff = vol * math.sqrt(dt)

        dW = rng.standard_normal(n_paths)
        S = S * np.exp(drift + diff * dW)

        df_t = math.exp(-rate_dom * t_obs)

        # Check autocall trigger
        triggered = alive & (S >= autocall_barrier)

        # Coupon for this period
        period_coupon = np.where(alive, coupon, 0.0)

        if has_memory:
            # On trigger, pay all missed + current coupon
            payout_triggered = missed_coupons + period_coupon
            pv += np.where(triggered, (notional + payout_triggered) * df_t, 0.0)
            # Missed coupons accumulate for non-triggered alive paths
            missed_coupons = np.where(alive & ~triggered,
                                       missed_coupons + period_coupon,
                                       missed_coupons)
        else:
            # Always pay period coupon, regardless of trigger
            pv += np.where(alive, period_coupon * df_t, 0.0)
            # On trigger, also pay notional
            pv += np.where(triggered, notional * df_t, 0.0)

        autocall_time = np.where(triggered & alive, t_obs, autocall_time)
        alive &= ~triggered
        t_prev = t_obs

    # At maturity, for paths still alive: pay notional (with optional protection)
    if protection_barrier is None:
        final_payout = notional
        pv += np.where(alive, final_payout * df_T, 0.0)
    else:
        # Soft protection: if S_T < protection, lose proportionally
        terminal = np.where(S < protection_barrier, notional * S / spot, notional)
        pv += np.where(alive, terminal * df_T, 0.0)

    price = float(pv.mean())
    ac_prob = float(1 - alive.mean())
    if ac_prob > 1e-6:
        # Conditional mean autocall time
        ac_mask = (autocall_time < T)
        mean_ac = float(autocall_time[ac_mask].mean()) if ac_mask.sum() > 0 else T
    else:
        mean_ac = T

    return AutocallableResult(price, ac_prob, mean_ac, coupon / notional)


# ---- Dual Currency Deposit ----

@dataclass
class DCDResult:
    """Dual-Currency Deposit result."""
    enhanced_yield: float
    base_yield: float
    embedded_option_value: float
    notional: float
    strike: float


def fx_dual_currency_deposit(
    notional: float,
    spot: float,
    strike: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
) -> DCDResult:
    """Dual-Currency Deposit: higher yield with FX conversion risk.

    Client deposits notional in ccy1 for maturity T.
    At maturity:
    - If S_T ≥ K: return notional × (1 + r_enhanced) in ccy1.
    - If S_T < K: return notional × (1 + r_enhanced) / K in ccy2.

    Equivalent to a deposit + selling a put on ccy2.
    Higher yield compensates for the put premium.

    Args:
        notional: deposit amount in ccy1.
        strike: conversion strike.
    """
    from pricebook.black76 import black76_price, OptionType

    # Value of short put on FX
    F = spot * math.exp((rate_dom - rate_for) * T)
    df = math.exp(-rate_dom * T)
    put_value = black76_price(F, strike, vol, T, df, OptionType.PUT)

    # Base yield
    base_yield = rate_dom

    # Enhanced yield: compensate for put
    # put_value_per_unit = put_value / notional (if notional = 1)
    # Enhanced coupon ≈ base + put_value / (T × DF)
    if T > 0:
        enhancement = put_value / max(df, 1e-6) / T
    else:
        enhancement = 0.0

    enhanced = base_yield + enhancement

    return DCDResult(
        enhanced_yield=enhanced,
        base_yield=base_yield,
        embedded_option_value=put_value * notional,
        notional=notional,
        strike=strike,
    )


# ---- Pivot / range option ----

@dataclass
class PivotResult:
    """Pivot/digital range option result."""
    price: float
    prob_in_range: float
    range_low: float
    range_high: float


def fx_pivot_option(
    spot: float,
    range_low: float,
    range_high: float,
    rate_dom: float,
    rate_for: float,
    vol: float,
    T: float,
    payout: float = 1.0,
    is_european: bool = True,
) -> PivotResult:
    """Pivot / digital range option.

    European: pays `payout` if S_T ∈ [range_low, range_high] at expiry.
    Path-dependent: use no-touch on both barriers.

    Decomposition: Digital(range_low) - Digital(range_high).
    """
    from scipy.stats import norm

    F = spot * math.exp((rate_dom - rate_for) * T)
    df = math.exp(-rate_dom * T)

    if is_european:
        # European: P(range_low < S_T ≤ range_high)
        d_lo = (math.log(F / range_low) - 0.5 * vol**2 * T) / (vol * math.sqrt(T))
        d_hi = (math.log(F / range_high) - 0.5 * vol**2 * T) / (vol * math.sqrt(T))
        prob = norm.cdf(d_lo) - norm.cdf(d_hi)
    else:
        # Path-dependent via MC (approximated for simplicity)
        from pricebook.fx_exotic import fx_double_no_touch
        dnt = fx_double_no_touch(spot, range_low, range_high,
                                  rate_dom, rate_for, vol, T, payout)
        return PivotResult(dnt.price, dnt.price / max(df * payout, 1e-10),
                           range_low, range_high)

    price = df * payout * prob

    return PivotResult(float(price), float(prob), range_low, range_high)
