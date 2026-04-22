"""Variance swap pricing via Breeden-Litzenberger replication.

Fair variance strike from the strip of OTM options:
    K_var = (2/T) × ∫ [OTM_option(K) / K²] dK

This is model-free: it depends only on observable option prices,
not on any specific vol model.

    from pricebook.variance_swap import fair_variance, variance_swap_pv

References:
    Demeterfi, Derman, Kamal & Zou, *More Than You Ever Wanted to Know
    About Volatility Swaps*, GS QS, 1999.
    Carr & Madan, *Towards a Theory of Volatility Trading*, 1998.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.black76 import black76_price, OptionType


@dataclass
class VarianceSwapResult:
    """Variance swap pricing result."""
    fair_variance: float          # annualised fair variance strike (σ²)
    fair_vol: float               # √(fair_variance)
    replication_integral: float   # raw integral value
    n_strikes: int


def fair_variance(
    forward: float,
    df: float,
    T: float,
    strikes: list[float] | np.ndarray,
    option_prices: list[float] | np.ndarray,
    is_call: list[bool] | np.ndarray | None = None,
) -> VarianceSwapResult:
    """Fair variance strike from a strip of option prices.

    K_var = (2/T) × Σ [ΔK_i × price_i / K_i²]

    Uses OTM options: calls for K > F, puts for K < F.

    Args:
        forward: forward price.
        df: discount factor to expiry.
        T: time to expiry (years).
        strikes: sorted strike prices.
        option_prices: market prices of OTM options at each strike.
        is_call: whether each option is a call. If None, auto-selects
                 based on K vs F (calls for K > F, puts for K ≤ F).
    """
    K = np.asarray(strikes, dtype=float)
    P = np.asarray(option_prices, dtype=float)
    n = len(K)

    if is_call is None:
        is_call = K > forward

    integral = 0.0
    for i in range(n):
        # Width of strike interval
        if i == 0:
            dk = K[1] - K[0] if n > 1 else 1.0
        elif i == n - 1:
            dk = K[-1] - K[-2] if n > 1 else 1.0
        else:
            dk = 0.5 * (K[i + 1] - K[i - 1])

        integral += dk * P[i] / (K[i] ** 2)

    fair_var = (2.0 / (T * df)) * integral if T > 0 and df > 0 else 0.0

    return VarianceSwapResult(
        fair_variance=fair_var,
        fair_vol=math.sqrt(max(fair_var, 0.0)),
        replication_integral=integral,
        n_strikes=n,
    )


def fair_variance_from_vols(
    forward: float,
    df: float,
    T: float,
    strikes: list[float] | np.ndarray,
    vols: list[float] | np.ndarray,
) -> VarianceSwapResult:
    """Fair variance strike from implied vols (converts to prices first)."""
    K = np.asarray(strikes, dtype=float)
    V = np.asarray(vols, dtype=float)

    prices = []
    is_call = []
    for k, v in zip(K, V):
        if k > forward:
            p = black76_price(forward, k, v, T, df, OptionType.CALL)
            is_call.append(True)
        else:
            p = black76_price(forward, k, v, T, df, OptionType.PUT)
            is_call.append(False)
        prices.append(p)

    return fair_variance(forward, df, T, K, prices, is_call)


@dataclass
class VarianceSwapPV:
    """Variance swap mark-to-market."""
    pv: float
    fair_variance: float
    strike_variance: float
    notional_variance: float


def variance_swap_pv(
    fair_var: float,
    strike_var: float,
    notional_vega: float,
    T_remaining: float,
    df: float,
) -> VarianceSwapPV:
    """PV of a variance swap.

    PV = notional_vega × (fair_variance - strike_variance) × df

    Args:
        fair_var: current fair variance (from replication).
        strike_var: agreed variance strike at inception.
        notional_vega: vega notional ($ per variance point).
        T_remaining: remaining time to expiry.
        df: discount factor to expiry.
    """
    pv = notional_vega * (fair_var - strike_var) * df
    return VarianceSwapPV(
        pv=pv,
        fair_variance=fair_var,
        strike_variance=strike_var,
        notional_variance=notional_vega,
    )
