"""Variance and volatility derivatives for equity.

* :func:`variance_swap_replication` — Demeterfi et al. static replication.
* :func:`volatility_swap_heston` — vol swap pricing under Heston.
* :func:`brockhaus_long_approx` — Brockhaus-Long vol swap approximation.
* :func:`variance_future_price` — variance futures / VIX-like index.
* :func:`variance_risk_premium` — implied − realised variance decomposition.

References:
    Demeterfi, Derman, Kamal & Zou, *More Than You Ever Wanted to Know About
    Variance Swaps*, Goldman Sachs Quantitative Strategies Research Notes, 1999.
    Brockhaus & Long, *Volatility Swaps Made Simple*, Risk, 2000.
    Carr & Madan, *Towards a Theory of Volatility Trading*, 1998.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricebook.black76 import black76_price, OptionType


# ---- Variance swap replication ----

@dataclass
class VarianceSwapResult:
    """Variance swap pricing result."""
    fair_variance: float        # fair strike for variance swap (annualised variance)
    fair_vol_strike: float      # sqrt(fair_variance)
    replication_pv: float        # static replication value
    n_strikes_used: int
    method: str


def variance_swap_replication(
    spot: float,
    forward: float,
    rate: float,
    dividend_yield: float,
    T: float,
    strikes: list[float],
    call_vols: list[float],
    put_vols: list[float] | None = None,
) -> VarianceSwapResult:
    """Fair variance via Demeterfi-Derman-Kamal-Zou static replication.

    The fair variance K_var² satisfies:
        K_var² T = 2 × [∫₀^F P(K)/K² dK + ∫_F^∞ C(K)/K² dK] / DF
                   + 2(r-q)T + 2 log(S0/F)

    Simplified discrete form:
        K_var² T ≈ 2 Σ_i w_i × price_i / DF

    where w_i = ΔK_i / K_i², price_i = put for K < F else call.

    Args:
        strikes: sorted strike grid covering the range [0, ∞).
        call_vols: implied vols for calls at each strike.
        put_vols: implied vols for puts; if None, uses call_vols (flat smile).
    """
    K = np.array(strikes)
    df = math.exp(-rate * T)
    F = forward

    if put_vols is None:
        put_vols = call_vols

    # Select OTM options: puts for K < F, calls for K > F
    total = 0.0
    n_used = 0

    for i, strike in enumerate(K):
        # ΔK: average of adjacent gaps
        if i == 0:
            dK = K[1] - K[0]
        elif i == len(K) - 1:
            dK = K[-1] - K[-2]
        else:
            dK = 0.5 * (K[i + 1] - K[i - 1])

        w = dK / (strike ** 2)

        if strike <= F:
            vol = put_vols[i]
            price = black76_price(F, strike, vol, T, df, OptionType.PUT)
        else:
            vol = call_vols[i]
            price = black76_price(F, strike, vol, T, df, OptionType.CALL)

        total += w * price
        n_used += 1

    # Variance = 2 × (replication / DF) / T  (for the integral part)
    # Plus correction: 2(r-q) - 2 log(F/S) / T  — but since we replicate from F,
    # we skip this term (simplification when S = F × e^{-(r-q)T}).
    fair_var = 2 * total / (df * T)

    return VarianceSwapResult(
        fair_variance=float(max(fair_var, 0.0)),
        fair_vol_strike=math.sqrt(max(fair_var, 0.0)),
        replication_pv=float(total),
        n_strikes_used=n_used,
        method="demeterfi_replication",
    )


# ---- Volatility swap under Heston ----

@dataclass
class VolatilitySwapResult:
    """Volatility swap result."""
    fair_vol: float             # fair strike for vol swap (√var - convexity adj)
    variance_strike: float      # for comparison
    convexity_adjustment: float
    method: str


def volatility_swap_heston(
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    T: float,
) -> VolatilitySwapResult:
    """Fair vol swap strike under Heston via expected realised variance.

    Expected variance under Heston:
        E[∫₀^T v_s ds] = θT + (v0 − θ)(1 − e^{-κT}) / κ

    Fair vol strike ≈ √(E[var]) − convexity correction.
    Jensen: E[√var] < √(E[var]), so vol strike < √(var strike).

    Uses approximation: E[√var] ≈ √E[var] × (1 − Var[var] / (8 × E[var]²))

    Args:
        v0: initial variance.
        kappa, theta, xi: Heston parameters.
        T: maturity.
    """
    # Expected realised variance
    if kappa < 1e-10:
        E_var = v0 * T
    else:
        E_var = theta * T + (v0 - theta) * (1 - math.exp(-kappa * T)) / kappa

    # Variance of realised variance (Heston closed form)
    # Var[∫v ds] ≈ (ξ²/κ²) × [θT + 2(v0-θ)(1-e^{-κT})/κ + ... ]
    # Simplified approximation:
    if kappa > 1e-10:
        var_var = (xi**2 / kappa**2) * (theta * T)
    else:
        var_var = (xi**2 / 3) * T**3

    var_var = max(var_var, 0.0)
    E_var = max(E_var, 1e-10)

    # Jensen convexity adjustment
    fair_vol_naive = math.sqrt(E_var / T)
    conv_adj = var_var / (8 * E_var**2) * fair_vol_naive

    fair_vol = fair_vol_naive - conv_adj

    return VolatilitySwapResult(
        fair_vol=float(max(fair_vol, 0.0)),
        variance_strike=float(fair_vol_naive),
        convexity_adjustment=float(conv_adj),
        method="heston_jensen",
    )


def brockhaus_long_approx(
    atm_vol: float,
    skew: float,
    T: float,
) -> VolatilitySwapResult:
    """Brockhaus-Long approximation for vol swap.

    Fair vol ≈ ATM vol − convexity adjustment from skew
            ≈ σ_ATM − (skew² × T) / (8 × σ_ATM)

    Works well when skew is small. Skew here is dσ/dlog K.
    """
    if atm_vol <= 0:
        return VolatilitySwapResult(0.0, 0.0, 0.0, "brockhaus_long")

    conv_adj = (skew**2 * T) / (8 * atm_vol)
    fair_vol = atm_vol - conv_adj

    return VolatilitySwapResult(
        fair_vol=float(max(fair_vol, 0.0)),
        variance_strike=float(atm_vol),
        convexity_adjustment=float(conv_adj),
        method="brockhaus_long",
    )


# ---- Variance futures ----

@dataclass
class VarianceFuturesResult:
    """Variance futures / VIX-like index result."""
    variance_index: float       # annualised variance
    volatility_index: float     # sqrt of above (like VIX)
    n_strikes: int
    underlying_forward: float


def variance_future_price(
    forward: float,
    rate: float,
    T: float,
    strikes: list[float],
    call_vols: list[float],
    put_vols: list[float] | None = None,
) -> VarianceFuturesResult:
    """VIX-like variance index from option strip (CBOE methodology).

    VIX² = (2/T) × Σ ΔK_i / K_i² × e^{rT} × Q(K_i) − (F/K₀ − 1)² / T

    where Q(K) is the OTM option price and K₀ is the strike just below F.

    Simplified here: use the same replication as variance swap.

    Args:
        forward: forward price.
        rate: risk-free rate.
        T: time to expiry.
        strikes: strike grid.
        call_vols/put_vols: vols.
    """
    vs = variance_swap_replication(
        spot=forward, forward=forward, rate=rate, dividend_yield=0.0,
        T=T, strikes=strikes, call_vols=call_vols, put_vols=put_vols,
    )

    return VarianceFuturesResult(
        variance_index=vs.fair_variance,
        volatility_index=vs.fair_vol_strike,
        n_strikes=vs.n_strikes_used,
        underlying_forward=forward,
    )


# ---- Variance risk premium ----

@dataclass
class VRPResult:
    """Variance risk premium decomposition."""
    implied_variance: float
    realised_variance: float
    vrp: float                  # implied - realised
    vrp_as_ratio: float         # (implied - realised) / realised
    vrp_in_vol_terms: float     # sqrt(implied) - sqrt(realised)
    period: float


def variance_risk_premium(
    implied_variance: float,
    realised_variance: float,
    period: float = 1.0,
) -> VRPResult:
    """Variance risk premium = implied − realised variance.

    Positive VRP is the norm: sellers of variance (calls+puts) earn
    premium over realised delivery — the insurance rate.

    Args:
        implied_variance: annualised implied variance (e.g. from var swap strike).
        realised_variance: annualised realised variance.
        period: horizon in years (for context).
    """
    vrp = implied_variance - realised_variance
    vrp_ratio = vrp / realised_variance if realised_variance > 1e-10 else 0.0
    vrp_vol = math.sqrt(max(implied_variance, 0.0)) - math.sqrt(max(realised_variance, 0.0))

    return VRPResult(
        implied_variance=float(implied_variance),
        realised_variance=float(realised_variance),
        vrp=float(vrp),
        vrp_as_ratio=float(vrp_ratio),
        vrp_in_vol_terms=float(vrp_vol),
        period=period,
    )
