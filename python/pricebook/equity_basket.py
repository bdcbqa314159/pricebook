"""Equity basket options and correlation trading.

* :func:`margrabe_equity` — two-asset exchange option (Margrabe 1978).
* :func:`johnson_max_call` / :func:`johnson_min_call` — Johnson max/min on 2 assets.
* :func:`equity_basket_mc` — N-asset basket MC pricing.
* :func:`correlation_swap_price` — realised vs implied correlation swap.
* :func:`dispersion_trade_value` — index vs single-name dispersion.

References:
    Margrabe, *The Value of an Option to Exchange One Asset for Another*, JF, 1978.
    Johnson, *Options on the Maximum or the Minimum of Several Assets*, JFQA, 1987.
    Stulz, *Options on the Minimum or Maximum of Two Risky Assets*, JFE, 1982.
    Bossu, *Advanced Equity Derivatives: Volatility and Correlation*, Wiley, 2014.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


# ---- Margrabe ----

@dataclass
class MargrabeEquityResult:
    """Margrabe exchange option result."""
    price: float
    forward1: float
    forward2: float
    vol_combined: float


def margrabe_equity(
    spot1: float,
    spot2: float,
    rate: float,
    dividend_yield1: float,
    dividend_yield2: float,
    vol1: float,
    vol2: float,
    correlation: float,
    T: float,
    quantity1: float = 1.0,
    quantity2: float = 1.0,
) -> MargrabeEquityResult:
    """Margrabe exchange: max(q₁ S₁ − q₂ S₂, 0) at T.

    σ² = σ₁² + σ₂² − 2 ρ σ₁ σ₂
    Price = q₁ S₁ e^{-q₁T} N(d₁) − q₂ S₂ e^{-q₂T} N(d₂)
    """
    if T <= 0:
        return MargrabeEquityResult(max(quantity1 * spot1 - quantity2 * spot2, 0.0),
                                     spot1, spot2, 0.0)

    sigma_sq = vol1**2 + vol2**2 - 2 * correlation * vol1 * vol2
    sigma = math.sqrt(max(sigma_sq, 1e-12))
    sigma_sqrt_T = sigma * math.sqrt(T)

    F1 = spot1 * math.exp((rate - dividend_yield1) * T)
    F2 = spot2 * math.exp((rate - dividend_yield2) * T)

    d1 = (math.log((quantity1 * spot1) / (quantity2 * spot2))
          + (dividend_yield2 - dividend_yield1 + 0.5 * sigma_sq) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    price = (quantity1 * spot1 * math.exp(-dividend_yield1 * T) * norm.cdf(d1)
             - quantity2 * spot2 * math.exp(-dividend_yield2 * T) * norm.cdf(d2))

    return MargrabeEquityResult(float(max(price, 0.0)), F1, F2, sigma)


# ---- Johnson max/min on 2 assets ----

@dataclass
class MaxMinResult:
    """Max/min of 2 assets option result."""
    price: float
    option_style: str       # "max_call", "min_call", "max_put", "min_put"


def johnson_max_call(
    spot1: float,
    spot2: float,
    strike: float,
    rate: float,
    dividend_yield1: float,
    dividend_yield2: float,
    vol1: float,
    vol2: float,
    correlation: float,
    T: float,
) -> MaxMinResult:
    """Call on max(S₁, S₂) via Johnson (1987) / Stulz (1982).

    Payoff = max(max(S₁, S₂) − K, 0).

    Uses identity:
        max(S₁, S₂) = S₁ + max(S₂ − S₁, 0)
                    = S₂ + max(S₁ − S₂, 0)
    Then decompose:
        (max(S₁,S₂) − K)+ = (S₁ − K)+ + (S₂ − S₁)+ − (S₂ − max(S₁,K))+

    Simplified via MC to avoid bivariate-normal complexity.
    """
    rng = np.random.default_rng(42)
    n_paths = 50_000
    sqrt_T = math.sqrt(T)

    z1 = rng.standard_normal(n_paths)
    z2 = correlation * z1 + math.sqrt(1 - correlation**2) * rng.standard_normal(n_paths)

    S1_T = spot1 * np.exp((rate - dividend_yield1 - 0.5 * vol1**2) * T + vol1 * sqrt_T * z1)
    S2_T = spot2 * np.exp((rate - dividend_yield2 - 0.5 * vol2**2) * T + vol2 * sqrt_T * z2)

    max_S = np.maximum(S1_T, S2_T)
    payoff = np.maximum(max_S - strike, 0.0)

    df = math.exp(-rate * T)
    price = df * float(payoff.mean())

    return MaxMinResult(float(price), "max_call")


def johnson_min_call(
    spot1: float,
    spot2: float,
    strike: float,
    rate: float,
    dividend_yield1: float,
    dividend_yield2: float,
    vol1: float,
    vol2: float,
    correlation: float,
    T: float,
) -> MaxMinResult:
    """Call on min(S₁, S₂)."""
    rng = np.random.default_rng(42)
    n_paths = 50_000
    sqrt_T = math.sqrt(T)

    z1 = rng.standard_normal(n_paths)
    z2 = correlation * z1 + math.sqrt(1 - correlation**2) * rng.standard_normal(n_paths)

    S1_T = spot1 * np.exp((rate - dividend_yield1 - 0.5 * vol1**2) * T + vol1 * sqrt_T * z1)
    S2_T = spot2 * np.exp((rate - dividend_yield2 - 0.5 * vol2**2) * T + vol2 * sqrt_T * z2)

    min_S = np.minimum(S1_T, S2_T)
    payoff = np.maximum(min_S - strike, 0.0)

    df = math.exp(-rate * T)
    price = df * float(payoff.mean())

    return MaxMinResult(float(price), "min_call")


# ---- N-asset basket ----

@dataclass
class EquityBasketResult:
    """Equity basket option result."""
    price: float
    basket_type: str
    n_assets: int


def equity_basket_mc(
    spots: list[float],
    weights: list[float],
    strike: float,
    rate: float,
    dividend_yields: list[float],
    vols: list[float],
    correlations: np.ndarray,
    T: float,
    is_call: bool = True,
    basket_type: str = "average",    # "average", "max", "min"
    n_paths: int = 20_000,
    seed: int | None = 42,
) -> EquityBasketResult:
    """Multi-asset equity basket option.

    Args:
        basket_type: how to aggregate terminal values.
    """
    n = len(spots)
    spots_arr = np.array(spots)
    weights_arr = np.array(weights)
    vols_arr = np.array(vols)
    div_arr = np.array(dividend_yields)

    try:
        L = np.linalg.cholesky(correlations)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(correlations + 1e-8 * np.eye(n))

    rng = np.random.default_rng(seed)
    sqrt_T = math.sqrt(T)

    Z = rng.standard_normal((n_paths, n)) @ L.T
    drifts = (rate - div_arr - 0.5 * vols_arr**2) * T
    terminal = spots_arr * np.exp(drifts + vols_arr * sqrt_T * Z)

    if basket_type == "average":
        basket = terminal @ weights_arr
    elif basket_type == "min":
        basket = terminal.min(axis=1)
    elif basket_type == "max":
        basket = terminal.max(axis=1)
    else:
        raise ValueError(f"Unknown basket_type: {basket_type}")

    if is_call:
        payoff = np.maximum(basket - strike, 0.0)
    else:
        payoff = np.maximum(strike - basket, 0.0)

    df = math.exp(-rate * T)
    price = df * float(payoff.mean())

    return EquityBasketResult(float(price), basket_type, n)


# ---- Correlation swap ----

@dataclass
class CorrelationSwapResult:
    """Correlation swap result."""
    price: float
    implied_correlation: float
    fair_correlation_strike: float
    n_assets: int


def correlation_swap_price(
    realised_correlation: float,
    implied_correlation: float,
    notional: float = 1.0,
) -> CorrelationSwapResult:
    """Correlation swap: pays (realised − strike) × notional.

    Fair strike = implied correlation at inception.
    PV = notional × (implied − strike) for receiver of implied.

    Args:
        realised_correlation: actual realised correlation over the period.
        implied_correlation: strike correlation (implied from index/singles).
        notional: swap notional (per 1% correlation typically).
    """
    fair_strike = implied_correlation
    # For the receiver of realised vs strike: PV at inception = 0
    # At maturity: payoff = (realised − strike) × notional
    price = notional * (realised_correlation - implied_correlation)

    return CorrelationSwapResult(
        price=float(price),
        implied_correlation=implied_correlation,
        fair_correlation_strike=fair_strike,
        n_assets=2,
    )


def implied_correlation_from_dispersion(
    index_variance: float,
    component_variances: list[float],
    weights: list[float],
) -> float:
    """Implied correlation from index variance and constituent variances.

    σ²_index = Σᵢ wᵢ² σᵢ² + ρ × Σ_{i≠j} wᵢ wⱼ σᵢ σⱼ

    Solve:
        ρ_implied = (σ²_index − Σ wᵢ² σᵢ²) / (Σ_{i≠j} wᵢ wⱼ σᵢ σⱼ)
    """
    w = np.array(weights)
    sig = np.sqrt(np.maximum(component_variances, 0.0))

    # Diagonal term: Σ w_i² σ_i²
    diag = np.sum(w**2 * np.array(component_variances))
    # Off-diag: Σ_{i≠j} w_i w_j σ_i σ_j = (Σ w σ)² − Σ w² σ²
    ws_sum_sq = (np.sum(w * sig)) ** 2
    off_diag = ws_sum_sq - np.sum(w**2 * sig**2)

    if abs(off_diag) < 1e-10:
        return 0.0

    rho = (index_variance - diag) / off_diag
    return max(-1.0, min(1.0, float(rho)))


# ---- Dispersion trade ----

@dataclass
class DispersionTradeResult:
    """Dispersion trade value."""
    index_variance: float
    weighted_component_variance: float
    dispersion_gap: float          # weighted - index (positive → pays dispersion)
    implied_correlation: float
    trade_pnl: float


def dispersion_trade_value(
    index_variance: float,
    component_variances: list[float],
    weights: list[float],
    notional: float = 1.0,
) -> DispersionTradeResult:
    """Classic dispersion trade: long single-name variance, short index variance.

    P&L at maturity ≈ (Σ wᵢ² × realised_single_var_i) − realised_index_variance
    Positive when correlation realised < correlation implied.

    This function takes variances (implied or realised) and computes the gap.
    """
    w = np.array(weights)
    comp_var = np.array(component_variances)

    weighted_var = float(np.sum(w**2 * comp_var))
    gap = weighted_var - index_variance

    rho = implied_correlation_from_dispersion(index_variance,
                                                list(comp_var), list(w))

    pnl = notional * gap

    return DispersionTradeResult(
        index_variance=float(index_variance),
        weighted_component_variance=weighted_var,
        dispersion_gap=float(gap),
        implied_correlation=rho,
        trade_pnl=float(pnl),
    )
