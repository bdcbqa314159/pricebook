"""FX correlation and basket options.

Extends FX with multi-asset features:

* :func:`triangular_correlation` — consistency of EURJPY vol from EURUSD × USDJPY.
* :func:`fx_basket_option` — min/max/average of multiple FX pairs.
* :func:`fx_worst_of` / :func:`fx_best_of` — extremum options.
* :func:`implied_correlation_quanto` — correlation from quanto premium.
* :func:`margrabe_fx_exchange` — two-asset exchange option closed form.

References:
    Wystup, *FX Options and Structured Products*, 2nd ed., Wiley, 2017, Ch. 6.
    Margrabe, *The Value of an Option to Exchange One Asset for Another*, JF, 1978.
    Johnson, *Options on the Maximum or the Minimum of Several Assets*, JFQA, 1987.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricebook.black76 import black76_price, OptionType


# ---- Triangular correlation ----

@dataclass
class TriangularResult:
    """Triangular FX correlation consistency check."""
    vol_implied_cross: float   # vol implied by σ₁, σ₂, ρ
    vol_market_cross: float | None
    correlation: float
    basis: float | None        # σ_implied - σ_market


def triangular_correlation(
    vol_pair1: float,
    vol_pair2: float,
    correlation: float,
    vol_cross_market: float | None = None,
) -> TriangularResult:
    """Triangular FX vol consistency.

    Given vol(EURUSD) = σ₁, vol(USDJPY) = σ₂, and correlation ρ:
        vol(EURJPY)² = σ₁² + σ₂² + 2 ρ σ₁ σ₂

    (The + sign arises because EURJPY = EURUSD × USDJPY; the USD term cancels
    when we consider log returns.)

    Triangular arbitrage checks:
    - implied cross vol vs market cross vol
    - basis = σ_implied − σ_market

    Args:
        vol_pair1, vol_pair2: vols of the two "leg" pairs.
        correlation: correlation between log returns of the legs.
        vol_cross_market: market vol of the cross pair (optional).
    """
    # For EURUSD × USDJPY → EURJPY:
    #   log(EURJPY) = log(EURUSD) + log(USDJPY)
    #   vol² = σ₁² + σ₂² + 2 ρ σ₁ σ₂
    var_implied = vol_pair1**2 + vol_pair2**2 + 2 * correlation * vol_pair1 * vol_pair2
    vol_implied = math.sqrt(max(var_implied, 0.0))

    basis = None
    if vol_cross_market is not None:
        basis = vol_implied - vol_cross_market

    return TriangularResult(vol_implied, vol_cross_market, correlation, basis)


def implied_correlation_from_triangular(
    vol_pair1: float,
    vol_pair2: float,
    vol_cross: float,
) -> float:
    """Invert triangular relation for correlation.

    ρ = (σ_cross² − σ₁² − σ₂²) / (2 σ₁ σ₂)
    """
    if vol_pair1 <= 0 or vol_pair2 <= 0:
        return 0.0
    rho = (vol_cross**2 - vol_pair1**2 - vol_pair2**2) / (2 * vol_pair1 * vol_pair2)
    return max(-1.0, min(1.0, rho))


# ---- FX Basket ----

@dataclass
class BasketResult:
    """FX basket option result."""
    price: float
    basket_type: str        # "average", "min", "max", "worst_of", "best_of"
    n_assets: int
    is_call: bool


def fx_basket_option(
    spots: list[float],
    weights: list[float],
    strike: float,
    rates_dom: float,
    rates_for: list[float],
    vols: list[float],
    correlations: np.ndarray,
    T: float,
    is_call: bool = True,
    basket_type: str = "average",
    n_paths: int = 20_000,
    seed: int | None = 42,
) -> BasketResult:
    """Multi-asset FX basket option.

    Basket payoff types:
    - "average": weighted average of spots
    - "min": minimum of spots
    - "max": maximum of spots

    Args:
        spots: initial FX spots.
        weights: basket weights (sum to 1 typically).
        strike: option strike.
        rates_dom: common domestic rate.
        rates_for: foreign rate per pair.
        vols: vols per pair.
        correlations: n×n correlation matrix.
        basket_type: aggregation function.
    """
    n = len(spots)
    spots_arr = np.array(spots)
    weights_arr = np.array(weights)
    vols_arr = np.array(vols)
    rates_for_arr = np.array(rates_for)

    # Cholesky decomposition for correlated normals
    try:
        L = np.linalg.cholesky(correlations)
    except np.linalg.LinAlgError:
        # Add small diagonal
        L = np.linalg.cholesky(correlations + 1e-8 * np.eye(n))

    rng = np.random.default_rng(seed)
    sqrt_T = math.sqrt(T)

    # Generate correlated normals
    Z = rng.standard_normal((n_paths, n))
    Z_corr = Z @ L.T

    # Simulate terminal spots (one-step GBM)
    drifts = (rates_dom - rates_for_arr - 0.5 * vols_arr**2) * T
    terminal = spots_arr * np.exp(drifts + vols_arr * sqrt_T * Z_corr)

    # Compute basket value per path
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

    df = math.exp(-rates_dom * T)
    price = df * float(payoff.mean())

    return BasketResult(price, basket_type, n, is_call)


def fx_worst_of(
    spots: list[float],
    strike: float,
    rates_dom: float,
    rates_for: list[float],
    vols: list[float],
    correlations: np.ndarray,
    T: float,
    is_call: bool = True,
    n_paths: int = 20_000,
    seed: int | None = 42,
) -> BasketResult:
    """Worst-of FX option: payoff on min(S₁, S₂, ...).

    Worst-of call = (min(S) - K)+ — cheapest of the options.
    Worst-of put = (K - min(S))+ — expensive (pays on worst outcome).
    """
    n = len(spots)
    weights = [1.0 / n] * n

    basket_type = "min"
    return fx_basket_option(spots, weights, strike, rates_dom, rates_for,
                             vols, correlations, T, is_call, basket_type,
                             n_paths, seed)


def fx_best_of(
    spots: list[float],
    strike: float,
    rates_dom: float,
    rates_for: list[float],
    vols: list[float],
    correlations: np.ndarray,
    T: float,
    is_call: bool = True,
    n_paths: int = 20_000,
    seed: int | None = 42,
) -> BasketResult:
    """Best-of FX option: payoff on max(S₁, S₂, ...)."""
    n = len(spots)
    weights = [1.0 / n] * n
    return fx_basket_option(spots, weights, strike, rates_dom, rates_for,
                             vols, correlations, T, is_call, "max",
                             n_paths, seed)


# ---- Margrabe (two-asset exchange) ----

@dataclass
class MargrabeResult:
    """Margrabe exchange option result."""
    price: float
    forward1: float
    forward2: float
    vol_combined: float


def margrabe_fx_exchange(
    spot1: float,
    spot2: float,
    rate_dom: float,
    rate_for1: float,
    rate_for2: float,
    vol1: float,
    vol2: float,
    correlation: float,
    T: float,
    quantity1: float = 1.0,
    quantity2: float = 1.0,
) -> MargrabeResult:
    """Margrabe formula for exchange option on two FX assets.

    Payoff: max(q₁ S₁(T) − q₂ S₂(T), 0) at expiry.

    No domestic rate enters the formula (cancels). Only the convenience yields
    (foreign rates) enter via the asset-specific drift.

    Price = q₁ S₁ e^{-rf₁ T} N(d₁) − q₂ S₂ e^{-rf₂ T} N(d₂)
    where
        σ² = σ₁² + σ₂² − 2 ρ σ₁ σ₂
        d₁ = [ln(q₁ S₁ / (q₂ S₂)) + (rf₂ - rf₁ + σ²/2) T] / (σ√T)
        d₂ = d₁ − σ√T
    """
    if T <= 0:
        payoff = max(quantity1 * spot1 - quantity2 * spot2, 0.0)
        return MargrabeResult(payoff, spot1, spot2, 0.0)

    sigma_sq = vol1**2 + vol2**2 - 2 * correlation * vol1 * vol2
    sigma = math.sqrt(max(sigma_sq, 1e-12))
    sigma_sqrt_T = sigma * math.sqrt(T)

    F1 = spot1 * math.exp((rate_dom - rate_for1) * T)
    F2 = spot2 * math.exp((rate_dom - rate_for2) * T)

    d1 = (math.log((quantity1 * spot1) / (quantity2 * spot2))
          + (rate_for2 - rate_for1 + 0.5 * sigma_sq) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    price = (quantity1 * spot1 * math.exp(-rate_for1 * T) * norm.cdf(d1)
             - quantity2 * spot2 * math.exp(-rate_for2 * T) * norm.cdf(d2))

    return MargrabeResult(float(max(price, 0.0)), F1, F2, sigma)


# ---- Implied correlation ----

@dataclass
class ImpliedCorrelationResult:
    """Implied correlation from market prices."""
    correlation: float
    target_price: float
    model_price: float
    method: str


def implied_correlation_quanto(
    spot: float,
    strike: float,
    quanto_strike_vol: float,
    underlying_vol: float,
    fx_vol: float,
    quanto_adjustment_observed: float,
    T: float,
) -> ImpliedCorrelationResult:
    """Invert quanto adjustment for implied correlation.

    Quanto drift adjustment: μ_adj = −ρ × σ_FX × σ_asset
    Observed adjustment ≈ μ_adj × T ≈ −ρ × σ_FX × σ_asset × T

    Solve for ρ:
        ρ = −observed / (σ_FX × σ_asset × T)
    """
    if fx_vol <= 0 or underlying_vol <= 0 or T <= 0:
        rho = 0.0
    else:
        rho = -quanto_adjustment_observed / (fx_vol * underlying_vol * T)
        rho = max(-1.0, min(1.0, rho))

    return ImpliedCorrelationResult(
        correlation=rho,
        target_price=quanto_adjustment_observed,
        model_price=-rho * fx_vol * underlying_vol * T,
        method="quanto_inversion",
    )
