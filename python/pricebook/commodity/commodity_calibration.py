"""Commodity model calibration to futures strip.

Calibrate Schwartz one-factor, Gibson-Schwartz two-factor, and
Schwartz-Smith models to observed futures prices. Seasonal
decomposition of the forward curve.

* :func:`calibrate_schwartz` — Schwartz 1F to futures strip.
* :func:`calibrate_gibson_schwartz` — Gibson-Schwartz 2F.
* :func:`seasonal_decomposition` — extract seasonal + trend.
* :func:`implied_convenience_yield_term` — convenience yield curve.

References:
    Schwartz, *The Stochastic Behavior of Commodity Prices*, JF, 1997.
    Gibson & Schwartz, *Stochastic Convenience Yield and the Pricing
    of Oil Contingent Claims*, JF, 1990.
    Schwartz & Smith, *Short-Term Variations and Long-Term Dynamics
    in Commodity Prices*, MS, 2000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SchwartzCalibResult:
    """Schwartz 1F calibration result."""
    kappa: float            # mean reversion speed
    mu: float               # long-run mean (log price)
    sigma: float            # volatility
    rmse: float             # root mean square error
    n_contracts: int

    def to_dict(self) -> dict:
        return vars(self)


def calibrate_schwartz(
    spot: float,
    futures_prices: list[float],
    maturities: list[float],
    rate: float = 0.04,
) -> SchwartzCalibResult:
    """Calibrate Schwartz one-factor to futures strip.

    Model: dS = κ(μ − ln S)S dt + σS dW

    Futures price: F(T) = exp(e^{−κT} ln S + (1−e^{−κT})μ
                          + σ²(1−e^{−2κT})/(4κ) − (r−...)T)

    Simplified: fit κ, μ, σ to minimise Σ(F_model − F_market)².

    Args:
        spot: current spot price.
        futures_prices: observed futures prices.
        maturities: time to delivery (years) per contract.
        rate: risk-free rate.
    """
    from scipy.optimize import minimize

    log_spot = math.log(spot)

    def objective(params):
        kappa, mu, sigma = params
        kappa = max(kappa, 0.01)
        sigma = max(sigma, 0.001)
        sse = 0.0
        for F_mkt, T in zip(futures_prices, maturities):
            # Schwartz 1F futures formula
            e_kt = math.exp(-kappa * T)
            log_F = e_kt * log_spot + (1 - e_kt) * mu
            log_F += sigma**2 / (4 * kappa) * (1 - math.exp(-2 * kappa * T))
            F_model = math.exp(log_F)
            sse += (F_model - F_mkt) ** 2
        return sse

    x0 = [1.0, math.log(spot), 0.30]
    bounds = [(0.01, 20), (math.log(spot) - 2, math.log(spot) + 2), (0.01, 2.0)]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    kappa, mu, sigma = result.x

    rmse = math.sqrt(result.fun / len(futures_prices)) if futures_prices else 0

    return SchwartzCalibResult(
        kappa=kappa, mu=mu, sigma=sigma,
        rmse=rmse, n_contracts=len(futures_prices),
    )


@dataclass
class GibsonSchwartzCalibResult:
    """Gibson-Schwartz 2F calibration result."""
    kappa: float            # convenience yield mean reversion
    alpha: float            # long-run convenience yield
    sigma_s: float          # spot vol
    sigma_c: float          # convenience yield vol
    rho: float              # spot-convenience yield correlation
    rmse: float

    def to_dict(self) -> dict:
        return vars(self)


def calibrate_gibson_schwartz(
    spot: float,
    futures_prices: list[float],
    maturities: list[float],
    rate: float = 0.04,
) -> GibsonSchwartzCalibResult:
    """Calibrate Gibson-Schwartz two-factor model.

    Spot: dS/S = (r − δ) dt + σ_s dW₁
    Convenience yield: dδ = κ(α − δ) dt + σ_c dW₂

    Futures: F(T) = S × exp(A(T) − B(T)δ)
    where B(T) = (1 − e^{−κT})/κ
    """
    from scipy.optimize import minimize

    def objective(params):
        kappa, alpha, sigma_s, sigma_c, rho = params
        kappa = max(kappa, 0.01)
        sigma_s = max(sigma_s, 0.001)
        sigma_c = max(sigma_c, 0.001)
        rho = max(-0.99, min(rho, 0.99))

        sse = 0.0
        for F_mkt, T in zip(futures_prices, maturities):
            B = (1 - math.exp(-kappa * T)) / kappa
            A = (rate - alpha) * T + alpha * B
            A += sigma_c**2 / (2 * kappa**2) * (T - 2*B + (1 - math.exp(-2*kappa*T))/(2*kappa))
            A -= rho * sigma_s * sigma_c / kappa * (T - B)

            # Initial convenience yield: implied from first futures
            delta_0 = alpha  # assume at long-run level

            F_model = spot * math.exp(A - B * delta_0)
            sse += (F_model - F_mkt) ** 2
        return sse

    x0 = [1.0, 0.03, 0.30, 0.20, -0.3]
    bounds = [(0.01, 20), (-0.10, 0.30), (0.01, 2.0), (0.01, 1.0), (-0.99, 0.99)]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    kappa, alpha, sigma_s, sigma_c, rho = result.x

    rmse = math.sqrt(result.fun / len(futures_prices)) if futures_prices else 0

    return GibsonSchwartzCalibResult(
        kappa=kappa, alpha=alpha, sigma_s=sigma_s,
        sigma_c=sigma_c, rho=rho, rmse=rmse,
    )


@dataclass
class SeasonalDecompResult:
    """Seasonal decomposition of forward curve."""
    trend: np.ndarray           # de-seasonalised level
    seasonal_factors: np.ndarray  # multiplicative seasonal factors
    residual: np.ndarray
    period: int                 # seasonal period (months)

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "n_points": len(self.trend),
            "mean_seasonal_range": float(np.max(self.seasonal_factors) - np.min(self.seasonal_factors)),
        }


def seasonal_decomposition(
    prices: list[float],
    period: int = 12,
) -> SeasonalDecompResult:
    """Extract seasonal + trend from commodity forward curve.

    Uses multiplicative decomposition:
    price = trend × seasonal × residual.

    Args:
        prices: forward prices (ordered by delivery month).
        period: seasonal period (12 for monthly).
    """
    arr = np.array(prices)
    n = len(arr)

    if n < period:
        return SeasonalDecompResult(
            trend=arr, seasonal_factors=np.ones(n),
            residual=np.ones(n), period=period,
        )

    # Moving average for trend
    kernel = np.ones(period) / period
    # Pad for valid convolution
    padded = np.pad(arr, (period // 2, period // 2), mode='edge')
    trend = np.convolve(padded, kernel, mode='valid')[:n]

    # Seasonal: average ratio at each position in the cycle
    ratios = arr / np.maximum(trend, 1e-10)
    seasonal = np.ones(period)
    for p in range(period):
        indices = list(range(p, n, period))
        if indices:
            seasonal[p] = float(np.mean(ratios[indices]))

    # Normalise seasonal factors
    seasonal /= np.mean(seasonal)

    # Tile seasonal to match data length
    full_seasonal = np.tile(seasonal, n // period + 1)[:n]

    # Residual
    residual = arr / (trend * full_seasonal + 1e-15)

    return SeasonalDecompResult(
        trend=trend,
        seasonal_factors=seasonal,
        residual=residual,
        period=period,
    )


def implied_convenience_yield_term(
    spot: float,
    futures_prices: list[float],
    maturities: list[float],
    rate: float = 0.04,
    storage_cost: float = 0.0,
) -> list[tuple[float, float]]:
    """Implied convenience yield term structure.

    y(T) = r + c − (1/T) × ln(F/S)

    where c = storage cost.

    Args:
        spot: current spot.
        futures_prices: forward prices.
        maturities: time to delivery (years).
        rate: risk-free rate.
        storage_cost: annual storage cost as fraction of spot.

    Returns:
        List of (maturity, convenience_yield) pairs.
    """
    result = []
    for F, T in zip(futures_prices, maturities):
        if T > 0 and spot > 0 and F > 0:
            y = rate + storage_cost - math.log(F / spot) / T
            result.append((T, y))
    return result
