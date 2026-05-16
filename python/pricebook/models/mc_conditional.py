"""Conditional Monte Carlo for stochastic volatility models.

Instead of MC on both spot and vol, simulate vol paths only and
price analytically (Black-Scholes) conditional on the realised vol.
Typically reduces variance by 10-100x.

    from pricebook.models.mc_conditional import conditional_mc_heston, conditional_mc_generic

    result = conditional_mc_heston(s0=100, v0=0.04, kappa=2, theta=0.04,
                                    xi=0.3, rho=-0.7, strike=100, T=1.0, r=0.05)

References:
    Willard (1997). Calculating Prices and Sensitivities for Path-Independent
    Derivative Securities in Multifactor Models. JoD.
    Romano & Touzi (1997). Contingent Claims and Market Completeness in a
    Stochastic Volatility Model. Math. Finance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricebook.models.mc_engine import MCEngine, TimeGrid, MCResult


def _bs_call(s0, k, r, sigma, T):
    """Analytical Black-Scholes call."""
    if sigma <= 0 or T <= 0:
        return max(s0 * math.exp(r * T) - k, 0.0) * math.exp(-r * T)
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return s0 * norm.cdf(d1) - k * math.exp(-r * T) * norm.cdf(d2)


def _bs_put(s0, k, r, sigma, T):
    return _bs_call(s0, k, r, sigma, T) - s0 + k * math.exp(-r * T)


def conditional_mc_heston(
    s0: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    strike: float,
    T: float,
    r: float = 0.05,
    option_type: str = "call",
    n_paths: int = 50_000,
    n_steps: int = 100,
    seed: int = 42,
) -> MCResult:
    """Conditional MC for Heston: simulate vol, price BS conditional.

    Steps:
    1. Simulate variance paths v(t) via CIR dynamics.
    2. For each path, compute realised vol = √(∫v dt / T).
    3. Compute conditional spot drift adjustment for ρ ≠ 0:
       E[log(S_T) | v-path] = log(S₀) + (r - ½σ̄²)T + ρ/ξ × (v_T - v₀ - κθT + κ∫v dt)
    4. Price with BS using adjusted forward and realised vol × √(1-ρ²).

    This eliminates MC noise from the spot dimension entirely.
    Variance reduction: ~10-50x vs full 2D MC.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # 1. Simulate variance paths
    v_paths = np.zeros((n_paths, n_steps + 1))
    v_paths[:, 0] = v0

    for i in range(n_steps):
        v = np.maximum(v_paths[:, i], 0.0)
        sqrt_v = np.sqrt(v)
        dw = rng.standard_normal(n_paths) * np.sqrt(dt)
        v_paths[:, i + 1] = v + kappa * (theta - v) * dt + xi * sqrt_v * dw
        v_paths[:, i + 1] = np.maximum(v_paths[:, i + 1], 0.0)

    # 2. Realised (integrated) variance
    integrated_var = np.sum(v_paths[:, :-1] * dt, axis=1)  # ∫v dt
    realised_vol = np.sqrt(integrated_var / T)

    # 3. Conditional BS pricing
    v_T = v_paths[:, -1]

    # Drift adjustment for correlation (Romano-Touzi)
    drift_adj = (rho / xi) * (v_T - v0 - kappa * theta * T + kappa * integrated_var)

    # Conditional forward
    log_fwd = np.log(s0) + r * T + drift_adj - 0.5 * integrated_var
    fwd = np.exp(log_fwd)

    # Conditional vol (orthogonal component)
    cond_vol = realised_vol * np.sqrt(max(1 - rho ** 2, 0))

    # 4. Price BS for each path
    prices = np.zeros(n_paths)
    df = math.exp(-r * T)

    for i in range(n_paths):
        vol_i = float(cond_vol[i])
        fwd_i = float(fwd[i])
        if vol_i > 1e-10 and fwd_i > 0:
            if option_type == "call":
                prices[i] = _bs_call(fwd_i, strike, 0.0, vol_i, T)
            else:
                prices[i] = _bs_put(fwd_i, strike, 0.0, vol_i, T)
        else:
            if option_type == "call":
                prices[i] = max(fwd_i - strike, 0.0)
            else:
                prices[i] = max(strike - fwd_i, 0.0)

    discounted = prices * df
    price = float(np.mean(discounted))
    stderr = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))

    return MCResult(
        price=price, stderr=stderr,
        n_paths=n_paths, n_steps=n_steps,
        confidence_95=(price - 1.96 * stderr, price + 1.96 * stderr),
    )


def conditional_mc_generic(
    vol_engine: MCEngine,
    s0: float,
    r: float,
    strike: float,
    T: float,
    rho: float = 0.0,
    option_type: str = "call",
) -> MCResult:
    """Generic conditional MC: any vol process + analytical inner pricing.

    Uses pre-simulated vol paths from an MCEngine, computes realised vol
    per path, then prices analytically.

    Args:
        vol_engine: MCEngine with 1D vol/variance paths already generated.
        s0: initial spot.
        r: risk-free rate.
        strike: option strike.
        T: maturity.
        rho: spot-vol correlation.
        option_type: "call" or "put".
    """
    paths = vol_engine.paths
    times = vol_engine.time_grid
    dt_arr = times.dt

    if paths.ndim == 3:
        v_paths = paths[:, :, 0]
    else:
        v_paths = paths

    n_paths = v_paths.shape[0]

    # Integrated variance
    integrated_var = np.sum(v_paths[:, :-1] * dt_arr[np.newaxis, :], axis=1)
    realised_vol = np.sqrt(np.maximum(integrated_var / T, 1e-15))

    # Conditional vol (remove correlation component)
    cond_vol = realised_vol * np.sqrt(max(1 - rho ** 2, 0))

    # Price BS per path
    df = math.exp(-r * T)
    prices = np.zeros(n_paths)

    for i in range(n_paths):
        vol_i = float(cond_vol[i])
        if vol_i > 1e-10:
            if option_type == "call":
                prices[i] = _bs_call(s0, strike, r, vol_i, T)
            else:
                prices[i] = _bs_put(s0, strike, r, vol_i, T)
        else:
            fwd = s0 * math.exp(r * T)
            prices[i] = max(fwd - strike, 0.0) if option_type == "call" else max(strike - fwd, 0.0)
            prices[i] *= df

    price = float(np.mean(prices))
    stderr = float(np.std(prices, ddof=1) / np.sqrt(n_paths))

    return MCResult(
        price=price, stderr=stderr,
        n_paths=n_paths, n_steps=times.n_steps,
        confidence_95=(price - 1.96 * stderr, price + 1.96 * stderr),
    )
