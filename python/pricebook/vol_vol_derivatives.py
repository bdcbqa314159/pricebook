"""Vol-of-vol derivatives: options on variance, gamma swaps, corridor variance.

* :func:`option_on_variance_swap` — call/put on realised variance.
* :func:`gamma_swap_price` — spot-weighted variance swap.
* :func:`corridor_variance_swap` — conditional variance in a range.
* :func:`vix_option_price` — VIX-like option via Heston variance dynamics.

References:
    Carr & Lee, *Robust Replication of Volatility Derivatives*, 2009.
    Carr & Madan, *Towards a Theory of Volatility Trading*, 1998.
    Demeterfi, Derman, Kamal & Zou, *More Than You Ever Wanted to Know
    About Variance Swaps*, 1999.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class OptionOnVarianceResult:
    """Option on variance swap result."""
    price: float
    expected_variance: float
    variance_of_variance: float
    strike: float
    is_call: bool


def option_on_variance_swap(
    expected_variance: float,
    vol_of_variance: float,
    strike: float,
    T: float,
    rate: float = 0.0,
    is_call: bool = True,
    n_paths: int = 20_000,
    seed: int | None = 42,
) -> OptionOnVarianceResult:
    """Option on realised variance (vol-of-vol product).

    Payoff: max(realised_var − K, 0) for call (receiver of vol-of-vol).

    Priced via lognormal MC on the variance distribution.

    Args:
        expected_variance: E[σ²_realised] (from var swap fair strike).
        vol_of_variance: volatility of the variance itself.
        strike: variance strike.
    """
    rng = np.random.default_rng(seed)
    df = math.exp(-rate * T)

    # Simulate variance as lognormal
    log_mean = math.log(max(expected_variance, 1e-10)) - 0.5 * vol_of_variance**2 * T
    log_vol = vol_of_variance * math.sqrt(T)
    var_sim = np.exp(log_mean + log_vol * rng.standard_normal(n_paths))

    if is_call:
        payoff = np.maximum(var_sim - strike, 0.0)
    else:
        payoff = np.maximum(strike - var_sim, 0.0)

    price = df * float(payoff.mean())

    return OptionOnVarianceResult(
        price=float(price),
        expected_variance=expected_variance,
        variance_of_variance=float(var_sim.var()),
        strike=strike,
        is_call=is_call,
    )


@dataclass
class GammaSwapResult:
    """Gamma swap result."""
    fair_strike: float          # fair gamma swap strike (vol²)
    variance_swap_strike: float
    gamma_adjustment: float     # gamma - variance (positive for positive skew)
    method: str


def gamma_swap_price(
    spot_paths: np.ndarray,     # (n_paths, n_steps+1)
    rate: float = 0.0,
    T: float = 1.0,
) -> GammaSwapResult:
    """Gamma swap: variance swap weighted by S(t) / S(0).

    Payoff = Σ (S_t / S_0) × (log return)² vs fair strike.

    Key property: gamma swap is less sensitive to spot-vol correlation
    than variance swap. In fact, it has a model-free replication too.

    Args:
        spot_paths: (n_paths, n_steps+1) spot price paths.
    """
    n_paths, n_cols = spot_paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps
    S0 = spot_paths[:, 0]

    # Variance swap: Σ (log return)² / T
    log_returns = np.diff(np.log(spot_paths), axis=1)
    var_swap = (log_returns**2).sum(axis=1) / T

    # Gamma swap: Σ (S_t / S_0) × (log return)² / T
    weights = spot_paths[:, :-1] / S0[:, np.newaxis]
    gamma_swap = (weights * log_returns**2).sum(axis=1) / T

    var_fair = float(var_swap.mean())
    gamma_fair = float(gamma_swap.mean())

    return GammaSwapResult(
        fair_strike=float(gamma_fair),
        variance_swap_strike=float(var_fair),
        gamma_adjustment=float(gamma_fair - var_fair),
        method="mc_realised",
    )


@dataclass
class CorridorVarianceResult:
    """Corridor variance swap result."""
    conditional_variance: float
    total_variance: float
    time_in_corridor: float     # fraction of time in range
    range_low: float
    range_high: float


def corridor_variance_swap(
    spot_paths: np.ndarray,
    range_low: float,
    range_high: float,
    T: float = 1.0,
) -> CorridorVarianceResult:
    """Corridor variance swap: accumulates variance only when spot in range.

    Fair strike = E[Σ 1_{L < S < U} × (log return)²] / T.

    Cheaper than full variance swap; captures vol in a specific regime.
    """
    n_paths, n_cols = spot_paths.shape
    n_steps = n_cols - 1

    log_returns = np.diff(np.log(spot_paths), axis=1)
    in_range = (spot_paths[:, :-1] >= range_low) & (spot_paths[:, :-1] < range_high)

    corridor_var = (in_range * log_returns**2).sum(axis=1) / T
    total_var = (log_returns**2).sum(axis=1) / T
    time_frac = in_range.mean(axis=1)

    return CorridorVarianceResult(
        conditional_variance=float(corridor_var.mean()),
        total_variance=float(total_var.mean()),
        time_in_corridor=float(time_frac.mean()),
        range_low=range_low,
        range_high=range_high,
    )


@dataclass
class VIXOptionResult:
    """VIX-like option result."""
    price: float
    forward_vix: float
    strike: float
    is_call: bool


def vix_option_price(
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    T_option: float,
    T_variance: float,
    strike_vol: float,
    rate: float = 0.0,
    is_call: bool = True,
    n_paths: int = 20_000,
    n_steps: int = 50,
    seed: int | None = 42,
) -> VIXOptionResult:
    """VIX-like option via Heston variance simulation.

    VIX² ≈ E[∫ v_s ds over next 30 days] / τ  (30-day expected variance).

    Simulate v_t under Heston to T_option, then compute expected
    variance over [T_option, T_option + T_variance], and price a
    call/put on √(expected_var).

    Args:
        v0, kappa, theta, xi: Heston variance parameters.
        T_option: option expiry.
        T_variance: variance horizon (≈ 30/365 for VIX).
        strike_vol: VIX strike (in vol terms, e.g. 0.20 for 20%).
    """
    rng = np.random.default_rng(seed)
    dt = T_option / n_steps
    sqrt_dt = math.sqrt(dt)

    v = np.full(n_paths, v0)

    for _ in range(n_steps):
        dW = rng.standard_normal(n_paths) * sqrt_dt
        v_pos = np.maximum(v, 0.0)
        v = np.maximum(v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * dW, 0.0)

    # At T_option, approximate VIX² = E[v over next T_variance | v(T)]
    # Under Heston: E[v_T | v_t] = θ + (v_t − θ) e^{−κτ}
    # So E[∫ v ds] / τ ≈ θ + (v − θ)(1 − e^{−κτ})/(κτ)
    if kappa > 1e-10:
        e = math.exp(-kappa * T_variance)
        expected_var = theta + (v - theta) * (1 - e) / (kappa * T_variance)
    else:
        expected_var = v

    vix = np.sqrt(np.maximum(expected_var, 0.0))

    strike_sq = strike_vol

    if is_call:
        payoff = np.maximum(vix - strike_sq, 0.0)
    else:
        payoff = np.maximum(strike_sq - vix, 0.0)

    df = math.exp(-rate * T_option)
    price = df * float(payoff.mean())
    forward_vix = float(vix.mean())

    return VIXOptionResult(
        price=float(price),
        forward_vix=forward_vix,
        strike=strike_vol,
        is_call=is_call,
    )
