"""MC migration utilities: helpers to rewire existing MC functions through MCEngine.

Provides drop-in wrappers that existing instruments can call instead of
rolling their own Euler loops.

    # Old pattern (in each file):
    rng = np.random.default_rng(seed)
    for step in range(n_steps):
        Z = rng.standard_normal(n_paths)
        S = S * np.exp(...)
    payoff = np.maximum(S - K, 0)
    price = np.mean(payoff) * df

    # New pattern (one line):
    from pricebook.models.mc_migrate import gbm_mc, stochvol_mc, multi_asset_mc
    paths = gbm_mc(spot, rate, vol, T, n_steps, n_paths, seed)
    # Then use paths as before for custom payoff

These are PATH GENERATORS only — they return numpy arrays in the same
format as the old code, so existing payoff logic works unchanged.
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.models.mc_engine import MCEngine, TimeGrid
from pricebook.models.mc_processes import (
    BlackScholesProcess, HestonProcess, SABRProcess,
    OUProcess, CIRProcess, CorrelatedGBMProcess,
    BatesProcess, CEVProcess,
    HullWhiteProcess, SLVProcess,
)


# ---------------------------------------------------------------------------
# Path generators (drop-in replacements for inline Euler loops)
# ---------------------------------------------------------------------------

def gbm_paths(
    spot: float,
    rate: float,
    vol: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
    div_yield: float = 0.0,
    antithetic: bool = False,
) -> np.ndarray:
    """Generate GBM spot paths. Returns (n_paths, n_steps+1) in SPOT space.

    Drop-in replacement for:
        S = spot; for i: S = S * exp(...)
    """
    process = BlackScholesProcess(spot, rate - div_yield, vol)
    engine = MCEngine(process, TimeGrid.uniform(T, n_steps), n_paths, seed, antithetic=antithetic)
    return np.exp(engine.paths)  # convert from log-space to spot space


def heston_paths(
    spot: float,
    v0: float,
    rate: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Heston (spot, variance) paths.

    Returns: (spot_paths, var_paths) each (n_paths, n_steps+1).
    """
    process = HestonProcess(spot, v0, rate, kappa, theta, xi, rho)
    engine = MCEngine(process, TimeGrid.uniform(T, n_steps), n_paths, seed)
    paths = engine.paths  # (n_paths, n_steps+1, 2)
    return np.exp(paths[:, :, 0]), paths[:, :, 1]  # spot, variance


def ou_paths(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate Ornstein-Uhlenbeck paths. Returns (n_paths, n_steps+1)."""
    process = OUProcess(x0, kappa, theta, sigma)
    engine = MCEngine(process, TimeGrid.uniform(T, n_steps), n_paths, seed)
    return engine.paths


def cir_paths(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate CIR paths (non-negative). Returns (n_paths, n_steps+1)."""
    process = CIRProcess(x0, kappa, theta, sigma)
    engine = MCEngine(process, TimeGrid.uniform(T, n_steps), n_paths, seed)
    return engine.paths


def hw_paths(
    r0: float,
    a: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
    theta_func=None,
) -> np.ndarray:
    """Generate Hull-White short rate paths. Returns (n_paths, n_steps+1)."""
    process = HullWhiteProcess(r0, a, sigma, theta_func)
    engine = MCEngine(process, TimeGrid.uniform(T, n_steps), n_paths, seed)
    return engine.paths


def correlated_gbm_paths(
    spots: list[float],
    rates: list[float],
    vols: list[float],
    correlation: np.ndarray,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate correlated multi-asset GBM paths.

    Returns (n_paths, n_steps+1, n_assets) in SPOT space.
    """
    process = CorrelatedGBMProcess(spots, rates, vols, correlation)
    engine = MCEngine(process, TimeGrid.uniform(T, n_steps), n_paths, seed)
    return np.exp(engine.paths)  # log-space → spot space


def sabr_paths(
    f0: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SABR (forward, vol) paths.

    Returns: (forward_paths, vol_paths) each (n_paths, n_steps+1).
    """
    process = SABRProcess(f0, alpha, beta, rho, nu)
    engine = MCEngine(process, TimeGrid.uniform(T, n_steps), n_paths, seed)
    paths = engine.paths
    return paths[:, :, 0], paths[:, :, 1]


def bates_paths(
    spot: float,
    v0: float,
    rate: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Bates (Heston+jumps) paths.

    Returns: (spot_paths, var_paths) each (n_paths, n_steps+1).
    """
    process = BatesProcess(spot, v0, rate, kappa, theta, xi, rho,
                           jump_intensity, jump_mean, jump_vol)
    engine = MCEngine(process, TimeGrid.uniform(T, n_steps), n_paths, seed)
    paths = engine.paths
    return np.exp(paths[:, :, 0]), paths[:, :, 1]


def cev_paths(
    s0: float,
    mu: float,
    sigma: float,
    beta: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate CEV paths. Returns (n_paths, n_steps+1) in SPOT space."""
    process = CEVProcess(s0, mu, sigma, beta)
    engine = MCEngine(process, TimeGrid.uniform(T, n_steps), n_paths, seed)
    return engine.paths  # already in spot space


# ---------------------------------------------------------------------------
# Quick price helper (for simple European payoffs)
# ---------------------------------------------------------------------------

def quick_mc_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 1,
    seed: int = 42,
    antithetic: bool = True,
    div_yield: float = 0.0,
) -> dict:
    """Quick European option price via MCEngine.

    Returns dict with price, stderr, n_paths.
    """
    process = BlackScholesProcess(spot, rate - div_yield, vol)
    engine = MCEngine(process, TimeGrid.uniform(T, n_steps), n_paths, seed, antithetic=antithetic)
    df = math.exp(-rate * T)

    if option_type == "call":
        from pricebook.models.mc_payoffs import european_call
        payoff = european_call(strike)
    else:
        from pricebook.models.mc_payoffs import european_put
        payoff = european_put(strike)

    result = engine.price(payoff, df)
    return {"price": result.price, "stderr": result.stderr, "n_paths": result.n_paths}
