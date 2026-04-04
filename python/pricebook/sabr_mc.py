"""SABR Monte Carlo simulation.

Direct simulation of the SABR SDE for path-dependent pricing.
Absorbing boundary at F=0 for beta < 1.

Dynamics:
    dF = sigma * F^beta * dW1
    dsigma = alpha_vol * sigma * dW2     (alpha_vol = nu in Hagan notation)
    dW1 * dW2 = rho * dt
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.black76 import OptionType, black76_price
from pricebook.sabr import sabr_implied_vol


def sabr_mc_paths(
    forward: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    n_steps: int = 200,
    n_paths: int = 50_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate SABR forward and vol paths.

    Uses Euler-Maruyama with absorbing boundary at F=0.

    Returns:
        (F, sigma) arrays of shape (n_paths, n_steps+1).
    """
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    rng = np.random.default_rng(seed)

    F = np.zeros((n_paths, n_steps + 1))
    sig = np.zeros((n_paths, n_steps + 1))
    F[:, 0] = forward
    sig[:, 0] = alpha

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w1 = z1
        w2 = rho * z1 + math.sqrt(1.0 - rho**2) * z2

        f = F[:, t]
        s = sig[:, t]

        # Absorbing boundary: F stays at 0 once it hits
        alive = f > 0
        f_beta = np.where(alive, np.power(np.maximum(f, 1e-30), beta), 0.0)

        F[:, t + 1] = np.where(
            alive,
            f + s * f_beta * sqrt_dt * w1,
            0.0,
        )
        F[:, t + 1] = np.maximum(F[:, t + 1], 0.0)

        # Log-normal vol process
        sig[:, t + 1] = s * np.exp(-0.5 * nu**2 * dt + nu * sqrt_dt * w2)

    return F, sig


def sabr_mc_european(
    forward: float,
    strike: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    df: float = 1.0,
    option_type: OptionType = OptionType.CALL,
    n_steps: int = 200,
    n_paths: int = 100_000,
    seed: int = 42,
) -> float:
    """European option under SABR via MC."""
    F, _ = sabr_mc_paths(forward, T, alpha, beta, rho, nu, n_steps, n_paths, seed)
    FT = F[:, -1]

    if option_type == OptionType.CALL:
        payoff = np.maximum(FT - strike, 0.0)
    else:
        payoff = np.maximum(strike - FT, 0.0)

    return float(df * payoff.mean())


def sabr_mc_asian(
    forward: float,
    strike: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    df: float = 1.0,
    option_type: OptionType = OptionType.CALL,
    n_steps: int = 200,
    n_paths: int = 100_000,
    seed: int = 42,
) -> float:
    """Asian option (arithmetic average) under SABR via MC."""
    F, _ = sabr_mc_paths(forward, T, alpha, beta, rho, nu, n_steps, n_paths, seed)
    avg = F[:, 1:].mean(axis=1)

    if option_type == OptionType.CALL:
        payoff = np.maximum(avg - strike, 0.0)
    else:
        payoff = np.maximum(strike - avg, 0.0)

    return float(df * payoff.mean())


def sabr_mc_implied_vol(
    forward: float,
    strike: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    n_steps: int = 200,
    n_paths: int = 200_000,
    seed: int = 42,
) -> float:
    """Implied vol from SABR MC price (for comparison with Hagan)."""
    from pricebook.implied_vol import implied_vol_black76

    mc_price = sabr_mc_european(forward, strike, T, alpha, beta, rho, nu,
                                df=1.0, n_steps=n_steps, n_paths=n_paths, seed=seed)

    return implied_vol_black76(mc_price, forward, strike, T, df=1.0)
