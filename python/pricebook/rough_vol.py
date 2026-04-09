"""Rough volatility: fractional Brownian motion and rBergomi model.

The rBergomi model uses a fractional Brownian motion (H < 0.5) to drive
the variance process, producing power-law term structure of ATM skew.

    dS/S = √v dW
    v(t) = ξ(t) × exp(η W^H(t) - 0.5 η² t^{2H})

    from pricebook.rough_vol import simulate_fbm, rbergomi_mc, rbergomi_european
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.black76 import OptionType


# ---- Fractional Brownian motion ----

def fbm_covariance(n_steps: int, H: float) -> np.ndarray:
    """Covariance matrix of fBM increments on a uniform grid.

    For increments Δ_i = B^H(t_{i+1}) - B^H(t_i) with unit spacing:
    Cov(Δ_i, Δ_j) = 0.5(|i-j+1|^{2H} + |i-j-1|^{2H} - 2|i-j|^{2H})

    This is a Toeplitz matrix (depends only on |i-j|).
    """
    twoH = 2 * H
    # Compute the autocovariance function
    acf = np.zeros(n_steps)
    for k in range(n_steps):
        acf[k] = 0.5 * (abs(k + 1) ** twoH + abs(k - 1) ** twoH - 2 * abs(k) ** twoH)

    # Build Toeplitz matrix
    from scipy.linalg import toeplitz
    return toeplitz(acf)


def simulate_fbm(
    T: float,
    n_steps: int,
    n_paths: int,
    H: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Simulate fractional Brownian motion paths via Cholesky.

    Args:
        T: time horizon.
        H: Hurst parameter (0 < H < 1, H < 0.5 for rough).
        n_steps: number of time steps.
        n_paths: number of paths.

    Returns:
        Array of shape (n_paths, n_steps+1) with fBM values (starts at 0).
    """
    rng = np.random.default_rng(seed)

    # Covariance of increments
    cov = fbm_covariance(n_steps, H)

    # Regularise for numerical stability
    cov += 1e-10 * np.eye(n_steps)

    L = np.linalg.cholesky(cov)

    Z = rng.standard_normal((n_paths, n_steps))
    increments = Z @ L.T

    # Scale by dt^H
    dt = T / n_steps
    increments *= dt ** H

    # Cumulate
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 1:] = np.cumsum(increments, axis=1)

    return paths


# ---- rBergomi model ----

def rbergomi_mc(
    spot: float,
    rate: float,
    xi: float,
    eta: float,
    H: float,
    T: float,
    n_steps: int = 100,
    n_paths: int = 10_000,
    rho: float = -0.7,
    div_yield: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate rBergomi model paths.

    v(t) = ξ × exp(η × W^H(t) - 0.5 × η² × t^{2H})
    dS/S = (r - q)dt + √v dW_S
    dW_S dW^H = ρ dt (approximate)

    Args:
        xi: forward variance level (flat vol² equivalent).
        eta: vol of vol.
        H: Hurst parameter (typically 0.05-0.15).
        rho: correlation between spot and vol.

    Returns:
        Terminal spot values, shape (n_paths,).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    # Simulate fBM for the variance driver
    W_H = simulate_fbm(T, n_steps, n_paths, H, seed)

    # Standard BM for spot
    Z_perp = rng.standard_normal((n_paths, n_steps))

    S = np.full(n_paths, spot, dtype=float)

    for step in range(n_steps):
        t = (step + 1) * dt

        # Variance process
        v = xi * np.exp(eta * W_H[:, step + 1] - 0.5 * eta ** 2 * t ** (2 * H))
        v = np.maximum(v, 1e-10)
        sqrt_v = np.sqrt(v)

        # Correlated spot increment
        dW_H = W_H[:, step + 1] - W_H[:, step]
        dW_S = rho * dW_H / (dt ** H + 1e-15) * sqrt_dt + math.sqrt(1 - rho ** 2) * Z_perp[:, step] * sqrt_dt

        S = S * np.exp(
            (rate - div_yield - 0.5 * v) * dt + sqrt_v * dW_S
        )

    return S


def rbergomi_european(
    spot: float,
    rate: float,
    xi: float,
    eta: float,
    H: float,
    strike: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    n_steps: int = 100,
    n_paths: int = 50_000,
    rho: float = -0.7,
    div_yield: float = 0.0,
    seed: int = 42,
) -> float:
    """Price a European option under rBergomi via MC."""
    S_T = rbergomi_mc(spot, rate, xi, eta, H, T, n_steps, n_paths, rho, div_yield, seed)
    df = math.exp(-rate * T)

    if option_type == OptionType.CALL:
        payoffs = np.maximum(S_T - strike, 0)
    else:
        payoffs = np.maximum(strike - S_T, 0)

    return float(df * payoffs.mean())


# ---- Implied vol term structure shape ----

def implied_vol_term_structure(
    spot: float,
    rate: float,
    xi: float,
    eta: float,
    H: float,
    expiries: list[float],
    n_paths: int = 50_000,
    rho: float = -0.7,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Compute ATM implied vol at multiple expiries under rBergomi.

    Returns (T, implied_vol) pairs. The ATM skew should follow
    a power law ∝ T^{H-0.5}.
    """
    from pricebook.implied_vol import implied_vol_newton

    results = []
    for T in expiries:
        if T <= 0:
            continue
        price = rbergomi_european(
            spot, rate, xi, eta, H, spot, T,
            n_steps=max(int(T * 100), 20), n_paths=n_paths, rho=rho, seed=seed,
        )
        fwd = spot * math.exp(rate * T)
        df = math.exp(-rate * T)
        try:
            iv = implied_vol_newton(price, fwd, spot, T, df, OptionType.CALL)
        except Exception:
            iv = math.sqrt(xi)  # fallback
        results.append((T, iv))

    return results
