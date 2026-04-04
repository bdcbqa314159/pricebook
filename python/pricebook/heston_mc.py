"""Heston Monte Carlo simulation.

Full-truncation Euler and QE (quadratic exponential) schemes for the
variance process. Enables path-dependent pricing under stochastic vol.
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.black76 import OptionType


def heston_euler(
    spot: float,
    rate: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    n_steps: int = 100,
    n_paths: int = 10_000,
    div_yield: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Heston paths via full-truncation Euler.

    Full truncation: use max(v, 0) in both drift and diffusion to prevent
    negative variance without bias.

    Returns:
        (S, v) where S is (n_paths, n_steps+1) spot paths
        and v is (n_paths, n_steps+1) variance paths.
    """
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    rng = np.random.default_rng(seed)

    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = spot
    v[:, 0] = v0

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w1 = z1
        w2 = rho * z1 + math.sqrt(1.0 - rho**2) * z2

        v_pos = np.maximum(v[:, t], 0.0)
        sqrt_v = np.sqrt(v_pos)

        # Variance: full truncation
        v[:, t + 1] = v[:, t] + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * w2
        v[:, t + 1] = np.maximum(v[:, t + 1], 0.0)

        # Log-spot
        S[:, t + 1] = S[:, t] * np.exp(
            (rate - div_yield - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * w1
        )

    return S, v


def heston_qe(
    spot: float,
    rate: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    n_steps: int = 100,
    n_paths: int = 10_000,
    div_yield: float = 0.0,
    seed: int = 42,
    psi_crit: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Heston paths via QE (quadratic exponential) scheme.

    Andersen (2008): uses moment-matching for the variance process.
    More accurate than Euler, especially for low vol-of-vol or near
    the Feller boundary.

    Returns:
        (S, v) arrays of shape (n_paths, n_steps+1).
    """
    dt = T / n_steps
    rng = np.random.default_rng(seed)

    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = spot
    v[:, 0] = v0

    exp_kdt = math.exp(-kappa * dt)
    k1 = (exp_kdt * (1 - exp_kdt)) / kappa if kappa > 1e-10 else dt
    # Exact moments of v(t+dt) given v(t)
    # E[v(t+dt)] = theta + (v(t) - theta) * exp(-kappa*dt)
    # Var[v(t+dt)] = v(t) * xi^2 * exp(-kdt) * (1-exp(-kdt)) / kappa
    #              + theta * xi^2 * (1-exp(-kdt))^2 / (2*kappa)

    c1 = xi**2 * exp_kdt * (1.0 - exp_kdt) / kappa if kappa > 1e-10 else xi**2 * dt
    c2 = theta * xi**2 * (1.0 - exp_kdt)**2 / (2.0 * kappa) if kappa > 1e-10 else 0.0

    for t in range(n_steps):
        # Moments
        m = theta + (v[:, t] - theta) * exp_kdt
        s2 = v[:, t] * c1 + c2
        s2 = np.maximum(s2, 0.0)
        psi = s2 / np.maximum(m**2, 1e-30)

        # QE: switch between exponential and quadratic approximation
        v_next = np.zeros(n_paths)
        u = rng.uniform(size=n_paths)

        # Quadratic branch (psi <= psi_crit)
        quad_mask = psi <= psi_crit
        if np.any(quad_mask):
            psi_q = psi[quad_mask]
            m_q = m[quad_mask]
            inv_p = 2.0 / psi_q
            b2 = np.maximum(inv_p - 1.0 + np.sqrt(inv_p * (inv_p - 1.0)), 0.0)
            b = np.sqrt(b2)
            a = m_q / (1.0 + b2)
            z = rng.standard_normal(int(quad_mask.sum()))
            v_next[quad_mask] = a * (b + z)**2

        # Exponential branch (psi > psi_crit)
        exp_mask = ~quad_mask
        if np.any(exp_mask):
            psi_e = psi[exp_mask]
            m_e = m[exp_mask]
            p = (psi_e - 1.0) / (psi_e + 1.0)
            beta = (1.0 - p) / np.maximum(m_e, 1e-30)
            u_e = u[exp_mask]
            # Inverse CDF: v = (1/beta) * ln((1-p)/(1-u)) for u > p
            v_next[exp_mask] = np.where(
                u_e > p,
                np.log(np.maximum((1.0 - p) / np.maximum(1.0 - u_e, 1e-30), 1e-30)) / beta,
                0.0,
            )

        v[:, t + 1] = np.maximum(v_next, 0.0)

        # Log-spot: exact integration conditional on v
        z1 = rng.standard_normal(n_paths)
        k0 = -rho * kappa * theta / xi * dt
        k1_spot = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
        k2_spot = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
        k3 = 0.5 * dt * (1.0 - rho**2)

        ln_S = np.log(S[:, t]) + (rate - div_yield) * dt + k0 \
            + k1_spot * v[:, t] + k2_spot * v[:, t + 1] \
            + np.sqrt(k3 * (v[:, t] + v[:, t + 1])) * z1

        S[:, t + 1] = np.exp(ln_S)

    return S, v


# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------


def heston_mc_european(
    spot: float,
    strike: float,
    rate: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    n_steps: int = 100,
    n_paths: int = 50_000,
    scheme: str = "qe",
    seed: int = 42,
) -> float:
    """European option price under Heston via MC."""
    sim = heston_qe if scheme == "qe" else heston_euler
    S, _ = sim(spot, rate, T, v0, kappa, theta, xi, rho,
               n_steps, n_paths, div_yield, seed)

    ST = S[:, -1]
    df = math.exp(-rate * T)

    if option_type == OptionType.CALL:
        payoff = np.maximum(ST - strike, 0.0)
    else:
        payoff = np.maximum(strike - ST, 0.0)

    return float(df * payoff.mean())


def heston_mc_barrier(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    is_up: bool = False,
    is_knock_in: bool = False,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    n_steps: int = 200,
    n_paths: int = 50_000,
    scheme: str = "qe",
    seed: int = 42,
) -> float:
    """Barrier option under Heston via MC."""
    sim = heston_qe if scheme == "qe" else heston_euler
    S, _ = sim(spot, rate, T, v0, kappa, theta, xi, rho,
               n_steps, n_paths, div_yield, seed)

    df = math.exp(-rate * T)
    ST = S[:, -1]

    if option_type == OptionType.CALL:
        payoff = np.maximum(ST - strike, 0.0)
    else:
        payoff = np.maximum(strike - ST, 0.0)

    # Check barrier hit along path
    if is_up:
        hit = np.any(S >= barrier, axis=1)
    else:
        hit = np.any(S <= barrier, axis=1)

    if is_knock_in:
        payoff = np.where(hit, payoff, 0.0)
    else:
        payoff = np.where(hit, 0.0, payoff)

    return float(df * payoff.mean())
