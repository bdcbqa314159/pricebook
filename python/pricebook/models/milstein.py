"""Milstein scheme: strong order 1.0 SDE discretisation.

Extends Euler-Maruyama (strong order 0.5) with the Itô-Taylor correction
term involving the diffusion derivative:

    X_{n+1} = X_n + μ Δt + σ ΔW + 0.5 σ σ' (ΔW² − Δt)

where σ' = dσ/dX. This halves the strong error exponent.

* :func:`milstein_step` — one Milstein step.
* :func:`milstein_paths` — full path simulation.
* :func:`milstein_gbm` — GBM specialisation (σ' = σ, exact correction).
* :func:`milstein_cev` — CEV model (σ' = β σ S^{β-1}).
* :func:`milstein_cir` — CIR variance (σ' = ξ/(2√v), with absorption).

References:
    Kloeden & Platen, *Numerical Solution of Stochastic Differential
    Equations*, Springer, 1992, Ch. 10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class MilsteinResult:
    """Result of Milstein simulation."""
    paths: np.ndarray       # (n_paths, n_steps+1)
    times: np.ndarray       # (n_steps+1,)
    n_paths: int
    n_steps: int
    dt: float


# ---- Generic Milstein ----

def milstein_step(
    x: np.ndarray,
    drift: Callable[[np.ndarray], np.ndarray],
    diffusion: Callable[[np.ndarray], np.ndarray],
    diffusion_deriv: Callable[[np.ndarray], np.ndarray],
    dt: float,
    dW: np.ndarray,
) -> np.ndarray:
    """One Milstein step for a general SDE dX = μ(X)dt + σ(X)dW.

    X_{n+1} = X_n + μ(X_n)Δt + σ(X_n)ΔW + 0.5 σ(X_n) σ'(X_n) (ΔW² − Δt)

    Args:
        x: current state (n_paths,).
        drift: μ(x) function.
        diffusion: σ(x) function.
        diffusion_deriv: σ'(x) = dσ/dx function.
        dt: time step.
        dW: Brownian increments (n_paths,).

    Returns:
        Next state (n_paths,).
    """
    mu = drift(x)
    sigma = diffusion(x)
    sigma_prime = diffusion_deriv(x)
    return x + mu * dt + sigma * dW + 0.5 * sigma * sigma_prime * (dW * dW - dt)


def milstein_paths(
    x0: float,
    drift: Callable[[np.ndarray], np.ndarray],
    diffusion: Callable[[np.ndarray], np.ndarray],
    diffusion_deriv: Callable[[np.ndarray], np.ndarray],
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> MilsteinResult:
    """Simulate full paths with the Milstein scheme.

    Args:
        x0: initial value.
        drift: μ(x).
        diffusion: σ(x).
        diffusion_deriv: σ'(x) = dσ/dx.
        T: time horizon.
        n_steps: number of time steps.
        n_paths: number of paths.
        seed: random seed.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    times = np.linspace(0, T, n_steps + 1)

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0

    for i in range(n_steps):
        dW = rng.standard_normal(n_paths) * sqrt_dt
        paths[:, i + 1] = milstein_step(
            paths[:, i], drift, diffusion, diffusion_deriv, dt, dW,
        )

    return MilsteinResult(paths, times, n_paths, n_steps, dt)


# ---- GBM specialisation ----

def milstein_gbm(
    spot: float,
    rate: float,
    vol: float,
    T: float,
    n_steps: int,
    n_paths: int,
    div_yield: float = 0.0,
    seed: int | None = None,
) -> MilsteinResult:
    """Milstein for GBM: dS = (r−q)S dt + σS dW.

    For GBM, σ(S) = σS and σ'(S) = σ, so the Milstein correction is:
        0.5 σ²S (ΔW² − Δt)

    This actually makes the Milstein scheme equivalent to the exact
    log-Euler for GBM (strong order 1.0).
    """
    mu = rate - div_yield

    def drift(S):
        return mu * S

    def diffusion(S):
        return vol * S

    def diffusion_deriv(S):
        return vol * np.ones_like(S)

    return milstein_paths(spot, drift, diffusion, diffusion_deriv,
                          T, n_steps, n_paths, seed)


# ---- CEV specialisation ----

def milstein_cev(
    spot: float,
    rate: float,
    vol: float,
    beta: float,
    T: float,
    n_steps: int,
    n_paths: int,
    div_yield: float = 0.0,
    seed: int | None = None,
) -> MilsteinResult:
    """Milstein for CEV: dS = (r−q)S dt + σ S^β dW.

    σ(S) = σ S^β,  σ'(S) = β σ S^{β-1}.
    """
    mu = rate - div_yield

    def drift(S):
        return mu * S

    def diffusion(S):
        return vol * np.power(np.maximum(S, 1e-10), beta)

    def diffusion_deriv(S):
        return beta * vol * np.power(np.maximum(S, 1e-10), beta - 1)

    return milstein_paths(spot, drift, diffusion, diffusion_deriv,
                          T, n_steps, n_paths, seed)


# ---- CIR specialisation ----

def milstein_cir(
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> MilsteinResult:
    """Milstein for CIR: dv = κ(θ−v)dt + ξ√v dW.

    σ(v) = ξ√v,  σ'(v) = ξ/(2√v).

    Absorption at v=0 to prevent negative variance.
    """
    def drift(v):
        return kappa * (theta - v)

    def diffusion(v):
        return xi * np.sqrt(np.maximum(v, 0.0))

    def diffusion_deriv(v):
        safe_v = np.maximum(v, 1e-10)
        return xi / (2.0 * np.sqrt(safe_v))

    result = milstein_paths(v0, drift, diffusion, diffusion_deriv,
                            T, n_steps, n_paths, seed)
    # Absorb at zero
    result.paths = np.maximum(result.paths, 0.0)
    return result
