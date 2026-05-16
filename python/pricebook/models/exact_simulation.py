"""Exact simulation: non-central chi-squared CIR, implicit Euler for stiff SDEs.

* :func:`exact_cir` — exact CIR simulation via non-central chi-squared.
* :func:`exact_cir_zcb` — analytical CIR zero-coupon bond price.
* :func:`implicit_euler_step` — implicit Euler for stiff SDEs (iterative).
* :func:`implicit_euler_paths` — full path simulation with implicit Euler.

References:
    Broadie & Kaya, *Exact Simulation of Stochastic Volatility and
    Other Affine Jump Diffusion Processes*, Operations Research, 2006.
    Glasserman, *Monte Carlo Methods*, Ch. 3.5 (CIR exact sampling).
    Cox, Ingersoll & Ross, *A Theory of the Term Structure*, 1985.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import ncx2


# ---- Exact CIR simulation ----

@dataclass
class ExactCIRResult:
    """Result of exact CIR simulation."""
    paths: np.ndarray      # (n_paths, n_steps+1)
    times: np.ndarray      # (n_steps+1,)
    n_paths: int
    n_steps: int


def exact_cir(
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> ExactCIRResult:
    """Exact CIR simulation via non-central chi-squared sampling.

    The transition density of the CIR process
        dv = κ(θ − v)dt + ξ√v dW
    is a scaled non-central chi-squared:

        v(t+Δt) | v(t) ~ (ξ²(1−e^{−κΔt})/(4κ)) × χ²(d, λ)

    where:
        d = 4κθ/ξ² (degrees of freedom)
        λ = 4κ e^{−κΔt} v(t) / (ξ²(1−e^{−κΔt})) (non-centrality)

    This is exact — no discretisation bias whatsoever.

    Args:
        v0: initial variance/rate.
        kappa: mean-reversion speed.
        theta: long-run level.
        xi: vol-of-vol.
        T: time horizon.
        n_steps: number of time steps.
        n_paths: number of paths.
        seed: random seed.

    Reference:
        Glasserman, Monte Carlo Methods, Prop. 3.12.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = v0

    # CIR parameters for non-central chi-squared
    d = 4.0 * kappa * theta / (xi * xi)  # degrees of freedom
    exp_kdt = math.exp(-kappa * dt)
    c = xi * xi * (1.0 - exp_kdt) / (4.0 * kappa)  # scale factor

    for i in range(n_steps):
        v_prev = paths[:, i]
        # Non-centrality parameter
        lam = 4.0 * kappa * exp_kdt * v_prev / (xi * xi * (1.0 - exp_kdt))
        lam = np.maximum(lam, 0.0)

        # Sample from non-central chi-squared
        # ncx2.rvs(df, nc) gives χ²(d, λ)
        samples = ncx2.rvs(d, lam, random_state=rng)
        paths[:, i + 1] = c * samples

    return ExactCIRResult(paths, times, n_paths, n_steps)


# ---- Analytical CIR ZCB price ----

def exact_cir_zcb(
    r0: float,
    kappa: float,
    theta: float,
    xi: float,
    T: float,
) -> float:
    """Analytical zero-coupon bond price under the CIR model.

    P(0, T) = A(T) exp(−B(T) r₀)

    where:
        γ = √(κ² + 2ξ²)
        B(T) = 2(e^{γT} − 1) / ((γ+κ)(e^{γT} − 1) + 2γ)
        A(T) = [2γ exp((κ+γ)T/2) / ((γ+κ)(e^{γT} − 1) + 2γ)]^{2κθ/ξ²}

    Reference:
        Cox, Ingersoll & Ross, Econometrica, 1985, Eq. (23).
    """
    if T <= 0:
        return 1.0

    gamma = math.sqrt(kappa * kappa + 2 * xi * xi)
    exp_gt = math.exp(gamma * T)

    denom = (gamma + kappa) * (exp_gt - 1) + 2 * gamma
    B = 2.0 * (exp_gt - 1.0) / denom

    exponent = 2.0 * kappa * theta / (xi * xi)
    A_num = 2.0 * gamma * math.exp((kappa + gamma) * T / 2.0)
    A = (A_num / denom) ** exponent

    return A * math.exp(-B * r0)


# ---- Implicit Euler for stiff SDEs ----

def implicit_euler_step(
    x: np.ndarray,
    drift: Callable[[np.ndarray], np.ndarray],
    diffusion: Callable[[np.ndarray], np.ndarray],
    dt: float,
    dW: np.ndarray,
    n_iter: int = 5,
) -> np.ndarray:
    """One implicit Euler step for a stiff SDE.

    Solves x_{n+1} = x_n + μ(x_{n+1})Δt + σ(x_n)ΔW
    via fixed-point iteration:
        x^{(0)} = x_n + μ(x_n)Δt + σ(x_n)ΔW   (explicit predictor)
        x^{(k+1)} = x_n + μ(x^{(k)})Δt + σ(x_n)ΔW   (corrector)

    The diffusion term uses x_n (semi-implicit / drift-implicit).

    Args:
        x: current state (n_paths,).
        drift: μ(x) function.
        diffusion: σ(x) function.
        dt: time step.
        dW: Brownian increments (n_paths,).
        n_iter: number of fixed-point iterations.
    """
    sigma_dW = diffusion(x) * dW
    x_new = x + drift(x) * dt + sigma_dW  # explicit predictor

    for _ in range(n_iter):
        x_new = x + drift(x_new) * dt + sigma_dW  # corrector

    return x_new


def implicit_euler_paths(
    x0: float,
    drift: Callable[[np.ndarray], np.ndarray],
    diffusion: Callable[[np.ndarray], np.ndarray],
    T: float,
    n_steps: int,
    n_paths: int,
    n_iter: int = 5,
    seed: int | None = None,
) -> np.ndarray:
    """Full path simulation with implicit (drift-implicit) Euler.

    Particularly useful for stiff SDEs where explicit Euler diverges
    (e.g. CIR with large κ, mean-reverting models near zero).

    Returns:
        (n_paths, n_steps+1) array of paths.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0

    for i in range(n_steps):
        dW = rng.standard_normal(n_paths) * sqrt_dt
        paths[:, i + 1] = implicit_euler_step(
            paths[:, i], drift, diffusion, dt, dW, n_iter,
        )

    return paths
