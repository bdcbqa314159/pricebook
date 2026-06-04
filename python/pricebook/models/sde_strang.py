"""Strang splitting for jump-diffusion MC simulation.

Symmetric diffusion-jump-diffusion splitting reduces the splitting
error from O(dt) to O(dt²) compared to naive Lie-Trotter.

* :func:`strang_merton_mc` — Merton jump-diffusion via Strang splitting.
* :func:`strang_bates_mc` — Bates (Heston + jumps) via Strang splitting.

References:
    Glasserman, *Monte Carlo Methods in Financial Engineering*, Ch. 6.
    Platen & Bruti-Liberati, *Numerical Solution of SDEs with Jumps*, 2010.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class StrangMCResult:
    """Strang splitting MC result."""
    terminal_values: np.ndarray
    price: float                # discounted mean payoff
    stderr: float
    n_paths: int
    n_steps: int

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "stderr": self.stderr,
            "n_paths": self.n_paths,
        }


def strang_merton_mc(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    jump_intensity: float = 0.5,
    jump_mean: float = -0.05,
    jump_vol: float = 0.10,
    is_call: bool = True,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int = 42,
) -> StrangMCResult:
    """Merton jump-diffusion via Strang splitting.

    Each time step: diffusion(dt/2) → jump(dt) → diffusion(dt/2).

    Diffusion step (exact GBM):
    S → S × exp((r − q − λk − ½σ²)dt/2 + σ√(dt/2) Z)

    Jump step:
    N ~ Poisson(λ dt), J_i ~ N(μ_J, σ_J²)
    S → S × Π exp(J_i)

    The symmetric splitting gives O(dt²) weak error.

    Args:
        jump_intensity: λ (expected jumps per year).
        jump_mean: μ_J (mean of log-jump).
        jump_vol: σ_J (std of log-jump).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half_dt = dt / 2

    # Compensated drift
    kappa = math.exp(jump_mean + 0.5 * jump_vol**2) - 1
    drift = rate - jump_intensity * kappa - 0.5 * vol**2

    log_S = np.full(n_paths, math.log(spot))

    for step in range(n_steps):
        # Step 1: diffusion half-step (exact GBM in log-space)
        Z1 = rng.standard_normal(n_paths)
        log_S += drift * half_dt + vol * math.sqrt(half_dt) * Z1

        # Step 2: jump full-step
        N_jumps = rng.poisson(jump_intensity * dt, n_paths)
        for i in range(n_paths):
            if N_jumps[i] > 0:
                jumps = rng.normal(jump_mean, jump_vol, N_jumps[i])
                log_S[i] += np.sum(jumps)

        # Step 3: diffusion half-step (exact GBM)
        Z2 = rng.standard_normal(n_paths)
        log_S += drift * half_dt + vol * math.sqrt(half_dt) * Z2

    S_T = np.exp(log_S)

    # Payoff
    if is_call:
        payoff = np.maximum(S_T - strike, 0)
    else:
        payoff = np.maximum(strike - S_T, 0)

    df = math.exp(-rate * T)
    discounted = payoff * df
    price = float(np.mean(discounted))
    stderr = float(np.std(discounted, ddof=1) / math.sqrt(n_paths))

    return StrangMCResult(S_T, price, stderr, n_paths, n_steps)


def strang_bates_mc(
    spot: float,
    strike: float,
    rate: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    jump_intensity: float = 0.5,
    jump_mean: float = -0.05,
    jump_vol: float = 0.10,
    is_call: bool = True,
    div_yield: float = 0.0,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int = 42,
) -> StrangMCResult:
    """Bates (Heston + jumps) via Strang splitting.

    diffusion(dt/2): Heston step on (log S, v)
    jump(dt): Poisson jumps on log S
    diffusion(dt/2): Heston step on (log S, v)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half_dt = dt / 2

    kappa_j = math.exp(jump_mean + 0.5 * jump_vol**2) - 1
    mu = rate - div_yield - jump_intensity * kappa_j

    log_S = np.full(n_paths, math.log(spot))
    v_arr = np.full(n_paths, v0)

    for step in range(n_steps):
        # Step 1: Heston diffusion half-step
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        W1 = Z1
        W2 = rho * Z1 + math.sqrt(1 - rho**2) * Z2

        v_pos = np.maximum(v_arr, 0)
        sqrt_v = np.sqrt(v_pos)

        log_S += (mu - 0.5 * v_pos) * half_dt + sqrt_v * math.sqrt(half_dt) * W1
        v_arr += kappa * (theta - v_pos) * half_dt + xi * sqrt_v * math.sqrt(half_dt) * W2
        v_arr = np.maximum(v_arr, 0)

        # Step 2: jump full-step
        N_jumps = rng.poisson(jump_intensity * dt, n_paths)
        for i in range(n_paths):
            if N_jumps[i] > 0:
                jumps = rng.normal(jump_mean, jump_vol, N_jumps[i])
                log_S[i] += np.sum(jumps)

        # Step 3: Heston diffusion half-step
        Z3 = rng.standard_normal(n_paths)
        Z4 = rng.standard_normal(n_paths)
        W3 = Z3
        W4 = rho * Z3 + math.sqrt(1 - rho**2) * Z4

        v_pos = np.maximum(v_arr, 0)
        sqrt_v = np.sqrt(v_pos)

        log_S += (mu - 0.5 * v_pos) * half_dt + sqrt_v * math.sqrt(half_dt) * W3
        v_arr += kappa * (theta - v_pos) * half_dt + xi * sqrt_v * math.sqrt(half_dt) * W4
        v_arr = np.maximum(v_arr, 0)

    S_T = np.exp(log_S)
    if is_call:
        payoff = np.maximum(S_T - strike, 0)
    else:
        payoff = np.maximum(strike - S_T, 0)

    df = math.exp(-rate * T)
    discounted = payoff * df
    price = float(np.mean(discounted))
    stderr = float(np.std(discounted, ddof=1) / math.sqrt(n_paths))

    return StrangMCResult(S_T, price, stderr, n_paths, n_steps)
