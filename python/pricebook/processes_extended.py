"""Extended stochastic processes: CEV, 3/2 model, Kou, Bates, Hawkes.

Fills the gaps in the process library identified in the numerical audit.
Each process provides path simulation and, where available, a
characteristic function for Fourier pricing.

Phase M5 slices 188-190 consolidated (share this file).

* :class:`CEVProcess` — dS = μS dt + σS^β dW.
* :class:`ThreeHalvesProcess` — dv = κv(θ−v) dt + ε v^{3/2} dW.
* :class:`KouJumpDiffusion` — GBM + double-exponential jumps.
* :class:`BatesModel` — Heston + Merton jumps.
* :class:`HawkesProcess` — self-exciting point process.
* :func:`vg_full_paths` — Variance Gamma full path simulation.

References:
    Cox, *Notes on Option Pricing I: CEV*, 1975.
    Heston, *A Closed-Form Solution for Options with Stochastic Vol*, 1993.
    Kou, *A Jump-Diffusion Model for Option Pricing*, Management Sci., 2002.
    Bates, *Jumps and Stochastic Volatility*, RFS, 1996.
    Hawkes, *Spectra of Some Self-Exciting Point Processes*, Biometrika, 1971.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- CEV ----

@dataclass
class CEVResult:
    paths: np.ndarray
    times: np.ndarray


def cev_paths(
    spot: float,
    rate: float,
    vol: float,
    beta: float,
    T: float,
    n_steps: int,
    n_paths: int,
    div_yield: float = 0.0,
    seed: int | None = None,
) -> CEVResult:
    """CEV process: dS = (r−q)S dt + σ S^β dW.

    β = 1 → GBM, β = 0.5 → square-root diffusion on S,
    β < 1 → fatter left tail (negative skew).
    Uses Euler-Maruyama with absorption at S = 0.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    mu = rate - div_yield

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = spot

    for i in range(n_steps):
        S = paths[:, i]
        S_safe = np.maximum(S, 1e-10)
        dW = rng.standard_normal(n_paths) * sqrt_dt
        paths[:, i + 1] = S + mu * S * dt + vol * np.power(S_safe, beta) * dW
        paths[:, i + 1] = np.maximum(paths[:, i + 1], 0.0)

    return CEVResult(paths, np.linspace(0, T, n_steps + 1))


# ---- 3/2 model ----

@dataclass
class ThreeHalvesResult:
    paths: np.ndarray
    times: np.ndarray


def three_halves_paths(
    v0: float,
    kappa: float,
    theta: float,
    epsilon: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> ThreeHalvesResult:
    """3/2 stochastic vol model: dv = κv(θ − v) dt + ε v^{3/2} dW.

    Vol-of-vol explodes as v rises — better for commodities and
    high-vol regimes than Heston.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = v0

    for i in range(n_steps):
        v = np.maximum(paths[:, i], 1e-10)
        dW = rng.standard_normal(n_paths) * sqrt_dt
        paths[:, i + 1] = v + kappa * v * (theta - v) * dt + epsilon * np.power(v, 1.5) * dW
        paths[:, i + 1] = np.maximum(paths[:, i + 1], 0.0)

    return ThreeHalvesResult(paths, np.linspace(0, T, n_steps + 1))


# ---- Kou double-exponential jump-diffusion ----

@dataclass
class KouResult:
    paths: np.ndarray
    times: np.ndarray
    n_jumps: np.ndarray  # total jumps per path


def kou_paths(
    spot: float,
    rate: float,
    vol: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
    T: float,
    n_steps: int,
    n_paths: int,
    div_yield: float = 0.0,
    seed: int | None = None,
) -> KouResult:
    """Kou double-exponential jump-diffusion.

    dS/S = (r − q − λk) dt + σ dW + J dN

    where J ~ p × Exp(η₁) − (1−p) × Exp(η₂) (asymmetric jumps).
    k = E[e^J − 1] = p η₁/(η₁−1) + (1−p) η₂/(η₂+1) − 1.

    Args:
        lam: jump intensity.
        p: probability of upward jump.
        eta1: rate of upward jump size (η₁ > 1 for finite expectation).
        eta2: rate of downward jump size (η₂ > 0).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    # Compensator
    k = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1
    mu = rate - div_yield - lam * k

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = spot
    total_jumps = np.zeros(n_paths)

    for i in range(n_steps):
        S = paths[:, i]
        dW = rng.standard_normal(n_paths) * sqrt_dt

        # Poisson jumps in [t, t+dt]
        n_jump = rng.poisson(lam * dt, n_paths)
        total_jumps += n_jump

        # Jump sizes: double exponential
        log_jump = np.zeros(n_paths)
        for j in range(n_paths):
            for _ in range(n_jump[j]):
                if rng.random() < p:
                    log_jump[j] += rng.exponential(1.0 / eta1)
                else:
                    log_jump[j] -= rng.exponential(1.0 / eta2)

        paths[:, i + 1] = S * np.exp(
            (mu - 0.5 * vol**2) * dt + vol * dW + log_jump
        )

    return KouResult(paths, np.linspace(0, T, n_steps + 1), total_jumps)


# ---- Bates = Heston + jumps ----

@dataclass
class BatesResult:
    S_paths: np.ndarray
    v_paths: np.ndarray
    times: np.ndarray


def bates_paths(
    spot: float,
    v0: float,
    rate: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    T: float,
    n_steps: int,
    n_paths: int,
    div_yield: float = 0.0,
    seed: int | None = None,
) -> BatesResult:
    """Bates model: Heston stochastic vol + Merton log-normal jumps.

    dS/S = (r − q − λk) dt + √v dW₁ + J dN
    dv   = κ(θ − v) dt + ξ√v dW₂
    corr(dW₁, dW₂) = ρ
    J ~ N(μ_j, σ_j²),  k = exp(μ_j + 0.5σ_j²) − 1.

    Reduces to Heston when λ = 0.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    k = math.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift_adj = rate - div_yield - lam * k

    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = spot
    v[:, 0] = v0

    for i in range(n_steps):
        v_cur = np.maximum(v[:, i], 0.0)
        sqrt_v = np.sqrt(v_cur)

        # Correlated Brownian motions
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        dW1 = z1 * sqrt_dt
        dW2 = (rho * z1 + math.sqrt(1 - rho**2) * z2) * sqrt_dt

        # Jumps
        n_jump = rng.poisson(lam * dt, n_paths)
        log_jump = np.zeros(n_paths)
        for j in range(n_paths):
            if n_jump[j] > 0:
                log_jump[j] = np.sum(rng.normal(mu_j, sigma_j, n_jump[j]))

        # Euler for variance (full truncation)
        v[:, i + 1] = v_cur + kappa * (theta - v_cur) * dt + xi * sqrt_v * dW2
        v[:, i + 1] = np.maximum(v[:, i + 1], 0.0)

        # Log-Euler for spot
        S[:, i + 1] = S[:, i] * np.exp(
            (drift_adj - 0.5 * v_cur) * dt + sqrt_v * dW1 + log_jump
        )

    return BatesResult(S, v, np.linspace(0, T, n_steps + 1))


# ---- Hawkes process ----

@dataclass
class HawkesResult:
    event_times: list[list[float]]  # per-path list of event times
    intensities: np.ndarray          # (n_paths, n_grid) sampled intensity
    times: np.ndarray


def hawkes_paths(
    mu: float,
    alpha: float,
    beta: float,
    T: float,
    n_paths: int,
    n_grid: int = 200,
    seed: int | None = None,
) -> HawkesResult:
    """Univariate Hawkes process via thinning (Ogata).

    λ(t) = μ + Σ_{t_i < t} α exp(−β(t − t_i))

    Self-exciting: each event increases the intensity by α, which
    then decays exponentially at rate β.

    Args:
        mu: baseline intensity.
        alpha: excitation magnitude per event.
        beta: decay rate of excitation.
        T: time horizon.
        n_paths: number of independent paths.
        n_grid: grid points for sampled intensity output.
        seed: random seed.
    """
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, T, n_grid)

    all_events: list[list[float]] = []
    all_intensity = np.zeros((n_paths, n_grid))

    for p in range(n_paths):
        events: list[float] = []
        t = 0.0
        lam_star = mu  # upper bound for thinning

        while t < T:
            # Next candidate via exponential with rate lam_star
            u = rng.random()
            if u == 0:
                u = 1e-15
            dt_cand = -math.log(u) / lam_star
            t += dt_cand

            if t >= T:
                break

            # Current intensity at t
            lam_t = mu + sum(
                alpha * math.exp(-beta * (t - ti)) for ti in events
            )

            # Accept/reject
            if rng.random() < lam_t / lam_star:
                events.append(t)
                lam_star = lam_t + alpha  # update upper bound
            else:
                lam_star = lam_t  # tighten bound

        all_events.append(events)

        # Sample intensity on grid
        for gi, gt in enumerate(grid):
            all_intensity[p, gi] = mu + sum(
                alpha * math.exp(-beta * (gt - ti))
                for ti in events if ti < gt
            )

    return HawkesResult(all_events, all_intensity, grid)


# ---- VG full paths ----

@dataclass
class VGPathResult:
    paths: np.ndarray
    times: np.ndarray


def vg_full_paths(
    spot: float,
    rate: float,
    sigma: float,
    theta_vg: float,
    nu: float,
    T: float,
    n_steps: int,
    n_paths: int,
    div_yield: float = 0.0,
    seed: int | None = None,
) -> VGPathResult:
    """Variance Gamma full path simulation via Gamma time change.

    X(t) = θ G(t) + σ W(G(t))

    where G(t) is a Gamma process with rate 1/ν and shape t/ν.
    The VG process is obtained by evaluating BM at Gamma-subordinated time.

    Previously only terminal values were available; this gives full paths.

    Args:
        sigma: volatility of the BM component.
        theta_vg: drift of the BM component (skewness).
        nu: variance rate of the Gamma subordinator.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Compensator for risk-neutral drift
    omega = (1.0 / nu) * math.log(1.0 - theta_vg * nu - 0.5 * sigma**2 * nu)
    mu = rate - div_yield + omega

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = spot
    log_S = np.full(n_paths, math.log(spot))

    for i in range(n_steps):
        # Gamma increment: shape = dt/ν, scale = ν
        dG = rng.gamma(dt / nu, nu, n_paths)
        # BM evaluated at Gamma time
        dW = rng.standard_normal(n_paths) * np.sqrt(dG)
        dX = theta_vg * dG + sigma * dW
        log_S = log_S + mu * dt + dX
        paths[:, i + 1] = np.exp(log_S)

    return VGPathResult(paths, np.linspace(0, T, n_steps + 1))
