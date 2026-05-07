"""Standard stochastic processes for the MC engine.

Each function returns a ProcessSpec that plugs into MCEngine.

    from pricebook.mc_processes import GBMProcess, HestonProcess, OUProcess

    engine = MCEngine(process=GBMProcess(100, 0.05, 0.20), ...)
"""

from __future__ import annotations

import numpy as np

from pricebook.mc_engine import ProcessSpec


def GBMProcess(s0: float, mu: float, sigma: float) -> ProcessSpec:
    """Geometric Brownian Motion: dS = μS dt + σS dW.

    Log-exact Euler: uses log(S) dynamics for better accuracy.
    X = log(S), dX = (μ - σ²/2) dt + σ dW, S = exp(X).
    """
    def drift(x, t):
        return (mu - 0.5 * sigma ** 2) * np.ones_like(x)

    def diffusion(x, t):
        return sigma * np.ones_like(x)

    def exact_step(x, t, dt, dw):
        # Exact log-normal step: X_{t+dt} = X_t + (μ - σ²/2)dt + σ√dt Z
        return x + (mu - 0.5 * sigma ** 2) * dt + sigma * dw

    return ProcessSpec(
        x0=np.log(s0),  # work in log-space
        drift=drift,
        diffusion=diffusion,
        n_factors=1,
        exact_step=exact_step,
    )


def BlackScholesProcess(s0: float, r: float, sigma: float, q: float = 0.0) -> ProcessSpec:
    """Risk-neutral GBM: dS = (r-q)S dt + σS dW (for pricing)."""
    return GBMProcess(s0, r - q, sigma)


def HestonProcess(
    s0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> ProcessSpec:
    """Heston stochastic volatility: dS = μS dt + √v S dW_1,
                                      dv = κ(θ-v) dt + ξ√v dW_2.

    2-factor: X = [log(S), v]. Correlation ρ between W_1 and W_2.
    Uses full-truncation Euler for v (clamp at 0).
    """
    def drift(x, t):
        v = np.maximum(x[..., 1], 0.0)
        d = np.zeros_like(x)
        d[..., 0] = mu - 0.5 * v              # log(S) drift
        d[..., 1] = kappa * (theta - v)        # variance drift
        return d

    def diffusion(x, t):
        v = np.maximum(x[..., 1], 0.0)
        sqrt_v = np.sqrt(v)
        d = np.zeros_like(x)
        d[..., 0] = sqrt_v                     # log(S) vol
        d[..., 1] = xi * sqrt_v                # vol-of-vol
        return d

    def exact_step(x, t, dt, dw):
        v = np.maximum(x[..., 1], 0.0)
        sqrt_v = np.sqrt(v)

        new_x = np.zeros_like(x)
        # Log-price
        new_x[..., 0] = x[..., 0] + (mu - 0.5 * v) * dt + sqrt_v * dw[..., 0]
        # Variance (full truncation)
        new_x[..., 1] = v + kappa * (theta - v) * dt + xi * sqrt_v * dw[..., 1]
        new_x[..., 1] = np.maximum(new_x[..., 1], 0.0)
        return new_x

    corr = np.array([[1.0, rho], [rho, 1.0]])

    return ProcessSpec(
        x0=np.array([np.log(s0), v0]),
        drift=drift,
        diffusion=diffusion,
        n_factors=2,
        correlation=corr,
        exact_step=exact_step,
    )


def OUProcess(x0: float, kappa: float, theta: float, sigma: float) -> ProcessSpec:
    """Ornstein-Uhlenbeck: dX = κ(θ-X) dt + σ dW.

    Mean-reverting process for rates, spreads, vol.
    """
    def drift(x, t):
        return kappa * (theta - x)

    def diffusion(x, t):
        return sigma * np.ones_like(x)

    return ProcessSpec(x0=x0, drift=drift, diffusion=diffusion, n_factors=1)


def CIRProcess(x0: float, kappa: float, theta: float, sigma: float) -> ProcessSpec:
    """Cox-Ingersoll-Ross: dX = κ(θ-X) dt + σ√X dW.

    Non-negative mean-reverting (rates, intensity). Full truncation.
    Feller condition: 2κθ > σ² ensures positivity.
    """
    def drift(x, t):
        return kappa * (theta - np.maximum(x, 0.0))

    def diffusion(x, t):
        return sigma * np.sqrt(np.maximum(x, 0.0))

    def exact_step(x, t, dt, dw):
        xp = np.maximum(x, 0.0)
        new_x = xp + kappa * (theta - xp) * dt + sigma * np.sqrt(xp) * dw
        return np.maximum(new_x, 0.0)

    return ProcessSpec(x0=x0, drift=drift, diffusion=diffusion, n_factors=1,
                       exact_step=exact_step)


def JumpDiffusionProcess(
    s0: float,
    mu: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
) -> ProcessSpec:
    """Merton jump-diffusion: dS/S = (μ-λk) dt + σ dW + J dN.

    J ~ N(jump_mean, jump_vol²), N ~ Poisson(λ).
    k = E[e^J - 1] = exp(jump_mean + jump_vol²/2) - 1.
    """
    k = np.exp(jump_mean + 0.5 * jump_vol ** 2) - 1
    drift_adj = mu - jump_intensity * k

    def exact_step(x, t, dt, dw):
        # Diffusion part (log-space)
        new_x = x + (drift_adj - 0.5 * sigma ** 2) * dt + sigma * dw

        # Jump part: sample number of jumps, then jump sizes
        # Note: dw is from the engine's RNG; we need a separate source for jumps.
        # For simplicity, use a deterministic approximation:
        # average jump contribution per step
        n_jumps = np.random.poisson(jump_intensity * dt, size=x.shape)
        jump_sizes = n_jumps * jump_mean + np.sqrt(n_jumps.astype(float)) * jump_vol * np.random.randn(*x.shape)
        new_x += jump_sizes

        return new_x

    return ProcessSpec(
        x0=np.log(s0),
        drift=lambda x, t: (drift_adj - 0.5 * sigma ** 2) * np.ones_like(x),
        diffusion=lambda x, t: sigma * np.ones_like(x),
        n_factors=1,
        exact_step=exact_step,
    )


def CorrelatedGBMProcess(
    s0: list[float],
    mu: list[float],
    sigma: list[float],
    correlation: np.ndarray,
) -> ProcessSpec:
    """N-asset correlated GBM for basket/multi-asset pricing.

    Each asset: dS_i = μ_i S_i dt + σ_i S_i dW_i, with corr(dW_i, dW_j) = ρ_ij.
    Works in log-space for each asset.
    """
    n = len(s0)
    s0_arr = np.array(s0)
    mu_arr = np.array(mu)
    sigma_arr = np.array(sigma)

    def exact_step(x, t, dt, dw):
        new_x = np.zeros_like(x)
        for i in range(n):
            new_x[..., i] = x[..., i] + (mu_arr[i] - 0.5 * sigma_arr[i] ** 2) * dt + sigma_arr[i] * dw[..., i]
        return new_x

    return ProcessSpec(
        x0=np.log(s0_arr),
        drift=lambda x, t: np.broadcast_to(mu_arr - 0.5 * sigma_arr ** 2, x.shape),
        diffusion=lambda x, t: np.broadcast_to(sigma_arr, x.shape),
        n_factors=n,
        correlation=correlation,
        exact_step=exact_step,
    )
