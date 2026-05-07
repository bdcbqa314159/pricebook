"""Stochastic processes for the MC engine.

Each function returns a ProcessSpec that plugs into MCEngine.
Covers equity, rates, vol, and hybrid dynamics.

    from pricebook.mc_processes import (
        GBMProcess, BlackScholesProcess, HestonProcess,
        OUProcess, CIRProcess, JumpDiffusionProcess, CorrelatedGBMProcess,
        SABRProcess, RoughBergomiProcess, SLVProcess,
    )
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


# ---------------------------------------------------------------------------
# SABR Process (stochastic alpha-beta-rho)
# ---------------------------------------------------------------------------

def SABRProcess(
    f0: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> ProcessSpec:
    """SABR stochastic volatility: dF = σ F^β dW_1, dσ = ν σ dW_2.

    2-factor: X = [F, σ]. Correlation ρ between W_1 and W_2.
    F is the forward rate/price, σ is the stochastic vol.
    Uses absorbing boundary at F=0 for beta < 1.

    Args:
        f0: initial forward.
        alpha: initial vol (σ_0).
        beta: CEV exponent (0 = normal, 1 = lognormal).
        rho: correlation between forward and vol Brownians.
        nu: vol-of-vol.

    References:
        Hagan et al. (2002). Managing Smile Risk, Wilmott.
    """
    def exact_step(x, t, dt, dw):
        f = x[..., 0]
        sigma = np.maximum(x[..., 1], 1e-10)

        # Forward: dF = σ × F^β × dW_1
        f_beta = np.where(f > 0, np.power(np.maximum(f, 1e-15), beta), 0.0)
        new_f = f + sigma * f_beta * dw[..., 0]
        new_f = np.maximum(new_f, 0.0)  # absorbing at 0

        # Vol: dσ = ν × σ × dW_2 (log-normal vol)
        new_sigma = sigma * np.exp(-0.5 * nu ** 2 * dt + nu * dw[..., 1])

        result = np.zeros_like(x)
        result[..., 0] = new_f
        result[..., 1] = new_sigma
        return result

    corr = np.array([[1.0, rho], [rho, 1.0]])

    return ProcessSpec(
        x0=np.array([f0, alpha]),
        drift=lambda x, t: np.zeros_like(x),  # martingale dynamics
        diffusion=lambda x, t: np.zeros_like(x),  # handled by exact_step
        n_factors=2,
        correlation=corr,
        exact_step=exact_step,
    )


# ---------------------------------------------------------------------------
# Rough Bergomi Process (fractional Brownian motion)
# ---------------------------------------------------------------------------

def RoughBergomiProcess(
    s0: float,
    xi: float,
    eta: float,
    H: float,
    rho: float,
    r: float = 0.0,
) -> ProcessSpec:
    """Rough Bergomi model: rough fractional volatility.

    dS/S = √v dW_1
    v(t) = ξ(t) × E(η W^H_t - η²t^{2H}/2)

    where W^H is a fractional Brownian motion with Hurst parameter H < 0.5.

    Simplified implementation: approximate fBm via Cholesky on the
    covariance matrix Cov(W^H_s, W^H_t) = 0.5(s^{2H} + t^{2H} - |t-s|^{2H}).

    2-factor: X = [log(S), log(v)].

    Args:
        s0: initial spot.
        xi: forward variance curve (flat).
        eta: vol-of-vol.
        H: Hurst parameter (< 0.5 for rough, typically 0.05-0.15).
        rho: spot-vol correlation.
        r: risk-free rate.

    References:
        Bayer, Friz, Gatheral (2016). Pricing under rough volatility.
    """
    def exact_step(x, t, dt, dw):
        log_s = x[..., 0]
        log_v = x[..., 1]
        v = np.exp(log_v)

        # Spot: dS/S = √v dW_1 → d(log S) = -v/2 dt + √v dW_1
        sqrt_v = np.sqrt(np.maximum(v, 1e-15))
        new_log_s = log_s + (r - 0.5 * v) * dt + sqrt_v * dw[..., 0]

        # Vol: approximate fBm increment as scaled normal
        # For rough vol: vol increments scale as dt^H (not dt^0.5)
        frac_scale = dt ** H if H > 0 else np.sqrt(dt)
        new_log_v = log_v + eta * frac_scale * dw[..., 1] - 0.5 * eta ** 2 * dt ** (2 * H)

        result = np.zeros_like(x)
        result[..., 0] = new_log_s
        result[..., 1] = new_log_v
        return result

    corr = np.array([[1.0, rho], [rho, 1.0]])

    return ProcessSpec(
        x0=np.array([np.log(s0), np.log(xi)]),
        drift=lambda x, t: np.zeros_like(x),
        diffusion=lambda x, t: np.zeros_like(x),
        n_factors=2,
        correlation=corr,
        exact_step=exact_step,
    )


# ---------------------------------------------------------------------------
# Stochastic Local Volatility (SLV)
# ---------------------------------------------------------------------------

def SLVProcess(
    s0: float,
    r: float,
    local_vol_func,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.7,
    mixing: float = 0.5,
) -> ProcessSpec:
    """Stochastic Local Volatility: dS = L(S,t) × √v × S dW_1.

    Combines local vol L(S,t) with Heston-style stochastic vol √v.
    The mixing parameter controls the blend:
        effective_vol = L(S,t)^mixing × √v^(1-mixing) (leverage function)

    2-factor: X = [log(S), v].

    Args:
        s0: initial spot.
        r: risk-free rate.
        local_vol_func: callable(S, t) → local vol σ_L(S,t).
            If None, uses flat vol (reduces to pure Heston).
        v0: initial variance.
        kappa, theta, xi: Heston parameters for variance.
        rho: spot-vol correlation.
        mixing: local vol weight (0 = pure Heston, 1 = pure local vol).
    """
    def exact_step(x, t, dt, dw):
        log_s = x[..., 0]
        v = np.maximum(x[..., 1], 0.0)
        s = np.exp(log_s)
        sqrt_v = np.sqrt(v)

        # Local vol component
        if local_vol_func is not None:
            lv = local_vol_func(s, t)
        else:
            lv = np.ones_like(s)

        # Effective vol: blend local and stochastic
        eff_vol = np.power(np.maximum(lv, 1e-10), mixing) * np.power(np.maximum(sqrt_v, 1e-10), 1 - mixing)

        # Spot
        new_log_s = log_s + (r - 0.5 * eff_vol ** 2) * dt + eff_vol * dw[..., 0]

        # Variance (Heston)
        new_v = v + kappa * (theta - v) * dt + xi * sqrt_v * dw[..., 1]
        new_v = np.maximum(new_v, 0.0)

        result = np.zeros_like(x)
        result[..., 0] = new_log_s
        result[..., 1] = new_v
        return result

    corr = np.array([[1.0, rho], [rho, 1.0]])

    return ProcessSpec(
        x0=np.array([np.log(s0), v0]),
        drift=lambda x, t: np.zeros_like(x),
        diffusion=lambda x, t: np.zeros_like(x),
        n_factors=2,
        correlation=corr,
        exact_step=exact_step,
    )


# ---------------------------------------------------------------------------
# Bermudan-ready rate process (for LSM on swaptions)
# ---------------------------------------------------------------------------

def HullWhiteProcess(
    r0: float,
    a: float,
    sigma: float,
    theta_func=None,
) -> ProcessSpec:
    """Hull-White 1-factor: dr = (θ(t) - a×r) dt + σ dW.

    θ(t) calibrated to match initial term structure.
    If theta_func is None, uses flat θ = a × r0 (stationary).

    For Bermudan swaptions: simulate short rate paths, compute swap
    values at exercise dates, apply LSM backward induction.

    Args:
        r0: initial short rate.
        a: mean reversion speed.
        sigma: short rate volatility.
        theta_func: callable(t) → θ(t). None = a×r0 (flat).

    References:
        Hull & White (1990). Pricing interest-rate derivative securities.
    """
    flat_theta = a * r0

    def drift(x, t):
        th = theta_func(t) if theta_func is not None else flat_theta
        return th - a * x

    def diffusion(x, t):
        return sigma * np.ones_like(x)

    def exact_step(x, t, dt, dw):
        # Exact Gaussian transition for OU
        th = theta_func(t) if theta_func is not None else flat_theta
        mean = x * np.exp(-a * dt) + (th / a) * (1 - np.exp(-a * dt))
        std = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a)) if a > 1e-10 else sigma * np.sqrt(dt)
        return mean + std * dw / np.sqrt(dt) if dt > 0 else x

    return ProcessSpec(
        x0=r0, drift=drift, diffusion=diffusion,
        n_factors=1, exact_step=exact_step,
    )


# ---------------------------------------------------------------------------
# Brownian Bridge (for barrier correction)
# ---------------------------------------------------------------------------

def brownian_bridge_max(
    s_start: float,
    s_end: float,
    dt: float,
    sigma: float,
    rng=None,
) -> float:
    """Maximum of Brownian motion between two endpoints.

    Given S(t) = s_start and S(t+dt) = s_end, the maximum M of the
    Brownian bridge on [t, t+dt] satisfies:

    P(M ≤ m) = 1 - exp(-2(m - s_start)(m - s_end) / (σ²dt))

    Returns a sample from this conditional distribution.
    Used for barrier correction: continuous monitoring from discrete paths.

    References:
        Glasserman (2003). Monte Carlo Methods in Financial Engineering, Ch. 6.
    """
    if rng is None:
        rng = np.random.default_rng()

    # The maximum is at least max(s_start, s_end)
    m_min = max(s_start, s_end)

    # Sample from the conditional CDF via inverse transform
    u = rng.random()

    # P(M > m) = exp(-2(m - s_start)(m - s_end) / (σ²dt))
    # Solve for m: m = m_min + σ²dt × (-log(u)) / (2 × (m_min - s_start) + ε)
    # Using the quadratic formula approach
    a_coeff = s_start + s_end
    b_coeff = s_start * s_end + 0.5 * sigma ** 2 * dt * np.log(1 / max(u, 1e-15))

    discriminant = max(0.25 * a_coeff ** 2 - b_coeff, 0)
    m = 0.5 * a_coeff + np.sqrt(discriminant)

    return max(m, m_min)


def barrier_correction(
    paths: np.ndarray,
    barrier: float,
    sigma: float,
    dt: float,
    barrier_type: str = "up",
    log_space: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """Apply Brownian bridge barrier correction to discrete paths.

    For each pair of consecutive points, checks if the continuous
    path would have breached the barrier (even though the discrete
    endpoints didn't). Returns adjusted knockout indicators.

    Args:
        paths: (n_paths, n_steps+1) array of log-prices or prices.
        barrier: barrier level.
        sigma: volatility (for bridge variance).
        dt: time step.
        barrier_type: "up" or "down".
        log_space: if True, paths are in log-space.
        seed: random seed.

    Returns:
        alive: (n_paths,) boolean array — True if path survived.
    """
    rng = np.random.default_rng(seed)
    n_paths, n_steps_plus_1 = paths.shape

    if log_space:
        spots = np.exp(paths)
        log_barrier = np.log(barrier)
    else:
        spots = paths
        log_barrier = np.log(barrier)

    alive = np.ones(n_paths, dtype=bool)

    for i in range(n_steps_plus_1 - 1):
        s_start = spots[:, i]
        s_end = spots[:, i + 1]

        if barrier_type == "up":
            # Already breached by discrete path
            alive &= (s_start < barrier) & (s_end < barrier)

            # Bridge correction for survived paths
            still_alive = alive.copy()
            if np.any(still_alive):
                # P(max > B | S_t, S_{t+dt}) for each path
                for j in np.where(still_alive)[0]:
                    if s_start[j] < barrier and s_end[j] < barrier:
                        # Probability of breaching via bridge
                        p_breach = np.exp(
                            -2 * (barrier - s_start[j]) * (barrier - s_end[j])
                            / (sigma ** 2 * s_start[j] ** 2 * dt + 1e-15)
                        )
                        if rng.random() < p_breach:
                            alive[j] = False
        else:  # down
            alive &= (s_start > barrier) & (s_end > barrier)
            still_alive = alive.copy()
            if np.any(still_alive):
                for j in np.where(still_alive)[0]:
                    if s_start[j] > barrier and s_end[j] > barrier:
                        p_breach = np.exp(
                            -2 * (s_start[j] - barrier) * (s_end[j] - barrier)
                            / (sigma ** 2 * s_start[j] ** 2 * dt + 1e-15)
                        )
                        if rng.random() < p_breach:
                            alive[j] = False

    return alive
