"""Fokker-Planck (forward Kolmogorov) equation solver.

Evolve the risk-neutral density forward in time under GBM,
local vol, or Heston dynamics.

* :func:`fokker_planck_1d` — 1D density evolution (GBM/local vol).
* :func:`fokker_planck_density` — extract density at maturity.
* :func:`density_to_option_prices` — price options from density.

References:
    Risken, *The Fokker-Planck Equation*, Springer, 1996.
    Dupire, *Pricing with a Smile*, Risk, 1994.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# NumPy 2.x compat: trapz renamed to trapezoid
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


@dataclass
class FPResult:
    """Fokker-Planck solution result."""
    grid: np.ndarray            # spot grid
    density: np.ndarray         # probability density at T
    mean: float
    variance: float
    T: float

    def to_dict(self) -> dict:
        return {
            "n_points": len(self.grid),
            "mean": self.mean,
            "variance": self.variance,
            "T": self.T,
        }


def fokker_planck_1d(
    spot: float,
    rate: float,
    vol,
    T: float,
    div_yield: float = 0.0,
    n_space: int = 200,
    n_time: int = 200,
) -> FPResult:
    """Solve 1D Fokker-Planck equation forward in time.

    ∂p/∂t = −∂/∂x[μ(x)p] + ½∂²/∂x²[σ²(x)p]

    In log-space x = ln(S):
    ∂p/∂t = −∂/∂x[(r − q − ½σ²)p] + ½∂²/∂x²[σ²p]

    Initial condition: p(x, 0) = δ(x − ln(S₀)).

    Args:
        vol: constant float or callable(S, t) → σ for local vol.
    """
    mu = rate - div_yield

    # Log-space grid
    x0 = math.log(spot)
    if callable(vol):
        vol_approx = vol(spot, 0)
    else:
        vol_approx = float(vol)

    width = max(4 * vol_approx * math.sqrt(T), 2.0)
    x = np.linspace(x0 - width, x0 + width, n_space)
    dx = x[1] - x[0]
    dt = T / n_time

    # Initial condition: narrow Gaussian approximating delta
    sigma_init = dx * 2  # spread over ~4 grid points
    p = np.exp(-0.5 * ((x - x0) / sigma_init)**2) / (sigma_init * math.sqrt(2 * math.pi))
    p /= _trapz(p, x)  # normalise

    for step in range(n_time):
        t = step * dt

        # Compute coefficients at each grid point
        drift = np.zeros(n_space)
        diff = np.zeros(n_space)

        for i in range(n_space):
            S_i = math.exp(x[i])
            if callable(vol):
                sigma_i = vol(S_i, t)
            else:
                sigma_i = float(vol)
            drift[i] = mu - 0.5 * sigma_i**2
            diff[i] = sigma_i**2

        # Crank-Nicolson for Fokker-Planck
        # FP operator in log-space:
        # L[p] = -d/dx[drift * p] + 0.5 * d²/dx²[diff * p]

        # Explicit part
        rhs = p.copy()
        for i in range(1, n_space - 1):
            # -d/dx[drift * p]: central difference
            conv = -(drift[i + 1] * p[i + 1] - drift[i - 1] * p[i - 1]) / (2 * dx)
            # 0.5 d²/dx²[diff * p]: central second derivative
            diffusion = 0.5 * (
                diff[i + 1] * p[i + 1] - 2 * diff[i] * p[i] + diff[i - 1] * p[i - 1]
            ) / (dx**2)
            rhs[i] = p[i] + 0.5 * dt * (conv + diffusion)

        # Implicit part (simplified: just diffusion)
        # (I - 0.5 dt L_diff) p_new = rhs
        lower = np.zeros(n_space - 2)
        diag = np.zeros(n_space - 2)
        upper = np.zeros(n_space - 2)

        for i in range(1, n_space - 1):
            a = 0.5 * diff[i] / dx**2
            lower[i - 1] = -0.5 * dt * a
            diag[i - 1] = 1 + dt * diff[i] / dx**2
            upper[i - 1] = -0.5 * dt * a

        # Thomas solve
        p_new = p.copy()
        p_new[1:-1] = _thomas(lower, diag, upper, rhs[1:-1])

        # Boundary: zero density at edges
        p_new[0] = 0
        p_new[-1] = 0

        # Ensure non-negative and normalise
        p_new = np.maximum(p_new, 0)
        mass = _trapz(p_new, x)
        if mass > 1e-10:
            p_new /= mass

        p = p_new

    # Convert to physical space density
    S = np.exp(x)
    # p(x)dx = p_S(S)dS → p_S = p(x)/S
    density_S = p / S

    # Moments
    mean = float(_trapz(S * density_S, S))
    variance = float(_trapz(S**2 * density_S, S)) - mean**2

    return FPResult(grid=S, density=density_S, mean=mean, variance=variance, T=T)


def density_to_option_prices(
    fp_result: FPResult,
    strikes: list[float],
    rate: float,
    T: float,
) -> list[dict]:
    """Price European options from the risk-neutral density.

    C(K) = e^{-rT} ∫_K^∞ (S − K) p(S) dS
    P(K) = e^{-rT} ∫_0^K (K − S) p(S) dS

    Args:
        fp_result: Fokker-Planck density result.
        strikes: option strikes.
    """
    S = fp_result.grid
    p = fp_result.density
    df = math.exp(-rate * T)

    results = []
    for K in strikes:
        # Call
        mask = S > K
        call_payoff = np.where(mask, S - K, 0) * p
        call = df * float(_trapz(call_payoff, S))

        # Put
        mask = S < K
        put_payoff = np.where(mask, K - S, 0) * p
        put = df * float(_trapz(put_payoff, S))

        results.append({
            "strike": K,
            "call": max(call, 0),
            "put": max(put, 0),
        })

    return results


def _thomas(lower, diag, upper, rhs):
    n = len(diag)
    c = np.zeros(n)
    d = np.zeros(n)
    c[0] = upper[0] / diag[0] if abs(diag[0]) > 1e-15 else 0
    d[0] = rhs[0] / diag[0] if abs(diag[0]) > 1e-15 else 0
    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c[i - 1]
        if abs(denom) < 1e-15:
            c[i] = 0
            d[i] = 0
        else:
            c[i] = upper[i] / denom if i < n - 1 else 0
            d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x
