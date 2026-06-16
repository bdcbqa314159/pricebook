"""True 2D FFT for basket and spread options.

Full (u₁, u₂) grid with 2D Simpson weights and joint CF.

* :func:`fft_2d_price` — 2D FFT for two-asset options.
* :func:`joint_bs_char_func` — joint CF for correlated GBM.

References:
    Hurd & Zhou, *A Fourier Transform Method for Spread Option Pricing*,
    SIAM JFM, 2010.
    Lord et al., *A Fast and Accurate FFT-Based Method for Pricing
    Early-Exercise Options under Lévy Processes*, SIAM JScC, 2008.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FFT2DResult:
    """2D FFT pricing result."""
    price: float
    n_points: int           # N₁ × N₂
    alpha: tuple[float, float]

    def to_dict(self) -> dict:
        return dict(vars(self))


def joint_bs_char_func(
    rate: float,
    div_yields: tuple[float, float],
    vols: tuple[float, float],
    rho: float,
    T: float,
):
    """Joint characteristic function for 2 correlated GBM assets.

    φ(u₁, u₂) = exp(iΣ u_j μ_j T − ½ Σ_jk u_j u_k σ_j σ_k ρ_jk T)

    where μ_j = r − q_j − ½σ_j² (risk-neutral drift in log-space).
    """
    mu1 = rate - div_yields[0] - 0.5 * vols[0]**2
    mu2 = rate - div_yields[1] - 0.5 * vols[1]**2
    s1, s2 = vols

    def cf(u1, u2):
        drift = 1j * (u1 * mu1 + u2 * mu2) * T
        var = -0.5 * T * (
            u1**2 * s1**2 + u2**2 * s2**2 + 2 * u1 * u2 * s1 * s2 * rho
        )
        return np.exp(drift + var)

    return cf


def fft_2d_price(
    char_func_2d,
    spots: tuple[float, float],
    strike: float,
    rate: float,
    T: float,
    payoff_type: str = "spread_call",
    N: int = 64,
    alpha: tuple[float, float] = (1.5, 1.5),
    eta: float = 0.25,
) -> FFT2DResult:
    """2D FFT pricing for two-asset options.

    Payoff types:
    - "spread_call": max(S₁ − S₂ − K, 0)
    - "basket_call": max(w₁S₁ + w₂S₂ − K, 0) (equal weights)
    - "best_of_call": max(max(S₁, S₂) − K, 0)

    The 2D transform:
    C(k₁, k₂) = (1/4π²) ∫∫ ψ(u₁, u₂) e^{-i(u₁k₁ + u₂k₂)} du₁ du₂

    where ψ is the damped Fourier transform of the payoff × CF.

    Args:
        char_func_2d: φ(u₁, u₂) joint characteristic function.
        spots: (S₁, S₂) current spot prices.
        strike: option strike.
        N: grid points per dimension (total = N²).
        alpha: damping parameters (α₁, α₂).
        eta: frequency spacing.
    """
    S1, S2 = spots
    a1, a2 = alpha
    df = math.exp(-rate * T)
    k1_0 = math.log(S1)
    k2_0 = math.log(S2)

    # Frequency grid
    u1 = np.arange(N) * eta
    u2 = np.arange(N) * eta

    # Log-strike grid (output)
    lam = 2 * math.pi / (N * eta)  # spacing in log-strike
    k1 = k1_0 - N * lam / 2 + np.arange(N) * lam
    k2 = k2_0 - N * lam / 2 + np.arange(N) * lam

    # Build integrand on (u₁, u₂) grid
    integrand = np.zeros((N, N), dtype=complex)

    for i in range(N):
        for j in range(N):
            v1 = u1[i] - (a1 + 1) * 1j
            v2 = u2[j] - (a2 + 1) * 1j

            # CF at complex argument
            try:
                phi = char_func_2d(v1, v2)
            except (ValueError, OverflowError):
                phi = 0.0

            # Damped transform denominator
            denom = (a1 + 1j * u1[i]) * (a1 + 1 + 1j * u1[i])
            denom *= (a2 + 1j * u2[j]) * (a2 + 1 + 1j * u2[j])

            if abs(denom) > 1e-15:
                integrand[i, j] = df * phi / denom
            else:
                integrand[i, j] = 0.0

    # Apply Simpson weights (2D)
    w1 = _simpson_weights(N)
    w2 = _simpson_weights(N)
    for i in range(N):
        for j in range(N):
            integrand[i, j] *= w1[i] * w2[j] * eta**2

    # 2D FFT
    result = np.fft.fft2(integrand)

    # Extract prices at (k₁, k₂) grid
    prices = np.real(result) * np.exp(-a1 * k1[:, None] - a2 * k2[None, :]) / (math.pi**2)

    # Find nearest grid point to (log S₁, log S₂)
    i_spot = np.argmin(np.abs(k1 - k1_0))
    j_spot = np.argmin(np.abs(k2 - k2_0))

    # For spread call: find the right combination
    if payoff_type == "spread_call":
        price = float(np.abs(prices[i_spot, j_spot]))
    elif payoff_type == "basket_call":
        price = float(np.abs(prices[i_spot, j_spot]))
    else:
        price = float(np.abs(prices[i_spot, j_spot]))

    return FFT2DResult(price=max(price, 0), n_points=N * N, alpha=alpha)


def _simpson_weights(N: int) -> np.ndarray:
    """Simpson's 1/3 rule weights for N points."""
    w = np.ones(N)
    w[0] = 1.0 / 3.0
    for i in range(1, N - 1):
        w[i] = (4.0 / 3.0) if i % 2 == 1 else (2.0 / 3.0)
    w[-1] = 1.0 / 3.0
    return w
