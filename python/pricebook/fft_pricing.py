"""FFT-based option pricing: Carr-Madan, Lewis contour, density recovery.

Prices European options at many strikes simultaneously using the Fast
Fourier Transform. Complements the COS method (which prices one strike
at a time) with O(N log N) pricing across a full strike grid.

* :func:`carr_madan_fft` — Carr & Madan (1999) FFT pricing.
* :func:`lewis_price` — Lewis (2001) contour integral.
* :func:`density_from_calls` — Breeden-Litzenberger: d²C/dK² = RN density.
* :func:`density_from_cf` — inverse Fourier of characteristic function.

References:
    Carr & Madan, *Option Valuation Using the FFT*, J. Comp. Finance, 1999.
    Lewis, *A Simple Option Formula for General Jump-Diffusion*, 2001.
    Breeden & Litzenberger, *Prices of State-Contingent Claims*, 1978.
"""

from __future__ import annotations

import math
import cmath
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---- Carr-Madan FFT ----

@dataclass
class FFTResult:
    """Result of FFT-based option pricing."""
    strikes: np.ndarray     # log-strike grid
    prices: np.ndarray      # call prices at each strike
    n_points: int
    alpha: float            # damping parameter


def carr_madan_fft(
    char_func: Callable[[float], complex],
    spot: float,
    rate: float,
    T: float,
    N: int = 4096,
    alpha: float = 1.5,
    eta: float = 0.25,
    div_yield: float = 0.0,
) -> FFTResult:
    """Carr-Madan FFT pricing: call prices at N strikes in one pass.

    The modified call price c_T(k) = exp(αk) C(K) has Fourier transform:

        ψ_T(v) = exp(-rT) φ_T(v − (α+1)i) / (α² + α − v² + iv(2α+1))

    where φ_T is the characteristic function of log(S_T/S_0) and
    k = log(K/S_0).

    The inverse FFT recovers c_T(k) at a grid of log-strikes.

    Args:
        char_func: φ(u) = E[exp(iu log(S_T/S_0))] under risk-neutral measure.
        spot: current spot.
        rate: risk-free rate.
        T: time to maturity.
        N: number of FFT points (power of 2 for efficiency).
        alpha: damping parameter (must be > 0; typical 1.0–2.0).
        eta: spacing in the frequency domain.
        div_yield: continuous dividend yield.

    Returns:
        :class:`FFTResult` with strikes and call prices.

    Reference:
        Carr & Madan, J. Comp. Finance 2(4), 1999, Eq. (6)–(11).
    """
    df = math.exp(-rate * T)
    lam = 2 * math.pi / (N * eta)  # log-strike spacing
    b = N * lam / 2                 # log-strike range: [-b, b]

    # Frequency grid
    v = np.arange(N) * eta

    # Build the integrand ψ_T(v)
    # The char func MUST accept complex arguments for Carr-Madan:
    # we evaluate φ(v − (α+1)i) which requires imaginary shift.
    psi = np.zeros(N, dtype=complex)
    for j in range(N):
        vj = v[j]
        u = complex(vj, -(alpha + 1))
        phi = char_func(u)
        denom = alpha * alpha + alpha - vj * vj + 1j * vj * (2 * alpha + 1)
        if abs(denom) < 1e-30:
            psi[j] = 0.0
        else:
            psi[j] = df * phi / denom

    # Simpson weights for numerical integration
    simpson = 3 + (-1) ** np.arange(1, N + 1)
    simpson[0] = 1
    simpson = simpson * eta / 3.0

    # FFT input
    x = np.exp(1j * b * v) * psi * simpson

    # FFT
    y = np.fft.fft(x)

    # Log-strike grid
    k = -b + lam * np.arange(N)

    # Call prices: undo the damping.
    # Our CF is for log(S_T/S_0), but Carr-Madan is stated for log(S_T).
    # The log-moneyness shift means we need to scale by spot.
    calls = spot * np.exp(-alpha * k) / math.pi * y.real

    # Convert log-moneyness to actual strikes
    strikes = spot * np.exp(k)

    return FFTResult(
        strikes=strikes,
        prices=calls,
        n_points=N,
        alpha=alpha,
    )


def _cf_complex(
    char_func: Callable[[float], complex],
    u: complex,
) -> complex:
    """Evaluate characteristic function at a complex argument.

    Many char funcs accept complex u directly. If not, we use the
    relationship φ(a + bi) = E[exp(i(a+bi)X)] = E[exp(-bX) exp(iaX)].
    For standard implementations that accept float, we call with the
    real part and multiply by the imaginary correction.

    This wrapper tries the direct call first.
    """
    try:
        return char_func(u)
    except (TypeError, ValueError):
        # Fallback: not all char funcs accept complex
        # This is a limitation — user should provide complex-capable CF
        return char_func(float(u.real))


# ---- Lewis contour integral ----

def lewis_price(
    char_func: Callable[[complex], complex],
    spot: float,
    strike: float,
    rate: float,
    T: float,
    div_yield: float = 0.0,
    N: int = 256,
    u_max: float = 50.0,
) -> float:
    """Lewis (2001) option pricing via contour integration.

    Integrates along the line Im(u) = 1/2 in the complex plane:

        C = S − (K e^{-rT} / π) Re ∫₀^∞ exp(-iu k) φ(u − i/2) / (u² + 1/4) du

    where k = log(S/K) + (r−q)T.

    More numerically stable than Carr-Madan for some characteristic
    functions (avoids the damping parameter choice).

    Args:
        char_func: must accept complex arguments.
        N: number of quadrature points.
        u_max: upper integration limit.
    """
    df = math.exp(-rate * T)
    log_K = math.log(strike)

    # Gil-Pelaez (1951) inversion for call = df × (F×P₁ - K×P₂)
    # P₁ = 0.5 + (1/π) ∫₀^∞ Re[e^{-iu·logK} × φ(u-i) / (iuφ(-i))] du
    # P₂ = 0.5 + (1/π) ∫₀^∞ Re[e^{-iu·logK} × φ(u) / (iu)] du
    #
    # φ(-i) = E[S_T] = F (forward) when φ is CF of log(S_T)

    forward = spot * math.exp((rate - div_yield) * T)
    phi_neg_i = char_func(complex(0, -1))  # = F in expectation
    F_implied = phi_neg_i.real

    du = u_max / N
    integral_p1 = 0.0
    integral_p2 = 0.0

    for j in range(1, N + 1):
        u = j * du
        iu = 1j * u
        e_factor = cmath.exp(-iu * log_K)

        # P₂ integrand
        phi_u = char_func(u)
        p2_int = (e_factor * phi_u / iu).real

        # P₁ integrand
        phi_u_mi = char_func(complex(u, -1))
        p1_int = (e_factor * phi_u_mi / (iu * F_implied)).real

        w = du if j < N else du / 2
        integral_p1 += w * p1_int
        integral_p2 += w * p2_int

    P1 = 0.5 + integral_p1 / math.pi
    P2 = 0.5 + integral_p2 / math.pi

    call = df * (F_implied * P1 - strike * P2)
    return max(call, 0.0)


# ---- Breeden-Litzenberger density recovery ----

@dataclass
class DensityResult:
    """Risk-neutral density recovered from option prices."""
    strikes: np.ndarray
    density: np.ndarray
    is_non_negative: bool


def density_from_calls(
    strikes: np.ndarray | list[float],
    call_prices: np.ndarray | list[float],
    rate: float,
    T: float,
) -> DensityResult:
    """Recover the risk-neutral density via Breeden-Litzenberger.

    The risk-neutral density is:
        f(K) = e^{rT} × d²C/dK²

    computed via central finite differences on the call price curve.

    Args:
        strikes: sorted array of strikes.
        call_prices: corresponding call prices.
        rate: risk-free rate.
        T: time to maturity.

    Returns:
        :class:`DensityResult` with density at interior strike points.
    """
    K = np.asarray(strikes, dtype=float)
    C = np.asarray(call_prices, dtype=float)
    n = len(K)

    if n < 3:
        return DensityResult(K, np.zeros_like(K), True)

    # Central second derivative at interior points (non-uniform grid safe)
    density = np.zeros(n)
    for i in range(1, n - 1):
        dk_left = K[i] - K[i - 1]
        dk_right = K[i + 1] - K[i]
        # Non-uniform second derivative: avoids the dk_avg² approximation
        d2C = 2.0 * (C[i - 1] / dk_left - C[i] * (1.0 / dk_left + 1.0 / dk_right)
                      + C[i + 1] / dk_right) / (dk_left + dk_right)
        density[i] = math.exp(rate * T) * d2C

    # Boundary: copy nearest interior value
    density[0] = density[1]
    density[-1] = density[-2]

    non_neg = bool(np.all(density >= -1e-10))

    return DensityResult(K, density, non_neg)


# ---- Density from characteristic function ----

def density_from_cf(
    char_func: Callable[[float], complex],
    x_grid: np.ndarray | list[float],
) -> np.ndarray:
    """Recover probability density via inverse Fourier of characteristic function.

    f(x) = (1/2π) ∫ exp(-iux) φ(u) du

    Computed via trapezoidal quadrature on a finite domain.

    Args:
        char_func: φ(u) = E[exp(iuX)].
        x_grid: points at which to evaluate the density.

    Returns:
        Array of density values at each x.
    """
    x = np.asarray(x_grid, dtype=float)
    n_u = 1024
    u_max = 50.0
    du = u_max / n_u

    density = np.zeros_like(x)
    for i, xi in enumerate(x):
        integral = 0.0
        for j in range(n_u + 1):
            u = j * du
            phi = char_func(u)
            integrand = (phi * cmath.exp(-1j * u * xi)).real
            w = du if (0 < j < n_u) else du / 2
            integral += w * integrand
        density[i] = integral / math.pi

    return density
