"""
COS method: Fourier-cosine series expansion for option pricing.

Given a characteristic function φ(u) of log(S_T/K), prices European
options with O(N) complexity and exponential convergence.

Fang & Oosterlee (2008), "A Novel Pricing Method for European Options
Based on Fourier-Cosine Series Expansions."

    price = cos_price(char_func=bs_char, spot=100, strike=105,
                      rate=0.05, T=1.0, N=128)
"""

from __future__ import annotations

import math
import cmath
from typing import Protocol

import numpy as np

from pricebook.black76 import OptionType


class CharFunc(Protocol):
    """Protocol for characteristic functions φ(u) = E[exp(iu*X)]."""

    def __call__(self, u: float) -> complex: ...


def _cos_truncation_range(
    c1: float, c2: float, c4: float = 0.0, L: float = 10.0,
) -> tuple[float, float]:
    """Compute truncation range [a, b] from cumulants.

    c1 = first cumulant (mean), c2 = second (variance),
    c4 = fourth. L controls how many stdevs to include.
    """
    a = c1 - L * math.sqrt(c2 + math.sqrt(c4))
    b = c1 + L * math.sqrt(c2 + math.sqrt(c4))
    return a, b


def _chi(k: int, a: float, b: float, c: float, d: float) -> float:
    """χ_k(c, d) = ∫_c^d exp(x) cos(kπ(x-a)/(b-a)) dx (analytical)."""
    w = k * math.pi / (b - a)
    if k == 0:
        return math.exp(d) - math.exp(c)
    denom = 1 + w * w
    return (
        (math.exp(d) * (math.cos(w * (d - a)) + w * math.sin(w * (d - a)))
         - math.exp(c) * (math.cos(w * (c - a)) + w * math.sin(w * (c - a))))
        / denom
    )


def _psi(k: int, a: float, b: float, c: float, d: float) -> float:
    """ψ_k(c, d) = ∫_c^d cos(kπ(x-a)/(b-a)) dx (analytical)."""
    if k == 0:
        return d - c
    w = k * math.pi / (b - a)
    return (math.sin(w * (d - a)) - math.sin(w * (c - a))) / w


def cos_price(
    char_func: CharFunc,
    spot: float,
    strike: float,
    rate: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    N: int = 128,
    L: float = 10.0,
) -> float:
    """European option price via COS method.

    Args:
        char_func: characteristic function of log(S_T/S_0) under risk-neutral measure.
            φ(u) = E[exp(iu * log(S_T/S_0))].
        spot: current spot price.
        strike: option strike.
        rate: risk-free rate.
        T: time to expiry.
        option_type: CALL or PUT.
        div_yield: continuous dividend yield.
        N: number of cosine terms (higher = more accurate).
        L: truncation parameter (number of stdevs).
    """
    x = math.log(spot / strike)
    df = math.exp(-rate * T)

    # Estimate cumulants numerically from the char func
    eps = 1e-4
    phi0 = char_func(0.0)
    phi_p = char_func(eps)
    phi_m = char_func(-eps)
    ln0 = cmath.log(phi0) if abs(phi0) > 1e-20 else 0
    ln_p = cmath.log(phi_p)
    ln_m = cmath.log(phi_m)
    c1 = float((ln_p - ln_m).imag / (2 * eps))
    c2 = max(float(-(ln_p + ln_m - 2 * ln0).real / eps**2), 0.001)
    # Shift by x = log(S/K) to center on log(S_T/K) = log(S_T/S_0) + x
    a = x + c1 - L * math.sqrt(c2)
    b = x + c1 + L * math.sqrt(c2)

    # COS series
    price = 0.0
    for k in range(N):
        # Characteristic function evaluated at kπ/(b-a)
        u_k = k * math.pi / (b - a)
        phi_k = char_func(u_k)

        # Shift char func from log(S_T/S_0) to log(S_T/K) by multiplying by exp(iu*x)
        re_part = (phi_k * cmath.exp(1j * u_k * x) * cmath.exp(-1j * u_k * a)).real

        # Payoff coefficients
        if option_type == OptionType.CALL:
            V_k = 2.0 / (b - a) * (
                _chi(k, a, b, 0, b) - _psi(k, a, b, 0, b)
            )
        else:
            V_k = 2.0 / (b - a) * (
                -_chi(k, a, b, a, 0) + _psi(k, a, b, a, 0)
            )

        weight = 0.5 if k == 0 else 1.0
        price += weight * re_part * V_k

    return df * strike * price


# ---------------------------------------------------------------------------
# Standard characteristic functions
# ---------------------------------------------------------------------------

def bs_char_func(rate: float, div_yield: float, vol: float, T: float) -> CharFunc:
    """Black-Scholes characteristic function of log(S_T/S_0)."""
    mu = (rate - div_yield - 0.5 * vol**2) * T
    var = vol**2 * T

    def phi(u: float) -> complex:
        return cmath.exp(1j * u * mu - 0.5 * u**2 * var)

    return phi


def heston_char_func_cos(
    rate: float,
    div_yield: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
) -> CharFunc:
    """Heston characteristic function of log(S_T/S_0) for COS method."""
    from pricebook.heston import _heston_f

    x_dummy = 0.0  # log(S) cancels in the COS formulation

    def phi(u: float) -> complex:
        # Use measure j=2 (risk-neutral)
        f = _heston_f(u, T, rate, div_yield, v0, kappa, theta, xi, rho, x_dummy, 2)
        # Normalize: divide by exp(iu*x) since _heston_f includes exp(iu*x)
        # But x_dummy=0, so exp(iu*0)=1. However, _heston_f includes the
        # drift term exp(iu*(r-q)*T) which we want in our char func.
        return f

    return phi
