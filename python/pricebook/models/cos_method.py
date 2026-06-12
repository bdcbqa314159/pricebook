"""
COS method: Fourier-cosine series expansion for option pricing.

Given a characteristic function φ(u) of log(S_T/K), prices European
options with O(N) complexity and exponential convergence.

See REFERENCES.md (Fang & Oosterlee 2008).

    price = cos_price(char_func=bs_char, spot=100, strike=105,
                      rate=0.05, T=1.0, N=128)
"""

from __future__ import annotations

import math
import cmath

from pricebook.models.black76 import OptionType


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
    char_func,
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
    # Fix T3.12: pre-fix clamped c2 to a hard floor of 0.001 (absolute), which
    # is huge for low-variance pricing (e.g. σ=10 %, T=0.01y → true c2 ≈ 1e-4).
    # The clamp inflated the truncation half-width L·√c2 by ≈ 3× the truth,
    # spreading the COS support far past the actual density and degrading
    # convergence for low-vol or short-T options.  Use a tiny absolute floor
    # purely as a numerical-noise guard (negative c2 from round-off), not as
    # a model-implied minimum.
    c2 = max(float(-(ln_p + ln_m - 2 * ln0).real / eps**2), 1e-12)
    a = x + c1 - L * math.sqrt(c2)
    b = x + c1 + L * math.sqrt(c2)

    # COS series
    x_minus_a = x - a
    price = 0.0
    for k in range(N):
        u_k = k * math.pi / (b - a)
        phi_k = char_func(u_k)

        re_part = (phi_k * cmath.exp(1j * u_k * x_minus_a)).real

        # Fix T1.8: V_k integration domain must be intersected with the
        # truncation interval [a, b].
        # Call payoff (e^y - 1) is positive on y ≥ 0, so V_k^call integrates
        # over [max(0,a), b].  Pre-fix: hardcoded [0, b], which over-counts
        # when a > 0 (deep-ITM call, low vol — the segment [0, a] is outside
        # truncation and shouldn't be part of the V_k integral).
        # Symmetric story for puts (payoff positive on y ≤ 0): pre-fix used
        # [a, 0] which over-counts when b < 0 (deep-ITM put).
        if option_type == OptionType.CALL:
            c = max(0.0, a)
            d = b
            if c < d:
                V_k = 2.0 / (b - a) * (
                    _chi(k, a, b, c, d) - _psi(k, a, b, c, d)
                )
            else:
                V_k = 0.0
        else:
            c = a
            d = min(0.0, b)
            if c < d:
                V_k = 2.0 / (b - a) * (
                    -_chi(k, a, b, c, d) + _psi(k, a, b, c, d)
                )
            else:
                V_k = 0.0

        weight = 0.5 if k == 0 else 1.0
        price += weight * re_part * V_k

    return df * strike * price


# ---------------------------------------------------------------------------
# Standard characteristic functions
# ---------------------------------------------------------------------------

def bs_char_func(rate: float, div_yield: float, vol: float, T: float):
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
):
    """Heston characteristic function of log(S_T/S_0) for COS method."""
    from pricebook.options.heston import _heston_f

    def phi(u: float) -> complex:
        return _heston_f(u, T, rate, div_yield, v0, kappa, theta, xi, rho, 0.0, 2)

    return phi
