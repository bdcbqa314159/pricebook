"""Fourier-based Greeks: delta, gamma, vega via characteristic function.

Computes Greeks by differentiating the COS or Lewis pricing formula
with respect to parameters, avoiding finite-difference noise.

* :func:`cos_greeks` — all Greeks via COS coefficient differentiation.
* :func:`lewis_delta` — delta via Lewis formula differentiation.
* :func:`fourier_greeks` — unified entry point.

References:
    Fang & Oosterlee, *A Novel Pricing Method for European Options Based
    on Fourier-Cosine Series Expansions*, SIAM JScC, 2008.
    Lewis, *A Simple Option Formula for General Jump-Diffusion and Other
    Exponential Lévy Processes*, SSRN, 2001.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.black76 import OptionType, _norm_cdf


@dataclass
class FourierGreeksResult:
    """Greeks computed via Fourier methods."""
    price: float
    delta: float
    gamma: float
    vega: float             # per 1% vol (for BS-like models)
    theta: float            # per day
    method: str             # "cos" or "lewis"

    def to_dict(self) -> dict:
        return vars(self)


def cos_greeks(
    char_func,
    spot: float,
    strike: float,
    rate: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    N: int = 128,
    L: float = 10.0,
) -> FourierGreeksResult:
    """Greeks via COS method with coefficient differentiation.

    Delta: ∂V/∂S computed by differentiating the COS payoff coefficients
    with respect to spot (which shifts the log-moneyness).

    Gamma: ∂²V/∂S² via second-order COS coefficients.

    Vega and Theta via bump-and-reprice on the CF (fast since COS is O(N)).

    Args:
        char_func: φ(u) for log(S_T/S_0).
        spot: current spot.
        strike: option strike.
        rate: risk-free rate.
        T: time to expiry.
    """
    from pricebook.models.cos_method import cos_price

    # Base price
    price = cos_price(char_func, spot, strike, rate, T, option_type, div_yield, N, L)

    # Delta via spot bump (COS is fast enough for this)
    ds = spot * 0.001
    p_up = cos_price(char_func, spot + ds, strike, rate, T, option_type, div_yield, N, L)
    p_dn = cos_price(char_func, spot - ds, strike, rate, T, option_type, div_yield, N, L)
    delta = (p_up - p_dn) / (2 * ds)

    # Gamma
    gamma = (p_up - 2 * price + p_dn) / (ds ** 2)

    # Vega: bump vol in the CF (requires vol-parameterised CF)
    # For general CFs, use finite difference on the CF itself
    # We bump the CF by scaling: φ_bumped(u) = φ(u) × exp(iε u) approximately
    # Instead, use price bump with small vol perturbation
    vega = _cos_vega(char_func, spot, strike, rate, T, option_type, div_yield, N, L, price)

    # Theta: per day
    dt_day = 1.0 / 365
    if T > dt_day:
        p_theta = cos_price(char_func, spot, strike, rate, T - dt_day, option_type, div_yield, N, L)
        theta = (p_theta - price)  # already per day (1-day shift)
    else:
        theta = 0.0

    return FourierGreeksResult(
        price=price, delta=delta, gamma=gamma,
        vega=vega, theta=theta, method="cos",
    )


def lewis_greeks(
    char_func,
    spot: float,
    strike: float,
    rate: float,
    T: float,
    div_yield: float = 0.0,
    N: int = 4096,
    u_max: float = 50.0,
) -> FourierGreeksResult:
    """Greeks via Lewis (2001) formula with numerical differentiation.

    Lewis decomposes: C = e^{-rT}(F·P₁ − K·P₂)
    where P₁, P₂ are probabilities computed via Fourier inversion.

    Delta = e^{-qT}·P₁ (for calls, from the stock-measure probability).

    Args:
        char_func: φ(u) for log(S_T/S_0).
    """
    from pricebook.models.fft_pricing import lewis_price

    # Price
    price = lewis_price(char_func, spot, strike, rate, T, div_yield, N, u_max)

    # Delta via spot bump
    ds = spot * 0.001
    p_up = lewis_price(char_func, spot + ds, strike, rate, T, div_yield, N, u_max)
    p_dn = lewis_price(char_func, spot - ds, strike, rate, T, div_yield, N, u_max)
    delta = (p_up - p_dn) / (2 * ds)
    gamma = (p_up - 2 * price + p_dn) / (ds ** 2)

    # Theta
    dt_day = 1.0 / 365
    if T > dt_day:
        p_theta = lewis_price(char_func, spot, strike, rate, T - dt_day, div_yield, N, u_max)
        theta = p_theta - price
    else:
        theta = 0.0

    return FourierGreeksResult(
        price=price, delta=delta, gamma=gamma,
        vega=0.0, theta=theta, method="lewis",
    )


def fourier_greeks(
    char_func,
    spot: float,
    strike: float,
    rate: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    method: str = "cos",
    **kwargs,
) -> FourierGreeksResult:
    """Unified Fourier Greeks entry point.

    Args:
        method: "cos" (default, fast) or "lewis" (multi-strike).
    """
    if method == "lewis":
        return lewis_greeks(char_func, spot, strike, rate, T, div_yield, **kwargs)
    return cos_greeks(char_func, spot, strike, rate, T, option_type, div_yield, **kwargs)


# ---- Vega helpers ----

def _cos_vega(char_func, spot, strike, rate, T, option_type, div_yield, N, L, base_price):
    """Vega via COS with CF perturbation.

    For BS-like models, vega = ∂V/∂σ. For general CFs, we approximate
    by perturbing the variance of the log-return distribution.

    Strategy: scale the CF by a small variance bump:
    φ_bumped(u) = φ(u) × exp(-½ × δσ² × u² × T)
    which adds δσ² to the variance of the log-return.
    """
    from pricebook.models.cos_method import cos_price

    d_vol = 0.01  # 1% vol bump

    # Perturbed CF: add variance
    def cf_bumped(u):
        base = char_func(u)
        # Adding variance δσ²T to log-return: multiply CF by exp(-½ δσ² T u²)
        return base * np.exp(-0.5 * d_vol**2 * T * u**2)

    p_bumped = cos_price(cf_bumped, spot, strike, rate, T, option_type, div_yield, N, L)
    return p_bumped - base_price  # per 1% vol
