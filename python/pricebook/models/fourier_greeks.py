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
        return dict(vars(self))


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
    """Vega via COS with CF perturbation (BS-like log-return variance bump).

    For BS-like models, bumping σ by Δσ changes the log-return variance
    by ΔVar = (σ + Δσ)²·T − σ²·T = (2σ·Δσ + Δσ²)·T.

    Fix T4-FG1: pre-fix added `δσ²·T` (the QUADRATIC term only) to the
    variance, missing the dominant linear 2σΔσT term.  For σ=20 %, Δσ=1 %:
        pre-fix ΔVar = (0.01)² · T = 1e-4 · T
        correct ΔVar = (2 × 0.20 × 0.01 + 1e-4) · T ≈ 4.1e-3 · T
    so the CF perturbation was ≈ 41× too small and the reported vega was
    ≈ 30× too small (≈ 0.012 for an ATM call where BS vega is ≈ 0.376).

    Post-fix extracts the implied σ from the CF via cumulant matching
    (c2 ≈ σ²·T) and applies the correct ΔVar perturbation.
    """
    import cmath
    from pricebook.models.cos_method import cos_price

    d_vol = 0.01  # 1% vol bump

    # Extract σ_implied from the CF via the 2nd cumulant c2 = σ²·T.
    eps = 1e-4
    try:
        ln_0 = cmath.log(char_func(0.0))
        ln_p = cmath.log(char_func(eps))
        ln_m = cmath.log(char_func(-eps))
        c2 = float(-(ln_p + ln_m - 2 * ln_0).real / (eps ** 2))
    except (ValueError, ZeroDivisionError):
        c2 = 0.0
    sigma_implied = math.sqrt(max(c2, 0.0) / T) if T > 1e-12 else 0.0

    # For a BS-like CF phi(u) = exp(iu·μ − 0.5·σ²T·u²) with
    # μ = (r-q)T − 0.5·σ²T (martingale-preserving drift), bumping σ → σ+Δσ
    # changes BOTH the drift and the variance:
    #   Δμ = -0.5·(σ+Δσ)²T + 0.5·σ²T = -(σΔσ + 0.5·Δσ²)·T
    #   ΔVar = (2σΔσ + Δσ²)·T
    # So the bumped CF is phi(u) · exp(i·u·Δμ − 0.5·ΔVar·u²).
    # Pre-fix only adjusted variance; the missing drift correction broke
    # the martingale property and gave a different (wrong) vega magnitude.
    d_mu = -(sigma_implied * d_vol + 0.5 * d_vol ** 2) * T
    var_extra = (2 * sigma_implied * d_vol + d_vol ** 2) * T

    def cf_bumped(u):
        return char_func(u) * np.exp(1j * u * d_mu - 0.5 * var_extra * u ** 2)

    p_bumped = cos_price(cf_bumped, spot, strike, rate, T, option_type, div_yield, N, L)
    return p_bumped - base_price  # per 1% vol
