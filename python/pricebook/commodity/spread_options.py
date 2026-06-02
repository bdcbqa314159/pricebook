"""Spread options on commodity futures.

Kirk's approximation for two-asset spread options, with applications
to crack spreads, calendar spreads, and inter-commodity spreads.

* :class:`SpreadOptionResult` — pricing result with Greeks.
* :func:`kirk_spread_option` — Kirk's approximation for spread options.
* :func:`crack_spread_option` — option on refining margin.
* :func:`calendar_spread_option` — option on front-back spread.
* :func:`intercommodity_spread_option` — option on cross-commodity spread.

References:
    Kirk, *Correlation in the Energy Markets*, in Managing Energy Price
    Risk, Risk Publications, 1995.
    Margrabe, *The Value of an Option to Exchange One Asset for Another*,
    JF, 1978.
    Carmona & Durrleman, *Pricing and Hedging Spread Options*, SIAM Rev, 2003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.models.black76 import OptionType, _norm_cdf, _norm_pdf


@dataclass
class SpreadOptionResult:
    """Spread option pricing result."""
    price: float
    delta_asset1: float     # ∂V/∂F₁
    delta_asset2: float     # ∂V/∂F₂
    gamma_asset1: float
    gamma_asset2: float
    cross_gamma: float      # ∂²V/∂F₁∂F₂
    vega1: float            # per 1% vol₁
    vega2: float            # per 1% vol₂
    correlation_sensitivity: float  # ∂V/∂ρ
    spread: float           # F₁ − F₂ (or weighted)
    strike: float
    option_type: str

    def to_dict(self) -> dict:
        return vars(self)


# ---- Kirk's approximation ----

def kirk_spread_option(
    F1: float,
    F2: float,
    K: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    T: float,
    r: float = 0.04,
    option_type: str = "call",
    w1: float = 1.0,
    w2: float = 1.0,
) -> SpreadOptionResult:
    """Kirk's approximation for spread options.

    Prices an option on the spread: max(w₁F₁ − w₂F₂ − K, 0).

    Kirk's key insight: treat F₂ + K as a modified strike, then
    apply Black-76 with an adjusted vol:

    σ_adj² = σ₁² − 2ρσ₁σ₂(F₂/(F₂+K)) + σ₂²(F₂/(F₂+K))²

    Args:
        F1: futures price of asset 1 (e.g. gasoline).
        F2: futures price of asset 2 (e.g. crude oil).
        K: spread strike.
        sigma1: vol of asset 1.
        sigma2: vol of asset 2.
        rho: correlation between assets.
        T: time to expiry (years).
        r: risk-free rate.
        option_type: "call" or "put".
        w1: weight on asset 1.
        w2: weight on asset 2.
    """
    df = math.exp(-r * T)

    # Kirk's modified forward and vol
    adj_F2 = w2 * F2 + K
    if adj_F2 <= 0:
        # Degenerate case: spread option reduces to vanilla on F1
        intrinsic = max(w1 * F1 - w2 * F2 - K, 0) if option_type == "call" else max(w2 * F2 + K - w1 * F1, 0)
        return SpreadOptionResult(
            price=df * intrinsic, delta_asset1=0, delta_asset2=0,
            gamma_asset1=0, gamma_asset2=0, cross_gamma=0,
            vega1=0, vega2=0, correlation_sensitivity=0,
            spread=w1 * F1 - w2 * F2, strike=K, option_type=option_type,
        )

    ratio = w2 * F2 / adj_F2

    sigma_adj = math.sqrt(
        sigma1 ** 2
        - 2 * rho * sigma1 * sigma2 * ratio
        + (sigma2 * ratio) ** 2
    )

    # Apply Black-76 with modified forward = w1*F1, modified strike = adj_F2
    fwd = w1 * F1
    strike = adj_F2

    if T <= 0 or sigma_adj <= 0:
        intrinsic = max(fwd - strike, 0) if option_type == "call" else max(strike - fwd, 0)
        price = df * intrinsic
    else:
        sqrt_t = math.sqrt(T)
        d1 = (math.log(fwd / strike) + 0.5 * sigma_adj ** 2 * T) / (sigma_adj * sqrt_t)
        d2 = d1 - sigma_adj * sqrt_t

        if option_type == "call":
            price = df * (fwd * _norm_cdf(d1) - strike * _norm_cdf(d2))
        else:
            price = df * (strike * _norm_cdf(-d2) - fwd * _norm_cdf(-d1))

    # Greeks via finite differences
    bump = max(F1 * 0.001, 0.01)
    bump2 = max(F2 * 0.001, 0.01)

    p_up1 = _kirk_price(F1 + bump, F2, K, sigma1, sigma2, rho, T, r, option_type, w1, w2)
    p_dn1 = _kirk_price(F1 - bump, F2, K, sigma1, sigma2, rho, T, r, option_type, w1, w2)
    delta1 = (p_up1 - p_dn1) / (2 * bump)
    gamma1 = (p_up1 - 2 * price + p_dn1) / (bump ** 2)

    p_up2 = _kirk_price(F1, F2 + bump2, K, sigma1, sigma2, rho, T, r, option_type, w1, w2)
    p_dn2 = _kirk_price(F1, F2 - bump2, K, sigma1, sigma2, rho, T, r, option_type, w1, w2)
    delta2 = (p_up2 - p_dn2) / (2 * bump2)
    gamma2 = (p_up2 - 2 * price + p_dn2) / (bump2 ** 2)

    # Cross-gamma
    p_uu = _kirk_price(F1 + bump, F2 + bump2, K, sigma1, sigma2, rho, T, r, option_type, w1, w2)
    p_ud = _kirk_price(F1 + bump, F2 - bump2, K, sigma1, sigma2, rho, T, r, option_type, w1, w2)
    p_du = _kirk_price(F1 - bump, F2 + bump2, K, sigma1, sigma2, rho, T, r, option_type, w1, w2)
    p_dd = _kirk_price(F1 - bump, F2 - bump2, K, sigma1, sigma2, rho, T, r, option_type, w1, w2)
    cross_g = (p_uu - p_ud - p_du + p_dd) / (4 * bump * bump2)

    # Vega per 1% vol
    dv = 0.01
    vega1 = _kirk_price(F1, F2, K, sigma1 + dv, sigma2, rho, T, r, option_type, w1, w2) - price
    vega2 = _kirk_price(F1, F2, K, sigma1, sigma2 + dv, rho, T, r, option_type, w1, w2) - price

    # Correlation sensitivity per 0.01
    drho = 0.01
    rho_up = min(rho + drho, 0.999)
    corr_sens = _kirk_price(F1, F2, K, sigma1, sigma2, rho_up, T, r, option_type, w1, w2) - price

    return SpreadOptionResult(
        price=price,
        delta_asset1=delta1,
        delta_asset2=delta2,
        gamma_asset1=gamma1,
        gamma_asset2=gamma2,
        cross_gamma=cross_g,
        vega1=vega1,
        vega2=vega2,
        correlation_sensitivity=corr_sens,
        spread=w1 * F1 - w2 * F2,
        strike=K,
        option_type=option_type,
    )


def _kirk_price(F1, F2, K, s1, s2, rho, T, r, otype, w1, w2) -> float:
    """Internal: Kirk's price without Greeks."""
    df = math.exp(-r * T)
    adj_F2 = w2 * F2 + K
    if adj_F2 <= 0 or T <= 0:
        intrinsic = max(w1 * F1 - w2 * F2 - K, 0) if otype == "call" else max(w2 * F2 + K - w1 * F1, 0)
        return df * intrinsic

    ratio = w2 * F2 / adj_F2
    sigma_adj = math.sqrt(max(s1**2 - 2*rho*s1*s2*ratio + (s2*ratio)**2, 1e-12))
    fwd, strike = w1 * F1, adj_F2

    if sigma_adj <= 0:
        intrinsic = max(fwd - strike, 0) if otype == "call" else max(strike - fwd, 0)
        return df * intrinsic

    sqrt_t = math.sqrt(T)
    d1 = (math.log(fwd / strike) + 0.5 * sigma_adj**2 * T) / (sigma_adj * sqrt_t)
    d2 = d1 - sigma_adj * sqrt_t

    if otype == "call":
        return df * (fwd * _norm_cdf(d1) - strike * _norm_cdf(d2))
    return df * (strike * _norm_cdf(-d2) - fwd * _norm_cdf(-d1))


# ---- Application-specific spread options ----

def crack_spread_option(
    gasoline_price: float,
    crude_price: float,
    strike: float,
    vol_gasoline: float,
    vol_crude: float,
    correlation: float,
    T: float,
    r: float = 0.04,
    option_type: str = "call",
    ratio: tuple[float, float] = (1.0, 1.0),
) -> SpreadOptionResult:
    """Option on crack spread (refining margin).

    Payoff = max(w_gas × gasoline − w_crude × crude − K, 0).

    Standard 1:1 crack. For 3:2:1 use a portfolio of spread options.

    Args:
        gasoline_price: gasoline futures price ($/gal or $/bbl).
        crude_price: crude futures price ($/bbl).
        ratio: (gasoline_weight, crude_weight).
    """
    return kirk_spread_option(
        gasoline_price, crude_price, strike,
        vol_gasoline, vol_crude, correlation,
        T, r, option_type, ratio[0], ratio[1],
    )


def calendar_spread_option(
    front_price: float,
    back_price: float,
    strike: float,
    vol_front: float,
    vol_back: float,
    correlation: float,
    T: float,
    r: float = 0.04,
    option_type: str = "call",
) -> SpreadOptionResult:
    """Option on calendar spread (front − back).

    Payoff = max(F_front − F_back − K, 0).

    Args:
        front_price: near-month futures price.
        back_price: far-month futures price.
        vol_front: front-month vol.
        vol_back: back-month vol (typically lower, Samuelson).
        correlation: front-back correlation (typically high, 0.85–0.98).
    """
    return kirk_spread_option(
        front_price, back_price, strike,
        vol_front, vol_back, correlation,
        T, r, option_type,
    )


def intercommodity_spread_option(
    F1: float,
    F2: float,
    strike: float,
    sigma1: float,
    sigma2: float,
    correlation: float,
    T: float,
    r: float = 0.04,
    option_type: str = "call",
    commodity1: str = "WTI",
    commodity2: str = "Brent",
) -> SpreadOptionResult:
    """Option on inter-commodity spread (e.g. WTI-Brent).

    Args:
        commodity1: name of first commodity (for labeling).
        commodity2: name of second commodity.
    """
    return kirk_spread_option(
        F1, F2, strike, sigma1, sigma2, correlation,
        T, r, option_type,
    )
