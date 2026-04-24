"""
European equity option pricing via Black-Scholes.

Reuses Black-76 with F = S * exp((r-q)*T). Provides spot-based Greeks
(delta w.r.t. spot, not forward).

    price = equity_option_price(spot=100, strike=105, rate=0.05, vol=0.20,
                                T=1.0, option_type=OptionType.CALL)
"""

from __future__ import annotations

import math

from pricebook.black76 import (
    OptionType,
    black76_price,
    black76_vega,
    black76_theta,
    _norm_cdf,
    _norm_pdf,
)
from pricebook.greeks import Greeks


def _forward_and_df(spot, rate, div_yield, T):
    forward = spot * math.exp((rate - div_yield) * T)
    df = math.exp(-rate * T)
    return forward, df


def equity_option_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> float:
    """European equity option price (Black-Scholes via Black-76)."""
    forward, df = _forward_and_df(spot, rate, div_yield, T)
    return black76_price(forward, strike, vol, T, df, option_type)


def equity_delta(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> float:
    """Spot delta: dPrice/dSpot.

    Spot delta = exp(-q*T) * forward_delta.
    """
    forward, df = _forward_and_df(spot, rate, div_yield, T)
    if T <= 0 or vol <= 0:
        if option_type == OptionType.CALL:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0

    sqrt_t = math.sqrt(T)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * T) / (vol * sqrt_t)
    eq = math.exp(-div_yield * T)
    if option_type == OptionType.CALL:
        return eq * _norm_cdf(d1)
    return eq * (_norm_cdf(d1) - 1.0)


def equity_gamma(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    div_yield: float = 0.0,
) -> float:
    """Spot gamma: d²Price/dSpot². Same for calls and puts."""
    forward, df = _forward_and_df(spot, rate, div_yield, T)
    if T <= 0 or vol <= 0:
        return 0.0

    sqrt_t = math.sqrt(T)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * T) / (vol * sqrt_t)
    eq = math.exp(-div_yield * T)
    return eq * _norm_pdf(d1) / (spot * vol * sqrt_t)


def equity_vega(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    div_yield: float = 0.0,
) -> float:
    """Vega: dPrice/dVol. Same for calls and puts."""
    forward, df = _forward_and_df(spot, rate, div_yield, T)
    return black76_vega(forward, strike, vol, T, df)


def equity_theta(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> float:
    """Theta: dPrice/dTime (per year, negative for long options).

    Full Hull formula:
        call: -S×n(d1)×σ×e^{-qT}/(2√T) - r×K×e^{-rT}×N(d2) + q×S×e^{-qT}×N(d1)
        put:  -S×n(d1)×σ×e^{-qT}/(2√T) + r×K×e^{-rT}×N(-d2) - q×S×e^{-qT}×N(-d1)
    """
    import math
    from scipy.stats import norm

    forward, df = _forward_and_df(spot, rate, div_yield, T)
    # Black-76 theta gives the first term only
    theta_b76 = black76_theta(forward, strike, vol, T, df, option_type)

    if T <= 0 or vol <= 0:
        return theta_b76

    sqrt_t = math.sqrt(T)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * T) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t

    if option_type == OptionType.CALL:
        theta_r = -rate * strike * math.exp(-rate * T) * norm.cdf(d2)
        theta_q = div_yield * spot * math.exp(-div_yield * T) * norm.cdf(d1)
    else:
        theta_r = rate * strike * math.exp(-rate * T) * norm.cdf(-d2)
        theta_q = -div_yield * spot * math.exp(-div_yield * T) * norm.cdf(-d1)

    return theta_b76 + theta_r + theta_q


def equity_rho(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> float:
    """Rho: dPrice/dRate (per 1 unit rate change)."""
    if T <= 0 or vol <= 0:
        return 0.0

    forward, df = _forward_and_df(spot, rate, div_yield, T)
    sqrt_t = math.sqrt(T)
    d2 = (math.log(forward / strike) + 0.5 * vol * vol * T) / (vol * sqrt_t) - vol * sqrt_t

    if option_type == OptionType.CALL:
        return strike * T * df * _norm_cdf(d2)
    return -strike * T * df * _norm_cdf(-d2)


def equity_greeks(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> Greeks:
    """All equity option Greeks in one call."""
    return Greeks(
        price=equity_option_price(spot, strike, rate, vol, T, option_type, div_yield),
        delta=equity_delta(spot, strike, rate, vol, T, option_type, div_yield),
        gamma=equity_gamma(spot, strike, rate, vol, T, div_yield),
        vega=equity_vega(spot, strike, rate, vol, T, div_yield),
        theta=equity_theta(spot, strike, rate, vol, T, option_type, div_yield),
        rho=equity_rho(spot, strike, rate, vol, T, option_type, div_yield),
    )
