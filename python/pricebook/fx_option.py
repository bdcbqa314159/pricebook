"""
FX vanilla option pricing (Garman-Kohlhagen).

An FX call is the right to buy the base currency (sell quote) at strike K.
An FX put is the right to sell the base currency (buy quote) at strike K.

Garman-Kohlhagen = Black-Scholes with domestic rate r_d and foreign rate r_f:
    F = S * exp((r_d - r_f) * T)
    price = Black76(F, K, vol, T, df_d)

where df_d = exp(-r_d * T) discounts in domestic (quote) currency.

    price = fx_option_price(spot=1.0850, strike=1.10, r_d=0.05, r_f=0.03,
                            vol=0.08, T=1.0, option_type=OptionType.CALL)
"""

from __future__ import annotations

import math

from pricebook.black76 import (
    OptionType,
    black76_price,
    black76_vega,
    _norm_cdf,
)
from pricebook.greeks import Greeks


def fx_forward(spot: float, r_d: float, r_f: float, T: float) -> float:
    """CIP forward: F = S * exp((r_d - r_f) * T)."""
    return spot * math.exp((r_d - r_f) * T)


def fx_option_price(
    spot: float,
    strike: float,
    r_d: float,
    r_f: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """European FX option price (Garman-Kohlhagen via Black-76)."""
    fwd = fx_forward(spot, r_d, r_f, T)
    df_d = math.exp(-r_d * T)
    return black76_price(fwd, strike, vol, T, df_d, option_type)


def fx_spot_delta(
    spot: float,
    strike: float,
    r_d: float,
    r_f: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Spot delta: dPrice/dSpot = exp(-r_f*T) * N(d1) for call."""
    if T <= 0 or vol <= 0:
        if option_type == OptionType.CALL:
            return math.exp(-r_f * T) if spot > strike else 0.0
        return -math.exp(-r_f * T) if spot < strike else 0.0

    fwd = fx_forward(spot, r_d, r_f, T)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(fwd / strike) + 0.5 * vol * vol * T) / (vol * sqrt_t)
    eq = math.exp(-r_f * T)
    if option_type == OptionType.CALL:
        return eq * _norm_cdf(d1)
    return eq * (_norm_cdf(d1) - 1.0)


def fx_forward_delta(
    spot: float,
    strike: float,
    r_d: float,
    r_f: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Forward delta: dPrice/dForward * exp(r_d*T) = N(d1) for call."""
    if T <= 0 or vol <= 0:
        if option_type == OptionType.CALL:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0

    fwd = fx_forward(spot, r_d, r_f, T)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(fwd / strike) + 0.5 * vol * vol * T) / (vol * sqrt_t)
    if option_type == OptionType.CALL:
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def fx_premium_adjusted_delta(
    spot: float,
    strike: float,
    r_d: float,
    r_f: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Premium-adjusted (forward) delta.

    Used when the premium is paid in foreign currency (e.g. USD/JPY):
        delta_pa = delta_fwd - price / (spot * exp(-r_f*T)) ... simplified:
        delta_pa = N(d1) - (price/spot) * exp(r_f*T)  for call
    Alternatively: delta_pa = sign * N(sign * d2) where sign=+1 call, -1 put.
    """
    if T <= 0 or vol <= 0:
        return fx_forward_delta(spot, strike, r_d, r_f, vol, T, option_type)

    fwd = fx_forward(spot, r_d, r_f, T)
    sqrt_t = math.sqrt(T)
    d2 = (math.log(fwd / strike) - 0.5 * vol * vol * T) / (vol * sqrt_t)
    if option_type == OptionType.CALL:
        return _norm_cdf(d2)
    return _norm_cdf(d2) - 1.0


def fx_vega(
    spot: float,
    strike: float,
    r_d: float,
    r_f: float,
    vol: float,
    T: float,
) -> float:
    """FX vega: dPrice/dVol."""
    fwd = fx_forward(spot, r_d, r_f, T)
    df_d = math.exp(-r_d * T)
    return black76_vega(fwd, strike, vol, T, df_d)


def strike_from_delta(
    spot: float,
    delta: float,
    r_d: float,
    r_f: float,
    vol: float,
    T: float,
    delta_type: str = "spot",
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Compute strike from a delta value.

    Args:
        delta_type: "spot", "forward", or "premium_adjusted".
    """
    fwd = fx_forward(spot, r_d, r_f, T)
    sqrt_t = math.sqrt(T)

    if delta_type == "forward":
        # delta_fwd = N(d1) for call → d1 = N^{-1}(delta)
        from scipy.stats import norm
        sign = 1.0 if option_type == OptionType.CALL else -1.0
        d1 = sign * norm.ppf(sign * delta)
        return fwd * math.exp(-d1 * vol * sqrt_t + 0.5 * vol * vol * T)

    elif delta_type == "premium_adjusted":
        from scipy.stats import norm
        sign = 1.0 if option_type == OptionType.CALL else -1.0
        d2 = sign * norm.ppf(sign * delta)
        return fwd * math.exp(-d2 * vol * sqrt_t - 0.5 * vol * vol * T)

    else:  # spot delta
        from scipy.stats import norm
        sign = 1.0 if option_type == OptionType.CALL else -1.0
        adj_delta = delta / math.exp(-r_f * T)  # remove discount
        d1 = sign * norm.ppf(sign * adj_delta)
        return fwd * math.exp(-d1 * vol * sqrt_t + 0.5 * vol * vol * T)


def fx_greeks(
    spot: float,
    strike: float,
    r_d: float,
    r_f: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
) -> Greeks:
    """All FX option Greeks in one call."""
    return Greeks(
        price=fx_option_price(spot, strike, r_d, r_f, vol, T, option_type),
        delta=fx_spot_delta(spot, strike, r_d, r_f, vol, T, option_type),
        vega=fx_vega(spot, strike, r_d, r_f, vol, T),
    )
