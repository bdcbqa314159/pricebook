"""Black-76 and Bachelier (normal) option pricing models."""

import math
from enum import Enum


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black76_price(
    forward: float,
    strike: float,
    vol: float,
    time_to_expiry: float,
    df: float,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """
    Black-76 option price on a forward.

    C = df * [F*N(d1) - K*N(d2)]
    P = df * [K*N(-d2) - F*N(-d1)]

    where:
        d1 = [ln(F/K) + 0.5*vol^2*T] / (vol*sqrt(T))
        d2 = d1 - vol*sqrt(T)
    """
    if time_to_expiry <= 0:
        intrinsic = max(forward - strike, 0.0) if option_type == OptionType.CALL \
            else max(strike - forward, 0.0)
        return df * intrinsic
    if vol <= 0:
        intrinsic = max(forward - strike, 0.0) if option_type == OptionType.CALL \
            else max(strike - forward, 0.0)
        return df * intrinsic

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * time_to_expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t

    if option_type == OptionType.CALL:
        return df * (forward * _norm_cdf(d1) - strike * _norm_cdf(d2))
    return df * (strike * _norm_cdf(-d2) - forward * _norm_cdf(-d1))


def black76_delta(
    forward: float, strike: float, vol: float,
    time_to_expiry: float, df: float,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Black-76 delta: dPrice/dForward."""
    if time_to_expiry <= 0 or vol <= 0:
        if option_type == OptionType.CALL:
            return df if forward > strike else 0.0
        return -df if forward < strike else 0.0

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * time_to_expiry) / (vol * sqrt_t)
    if option_type == OptionType.CALL:
        return df * _norm_cdf(d1)
    return df * (_norm_cdf(d1) - 1.0)


def black76_gamma(
    forward: float, strike: float, vol: float,
    time_to_expiry: float, df: float,
) -> float:
    """Black-76 gamma: d²Price/dForward². Same for calls and puts."""
    if time_to_expiry <= 0 or vol <= 0:
        return 0.0

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * time_to_expiry) / (vol * sqrt_t)
    return df * _norm_pdf(d1) / (forward * vol * sqrt_t)


def black76_vega(
    forward: float, strike: float, vol: float,
    time_to_expiry: float, df: float,
) -> float:
    """Black-76 vega: dPrice/dVol. Same for calls and puts."""
    if time_to_expiry <= 0 or vol <= 0:
        return 0.0

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * time_to_expiry) / (vol * sqrt_t)
    return df * forward * sqrt_t * _norm_pdf(d1)


def black76_theta(
    forward: float, strike: float, vol: float,
    time_to_expiry: float, df: float,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """
    Black-76 theta: dPrice/dTime (negative for long options).

    This is the partial derivative with respect to time_to_expiry,
    holding forward and df constant.
    """
    if time_to_expiry <= 0 or vol <= 0:
        return 0.0

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * time_to_expiry) / (vol * sqrt_t)
    return -df * forward * vol * _norm_pdf(d1) / (2.0 * sqrt_t)


def bachelier_price(
    forward: float,
    strike: float,
    vol_normal: float,
    time_to_expiry: float,
    df: float,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """
    Bachelier (normal) model option price.

    Assumes forward follows arithmetic Brownian motion: dF = sigma_n * dW.
    Handles negative forwards/strikes naturally.

    C = df * [(F-K)*N(d) + sigma_n*sqrt(T)*n(d)]
    P = df * [(K-F)*N(-d) + sigma_n*sqrt(T)*n(d)]

    where d = (F - K) / (sigma_n * sqrt(T))
    """
    if time_to_expiry <= 0:
        intrinsic = max(forward - strike, 0.0) if option_type == OptionType.CALL \
            else max(strike - forward, 0.0)
        return df * intrinsic
    if vol_normal <= 0:
        intrinsic = max(forward - strike, 0.0) if option_type == OptionType.CALL \
            else max(strike - forward, 0.0)
        return df * intrinsic

    sqrt_t = math.sqrt(time_to_expiry)
    stdev = vol_normal * sqrt_t
    d = (forward - strike) / stdev

    if option_type == OptionType.CALL:
        return df * ((forward - strike) * _norm_cdf(d) + stdev * _norm_pdf(d))
    return df * ((strike - forward) * _norm_cdf(-d) + stdev * _norm_pdf(d))
