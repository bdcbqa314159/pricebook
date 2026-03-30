"""
Implied volatility solvers.

Given a market price, recover the volatility that reproduces it under
Black-76 or Bachelier. Newton-Raphson (using vega) as primary solver,
bisection as fallback.

    vol = implied_vol_black76(
        market_price=5.0, forward=100, strike=105,
        T=1.0, df=0.95, option_type=OptionType.CALL,
    )
"""

from __future__ import annotations

import math

from pricebook.black76 import (
    OptionType,
    black76_price,
    black76_vega,
    bachelier_price,
    _norm_pdf,
)


def implied_vol_black76(
    market_price: float,
    forward: float,
    strike: float,
    T: float,
    df: float,
    option_type: OptionType = OptionType.CALL,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """
    Implied lognormal vol from a Black-76 price.

    Uses Newton-Raphson with vega, falling back to bisection if Newton
    fails to converge.

    Args:
        market_price: observed option price.
        forward: forward price.
        strike: option strike.
        T: time to expiry in years.
        df: discount factor to expiry.
        option_type: CALL or PUT.
        tol: convergence tolerance on vol.
        max_iter: maximum iterations.

    Returns:
        Implied lognormal volatility.

    Raises:
        ValueError: if price is below intrinsic or above upper bound.
    """
    if T <= 0:
        raise ValueError("T must be positive for implied vol")

    # Intrinsic value check
    if option_type == OptionType.CALL:
        intrinsic = df * max(forward - strike, 0.0)
    else:
        intrinsic = df * max(strike - forward, 0.0)

    if market_price < intrinsic - tol:
        raise ValueError(
            f"market_price {market_price:.6f} is below intrinsic {intrinsic:.6f}"
        )

    # Upper bound: call <= df*F, put <= df*K
    upper = df * forward if option_type == OptionType.CALL else df * strike
    if market_price > upper + tol:
        raise ValueError(
            f"market_price {market_price:.6f} exceeds upper bound {upper:.6f}"
        )

    # Newton-Raphson
    vol = 0.20  # initial guess
    for _ in range(max_iter):
        price = black76_price(forward, strike, vol, T, df, option_type)
        diff = price - market_price
        if abs(diff) < tol:
            return vol

        vega = black76_vega(forward, strike, vol, T, df)
        if vega < 1e-15:
            break  # vega too small, fall back to bisection
        vol -= diff / vega
        if vol <= 0:
            break  # went negative, fall back to bisection

    # Bisection fallback
    lo, hi = 1e-6, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        price = black76_price(forward, strike, mid, T, df, option_type)
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            return mid

    return 0.5 * (lo + hi)


def implied_vol_bachelier(
    market_price: float,
    forward: float,
    strike: float,
    T: float,
    df: float,
    option_type: OptionType = OptionType.CALL,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """
    Implied normal vol from a Bachelier price.

    Uses Newton-Raphson with Bachelier vega, falling back to bisection.

    Returns:
        Implied normal volatility (in price units, not percentage).
    """
    if T <= 0:
        raise ValueError("T must be positive for implied vol")

    if option_type == OptionType.CALL:
        intrinsic = df * max(forward - strike, 0.0)
    else:
        intrinsic = df * max(strike - forward, 0.0)

    if market_price < intrinsic - tol:
        raise ValueError(
            f"market_price {market_price:.6f} is below intrinsic {intrinsic:.6f}"
        )

    # Newton-Raphson
    # Bachelier vega = df * sqrt(T) * n(d) where d = (F-K)/(vol_n*sqrt(T))
    vol_n = market_price / (df * math.sqrt(T)) if df > 0 and T > 0 else 0.01
    vol_n = max(vol_n, 1e-6)

    for _ in range(max_iter):
        price = bachelier_price(forward, strike, vol_n, T, df, option_type)
        diff = price - market_price
        if abs(diff) < tol:
            return vol_n

        # Bachelier vega
        sqrt_t = math.sqrt(T)
        stdev = vol_n * sqrt_t
        if stdev < 1e-15:
            break
        d = (forward - strike) / stdev
        vega = df * sqrt_t * _norm_pdf(d)

        if vega < 1e-15:
            break
        vol_n -= diff / vega
        if vol_n <= 0:
            break

    # Bisection fallback
    lo, hi = 1e-6, forward * 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        price = bachelier_price(forward, strike, mid, T, df, option_type)
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            return mid

    return 0.5 * (lo + hi)
