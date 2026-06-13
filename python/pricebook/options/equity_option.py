"""
European equity option pricing via Black-Scholes.

Reuses Black-76 with F = S * exp((r-q)*T). Provides spot-based Greeks
(delta w.r.t. spot, not forward).

    price = equity_option_price(spot=100, strike=105, rate=0.05, vol=0.20,
                                T=1.0, option_type=OptionType.CALL)
"""

from __future__ import annotations

import math

from pricebook.models.black76 import (
    OptionType,
    black76_price,
    black76_vega,
    black76_theta,
    _norm_cdf,
    _norm_pdf,
)
from pricebook.core.greeks import Greeks


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

    Fix T4-EQ1: pre-fix the degenerate branch (``T <= 0 or vol <= 0``)
    had three coupled bugs:

    1. It compared ``spot`` to ``strike`` instead of ``forward`` to
       ``strike``.  For a non-zero dividend yield, ``forward = spot · exp((r-q)T)``
       can differ materially from spot — e.g. q > r yields forward < spot,
       so a call that is "ITM on spot" can still be OTM on forward.
       The payoff at expiry depends on ``forward`` (under zero vol the
       terminal spot equals the forward to no-arbitrage).
    2. The ITM magnitude was a literal ``1.0`` (or ``-1.0``) instead of
       ``exp(-q·T)``.  Spot delta of an ITM call = ``∂(S - K·exp(-rT))/∂S
       = exp(-qT)`` for the FX-/dividend-adjusted forward replication.
    3. The exactly-ATM case (forward == strike) returned 0 (call) /
       0 (put) instead of the standard ``±0.5·exp(-qT)`` limit —
       asymmetric vs ``black76_delta`` which gets this right.
    """
    forward, df = _forward_and_df(spot, rate, div_yield, T)
    if T <= 0 or vol <= 0:
        eq = math.exp(-div_yield * T)
        if option_type == OptionType.CALL:
            if forward > strike:
                return eq
            if forward < strike:
                return 0.0
            return 0.5 * eq
        # PUT
        if forward < strike:
            return -eq
        if forward > strike:
            return 0.0
        return -0.5 * eq

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

    Fix T4-EQ3: pre-fix the ``T <= 0 or vol <= 0`` branch returned
    ``theta_b76`` (the first σ·n(d1) Black-76 term, which is 0 at σ=0)
    and silently dropped both the ``theta_r`` (rate-discount) and
    ``theta_q`` (dividend) corrections.  But at ``vol=0, T>0`` the
    price is ``S·exp(-qT) − K·exp(-rT)`` (ITM call, deterministic) and
    its theta is ``q·S·exp(-qT) − r·K·exp(-rT)`` — non-zero, and
    sometimes the dominant component of total theta when rate/div are
    far from zero.  Now compute the deterministic limit explicitly.
    """
    import math
    from scipy.stats import norm

    forward, df = _forward_and_df(spot, rate, div_yield, T)
    # Black-76 theta gives the first term only
    theta_b76 = black76_theta(forward, strike, vol, T, df, option_type)

    if T <= 0:
        # At expiry, no time decay (convention).
        return theta_b76
    if vol <= 0:
        # Deterministic limit: price = max(F·exp(-qT)·indicator - K·exp(-rT)·indicator, 0)
        # collapses by forward vs strike.  Take ∂/∂T of price and negate (theta convention).
        sgn = 1.0 if option_type == OptionType.CALL else -1.0
        if sgn * (forward - strike) > 0:
            # ITM (under deterministic forward).
            theta_q = div_yield * spot * math.exp(-div_yield * T)
            theta_r = -rate * strike * math.exp(-rate * T)
            # For put, both signs flip.
            return sgn * (theta_q + theta_r)
        if sgn * (forward - strike) < 0:
            # OTM: price = 0 → theta = 0.
            return 0.0
        # ATM: one-sided limit = half of the ITM value.
        theta_q = div_yield * spot * math.exp(-div_yield * T)
        theta_r = -rate * strike * math.exp(-rate * T)
        return 0.5 * sgn * (theta_q + theta_r)

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
    """Rho: dPrice/dRate (per 1 unit rate change).

    Fix T4-EQ2: pre-fix the ``T <= 0 or vol <= 0`` branch returned 0
    unconditionally.  At T=0 (expiry) rho is genuinely 0 (no time for
    rate to act).  But at vol=0 with T>0 the deterministic-payoff limit
    is non-trivial: an ITM call has price ``S·exp(-qT) - K·exp(-rT)``
    so ``rho = T·K·exp(-rT)`` (positive).  ITM put: ``rho = -T·K·exp(-rT)``.
    OTM (either side): 0.  At-the-money (forward == strike): the
    standard ATM-at-expiry-style one-sided limit is ``±0.5·T·K·exp(-rT)``.
    Pre-fix the zero return silently dropped this deterministic rho.
    """
    if T <= 0:
        return 0.0
    if vol <= 0:
        # Deterministic-payoff limit at vol=0, T>0.
        forward, df = _forward_and_df(spot, rate, div_yield, T)
        rho_itm = strike * T * df  # = K·T·exp(-rT)
        if option_type == OptionType.CALL:
            if forward > strike:
                return rho_itm
            if forward < strike:
                return 0.0
            return 0.5 * rho_itm
        # PUT
        if forward < strike:
            return -rho_itm
        if forward > strike:
            return 0.0
        return -0.5 * rho_itm

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
