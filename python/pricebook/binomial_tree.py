"""
Binomial tree option pricing (Cox-Ross-Rubinstein).

CRR parameters:
    u = exp(vol * sqrt(dt))
    d = 1/u
    p = (exp((r-q)*dt) - d) / (u - d)

Forward induction builds stock prices, backward induction discounts
expected payoffs. American options check for early exercise at each node.

    price = binomial_european(spot=100, strike=105, rate=0.05, vol=0.20,
                              T=1.0, n_steps=500, option_type=OptionType.CALL)
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.black76 import OptionType


def _crr_params(rate: float, div_yield: float, vol: float, dt: float):
    """CRR tree parameters."""
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-rate * dt)
    p = (math.exp((rate - div_yield) * dt) - d) / (u - d)
    return u, d, p, disc


def binomial_european(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_steps: int,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> float:
    """European option price via CRR binomial tree.

    Args:
        spot: initial price.
        strike: option strike.
        rate: risk-free rate (continuous).
        vol: lognormal volatility.
        T: time to expiry.
        n_steps: number of time steps.
        option_type: CALL or PUT.
        div_yield: continuous dividend yield.
    """
    dt = T / n_steps
    u, d, p, disc = _crr_params(rate, div_yield, vol, dt)

    # Terminal stock prices: S * u^j * d^(n-j) for j = 0..n
    st = spot * d ** np.arange(n_steps, -1, -1) * u ** np.arange(0, n_steps + 1)

    # Terminal payoffs
    if option_type == OptionType.CALL:
        values = np.maximum(st - strike, 0.0)
    else:
        values = np.maximum(strike - st, 0.0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])

    return float(values[0])


def binomial_american(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_steps: int,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> float:
    """American option price via CRR binomial tree.

    At each node: max(continuation value, exercise value).
    """
    dt = T / n_steps
    u, d, p, disc = _crr_params(rate, div_yield, vol, dt)

    # Terminal stock prices
    st = spot * d ** np.arange(n_steps, -1, -1) * u ** np.arange(0, n_steps + 1)

    # Terminal payoffs
    if option_type == OptionType.CALL:
        values = np.maximum(st - strike, 0.0)
    else:
        values = np.maximum(strike - st, 0.0)

    # Backward induction with early exercise
    for i in range(n_steps - 1, -1, -1):
        # Stock prices at step i
        si = spot * d ** np.arange(i, -1, -1) * u ** np.arange(0, i + 1)

        # Continuation value
        continuation = disc * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])

        # Exercise value
        if option_type == OptionType.CALL:
            exercise = np.maximum(si - strike, 0.0)
        else:
            exercise = np.maximum(strike - si, 0.0)

        values = np.maximum(continuation, exercise)

    return float(values[0])
