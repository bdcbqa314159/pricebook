"""
Trinomial tree option pricing.

Kamrad-Ritchren parameterisation with stretch parameter lambda:
    u = exp(lambda * vol * sqrt(dt))
    d = 1/u
    m = 1
    p_u = 1/(2*lambda^2) + (r-q-0.5*vol^2)*sqrt(dt)/(2*lambda*vol)
    p_d = 1/(2*lambda^2) - (r-q-0.5*vol^2)*sqrt(dt)/(2*lambda*vol)
    p_m = 1 - p_u - p_d

lambda = sqrt(3/2) gives the standard trinomial. lambda = 1 gives a
special case that collapses to the binomial-like structure.

    price = trinomial_european(spot=100, strike=105, rate=0.05, vol=0.20,
                               T=1.0, n_steps=200, option_type=OptionType.CALL)
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.black76 import OptionType


def _kr_params(rate: float, div_yield: float, vol: float, dt: float,
               lam: float = math.sqrt(1.5)):
    """Kamrad-Ritchren trinomial parameters."""
    u = math.exp(lam * vol * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-rate * dt)
    nu = (rate - div_yield - 0.5 * vol**2) * math.sqrt(dt) / (lam * vol)
    p_u = 1.0 / (2.0 * lam**2) + nu / 2.0
    p_d = 1.0 / (2.0 * lam**2) - nu / 2.0
    p_m = 1.0 - p_u - p_d
    return u, d, p_u, p_m, p_d, disc


def trinomial_european(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_steps: int,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    lam: float = math.sqrt(1.5),
) -> float:
    """European option price via trinomial tree."""
    dt = T / n_steps
    u, d, p_u, p_m, p_d, disc = _kr_params(rate, div_yield, vol, dt, lam)

    # Terminal stock prices: 2*n_steps + 1 nodes
    # Node j at step n: S * u^j * d^(n-j) ... but trinomial has -n to +n
    # S_j = spot * u^j for j in [-n, n]
    n = n_steps
    j_vals = np.arange(-n, n + 1)
    st = spot * u ** j_vals

    if option_type == OptionType.CALL:
        values = np.maximum(st - strike, 0.0)
    else:
        values = np.maximum(strike - st, 0.0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        new_size = 2 * i + 1
        new_values = np.empty(new_size)
        for j in range(new_size):
            new_values[j] = disc * (
                p_u * values[j + 2] + p_m * values[j + 1] + p_d * values[j]
            )
        values = new_values

    return float(values[0])


def trinomial_american(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_steps: int,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    lam: float = math.sqrt(1.5),
) -> float:
    """American option price via trinomial tree with early exercise."""
    dt = T / n_steps
    u, d, p_u, p_m, p_d, disc = _kr_params(rate, div_yield, vol, dt, lam)

    n = n_steps
    j_vals = np.arange(-n, n + 1)
    st = spot * u ** j_vals

    if option_type == OptionType.CALL:
        values = np.maximum(st - strike, 0.0)
    else:
        values = np.maximum(strike - st, 0.0)

    for i in range(n_steps - 1, -1, -1):
        new_size = 2 * i + 1
        j_i = np.arange(-i, i + 1)
        si = spot * u ** j_i

        new_values = np.empty(new_size)
        for j in range(new_size):
            continuation = disc * (
                p_u * values[j + 2] + p_m * values[j + 1] + p_d * values[j]
            )
            if option_type == OptionType.CALL:
                exercise = max(si[j] - strike, 0.0)
            else:
                exercise = max(strike - si[j], 0.0)
            new_values[j] = max(continuation, exercise)
        values = new_values

    return float(values[0])
