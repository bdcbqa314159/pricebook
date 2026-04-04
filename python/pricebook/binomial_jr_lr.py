"""Jarrow-Rudd and Leisen-Reimer binomial trees.

JR: equal probabilities (p = 0.5), drift-adjusted up/down moves.
LR: Peizer-Pratt inversion for much faster convergence to Black-Scholes.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from pricebook.black76 import OptionType


# ---------------------------------------------------------------------------
# Jarrow-Rudd
# ---------------------------------------------------------------------------


def _jr_params(rate: float, div_yield: float, vol: float, dt: float):
    """JR tree parameters: equal probabilities, adjusted drift."""
    drift = (rate - div_yield - 0.5 * vol * vol) * dt
    u = math.exp(drift + vol * math.sqrt(dt))
    d = math.exp(drift - vol * math.sqrt(dt))
    p = 0.5
    disc = math.exp(-rate * dt)
    return u, d, p, disc


def jr_european(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_steps: int,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> float:
    """European option via Jarrow-Rudd binomial tree."""
    dt = T / n_steps
    u, d, p, disc = _jr_params(rate, div_yield, vol, dt)
    return _tree_european(spot, strike, u, d, p, disc, n_steps, option_type)


def jr_american(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_steps: int,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> float:
    """American option via Jarrow-Rudd binomial tree."""
    dt = T / n_steps
    u, d, p, disc = _jr_params(rate, div_yield, vol, dt)
    return _tree_american(spot, strike, u, d, p, disc, n_steps, option_type)


# ---------------------------------------------------------------------------
# Leisen-Reimer
# ---------------------------------------------------------------------------


def _peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt inversion: maps a standard normal quantile to a binomial probability.

    PP2 formula (Leisen & Reimer 1996):
        h(z, n) ≈ 0.5 + sign(z) * sqrt(0.25 - 0.25 * exp(-(z/(n+1/3+0.1/(n+1)))^2 * (n+1/6)))
    """
    if abs(z) < 1e-14:
        return 0.5
    m = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
    return 0.5 + math.copysign(
        math.sqrt(0.25 - 0.25 * math.exp(-(z / m) ** 2 * (n + 1.0 / 6.0))),
        z,
    )


def _lr_params(
    spot: float, strike: float, rate: float, div_yield: float,
    vol: float, T: float, n_steps: int,
):
    """LR tree parameters via Peizer-Pratt inversion.

    Uses odd n_steps for centring. d1/d2 from Black-Scholes mapped to
    binomial probabilities for fast convergence.
    """
    # Force odd steps
    if n_steps % 2 == 0:
        n_steps += 1

    dt = T / n_steps
    d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)

    p_prime = _peizer_pratt(d1, n_steps)  # probability for S paths
    p = _peizer_pratt(d2, n_steps)        # risk-neutral probability

    u = math.exp((rate - div_yield) * dt) * p_prime / p
    d = (math.exp((rate - div_yield) * dt) - p * u) / (1.0 - p)
    disc = math.exp(-rate * dt)

    return u, d, p, disc, n_steps


def lr_european(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_steps: int,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> float:
    """European option via Leisen-Reimer binomial tree.

    Much faster convergence than CRR: typically N=51 matches BS to 4+ digits.
    """
    u, d, p, disc, n_steps = _lr_params(spot, strike, rate, div_yield, vol, T, n_steps)
    return _tree_european(spot, strike, u, d, p, disc, n_steps, option_type)


def lr_american(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_steps: int,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> float:
    """American option via Leisen-Reimer binomial tree."""
    u, d, p, disc, n_steps = _lr_params(spot, strike, rate, div_yield, vol, T, n_steps)
    return _tree_american(spot, strike, u, d, p, disc, n_steps, option_type)


# ---------------------------------------------------------------------------
# Shared backward induction
# ---------------------------------------------------------------------------


def _tree_european(
    spot: float, strike: float, u: float, d: float, p: float,
    disc: float, n_steps: int, option_type: OptionType,
) -> float:
    """Generic European tree pricing given (u, d, p, disc)."""
    st = spot * d ** np.arange(n_steps, -1, -1) * u ** np.arange(0, n_steps + 1)

    if option_type == OptionType.CALL:
        values = np.maximum(st - strike, 0.0)
    else:
        values = np.maximum(strike - st, 0.0)

    for i in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])

    return float(values[0])


def _tree_american(
    spot: float, strike: float, u: float, d: float, p: float,
    disc: float, n_steps: int, option_type: OptionType,
) -> float:
    """Generic American tree pricing given (u, d, p, disc)."""
    st = spot * d ** np.arange(n_steps, -1, -1) * u ** np.arange(0, n_steps + 1)

    if option_type == OptionType.CALL:
        values = np.maximum(st - strike, 0.0)
    else:
        values = np.maximum(strike - st, 0.0)

    for i in range(n_steps - 1, -1, -1):
        si = spot * d ** np.arange(i, -1, -1) * u ** np.arange(0, i + 1)
        continuation = disc * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])

        if option_type == OptionType.CALL:
            exercise = np.maximum(si - strike, 0.0)
        else:
            exercise = np.maximum(strike - si, 0.0)

        values = np.maximum(continuation, exercise)

    return float(values[0])
