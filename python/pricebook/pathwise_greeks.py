"""Pathwise Greeks: differentiate payoff along MC paths.

No re-simulation needed — delta computed from the same paths used
for pricing. For GBM: ∂S(t)/∂S₀ = S(t)/S₀.

    from pricebook.pathwise_greeks import pathwise_asian_delta

    delta = pathwise_asian_delta(paths, strike, spot, rate, T)

References:
    Glasserman, *Monte Carlo Methods in Financial Engineering*, Ch. 7.
    Broadie & Glasserman (1996). Estimating Security Price Derivatives
    Using Simulation. Management Science, 42(2).
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.black76 import OptionType


def pathwise_asian_delta(
    paths: np.ndarray,
    strike: float,
    spot: float,
    rate: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Pathwise delta for a fixed-strike arithmetic Asian option.

    For GBM paths, ∂S(t_i)/∂S₀ = S(t_i)/S₀. So:

    ∂payoff/∂S₀ = (1/n) Σ_i S(t_i)/S₀ × 1_{A > K}  (for call)
                = (1/n) Σ_i S(t_i)/S₀ × 1_{payoff > 0}

    This is exact (no bias) and uses the same paths as pricing.

    Args:
        paths: (n_paths, n_steps+1) including S(0).
        strike: option strike.
        spot: initial price S₀.
        rate: risk-free rate (for discounting).
        T: time to maturity.
        option_type: CALL or PUT.

    Returns:
        Pathwise delta estimate.
    """
    monitoring = paths[:, 1:]  # exclude t=0
    n_steps = monitoring.shape[1]
    arith_avg = monitoring.mean(axis=1)

    df = math.exp(-rate * T)

    # Indicator: payoff > 0
    if option_type == OptionType.CALL:
        in_the_money = arith_avg > strike
    else:
        in_the_money = arith_avg < strike

    # ∂A/∂S₀ = (1/n) Σ S(t_i)/S₀
    dA_dS0 = monitoring.mean(axis=1) / spot

    # ∂payoff/∂S₀ = ∂A/∂S₀ × indicator (call) or -∂A/∂S₀ × indicator (put)
    if option_type == OptionType.CALL:
        dpayoff = dA_dS0 * in_the_money
    else:
        dpayoff = -dA_dS0 * in_the_money

    return float(df * dpayoff.mean())


def pathwise_european_delta(
    terminals: np.ndarray,
    strike: float,
    spot: float,
    rate: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Pathwise delta for a European option.

    ∂payoff/∂S₀ = S(T)/S₀ × 1_{S(T) > K}  (call)

    Args:
        terminals: (n_paths,) terminal spot values.
        strike: option strike.
        spot: initial price.
    """
    df = math.exp(-rate * T)

    if option_type == OptionType.CALL:
        in_the_money = terminals > strike
    else:
        in_the_money = terminals < strike

    dS_dS0 = terminals / spot

    if option_type == OptionType.CALL:
        dpayoff = dS_dS0 * in_the_money
    else:
        dpayoff = -dS_dS0 * in_the_money

    return float(df * dpayoff.mean())
