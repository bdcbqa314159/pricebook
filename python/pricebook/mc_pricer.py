"""
Monte Carlo pricer for European options.

Prices European call/put via simulation of GBM terminal values,
with optional variance reduction (antithetic variates, control variate).

    result = mc_european(
        spot=100, strike=105, rate=0.05, vol=0.20, T=1.0,
        option_type=OptionType.CALL, n_paths=100_000,
    )
    print(f"Price: {result.price:.4f} ± {result.std_error:.4f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.black76 import OptionType, black76_price
from pricebook.gbm import GBMGenerator
from pricebook.rng import PseudoRandom, QuasiRandom


@dataclass
class MCResult:
    """Result of a Monte Carlo pricing run.

    Attributes:
        price: estimated option price.
        std_error: standard error of the estimate.
        n_paths: effective number of paths used.
    """

    price: float
    std_error: float
    n_paths: int


def mc_european(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    n_paths: int = 100_000,
    seed: int = 42,
    antithetic: bool = False,
    control_variate: bool = False,
) -> MCResult:
    """
    Price a European option via Monte Carlo.

    Args:
        spot: initial price.
        strike: option strike.
        rate: risk-free rate.
        vol: lognormal volatility.
        T: time to expiry in years.
        option_type: CALL or PUT.
        div_yield: continuous dividend yield.
        n_paths: number of simulation paths.
        seed: random seed.
        antithetic: use antithetic variates.
        control_variate: use the analytical Black-76 price as control.

    Returns:
        MCResult with price, standard error, and effective path count.
    """
    gen = GBMGenerator(spot=spot, rate=rate, vol=vol, div_yield=div_yield)
    rng = PseudoRandom(seed=seed)
    st = gen.terminal(T=T, n_paths=n_paths, rng=rng, antithetic=antithetic)

    if option_type == OptionType.CALL:
        payoffs = np.maximum(st - strike, 0.0)
    else:
        payoffs = np.maximum(strike - st, 0.0)

    df = math.exp(-rate * T)

    if control_variate:
        forward = spot * math.exp((rate - div_yield) * T)
        analytical = black76_price(forward, strike, vol, T, df, option_type)
        control = st - forward

        discounted_payoffs = df * payoffs
        cov = np.cov(discounted_payoffs, control)[0, 1]
        var_control = np.var(control)
        beta = cov / var_control if var_control > 0 else 0.0

        adjusted = discounted_payoffs - beta * control
        price = float(adjusted.mean())
        std_error = float(adjusted.std(ddof=1) / math.sqrt(len(adjusted)))
    else:
        discounted = df * payoffs
        price = float(discounted.mean())
        std_error = float(discounted.std(ddof=1) / math.sqrt(len(discounted)))

    return MCResult(price=price, std_error=std_error, n_paths=len(st))
