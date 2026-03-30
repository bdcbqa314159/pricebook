"""
Asian option pricing via Monte Carlo.

Arithmetic average price options have no closed form — Monte Carlo is
required. Geometric average options DO have a closed form (under GBM),
making them an ideal control variate for the arithmetic version.

Fixed strike: payoff = max(A - K, 0) for call, max(K - A, 0) for put
Floating strike: payoff = max(S(T) - A, 0) for call, max(A - S(T), 0) for put

where A is the average price over the monitoring period.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.black76 import OptionType, black76_price
from pricebook.gbm import GBMGenerator
from pricebook.mc_pricer import MCResult
from pricebook.rng import PseudoRandom


def geometric_asian_analytical(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_steps: int,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> float:
    """
    Closed-form price of a geometric average Asian option under GBM.

    The geometric average of GBM is itself lognormal, so Black-76 applies
    with adjusted forward and volatility.

    Uses discrete monitoring at n_steps equally spaced points.
    """
    dt = T / n_steps
    n = n_steps

    # Adjusted vol for geometric average
    vol_g = vol * math.sqrt((2 * n + 1) / (6 * (n + 1)))

    # Adjusted drift for geometric average
    mu = rate - div_yield
    drift_g = (mu - 0.5 * vol**2) * (n + 1) / (2 * n) + 0.5 * vol_g**2

    # Geometric average forward
    forward_g = spot * math.exp(drift_g * T)

    df = math.exp(-rate * T)

    return black76_price(forward_g, strike, vol_g, T, df, option_type)


def mc_asian_arithmetic(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_steps: int,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    n_paths: int = 100_000,
    seed: int = 42,
    antithetic: bool = False,
    control_variate: bool = False,
    floating_strike: bool = False,
) -> MCResult:
    """
    Price an arithmetic average Asian option via Monte Carlo.

    Args:
        spot: initial price.
        strike: fixed strike (ignored if floating_strike=True).
        rate: risk-free rate.
        vol: lognormal volatility.
        T: time to expiry.
        n_steps: number of averaging points.
        option_type: CALL or PUT.
        div_yield: continuous dividend yield.
        n_paths: number of simulation paths.
        seed: random seed.
        antithetic: use antithetic variates.
        control_variate: use geometric Asian as control variate.
        floating_strike: if True, payoff uses S(T) - A (call) or A - S(T) (put).
    """
    gen = GBMGenerator(spot=spot, rate=rate, vol=vol, div_yield=div_yield)
    rng = PseudoRandom(seed=seed)
    paths = gen.generate(T=T, n_steps=n_steps, n_paths=n_paths,
                         rng=rng, antithetic=antithetic)

    # Average over monitoring points (exclude time 0)
    monitoring = paths[:, 1:]  # shape (n_eff, n_steps)
    arith_avg = monitoring.mean(axis=1)

    df = math.exp(-rate * T)

    if floating_strike:
        terminal = paths[:, -1]
        if option_type == OptionType.CALL:
            payoffs = np.maximum(terminal - arith_avg, 0.0)
        else:
            payoffs = np.maximum(arith_avg - terminal, 0.0)
    else:
        if option_type == OptionType.CALL:
            payoffs = np.maximum(arith_avg - strike, 0.0)
        else:
            payoffs = np.maximum(strike - arith_avg, 0.0)

    discounted = df * payoffs

    if control_variate:
        # Geometric average as control variate
        geom_avg = np.exp(np.log(monitoring).mean(axis=1))

        if floating_strike:
            terminal = paths[:, -1]
            if option_type == OptionType.CALL:
                geom_payoffs = np.maximum(terminal - geom_avg, 0.0)
            else:
                geom_payoffs = np.maximum(geom_avg - terminal, 0.0)
        else:
            if option_type == OptionType.CALL:
                geom_payoffs = np.maximum(geom_avg - strike, 0.0)
            else:
                geom_payoffs = np.maximum(strike - geom_avg, 0.0)

        geom_discounted = df * geom_payoffs
        geom_analytical = geometric_asian_analytical(
            spot, strike, rate, vol, T, n_steps, option_type, div_yield,
        )

        # Optimal beta
        cov = np.cov(discounted, geom_discounted)[0, 1]
        var_g = np.var(geom_discounted)
        beta = cov / var_g if var_g > 0 else 0.0

        adjusted = discounted - beta * (geom_discounted - geom_analytical)
        price = float(adjusted.mean())
        std_error = float(adjusted.std(ddof=1) / math.sqrt(len(adjusted)))
    else:
        price = float(discounted.mean())
        std_error = float(discounted.std(ddof=1) / math.sqrt(len(discounted)))

    return MCResult(price=price, std_error=std_error, n_paths=len(payoffs))
