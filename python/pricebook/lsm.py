"""
Longstaff-Schwartz Monte Carlo for American/Bermudan options.

At each exercise date, regress continuation value on basis functions
of the state variable (spot price). Exercise if intrinsic > continuation.

    result = lsm_american(spot=100, strike=105, rate=0.05, vol=0.20,
                          T=1.0, n_steps=50, n_paths=100_000,
                          option_type=OptionType.PUT)
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.black76 import OptionType
from pricebook.gbm import GBMGenerator
from pricebook.mc_pricer import MCResult
from pricebook.rng import PseudoRandom


def lsm_american(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_steps: int = 50,
    n_paths: int = 100_000,
    option_type: OptionType = OptionType.PUT,
    div_yield: float = 0.0,
    seed: int = 42,
    n_basis: int = 3,
) -> MCResult:
    """American option price via Longstaff-Schwartz.

    Args:
        n_steps: number of exercise dates (equally spaced).
        n_paths: number of simulation paths.
        option_type: CALL or PUT.
        n_basis: number of polynomial basis functions for regression.
        seed: random seed.
    """
    gen = GBMGenerator(spot=spot, rate=rate, vol=vol, div_yield=div_yield)
    rng = PseudoRandom(seed=seed)
    paths = gen.generate(T=T, n_steps=n_steps, n_paths=n_paths, rng=rng)

    dt = T / n_steps
    df = math.exp(-rate * dt)

    # Payoff function
    if option_type == OptionType.PUT:
        def payoff(s): return np.maximum(strike - s, 0.0)
    else:
        def payoff(s): return np.maximum(s - strike, 0.0)

    # Cash flows: what each path receives and when
    cashflow = payoff(paths[:, -1])  # terminal payoff
    cashflow_time = np.full(n_paths, n_steps)  # when it's received

    # Backward induction
    for step in range(n_steps - 1, 0, -1):
        s = paths[:, step]
        exercise_value = payoff(s)

        # Only consider paths that are in the money
        itm = exercise_value > 0
        if itm.sum() < n_basis + 1:
            continue

        # Discounted future cashflows for ITM paths
        s_itm = s[itm]
        steps_to_cf = cashflow_time[itm] - step
        discounted_cf = cashflow[itm] * np.power(df, steps_to_cf)

        # Regression: continuation value = f(S)
        # Basis: 1, S, S^2, ... (normalized)
        s_norm = s_itm / spot  # normalize for numerical stability
        basis = np.column_stack([s_norm**k for k in range(n_basis)])

        try:
            coeffs = np.linalg.lstsq(basis, discounted_cf, rcond=None)[0]
            continuation = basis @ coeffs
        except np.linalg.LinAlgError:
            continue

        # Exercise decision: exercise if intrinsic > continuation
        exercise = exercise_value[itm] > continuation
        exercise_idx = np.where(itm)[0][exercise]

        cashflow[exercise_idx] = exercise_value[exercise_idx]
        cashflow_time[exercise_idx] = step

    # Discount all cashflows to time 0
    steps_to_cf = cashflow_time
    discounted = cashflow * np.power(df, steps_to_cf)

    price = float(discounted.mean())
    std_error = float(discounted.std(ddof=1) / math.sqrt(n_paths))

    return MCResult(price=price, std_error=std_error, n_paths=n_paths)
