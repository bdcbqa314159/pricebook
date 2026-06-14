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

import numpy as np

from pricebook.models.black76 import OptionType, black76_price
from pricebook.models.gbm import GBMGenerator
from pricebook.models.mc_pricer import MCResult
from pricebook.statistics.rng import PseudoRandom


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

    Uses discrete monitoring at n equally spaced points
    ``t_i = i·T/n`` for ``i = 1..n`` (i.e. n random observations,
    NOT including the deterministic t_0 = 0).

    Fix T4-OPTIONS: pre-fix ``vol_g = σ·sqrt((2n+1)/(6(n+1)))`` matched
    the case of n+1 monitoring points *including* the deterministic
    t_0=0, while the drift formula and the MC monitoring (``paths[:, 1:]``
    in mc_asian_arithmetic) use n points starting at t_1.  The
    inconsistency biased σ_g LOW by a factor of n/(n+1) — about 7.7%
    for n=12.  As a control variate this produces a biased adjustment.

    Correct formula for n monitoring points (Kemna-Vorst):
        σ_g² = σ² · (n+1)(2n+1) / (6n²)
    """
    n = n_steps

    # Adjusted vol for geometric average (n monitoring points, t_i = i·T/n).
    vol_g = vol * math.sqrt((n + 1) * (2 * n + 1) / (6 * n * n))

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


# ---------------------------------------------------------------------------
# Unified MC Engine migration
# ---------------------------------------------------------------------------

def mc_asian_arithmetic_via_engine(
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
    use_sobol: bool = False,
) -> MCResult:
    """Asian arithmetic option priced via the unified MC engine.

    Drop-in replacement for mc_asian_arithmetic() using MCEngine.
    Supports antithetic, control variate, and Sobol.
    """
    from pricebook.models.mc_engine import MCEngine, TimeGrid
    from pricebook.models.mc_processes import BlackScholesProcess
    from pricebook.models.mc_payoffs import asian_arithmetic, asian_geometric

    process = BlackScholesProcess(spot, rate - div_yield, vol)
    grid = TimeGrid.uniform(T, n_steps)
    df = math.exp(-rate * T)

    if use_sobol:
        from pricebook.models.mc_extensions import sobol_engine
        engine = sobol_engine(process, grid, n_paths, seed)
    else:
        engine = MCEngine(process, grid, n_paths, seed, antithetic=antithetic)

    if option_type == OptionType.CALL:
        payoff = asian_arithmetic(strike, log_space=True)
    else:
        # Asian put
        def payoff(paths, times):
            p = paths[:, :, 0] if paths.ndim == 3 else paths
            spots = np.exp(p)
            avg = np.mean(spots[:, 1:], axis=1)
            return np.maximum(strike - avg, 0.0)

    if control_variate:
        from pricebook.models.mc_variance_reduction import control_variate as cv_fn
        geo_payoff = asian_geometric(strike, log_space=True)
        geo_exact = geometric_asian_analytical(spot, strike, rate, vol, T, n_steps, option_type, div_yield)
        result = cv_fn(engine, payoff, geo_payoff, geo_exact, df)
    else:
        result = engine.price(payoff, df)

    # Convert to old MCResult format
    return MCResult(price=result.price, std_error=result.stderr, n_paths=result.n_paths)
