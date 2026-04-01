"""
Advanced Monte Carlo variance reduction techniques.

- Stratified sampling: divide uniform space into strata for better coverage
- Importance sampling: shift drift to concentrate paths where payoff is large
- Multi-level Monte Carlo (MLMC): telescoping estimator with coarse+fine paths

    result = mc_stratified(spot=100, strike=105, rate=0.05, vol=0.20,
                           T=1.0, option_type=OptionType.CALL, n_paths=50_000)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from pricebook.black76 import OptionType
from pricebook.mc_pricer import MCResult


def mc_stratified(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    n_paths: int = 100_000,
    seed: int = 42,
) -> MCResult:
    """European option via stratified sampling.

    Divides [0,1] into n_paths equal strata, draws one uniform per stratum,
    then applies inverse normal CDF.
    """
    rng = np.random.default_rng(seed)

    # Stratified uniform samples
    u = (np.arange(n_paths) + rng.uniform(size=n_paths)) / n_paths
    z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))

    drift = (rate - div_yield - 0.5 * vol**2) * T
    diffusion = vol * math.sqrt(T)
    st = spot * np.exp(drift + diffusion * z)

    if option_type == OptionType.CALL:
        payoffs = np.maximum(st - strike, 0.0)
    else:
        payoffs = np.maximum(strike - st, 0.0)

    df = math.exp(-rate * T)
    discounted = df * payoffs
    price = float(discounted.mean())
    std_error = float(discounted.std(ddof=1) / math.sqrt(n_paths))

    return MCResult(price=price, std_error=std_error, n_paths=n_paths)


def mc_importance(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    n_paths: int = 100_000,
    seed: int = 42,
    shift: float | None = None,
) -> MCResult:
    """European option via importance sampling.

    Shifts the mean of the normal distribution toward the region where
    the payoff is large, then corrects with likelihood ratios.

    Args:
        shift: drift shift in standard deviations. If None, auto-computed
            to center on the strike.
    """
    rng = np.random.default_rng(seed)

    drift = (rate - div_yield - 0.5 * vol**2) * T
    diffusion = vol * math.sqrt(T)

    if shift is None:
        # Shift so that E[S_T] under the new measure = strike
        shift = (math.log(strike / spot) - drift) / diffusion

    z = rng.standard_normal(n_paths) + shift

    st = spot * np.exp(drift + diffusion * z)

    if option_type == OptionType.CALL:
        payoffs = np.maximum(st - strike, 0.0)
    else:
        payoffs = np.maximum(strike - st, 0.0)

    # Likelihood ratio: weight = phi_0(z) / phi_shift(z) = exp(-shift*z + shift^2/2)
    lr = np.exp(-shift * z + 0.5 * shift**2)

    df = math.exp(-rate * T)
    discounted = df * payoffs * lr
    price = float(discounted.mean())
    std_error = float(discounted.std(ddof=1) / math.sqrt(n_paths))

    return MCResult(price=price, std_error=std_error, n_paths=n_paths)


def mc_mlmc(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    n_levels: int = 5,
    n_paths_base: int = 50_000,
    seed: int = 42,
) -> MCResult:
    """European option via Multi-Level Monte Carlo.

    MLMC telescoping: E[P_L] = E[P_0] + sum_{l=1}^L E[P_l - P_{l-1}]

    Each level uses 2^l time steps. Coarser levels use more paths
    (variance is higher at fine levels but corrections are smaller).

    Args:
        n_levels: number of MLMC levels (L).
        n_paths_base: paths at the finest level (coarser levels get more).
    """
    rng_base = np.random.default_rng(seed)
    drift_rate = rate - div_yield - 0.5 * vol**2
    df = math.exp(-rate * T)

    total_price = 0.0
    total_var = 0.0
    total_paths = 0

    for level in range(n_levels + 1):
        n_steps = 2**level
        dt = T / n_steps

        # More paths at coarser levels (variance is higher for corrections)
        n_paths = n_paths_base * (2 ** (n_levels - level))
        n_paths = max(n_paths, 1000)

        seed_l = seed + level * 1000
        rng = np.random.default_rng(seed_l)
        z_all = rng.standard_normal((n_paths, n_steps))

        # Fine path: n_steps steps (vectorized)
        increments = drift_rate * dt + vol * math.sqrt(dt) * z_all
        log_s_fine = increments.sum(axis=1)
        s_fine = spot * np.exp(log_s_fine)

        if option_type == OptionType.CALL:
            payoff_fine = np.maximum(s_fine - strike, 0.0)
        else:
            payoff_fine = np.maximum(strike - s_fine, 0.0)

        if level == 0:
            Y = df * payoff_fine
        else:
            # Coarse path: pair consecutive fine increments (vectorized)
            z_pairs = z_all[:, 0::2] + z_all[:, 1::2]
            coarse_dt = 2 * dt
            coarse_increments = drift_rate * coarse_dt + vol * math.sqrt(coarse_dt) * z_pairs / math.sqrt(2)
            log_s_coarse = coarse_increments.sum(axis=1)
            s_coarse = spot * np.exp(log_s_coarse)

            if option_type == OptionType.CALL:
                payoff_coarse = np.maximum(s_coarse - strike, 0.0)
            else:
                payoff_coarse = np.maximum(strike - s_coarse, 0.0)

            Y = df * (payoff_fine - payoff_coarse)

        level_mean = float(Y.mean())
        level_var = float(Y.var(ddof=1))

        total_price += level_mean
        total_var += level_var / n_paths
        total_paths += n_paths

    std_error = math.sqrt(total_var)

    return MCResult(price=total_price, std_error=std_error, n_paths=total_paths)
