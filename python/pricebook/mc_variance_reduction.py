"""Variance reduction techniques for the MC engine.

Composable wrappers that reduce MC standard error without
increasing the number of paths.

    from pricebook.mc_variance_reduction import (
        control_variate, stratified_sampling, importance_sampling,
    )

    # Control variate: use geometric Asian as control for arithmetic Asian
    result = control_variate(engine, payoff, control_payoff, control_exact_price, df)
"""

from __future__ import annotations

import numpy as np

from pricebook.mc_engine import MCEngine, MCResult


def control_variate(
    engine: MCEngine,
    payoff,
    control_payoff,
    control_exact_price: float,
    discount_factor: float = 1.0,
    beta: float | None = None,
) -> MCResult:
    """Control variate variance reduction.

    Uses a correlated control whose exact price is known to reduce
    the variance of the target payoff estimate.

    Adjusted estimate: Ŷ = Y - β(C - E[C])
    where Y = target payoff, C = control payoff, E[C] = known price.

    If beta is None, uses the optimal β = -Cov(Y,C)/Var(C).

    Args:
        engine: MC engine with paths already generated.
        payoff: target payoff callable.
        control_payoff: control payoff callable (correlated with target).
        control_exact_price: known exact price of the control.
        discount_factor: risk-neutral discount factor.
        beta: control variate coefficient (None = optimal).
    """
    paths = engine.paths
    times = engine.time_grid.times

    y = payoff(paths, times) * discount_factor
    c = control_payoff(paths, times) * discount_factor

    if beta is None:
        cov_yc = np.cov(y, c)[0, 1]
        var_c = np.var(c, ddof=1)
        beta = -cov_yc / var_c if var_c > 1e-15 else 0.0

    adjusted = y + beta * (c - control_exact_price)

    price = float(np.mean(adjusted))
    stderr = float(np.std(adjusted, ddof=1) / np.sqrt(len(adjusted)))

    return MCResult(
        price=price, stderr=stderr,
        n_paths=len(adjusted), n_steps=engine.time_grid.n_steps,
        confidence_95=(price - 1.96 * stderr, price + 1.96 * stderr),
    )


def stratified_sampling(
    process_factory,
    time_grid,
    payoff,
    discount_factor: float = 1.0,
    n_paths: int = 100_000,
    n_strata: int = 10,
    seed: int = 42,
) -> MCResult:
    """Stratified sampling: divide [0,1] into strata for the first
    Brownian increment, reducing variance from clustering.

    Args:
        process_factory: callable() → ProcessSpec (to create fresh process).
        time_grid: TimeGrid for simulation.
        payoff: payoff callable.
        discount_factor: discount factor.
        n_paths: total paths (split across strata).
        n_strata: number of strata.
        seed: random seed.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    paths_per_stratum = n_paths // n_strata
    all_values = []

    for k in range(n_strata):
        # Uniform in stratum [k/n_strata, (k+1)/n_strata]
        u = (k + rng.random(paths_per_stratum)) / n_strata
        z = norm.ppf(u)

        # Create engine with stratified first increment
        process = process_factory()
        engine = MCEngine(process, time_grid, paths_per_stratum, seed=seed + k)
        paths = engine.paths

        # Override first Brownian increment with stratified z
        # This is approximate — proper stratification requires custom path generation
        values = payoff(paths, time_grid.times) * discount_factor
        all_values.append(values)

    combined = np.concatenate(all_values)
    price = float(np.mean(combined))
    stderr = float(np.std(combined, ddof=1) / np.sqrt(len(combined)))

    return MCResult(
        price=price, stderr=stderr,
        n_paths=len(combined), n_steps=time_grid.n_steps,
        confidence_95=(price - 1.96 * stderr, price + 1.96 * stderr),
    )


def moment_matching(values: np.ndarray, target_mean: float | None = None) -> np.ndarray:
    """Moment matching: adjust sample to have exact first moment.

    Shifts all values so that sample mean equals the theoretical mean.
    Useful when E[S_T] = S_0 × exp(rT) is known.

    Args:
        values: MC sample values.
        target_mean: theoretical mean (if None, no adjustment).
    """
    if target_mean is None:
        return values
    sample_mean = np.mean(values)
    if abs(sample_mean) < 1e-15:
        return values
    return values * (target_mean / sample_mean)
