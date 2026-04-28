"""Spread dynamics for FVA: stochastic funding spread simulation.

Wires StochasticBasis into the XVA framework to compute FVA with
stochastic funding spread paths, capturing the convexity adjustment
between deterministic and stochastic spread assumptions.

    from pricebook.spread_dynamics import fva_with_spread_dynamics

    result = fva_with_spread_dynamics(
        epe, time_grid, ois_curve, stochastic_basis,
        initial_spread=0.005)

References:
    Piterbarg, V. (2010). Funding beyond discounting.
    Burgard & Kjaer (2011). PDE approach to CVA with funding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.discount_curve import DiscountCurve
from pricebook.day_count import date_from_year_fraction
from pricebook.rfr import StochasticBasis
from pricebook.survival_curve import SurvivalCurve
from pricebook.xva import fva as fva_deterministic, total_xva_decomposition, TotalXVAResult


@dataclass
class SpreadDynamicsResult:
    """Result of FVA computation with spread dynamics."""
    fva_deterministic: float    # FVA with constant spread = E[s]
    fva_stochastic: float       # FVA with simulated spread paths
    convexity_adjustment: float  # stochastic - deterministic
    mean_spread_path: list[float]
    n_paths: int


def fva_with_spread_dynamics(
    epe: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    stochastic_basis: StochasticBasis,
    initial_spread: float,
    n_spread_paths: int = 1000,
    seed: int | None = 42,
) -> SpreadDynamicsResult:
    """FVA with stochastic funding spread.

    Instead of FVA = Σ EPE(t) × s × dt × df(t) with constant s,
    simulates spread paths s(t) and computes:

    FVA_stochastic = E[ Σ EPE(t) × s(t) × dt × df(t) ]

    The difference (convexity adjustment) arises because E[EPE × s] ≠ E[EPE] × E[s]
    when spread and exposure are correlated, or simply from Jensen's inequality.

    Args:
        epe: expected positive exposure profile, shape (n_times,).
        time_grid: time points in years.
        discount_curve: OIS curve for discounting.
        stochastic_basis: OU process for funding spread.
        initial_spread: current funding spread.
        n_spread_paths: number of MC paths for spread simulation.
        seed: random seed.
    """
    n_times = len(time_grid)
    T = time_grid[-1]

    # Simulate spread paths: shape (n_paths, n_steps + 1)
    spread_paths = stochastic_basis.simulate(
        s0=initial_spread, T=T, n_steps=n_times, n_paths=n_spread_paths,
    )

    # Deterministic FVA with mean spread
    mean_spread = stochastic_basis.stationary_mean()
    fva_det = fva_deterministic(epe, time_grid, discount_curve, mean_spread)

    # Stochastic FVA: average over paths
    ref = discount_curve.reference_date
    dts = [time_grid[0]] + [
        time_grid[i] - time_grid[i - 1] for i in range(1, n_times)
    ]

    path_fvas = np.zeros(n_spread_paths)
    for p in range(n_spread_paths):
        pf = 0.0
        for j in range(n_times):
            d = date_from_year_fraction(ref, time_grid[j])
            df = discount_curve.df(d)
            # spread_paths has n_steps+1 columns; skip index 0 (initial)
            s_t = max(spread_paths[p, j + 1], 0.0)  # floor at 0
            pf += epe[j] * s_t * dts[j] * df
        path_fvas[p] = pf

    fva_stoch = float(path_fvas.mean())

    # Mean spread path
    mean_path = [float(spread_paths[:, j + 1].mean()) for j in range(n_times)]

    return SpreadDynamicsResult(
        fva_deterministic=fva_det,
        fva_stochastic=fva_stoch,
        convexity_adjustment=fva_stoch - fva_det,
        mean_spread_path=mean_path,
        n_paths=n_spread_paths,
    )


def xva_with_spread_dynamics(
    epe: np.ndarray,
    ene: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    cpty_survival: SurvivalCurve,
    own_survival: SurvivalCurve,
    stochastic_basis: StochasticBasis,
    initial_spread: float,
    n_spread_paths: int = 1000,
    seed: int | None = 42,
    **xva_kwargs,
) -> TotalXVAResult:
    """Total XVA decomposition with stochastic funding spread.

    Computes CVA, DVA, etc. deterministically, but replaces FVA
    with the stochastic spread result.
    """
    # Base XVA with deterministic spread
    mean_spread = stochastic_basis.stationary_mean()
    base = total_xva_decomposition(
        epe=epe, ene=ene, time_grid=time_grid,
        discount_curve=discount_curve,
        cpty_survival=cpty_survival,
        own_survival=own_survival,
        funding_spread=mean_spread,
        **xva_kwargs,
    )

    # Replace FVA with stochastic version
    spread_result = fva_with_spread_dynamics(
        epe, time_grid, discount_curve,
        stochastic_basis, initial_spread,
        n_spread_paths, seed,
    )

    return TotalXVAResult(
        cva=base.cva, dva=base.dva,
        cfa=base.cfa, dfa=base.dfa,
        colva=base.colva,
        fva_val=spread_result.fva_stochastic,
        mva_val=base.mva_val,
        kva_val=base.kva_val,
    )
