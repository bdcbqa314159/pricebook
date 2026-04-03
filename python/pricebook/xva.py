"""XVA: CVA — exposure simulation and credit valuation adjustment."""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.pricing_context import PricingContext


# ---------------------------------------------------------------------------
# Exposure simulation
# ---------------------------------------------------------------------------


def simulate_exposures(
    pricer,
    ctx: PricingContext,
    time_grid: list[float],
    n_paths: int = 1000,
    rate_vol: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Simulate portfolio PV at future dates under diffused rates.

    Returns an (n_paths, n_times) array of mark-to-market values.

    Simple model: parallel rate shifts drawn from N(0, rate_vol * sqrt(dt)).
    """
    rng = np.random.default_rng(seed)
    n_times = len(time_grid)
    pvs = np.zeros((n_paths, n_times))

    for j, t in enumerate(time_grid):
        shifts = rng.normal(0, rate_vol * math.sqrt(t), n_paths)
        for i, shift in enumerate(shifts):
            bumped_curve = ctx.discount_curve.bumped(shift)
            bumped_ctx = PricingContext(
                valuation_date=ctx.valuation_date,
                discount_curve=bumped_curve,
                projection_curves=ctx.projection_curves,
                vol_surfaces=ctx.vol_surfaces,
                credit_curves=ctx.credit_curves,
                fx_spots=ctx.fx_spots,
            )
            pvs[i, j] = pricer(bumped_ctx)

    return pvs


def expected_positive_exposure(pvs: np.ndarray) -> np.ndarray:
    """EPE at each time point: E[max(V, 0)]."""
    return np.maximum(pvs, 0).mean(axis=0)


def expected_negative_exposure(pvs: np.ndarray) -> np.ndarray:
    """ENE at each time point: E[max(-V, 0)]."""
    return np.maximum(-pvs, 0).mean(axis=0)


def expected_exposure(pvs: np.ndarray) -> np.ndarray:
    """EE at each time point: E[V]."""
    return pvs.mean(axis=0)


# ---------------------------------------------------------------------------
# CVA
# ---------------------------------------------------------------------------


def cva(
    epe: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
) -> float:
    """Unilateral CVA = sum_i EPE(t_i) * df(t_i) * delta_PD(t_i) * (1-R).

    Args:
        epe: expected positive exposure at each time point.
        time_grid: year fractions from valuation date.
        discount_curve: risk-free discounting.
        survival_curve: counterparty survival curve.
        recovery: counterparty recovery rate.
    """
    ref = discount_curve.reference_date
    lgd = 1.0 - recovery
    result = 0.0

    for i, t in enumerate(time_grid):
        d = date.fromordinal(ref.toordinal() + int(t * 365))
        df = discount_curve.df(d)
        sp = survival_curve.survival(d)

        if i == 0:
            sp_prev = 1.0
        else:
            d_prev = date.fromordinal(ref.toordinal() + int(time_grid[i - 1] * 365))
            sp_prev = survival_curve.survival(d_prev)

        delta_pd = sp_prev - sp  # probability of default in bucket
        result += epe[i] * df * delta_pd * lgd

    return result
