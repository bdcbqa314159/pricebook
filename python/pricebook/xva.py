"""XVA: CVA, DVA, bilateral CVA, FVA, MVA, KVA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.pricing_context import PricingContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _time_to_date(ref: date, t: float) -> date:
    return date.fromordinal(ref.toordinal() + int(t * 365))


def _time_grid_dts(time_grid: list[float]) -> list[float]:
    return [time_grid[0]] + [
        time_grid[i] - time_grid[i - 1] for i in range(1, len(time_grid))
    ]


def _default_leg(
    exposure: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    lgd: float,
) -> float:
    """Shared computation for CVA and DVA: sum_i exposure_i * df_i * delta_PD_i * lgd."""
    ref = discount_curve.reference_date
    result = 0.0
    sp_prev = 1.0

    for i, t in enumerate(time_grid):
        d = _time_to_date(ref, t)
        df = discount_curve.df(d)
        sp = survival_curve.survival(d)
        delta_pd = sp_prev - sp
        result += exposure[i] * df * delta_pd * lgd
        sp_prev = sp

    return result


def _discounted_integral(
    profile: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    spread: float,
) -> float:
    """Shared computation for FVA, MVA, KVA: sum_i profile_i * spread * dt_i * df_i."""
    ref = discount_curve.reference_date
    dts = _time_grid_dts(time_grid)
    result = 0.0

    for i, t in enumerate(time_grid):
        d = _time_to_date(ref, t)
        df = discount_curve.df(d)
        result += profile[i] * spread * dts[i] * df

    return result


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
# CVA / DVA
# ---------------------------------------------------------------------------


def cva(
    epe: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
) -> float:
    """Unilateral CVA = sum_i EPE(t_i) * df(t_i) * delta_PD(t_i) * (1-R)."""
    return _default_leg(epe, time_grid, discount_curve, survival_curve, 1.0 - recovery)


def dva(
    ene: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    own_survival: SurvivalCurve,
    own_recovery: float = 0.4,
) -> float:
    """DVA = sum_i ENE(t_i) * df(t_i) * delta_PD_own(t_i) * (1-R_own)."""
    return _default_leg(ene, time_grid, discount_curve, own_survival, 1.0 - own_recovery)


def bilateral_cva(cva_val: float, dva_val: float) -> float:
    """BCVA = CVA - DVA."""
    return cva_val - dva_val


# ---------------------------------------------------------------------------
# FVA / MVA / KVA
# ---------------------------------------------------------------------------


def fva(
    ee: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    funding_spread: float,
) -> float:
    """FVA = sum_i EE(t_i) * funding_spread * dt_i * df(t_i)."""
    return _discounted_integral(ee, time_grid, discount_curve, funding_spread)


def mva(
    im_profile: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    funding_spread: float,
) -> float:
    """MVA = sum_i IM(t_i) * funding_spread * dt_i * df(t_i)."""
    return _discounted_integral(im_profile, time_grid, discount_curve, funding_spread)


def kva(
    capital_profile: np.ndarray,
    time_grid: list[float],
    discount_curve: DiscountCurve,
    hurdle_rate: float,
) -> float:
    """KVA = sum_i K(t_i) * hurdle_rate * dt_i * df(t_i)."""
    return _discounted_integral(capital_profile, time_grid, discount_curve, hurdle_rate)


# ---------------------------------------------------------------------------
# Total XVA
# ---------------------------------------------------------------------------


@dataclass
class XVAResult:
    """Aggregated XVA components."""

    cva: float = 0.0
    dva: float = 0.0
    fva: float = 0.0
    mva: float = 0.0
    kva: float = 0.0

    @property
    def bcva(self) -> float:
        return self.cva - self.dva

    @property
    def total(self) -> float:
        return self.cva - self.dva + self.fva + self.mva + self.kva
