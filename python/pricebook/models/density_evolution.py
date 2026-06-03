"""Density evolution toolkit: forward + backward Kolmogorov + Fourier.

Three routes to the same risk-neutral density — cross-validate.

* :func:`density_three_ways` — compute density via FP, backward PDE, Fourier.
* :func:`cross_validate_density` — compare all three methods.

References:
    Shreve, *Stochastic Calculus for Finance II*, Ch. 6.
    Dupire, *Pricing with a Smile*, Risk, 1994.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DensityComparisonResult:
    """Cross-validation of density from multiple methods."""
    grid: np.ndarray
    density_fp: np.ndarray          # Fokker-Planck (forward PDE)
    density_fourier: np.ndarray     # Fourier inversion of CF
    density_breeden: np.ndarray     # Breeden-Litzenberger from option prices
    max_diff_fp_fourier: float
    max_diff_fp_breeden: float
    consistent: bool                # all agree within tolerance

    def to_dict(self) -> dict:
        return {
            "max_diff_fp_fourier": self.max_diff_fp_fourier,
            "max_diff_fp_breeden": self.max_diff_fp_breeden,
            "consistent": self.consistent,
        }


def density_three_ways(
    spot: float,
    rate: float,
    vol: float,
    T: float,
    div_yield: float = 0.0,
    n_points: int = 200,
) -> DensityComparisonResult:
    """Compute risk-neutral density via three independent methods.

    1. Fokker-Planck (forward PDE evolution)
    2. Fourier inversion of BS characteristic function
    3. Breeden-Litzenberger from BS call prices (d²C/dK²)

    If all three agree, the implementation is cross-validated.

    Args:
        n_points: grid resolution.
    """
    # Common grid in S-space
    S_min = spot * 0.2
    S_max = spot * 3.0
    S_grid = np.linspace(S_min, S_max, n_points)

    # 1. Fokker-Planck
    from pricebook.models.fokker_planck import fokker_planck_1d
    fp = fokker_planck_1d(spot, rate, vol, T, div_yield, n_space=300, n_time=300)
    density_fp = np.interp(S_grid, fp.grid, fp.density)

    # 2. Fourier inversion
    from pricebook.models.cos_method import bs_char_func
    from pricebook.models.fft_pricing import density_from_cf
    cf = bs_char_func(rate, div_yield, vol, T)
    x_grid = np.log(S_grid / spot)
    density_x = density_from_cf(cf, x_grid)
    # Convert log-density to S-density: p_S = p_x / S
    density_fourier = density_x / S_grid
    density_fourier = np.maximum(density_fourier, 0)
    # Normalise
    mass = float(np.trapezoid(density_fourier, S_grid))
    if mass > 0:
        density_fourier /= mass

    # 3. Breeden-Litzenberger from BS prices
    from pricebook.models.black76 import black76_price, OptionType
    fwd = spot * math.exp((rate - div_yield) * T)
    df = math.exp(-rate * T)
    density_bl = np.zeros(n_points)
    dK = S_grid[1] - S_grid[0]
    for i in range(1, n_points - 1):
        K = S_grid[i]
        c_up = black76_price(fwd, K + dK, vol, T, df, OptionType.CALL)
        c_mid = black76_price(fwd, K, vol, T, df, OptionType.CALL)
        c_dn = black76_price(fwd, K - dK, vol, T, df, OptionType.CALL)
        density_bl[i] = math.exp(rate * T) * (c_up - 2 * c_mid + c_dn) / (dK**2)
    density_bl = np.maximum(density_bl, 0)
    mass_bl = float(np.trapezoid(density_bl, S_grid))
    if mass_bl > 0:
        density_bl /= mass_bl

    # Compare
    # Only compare in the bulk (exclude tails where densities are tiny)
    bulk = (density_fp > 1e-6) | (density_fourier > 1e-6)
    if np.any(bulk):
        diff_ff = float(np.max(np.abs(density_fp[bulk] - density_fourier[bulk])))
        diff_fb = float(np.max(np.abs(density_fp[bulk] - density_bl[bulk])))
    else:
        diff_ff = 0.0
        diff_fb = 0.0

    # Consistent if max differences are small relative to peak density
    peak = max(float(np.max(density_fp)), 1e-6)
    consistent = diff_ff / peak < 0.15 and diff_fb / peak < 0.15

    return DensityComparisonResult(
        grid=S_grid,
        density_fp=density_fp,
        density_fourier=density_fourier,
        density_breeden=density_bl,
        max_diff_fp_fourier=diff_ff,
        max_diff_fp_breeden=diff_fb,
        consistent=consistent,
    )
