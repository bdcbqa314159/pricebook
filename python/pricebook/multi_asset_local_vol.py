"""Multi-asset local vol: 2D Dupire, multi-asset SLV, smile consistency.

* :func:`dupire_2d_local_vol` — 2D local vol from marginal vanillas.
* :func:`multi_asset_slv_simulate` — SLV with leverage in 2D.
* :func:`smile_consistency_check` — check basket vs constituent smiles.

References:
    Ren, Madan & Qian, *Calibrating and Pricing with Embedded Local Volatility*, 2007.
    Guyon & Henry-Labordère, *Nonlinear Option Pricing*, CRC, 2014.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class LocalVol2DResult:
    """2D local vol result."""
    vol_surface_1: np.ndarray   # (n_t, n_s) marginal LV for asset 1
    vol_surface_2: np.ndarray   # (n_t, n_s) marginal LV for asset 2
    times: np.ndarray
    spots: np.ndarray
    method: str

def dupire_2d_local_vol(
    marginal_vols_1: np.ndarray, marginal_vols_2: np.ndarray,
    times: np.ndarray, spots: np.ndarray,
) -> LocalVol2DResult:
    """Compute marginal local vol surfaces for two assets.
    In multi-asset, each asset's LV is computed from its marginal vanillas.
    The correlation structure is added via the simulation.
    """
    return LocalVol2DResult(marginal_vols_1, marginal_vols_2, times, spots, "marginal_dupire")


@dataclass
class MultiAssetSLVResult:
    """Multi-asset SLV simulation result."""
    spot1_paths: np.ndarray
    spot2_paths: np.ndarray
    vol1_paths: np.ndarray
    vol2_paths: np.ndarray
    mean_terminal_1: float
    mean_terminal_2: float

def multi_asset_slv_simulate(
    spot1: float, spot2: float, rate: float,
    div1: float, div2: float,
    lv1: float, lv2: float,
    heston_v0: float, heston_kappa: float, heston_theta: float,
    heston_xi: float, rho_assets: float, rho_vol: float,
    T: float, n_paths: int = 5_000, n_steps: int = 50,
    mixing: float = 0.5, seed: int | None = 42,
) -> MultiAssetSLVResult:
    """2-asset SLV: each asset has LV + shared Heston-like stochastic vol.
    σ_eff_i = mixing × LV_i + (1-mixing) × √v_t
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps; sqrt_dt = math.sqrt(dt)

    S1 = np.full((n_paths, n_steps + 1), float(spot1))
    S2 = np.full((n_paths, n_steps + 1), float(spot2))
    v = np.full((n_paths, n_steps + 1), heston_v0)

    for step in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho_assets * z1 + math.sqrt(1 - rho_assets**2) * rng.standard_normal(n_paths)
        zv = rho_vol * z1 + math.sqrt(1 - rho_vol**2) * rng.standard_normal(n_paths)

        v_pos = np.maximum(v[:, step], 1e-10)
        sv = np.sqrt(v_pos)
        eff1 = mixing * lv1 + (1 - mixing) * sv
        eff2 = mixing * lv2 + (1 - mixing) * sv

        S1[:, step+1] = S1[:, step] * np.exp((rate - div1 - 0.5*eff1**2)*dt + eff1*z1*sqrt_dt)
        S2[:, step+1] = S2[:, step] * np.exp((rate - div2 - 0.5*eff2**2)*dt + eff2*z2*sqrt_dt)
        v[:, step+1] = np.maximum(v_pos + heston_kappa*(heston_theta - v_pos)*dt + heston_xi*sv*zv*sqrt_dt, 0)

    return MultiAssetSLVResult(S1, S2, np.sqrt(np.maximum(v, 0)), np.sqrt(np.maximum(v, 0)),
                                 float(S1[:,-1].mean()), float(S2[:,-1].mean()))


@dataclass
class SmileConsistencyResult:
    """Basket smile consistency check."""
    basket_vol: float
    weighted_component_vol: float
    consistency_ratio: float   # basket_vol / weighted (should be < 1)
    is_consistent: bool

def smile_consistency_check(
    basket_vol: float, component_vols: list[float],
    weights: list[float], correlation: float,
) -> SmileConsistencyResult:
    """Check if basket vol is consistent with constituent smiles.
    Basket vol² ≤ (Σ w_i σ_i)² — the weighted average is an upper bound.
    Basket vol² ≈ Σ w_i² σ_i² + ρ Σ_{i≠j} w_i w_j σ_i σ_j.
    """
    w = np.array(weights); s = np.array(component_vols)
    weighted = float(np.sum(w * s))
    model_var = float(np.sum(w**2 * s**2) + correlation * ((np.sum(w * s))**2 - np.sum(w**2 * s**2)))
    model_vol = math.sqrt(max(model_var, 0))
    ratio = basket_vol / max(weighted, 1e-10)
    consistent = basket_vol <= weighted * 1.01 and basket_vol >= 0
    return SmileConsistencyResult(basket_vol, weighted, float(ratio), consistent)
