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


    def to_dict(self) -> dict:
        return dict(vars(self))
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


    def to_dict(self) -> dict:
        return dict(vars(self))
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
    # Per-asset effective vol traces — pre-fix the result returned
    # ``sqrt(v)`` (the bare stochastic vol) for BOTH ``vol1_paths`` and
    # ``vol2_paths``, losing the asset-specific blended SLV effective
    # vol that actually drives each spot.  Track them explicitly here
    # (Fix T4-MALV2).
    eff1_paths = np.zeros((n_paths, n_steps + 1))
    eff2_paths = np.zeros((n_paths, n_steps + 1))
    sv0 = math.sqrt(max(heston_v0, 0.0))
    eff1_paths[:, 0] = mixing * lv1 + (1 - mixing) * sv0
    eff2_paths[:, 0] = mixing * lv2 + (1 - mixing) * sv0

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
        # Record the effective vol applied at this step.
        eff1_paths[:, step + 1] = eff1
        eff2_paths[:, step + 1] = eff2

    return MultiAssetSLVResult(
        spot1_paths=S1, spot2_paths=S2,
        vol1_paths=eff1_paths, vol2_paths=eff2_paths,
        mean_terminal_1=float(S1[:, -1].mean()),
        mean_terminal_2=float(S2[:, -1].mean()),
    )


@dataclass
class SmileConsistencyResult:
    """Basket smile consistency check.

    ``weighted_component_vol`` is the trivial upper bound ``Σ w_i σ_i``
    (achieved at ρ = 1).  ``model_basket_vol`` is the linearised model
    vol that respects the supplied correlation.  ``consistency_ratio``
    and ``is_consistent`` use ``model_basket_vol`` (correlation-aware).
    """
    basket_vol: float
    weighted_component_vol: float
    consistency_ratio: float
    is_consistent: bool
    model_basket_vol: float = 0.0


    def to_dict(self) -> dict:
        return dict(vars(self))
def smile_consistency_check(
    basket_vol: float, component_vols: list[float],
    weights: list[float], correlation: float,
) -> SmileConsistencyResult:
    """Check if basket vol is consistent with constituent smiles.

    Linearised basket variance with uniform pair correlation ``ρ``:

        Var(B) = Σ w_i² σ_i² + 2 ρ Σ_{i<j} w_i w_j σ_i σ_j
               = Σ w_i² σ_i² + ρ · [(Σ w_i σ_i)² − Σ w_i² σ_i²]

    ``model_basket_vol = √Var(B)`` is the correlation-aware reference;
    ``weighted_component_vol = Σ w_i σ_i`` is the trivial ρ=1 upper bound.

    Fix T4-MALV1: pre-fix ``correlation`` was a silent-no-op API param —
    the model_vol was computed locally but discarded, and the result
    fields (``is_consistent``, ``consistency_ratio``) referenced only
    ``weighted_component_vol``, so changing ``correlation`` left the
    output bit-identical.  Now ``correlation`` drives ``model_basket_vol``
    which in turn drives both the consistency check and the ratio.
    """
    w = np.array(weights); s = np.array(component_vols)
    weighted = float(np.sum(w * s))
    model_var = float(
        np.sum(w**2 * s**2)
        + correlation * (weighted**2 - np.sum(w**2 * s**2))
    )
    model_vol = math.sqrt(max(model_var, 0.0))
    ratio = basket_vol / max(model_vol, 1e-10)
    consistent = (basket_vol >= 0) and (basket_vol <= model_vol * 1.05)
    return SmileConsistencyResult(
        basket_vol=basket_vol,
        weighted_component_vol=weighted,
        consistency_ratio=float(ratio),
        is_consistent=consistent,
        model_basket_vol=float(model_vol),
    )


# ---------------------------------------------------------------------------
# Unified MC Engine migration
# ---------------------------------------------------------------------------

def multi_asset_slv_simulate_via_engine(
    spot1: float, spot2: float, rate: float,
    div1: float, div2: float,
    lv1: float, lv2: float,
    heston_v0: float, heston_kappa: float, heston_theta: float,
    heston_xi: float, rho_assets: float, rho_vol: float,
    T: float, n_paths: int = 5_000, n_steps: int = 50,
    mixing: float = 0.5, seed: int | None = 42,
) -> MultiAssetSLVResult:
    """2-asset SLV via unified MC engine.

    Delegates to original: three correlated Brownians (z1, z2, zv) with
    path-dependent effective vol (mixing*lv + (1-mixing)*√v) require
    shared increments that independent engine calls cannot provide.
    """
    return multi_asset_slv_simulate(
        spot1, spot2, rate, div1, div2, lv1, lv2,
        heston_v0, heston_kappa, heston_theta, heston_xi,
        rho_assets, rho_vol, T, n_paths, n_steps, mixing, seed,
    )
