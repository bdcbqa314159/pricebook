"""Vol surface stress testing: bumps, historical replays, cross-asset.

* :func:`parallel_vol_bump` — uniform shift across surface.
* :func:`tilt_vol_bump` — tilt (slope change) in vol term structure.
* :func:`twist_vol_bump` — butterfly bump (curvature change).
* :func:`vol_scenario_replay` — replay historical vol moves.
* :func:`cross_asset_vol_stress` — correlated vol bumps across classes.

References:
    Alexander, *Market Risk Analysis*, Vol. IV, Wiley, 2008.
    FRTB SA, *Sensitivities-Based Method for Vega Risk*, BCBS, 2019.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class VolBumpResult:
    bumped_vols: np.ndarray
    pnl_estimate: float
    bump_type: str
    bump_size: float

def parallel_vol_bump(
    base_vols: list[float], vega_ladder: list[float], bump_bps: float = 100,
) -> VolBumpResult:
    """Parallel vol bump: shift all vols by bump_bps basis points."""
    shift = bump_bps / 10000
    bumped = np.array(base_vols) + shift
    pnl = sum(v * shift for v in vega_ladder)
    return VolBumpResult(bumped, float(pnl), "parallel", bump_bps)


def tilt_vol_bump(
    tenors: list[float], base_vols: list[float], vega_ladder: list[float],
    tilt_bps: float = 50,
) -> VolBumpResult:
    """Tilt: short-end up, long-end down (steepening)."""
    T = np.array(tenors); T_mid = (T[0] + T[-1]) / 2
    weights = (T - T_mid) / max(T[-1] - T[0], 1e-10)
    shift = tilt_bps / 10000 * weights
    bumped = np.array(base_vols) + shift
    pnl = sum(v * s for v, s in zip(vega_ladder, shift))
    return VolBumpResult(bumped, float(pnl), "tilt", tilt_bps)


def twist_vol_bump(
    tenors: list[float], base_vols: list[float], vega_ladder: list[float],
    twist_bps: float = 30,
) -> VolBumpResult:
    """Twist/butterfly: wings up, belly down."""
    T = np.array(tenors); T_mid = (T[0] + T[-1]) / 2
    weights = ((T - T_mid) / max(T[-1] - T[0], 1e-10))**2 - 0.25
    shift = twist_bps / 10000 * weights
    bumped = np.array(base_vols) + shift
    pnl = sum(v * s for v, s in zip(vega_ladder, shift))
    return VolBumpResult(bumped, float(pnl), "twist", twist_bps)


@dataclass
class VolReplayResult:
    scenario_name: str
    vol_changes: np.ndarray
    pnl: float
    max_vol_change: float

def vol_scenario_replay(
    scenario_name: str,
    historical_vol_changes: list[float],
    vega_ladder: list[float],
) -> VolReplayResult:
    """Replay a historical vol scenario on the current book."""
    changes = np.array(historical_vol_changes)
    n = min(len(changes), len(vega_ladder))
    pnl = sum(v * c for v, c in zip(vega_ladder[:n], changes[:n]))
    return VolReplayResult(scenario_name, changes, float(pnl),
                             float(np.abs(changes).max()))


@dataclass
class CrossAssetVolStressResult:
    per_asset_pnl: dict[str, float]
    total_pnl: float
    worst_asset: str
    diversification_benefit: float

def cross_asset_vol_stress(
    asset_vegas: dict[str, list[float]],
    bumps_bps: dict[str, float],
    correlations: dict[tuple[str, str], float] | None = None,
) -> CrossAssetVolStressResult:
    """Correlated vol bump across asset classes."""
    per_asset = {}
    for asset, vegas in asset_vegas.items():
        bump = bumps_bps.get(asset, 100) / 10000
        pnl = sum(v * bump for v in vegas)
        per_asset[asset] = float(pnl)

    total = sum(per_asset.values())
    worst = min(per_asset, key=lambda a: per_asset[a])
    gross = sum(abs(v) for v in per_asset.values())
    div_benefit = gross - abs(total) if gross > 0 else 0.0

    return CrossAssetVolStressResult(per_asset, float(total), worst, float(div_benefit))
