"""Vol risk premium: implied-realised decomposition, VRP term structure.

* :func:`vrp_single_asset` — implied − realised for one asset class.
* :func:`vrp_term_structure` — VRP across tenors.
* :func:`cross_asset_vrp_comparison` — compare VRP across asset classes.
* :func:`vrp_strategy_signal` — sell vol when VRP is high.

References:
    Carr & Wu, *Variance Risk Premiums*, RFS, 2009.
    Bollerslev, Tauchen & Zhou, *Expected Stock Returns and Variance Risk Premia*, RFS, 2009.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class VRPResult:
    asset_class: str
    implied_vol: float
    realised_vol: float
    vrp_vol: float              # implied − realised (vol terms)
    vrp_var: float              # implied² − realised² (variance terms)
    vrp_ratio: float            # implied / realised

def vrp_single_asset(asset_class: str, implied_vol: float, realised_vol: float) -> VRPResult:
    vrp_v = implied_vol - realised_vol
    vrp_var = implied_vol**2 - realised_vol**2
    ratio = implied_vol / max(realised_vol, 1e-10)
    return VRPResult(asset_class, implied_vol, realised_vol, float(vrp_v), float(vrp_var), float(ratio))


@dataclass
class VRPTermStructureResult:
    tenors: list[float]
    vrp_by_tenor: list[float]
    slope: float                # long − short VRP
    is_inverted: bool

def vrp_term_structure(tenors: list[float], implied_vols: list[float],
                        realised_vol: float) -> VRPTermStructureResult:
    vrps = [iv - realised_vol for iv in implied_vols]
    slope = vrps[-1] - vrps[0] if len(vrps) > 1 else 0
    return VRPTermStructureResult(tenors, [float(v) for v in vrps], float(slope), slope < -0.005)


@dataclass
class CrossAssetVRPResult:
    rankings: list[tuple[str, float]]   # (asset_class, VRP) sorted descending
    highest_vrp: str
    lowest_vrp: str
    spread: float

def cross_asset_vrp_comparison(
    vrps: dict[str, tuple[float, float]],   # {asset_class: (implied, realised)}
) -> CrossAssetVRPResult:
    entries = [(ac, iv - rv) for ac, (iv, rv) in vrps.items()]
    entries.sort(key=lambda x: -x[1])
    return CrossAssetVRPResult(
        entries, entries[0][0], entries[-1][0],
        float(entries[0][1] - entries[-1][1]))


@dataclass
class VRPSignalResult:
    signal: str
    vrp_level: float
    z_score: float
    recommended_action: str

def vrp_strategy_signal(
    current_vrp: float, historical_vrps: list[float],
) -> VRPSignalResult:
    arr = np.array(historical_vrps)
    mu = float(arr.mean()); sigma = float(arr.std())
    z = (current_vrp - mu) / max(sigma, 1e-10)
    if z > 1.5: signal, action = "very_high", "sell_vol_aggressively"
    elif z > 0.5: signal, action = "high", "sell_vol"
    elif z > -0.5: signal, action = "neutral", "no_action"
    elif z > -1.5: signal, action = "low", "buy_vol"
    else: signal, action = "very_low", "buy_vol_aggressively"
    return VRPSignalResult(signal, current_vrp, float(z), action)
