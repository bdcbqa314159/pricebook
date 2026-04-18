"""Correlation monitoring: implied vs realised, term structure, arb, stress.

* :func:`implied_vs_realised_correlation` — spread tracking.
* :func:`correlation_term_structure` — short vs long implied ρ.
* :func:`correlation_stress_matrix` — stressed correlation scenarios.
* :func:`multi_asset_smile_arb_check` — basket vs constituent smile arb.

References:
    Bossu, *Advanced Equity Derivatives*, Wiley, 2014.
    Alexander, *Market Risk Analysis*, Vol. IV, Wiley, 2008.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class ImpliedRealisedCorrResult:
    implied_correlation: float
    realised_correlation: float
    spread: float
    z_score: float
    historical_spreads: np.ndarray | None

def implied_vs_realised_correlation(
    implied: float, realised: float,
    historical_spreads: list[float] | None = None,
) -> ImpliedRealisedCorrResult:
    """Track implied − realised correlation spread."""
    spread = implied - realised
    if historical_spreads and len(historical_spreads) > 1:
        arr = np.array(historical_spreads)
        z = (spread - arr.mean()) / max(arr.std(), 1e-10)
    else:
        z = 0.0
    return ImpliedRealisedCorrResult(implied, realised, float(spread), float(z),
                                      np.array(historical_spreads) if historical_spreads else None)


@dataclass
class CorrTermStructureResult:
    tenors: list[float]
    implied_correlations: list[float]
    is_inverted: bool          # short > long (unusual)
    slope: float               # long − short

def correlation_term_structure(
    tenors: list[float], implied_correlations: list[float],
) -> CorrTermStructureResult:
    """Term structure of implied correlation."""
    slope = implied_correlations[-1] - implied_correlations[0] if len(implied_correlations) > 1 else 0.0
    inverted = slope < -0.05
    return CorrTermStructureResult(tenors, implied_correlations, inverted, float(slope))


@dataclass
class CorrStressResult:
    base_correlation: np.ndarray
    stressed_correlation: np.ndarray
    scenario_name: str
    max_shift: float

def correlation_stress_matrix(
    base_corr: np.ndarray, scenario: str = "uniform_up",
    shift: float = 0.2,
) -> CorrStressResult:
    """Apply stress to correlation matrix.
    Scenarios: uniform_up, uniform_down, tail_stress, sector_decorrelation.
    """
    n = base_corr.shape[0]
    stressed = base_corr.copy()
    if scenario == "uniform_up":
        for i in range(n):
            for j in range(i+1, n):
                stressed[i, j] = min(base_corr[i, j] + shift, 0.999)
                stressed[j, i] = stressed[i, j]
    elif scenario == "uniform_down":
        for i in range(n):
            for j in range(i+1, n):
                stressed[i, j] = max(base_corr[i, j] - shift, -0.999)
                stressed[j, i] = stressed[i, j]
    elif scenario == "tail_stress":
        # In tail: all correlations go to 1
        for i in range(n):
            for j in range(i+1, n):
                stressed[i, j] = min(base_corr[i, j] + 2 * shift, 0.999)
                stressed[j, i] = stressed[i, j]
    elif scenario == "sector_decorrelation":
        # Reduce inter-sector correlation
        for i in range(n):
            for j in range(i+1, n):
                stressed[i, j] = base_corr[i, j] * (1 - shift)
                stressed[j, i] = stressed[i, j]
    max_shift_val = float(np.abs(stressed - base_corr).max())
    return CorrStressResult(base_corr, stressed, scenario, max_shift_val)


@dataclass
class SmileArbCheckResult:
    is_arbitrage_free: bool
    basket_vol: float
    model_vol: float
    residual: float

def multi_asset_smile_arb_check(
    basket_vol: float, component_vols: list[float],
    weights: list[float], implied_correlation: float,
) -> SmileArbCheckResult:
    """Check if basket vol is consistent with constituent smiles + implied ρ.
    Model vol² = Σ w²σ² + ρ × (Σ wσ)² − Σ w²σ²).
    Arb-free if basket_vol ≈ model_vol (within tolerance).
    """
    w = np.array(weights); s = np.array(component_vols)
    diag = float(np.sum(w**2 * s**2))
    off = float((np.sum(w * s))**2 - np.sum(w**2 * s**2))
    model_var = diag + implied_correlation * off
    model_vol = math.sqrt(max(model_var, 0))
    residual = abs(basket_vol - model_vol)
    arb_free = residual < 0.02  # within 2 vol points
    return SmileArbCheckResult(arb_free, basket_vol, float(model_vol), float(residual))
