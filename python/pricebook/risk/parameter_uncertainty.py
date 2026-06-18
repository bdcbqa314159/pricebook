"""Parameter uncertainty quantification.

Bootstrap confidence intervals on calibrated parameters, sensitivity
ladders, and joint parameter surfaces.

    from pricebook.risk.parameter_uncertainty import (
        ParameterBand, calibration_uncertainty, sensitivity_ladder,
    )

References:
    Efron & Tibshirani (1993). An Introduction to the Bootstrap.
    Cont (2006). Model Uncertainty and its Impact on the Pricing of
        Derivative Instruments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ParameterBand:
    """Confidence interval for a calibrated parameter."""
    name: str
    base: float
    low: float
    high: float
    confidence: float  # e.g. 0.95

    def width(self) -> float:
        return self.high - self.low

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class SensitivityEntry:
    """PV sensitivity to a parameter at its band edges."""
    param_name: str
    base_pv: float
    low_pv: float
    high_pv: float
    impact: float       # max(|low_pv - base|, |high_pv - base|)

    def to_dict(self) -> dict:
        return dict(vars(self))


def calibration_uncertainty(
    pricer: Callable,
    base_params: dict[str, float],
    market_data: np.ndarray,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int = 42,
) -> list[ParameterBand]:
    """Estimate parameter confidence intervals via bootstrap.

    Resamples market_data with replacement, re-calibrates each time,
    and computes percentile-based confidence intervals.

    Args:
        pricer: callable(params_dict, data) → calibrated_params_dict.
            Takes market data, returns fitted parameters.
        base_params: baseline calibrated parameters.
        market_data: 1D or 2D array of market observations.
        n_bootstrap: number of bootstrap replications.
        confidence: confidence level (default 95%).
        seed: random seed.

    Returns:
        List of ParameterBand, one per parameter.
    """
    rng = np.random.default_rng(seed)
    n = len(market_data)
    alpha = (1 - confidence) / 2

    # Bootstrap replications
    bootstrap_params = {name: [] for name in base_params}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled = market_data[idx]
        try:
            fitted = pricer(base_params, resampled)
            for name in base_params:
                bootstrap_params[name].append(fitted[name])
        except Exception:
            continue

    bands = []
    for name, base_val in base_params.items():
        samples = np.array(bootstrap_params[name])
        if len(samples) < 10:
            bands.append(ParameterBand(name, base_val, base_val, base_val, confidence))
            continue
        lo = float(np.percentile(samples, alpha * 100))
        hi = float(np.percentile(samples, (1 - alpha) * 100))
        bands.append(ParameterBand(name, base_val, lo, hi, confidence))

    return bands


def sensitivity_ladder(
    pricer: Callable,
    base_params: dict[str, float],
    bands: list[ParameterBand],
) -> list[SensitivityEntry]:
    """PV impact of each parameter at its confidence band edges.

    Args:
        pricer: callable(params_dict) → float (PV).
        base_params: baseline parameters.
        bands: parameter confidence bands.

    Returns:
        List of SensitivityEntry sorted by impact (largest first).
    """
    base_pv = pricer(base_params)

    entries = []
    for band in bands:
        # Price at low end
        params_lo = dict(base_params)
        params_lo[band.name] = band.low
        pv_lo = pricer(params_lo)

        # Price at high end
        params_hi = dict(base_params)
        params_hi[band.name] = band.high
        pv_hi = pricer(params_hi)

        impact = max(abs(pv_lo - base_pv), abs(pv_hi - base_pv))
        entries.append(SensitivityEntry(band.name, base_pv, pv_lo, pv_hi, impact))

    return sorted(entries, key=lambda e: e.impact, reverse=True)


def joint_parameter_surface(
    pricer: Callable,
    base_params: dict[str, float],
    param1_band: ParameterBand,
    param2_band: ParameterBand,
    n_grid: int = 10,
) -> dict:
    """2D PV surface over two parameter bands.

    Returns dict with param1_values, param2_values, pv_surface (n_grid × n_grid).
    """
    p1_vals = np.linspace(param1_band.low, param1_band.high, n_grid)
    p2_vals = np.linspace(param2_band.low, param2_band.high, n_grid)

    surface = np.zeros((n_grid, n_grid))
    for i, v1 in enumerate(p1_vals):
        for j, v2 in enumerate(p2_vals):
            params = dict(base_params)
            params[param1_band.name] = v1
            params[param2_band.name] = v2
            surface[i, j] = pricer(params)

    return {
        "param1": param1_band.name,
        "param2": param2_band.name,
        "param1_values": p1_vals.tolist(),
        "param2_values": p2_vals.tolist(),
        "pv_surface": surface.tolist(),
        "base_pv": pricer(base_params),
    }
