"""Calibration robustness utilities.

Tikhonov regularisation, parameter bounds enforcement, quality metrics,
and multi-start optimisation for stable model calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from pricebook.optimization import minimize


@dataclass
class CalibrationResult:
    """Result of a robust calibration."""

    params: list[float]
    objective: float
    rmse: float
    n_evaluations: int
    converged: bool
    condition_number: float = 0.0
    residuals: list[float] = field(default_factory=list)
    method: str = ""


def tikhonov_regularise(
    objective: Callable,
    prior: list[float],
    lambda_reg: float = 0.01,
) -> Callable:
    """Add Tikhonov regularisation to an objective function.

    Penalises deviation from prior parameters:
        obj_reg(x) = obj(x) + lambda * ||x - prior||^2
    """
    prior_arr = np.array(prior)

    def regularised(params):
        base = objective(params)
        penalty = lambda_reg * np.sum((np.array(params) - prior_arr) ** 2)
        return base + penalty

    return regularised


def enforce_bounds(
    params: list[float],
    bounds: list[tuple[float, float]],
) -> list[float]:
    """Clip parameters to bounds."""
    return [max(lo, min(hi, p)) for p, (lo, hi) in zip(params, bounds)]


def calibration_quality(
    model_func: Callable,
    params: list[float],
    targets: list[float],
) -> dict[str, float]:
    """Compute calibration quality metrics.

    Args:
        model_func: function(params) → list of model values.
        params: calibrated parameters.
        targets: market target values.

    Returns:
        dict with rmse, max_error, mean_error, residuals.
    """
    model_vals = model_func(params)
    residuals = [m - t for m, t in zip(model_vals, targets)]
    sq_residuals = [r**2 for r in residuals]

    rmse = math.sqrt(sum(sq_residuals) / len(residuals))
    max_err = max(abs(r) for r in residuals)
    mean_err = sum(abs(r) for r in residuals) / len(residuals)

    return {
        "rmse": rmse,
        "max_error": max_err,
        "mean_error": mean_err,
        "residuals": residuals,
    }


def multi_start_calibrate(
    objective: Callable,
    bounds: list[tuple[float, float]],
    n_starts: int = 10,
    method: str = "nelder_mead",
    seed: int = 42,
    **kwargs,
) -> CalibrationResult:
    """Multi-start optimisation: run from multiple random starting points.

    Picks the best result across all starts to avoid local minima.
    """
    rng = np.random.default_rng(seed)
    best_result = None
    best_obj = float("inf")
    total_evals = 0

    for _ in range(n_starts):
        x0 = [rng.uniform(lo, hi) for lo, hi in bounds]
        try:
            result = minimize(objective, x0=x0, method=method,
                              bounds=bounds, **kwargs)
            total_evals += result.iterations
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return CalibrationResult(
            params=[0.5 * (lo + hi) for lo, hi in bounds],
            objective=float("inf"), rmse=float("inf"),
            n_evaluations=total_evals, converged=False, method=method,
        )

    params = enforce_bounds(list(best_result.x), bounds)
    rmse = math.sqrt(best_obj / max(1, len(bounds)))

    return CalibrationResult(
        params=params,
        objective=best_obj,
        rmse=rmse,
        n_evaluations=total_evals,
        converged=best_result.converged,
        method=method,
    )


def perturbation_stability(
    calibrate_func: Callable,
    base_inputs: list[float],
    perturbation: float = 0.01,
    n_trials: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Test calibration stability under perturbed inputs.

    Runs calibration with small random perturbations to inputs
    and measures parameter variance.

    Args:
        calibrate_func: function(inputs) → params (list of floats).
        base_inputs: original market inputs.
        perturbation: relative perturbation size.
        n_trials: number of perturbed calibrations.

    Returns:
        dict with param_mean, param_std, max_deviation.
    """
    rng = np.random.default_rng(seed)
    all_params = []

    base_params = calibrate_func(base_inputs)
    all_params.append(base_params)

    for _ in range(n_trials):
        perturbed = [v * (1.0 + rng.uniform(-perturbation, perturbation))
                     for v in base_inputs]
        try:
            params = calibrate_func(perturbed)
            all_params.append(params)
        except Exception:
            continue

    arr = np.array(all_params)
    return {
        "param_mean": arr.mean(axis=0).tolist(),
        "param_std": arr.std(axis=0).tolist(),
        "max_deviation": float(np.max(np.abs(arr - arr[0]))),
        "n_successful": len(all_params),
    }
