"""Calibration of jump/Lévy models to implied volatility surfaces.

Fits model parameters by minimising the difference between model-implied
and market-implied volatilities using COS pricing + scipy optimisation.

    from pricebook.models.jump_calibration import (
        calibrate_jump_model, calibrate_jump_surface, jump_model_comparison,
    )

    result = calibrate_jump_model("merton", strikes, vols, spot, rate, T)

References:
    Cont & Tankov (2004). Financial Modelling with Jump Processes.
    Schoutens (2003). Lévy Processes in Finance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize as scipy_minimize

from pricebook.models.black76 import OptionType, black76_price
from pricebook.models.cos_method import cos_price
from pricebook.options.implied_vol import implied_vol_black76
from pricebook.models.char_func_protocol import (
    merton_char_func, vg_char_func, kou_char_func, bates_char_func,
)
from pricebook.models.levy_processes import nig_char_func, cgmy_char_func


# ═══════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════

@dataclass
class JumpCalibrationResult:
    """Result of fitting a jump model to market implied vols."""
    model_type: str
    params: dict
    rmse_vol: float                    # RMSE in vol terms (e.g. 0.005 = 0.5 vol pt)
    market_vols: list[float]
    model_vols: list[float]
    strikes: list[float]
    n_params: int

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type, "params": self.params,
            "rmse_vol": self.rmse_vol, "n_params": self.n_params,
        }


@dataclass
class ModelComparisonResult:
    """Comparison of multiple jump models on the same data."""
    results: list[JumpCalibrationResult]
    ranking: list[str]             # model names sorted by AIC
    aic_values: dict[str, float]

    def to_dict(self) -> dict:
        return {"ranking": self.ranking, "aic": self.aic_values}


# ═══════════════════════════════════════════════════════════════
# Model parameter specs
# ═══════════════════════════════════════════════════════════════

_MODEL_SPECS = {
    "merton": {
        "names": ["sigma", "lam", "mu_j", "sigma_j"],
        "bounds": [(0.05, 0.80), (0.1, 10.0), (-0.5, 0.1), (0.01, 0.5)],
        "build_cf": lambda p, rate, T, **kw: merton_char_func(rate, p[0], p[1], p[2], p[3], T),
    },
    "vg": {
        "names": ["sigma", "nu", "theta"],
        "bounds": [(0.05, 0.80), (0.01, 2.0), (-0.5, 0.1)],
        "build_cf": lambda p, rate, T, **kw: vg_char_func(p[0], p[1], p[2], rate, T),
    },
    "kou": {
        "names": ["sigma", "lam", "p", "eta1", "eta2"],
        "bounds": [(0.05, 0.80), (0.1, 10.0), (0.1, 0.9), (2.0, 50.0), (2.0, 50.0)],
        "build_cf": lambda p, rate, T, **kw: kou_char_func(rate, p[0], T, p[1], p[2], p[3], p[4], kw.get("div_yield", 0.0)),
    },
    "nig": {
        "names": ["alpha", "beta", "delta"],
        "bounds": [(1.0, 50.0), (-25.0, 5.0), (0.01, 2.0)],
        "build_cf": lambda p, rate, T, **kw: nig_char_func(p[0], p[1], p[2], rate, T),
    },
    "cgmy": {
        "names": ["C", "G", "M", "Y"],
        "bounds": [(0.1, 20.0), (1.0, 50.0), (1.0, 50.0), (-0.5, 1.9)],
        "build_cf": lambda p, rate, T, **kw: cgmy_char_func(p[0], p[1], p[2], p[3], rate, T),
    },
    "bates": {
        "names": ["v0", "kappa", "theta", "xi", "rho", "lam", "mu_j", "sigma_j"],
        "bounds": [(0.01, 0.25), (0.1, 10.0), (0.01, 0.25), (0.05, 1.0),
                   (-0.95, 0.0), (0.0, 5.0), (-0.3, 0.05), (0.01, 0.3)],
        "build_cf": lambda p, rate, T, **kw: bates_char_func(rate, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], T, kw.get("div_yield", 0.0)),
    },
}


# ═══════════════════════════════════════════════════════════════
# Core calibration
# ═══════════════════════════════════════════════════════════════

def _cos_implied_vol(
    phi: Callable,
    spot: float,
    strike: float,
    rate: float,
    T: float,
    div_yield: float = 0.0,
) -> float:
    """Price via COS, then invert to implied vol."""
    price = cos_price(phi, spot, strike, rate, T, OptionType.CALL, div_yield, N=128)
    if price <= 0:
        return 0.0
    fwd = spot * math.exp((rate - div_yield) * T)
    df = math.exp(-rate * T)
    intrinsic = max(df * (fwd - strike), 0)
    if price <= intrinsic + 1e-10:
        return 0.0
    try:
        return implied_vol_black76(price, fwd, strike, T, df, OptionType.CALL)
    except Exception:
        return 0.0


def calibrate_jump_model(
    model_type: str,
    strikes: list[float],
    market_vols: list[float],
    spot: float,
    rate: float,
    T: float,
    div_yield: float = 0.0,
    maxiter: int = 200,
    seed: int = 42,
) -> JumpCalibrationResult:
    """Calibrate a jump model to market implied vols at a single expiry.

    Args:
        model_type: one of "merton", "vg", "kou", "nig", "cgmy", "bates".
        strikes: list of strike prices.
        market_vols: corresponding implied vols.
        spot: current spot price.
        rate: risk-free rate.
        T: time to expiry.
        div_yield: continuous dividend yield.
        maxiter: max iterations for differential evolution.
        seed: random seed for optimiser.

    Returns:
        JumpCalibrationResult with fitted params, RMSE, model vols.
    """
    if model_type not in _MODEL_SPECS:
        raise ValueError(f"Unknown model '{model_type}'. Available: {list(_MODEL_SPECS.keys())}")

    spec = _MODEL_SPECS[model_type]
    bounds = spec["bounds"]
    param_names = spec["names"]
    n_strikes = len(strikes)

    # Constraint for NIG: alpha > |beta|
    def nig_constraint(params):
        if model_type == "nig":
            return params[0] - abs(params[1]) - 0.1  # alpha > |beta| + margin
        return 1.0

    def objective(params):
        if model_type == "nig" and params[0] <= abs(params[1]):
            return 1e6
        try:
            phi = spec["build_cf"](params, rate, T, div_yield=div_yield)
        except (ValueError, ZeroDivisionError, OverflowError):
            return 1e6

        sse = 0.0
        for i in range(n_strikes):
            model_vol = _cos_implied_vol(phi, spot, strikes[i], rate, T, div_yield)
            sse += (model_vol - market_vols[i]) ** 2
        return sse

    # Global search
    result = differential_evolution(
        objective, bounds, maxiter=maxiter, seed=seed,
        tol=1e-8, atol=1e-10, polish=True,
    )

    best_params = result.x

    # Compute final model vols
    phi = spec["build_cf"](best_params, rate, T, div_yield=div_yield)
    model_vols = [_cos_implied_vol(phi, spot, k, rate, T, div_yield) for k in strikes]
    rmse = math.sqrt(sum((mv - mkv)**2 for mv, mkv in zip(model_vols, market_vols)) / n_strikes)

    params_dict = dict(zip(param_names, best_params.tolist()))

    return JumpCalibrationResult(
        model_type=model_type,
        params=params_dict,
        rmse_vol=rmse,
        market_vols=list(market_vols),
        model_vols=model_vols,
        strikes=list(strikes),
        n_params=len(param_names),
    )


# ═══════════════════════════════════════════════════════════════
# Multi-expiry calibration
# ═══════════════════════════════════════════════════════════════

def calibrate_jump_surface(
    model_type: str,
    market_data: list[dict],
    spot: float,
    rate: float,
    div_yield: float = 0.0,
    maxiter: int = 300,
    seed: int = 42,
) -> list[JumpCalibrationResult]:
    """Calibrate a jump model across multiple expiries independently.

    Args:
        market_data: list of {"T": float, "strikes": [...], "vols": [...]}.

    Returns:
        List of JumpCalibrationResult, one per expiry.
    """
    results = []
    for md in market_data:
        r = calibrate_jump_model(
            model_type, md["strikes"], md["vols"], spot, rate,
            md["T"], div_yield, maxiter, seed,
        )
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# Model comparison
# ═══════════════════════════════════════════════════════════════

def jump_model_comparison(
    strikes: list[float],
    market_vols: list[float],
    spot: float,
    rate: float,
    T: float,
    div_yield: float = 0.0,
    models: list[str] | None = None,
    maxiter: int = 200,
    seed: int = 42,
) -> ModelComparisonResult:
    """Fit multiple jump models and rank by AIC.

    AIC = n·log(MSE) + 2·k, where k = number of parameters.
    Lower AIC = better (penalises overfitting).

    Args:
        models: list of model types to compare. Default: all.
    """
    if models is None:
        models = ["merton", "vg", "kou", "nig", "cgmy"]  # exclude bates (slow)

    n = len(strikes)
    results = []
    aic_values = {}

    for m in models:
        r = calibrate_jump_model(m, strikes, market_vols, spot, rate, T,
                                  div_yield, maxiter, seed)
        results.append(r)

        mse = r.rmse_vol ** 2
        aic = n * math.log(max(mse, 1e-20)) + 2 * r.n_params
        aic_values[m] = aic

    ranking = sorted(aic_values, key=lambda m: aic_values[m])

    return ModelComparisonResult(results, ranking, aic_values)
