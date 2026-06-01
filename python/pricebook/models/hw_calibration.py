"""Hull-White model calibration from swaption volatilities.

Fits the Hull-White parameters (a, sigma) by minimising the difference
between model-implied and market swaption prices/vols.

    from pricebook.models.hw_calibration import (
        calibrate_hull_white, HWCalibrationResult,
    )

    hw = calibrate_hull_white(curve, swaption_vols).model

References:
    Brigo & Mercurio (2006). Interest Rate Models, Ch. 3.
    Hull & White (1990). Pricing Interest-Rate Derivative Securities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import minimize

from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.hull_white import HullWhite


@dataclass
class HWCalibrationResult:
    """Result of Hull-White calibration."""
    model: HullWhite
    a: float
    sigma: float
    rmse_vol: float                          # RMSE in vol terms
    per_swaption_errors: list[dict]          # per-instrument fit diagnostics
    n_swaptions: int
    converged: bool

    def to_dict(self) -> dict:
        return {
            "a": self.a, "sigma": self.sigma,
            "rmse_vol": self.rmse_vol,
            "n_swaptions": self.n_swaptions,
            "converged": self.converged,
        }


def _hw_swaption_price(
    a: float,
    sigma: float,
    curve: DiscountCurve,
    expiry_years: float,
    tenor_years: float,
    strike: float,
    is_payer: bool = True,
    n_steps: int = 50,
) -> float:
    """Price a European swaption under HW with given params."""
    try:
        hw = HullWhite(a=max(a, 1e-4), sigma=max(sigma, 1e-6), curve=curve)
        swap_end = expiry_years + tenor_years
        return hw.tree_european_swaption(expiry_years, swap_end, strike,
                                          n_steps=n_steps, is_payer=is_payer)
    except Exception:
        return 0.0


def _hw_implied_vol(
    a: float,
    sigma: float,
    curve: DiscountCurve,
    expiry_years: float,
    tenor_years: float,
    strike: float,
    n_steps: int = 50,
) -> float:
    """Compute HW-implied Black vol for a swaption."""
    from pricebook.models.black76 import black76_price, OptionType
    from pricebook.options.implied_vol import implied_vol_black76
    from pricebook.core.day_count import date_from_year_fraction, DayCountConvention, year_fraction

    ref = curve.reference_date
    price = _hw_swaption_price(a, sigma, curve, expiry_years, tenor_years,
                                 strike, True, n_steps)
    if price <= 0:
        return 0.0

    # Forward swap rate and annuity for Black vol inversion
    swap_end = expiry_years + tenor_years
    df_expiry = curve.df(date_from_year_fraction(ref, expiry_years))
    df_end = curve.df(date_from_year_fraction(ref, swap_end))

    n_payments = max(1, int(tenor_years))
    annuity = 0.0
    for k in range(1, n_payments + 1):
        t_pay = expiry_years + k
        if t_pay <= swap_end:
            annuity += curve.df(date_from_year_fraction(ref, t_pay))

    if annuity <= 0:
        return 0.0

    fwd_swap = (df_expiry - df_end) / annuity
    if fwd_swap <= 0:
        fwd_swap = strike

    try:
        iv = implied_vol_black76(
            price / annuity,  # normalise to Black price per unit annuity
            fwd_swap, strike, expiry_years, 1.0, OptionType.CALL,
        )
        return iv
    except Exception:
        return 0.0


def calibrate_hull_white(
    curve: DiscountCurve,
    swaption_vols: dict[tuple[float, float], float],
    strike: float | None = None,
    method: str = "nelder_mead",
    a_bounds: tuple[float, float] = (0.001, 0.50),
    sigma_bounds: tuple[float, float] = (0.001, 0.10),
    n_steps: int = 50,
) -> HWCalibrationResult:
    """Calibrate Hull-White (a, sigma) from ATM swaption vols.

    Minimises sum of squared differences between market and model
    implied volatilities across a grid of swaptions.

    Args:
        curve: initial discount curve.
        swaption_vols: {(expiry_years, tenor_years): black_vol}.
            Example: {(1, 5): 0.0065, (5, 5): 0.0055, (10, 10): 0.0045}
        strike: ATM strike. If None, computed from forward swap rates.
        method: "nelder_mead" or "differential_evolution".
        a_bounds: bounds for mean reversion parameter.
        sigma_bounds: bounds for volatility parameter.
        n_steps: tree steps for swaption pricing.

    Returns:
        HWCalibrationResult with calibrated HullWhite model.
    """
    from pricebook.core.day_count import date_from_year_fraction

    ref = curve.reference_date
    keys = list(swaption_vols.keys())
    market_vols = [swaption_vols[k] for k in keys]

    # Compute ATM strikes if not provided
    if strike is None:
        # Use forward swap rate as ATM
        strikes = {}
        for (exp_y, tenor_y) in keys:
            swap_end = exp_y + tenor_y
            df_exp = curve.df(date_from_year_fraction(ref, exp_y))
            df_end = curve.df(date_from_year_fraction(ref, swap_end))
            n_pay = max(1, int(tenor_y))
            ann = sum(curve.df(date_from_year_fraction(ref, exp_y + k))
                      for k in range(1, n_pay + 1) if exp_y + k <= swap_end)
            strikes[(exp_y, tenor_y)] = (df_exp - df_end) / ann if ann > 0 else 0.04
    else:
        strikes = {k: strike for k in keys}

    def objective(params):
        a, sig = params[0], params[1]
        if a <= 0 or sig <= 0:
            return 1e6
        total = 0.0
        for i, (exp_y, tenor_y) in enumerate(keys):
            k = strikes[(exp_y, tenor_y)]
            model_vol = _hw_implied_vol(a, sig, curve, exp_y, tenor_y, k, n_steps)
            total += (model_vol - market_vols[i]) ** 2
        return total

    if method == "nelder_mead":
        result = minimize(objective, [0.03, 0.01], method="Nelder-Mead",
                           options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-10})
    elif method == "differential_evolution":
        from scipy.optimize import differential_evolution
        bounds = [a_bounds, sigma_bounds]
        result = differential_evolution(objective, bounds, maxiter=200, tol=1e-8)
    else:
        result = minimize(objective, [0.03, 0.01], method="L-BFGS-B",
                           bounds=[a_bounds, sigma_bounds])

    a_opt, sigma_opt = max(result.x[0], 1e-4), max(result.x[1], 1e-6)
    hw = HullWhite(a=a_opt, sigma=sigma_opt, curve=curve)

    # Compute per-swaption diagnostics
    errors = []
    for i, (exp_y, tenor_y) in enumerate(keys):
        k = strikes[(exp_y, tenor_y)]
        model_vol = _hw_implied_vol(a_opt, sigma_opt, curve, exp_y, tenor_y, k, n_steps)
        errors.append({
            "expiry": exp_y, "tenor": tenor_y,
            "market_vol": market_vols[i], "model_vol": model_vol,
            "error_bp": (model_vol - market_vols[i]) * 10_000,
        })

    rmse = math.sqrt(sum(e["error_bp"]**2 for e in errors) / len(errors)) if errors else 0

    return HWCalibrationResult(
        model=hw, a=a_opt, sigma=sigma_opt,
        rmse_vol=rmse / 10_000,
        per_swaption_errors=errors,
        n_swaptions=len(keys),
        converged=result.success if hasattr(result, 'success') else True,
    )
