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
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationResult,
    CanonicalCalibrationResult,
    SolveReport,
    model_calibration_record,
)
from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.hull_white import HullWhite

if TYPE_CHECKING:
    from pricebook.market_data import MarketSnapshot


@dataclass
class HWCalibrationResult(CanonicalCalibrationResult):
    """Result of Hull-White calibration.

    `calibration_result` carries the canonical provenance artefact; the
    other fields are HW-specific outputs (the model itself, per-swaption
    diagnostics). The `to_calibration_result()` method returns the stored
    instance when populated, or builds one on-demand from the existing
    fields (back-compat path for hand-constructed instances).
    """
    model: HullWhite
    a: float
    sigma: float
    rmse_vol: float                          # RMSE in vol terms
    per_swaption_errors: list[dict]          # per-instrument fit diagnostics
    n_swaptions: int
    converged: bool
    # New in G1 P1 Slice 3 — canonical calibration artefact.
    calibration_result: CalibrationResult | None = None

    def to_dict(self) -> dict:
        return {
            "a": self.a, "sigma": self.sigma,
            "rmse_vol": self.rmse_vol,
            "n_swaptions": self.n_swaptions,
            "converged": self.converged,
            "calibration_id": self.calibration_id,
        }

    def _build_calibration_record(self) -> CalibrationResult:
        # Reconstructed fallback for hand-built instances (calibrate_hull_white
        # populates the record eagerly with the real optimiser run).
        residuals = [e["error_bp"] for e in self.per_swaption_errors]
        quotes = [f"swaption_{e['expiry']}x{e['tenor']}" for e in self.per_swaption_errors]
        solve = SolveReport.external(algorithm="unspecified", converged=self.converged, iterations=0)
        return model_calibration_record(
            model_class="hull_white",
            parameters={"a": self.a, "sigma": self.sigma},
            residuals=residuals,
            quotes_fitted=quotes,
            solve=solve,
            diagnostics=CalibrationDiagnostics(
                extra={"rmse_vol": float(self.rmse_vol)}, reconstructed=True,
            ),
        )


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
    """Price a European swaption under HW with given params.

    Fix T4-HW1 (mirror of T2.11 for ``g2pp_swaption_price``): pre-fix this
    routine wrapped the entire body in ``try: ... except Exception:
    return 0.0``, masking every error mode (bracketing failure, brentq
    divergence, overflow in the tree formulas, calibration bugs) as a
    silent zero price.  Callers had no way to distinguish "swaption
    worth ≈ 0" from "pricer crashed".  Let real exceptions propagate;
    specific recoverable cases (degenerate params) are clamped at the
    call boundary via the ``max()`` guards on ``a`` and ``sigma``.
    """
    hw = HullWhite(a=max(a, 1e-4), sigma=max(sigma, 1e-6), curve=curve)
    swap_end = expiry_years + tenor_years
    return hw.tree_european_swaption(expiry_years, swap_end, strike,
                                      n_steps=n_steps, is_payer=is_payer)


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

    # Fix T4-HW1: vol inversion can legitimately fail at arbitrage-violating
    # prices (intrinsic-floor breach, calibration of degenerate params).
    # Catch only the specific ``ValueError`` that ``implied_vol_black76``
    # raises in those cases — let other exceptions surface instead of
    # silently masking calibration bugs as a "zero implied vol".
    try:
        return implied_vol_black76(
            price / annuity,  # normalise to Black price per unit annuity
            fwd_swap, strike, expiry_years, 1.0, OptionType.CALL,
        )
    except ValueError:
        return 0.0


def calibrate_hull_white(
    curve: DiscountCurve,
    swaption_vols: dict[tuple[float, float], float],
    strike: float | None = None,
    method: str = "nelder_mead",
    a_bounds: tuple[float, float] = (0.001, 0.50),
    sigma_bounds: tuple[float, float] = (0.001, 0.10),
    n_steps: int = 50,
    *,
    market_snapshot: MarketSnapshot | None = None,
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
        algo_name = "Nelder-Mead"
        algo_tol = 1e-10
        algo_maxiter = 500
    elif method == "differential_evolution":
        from scipy.optimize import differential_evolution
        bounds = [a_bounds, sigma_bounds]
        result = differential_evolution(objective, bounds, maxiter=200, tol=1e-8)
        algo_name = "differential_evolution"
        algo_tol = 1e-8
        algo_maxiter = 200
    else:
        result = minimize(objective, [0.03, 0.01], method="L-BFGS-B",
                           bounds=[a_bounds, sigma_bounds])
        algo_name = "L-BFGS-B"
        algo_tol = 1e-9
        algo_maxiter = 100

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

    solve = SolveReport.from_scipy(
        result, algorithm=algo_name, tolerance=algo_tol, max_iterations=algo_maxiter,
    )
    cr = model_calibration_record(
        model_class="hull_white",
        parameters={"a": float(a_opt), "sigma": float(sigma_opt)},
        residuals=[e["error_bp"] for e in errors],
        quotes_fitted=[f"swaption_{exp_y}x{tenor_y}" for (exp_y, tenor_y) in keys],
        solve=solve,
        market_snapshot_id=market_snapshot.id if market_snapshot is not None else None,
        optimiser_extra={"n_steps": n_steps},
        diagnostics=CalibrationDiagnostics(extra={"rmse_vol": rmse / 10_000.0, "n_steps": n_steps}),
    )

    return HWCalibrationResult(
        model=hw, a=a_opt, sigma=sigma_opt,
        rmse_vol=rmse / 10_000,
        per_swaption_errors=errors,
        n_swaptions=len(keys),
        converged=bool(solve.converged),
        calibration_result=cr,
    )
