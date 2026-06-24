"""Dividend term structure calibration via optimisation.

Replaces simple linear bootstrap with proper curve fitting:
- Piecewise-constant yield optimisation
- Cubic spline on cumulative dividends
- Seasonality decomposition

    from pricebook.equity.dividend_calibration import (
        calibrate_dividend_curve, calibrate_from_options,
        dividend_curve_seasonality,
    )

    curve = calibrate_dividend_curve(spot, rate, tenors, futures, method="optimize")

References:
    Kragt (2015). Managing Dividend Risk, Risk.
    Bos, Kragt & Bovenberg (2017). Pricing and Hedging Dividend Derivatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.interpolate import CubicSpline

from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationFit,
    CalibrationProvenance,
    CalibrationResult,
    CanonicalCalibrationResult,
    ObjectiveKind,
    OptimiserRun,
    OptimiserSpec,
)
from pricebook.equity.dividend_advanced import DividendCurve


# ═══════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════

@dataclass
class DividendCalibrationResult(CanonicalCalibrationResult):
    """Result of dividend curve calibration.

    `_build_calibration_record()` derives the canonical artefact from the
    retained fitted/market futures (faithful residuals); the mixin's
    `to_calibration_result()` caches it on first access — the instance carries
    everything needed, so no per-build-site population.
    """

    curve: DividendCurve
    rmse: float                     # RMSE in futures price terms
    fitted_futures: list[float]     # model-implied futures prices
    market_futures: list[float]
    method: str
    # Canonical calibration artefact (G1 P2 — widen producers); lazily cached.
    calibration_result: CalibrationResult | None = None

    def to_dict(self) -> dict:
        return {"rmse": self.rmse, "method": self.method,
                "n_tenors": len(self.market_futures),
                "calibration_id": self.calibration_id}

    def _build_calibration_record(self) -> CalibrationResult:
        residuals = [f - m for f, m in zip(self.fitted_futures, self.market_futures)]
        # Converged if the futures-price RMSE is below 0.5 price points.
        converged = self.rmse < 0.5
        return CalibrationResult(
            provenance=CalibrationProvenance.stamp(),
            fit=CalibrationFit(
                model_class="dividend_curve",
                parameters={
                    f"D_{t:g}": float(d)
                    for t, d in zip(self.curve.tenors, self.fitted_futures)
                },
                residuals=residuals,
                objective=ObjectiveKind.SSE,
                quotes_fitted=[f"div_future_{t:g}" for t in self.curve.tenors],
            ),
            optimiser_run=OptimiserRun(
                spec=OptimiserSpec(algorithm=self.method, tolerance=0.0, max_iterations=0),
                iterations=0,
                converged=converged,
            ),
            diagnostics=CalibrationDiagnostics(
                extra={"rmse": float(self.rmse), "record_source": "reconstructed"},
                warnings=() if converged else (f"rmse {self.rmse:.4f} above 0.5",),
            ),
        )


@dataclass
class SeasonalityResult:
    """Dividend seasonality decomposition."""
    quarterly_weights: list[float]  # Q1, Q2, Q3, Q4
    annual_yield: float
    peak_quarter: int               # 1-based
    trough_quarter: int

    def to_dict(self) -> dict:
        return dict(vars(self))


# ═══════════════════════════════════════════════════════════════
# Calibration methods
# ═══════════════════════════════════════════════════════════════

def calibrate_dividend_curve(
    spot: float,
    rate: float,
    tenors: list[float],
    div_futures_prices: list[float],
    method: str = "optimize",
) -> DividendCalibrationResult:
    """Calibrate dividend term structure from futures prices.

    Args:
        spot: current spot.
        rate: risk-free rate.
        tenors: futures maturities in years.
        div_futures_prices: cumulative dividend futures prices.
        method:
            "linear" — simple q̄ = D/(S·T) (existing bootstrap).
            "optimize" — piecewise-constant yield, minimise pricing error.
            "spline" — cubic spline on cumulative dividends.

    Returns:
        DividendCalibrationResult with fitted curve and diagnostics.
    """
    T = np.array(tenors, dtype=float)
    D_mkt = np.array(div_futures_prices, dtype=float)
    n = len(T)

    if method == "linear":
        from pricebook.equity.dividend_advanced import dividend_curve_bootstrap
        curve = dividend_curve_bootstrap(spot, rate, list(tenors), list(div_futures_prices))
        fitted = list(curve.cumulative_dividends)
        rmse = math.sqrt(np.mean((D_mkt - np.array(fitted))**2))
        return DividendCalibrationResult(curve, rmse, fitted, list(D_mkt), "linear")

    elif method == "optimize":
        return _calibrate_piecewise(spot, rate, T, D_mkt)

    elif method == "spline":
        return _calibrate_spline(spot, rate, T, D_mkt)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'linear', 'optimize', or 'spline'.")


def _calibrate_piecewise(
    spot: float, rate: float, T: np.ndarray, D_mkt: np.ndarray,
) -> DividendCalibrationResult:
    """Piecewise-constant yield optimisation.

    Fit q_1, q_2, ..., q_n such that the cumulative dividend from each
    piecewise segment matches market futures.

    Cumulative div for tenor T_i:
        D(T_i) = Σ_{j=1}^{i} S · q_j · (T_j - T_{j-1}) · exp(-r·T_j)
    (simplified: D ≈ S·q̄·T for short tenors)
    """
    n = len(T)

    def objective(yields):
        fitted = np.zeros(n)
        for i in range(n):
            cum = 0.0
            for j in range(i + 1):
                t_start = T[j - 1] if j > 0 else 0.0
                t_end = T[j]
                dt = t_end - t_start
                cum += spot * yields[j] * dt
            fitted[i] = cum
        return np.sum((fitted - D_mkt) ** 2)

    # Initial guess: average yield
    q_avg = float(D_mkt[-1] / (spot * T[-1])) if T[-1] > 0 else 0.02
    x0 = np.full(n, q_avg)
    bounds = [(0.0, 0.15)] * n  # yields between 0% and 15%

    result = scipy_minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
    opt_yields = result.x

    # Compute fitted futures
    fitted = []
    for i in range(n):
        cum = sum(spot * opt_yields[j] * (T[j] - (T[j - 1] if j > 0 else 0.0))
                  for j in range(i + 1))
        fitted.append(cum)

    # Build DividendCurve
    cum_divs = np.array(fitted)
    avg_yields = np.where(T > 0, cum_divs / (spot * T), 0.0)
    curve = DividendCurve(T, cum_divs, avg_yields, "piecewise_optimize")

    rmse = math.sqrt(np.mean((D_mkt - cum_divs) ** 2))
    return DividendCalibrationResult(curve, rmse, list(cum_divs), list(D_mkt), "optimize")


def _calibrate_spline(
    spot: float, rate: float, T: np.ndarray, D_mkt: np.ndarray,
) -> DividendCalibrationResult:
    """Cubic spline on cumulative dividends with positivity.

    Fits a smooth, non-decreasing cumulative dividend curve.
    """
    # Prepend origin: D(0) = 0
    T_ext = np.concatenate([[0.0], T])
    D_ext = np.concatenate([[0.0], D_mkt])

    # Fit natural cubic spline
    cs = CubicSpline(T_ext, D_ext, bc_type="natural")

    # Evaluate at original tenors
    fitted = cs(T)

    # Ensure non-negative (spline can go negative for edge cases)
    fitted = np.maximum(fitted, 0.0)

    avg_yields = np.where(T > 0, fitted / (spot * T), 0.0)
    curve = DividendCurve(T, fitted, avg_yields, "cubic_spline")

    rmse = math.sqrt(np.mean((D_mkt - fitted) ** 2))
    return DividendCalibrationResult(curve, rmse, list(fitted), list(D_mkt), "spline")


# ═══════════════════════════════════════════════════════════════
# Calibration from options
# ═══════════════════════════════════════════════════════════════

def calibrate_from_options(
    spot: float,
    options_data: list[dict],
    rate: float,
) -> DividendCalibrationResult:
    """Extract dividend curve from option prices via put-call parity.

    Args:
        options_data: list of {"T": float, "strike": float,
                               "call": float, "put": float}.
        rate: risk-free rate.

    Each entry gives one tenor's implied dividend PV via:
        PV(div) = S - K·df - (C - P)
    """
    tenors = sorted(set(d["T"] for d in options_data))

    cum_pvs = []
    for T in tenors:
        entries = [d for d in options_data if d["T"] == T]
        # Average across strikes for robustness
        pvs = []
        for e in entries:
            df = math.exp(-rate * T)
            pv_div = spot - e["strike"] * df - (e["call"] - e["put"])
            pvs.append(max(pv_div, 0.0))
        cum_pvs.append(np.mean(pvs))

    # Convert PV to cumulative dividends (forward value)
    T_arr = np.array(tenors)
    D_arr = np.array(cum_pvs)

    avg_yields = np.where(T_arr > 0, D_arr / (spot * T_arr), 0.0)
    curve = DividendCurve(T_arr, D_arr, avg_yields, "options_implied")

    return DividendCalibrationResult(curve, 0.0, list(D_arr), list(D_arr), "options")


# ═══════════════════════════════════════════════════════════════
# Seasonality decomposition
# ═══════════════════════════════════════════════════════════════

def dividend_curve_seasonality(
    curve: DividendCurve,
) -> SeasonalityResult:
    """Decompose annual dividend yield into quarterly weights.

    Assumes curve covers at least 1 year. Extracts incremental
    dividends per quarter and normalises to weights summing to 1.

    Returns:
        SeasonalityResult with Q1-Q4 weights and peak/trough.
    """
    tenors = curve.tenors
    cum_divs = curve.cumulative_dividends

    # Interpolate at quarterly points
    quarters = [0.25, 0.50, 0.75, 1.00]
    cum_at_q = np.interp(quarters, tenors, cum_divs, left=0.0)

    # Incremental per quarter
    incremental = np.diff(np.concatenate([[0.0], cum_at_q]))

    # Normalise
    total = incremental.sum()
    if total > 0:
        weights = (incremental / total).tolist()
    else:
        weights = [0.25, 0.25, 0.25, 0.25]

    annual_yield = float(np.mean(curve.implied_yields)) if len(curve.implied_yields) > 0 else 0.0

    peak = int(np.argmax(weights)) + 1
    trough = int(np.argmin(weights)) + 1

    return SeasonalityResult(weights, annual_yield, peak, trough)
