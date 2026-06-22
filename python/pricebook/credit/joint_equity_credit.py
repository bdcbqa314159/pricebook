"""Joint equity-credit calibration.

Simultaneously fit a structural credit model to both equity volatility
and CDS spreads. This links the equity and credit markets through the
Merton framework, ensuring consistency.

    from pricebook.credit.joint_equity_credit import (
        joint_calibrate, JointCalibrationResult,
    )

References:
    Cremers, Driessen & Maenhout (2008). Explaining the Level of Credit
    Spreads: Option-Implied Jump Risk Premia in a Firm Value Model.
    Finger et al. (2002). CreditGrades Technical Document.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from pricebook.calibration import (
    CalibrationFit,
    CalibrationProvenance,
    CalibrationResult,
    CanonicalCalibrationResult,
    ObjectiveKind,
    OptimiserRun,
    OptimiserSpec,
)
from pricebook.credit.credit_grades import CreditGrades


@dataclass
class JointCalibrationResult(CanonicalCalibrationResult):
    """Result of joint equity-credit calibration."""
    asset_vol: float             # calibrated asset volatility
    leverage: float              # calibrated leverage (D/V)
    recovery_mean: float
    recovery_vol: float
    equity_vol_model: float      # model-implied equity vol
    equity_vol_market: float     # market equity vol (target)
    cds_spread_model_bp: float   # model-implied 5Y CDS spread
    cds_spread_market_bp: float  # market CDS spread (target)
    equity_vol_error_pct: float  # % error on equity vol
    cds_spread_error_bp: float   # bp error on CDS spread
    fit_quality: float           # objective function value (lower = better)
    # Canonical calibration artefact (G1 P2 — widen producers).
    calibration_result: CalibrationResult | None = None

    def to_dict(self) -> dict:
        d = {k: v for k, v in vars(self).items() if k != "calibration_result"}
        d["calibration_id"] = self.calibration_id
        return d

    def _relative_residuals(self) -> list[float]:
        """Dimensionless relative residuals for the two fitted quotes."""
        vol_res = (self.equity_vol_model / self.equity_vol_market - 1.0
                   if self.equity_vol_market else 0.0)
        cds_res = (self.cds_spread_model_bp / self.cds_spread_market_bp - 1.0
                   if self.cds_spread_market_bp else 0.0)
        return [vol_res, cds_res]

    def _build_calibration_record(self) -> CalibrationResult:
        return CalibrationResult(
            provenance=CalibrationProvenance.stamp(),
            fit=CalibrationFit(
                model_class="joint_equity_credit",
                parameters={"asset_vol": float(self.asset_vol), "leverage": float(self.leverage)},
                residuals=self._relative_residuals(),
                objective=ObjectiveKind.SSE,
                quotes_fitted=["equity_vol", "cds_spread"],
            ),
            optimiser_run=OptimiserRun(
                spec=OptimiserSpec(algorithm="L-BFGS-B", tolerance=0.0, max_iterations=0),
                iterations=0,
                converged=True,
            ),
        )


def joint_calibrate(
    equity_vol: float,
    cds_spread_bp: float,
    initial_leverage: float = 0.40,
    recovery_mean: float = 0.40,
    recovery_vol: float = 0.25,
    cds_tenor: float = 5.0,
    vol_weight: float = 1.0,
    spread_weight: float = 1.0,
) -> JointCalibrationResult:
    """Jointly calibrate asset vol and leverage to equity vol + CDS spread.

    The Merton/CreditGrades model links:
    - Asset vol σ_A → equity vol σ_E via leverage: σ_E ≈ σ_A / (1 - L)
    - Asset vol σ_A + leverage L → CDS spread via CreditGrades

    We find (σ_A, L) that minimise:
        w₁ × (σ_E_model / σ_E_market - 1)² + w₂ × (s_model / s_market - 1)²

    Args:
        equity_vol: market-observed equity volatility (e.g. 0.30 = 30%).
        cds_spread_bp: market CDS spread in bp (e.g. 150).
        initial_leverage: starting guess for leverage.
        recovery_mean: recovery assumption for CreditGrades.
        recovery_vol: recovery vol (λ) for CreditGrades.
        cds_tenor: CDS tenor for spread matching (default 5Y).
        vol_weight: weight on equity vol matching.
        spread_weight: weight on CDS spread matching.

    Returns:
        JointCalibrationResult with calibrated parameters and fit quality.
    """
    cds_target = cds_spread_bp / 10_000

    def objective(x: np.ndarray) -> float:
        sigma_a = max(x[0], 0.01)
        lev = max(min(x[1], 0.99), 0.01)

        # Model equity vol: σ_E ≈ σ_A × V/E = σ_A / (1 - L)
        equity_factor = 1.0 / (1.0 - lev)
        sigma_e_model = sigma_a * equity_factor

        # Model CDS spread via CreditGrades
        try:
            model = CreditGrades(sigma_a, lev, recovery_mean, recovery_vol)
            cds_model = model.cds_spread(cds_tenor, recovery_mean)
        except (ValueError, ZeroDivisionError):
            return 1e10

        # Relative errors
        vol_err = (sigma_e_model / equity_vol - 1.0) ** 2 if equity_vol > 0 else 0
        spread_err = (cds_model / cds_target - 1.0) ** 2 if cds_target > 0 else 0

        return vol_weight * vol_err + spread_weight * spread_err

    # Initial guess: back out asset vol from equity vol and leverage
    sigma_a_guess = equity_vol * (1.0 - initial_leverage)
    x0 = np.array([sigma_a_guess, initial_leverage])

    bounds = [(0.01, 1.0), (0.01, 0.99)]
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 200, "ftol": 1e-12})

    sigma_a = max(result.x[0], 0.01)
    lev = max(min(result.x[1], 0.99), 0.01)

    # Evaluate at solution
    sigma_e_model = sigma_a / (1.0 - lev)
    model = CreditGrades(sigma_a, lev, recovery_mean, recovery_vol)
    cds_model = model.cds_spread(cds_tenor, recovery_mean)

    cr = CalibrationResult(
        provenance=CalibrationProvenance.stamp(),
        fit=CalibrationFit(
            model_class="joint_equity_credit",
            parameters={"asset_vol": float(sigma_a), "leverage": float(lev)},
            residuals=[
                sigma_e_model / equity_vol - 1.0 if equity_vol > 0 else 0.0,
                cds_model / cds_target - 1.0 if cds_target > 0 else 0.0,
            ],
            objective=ObjectiveKind.WEIGHTED_SSE,
            quotes_fitted=["equity_vol", f"cds_spread_{cds_tenor:g}Y"],
            weights=[vol_weight, spread_weight],
        ),
        optimiser_run=OptimiserRun(
            spec=OptimiserSpec(algorithm="L-BFGS-B", tolerance=1e-12, max_iterations=200),
            iterations=int(getattr(result, "nit", 0)),
            converged=bool(getattr(result, "success", True)),
        ),
    )

    return JointCalibrationResult(
        asset_vol=sigma_a,
        leverage=lev,
        recovery_mean=recovery_mean,
        recovery_vol=recovery_vol,
        equity_vol_model=sigma_e_model,
        equity_vol_market=equity_vol,
        cds_spread_model_bp=cds_model * 10_000,
        cds_spread_market_bp=cds_spread_bp,
        equity_vol_error_pct=abs(sigma_e_model / equity_vol - 1.0) * 100 if equity_vol > 0 else 0,
        cds_spread_error_bp=abs(cds_model - cds_target) * 10_000,
        fit_quality=result.fun,
        calibration_result=cr,
    )
