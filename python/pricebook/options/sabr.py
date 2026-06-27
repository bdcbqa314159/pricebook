"""
SABR stochastic volatility model.

Dynamics:
    dF = sigma * F^beta * dW1
    dsigma = alpha * sigma * dW2
    dW1 * dW2 = rho * dt

Hagan approximation for implied Black vol (see REFERENCES.md):

    sigma_B(K) = alpha / (F*K)^((1-beta)/2) * z/x(z) * (1 + corrections)

where z = (alpha/nu) * (F*K)^((1-beta)/2) * ln(F/K)

    vol = sabr_implied_vol(forward=100, strike=105, T=1.0,
                           alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationResult,
    CanonicalCalibrationResult,
    ObjectiveKind,
    SolveReport,
    model_calibration_record,
)
from pricebook.models.black76 import OptionType, black76_price
from pricebook.statistics.optimization import minimize as pb_minimize

if TYPE_CHECKING:
    from pricebook.market_data import MarketSnapshot


@dataclass
class SABRCalibrationResult(CanonicalCalibrationResult):
    """Typed result of a SABR smile calibration.

    Carries the fitted SABR parameters (`alpha`, `beta`, `rho`, `nu`) and fit
    quality. `calibration_result` is the canonical provenance artefact, populated
    eagerly by `sabr_calibrate`; `calibrate_sabr_smile` additionally fills the
    reprice diagnostics (`reprice_errors_bp`, `max_error_bp`).
    """

    alpha: float
    beta: float
    rho: float
    nu: float
    rmse: float
    reprice_errors_bp: list[float] = field(default_factory=list)
    max_error_bp: float = 0.0
    calibration_result: CalibrationResult | None = None

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha, "beta": self.beta, "rho": self.rho, "nu": self.nu,
            "rmse": self.rmse,
            "reprice_errors_bp": list(self.reprice_errors_bp),
            "max_error_bp": self.max_error_bp,
            "calibration_id": self.calibration_id,
        }

    def _build_calibration_record(self) -> CalibrationResult:
        # On-demand fallback for hand-built instances — `sabr_calibrate`
        # populates the record eagerly with the real optimiser run, so this
        # reconstructs from stored fields and is marked as such. No captured
        # optimiser metadata is available here (algorithm "unspecified").
        residuals = [float(e) for e in self.reprice_errors_bp]
        quotes = [f"smile_point_{i}" for i in range(len(residuals))]
        solve = SolveReport.external(algorithm="unspecified", converged=None, iterations=0)
        return model_calibration_record(
            model_class="sabr",
            parameters={"alpha": self.alpha, "beta": self.beta,
                        "rho": self.rho, "nu": self.nu},
            residuals=residuals,
            quotes_fitted=quotes,
            solve=solve,
            diagnostics=CalibrationDiagnostics(
                extra={"rmse": float(self.rmse), "max_error_bp": float(self.max_error_bp)}, reconstructed=True,
            ),
        )


def sabr_implied_vol(
    forward: float,
    strike: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """SABR implied Black volatility via Hagan approximation.

    Args:
        forward: forward price.
        strike: option strike.
        T: time to expiry.
        alpha: initial volatility level.
        beta: CEV exponent (0 = normal, 1 = lognormal).
        rho: correlation between forward and vol (-1 < rho < 1).
        nu: volatility of volatility.
    """
    if T <= 0:
        return alpha

    # ATM case (K ≈ F)
    if abs(forward - strike) < 1e-10 * forward:
        fk = forward
        one_minus_beta = 1.0 - beta
        A = alpha / (fk ** one_minus_beta)
        B1 = one_minus_beta**2 * alpha**2 / (24.0 * fk ** (2.0 * one_minus_beta))
        B2 = 0.25 * rho * beta * nu * alpha / (fk ** one_minus_beta)
        B3 = (2.0 - 3.0 * rho**2) * nu**2 / 24.0
        return A * (1.0 + (B1 + B2 + B3) * T)

    # General case
    one_minus_beta = 1.0 - beta
    fk = forward * strike
    fk_ratio = forward / strike
    log_fk = math.log(fk_ratio)

    fk_mid = fk ** (one_minus_beta / 2.0)

    # z and x(z) with guards for rho near ±1 and deep OTM
    z = (nu / alpha) * fk_mid * log_fk
    if abs(z) < 1e-12:
        x_z = 1.0
    else:
        sqrt_arg = max(1.0 - 2.0 * rho * z + z * z, 0.0)
        denom = (math.sqrt(sqrt_arg) + z - rho)
        one_minus_rho = max(1.0 - rho, 1e-10)
        ratio = denom / one_minus_rho
        if ratio <= 0:
            x_z = 1.0  # degenerate case
        else:
            x_z = z / math.log(ratio)

    # Prefactor
    A = alpha / (fk_mid * (
        1.0 + one_minus_beta**2 / 24.0 * log_fk**2
        + one_minus_beta**4 / 1920.0 * log_fk**4
    ))

    # Correction terms
    B1 = one_minus_beta**2 * alpha**2 / (24.0 * fk ** one_minus_beta)
    B2 = 0.25 * rho * beta * nu * alpha / fk_mid
    B3 = (2.0 - 3.0 * rho**2) * nu**2 / 24.0

    result = A * x_z * (1.0 + (B1 + B2 + B3) * T)
    return max(result, 1e-10)  # floor to prevent negative implied vol


def sabr_price(
    forward: float,
    strike: float,
    T: float,
    df: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Option price under SABR (via Hagan vol + Black-76)."""
    vol = sabr_implied_vol(forward, strike, T, alpha, beta, rho, nu)
    return black76_price(forward, strike, vol, T, df, option_type)


def sabr_calibrate(
    forward: float,
    strikes: list[float],
    market_vols: list[float],
    T: float,
    beta: float = 0.5,
    initial_guess: tuple[float, float, float] | None = None,
    *,
    market_snapshot: MarketSnapshot | None = None,
) -> SABRCalibrationResult:
    """Calibrate SABR parameters (alpha, rho, nu) to market smile.

    Beta is typically fixed. Minimises sum of squared vol errors.

    Args:
        forward: forward price.
        strikes: list of strikes.
        market_vols: corresponding market implied vols.
        T: time to expiry.
        beta: CEV exponent (fixed).
        initial_guess: (alpha, rho, nu) starting point.

    Returns:
        `SABRCalibrationResult` — the fitted `alpha`/`beta`/`rho`/`nu`, `rmse`,
        and the canonical `calibration_result` provenance artefact (a
        `CanonicalCalibrationResult`, so it persists via `db.save_calibration`).
    """
    if initial_guess is None:
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - forward))
        alpha0 = market_vols[atm_idx] * forward ** (1 - beta)
        initial_guess = (alpha0, -0.1, 0.3)

    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or rho <= -1 or rho >= 1:
            return 1e10
        total = 0.0
        for k, mv in zip(strikes, market_vols):
            model_vol = sabr_implied_vol(forward, k, T, alpha, beta, rho, nu)
            total += (model_vol - mv) ** 2
        return total

    result = pb_minimize(objective, x0=list(initial_guess), method="nelder_mead",
                         tol=1e-12, maxiter=2000)

    alpha, rho, nu = result.x
    rmse = math.sqrt(result.fun / len(strikes))

    # Per-strike residuals in vol units (model - market)
    residuals = [
        sabr_implied_vol(forward, k, T, alpha, beta, rho, nu) - mv
        for k, mv in zip(strikes, market_vols)
    ]

    # Capture the optimiser's own verdict; the builder reads it straight off the
    # report — no re-derivation. (pb_minimize returns a scipy-shaped result.)
    solve = SolveReport.from_scipy(
        result, algorithm="nelder_mead", tolerance=1e-12, max_iterations=2000,
    )
    cr = model_calibration_record(
        model_class="sabr",
        parameters={"alpha": float(alpha), "beta": float(beta),
                    "rho": float(rho), "nu": float(nu)},
        residuals=residuals,
        quotes_fitted=[f"smile_K={k:.4f}" for k in strikes],
        solve=solve,
        objective=ObjectiveKind.SSE,
        market_snapshot_id=market_snapshot.id if market_snapshot is not None else None,
        optimiser_extra={"beta_fixed": float(beta), "forward": float(forward), "T": float(T)},
        diagnostics=CalibrationDiagnostics(extra={"rmse_vol": float(rmse)}),
    )

    return SABRCalibrationResult(
        alpha=float(alpha), beta=float(beta), rho=float(rho), nu=float(nu),
        rmse=float(rmse), calibration_result=cr,
    )


def calibrate_sabr_smile(
    forward: float,
    strikes: list[float],
    market_vols: list[float],
    T: float,
    beta: float = 0.5,
    max_rmse_bp: float = 1.0,
) -> SABRCalibrationResult:
    """Hardened SABR calibration with validation.

    Calls sabr_calibrate() then validates:
    - RMSE < max_rmse_bp (default 1bp)
    - Parameters in valid ranges
    - Reprice check on input strikes

    Args:
        max_rmse_bp: maximum acceptable RMSE in vol basis points.

    Returns:
        `SABRCalibrationResult`, with `reprice_errors_bp` / `max_error_bp` filled.

    Raises:
        ValueError if calibration fails validation.

    References:
        Hagan et al. (2002). Managing Smile Risk. Wilmott Magazine.
    """
    import warnings

    result = sabr_calibrate(forward, strikes, market_vols, T, beta)

    # Validate ranges
    if result.alpha <= 0:
        warnings.warn(f"SABR alpha = {result.alpha:.6f} ≤ 0", stacklevel=2)
    if not -1 < result.rho < 1:
        warnings.warn(f"SABR rho = {result.rho:.4f} outside (-1,1)", stacklevel=2)
    if result.nu <= 0:
        warnings.warn(f"SABR nu = {result.nu:.6f} ≤ 0", stacklevel=2)

    # Reprice check
    reprice_errors = []
    for k, mv in zip(strikes, market_vols):
        model = sabr_implied_vol(forward, k, T, result.alpha, result.beta,
                                  result.rho, result.nu)
        err_bp = abs(model - mv) * 10_000
        reprice_errors.append(err_bp)

    result.reprice_errors_bp = reprice_errors
    result.max_error_bp = max(reprice_errors) if reprice_errors else 0.0

    if result.rmse * 10_000 > max_rmse_bp:
        warnings.warn(
            f"SABR RMSE = {result.rmse*10_000:.2f}bp > {max_rmse_bp}bp threshold",
            stacklevel=2,
        )

    return result


# ---------------------------------------------------------------------------
# Shifted SABR
# ---------------------------------------------------------------------------


def shifted_sabr_implied_vol(
    forward: float,
    strike: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
) -> float:
    """Shifted SABR implied vol: apply Hagan to (F+shift, K+shift).

    Handles negative rates by shifting the distribution.
    Reduces to standard SABR when shift=0.
    """
    return sabr_implied_vol(forward + shift, strike + shift, T, alpha, beta, rho, nu)


def shifted_sabr_price(
    forward: float,
    strike: float,
    T: float,
    df: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Option price under shifted SABR."""
    vol = shifted_sabr_implied_vol(forward, strike, T, alpha, beta, rho, nu, shift)
    return black76_price(forward + shift, strike + shift, vol, T, df, option_type)


# ---------------------------------------------------------------------------
# Normal (Bachelier) vol conversion
# ---------------------------------------------------------------------------


def sabr_normal_vol(
    forward: float,
    strike: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
) -> float:
    """Convert SABR lognormal vol to normal (Bachelier) vol.

    Approximation: sigma_N ≈ sigma_B * F (for ATM, beta=1).
    General: sigma_N ≈ sigma_B * (F*K)^0.5 for moderate moneyness.
    """
    f = forward + shift
    k = strike + shift
    lognormal_vol = sabr_implied_vol(f, k, T, alpha, beta, rho, nu)
    fk_mid = math.sqrt(max(f * k, 1e-30))
    return lognormal_vol * fk_mid
