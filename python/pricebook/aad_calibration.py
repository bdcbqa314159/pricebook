"""AAD calibration Jacobian: d(model_params) / d(market_data) in one pass.

Computes the sensitivity of calibrated parameters to market inputs
via AAD, enabling risk of calibrated models to market moves.
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.aad import Number, Tape


def sabr_jacobian(
    forward: float,
    strikes: list[float],
    market_vols: list[float],
    T: float,
    beta: float = 0.5,
    alpha: float = 0.2,
    rho: float = -0.3,
    nu: float = 0.4,
) -> dict:
    """Compute d(SABR_vols) / d(alpha, rho, nu) via AAD.

    For each strike, computes the sensitivity of the model implied vol
    to the SABR parameters. This is the Jacobian of the calibration
    mapping: market_vols → (alpha, rho, nu).

    Returns:
        dict with:
            - model_vols: list of model vols at each strike
            - d_alpha: list of d(vol)/d(alpha) per strike
            - d_rho: list of d(vol)/d(rho) per strike
            - d_nu: list of d(vol)/d(nu) per strike
    """
    from pricebook.sabr import sabr_implied_vol

    model_vols = []
    d_alpha_list = []
    d_rho_list = []
    d_nu_list = []

    for K in strikes:
        with Tape():
            a = Number(alpha)
            r = Number(rho)
            n = Number(nu)

            # Inline SABR Hagan approximation with Number arithmetic
            vol = _sabr_hagan_aad(forward, K, T, a, beta, r, n)

            vol.propagate_to_start()

            model_vols.append(vol.value)
            d_alpha_list.append(a.adjoint)
            d_rho_list.append(r.adjoint)
            d_nu_list.append(n.adjoint)

    return {
        "model_vols": model_vols,
        "d_alpha": d_alpha_list,
        "d_rho": d_rho_list,
        "d_nu": d_nu_list,
    }


def _sabr_hagan_aad(
    forward: float, strike: float, T: float,
    alpha: Number, beta: float, rho: Number, nu: Number,
) -> Number:
    """SABR Hagan formula with Number inputs for AAD."""
    from pricebook.aad import log, sqrt

    if T <= 0:
        return alpha

    one_minus_beta = 1.0 - beta
    fk = forward * strike
    fk_mid = fk ** (one_minus_beta / 2.0)

    # ATM case
    if abs(forward - strike) < 1e-10 * forward:
        A = alpha / (forward ** one_minus_beta)
        B1 = one_minus_beta**2 / 24.0 * alpha * alpha / (forward ** (2.0 * one_minus_beta))
        B2 = 0.25 * rho * beta * nu * alpha / (forward ** one_minus_beta)
        B3 = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
        return A * (Number(1.0) + (B1 + B2 + B3) * T)

    # General case
    log_fk = math.log(forward / strike)
    z = nu / alpha * fk_mid * log_fk

    # x(z) = z / log((sqrt(1-2*rho*z+z^2)+z-rho)/(1-rho))
    z_val = float(z)
    sqrt_arg = Number(1.0) - 2.0 * rho * z + z * z
    if abs(z_val) < 1e-12:
        x_z = Number(1.0)
    else:
        inner = (sqrt_arg.sqrt() + z - rho) / (1.0 - rho)
        x_z = z / inner.log()

    denom = 1.0 + one_minus_beta**2 / 24.0 * log_fk**2 + one_minus_beta**4 / 1920.0 * log_fk**4
    A = alpha / (fk_mid * denom)

    B1 = one_minus_beta**2 * alpha * alpha / (24.0 * fk ** one_minus_beta)
    B2 = 0.25 * rho * beta * nu * alpha / fk_mid
    B3 = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0

    return A * x_z * (Number(1.0) + (B1 + B2 + B3) * T)


def calibration_risk(
    forward: float,
    strikes: list[float],
    market_vols: list[float],
    T: float,
    alpha: float,
    rho: float,
    nu: float,
    beta: float = 0.5,
) -> dict:
    """Sensitivity of calibrated SABR params to market vol changes.

    Uses the implicit function theorem:
        d(params)/d(market) = -J_params^{-1} @ J_market

    where J_params = d(model_vols)/d(params) from AAD.

    Returns:
        dict with d_alpha_d_mkt, d_rho_d_mkt, d_nu_d_mkt arrays.
    """
    jac = sabr_jacobian(forward, strikes, market_vols, T, beta, alpha, rho, nu)

    n = len(strikes)
    # Build Jacobian matrix: J[i, j] = d(model_vol_i) / d(param_j)
    J = np.zeros((n, 3))
    J[:, 0] = jac["d_alpha"]
    J[:, 1] = jac["d_rho"]
    J[:, 2] = jac["d_nu"]

    # Pseudo-inverse for overdetermined system
    J_pinv = np.linalg.pinv(J)  # (3, n)

    # d(params)/d(market_vol_i) = J_pinv (since model_vol = market_vol at calibration)
    return {
        "d_alpha_d_mkt": J_pinv[0, :].tolist(),
        "d_rho_d_mkt": J_pinv[1, :].tolist(),
        "d_nu_d_mkt": J_pinv[2, :].tolist(),
    }
