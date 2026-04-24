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

from pricebook.black76 import OptionType, black76_price


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
) -> dict[str, float]:
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
        dict with keys: alpha, beta, rho, nu, rmse.
    """
    from pricebook.optimization import minimize as pb_minimize

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

    return {
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "nu": nu,
        "rmse": rmse,
    }


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
