"""Stochastic calculus utilities.

Ito formula, Stratonovich conversion, quadratic variation, and
stochastic integration helpers.

    from pricebook.numerical._stochastic import (
        ito_formula, stratonovich_to_ito, quadratic_variation,
        ito_isometry_check,
    )

References:
    Shreve (2004). Stochastic Calculus for Finance II.
    Kloeden & Platen (1992). Numerical Solution of SDEs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ItoFormulaResult:
    """Result of Ito formula application."""
    drift_correction: float      # the extra ½σ²f'' term
    total_drift: float           # μf' + ½σ²f''
    diffusion: float             # σf'
    description: str

    def to_dict(self) -> dict:
        return dict(vars(self))


def ito_formula(
    f_prime: float,
    f_double_prime: float,
    mu: float,
    sigma: float,
) -> ItoFormulaResult:
    """Apply Ito's formula to f(X_t) where dX = μdt + σdW.

    df(X) = f'(X)(μdt + σdW) + ½f''(X)σ²dt
          = [μf' + ½σ²f'']dt + σf'dW

    The ½σ²f'' term is the Ito correction (absent in ordinary calculus).

    Args:
        f_prime: f'(X) at current point.
        f_double_prime: f''(X) at current point.
        mu: drift of X.
        sigma: diffusion of X.
    """
    correction = 0.5 * sigma**2 * f_double_prime
    total_drift = mu * f_prime + correction
    diffusion = sigma * f_prime

    return ItoFormulaResult(
        drift_correction=correction,
        total_drift=total_drift,
        diffusion=diffusion,
        description=f"df = ({total_drift:.4f})dt + ({diffusion:.4f})dW",
    )


def ito_log_transform(mu: float, sigma: float) -> dict:
    """Ito formula for f(X) = log(X) where dX = μXdt + σXdW.

    d(log X) = (μ - ½σ²)dt + σdW

    This is why GBM drift in log-space is μ - ½σ², not μ.
    """
    return {
        "log_drift": mu - 0.5 * sigma**2,
        "log_diffusion": sigma,
        "ito_correction": -0.5 * sigma**2,
    }


def stratonovich_to_ito(
    drift_strat: float,
    diffusion: float,
    diffusion_prime: float,
) -> dict:
    """Convert Stratonovich SDE to Ito form.

    Stratonovich: dX = a(X)dt + b(X)∘dW
    Ito:          dX = [a(X) - ½b(X)b'(X)]dt + b(X)dW

    The correction term ½b(X)b'(X) converts between the two conventions.

    Args:
        drift_strat: Stratonovich drift a(X).
        diffusion: b(X).
        diffusion_prime: b'(X) = db/dX.
    """
    correction = 0.5 * diffusion * diffusion_prime
    return {
        "ito_drift": drift_strat - correction,
        "ito_diffusion": diffusion,
        "correction": correction,
    }


def ito_to_stratonovich(
    drift_ito: float,
    diffusion: float,
    diffusion_prime: float,
) -> dict:
    """Convert Ito SDE to Stratonovich form.

    Inverse of stratonovich_to_ito.
    """
    correction = 0.5 * diffusion * diffusion_prime
    return {
        "strat_drift": drift_ito + correction,
        "strat_diffusion": diffusion,
        "correction": correction,
    }


def quadratic_variation(
    path: np.ndarray,
    times: np.ndarray | None = None,
) -> float:
    """Compute [X]_T = Σ (X_{t_{i+1}} - X_{t_i})² (quadratic variation).

    For Brownian motion: [W]_T = T.
    For GBM: [log S]_T = σ²T.
    """
    increments = np.diff(path)
    return float(np.sum(increments**2))


def realized_variance(
    log_returns: np.ndarray,
    annualize: float = 252.0,
) -> float:
    """Realized variance from high-frequency log returns.

    RV = annualize × Σ r²_i

    This is a consistent estimator of integrated variance.
    """
    return float(annualize * np.sum(log_returns**2))


def realized_volatility(
    log_returns: np.ndarray,
    annualize: float = 252.0,
) -> float:
    """Realized volatility = √(realized variance)."""
    return math.sqrt(realized_variance(log_returns, annualize))


def bipower_variation(
    log_returns: np.ndarray,
    annualize: float = 252.0,
) -> float:
    """Bipower variation — robust to jumps (Barndorff-Nielsen & Shephard 2004).

    BV = (π/2) × annualize × Σ |r_i| × |r_{i-1}|

    Converges to integrated variance even in the presence of jumps.
    """
    if len(log_returns) < 2:
        return 0.0
    abs_r = np.abs(log_returns)
    bv = np.sum(abs_r[1:] * abs_r[:-1])
    return float(math.pi / 2 * annualize * bv)


def jump_test(
    log_returns: np.ndarray,
    annualize: float = 252.0,
    confidence: float = 0.95,
) -> dict:
    """Barndorff-Nielsen & Shephard jump test.

    Compares realized variance (sensitive to jumps) with bipower
    variation (robust to jumps). Large difference → jumps detected.

    z = (RV - BV) / √(var_estimate)
    """
    from scipy.stats import norm

    rv = realized_variance(log_returns, annualize)
    bv = bipower_variation(log_returns, annualize)
    n = len(log_returns)

    # Variance of the test statistic (simplified)
    var_est = max((math.pi**2 / 4 + math.pi - 5) * bv**2 / n, 1e-20)
    z = (rv - bv) / math.sqrt(var_est)
    p_value = 1 - norm.cdf(z)
    z_crit = norm.ppf(confidence)

    return {
        "realized_variance": rv,
        "bipower_variation": bv,
        "jump_component": max(rv - bv, 0),
        "z_statistic": z,
        "p_value": p_value,
        "jumps_detected": z > z_crit,
        "confidence": confidence,
    }


def milstein_correction(
    sigma: callable,
    sigma_prime: callable,
    x: float,
    dW: float,
    dt: float,
) -> float:
    """Milstein correction term for SDE discretisation.

    Milstein: X_{n+1} = X_n + μΔt + σΔW + ½σσ'(ΔW² - Δt)

    The last term ½σσ'(ΔW² - Δt) is the Milstein correction.
    It gives strong order 1.0 (vs 0.5 for Euler-Maruyama).

    Returns the correction term only.
    """
    s = sigma(x)
    sp = sigma_prime(x)
    return 0.5 * s * sp * (dW**2 - dt)
