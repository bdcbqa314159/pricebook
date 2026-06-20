"""Volatility forecasting: GARCH(1,1), EGARCH, EWMA, realized volatility.

    from pricebook.statistics.garch import (
        ewma_vol, realized_vol,
        garch_11_fit, garch_11_forecast, garch_var,
        egarch_fit,
    )

References:
    Engle (1982). Autoregressive Conditional Heteroscedasticity.
    Bollerslev (1986). Generalized ARCH.
    Nelson (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach (EGARCH).
    J.P. Morgan (1996). RiskMetrics Technical Document (EWMA).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ═══════════════════════════════════════════════════════════════
# EWMA (RiskMetrics)
# ═══════════════════════════════════════════════════════════════

def ewma_vol(
    returns: np.ndarray,
    decay: float = 0.94,
    annualise: int = 252,
) -> np.ndarray:
    """EWMA (RiskMetrics) volatility.

    sigma^2_t = lambda × sigma^2_{t-1} + (1 - lambda) × r^2_{t-1}

    Args:
        returns: array of returns.
        decay: lambda (0.94 for daily, 0.97 for monthly).
        annualise: annualisation factor.

    Returns:
        Array of annualised volatilities (same length as returns).
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n == 0:
        return np.array([])

    var = np.zeros(n)
    var[0] = r[0] ** 2
    for t in range(1, n):
        var[t] = decay * var[t - 1] + (1 - decay) * r[t - 1] ** 2

    return np.sqrt(var * annualise)


# ═══════════════════════════════════════════════════════════════
# Realized Volatility
# ═══════════════════════════════════════════════════════════════

def realized_vol(
    prices: np.ndarray,
    window: int = 20,
    annualise: int = 252,
) -> np.ndarray:
    """Realized volatility from close-to-close returns.

    RV_t = std(r_{t-window+1}, ..., r_t) × sqrt(annualise)

    Args:
        prices: array of strictly-positive prices. Passing returns instead of
            prices is a common confusion; non-positive entries raise
            ``ValueError`` rather than silently producing NaN via ``log(≤0)``.
        window: rolling window length.
        annualise: annualisation factor.

    Returns array same length as prices (NaN for initial window).
    """
    p = np.asarray(prices, dtype=float)
    if len(p) < 2:
        return np.full(len(p), np.nan)
    if not np.all(p > 0):
        raise ValueError(
            "realized_vol requires strictly-positive prices; got entries ≤ 0. "
            "If you passed returns by mistake, convert with "
            "`prices = base * np.cumprod(1 + returns)` first."
        )

    r = np.diff(np.log(p))
    rv = np.full(len(p), np.nan)
    for t in range(window, len(r) + 1):
        rv[t] = np.std(r[t - window:t]) * math.sqrt(annualise)

    return rv


# ═══════════════════════════════════════════════════════════════
# GARCH(1,1)
# ═══════════════════════════════════════════════════════════════

@dataclass
class GARCH11Result:
    """GARCH(1,1) fit result."""
    omega: float        # intercept
    alpha: float        # ARCH coefficient (news)
    beta: float         # GARCH coefficient (persistence)
    persistence: float  # alpha + beta (should be < 1)
    long_run_var: float # omega / (1 - alpha - beta)
    log_likelihood: float
    n_obs: int
    conditional_vol: np.ndarray  # fitted conditional volatility path

    def to_dict(self) -> dict:
        return {
            "omega": self.omega, "alpha": self.alpha, "beta": self.beta,
            "persistence": self.persistence, "long_run_var": self.long_run_var,
            "log_likelihood": self.log_likelihood, "n_obs": self.n_obs,
        }


def _garch_11_loglik(params: np.ndarray, returns: np.ndarray) -> float:
    """Negative log-likelihood for GARCH(1,1) with normal innovations."""
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10

    n = len(returns)
    var = np.zeros(n)
    var[0] = returns.var()

    for t in range(1, n):
        var[t] = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
        if var[t] <= 0:
            return 1e10

    # Normal log-likelihood: -0.5 * sum(log(var_t) + r_t^2/var_t)
    ll = -0.5 * np.sum(np.log(var) + returns ** 2 / var)
    return -ll  # negative for minimisation


def garch_11_fit(
    returns: np.ndarray,
    max_iter: int = 500,
) -> GARCH11Result:
    """Fit GARCH(1,1) via maximum likelihood.

    sigma^2_t = omega + alpha × r^2_{t-1} + beta × sigma^2_{t-1}

    Estimated via Nelder-Mead on the normal log-likelihood.
    """
    from scipy.optimize import minimize

    r = np.asarray(returns, dtype=float)
    n = len(r)
    sample_var = r.var()

    # Initial guess: alpha=0.05, beta=0.90, omega from unconditional
    x0 = np.array([sample_var * 0.05, 0.05, 0.90])

    result = minimize(
        _garch_11_loglik, x0, args=(r,),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-10, "fatol": 1e-10},
    )

    omega, alpha, beta = result.x
    omega = max(omega, 1e-10)
    alpha = max(alpha, 0.0)
    beta = max(beta, 0.0)
    persistence = alpha + beta

    # Compute fitted conditional volatility
    var = np.zeros(n)
    var[0] = sample_var
    for t in range(1, n):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]

    long_run = omega / max(1 - persistence, 1e-10) if persistence < 1 else sample_var

    return GARCH11Result(
        omega=float(omega),
        alpha=float(alpha),
        beta=float(beta),
        persistence=float(persistence),
        long_run_var=float(long_run),
        log_likelihood=float(-result.fun),
        n_obs=n,
        conditional_vol=np.sqrt(var),
    )


def garch_11_forecast(
    omega: float,
    alpha: float,
    beta: float,
    last_return: float,
    last_var: float,
    n_ahead: int = 10,
) -> np.ndarray:
    """Multi-step GARCH(1,1) variance forecast.

    E[sigma^2_{t+k}] = long_run_var + (alpha+beta)^{k-1} × (sigma^2_{t+1} - long_run_var)
    """
    persistence = alpha + beta
    long_run = omega / max(1 - persistence, 1e-10) if persistence < 1 else last_var

    # One-step: sigma^2_{t+1} = omega + alpha*r_t^2 + beta*sigma^2_t
    var_1 = omega + alpha * last_return ** 2 + beta * last_var

    forecast = np.zeros(n_ahead)
    for k in range(n_ahead):
        forecast[k] = long_run + persistence ** k * (var_1 - long_run)

    return np.sqrt(forecast)  # return as volatility


def garch_var(
    returns: np.ndarray,
    confidence: float = 0.99,
) -> float:
    """GARCH-based VaR: conditional sigma × z_alpha.

    Fits GARCH(1,1), uses the last conditional variance for 1-day VaR.
    """
    result = garch_11_fit(returns)
    last_vol = float(result.conditional_vol[-1])
    z = -2.326 if confidence >= 0.99 else -1.645  # normal quantile
    return last_vol * abs(z)


# ═══════════════════════════════════════════════════════════════
# EGARCH(1,1)
# ═══════════════════════════════════════════════════════════════

@dataclass
class EGARCHResult:
    """EGARCH(1,1) fit result."""
    omega: float
    alpha: float        # magnitude effect
    gamma: float        # leverage effect (negative → asymmetric)
    beta: float         # persistence
    log_likelihood: float
    n_obs: int
    conditional_vol: np.ndarray

    def to_dict(self) -> dict:
        return {
            "omega": self.omega, "alpha": self.alpha,
            "gamma": self.gamma, "beta": self.beta,
            "log_likelihood": self.log_likelihood, "n_obs": self.n_obs,
        }


def _egarch_loglik(params: np.ndarray, returns: np.ndarray) -> float:
    """Negative log-likelihood for EGARCH(1,1)."""
    omega, alpha, gamma, beta = params

    n = len(returns)
    log_var = np.zeros(n)
    log_var[0] = math.log(max(returns.var(), 1e-20))

    for t in range(1, n):
        vol_prev = math.exp(0.5 * log_var[t - 1])
        z = returns[t - 1] / max(vol_prev, 1e-10)
        g_z = alpha * (abs(z) - math.sqrt(2 / math.pi)) + gamma * z
        log_var[t] = omega + g_z + beta * log_var[t - 1]

    var = np.exp(log_var)
    ll = -0.5 * np.sum(log_var + returns ** 2 / var)
    return -ll


def egarch_fit(
    returns: np.ndarray,
    max_iter: int = 500,
) -> EGARCHResult:
    """Fit EGARCH(1,1) via maximum likelihood.

    log(sigma^2_t) = omega + alpha × (|z_{t-1}| - E|z|) + gamma × z_{t-1} + beta × log(sigma^2_{t-1})

    Captures leverage effect: negative gamma means negative returns
    increase vol more than positive returns of the same magnitude.
    """
    from scipy.optimize import minimize

    r = np.asarray(returns, dtype=float)
    n = len(r)
    sample_var = r.var()

    x0 = np.array([math.log(sample_var) * 0.05, 0.1, -0.05, 0.95])

    result = minimize(
        _egarch_loglik, x0, args=(r,),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-10, "fatol": 1e-10},
    )

    omega, alpha, gamma, beta = result.x

    # Compute conditional vol
    log_var = np.zeros(n)
    log_var[0] = math.log(max(sample_var, 1e-20))
    for t in range(1, n):
        vol_prev = math.exp(0.5 * log_var[t - 1])
        z = r[t - 1] / max(vol_prev, 1e-10)
        g_z = alpha * (abs(z) - math.sqrt(2 / math.pi)) + gamma * z
        log_var[t] = omega + g_z + beta * log_var[t - 1]

    return EGARCHResult(
        omega=float(omega),
        alpha=float(alpha),
        gamma=float(gamma),
        beta=float(beta),
        log_likelihood=float(-result.fun),
        n_obs=n,
        conditional_vol=np.exp(0.5 * log_var),
    )
