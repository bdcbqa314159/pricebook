"""
Heston stochastic volatility model.

Dynamics:
    dS/S = (r-q)dt + sqrt(v)dW1
    dv = kappa*(theta-v)dt + xi*sqrt(v)dW2
    dW1*dW2 = rho*dt

European options priced via the semi-analytical Heston (1993)
decomposition: C = S*exp(-qT)*P1 - K*exp(-rT)*P2.
"""

from __future__ import annotations

import math
import cmath

import numpy as np

from pricebook.black76 import OptionType
from pricebook.quadrature import gauss_legendre


def _heston_f(
    u: complex,
    T: float,
    rate: float,
    div_yield: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    x: float,
    j: int,
) -> complex:
    """Heston characteristic function f_j for measure j (1 or 2)."""
    if j == 1:
        u_j = 0.5
        b_j = kappa - rho * xi
    else:
        u_j = -0.5
        b_j = kappa

    i = 1j
    d = cmath.sqrt(
        (rho * xi * i * u - b_j) ** 2 - xi**2 * (2 * u_j * i * u - u**2)
    )

    g = (b_j - rho * xi * i * u + d) / (b_j - rho * xi * i * u - d)

    exp_dT = cmath.exp(d * T)

    if abs(1.0 - g * exp_dT) < 1e-20:
        return 0.0

    C = (rate - div_yield) * i * u * T + \
        kappa * theta / xi**2 * (
            (b_j - rho * xi * i * u + d) * T
            - 2.0 * cmath.log((1.0 - g * exp_dT) / (1.0 - g))
        )

    D = (b_j - rho * xi * i * u + d) / xi**2 * \
        (1.0 - exp_dT) / (1.0 - g * exp_dT)

    return cmath.exp(C + D * v0 + i * u * x)


def heston_price(
    spot: float,
    strike: float,
    rate: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    n_quad: int = 64,
) -> float:
    """European option price under Heston (1993).

    C = S*exp(-q*T)*P1 - K*exp(-r*T)*P2

    P_j = 0.5 + (1/pi) * ∫ Re[exp(-iu*ln(K)) * f_j(u) / (iu)] du
    """
    x = math.log(spot)
    log_K = math.log(strike)
    df = math.exp(-rate * T)

    def _integrand_p1(u):
        if u < 1e-10:
            return 0.0
        f1 = _heston_f(u, T, rate, div_yield, v0, kappa, theta, xi, rho, x, 1)
        return (cmath.exp(-1j * u * log_K) * f1 / (1j * u)).real

    def _integrand_p2(u):
        if u < 1e-10:
            return 0.0
        f2 = _heston_f(u, T, rate, div_yield, v0, kappa, theta, xi, rho, x, 2)
        return (cmath.exp(-1j * u * log_K) * f2 / (1j * u)).real

    r1 = gauss_legendre(_integrand_p1, 1e-6, 100.0, n=n_quad)
    r2 = gauss_legendre(_integrand_p2, 1e-6, 100.0, n=n_quad)

    P1 = 0.5 + r1.value / math.pi
    P2 = 0.5 + r2.value / math.pi

    call = spot * math.exp(-div_yield * T) * P1 - strike * df * P2

    if option_type == OptionType.CALL:
        return max(call, 0.0)
    put = call - spot * math.exp(-div_yield * T) + strike * df
    return max(put, 0.0)


def heston_calibrate(
    spot: float,
    strikes: list[float],
    market_prices: list[float],
    rate: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
) -> dict[str, float]:
    """Calibrate Heston parameters to market option prices.

    Fits: v0, kappa, theta, xi, rho via differential evolution.
    """
    from scipy.optimize import differential_evolution

    def objective(params):
        v0, kappa, theta, xi, rho = params
        total = 0.0
        for k, mp in zip(strikes, market_prices):
            try:
                model_price = heston_price(
                    spot, k, rate, T, v0, kappa, theta, xi, rho,
                    option_type, div_yield, n_quad=32,
                )
                total += (model_price - mp) ** 2
            except (ValueError, ZeroDivisionError, OverflowError):
                return 1e10
        return total

    bounds = [
        (0.001, 1.0),    # v0
        (0.01, 10.0),    # kappa
        (0.001, 1.0),    # theta
        (0.01, 2.0),     # xi
        (-0.99, 0.99),   # rho
    ]

    result = differential_evolution(objective, bounds, seed=42, maxiter=200, tol=1e-8)
    v0, kappa, theta, xi, rho = result.x
    rmse = math.sqrt(result.fun / len(strikes))

    return {
        "v0": v0, "kappa": kappa, "theta": theta,
        "xi": xi, "rho": rho, "rmse": rmse,
    }
