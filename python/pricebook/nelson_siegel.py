"""Nelson-Siegel and Svensson parametric yield curves.

Nelson-Siegel (1987): 3 factors (level, slope, curvature) + 1 decay.
    y(t) = beta0 + beta1 * (1-exp(-t/tau)) / (t/tau)
                  + beta2 * ((1-exp(-t/tau)) / (t/tau) - exp(-t/tau))

Svensson (1994): adds a second hump (6 parameters).
    y(t) = NS(t) + beta3 * ((1-exp(-t/tau2)) / (t/tau2) - exp(-t/tau2))
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from pricebook.discount_curve import DiscountCurve
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.optimization import minimize


def _ns_factor1(t: float, tau: float) -> float:
    """(1 - exp(-t/tau)) / (t/tau)."""
    x = t / tau
    if x < 1e-10:
        return 1.0
    return (1.0 - math.exp(-x)) / x


def _ns_factor2(t: float, tau: float) -> float:
    """(1 - exp(-t/tau)) / (t/tau) - exp(-t/tau)."""
    x = t / tau
    if x < 1e-10:
        return 0.0
    return (1.0 - math.exp(-x)) / x - math.exp(-x)


def nelson_siegel_yield(t: float, beta0: float, beta1: float, beta2: float, tau: float) -> float:
    """Nelson-Siegel zero rate at maturity t."""
    if t <= 0:
        return beta0 + beta1
    return beta0 + beta1 * _ns_factor1(t, tau) + beta2 * _ns_factor2(t, tau)


def svensson_yield(
    t: float, beta0: float, beta1: float, beta2: float, tau1: float,
    beta3: float, tau2: float,
) -> float:
    """Svensson zero rate at maturity t."""
    ns = nelson_siegel_yield(t, beta0, beta1, beta2, tau1)
    if t <= 0:
        return ns
    return ns + beta3 * _ns_factor2(t, tau2)


def ns_discount_curve(
    reference_date: date,
    beta0: float,
    beta1: float,
    beta2: float,
    tau: float,
    tenors: list[float] | None = None,
) -> DiscountCurve:
    """Build a DiscountCurve from Nelson-Siegel parameters."""
    if tenors is None:
        tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    dates = [date.fromordinal(reference_date.toordinal() + int(t * 365)) for t in tenors]
    dfs = [math.exp(-nelson_siegel_yield(t, beta0, beta1, beta2, tau) * t) for t in tenors]
    return DiscountCurve(reference_date, dates, dfs)


def svensson_discount_curve(
    reference_date: date,
    beta0: float,
    beta1: float,
    beta2: float,
    tau1: float,
    beta3: float,
    tau2: float,
    tenors: list[float] | None = None,
) -> DiscountCurve:
    """Build a DiscountCurve from Svensson parameters."""
    if tenors is None:
        tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    dates = [date.fromordinal(reference_date.toordinal() + int(t * 365)) for t in tenors]
    dfs = [math.exp(-svensson_yield(t, beta0, beta1, beta2, tau1, beta3, tau2) * t) for t in tenors]
    return DiscountCurve(reference_date, dates, dfs)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_nelson_siegel(
    tenors: list[float],
    market_yields: list[float],
    initial_guess: tuple[float, float, float, float] | None = None,
) -> dict[str, float]:
    """Calibrate NS parameters to market zero rates.

    Returns dict with beta0, beta1, beta2, tau, rmse.
    """
    if initial_guess is None:
        initial_guess = (market_yields[-1], market_yields[0] - market_yields[-1], 0.0, 2.0)

    def objective(params):
        b0, b1, b2, tau = params
        if tau <= 0.01:
            return 1e10
        return sum(
            (nelson_siegel_yield(t, b0, b1, b2, tau) - y) ** 2
            for t, y in zip(tenors, market_yields)
        )

    result = minimize(objective, x0=list(initial_guess), method="nelder_mead",
                      tol=1e-14, maxiter=5000)

    b0, b1, b2, tau = result.x
    rmse = math.sqrt(result.fun / len(tenors))
    return {"beta0": b0, "beta1": b1, "beta2": b2, "tau": tau, "rmse": rmse}


def calibrate_svensson(
    tenors: list[float],
    market_yields: list[float],
    initial_guess: tuple | None = None,
) -> dict[str, float]:
    """Calibrate Svensson parameters to market zero rates."""
    if initial_guess is None:
        initial_guess = (market_yields[-1], market_yields[0] - market_yields[-1],
                         0.0, 2.0, 0.0, 5.0)

    def objective(params):
        b0, b1, b2, tau1, b3, tau2 = params
        if tau1 <= 0.01 or tau2 <= 0.01:
            return 1e10
        return sum(
            (svensson_yield(t, b0, b1, b2, tau1, b3, tau2) - y) ** 2
            for t, y in zip(tenors, market_yields)
        )

    result = minimize(objective, x0=list(initial_guess), method="nelder_mead",
                      tol=1e-14, maxiter=10000)

    b0, b1, b2, tau1, b3, tau2 = result.x
    rmse = math.sqrt(result.fun / len(tenors))
    return {"beta0": b0, "beta1": b1, "beta2": b2, "tau1": tau1,
            "beta3": b3, "tau2": tau2, "rmse": rmse}
