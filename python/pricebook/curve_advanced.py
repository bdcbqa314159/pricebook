"""Advanced yield curve construction and fitting.

Extends :mod:`pricebook.bootstrap` with parametric and smooth methods:

* :func:`nelson_siegel_fit` — Nelson-Siegel 4-parameter yield curve.
* :func:`svensson_fit` — Svensson 6-parameter extension.
* :func:`smooth_forward_curve` — monotone-convex forward interpolation.
* :func:`turn_of_year_adjustment` — seasonal funding premium correction.

References:
    Nelson & Siegel, *Parsimonious Modeling of Yield Curves*, J. Business, 1987.
    Svensson, *Estimating Forward Rates with the Extended NS Model*, IMF, 1994.
    Hagan & West, *Interpolation Methods for Curve Construction*, AMF, 2006.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# ---- Nelson-Siegel ----

@dataclass
class NSFitResult:
    """Nelson-Siegel fit result."""
    beta0: float   # long-run level
    beta1: float   # slope
    beta2: float   # curvature
    tau: float      # decay parameter
    residual: float
    method: str


def _ns_yield(t: float, beta0: float, beta1: float, beta2: float, tau: float) -> float:
    """Nelson-Siegel yield at maturity t."""
    if t < 1e-10:
        return beta0 + beta1
    x = t / tau
    exp_x = math.exp(-x)
    factor1 = (1 - exp_x) / x
    factor2 = factor1 - exp_x
    return beta0 + beta1 * factor1 + beta2 * factor2


def nelson_siegel_fit(
    maturities: list[float],
    yields: list[float],
    tau_init: float = 1.5,
) -> NSFitResult:
    """Fit Nelson-Siegel model to observed yield curve.

    y(t) = β₀ + β₁ [(1−e^{−t/τ})/(t/τ)]
             + β₂ [(1−e^{−t/τ})/(t/τ) − e^{−t/τ}]

    β₀ = long-run yield, β₁ = slope (short−long), β₂ = curvature (hump).

    Args:
        maturities: observed maturities (years).
        yields: observed zero-coupon yields.
        tau_init: initial guess for τ.
    """
    T = np.array(maturities)
    Y = np.array(yields)

    def objective(params):
        b0, b1, b2, tau = params
        if tau < 0.01:
            return 1e10
        model = np.array([_ns_yield(t, b0, b1, b2, tau) for t in T])
        return float(np.sum((model - Y) ** 2))

    # Initial guess from endpoints
    b0_init = float(Y[-1]) if len(Y) > 0 else 0.05
    b1_init = float(Y[0] - Y[-1]) if len(Y) > 1 else 0.0
    b2_init = 0.0

    x0 = [b0_init, b1_init, b2_init, tau_init]
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})

    b0, b1, b2, tau = result.x
    residual = math.sqrt(result.fun / len(T))

    return NSFitResult(b0, b1, b2, max(tau, 0.01), residual, "nelson_siegel")


def ns_yield_curve(
    fit: NSFitResult,
    maturities: np.ndarray | list[float],
) -> np.ndarray:
    """Evaluate Nelson-Siegel yield at given maturities."""
    return np.array([_ns_yield(t, fit.beta0, fit.beta1, fit.beta2, fit.tau)
                     for t in maturities])


# ---- Svensson ----

@dataclass
class SvenssonFitResult:
    """Svensson fit result."""
    beta0: float
    beta1: float
    beta2: float
    beta3: float   # second hump
    tau1: float
    tau2: float
    residual: float
    method: str


def _svensson_yield(t: float, beta0: float, beta1: float, beta2: float,
                    beta3: float, tau1: float, tau2: float) -> float:
    """Svensson yield at maturity t."""
    if t < 1e-10:
        return beta0 + beta1
    x1 = t / tau1
    x2 = t / tau2
    exp1 = math.exp(-x1)
    exp2 = math.exp(-x2)
    f1 = (1 - exp1) / x1
    f2 = f1 - exp1
    f3 = (1 - exp2) / x2 - exp2
    return beta0 + beta1 * f1 + beta2 * f2 + beta3 * f3


def svensson_fit(
    maturities: list[float],
    yields: list[float],
    tau1_init: float = 1.5,
    tau2_init: float = 5.0,
) -> SvenssonFitResult:
    """Fit Svensson (extended Nelson-Siegel) to yield curve.

    Adds a second hump term with independent decay parameter τ₂.
    Better fit for curves with double humps (e.g., post-crisis).

    Args:
        maturities: observed maturities.
        yields: observed zero-coupon yields.
    """
    T = np.array(maturities)
    Y = np.array(yields)

    def objective(params):
        b0, b1, b2, b3, tau1, tau2 = params
        if tau1 < 0.01 or tau2 < 0.01:
            return 1e10
        model = np.array([_svensson_yield(t, b0, b1, b2, b3, tau1, tau2) for t in T])
        return float(np.sum((model - Y) ** 2))

    b0_init = float(Y[-1]) if len(Y) > 0 else 0.05
    b1_init = float(Y[0] - Y[-1]) if len(Y) > 1 else 0.0

    x0 = [b0_init, b1_init, 0.0, 0.0, tau1_init, tau2_init]
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})

    b0, b1, b2, b3, tau1, tau2 = result.x
    residual = math.sqrt(result.fun / len(T))

    return SvenssonFitResult(b0, b1, b2, b3, max(tau1, 0.01), max(tau2, 0.01),
                             residual, "svensson")


def svensson_yield_curve(
    fit: SvenssonFitResult,
    maturities: np.ndarray | list[float],
) -> np.ndarray:
    """Evaluate Svensson yield at given maturities."""
    return np.array([_svensson_yield(t, fit.beta0, fit.beta1, fit.beta2,
                                     fit.beta3, fit.tau1, fit.tau2)
                     for t in maturities])


# ---- Smooth forward interpolation ----

@dataclass
class SmoothForwardResult:
    """Smooth forward curve result."""
    pillars: np.ndarray
    forwards: np.ndarray
    discount_factors: np.ndarray
    method: str


def smooth_forward_curve(
    pillar_times: list[float],
    zero_rates: list[float],
    n_output: int = 100,
) -> SmoothForwardResult:
    """Monotone-preserving forward rate interpolation.

    Computes instantaneous forward rates from zero rates using
    log-linear interpolation on discount factors (piecewise constant
    forward rates), then applies Hagan-West monotone convex smoothing.

    The key property: forward rates are continuous and the curve
    does not exhibit spurious oscillations.

    Args:
        pillar_times: tenor pillar times.
        zero_rates: zero rates at each pillar.
        n_output: number of output points.
    """
    T = np.array(pillar_times)
    R = np.array(zero_rates)
    n = len(T)

    # Discount factors at pillars
    df = np.exp(-R * T)

    # Forward rates between pillars (piecewise constant)
    fwd_pillars = np.zeros(n)
    fwd_pillars[0] = R[0]  # instantaneous rate at t=0
    for i in range(1, n):
        dt = T[i] - T[i - 1]
        if dt > 0:
            fwd_pillars[i] = -math.log(df[i] / df[i - 1]) / dt
        else:
            fwd_pillars[i] = fwd_pillars[i - 1]

    # Monotone-preserving interpolation (Hyman filter)
    # Compute slopes
    slopes = np.zeros(n)
    for i in range(1, n - 1):
        d_left = (fwd_pillars[i] - fwd_pillars[i - 1]) / (T[i] - T[i - 1]) if T[i] > T[i - 1] else 0.0
        d_right = (fwd_pillars[i + 1] - fwd_pillars[i]) / (T[i + 1] - T[i]) if T[i + 1] > T[i] else 0.0
        if d_left * d_right > 0:
            slopes[i] = 2 * d_left * d_right / (d_left + d_right)  # harmonic mean
        else:
            slopes[i] = 0.0

    # Evaluate on fine grid
    t_out = np.linspace(T[0], T[-1], n_output)
    f_out = np.zeros(n_output)

    for k, t in enumerate(t_out):
        # Find interval
        idx = min(np.searchsorted(T, t, side='right') - 1, n - 2)
        idx = max(idx, 0)

        dt = T[idx + 1] - T[idx]
        if dt < 1e-10:
            f_out[k] = fwd_pillars[idx]
            continue

        s = (t - T[idx]) / dt
        # Hermite interpolation
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2

        f_out[k] = (h00 * fwd_pillars[idx] + h10 * dt * slopes[idx]
                     + h01 * fwd_pillars[idx + 1] + h11 * dt * slopes[idx + 1])

    # Reconstruct discount factors from forwards
    df_out = np.ones(n_output)
    for k in range(1, n_output):
        dt_k = t_out[k] - t_out[k - 1]
        df_out[k] = df_out[k - 1] * math.exp(-f_out[k - 1] * dt_k)

    return SmoothForwardResult(t_out, f_out, df_out, "monotone_hermite")


# ---- Turn-of-year adjustment ----

@dataclass
class TOYResult:
    """Turn-of-year adjustment result."""
    adjusted_rates: np.ndarray
    toy_spreads: np.ndarray
    total_adjustment: float


def turn_of_year_adjustment(
    pillar_times: list[float],
    zero_rates: list[float],
    toy_spread: float = 0.0010,
    toy_window: float = 1.0 / 52,
    reference_date_fraction: float = 0.0,
) -> TOYResult:
    """Apply turn-of-year funding premium to yield curve.

    At year-end, funding rates typically spike due to regulatory
    reporting (Basel III leverage ratio, window dressing). This adds
    a Gaussian-shaped bump around each Dec 31.

    The TOY effect is modelled as:
        r_adj(t) = r(t) + Σ_k s × exp(−(t − t_k)² / (2w²))

    where t_k are year-end dates and w is the window width.

    Args:
        pillar_times: tenor times (in years from reference date).
        zero_rates: unadjusted zero rates.
        toy_spread: magnitude of year-end spread (in rate terms).
        toy_window: half-width of the effect (in years).
        reference_date_fraction: fraction of year for reference date.
    """
    T = np.array(pillar_times)
    R = np.array(zero_rates)

    # Find year-end times relative to reference date
    max_t = T[-1] if len(T) > 0 else 1.0
    year_ends = []
    for y in range(int(max_t) + 2):
        t_ye = (1.0 - reference_date_fraction) + y
        if 0 < t_ye <= max_t + 0.5:
            year_ends.append(t_ye)

    # Compute TOY adjustment at each pillar
    toy_adj = np.zeros(len(T))
    for t_ye in year_ends:
        toy_adj += toy_spread * np.exp(-0.5 * ((T - t_ye) / toy_window) ** 2)

    adjusted = R + toy_adj
    total = float(toy_adj.sum()) / max(len(T), 1)

    return TOYResult(adjusted, toy_adj, total)
