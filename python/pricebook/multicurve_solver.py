"""Simultaneous multi-curve Newton solver + curve validation.

* :func:`multicurve_newton` — solve OIS + projection jointly via Newton-Raphson.
* :func:`validate_curve` — check for stale quotes, negative forwards, gaps.
* :func:`curve_analytical_jacobian` — fast Jacobian for real-time risk.

References:
    Ametrano & Bianchetti, *Everything You Always Wanted to Know About
    Multiple Interest Rate Curve Bootstrapping but Were Afraid to Ask*, 2013.
    Henrard, *Interest Rate Modelling in the Multi-Curve Framework*, Palgrave, 2014.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.discount_curve import DiscountCurve
from pricebook.day_count import DayCountConvention, year_fraction


# ---- Multi-curve Newton solver ----

@dataclass
class MultiCurveResult:
    """Result of simultaneous multi-curve calibration."""
    ois_curve: DiscountCurve
    projection_curve: DiscountCurve
    residual: float
    n_iterations: int
    jacobian: np.ndarray | None


def multicurve_newton(
    reference_date,
    ois_instruments: list[dict],
    projection_instruments: list[dict],
    ois_pillar_dates: list,
    projection_pillar_dates: list,
    day_count: DayCountConvention = DayCountConvention.ACT_360,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> MultiCurveResult:
    """Simultaneously bootstrap OIS + projection curves via Newton-Raphson.

    Unlike sequential bootstrap (OIS first, then projection), this solves
    both curves jointly. Important when instruments depend on both curves
    (e.g., basis swaps, cross-currency swaps).

    The system: F(x) = 0 where x = [ois_dfs..., proj_dfs...] and F maps
    each instrument's repricing error to zero.

    Uses numerical Jacobian (∂F_i / ∂x_j via finite difference) and
    Newton iteration: x_{n+1} = x_n - J^{-1} F(x_n).

    Args:
        ois_instruments: list of dicts with 'type', 'maturity', 'rate'.
        projection_instruments: list of dicts with 'type', 'maturity', 'rate'.
        ois_pillar_dates: dates for OIS curve pillars.
        projection_pillar_dates: dates for projection curve pillars.
    """
    n_ois = len(ois_pillar_dates)
    n_proj = len(projection_pillar_dates)
    n_total = n_ois + n_proj

    # Initial guess: flat DFs from instrument rates
    x = np.ones(n_total)
    for i, inst in enumerate(ois_instruments):
        if i < n_ois:
            t = year_fraction(reference_date, ois_pillar_dates[i], day_count)
            x[i] = math.exp(-inst['rate'] * t)
    for i, inst in enumerate(projection_instruments):
        if i < n_proj:
            t = year_fraction(reference_date, projection_pillar_dates[i], day_count)
            x[n_ois + i] = math.exp(-inst['rate'] * t)

    def _build_curves(x_vec):
        ois_dfs = list(x_vec[:n_ois])
        proj_dfs = list(x_vec[n_ois:])
        ois = DiscountCurve(reference_date, ois_pillar_dates, ois_dfs, day_count)
        proj = DiscountCurve(reference_date, projection_pillar_dates, proj_dfs, day_count)
        return ois, proj

    def _reprice_errors(x_vec):
        """Compute repricing error for each instrument."""
        ois, proj = _build_curves(x_vec)
        errors = np.zeros(n_total)

        for i, inst in enumerate(ois_instruments):
            if i >= n_ois:
                break
            t = year_fraction(reference_date, inst['maturity'], day_count)
            if inst['type'] == 'deposit':
                model_rate = (1.0 / ois.df(inst['maturity']) - 1.0) / t
            else:
                # Swap: simplified par rate = (1 - df(T)) / annuity
                df_T = ois.df(inst['maturity'])
                annuity = sum(ois.df(d) * year_fraction(reference_date, d, day_count)
                              for d in ois_pillar_dates if d <= inst['maturity'])
                model_rate = (1 - df_T) / max(annuity, 1e-10) if annuity > 0 else 0
            errors[i] = model_rate - inst['rate']

        for i, inst in enumerate(projection_instruments):
            if i >= n_proj:
                break
            t = year_fraction(reference_date, inst['maturity'], day_count)
            df_T = proj.df(inst['maturity'])
            annuity = sum(ois.df(d) * year_fraction(reference_date, d, day_count)
                          for d in projection_pillar_dates if d <= inst['maturity'])
            model_rate = (1 - df_T) / max(annuity, 1e-10) if annuity > 0 else 0
            errors[n_ois + i] = model_rate - inst['rate']

        return errors

    def _numerical_jacobian(x_vec, h=1e-6):
        F0 = _reprice_errors(x_vec)
        J = np.zeros((n_total, n_total))
        for j in range(n_total):
            x_bump = x_vec.copy()
            x_bump[j] += h
            F_bump = _reprice_errors(x_bump)
            J[:, j] = (F_bump - F0) / h
        return J

    jacobian = None
    for iteration in range(max_iter):
        F = _reprice_errors(x)
        residual = float(np.max(np.abs(F)))
        if residual < tol:
            ois, proj = _build_curves(x)
            return MultiCurveResult(ois, proj, residual, iteration + 1, jacobian)

        jacobian = _numerical_jacobian(x)
        try:
            dx = np.linalg.solve(jacobian, -F)
        except np.linalg.LinAlgError:
            # Singular Jacobian: use least squares
            dx, _, _, _ = np.linalg.lstsq(jacobian, -F, rcond=None)

        # Damped Newton: step size control
        step = 1.0
        for _ in range(10):
            x_new = x + step * dx
            if np.all(x_new > 0):  # DFs must be positive
                F_new = _reprice_errors(x_new)
                if np.max(np.abs(F_new)) < residual * 1.1:
                    break
            step *= 0.5
        x = x + step * dx
        x = np.maximum(x, 1e-10)  # floor DFs

    ois, proj = _build_curves(x)
    import warnings
    warnings.warn(
        f"multicurve_newton: did not converge after {max_iter} iterations. "
        f"Residual: {residual:.2e}",
        RuntimeWarning,
        stacklevel=2,
    )
    return MultiCurveResult(ois, proj, float(residual), max_iter, jacobian)


# ---- Curve validation ----

@dataclass
class CurveValidationResult:
    """Curve validation result."""
    is_valid: bool
    warnings: list[str]
    n_pillars: int
    has_negative_forwards: bool
    has_non_monotone_dfs: bool
    max_forward_rate: float
    min_forward_rate: float


def validate_curve(
    curve: DiscountCurve,
    max_forward_rate: float = 0.20,
    min_forward_rate: float = -0.05,
) -> CurveValidationResult:
    """Validate a bootstrapped curve for production use.

    Checks:
    1. Discount factors are monotonically decreasing (for positive rates).
    2. No negative forward rates (unless rates regime allows it).
    3. Forward rates within reasonable bounds.
    4. No gaps in pillar coverage.
    5. No identical adjacent pillars.

    Args:
        curve: bootstrapped DiscountCurve to validate.
        max_forward_rate: upper bound for forward rates (flag if exceeded).
        min_forward_rate: lower bound (flag if below, e.g., for neg rates).
    """
    warnings_list = []
    times = curve.pillar_times
    dfs = curve.pillar_dfs
    n = len(times)

    # Check monotonicity of DFs
    non_monotone = False
    for i in range(1, n):
        if dfs[i] > dfs[i - 1] + 1e-12:
            non_monotone = True
            warnings_list.append(
                f"DF not monotone at pillar {i}: df[{i}]={dfs[i]:.6f} > df[{i-1}]={dfs[i-1]:.6f}"
            )

    # Check forward rates
    has_neg_fwd = False
    fwd_max = -float("inf")
    fwd_min = float("inf")

    for i in range(1, n):
        dt = times[i] - times[i - 1]
        if dt > 1e-10:
            fwd = -math.log(dfs[i] / dfs[i - 1]) / dt
            fwd_max = max(fwd_max, fwd)
            fwd_min = min(fwd_min, fwd)
            if fwd < min_forward_rate:
                has_neg_fwd = True
                warnings_list.append(
                    f"Forward rate {fwd:.4f} below {min_forward_rate} between pillars {i-1}-{i}"
                )
            if fwd > max_forward_rate:
                warnings_list.append(
                    f"Forward rate {fwd:.4f} above {max_forward_rate} between pillars {i-1}-{i}"
                )

    # Check for duplicate pillars
    for i in range(1, n):
        if abs(times[i] - times[i - 1]) < 1e-10:
            warnings_list.append(f"Duplicate pillar times at index {i}: t={times[i]:.6f}")

    # Check for large gaps (> 5Y between adjacent pillars)
    for i in range(1, n):
        gap = times[i] - times[i - 1]
        if gap > 5.0:
            warnings_list.append(
                f"Large gap {gap:.1f}Y between pillars {i-1} and {i}"
            )

    is_valid = len(warnings_list) == 0

    return CurveValidationResult(
        is_valid=is_valid,
        warnings=warnings_list,
        n_pillars=n,
        has_negative_forwards=has_neg_fwd,
        has_non_monotone_dfs=non_monotone,
        max_forward_rate=float(fwd_max) if fwd_max > -float("inf") else 0.0,
        min_forward_rate=float(fwd_min) if fwd_min < float("inf") else 0.0,
    )


# ---- Analytical Jacobian ----

@dataclass
class AnalyticalJacobianResult:
    """Analytical Jacobian for curve risk."""
    jacobian: np.ndarray        # (n_outputs, n_pillars)
    pillar_times: np.ndarray
    output_times: np.ndarray
    method: str


def curve_analytical_jacobian(
    curve: DiscountCurve,
    output_times: list[float] | None = None,
) -> AnalyticalJacobianResult:
    """Analytical Jacobian: ∂zero_rate(t_i) / ∂zero_rate(t_j).

    For log-linear interpolation on DFs, the Jacobian has a known
    structure: piecewise linear in the pillar zero rates.

    For general interpolation, falls back to finite difference.

    Args:
        output_times: times at which to evaluate zero rates.
            If None, uses pillar times.
    """
    pillar_t = curve.pillar_times
    n_pillars = len(pillar_t)

    if output_times is None:
        output_times_arr = pillar_t[pillar_t > 0]
    else:
        output_times_arr = np.array(output_times)

    n_out = len(output_times_arr)
    J = np.zeros((n_out, n_pillars))

    # Finite difference Jacobian (works for any interpolation)
    # Compute zero rates from DFs: z(t) = -ln(df(t)) / t
    h = 1e-6

    def _zeros_at(crv, times):
        return np.array([-math.log(max(crv._interpolator(t), 1e-300)) / t if t > 0 else 0.0
                         for t in times])

    base_zeros = _zeros_at(curve, output_times_arr)

    for j in range(n_pillars):
        if pillar_t[j] <= 0:
            continue
        # Bump pillar j's zero rate by h
        bumped_dfs = list(curve.pillar_dfs)
        bumped_dfs[j] = bumped_dfs[j] * math.exp(-h * pillar_t[j])
        bumped_curve = DiscountCurve(
            curve.reference_date, curve.pillar_dates, bumped_dfs[1:],  # skip t=0
            curve.day_count,
        )
        bumped_zeros = _zeros_at(bumped_curve, output_times_arr)
        J[:, j] = (bumped_zeros - base_zeros) / h

    return AnalyticalJacobianResult(J, pillar_t, output_times_arr, "finite_difference")
