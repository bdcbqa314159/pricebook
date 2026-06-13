"""Global multi-curve solver: simultaneous Newton for all pillar DFs.

Instead of sequential bootstrap (one pillar at a time), solves the
entire system F(df) = 0 simultaneously using Newton's method with
a finite-difference Jacobian.

Enables coupled multi-curve solving where OIS and projection curves
are interdependent.
"""

from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.interpolation import InterpolationMethod
from pricebook.core.schedule import Frequency, generate_schedule

if TYPE_CHECKING:
    from pricebook.market_data import MarketSnapshot


# ---------------------------------------------------------------------------
# Single-curve global Newton
# ---------------------------------------------------------------------------


def global_bootstrap(
    reference_date: date,
    deposits: list[tuple[date, float]],
    swaps: list[tuple[date, float]],
    deposit_dc: DayCountConvention = DayCountConvention.ACT_360,
    swap_dc: DayCountConvention = DayCountConvention.THIRTY_360,
    swap_frequency: Frequency = Frequency.SEMI_ANNUAL,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
    tol: float = 1e-12,
    max_iter: int = 50,
    *,
    market_snapshot: MarketSnapshot | None = None,
) -> DiscountCurve:
    """Bootstrap a discount curve via global Newton iteration.

    Solves F(df_vector) = 0 simultaneously for all pillar DFs,
    where F_i is the pricing error of instrument i.

    Each maturity must be unique across deposits + swaps: the residual
    vector is indexed by pillar date, so two instruments at the same
    maturity would silently overwrite each other (fix L1 A.2 B1).
    """
    # Collect all instruments and detect duplicate maturities.
    # Pre-fix: a 1Y deposit @ 5% + 1Y swap @ 4% silently dropped the deposit
    # constraint; the resulting curve only honoured the swap.
    all_instruments = []
    pillar_dates: list[date] = []
    seen_maturities: dict[date, str] = {}  # mat -> instrument-type label for diagnostic

    for mat, rate in sorted(deposits, key=lambda x: x[0]):
        if mat in seen_maturities:
            raise ValueError(
                f"Duplicate maturity {mat}: already provided by "
                f"{seen_maturities[mat]!r}, also requested as 'deposit'. "
                f"Each maturity must appear at most once across deposits + swaps."
            )
        seen_maturities[mat] = "deposit"
        all_instruments.append(("deposit", mat, rate))
        pillar_dates.append(mat)

    for mat, rate in sorted(swaps, key=lambda x: x[0]):
        if mat in seen_maturities:
            raise ValueError(
                f"Duplicate maturity {mat}: already provided by "
                f"{seen_maturities[mat]!r}, also requested as 'swap'. "
                f"Each maturity must appear at most once across deposits + swaps."
            )
        seen_maturities[mat] = "swap"
        all_instruments.append(("swap", mat, rate))
        pillar_dates.append(mat)

    pillar_dates = sorted(set(pillar_dates))
    n = len(pillar_dates)
    pillar_idx = {d: i for i, d in enumerate(pillar_dates)}

    # Initial guess: flat 5% curve
    dfs = np.array([math.exp(-0.05 * year_fraction(reference_date, d, deposit_dc))
                    for d in pillar_dates])

    def _make_curve(df_vec):
        return DiscountCurve(reference_date, pillar_dates, list(df_vec),
                             interpolation=interpolation)

    def _residuals(df_vec):
        curve = _make_curve(df_vec)
        res = np.zeros(n)

        for inst_type, mat, rate in all_instruments:
            idx = pillar_idx[mat]

            if inst_type == "deposit":
                tau = year_fraction(reference_date, mat, deposit_dc)
                # PV condition: df = 1/(1 + r*tau)
                res[idx] = df_vec[idx] - 1.0 / (1.0 + rate * tau)

            elif inst_type == "swap":
                schedule = generate_schedule(reference_date, mat, swap_frequency)
                # Swap PV = 0: fixed_pv = float_pv
                # fixed_pv = rate * sum(tau_i * df_i)
                # float_pv = df(start) - df(end) = 1 - df(mat)
                fixed_pv = 0.0
                for i in range(1, len(schedule)):
                    tau = year_fraction(schedule[i-1], schedule[i], swap_dc)
                    fixed_pv += rate * tau * curve.df(schedule[i])
                float_pv = 1.0 - curve.df(mat)
                res[idx] = fixed_pv - float_pv

        return res

    def _jacobian_analytical(df_vec):
        """Analytical Jacobian: O(n) instead of O(n²).

        For deposits: ∂res_i/∂df_i = 1 (direct residual).
        For swaps: res = rate × Σ(tau_j × df_j) - 1 + df_mat.
            ∂res/∂df_j = rate × tau_j  (if j is a coupon date)
            ∂res/∂df_mat += 1           (principal at maturity)

        The Jacobian is sparse: each swap row has non-zero entries
        only at its coupon dates and maturity.
        """
        curve = _make_curve(df_vec)
        J = np.zeros((n, n))

        for inst_type, mat, rate in all_instruments:
            row = pillar_idx[mat]

            if inst_type == "deposit":
                J[row, row] = 1.0

            elif inst_type == "swap":
                schedule = generate_schedule(reference_date, mat, swap_frequency)
                for k in range(1, len(schedule)):
                    tau = year_fraction(schedule[k - 1], schedule[k], swap_dc)
                    # Find which pillar this coupon date maps to
                    # Coupon dates may not be exact pillars — use interpolation derivative
                    # For simplicity, if schedule[k] is a pillar date, direct entry
                    if schedule[k] in pillar_idx:
                        col = pillar_idx[schedule[k]]
                        J[row, col] += rate * tau
                    else:
                        # Coupon date between pillars: df comes from interpolation
                        # Use FD for this entry only (rare case)
                        for j in range(n):
                            df_up = df_vec.copy()
                            df_up[j] += 1e-8
                            c_up = DiscountCurve(reference_date, pillar_dates,
                                                  list(df_up), interpolation=interpolation)
                            J[row, j] += rate * tau * (c_up.df(schedule[k]) - curve.df(schedule[k])) / 1e-8

                # Principal: ∂(1 - df_mat)/∂df_mat = -1, so ∂res/∂df_mat gets +1
                # (res = fixed - float = fixed - (1 - df_mat), so ∂res/∂df_mat = +1)
                J[row, row] += 1.0

        return J

    def _jacobian_fd(df_vec, eps=None):
        """Finite-difference Jacobian (fallback). Auto-scales eps per element."""
        f0 = _residuals(df_vec)
        J = np.zeros((n, n))
        for j in range(n):
            h = eps or max(abs(df_vec[j]) * 1e-7, 1e-10)
            df_bump = df_vec.copy()
            df_bump[j] += h
            f_bump = _residuals(df_bump)
            J[:, j] = (f_bump - f0) / h
        return J

    _jacobian = _jacobian_analytical

    # Newton iteration
    import warnings
    converged = False
    for iteration in range(max_iter):
        res = _residuals(dfs)
        if np.max(np.abs(res)) < tol:
            converged = True
            break
        J = _jacobian(dfs)
        try:
            delta = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            break
        dfs += delta
        dfs = np.maximum(dfs, 1e-10)  # keep positive

    if not converged:
        warnings.warn(
            f"global_bootstrap did not converge after {max_iter} iterations "
            f"(residual={np.max(np.abs(_residuals(dfs))):.2e})",
            RuntimeWarning,
        )

    curve = _make_curve(dfs)

    # Attach canonical CalibrationResult (G1 P1 Slice 5).
    final_res = _residuals(dfs)
    from pricebook.calibration import (
        CalibrationDiagnostics,
        CalibrationResult,
        ObjectiveKind,
        OptimiserSpec,
    )
    quotes = []
    for inst_type, mat, _rate in all_instruments:
        quotes.append(f"{inst_type}_{mat.isoformat()}")
    parameters = {f"df({d.isoformat()})": float(df) for d, df in zip(pillar_dates, dfs)}
    curve.calibration_result = CalibrationResult.new(
        model_class="discount_curve_global",
        parameters=parameters,
        residuals=[float(r) for r in final_res],
        objective=ObjectiveKind.SSE,
        optimiser=OptimiserSpec(
            algorithm="newton-global",
            tolerance=tol,
            max_iterations=max_iter,
            extra={
                "interpolation": str(interpolation.value),
                "deposit_dc": str(deposit_dc.value),
                "swap_dc": str(swap_dc.value),
            },
        ),
        iterations=int(iteration + 1) if 'iteration' in dir() else 0,
        converged=bool(converged),
        quotes_fitted=quotes,
        diagnostics=CalibrationDiagnostics(
            extra={
                "n_deposits": len(deposits),
                "n_swaps": len(swaps),
                "max_residual_abs": float(np.max(np.abs(final_res))),
            },
        ),
        market_snapshot_id=market_snapshot.id if market_snapshot is not None else None,
    )
    return curve


# ---------------------------------------------------------------------------
# Coupled multi-curve solve
# ---------------------------------------------------------------------------


def coupled_bootstrap(
    reference_date: date,
    ois_deposits: list[tuple[date, float]],
    ois_swaps: list[tuple[date, float]],
    projection_swaps: list[tuple[date, float]],
    deposit_dc: DayCountConvention = DayCountConvention.ACT_360,
    swap_dc: DayCountConvention = DayCountConvention.THIRTY_360,
    ois_frequency: Frequency = Frequency.ANNUAL,
    proj_frequency: Frequency = Frequency.SEMI_ANNUAL,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> tuple[DiscountCurve, DiscountCurve]:
    """Simultaneously solve for OIS discount and projection curves.

    The OIS curve discounts all cashflows. The projection curve
    determines floating forward rates. Both are solved together.

    Returns:
        (ois_curve, projection_curve)
    """
    # Collect pillar dates for each curve
    ois_dates = sorted(set(
        [d for d, _ in ois_deposits] + [d for d, _ in ois_swaps]
    ))
    proj_dates = sorted(set([d for d, _ in projection_swaps]))

    n_ois = len(ois_dates)
    n_proj = len(proj_dates)
    n_total = n_ois + n_proj

    ois_idx = {d: i for i, d in enumerate(ois_dates)}
    proj_idx = {d: i + n_ois for i, d in enumerate(proj_dates)}

    # Initial guess: flat curves
    x = np.zeros(n_total)
    for i, d in enumerate(ois_dates):
        t = year_fraction(reference_date, d, deposit_dc)
        x[i] = math.exp(-0.05 * t)
    for i, d in enumerate(proj_dates):
        t = year_fraction(reference_date, d, deposit_dc)
        x[n_ois + i] = math.exp(-0.05 * t)

    def _make_curves(xvec):
        ois = DiscountCurve(reference_date, ois_dates, list(xvec[:n_ois]),
                            interpolation=interpolation)
        proj = DiscountCurve(reference_date, proj_dates, list(xvec[n_ois:]),
                             interpolation=interpolation)
        return ois, proj

    def _residuals(xvec):
        ois, proj = _make_curves(xvec)
        res = np.zeros(n_total)

        # OIS deposits
        for d, rate in ois_deposits:
            idx = ois_idx[d]
            tau = year_fraction(reference_date, d, deposit_dc)
            res[idx] = xvec[idx] - 1.0 / (1.0 + rate * tau)

        # OIS swaps (single curve: discount = projection)
        for d, rate in ois_swaps:
            idx = ois_idx[d]
            schedule = generate_schedule(reference_date, d, ois_frequency)
            fixed_pv = sum(
                rate * year_fraction(schedule[i-1], schedule[i], swap_dc) * ois.df(schedule[i])
                for i in range(1, len(schedule))
            )
            float_pv = 1.0 - ois.df(d)
            res[idx] = fixed_pv - float_pv

        # Projection swaps (dual-curve: discount off OIS, forward off proj)
        for d, rate in projection_swaps:
            idx = proj_idx[d]
            schedule = generate_schedule(reference_date, d, proj_frequency)
            fixed_pv = sum(
                rate * year_fraction(schedule[i-1], schedule[i], swap_dc) * ois.df(schedule[i])
                for i in range(1, len(schedule))
            )
            # Floating: sum of fwd * tau * df_ois.
            # Fix T4-GS1: pre-fix a degenerate ``df2 <= 0`` or ``tau <= 0``
            # silently set ``fwd = 0.0`` for that period, contributing 0
            # to ``float_pv``.  This made the residual artificially LOW
            # (fixed_pv − too-small-float_pv), which Newton can drive
            # toward zero by adjusting DFs along an unphysical trajectory
            # — silently converging on a bad solution.  Raise instead so
            # the coupled bootstrap surfaces the upstream degeneracy.
            float_pv = 0.0
            for i in range(1, len(schedule)):
                tau = year_fraction(schedule[i-1], schedule[i], deposit_dc)
                df1 = proj.df(schedule[i-1])
                df2 = proj.df(schedule[i])
                if tau <= 0:
                    raise ValueError(
                        f"coupled_bootstrap: non-positive tau ({tau}) "
                        f"between {schedule[i-1]} and {schedule[i]}.  "
                        "Check schedule for duplicate/inverted dates."
                    )
                if df2 <= 0:
                    raise ValueError(
                        f"coupled_bootstrap: projection-curve DF went "
                        f"non-positive at {schedule[i]} (df={df2}).  "
                        "The Newton iterate has produced an arbitrageable "
                        "curve — check input deposit/swap rates."
                    )
                fwd = (df1 - df2) / (tau * df2)
                float_pv += fwd * tau * ois.df(schedule[i])
            res[idx] = fixed_pv - float_pv

        return res

    def _jacobian(xvec, eps=None):
        f0 = _residuals(xvec)
        J = np.zeros((n_total, n_total))
        for j in range(n_total):
            h = eps or max(abs(xvec[j]) * 1e-7, 1e-10)
            x_bump = xvec.copy()
            x_bump[j] += h
            J[:, j] = (_residuals(x_bump) - f0) / h
        return J

    import warnings
    converged = False
    for iteration in range(max_iter):
        res = _residuals(x)
        if np.max(np.abs(res)) < tol:
            converged = True
            break
        J = _jacobian(x)
        try:
            delta = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            break
        x += delta
        x = np.maximum(x, 1e-10)

    if not converged:
        warnings.warn(
            f"coupled_bootstrap did not converge after {max_iter} iterations "
            f"(residual={np.max(np.abs(_residuals(x))):.2e})",
            RuntimeWarning,
        )

    ois, proj = _make_curves(x)
    return ois, proj
