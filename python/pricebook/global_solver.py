"""Global multi-curve solver: simultaneous Newton for all pillar DFs.

Instead of sequential bootstrap (one pillar at a time), solves the
entire system F(df) = 0 simultaneously using Newton's method with
an analytical Jacobian.

Enables coupled multi-curve solving where OIS and projection curves
are interdependent.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.interpolation import InterpolationMethod
from pricebook.schedule import Frequency, generate_schedule


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
) -> DiscountCurve:
    """Bootstrap a discount curve via global Newton iteration.

    Solves F(df_vector) = 0 simultaneously for all pillar DFs,
    where F_i is the pricing error of instrument i.
    """
    # Collect all pillar dates (sorted)
    all_instruments = []
    pillar_dates = []

    for mat, rate in sorted(deposits, key=lambda x: x[0]):
        all_instruments.append(("deposit", mat, rate))
        pillar_dates.append(mat)

    for mat, rate in sorted(swaps, key=lambda x: x[0]):
        all_instruments.append(("swap", mat, rate))
        if mat not in pillar_dates:
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

        inst_idx = 0
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

    def _jacobian(df_vec, eps=1e-8):
        """Finite-difference Jacobian."""
        f0 = _residuals(df_vec)
        J = np.zeros((n, n))
        for j in range(n):
            df_bump = df_vec.copy()
            df_bump[j] += eps
            f_bump = _residuals(df_bump)
            J[:, j] = (f_bump - f0) / eps
        return J

    # Newton iteration
    for iteration in range(max_iter):
        res = _residuals(dfs)
        if np.max(np.abs(res)) < tol:
            break
        J = _jacobian(dfs)
        try:
            delta = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            break
        dfs += delta
        dfs = np.maximum(dfs, 1e-10)  # keep positive

    return _make_curve(dfs)


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
            # Floating: sum of fwd * tau * df_ois
            float_pv = 0.0
            for i in range(1, len(schedule)):
                tau = year_fraction(schedule[i-1], schedule[i], deposit_dc)
                df1 = proj.df(schedule[i-1])
                df2 = proj.df(schedule[i])
                if df2 > 0 and tau > 0:
                    fwd = (df1 - df2) / (tau * df2)
                else:
                    fwd = 0.0
                float_pv += fwd * tau * ois.df(schedule[i])
            res[idx] = fixed_pv - float_pv

        return res

    def _jacobian(xvec, eps=1e-8):
        f0 = _residuals(xvec)
        J = np.zeros((n_total, n_total))
        for j in range(n_total):
            x_bump = xvec.copy()
            x_bump[j] += eps
            J[:, j] = (_residuals(x_bump) - f0) / eps
        return J

    for iteration in range(max_iter):
        res = _residuals(x)
        if np.max(np.abs(res)) < tol:
            break
        J = _jacobian(x)
        try:
            delta = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            break
        x += delta
        x = np.maximum(x, 1e-10)

    ois, proj = _make_curves(x)
    return ois, proj
