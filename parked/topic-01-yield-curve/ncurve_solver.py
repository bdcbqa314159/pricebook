"""N-curve simultaneous global solver.

Generalises dual-curve bootstrap to N curves solved simultaneously via
damped Newton-Raphson. Needed for: SOFR OIS + 1M projection + 3M projection
+ tenor basis, or multi-currency xccy setups.

    from pricebook.curves.ncurve_solver import (
        ncurve_solve, CurveSpec, InstrumentPricer, NCurveResult,
    )

References:
    Ametrano & Bianchetti (2013). Everything You Always Wanted to Know
    About Multiple Interest Rate Curve Bootstrapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

import numpy as np

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.interpolation import InterpolationMethod


class InstrumentPricer(Protocol):
    """Protocol: an instrument that can reprice given named curves."""

    def reprice(self, curves: dict[str, DiscountCurve]) -> float:
        """Return pricing error (model - market). Should be 0 at solution."""
        ...


@dataclass
class CurveSpec:
    """Specification for one curve in the N-curve system."""
    name: str                    # e.g. "ois", "sofr_1m", "sofr_3m"
    pillar_dates: list[date]
    initial_guess: list[float] | None = None  # initial DFs (default: flat at 1.0)
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR


@dataclass
class NCurveResult:
    """Result of N-curve simultaneous solve."""
    curves: dict[str, DiscountCurve]
    residual: float              # max absolute pricing error
    n_iterations: int
    converged: bool
    n_curves: int
    n_instruments: int
    pillar_counts: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "n_curves": self.n_curves,
            "n_instruments": self.n_instruments,
            "residual": self.residual,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "pillar_counts": self.pillar_counts,
        }


# ═══════════════════════════════════════════════════════════════
# Concrete instrument pricers
# ═══════════════════════════════════════════════════════════════


class DepositPricer:
    """Deposit: df(T) should equal 1/(1 + r×τ)."""

    def __init__(self, curve_name: str, maturity: date, rate: float,
                 reference_date: date, dc: DayCountConvention = DayCountConvention.ACT_360):
        self.curve_name = curve_name
        self.maturity = maturity
        self.rate = rate
        self.reference_date = reference_date
        self.dc = dc

    def reprice(self, curves: dict[str, DiscountCurve]) -> float:
        curve = curves[self.curve_name]
        t = year_fraction(self.reference_date, self.maturity, self.dc)
        model_df = curve.df(self.maturity)
        target_df = 1.0 / (1.0 + self.rate * t)
        return model_df - target_df


class OISSwapPricer:
    """OIS swap: PV_fixed = PV_float on the OIS curve."""

    def __init__(self, curve_name: str, maturity: date, par_rate: float,
                 reference_date: date, frequency: int = 1,
                 dc: DayCountConvention = DayCountConvention.ACT_360):
        self.curve_name = curve_name
        self.maturity = maturity
        self.par_rate = par_rate
        self.reference_date = reference_date
        self.frequency = frequency
        self.dc = dc

    def reprice(self, curves: dict[str, DiscountCurve]) -> float:
        curve = curves[self.curve_name]
        n = max(int(round(
            year_fraction(self.reference_date, self.maturity, self.dc) * self.frequency
        )), 1)
        annuity = 0.0
        for i in range(1, n + 1):
            t = i / self.frequency
            y = self.reference_date.year + int(t)
            m = self.reference_date.month
            d_i = date(y, m, min(self.reference_date.day, 28))
            if d_i > self.maturity:
                d_i = self.maturity
            tau = 1.0 / self.frequency
            annuity += tau * curve.df(d_i)
        # PV_fixed = par × annuity, PV_float = 1 - df(T)
        pv_fixed = self.par_rate * annuity
        pv_float = 1.0 - curve.df(self.maturity)
        return pv_fixed - pv_float


class BasisSwapPricer:
    """Basis swap: spread between two projection curves."""

    def __init__(self, disc_curve: str, proj_curve_pay: str, proj_curve_recv: str,
                 maturity: date, basis_spread: float, reference_date: date,
                 dc: DayCountConvention = DayCountConvention.ACT_360):
        self.disc_curve = disc_curve
        self.proj_pay = proj_curve_pay
        self.proj_recv = proj_curve_recv
        self.maturity = maturity
        self.basis_spread = basis_spread
        self.reference_date = reference_date
        self.dc = dc

    def reprice(self, curves: dict[str, DiscountCurve]) -> float:
        disc = curves[self.disc_curve]
        pay = curves[self.proj_pay]
        recv = curves[self.proj_recv]
        # Simplified: PV_pay(proj_pay + spread) = PV_recv(proj_recv)
        # At par: float_pay + spread×annuity = float_recv
        # Error: (1-df_pay(T)) + spread×annuity - (1-df_recv(T))
        df_pay = pay.df(self.maturity)
        df_recv = recv.df(self.maturity)
        t = year_fraction(self.reference_date, self.maturity, self.dc)
        annuity = t * disc.df(self.maturity)  # simplified
        return (1 - df_pay) + self.basis_spread * annuity - (1 - df_recv)


# ═══════════════════════════════════════════════════════════════
# Solver
# ═══════════════════════════════════════════════════════════════


def ncurve_solve(
    reference_date: date,
    specs: list[CurveSpec],
    instruments: list[InstrumentPricer],
    tol: float = 1e-10,
    max_iter: int = 50,
    bump_size: float = 1e-6,
) -> NCurveResult:
    """Solve N curves simultaneously via damped Newton-Raphson.

    The state vector x = [df₁¹, df₂¹, ..., df₁², df₂², ...]
    contains all pillar DFs across all curves. The Jacobian is computed
    via finite differences and the system F(x) = 0 is solved.

    Args:
        reference_date: valuation date.
        specs: list of CurveSpec (one per curve).
        instruments: list of InstrumentPricer (total = sum of pillars).
        tol: convergence tolerance (max |F(x)|).
        max_iter: maximum Newton iterations.
        bump_size: finite-difference bump for Jacobian.

    Returns:
        NCurveResult with solved curves.
    """
    n_curves = len(specs)
    n_total = sum(len(s.pillar_dates) for s in specs)
    n_instr = len(instruments)

    if n_instr < n_total:
        pass  # under-determined is OK (least-squares)

    # Build initial state vector
    x = []
    offsets = {}
    idx = 0
    for spec in specs:
        offsets[spec.name] = (idx, len(spec.pillar_dates))
        if spec.initial_guess is not None:
            x.extend(spec.initial_guess)
        else:
            # Default: flat DFs from a ~3% rate
            for d in spec.pillar_dates:
                t = (d - reference_date).days / 365.0
                x.append(math.exp(-0.03 * t))
        idx += len(spec.pillar_dates)

    x = np.array(x, dtype=float)

    def build_curves(state: np.ndarray) -> dict[str, DiscountCurve]:
        curves = {}
        for spec in specs:
            start, count = offsets[spec.name]
            dfs = list(np.maximum(state[start:start + count], 1e-15))
            curves[spec.name] = DiscountCurve(
                reference_date, spec.pillar_dates, dfs,
                interpolation=spec.interpolation,
            )
        return curves

    def eval_residuals(state: np.ndarray) -> np.ndarray:
        curves = build_curves(state)
        return np.array([inst.reprice(curves) for inst in instruments])

    # Newton iterations
    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        n_iter = iteration + 1
        F = eval_residuals(x)
        max_res = float(np.max(np.abs(F)))

        if max_res < tol:
            converged = True
            break

        # Numerical Jacobian
        J = np.zeros((n_instr, n_total))
        for j in range(n_total):
            x_bump = x.copy()
            x_bump[j] += bump_size
            F_bump = eval_residuals(x_bump)
            J[:, j] = (F_bump - F) / bump_size

        # Solve: J × Δx = -F
        try:
            if n_instr == n_total:
                dx = np.linalg.solve(J, -F)
            else:
                dx, _, _, _ = np.linalg.lstsq(J, -F, rcond=None)
        except np.linalg.LinAlgError:
            dx, _, _, _ = np.linalg.lstsq(J, -F, rcond=None)

        # Damped step: ensure DFs stay positive
        step = 1.0
        for _ in range(10):
            x_new = x + step * dx
            if np.all(x_new > 0):
                F_new = eval_residuals(x_new)
                if np.max(np.abs(F_new)) < max_res * 1.1:
                    break
            step *= 0.5
        x = x + step * dx
        x = np.maximum(x, 1e-15)

    # Build final curves
    final_curves = build_curves(x)
    final_F = eval_residuals(x)

    return NCurveResult(
        curves=final_curves,
        residual=float(np.max(np.abs(final_F))),
        n_iterations=n_iter,
        converged=converged,
        n_curves=n_curves,
        n_instruments=n_instr,
        pillar_counts={s.name: len(s.pillar_dates) for s in specs},
    )
