"""AAD-aware discount and survival curves.

Number-valued pillar DFs/survivals, returning Number from df(t) and
survival(t) so that derivatives flow through pricing to curve inputs.
"""

from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING

from pricebook.calibration import curve_calibration_record, pillar_parameters
from pricebook.curves.aad import Number
from pricebook.curves.aad_interp import aad_log_linear_interp
from pricebook.core.day_count import DayCountConvention, year_fraction

if TYPE_CHECKING:
    from pricebook.calibration import CalibrationResult


class AADDiscountCurve:
    """Discount curve with Number-valued pillar DFs.

    Supports df(date) -> Number, so the adjoint graph links pricing
    output to each pillar DF.
    """

    def __init__(
        self,
        reference_date: date,
        dates: list[date],
        dfs: list[Number],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    ):
        if len(dates) != len(dfs):
            raise ValueError("dates and dfs must have the same length")
        if len(dates) < 1:
            raise ValueError("need at least 1 pillar point")

        self.reference_date = reference_date
        self.day_count = day_count

        times = [year_fraction(reference_date, d, day_count) for d in dates]

        if times[0] > 0:
            times = [0.0] + times
            dfs = [Number(1.0)] + list(dfs)

        self._times = times
        self._dfs = dfs

        # Canonical calibration provenance, attached by aad_bootstrap (None
        # until set). Mirrors DiscountCurve — the curve carries its own record.
        self.calibration_result: "CalibrationResult | None" = None

    def _time(self, d: date) -> float:
        if d <= self.reference_date:
            return 0.0
        return year_fraction(self.reference_date, d, self.day_count)

    def df(self, d: date) -> Number:
        """Discount factor at date d. Returns Number on tape."""
        t = self._time(d)
        if t <= 0:
            return Number(1.0)
        return aad_log_linear_interp(t, self._times, self._dfs)

    @property
    def pillar_dfs(self) -> list[Number]:
        return self._dfs

    @property
    def pillar_times(self) -> list[float]:
        return self._times


class AADSurvivalCurve:
    """Survival curve with Number-valued pillar survival probabilities.

    Supports survival(date) -> Number for AAD CS01 per pillar.
    """

    def __init__(
        self,
        reference_date: date,
        dates: list[date],
        survivals: list[Number],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    ):
        if len(dates) != len(survivals):
            raise ValueError("dates and survivals must have the same length")
        if len(dates) < 1:
            raise ValueError("need at least 1 pillar point")

        self.reference_date = reference_date
        self.day_count = day_count

        times = [year_fraction(reference_date, d, day_count) for d in dates]

        if times[0] > 0:
            times = [0.0] + times
            survivals = [Number(1.0)] + list(survivals)

        self._times = times
        self._survivals = survivals

    def _time(self, d: date) -> float:
        if d <= self.reference_date:
            return 0.0
        return year_fraction(self.reference_date, d, self.day_count)

    def survival(self, d: date) -> Number:
        """Survival probability at date d. Returns Number on tape."""
        t = self._time(d)
        if t <= 0:
            return Number(1.0)
        return aad_log_linear_interp(t, self._times, self._survivals)

    @property
    def pillar_survivals(self) -> list[Number]:
        return self._survivals

    @property
    def pillar_times(self) -> list[float]:
        return self._times


# ---------------------------------------------------------------------------
# AAD Bootstrap
# ---------------------------------------------------------------------------

def aad_bootstrap(
    reference_date: date,
    deposit_quotes: list[tuple[date, Number]],
    swap_quotes: list[tuple[date, Number]],
    deposit_dc: DayCountConvention = DayCountConvention.ACT_360,
    swap_dc: DayCountConvention = DayCountConvention.THIRTY_360,
) -> AADDiscountCurve:
    """Bootstrap a discount curve with AAD-aware quotes.

    Input quotes are ``Number`` objects on the active tape. The output
    curve has ``Number``-valued pillar DFs. After pricing an instrument
    on this curve, one ``propagate_to_start()`` gives sensitivities
    to every input quote.

    Args:
        deposit_quotes: list of (maturity_date, rate_as_Number).
        swap_quotes: list of (maturity_date, par_rate_as_Number),
            sorted by maturity.

    Returns:
        AADDiscountCurve with Number-valued DFs on the tape.

    Example::

        with Tape() as tape:
            dep = Number(0.05)
            swap = Number(0.04)
            curve = aad_bootstrap(ref, [(d1, dep)], [(d2, swap)])
            pv = instrument_price(curve)
            pv.propagate_to_start()
            print(dep.adjoint)   # dpv/d(dep_rate)
            print(swap.adjoint)  # dpv/d(swap_rate)
    """
    from pricebook.core.schedule import Frequency, generate_schedule

    pillar_dates: list[date] = []
    pillar_dfs: list[Number] = []

    # Phase 1: deposits — df = 1 / (1 + rate × tau)
    sorted_deps = sorted(deposit_quotes, key=lambda x: x[0])
    for mat, rate in sorted_deps:
        tau = year_fraction(reference_date, mat, deposit_dc)
        df = Number(1.0) / (Number(1.0) + rate * tau)
        pillar_dates.append(mat)
        pillar_dfs.append(df)

    # Phase 2: swaps — solve for df(mat) from par condition
    # Fix T1.13: pre-fix the intermediate-coupon DFs were drawn from a
    # `temp_curve` that lacked the new `mat` pillar.  When the new swap's
    # tenor extends beyond all existing pillars (the common case — e.g.
    # bootstrap a 5y swap against a 1y deposit), the intermediate coupons
    # at 1.5y, 2y, ..., 4.5y fell in the extrapolation region, where
    # `aad_log_linear_interp` flat-extrapolates at `df_last`.  This
    # silently flat-extrapolates the bootstrap and biases sensitivities.
    #
    # Fix: solve the par condition via a fixed-point iteration where the
    # trial curve at each step INCLUDES the current df_mat estimate as a
    # pillar.  Pre-solve numerically with `brentq` on floats for a sharp
    # initial guess, then iterate the AAD construction starting from it
    # until convergence — the AAD graph at the fixed point captures the
    # implicit-function dependency dx*/dp = −(∂A + ∂B) / (1 + ∂B/∂x).
    from pricebook.core.solvers import brentq

    sorted_swaps = sorted(swap_quotes, key=lambda x: x[0])
    for mat, par_rate in sorted_swaps:
        schedule = generate_schedule(reference_date, mat, Frequency.SEMI_ANNUAL)
        par_rate_val = par_rate.value

        # ---- Float pre-solve via brentq (sharp initial guess) ----
        pillar_dates_float = list(pillar_dates) + [mat]
        pillar_dfs_float = [d.value for d in pillar_dfs]

        def _par_residual_float(df_mat_guess: float) -> float:
            dfs_now = pillar_dfs_float + [df_mat_guess]
            curve_f = _FloatLogLinearCurve(
                reference_date, pillar_dates_float, dfs_now, swap_dc,
            )
            pv = 0.0
            for k in range(1, len(schedule)):
                tau_k = year_fraction(schedule[k - 1], schedule[k], swap_dc)
                pv += par_rate_val * tau_k * curve_f.df(schedule[k])
            return pv + df_mat_guess - 1.0

        df_mat_init = brentq(_par_residual_float, 1e-6, 3.0)

        # ---- AAD fixed-point starting from converged float value ----
        df_mat = Number(df_mat_init)
        for _ in range(50):
            trial_dfs = list(pillar_dfs) + [df_mat]
            trial_curve = AADDiscountCurve(
                reference_date, pillar_dates_float, trial_dfs, swap_dc,
            )
            annuity_before_last = Number(0.0)
            for k in range(1, len(schedule) - 1):
                tau_k = year_fraction(schedule[k - 1], schedule[k], swap_dc)
                df_k = trial_curve.df(schedule[k])
                annuity_before_last = annuity_before_last + par_rate * tau_k * df_k
            tau_last = year_fraction(schedule[-2], schedule[-1], swap_dc)
            df_mat_new = (Number(1.0) - annuity_before_last) / (Number(1.0) + par_rate * tau_last)
            if abs(df_mat_new.value - df_mat.value) < 1e-14:
                df_mat = df_mat_new
                break
            df_mat = df_mat_new

        pillar_dates.append(mat)
        pillar_dfs.append(df_mat)

    curve = AADDiscountCurve(reference_date, pillar_dates, pillar_dfs, swap_dc)

    # Provenance: per-instrument round-trip residuals on float values (the AAD
    # curve solves each par condition, so these are ~0 by construction).
    def _f(x):
        return x.value if hasattr(x, "value") else float(x)

    quotes: list[str] = []
    residuals: list[float] = []
    for mat, rate in sorted_deps:
        tau = year_fraction(reference_date, mat, deposit_dc)
        model_rate = (1.0 / curve.df(mat).value - 1.0) / tau if tau > 0 else 0.0
        quotes.append(f"deposit_{mat.isoformat()}")
        residuals.append(model_rate - _f(rate))
    for mat, par_rate in sorted_swaps:
        schedule = generate_schedule(reference_date, mat, Frequency.SEMI_ANNUAL)
        annuity = sum(
            year_fraction(schedule[k - 1], schedule[k], swap_dc) * curve.df(schedule[k]).value
            for k in range(1, len(schedule))
        )
        # par swap: par_rate * annuity + df(mat) - 1 ≈ 0
        quotes.append(f"swap_{mat.isoformat()}")
        residuals.append(_f(par_rate) * annuity + curve.df(mat).value - 1.0)
    curve.calibration_result = curve_calibration_record(
        model_class="aad_discount_curve_bootstrap",
        parameters=pillar_parameters(pillar_dates, [d.value for d in pillar_dfs], label="df"),
        residuals=residuals,
        quotes_fitted=quotes,
        algorithm="aad-fixed-point",
        iterations=len(pillar_dates),
        converged=True,
        diagnostics_extra={"n_deposits": len(sorted_deps), "n_swaps": len(sorted_swaps)},
    )
    return curve


class _FloatLogLinearCurve:
    """Plain-float log-linear DF curve — used inside `aad_bootstrap` for the
    brentq pre-solve.  Mirrors `AADDiscountCurve` interp logic but avoids
    the Number tape (we only need values for the numerical bracket-search)."""

    def __init__(self, reference_date: date, dates: list[date],
                 dfs: list[float], day_count: DayCountConvention):
        times = [year_fraction(reference_date, d, day_count) for d in dates]
        if times[0] > 0:
            times = [0.0] + times
            dfs = [1.0] + list(dfs)
        self._times = times
        self._dfs = dfs
        self._ref = reference_date
        self._dc = day_count

    def df(self, d: date) -> float:
        if d <= self._ref:
            return 1.0
        t = year_fraction(self._ref, d, self._dc)
        ts, ds = self._times, self._dfs
        if t <= ts[0]:
            return ds[0]
        if t >= ts[-1]:
            return ds[-1]
        import bisect
        i = max(0, min(bisect.bisect_right(ts, t) - 1, len(ts) - 2))
        w = (t - ts[i]) / (ts[i + 1] - ts[i])
        return math.exp((1 - w) * math.log(ds[i]) + w * math.log(ds[i + 1]))
