"""AAD-aware discount and survival curves.

Number-valued pillar DFs/survivals, returning Number from df(t) and
survival(t) so that derivatives flow through pricing to curve inputs.
"""

from __future__ import annotations

import math
from datetime import date

from pricebook.curves.aad import Number
from pricebook.curves.aad_interp import aad_log_linear_interp
from pricebook.core.day_count import DayCountConvention, year_fraction


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
    # PV = 0: par × Σ(tau_i × df_i) = 1 - df(mat)
    # df(mat) = (1 - par × Σ_{i<n}(tau_i × df_i)) / (1 + par × tau_n)
    sorted_swaps = sorted(swap_quotes, key=lambda x: x[0])
    for mat, par_rate in sorted_swaps:
        schedule = generate_schedule(reference_date, mat, Frequency.SEMI_ANNUAL)

        # Temporary AAD curve from known pillars
        if pillar_dates:
            temp_curve = AADDiscountCurve(reference_date, pillar_dates,
                                          pillar_dfs, swap_dc)
        else:
            temp_curve = None

        # Annuity for coupon dates before the last
        annuity_before_last = Number(0.0)
        for k in range(1, len(schedule) - 1):
            tau_k = year_fraction(schedule[k - 1], schedule[k], swap_dc)
            df_k = temp_curve.df(schedule[k]) if temp_curve else Number(1.0)
            annuity_before_last = annuity_before_last + par_rate * tau_k * df_k

        # Last period
        tau_last = year_fraction(schedule[-2], schedule[-1], swap_dc)

        # Solve: df(mat) = (1 - annuity_before) / (1 + par × tau_last)
        df_mat = (Number(1.0) - annuity_before_last) / (Number(1.0) + par_rate * tau_last)

        pillar_dates.append(mat)
        pillar_dfs.append(df_mat)

    return AADDiscountCurve(reference_date, pillar_dates, pillar_dfs, swap_dc)
