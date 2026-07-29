"""Shared pricing atoms — the building blocks the calibrator AND the engine compose (L1).

CLAUDE.md §3d: an atom is defined ONCE; every stage composes it. `rpv01` (the fixed
annuity) and `float_leg_pv` live here, below both the L3 calibrator and the L4 engine,
so neither re-derives a discount/annuity loop — there is no second day count and no
second interpolation for them to drift on. Both take the SAME `Schedule` from the one
L2 product, closing the accrual-date-identity axis (the F1 re-review catch).

Provenance:
  quarry: python/pricebook/fixed_income/interest_rate_swap.py (annuity / swap PV)
  source: CLAUDE.md §3d (shared atoms); single-curve par-float telescoping identity
  oracle: par swap reprices to zero NPV; annuity > 0; float telescopes to df0 − dfN
  slice:  swap-to-zero-npv (T1 slice 1)
"""

from __future__ import annotations

from pricebook_ng.foundation import DayCountConvention, Schedule, year_fraction
from pricebook_ng.market.curve import CurveHandle


def rpv01(schedule: Schedule, day_count: DayCountConvention, curve: CurveHandle) -> float:
    """The fixed annuity — RPV01 = Σ τᵢ·df(payᵢ) over the schedule's periods. THIS is
    the single definition of the annuity: the calibrator's `par_rate` and the engine's
    swap PV both call it (§3d), so they cannot disagree on day count or discounting."""
    return sum(
        year_fraction(period.accrual_start, period.accrual_end, day_count)
        * curve.df(period.payment_date)
        for period in schedule.periods
    )


def float_leg_pv(schedule: Schedule, curve: CurveHandle) -> float:
    """Single-curve floating-leg PV (unit notional): the par-float telescoping identity
    `df(start) − df(end)`. Exact when projection == discount and payments carry no lag
    (the slice's single-curve world); the per-period projection form arrives with the
    projection curve at multicurve."""
    first, last = schedule.periods[0], schedule.periods[-1]
    return curve.df(first.accrual_start) - curve.df(last.payment_date)
