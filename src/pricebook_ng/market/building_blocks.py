"""Shared pricing atoms — the building blocks the calibrator AND the engine compose (L1).

CLAUDE.md §3d: an atom is defined ONCE; every stage composes it. `rpv01` (the fixed
annuity, discount only), `forward` (the projection forward rate), and `float_leg_pv`
(which COMPOSES `forward`) live here, below both the L3 calibrator and the L4 engine, so
neither re-derives a discount/annuity/forward loop. Both take the SAME `Schedule` from
the one L2 product, closing the accrual-date-identity axis.

`float_leg_pv` is dual-curve: it discounts on `discount` and projects forwards on
`projection`. Single-curve is the degenerate case — with `projection is discount` and no
pay lag the per-period sum equals the telescoping identity `df(start₀) − df(endₙ)` (to
rounding). There is no separate telescoping code path (that would be a §3d exception).

Provenance:
  quarry: python/pricebook/fixed_income/interest_rate_swap.py (annuity / swap PV / forwards)
  source: CLAUDE.md §3d (shared atoms); multicurve leg PV = Σ df_disc·τ·L_proj
  oracle: EURIBOR swap → zero NPV dual-curve; float leg degenerates to df0 − dfN
  slice:  dual-curve-euribor-estr (T1 slice 2)
"""

from __future__ import annotations

from pricebook_ng.foundation import Accrual, DayCountConvention, Schedule
from pricebook_ng.market.curve import CurveHandle


def rpv01(schedule: Schedule, day_count: DayCountConvention, curve: CurveHandle) -> float:
    """The fixed annuity — RPV01 = Σ τᵢ·df(payᵢ), discount curve only. THIS is the single
    definition of the annuity: the calibrator's `par_rate` and the engine's swap PV both
    call it (§3d), so they cannot disagree on day count or discounting."""
    return sum(
        Accrual(p.accrual_start, p.accrual_end, day_count).year_fraction() * curve.df(p.payment_date)
        for p in schedule.periods
    )


def forward(projection: CurveHandle, accrual: Accrual) -> float:
    """The simple forward rate over `accrual`, off the projection curve —
    `(df(start)/df(end) − 1) / τ`. The single definition of the projection atom; the
    float-leg calibrator and the engine both reach it through the same `projection`
    curve (resolved from the leg's index via `CurveSet.projection`)."""
    return (projection.df(accrual.start) / projection.df(accrual.end) - 1.0) / accrual.year_fraction()


def float_leg_pv(
    schedule: Schedule,
    day_count: DayCountConvention,
    discount: CurveHandle,
    projection: CurveHandle,
) -> float:
    """Floating-leg PV (unit notional): Σ df_disc(payᵢ)·τᵢ·forward(proj, accrualᵢ). Composes
    `forward` (never inlines the df ratio, §3d). Degenerate single-curve case
    (`projection is discount`, no lag) equals `df(start₀) − df(endₙ)`."""
    total = 0.0
    for p in schedule.periods:
        accrual = Accrual(p.accrual_start, p.accrual_end, day_count)
        total += discount.df(p.payment_date) * accrual.year_fraction() * forward(projection, accrual)
    return total
