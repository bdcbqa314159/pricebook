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

from pricebook_ng.foundation import (
    Accrual,
    CouponPeriod,
    DayCountConvention,
    Schedule,
    SchedulePeriod,
    icma_coupon_period,
)
from pricebook_ng.market.curve import CurveHandle

# A day count that needs convention context beyond the two dates: ICMA needs the coupon frequency,
# BUS/252 needs the calendar. Both are read FROM THE SCHEDULE'S `terms` (§3d — context travels with
# the data), so the calibrator and engine cannot pass mismatched conventions for the same schedule.


def _coupon_context(schedule: Schedule, period: SchedulePeriod, day_count: DayCountConvention, is_first: bool):
    """The `(coupon_period, calendar)` a period needs under `day_count`, from the schedule's own
    `terms`. ICMA builds a `CouponPeriod` (its consumer for `is_stub`); every other convention gets
    `None` — INERT, so a non-ICMA leg is byte-identical (and `per_year()` is never called on a
    non-integer frequency). The calendar comes from the schedule's roll for BOTH legs (one source)."""
    coupon_period: CouponPeriod | None = None
    if day_count is DayCountConvention.ACT_ACT_ICMA:
        coupon_period = icma_coupon_period(
            period.accrual_start,
            period.accrual_end,
            front_stub=period.is_stub and is_first,
            frequency=schedule.terms.frequency.per_year(),
        )
    return coupon_period, schedule.terms.roll.calendar


def rpv01(schedule: Schedule, day_count: DayCountConvention, curve: CurveHandle) -> float:
    """The fixed annuity — RPV01 = Σ τᵢ·df(payᵢ), discount curve only. THIS is the single
    definition of the annuity: the calibrator's `par_rate` and the engine's swap PV both
    call it (§3d), so they cannot disagree on day count or discounting. Each τ reads its
    convention context (ICMA frequency, BUS/252 calendar) from the schedule (§3d)."""
    total = 0.0
    for i, p in enumerate(schedule.periods):
        cp, cal = _coupon_context(schedule, p, day_count, is_first=i == 0)
        tau = Accrual(p.accrual_start, p.accrual_end, day_count).year_fraction(
            coupon_period=cp, calendar=cal
        )
        total += tau * curve.df(p.payment_date)
    return total


def deposit_df(rate: float, accrual: Accrual) -> float:
    """The simple-compounding discount factor over `accrual`: `1/(1 + rate·τ)`. The single
    definition — both the L3 `DepositInstrument.residual` and the L4 deposit pricer compose
    it (§3d), so they cannot disagree."""
    return 1.0 / (1.0 + rate * accrual.year_fraction())


def forward(
    projection: CurveHandle,
    accrual: Accrual,
    *,
    coupon_period: CouponPeriod | None = None,
    calendar=None,
) -> float:
    """The simple forward rate over `accrual`, off the projection curve —
    `(df(start)/df(end) − 1) / τ`. The single definition of the projection atom; the
    float-leg calibrator and the engine both reach it through the same `projection`
    curve. `coupon_period`/`calendar` thread the ICMA/BUS252 context into `τ` (inert for
    other conventions) — the FRA/future calibrators pass none (their accruals are simple)."""
    return (projection.df(accrual.start) / projection.df(accrual.end) - 1.0) / accrual.year_fraction(
        coupon_period=coupon_period, calendar=calendar
    )


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
    for i, p in enumerate(schedule.periods):
        cp, cal = _coupon_context(schedule, p, day_count, is_first=i == 0)
        accrual = Accrual(p.accrual_start, p.accrual_end, day_count)
        tau = accrual.year_fraction(coupon_period=cp, calendar=cal)
        total += discount.df(p.payment_date) * tau * forward(
            projection, accrual, coupon_period=cp, calendar=cal
        )
    return total
