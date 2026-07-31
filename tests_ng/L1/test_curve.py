"""L1 oracle — the discount curve, the shared atoms, and the degeneracy guard.

Flat-curve `df(t) = exp(-r·t)` is the discounting anchor. The §3d atoms are `rpv01`
(discount annuity) and, new this slice, `forward` (the single projection-rate atom)
which `float_leg_pv` composes. The degeneracy oracle: with projection == discount the
generalised float leg equals the telescoping identity `df(start₀) − df(endₙ)` to a
tight tolerance (NOT bit-identity — a df ratio is not bit-exact).
"""

import math
from datetime import date

from pricebook_ng.foundation import (
    Accrual,
    DayCountConvention,
    Frequency,
    RollRule,
    ScheduleTerms,
    Tenor,
    TenorUnit,
    TimeMeasure,
    build_schedule,
)
from pricebook_ng.market.building_blocks import deposit_df, float_leg_pv, forward, rpv01
from pricebook_ng.market.curve import DiscountCurve

VAL = date(2026, 1, 15)
TM = TimeMeasure(VAL, DayCountConvention.ACT_365_FIXED)
DC = DayCountConvention.ACT_365_FIXED
FIVE_Y = VAL + Tenor(5, TenorUnit.YEAR)
ONE_Y = VAL + Tenor(1, TenorUnit.YEAR)


def test_flat_curve_df_is_closed_form() -> None:
    r = 0.03
    curve = DiscountCurve.flat(TM, r, until=FIVE_Y)
    t = TM.year_fraction(FIVE_Y)
    assert abs(curve.df(FIVE_Y) - math.exp(-r * t)) < 1e-12
    notional = 10_000_000.0
    assert abs(notional * curve.df(FIVE_Y) - notional * math.exp(-r * t)) < 1e-6


def test_flat_curve_df_is_one_at_the_anchor() -> None:
    assert DiscountCurve.flat(TM, 0.03, until=FIVE_Y).df(VAL) == 1.0


def test_forward_rate_is_the_simple_forward() -> None:
    curve = DiscountCurve.flat(TM, 0.03, until=FIVE_Y)
    accrual = Accrual(VAL, ONE_Y, DC)  # ACT/365F ⇒ τ = 1.0 over the year
    # forward = (df(start)/df(end) − 1)/τ = (1/exp(-0.03) − 1)/1 = exp(0.03) − 1
    assert abs(forward(curve, accrual) - (math.exp(0.03) - 1.0)) < 1e-12


def test_float_leg_degenerates_to_the_telescoping_identity() -> None:
    # projection == discount ⇒ the general per-period form equals df(start₀) − df(endₙ),
    # to ~1e-12 relative (NOT ==). This is the "single == multi degenerate" guard (doc 18 §8).
    terms = ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=None))
    schedule = build_schedule(VAL, FIVE_Y, terms)
    curve = DiscountCurve.flat(TM, 0.03, until=FIVE_Y)
    general = float_leg_pv(schedule, DC, curve, curve)
    telescoping = curve.df(schedule.periods[0].accrual_start) - curve.df(
        schedule.periods[-1].payment_date
    )
    assert abs(general - telescoping) < 1e-12 * abs(telescoping)
    assert rpv01(schedule, DC, curve) > 0.0


def test_deposit_df_is_simple_compounding() -> None:
    accrual = Accrual(VAL, VAL + Tenor(6, TenorUnit.MONTH), DC)
    rate = 0.03
    assert abs(deposit_df(rate, accrual) - 1.0 / (1.0 + rate * accrual.year_fraction())) < 1e-15
