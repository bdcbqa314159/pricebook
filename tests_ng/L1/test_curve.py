"""L1 oracle — the discount curve and the shared building blocks.

The discounting anchor: on a flat continuously-compounded curve a discount factor
is `exp(-r·t)` exactly, so a single cashflow `N` is worth `N·exp(-r·t)`. And the
§3d atoms (`rpv01`, `float_leg_pv`) compose that `df` — the float leg telescopes.
"""

import math
from datetime import date

from pricebook_ng.foundation import (
    DayCountConvention,
    Frequency,
    RollRule,
    ScheduleTerms,
    Tenor,
    TenorUnit,
    TimeMeasure,
    build_schedule,
)
from pricebook_ng.market.building_blocks import float_leg_pv, rpv01
from pricebook_ng.market.curve import DiscountCurve

VAL = date(2026, 1, 15)
TM = TimeMeasure(VAL, DayCountConvention.ACT_365_FIXED)
FIVE_Y = VAL + Tenor(5, TenorUnit.YEAR)


def test_flat_curve_df_is_closed_form() -> None:
    # single fixed cashflow N on a flat curve: PV = N·exp(-r·t), to 1e-12.
    r = 0.03
    curve = DiscountCurve.flat(TM, r, until=FIVE_Y)
    t = TM.year_fraction(FIVE_Y)
    assert abs(curve.df(FIVE_Y) - math.exp(-r * t)) < 1e-12
    notional = 10_000_000.0
    assert abs(notional * curve.df(FIVE_Y) - notional * math.exp(-r * t)) < 1e-6


def test_flat_curve_df_is_one_at_the_anchor() -> None:
    curve = DiscountCurve.flat(TM, 0.03, until=FIVE_Y)
    assert curve.df(VAL) == 1.0


def test_flat_curve_df_exact_between_pillars() -> None:
    # log-linear DF interpolation is a constant forward — exact exp(-r·t) inside the range.
    r = 0.03
    curve = DiscountCurve.flat(TM, r, until=VAL + Tenor(10, TenorUnit.YEAR))
    mid = VAL + Tenor(3, TenorUnit.YEAR)
    assert abs(curve.df(mid) - math.exp(-r * TM.year_fraction(mid))) < 1e-12


def test_building_blocks_telescope_and_annuity_positive() -> None:
    terms = ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=None))
    schedule = build_schedule(VAL, FIVE_Y, terms)
    curve = DiscountCurve.flat(TM, 0.03, until=FIVE_Y)
    # float leg (unit notional) telescopes to df(start) - df(end) = 1 - df(end)
    assert abs(float_leg_pv(schedule, curve) - (1.0 - curve.df(FIVE_Y))) < 1e-14
    assert rpv01(schedule, DayCountConvention.ACT_365_FIXED, curve) > 0.0
