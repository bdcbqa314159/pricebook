"""L4 oracle — ACT/ACT ICMA + BUS/252 leg pricing (audit #7).

The atoms read convention context from the self-describing Schedule (its `terms`): ICMA gets the
coupon frequency, BUS/252 gets the calendar. Both strict conventions used to RAISE (repro C); now
they price. `is_stub` finally has a consumer (the ICMA CouponPeriod). Non-ICMA/BUS252 legs are inert.
"""

from datetime import date

from pricebook_ng.engine import price
from pricebook_ng.foundation import (
    DayCountConvention,
    Frequency,
    PricingResult,
    RollRule,
    ScheduleTerms,
    TimeMeasure,
    business_days_between,
    build_schedule,
    get_calendar,
    get_rate_index,
)
from pricebook_ng.market.building_blocks import float_leg_pv, rpv01
from pricebook_ng.market.curve import DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.swap import FixedLeg, FloatLeg, VanillaSwap

VAL = date(2026, 1, 15)
END = date(2029, 1, 15)
INDEX = get_rate_index("EURIBOR_3M")
CCY = INDEX.id.currency
TM = TimeMeasure(VAL, DayCountConvention.ACT_365_FIXED)
ICMA = DayCountConvention.ACT_ACT_ICMA
BUS252 = DayCountConvention.BUS_252
SP = get_calendar("SAO_PAULO")

ICMA_SCHED = build_schedule(VAL, END, ScheduleTerms(frequency=Frequency.SEMI_ANNUAL, roll=RollRule(calendar=None)))
CDI_SCHED = build_schedule(VAL, END, ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=SP)))


def _disc(rate: float) -> DiscountCurve:
    return DiscountCurve.flat(TM, rate, until=END)


def test_icma_rpv01_regular_semi_is_half_per_period() -> None:
    curve = _disc(0.03)
    got = rpv01(ICMA_SCHED, ICMA, curve)  # each regular semi coupon → ICMA year fraction 0.5 exactly
    expected = sum(0.5 * curve.df(p.payment_date) for p in ICMA_SCHED.periods)
    assert abs(got - expected) < 1e-12


def test_bus252_rpv01_matches_business_day_count() -> None:
    curve = _disc(0.10)
    got = rpv01(CDI_SCHED, BUS252, curve)  # τ = business_days/252 on the schedule's calendar
    expected = sum(
        business_days_between(p.accrual_start, p.accrual_end, SP) / 252.0 * curve.df(p.payment_date)
        for p in CDI_SCHED.periods
    )
    assert abs(got - expected) < 1e-12


def test_icma_and_bus252_swaps_price_without_raising() -> None:
    # repro C: both strict-convention swaps used to raise; now they price
    disc, proj = _disc(0.03), _disc(0.035)
    market = MarketSnapshot(VAL, CurveSet({CurveKey(CurveRole.DISCOUNT, CCY): disc, CurveKey(CurveRole.PROJECTION, INDEX): proj}))
    model = DiscountingModel(market)
    icma_swap = VanillaSwap(1.0, CCY, FixedLeg(ICMA_SCHED, ICMA, 0.03), FloatLeg(ICMA_SCHED, ICMA, INDEX))
    cdi_swap = VanillaSwap(1.0, CCY, FixedLeg(CDI_SCHED, BUS252, 0.10), FloatLeg(CDI_SCHED, BUS252, INDEX))
    assert isinstance(price(icma_swap, model), PricingResult)
    assert isinstance(price(cdi_swap, model), PricingResult)


def test_icma_3d_identity_calibrator_equals_engine() -> None:
    # §3d: the par rate from the atoms (calibrator's composition) makes the engine price to zero —
    # both compose the SAME rpv01/float_leg_pv with the SAME schedule-carried ICMA context
    disc, proj = _disc(0.03), _disc(0.035)
    annuity = rpv01(ICMA_SCHED, ICMA, disc)
    floating = float_leg_pv(ICMA_SCHED, ICMA, disc, proj)
    par = floating / annuity
    swap = VanillaSwap(1.0, CCY, FixedLeg(ICMA_SCHED, ICMA, par), FloatLeg(ICMA_SCHED, ICMA, INDEX))
    market = MarketSnapshot(VAL, CurveSet({CurveKey(CurveRole.DISCOUNT, CCY): disc, CurveKey(CurveRole.PROJECTION, INDEX): proj}))
    result = price(swap, DiscountingModel(market))
    assert isinstance(result, PricingResult)
    assert abs(result.pv.amount) < 1e-12


def test_non_icma_leg_is_byte_identical() -> None:
    # an ACT/360 leg is inert to the new context path — same value as a direct year-fraction sum
    curve = _disc(0.03)
    act360 = DayCountConvention.ACT_360
    sched = build_schedule(VAL, END, ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=None)))
    got = rpv01(sched, act360, curve)
    expected = sum(
        (p.accrual_end - p.accrual_start).days / 360.0 * curve.df(p.payment_date) for p in sched.periods
    )
    assert got == expected  # exact — no context perturbation for a non-ICMA/BUS252 leg
