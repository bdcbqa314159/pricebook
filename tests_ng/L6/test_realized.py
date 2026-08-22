"""L6 oracle — the stateful shell: BookedTrade + benefit table + realized P&L (closes C3/T1).

The engine computes the MARK (future PV, historical excluded — invariant 6); the shell REMEMBERS
realized (past coupons, undiscounted). Total = realized + mark. Oracles: seasoned realized vs a
hand-computed benefit table, future-only mark (closes the silent misprice), total identity, the spot
degenerate (realized 0, total == mark), realized-undiscounted, and missing-fixing failure-as-value.
"""

from datetime import date

from pricebook_ng.engine import price
from pricebook_ng.foundation import (
    Accrual,
    DayCountConvention,
    FixingHistory,
    Frequency,
    Money,
    PricingFailure,
    PricingResult,
    RollRule,
    ScheduleTerms,
    Tenor,
    TenorUnit,
    TimeMeasure,
    accrued_rate,
    build_schedule,
    future_periods,
    get_rate_index,
)
from pricebook_ng.market.building_blocks import float_leg_pv, rpv01
from pricebook_ng.market.curve import DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.swap import FixedLeg, FloatLeg, VanillaSwap
from pricebook_ng.shell import BookedTrade, Trade, mark, realized, total

VAL = date(2026, 1, 15)
INDEX = get_rate_index("EURIBOR_3M")
CCY = INDEX.id.currency
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
TM = TimeMeasure(VAL, DC)
CAL = INDEX.accrual.roll.calendar
LAG = INDEX.fixing.fixing_lag
K = 0.03
NOTIONAL = 1_000_000.0
_TERMS = ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=None))


def _curves() -> CurveSet:
    disc = DiscountCurve.flat(TM, 0.030, until=date(2030, 1, 15))
    proj = DiscountCurve.flat(TM, 0.035, until=date(2030, 1, 15))
    return CurveSet({CurveKey(CurveRole.DISCOUNT, CCY): disc, CurveKey(CurveRole.PROJECTION, INDEX): proj})


def _swap(start: date, end: date) -> VanillaSwap:
    sched = build_schedule(start, end, _TERMS)
    return VanillaSwap(NOTIONAL, CCY, FixedLeg(sched, DC, K), FloatLeg(sched, DC, INDEX))


# a SEASONED swap: started 2024-01-15, matures 2029; past pays 2025-01-15 and 2026-01-15 (== VAL)
SEASONED = _swap(date(2024, 1, 15), date(2029, 1, 15))
_FIX = {  # EURIBOR fixings at (accrual_start − 2 business days) for the two past periods
    "EURIBOR_3M": {
        CAL.add_business_days(date(2024, 1, 15), -LAG): 0.031,
        CAL.add_business_days(date(2025, 1, 15), -LAG): 0.028,
    }
}
FIXINGS = FixingHistory(_FIX)


def _hand_realized() -> float:
    fixings = FixingHistory(_FIX)
    net = 0.0
    for p in SEASONED.float_leg.schedule.periods:
        if p.payment_date <= VAL:  # a past (already-paid) period
            accrual = Accrual(p.accrual_start, p.accrual_end, DC)
            tau = accrual.year_fraction()
            fwd = accrued_rate(INDEX, accrual, fixings)
            net += tau * fwd - tau * K  # payer: float received − fixed paid
    return NOTIONAL * net


def test_seasoned_realized_matches_benefit_table() -> None:
    booked = BookedTrade(Trade((SEASONED,), date(2024, 1, 15)), FIXINGS)
    r = realized(booked, VAL)
    assert isinstance(r, Money)
    assert abs(r.amount - _hand_realized()) < 1e-9


def test_future_only_mark_closes_the_silent_misprice() -> None:
    model = DiscountingModel(MarketSnapshot(VAL, _curves()))
    m = mark(Trade((SEASONED,), date(2024, 1, 15)), model)
    assert isinstance(m, Money)  # no raise / no df>1 garbage
    # mark == the FUTURE periods only, priced through the atoms
    curves = _curves()
    disc, proj = curves.discount(CCY), curves.projection(INDEX)
    fixed_f = future_periods(SEASONED.fixed_leg.schedule, VAL)
    float_f = future_periods(SEASONED.float_leg.schedule, VAL)
    expected = NOTIONAL * (float_leg_pv(float_f, DC, disc, proj) - K * rpv01(fixed_f, DC, disc))
    assert abs(m.amount - expected) < 1e-9


def test_total_is_realized_plus_mark() -> None:
    model = DiscountingModel(MarketSnapshot(VAL, _curves()))
    booked = BookedTrade(Trade((SEASONED,), date(2024, 1, 15)), FIXINGS)
    t = total(booked, model)
    r = realized(booked, VAL)
    m = mark(booked.trade, model)
    assert isinstance(t, Money) and isinstance(r, Money) and isinstance(m, Money)
    assert abs(t.amount - (r.amount + m.amount)) < 1e-12


def test_spot_degenerate_realized_zero_total_is_mark() -> None:
    spot = _swap(VAL, VAL + Tenor(5, Y))  # starts at valuation → no past periods
    model = DiscountingModel(MarketSnapshot(VAL, _curves()))
    booked = BookedTrade(Trade((spot,), VAL), FixingHistory({}))
    r = realized(booked, VAL)
    t = total(booked, model)
    m = mark(booked.trade, model)
    assert isinstance(r, Money) and r.amount == 0.0  # benefit table empty
    assert isinstance(t, Money) and isinstance(m, Money)
    assert abs(t.amount - m.amount) < 1e-12  # total == mark == full future PV


def test_realized_is_undiscounted() -> None:
    # the benefit table is raw notional·τ·rate — a df would shrink it; assert it equals the
    # un-discounted hand value (which applies no df) exactly
    booked = BookedTrade(Trade((SEASONED,), date(2024, 1, 15)), FIXINGS)
    r = realized(booked, VAL)
    assert isinstance(r, Money)
    assert abs(r.amount - _hand_realized()) < 1e-9  # matches the no-df computation


def test_missing_fixing_is_failure_as_value() -> None:
    booked = BookedTrade(Trade((SEASONED,), date(2024, 1, 15)), FixingHistory({}))  # no fixings
    assert isinstance(realized(booked, VAL), PricingFailure)
    model = DiscountingModel(MarketSnapshot(VAL, _curves()))
    assert isinstance(total(booked, model), PricingFailure)
