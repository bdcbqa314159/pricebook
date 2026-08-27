"""L4 audit Batch F / #3b — the engine prices a seasoned trade on a MID-PERIOD valuation date.

The current in-progress period's float coupon was fixed in the past; the engine reads it from the
snapshot's fixings (A1) and splices it into the FUTURE-PV mark, and populates `PricingResult.accrued`
(the earned-but-unpaid slice) so `clean = pv − accrued`. The splice lives in the engine — the atoms
(`float_leg_pv`/`rpv01`) are unchanged, so spot/boundary results are byte-identical (§3d).
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
    build_schedule,
    get_rate_index,
)
from pricebook_ng.market.building_blocks import float_leg_pv, rpv01
from pricebook_ng.market.curve import DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.swap import FixedLeg, FloatLeg, VanillaSwap

INDEX = get_rate_index("EURIBOR_3M")  # FLAT/IBOR — the current coupon is the single fixing
CCY = INDEX.id.currency
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
K = 0.03
N = 1_000_000.0
FAR = date(2032, 1, 15)
CAL = INDEX.accrual.roll.calendar
LAG = INDEX.fixing.fixing_lag
_TERMS = ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=None))

# a seasoned swap valued MID-PERIOD: current coupon period is 2025-01-15 → 2026-01-15
SEASONED = VanillaSwap(
    N, CCY,
    FixedLeg(build_schedule(date(2024, 1, 15), date(2029, 1, 15), _TERMS), DC, K),
    FloatLeg(build_schedule(date(2024, 1, 15), date(2029, 1, 15), _TERMS), DC, INDEX),
)
VAL = date(2025, 6, 15)  # inside the 2025-01-15 → 2026-01-15 period
CUR_START = date(2025, 1, 15)
FIXING = 0.028
_FIX_DATE = CAL.add_business_days(CUR_START, -LAG)
FIXINGS = FixingHistory({"EURIBOR_3M": {_FIX_DATE: FIXING}})
TM = TimeMeasure(VAL, DC)


def _curves() -> CurveSet:
    disc = DiscountCurve.flat(TM, 0.030, until=FAR)
    proj = DiscountCurve.flat(TM, 0.035, until=FAR)
    return CurveSet({CurveKey(CurveRole.DISCOUNT, CCY): disc, CurveKey(CurveRole.PROJECTION, INDEX): proj})


def _model(fixings=FIXINGS) -> DiscountingModel:
    return DiscountingModel(MarketSnapshot(VAL, _curves(), fixings=fixings))


def _expected_mark() -> float:
    disc = _curves().discount(CCY)
    proj = _curves().projection(INDEX)
    periods = SEASONED.float_leg.schedule.periods
    # fixed: rpv01 over all FUTURE periods (payment > VAL) — includes the current fixed coupon
    fixed_future = [p for p in periods if p.payment_date > VAL]
    annuity = sum(Accrual(p.accrual_start, p.accrual_end, DC).year_fraction() * disc.df(p.payment_date) for p in fixed_future)
    # float: the current period uses the FIXING; strictly-future use the projected forward
    float_pv = 0.0
    for p in fixed_future:
        acc = Accrual(p.accrual_start, p.accrual_end, DC)
        if p.accrual_start <= VAL:  # current in-progress period → the single IBOR fixing
            rate = FIXING
        else:  # strictly future → projected forward
            rate = (proj.df(p.accrual_start) / proj.df(p.accrual_end) - 1.0) / acc.year_fraction()
        float_pv += disc.df(p.payment_date) * acc.year_fraction() * rate
    return N * (float_pv - K * annuity)


def test_seasoned_mid_period_prices() -> None:  # repro_P — was current_period_failure
    result = price(SEASONED, _model())
    assert isinstance(result, PricingResult)
    assert abs(result.pv.amount - _expected_mark()) < 1e-6


def test_accrued_and_clean() -> None:
    result = price(SEASONED, _model())
    assert isinstance(result, PricingResult) and result.accrued is not None
    tau_elapsed = Accrual(CUR_START, VAL, DC).year_fraction()
    expected_accrued = N * tau_elapsed * (FIXING - K)  # net earned slice (float − fixed)
    assert abs(result.accrued.amount - expected_accrued) < 1e-9
    assert result.clean == result.pv - result.accrued  # dirty − accrued


def test_boundary_valuation_has_no_accrued() -> None:  # byte-identical, accrued None
    # value exactly on a coupon boundary → no in-progress period → accrued None, pv unchanged
    boundary = date(2026, 1, 15)
    tm = TimeMeasure(boundary, DC)
    curves = CurveSet({
        CurveKey(CurveRole.DISCOUNT, CCY): DiscountCurve.flat(tm, 0.03, until=FAR),
        CurveKey(CurveRole.PROJECTION, INDEX): DiscountCurve.flat(tm, 0.035, until=FAR),
    })
    result = price(SEASONED, DiscountingModel(MarketSnapshot(boundary, curves, fixings=FixingHistory({}))))
    assert isinstance(result, PricingResult) and result.accrued is None


def test_missing_current_fixing_is_failure() -> None:
    out = price(SEASONED, _model(FixingHistory({})))  # no fixing for the current period
    assert isinstance(out, PricingFailure)


def test_spot_swap_3d_identity_splice_never_fires() -> None:
    # a spot swap: no current period → the engine's mark == calibrator's atoms (par rate → zero PV)
    sched = build_schedule(VAL, VAL + Tenor(5, Y), _TERMS)
    disc, proj = _curves().discount(CCY), _curves().projection(INDEX)
    annuity = rpv01(sched, DC, disc)
    par = float_leg_pv(sched, DC, disc, proj) / annuity
    swap = VanillaSwap(N, CCY, FixedLeg(sched, DC, par), FloatLeg(sched, DC, INDEX))
    result = price(swap, DiscountingModel(MarketSnapshot(VAL, _curves(), fixings=FixingHistory({}))))
    assert isinstance(result, PricingResult)
    assert abs(result.pv.amount) < 1e-6 and result.accrued is None
