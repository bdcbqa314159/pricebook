"""L4 audit Batch C — #5: exact underlying swaption tenor (no whole-year rounding). repro_E.

The swaption vol key must carry the EXACT underlying tenor (6M ≠ 1Y ≠ 18M), from the schedule's
period count × its frequency — not `round(days/365)`.
"""

from datetime import date

from pricebook_ng.engine import price
from pricebook_ng.engine.vanilla_option import underlying_tenor
from pricebook_ng.foundation import (
    DayCountConvention,
    Frequency,
    PricingResult,
    RollRule,
    ScheduleTerms,
    Tenor,
    TenorUnit,
    TimeMeasure,
    build_schedule,
    get_rate_index,
)
from pricebook_ng.market.curve import DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.market.vol_surface import Surface, SwaptionSurfaceKey
from pricebook_ng.models.black_model import BlackModel
from pricebook_ng.products.option import OptionType, Swaption
from pricebook_ng.products.swap import FixedLeg, FloatLeg, VanillaSwap

VAL = date(2026, 1, 15)
INDEX = get_rate_index("EURIBOR_3M")
CCY = INDEX.id.currency
DC = DayCountConvention.ACT_365_FIXED
M, Y = TenorUnit.MONTH, TenorUnit.YEAR
TM = TimeMeasure(VAL, DC)
FAR = VAL + Tenor(10, Y)
EXPIRY = VAL + Tenor(1, Y)


def _swap(months: int) -> VanillaSwap:
    sched = build_schedule(EXPIRY, EXPIRY + Tenor(months, M), ScheduleTerms(frequency=Frequency.SEMI_ANNUAL, roll=RollRule(calendar=None)))
    return VanillaSwap(1.0, CCY, FixedLeg(sched, DC, 0.03), FloatLeg(sched, DC, INDEX))


def test_underlying_tenor_is_exact() -> None:  # repro_E
    assert underlying_tenor(_swap(6)) == Tenor(6, M)    # was Tenor(0, YEAR) → raise
    assert underlying_tenor(_swap(18)) == Tenor(18, M)  # was 1Y (silently wrong vol)
    assert underlying_tenor(_swap(60)) == Tenor(5, Y)   # whole-year normalizes to YEAR (unchanged)


def test_18m_swaption_prices_on_an_18m_keyed_surface() -> None:
    swap = _swap(18)
    curves = CurveSet({
        CurveKey(CurveRole.DISCOUNT, CCY): DiscountCurve.flat(TM, 0.03, until=FAR),
        CurveKey(CurveRole.PROJECTION, INDEX): DiscountCurve.flat(TM, 0.035, until=FAR),
    })
    surfaces = {SwaptionSurfaceKey(INDEX, Tenor(18, M)): Surface.flat(0.2)}
    model = BlackModel(MarketSnapshot(VAL, curves, surfaces=surfaces))
    assert isinstance(price(Swaption(swap, EXPIRY, OptionType.CALL), model), PricingResult)
