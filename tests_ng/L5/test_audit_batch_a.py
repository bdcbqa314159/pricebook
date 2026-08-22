"""L5 audit Batch A — #8: ir_delta/vol_vega on an absent key returns a value (invariant 4).

Ports repro_O: a greek keyed to market data the snapshot doesn't carry used to raise KeyError.
"""

from datetime import date

from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    PricingFailure,
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
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.swap import FixedLeg, FloatLeg, VanillaSwap
from pricebook_ng.risk import ir_delta, priceable

VAL = date(2026, 1, 15)
INDEX = get_rate_index("EURIBOR_3M")
EUR = INDEX.id.currency
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)
_SCHED = build_schedule(VAL, VAL + Tenor(5, TenorUnit.YEAR), ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=None)))
SWAP = VanillaSwap(1.0, EUR, FixedLeg(_SCHED, DC, 0.02), FloatLeg(_SCHED, DC, INDEX))


def _eur_market() -> MarketSnapshot:
    d = DiscountCurve.flat(TM, 0.03, until=VAL + Tenor(6, TenorUnit.YEAR))
    p = DiscountCurve.flat(TM, 0.035, until=VAL + Tenor(6, TenorUnit.YEAR))
    return MarketSnapshot(VAL, CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): d, CurveKey(CurveRole.PROJECTION, INDEX): p}))


def test_ir_delta_absent_key_returns_failure_not_keyerror() -> None:  # #8, repro_O
    market = _eur_market()
    usd_key = CurveKey(CurveRole.DISCOUNT, Currency.USD)  # not in the EUR-only snapshot
    out = ir_delta(priceable(SWAP, DiscountingModel), market, usd_key)
    assert isinstance(out, PricingFailure)
