"""L4 oracle — CSA collateral-keyed discounting (slice 6b).

A swap carries its `collateral` currency; the engine resolves `discount(ccy, collateral)`
through `model.market` (A1) and records it as `PricingResult.basis`. Own-currency collateral
normalizes to the domestic OIS curve (degenerate — identical to slices 1–5); a foreign
collateral selects its keyed curve (the xccy curve lands in 6c).
"""

from dataclasses import replace
from datetime import date

from pricebook_ng.calibration.calibrate import CalibrationSpec, CurveBuild, ParSwapQuote, calibrate, par_swap
from pricebook_ng.engine import price
from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    PricingResult,
    Tenor,
    TenorUnit,
    TimeMeasure,
    get_rate_index,
)
from pricebook_ng.market.curve import DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel

VAL = date(2026, 1, 15)
ESTR = get_rate_index("ESTR")
EURIBOR_3M = get_rate_index("EURIBOR_3M")
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
DISCOUNT = CurveBuild(ESTR, Frequency.ANNUAL, DC, tuple(ParSwapQuote(Tenor(y, Y), r) for y, r in [(1, .030), (2, .032), (3, .034), (5, .036)]))
PROJECTION = CurveBuild(EURIBOR_3M, Frequency.ANNUAL, DC, tuple(ParSwapQuote(Tenor(y, Y), r) for y, r in [(1, .0312), (2, .0332), (3, .0352), (5, .0372)]))
SPEC = CalibrationSpec.single_currency(valuation_date=VAL, currency=Currency.EUR, discount=DISCOUNT, projection=PROJECTION)


def _swap(collateral):
    return replace(par_swap(SPEC, PROJECTION, Tenor(5, Y), 0.0372), collateral=collateral)


def test_own_currency_collateral_is_domestic_ois_degenerate() -> None:
    model, _ = calibrate(SPEC)
    none_r = price(_swap(None), model)
    eur_r = price(_swap(Currency.EUR), model)  # collateral == ccy → normalizes to domestic OIS
    assert isinstance(none_r, PricingResult) and isinstance(eur_r, PricingResult)
    assert abs(none_r.pv.amount - eur_r.pv.amount) < 1e-12  # identical to the domestic price
    assert none_r.basis is None and eur_r.basis is None  # own-currency → basis None


def test_foreign_collateral_uses_keyed_curve_and_records_basis() -> None:
    model, _ = calibrate(SPEC)
    # augment the CurveSet with a distinct EUR-in-USD collateral curve so discount(EUR, USD) resolves
    usd_coll = DiscountCurve.flat(TimeMeasure(VAL, DC), 0.05, until=VAL + Tenor(5, Y))
    curves = dict(model.market.curves.curves)
    curves[CurveKey(CurveRole.DISCOUNT, Currency.EUR, Currency.USD)] = usd_coll
    m2 = DiscountingModel(MarketSnapshot(VAL, CurveSet(curves)))
    usd_r = price(_swap(Currency.USD), m2)
    assert isinstance(usd_r, PricingResult)
    assert usd_r.basis == Currency.USD  # foreign collateral recorded on the result
    assert abs(usd_r.pv.amount - price(_swap(None), m2).pv.amount) > 1e-6  # different discount curve
