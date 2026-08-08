"""L4 oracle — CSA collateral-keyed discounting (slice 6b, extended by 6c).

A swap carries its `collateral` currency; the engine resolves `discount(ccy, collateral)`
through `model.market` (A1) and records it as `PricingResult.basis`. Own-currency collateral
normalizes to the domestic OIS curve (degenerate — identical to slices 1–5); a foreign collateral
selects the xccy-basis curve — now a REAL calibrated curve (6c), no longer a manual injection.
"""

from dataclasses import replace
from datetime import date

from pricebook_ng.calibration.calibrate import (
    CalibrationSpec,
    CurrencyCurves,
    CurveBuild,
    ParSwapQuote,
    XccyBasisQuote,
    XccyBuild,
    calibrate,
    par_swap,
)
from pricebook_ng.engine import price
from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    PricingResult,
    Tenor,
    TenorUnit,
    fx_pair,
    get_rate_index,
)
from pricebook_ng.market.snapshot import ScalarKey

VAL = date(2026, 1, 15)
SOFR = get_rate_index("SOFR")
ESTR = get_rate_index("ESTR")
EURIBOR_3M = get_rate_index("EURIBOR_3M")
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
A = Frequency.ANNUAL


def _q(pairs):
    return tuple(ParSwapQuote(Tenor(t, Y), r) for t, r in pairs)


DISCOUNT = CurveBuild(ESTR, A, DC, _q([(1, .030), (2, .032), (3, .034), (5, .036)]))
PROJECTION = CurveBuild(EURIBOR_3M, A, DC, _q([(1, .0312), (2, .0332), (3, .0352), (5, .0372)]))
SPEC = CalibrationSpec.single_currency(valuation_date=VAL, currency=Currency.EUR, discount=DISCOUNT, projection=PROJECTION)

# a EUR trade collateralised in USD → a real multi-currency + xccy calibration (6c)
USD = CurrencyCurves(
    Currency.USD,
    CurveBuild(SOFR, A, DC, _q([(1, .040), (2, .041), (3, .042), (5, .043)])),
    CurveBuild(SOFR, A, DC, _q([(1, .040), (2, .041), (3, .042), (5, .043)])),
)
EUR = CurrencyCurves(Currency.EUR, DISCOUNT, PROJECTION)
XCCY_SPEC = CalibrationSpec(
    VAL,
    (USD, EUR),
    xccy=XccyBuild(Currency.EUR, Currency.USD, tuple(XccyBasisQuote(Tenor(t, Y), 0.0015) for t in (1, 2, 3, 5))),
    fx={ScalarKey(fx_pair(Currency.EUR, Currency.USD)): 1.08},
)


def _swap(collateral):
    return replace(par_swap(SPEC, PROJECTION, Tenor(5, Y), 0.0372), collateral=collateral)


def test_own_currency_collateral_is_domestic_ois_degenerate() -> None:
    model, _ = calibrate(SPEC)
    none_r = price(_swap(None), model)
    eur_r = price(_swap(Currency.EUR), model)  # collateral == ccy → normalizes to domestic OIS
    assert isinstance(none_r, PricingResult) and isinstance(eur_r, PricingResult)
    assert abs(none_r.pv.amount - eur_r.pv.amount) < 1e-12  # identical to the domestic price
    assert none_r.basis is None and eur_r.basis is None  # own-currency → basis None


def test_foreign_collateral_uses_calibrated_xccy_curve_and_records_basis() -> None:
    model, _ = calibrate(XCCY_SPEC)  # real calibration, not a manual dict injection
    usd_r = price(_swap(Currency.USD), model)
    dom_r = price(_swap(None), model)
    assert isinstance(usd_r, PricingResult) and isinstance(dom_r, PricingResult)
    assert usd_r.basis == Currency.USD  # foreign collateral recorded on the result
    assert abs(usd_r.pv.amount - dom_r.pv.amount) > 1e-6  # discounted on the xccy curve, not domestic
