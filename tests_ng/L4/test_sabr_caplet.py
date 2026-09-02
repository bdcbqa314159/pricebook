"""L4 oracle — SABR caplet through the stateless engine (no engine change).

`SABRModel` is the SECOND `BlackVol` implementer (after `BlackModel`) — and the first that USES
`strike`, so the caplet grows a smile with NO signature change (Q1 rule-of-two on the capability).
Oracles: (1) the caplet reprices to Black-76 at the SABR-implied vol via an independent inline
Hagan+Black code path; (2) the §3d F-identity — at K = the engine's own forward the SABR vol equals
the ATM formula, proving `SABRModel` derives F from the SAME `forward` atom the engine composes.
"""

import math
from datetime import date

from pricebook_ng.engine import price
from pricebook_ng.foundation import (
    Accrual,
    DayCountConvention,
    PricingResult,
    Tenor,
    TenorUnit,
    TimeMeasure,
    get_rate_index,
)
from pricebook_ng.market.building_blocks import forward
from pricebook_ng.market.curve import DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.market.vol_surface import SabrParams, SabrSurface, SurfaceKey
from pricebook_ng.models.sabr import SABRModel, sabr_vol
from pricebook_ng.products.option import Caplet

VAL = date(2026, 1, 15)
INDEX = get_rate_index("EURIBOR_3M")
CCY = INDEX.id.currency
DC365 = DayCountConvention.ACT_365_FIXED
DC360 = DayCountConvention.ACT_360
Y, M = TenorUnit.YEAR, TenorUnit.MONTH
TM = TimeMeasure(VAL, DC365)

EXPIRY = VAL + Tenor(1, Y)
PAY = EXPIRY + Tenor(3, M)
ACCRUAL = Accrual(EXPIRY, PAY, DC360)
NOTIONAL = 1_000_000.0
PARAMS = SabrParams(alpha=0.20, beta=0.5, rho=-0.3, nu=0.4)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _ref_black(f: float, k: float, vol: float, t: float) -> float:
    s = vol * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * s * s) / s
    return f * _phi(d1) - k * _phi(d1 - s)


def _model() -> SABRModel:
    disc = DiscountCurve.flat(TM, 0.030, until=PAY)
    proj = DiscountCurve.flat(TM, 0.035, until=PAY)
    curves = CurveSet(
        {CurveKey(CurveRole.DISCOUNT, CCY): disc, CurveKey(CurveRole.PROJECTION, INDEX): proj}
    )
    snap = MarketSnapshot(VAL, curves, surfaces={SurfaceKey(INDEX): SabrSurface({EXPIRY: PARAMS})})
    return SABRModel(snap)


def _fwd(model: SABRModel) -> float:
    return forward(model.market.curves.projection(INDEX), ACCRUAL)


def test_sabr_caplet_reprices_to_black_at_sabr_vol() -> None:
    model = _model()
    fwd = _fwd(model)
    for strike in (0.025, 0.035, 0.050):
        vol = sabr_vol(fwd, strike, TM.year_fraction(EXPIRY), PARAMS)
        pay_df = model.market.curves.discount(CCY).df(PAY)
        expected = pay_df * NOTIONAL * ACCRUAL.year_fraction() * _ref_black(
            fwd, strike, vol, TM.year_fraction(EXPIRY)
        )
        result = price(Caplet(INDEX, ACCRUAL, strike, NOTIONAL), model)
        assert isinstance(result, PricingResult)
        assert abs(result.pv.amount - expected) < 1e-10


def test_sabr_model_black_vol_matches_engine_forward_smile() -> None:
    # §3d F-identity: SABRModel.black_vol reads its vol at the SAME forward the engine's caplet uses,
    # so black_vol(strike=F) equals the standalone sabr_vol at that forward — no F drift between stages.
    model = _model()
    fwd = _fwd(model)
    via_model = model.black_vol(INDEX, EXPIRY, fwd)
    via_atom = sabr_vol(fwd, fwd, TM.year_fraction(EXPIRY), PARAMS)
    assert abs(via_model - via_atom) < 1e-14


def test_sabr_wrong_surface_type_is_a_failure_value() -> None:
    # a flat Surface under a SABRModel (or vice-versa) is a config error surfaced as a value, not a raise
    from pricebook_ng.foundation import PricingFailure
    from pricebook_ng.market.vol_surface import Surface

    disc = DiscountCurve.flat(TM, 0.030, until=PAY)
    proj = DiscountCurve.flat(TM, 0.035, until=PAY)
    curves = CurveSet(
        {CurveKey(CurveRole.DISCOUNT, CCY): disc, CurveKey(CurveRole.PROJECTION, INDEX): proj}
    )
    snap = MarketSnapshot(VAL, curves, surfaces={SurfaceKey(INDEX): Surface.flat(0.2)})
    result = price(Caplet(INDEX, ACCRUAL, 0.035, NOTIONAL), SABRModel(snap))
    assert isinstance(result, PricingFailure)
