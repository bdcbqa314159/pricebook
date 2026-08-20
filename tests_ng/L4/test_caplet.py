"""L4 oracle — Black-76 European caplet (C2 slice 1).

Closed-form top-tier oracle: the engine caplet reprices to an INDEPENDENT Black-76 evaluation
(inline `erf`-based formula, a different code path from the production `norm_cdf`/scipy adapter),
plus put-call parity and the vol→0 degenerate intrinsic.
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
from pricebook_ng.market.vol_surface import Surface, SurfaceKey
from pricebook_ng.models.black_model import BlackModel
from pricebook_ng.products.option import Caplet

VAL = date(2026, 1, 15)
INDEX = get_rate_index("EURIBOR_3M")
CCY = INDEX.id.currency
DC365 = DayCountConvention.ACT_365_FIXED
DC360 = DayCountConvention.ACT_360
Y = TenorUnit.YEAR
M = TenorUnit.MONTH
TM = TimeMeasure(VAL, DC365)

EXPIRY = VAL + Tenor(1, Y)  # fixing
PAY = EXPIRY + Tenor(3, M)  # payment
ACCRUAL = Accrual(EXPIRY, PAY, DC360)
STRIKE = 0.035
NOTIONAL = 1_000_000.0


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _ref_black(f: float, k: float, vol: float, t: float, call: bool = True) -> float:
    """Independent undiscounted Black-76 (inline erf) — the oracle's reference code path."""
    if vol <= 0 or t <= 0:
        return max(f - k, 0.0) if call else max(k - f, 0.0)
    s = vol * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * s * s) / s
    d2 = d1 - s
    if call:
        return f * _phi(d1) - k * _phi(d2)
    return k * _phi(-d2) - f * _phi(-d1)


def _market(vol: float) -> BlackModel:
    disc = DiscountCurve.flat(TM, 0.030, until=PAY)
    proj = DiscountCurve.flat(TM, 0.035, until=PAY)
    curves = CurveSet(
        {CurveKey(CurveRole.DISCOUNT, CCY): disc, CurveKey(CurveRole.PROJECTION, INDEX): proj}
    )
    snap = MarketSnapshot(VAL, curves, surfaces={SurfaceKey(INDEX): Surface.flat(vol)})
    return BlackModel(snap)


def _inputs(model: BlackModel) -> tuple[float, float, float, float]:
    curves = model.market.curves
    fwd = forward(curves.projection(INDEX), ACCRUAL)
    pay_df = curves.discount(CCY).df(PAY)
    tau = ACCRUAL.year_fraction()
    t = TM.year_fraction(EXPIRY)
    return fwd, pay_df, tau, t


def test_caplet_reprices_to_black_closed_form() -> None:
    model = _market(0.20)
    fwd, pay_df, tau, t = _inputs(model)
    expected = pay_df * NOTIONAL * tau * _ref_black(fwd, STRIKE, 0.20, t, call=True)
    result = price(Caplet(INDEX, ACCRUAL, STRIKE, NOTIONAL), model)
    assert isinstance(result, PricingResult)
    assert result.pv.currency == CCY
    assert abs(result.pv.amount - expected) < 1e-12


def test_caplet_floorlet_put_call_parity() -> None:
    model = _market(0.20)
    fwd, pay_df, tau, t = _inputs(model)
    caplet = price(Caplet(INDEX, ACCRUAL, STRIKE, NOTIONAL), model)
    assert isinstance(caplet, PricingResult)
    floorlet = pay_df * NOTIONAL * tau * _ref_black(fwd, STRIKE, 0.20, t, call=False)
    parity = pay_df * NOTIONAL * tau * (fwd - STRIKE)
    assert abs((caplet.pv.amount - floorlet) - parity) < 1e-12


def test_caplet_zero_vol_is_discounted_intrinsic() -> None:
    model = _market(0.0)
    fwd, pay_df, tau, _ = _inputs(model)
    result = price(Caplet(INDEX, ACCRUAL, STRIKE, NOTIONAL), model)
    assert isinstance(result, PricingResult)
    intrinsic = pay_df * NOTIONAL * tau * max(fwd - STRIKE, 0.0)
    assert abs(result.pv.amount - intrinsic) < 1e-12
