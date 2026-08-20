"""L3 oracle — caplet-vol stripping (C2 slice 3), the first solved surface.

Sequential 1-D Brent strip: given ascending flat cap vols, strip each maturity's marginal caplet
vol so the caps reprice. Oracles: flat round-trip (closed-form anchor), reprice-to-quote,
reprice-through-the-L4-engine backstop (§3d shared atoms), and invariant-4 on an infeasible input.
"""

import math
from datetime import date

from pricebook_ng.calibration.calibrate import CalibrationFailure
from pricebook_ng.calibration.vol_strip import VolCalibrationSpec, strip_caplet_vols
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
from pricebook_ng.market.quotes import CapQuote
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.market.vol_surface import Surface, SurfaceKey
from pricebook_ng.models.black_model import BlackModel
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.option import Caplet

VAL = date(2026, 1, 15)
INDEX = get_rate_index("EURIBOR_3M")
CCY = INDEX.id.currency
DC = INDEX.accrual.day_count  # ACT/360 — the caplet accrual convention
Y = TenorUnit.YEAR
TM365 = TimeMeasure(VAL, DayCountConvention.ACT_365_FIXED)
STRIKE = 0.035
MATS = tuple(VAL + Tenor(y, Y) for y in (1, 2, 3, 5))
_END = VAL + Tenor(6, Y)


def _curves() -> CurveSet:
    disc = DiscountCurve.flat(TimeMeasure(VAL, DC), 0.030, until=_END)
    proj = DiscountCurve.flat(TimeMeasure(VAL, DC), 0.035, until=_END)
    return CurveSet({CurveKey(CurveRole.DISCOUNT, CCY): disc, CurveKey(CurveRole.PROJECTION, INDEX): proj})


def _spec(vols) -> VolCalibrationSpec:
    curves = _curves()
    model = DiscountingModel(MarketSnapshot(VAL, curves))
    quotes = tuple(CapQuote(m, STRIKE, v) for m, v in zip(MATS, vols))
    return VolCalibrationSpec(model, INDEX, quotes)


def test_flat_roundtrip_is_exact() -> None:
    # closed-form anchor: strip a flat cap → every stripped caplet vol == the flat vol
    flat = 0.22
    out = strip_caplet_vols(_spec((flat,) * len(MATS)))
    assert not isinstance(out, CalibrationFailure), out
    surface, result = out
    assert result.converged
    for v in surface.vols:
        assert abs(v - flat) < 1e-12


def test_backstop_caplet_reprices_through_engine() -> None:
    # §3d: a caplet priced through the L4 engine off the stripped surface matches an independent
    # Black on the same forward/df — proves the strip and engine share black/forward/df.
    out = strip_caplet_vols(_spec((0.24, 0.22, 0.20, 0.19)))
    assert not isinstance(out, CalibrationFailure), out
    surface, _ = out
    curves = _curves()
    snap = MarketSnapshot(VAL, curves, surfaces={SurfaceKey(INDEX): surface})
    for mat, vol in zip(MATS, surface.vols):
        accrual = Accrual(mat, mat + INDEX.id.tenor, DC)
        result = price(Caplet(INDEX, accrual, STRIKE, 1.0), BlackModel(snap))
        assert isinstance(result, PricingResult)
        fwd = forward(curves.projection(INDEX), accrual)
        pay_df = curves.discount(CCY).df(accrual.end)
        t = TM365.year_fraction(mat)
        s = vol * math.sqrt(t)
        d1 = (math.log(fwd / STRIKE) + 0.5 * s * s) / s
        d2 = d1 - s
        phi = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        expected = pay_df * accrual.year_fraction() * (fwd * phi(d1) - STRIKE * phi(d2))
        assert abs(result.pv.amount - expected) < 1e-12


def test_stripped_surface_reprices_quotes() -> None:
    out = strip_caplet_vols(_spec((0.24, 0.22, 0.20, 0.19)))
    assert not isinstance(out, CalibrationFailure), out
    _, result = out
    assert result.converged
    assert max(abs(r) for r in result.residuals) < 1e-9


def test_infeasible_input_returns_failure() -> None:
    # a later cap quoted far BELOW the earlier ones → the marginal caplet PV is negative, no
    # positive vol reproduces it → CalibrationFailure (invariant 4), not a garbage vol.
    out = strip_caplet_vols(_spec((0.60, 0.01, 0.01, 0.01)))
    assert isinstance(out, CalibrationFailure)
