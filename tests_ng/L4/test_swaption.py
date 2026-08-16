"""L4 oracle — Black European swaption on the annuity numeraire (C2 slice 2).

The clean §3d payoff: numeraire = the annuity `rpv01`, underlying = the forward swap rate
`S = float_leg_pv/rpv01` — the SAME shared atoms the swap calibrator/engine compose. `black()` is
reused verbatim; only the vol's meaning changes (a sibling `SwaptionVol` capability).
"""

import math
from datetime import date

from pricebook_ng.engine import price
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
from pricebook_ng.market.building_blocks import float_leg_pv, rpv01
from pricebook_ng.market.curve import DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.market.vol_surface import Surface, SwaptionSurfaceKey
from pricebook_ng.models.black_model import BlackModel
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.option import OptionType, Swaption
from pricebook_ng.products.swap import FixedLeg, FloatLeg, VanillaSwap

VAL = date(2026, 1, 15)
INDEX = get_rate_index("EURIBOR_3M")
CCY = INDEX.id.currency
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
TM = TimeMeasure(VAL, DC)
EXPIRY = VAL + Tenor(1, Y)  # 1Y into 5Y
TENOR = Tenor(5, Y)
STRIKE = 0.035
# unit notional: PV is exactly linear in notional, so this tests the identical computation while
# keeping quantities O(1) — the ratified <1e-12 absolute tolerance is then a genuine tight check,
# not machine-epsilon-relative at 1e6 scale (§7c: tolerances sized to the transcendentals).
NOTIONAL = 1.0
_SCHED = build_schedule(EXPIRY, EXPIRY + TENOR, ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=None)))
_SWAP = VanillaSwap(NOTIONAL, CCY, FixedLeg(_SCHED, DC, STRIKE), FloatLeg(_SCHED, DC, INDEX))


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _ref_black(f: float, k: float, vol: float, t: float, call: bool) -> float:
    if vol <= 0 or t <= 0:
        return max(f - k, 0.0) if call else max(k - f, 0.0)
    s = vol * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * s * s) / s
    d2 = d1 - s
    return (f * _phi(d1) - k * _phi(d2)) if call else (k * _phi(-d2) - f * _phi(-d1))


def _snapshot(vol: float) -> MarketSnapshot:
    disc = DiscountCurve.flat(TM, 0.030, until=EXPIRY + TENOR)
    proj = DiscountCurve.flat(TM, 0.035, until=EXPIRY + TENOR)
    curves = CurveSet(
        {CurveKey(CurveRole.DISCOUNT, CCY): disc, CurveKey(CurveRole.PROJECTION, INDEX): proj}
    )
    return MarketSnapshot(VAL, curves, surfaces={SwaptionSurfaceKey(INDEX, TENOR): Surface(vol)})


def _S_annuity(snap: MarketSnapshot) -> tuple[float, float]:
    disc = snap.curves.discount(CCY)
    proj = snap.curves.projection(INDEX)
    annuity = rpv01(_SCHED, DC, disc)
    s = float_leg_pv(_SCHED, DC, disc, proj) / annuity
    return s, annuity


def test_swaption_reprices_to_black_annuity_closed_form() -> None:
    snap = _snapshot(0.25)
    s, annuity = _S_annuity(snap)
    t = TM.year_fraction(EXPIRY)
    expected = NOTIONAL * annuity * _ref_black(s, STRIKE, 0.25, t, call=True)
    result = price(Swaption(_SWAP, EXPIRY, OptionType.CALL), BlackModel(snap))
    assert isinstance(result, PricingResult)
    assert result.pv.currency == CCY
    assert abs(result.pv.amount - expected) < 1e-12


def test_swaption_payer_receiver_parity() -> None:
    snap = _snapshot(0.25)
    s, annuity = _S_annuity(snap)
    payer = price(Swaption(_SWAP, EXPIRY, OptionType.CALL), BlackModel(snap))
    receiver = price(Swaption(_SWAP, EXPIRY, OptionType.PUT), BlackModel(snap))
    assert isinstance(payer, PricingResult) and isinstance(receiver, PricingResult)
    assert abs((payer.pv.amount - receiver.pv.amount) - NOTIONAL * annuity * (s - STRIKE)) < 1e-12


def test_swaption_zero_vol_is_annuity_intrinsic() -> None:
    snap = _snapshot(0.0)
    s, annuity = _S_annuity(snap)
    result = price(Swaption(_SWAP, EXPIRY, OptionType.CALL), BlackModel(snap))
    assert isinstance(result, PricingResult)
    assert abs(result.pv.amount - NOTIONAL * annuity * max(s - STRIKE, 0.0)) < 1e-12


def test_swaption_parity_matches_swap_engine_3d_identity() -> None:
    # §3d: the swaption engine's S/annuity MUST match the swap engine's composition, so
    # payer − receiver (swaption engine) == the underlying swap's forward value (swap engine).
    snap = _snapshot(0.25)
    payer = price(Swaption(_SWAP, EXPIRY, OptionType.CALL), BlackModel(snap))
    receiver = price(Swaption(_SWAP, EXPIRY, OptionType.PUT), BlackModel(snap))
    swap_pv = price(_SWAP, DiscountingModel(snap))
    assert isinstance(swap_pv, PricingResult)
    assert abs((payer.pv.amount - receiver.pv.amount) - swap_pv.pv.amount) < 1e-9
