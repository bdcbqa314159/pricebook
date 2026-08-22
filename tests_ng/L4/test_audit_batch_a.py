"""L4 audit Batch A — engine invariant-4/6 hardening: #15, #16, #20, #3a.

Every ported repro must return a VALUE (PricingFailure / PV 0), never raise.
"""

from datetime import date

import pytest

from pricebook_ng.engine import price
from pricebook_ng.foundation import (
    Accrual,
    DayCountConvention,
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
from pricebook_ng.market.curve import DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.market.vol_surface import Surface, SurfaceKey, SwaptionSurfaceKey
from pricebook_ng.models.black_model import BlackModel
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.cash import FRA
from pricebook_ng.products.option import Caplet, OptionType, Swaption
from pricebook_ng.products.swap import FixedLeg, FloatLeg, VanillaSwap

VAL = date(2026, 1, 15)
INDEX = get_rate_index("EURIBOR_3M")
CCY = INDEX.id.currency
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
TM = TimeMeasure(VAL, DC)
FAR = date(2032, 1, 15)


def _curves(disc_rate: float, proj_rate: float) -> CurveSet:
    d = DiscountCurve.flat(TM, disc_rate, until=FAR)
    p = DiscountCurve.flat(TM, proj_rate, until=FAR)
    return CurveSet({CurveKey(CurveRole.DISCOUNT, CCY): d, CurveKey(CurveRole.PROJECTION, INDEX): p})


# ── #16 — ZeroDivisionError through the cash pricers ───────────────────────────────
def test_fra_zero_year_fraction_is_failure_not_zerodiv() -> None:  # repro_R
    market = MarketSnapshot(VAL, _curves(0.03, 0.035))
    # NL/365 accrual spanning only 29-Feb → year fraction 0 → forward() divides by zero
    fra = FRA(1.0, CCY, Accrual(date(2028, 2, 29), date(2028, 3, 1), DayCountConvention.NL_365), INDEX, 0.03)
    assert isinstance(price(fra, DiscountingModel(market)), PricingFailure)


# ── #15 — negative-forward options silently priced with zero time value ────────────
def _neg_market(vol: float) -> BlackModel:
    curves = _curves(-0.005, -0.005)  # the 2015–2022 EUR/JPY regime
    surfaces = {SurfaceKey(INDEX): Surface.flat(vol)}
    return BlackModel(MarketSnapshot(VAL, curves, surfaces=surfaces))


def test_negative_forward_caplet_is_failure_not_silent_zero() -> None:  # repro_Q
    exp = VAL + Tenor(1, Y)
    caplet = Caplet(INDEX, Accrual(exp, exp + INDEX.id.tenor, DayCountConvention.ACT_360), 0.01, 1.0)
    out = price(caplet, _neg_market(0.20))  # forward < 0 → lognormal Black undefined
    assert isinstance(out, PricingFailure)
    assert "forward" in out.reason.lower()


# ── #20 — expired swaption raises where the caplet returns 0 ───────────────────────
def test_expired_swaption_returns_zero_not_raw_date_error() -> None:  # repro_S
    past = date(2025, 1, 15)  # expiry before valuation
    sched = build_schedule(past + Tenor(1, Y), past + Tenor(6, Y), ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=None)))
    swap = VanillaSwap(1.0, CCY, FixedLeg(sched, DC, 0.03), FloatLeg(sched, DC, INDEX))
    surfaces = {SwaptionSurfaceKey(INDEX, Tenor(5, Y)): Surface.flat(0.2)}
    model = BlackModel(MarketSnapshot(VAL, _curves(0.03, 0.035), surfaces=surfaces))
    out = price(Swaption(swap, past, OptionType.CALL), model)
    assert isinstance(out, PricingResult)
    assert out.pv == Money(0.0, CCY)


def test_surface_validates_vols_vs_expiries() -> None:  # #20 companion
    with pytest.raises(ValueError):
        Surface((0.2, 0.25), (VAL,))  # 2 vols, 1 expiry — malformed


# ── #3a — seasoned mid-period marking fails HONESTLY (named, not a raw date error) ──
def test_seasoned_mid_period_swap_names_the_situation() -> None:  # repro_P
    sched = build_schedule(date(2024, 1, 15), date(2030, 1, 15), ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=None)))
    swap = VanillaSwap(1.0, CCY, FixedLeg(sched, DC, 0.03), FloatLeg(sched, DC, INDEX))
    # VAL = 2026-01-15 is a period boundary; value mid-period instead
    mid = date(2026, 7, 15)
    tm = TimeMeasure(mid, DC)
    curves = CurveSet({
        CurveKey(CurveRole.DISCOUNT, CCY): DiscountCurve.flat(tm, 0.03, until=FAR),
        CurveKey(CurveRole.PROJECTION, INDEX): DiscountCurve.flat(tm, 0.035, until=FAR),
    })
    out = price(swap, DiscountingModel(MarketSnapshot(mid, curves)))
    assert isinstance(out, PricingFailure)
    assert "current period" in out.reason.lower() and "fixings" in out.reason.lower()
