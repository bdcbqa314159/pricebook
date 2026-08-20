"""L5 oracle — generic bump-and-reprice greeks (C3 opening).

DV01 (curve bump) and vega (surface bump) flow through ONE key-blind finite-difference core, two
`Bump` strategies. Oracles: DV01 vs the analytic curve delta; vega vs closed-form Black vega;
one shared core (no key-type switch); risk never isinstances an instrument; immutability + failure
as a value.
"""

import math
from datetime import date
from pathlib import Path

from pricebook_ng.engine import price
from pricebook_ng.foundation import (
    Accrual,
    DayCountConvention,
    Frequency,
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
from pricebook_ng.market.building_blocks import forward
from pricebook_ng.market.curve import DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.market.vol_surface import Surface, SurfaceKey
from pricebook_ng.models.black import black_vega
from pricebook_ng.models.black_model import BlackModel
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.option import Caplet
from pricebook_ng.products.swap import FixedLeg, FloatLeg, VanillaSwap
from pricebook_ng.risk import CurveBump, SurfaceBump, central_diff, ir_delta, priceable, vol_vega

VAL = date(2026, 1, 15)
INDEX = get_rate_index("EURIBOR_3M")
CCY = INDEX.id.currency
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
TM = TimeMeasure(VAL, DC)
_END = VAL + Tenor(6, Y)
DISC_KEY = CurveKey(CurveRole.DISCOUNT, CCY)


def _curves() -> CurveSet:
    disc = DiscountCurve.flat(TM, 0.030, until=_END)
    proj = DiscountCurve.flat(TM, 0.035, until=_END)
    return CurveSet({DISC_KEY: disc, CurveKey(CurveRole.PROJECTION, INDEX): proj})


# an OFF-par swap (K far from par) so the discount-curve delta is meaningfully non-zero
_SCHED = build_schedule(VAL, VAL + Tenor(5, Y), ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=None)))
STRIKE = 0.02
SWAP = VanillaSwap(1.0, CCY, FixedLeg(_SCHED, DC, STRIKE), FloatLeg(_SCHED, DC, INDEX))


def test_ir_delta_matches_analytic_curve_delta() -> None:
    market = MarketSnapshot(VAL, _curves())
    p = priceable(SWAP, DiscountingModel)
    fd = ir_delta(p, market, DISC_KEY)
    assert isinstance(fd, float)
    # analytic dPV/dr under a parallel discount bump: only df(pay) moves (projection is a
    # separate key) → dPV/dr = −Σ N·τ·(fwd−K)·t·df(pay)
    disc = market.curves.discount(CCY)
    proj = market.curves.projection(INDEX)
    analytic = 0.0
    for p_ in _SCHED.periods:
        accrual = Accrual(p_.accrual_start, p_.accrual_end, DC)
        fwd = forward(proj, accrual)
        df = disc.df(p_.payment_date)
        t = TM.year_fraction(p_.payment_date)
        analytic += -1.0 * accrual.year_fraction() * (fwd - STRIKE) * t * df
    assert abs(fd - analytic) < 1e-6


def test_vol_vega_matches_black_closed_form() -> None:
    expiry = VAL + Tenor(1, Y)
    accrual = Accrual(expiry, expiry + INDEX.id.tenor, DayCountConvention.ACT_360)
    caplet = Caplet(INDEX, accrual, 0.035, 1.0)
    vol = 0.25
    market = MarketSnapshot(VAL, _curves(), surfaces={SurfaceKey(INDEX): Surface.flat(vol)})
    p = priceable(caplet, BlackModel)
    fd = vol_vega(p, market, SurfaceKey(INDEX))
    assert isinstance(fd, float)
    fwd = forward(market.curves.projection(INDEX), accrual)
    pay_df = market.curves.discount(CCY).df(accrual.end)
    t = TM.year_fraction(expiry)
    analytic = pay_df * 1.0 * accrual.year_fraction() * black_vega(fwd, 0.035, vol, t)
    assert abs(fd - analytic) < 1e-6


def test_greeks_share_one_key_blind_core() -> None:
    # both greeks are thin wrappers over the SAME central_diff; the core takes a Bump strategy
    market = MarketSnapshot(VAL, _curves(), surfaces={SurfaceKey(INDEX): Surface.flat(0.25)})
    p_swap = priceable(SWAP, DiscountingModel)
    via_core = central_diff(p_swap, market, CurveBump(DISC_KEY), 1e-4)
    via_greek = ir_delta(p_swap, market, DISC_KEY)
    assert isinstance(via_core, float) and isinstance(via_greek, float)
    assert abs(via_core - via_greek) < 1e-15  # ir_delta IS central_diff with a CurveBump


def test_central_diff_core_has_no_key_type_switch() -> None:
    # §1 no-isinstance / §3d no type-switch: the FD core must be key-blind. The only isinstance in
    # risk/ is the failure-as-value check on PricingFailure — never on a Bump/key/instrument type.
    risk_dir = Path(__file__).resolve().parents[2] / "src" / "pricebook_ng" / "risk"
    for src in risk_dir.glob("*.py"):
        for line in src.read_text(encoding="utf-8").splitlines():
            if "isinstance(" in line:  # the CALL, not the word in prose
                assert "PricingFailure" in line, f"non-failure isinstance in {src.name}: {line.strip()}"


def test_bump_is_immutable_and_failure_is_a_value() -> None:
    market = MarketSnapshot(VAL, _curves())
    base_dfs = market.curves.discount(CCY).dfs
    _ = ir_delta(priceable(SWAP, DiscountingModel), market, DISC_KEY)
    assert market.curves.discount(CCY).dfs == base_dfs  # base untouched (invariant 3)
    # a caplet through a DiscountingModel lacks BlackVol → PricingFailure, propagated as a VALUE
    expiry = VAL + Tenor(1, Y)
    caplet = Caplet(INDEX, Accrual(expiry, expiry + INDEX.id.tenor, DayCountConvention.ACT_360), 0.035, 1.0)
    out = ir_delta(priceable(caplet, DiscountingModel), market, DISC_KEY)
    assert isinstance(out, PricingFailure)
