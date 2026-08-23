"""L5 audit #1 (Batch B) — CurveBump by curve IDENTITY (repro_A / repro_N).

For an OIS swap the discount curve IS the projection curve (one object under two keys). The old
single-key bump shifted one role and not the other — an impossible market state, and the DV01 came
out ~750× wrong with the wrong sign. `ir_delta` now bumps by IDENTITY (every aliased entry moves at
once); the single-key partial is kept, explicitly named, as a basis delta.
"""

from datetime import date

from pricebook_ng.foundation import (
    DayCountConvention,
    Frequency,
    RollRule,
    ScheduleTerms,
    Tenor,
    TenorUnit,
    TimeMeasure,
    build_schedule,
    get_rate_index,
)
from pricebook_ng.engine import price
from pricebook_ng.market.curve import DiscountCurve
from pricebook_ng.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.swap import FixedLeg, FloatLeg, VanillaSwap
from pricebook_ng.risk import CurveBasisBump, CurveIdentityBump, ir_basis_delta, ir_delta, priceable
from pricebook_ng.shell import Book, Trade, book_dv01

VAL = date(2026, 1, 15)
ESTR = get_rate_index("ESTR")  # an OIS index: its projection curve IS the discount curve
EUR = ESTR.id.currency
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
TM = TimeMeasure(VAL, DC)
FAR = VAL + Tenor(6, Y)
DISC_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
PROJ_KEY = CurveKey(CurveRole.PROJECTION, ESTR)
_SCHED = build_schedule(VAL, VAL + Tenor(5, Y), ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=None)))
OIS_SWAP = VanillaSwap(1_000_000.0, EUR, FixedLeg(_SCHED, DC, 0.02), FloatLeg(_SCHED, DC, ESTR))
SHIFT = 1e-4


def _ois_market(rate: float = 0.03) -> MarketSnapshot:
    curve = DiscountCurve.flat(TM, rate, until=FAR)  # ONE object shared by both keys (the OIS alias)
    return MarketSnapshot(VAL, CurveSet({DISC_KEY: curve, PROJ_KEY: curve}))


def _true_parallel_dv01() -> float:
    # independent ground truth: reprice on flat curves shifted ± a bp (both roles move together)
    def pv(rate: float) -> float:
        return price(OIS_SWAP, DiscountingModel(_ois_market(rate))).pv.amount
    return (pv(0.03 + SHIFT) - pv(0.03 - SHIFT)) / (2.0 * SHIFT)


def test_identity_delta_is_the_true_parallel() -> None:  # repro_A
    got = ir_delta(priceable(OIS_SWAP, DiscountingModel), _ois_market(), DISC_KEY)
    assert isinstance(got, float)
    assert abs(got - _true_parallel_dv01()) < 1e-3  # FD-second-order noise on a 1e6 notional


def test_sum_of_basis_deltas_equals_the_identity_delta() -> None:  # hardening: partials sum to parallel
    market = _ois_market()
    p = priceable(OIS_SWAP, DiscountingModel)
    identity = ir_delta(p, market, DISC_KEY)
    disc_basis = ir_basis_delta(p, market, DISC_KEY)
    proj_basis = ir_basis_delta(p, market, PROJ_KEY)
    assert abs((disc_basis + proj_basis) - identity) < 1e-3


def test_float_leg_participates_identity_differs_from_fixed_only_basis() -> None:  # oracle #2
    market = _ois_market()
    p = priceable(OIS_SWAP, DiscountingModel)
    # the discount-only basis delta is the fixed-leg-shaped answer the OLD oracle checked; the identity
    # delta additionally moves the float leg's (aliased) projection — they must differ materially
    assert abs(ir_delta(p, market, DISC_KEY) - ir_basis_delta(p, market, DISC_KEY)) > 1.0


def test_identity_bump_preserves_the_alias() -> None:  # oracle #5
    market = _ois_market()
    bumped = CurveIdentityBump(DISC_KEY).apply(market, SHIFT)
    assert bumped.curves.curves[DISC_KEY] is bumped.curves.curves[PROJ_KEY]  # still one object


def test_book_dv01_correct_for_ois_trade() -> None:  # repro_N
    market = _ois_market()
    agg = book_dv01(Book((Trade((OIS_SWAP,), VAL),)), DiscountingModel(market), DISC_KEY)
    assert isinstance(agg, float)
    assert abs(agg - _true_parallel_dv01()) < 1e-3


def test_basis_bump_replaces_only_one_key() -> None:  # oracle #6: the named partial still works
    market = _ois_market()
    bumped = CurveBasisBump(DISC_KEY).apply(market, SHIFT)
    assert bumped.curves.curves[DISC_KEY] is not bumped.curves.curves[PROJ_KEY]  # only DISC moved
