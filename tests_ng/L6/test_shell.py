"""L6 oracle — the imperative shell: frozen Trade/Book + marking + portfolio risk (C3, L6 opening).

The shell holds descriptions and CALLS the core: `mark` is a shell function (no `pv()` on any
product/trade/book), portfolio DV01 is Σ of the L5 greek. Oracles: trade-of-1 == engine, book
additivity, DV01 additivity, frozen/immutable, failure-as-value, and a MIXED book under one model.
"""

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from pricebook_ng.engine import price
from pricebook_ng.foundation import (
    Accrual,
    DayCountConvention,
    Frequency,
    Money,
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
from pricebook_ng.market.vol_surface import Surface, SurfaceKey
from pricebook_ng.models.black_model import BlackModel
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.option import Caplet
from pricebook_ng.products.swap import FixedLeg, FloatLeg, VanillaSwap
from pricebook_ng.risk import ir_delta, priceable
from pricebook_ng.shell import Book, Trade, book_dv01, mark, mark_book

VAL = date(2026, 1, 15)
INDEX = get_rate_index("EURIBOR_3M")
CCY = INDEX.id.currency
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
TM = TimeMeasure(VAL, DC)
_END = VAL + Tenor(6, Y)
DISC_KEY = CurveKey(CurveRole.DISCOUNT, CCY)


def _market() -> MarketSnapshot:
    disc = DiscountCurve.flat(TM, 0.030, until=_END)
    proj = DiscountCurve.flat(TM, 0.035, until=_END)
    curves = CurveSet({DISC_KEY: disc, CurveKey(CurveRole.PROJECTION, INDEX): proj})
    return MarketSnapshot(VAL, curves, surfaces={SurfaceKey(INDEX): Surface.flat(0.25)})


_SCHED = build_schedule(VAL, VAL + Tenor(5, Y), ScheduleTerms(frequency=Frequency.ANNUAL, roll=RollRule(calendar=None)))
SWAP = VanillaSwap(1.0, CCY, FixedLeg(_SCHED, DC, 0.02), FloatLeg(_SCHED, DC, INDEX))  # off-par
_EXP = VAL + Tenor(1, Y)
CAPLET = Caplet(INDEX, Accrual(_EXP, _EXP + INDEX.id.tenor, DayCountConvention.ACT_360), 0.035, 1.0)


def test_trade_of_one_equals_engine() -> None:
    model = DiscountingModel(_market())
    t = Trade((SWAP,), VAL)
    m = mark(t, model)
    engine = price(SWAP, model)
    assert isinstance(m, Money) and not isinstance(engine, PricingFailure)
    assert m == engine.pv  # thin pass-through — the shell calls the engine, adds nothing


def test_book_marks_to_sum_of_trades() -> None:
    model = BlackModel(_market())  # ONE rich model marks a MIXED book (swaps + caplets)
    t_swap, t_caplet = Trade((SWAP,), VAL), Trade((CAPLET,), VAL)
    book = Book((t_swap, t_caplet), "mixed")
    total = mark_book(book, model)
    per_trade = mark(t_swap, model), mark(t_caplet, model)
    assert isinstance(total, Money) and all(isinstance(x, Money) for x in per_trade)
    assert abs(total.amount - (per_trade[0].amount + per_trade[1].amount)) < 1e-12
    # and == the raw engine sum
    s = price(SWAP, model).pv.amount + price(CAPLET, model).pv.amount
    assert abs(total.amount - s) < 1e-12


def test_portfolio_dv01_is_sum_of_per_product() -> None:
    model = BlackModel(_market())
    book = Book((Trade((SWAP,), VAL), Trade((CAPLET,), VAL)))
    agg = book_dv01(book, model, DISC_KEY)
    per_product = sum(
        ir_delta(priceable(p, BlackModel), model.market, DISC_KEY) for p in (SWAP, CAPLET)
    )
    assert isinstance(agg, float)
    assert abs(agg - per_product) < 1e-12  # greek additivity across the shell


def test_trade_and_book_are_frozen_and_marking_does_not_mutate() -> None:
    market = _market()
    model = DiscountingModel(market)
    base_dfs = market.curves.discount(CCY).dfs
    t = Trade((SWAP,), VAL)
    _ = mark(t, model)
    assert market.curves.discount(CCY).dfs == base_dfs  # marking mutates nothing
    with pytest.raises(FrozenInstanceError):
        setattr(t, "start_date", VAL)  # frozen — mutation is refused at runtime


def test_failure_propagates_as_a_value() -> None:
    # a caplet needs BlackVol; under a DiscountingModel it can't price → mark returns the failure
    model = DiscountingModel(_market())
    out = mark(Trade((CAPLET,), VAL), model)
    assert isinstance(out, PricingFailure)
    assert isinstance(mark_book(Book((Trade((CAPLET,), VAL),)), model), PricingFailure)
