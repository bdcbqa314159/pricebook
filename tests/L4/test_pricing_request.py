"""T84 — R1.3: `PricingRequest` and `value` (marking a book as of one date).

**The model assignment is a lookup; the as-of semantics are the new content.** The shape was ruled at
0453 after the build-time finding that a product does **not** determine its model: five products map to
multiple models (`EuropeanEquityOption` to four), and §3 makes every model carry its own calibrated
market — so "supply a market, get a model" is calibration, not representation. The request therefore
carries an explicit `{product_type → model}` assignment: *choices*, supplied, never resolved.

**The as-of date is a DECLARED field, not derived from the models**, and the reason is the one Cowork
accepted for `Position.currency`: declaring it catches the failure deriving cannot. Derived, a book
accidentally assembled from yesterday's models values itself as-of yesterday, consistently, and reports
nothing wrong. Declared, the caller states the date they *meant* and every model is checked against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.pricing_request import value
from pricebook.engine.registry import register
from pricebook.engine.trade import trade_value
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import EquityAsset, EquityUnderlying
from pricebook.models.equity_black import EquityBlackModel
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption
from pricebook.products.position import Position
from pricebook.products.pricing_request import PricingRequest
from pricebook.products.trade import Trade, TradeDirection

TODAY = date(2026, 1, 15)
YESTERDAY = TODAY - timedelta(days=1)
USD = Currency("USD")
ACME = EquityAsset("ACME")
DC = DayCountConvention.ACT_365_FIXED
LIVE_EXPIRY = TODAY + timedelta(days=400)
SETTLED_EXPIRY = TODAY - timedelta(days=30)
SPOT = 100.0


def _market(valuation: date = TODAY) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=valuation,
        curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): DiscountCurve.flat(valuation, 0.03, DC)}),
        equities={ACME: EquityUnderlying(spot=SPOT, repo_rate=0.01)},
    )


# ============================================================================================
# The fixture population: two product types with DIFFERENT models, and two estimate pricers.
# ============================================================================================

@dataclass(frozen=True)
class _EstimateA:
    currency: Currency


@dataclass(frozen=True)
class _EstimateB:
    currency: Currency


@dataclass(frozen=True)
class _ModelA:
    market: MarketSnapshot


@dataclass(frozen=True)
class _ModelB:
    market: MarketSnapshot


PV_A, SE_A = 10.0, 0.30
PV_B, SE_B = 20.0, 0.40


@register(_EstimateA, _ModelA)
def _price_a(product: _EstimateA, model: _ModelA) -> PricingResult | PricingFailure:
    del model
    return PricingResult(pv=Money(PV_A, product.currency), standard_error=SE_A)


@register(_EstimateB, _ModelB)
def _price_b(product: _EstimateB, model: _ModelB) -> PricingResult | PricingFailure:
    del model
    return PricingResult(pv=Money(PV_B, product.currency), standard_error=SE_B)


#: **A second model for `_EstimateA`, registered deliberately**, so the one-to-many ambiguity that
#: shaped this tranche is present in the fixture rather than only in the argument: resolving
#: `_EstimateA` to `_ModelA` has to be a *choice* the assignment makes, not the only option.
@dataclass(frozen=True)
class _ModelAlt:
    market: MarketSnapshot


PV_ALT = 999.0


@register(_EstimateA, _ModelAlt)
def _price_a_alt(product: _EstimateA, model: _ModelAlt) -> PricingResult | PricingFailure:
    del model
    return PricingResult(pv=Money(PV_ALT, product.currency))


def _two_type_book(as_of: date = TODAY) -> PricingRequest:
    """Two product types, **different** assigned models — so the lookup does real work."""
    position = Position(
        currency=USD,
        trades=(
            Trade(product=_EstimateA(USD), units=2.0, direction=TradeDirection.LONG),
            Trade(product=_EstimateB(USD), units=3.0, direction=TradeDirection.LONG),
        ),
    )
    market = _market(as_of)
    return PricingRequest(
        position=position,
        assignment={_EstimateA: _ModelA(market), _EstimateB: _ModelB(market)},
        as_of=as_of,
    )


# ============================================================================================
# 1. Resolution — the lookup picks the ASSIGNED model, not merely a registered one.
# ============================================================================================

def test_the_assignment_resolves_each_trade_to_ITS_model_and_reduces_to_the_catalogue():
    request = _two_type_book()
    book = value(request)
    assert isinstance(book, PricingResult)
    assert book.pv.amount == pytest.approx(2.0 * PV_A + 3.0 * PV_B, rel=1e-15, abs=0.0)


def test_the_RESOLUTION_is_a_choice_because_the_product_admits_MORE_THAN_ONE_model():
    """The finding that shaped this tranche, made a fixture rather than left an argument.

    `_EstimateA` has **two** registered pricers (`_ModelA` and `_ModelAlt`, worth wildly different
    amounts). So resolving it is a *choice the assignment makes*, and an implementation that reached
    into the registry for "the" model for a product could not do it at all. This is the shape of the
    real tree: `EuropeanEquityOption` has four registered models, `NominalCap` four, `VarianceSwap`
    three.
    """
    position = Position(
        currency=USD,
        trades=(Trade(product=_EstimateA(USD), units=1.0, direction=TradeDirection.LONG),),
    )
    market = _market()
    chosen = value(
        PricingRequest(position=position, assignment={_EstimateA: _ModelA(market)}, as_of=TODAY)
    )
    alternative = value(
        PricingRequest(position=position, assignment={_EstimateA: _ModelAlt(market)}, as_of=TODAY)
    )
    assert isinstance(chosen, PricingResult) and isinstance(alternative, PricingResult)
    assert chosen.pv.amount == pytest.approx(PV_A, rel=1e-15, abs=0.0)
    assert alternative.pv.amount == pytest.approx(PV_ALT, rel=1e-15, abs=0.0)
    # the premise: the two models disagree enormously, so a mis-resolution is unmissable
    assert abs(PV_ALT / PV_A - 1.0) > 50.0


def test_a_single_trade_request_is_that_TRADE_exactly():
    market = _market()
    trade = Trade(product=_EstimateB(USD), units=4.0, direction=TradeDirection.SHORT)
    request = PricingRequest(
        position=Position(currency=USD, trades=(trade,)),
        assignment={_EstimateB: _ModelB(market)},
        as_of=TODAY,
    )
    through_request = value(request)
    direct = trade_value(trade, _ModelB(market))
    assert isinstance(through_request, PricingResult) and isinstance(direct, PricingResult)
    assert through_request.pv.amount == direct.pv.amount
    assert through_request.standard_error == direct.standard_error


# ============================================================================================
# 2. An unassigned product type is a REFUSAL.
# ============================================================================================

def test_an_UNASSIGNED_product_type_is_refused_not_skipped():
    """A book holding something the configuration does not cover cannot be priced.

    Skipping it would return a **number** that silently omits a position — the §3.4 reasoning that
    T83 applied at book level, one layer up again.
    """
    market = _market()
    position = Position(
        currency=USD,
        trades=(
            Trade(product=_EstimateA(USD), units=1.0, direction=TradeDirection.LONG),
            Trade(product=_EstimateB(USD), units=1.0, direction=TradeDirection.LONG),
        ),
    )
    request = PricingRequest(
        position=position, assignment={_EstimateA: _ModelA(market)}, as_of=TODAY
    )
    outcome = value(request)
    assert isinstance(outcome, PricingFailure)
    assert "_EstimateB" in outcome.reason, outcome.reason


# ============================================================================================
# 3. The AS-OF semantics — the tranche's real content, both branches (T33b).
# ============================================================================================

def test_a_model_as_of_a_DIFFERENT_date_than_the_request_is_refused():
    """You cannot mark a book half as-of today and half as-of yesterday.

    The currency-refusal shape on the **date** axis. Every assigned model carries its own
    `market.valuation_date` (§3 invariant 2), and the request states the date it *means* — so the
    check is declared-against-carried, not a consistency scan among the models.
    """
    today_market, stale_market = _market(TODAY), _market(YESTERDAY)
    position = Position(
        currency=USD,
        trades=(
            Trade(product=_EstimateA(USD), units=1.0, direction=TradeDirection.LONG),
            Trade(product=_EstimateB(USD), units=1.0, direction=TradeDirection.LONG),
        ),
    )
    request = PricingRequest(
        position=position,
        assignment={_EstimateA: _ModelA(today_market), _EstimateB: _ModelB(stale_market)},
        as_of=TODAY,
    )
    outcome = value(request)
    assert isinstance(outcome, PricingFailure)
    assert "valuation date" in outcome.reason.lower(), outcome.reason
    assert str(YESTERDAY) in outcome.reason, outcome.reason


def test_the_date_check_does_NOT_fire_when_every_model_agrees():
    """The other branch — a guard that always fired would pass the test above and break everything."""
    book = value(_two_type_book())
    assert isinstance(book, PricingResult)


def test_models_that_AGREE_WITH_EACH_OTHER_but_not_with_the_request_are_still_refused():
    """Why the date is **declared** rather than derived, asserted rather than argued.

    Derived from the models, this book is perfectly self-consistent: both models are as-of yesterday,
    so a consistency *scan* finds nothing wrong and the book values as-of yesterday while the caller
    believes it is today's mark. Declaring the date is what turns that into a refusal.
    """
    stale = _market(YESTERDAY)
    position = Position(
        currency=USD,
        trades=(
            Trade(product=_EstimateA(USD), units=1.0, direction=TradeDirection.LONG),
            Trade(product=_EstimateB(USD), units=1.0, direction=TradeDirection.LONG),
        ),
    )
    request = PricingRequest(
        position=position,
        assignment={_EstimateA: _ModelA(stale), _EstimateB: _ModelB(stale)},
        as_of=TODAY,
    )
    outcome = value(request)
    assert isinstance(outcome, PricingFailure)
    assert "valuation date" in outcome.reason.lower(), outcome.reason
    # and the same assignment IS accepted when the request says what it means
    agreed = value(
        PricingRequest(position=position, assignment=request.assignment, as_of=YESTERDAY)
    )
    assert isinstance(agreed, PricingResult)


# ============================================================================================
# 4. Invariant 6 is INHERITED through the wrapper, not re-implemented.
# ============================================================================================

def _equity_call(expiry: date, strike: float) -> EuropeanEquityOption:
    return EuropeanEquityOption(
        asset=ACME,
        strike=strike,
        expiry=expiry,
        kind=EquityOptionKind.CALL,
        notional=Money(1.0, USD),
    )


def test_a_SETTLED_trade_contributes_ZERO_through_the_request_and_a_live_one_does_not():
    """Invariant 6 survives two wrappers (`trade_value`, then `value`) — a cross-level check.

    The request adds **no** date filtering; it inherits the engines'. The settled option is
    deliberately **in the money** (strike 80 against spot 100), so PV 0 differs from its intrinsic —
    the F-4.1/F-4.2 premise, where an out-of-the-money expired option's intrinsic is also 0 and the
    guard stays invisible.
    """
    market = _market(TODAY)
    model = EquityBlackModel(market=market, volatility=0.28)
    settled_itm = _equity_call(SETTLED_EXPIRY, 80.0)
    live = _equity_call(LIVE_EXPIRY, 110.0)

    # the premise: the settled option is ITM, so a bypass would show a large positive intrinsic
    assert SPOT - 80.0 > 15.0

    position = Position(
        currency=USD,
        trades=(
            Trade(product=settled_itm, units=100.0, direction=TradeDirection.LONG),
            Trade(product=live, units=100.0, direction=TradeDirection.LONG),
        ),
    )
    request = PricingRequest(
        position=position, assignment={EuropeanEquityOption: model}, as_of=TODAY
    )
    book = value(request)
    assert isinstance(book, PricingResult)

    live_only = trade_value(position.trades[1], model)
    assert isinstance(live_only, PricingResult)
    # the settled leg contributes exactly nothing, so the book is the live leg alone
    assert book.pv.amount == pytest.approx(live_only.pv.amount, rel=1e-15, abs=0.0)
    # and the live leg is materially non-zero, so this is not two zeros compared
    assert live_only.pv.amount > 100.0


# ============================================================================================
# 5. T83's aggregation rides through unchanged.
# ============================================================================================

def test_the_LINEAR_SE_combination_rides_through_the_request():
    request = _two_type_book()
    book = value(request)
    assert isinstance(book, PricingResult)
    assert book.standard_error == pytest.approx(2.0 * SE_A + 3.0 * SE_B, rel=1e-15, abs=0.0)


def test_a_CURRENCY_mismatch_still_refuses_through_the_request():
    euro = Currency("EUR")
    market = _market()
    position = Position(
        currency=USD,
        trades=(Trade(product=_EstimateA(euro), units=1.0, direction=TradeDirection.LONG),),
    )
    outcome = value(
        PricingRequest(position=position, assignment={_EstimateA: _ModelA(market)}, as_of=TODAY)
    )
    assert isinstance(outcome, PricingFailure)
    assert "EUR" in outcome.reason


def test_an_EMPTY_book_through_the_request_is_PV_ZERO_with_no_band():
    book = value(
        PricingRequest(position=Position(currency=USD, trades=()), assignment={}, as_of=TODAY)
    )
    assert isinstance(book, PricingResult)
    assert book.pv.amount == 0.0
    assert book.standard_error is None


def test_a_request_is_FROZEN():
    from dataclasses import FrozenInstanceError

    request = _two_type_book()
    with pytest.raises(FrozenInstanceError):
        setattr(request, "as_of", YESTERDAY)
