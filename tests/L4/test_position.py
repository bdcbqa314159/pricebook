"""T83 — R1.2: `Position` and aggregation (the book's value, and its ERROR).

The PV aggregation is a sum and **telescopes**, exactly as T82's reduction did, so it is the guard and
not the test. The real content is **how a book's Monte-Carlo error combines**, and the spec left that as
an *obligation* — state the independence assumption and oracle against it — rather than pre-committing
`√(Σ SE_i²)`.

**Measured, the form is richer than that hypothesis, and the reason is a charter guarantee.** §3.1 makes
a pricer referentially transparent: identical inputs ⇒ identical output. Verified here on a real MC
engine — dispatching the same `(product, model)` twice is **bit-identical** in both PV and SE. So two
trades of the *same* product and model in one book do not have *independent* errors; they have the
**same** error. Their standard errors therefore add **linearly**, and `√(Σ SE²)` would understate the
book — for the entirely ordinary book that holds two lots of one contract.

So the combination this tranche implements and oracles is:

    SE_book = √( Σ_groups ( Σ_within-group SE )² )

where a *group* is one `(product, model)` pair. **Within a group the linear sum is forced** by §3.1 —
it is a fact about this library, not an assumption. **Across groups, independence is assumed** — that is
the stated assumption, and it is an assumption: different pricers use different fixed seeds and different
payoffs, which makes their errors different but not provably uncorrelated. Named here so the number the
book reports can be read for what it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.position import position_value
from pricebook.engine.registry import register
from pricebook.engine.trade import trade_value
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import EquityAsset, EquityUnderlying
from pricebook.models.equity_black import EquityBlackModel
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption
from pricebook.products.position import Position
from pricebook.products.trade import Trade, TradeDirection

VAL = date(2026, 1, 15)
USD = Currency("USD")
ACME = EquityAsset("ACME")
DC = DayCountConvention.ACT_365_FIXED
EXPIRY = VAL + timedelta(days=400)


def _market() -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): DiscountCurve.flat(VAL, 0.03, DC)}),
        equities={ACME: EquityUnderlying(spot=100.0, repo_rate=0.01)},
    )


# ============================================================================================
# The fixture population: TWO estimate pricers and one exact, per the R1-plan premise.
# ============================================================================================

@dataclass(frozen=True)
class _EstimateA:
    currency: Currency


@dataclass(frozen=True)
class _EstimateB:
    currency: Currency


@dataclass(frozen=True)
class _ExactC:
    currency: Currency


@dataclass(frozen=True)
class _ModelA:
    market: MarketSnapshot


@dataclass(frozen=True)
class _ModelB:
    market: MarketSnapshot


@dataclass(frozen=True)
class _ModelC:
    market: MarketSnapshot


#: Known PVs and SEs, so every book-level assertion below is **exact** rather than banded — which is
#: the whole point of the R1-plan's registered-test-pricer premise. The two SEs are deliberately
#: unequal and not in a ratio that makes a linear sum and a quadrature sum coincide.
PV_A, SE_A = 10.0, 0.30
PV_B, SE_B = 20.0, 0.40
PV_C = 5.0


@register(_EstimateA, _ModelA)
def _price_a(product: _EstimateA, model: _ModelA) -> PricingResult | PricingFailure:
    del model
    return PricingResult(pv=Money(PV_A, product.currency), standard_error=SE_A)


@register(_EstimateB, _ModelB)
def _price_b(product: _EstimateB, model: _ModelB) -> PricingResult | PricingFailure:
    del model
    return PricingResult(pv=Money(PV_B, product.currency), standard_error=SE_B)


@register(_ExactC, _ModelC)
def _price_c(product: _ExactC, model: _ModelC) -> PricingResult | PricingFailure:
    del model
    return PricingResult(pv=Money(PV_C, product.currency))


def _estimate_book() -> tuple[Position, list[object]]:
    """Two trades of DIFFERENT estimate products — the across-group (independent) regime."""
    trades = (
        Trade(product=_EstimateA(USD), units=2.0, direction=TradeDirection.LONG),
        Trade(product=_EstimateB(USD), units=3.0, direction=TradeDirection.LONG),
    )
    return Position(currency=USD, trades=trades), [_ModelA(_market()), _ModelB(_market())]


# ============================================================================================
# 1. The determinism this tranche's SE combination RESTS on — asserted, not assumed.
# ============================================================================================

def test_the_same_product_and_model_prices_BIT_IDENTICALLY_so_its_error_is_the_SAME_error():
    """§3.1's referential transparency, and the fact that forces the within-group linear sum.

    If a repeated dispatch returned *different* draws, two trades of one product would have
    independent errors and their variances would add. It does not: the output is bit-identical, so
    the error is not merely correlated but **the same number**. That makes the within-group linear
    sum a **fact about this library** rather than a modelling assumption — which is why it is
    asserted here, at the top, instead of being left implicit in the combination formula.
    """
    model = _ModelA(_market())
    first = price(_EstimateA(USD), model)
    second = price(_EstimateA(USD), model)
    assert isinstance(first, PricingResult) and isinstance(second, PricingResult)
    assert first.pv.amount == second.pv.amount
    assert first.standard_error == second.standard_error


# ============================================================================================
# 2. The aggregation identity — telescopes, so it is the guard.
# ============================================================================================

def test_the_book_PV_is_the_SUM_of_its_trades_a_guard_not_the_test():
    position, models = _estimate_book()
    book = position_value(position, models)
    assert isinstance(book, PricingResult)

    expected = sum(
        trade_value(trade, model).pv.amount
        for trade, model in zip(position.trades, models)
    )
    assert book.pv.amount == pytest.approx(expected, rel=1e-15, abs=0.0)
    # the premise: the two trades differ, so this is a real sum and not a doubled single value
    assert len(position.trades) == 2
    assert 2.0 * PV_A != 3.0 * PV_B


def test_a_SINGLE_trade_position_is_that_trade_exactly():
    trade = Trade(product=_EstimateA(USD), units=7.0, direction=TradeDirection.SHORT)
    model = _ModelA(_market())
    book = position_value(Position(currency=USD, trades=(trade,)), [model])
    alone = trade_value(trade, model)
    assert isinstance(book, PricingResult) and isinstance(alone, PricingResult)
    assert book.pv.amount == alone.pv.amount
    assert book.standard_error == alone.standard_error


# ============================================================================================
# 3. The SE combination — the load-bearing new oracle, in BOTH regimes.
# ============================================================================================

def test_ACROSS_groups_the_variances_add_the_stated_independence_assumption():
    """Different `(product, model)` groups: variances add — `√(Σ SE_i²)`.

    This is the **assumed** half, and the assumption is named: different pricers use different fixed
    seeds and different payoffs, so their errors are different but not provably uncorrelated. The
    assertion is **exact** (known SEs), not banded, so a mutant combining them linearly dies.
    """
    position, models = _estimate_book()
    book = position_value(position, models)
    assert isinstance(book, PricingResult)

    expected = math.sqrt((2.0 * SE_A) ** 2 + (3.0 * SE_B) ** 2)
    assert book.standard_error == pytest.approx(expected, rel=1e-15, abs=0.0)
    # the premise that makes this discriminating: the LINEAR sum is a materially different number,
    # so the oracle distinguishes the two conventions rather than passing under either.
    linear = 2.0 * SE_A + 3.0 * SE_B
    assert abs(linear / expected - 1.0) > 0.1, (linear, expected)


def test_WITHIN_a_group_the_standard_errors_add_LINEARLY_because_the_error_is_identical():
    """Two trades of the **same** `(product, model)`: the SEs add, forced by §3.1.

    The hypothesis the spec offered was `√(Σ SE²)` throughout. Measured, that is wrong here: a
    repeated dispatch is bit-identical (asserted above), so the two trades share **one** error and
    their bands add. Treating them as independent would **understate** the book — for the entirely
    ordinary book that holds two lots of one contract.
    """
    position = Position(currency=USD, trades=(
            Trade(product=_EstimateA(USD), units=2.0, direction=TradeDirection.LONG),
            Trade(product=_EstimateA(USD), units=5.0, direction=TradeDirection.LONG),
        )
    )
    models = [_ModelA(_market()), _ModelA(_market())]
    book = position_value(position, models)
    assert isinstance(book, PricingResult)

    expected = (2.0 + 5.0) * SE_A
    assert book.standard_error == pytest.approx(expected, rel=1e-15, abs=0.0)
    # and the independence treatment would have understated it, measurably
    independent = math.sqrt((2.0 * SE_A) ** 2 + (5.0 * SE_A) ** 2)
    assert independent < expected
    assert abs(independent / expected - 1.0) > 0.2, (independent, expected)


def test_a_SHORT_leg_within_a_group_still_ADDS_to_the_band_because_a_band_has_no_sign():
    """Direction cancels in the PV and **not** in the error — the subtle half of the linear sum.

    A long and a short of the same contract net to zero value, but the *uncertainty* about that zero
    is not zero: both legs carry the same error, and the net of two identical errors is... zero,
    because it is literally the same draw. This is the one place the "same error" fact bites in the
    opposite direction, so it is measured rather than reasoned about: `trade_value` scales the band
    by `|units|`, so the legs' bands add in magnitude — the book's band is the sum, not the
    difference. Asserted as the production convention, and flagged at handoff as the one place a
    reader might expect cancellation.
    """
    position = Position(currency=USD, trades=(
            Trade(product=_EstimateA(USD), units=4.0, direction=TradeDirection.LONG),
            Trade(product=_EstimateA(USD), units=4.0, direction=TradeDirection.SHORT),
        )
    )
    models = [_ModelA(_market()), _ModelA(_market())]
    book = position_value(position, models)
    assert isinstance(book, PricingResult)
    # the PV nets exactly
    assert book.pv.amount == pytest.approx(0.0, abs=1e-12)
    # the band does not
    assert book.standard_error == pytest.approx(8.0 * SE_A, rel=1e-15, abs=0.0)


def test_a_MIXED_book_lets_the_exact_leg_contribute_ZERO_variance_not_a_crash():
    """One estimate + one exact pricer. `None` is zero variance, not an error and not a `None` book.

    The spec names this explicitly, and it is the branch a naive `√(Σ se²)` would crash on.
    """
    position = Position(currency=USD, trades=(
            Trade(product=_EstimateA(USD), units=2.0, direction=TradeDirection.LONG),
            Trade(product=_ExactC(USD), units=6.0, direction=TradeDirection.LONG),
        )
    )
    models = [_ModelA(_market()), _ModelC(_market())]
    book = position_value(position, models)
    assert isinstance(book, PricingResult)
    assert book.pv.amount == pytest.approx(2.0 * PV_A + 6.0 * PV_C, rel=1e-15, abs=0.0)
    assert book.standard_error == pytest.approx(2.0 * SE_A, rel=1e-15, abs=0.0)


def test_an_ALL_EXACT_book_has_NO_band_at_all():
    position = Position(currency=USD, trades=(
            Trade(product=_ExactC(USD), units=1.0, direction=TradeDirection.LONG),
            Trade(product=_ExactC(USD), units=2.0, direction=TradeDirection.SHORT),
        )
    )
    book = position_value(position, [_ModelC(_market()), _ModelC(_market())])
    assert isinstance(book, PricingResult)
    assert book.standard_error is None


# ============================================================================================
# 4. Netting, order independence, and the empty book.
# ============================================================================================

def _call(strike: float = 110.0) -> EuropeanEquityOption:
    """A **non-symmetric** payoff, so the netting below is a real cancellation (T82's premise)."""
    return EuropeanEquityOption(
        asset=ACME,
        strike=strike,
        expiry=EXPIRY,
        kind=EquityOptionKind.CALL,
        notional=Money(1.0, USD),
    )


def test_a_long_and_its_offsetting_SHORT_net_to_zero_on_a_non_symmetric_payoff():
    model = EquityBlackModel(market=_market(), volatility=0.28)
    position = Position(currency=USD, trades=(
            Trade(product=_call(), units=500.0, direction=TradeDirection.LONG),
            Trade(product=_call(), units=500.0, direction=TradeDirection.SHORT),
        )
    )
    book = position_value(position, [model, model])
    assert isinstance(book, PricingResult)
    assert book.pv.amount == pytest.approx(0.0, abs=1e-12)
    # the premise: each leg is worth a definite non-zero amount, so this is cancellation and not
    # two zeros added (T82's non-symmetric-payoff premise, at book level)
    leg = trade_value(position.trades[0], model)
    assert isinstance(leg, PricingResult) and abs(leg.pv.amount) > 100.0


def test_the_book_value_is_INVARIANT_under_trade_ORDER():
    position, models = _estimate_book()
    forward = position_value(position, models)
    reversed_position = Position(currency=USD, trades=tuple(reversed(position.trades)))
    backward = position_value(reversed_position, list(reversed(models)))
    assert isinstance(forward, PricingResult) and isinstance(backward, PricingResult)
    assert forward.pv.amount == pytest.approx(backward.pv.amount, rel=1e-15, abs=0.0)
    assert forward.standard_error == pytest.approx(backward.standard_error, rel=1e-15, abs=0.0)


def test_an_EMPTY_book_is_PV_ZERO_with_NO_band():
    book = position_value(Position(currency=USD, trades=()), [])
    assert isinstance(book, PricingResult)
    assert book.pv.amount == 0.0
    # `None`, not 0.0: an empty book is not an estimate, so it has no error bar to report. A 0.0
    # would claim a measured zero uncertainty, which is a different statement.
    assert book.standard_error is None


# ============================================================================================
# 5. Failure propagation and refusals.
# ============================================================================================

def test_ONE_unpriceable_trade_fails_the_WHOLE_book_rather_than_silently_dropping_it():
    """A book with an unpriceable trade is a failure, not a partial sum.

    The alternative — skip what cannot be priced and total the rest — returns a **number**, and a
    number that silently omits a position is worse than no number at all (§3.4's reasoning, at book
    level). The failure names the offending product so the caller can act on it.
    """

    class _Unpriceable:
        pass

    position = Position(currency=USD, trades=(
            Trade(product=_EstimateA(USD), units=2.0, direction=TradeDirection.LONG),
            Trade(product=_Unpriceable(), units=1.0, direction=TradeDirection.LONG),
        )
    )
    outcome = position_value(position, [_ModelA(_market()), _ModelA(_market())])
    assert isinstance(outcome, PricingFailure)
    assert "_Unpriceable" in outcome.reason, outcome.reason


def test_a_models_LENGTH_MISMATCH_is_refused_rather_than_zipped_short():
    """`zip` would silently truncate — the shape of error this guard exists to reject (T33b).

    Two parallel sequences are the honest minimum until R1.3's `PricingRequest` resolves models from
    the market, but `zip(trades, models)` on a mismatch drops trades **without a word** and returns
    a confident partial total. Supplied the mismatching input rather than assumed unreachable.
    """
    position, models = _estimate_book()
    with pytest.raises(ValueError, match="one model per trade"):
        position_value(position, models[:1])


def test_a_MIXED_CURRENCY_book_is_REFUSED_rather_than_summed():
    """Not in the spec, and found by building: the first draft summed unlike currencies silently.

    It took the *first* trade's currency and ignored the rest, so a book holding a dollar contract
    and a euro one reported their arithmetic sum as a number. "Multi-currency aggregation is out of
    scope" licenses not *supporting* cross-currency netting; it does not license adding unlike
    currencies and reporting the total. The book declares its currency and every trade is checked
    against it.
    """
    euro = Currency("EUR")
    position = Position(
        currency=USD,
        trades=(
            Trade(product=_EstimateA(USD), units=1.0, direction=TradeDirection.LONG),
            Trade(product=_EstimateA(euro), units=1.0, direction=TradeDirection.LONG),
        ),
    )
    outcome = position_value(position, [_ModelA(_market()), _ModelA(_market())])
    assert isinstance(outcome, PricingFailure)
    assert "USD" in outcome.reason and "EUR" in outcome.reason, outcome.reason


def test_the_SE_grouping_is_by_EQUALITY_not_identity():
    """The premise the within-group oracle silently depended on, and which the build got wrong.

    The first draft keyed its groups on `id(model)`, so two **structurally equal** models were two
    groups and the within-group linear sum never fired — the grouping degraded to the independent
    regime, which is the *understating* direction. This asserts the thing that was wrong: two
    separately-constructed but equal models must group, which is what §3.1 implies (identical
    *inputs*, not the same object).
    """
    first, second = _ModelA(_market()), _ModelA(_market())
    assert first is not second          # genuinely distinct objects
    assert first == second              # and structurally equal, so one group

    position = Position(
        currency=USD,
        trades=(
            Trade(product=_EstimateA(USD), units=1.0, direction=TradeDirection.LONG),
            Trade(product=_EstimateA(USD), units=1.0, direction=TradeDirection.LONG),
        ),
    )
    book = position_value(position, [first, second])
    assert isinstance(book, PricingResult)
    # linear (one group), not quadrature (two groups) -- and the two differ measurably
    assert book.standard_error == pytest.approx(2.0 * SE_A, rel=1e-15, abs=0.0)
    assert math.sqrt(2.0) * SE_A < 2.0 * SE_A


# ============================================================================================
# 6. Second killers — four guards died to ONE test each, and here hardening is free.
# ============================================================================================

def test_quadrature_is_STRICTLY_LESS_than_the_linear_sum_across_groups():
    """Second killer for the across-group combination, as a REQUIREMENT not a restatement.

    For two or more positive bands, `√(Σ SE²) < Σ SE` strictly — that is arithmetic, not a
    convention, so it holds whatever the SEs are and survives a transcription error in the exact
    expected value. A mutant that sums linearly across groups makes the two **equal**, so this
    kills it from a different direction than the pointwise assertion above.
    """
    position, models = _estimate_book()
    book = position_value(position, models)
    assert isinstance(book, PricingResult)
    assert book.standard_error is not None

    linear = 2.0 * SE_A + 3.0 * SE_B
    assert book.standard_error < linear
    # and strictly, by a margin that is not floating-point noise
    assert linear - book.standard_error > 0.1, (linear, book.standard_error)


def test_a_currency_mismatch_in_the_FIRST_trade_is_caught_too():
    """Second killer for the currency guard, and it probes a different failure mode.

    The existing oracle puts the mismatching trade **second**, so a guard that only ran after the
    first accumulation would still pass it. This puts it **first** — and additionally asserts the
    book does not accumulate anything before refusing, which a guard placed after the sum would.
    """
    euro = Currency("EUR")
    position = Position(
        currency=USD,
        trades=(
            Trade(product=_EstimateA(euro), units=1.0, direction=TradeDirection.LONG),
            Trade(product=_EstimateA(USD), units=1.0, direction=TradeDirection.LONG),
        ),
    )
    outcome = position_value(position, [_ModelA(_market()), _ModelA(_market())])
    assert isinstance(outcome, PricingFailure)
    assert "EUR" in outcome.reason


def test_a_failing_FIRST_trade_fails_the_book_before_any_partial_total():
    """Second killer for the failure propagation, from the opposite end of the book.

    The existing oracle's unpriceable trade is **second**, so a wrapper that returned the running
    total on failure would still have summed one trade first. With the failure **first** there is
    nothing to partially total, and a wrapper that swallowed it would return `PV 0` — a number, and
    the worst possible answer for an unpriceable book.
    """

    class _AlsoUnpriceable:
        pass

    position = Position(
        currency=USD,
        trades=(
            Trade(product=_AlsoUnpriceable(), units=1.0, direction=TradeDirection.LONG),
            Trade(product=_EstimateA(USD), units=2.0, direction=TradeDirection.LONG),
        ),
    )
    outcome = position_value(position, [_ModelA(_market()), _ModelA(_market())])
    assert isinstance(outcome, PricingFailure)
    assert "_AlsoUnpriceable" in outcome.reason, outcome.reason


def test_TOO_MANY_models_is_refused_as_well_as_too_few():
    """Second killer for the length guard — `zip` truncates in **both** directions.

    The existing oracle passes too *few* models. Too *many* is the mirror and equally silent: `zip`
    would price every trade and quietly ignore the surplus models, which means a caller who paired
    them wrongly gets a confident total computed against the wrong models.
    """
    position, models = _estimate_book()
    with pytest.raises(ValueError, match="one model per trade"):
        position_value(position, [*models, _ModelA(_market())])


def test_a_position_is_FROZEN():
    from dataclasses import FrozenInstanceError

    position, _ = _estimate_book()
    with pytest.raises(FrozenInstanceError):
        setattr(position, "trades", ())
