"""T82 — R1.1: the `Trade` type and `trade_value` (the structural track's first representation unit).

**Architecture-first, so the oracle is the catalogue.** There is no held source here: a trade's value is
*defined* to reduce to `price(product, model)`, and the 46 verified engines are the reference. That makes
the headline identity nearly a tautology — `trade_value` calls `price` and multiplies — which is §4's
**self-consistency-telescopes** shape in a new coordinate. So the reduction is the *foundation* and not
the test; what does the testing is the set of companions, each reported with its **measured separation**:

* the long/short asymmetry, on a payoff that is **not** symmetric about zero (else a dropped sign is
  invisible);
* the notional's **absolute** level against an independently composed reference, because `2N → 2V`
  pins proportionality and is blind to a uniform mis-scaling (T43's trap);
* the currency guard, supplied the input it exists to reject (T33b).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.registry import register
from pricebook.engine.trade import trade_value
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import EquityAsset, EquityUnderlying
from pricebook.models.equity_black import EquityBlackModel
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption
from pricebook.products.trade import Trade, TradeDirection

VAL = date(2026, 1, 15)
USD = Currency("USD")
EUR = Currency("EUR")
ACME = EquityAsset("ACME")
DC = DayCountConvention.ACT_365_FIXED
EXPIRY = VAL + timedelta(days=400)

#: A **non-round, non-unit** notional (the spec's degeneracy premise): not 1, which would collapse the
#: scaling oracle into the reduction oracle, and not a power of ten, which can hide a factor.
NOTIONAL = 37_500.0


def _market() -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): DiscountCurve.flat(VAL, 0.03, DC)}),
        equities={ACME: EquityUnderlying(spot=100.0, repo_rate=0.01)},
    )


def _call(strike: float = 110.0) -> EuropeanEquityOption:
    """**Deliberately out-of-the-money and a CALL** — a payoff that is not symmetric about zero.

    A forward struck at the money is worth ~0, so `−long == long` holds for it whatever the sign code
    does; the asymmetry oracle would pass on a dropped sign. An OTM call is worth a definite positive
    amount, so flipping the direction is a measurable change.
    """
    return EuropeanEquityOption(
        asset=ACME,
        strike=strike,
        expiry=EXPIRY,
        kind=EquityOptionKind.CALL,
        notional=Money(1.0, USD),
    )


def _model() -> EquityBlackModel:
    return EquityBlackModel(market=_market(), volatility=0.28)


def _trade(
    direction: TradeDirection = TradeDirection.LONG,
    notional: float = NOTIONAL,
    strike: float = 110.0,
) -> Trade:
    return Trade(product=_call(strike), notional=Money(notional, USD), direction=direction)


# ============================================================================================
# 1. The reduction to the catalogue — the foundation, and stated as near-tautological.
# ============================================================================================

def test_a_LONG_UNIT_trade_is_BIT_IDENTICAL_to_the_bare_price():
    """`trade_value(long, 1 unit) == price(product, model)`, exactly.

    **Bit-identical, not approximately**: a long unit trade multiplies by `+1.0 × 1.0`, which is
    exact in binary, so any difference would mean the wrapper reordered or re-derived something
    rather than delegating. Recorded as the *foundation* of the oracle set and explicitly **not** its
    strength — it is satisfied by construction the moment `trade_value` calls `price` and multiplies,
    so it cannot see a wrong direction sign or a notional applied to the wrong side. Those are the
    companions' job.
    """
    model = _model()
    bare = price(_call(), model)
    assert isinstance(bare, PricingResult)

    wrapped = trade_value(_trade(notional=1.0), model)
    assert isinstance(wrapped, PricingResult)
    assert wrapped.pv.amount == bare.pv.amount
    assert wrapped.pv.currency == bare.pv.currency


def test_the_reduction_is_SATISFIED_BY_CONSTRUCTION_which_is_why_it_needs_companions():
    """The premise behind the docstring above, asserted rather than asserted-in-prose.

    A `trade_value` that ignored the direction entirely would still pass the reduction oracle: a long
    unit trade has `sign = +1`, so a wrapper that never consults the sign is indistinguishable there.
    This demonstrates it — the same call with the direction *flipped* must differ, and the reduction
    oracle above says nothing about that.
    """
    model = _model()
    long_unit = trade_value(_trade(notional=1.0), model)
    short_unit = trade_value(_trade(TradeDirection.SHORT, notional=1.0), model)
    assert isinstance(long_unit, PricingResult) and isinstance(short_unit, PricingResult)
    # the reduction oracle constrains the first and is blind to the second
    assert long_unit.pv.amount != short_unit.pv.amount


# ============================================================================================
# 2. The long/short asymmetry — on a payoff where the sign does real work.
# ============================================================================================

def test_a_SHORT_trade_is_the_NEGATION_of_the_long_one():
    model = _model()
    long_value = trade_value(_trade(TradeDirection.LONG), model)
    short_value = trade_value(_trade(TradeDirection.SHORT), model)
    assert isinstance(long_value, PricingResult) and isinstance(short_value, PricingResult)
    assert short_value.pv.amount == -long_value.pv.amount


def test_the_asymmetry_fixtures_payoff_is_NOT_symmetric_so_the_sign_is_measurable():
    """The degeneracy premise, measured: how far apart a dropped sign would land.

    On a payoff worth ~0 (an at-the-money forward) `−V == V` to within noise and the oracle above is
    vacuous. This fixture's OTM call is worth a definite positive amount, so the separation between
    long and short is the whole notional-scaled value — reported rather than assumed.
    """
    model = _model()
    long_value = trade_value(_trade(TradeDirection.LONG), model)
    assert isinstance(long_value, PricingResult)
    # the value is materially non-zero: a sign error moves it by twice this
    assert long_value.pv.amount > 100.0, long_value.pv.amount
    separation = 2.0 * abs(long_value.pv.amount)
    assert separation > 200.0, separation


# ============================================================================================
# 3. The notional — proportionality AND the absolute level (T43).
# ============================================================================================

def test_the_value_is_PROPORTIONAL_to_the_notional():
    model = _model()
    single = trade_value(_trade(notional=NOTIONAL), model)
    double = trade_value(_trade(notional=2.0 * NOTIONAL), model)
    assert isinstance(single, PricingResult) and isinstance(double, PricingResult)
    assert double.pv.amount == pytest.approx(2.0 * single.pv.amount, rel=1e-15, abs=0.0)


def test_the_ABSOLUTE_level_is_pinned_independently_because_scaling_cannot_see_it():
    """**T43's trap, in the first representation tranche.** Proportionality pins shape, not level.

    `V(2N) == 2·V(N)` is homogeneous, so a `trade_value` that applied the notional **twice**, or
    halved it, or used the product's own notional instead of the trade's, would satisfy it perfectly.
    So the level is asserted against a reference composed **outside** the trade path: the bare price
    from the registry, multiplied by the notional here in the test.
    """
    model = _model()
    bare = price(_call(), model)
    assert isinstance(bare, PricingResult)
    expected = NOTIONAL * bare.pv.amount

    actual = trade_value(_trade(notional=NOTIONAL), model)
    assert isinstance(actual, PricingResult)
    assert actual.pv.amount == pytest.approx(expected, rel=1e-15, abs=0.0)
    # and the premise: the notional is NOT 1, so this is a different claim from the reduction oracle
    assert abs(expected / bare.pv.amount - 1.0) > 1000.0


# ============================================================================================
# 4. The §3 contract rides through the wrapper.
# ============================================================================================

def test_an_UNREGISTERED_pair_is_a_PRICING_FAILURE_not_a_raise_or_a_silent_zero():
    """§3.4: failure is a value. A wrapper that let the registry's `PricingFailure` become an
    exception — or worse, scored it as zero — would break the contract one layer up."""

    class UnpriceableWidget:
        pass

    outcome = trade_value(
        Trade(
            product=UnpriceableWidget(),
            notional=Money(NOTIONAL, USD),
            direction=TradeDirection.LONG,
        ),
        _model(),
    )
    assert isinstance(outcome, PricingFailure)


def test_a_MATURED_product_keeps_its_PV_ZERO_through_the_wrapper():
    # invariant 6 belongs to the engine; the wrapper must not turn a legitimate zero into something
    # else, nor multiply a failure. A zero scaled by any notional is still zero.
    late = MarketSnapshot(
        valuation_date=EXPIRY + timedelta(days=5),
        curves=CurveSet(
            {
                CurveKey(CurveRole.DISCOUNT, USD): DiscountCurve.flat(
                    EXPIRY + timedelta(days=5), 0.03, DC
                )
            }
        ),
        equities={ACME: EquityUnderlying(spot=100.0, repo_rate=0.01)},
    )
    outcome = trade_value(_trade(), EquityBlackModel(market=late, volatility=0.28))
    assert isinstance(outcome, PricingResult)
    assert outcome.pv.amount == 0.0


# ============================================================================================
# 4b. The STANDARD ERROR — the §3.7 invariant, and the hole the sweep found.
# ============================================================================================

@dataclass(frozen=True)
class _EstimatedProduct:
    """A private product whose pricer returns an ESTIMATE — the fixture population's missing half."""

    currency: Currency


@dataclass(frozen=True)
class _EstimatingModel:
    """A private model, so registering this pair collides with nothing real."""

    market: MarketSnapshot


#: A known, exact standard error. Registering a test-only pricer is the established pattern in this
#: tree (`tests/tooling/test_engine_registry.py` does it with private `_ProductA`/`_ModelOne` types).
_ESTIMATE_PV = 12.5
_ESTIMATE_SE = 0.25


@register(_EstimatedProduct, _EstimatingModel)
def _price_estimated(
    product: _EstimatedProduct, model: _EstimatingModel
) -> PricingResult | PricingFailure:
    del model
    return PricingResult(
        pv=Money(_ESTIMATE_PV, product.currency), standard_error=_ESTIMATE_SE
    )


def test_the_STANDARD_ERROR_scales_with_the_notional_and_ignores_the_direction():
    """**The sweep's one survivor, and its cause was my fixture POPULATION.**

    Dropping `abs(scale) * priced.standard_error` to a bare pass-through survived all 3314 tests,
    because every *real* product in this file is priced by an **exact** engine — so
    `standard_error` is always `None` and the scaling line never executed with a real value. Not a
    degenerate value and not a degenerate schedule: a degenerate **population of pricers** (T33b's
    input-population shape, one coordinate over). Expect it again at R1.2, R1.4 and R2.

    **The fixture is a registered test-only pricer returning a KNOWN estimate, not a real MC
    engine, and that is the second lesson.** The first version dispatched a `TimerOption` under
    Heston — the only registry-dispatchable pair that carries an SE — which cost 7.31s locally and
    **16.0s / 17.9s on the two CI runners**, breaking the 10s ceiling. §9 says plainly that for a
    file with real numerical cost **CI is the reading and a local re-time is not**, and this is the
    third time the project has made that mistake (T42, T44a). The fix is lossless *and* better: a
    known SE makes the assertion **exact** rather than banded, costs nothing, and tests the
    **wrapper's** arithmetic — which is what T82 owns — instead of incidentally re-testing Heston.

    Two claims in one assertion: the band **scales** with the units, and is **sign-independent** —
    an error bar has no direction, so `abs(scale)` is load-bearing. A **short** trade's band
    compared against a *positive* reference tests both at once, since a signed `scale` would return
    a negative band and fail the equality.
    """
    product = _EstimatedProduct(currency=USD)
    model = _EstimatingModel(market=_market())

    bare = price(product, model)
    assert isinstance(bare, PricingResult)
    # the premise: this pricer DOES carry an error bar, unlike every other fixture in this file
    assert bare.standard_error == _ESTIMATE_SE

    short_trade = trade_value(
        Trade(product=product, notional=Money(NOTIONAL, USD), direction=TradeDirection.SHORT),
        model,
    )
    assert isinstance(short_trade, PricingResult)
    assert short_trade.standard_error == pytest.approx(
        NOTIONAL * _ESTIMATE_SE, rel=1e-15, abs=0.0
    )
    # the premise that makes the above a SIGN claim and not only a scaling one: this really is a
    # short, so a signed band would have come back negative.
    assert short_trade.pv.amount == pytest.approx(-NOTIONAL * _ESTIMATE_PV, rel=1e-15, abs=0.0)
    assert short_trade.pv.amount < 0.0


def test_the_RELATIVE_error_is_invariant_under_size_and_direction():
    """The independent second killer for the SE's scaling — and now it is cheap enough to have.

    Scaling a position does not change how *uncertain* it is: the coefficient of variation
    `SE/|PV|` is invariant under both the size and the direction. That is a finance requirement
    rather than a restatement of `abs(scale) * se`, so it is independent of the assertion above —
    and it kills both defects for structural reasons: if the band fails to scale while the value
    does, the ratio moves; if the band carries the sign, the ratio goes negative.

    Recorded in the audit log as a single-killer when the fixture was a 7s Monte-Carlo dispatch and
    a second killer would have cost a `slow` exemption. The registered known-estimate fixture made
    hardening free, so §9's *ceiling* applies instead of its floor and the log row is retracted.
    """
    product = _EstimatedProduct(currency=USD)
    model = _EstimatingModel(market=_market())
    reference = _ESTIMATE_SE / abs(_ESTIMATE_PV)

    for direction in (TradeDirection.LONG, TradeDirection.SHORT):
        for units in (1.0, NOTIONAL, 3.0 * NOTIONAL):
            outcome = trade_value(
                Trade(product=product, notional=Money(units, USD), direction=direction), model
            )
            assert isinstance(outcome, PricingResult)
            assert outcome.standard_error is not None
            ratio = outcome.standard_error / abs(outcome.pv.amount)
            assert ratio == pytest.approx(reference, rel=1e-14, abs=0.0), (direction, units, ratio)


def test_an_EXACT_pricer_keeps_a_None_band_through_the_wrapper():
    """The other branch, and the reason the test above needed a new fixture at all.

    An exact pricer has no error bar, and the wrapper must not invent one — `None` in, `None` out.
    This is the branch every *other* fixture in the file exercises, which is exactly why none of
    them could see the scaling defect.
    """
    outcome = trade_value(_trade(), _model())
    assert isinstance(outcome, PricingResult)
    assert outcome.standard_error is None


# ============================================================================================
# 5. Refusals — supplying the input each guard exists to reject (T33b).
# ============================================================================================

def test_a_notional_in_the_WRONG_CURRENCY_is_refused():
    """The guard that gives the `Money` notional its meaning.

    The notional's **amount** is how many units of the contract the trade holds; its **currency** is
    what makes it a `Money` rather than a bare float, and the only thing that can be checked about it
    is that it agrees with what the product pays in. Supplied the mismatching input rather than
    assumed unreachable (T33b).
    """
    model = _model()
    mismatched = Trade(
        product=_call(), notional=Money(NOTIONAL, EUR), direction=TradeDirection.LONG
    )
    outcome = trade_value(mismatched, model)
    assert isinstance(outcome, PricingFailure)
    assert "currenc" in outcome.reason.lower(), outcome.reason


def test_the_currency_failure_NAMES_BOTH_currencies_a_second_killer():
    """The second killer for the currency guard, which the sweep measured at **one**.

    §9 prefers two for anything load-bearing: with a single killer, deleting that one test reopens
    the hole silently. This one is independent of the first — it asserts the failure's *content*
    (both currency codes, so a reader can see which way the mismatch runs) rather than merely that
    a failure occurred.
    """
    outcome = trade_value(
        Trade(product=_call(), notional=Money(NOTIONAL, EUR), direction=TradeDirection.LONG),
        _model(),
    )
    assert isinstance(outcome, PricingFailure)
    assert "EUR" in outcome.reason and "USD" in outcome.reason, outcome.reason


def test_an_underlying_failures_REASON_is_preserved_verbatim_a_second_killer():
    """The second killer for the failure pass-through, also measured at **one**.

    Independent of the `isinstance` check: it asserts the underlying engine's own reason survives
    unwrapped. A wrapper that caught the failure and re-worded it would pass the first test and lose
    the diagnostic that says *why* the trade could not be priced.
    """

    class UnpriceableWidget:
        pass

    direct = price(UnpriceableWidget(), _model())
    assert isinstance(direct, PricingFailure)

    through_trade = trade_value(
        Trade(
            product=UnpriceableWidget(),
            notional=Money(NOTIONAL, USD),
            direction=TradeDirection.LONG,
        ),
        _model(),
    )
    assert isinstance(through_trade, PricingFailure)
    assert through_trade.reason == direct.reason, (through_trade.reason, direct.reason)


@pytest.mark.parametrize("notional", [0.0, -1.0, float("nan"), float("inf")])
def test_the_trade_REFUSES_a_non_positive_or_non_finite_notional(notional):
    with pytest.raises(ValueError):
        Trade(
            product=_call(),
            notional=Money(notional, USD),
            direction=TradeDirection.LONG,
        )


def test_the_direction_SIGNS_are_plus_and_minus_one():
    assert TradeDirection.LONG.sign == 1.0
    assert TradeDirection.SHORT.sign == -1.0


def test_a_trade_is_FROZEN():
    from dataclasses import FrozenInstanceError

    trade = _trade()
    with pytest.raises(FrozenInstanceError):
        setattr(trade, "notional", Money(1.0, USD))
