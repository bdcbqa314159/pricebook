"""T86 — R1.5: the mixed-asset book capstone. The R1 arc's proof that R1.1-R1.4 COMPOSE.

Three asset classes — **rates** (an interest-rate swap), **credit** (a CDS), **equity** (a European
option) — in one `Position`, priced by three different models off **one `MarketSnapshot`**, valued by
one `mark_to_market(request)` call into one `MtMReport`. Nothing here is new machinery; that is the
point. The earlier tranches each proved a piece against test pricers, and this proves the pieces
compose over the real catalogue.

**There is no RED commit for this tranche, and that is not a lapse.** §4 pre-registers a failing
reference-value test before *the code that satisfies it* — but a capstone adds no `src/` surface, so
there is no code to write after the test and nothing to bend toward a number. What §4 asks for in
that position is the mutation sweep: the empirical proof that these oracles have power, which is run
and reported rather than assumed.

**Two axes are exercised by registered TEST pricers, and the reason is measured, not preferential.**
The book's standard error and its clean/accrued decomposition need an estimate pricer and a
decomposing pricer respectively. On the real catalogue:

* **no engine populates `accrued`** (0 of 50 at T85), so the clean/accrued columns have no production
  path at all — the T33b input-population degeneracy, which is why 0459(b) ruled the test pricer;
* the **only registered estimate pricer** is `TimerOption <- HestonModel`, and it costs **~2.8s per
  dispatch locally** (measured at T86), so a capstone exercising a real Monte-Carlo band would cost
  6-9s *per call* on the CI runners against a ~10s per-file budget. §9 is explicit that CI is the
  reading and that an oracle must not be traded for the stopwatch, so the band is exercised by the
  T82 known-estimate pricer and the cost is recorded in `to_go_deeper.md` instead of gambling the
  gate. The real-catalogue trades carry the integration; the test pricers carry the two axes the real
  catalogue cannot reach.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.engine import price
from pricebook.engine.mtm import mark_to_market
from pricebook.engine.pricing_request import value
from pricebook.engine.registry import register
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.money import Currency, Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.schedule import CouponSchedule
from pricebook.foundation.tenor import Tenor
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.curve_set import ProjectionKey
from pricebook.market.equity import EquityAsset, EquityUnderlying
from pricebook.market.survival_curve import ReferenceEntity, SurvivalCurve
from pricebook.models.credit import CreditModel
from pricebook.models.discounting_model import DiscountingModel
from pricebook.models.equity_black import EquityBlackModel
from pricebook.products.credit_default_swap import CreditDefaultSwap, ProtectionDirection
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption
from pricebook.products.interest_rate_swap import InterestRateSwap, SwapDirection
from pricebook.products.position import Position
from pricebook.products.pricing_request import PricingRequest
from pricebook.products.trade import Trade, TradeDirection

VAL = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
TM = TimeMeasure(VAL, DC)
USD = Currency("USD")
EUR = Currency("EUR")
SIX_MONTH = Tenor(months=6)
DISCOUNT = CurveKey(CurveRole.DISCOUNT, USD)

#: Sloped, not flat: a flat curve hides anything that depends on *where* on the curve a flow lands.
ZERO_RATES = (0.020, 0.026, 0.030, 0.033)
#: Term-structured hazard, for the same reason (T39: a flat hazard has no interior pillars).
HAZARD = SurvivalCurve(TM, (0.0, 1.0, 2.0), (0.010, 0.022, 0.040))
ENTITY = ReferenceEntity("ACME")
STOCK = EquityAsset("ACME")
SPOT, VOL, RECOVERY = 100.0, 0.25, 0.4
SEMI_ANNUAL = CouponSchedule(tuple(VAL + timedelta(days=182 * i) for i in range(7)), DC)
SHOCK_BP = 0.01


def _curve(shift: float = 0.0) -> DiscountCurve:
    return DiscountCurve(TM, (0.0, 1.0, 2.0, 3.0), tuple(r + shift for r in ZERO_RATES))


def _market(shift: float = 0.0) -> MarketSnapshot:
    """**One** snapshot serving three asset classes — which is the §0 contract's whole claim.

    `shift` bumps the discount curve in parallel, and the same curve is the projection curve, so the
    shock reaches the swap's forwards, the CDS's discounting *and* the option's carry — a factor the
    book genuinely depends on, which the scenario premise requires (a shock to an unused factor would
    make the preview vacuous).
    """
    curve = _curve(shift)
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet(
            {DISCOUNT: curve},
            projections={ProjectionKey(USD, SIX_MONTH): curve},
            survivals={ENTITY: HAZARD},
        ),
        equities={STOCK: EquityUnderlying(spot=SPOT, repo_rate=0.01)},
    )


SWAP = InterestRateSwap(
    schedule=SEMI_ANNUAL,
    fixed_rate=0.028,
    notional=Money(10_000_000.0, USD),
    tenor=SIX_MONTH,
    direction=SwapDirection.PAYER,
)
CDS = CreditDefaultSwap(
    schedule=SEMI_ANNUAL,
    spread=0.01,
    notional=Money(5_000_000.0, USD),
    direction=ProtectionDirection.BUYER,
    reference_entity=ENTITY,
)
OPTION = EuropeanEquityOption(
    asset=STOCK,
    strike=105.0,
    expiry=VAL + timedelta(days=365),
    notional=Money(100_000.0, USD),
    kind=EquityOptionKind.CALL,
)

#: Varied units and a SHORT, so the scaling and the direction both do real work in the aggregate.
#: A book of unit-sized longs would be degenerate in exactly the way the catalogue warns about.
BOOK_UNITS = ((SWAP, 1.0, TradeDirection.LONG), (CDS, 2.0, TradeDirection.LONG),
              (OPTION, 3.0, TradeDirection.SHORT))


def _models(shift: float = 0.0) -> dict[type, object]:
    market = _market(shift)
    return {
        InterestRateSwap: DiscountingModel(market),
        CreditDefaultSwap: CreditModel(market, RECOVERY),
        EuropeanEquityOption: EquityBlackModel(market, VOL),
    }


def _trades() -> tuple[Trade, ...]:
    return tuple(
        Trade(product=product, units=units, direction=direction)
        for product, units, direction in BOOK_UNITS
    )


def _request(shift: float = 0.0, currency: Currency = USD) -> PricingRequest:
    return PricingRequest(
        position=Position(currency=currency, trades=_trades()),
        assignment=_models(shift),
        as_of=VAL,
    )


def _manual_total(shift: float = 0.0) -> float:
    """The book PV composed by hand: price each product independently, then scale and add.

    Deliberately **not** routed through `trade_value` or `position_value` — it calls `price` directly
    and does its own arithmetic, so it is an independent reference for the whole R1 stack rather than
    a restatement of it (§4 independence). This is what gives the reconciliation content.
    """
    models = _models(shift)
    total = 0.0
    for product, units, direction in BOOK_UNITS:
        priced = price(product, models[type(product)])
        assert not isinstance(priced, PricingFailure), priced
        total += direction.sign * units * priced.pv.amount
    return total


# ============================================================================================
# 0. The fixture's own premises — asserted, not trusted (§4: assert the non-degeneracy).
# ============================================================================================


def test_the_capstone_book_is_GENUINELY_mixed_in_products_models_and_asset_classes():
    """A single-asset or single-model capstone would test nothing the earlier tranches did not.

    Three distinct product types, three distinct model types, three distinct `(product, model)`
    dispatch pairs — so cross-asset dispatch, aggregation, the currency check and the as-of check all
    fire together, which is the integration a one-asset book cannot reach.
    """
    request = _request()
    product_types = {type(trade.product) for trade in request.position.trades}
    model_types = {type(model) for model in request.assignment.values()}
    assert len(product_types) == 3, product_types
    assert len(model_types) == 3, model_types
    assert len(request.position.trades) == 3

    # And the three really are different asset classes, not three rates products.
    assert product_types == {InterestRateSwap, CreditDefaultSwap, EuropeanEquityOption}


def test_every_trade_in_the_book_prices_to_a_NON_ZERO_value():
    """A zero makes a factor invisible: a trade worth nothing cannot witness its own mis-scaling.

    The equity option in particular is checked, because an out-of-the-money option near expiry would
    price to approximately zero and silently drop out of every sum below (F-4.1's shape).
    """
    models = _models()
    for product, _, _ in BOOK_UNITS:
        priced = price(product, models[type(product)])
        assert not isinstance(priced, PricingFailure), priced
        assert abs(priced.pv.amount) > 1.0, (type(product).__name__, priced.pv.amount)


# ============================================================================================
# 1. The capstone identity — one request, one report, reconciled against the manual composition.
# ============================================================================================


def test_the_mixed_book_reconciles_with_the_MANUAL_composition_of_its_three_trades():
    """The R1 arc's headline: `PricingRequest -> MtMReport` equals pricing by hand and adding up.

    The reference calls `price` directly and scales by hand, so it shares no code with the stack
    under test — this is a reconciliation, not a telescope.
    """
    report = mark_to_market(_request())
    assert not isinstance(report, PricingFailure), report
    expected = _manual_total()
    assert report.total.pv.currency == USD
    assert report.total.pv.amount == expected


def test_the_report_and_the_scalar_value_agree_on_the_mixed_book():
    """The guard (they share the path), asserted on the real catalogue rather than on test pricers."""
    request = _request()
    report = mark_to_market(request)
    scalar = value(request)
    assert not isinstance(report, PricingFailure), report
    assert not isinstance(scalar, PricingFailure), scalar
    assert report.total.pv == scalar.pv
    assert report.as_of == VAL


def test_each_row_scales_its_own_products_price_by_that_trades_units_and_direction():
    """Per row, against an independent `price` call — so a mis-scaled or mis-routed row dies.

    The short equity leg is what makes the direction observable: were the sign dropped, the total
    would move by twice the option leg.
    """
    report = mark_to_market(_request())
    assert not isinstance(report, PricingFailure), report
    models = _models()
    assert len(report.rows) == 3
    for row, (product, units, direction) in zip(report.rows, BOOK_UNITS):
        independent = price(product, models[type(product)])
        assert not isinstance(independent, PricingFailure), independent
        assert row.result.pv.amount == direction.sign * units * independent.pv.amount

    assert sum(row.result.pv.amount for row in report.rows) == report.total.pv.amount


# ============================================================================================
# 2. Attribution ACROSS asset classes — a desk reads which model marked the CDS.
# ============================================================================================


def test_each_row_names_the_right_product_AND_the_right_model_across_three_asset_classes():
    """Numbers right with attribution wrong is the failure a reconciliation cannot see.

    Three distinct model types, so a mutation mis-routing one asset class's model changes an
    observable here — which a book sharing one model across its trades could not detect.
    """
    request = _request()
    report = mark_to_market(request)
    assert not isinstance(report, PricingFailure), report

    assert [type(row.trade.product) for row in report.rows] == [
        InterestRateSwap, CreditDefaultSwap, EuropeanEquityOption
    ]
    assert isinstance(report.rows[0].model, DiscountingModel)
    assert isinstance(report.rows[1].model, CreditModel)
    assert isinstance(report.rows[2].model, EquityBlackModel)
    for row in report.rows:
        assert row.model is request.assignment[type(row.trade.product)]


# ============================================================================================
# 3. The scenario PREVIEW — the seam to R3. A reprice, because the snapshot is immutable.
# ============================================================================================


def test_a_shocked_snapshot_reprices_the_whole_book_and_the_PV_MOVES():
    """The functional-core payoff: a scenario is a pure reprice against a new snapshot, no engine.

    The shock must move a factor the book depends on, else the preview is vacuous — so the move is
    asserted to be **material** (basis points of the notional, not floating-point noise) rather than
    merely non-zero.
    """
    base = mark_to_market(_request())
    shocked = mark_to_market(_request(shift=SHOCK_BP))
    assert not isinstance(base, PricingFailure), base
    assert not isinstance(shocked, PricingFailure), shocked

    move = shocked.total.pv.amount - base.total.pv.amount
    assert abs(move) > 1_000.0, move


def test_the_shocked_reprice_equals_an_INDEPENDENT_manual_reprice_under_the_same_shock():
    """Exactly — the scenario is not approximated, it is the same composition on other inputs."""
    shocked = mark_to_market(_request(shift=SHOCK_BP))
    assert not isinstance(shocked, PricingFailure), shocked
    assert shocked.total.pv.amount == _manual_total(shift=SHOCK_BP)


def test_the_shock_does_NOT_leak_into_the_base_mark():
    """The immutability guarantee, asserted rather than trusted: repricing the base after the shock
    gives bit-identical numbers.

    A scenario engine built on a mutable snapshot would corrupt the mark it was measuring against,
    and the symptom — a base PV that drifts once a scenario has been run — is the kind of thing that
    is noticed months later. Here it is a one-line check because the snapshot is frozen.
    """
    before = mark_to_market(_request())
    assert not isinstance(before, PricingFailure), before
    shocked = mark_to_market(_request(shift=SHOCK_BP))
    assert not isinstance(shocked, PricingFailure), shocked
    after = mark_to_market(_request())
    assert not isinstance(after, PricingFailure), after

    assert after.total.pv.amount == before.total.pv.amount
    assert [row.result.pv.amount for row in after.rows] == [
        row.result.pv.amount for row in before.rows
    ]
    assert shocked.total.pv.amount != before.total.pv.amount


def test_every_row_moves_under_the_shock_so_no_leg_is_insensitive_to_it():
    """The premise behind the preview: the shock reaches all three asset classes, not just one.

    Were it to move only the swap, the capstone would be demonstrating a rates scenario and calling
    it a book scenario — the degenerate-coverage shape, at book scale.
    """
    base = mark_to_market(_request())
    shocked = mark_to_market(_request(shift=SHOCK_BP))
    assert not isinstance(base, PricingFailure), base
    assert not isinstance(shocked, PricingFailure), shocked
    for before_row, after_row in zip(base.rows, shocked.rows):
        assert before_row.result.pv.amount != after_row.result.pv.amount, (
            type(before_row.trade.product).__name__
        )


# ============================================================================================
# 4. The whole-book refusals, inside the MIXED book — each pinned to the guard that fired.
# ============================================================================================


def test_one_unassigned_asset_class_refuses_the_whole_mixed_book():
    """Two of three assigned: the book is refused rather than marked at two-thirds of itself."""
    assignment = dict(_models())
    del assignment[CreditDefaultSwap]
    request = PricingRequest(
        position=Position(currency=USD, trades=_trades()), assignment=assignment, as_of=VAL
    )
    outcome = mark_to_market(request)
    assert isinstance(outcome, PricingFailure)
    assert "assigns no model" in outcome.reason, outcome.reason
    assert "CreditDefaultSwap" in outcome.reason, outcome.reason
    assert "valuation date" not in outcome.reason.lower(), outcome.reason


def test_one_model_as_of_the_wrong_date_refuses_the_whole_mixed_book():
    """A book cannot be marked half as of today and half as of yesterday, even if it would sum."""
    # A *self-consistent* yesterday: its curves are anchored to yesterday too, so nothing inside
    # the model is mismatched. That is the point of the §3 refusal — a book assembled from
    # yesterday's snapshots values itself perfectly well, and only the request's declared date
    # reveals that it is not the mark anyone asked for.
    yesterday = VAL - timedelta(days=1)
    stale_measure = TimeMeasure(yesterday, DC)
    stale = MarketSnapshot(
        valuation_date=yesterday,
        curves=CurveSet(
            {CurveKey(CurveRole.DISCOUNT, USD): DiscountCurve(
                stale_measure, (0.0, 1.0, 2.0, 3.0), ZERO_RATES
            )},
            survivals={ENTITY: SurvivalCurve(stale_measure, (0.0, 1.0, 2.0), (0.010, 0.022, 0.040))},
        ),
    )
    assignment = dict(_models())
    assignment[CreditDefaultSwap] = CreditModel(stale, RECOVERY)
    request = PricingRequest(
        position=Position(currency=USD, trades=_trades()), assignment=assignment, as_of=VAL
    )
    outcome = mark_to_market(request)
    assert isinstance(outcome, PricingFailure)
    assert "valuation date" in outcome.reason.lower(), outcome.reason
    assert "CreditDefaultSwap" in outcome.reason, outcome.reason
    assert "assigns no model" not in outcome.reason, outcome.reason


def test_a_book_declaring_the_wrong_currency_refuses_rather_than_summing_unlike_money():
    """The trades price in USD; the book says EUR. Adding them would report a meaningless number."""
    outcome = mark_to_market(_request(currency=EUR))
    assert isinstance(outcome, PricingFailure)
    assert "cannot be summed" in outcome.reason, outcome.reason


# ============================================================================================
# 5. The two axes the real catalogue cannot reach — band and decomposition, via test pricers.
# ============================================================================================


@dataclass(frozen=True)
class _Estimated:
    currency: Currency


@dataclass(frozen=True)
class _Accruing:
    currency: Currency


@dataclass(frozen=True)
class _TestModel:
    market: MarketSnapshot


PV_EST, SE_EST = 25_000.0, 400.0
PV_ACC, ACCRUED_ACC = 60_000.0, 1_500.0


@register(_Estimated, _TestModel)
def _price_estimated(product: _Estimated, model: _TestModel) -> PricingResult | PricingFailure:
    """A known Monte-Carlo-shaped estimate. Stands in for a real MC pricer at ~0.0s rather than 2.8s."""
    del model
    return PricingResult(pv=Money(PV_EST, product.currency), standard_error=SE_EST)


@register(_Accruing, _TestModel)
def _price_accruing(product: _Accruing, model: _TestModel) -> PricingResult | PricingFailure:
    """Decomposes — which no engine in the tree does (0 of 50), hence this pricer exists."""
    del model
    return PricingResult(
        pv=Money(PV_ACC, product.currency), accrued=Money(ACCRUED_ACC, product.currency)
    )


def _augmented_request() -> PricingRequest:
    """The mixed book plus the two axes the real catalogue cannot exercise."""
    market = _market()
    trades = _trades() + (
        Trade(product=_Estimated(USD), units=2.0, direction=TradeDirection.LONG),
        Trade(product=_Accruing(USD), units=3.0, direction=TradeDirection.LONG),
    )
    assignment = dict(_models())
    assignment[_Estimated] = _TestModel(market)
    assignment[_Accruing] = _TestModel(market)
    return PricingRequest(
        position=Position(currency=USD, trades=trades), assignment=assignment, as_of=VAL
    )


def test_a_real_book_with_one_estimate_carries_a_BAND_and_the_exact_trades_do_not():
    """The band is the estimate's alone: five rows, one SE, and the book band equals it exactly.

    This is what the exact catalogue cannot show — every real pricer in the book returns `None`, so
    without an estimate in the population the book's band would be `None` whatever the aggregation
    did (T33b, the input-population degeneracy).
    """
    report = mark_to_market(_augmented_request())
    assert not isinstance(report, PricingFailure), report
    assert len(report.rows) == 5
    assert [row.result.standard_error for row in report.rows[:3]] == [None, None, None]
    assert report.rows[3].result.standard_error == 2.0 * SE_EST
    assert report.rows[4].result.standard_error is None
    assert report.total.standard_error == 2.0 * SE_EST


def test_a_real_book_with_one_decomposing_trade_reports_clean_and_accrued_exactly():
    """`accrued` sums only the decomposing row; `clean = dirty − accrued` across the whole book.

    The four non-decomposing rows contribute zero accrued, which the sum identity forces (T85): their
    own `clean` is their `pv`, so the book's clean is the sum of the rows' cleans only this way.
    """
    report = mark_to_market(_augmented_request())
    assert not isinstance(report, PricingFailure), report
    assert report.total.accrued == Money(3.0 * ACCRUED_ACC, USD)
    assert report.total.clean == report.total.pv - Money(3.0 * ACCRUED_ACC, USD)
    assert report.rows[4].result.accrued == Money(3.0 * ACCRUED_ACC, USD)

    # And the real-catalogue rows are genuinely silent on it, which is the premise above.
    assert all(row.result.accrued is None for row in report.rows[:4])


def test_the_augmented_book_still_reconciles_with_the_manual_composition():
    """The real trades and the test-pricer trades total together, so neither path is special-cased."""
    report = mark_to_market(_augmented_request())
    assert not isinstance(report, PricingFailure), report
    expected = _manual_total() + 2.0 * PV_EST + 3.0 * PV_ACC
    assert report.total.pv.amount == expected
