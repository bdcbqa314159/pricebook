"""T85 — R1.4: the mark-to-market breakdown report.

`value(request)` (T84) answers *what the book is worth*; the report answers *where that number comes
from*. The totals reconciling with `value()` is the **guard** — the two share the aggregation, so it
telescopes by construction and is satisfied by a no-op. The **test** is the breakdown: each row is its
own trade's value, the rows sum to the total in every column, and each row names the model that
actually priced it.

**The accrued decomposition is oracled against a known-accrued test pricer**, because `accrued` is
populated by **no engine in the tree** (0 of 50 at T85): on every real path `accrued is None` and
`clean` is identically `pv`, so an oracle built on production pricers alone could not tell a correct
accrued aggregation from one that dropped it — the T33b input-population degeneracy, with the one
respect the code discriminates on uniform across the whole input population. Ruled 0459(b).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.engine.mtm import mark_to_market
from pricebook.engine.pricing_request import value
from pricebook.engine.registry import register
from pricebook.engine.trade import trade_value
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.products.position import Position
from pricebook.products.pricing_request import PricingRequest
from pricebook.products.trade import Trade, TradeDirection

TODAY = date(2026, 1, 15)
YESTERDAY = TODAY - timedelta(days=1)
USD = Currency("USD")
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED


def _market(valuation: date = TODAY) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=valuation,
        curves=CurveSet(
            {CurveKey(CurveRole.DISCOUNT, USD): DiscountCurve.flat(valuation, 0.03, DC)}
        ),
    )


# ============================================================================================
# The fixture population. Four pricers, each present for a reason the oracles below name:
# two estimates (the total SE does real work), one KNOWN-ACCRUED exact pricer (the clean/accrued
# columns are non-degenerate), and one whose accrued is in the WRONG currency (the aggregation's
# refusal is reachable rather than dead code).
# ============================================================================================


@dataclass(frozen=True)
class _EstimateA:
    currency: Currency


@dataclass(frozen=True)
class _EstimateB:
    currency: Currency


@dataclass(frozen=True)
class _Accruing:
    currency: Currency


@dataclass(frozen=True)
class _BadAccrual:
    currency: Currency


@dataclass(frozen=True)
class _ModelA:
    market: MarketSnapshot


@dataclass(frozen=True)
class _ModelB:
    market: MarketSnapshot


@dataclass(frozen=True)
class _ModelAcc:
    market: MarketSnapshot


@dataclass(frozen=True)
class _ModelBad:
    market: MarketSnapshot


PV_A, SE_A = 10.0, 0.30
PV_B, SE_B = 20.0, 0.40
PV_ACC, ACCRUED_ACC = 50.0, 1.50


@register(_EstimateA, _ModelA)
def _price_a(product: _EstimateA, model: _ModelA) -> PricingResult | PricingFailure:
    del model
    return PricingResult(pv=Money(PV_A, product.currency), standard_error=SE_A)


@register(_EstimateB, _ModelB)
def _price_b(product: _EstimateB, model: _ModelB) -> PricingResult | PricingFailure:
    del model
    return PricingResult(pv=Money(PV_B, product.currency), standard_error=SE_B)


@register(_Accruing, _ModelAcc)
def _price_accruing(product: _Accruing, model: _ModelAcc) -> PricingResult | PricingFailure:
    """Exact, and it DECOMPOSES: `clean = 50 − 1.5 = 48.5`. No engine in the tree does this yet."""
    del model
    return PricingResult(
        pv=Money(PV_ACC, product.currency), accrued=Money(ACCRUED_ACC, product.currency)
    )


@register(_BadAccrual, _ModelBad)
def _price_bad_accrual(product: _BadAccrual, model: _ModelBad) -> PricingResult | PricingFailure:
    """A pricer whose `accrued` currency disagrees with its own `pv` — an engine bug, by construction.

    It exists so the aggregation's accrued-currency refusal is **reachable from a test** rather than
    dead defensive code (§4's third survivor kind). `Money` raises on mixing, so without the check
    the aggregation would raise instead of returning a failure value, breaking §3.4.
    """
    del model
    return PricingResult(pv=Money(5.0, product.currency), accrued=Money(0.25, EUR))


def _request(*trades: Trade, as_of: date = TODAY, currency: Currency = USD) -> PricingRequest:
    market = _market(as_of)
    return PricingRequest(
        position=Position(currency=currency, trades=trades),
        assignment={
            _EstimateA: _ModelA(market),
            _EstimateB: _ModelB(market),
            _Accruing: _ModelAcc(market),
            _BadAccrual: _ModelBad(market),
        },
        as_of=as_of,
    )


def _long(product: object, units: float = 1.0) -> Trade:
    return Trade(product=product, units=units, direction=TradeDirection.LONG)


# ============================================================================================
# 1. The reconciliation GUARD — the report and the scalar are two views of one computation.
# ============================================================================================


def test_the_report_totals_reconcile_with_value_exactly():
    """The guard, not the test: they share the aggregation, so this telescopes by construction.

    It is still worth asserting — it is what makes "two views of one computation" a fact rather than
    an intention, and it would catch a future fork of the aggregation (the C16/C17 duplication the
    re-audits hunt). The breakdown oracles below are what have real power.
    """
    request = _request(_long(_EstimateA(USD), 2.0), _long(_EstimateB(USD), 3.0))
    report = mark_to_market(request)
    scalar = value(request)
    assert not isinstance(report, PricingFailure)
    assert not isinstance(scalar, PricingFailure)
    assert report.total.pv == scalar.pv
    assert report.total.standard_error == scalar.standard_error
    assert report.total.accrued == scalar.accrued
    assert report.total.clean == scalar.clean


def test_the_report_carries_the_as_of_date_it_was_marked_on():
    """A mark without its date is not a mark — constitutive of the report's identity (§6)."""
    report = mark_to_market(_request(_long(_EstimateA(USD))))
    assert not isinstance(report, PricingFailure)
    assert report.as_of == TODAY


# ============================================================================================
# 2. The breakdown — each row is its own trade's value, and the rows SUM to the total.
# ============================================================================================


def test_each_row_equals_its_own_trade_value_and_the_rows_sum_to_the_total():
    trade_a, trade_b = _long(_EstimateA(USD), 2.0), _long(_EstimateB(USD), 3.0)
    request = _request(trade_a, trade_b)
    report = mark_to_market(request)
    assert not isinstance(report, PricingFailure)

    assert len(report.rows) == 2
    assert [row.trade for row in report.rows] == [trade_a, trade_b]
    for row in report.rows:
        independent = trade_value(row.trade, row.model)
        assert not isinstance(independent, PricingFailure)
        assert row.result.pv == independent.pv
        assert row.result.standard_error == independent.standard_error

    # 2 x 10 + 3 x 20 = 80, exactly.
    assert report.total.pv == Money(80.0, USD)
    assert sum(row.result.pv.amount for row in report.rows) == report.total.pv.amount


def test_a_row_is_present_for_every_trade_including_a_repeated_product():
    """The breakdown is per TRADE, not per product: two trades on one product are two rows.

    A report that keyed its rows by product type would silently net them, which is the one thing a
    breakdown exists not to do — a desk reading two rows of 10 cannot see them as one row of 20.
    """
    request = _request(_long(_EstimateA(USD)), _long(_EstimateA(USD)))
    report = mark_to_market(request)
    assert not isinstance(report, PricingFailure)
    assert len(report.rows) == 2
    assert report.total.pv == Money(2 * PV_A, USD)


def test_the_empty_book_reports_no_rows_and_a_zero_total():
    report = mark_to_market(_request())
    assert not isinstance(report, PricingFailure)
    assert report.rows == ()
    assert report.total.pv == Money(0.0, USD)
    assert report.total.standard_error is None


# ============================================================================================
# 3. The ATTRIBUTION — numbers right and attribution wrong is the "plausible but wrong" hazard.
# ============================================================================================


def test_each_row_names_the_model_that_actually_priced_that_trade():
    """A desk reads the attribution, so a label-swap must die even though every number is right.

    The fixture's two trades are **different product types with different models**, so a mutation
    that paired the rows with the wrong models changes an observable here. With one model for both
    trades the swap would be invisible (the resolution-unexercised shape).
    """
    request = _request(_long(_EstimateA(USD)), _long(_EstimateB(USD)))
    report = mark_to_market(request)
    assert not isinstance(report, PricingFailure)
    assert isinstance(report.rows[0].model, _ModelA)
    assert isinstance(report.rows[1].model, _ModelB)
    assert report.rows[0].model is request.assignment[_EstimateA]
    assert report.rows[1].model is request.assignment[_EstimateB]


def test_the_attribution_is_what_the_assignment_chose_not_the_products_only_option():
    """`_EstimateA` is priced by the model the request assigned; the row says so by identity."""
    request = _request(_long(_EstimateA(USD)))
    report = mark_to_market(request)
    assert not isinstance(report, PricingFailure)
    assert report.rows[0].model is request.assignment[_EstimateA]
    assert report.rows[0].result.pv == Money(PV_A, USD)


# ============================================================================================
# 4. The ACCRUED decomposition — non-degenerate only because of the known-accrued test pricer.
# ============================================================================================


def test_a_known_accrued_pricer_gives_the_row_an_exact_clean_and_accrued():
    """Exact values, so a dropped or mis-scaled accrued cannot hide in a tolerance.

    `units = 2` is deliberately non-unit: `accrued` scales with the trade like `pv` does, which is
    forced rather than chosen — `clean = pv − accrued` must commute with scaling, so
    `scale·pv − scale·accrued = scale·clean`.
    """
    report = mark_to_market(_request(_long(_Accruing(USD), 2.0)))
    assert not isinstance(report, PricingFailure)
    row = report.rows[0]
    assert row.result.pv == Money(100.0, USD)
    assert row.result.accrued == Money(3.0, USD)
    assert row.result.clean == Money(97.0, USD)


def test_a_SHORT_accruing_trade_carries_a_SIGNED_accrued():
    """Accrued is a money amount, not a magnitude: a short owes it, so it flips with the direction.

    The standard error does not flip (a band has no sign, T83); accrued does. Asserting both in one
    fixture is what keeps the two from being conflated.
    """
    trade = Trade(product=_Accruing(USD), units=2.0, direction=TradeDirection.SHORT)
    report = mark_to_market(_request(trade))
    assert not isinstance(report, PricingFailure)
    row = report.rows[0]
    assert row.result.pv == Money(-100.0, USD)
    assert row.result.accrued == Money(-3.0, USD)
    assert row.result.clean == Money(-97.0, USD)


def test_the_book_accrued_sums_the_rows_and_the_book_clean_follows():
    """A MIXED book: one row decomposes, one does not. The sum identity is what fixes the rule.

    `None` contributes **zero** to the accrued total rather than poisoning it, because the row's own
    `clean` is `pv` when `accrued is None` — so book clean = Σ row cleans only if the undecomposed
    rows contribute zero accrued. The aggregation rule is derived from the identity, not chosen.
    """
    request = _request(_long(_Accruing(USD), 2.0), _long(_EstimateB(USD), 3.0))
    report = mark_to_market(request)
    assert not isinstance(report, PricingFailure)

    assert report.rows[1].result.accrued is None
    assert report.total.pv == Money(100.0 + 60.0, USD)
    assert report.total.accrued == Money(3.0, USD)
    assert report.total.clean == Money(157.0, USD)
    assert report.total.clean == sum(
        (row.result.clean for row in report.rows[1:]), report.rows[0].result.clean
    )


def test_a_book_where_NOTHING_decomposes_reports_accrued_as_None_not_zero():
    """`None` is "not decomposed", and a `0.0` would claim a *measured* zero — the 0450(c) rule.

    This is the production case today: no engine populates `accrued`, so every real book lands here.
    """
    report = mark_to_market(_request(_long(_EstimateA(USD)), _long(_EstimateB(USD))))
    assert not isinstance(report, PricingFailure)
    assert report.total.accrued is None
    assert report.total.clean == report.total.pv


# ============================================================================================
# 5. The SE in the breakdown — per row faithfully, and linear in the total (T83).
# ============================================================================================


def test_a_mixed_book_reports_each_rows_SE_faithfully_and_sums_them_linearly():
    """One exact row keeps `None`; it must not become a `0.0` on its way into the report."""
    request = _request(_long(_EstimateA(USD), 2.0), _long(_Accruing(USD)))
    report = mark_to_market(request)
    assert not isinstance(report, PricingFailure)
    assert report.rows[0].result.standard_error == 2.0 * SE_A
    assert report.rows[1].result.standard_error is None
    assert report.total.standard_error == 2.0 * SE_A


def test_two_estimate_rows_add_their_bands_linearly_in_the_total():
    """Two estimates, so the total SE does real work: 2 x 0.30 + 3 x 0.40 = 1.80, not quadrature."""
    request = _request(_long(_EstimateA(USD), 2.0), _long(_EstimateB(USD), 3.0))
    report = mark_to_market(request)
    assert not isinstance(report, PricingFailure)
    assert report.total.standard_error == 2.0 * SE_A + 3.0 * SE_B
    assert report.total.standard_error == sum(
        row.result.standard_error for row in report.rows
    )


# ============================================================================================
# 6. The refusals ride through — and each is pinned to the RIGHT guard (the T84 watch).
# ============================================================================================


def test_an_unassigned_product_type_refuses_the_report_naming_THAT_guard():
    """The T84 watch: two guards can catch this input, so pin which one fired, not merely that one did."""
    position = Position(currency=USD, trades=(_long(_EstimateA(USD)),))
    outcome = mark_to_market(PricingRequest(position=position, assignment={}, as_of=TODAY))
    assert isinstance(outcome, PricingFailure)
    assert "assigns no model" in outcome.reason, outcome.reason
    assert "_EstimateA" in outcome.reason, outcome.reason
    assert "valuation date" not in outcome.reason.lower(), outcome.reason


def test_a_model_as_of_the_wrong_date_refuses_the_report_naming_THAT_guard():
    market = _market(YESTERDAY)
    position = Position(currency=USD, trades=(_long(_EstimateA(USD)),))
    outcome = mark_to_market(
        PricingRequest(position=position, assignment={_EstimateA: _ModelA(market)}, as_of=TODAY)
    )
    assert isinstance(outcome, PricingFailure)
    assert "valuation date" in outcome.reason.lower(), outcome.reason
    assert "assigns no model" not in outcome.reason, outcome.reason


def test_a_trade_pricing_in_the_wrong_currency_refuses_the_whole_report():
    """A report is not a way to smuggle an unpriceable book past the guards."""
    outcome = mark_to_market(_request(_long(_EstimateA(EUR))))
    assert isinstance(outcome, PricingFailure)
    assert "cannot be summed" in outcome.reason, outcome.reason


def test_an_accrued_in_a_foreign_currency_is_REFUSED_rather_than_raising():
    """`Money` raises on mixing, so without this guard the aggregation would raise — breaking §3.4.

    The test pricer exists to make the branch reachable; with only real engines (every one of which
    returns `accrued=None`) it would be untestable surface, which §4 says should not exist.
    """
    outcome = mark_to_market(_request(_long(_BadAccrual(USD))))
    assert isinstance(outcome, PricingFailure)
    assert "accrued" in outcome.reason.lower(), outcome.reason
    assert "EUR" in outcome.reason, outcome.reason
