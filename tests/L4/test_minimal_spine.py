"""Tranche 1 — a single fixed cashflow discounted on a flat curve, end to end.

The headline oracle is **arithmetic written independently of the code under it**: `PV = N·e^{−r·t}`.
That independence is the whole point. The predecessor tree shipped a 5.14% error because its
oracle restated the builder's own composition and so validated it against itself; `N·e^{−rt}`
restates nothing, so if the curve, the day count and the engine disagree about anything, the
number moves.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.foundation import (
    Cashflow,
    Currency,
    DayCountConvention,
    Money,
    PricingFailure,
    PricingResult,
)
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models import DiscountingModel
from pricebook.products import FixedCashflow

VAL = date(2026, 1, 15)
USD = Currency("USD")
DC = DayCountConvention.ACT_365_FIXED
DISCOUNT = CurveKey(CurveRole.DISCOUNT, USD)


def _market(rate: float = 0.05, valuation: date = VAL) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=valuation,
        curves=CurveSet({DISCOUNT: DiscountCurve.flat(valuation, rate, DC)}),
    )


def _flow(amount: float, when: date, currency: Currency = USD) -> FixedCashflow:
    return FixedCashflow(Cashflow(when, Money(amount, currency)))


def _pv(product: FixedCashflow, rate: float = 0.05, valuation: date = VAL) -> float:
    out = price(product, DiscountingModel(_market(rate, valuation)))
    assert isinstance(out, PricingResult), out
    return out.pv.amount


# ── oracle 1: the closed form, written independently ────────────────────────────────────────
def test_pv_is_notional_times_exp_minus_r_t() -> None:
    for notional in (1.0, 1_000_000.0, -250_000.0):
        for rate in (0.0, 0.01, 0.05, 0.12):
            for days in (1, 30, 365, 365 * 7 + 3):
                when = VAL + timedelta(days=days)
                t = days / 365.0  # ACT/365F, stated here rather than asked of the library
                want = notional * math.exp(-rate * t)
                got = _pv(_flow(notional, when), rate)
                assert got == pytest.approx(want, abs=1e-12, rel=1e-12), (notional, rate, days)


def test_a_zero_rate_leaves_the_amount_untouched() -> None:
    assert _pv(_flow(1_000.0, VAL + timedelta(days=500)), rate=0.0) == pytest.approx(1_000.0)


# ── oracle 2: analytic vs finite-difference DV01 ─────────────────────────────────────────────
def test_dv01_analytic_matches_finite_difference() -> None:
    """`∂PV/∂r = −t·N·e^{−rt}`, so DV01 (per 1bp) is `−t·PV·1e-4`. The analytic form and a
    bump-reprice are two different computations — their agreement is not tautological, which is
    what makes this an oracle rather than a restatement."""
    notional, rate, days = 1_000_000.0, 0.05, 365 * 3 + 1
    when = VAL + timedelta(days=days)
    t = days / 365.0
    analytic = -t * notional * math.exp(-rate * t) * 1e-4

    bump = 1e-6  # a small central bump, so truncation sits far below the 1e-6 tolerance
    up, down = _pv(_flow(notional, when), rate + bump), _pv(_flow(notional, when), rate - bump)
    fd = (up - down) / (2.0 * bump) * 1e-4

    assert fd == pytest.approx(analytic, rel=1e-6), (fd, analytic)


# ── oracle 3: the valuation-date boundary, where off-by-one lives ───────────────────────────
def test_a_flow_ON_the_valuation_date_is_EXCLUDED_from_pv() -> None:
    """Invariant: a cashflow on or before the valuation date is historical — the shell's business,
    never discounted with a non-positive `t`. The `date == valuation_date` case is the one that
    gets written wrong, so it is tested explicitly rather than by implication."""
    assert _pv(_flow(5_000.0, VAL)) == pytest.approx(0.0, abs=1e-12)
    assert _pv(_flow(5_000.0, VAL - timedelta(days=1))) == pytest.approx(0.0, abs=1e-12)


def test_a_flow_ONE_DAY_AFTER_the_valuation_date_is_INCLUDED() -> None:
    """The other side of the same boundary — without this, 'excluded' could be implemented as
    'always zero' and oracle 3 would still pass."""
    got = _pv(_flow(5_000.0, VAL + timedelta(days=1)))
    assert got == pytest.approx(5_000.0 * math.exp(-0.05 / 365.0), abs=1e-12)
    assert got > 0.0


# ── oracle 4: statelessness and immutability ────────────────────────────────────────────────
def test_repricing_is_byte_identical_and_the_snapshot_is_never_mutated() -> None:
    market = _market()
    product = _flow(1_000_000.0, VAL + timedelta(days=200))
    model = DiscountingModel(market)
    first, second = price(product, model), price(product, model)
    assert isinstance(first, PricingResult) and isinstance(second, PricingResult)
    assert first.pv.amount == second.pv.amount  # identical inputs, identical output, to the bit
    assert market.valuation_date == VAL
    assert model.market is market  # the model carries it; the engine never swaps it


# ── oracle 5: failure is a value ────────────────────────────────────────────────────────────
def test_a_missing_curve_is_a_FAILURE_not_an_exception_or_a_silent_zero() -> None:
    empty = MarketSnapshot(valuation_date=VAL, curves=CurveSet({}))
    out = price(_flow(1_000.0, VAL + timedelta(days=90)), DiscountingModel(empty))
    assert isinstance(out, PricingFailure)
    assert out.reason  # it says what went wrong


def test_an_unregistered_product_type_is_a_FAILURE() -> None:
    out = price(object(), DiscountingModel(_market()))
    assert isinstance(out, PricingFailure)


# ── oracle 6: currency mixing is an error, not a silent addition ────────────────────────────
def test_adding_money_in_different_currencies_is_refused() -> None:
    with pytest.raises(ValueError, match="currenc"):
        Money(1.0, USD) + Money(1.0, Currency("EUR"))


def test_pricing_in_a_currency_the_snapshot_does_not_carry_is_a_FAILURE() -> None:
    out = price(_flow(1_000.0, VAL + timedelta(days=90), Currency("JPY")), DiscountingModel(_market()))
    assert isinstance(out, PricingFailure)


# ── the representation contract (the scope line's durable public contract) ──────────────────
def test_a_product_round_trips_through_its_serialised_form() -> None:
    """The scope line makes *how a product is represented* the durable public contract, so the
    round trip is tested from the first product rather than retrofitted. A pricing server is the
    consumer that will prove it; this is the contract it will consume."""
    product = _flow(1_234.56, VAL + timedelta(days=77))
    assert FixedCashflow.from_dict(product.to_dict()) == product


def test_the_serialised_form_carries_a_schema_version() -> None:
    """Without a version the format cannot change without silently breaking a reader."""
    assert FixedCashflow(_flow(1.0, VAL).cashflow).to_dict()["schema_version"] >= 1
