"""Deposit oracle (L4) — CP-2c #3, fixed_income spine.

A money-market deposit: place `notional` at `start`, receive `notional·(1 + rate·τ)`
at `maturity`. Modelled as its two cashflows (`−N` at start, `+N(1+rτ)` at maturity)
priced by the existing `DiscountingEngine` — no new engine. A2 reconciles the views:
  - forward-starting (start > valuation): both legs count, so a par deposit reprices to
    zero (self-consistency with the curve);
  - spot (start = valuation): the principal-today is realized cash (excluded from the
    mark, A2/A3), so the mark is the redemption value — `N` at par.

Oracles: forward par → 0; spot par → the principal; off-par matches the closed form;
par reprices to 0 on a bootstrapped curve.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.calibration.discount_curve import bootstrap_discount_curve
from pricebook_ng.engine.discounting import DiscountingEngine
from pricebook_ng.market.discount_curve import DepositQuote
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.deposit import Deposit, deposit

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
ACT360 = DC.ACT_360
USD = Currency.USD
NOTIONAL = 10_000_000.0
NUM = NumericalConfig()
ENGINE = DiscountingEngine()


def _market(curve):
    return MarketSnapshot(valuation_date=D0, discount_curve=curve)


def _par_rate(curve, start, maturity):
    tau = year_fraction(start, maturity, ACT360)
    return (curve.df(start) / curve.df(maturity) - 1.0) / tau


def _price(dep, curve):
    return ENGINE.price(dep, DiscountingModel(_market(curve)), NUM).pv.amount


def test_forward_par_deposit_prices_to_zero():
    curve = FlatDiscountCurve(0.03, D0, ACT365)
    start, maturity = date(2027, 1, 15), date(2027, 7, 15)
    dep = deposit(Money(NOTIONAL, USD), _par_rate(curve, start, maturity), start, maturity, ACT360)
    assert _price(dep, curve) == pytest.approx(0.0, abs=1e-6)


def test_spot_par_deposit_marks_to_the_principal():
    curve = FlatDiscountCurve(0.03, D0, ACT365)
    maturity = date(2026, 7, 15)
    dep = deposit(Money(NOTIONAL, USD), _par_rate(curve, D0, maturity), D0, maturity, ACT360)
    # spot: principal-today is realized (A2), so the mark is the redemption value == N at par
    assert _price(dep, curve) == pytest.approx(NOTIONAL, abs=1e-4)


def test_off_par_forward_matches_closed_form():
    curve = FlatDiscountCurve(0.03, D0, ACT365)
    start, maturity = date(2027, 1, 15), date(2027, 7, 15)
    rate = _par_rate(curve, start, maturity) + 0.01
    tau = year_fraction(start, maturity, ACT360)
    expected = -NOTIONAL * curve.df(start) + NOTIONAL * (1.0 + rate * tau) * curve.df(maturity)
    dep = deposit(Money(NOTIONAL, USD), rate, start, maturity, ACT360)
    assert _price(dep, curve) == pytest.approx(expected, abs=1e-4)


def test_par_reprices_to_zero_on_bootstrapped_curve():
    curve = bootstrap_discount_curve(
        D0,
        [DepositQuote(date(2027, 1, 15), 0.030, ACT360), DepositQuote(date(2028, 1, 15), 0.036, ACT360)],
        [],
    )
    start, maturity = date(2027, 1, 15), date(2027, 7, 15)
    dep = deposit(Money(NOTIONAL, USD), _par_rate(curve, start, maturity), start, maturity, ACT360)
    assert _price(dep, curve) == pytest.approx(0.0, abs=1e-6)


# ── serialisation (CP-3 #2 — the genuine residual that retires quarry deposit) ──


def test_round_trips_through_dict():
    dep = deposit(Money(NOTIONAL, USD), 0.05, date(2027, 1, 15), date(2027, 7, 15), ACT360)
    assert Deposit.from_dict(dep.to_dict()) == dep


def test_to_dict_carries_schema_version():
    dep = deposit(Money(1.0, USD), 0.03, date(2027, 1, 15), date(2027, 4, 15), ACT360)
    assert dep.to_dict()["schema_version"] == 1


def test_missing_version_reads_as_v1():
    dep = deposit(Money(1.0, USD), 0.03, date(2027, 1, 15), date(2027, 4, 15), ACT360)
    data = dep.to_dict()
    del data["schema_version"]  # legacy payload
    assert Deposit.from_dict(data) == dep


def test_future_version_is_rejected_loudly():
    dep = deposit(Money(1.0, USD), 0.03, date(2027, 1, 15), date(2027, 4, 15), ACT360)
    data = dep.to_dict()
    data["schema_version"] = 99
    with pytest.raises(ValueError):
        Deposit.from_dict(data)
