"""FRA oracle (L4) — CP-2b #3, fixed_income spine to parity.

A forward rate agreement: pay fixed `K`, receive the simply-compounded forward
`L(T1,T2) = (P(0,T1)/P(0,T2) - 1)/tau` over `[T1, T2]`, settled at `T2`. Single-curve,
priced by composing the curve's discount factors (pricing lives in L4) —
    PV(pay-fixed) = notional · tau · (L - K) · P(0,T2).
Works on any curve, so this exercises the fixed-income spine on a real bootstrapped
curve, not just the flat skeleton.

Oracles: a par FRA (K = L) prices to zero; off-par matches the closed form; receiving
fixed flips the sign; and on a bootstrapped curve the implied forward reprices to par.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.cashflow import Accrual
from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.calibration.discount_curve import bootstrap_discount_curve
from pricebook_ng.engine.fra import FRAEngine
from pricebook_ng.market.discount_curve import DepositQuote
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.fra import ForwardRateAgreement

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
ACT360 = DC.ACT_360
USD = Currency.USD
NOTIONAL = 10_000_000.0
T1, T2 = date(2027, 1, 15), date(2027, 7, 15)  # forward-starting 6M FRA
NUM = NumericalConfig()
ENGINE = FRAEngine()


def _market(curve):
    return MarketSnapshot(valuation_date=D0, discount_curve=curve)


def _forward(curve):
    tau = year_fraction(T1, T2, ACT360)
    return (curve.df(T1) / curve.df(T2) - 1.0) / tau


def _fra(rate, pay_fixed=True):
    return ForwardRateAgreement(Money(NOTIONAL, USD), rate, Accrual(T1, T2, ACT360), pay_fixed)


def test_par_fra_prices_to_zero():
    curve = FlatDiscountCurve(0.03, D0, ACT365)
    model = DiscountingModel(_market(curve))
    par = _forward(curve)
    assert ENGINE.price(_fra(par), model, NUM).pv.amount == pytest.approx(0.0, abs=1e-8)


def test_off_par_matches_closed_form():
    curve = FlatDiscountCurve(0.03, D0, ACT365)
    model = DiscountingModel(_market(curve))
    fwd = _forward(curve)
    strike = fwd + 0.01
    tau = year_fraction(T1, T2, ACT360)
    expected = NOTIONAL * tau * (fwd - strike) * curve.df(T2)
    assert ENGINE.price(_fra(strike), model, NUM).pv.amount == pytest.approx(expected, abs=1e-6)


def test_receive_fixed_flips_sign():
    curve = FlatDiscountCurve(0.03, D0, ACT365)
    model = DiscountingModel(_market(curve))
    strike = _forward(curve) + 0.005
    pay = ENGINE.price(_fra(strike, pay_fixed=True), model, NUM).pv.amount
    receive = ENGINE.price(_fra(strike, pay_fixed=False), model, NUM).pv.amount
    assert pay == pytest.approx(-receive, abs=1e-9)


def test_par_reprices_to_zero_on_bootstrapped_curve():
    curve = bootstrap_discount_curve(
        D0,
        [DepositQuote(date(2027, 1, 15), 0.030, ACT360), DepositQuote(date(2028, 1, 15), 0.036, ACT360)],
        [],
    )
    model = DiscountingModel(_market(curve))
    assert ENGINE.price(_fra(_forward(curve)), model, NUM).pv.amount == pytest.approx(0.0, abs=1e-8)
