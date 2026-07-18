"""Seasoned FRA oracle (L4) — CP-2c #2, fixings / seasoned-float.

When a FRA's accrual period has already started (`accrual.start < valuation`), its
float rate was fixed at the reset — a realized value, looked up in the snapshot's
`FixingHistory`, not a curve forward (A2: reset dates ≤ valuation use fixings). The
payoff is then a known cashflow, `face·τ·(fixing − K)·P(0,end)`, still live until it
settles at `end`. This is the first `FixingHistory`-consuming engine; the swap float
leg and the L6 float realized P&L follow the same pattern.

Oracles: a seasoned FRA with the fixing present prices to the deterministic payoff;
at K = fixing it is zero; a missing fixing is a `PricingFailure`; a fully-paid FRA
(`end ≤ valuation`) is zero; and a forward-starting FRA is unchanged.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.cashflow import Accrual
from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.fra import FRAEngine
from pricebook_ng.market.snapshot import FixingHistory, FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.fra import ForwardRateAgreement

D0 = date(2026, 1, 15)
ACT365 = DC.ACT_365_FIXED
ACT360 = DC.ACT_360
USD = Currency.USD
NOTIONAL = 10_000_000.0
RESET = date(2025, 7, 15)       # < valuation: the period has started, rate fixed here
END = date(2026, 7, 15)         # > valuation: not yet settled
FIXING = 0.035
NUM = NumericalConfig()
ENGINE = FRAEngine()


def _market(fixings=None, rate=0.03):
    return MarketSnapshot(
        valuation_date=D0, discount_curve=FlatDiscountCurve(rate, D0, ACT365),
        fixings=FixingHistory(fixings or {}),
    )


def _fra(fixed_rate, start=RESET, end=END, pay_fixed=True):
    return ForwardRateAgreement(Money(NOTIONAL, USD), fixed_rate, Accrual(start, end, ACT360), pay_fixed)


def test_seasoned_fra_uses_the_fixing():
    market = _market({RESET: FIXING})
    strike = 0.030
    tau = year_fraction(RESET, END, ACT360)
    expected = NOTIONAL * tau * (FIXING - strike) * market.discount_curve.df(END)
    assert ENGINE.price(_fra(strike), DiscountingModel(market), NUM).pv.amount == pytest.approx(
        expected, abs=1e-6
    )


def test_seasoned_fra_at_the_fixing_is_zero():
    market = _market({RESET: FIXING})
    assert ENGINE.price(_fra(FIXING), DiscountingModel(market), NUM).pv.amount == pytest.approx(
        0.0, abs=1e-8
    )


def test_missing_fixing_is_a_failure():
    from pricebook_ng.foundation.results import PricingFailure

    market = _market({})  # no fixing at RESET
    assert isinstance(ENGINE.price(_fra(0.03), DiscountingModel(market), NUM), PricingFailure)


def test_fully_paid_fra_is_zero():
    market = _market({RESET: FIXING})
    paid = _fra(0.03, start=date(2024, 7, 15), end=date(2025, 1, 15))  # end < valuation
    assert ENGINE.price(paid, DiscountingModel(market), NUM).pv.amount == pytest.approx(0.0, abs=1e-12)


def test_forward_fra_still_prices_off_the_curve():
    market = _market({})  # no fixings needed — forward-starting
    t1, t2 = date(2027, 1, 15), date(2027, 7, 15)
    curve = market.discount_curve
    tau = year_fraction(t1, t2, ACT360)
    par = (curve.df(t1) / curve.df(t2) - 1.0) / tau
    assert ENGINE.price(_fra(par, start=t1, end=t2), DiscountingModel(market), NUM).pv.amount == pytest.approx(
        0.0, abs=1e-8
    )
