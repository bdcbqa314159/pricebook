"""S08 oracle — Hull-White European swaption via Jamshidian (L4).

A European swaption is decomposed (Jamshidian) into a portfolio of options on
zero-coupon bonds, priced with the S07 HW analytic ZCB-option formula. Closed-form
self-consistency oracles (no MC — that is the S09 cross-check):
  1. put-call parity: payer - receiver == P(0,T0)*notional - sum(cf_i * P(0,t_i));
  2. ATM symmetry: struck at the forward par swap rate, payer == receiver;
  3. sigma -> 0 collapses to the discounted intrinsic = max(forward swap PV, 0),
     cross-checked against the S06 SwapEngine.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingResult
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms, generate_schedule
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.engine.swap import SwapEngine
from pricebook_ng.engine.swaption import SwaptionEngine
from pricebook_ng.instruments.swap import SwapTerms, vanilla_swap
from pricebook_ng.instruments.swaption import Swaption
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.hull_white import HullWhite

ABS = 1e-8
D0 = date(2026, 1, 5)
EXPIRY = date(2027, 1, 5)      # 1Y into a ...
MATURITY = date(2032, 1, 5)    # ... 5Y swap
NOTIONAL = 1_000_000.0
CCY = Currency.USD
FIXED = ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360)
FLOAT = ScheduleTerms(Frequency.ANNUAL, DC.ACT_360)


def _curve(rate=0.03):
    return FlatDiscountCurve(rate=rate, anchor=D0, day_count=DC.ACT_365_FIXED)


def _market(rate=0.03):
    return MarketSnapshot(valuation_date=D0, discount_curve=_curve(rate))


def _swaption(fixed_rate, pay_fixed=True):
    swap = vanilla_swap(
        face=Money(NOTIONAL, CCY), fixed_rate=fixed_rate, start=EXPIRY, maturity=MATURITY,
        terms=SwapTerms(fixed_schedule=FIXED, float_schedule=FLOAT, pay_fixed=pay_fixed),
    )
    return Swaption(expiry=EXPIRY, swap=swap), swap


def _forward_par_rate(curve):
    sched = generate_schedule(EXPIRY, MATURITY, FIXED.frequency, FIXED.roll)
    annuity = sum(
        year_fraction(sched[i - 1], sched[i], FIXED.day_count) * curve.df(sched[i])
        for i in range(1, len(sched))
    )
    return (curve.df(EXPIRY) - curve.df(MATURITY)) / annuity


def _price(swaption, hw, market, sigma=0.01):
    return SwaptionEngine().price(swaption, hw, market, NumericalConfig())


def test_put_call_parity():
    market, curve = _market(), _curve()
    hw = HullWhite(a=0.05, sigma=0.012, curve=curve)
    payer, swap = _swaption(0.035, pay_fixed=True)
    receiver, _ = _swaption(0.035, pay_fixed=False)
    p = SwaptionEngine().price(payer, hw, market, NumericalConfig())
    r = SwaptionEngine().price(receiver, hw, market, NumericalConfig())
    assert isinstance(p, PricingResult)

    # coupon-bond cashflows: fixed amounts, notional added to the last date
    amounts = [(cf.date, cf.amount.amount) for cf in swap.fixed_leg.cashflows]
    amounts[-1] = (amounts[-1][0], amounts[-1][1] + NOTIONAL)
    rhs = curve.df(EXPIRY) * NOTIONAL - sum(a * curve.df(d) for d, a in amounts)
    assert (p.pv.amount - r.pv.amount) == pytest.approx(rhs, abs=1e-4)


def test_atm_payer_equals_receiver():
    market, curve = _market(), _curve()
    hw = HullWhite(a=0.05, sigma=0.012, curve=curve)
    par = _forward_par_rate(curve)
    payer, _ = _swaption(par, pay_fixed=True)
    receiver, _ = _swaption(par, pay_fixed=False)
    p = SwaptionEngine().price(payer, hw, market, NumericalConfig()).pv.amount
    r = SwaptionEngine().price(receiver, hw, market, NumericalConfig()).pv.amount
    assert p == pytest.approx(r, abs=1e-4)


def test_zero_vol_is_discounted_intrinsic():
    market, curve = _market(), _curve()
    hw = HullWhite(a=0.05, sigma=0.0, curve=curve)
    payer, swap = _swaption(0.02, pay_fixed=True)  # in-the-money payer (low fixed)
    swaption_pv = SwaptionEngine().price(payer, hw, market, NumericalConfig()).pv.amount
    fwd_swap_pv = SwapEngine().price(swap, None, market, NumericalConfig()).pv.amount
    assert swaption_pv == pytest.approx(max(fwd_swap_pv, 0.0), abs=1e-4)


def test_out_of_the_money_zero_vol_is_zero():
    market, curve = _market(), _curve()
    hw = HullWhite(a=0.05, sigma=0.0, curve=curve)
    payer, _ = _swaption(0.06, pay_fixed=True)  # high fixed -> payer worthless
    assert SwaptionEngine().price(payer, hw, market, NumericalConfig()).pv.amount == pytest.approx(
        0.0, abs=1e-4
    )
