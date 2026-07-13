"""S09 oracle — Hull-White swaption by Monte Carlo, vs the Jamshidian analytic (L4).

The MC engine simulates the short rate at expiry under the T0-forward measure
(one exact Gaussian draw of x(T0)), reconstitutes the coupon-bond value, and
averages the swaption payoff. Oracle: it converges to the S08 analytic price
(the named "analytic vs MC convergence" check for the whole HW arc); is exact at
sigma=0 (deterministic); and is reproducible under a fixed seed.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.schedule import Frequency, ScheduleTerms
from pricebook_ng.foundation.time import DayCountConvention as DC
from pricebook_ng.engine.swaption import SwaptionEngine
from pricebook_ng.engine.swaption_mc import SwaptionMCEngine
from pricebook_ng.products.swap import SwapTerms, vanilla_swap
from pricebook_ng.products.swaption import Swaption
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.hull_white import HullWhite

D0 = date(2026, 1, 5)
EXPIRY = date(2027, 1, 5)
MATURITY = date(2032, 1, 5)
NOTIONAL = 1_000_000.0
CCY = Currency.USD


def _market(rate=0.03):
    return MarketSnapshot(valuation_date=D0, discount_curve=FlatDiscountCurve(rate, D0, DC.ACT_365_FIXED))


def _swaption(fixed_rate=0.035, pay_fixed=True):
    swap = vanilla_swap(
        face=Money(NOTIONAL, CCY), fixed_rate=fixed_rate, start=EXPIRY, maturity=MATURITY,
        terms=SwapTerms(ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
                        ScheduleTerms(Frequency.ANNUAL, DC.ACT_360), pay_fixed),
    )
    return Swaption(expiry=EXPIRY, swap=swap)


def test_mc_converges_to_analytic():
    hw = HullWhite(a=0.05, sigma=0.012, market=_market())
    swaption = _swaption()
    analytic = SwaptionEngine().price(swaption, hw, NumericalConfig()).pv.amount
    mc = SwaptionMCEngine().price(swaption, hw, NumericalConfig(mc_paths=200_000, mc_seed=42)).pv.amount
    assert mc == pytest.approx(analytic, rel=0.02)  # within ~2% at 200k seeded paths


def test_mc_receiver_converges():
    hw = HullWhite(a=0.05, sigma=0.012, market=_market())
    swaption = _swaption(pay_fixed=False)
    analytic = SwaptionEngine().price(swaption, hw, NumericalConfig()).pv.amount
    mc = SwaptionMCEngine().price(swaption, hw, NumericalConfig(mc_paths=200_000, mc_seed=7)).pv.amount
    assert mc == pytest.approx(analytic, rel=0.02)


def test_mc_exact_at_zero_vol():
    hw = HullWhite(a=0.05, sigma=0.0, market=_market())
    swaption = _swaption(fixed_rate=0.02)  # ITM payer
    analytic = SwaptionEngine().price(swaption, hw, NumericalConfig()).pv.amount
    mc = SwaptionMCEngine().price(swaption, hw, NumericalConfig(mc_paths=1000, mc_seed=1)).pv.amount
    assert mc == pytest.approx(analytic, abs=1e-6)  # deterministic, no variance


def test_mc_is_reproducible():
    hw = HullWhite(a=0.05, sigma=0.012, market=_market())
    swaption = _swaption()
    num = NumericalConfig(mc_paths=10_000, mc_seed=99)
    a = SwaptionMCEngine().price(swaption, hw, num).pv.amount
    b = SwaptionMCEngine().price(swaption, hw, num).pv.amount
    assert a == b  # same seed -> identical (referential transparency)


def test_numerical_config_mc_knobs():
    num = NumericalConfig(mc_paths=5000, mc_seed=3)
    assert num.mc_paths == 5000
    assert num.mc_seed == 3
    with pytest.raises(ValueError):
        NumericalConfig(mc_paths=0)
