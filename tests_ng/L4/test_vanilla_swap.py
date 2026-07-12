"""S06 oracle — vanilla single-curve interest-rate swap (L4).

Fixed leg: known coupons discounted on the curve. Float leg: coupons are the
curve's forwards, computed at pricing time by the SwapEngine (not stored on the
instrument). Single-curve (discount = projection), no basis spread.

Oracles (all self-consistency, exact):
  1. a swap struck at the curve's par rate reprices to zero NPV;
  2. the float leg PV telescopes to notional * (DF(start) - DF(maturity));
  3. an off-par NPV equals notional * annuity * (par_rate - contract_rate);
  4. receiver = -payer; and a swap matching an S03 bootstrap input reprices to ~0.
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
from pricebook_ng.instruments.swap import SwapTerms, VanillaSwap, vanilla_swap
from pricebook_ng.market.discount_curve import ParSwapQuote, bootstrap_discount_curve
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel

ABS = 1e-6                      # notional 1e6 -> relative ~1e-12
START = date(2026, 1, 15)
MATURITY = date(2031, 1, 15)    # 5Y
NOTIONAL = 1_000_000.0
CCY = Currency.USD
FIXED = ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360)      # fixed: annual 30/360
FLOAT = ScheduleTerms(Frequency.SEMI_ANNUAL, DC.ACT_360)    # float: semi-annual ACT/360


def _flat_market(rate=0.03):
    curve = FlatDiscountCurve(rate=rate, anchor=START, day_count=DC.ACT_365_FIXED)
    return MarketSnapshot(valuation_date=START, discount_curve=curve)


def _par_rate(market, start=START, maturity=MATURITY, fixed=FIXED):
    """Par fixed rate = (DF(start) - DF(maturity)) / fixed-leg annuity."""
    df = market.discount_curve.df
    sched = generate_schedule(start, maturity, fixed.frequency, fixed.roll)
    annuity = sum(
        year_fraction(sched[i - 1], sched[i], fixed.day_count) * df(sched[i])
        for i in range(1, len(sched))
    )
    return (df(start) - df(maturity)) / annuity, annuity


def _swap(rate, pay_fixed=True):
    return vanilla_swap(
        face=Money(NOTIONAL, CCY), fixed_rate=rate, start=START, maturity=MATURITY,
        terms=SwapTerms(fixed_schedule=FIXED, float_schedule=FLOAT, pay_fixed=pay_fixed),
    )


def test_par_swap_reprices_to_zero():
    market = _flat_market()
    par, _ = _par_rate(market)
    result = SwapEngine().price(_swap(par), DiscountingModel(market), NumericalConfig())
    assert isinstance(result, PricingResult)
    assert result.pv.currency is CCY
    assert result.pv.amount == pytest.approx(0.0, abs=ABS)


def test_float_leg_telescopes():
    market = _flat_market()
    swap = _swap(0.0)  # zero fixed rate -> payer NPV = float PV
    df = market.discount_curve.df
    expected_float = NOTIONAL * (df(START) - df(MATURITY))
    npv = SwapEngine().price(swap, DiscountingModel(market), NumericalConfig()).pv.amount
    assert npv == pytest.approx(expected_float, abs=ABS)


def test_off_par_npv_is_annuity_times_rate_gap():
    market = _flat_market()
    par, annuity = _par_rate(market)
    contract = par - 0.005  # 50bp below par -> payer swap in the money
    npv = SwapEngine().price(_swap(contract), DiscountingModel(market), NumericalConfig()).pv.amount
    assert npv == pytest.approx(NOTIONAL * annuity * (par - contract), abs=ABS)


def test_receiver_is_negative_payer():
    market = _flat_market()
    par, _ = _par_rate(market)
    contract = par - 0.005
    payer = SwapEngine().price(_swap(contract, pay_fixed=True), DiscountingModel(market), NumericalConfig())
    recv = SwapEngine().price(_swap(contract, pay_fixed=False), DiscountingModel(market), NumericalConfig())
    assert recv.pv.amount == pytest.approx(-payer.pv.amount, abs=ABS)


def test_is_pure_data_no_pricing():
    swap = _swap(0.03)
    assert isinstance(swap, VanillaSwap)
    assert not hasattr(swap, "pv")


def test_swap_matching_bootstrap_input_reprices_near_zero():
    # A swap identical to an S03 bootstrap input must reprice to ~0 on that curve.
    curve = bootstrap_discount_curve(
        START,
        [],
        [ParSwapQuote(date(2027, 1, 15), 0.030, Frequency.ANNUAL, DC.THIRTY_360),
         ParSwapQuote(date(2028, 1, 15), 0.032, Frequency.ANNUAL, DC.THIRTY_360)],
    )
    market = MarketSnapshot(valuation_date=START, discount_curve=curve)
    swap = vanilla_swap(
        face=Money(NOTIONAL, CCY), fixed_rate=0.032, start=START, maturity=date(2028, 1, 15),
        terms=SwapTerms(
            fixed_schedule=ScheduleTerms(Frequency.ANNUAL, DC.THIRTY_360),
            float_schedule=ScheduleTerms(Frequency.ANNUAL, DC.ACT_360),
        ),
    )
    npv = SwapEngine().price(swap, DiscountingModel(market), NumericalConfig()).pv.amount
    assert npv == pytest.approx(0.0, abs=ABS)
