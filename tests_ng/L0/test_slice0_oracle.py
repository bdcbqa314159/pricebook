"""Slice 0 oracle — the walking skeleton, priced end-to-end L0->L6.

A single fixed cashflow discounted on a flat, continuously-compounded curve.
Closed form is exact to machine precision, so the oracle is a closed-form
red/green check (redesign/04_slice_plan.md):

    t   = year_fraction(valuation, T, ACT/365F)
    DF  = exp(-r * t)
    PV  = notional * DF

Three oracles: price (closed form), risk (analytic vs finite-difference DV01),
statelessness (repricing is byte-identical).
"""

import math
from datetime import date

from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.foundation.cashflow import Cashflow
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingResult
from pricebook_ng.market.snapshot import FlatDiscountCurve, MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.instruments.fixed_cashflow import FixedCashflowTrade
from pricebook_ng.engine.discounting import DiscountingEngine
from pricebook_ng.risk.dv01 import dv01
from pricebook_ng.shell.booking import book

# --- the fixture trade & market ------------------------------------------------
VALUATION = date(2026, 1, 1)
MATURITY = date(2028, 1, 1)
RATE = 0.03
NOTIONAL = 1_000_000.0
CCY = Currency.USD
DC = DayCountConvention.ACT_365_FIXED


def _setup():
    curve = FlatDiscountCurve(rate=RATE, anchor=VALUATION, day_count=DC)
    market = MarketSnapshot(valuation_date=VALUATION, discount_curve=curve)
    trade = FixedCashflowTrade(
        Cashflow(date=MATURITY, amount=Money(NOTIONAL, CCY))
    )
    return trade, market, NumericalConfig(), DiscountingEngine()


def _expected_pv() -> float:
    t = year_fraction(VALUATION, MATURITY, DC)
    return NOTIONAL * math.exp(-RATE * t)


def test_price_oracle_closed_form():
    trade, market, numerics, engine = _setup()
    result = engine.price(trade, DiscountingModel(market), numerics)
    assert isinstance(result, PricingResult)
    assert result.pv.currency is CCY
    assert abs(result.pv.amount - _expected_pv()) < 1e-12


def test_risk_oracle_dv01_analytic_vs_fd():
    trade, market, numerics, engine = _setup()
    t = year_fraction(VALUATION, MATURITY, DC)
    analytic = -NOTIONAL * t * math.exp(-RATE * t) * 1e-4
    fd = dv01(engine, trade, DiscountingModel(market), numerics)
    assert abs(fd - analytic) < 1e-6


def test_statelessness_reprice_byte_identical():
    trade, market, numerics, engine = _setup()
    first = engine.price(trade, DiscountingModel(market), numerics).pv.amount
    # a risk bump reprices under mutated snapshots; the original must be untouched
    dv01(engine, trade, DiscountingModel(market), numerics)
    second = engine.price(trade, DiscountingModel(market), numerics).pv.amount
    assert first == second  # same-process reproducibility: identical bits
    assert first.hex() == second.hex()


def test_shell_path_matches_core():
    trade, market, numerics, engine = _setup()
    booked = book(trade)
    result = booked.value(market, numerics, engine)
    assert isinstance(result, PricingResult)
    assert result.pv.amount == engine.price(trade, DiscountingModel(market), numerics).pv.amount
    # the shell stored the result (it remembers; it never re-computes price)
    assert booked.results[-1] is result
