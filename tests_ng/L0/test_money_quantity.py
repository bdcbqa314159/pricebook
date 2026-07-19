"""Money / Quantity / Cashflow / Leg / Accrual oracles (L0) — Topic 0 Slice 4.

Currency-mixing is a type error; commodity `Quantity` arithmetic is closed under the
same unit only; `Accrual.year_fraction` is the ergonomic wrapper over the Slice-2
`year_fraction` primitive (bundling start+end+day_count). Serialisation is deferred to
Slice 6.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.cashflow import Cashflow, Leg
from pricebook_ng.foundation.day_count import Accrual, CouponPeriod
from pricebook_ng.foundation.day_count import DayCountConvention as DC
from pricebook_ng.foundation.day_count import year_fraction
from pricebook_ng.foundation.money import (
    Currency,
    CurrencyPair,
    Money,
    Quantity,
    Unit,
    currency,
    list_currencies,
    register_currency,
)

USD, EUR = Currency.USD, Currency.EUR


# ── Money: currency-mixing is a type error ──
def test_money_same_currency_arithmetic():
    assert Money(2.0, USD) + Money(3.0, USD) == Money(5.0, USD)
    assert Money(5.0, USD) - Money(2.0, USD) == Money(3.0, USD)
    assert -Money(1.0, USD) == Money(-1.0, USD)
    assert Money(2.0, USD) * 3 == Money(6.0, USD)


def test_money_mixing_currencies_raises():
    with pytest.raises(TypeError):
        Money(1.0, USD) + Money(1.0, EUR)
    with pytest.raises(TypeError):
        Money(1.0, USD) - Money(1.0, EUR)


def test_money_currency_must_be_a_currency():
    with pytest.raises(TypeError):
        Money(1.0, "USD")


def test_all_37_currencies_declared():
    # matches the 37 market calendars (Slice 1)
    assert len(list_currencies()) == 37


# ── S1: Currency is an OPEN registry (+ minor_units) ──
def test_currency_minor_units():
    assert Currency.USD.minor_units == 2
    assert Currency.JPY.minor_units == 0        # zero-decimal — matters at settlement rounding
    assert currency("USD") is Currency.USD       # interned lookup


def test_currency_registry_is_open():
    # a new market is a DECLARATION, not an L0 enum edit
    kwd = register_currency("KWD", 3)            # Kuwaiti dinar — 3 minor units
    assert currency("KWD") is kwd
    assert kwd.minor_units == 3


def test_brazilian_instrument_end_to_end():
    # the S1 gate regression: a BRL instrument is constructible — Currency · BUS/252 · São Paulo
    from pricebook_ng.foundation.market_calendars import get_calendar
    sp = get_calendar("SAO_PAULO")
    accrual = Accrual(date(2024, 6, 10), date(2024, 6, 14), DC.BUS_252)
    coupon = Cashflow(date(2024, 6, 14), Money(1_000_000.0, Currency.BRL), accrual=accrual)
    assert coupon.amount.currency.code == "BRL"
    assert year_fraction(accrual.start, accrual.end, DC.BUS_252, calendar=sp) > 0


# ── Quantity: closed under same-unit only ──
def test_quantity_same_unit_arithmetic():
    assert Quantity(1.0, Unit.BARREL) + Quantity(2.0, Unit.BARREL) == Quantity(3.0, Unit.BARREL)
    assert Quantity(5.0, Unit.MWH) * 2 == Quantity(10.0, Unit.MWH)


def test_quantity_mixing_units_raises():
    with pytest.raises(TypeError):
        Quantity(1.0, Unit.BARREL) + Quantity(1.0, Unit.MWH)


# ── CurrencyPair ──
def test_currency_pair_name_and_spot_lag():
    eurusd = CurrencyPair(EUR, USD)
    assert eurusd.name == "EURUSD"
    assert eurusd.spot_lag == 2                      # T+2 is the market default
    usdcad = CurrencyPair(USD, Currency.CAD, spot_lag=1)
    assert usdcad.name == "USDCAD"
    assert usdcad.spot_lag == 1


# ── Accrual.year_fraction — ergonomic wrapper over the S2 primitive ──
def test_accrual_year_fraction_matches_primitive():
    acc = Accrual(date(2024, 1, 1), date(2024, 7, 1), DC.ACT_360)
    assert acc.year_fraction() == year_fraction(date(2024, 1, 1), date(2024, 7, 1), DC.ACT_360)


def test_accrual_year_fraction_threads_icma_anchors():
    rs, re = date(2024, 2, 15), date(2024, 8, 15)
    acc = Accrual(rs, re, DC.ACT_ACT_ICMA)
    assert acc.year_fraction(coupon_period=CouponPeriod(rs, re, 2)) == 0.5
    with pytest.raises(ValueError):
        acc.year_fraction()                          # ICMA strict — anchors required


# ── Cashflow & Leg ──
def test_cashflow_and_leg():
    cf = Cashflow(date(2024, 7, 1), Money(1_000.0, USD),
                  accrual=Accrual(date(2024, 1, 1), date(2024, 7, 1), DC.THIRTY_360))
    leg = Leg(flows=(cf,), day_count=DC.THIRTY_360)
    assert leg.flows[0].amount == Money(1_000.0, USD)
    assert cf.accrual.year_fraction() == 0.5


def test_leg_holds_cashflows_and_deliveries():
    # S2: a leg is a run of flows, cash OR physical — a commodity/physical leg is expressible
    from pricebook_ng.foundation.settlement import Delivery
    cash = Cashflow(date(2024, 7, 1), Money(-1_000.0, USD))          # S13: paying = negative
    delivery = Delivery(date(2024, 7, 1), Quantity(10.0, Unit.BARREL))
    leg = Leg(flows=(cash, delivery), day_count=DC.ACT_360)
    assert leg.flows[0].amount.amount == -1_000.0                    # signed direction, no flag
    assert leg.flows[1].quantity == Quantity(10.0, Unit.BARREL)


def test_degenerate_periods_raise():
    # S14: an accrual is ordered by construction; reversed / zero-length raise
    with pytest.raises(ValueError):
        Accrual(date(2024, 7, 1), date(2024, 1, 1), DC.ACT_360)     # reversed
    with pytest.raises(ValueError):
        Accrual(date(2024, 1, 1), date(2024, 1, 1), DC.ACT_360)     # zero-length
    from pricebook_ng.foundation.day_count import year_fraction
    with pytest.raises(ValueError):
        year_fraction(date(2024, 7, 1), date(2024, 1, 1), DC.ACT_360)  # reversed primitive
