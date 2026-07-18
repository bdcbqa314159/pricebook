"""Settlement oracles (L0) — Topic 0 Slice 4b.

Completes the flow vocabulary: a physical `Delivery(date, Quantity)` alongside
`Cashflow(date, Money)`, cash/physical/auction settlement types, a settlement currency
that may differ from the contract currency (quanto/NDF), and settlement-date = trade
date + lag business days under a calendar (FX T+2). `AUCTION` is a marker only — the
CDS credit-event mechanics are the credit topic, not L0.
"""

from datetime import date

import pytest

from pricebook_ng.foundation.cashflow import Cashflow
from pricebook_ng.foundation.market_calendars import get_calendar
from pricebook_ng.foundation.money import Currency, Money, Quantity, Unit
from pricebook_ng.foundation.settlement import (
    Delivery,
    SettlementTerms,
    SettlementType,
    settlement_date,
)

NY = get_calendar("NEW_YORK_SIFMA")


# ── settlement date = trade + lag business days under a calendar ──
def test_fx_spot_t_plus_2_skips_weekend():
    fx = SettlementTerms(SettlementType.CASH, currency=Currency.USD, lag=2)
    # Fri 2024-06-07 + 2 business days = Tue 2024-06-11 (skips Sat/Sun)
    assert settlement_date(date(2024, 6, 7), fx, NY) == date(2024, 6, 11)


def test_lag_zero_is_same_day_when_business():
    terms = SettlementTerms(SettlementType.CASH, currency=Currency.EUR, lag=0)
    assert settlement_date(date(2024, 6, 10), terms, NY) == date(2024, 6, 10)


# ── cash vs physical produce different flow types ──
def test_physical_delivers_a_quantity_cash_pays_money():
    delivery = Delivery(date(2024, 6, 11), Quantity(1000.0, Unit.BARREL))
    cash = Cashflow(date(2024, 6, 11), Money(75_000.0, Currency.USD))
    assert isinstance(delivery.quantity, Quantity)
    assert isinstance(cash.amount, Money)
    assert type(delivery) is not type(cash)


# ── settlement currency ≠ contract currency (quanto / NDF) ──
def test_settlement_currency_can_differ_from_contract():
    # an NDF on USD/BRL settling in USD, though the notional leg is BRL
    ndf = SettlementTerms(SettlementType.CASH, currency=Currency.USD, lag=2)
    assert ndf.currency is Currency.USD
    assert ndf.settlement_type is SettlementType.CASH


def test_physical_terms_carry_no_settlement_currency():
    physical = SettlementTerms(SettlementType.PHYSICAL, currency=None, lag=2)
    assert physical.currency is None
    assert physical.settlement_type is SettlementType.PHYSICAL


# ── auction is a marker only (CDS credit event) ──
def test_auction_is_a_type_marker():
    cds = SettlementTerms(SettlementType.AUCTION, currency=Currency.USD, lag=5)
    assert cds.settlement_type is SettlementType.AUCTION


def test_cash_terms_require_a_currency():
    with pytest.raises(ValueError):
        SettlementTerms(SettlementType.CASH, currency=None, lag=2)  # cash must name a currency
