"""Serialisation-pattern oracle (L0) — Topic 0 gate Slice 8.

Demonstrates the per-class `to_dict`/`from_dict` + `schema_version` pattern on a **hard
case** — a `Leg`, which exercises all three dimensions at once: a **collection** (its flows),
an **enum** (`DayCountConvention`), and **nested value objects** (`Money`/`Quantity`/`Accrual`),
plus a `Cashflow | Delivery` **union** carried by a `kind` discriminator. `Schedule`/`FixingHistory`
were the suggested collections, but neither carries an enum or a nested value object — `Leg` does,
so it is the honest demonstration. The round-trip is JSON-clean (only str/float/list/dict/None).
"""

import json
from datetime import date

import pytest

from pricebook_ng.foundation.cashflow import SCHEMA_VERSION, Accrual, Cashflow, Leg
from pricebook_ng.foundation.day_count import DayCountConvention as DC
from pricebook_ng.foundation.money import Currency, Money, Quantity, Unit
from pricebook_ng.foundation.settlement import Delivery


def _leg() -> Leg:
    coupon = Cashflow(
        date(2024, 7, 1),
        Money(-1_000.0, Currency.USD),                                   # signed: paying = negative
        accrual=Accrual(date(2024, 1, 1), date(2024, 7, 1), DC.THIRTY_360),
    )
    physical = Delivery(date(2024, 7, 1), Quantity(10.0, Unit.BARREL))
    bullet = Cashflow(date(2024, 12, 31), Money(1_000.0, Currency.USD))  # no accrual
    return Leg(flows=(coupon, physical, bullet), day_count=DC.THIRTY_360)


def test_leg_round_trips_through_dict():
    leg = _leg()
    assert Leg.from_dict(leg.to_dict()) == leg           # nested + enum + collection + union all survive


def test_leg_dict_is_json_clean():
    d = _leg().to_dict()
    assert json.loads(json.dumps(d)) == d                # only JSON primitives
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["day_count"] == DC.THIRTY_360.value
    assert {f["kind"] for f in d["flows"]} == {"cash", "delivery"}


def test_nested_value_objects_encode_by_identity():
    leg = _leg()
    d = leg.to_dict()
    cash = next(f for f in d["flows"] if f["kind"] == "cash")
    assert cash["amount"] == {"amount": -1_000.0, "currency": "USD"}     # Money by currency code
    delivery = next(f for f in d["flows"] if f["kind"] == "delivery")
    assert delivery["quantity"] == {"amount": 10.0, "unit": "bbl"}       # Quantity by unit symbol


def test_bullet_cashflow_accrual_is_none():
    leg = _leg()
    d = leg.to_dict()
    bullet = [f for f in d["flows"] if f["kind"] == "cash"][1]
    assert bullet["accrual"] is None
    assert Leg.from_dict(d).flows[2].accrual is None


def test_unknown_schema_version_rejected():
    d = _leg().to_dict()
    d["schema_version"] = 999
    with pytest.raises(ValueError):
        Leg.from_dict(d)


def test_money_and_accrual_round_trip_standalone():
    m = Money(2_500.0, Currency.JPY)
    assert Money.from_dict(m.to_dict()) == m
    a = Accrual(date(2024, 2, 15), date(2024, 8, 15), DC.ACT_ACT_ICMA)
    assert Accrual.from_dict(a.to_dict()) == a
