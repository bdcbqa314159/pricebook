"""Cash / linear-rates products — Deposit, FRA, Future (L2, pure data).

Frozen descriptions, no behaviour (pricing lives in L4, dispatched by the engine registry).
A `Deposit` discounts on its currency's discount curve; an `FRA`/`Future` project a forward
off the curve keyed by their `index`. A `Future` is IMM-dated and quoted as a `price` (its
forward rate is `1 − price`; convexity deferred to the models topic).

Provenance:
  quarry: python/pricebook/fixed_income/ (deposit / FRA / future)
  source: doc 18 §1 (pillar instruments); §2 (futures as forwards); CLAUDE.md §2 (pure data)
  oracle: each reprices to zero through the L4 engine off the built curves
  slice:  cash-instruments (T1 slice 3, Step B)
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook_ng.foundation import Accrual, Currency, RateIndex


@dataclass(frozen=True)
class Deposit:
    """A cash deposit: place `notional` over `accrual`, earn simple `rate`."""

    notional: float
    currency: Currency
    accrual: Accrual
    rate: float


@dataclass(frozen=True)
class FRA:
    """A forward-rate agreement: exchange fixed `rate` for the `index` forward over `accrual`."""

    notional: float
    currency: Currency
    accrual: Accrual
    index: RateIndex
    rate: float


@dataclass(frozen=True)
class Future:
    """An IMM-dated interest-rate future, quoted as `price` (forward rate = 1 − price;
    convexity deferred). Projects the `index` forward over its IMM `accrual`."""

    notional: float
    currency: Currency
    accrual: Accrual
    index: RateIndex
    price: float
