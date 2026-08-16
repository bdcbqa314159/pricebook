"""Optionlet products — pure data (L2).

A `Caplet` is a call on a forward rate: it pays `notional · τ · (F − strike)+` at the accrual
end, where `F` is the `index` forward over the accrual. Pure data — it does not price itself
(pricing is L4). `OptionType` (call/put) is the payoff parity a pricer reads; a caplet is a CALL
(a floorlet — the PUT — arrives with its consumer).

Provenance:
  quarry: python/pricebook/options/capfloor.py (caplet composition)
  source: CLAUDE.md §2 (products are pure data); redesign/12 §B5 (caps/floors)
  oracle: caplet reprices to Black-76 closed form through the L4 engine
  slice:  black-caplet (C2 slice 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from datetime import date

from pricebook_ng.foundation import Accrual, RateIndex
from pricebook_ng.products.swap import VanillaSwap


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class Caplet:
    """A caplet: a call on the `index` forward over `accrual`, struck at `strike`, on `notional`.
    Fixes at `accrual.start`, pays `notional · τ · (F − strike)+` at `accrual.end`."""

    index: RateIndex
    accrual: Accrual
    strike: float
    notional: float


@dataclass(frozen=True)
class Swaption:
    """A European swaption on `swap` (a `VanillaSwap`), exercisable at `expiry` into physical
    settlement. The strike IS `swap.fixed_leg.rate`; `option_type` is the parity — a PAYER
    swaption (option to pay fixed) is a CALL on the swap rate, a RECEIVER a PUT. Pure data."""

    swap: VanillaSwap
    expiry: date
    option_type: OptionType
