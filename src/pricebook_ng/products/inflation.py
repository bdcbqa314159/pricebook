"""Zero-coupon inflation swap as pure data (L2).

A ZCIS on a price `index`: at maturity one leg pays realized inflation
`notional*(I(T) - 1)`, the other a fixed compounded rate `notional*((1+K)^T - 1)`.
`receive_inflation=True` receives the inflation leg and pays fixed. Pure data — no
`pv` method (CLAUDE.md 2); pricing is L4.

Provenance:
  quarry: python/pricebook/inflation/ (ZCIS)
  source: zero-coupon inflation swap; Fisher relation
  oracle: par ZCIS reprices to zero (inflation-zcis slice)
  slice:  inflation-zcis
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.money import Money


@dataclass(frozen=True)
class ZeroCouponInflationSwap:
    """A ZCIS on `index`: inflation leg vs fixed compounded rate `fixed_rate` (K) on
    `face` (notional + currency)."""

    index: str
    face: Money
    fixed_rate: float
    maturity: date
    receive_inflation: bool = True
