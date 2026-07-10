"""Cashflow — the instrument atom, promoted to L0.

The shared atom of every instrument must not live inside one asset class
(vocabulary ratification: Cashflow L3 -> L0). A single payment: `amount` on
`date`. Frozen pure data — it does not price itself.

Provenance:
  quarry: python/pricebook/fixed_income/fixed_leg.py (Cashflow)
  source: promoted per redesign/03_vocabulary.md
  oracle: N/A (pure value type; exercised by the Slice 0 closed-form oracle)
  slice:  S00
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.money import Money


@dataclass(frozen=True)
class Cashflow:
    """A single payment of `amount` on `date`."""

    date: date
    amount: Money
