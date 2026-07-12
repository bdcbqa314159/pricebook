"""FixedCashflow — the Slice 0 product: one fixed cashflow.

L2: products are pure data. This describes a single payment and holds NO
`pv` method — pricing lives in the L4 engine (CLAUDE.md 2).

Provenance:
  quarry: python/pricebook/fixed_income/ (zero-coupon / single cashflow)
  source: redesign/04_slice_plan.md (Slice 0 product)
  oracle: N/A (pure data; priced by DiscountingEngine under the Slice 0 oracle)
  slice:  S00; A3 (instrument -> product rename)
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook_ng.foundation.cashflow import Cashflow


@dataclass(frozen=True)
class FixedCashflow:
    """A product that pays a single fixed `Cashflow`."""

    cashflow: Cashflow

    @property
    def cashflows(self) -> tuple[Cashflow, ...]:
        """The engine's `CashflowProduct` view: a one-cashflow leg."""
        return (self.cashflow,)
