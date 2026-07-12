"""Cashflow — the instrument atom, promoted to L0.

The shared atom of every instrument must not live inside one asset class
(vocabulary ratification: Cashflow L3 -> L0). A single payment: `amount` on
`date`. Frozen pure data — it does not price itself.

Provenance:
  quarry: python/pricebook/fixed_income/fixed_leg.py (Cashflow)
  source: promoted per redesign/03_vocabulary.md
  oracle: N/A (pure value type; exercised by the Slice 0 closed-form oracle)
  slice:  S00; A2 (optional Accrual, for accrued interest / clean-vs-dirty)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.time import DayCountConvention, year_fraction


@dataclass(frozen=True)
class Accrual:
    """The accrual period behind a coupon: its span and day-count convention.
    Lets the engine compute the earned-but-unpaid slice at a valuation date."""

    start: date
    end: date
    day_count: DayCountConvention

    def earned_fraction(self, valuation: date) -> float:
        """Fraction of the period accrued by `valuation` (0 before start, 1 after
        end), on the coupon's own day count."""
        if valuation <= self.start:
            return 0.0
        if valuation >= self.end:
            return 1.0
        return (year_fraction(self.start, valuation, self.day_count)
                / year_fraction(self.start, self.end, self.day_count))


@dataclass(frozen=True)
class Cashflow:
    """A single payment of `amount` on `date`. A coupon additionally carries its
    `accrual` period; a bullet payment leaves it `None`."""

    date: date
    amount: Money
    accrual: Accrual | None = None
