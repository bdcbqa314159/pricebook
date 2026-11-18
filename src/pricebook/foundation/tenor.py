"""A floating-rate tenor (L0) — finance-free, and the dimension a projection curve is keyed by.

Post-crisis, a floating leg's forwards come from a curve specific to its **tenor**: a 3M leg and a
6M leg on the same dates are different trades, priced off different curves, and the gap between them
is the tenor basis. So the tenor has to be something a curve can be keyed by and a product can
carry.

**There was no such type.** The tree's tenor was implicit in a `CouponSchedule` — recoverable from
the dates, but not sayable, so nothing could key a mapping by it or state that two legs share one.
Introduced here, at tranche 55, where it is first consumed (§6: new vocabulary arrives with its
consumer, not speculatively — it was deliberately *not* added at tranche 54, which built the curves
and needed no key).

**Months, not a year fraction.** A tenor is a quoted market convention — "3M", "6M" — and quoting it
as `0.25` would invite two tenors that are the same convention to compare unequal through a day-count
difference. An integer month count is exact, hashable, and compares the way the market means it.

**Finance-free, so L0.** A count of months with a label. It knows nothing of curves, legs or
discounting; what a tenor is *for* lives at L1 and above.

Provenance:
  stock:  none - the archived predecessor had no tenor type either
  source: LICHTERS-2015 (docs/references.md) ch 3.5 / ch 5 - the tenor-specific projection curve,
          whose keying dimension this is; the type itself is vocabulary, not a transcription
  oracle: tests/L4/test_csa_discounting.py - value equality and hashability (so it can key the
          projections store), and the non-positive refusal
  slice:  tranche 55 - CSA discounting
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tenor:
    """A floating-rate tenor as a whole number of months: `Tenor(3)` is 3M.

    Frozen so it can key a mapping, and compared by value so two legs quoting the same convention
    are the same tenor."""

    months: int

    def __post_init__(self) -> None:
        if self.months <= 0:
            raise ValueError(f"a tenor must be a positive number of months, got {self.months}")

    @property
    def label(self) -> str:
        """`"6M"` / `"1Y"` — how the market says it, for error messages and keys a human reads."""
        if self.months % 12 == 0:
            return f"{self.months // 12}Y"
        return f"{self.months}M"
