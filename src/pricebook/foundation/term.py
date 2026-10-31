"""A contract term: two dates and the basis they are measured on (L0).

**Why this type exists rather than two loose fields.** `ZeroCouponInflationSwap` needs the index,
the base level `I0`, the break-even `b`, the notional, *and* both ends of its term — six loose
fields, one over the five-field limit. The rule's answer to that is never to raise the limit but to
**group cohesive parameters into a frozen value object**, and `(start, maturity, day_count)` is
cohesive in a way that matters here: the fixed leg pays `(1+b)^TM − 1` where `TM` is the term's
own year fraction, so the dates and the basis they are measured on are one fact, not three.

It also puts the year fraction in one place. A ZC inflation swap's compounding exponent is measured
from **inception**, not from the valuation date — the fixed payout was fixed at trade time — and
that is the kind of detail that goes wrong when each caller computes its own `T`.

Provenance:
  stock:  — (built fresh; the stock's equivalent is `Accrual`, which carries an accrual period for
          coupon purposes rather than a contract's whole term)
  source: BERHAN-2012 (docs/references.md) §1.2.2, eq 1.2.7 — the fixed payout's exponent `TM`
  oracle: `years()` agrees with the L0 primitive; the par identity depends on it being the
          inception-to-maturity fraction and not a valuation-date-relative one
  slice:  tranche 2 — the inflation spine
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.foundation.day_count import DayCountConvention, year_fraction


@dataclass(frozen=True)
class Term:
    """`start` to `maturity`, measured on `day_count`."""

    start: date
    maturity: date
    day_count: DayCountConvention

    def __post_init__(self) -> None:
        if self.maturity <= self.start:
            raise ValueError(f"a term must end after it starts: {self.start} → {self.maturity}")

    def years(self) -> float:
        """The term's length — **from inception**, which is what a compounded fixed payout uses."""
        return year_fraction(self.start, self.maturity, self.day_count)

    def to_dict(self) -> dict:
        return {"start": self.start.isoformat(), "maturity": self.maturity.isoformat(),
                "day_count": self.day_count.value}

    @classmethod
    def from_dict(cls, d: dict) -> Term:
        return cls(
            date.fromisoformat(d["start"]),
            date.fromisoformat(d["maturity"]),
            DayCountConvention(d["day_count"]),
        )
