"""The mark-to-market report (L2) — pure frozen data. R1.4.

`value(request)` returns one number about a book; this is the other half of a desk's mark — **where
that number comes from**. It is a distinct type and not a richer `value()` return (ruled 0456), so
the engine's return shape never varies with how much detail a caller wants.

**The totals are a `PricingResult`, not four restated fields.** A `PricingResult` already *is* the
decomposition — `pv`, `accrued`, `basis`, `standard_error`, with `clean` derived — so restating them
here would be a second home for the same vocabulary, free to drift from the rows it is supposed to
total (§6). The book's currency likewise lives in `total.pv.currency`, where a `Money` keeps it.

**What is NOT here: the cashflow ladder.** No engine exposes dated cashflows, so a ladder would need
the schedule/calendar foundation first; R1.4 is the breakdown and the ladder waits for that arc
(recorded in `to_go_deeper.md`). And the clean/accrued columns are **type-present but
engine-dormant**: no engine populates `accrued` today, so on every real book `accrued is None` and
`clean == pv`. The aggregation is still oracled exactly, against a test pricer that decomposes —
otherwise the one respect the code discriminates on would be uniform across the whole input
population, which is the T33b degeneracy (ruled 0459(b)).

Provenance:
  stock:  none — architecture-first
  source: CLAUDE.md §0 (the representation contract) and §3 (the result is a decomposition, not a
          scalar — which is why the totals are a `PricingResult` and the columns come free)
  oracle: tests/L4/test_mtm_report.py — each row against its own `trade_value`, the rows summing to
          the total in every column, the model attribution per row, and the accrued decomposition
          against a known-accrued test pricer
  crosscheck: none — frozen representation data
  slice:  tranche 85 — R1.4, the mark-to-market breakdown report
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.foundation.results import PricingResult
from pricebook.products.trade import Trade


@dataclass(frozen=True)
class MtMRow:
    """One trade's contribution, and **which model produced it**.

    The model is carried because a breakdown whose numbers are right and whose attribution is wrong
    is the plausible-but-wrong failure a desk cannot see: the PV reconciles, the row reads as
    explained, and the explanation names the wrong model. It is typed `object` for the same reason
    `PricingRequest.assignment` is — a model is L3 and this is L2, so the layering forbids the
    import, and §3 gives a model no common base class to name anyway.
    """

    trade: Trade
    model: object
    result: PricingResult


@dataclass(frozen=True)
class MtMReport:
    """A book's mark: the totals, the per-trade breakdown, and the date it was marked as of.

    `as_of` is carried because a mark without its date is not a mark — constitutive of this type's
    identity, which is the §6 test for a field that nothing computes with.

    The rows are **per trade, not per product**: two trades on one contract are two rows. Netting
    them would destroy the one thing a breakdown exists to show.
    """

    total: PricingResult
    rows: tuple[MtMRow, ...]
    as_of: date
