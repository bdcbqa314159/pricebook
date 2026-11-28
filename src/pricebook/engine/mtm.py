"""`mark_to_market` — a book's mark, broken down by trade (L4). R1.4.

The breakdown and the scalar `value(request)` are **two views of one computation**, and that is
structural rather than asserted: both resolve through `resolve_models`, price through
`priced_trades`, and total through `combine`. This function's entire difference from `value` is that
it *keeps* the per-trade results instead of discarding them after the sum. A second aggregation here
would be the C16/C17 duplication the re-audits hunt, and the two views could then disagree about the
same book — the one failure a reconciliation report must not have.

So the reconciliation oracle (`report.total == value(request)`) **telescopes** and is the guard, not
the test. The tests are the rows: each against its own `trade_value`, the rows summing to the total
in every column, and the model attribution per row.

Provenance:
  stock:  none — architecture-first
  source: CLAUDE.md §0 (the representation contract) and §3 (a decomposition, not a scalar; and §3.4
          — a refusal is a value, which is why a foreign `accrued` is refused rather than raised)
  oracle: tests/L4/test_mtm_report.py — the rows against `trade_value`, the column sums, the
          attribution, the signed accrued scaling, and both request refusals pinned to the guard
          that fired
  crosscheck: none — no library marks an abstract book; the underlying engines carry theirs
  slice:  tranche 85 — R1.4, the mark-to-market breakdown report
"""

from __future__ import annotations

from pricebook.engine.position import combine, priced_trades
from pricebook.engine.pricing_request import resolve_models
from pricebook.foundation.results import PricingFailure
from pricebook.products.mtm_report import MtMReport, MtMRow
from pricebook.products.pricing_request import PricingRequest


def mark_to_market(request: PricingRequest) -> MtMReport | PricingFailure:
    """The book's mark with its breakdown, or the first refusal that prevents valuing it.

    A report is not a way to smuggle an unpriceable book past the guards: every refusal `value`
    makes, this makes, with the same wording — so a caller cannot get a partial report where the
    scalar would have refused.
    """
    models = resolve_models(request)
    if isinstance(models, PricingFailure):
        return models

    results = priced_trades(request.position, models)
    if isinstance(results, PricingFailure):
        return results

    rows = tuple(
        MtMRow(trade=trade, model=model, result=result)
        for trade, model, result in zip(request.position.trades, models, results)
    )
    return MtMReport(
        total=combine(results, request.position.currency), rows=rows, as_of=request.as_of
    )
