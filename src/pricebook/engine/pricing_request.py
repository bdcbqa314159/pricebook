"""`value` — a book marked as of one date, under an explicit set of models (L4). R1.3.

`value(request)` resolves each trade to `request.assignment[type(trade.product)]`, then reuses T83's
`position_value` wholesale: the summed PV, the linear standard-error combination, the declared-currency
check. It adds exactly two things of its own, and both are refusals:

* **an unassigned product type** — a book holding something the pricing configuration does not cover
  cannot be valued, and skipping it would return a *number* that silently omits a position (§3.4's
  reasoning, now two layers up);
* **a model whose market is not as of the requested date** — you cannot mark a book half as-of today and
  half as-of yesterday. This is the currency-refusal shape on the date axis, and it is checked against the
  request's **declared** date rather than among the models, because mutual agreement is the weaker test:
  models that agree with each other but not with the caller's intent are exactly the silent failure
  (a book assembled from yesterday's snapshots, valuing itself consistently as of yesterday).

**Invariant 6 is inherited, not re-implemented.** A trade whose flows have settled already prices to PV 0
inside its engine (§3 invariant 6), so this function adds no date filtering at all — it would be a second
copy of a guard the engines already carry, and §4's consolidation rule points the other way. The oracle
asserts the inheritance *through* both wrappers with an **in-the-money** settled option, because an
out-of-the-money one has intrinsic 0 and the guard stays invisible (F-4.1's premise).

Provenance:
  stock:  none — architecture-first
  source: CLAUDE.md §0/§1 — the representation contract and the purpose-defined 1.0; §3 invariant 2
          (a model carries its market, hence its valuation date) and invariant 6 (settled flows are the
          shell's business) are what make the two refusals checkable
  oracle: tests/L4/test_pricing_request.py — resolution as a genuine choice, the two refusals in both
          branches, invariant 6 surviving two wrappers, and T83's aggregation riding through unchanged
  crosscheck: none — no library prices an abstract valuation request; the engines carry theirs
  slice:  tranche 84 — R1.3, the PricingRequest type
"""

from __future__ import annotations

from pricebook.engine.position import position_value
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.products.pricing_request import PricingRequest


def value(request: PricingRequest) -> PricingResult | PricingFailure:
    """The book's value under the requested models, or the first refusal that prevents it."""
    models: list[object] = []
    for trade in request.position.trades:
        product_type = type(trade.product)
        model = request.assignment.get(product_type)
        if model is None:
            return PricingFailure(
                f"the pricing configuration assigns no model to {product_type.__name__} — a book "
                "holding it cannot be valued, and omitting the position would report a number that "
                "is silently incomplete"
            )
        carried = getattr(getattr(model, "market", None), "valuation_date", None)
        if carried != request.as_of:
            return PricingFailure(
                f"the request is as of {request.as_of} but the model assigned to "
                f"{product_type.__name__} carries a market with valuation date {carried} — a book "
                "cannot be marked as of two dates"
            )
        models.append(model)

    return position_value(request.position, models)
