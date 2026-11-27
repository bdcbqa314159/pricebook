"""The `PricingRequest` — value *this book*, under *these models*, as of *this date* (L2).

**R1.3, and §0's missing half.** T82 gave a holding, T83 a book; this says what it means to *ask* for a
valuation. Frozen data; prices nothing.

**The model assignment is supplied, never resolved — and that was a build-time finding, not a
preference** (ruled 0453). The R1 plan originally had the request "resolve the correct snapshot per
trade". It cannot: the product→model mapping is **one-to-many** and materially so — `EuropeanEquityOption`
has four registered models, `NominalCap` four, `VarianceSwap` three (replication, 3/2, Lévy) — and *which*
one prices a variance swap is a modelling choice nothing in the trade records. Compounding it, §3 makes
every model carry the `MarketSnapshot` it was calibrated to, so "supply a market, get a model" would be
calibration rather than representation. So the request carries the desk's **pricing configuration**: a
`{product_type → model}` map of already-calibrated choices.

**It is a mapping, not a policy** (§6): a house-model choice is *data*. A rule that picked a model per
product would be the speculative engine the charter forbids — and there is no second consumer asking for
one.

**`as_of` is DECLARED, not derived, and that is the one design decision this type turns on.** Every
assigned model already carries its own `market.valuation_date` (§3 invariant 2), so the date *could* be
read off them and checked for mutual agreement. That check is strictly weaker: a book accidentally
assembled from yesterday's models is perfectly self-consistent, values itself as-of yesterday, and reports
nothing wrong. Declaring the date states what the caller **meant**, so every model is checked against the
intent rather than against each other. The same argument Cowork accepted for `Position.currency` at
0446 — declare, then refuse a mismatch — and the oracle for it asserts exactly the case deriving cannot
catch.

Provenance:
  stock:  none — architecture-first; the structural track anchors on the catalogue
  source: CLAUDE.md §0 (the representation contract: a product and a *pricing request* are the durable
          public half) and §1 (the purpose-defined 1.0); §3 invariant 2 is what makes `as_of` checkable
  oracle: tests/L4/test_pricing_request.py — the assignment resolving a genuine *choice* (a fixture
          product with two registered models), the unassigned-type refusal, the declared-date refusal in
          both branches including self-consistent-but-wrong-dated models, and invariant 6 inherited
          through two wrappers
  crosscheck: none — frozen representation data
  slice:  tranche 84 — R1.3, the PricingRequest type
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from pricebook.products.position import Position


@dataclass(frozen=True)
class PricingRequest:
    """A book, the models to price it with, and the date it is marked as of.

    `assignment` is keyed by **product type**, which is the desk-natural granularity: a house-model set
    says "we price equity options with Black and caps with Hull-White". A per-*trade* override (two
    equity trades on different models) is a rule-of-two pull when a second consumer wants it, not a
    hook built now (§6).
    """

    position: Position
    assignment: Mapping[type, object]
    as_of: date
