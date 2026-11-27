"""`position_value` — a book's value and its ERROR (L4). R1.2.

`PV = Σ direction_i · units_i · price(product_i, model_i)`, which telescopes and is therefore the
guard rather than the test. The tranche's real content is the **standard error**:

    SE_book = Σ SE_i          a conservative upper bound under unknown cross-trade correlation

**Why a linear sum and not `√(Σ SE_i²)`, which is the form one reaches for first.** Two separate facts
point the same way, and the first was verified rather than assumed:

* **Within one `(product, model)` pair the errors are the SAME error.** §3.1 makes every pricer
  referentially transparent, and dispatching the same pair twice is **bit-identical** in PV *and* SE
  (asserted in the oracle). So `ρ = 1` exactly, and bands add linearly. This half is a **fact** about
  this library, not a convention.
* **Across pairs the errors are CORRELATED, not independent.** Every Monte-Carlo dispatch draws from
  `UniformSource(config.seed)` and **nothing assigns distinct seeds per trade** — production has a
  single hardcoded `seed=7` — so two trades whose configs share a seed consume **overlapping streams**
  (common random numbers). Quadrature assumes `ρ = 0`, which the architecture does not support, and it
  fails in the **understating** direction: the dangerous one for a book's risk number. *(Ruled 0450 by
  Cowork, which checked `rng.py`/`mc_path_engine`/`heston_mc`; verified here independently — the same
  seed yields an identical uniform stream.)*

So the combination is linear throughout: **exact** within a group, and a **conservative upper bound**
across them. It tightens to quadrature only once the infrastructure guarantees independent per-trade
seeding, which is recorded in `to_go_deeper.md` as an R-track MC-infrastructure item — R2 (greeks of an
estimate) and R3 (scenario) meet the identical question.

**The `(product, model)` grouping is NOT kept as a seam, deliberately, and that departs from the
ruling's wording.** With a linear combination, `Σ_groups Σ_within = Σ_all`: no grouping changes the
answer, so grouping code would be **unexercisable** — a mutation to the grouping key would survive every
test, which is §4's third survivor kind (surface that cannot be tested should not exist) and §6's
nothing-speculative rule. Keeping it would trade a real hole for a hypothetical future. The *reasoning*
is preserved here and in `to_go_deeper`, which is where a future seam belongs; reinstating the grouping
is a few lines when quadrature becomes valid, and it will then be testable.

A band has **no sign**, so a long and an offsetting short of the same contract net to zero *value* with
their bands **adding**: the uncertainty about a netted zero is not zero. That falls out of
`trade_value`'s `abs(scale)` and is oracled explicitly, because a reader may reasonably expect the
cancellation to carry through.

Provenance:
  stock:  none — architecture-first
  source: CLAUDE.md §0/§1 — the representation contract; §3.1 (referential transparency) is what makes
          the within-group combination a fact rather than a convention, and §3.7 is why a book carries
          a band at all
  oracle: tests/L4/test_position.py — determinism asserted first, then both SE regimes against exact
          known values, the mixed exact/estimate book, netting on a non-symmetric payoff, order
          invariance, and the whole-book failure on one unpriceable trade
  crosscheck: none — no library prices an abstract book wrapper; the underlying engines carry theirs
  slice:  tranche 83 — R1.2, the Position type
"""

from __future__ import annotations

from collections.abc import Sequence

from pricebook.engine.trade import trade_value
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.products.position import Position


def position_value(
    position: Position, models: Sequence[object]
) -> PricingResult | PricingFailure:
    """The book's value, or the failure of the first trade that could not be priced.

    **One unpriceable trade fails the whole book.** Skipping it and totalling the rest would return a
    *number* that silently omits a position, which is worse than no number — §3.4's reasoning one
    layer up. The failure names the product so the caller can act on it.

    `models` is one model per trade, positionally. Two parallel sequences are the honest minimum
    until R1.3's `PricingRequest` resolves models from the market; the length is **checked** rather
    than `zip`-ed, because `zip` truncates silently and would return a confident partial total.
    """
    if len(models) != len(position.trades):
        raise ValueError(
            f"position_value needs one model per trade: {len(position.trades)} trades, "
            f"{len(models)} models"
        )

    total = 0.0
    # The bands sum LINEARLY (ruled 0450): exact within a `(product, model)` pair, where §3.1 forces
    # the same error, and a conservative upper bound across pairs, where the shared-seed architecture
    # makes the errors correlated rather than independent. No grouping, because with a linear sum no
    # grouping changes the answer and the code would be unexercisable -- see the module docstring.
    bands = 0.0
    seen_estimate = False

    for trade, model in zip(position.trades, models):
        priced = trade_value(trade, model)
        if isinstance(priced, PricingFailure):
            return PricingFailure(
                f"the book cannot be valued: {type(trade.product).__name__} — {priced.reason}"
            )
        if priced.pv.currency != position.currency:
            return PricingFailure(
                f"the book reports in {position.currency.code} but "
                f"{type(trade.product).__name__} prices in {priced.pv.currency.code} — unlike "
                "currencies cannot be summed, and cross-currency netting is not modelled yet"
            )
        total += priced.pv.amount
        if priced.standard_error is not None:
            bands += priced.standard_error
            seen_estimate = True

    # No band when nothing in the book is an estimate — including the empty book, which is an exact
    # sum of nothing. `None` is §3's marker for "not a Monte-Carlo estimate"; a `0.0` would claim a
    # *measured* zero uncertainty, a stronger and different statement (ruled 0450(c)).
    combined = bands if seen_estimate else None
    return PricingResult(pv=Money(total, position.currency), standard_error=combined)
