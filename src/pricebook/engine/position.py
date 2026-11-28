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
  oracle: tests/L4/test_mtm_report.py (the shared loop's accrued aggregation and its
          foreign-accrued refusal, T85); tests/L4/test_position.py — determinism asserted first, then both SE regimes against exact
          known values, the mixed exact/estimate book, netting on a non-symmetric payoff, order
          invariance, and the whole-book failure on one unpriceable trade
  crosscheck: none — no library prices an abstract book wrapper; the underlying engines carry theirs
  slice:  tranche 83 — R1.2, the Position type
"""

from __future__ import annotations

from collections.abc import Sequence

from pricebook.engine.trade import trade_value
from pricebook.foundation.money import Currency, Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.products.position import Position


def priced_trades(
    position: Position, models: Sequence[object]
) -> list[PricingResult] | PricingFailure:
    """Each trade's own value, in order — or the first refusal that stops the book being valued.

    **Extracted at its second consumer** (T85's `mark_to_market`), which is the rule-of-two trigger:
    the scalar total and the breakdown report are two views of one computation, so they must price
    and refuse through the *same* code. A second implementation of this loop would be the C16/C17
    duplication the re-audits hunt, and the two views could then disagree.

    `models` is one model per trade, positionally; the length is **checked** rather than `zip`-ed,
    because `zip` truncates silently and would return a confident partial total.

    Three refusals live here, and all three are about summing things that cannot be summed:

    * **one unpriceable trade fails the whole book** — skipping it and totalling the rest would
      report a number that silently omits a position, which is worse than no number (§3.4 one layer
      up);
    * **a PV in another currency** than the book declares;
    * **an `accrued` in another currency** than the book declares. That one is not symmetry for its
      own sake: `Money` *raises* on mixing, so without the check the aggregation would raise instead
      of returning a failure value, which §3.4 forbids. No engine in the tree populates `accrued`
      today, so the branch is reachable only from a test pricer that does — which is how it is
      oracled, rather than being left as the untestable surface §4 says should not exist.
    """
    if len(models) != len(position.trades):
        raise ValueError(
            f"position_value needs one model per trade: {len(position.trades)} trades, "
            f"{len(models)} models"
        )

    results: list[PricingResult] = []
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
        if priced.accrued is not None and priced.accrued.currency != position.currency:
            return PricingFailure(
                f"the book reports in {position.currency.code} but "
                f"{type(trade.product).__name__} reports accrued interest in "
                f"{priced.accrued.currency.code} — the accrued column cannot be summed across "
                "currencies, and a decomposition whose parts are in different units is not one"
            )
        results.append(priced)

    return results


def combine(results: Sequence[PricingResult], currency: Currency) -> PricingResult:
    """The book's total from its parts: PV added, bands added LINEARLY, accrued added where it exists.

    The linear band is ruled 0450 — see the module docstring for why it is not quadrature. `accrued`
    follows the **sum identity**, which fixes the rule rather than leaving it to taste: a row whose
    `accrued is None` has `clean == pv`, so the book's clean equals the sum of the rows' cleans only
    if an undecomposed row contributes **zero** accrued. Hence `None` contributes nothing to the
    total, and the total is `None` only when **no** row decomposed — which is every real book today,
    since no engine populates `accrued` (tracked in `to_go_deeper.md`).

    A `0.0` in place of either `None` would be a stronger and different statement — a *measured* zero
    band, or a decomposition asserting there is no earned-but-unpaid interest (ruled 0450(c)).
    """
    total = 0.0
    bands = 0.0
    accrued_total = 0.0
    seen_estimate = False
    seen_accrual = False

    for priced in results:
        total += priced.pv.amount
        if priced.standard_error is not None:
            bands += priced.standard_error
            seen_estimate = True
        if priced.accrued is not None:
            accrued_total += priced.accrued.amount
            seen_accrual = True

    return PricingResult(
        pv=Money(total, currency),
        accrued=Money(accrued_total, currency) if seen_accrual else None,
        standard_error=bands if seen_estimate else None,
    )


def position_value(
    position: Position, models: Sequence[object]
) -> PricingResult | PricingFailure:
    """The book's value, or the failure of the first trade that could not be priced.

    Two steps, both shared with T85's breakdown report: price every trade, then combine. This
    function keeps only the total; the report keeps the rows as well. That is the *whole* difference
    between them, which is what makes "two views of one computation" a fact rather than a hope.
    """
    priced = priced_trades(position, models)
    if isinstance(priced, PricingFailure):
        return priced
    return combine(priced, position.currency)
