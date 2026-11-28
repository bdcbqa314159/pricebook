# T84 — R1.3: the `PricingRequest` (an explicit model assignment + the as-of date)

**Cowork spec, 2026-11-27.** The third representation tranche. A `PricingRequest` is §0's missing half — the
durable "value *this book* as of *this date* with *these models*" — and its central fact, ruled 0453 after
Code's registry finding, is that **a product does not determine its model** (one-to-many), so the request
carries an **explicit model assignment**, it does not resolve one.

## Target
A frozen `PricingRequest` (a `Position` + a model assignment) and a pure
`value(request) → PricingResult` that resolves each trade to its assigned model, dispatches, and aggregates
— the `Position` MtM under the chosen models, marked consistently as of one date.

## The shape (ruled 0453 — do not reinvent the resolution)
- The request carries a **model assignment `{product_type → model}`** — the desk "pricing configuration" /
  house-model set. Each trade resolves to `assignment[type(trade.product)]`. Each assigned model is already
  calibrated and **carries its own `MarketSnapshot`** (§3) — the request supplies *choices*, not calibration.
- **It is a mapping, not a policy** (§6): a house-model choice is data, not a rule engine.
- **"Resolve the snapshot from the product" is retired** — it cannot be done and is not attempted.

## Pieces, layer, reuse
- **`PricingRequest`** (new, **L2-level frozen representation data**) — a `Position` (T83) + the
  `{product_type → model}` assignment. ≤5 fields (§7; the assignment is one field, a frozen mapping). Does
  not price itself.
- **`value(request)`** (new, **L4-level pure function**) — for each trade, look up its model by product
  type, call `trade_value(trade, model)` (T82), aggregate via the `Position` linear SE combination (T83).
  Reuses T82/T83 and the registry wholesale.
- Build confirms placement; `acyclic`/`layers` green (the T82(b) unmapped-package gate holds).

## The genuinely new content — the AS-OF semantics (the obligation, not a form; §4)
The model assignment is a lookup (modest). The real new concept is marking a book **as of one date**:
- **Valuation-date consistency:** every assigned model carries `model.market.valuation_date` (§3 invariant
  2). The request **asserts they share one date** and **refuses a mixed-date assignment** — you cannot mark a
  book half as-of today and half as-of yesterday (the currency-refusal shape, on the date axis). State the
  obligation (the book is marked as of a single, explicit date); let the build decide whether the date is a
  carried field checked against the models, or derived-and-asserted-consistent.
- **Invariant-6 is INHERITED, not re-implemented:** a trade whose flows have settled before the valuation
  date already prices to PV 0 inside its engine (§3 invariant 6). The request adds no date filtering — it
  inherits it. Assert that a settled trade in a book contributes 0 *through* the request (a cross-level
  check that invariant-6 survives the wrapper), not a new filter.

## Oracles (RED-first, §4)
1. **Resolution + reduction:** `value(request)` for a single-trade book == `trade_value(trade,
   assignment[type(product)])` == `price(product, model)` scaled — the assignment resolves to the right
   model, exact. A **mutation resolving to the wrong model** (or to `None`) dies — so the fixture's book has
   **≥2 product types with different assigned models**, else the resolution is unexercised (the τ=1 /
   degenerate-population family: the lookup must do real work).
2. **Unassigned product-type → `PricingFailure` (refusal), not a silent skip or a raise** — a book with a
   product the assignment does not cover cannot be priced; the §3 contract rides through.
3. **Valuation-date consistency refusal:** an assignment whose models carry **different** valuation dates is
   refused (build a fixture with two models as-of different dates; assert the refusal, and assert it does NOT
   fire when the dates agree — both branches, the T33b validation-guard discipline).
4. **Invariant-6 through the wrapper:** a book containing a **settled** trade (a flow before the valuation
   date) values it at PV 0 via the request, and a non-settled companion is unaffected — the cross-level
   inheritance check (assert a mutation that bypasses the settled trade would change the book value, i.e. the
   fixture's settled trade is in-the-money so PV 0 ≠ its fall-through, the F-4.1 premise).
5. **Aggregation + SE** ride through from T83 (linear SE, declared currency, empty/mismatch refusals) — a
   regression check that wrapping in a request changes none of it.

## Degeneracy premises (§4 / §9, the standing R1 premises)
- The fixture **population includes ≥2 estimate pricers** (the SE combination is still live through the
  request) and **≥2 product types with different models** (the resolution does real work).
- The settled-trade fixture is **in-the-money** (so PV 0 ≠ intrinsic — invariant-6 is visible, the F-4.1/4.2
  straddle-plus-ITM premise).
- The date-consistency fixture has models **genuinely as-of different dates** (else the refusal is vacuous).

## Provenance / gates
- Provenance headers; `fields` ≤5; `acyclic`/`layers` green. No `references.md` flip (architecture-first).
- Version → **v0.86.0**; floor from the collected count; one `CHANGELOG` entry.
- **Status page:** Representation advances toward **3 of 5**.

## Out of scope (→ R1.4+)
- The **rich MtM report** — per-trade breakdown, the cashflow ladder — is R1.4; T84 delivers the single
  aggregated `value(request)`. *(If at build the request + its value and the breakdown read as one unit, flag
  it — R1.3/R1.4 may merge; the plan separated them but the seam is the report, not the valuation.)*
- Per-**trade** model override (two trades of one product, different models) — a rule-of-two pull later.
- Re-calibration / as-of-a-new-date (producing new models for a new market) — that is calibration, a later
  arc, not representation.
- Multi-currency (FX, after R1).
