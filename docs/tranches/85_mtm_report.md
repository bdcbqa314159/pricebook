# T85 — R1.4: the mark-to-market breakdown report

**Cowork spec, 2026-11-28.** The fourth representation tranche. `value(request)` (T84) returns one number
about the book; the **MtM report** is the other half of a desk's mark — *where the number comes from*: a
per-trade breakdown with the model that priced each. Ruled a **distinct type**, not a richer `value()`
return (0456), so the engine's return shape never varies by caller appetite (§3).

## Target
A frozen `MtMReport` and a pure `mark_to_market(request) → MtMReport` that carries the book totals and a
**per-trade breakdown** — each trade's `PricingResult` contribution and which model priced it.

## Scope — the breakdown, NOT the cashflow ladder
`PricingResult` carries `pv`/`accrued`/`clean`/`basis`/`standard_error` but **no dated cashflows**, and no
engine exposes them (checked, 0456). So:
- **In scope:** the per-trade breakdown + book totals, with **dirty/clean/accrued for free** (the result
  already decomposes — the report surfaces that decomposition per trade and aggregated).
- **Deferred:** the **cashflow ladder** — it needs cashflow exposure, which is the **schedule/calendar rework
  arc**'s to build. R1.4 does not dig that foundation; the report is the breakdown.

## Pieces, layer, reuse
- **`MtMReport`** (new, **L2-level frozen representation data**) — the book totals (dirty PV, clean, accrued,
  SE, currency) + an ordered per-trade breakdown (for each trade: the trade, the model that priced it, its
  `PricingResult`). ≤5 fields (§7; the breakdown is one field, a frozen sequence of a small row value object —
  which is itself ≤5 fields).
- **`mark_to_market(request)`** (new, **L4-level pure function**) — resolves and prices each trade exactly as
  `value()` does (reuse that path, do not re-derive it — §6; `value()` and the report must agree by
  construction), and additionally **keeps** each trade's result and model rather than only summing. Reuses
  T82/T83/T84 wholesale.
- The totals aggregation (PV sum, **linear** SE per T83, declared currency, the refusals) is the **same**
  combination T84 uses — share it, do not fork it (a second SE-combination implementation is the C16/C17
  duplication the re-audits hunt).

## Oracles (RED-first, §4)
1. **Totals reconcile with `value()`, exactly:** `mark_to_market(request).total_pv == value(request).pv`
   (and SE, clean, accrued) — the report and the scalar are **two views of one computation**, so this is the
   consistency guard. It telescopes if they share the code (they should), so it is the guard; the breakdown
   oracles below are the test (the T82 telescoping shape again).
2. **Each breakdown row is its own trade's value:** row_i.result == `trade_value(trade_i, model_i)` exactly,
   and **the rows sum to the total** (dirty, clean, accrued each) — a mutation that drops, duplicates, or
   mis-attributes a row dies. The fixture book has **≥2 trades of different products with different models**,
   so attribution is a demonstrable choice (the τ=1/resolution-unexercised family).
3. **The model attribution is correct per row** — row_i names the model that actually priced trade_i (a
   mutation swapping two rows' model labels dies) — because a breakdown whose *numbers* are right but whose
   *attribution* is wrong is the structural track's "plausible but wrong" hazard (T83), and a desk reads the
   attribution.
4. **SE in the breakdown:** a mixed book (one exact `None`, one estimate) reports each row's SE faithfully
   (None stays None, not 0) and the **total SE is the linear combination** (T83) — the estimate-population
   premise carries (≥2 estimate pricers so the total SE does real work).
5. **The refusals ride through** (unassigned model, mixed currency, mixed as-of date) — `mark_to_market`
   returns the same `PricingFailure`/refusal `value()` does; a report is not a way to smuggle an unpriceable
   book past the guards (assert the fall-through is pinned to the *right* refusal — the T84 watch, since the
   report path re-enters the guard-heavy code).

## Degeneracy premises (§4 / §9, the standing R1 premises)
- Fixture population: **≥2 estimate pricers** (total SE live), **≥2 product types with different models**
  (attribution does real work), a **mix of exact and estimate** (the None-SE row is exercised).
- The model-attribution fixture pairs the **same product type is not reused with one model only** — two rows
  must differ in model for a label-swap mutation to be killable.

## Provenance / gates
- Provenance headers; `fields` ≤5 (the row value object too); `acyclic`/`layers` green. No `references.md`
  flip.
- Version → **v0.87.0**; floor from the collected count; one `CHANGELOG` entry.
- **Status page:** Representation advances toward **4 of 5**.

## Out of scope (→ R1.5)
- The **cashflow ladder** (deferred to the schedule/calendar rework arc — no cashflow exposure exists).
- The **mixed-asset book capstone** (R1.5 — value a swap + swaption + CDS end-to-end, the "it's a library
  now" demo, with a one-shock scenario *preview* as the seam to R3).
- Greeks (R2), scenario proper (R3), lifecycle (R4-of-the-plan / the stateful L6 tier).
