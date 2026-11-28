# T86 — R1.5: the mixed-asset book capstone ("it's a library now")

**Cowork spec, 2026-11-28.** The R1 capstone. Not a new type (Portfolio is deferred, §6 — ruled 0461): the
proof that R1.1–R1.4 **compose** — a genuinely mixed book valued end-to-end through
`PricingRequest → MtMReport`, reconciled against the manual composition — plus a one-shock **scenario
preview** that seams to R3. This is the tranche where "a catalog of pricers" becomes "value and risk a book."

## Target
An end-to-end valuation of a **mixed-asset book** (≥3 asset classes, multiple models) via the R1 machinery,
oracled against the manual sum of independent prices; and a **scenario preview** — the same book repriced
under one shocked `MarketSnapshot`, demonstrating the functional-core payoff (reprice = a pure call on a new
snapshot), as the seam to R3, not the R3 engine.

## Pieces — mostly integration, little new code
- **No new representation type.** `Portfolio` (a hierarchy of positions) is a §6 later-pull; a flat
  multi-product `Position` (T83) is the book. Reuses `Trade`/`Position`/`PricingRequest`/`MtMReport`/the
  registry wholesale.
- **The capstone fixture:** a book of, e.g., an **interest-rate swap** (rates), an **equity/swaption**
  (vol), and a **CDS** (credit) — three asset classes, each with its assigned model, one currency, one
  as-of date — valued through `mark_to_market(request)`.
- **A one-shock scenario preview** (test-level, no new API): build a shocked `MarketSnapshot` (bump one
  curve), re-run the pricing, show the book PV moves. This exercises the architecture's scenario readiness
  (immutable snapshot ⇒ a scenario is a reprice) without building R3's scenario engine.
- **Status roster correction (ruled 0461/0464):** **MARK `Portfolio` a deferred §6 pull** (not drop the
  row) — a scoreboard should keep the *decision* visible, so the row reads "deferred (§6 — a hierarchy of
  positions; a flat multi-product `Position` is the book)", and the R1 count reads **4 of 4 built** at the
  capstone, not a perpetual "4 of 5". And **T86's `CHANGELOG` entry names the supersession explicitly**
  (v0.87.0 said "4 of 5, `Portfolio` outstanding"; T86 makes it 4 of 4) so the two releases do not read as a
  silent reversal (§9). Follow the plan, not the T81 forecast (as `MtMReport` did).

## Oracles (RED-first, §4)
1. **End-to-end reconciliation (the capstone identity):** the mixed book's `MtMReport` totals ==
   the manual sum of the three trades priced **independently** (`price(product, model)` each, scaled) —
   dirty PV, clean, accrued, and the **linear** SE (T83). It telescopes through the shared path (guard); the
   genuinely-mixed fixture is what gives it content (≥3 asset classes, so cross-asset dispatch, aggregation,
   currency and as-of consistency all fire together — the integration a single-asset book cannot test).
2. **Per-trade attribution across asset classes:** each breakdown row names the right product **and** the
   right model, across three different `(product, model)` pairs — a mutation mis-routing one asset class's
   model dies (the T83 "plausible but wrong" hazard at book scale; a desk reads which model marked the CDS).
3. **The scenario preview is a real check, not a smoke test:** the shocked-snapshot reprice == an
   **independent** manual reprice of each trade under the same shock (exact), and the **unshocked** book is
   unchanged (the shock is isolated to the scenario — the immutable-snapshot guarantee: the original mark is
   not mutated). A mutation that leaks the shock into the base, or fails to move the shocked PV, dies.
4. **The whole-book refusals** (one trade in an unassigned model / foreign currency / mismatched as-of date
   inside the mixed book) still refuse, pinned to the right guard (the T84 watch at book scale).

## Degeneracy premises (§4 / §9, the standing R1 premises)
- The book is **genuinely mixed** (≥3 asset classes, ≥3 `(product, model)` pairs) — a single-asset or
  single-model capstone tests nothing the earlier tranches did not (the integration must do real work).
- The population includes **≥1 estimate pricer** (the book SE is live) and **≥1 known-accrued pricer** (the
  clean/accrued totals are non-degenerate — the T85 axis).
- The scenario shock **moves a factor the book actually depends on** (bump a curve a trade discounts on, not
  an unused one — else the preview is vacuous).

## Provenance / gates
- Provenance headers if any new `src/` surface (likely none — integration + a scenario-preview test);
  `fields`/`acyclic`/`layers` green. No `references.md` flip.
- Version → **v0.88.0**; floor from the collected count; one `CHANGELOG` entry.
- **Status page:** Representation reads **4 of 4** (Portfolio deferred); the R1 arc is complete.

## Out of scope / what comes next
- `Portfolio` hierarchy (§6 later-pull), the **scenario engine** (R3 — the preview is only the seam), the
  **cashflow ladder** (schedule/calendar rework arc), greeks (R2).
- After R1.5: **re-audit #10** (the first structural-arc close — its sweep scope includes the new L2/L4
  representation surface, the test-pricer-only axes, and the fall-through-guard watch), then **R2 (greeks)**.
