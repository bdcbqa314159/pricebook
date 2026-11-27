# T83 — R1.2: the `Position` type and aggregation

**Cowork spec, 2026-11-26.** The second representation tranche. A `Position` is a frozen collection of
trades; its value aggregates the trades'. The headline aggregation identity telescopes (as T82's reduction
did), so the **real content is the companions — above all how a book's Monte-Carlo ERROR combines**, which
is where T82's recorded estimate-population premise pays off.

## Target
A frozen `Position` (a collection of `Trade`s) and a pure
`position_value(position, models) → PricingResult` whose PV is the sum of the trades' values and whose SE
combines the trades' SEs under a stated independence assumption.

## Pieces, layer, reuse
- **`Position`** (new, **L2-level frozen representation data**) — a collection of `Trade`s (T82). ≤5 fields
  (§7). Does not price itself.
- **`position_value(position, models)`** (new, **L4-level pure function**) — `Σ trade_value(trade, model)`
  over the trades, projected to one `PricingResult`. **Reuses `trade_value` (T82) and the registry**; the
  models are supplied per trade here (resolving *which* model from the market is R1.3's `PricingRequest`, not
  this tranche). `PricingFailure` from any trade propagates (a book with one unpriceable trade is a failure,
  not a silent partial — confirm this is the right convention at build; the §3 contract rides through).
- Build confirms the module placement and `verify.py acyclic/layers` green (the new `acyclic` gate from T82(b)
  now *requires* the package be mapped).

## The one genuinely new decision — name the OBLIGATION, not the form (§4; my four-error watch)
The PV aggregation is a sum and telescopes. **The SE aggregation is a real convention with a right answer**:
how does the standard error of a *book* combine from its trades' SEs? The **obligation** is that
`position_value`'s SE correctly reflects the combination **under a stated independence assumption** — state
it, and oracle the SE against that statement. Do **not** let me pre-commit the form; *measure* it. (The
expected form, to confirm not assume: trades priced by **different** MC runs have **independent** errors, so
**variances add** — `SE_book = √(Σ SE_i²)` — while trades sharing one MC run/seed are correlated and SEs add
linearly. State which regime the fixture is in and which the production path assumes.)

## Oracles (RED-first, §4) — the telescoping headline, the real companions
1. **Aggregation identity (telescopes — a guard, not the test):** `position_value == Σ trade_value(trade_i,
   model_i)` exactly. Satisfied by construction, so it is the consistency guard; the companions below are the
   test (the T82 self-consistency-telescopes shape, one level up).
2. **The SE combination — the load-bearing new oracle.** Build a **fixture with ≥2 ESTIMATE pricers**
   (registered test-pricers with *known* SEs — the R1-plan premise, exactly why it exists), so the book SE is
   checked against the **stated** combination to an **exact** value (known SEs → known `√(Σ SE_i²)`), not a
   band. A mutant using the wrong combination (linear sum where variances should add, or dropping a term)
   dies here. **A mixed book (one exact `SE=None` + one estimate)** must also combine correctly — the `None`
   contributes zero variance, not a crash.
3. **Empty and singleton reductions:** an empty position is `PV 0, SE 0` (or `None` — state which);
   a **single-trade position == that trade** exactly (the reduction anchor).
4. **Netting:** a long + the offsetting short of the **same** trade nets to `PV 0` (the sign companion from
   T82 at book level) — on a non-symmetric payoff so it is a real cancellation, not a symmetric triviality.
5. **Order independence:** `position_value` is invariant under trade order (the sum commutes) — assert it,
   and a mutation that (say) overwrites instead of accumulates dies.

## Degeneracy premises (§4 / §9, the R1-plan standing premises)
- The fixture **population includes ≥2 estimate pricers** (else the SE-combination oracle is vacuous — the
  T33b/T82 estimate-population hole; this is its first *intended* exercise).
- The aggregation fixture has **≥2 trades of different products** (a one-trade or single-product book hides a
  dispatch/accumulation error — the τ=1 family).
- The netting fixture's payoff is **non-symmetric** (T82 premise at book level).

## Provenance / gates
- Provenance headers; `fields` ≤5; `acyclic`/`layers` green (the package is mapped, per T82(b)). No
  `references.md` flip.
- Version → **v0.85.0**; floor derived from the collected count (T82's lesson — not hand-stamped); one
  `CHANGELOG` entry.
- **Status page:** Representation advances toward **2 of 5**.

## Out of scope (→ R1.3+)
- `PricingRequest` (R1.3 — resolving *which* model per trade from the market, and the valuation date);
  the MtM procedure + cashflow ladder (R1.4); the mixed-asset book capstone (R1.5).
- Correlated-error books beyond the stated independence assumption (if a later unit shares MC paths across
  trades, the linear-SE regime is its own note — `to_go_deeper` unless pulled).
- Multi-currency aggregation (FX, after R1 — a position is single-currency here; cross-currency netting is
  the test FX brings).
