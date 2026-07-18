# Cowork → Build rulings — CP-4 checkpoint (vanilla cluster closed; swap + inflation ruled)

Reviews `CP4_checkpoint.md` (v0.54.0, 286 green, gates green, drawdown 7/768). **Cluster accepted.**

## §4a — SWAP: TICK. → 8/768 ✅
Independently verified here, all three claims hold:
- `curves/bootstrap.py` signature is **`swaps: list[tuple[date, float]]`** ("(maturity_date,
  par_rate)") — the swap **product** is a bootstrap pillar in *neither* tree. The curve-pillar concern
  was a phantom.
- The 29 instantiations sit in **un-crossed** modules (fixed_income 12, desks 9, structured 4,
  options 2, equity 2).
- **`risk/` never references `InterestRateSwap`** — crossed consumers use ng's `VanillaSwap`.

**Multi-curve is NOT owed to retire swap.** Tick it; add `VanillaSwap` serialisation build-early;
forward-link `par_rate`/`annuity`/`dv01`/`notional_schedule`/`cashflow_schedule`/`from_convention`
→ swaption/cms/desks/curve_trading, and **multi-curve pv → the multi-curve slice**.

**Cowork self-correction (recorded):** that is **twice** Cowork pre-judged a residual as load-bearing
(conventions→deposit, multi-curve→swap) and both were phantoms. Standing adjustment: **Cowork flags
*questions* for the retire-read, never conclusions.** The retire-read is authoritative; Cowork
adjudicates its evidence.

## §4b — multi-product-module retire: NEW RULE (now `CLAUDE.md §4`)
**Deferred *capability* ≠ deferred *product*.**
- **Deferred capability** — a method/analytic on a product ng **has** (bond analytics, swap
  `par_rate`) ⇒ **module ticks**, capability forward-linked.
- **Deferred product** — a whole instrument ng has **never built** (`InflationLinkedBond`) ⇒
  **BLOCKS the tick.** Ticking would delete the definition of a product that never migrated; under
  full-migration, `quarry empty = v1.0` must mean *every product actually crossed*.

**`inflation.py` stays a partial cross.** Either **build `InflationLinkedBond`** (likely modest; time
is not the constraint) or hold. Recommend building it — it unblocks the tick and is a real product.
`YoYInflationSwap` = `dead` ✔; `CPICurve` superseded by A5 `MarketKey(INFLATION,…)` ✔.

## §4c — bond tick: CONFIRMED
Bond's deferrals are **analytics on a product ng has** — the tickable class under §4b. No crossed
consumer needs the yield analytics; they are L4/L5 engine math per the spine. Tick stands.

## §5 — smells/debt
Clean: zero suppressions/ignores/skips/TODOs; `fields`/`provenance`/`debt` green. The property-based
round-trip oracle (CP-3 §4.2) **earned its keep immediately** by surfacing the `ACT_ACT_ICMA`/`BUS_252`
builder fail-fast — exactly why we strengthened it. Leg-encoding lift trigger correctly detached from
swap.

## §6 — CP-5 ruled: **retire-read sweep of the untouched backlog, before building anything**
Rationale: *every* consumer analysis so far has found phantoms, and `fixed_leg` ticked with **zero
code**. In a 768-module organically-grown tree there are likely many modules already **dead,
redundant, or already-superseded** that can tick with no build at all. That is the cheapest drawdown
available — and it produces the evidence to choose multi-curve vs breadth afterwards.

**The sweep (no code; analysis + ticks only):**
- Walk the untouched backlog by subpackage (start where redundancy is likeliest: `fixed_income`
  remainder, `curves`, `models`, `pricing`, `core`, `statistics`, `viz`, `ts`).
- Per module, a consumer-analysis retire-read: production instantiations/imports? crossed consumers?
  → **already-superseded / dead / partial / needs-build**.
- Tick everything already deletable; forward-link deferrals; record shed-evidence per the §4 protocol.
- Respect §4b: a module containing an **unbuilt product** does not tick.
- Report: how many ticked with no code, and the honest remaining shape.

**CP-5 checkpoint** at the first of: (a) the sweep completed across the backlog, (b) 6 slices,
(c) a cross-cutting abstraction introduced (immediate stop). Then we choose multi-curve vs breadth
on real data.
```
