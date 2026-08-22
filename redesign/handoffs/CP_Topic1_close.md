# Checkpoint — C3 cluster close + TOPIC 1 CLOSE (the first complete vertical, L0→L6)

**Version:** v0.98.0 · **Slice:** `slice/12-l6-realized` · **Baseline:** v0.97.0.

This slice lands `BookedTrade` + benefit table + realized P&L, **closes the C3 cluster**, and with it
**closes Topic 1** — the first complete instrument vertical proven end-to-end through all seven layers.
Two review inputs: the C3 cluster close, then the Topic-1 close retrospective.

---

## Part 1 — C3 cluster close (trade / portfolio / risk)

C3 delivered: L5 risk greeks (DV01+vega on `Priceable`), L6 frozen shell (Trade/Book + marking +
portfolio DV01), L6 stateful shell (BookedTrade + benefit table + realized/total).
- **The spine's two structural fixes for C3 both held:** risk at L5 on a `Priceable` protocol (no
  isinstance-on-instrument), and the shell at L6 calling the core (no `pv()` on products). The engine
  registry owns all product dispatch; risk and the shell are type-blind.
- **`Total = realized + mark` is now real** (§2): the engine computes the mark (future PV, invariant 6
  excludes historical), the shell remembers realized (undiscounted benefit table). The last unexercised
  engine invariant (6) is now exercised.
- **Oracle quality:** DV01/vega vs closed form (<1e-6); shell additivity exact; realized vs a hand-computed
  benefit table (<1e-9); the spot degenerate keeps every prior slice byte-identical.

## Part 2 — Topic-1 close retrospective

### The milestone
**Topic 1 is the first complete vertical: a full instrument lifecycle priced and risked end-to-end
through L0→L6.** From foundation conventions (L0) → market snapshot + curves + surfaces (L1) → products
(L2) → calibrated models + vol (L3) → the stateless engine (L4) → generic greeks (L5) → the imperative
shell that books, marks, and remembers realized P&L (L6). Every layer is now load-bearing on a real path.
This is the proof that the ratified spine *works* as a whole — not just per-layer.

### Invariant 6 — the seasoned-trade misprice, closed
Before this slice the atoms summed ALL periods; a seasoned trade (past pay date) silently mispriced
(negative t → `df > 1` extrapolation, or a raise). Now `future_periods` pre-filters historical periods
ABOVE the atoms (which are UNCHANGED — §3d preserved), the engine marks future-only, and the shell records
the past as realized. **The pre-filter is a no-op for spot trades → slices 1–11 are byte-identical** (full
suite 234, the spot subset unchanged).

### The parking event (Topic-1 close, one git-mv)
The 6 accumulated Topic-1 deletables were **physically parked** into `parked/topic-01-yield-curve/` in one
event (mirroring the Topic-0 close): `bootstrap`, `discount_curve`, `ncurve_solver`, `global_solver`,
`multicurve_solver`, `forward_interpolation`. **Drawdown 19/793 — now ALL physically parked** (13 Topic-0 +
6 Topic-1). The numerator is unchanged this slice (the booking cluster stays partial — see below); the
parking is the physical realization of already-counted deletables. ng is unaffected (it never imports the
quarry; the merge gate never touches `python/`) — verified: all `verify.py` gates + full suite green after
the git-mv.

### Drawdown (§4) — booking cluster retire-read
`core/trade.py`/`core/book.py`/`core/fixings.py`: the realized-vs-mark / benefit-table CONCEPT crosses (ng
`BookedTrade`/`realized`/`total` + invariant-6 exclusion; `Trade.pv(ctx)` realigned to `mark`). But the
files stay resident and **do NOT tick** — serialisation, `Desk`, limits/positions/netting, lifecycle events,
the mutable fixing store + file I/O, and ~10 un-crossed desk consumers (`desks/*`, `credit_risk`). Partial.

### Challenge-me (Topic-1 as a whole)
- **The benefit table dispatches via a registry** (`register_benefit`), mirroring the engine registry — the
  shell stays isinstance-free (a `cast` after registry dispatch, not an isinstance ladder). Only `VanillaSwap`
  is registered; other products register their benefit with their slices.
- **`realized` supports the swap only** this slice — the oracle product. Caplet/swaption realized (exercise
  cash) travels with lifecycle events. Honest scope, forward-linked.
- **Single-currency** — `Money.__add__` guards mixing; cross-currency realized (FX-to-base) deferred.
- **Accrued/clean-dirty deferred** — this slice is realized + future-PV mark; accrued (the engine's
  current-period slice) + clean/dirty + fixings-on-snapshot is the paired follow-up.

### Smell + debt scan
`verify.py` acyclic/fields/layers/provenance/debt all green. Field discipline: `BookedTrade` 2 fields. No new
suppressions (the frozenness test uses `setattr`, not `# type: ignore`). Exception-count (§3d): zero
isinstance-on-product; the benefit registry is structural dispatch. The atoms were not touched (invariant 6
lives in the pre-filter, above them).

## Deferred (named triggers)
Accrued + clean/dirty + fixings-on-`MarketSnapshot` (paired follow-up) · lifecycle events
(amend/exercise/novate → quarry `trade_lifecycle`) · booking persistence/DB · daily P&L / attribution
(quarry `daily_pnl`) · cross-currency benefit table · physical-delivery cashflows · caplet/swaption realized ·
`pe/` modules · the resident booking breadth (`Desk`/limits/positions/netting).

## Named next checkpoint
**Topic 2 opening** — the next asset/topic vertical (credit? inflation? a deeper rates topic?), or the
paired **accrued/clean-dirty** follow-up that completes the reporting split. Topic selection is a Cowork
decision (doc 12 domain build order). Checkpoint at the first of ≤6 slices or the next topic boundary.
