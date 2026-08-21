# Checkpoint — L6 opening (the imperative shell) · ALL SEVEN LAYERS LIVE

**Version:** v0.97.0 · **Slice:** `slice/11-l6-shell` · **Baseline:** v0.96.0 (L5 risk landed).

**Milestone.** L6 is the last spine layer. With it, **every layer L0–L6 now has code** — the full
"functional core (L0–L5), imperative shell (L6)" spine is realised. New layer → checkpoint.

## 1. The shell/core boundary — proven, no leak
The shell holds descriptions and CALLS the core; the core never remembers, the shell never computes.
This slice lands the FROZEN half:
- **`Trade`/`Book` are pure data with NO `pv()`** — the quarry's `Trade.pv(ctx)` self-pricing is
  **realigned** into the shell function `mark()` (products stay pure data, CLAUDE.md §2/§3). The stop
  condition "marking requires a `pv()` on a product/trade" did NOT trigger.
- **`mark` is the shell calling the engine** — `Σ engine.price(product, model).pv`, type-blind via the
  registry. The stop condition "an isinstance on product type" did NOT trigger — a MIXED swap+caplet book
  marks under ONE `BlackModel` in a single pass, purely through registry dispatch.
- **Portfolio risk = Σ of the L5 greek** — `book_priceable` adapts a `Book` to the L5 `Priceable`, so every
  greek works on a portfolio for free; `book_dv01` == Σ per-product `ir_delta` (linearity). No new risk code.
- **No ratified L2/L4/L5 type's meaning changed** — the shell depends *down*; nothing below L6 was touched.
- **`acyclic` confirms L6 is the top leaf** — imported by nothing (the mirror of the L5 check).

## 2. Oracle-quality audit
| Oracle | Class | Result |
|---|---|---|
| Trade-of-1 == engine: `mark(Trade((p,), s), model) == price(p, model).pv` | identity (thin pass-through) | exact |
| Book == Σ trade marks (and == raw engine sum) | additivity | <1e-12 |
| Portfolio DV01 == Σ per-product `ir_delta` | greek additivity across the shell | <1e-12 |
| Frozen/immutable: `Trade`/`Book` frozen; marking mutates nothing | invariant 3 | asserted |
| Failure-as-value: an unpriceable product → `mark`/`mark_book` return `PricingFailure` | invariant 4 | asserted |
| **Mixed book under one model** (swap+caplet, one `BlackModel`) | single-model ≠ homogeneous | passes |

The mixed-book oracle is the load-bearing one: it proves "single model" is not "single product type" — one
rich model (curves + surfaces) marks heterogeneous products via dispatch, with zero product introspection.

## 3. Challenge-me
- **`book_dv01` gained a `key` parameter** vs the ratified `book_dv01(book, model)`. DV01 must bump a
  *specific* curve, and the signature couldn't say which; I added `key: CurveKey` — the same shape as L5
  `ir_delta(p, market, key)`, so it's consistent, not surprising. *Flagged:* if a keyless "total parallel
  DV01" (bump every curve) is wanted instead, that's a distinct greek (an all-curves `Bump`) — deferrable.
- **`book_priceable` makes a `Book` a first-class `Priceable`** — arguably the more fundamental primitive
  than `book_dv01` (every L5/future greek works on a portfolio through it). `book_dv01` is a thin convenience
  over it. Kept both: the adapter (general) + the named DV01 (the ratified ask).
- **Single-currency marking** — `Money.__add__` guards currency-mixing, so a mixed-currency book raises today.
  Cross-currency aggregation (FX-to-base) is deferred to its first consumer (named).
- **`mark` is the future PV only** — accrued/realized/clean-dirty split is the stateful half (next slice);
  for a spot-started trade with no past cashflows this IS the full mark (invariant 6 excludes past flows upstream).

## 4. Smell + debt scan
- `verify.py` acyclic/fields/layers/provenance/debt all green. Field discipline: `Trade` 2 fields, `Book` 2.
  No new suppressions — the frozenness test uses `setattr` (runtime `FrozenInstanceError`) rather than a
  `# type: ignore`, keeping `debt` balanced.
- Exception-count (§3d): zero isinstance-on-product in the shell; the only isinstance is failure-as-value on
  `PricingFailure`. `Product = object` alias keeps the shell type-blind (the registry owns dispatch).

## Drawdown (§4) — 19/793, tick 0 (PARTIAL cross)
`core/trade.py`/`core/book.py`: the frozen-description + mark + DV01 concept crosses (realigned), but the
files stay resident — serialisation, mutable `Book.add`, `Desk`, `Position`/`BookLimits`/limit-breaches,
tenor-bucketed DV01, counterparty netting, AND ~10+ un-crossed desk consumers (`desks/*`, `credit_risk`).
Realized-P&L/benefit-table/`BookedTrade` → next slice. Partial → tick 0.

## Deferred (named triggers)
`BookedTrade` + benefit table + realized P&L (the STATEFUL half → next L6 slice, closes C3/T1) ·
accrued/realized/clean-dirty L6 reporting split · per-product model resolution / multi-market heterogeneous
books · lifecycle events (fixings/exercise/settlement) · cross-currency book aggregation (FX-to-base) ·
netting/collateral/limits/desks at book level · keyless total-parallel `book_dv01` · Trade/Book serialisation.

## Named next checkpoint
**The stateful L6 slice** — `BookedTrade` + benefit table + realized P&L (the "shell remembers realized
cash" half) — which closes the C3 trade/portfolio/risk cluster and the T1 arc. Checkpoint at the first of
≤6 slices or that cluster boundary.
