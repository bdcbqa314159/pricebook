# T82 — R1.1: the `Trade` type (R1 representation & MtM begins)

**Cowork spec, 2026-11-26.** The first representation tranche of the structural track's R1 arc. A `Trade` is
the smallest piece of "a book": a product + how much of it + which way. Architecture-first — the oracle is
the verified catalogue (a trade's value reduces to `price(product, model)`), not a held source.

## Target
A frozen `Trade` (a product + notional + direction) and a pure function
`trade_value(trade, model) → PricingResult` whose value reduces exactly to the underlying pricer:
`trade_value = direction_sign · notional · price(product, model)`.

## Pieces, layer, reuse
- **`Trade`** (new, **L2-level frozen representation data** — describe, never price; **not L6**, which is
  stateful) — fields: the **product** (any existing L2 product), a **notional** (`Money`, L0), and a
  **direction** (long/short — an enum or a signed marker). ≤5 fields (§7); if trade-level economics grow
  (trade date, id) bundle them into a value object rather than widen. The `Trade` does **not** price itself
  (the §2 L2 contract); it is pure data.
- **`trade_value(trade, model)`** (new, **L4-level pure function** — it consumes the registry) — dispatches
  `price(trade.product, model)` and applies the direction sign and notional, projecting to a `PricingResult`
  (SE scales with notional too, where present; §3). **Reuses the registry and every existing engine
  untouched** — this is the whole point of building above the pricers. Returns `PricingFailure` faithfully
  if the underlying pair is unregistered (the §3 contract rides through).
- Build confirms the exact module/dir and that `verify.py acyclic/layers` stays green (the dependency points
  DOWN into L2/L4; nothing above is imported).

## Oracles (RED-first, §4) — anchored on the catalogue
1. **The reduction identity, pointwise:** `trade_value(long 1-unit trade, model) == price(product, model)`
   exactly (a long unit trade is the bare price — bit-identical, no reordering). The foundation anchor.
2. **The long/short asymmetry on a NON-SYMMETRIC payoff:** `trade_value(short, model) ==
   −trade_value(long, model)`, exercised on a product whose value is **not** symmetric about zero (a call,
   not a forward struck at the money), so `−long ≠ long` is a real check and a dropped/!wrong sign is caught
   (the degeneracy premise below).
3. **Notional scaling — and its absolute companion (T43).** `trade_value(2N) == 2·trade_value(N)` is
   necessary but **scale-invariant**, so it pins proportionality and says nothing about the level; add one
   oracle asserting the **absolute** value against an independent reference (`N · price(...)` computed
   independently of the trade path), so a uniform mis-scaling of the notional cannot survive (the T43
   shared-scale-invariance trap).
4. **Failure rides through:** an unregistered `(product, model)` pair yields `PricingFailure` from
   `trade_value`, not a raise or a silent 0 (the §3 contract is preserved across the wrapper).

## Degeneracy premises (§4 / §9)
- The long/short fixture's product is **non-symmetric** (a short ≠ a sign-flip of a symmetric payoff) — else
  oracle 2 is vacuous (the T43/degenerate-fixture family: the fixture must make the sign do real work).
- The notional fixture uses a **non-unit, non-round notional** (not 1, not a power of ten that hides a factor)
  so the scaling and the reduction identity are distinguishable.
- A **direction mutation** (drop the sign, or always-long) must be killed by oracle 2 — assert it (the
  mutation-as-proof rule).

## Provenance / gates
- Provenance headers on the new modules (`stock: none` — architecture-first; `source: CLAUDE.md §0/§1
  representation`; `oracle: tests/...`; `slice: T82`). No `references.md` flip (no held source). `fields` ≤5,
  `acyclic`/`layers` green.
- Version → **v0.84.0**; one floor bump; one `CHANGELOG` entry.
- **Status page:** the generated inventory gains the first representation row (a `Trade` type now exists) —
  the scoreboard's representation column begins to fill, as intended.

## Out of scope (→ R1.2+)
- `Position`/`Portfolio` (R1.2 — a collection of trades; the aggregation identity), `PricingRequest` (R1.3),
  the MtM procedure + cashflow ladder (R1.4), the mixed-asset book capstone (R1.5). T82 is the single trade.
- Multi-currency (FX, the literature arc after R1 — a `Trade` is single-currency here; `Money` carries the
  currency but cross-currency netting is not modelled yet).
- Greeks (R2), scenario (R3), lifecycle/fixings (R4 — the genuinely stateful L6 tier).
