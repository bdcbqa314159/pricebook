# Arc plan — R1: representation & mark-to-market (the first STRUCTURAL arc)

**Cowork, 2026-11-26. ADOPTED — the two-track amendment is ruled; the structural track is open.**
**The track opens with T81 — the live status instrument** (`docs/tranches/81_status_instrument.md`): a
tooling-emitted `docs/STATUS.md` scoreboard (the `mutation_sweep.py` instrument pattern) that the structural
track then fills column by column, R1→R5. **R1 proper begins at T82** (the `Trade` type below). The first structural arc — architecture-first, not
book-grounded: it builds the §0-promised representation layer *above* the existing pricers and does not
rewrite any of the 45 engines. Oracle set = the existing verified catalogue (reduce a position's value to
`price(product, model)` and its sum).

## The gap it closes
Today there is no `Trade`, `Position`/`Portfolio`, or `PricingRequest` type, no calendar/business-day engine,
and schedules are half-unified. You can price a *product*; you cannot represent or value *a book*. This arc
makes "value and risk my position" expressible — the thing that turns the catalog-above-the-pricers into a
library from the top.

## Target
`value(position, market) → PositionValue` — a book of trades marked to a `MarketSnapshot`, producing PV, a
per-trade breakdown, and a cashflow ladder, by dispatching each trade through the existing registry and
aggregating. End state: value a small **mixed-asset book** (a swap + a swaption + a CDS) in one report.

## Layer — corrected (the representation is FROZEN data, not L6)
`Trade`/`Position`/`PricingRequest` are **frozen representation data at the L2 level** — they *describe,
never price themselves* (the L2 contract), composing products; they are **not L6**, because L6 is *stateful*
("remembers") and these are frozen. The **MtM procedure is a PURE function at the L4 level** — it dispatches
each trade through the registry and aggregates, no state (the functional core, like every engine). **L6
enters only at R4** (lifecycle — fixings as they set, rolls, the trading cycle; that is the genuinely
stateful shell). Dependencies point DOWN into the existing pricers; the engines are consumed, not touched
(`verify.py acyclic/layers` stays green — the build confirms the exact module placement).

## The tranche spine (~5–7 tranches, each oracled against the catalogue)
- **R1.1 — `Trade`** — a product + notional + direction (buy/sell) + trade-level economics. Frozen (§2
  products are frozen; a trade is too). **Oracle:** a `Trade`'s value = `±notional × price(product, model)`;
  the sign/notional conventions pinned by a long/short reconciliation (a short's value is the negated long).
- **R1.2 — `Position` / `Portfolio`** — a frozen collection of trades. **Oracle:** portfolio value = the sum
  of its trades' values (the aggregation/linearity identity, exact); an empty portfolio is 0; netting
  conventions (two offsetting trades net to ~0 up to bid/offer, which this layer does not model yet).
- **R1.3 — `PricingRequest` / `Valuation`** — "value *this position* as of *this date* under *this market*"
  (§0's missing half). **Oracle:** a one-trade request == `price(product, model)`; the request resolves the
  correct snapshot per trade (a multi-asset book reaches each asset's curves/surfaces); valuation-date
  handling reuses invariant-6 (a settled trade is PV 0, §3).
- **R1.4 — the mark-to-market procedure** — dispatch each trade, aggregate, emit PV + per-trade breakdown +
  cashflow ladder. **Oracle:** MtM == the manual sum of `price()` calls (a reconciliation identity, exact);
  the cashflow ladder ties to each product's own schedule. **The imperative shell** — it orchestrates, the
  pure engines compute (§3 unchanged).
- **R1.5 (capstone) — the mixed-asset book** — value a swap + swaption + CDS book end-to-end in one report.
  **Oracle:** the book PV == the sum of the three independently-priced trades; a scenario preview (reprice the
  book under one shocked snapshot) as a seam to R3, not built here.

*(The schedule/calendar rework — the ~15 per-product schedulers consolidated + the missing holiday/business-
day engine — is its own arc, R-track, after R1: MtM runs on the existing per-product schedules, so it is not
blocked on the rework. Recorded so R1 stays scoped to representation, not foundation-tidying.)*

## Reuse / what it does NOT do
- Reuses the **registry** (dispatch is already `(product, model) → pricer`), every existing engine, the
  `MarketSnapshot`, `PricingResult`, invariant-6. The new code is representation types + an orchestration
  shell.
- **Does not** compute greeks (R2), shock/scenario beyond a preview (R3), handle lifecycle/fixings-rolls
  (R4), or add FX/multi-currency (the literature arc after R1 is where multi-currency first tests the
  representation — build R1 single-currency, let FX prove it generalizes).

## Degeneracy premises (§4 / §9)
- The aggregation oracle is **non-degenerate**: the book has **≥2 trades of different products** (a one-trade
  or single-product book hides a dispatch/aggregation error — the τ=1-family shape: the fixture must make the
  sum do real work).
- The long/short oracle uses a **non-symmetric payoff** so `−long ≠ long` is a real check.
- A **multi-asset** MtM fixture (so "resolve the right snapshot per trade" is exercised — a single-asset book
  never tests cross-asset snapshot resolution).

## Gates / provenance
- `provenance` headers on the new L6 modules (`stock:`/`source: CLAUDE.md §0 representation`/`oracle:`/
  `slice:`); `acyclic`/`layers` green (L6 depends down only); `fields` ≤5 (bundle trade economics into a
  value object if over). No new source (architecture-first); no `references.md` flip.
- Version bumps per tranche; re-audit #10 at the arc close (then R2 — greeks).
