# Artifact #12 — Domain Build Order (the migration method, corrected)

**Status:** Draft for ratification. **Supersedes demand-driven / retire-driven progress** as the
*method*. The spine (#02) stays as the **static structure**; this supplies the **build order** the
spine never specified.

---

## 1. Why the method changed

Two compounding errors produced a patchwork rather than a library:

**(a) The metric became the steering wheel.** `drawdown > 0` was the right *diagnostic* when
retirement was stuck at 0; it became the *objective*. Retire-reads then pulled the build wherever
the cheapest tick sat — some FX, some equity, XVA on a flat curve, five vanillas. Every slice was
individually clean and oracle-gated; the whole was incoherent. Nothing forces coherence when the
objective is "modules retired."

**(b) §6b (YAGNI / rule-of-two) was applied to *domain* decisions, where it does not belong.**
Rule-of-two is correct for *software abstractions* (no engine registry until the second engine). It
is **wrong** for *domain architecture* — "how many currencies," "single- or multi-curve," "how xccy
discounting works" are not discovered from a second consumer. They determine the shape of the curve
layer, the snapshot, discounting, and every product above. Deferring them produced single-curve,
effectively single-currency code **underneath an XVA stack** — debt at the foundation, which is why
it does not feel solid.

**The corrective:** settle domain architecture *up front*; migrate **block by block in
financial-engineering dependency order**; complete a block before opening the next.
**Drawdown becomes reporting, never steering.**

**Use the material.** The quarry already encodes answers (multi-curve solver, RFR bootstraps, xccy,
33 sovereign markets, conventions tables). Each block **mines the quarry's decisions** rather than
rediscovering them — that is the point of migrating rather than rewriting.

---

## 2. Ratified domain decisions (2026-07, Bernardo)

These are **architecture, not abstractions** — fixed now, not deferred to a consumer.

- **D1 — Curve framework: single *and* multi-curve from the start.** The curve container is
  **multi-curve-capable natively**: discounting curve + projection curve(s) per index. **Single-curve
  is the degenerate configuration** (projection ≡ discount), not a separate code path. One design,
  configurable either way; no retrofit.
- **D2 — Currencies: multi-currency *structure*, one currency populated.** Curves/snapshot keyed by
  currency from day one (structure supports N); only one currency (EUR or USD) populated until B3.
  No single-currency assumption may bake into products or engines.
- **D3 — xccy: designed-in at B1, built at B3.** The curve/discounting model accounts for
  **collateral currency / basis-adjusted discounting as a concept** from the start; the basis curves
  themselves are constructed in B3. Structure never needs rework; work is sequenced.
- **D4 — Product / Trade / Book** (A3, unchanged): `Product` = priceable atom (L2, pure data);
  `Trade` = collection of products + start date + lifecycle (L6); `Book` = trades.
- **D5 — Valuation engine** (A1, unchanged): `price(product, model, numerics)`; the model carries its
  snapshot; engines per family behind one facade. The **registry lands in B5**, where multiple
  methods per product genuinely exist (swaption analytic vs MC).

---

## 3. The blocks

A block is **complete** when: every quarry module mapped to it has crossed or is explicitly
classified; its products/curves are oracle-gated; `verify.py all` (incl. `layers`) is green; and the
block's domain decisions are exercised, not just declared.

| Block | Contents | Completion gate |
|---|---|---|
| **B0 conventions & foundations** | dates · calendars · day counts (full set) · schedules · roll/stub rules · currencies · `Money` · cashflow/leg · `RateIndex` | published convention test vectors (ISDA/ICMA) |
| **B1 market data & curves** | curve **framework** (discount + projection, currency-keyed, xccy-aware per D1–D3) · interpolation (pluggable) · snapshot · FX spots | curve invariants; single-curve == multi-curve degenerate case |
| **B2 linear product definitions** | deposit · FRA · future · swap · OIS · basis swap · bond · FX forward — **pure data, multi-curve aware** | each prices closed-form on a given curve set |
| **B3 curve construction** | bootstrap/solve using B2 as pillars · multi-curve solver · **xccy basis** · per-currency markets | reprice-to-par on every pillar; multi-curve consistency |
| **B4 models** | Black-76 · Hull-White/LGM · SABR · (jump/rough later) — **at L3, never L0** | analytic vs MC/PDE convergence |
| **B5 non-linear products + engines** | caps/floors · swaptions · FX/equity options · **engine registry** | closed-form vs numerical agreement |
| **B6 credit** | hazard/survival curves · CDS · credit products | par-spread reprices to zero; ISDA cross-check |
| **B7 risk** | greeks · scenarios → XVA · regulatory capital | analytic-vs-FD; the A6.1 measure oracle |
| **B8 portfolio & lifecycle** | trade · book · benefit table · P&L · persistence/data spine | realized + mark reconcile over a trade's life |

**Ordering note:** B2 before B3 is deliberate — curves are bootstrapped *from* linear instruments, so
the product *definitions* (pure data, L2) must exist before curve construction consumes them. This is
the domain order; it is consistent with the spine (products never price themselves).

---

## 4. What this means for the existing tree (honest)

The ng tree is **ahead of its blocks in places and unfinished beneath them**. Roughly 54 modules
exist, but B0/B1 are incomplete and single-curve while B7 (XVA/capital) is deep.

- **Not wasted:** the spine, vocabulary, engine contract, `Priceable`, the shell, the oracle
  patterns, and the property-based serialisation sweep all hold and carry forward.
- **To rebuild properly:** **B0 and B1** — full conventions; the multi-curve-capable,
  currency-keyed, xccy-aware curve framework (D1–D3).
- **To re-base:** **B2** products become multi-curve aware; **B4** HW already de-flattened (good);
  **B7** XVA/capital gets re-based onto real curves once B1–B4 are solid. Its *oracles* survive.
- **The 8 retires stand** — those quarry modules remain superseded; their ng counterparts simply get
  better (and deferred multi-curve obligations get discharged in B2/B3).

Paying this at ~8/768 is far cheaper than at 100/768. That is the argument for correcting now.

---

## 5. Method per block
1. **Map** — list every quarry module belonging to the block (from `quarry_reconciliation.md`), and
   order them by dependency *within* the block; identify the independent/leaf modules first.
2. **Mine** — read the quarry's implementation for the domain decisions it already encodes
   (conventions tables, multi-curve solver, xccy). Adopt what is right; realign what is not.
3. **Build** — bottom-up inside the block, copy-ADAPT, each slice oracle-gated, ≤5 args/fields,
   provenance headers, spine-conformant placement.
4. **Retire** — consumer-analysis retire-read per module; tick what is deletable; forward-link
   deferrals. Drawdown is *reported*, not chased.
5. **Checkpoint** — the block boundary is a checkpoint (cadence #11 still applies: ≤6 slices, or the
   boundary, whichever first), carrying the five review inputs.

---

## 6. Immediate next
**B0 + B1**, in that order — audit what exists, complete the conventions set, then build the
multi-curve-capable, currency-keyed, xccy-aware curve framework. No work above B1 until B1's gate is
green. The spine-conformance correction (`rulings_spine_conformance.md`) folds into B0/B1 as it is
the same territory.
