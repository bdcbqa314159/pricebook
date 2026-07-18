# CLAUDE.md — Pricebook build guardrails

These are **guardrails, not suggestions.** They are the ratified output of the design
phase (`redesign/01`–`05`). When a change conflicts with this file, this file wins.
The design *rationale* lives in `redesign/`; this is the enforceable subset.

The build proceeds by **migration, one thin slice at a time, bottom-up**, starting with
Slice 0. Do not rewrite; do not chase feature parity; do not price outside the engine.

---

## 0. The one idea

**Functional core, imperative shell.** Pricing is a pure function
`price(product, model, numerics) → result` — no state. The **model is calibrated to a
`MarketSnapshot` and carries it**; market is upstream of the model
(`market → calibrate → model → price`), never a separate peer input to the engine. Linear
products use a thin `DiscountingModel` that wraps the curve. Everything that persists (a
booked trade, its life, the database) lives in a thin shell around the core and only calls
into it. The core never remembers; the shell never computes.

---

## 1. The spine (layers — dependencies point DOWN only, never up)

```
L6  LIFECYCLE / BOOKING / PORTFOLIO   (stateful shell: trade=book of products, benefit table, P&L, desks)
L5  RISK & CAPITAL                    (greeks · XVA · RWA — on the engine + Priceable protocol)
L4  ENGINES                           (the stateless heart: bind product+model → mark = future PV + accrued)
L3  MODELS + CALIBRATION              (dynamics; calibrated to a MarketSnapshot, carry it)
L2  PRODUCTS                          (pure-data product descriptions: legs, cashflows, payoffs)
L1  MARKET DATA                       (immutable MarketSnapshot · quotes · fixings · curves · vols)
L0  FOUNDATION                        (time & conventions · value types · finance-free numerics)
     DATA SPINE (side)                (SQLite→DuckDB ingestion → feeds L1; never imported by core)
```

**The law:** a lower layer must never import a higher one. This is enforced by the
per-commit acyclic check (Tarjan SCC on the import graph) — keep it green. If a change
needs an upward import, the design is wrong; fix the layering, not the check.

**Three structural fixes to honour as entries migrate:**
- `risk` lands at **L5** (above the engine), depends only on the engine + a `Priceable`
  protocol — **no `isinstance`-on-instrument ladders.**
- Calibration lands as a **unified front at L3** over per-family solvers.
- The engine registry lands **inside L4** (no leaky top-level `registry.py`).
- `pe` lands in the **L6 shell** (not L0); `Cashflow` lands at **L0** (not inside an asset class).

---

## 2. The stateless-engine contract (L4) — five invariants

```python
result = engine.price(instrument, model, numerics)  # -> PricingResult
```
The **model carries the `MarketSnapshot` it was calibrated to** (`model.market`); the
engine reads curves/vols through the model, never from a separate `market` argument. For
linear products the model is a `DiscountingModel` wrapping the curve. Market is never a
peer input — this makes model/market mismatch structurally impossible.

1. **Referential transparency** — identical inputs ⇒ identical output, always.
2. **No ambient state** — no thread-locals, globals, or clock reads; "today" is
   `model.market.valuation_date`.
3. **No mutation** — `instrument`, `model` (and the `MarketSnapshot` it carries) are
   frozen; never mutated in place.
4. **Failure is a value** — return `PricingFailure`, never raise-and-hope or emit silent
   `NaN`.
5. **Config is explicit** — all reproducibility knobs (seeds, MC paths, PDE grid, tol)
   arrive in `NumericalConfig`; never a hidden default.
6. **Valuation-date-aware** — cashflows on or before `valuation_date` are historical:
   **excluded from PV** (handled by the L6 shell via fixings/settlement), never discounted
   with a non-positive `t`. Future cashflows discount from `valuation_date`. Clean/dirty
   and accrued interest are explicit, not incidental.

**Products are pure data.** Frozen dataclasses that *describe* (legs, cashflows); they
do **not** price themselves. No `pv()`/`pv_ctx()` on products — pricing lives in L4.

**Trade vs mark vs realized.** A **trade** (L6) holds a *collection of products* + a start
date + lifecycle. At the valuation date its economics split three ways: **realized P&L**
(cashflows that already paid — the **benefit table**, actual cash, recorded in the L6 shell,
never discounted), **accrued** (earned-but-unpaid slice of the current period — part of the
*mark*), and **future PV** (remaining flows, discounted). The **engine computes the mark**
(future PV + accrued); the **shell remembers realized P&L**. Total economics = realized +
mark; dirty = clean + accrued.

---

## 3. Vocabulary (the stable types every layer speaks — ratified)

- **Time:** `Date` (stdlib), `DayCountConvention`+`year_fraction`, `Calendar`,
  `Frequency`/`Schedule`, `CompoundingMethod`, `RateIndex`. `Tenor` stays a string +
  helpers; `Rate` stays a float.
- **Money:** `Money(amount, Currency)` **at boundaries** (cashflows, PVs); plain floats
  inside hot numerical loops. Currency-mixing must be a type error where it matters.
- **Instrument atom:** `Cashflow` (at L0), `Leg` = ordered cashflows + convention.
- **Market (L1):** `Quote`/`QuoteId`/`QuoteKind`, immutable `MarketSnapshot` (carries the
  `valuation_date`). **All market data is first-class in the snapshot** (rule: *if risk bumps
  it, it lives in the snapshot*), shaped as a **keyed registry** (Amendment A5): the home
  `discount_curve` stays a named field (numeraire), everything else lives in
  `curves`/`spots`/`vols` maps keyed by `MarketKey(asset: AssetClass, id: str)` — survival,
  dividend, foreign-discount, FX/equity/commodity spots & vols. A new asset class adds *keys,
  not fields*, and greeks are generic (`bump_spot`/`vol_vega`…), not per-asset. Curves are
  reached through a **`CurveHandle`** protocol (`df`, `survival`) — depend on the capability,
  not the concrete curve; **never mutated in place.** A curve/model may expose closed-form **building blocks** (`df`, `RPV01`, `B(t,T)`,
  zero-bond-option) reused upward; the **L4 engine composes them to price the product** —
  "pricing lives in L4" governs *product* pricing, not every scalar of math.
- **Models (L3):** a model is a **`CalibratedModel` bound to the `MarketSnapshot`** it was
  calibrated to (`model.market`). `DiscountingModel` wraps a curve for linear products. The
  engine depends on the model; market is reached through it, never passed alongside.
- **Time semantics:** valuation partitions cashflows into historical (`date ≤
  valuation_date`, excluded from PV) and future (discounted from `valuation_date`).
  Clean vs dirty price and accrued interest are explicit value concepts, not by-products.
- **Engine I/O:** `NumericalConfig`; `PricingResult` is a **decomposition** (dirty PV +
  cashflow/accrual breakdown ⇒ clean, accrued; plus sensitivities, diagnostics), not a
  scalar; `PricingFailure`. The economy a model is built on = **curves + `FixingHistory`**
  (fixings are first-class in the `MarketSnapshot`; the core needs them to resolve
  current-period amounts).
- **Identity/state (Product → Trade → Book):** `Product` = the priceable atom (**frozen**,
  L2, needs a model). `Trade` (**frozen** description, L6) holds ≥1 products + a start date;
  `BookedTrade` = trade + lifecycle events + its **benefit table** (realized-cash P&L).
  `Book` = collection of trades (L6). `PricingContext` is **frozen** and kept as the
  engine's built-market-state input (reached through `model.market`).

---

## 3b. Signature discipline

- **Max 5 arguments** per function/method in `src/pricebook_ng/**` (ruff `PLR0913`,
  `max-args=5`; the quarry `python/pricebook/**` is exempt).
- A signature over the limit is **un-bundled vocabulary**, never a reason to raise the
  limit. The fix is ALWAYS to group cohesive parameters into a frozen value object, or
  fold them into an existing ratified type: market state → `MarketSnapshot`;
  reproducibility knobs → `NumericalConfig`; `amount`+`currency` → `Money`; ICMA coupon
  anchors → `CouponPeriod`; calendar/convention/eom → `RollRule`; schedule terms →
  `ScheduleTerms`.
- New grouping types obey the rule of two (≥2 real consumers) — no speculative wrappers,
  and no 9-field "god dataclass" that just relocates the smell.
- **Never suppress.** No `# noqa: PLR0913`. A genuinely irreducible mathematical
  signature (a closed-form formula) is the sole exception and goes in `OPEN.md` with a
  rationale and a re-open trigger.
- **The limit also applies to dataclass fields** (products & value types), because a frozen
  product with 8 loose primitives is the same un-bundled-vocabulary smell as an 8-arg
  function — `PLR0913` just can't see it. A `products/` or `foundation/` value dataclass has
  **≤5 fields**; bundle primitives into ratified value objects (`Money` for amount+currency,
  `Accrual` for start+end+day_count, `ScheduleTerms`, …). Enforced by **`verify.py fields`**.
  Legitimately-wide **aggregate / output-record / config** types (e.g. `MarketSnapshot` — the
  A5 shape, `XvaReport`) carry an explicit `# fields-exempt: <reason>` marker — an *explicit*
  exemption, never silent tolerance.

---

## 4. Migration rules (quarry → new tree)

- **Quarry is read-only.** The old tree is never edited or deleted. Done = quarry empty.
- **"Crossed" = quarry-deletable, not concept-adapted.** A quarry module counts as migrated
  only when its ng counterpart reaches *realigned parity* and the quarry module could be
  deleted — a simplified skeleton (e.g. flat-curve HW standing in for the general model) is a
  *partial* cross with a recorded **parity gap**, not a cross. Drawdown (`deletable / 768`) is
  the honest progress bar; it is refreshed at every checkpoint
  (`redesign/handoffs/quarry_reconciliation.md`).
- **Copy-ADAPT, never copy-paste.** Every crossing conforms to a layer, speaks the
  vocabulary, moves behaviour into the engine, or sheds debt. A byte-for-byte copy is a
  **failed** migration.
- **Green-oracle gate — nothing crosses grey.** No entry lands until a red/green oracle
  proves it against a known value: closed form > QuantLib/ISDA cross-check >
  self-consistency (reprice to par / zero NPV) > trusted mark. "Runs and looks right" is
  not an oracle.
- **Demand-driven (vertical).** Slices pull quarry entries as they need them
  (`ng-migration-mode`); an entry still lands only after everything it depends on has landed,
  but the quarry empties by demand, not by exhaustively finishing a layer first. Progress is
  tracked via each build report's ledger-deltas table.
- **Provenance.** Each landed entry records: quarry path, the paper/book/model it
  implements, and its oracle. (Educational constraint — keep it legible.)
- **Deletable-bar rigor.** A parity slice ends by **reading its quarry counterpart end-to-end**
  and listing the residual gap in `quarry_reconciliation.md`. A module ticks to *deletable*
  (drawdown +1) **only when the genuine residual is empty** — never asserted from "looks covered."
- **"Deletable" = SUPERSEDE, not clone.** ng is deliberately minimal (copy-ADAPT sheds debt, §6b
  forbids speculative fields), so ng ≠ quarry by design. A quarry feature ng omits is *shed debt*,
  not residual — **but only with evidence.** Every omission is classified in the module's `shed:`
  list by grepping the quarry for consumers:
  - **`dead`** — no consumer anywhere in the quarry (incl. its tests) ⇒ genuinely shed, no obligation.
  - **`deferred→X`** — consumed by quarry module(s) X not yet crossed ⇒ shed now; the feature
    **travels with X's crossing slice** (a named future trigger, never a vague "later").
  - **`needed-now`** — an ng module already requires it ⇒ **not shed**; a genuine residual that must
    be built before the tick.
  A module ticks deletable only when the genuine residual is empty **and** every omission is
  classified with evidence. Cowork spot-checks `shed:` calls at each checkpoint and may **un-tick**
  (drawdown −1); the quarry is git-tracked, so a reversal is cheap and nothing is lost.
- **Evidence protocol for a `dead` claim** (a narrow grep produced a false negative on retire #1):
  search the **bare name** across `python/` source *and* tests — never just `\.name` — and explicitly
  check **constructor kwargs**, **dict/string keys**, `getattr`, `**kwargs` forwarding, and
  **serialisation round-trips**. Anything reachable dynamically is **not** `dead`. The shed-list
  records *how* it was verified (patterns + hit counts), not just the verdict.
- **Forward-link every `deferred→X`.** The obligation is written on **X's row** in
  `quarry_reconciliation.md` ("on crossing: add …"), not only on the retired module's entry — the
  retired entry is never re-read; X's row is read the moment X is picked up.
- **Retire flow (per module, just-in-time):** migrate → read the old module end-to-end → assess each
  omitted feature's status *in the quarry* (a fact, not a taste judgment) → **then** tick. The
  assessment is the evidence for the tick, so it completes before it.
- **Cross-cutting work is justified by what it retires.** Shared capabilities (conventions,
  multi-curve, serialisation…) may be built when residuals cluster, but **every such slice must
  tick ≥1 quarry module to deletable** — infrastructure is never built speculatively ahead of the
  module that needs it. Breadth (new asset classes) waits while drawdown is stuck at 0.

Per-entry checklist: `AUDIT → ALIGN → ORACLE → DEPS → PROVEN → PROVENANCE → MARK`.

---

## 5. Debt rule

**No debt is silenced silently.** Every suppressed warning, `# type: ignore`, empty
`except`, skipped test, shim, or load-bearing TODO goes into the debt ledger (`OPEN.md`)
with a rationale and a re-open trigger — or it does not exist.

- Debt may **never** buy a green oracle. A slice that can't go green honestly is not done.
- Debt is allowed only for *deferred scope*, never for *hidden wrongness*.
- **Invariant (CI-checkable):** suppressions − ledger entries = 0.

---

## 6. Slice discipline

- One **vertical** cut through only the layers it needs. Nothing speculative.
- **Ships with a named oracle**, green, before it counts as done.
- Small enough to check against its oracle **in one pass** — if it isn't, split it.

**Slice 0 (first build task):** a single fixed cashflow discounted on a flat curve, priced
end-to-end L0→L6 through the stateless engine. Oracle: `PV = notional·exp(−r·t)` to
<1e-12; analytic vs finite-difference DV01 to <1e-6; repricing is byte-identical
(statelessness). See `redesign/04_slice_plan.md`.

**Checkpoint cadence (hard stop — planned, not ad-hoc):** stop and write a checkpoint report
at the **first** of: **≤6 slices** since the last checkpoint, **or** a capability-cluster
boundary (asset class / risk family / layer vertical / calibration family) — **whichever comes
first**. Plus **immediate** stops on: any design drift or contract/measure question; any new
cross-cutting abstraction/vocabulary type; any debt logged or suppression added; any slice whose
best oracle is only self-consistency; any quarry entry that resists clean realignment. Every
report carries the four review inputs — **oracle-quality audit, quarry-drawdown reconciliation
(`N/768`), a "challenge-me" list of design choices, and a smell+debt scan** — plus the **named
next checkpoint**. Do not begin the next slice until the checkpoint is ruled. Full detail:
`redesign/11_checkpoint_and_review_cadence.md` (extends `08_handoff_protocol.md`).

**Migration stance:** full migration, corrected. Every quarry module crosses, *realigned* to
this design (copy-ADAPT, shed debt) — nothing archived-instead-of-migrated. `done = quarry
empty = v1.0`, taken literally; drawdown % is the progress bar.

---

## 6b. Simplicity — solve the real problem, not the imagined one

- Build for the problem the slice actually presents. **Nothing speculative.**
- **No abstraction without two present, real consumers.** Introduce it when the second
  arrives, not in anticipation (rule of two). One caller ⇒ write it concrete.
- **No hooks, config knobs, or extension points for hypothetical futures.** Every
  abstraction is complete and exercised by real code, or it does not exist. No half-built
  generality "for later" — a partial abstraction is debt, finished within the slice or
  not introduced.
- **Depth of abstraction is a cost, not a virtue.** Prefer concrete, legible code over
  configurable cleverness. The library is also a teaching text: legible beats clever.
- If a design adds a layer of indirection, it must justify it with a present need, not a
  possible one. When in doubt, do the simpler thing and let the second real use force the
  generalisation.

## 7. Release & versioning

- **New tree starts at `0.0.0`.** `0.x` = migration in progress; **`1.0.0` is reached
  exactly when the quarry is empty.** True semver after 1.0: MAJOR = public-API or
  serialisation-schema break, MINOR = new capability, PATCH = fix/numerical correction.
  `pricebook.__version__` is the single source of truth.
- **`RELEASE_NOTES.md` is frozen** (quarry record — read-only, never appended). The new
  tree logs to **`CHANGELOG.md`** (Keep-a-Changelog format).
- **Every landed slice = one version bump + one `CHANGELOG.md` entry**, carrying the same
  provenance the ledger requires (what changed, oracle, quarry path). CI asserts
  `__version__` matches the top changelog entry.

---

## 7b. Verification & audit (one tool, staged)

- The new tree's entire audit setup is a **single `verify.py`** — `acyclic`,
  `tests --layer N`, `debt`, `provenance`, `version`, `all`. No audit framework; a flat
  script of small functions. The quarry's `L*_DEPS.md`, regenerated `ARCHITECTURE.md`,
  and `AUDIT_PLAN.md` are **not carried forward** (frozen quarry history).
- **Staged tests:** each slice runs only its layer tier (`verify.py tests --layer <L>`),
  not the full suite. Merge gate = layer tier green + `acyclic` + `debt` + `version`.
  Full sweep runs nightly / at layer checkpoints.
- **Provenance header enforced:** every `src/pricebook_ng/` module carries four lines —
  `quarry:` path, `source:` paper/book, `oracle:`, `slice:` — or `verify.py provenance`
  fails CI.
- **Pre-commit stays surgical** (fast auto-fixes only); heavy checks live in CI.

## 7c. CI & cross-platform (Linux + Windows)

- **CI matrix:** `ubuntu-latest` + `windows-latest`, Python `3.12`, one workflow file.
  Per-PR runs `verify.py acyclic/debt/version/provenance` + the slice's layer tier on
  both OSes; merge gate = green on both. Full sweep + quarry regression nightly.
- **Tolerance-based oracles are a HARD rule.** `exp/log/pow/trig` differ 1–2 ULP between
  glibc and the MSVC runtime, so oracles assert tolerances (closed-form `1e-12`; MC/PDE
  their convergence tol), never `==` on transcendental results. The statelessness
  byte-identical check is same-process only, not a cross-OS claim.
- **Hygiene:** `.gitattributes` normalises to LF; UTF-8 forced (`PYTHONUTF8=1`, explicit
  `encoding="utf-8"`); `pathlib` everywhere, no hardcoded separators.
- **Tracking:** `OPEN.md`, `CHANGELOG.md`, `CLAUDE.md`, `redesign/` are tracked so CI has
  `verify.py`'s inputs.

## 8. Branching & commits

- **One branch per slice:** `slice/<NN>-<short-kebab-title>`, off `main`. `main` is
  protected and green-only — it advances solely by whole, oracle-passed slices.
- **A slice is several meaningful commits, not a squash.** Commits follow the checklist:
  `audit → align → test (oracle RED) → feat (oracle GREEN) → docs (provenance)`.
- **Red before green is a hard rule:** the failing reference-value test is committed
  *before* the code that satisfies it. (Carve-out: a pure refactor guarded by an already-
  green oracle need not re-introduce a red, but must run that oracle.)
- **Land with rebase-and-merge** (linear history, commits preserved). The version bump +
  `CHANGELOG.md` entry is the slice's final `chore(release):` commit at the tip.
- **Merge gate:** land only when the oracle is green *and* the layer-scoped test tier
  passes.

---

*Design rationale and full detail: `redesign/01_scope_contract.md` … `07_branching_and_commit_policy.md`.
Bottom-up worklist: `redesign/L0_ledger.xlsx`.*
