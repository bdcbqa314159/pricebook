# CLAUDE.md — Pricebook build guardrails

These are **guardrails, not suggestions.** They are the ratified output of the design
phase (`redesign/01`–`05`). When a change conflicts with this file, this file wins.
The design *rationale* lives in `redesign/`; this is the enforceable subset.

The build proceeds by **migration, one thin slice at a time, bottom-up**, starting with
Slice 0. Do not rewrite; do not chase feature parity; do not price outside the engine.

---

## 0. The one idea

**Functional core, imperative shell.** Pricing is a pure function
`price(instrument, model, market, numerics) → result` — no state. Everything that
persists (a booked trade, its life, the database) lives in a thin shell around the core
and only calls into it. The core never remembers; the shell never computes.

---

## 1. The spine (layers — dependencies point DOWN only, never up)

```
L6  LIFECYCLE / BOOKING / PORTFOLIO   (stateful shell: book, monitor, P&L, desks, viz)
L5  RISK & CAPITAL                    (greeks · XVA · RWA — on the engine + Pricable protocol)
L4  ENGINES                           (the stateless heart: bind instrument+model+market → PV/risk)
L3  MODELS + CALIBRATION              (dynamics; calibrated to a snapshot, else stateless)
L2  INSTRUMENTS                       (pure-data trade descriptions: legs, cashflows, payoffs)
L1  MARKET DATA                       (immutable MarketSnapshot · quotes · fixings · curves · vols)
L0  FOUNDATION                        (time & conventions · value types · finance-free numerics)
     DATA SPINE (side)                (SQLite→DuckDB ingestion → feeds L1; never imported by core)
```

**The law:** a lower layer must never import a higher one. This is enforced by the
per-commit acyclic check (Tarjan SCC on the import graph) — keep it green. If a change
needs an upward import, the design is wrong; fix the layering, not the check.

**Three structural fixes to honour as entries migrate:**
- `risk` lands at **L5** (above the engine), depends only on the engine + a `Pricable`
  protocol — **no `isinstance`-on-instrument ladders.**
- Calibration lands as a **unified front at L3** over per-family solvers.
- The engine registry lands **inside L4** (no leaky top-level `registry.py`).
- `pe` lands in the **L6 shell** (not L0); `Cashflow` lands at **L0** (not inside an asset class).

---

## 2. The stateless-engine contract (L4) — five invariants

```python
result = engine.price(instrument, model, market, numerics)  # -> PricingResult
```
1. **Referential transparency** — identical inputs ⇒ identical output, always.
2. **No ambient state** — no thread-locals, globals, or clock reads; "today" comes from
   the `MarketSnapshot`.
3. **No mutation** — `instrument`, `model`, `market` are frozen; never mutated in place.
4. **Failure is a value** — return `PricingFailure`, never raise-and-hope or emit silent
   `NaN`.
5. **Config is explicit** — all reproducibility knobs (seeds, MC paths, PDE grid, tol)
   arrive in `NumericalConfig`; never a hidden default.

**Instruments are pure data.** Frozen dataclasses that *describe* (legs, cashflows); they
do **not** price themselves. No `pv()`/`pv_ctx()` on instruments — pricing lives in L4.

---

## 3. Vocabulary (the stable types every layer speaks — ratified)

- **Time:** `Date` (stdlib), `DayCountConvention`+`year_fraction`, `Calendar`,
  `Frequency`/`Schedule`, `CompoundingMethod`, `RateIndex`. `Tenor` stays a string +
  helpers; `Rate` stays a float.
- **Money:** `Money(amount, Currency)` **at boundaries** (cashflows, PVs); plain floats
  inside hot numerical loops. Currency-mixing must be a type error where it matters.
- **Instrument atom:** `Cashflow` (at L0), `Leg` = ordered cashflows + convention.
- **Market (L1):** `Quote`/`QuoteId`/`QuoteKind`, immutable `MarketSnapshot`,
  `FixingHistory`. Curves are reached through a **`CurveHandle`** protocol
  (`df(date)`, `survival(date)`) — higher layers depend on the capability, not the
  concrete curve; **curves are never mutated in place.**
- **Engine I/O:** `NumericalConfig`, `PricingResult`, `PricingFailure`.
- **Identity/state:** `Trade` (**frozen**), `Portfolio`; `PricingContext` is **frozen**
  and kept as the engine's built-market-state input (linked to the `MarketSnapshot` it
  was built from); `BookedTrade` (L6) = description + lifecycle events.

---

## 4. Migration rules (quarry → new tree)

- **Quarry is read-only.** The old tree is never edited or deleted. Done = quarry empty.
- **Copy-ADAPT, never copy-paste.** Every crossing conforms to a layer, speaks the
  vocabulary, moves behaviour into the engine, or sheds debt. A byte-for-byte copy is a
  **failed** migration.
- **Green-oracle gate — nothing crosses grey.** No entry lands until a red/green oracle
  proves it against a known value: closed form > QuantLib/ISDA cross-check >
  self-consistency (reprice to par / zero NPV) > trusted mark. "Runs and looks right" is
  not an oracle.
- **Bottom-up.** An entry lands only if everything it depends on has landed.
- **Provenance.** Each landed entry records: quarry path, the paper/book/model it
  implements, and its oracle. (Educational constraint — keep it legible.)

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

**Layer checkpoint (hard stop):** migration is bottom-up *and* review-gated. On completing
all slices in a layer, write a **Layer Completion Report** to
`redesign/handoffs/L<N>_report.md` (template + required §5 design-drift section in
`redesign/08_handoff_protocol.md`), emit the one-line return message, and **stop** for
design review before starting the next layer. Do not begin L<N+1> until the L<N> ledger
is ruled.

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
