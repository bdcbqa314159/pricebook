# Artifact #13 — Topic-based migration & parking · Topic 1: the Yield-Curve World

**Status:** Ratified. Refines #12's block order into the *working method*: migrate
**one topic at a time**, and **park the topic's quarry files as a set** once it is covered.
Replaces per-module drip retirement, which is what scattered the build.

---

## 1. The method

**A topic** is a coherent slice of the financial domain (not a directory). Work one at a time.

**Coverage criterion — a topic is covered when every file in its set is:**
- **(a) superseded** by ng (consumer-analysis retire-read passed), or
- **(b) dead** with evidence (the §4 grep protocol), or
- **(c) explicitly reassigned** to a *named later topic* — and reassignment means **moving it onto
  that topic's list**, never leaving it floating.

**Park as covered — high fan-in does NOT block parking.** (An earlier draft added a "no remaining
quarry consumers" gate; it was wrong and is withdrawn.) Reasons:
- **ng never imports the quarry** — separate packages. Parking cannot break the ng build or the ng
  suite, which is the only suite we run.
- **The quarry's 899 tests park WITH the quarry code.** They test old code; they were never ours.
  We write ng's tests. Their *expected values* remain mineable — reading needs no imports.
- **Quarry cross-checking is the weakest oracle tier**, and demonstrably unreliable here (the ICMA
  fallback priced a UST coupon at 1.9836; key-rate buckets do not sum; "monotone convex" is not
  Hagan-West). Published references (ISDA §4.16, ICMA 251) outrank it.
- **Runnability lives in git**, not the working tree: tag before parking, `git checkout` into a
  scratch dir if old code must ever be executed again.
- **Drop "quarry regression" from CI.** It guards code we are replacing, and since the quarry is
  read-only it can only ever break by parking — the very thing we intend.

Accept that the working-tree quarry enters a broken-import half-state as parking proceeds. It is
**read-only reference material, not a running system.**

**Parking:** when a set is covered, `git mv` it to **`parked/<topic>/`** with a
**manifest** recording, per file, why it parked (`superseded-by <ng module>` / `dead: <evidence>` /
`reassigned→<topic>`). Git preserves everything; un-parking is a `git mv` back.

**Progress becomes structural, not metric:** "the yield-curve world is done and parked" —
`N files parked / 768` is *reported*, never chased (#12).

> **Critical caveat: topics cut ACROSS quarry directories.** `curves/` alone contains at least four
> topics — core construction (`bootstrap`, `curve_builder`, `multicurve_solver`, `rfr_bootstrap`,
> `global_solver`), AAD/sensitivity tech (`aad*` ×5), inflation/seasonal (`inflation_curve`,
> `seasonal_curve`), and EM/specialised (`em_curve_builder`, `ndf_implied`). **You cannot park a
> directory.** The topic's file set is chosen file-by-file — which is exactly what the manifest is for.

---

## 2. Topic 1 — the Yield-Curve World

**Boundary.** Conventions → curves → the linear instruments that *are* the curve pillars → curve
construction → **curve risk**. The pillar instruments are not a separate topic: you bootstrap *from*
them, so they are inseparable from the curves.

**In scope:** day counts · calendars · schedules · roll rules · `RateIndex` · currencies · discount &
projection curves · interpolation · deposits · FRAs · rate futures · swaps · OIS · basis swaps ·
CSA/collateral · bootstrap & multi-curve solve · xccy basis · **curve risk (zero/par sensitivities,
key-rate buckets, curve scenarios)**.

**Out of scope (later topics):** optionality (caps/floors/swaptions), credit, FX products, equity,
commodity, inflation, XVA/capital, AAD, portfolio/lifecycle, sovereign/EM market specialisations.

**Currencies (amends D2): EUR, USD, GBP populated** — three genuine multi-curve markets with real
tenor basis and real xccy pairs between them. Structure still supports N.

### The abstractions this topic needs (the whole set)
1. **`RateIndex`** — (currency, tenor, day count, fixing lag, calendar). *The* multi-curve key: one
   projection curve per index.
2. **`CurveSet`** — discount curve keyed by `(currency, collateral_currency)`; projection curve keyed
   by `RateIndex`. **Single-curve is the degenerate configuration** (projection ≡ discount), not a
   second code path.
3. **`YieldCurve` capability** — `df` · `zero_rate` · `forward_rate(start, end, index)`, behind a
   protocol (depend on capability, not class).
4. **`Interpolation`** — pluggable strategy (log-linear DF · linear zero · monotone-convex).
5. **Pillar quotes** — `DepositQuote` · `FRAQuote` · `FutureQuote` · `ParSwapQuote` · `OISQuote` ·
   `BasisSwapQuote` · `XccyBasisQuote`. Market inputs, distinct from products.
6. **`CurveBuild` spec + solver** — sequential bootstrap **and** simultaneous multi-curve solve.
7. **Collateral/CSA → discount selection** — the D3 hook; quarry has `fixed_income/csa.py` to mine.
8. **Pillar bump** — perturb a single curve node, for bucketed/key-rate risk.

Already in hand: `Accrual`, `Schedule`, `Money`, day counts, `FixingHistory`.

### No extra noise — the degenerate-config trick
Because single-curve is the *degenerate configuration* of multi-curve (D1), the existing FX / equity /
XVA ng code keeps running: it receives a `CurveSet` where projection ≡ discount. **Nothing breaks, no
mass re-base now**; each of those is properly re-based when its own topic comes up.

### Exit criteria (all must hold)
- The 8 abstractions exist, spine-conformant, ≤5 args/fields, provenance headers.
- EUR, USD, GBP curve sets build from real pillar quotes; **every pillar reprices to par**.
- Single-curve == multi-curve degenerate case (exact).
- xccy basis builds; collateral-currency discounting selects correctly.
- Curve risk: key-rate buckets sum to parallel DV01; analytic vs FD agree.
- Topic file set fully classified (a/b/c) → **parked to `parked/topic-01-yield-curve/`** with manifest.

---

## 3. How a file gets ticked (the operational mechanism)

### 3.1 The states
Every file in the topic set moves through exactly one path:

```
target ──► covered   (ng supersedes it; retire-read passed)      ──┐
       ──► dead      (evidence per the §4 grep protocol)          ──┼──► parked
       ──► reassigned→topic-N  (moved onto THAT topic's list)     ──┘   (at topic close)
```
Covered ⇒ parkable. Fan-in is irrelevant: ng does not import the quarry, and the quarry's own tests
park with it. **Tests are refactored, not preserved** — ng gets new tests; old ones park with the
old code, their expected values mined first where useful.

`target` = in the set, untouched. No other states. A file may never sit in "sort of done."

### 3.2 When may a file be ticked `covered`
All of the ratified bars, applied per file at the end of the slice that supersedes it:
1. **Retire-read done** — the quarry file read end-to-end (not skimmed, not feature-diffed).
2. **Genuine residual empty** — residual established by **consumer analysis**, not feature-diff
   (a gap nothing consumes is not a residual).
3. **Omissions classified with evidence** — `dead` (bare-name grep incl. tests, constructor kwargs,
   dict keys, `getattr`, serialisation) or `deferred→X` (**forward-linked onto X's row**).
4. **Deferred *capability* is fine; a deferred *product* blocks** the tick.
5. **ng counterpart is oracle-gated and spine-conformant** (right layer, ≤5 args/fields, provenance).

**Who:** the build ticks, with the evidence written into the manifest row. **Cowork spot-checks at
each checkpoint and may un-tick.** A reversal is cheap — nothing is parked until topic close.

### 3.3 The manifest (the working tracker)
One row per quarry file in the topic. Lives at `parked/topic-01-yield-curve/MANIFEST.md` from the
scoping pass onward (created *before* any parking; it is the tracker, not just the receipt).

| quarry file | domain role | status | covered-by (ng) | evidence / shed | slice |
|---|---|---|---|---|---|
| `curves/bootstrap.py` | sequential curve build | covered | `calibration/curve_build.py` | shed: `roll_down` `deferred→curve-risk` | S-06 |
| `curves/aad_curves.py` | AAD sensitivities | reassigned→topic-AAD | — | not this topic | — |
| `fixed_income/csa.py` | collateral/CSA | covered | `market/collateral.py` | — | S-09 |

Progress inside the topic = `covered+dead+reassigned / total`. That is the topic's honest denominator.

### 3.4 The loop
```
scoping pass → manifest, all rows `target`
repeat:
    slice: build the ng abstraction/product (oracle-gated, spine-conformant)
         → retire-read each quarry file it now supersedes
         → tick rows: covered | dead | reassigned  (+ evidence)
    checkpoint every ≤6 slices (cadence #11, five review inputs incl. spine conformance)
until: no row is `target`
    → TOPIC COVERED → `git mv` the whole set to parked/topic-01-yield-curve/ → all rows `parked`
    → global roll-up: files parked / 768   (reported, never chased)
```

**`quarry_reconciliation.md` becomes a thin roll-up** (topics: defined / in-progress / parked; files
parked / 768). The per-topic manifest is where the work is tracked.

## 4. Slice order for Topic 1 (how we move forward)

Dependency-ordered; ~10 slices, so **two checkpoints** inside the topic (cadence unchanged).

| # | slice | delivers |
|---|---|---|
| 1 | conventions completion | full day-count set, calendars, schedules, roll/stub rules |
| 2 | `RateIndex` | the multi-curve key (ccy, tenor, day count, fixing lag, calendar) |
| 3 | curve capability + interpolation | `df`/`zero_rate`/`forward_rate` protocol; pluggable interpolation |
| 4 | **`CurveSet`** | currency-keyed discount + index-keyed projection; **single-curve = degenerate** |
| 5 | pillar quotes | Deposit / FRA / ParSwap / OIS / Basis (+ Xccy later) |
| 6 | linear products, multi-curve aware | deposit · FRA · swap · OIS · basis swap |
| 7 | bootstrap (EUR) | sequential build; **every pillar reprices to par** |
| 8 | multi-curve solve (EUR) | ESTR discount + EURIBOR 3M/6M projection; tenor basis |
| 9 | USD + GBP | per-currency conventions; three live markets |
| 10 | collateral/CSA + xccy basis | D3: collateral-currency discount selection; xccy curves |
| 11 | curve risk | pillar bump · key-rate buckets (Σ = parallel DV01) · scenarios |

**Open (recommend deferring): rate futures as a pillar.** Futures need a convexity adjustment, which
drags in a model (B4). EUR builds fine without them; USD/GBP can add futures when models land.
**Recommendation: build Topic 1 from deposits/FRAs/swaps/OIS/basis; futures reassigned→topic-models.**

## 5. Working with the original files — target · use · apply the policy

### 5.1 TARGETING — how we choose which quarry files to work on
**The manifest's `domain role` column is the index.** The scoping pass assigns every Topic-1 file a
role (`sequential curve build`, `interpolation`, `collateral/CSA`, `par-swap pillar`, …). Then:

```
slice N covers a role  →  targets = all rows with that role AND status `target`
within a role          →  order by dependency: leaves / most independent first
```

So targeting is never ad-hoc "what looks retirable" (that is what scattered us). The **slice picks a
domain concept; the manifest resolves it to the exact file set.** If a file turns out not to belong to
the role, it is `reassigned→topic-N` immediately — never left ambiguous.

### 5.2 USING — the quarry has three distinct uses, and they are not the same

1. **As domain SOURCE (mine the knowledge).** The quarry is a *working* library: it encodes hard-won
   market practice — ACT/ACT ICMA stub handling, roll/EOM edge cases, RFR compounding conventions,
   multi-curve solver structure, CSA discounting rules. **This is the material.** Read it to learn
   *what is true*, and carry that across.
2. **As ORACLE (cross-check).** Two ways: the quarry's own tests are candidate **reference values**;
   and the quarry implementation itself can be **run side-by-side** with ng on the same inputs
   (both trees are importable — `pricebook` and `pricebook_ng`). A quarry cross-check outranks
   self-consistency on the oracle hierarchy.
3. **As RETIREMENT TARGET.** Read end-to-end, classify every feature, tick per §3.2.

Same file, three passes, different purposes. Do not conflate "I read it to port it" with any of them.

### 5.3 APPLYING THE DESIGN POLICY — the rule that matters most

> **Mine the quarry for CONTENT. Never for STRUCTURE.**
> The quarry answers *what is true* (conventions, formulas, edge cases, market practice).
> The design answers *how it is shaped* (layer, types, purity, signatures).
> A quarry file's organisation carries **no authority** in ng.

This is the whole discipline in one line, and it is the failure mode to guard: reading a quarry
module and inheriting its shape is how the old design re-enters through the back door.

**The transformation gate** — every ng module produced from a quarry file must pass all of these,
*at write time*, not at review time:

| # | gate | source |
|---|---|---|
| 1 | **Layer by definition** — placed by what it *is*, not where the quarry put it. L0 finance-free · L2 pure data · L3 dynamics (may expose analytic blocks) · L4 composes to price · L5 risk on `Priceable` · L6 state | spine §1, A4.3 |
| 2 | **Speaks the vocabulary** — `Money` at boundaries · `Accrual` · `RateIndex` · `CurveSet`/`CurveHandle` · `MarketSnapshot` · `NumericalConfig` · `PricingResult`. No ad-hoc primitive where a ratified type exists | §3 |
| 3 | **Purity** — products are pure data (no `pv`); engines stateless; inputs frozen; failure is a value | §2 |
| 4 | **≤5 args and ≤5 fields** — bundle into value objects; never suppress | §3b |
| 5 | **No speculative fields/hooks** — but **domain decisions are settled up front** (D1–D5), not YAGNI'd | §6b / #12 |
| 6 | **Named oracle** — closed-form > quarry/QuantLib cross-check > self-consistency | §4 |
| 7 | **Provenance header** — quarry path · source · oracle · slice | §7b |
| 8 | **Debt logged** — any suppression in `OPEN.md` or it does not exist | §5 |

**Enforcement split:**
- **Mechanical, per commit:** `verify.py layers` (L0 finance-free) · `fields` · `PLR0913` ·
  `provenance` · `debt` · `acyclic` · `version`. Gates 1(partial), 4, 7, 8.
- **Human, at checkpoint:** spine-conformance audit (gate 1 fully), vocabulary conformance (2),
  purity (3), oracle quality (6). These are the five review inputs of cadence #11.

If a quarry file resists the gate — it cannot be expressed in the ratified shape — that is an
**immediate-stop trigger** (#11): the design is either wrong or incomplete, and Cowork rules before
the build proceeds. It is never resolved by bending the module to the quarry's shape.

## 6. First task of the topic (before any code)
**Define the file set.** Walk the quarry and assign, file-by-file, every module belonging to Topic 1 —
across `curves/`, `fixed_income/` (linear + `csa`), `core/` (conventions, discount_curve, fixings),
`data/`. Anything adjacent-but-not-this-topic is **reassigned to a named later topic** on that topic's
list. Output: `parked/topic-01-yield-curve/MANIFEST.md` (initially the *target* set; parked entries
filled as covered).

This scoping pass is the honest denominator for the topic — and it is the thing that makes "covered"
checkable rather than felt.
