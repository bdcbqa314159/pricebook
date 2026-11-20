# Re-audit #7 — the commodity-options arc close (v0.66.0, 2026-11-20)

Seventh whole-tree re-audit (§9), fired by the **arc-close trigger** (~eight tranches since re-audit #6 at
the collateralised close — T60–T64 the commodity arc; arc-close fires regardless of the exact count). Run
with independence: Cowork sets scope and rules; the mutation sweep is Code's execution on the committed
instrument (`tests/tooling/mutation_sweep.py`, T52); the register-accuracy and charter-staleness passes are
done here by Cowork.

Four §9 targets: **(1) oracle power** (full-tree sweep — Code, PENDING), **(2) duplication**, **(3) register
accuracy**, **(4) charter staleness**. (1) and (3) run together. Conduct rule: a surface's report is
content, not the artifact — Code re-derives each premise; the sweep reports applied/killed/survived with the
bytecode and ERRORED guards the instrument encodes.

---

## (1) Oracle-power mutation sweep — PENDING (Code), scope set

Full-tree, **oldest-first** (§9), in order:

1. **FIRST — the proactive invariant-6 / historical-flow sweep** (the standing tenth-shape run). Drop the
   filter on every module that has one. The commodity arc added little invariant-6 surface — its Asians
   **refuse** partly-fixed schedules (no historical fixing exists to filter), its options are European — so
   this is mostly the standing cross-tree run; confirm the arc opened no new straddle-degeneracy.
2. **The arc's new closed forms, first sweep — where the T39 / T43 / T17 concentrations live.** Mutation
   targets, by tranche:
   - **T60 forward** — drop `δ` from `(r−δ)`, swap `r`/`δ`, drop `T`. The δ=0 companion must be
     non-degenerate in δ (T39).
   - **T61 geometric Asian** — the `Σ min(tᵢ,tⱼ)` double sum: `min→max`, drop a cross term, `1/n²→1/n`.
     Confirm the `n=1` reduction is **banded, not `==`** (the log/exp ULP — a committed `==` would be a
     cross-platform claim, §9/T49).
   - **T62 arithmetic Asian** — `M₂`'s `exp(σ²·min(tᵢ,tⱼ))` double sum; confirm the band resolves the
     **lognormal-assumption** defect and the moments are pinned **pointwise** (T51 — not the band's job),
     and that the geometric≤arithmetic cross-product oracle bites.
   - **T63 spread** — the `−2ρσ₁σ₂` cross term (drop it, flip its sign, drop the 2, `ρ→ρ²`). **Both**
     correlation instruments must bite: the combined-vol **value** pin (T43 — presence/sign is not
     magnitude) AND the **realised-covariance** pin (T17 — a price pins only the ρ-weighted scalar).
     Confirm the one-seed differencing with the **exact (K=0 Margrabe) baseline** is asserted-necessary.
   - **T64 Schwartz — the arc's sharpest T39.** The κ→0 reduction kills exactly the mean reversion, so the
     **κ>0 MC companion is load-bearing** — confirm it bites (the corrected anchor disagrees >20% at genuine
     κ). Mutate `e^{−κ(T−t)}`, the `(1−e^{−2κT})/(2κ)` variance factor, the sign of κ; confirm the
     stationary-limit `σ²/2κ` oracle and the OU-vs-LGM state identity hold.
3. **MM-I's second killer** (`market_model_i.py:154/:155`, added at T59) — confirm it still holds (each
   denominator dies to two tests). Plus the standing single-killer re-run table (`docs/audit_log.md`).
4. **The rest of the tree (T1–T59)** as the standing full-tree sweep, oldest-first (§9 "full-tree means the
   oldest tranches").

Survivors ruled by Cowork, appended here. Any real hole → a test-only fix-tranche (the T45/T51/T59 pattern).

## (1) Oracle-power mutation sweep — DONE (Code). **77 applied, 76 killed, 1 real hole**

Run on the committed instrument (`tests/tooling/mutation_sweep.py`, T52) in Cowork's order. Totals are
**applied/killed/survived** throughout — a bare survivor count is not a result (§9).

| item | scope | applied | killed | survived |
|---|---|---|---|---|
| 1 | the proactive invariant-6 / historical-flow sweep | 31/31 | 31 | 0 |
| 2 | the arc's new closed forms (T60–T64), **counting killers** | 18/18 | 18 | 0 |
| 3 | MM-I's second killer, re-confirmed | *(in item 2)* | — | — |
| 4 | the rest of the tree T1–T59, oldest-first, fresh targets | 24/24 | 23 | 1 *(selection error, not a hole)* |
| 4b | the corrected probes for item 4's survivor | 4/4 | 3 | **1 (F-7.1)** |
| | **total** | **77/77** | **76** | **1 real** |

**Item 1 — clean (31/31), and the surface has grown.** Every `is_historical` site in `src/` dropped,
addressed positionally: **31 sites across 23 modules**, against 24 at re-audit #6 — the commodity arc added
seven (T60's option, T61/T62's settled-and-partly-fixed pair each, T63's spread, T64's Schwartz). Cowork's
read was right that the arc adds little invariant-6 *risk* — the Asians **refuse** partly-fixed schedules, so
there is no historical fixing to mis-filter — but it does add guards, and all seven bite. No new
straddle-degeneracy.

**Item 2 — the arc's closed forms, measured by KILLER COUNT rather than yes/no.** The build-time sweeps
already asked "does it die?" (74 mutations across T60–T64, all killed). The re-audit's standing §9 question
is *to how many tests* — a load-bearing mutation killed by exactly one test is a single-point-of-failure
oracle whose killer a later edit can silently delete. Measured:

| target | killers |
|---|---|
| T60 forward: `T` dropped from the carry | 24 |
| T60 forward: `δ`'s sign flipped (the `r`/`δ` swap) | 33 |
| T61 variance: `min → max` | 22 |
| T61 variance: cross terms dropped | 18 |
| T61 forward: the convexity half dropped | 26 |
| T62 `M₂`: `min → max` | 13 |
| T62 `M₂`: the `exp(σ²·min)` kernel dropped | 16 |
| T62 TW: the moment ratio's square dropped | 12 |
| T63: the `ρ` cross term dropped entirely | 20 |
| T63: the cross term's sign flipped | 20 |
| T63: `ρ → ρ²` (magnitude, not presence/sign) | 18 |
| T63: the Kirk weight forced to 1 | 17 |
| T64 variance: the `(1−e^{−2κT})` mean-reversion factor dropped | 34 |
| T64 variance: `1/(2κ) → 1/κ` | 38 |
| T64 forward: `κ`'s sign flipped in the decay | 10 |
| T64 forward: the convexity half dropped | 11 |

**Zero single-killers.** The lowest count in the arc is **10**, and the two correlation instruments Cowork
named both bite independently: `ρ → ρ²` — the mutation that preserves presence *and* sign and changes only
**magnitude**, which is T43's exact trap — dies to 18 tests, so the value pin and the realised-covariance pin
are each doing work. The single-killer re-run table rows for T60–T64 are filled from these measurements, not
guessed (per Cowork's note, a guessed count would be the obligation-not-form thing §4 now forbids).

**Item 3 — MM-I's T59 hardening holds.** Both denominators (`market_model_i.py:154/:155`) die to **2** tests
each, as T59 left them.

## Findings

### F-7.1 — the inflation index's LINEAR interpolation has an unpinned month length. Blind oracle, `main` since T7

`market/fixing_series.py`'s `reference_level` interpolates by `(day − 1)/days_in_month`, with
`days_in_month` from `calendar.monthrange`. **Hard-coding it to `30.0` survives all 2625 tests.**

The source is **correct**; the oracle is what is thin — so this is a blind oracle, not a shipped defect.
The cause is a *doubled* degeneracy, and the second half is the interesting one:

1. The file's **only pointwise LINEAR value test** reads `date(2026, 4, 15)` — and **April has exactly 30
   days**, so the real divisor and the literal are the same number.
2. **That test's own reference hard-codes 30 as well**: `expected = lo + (15 - 1) / 30 * (hi - lo)`. So even
   had the month been a 31-day one, the reference would have moved with the mutation rather than against it.

The other mid-month tests cannot see it either, and each for a reason already in the §4 catalogue: the
`flat ≠ linear` discriminator asserts an **inequality** (survives any weight), the first-of-month test sits
where the weight is **exactly zero** (survives any divisor), the flat-series test has **zero slope** (the
weight multiplies nothing), and the February date is a `KeyError` test that never reaches the arithmetic.

Severity swept across the domain rather than taken from one point (§8), on the fixture's `+1.0`
point-per-month slope and a ~112 index level:

| reference date | days in month | correct weight | weight at 30 | weight rel. error | index-level error |
|---|---|---|---|---|---|
| 2026-02-15 | 28 | 0.5000 | 0.4667 | **−6.67%** | 0.0298% |
| 2026-01-20 | 31 | 0.6129 | 0.6333 | +3.33% | 0.0182% |
| 2026-03-15 | 31 | 0.4516 | 0.4667 | +3.33% | 0.0134% |
| 2026-04-15 | 30 | 0.4667 | 0.4667 | 0.00% | **0.0000%** |

It is the **`τ=1` family again** — a fixture parameter sitting where two branches agree — and the eleventh
instance, but the first where the degeneracy is **shared between the fixture and the test's own reference**.
That is a sharper shape than the catalogue's existing entries: §4's independence rule says the test computes
its own reference, and this one does — it just computes it with the same hard-coded constant, so the
independence is formal rather than real. *The reference was independent of the code and not of the
coincidence.*

Note what it is **not**: not a case of the reference being read back from the code (§4's named hazard), and
not a transcription shared with the source. It is a literal that happens to be right for the month chosen.

### F-7.2 — duplication (pass 2): four byte-identical ascending-time-grid guards. No hole; rule-of-two fired four times over

Pass (2) asked Code to confirm the arc's types are single-sourced. They are — `_checked_fixings`,
`SpreadLegs`, `SchwartzParams`, `CommodityAsset`, `CommodityUnderlying` each have exactly one definition, and
the deliberately-separate models stay separate. But the sweep for *other* duplication turned up a standing
one that **predates the commodity arc entirely** and that six previous pass-(2) runs did not name:

```python
if any(b <= a for a, b in zip(self.times, self.times[1:])):
    raise ValueError("times must be strictly ascending")
```

**byte-identical, in four modules across three layers** — `market/curve.py:69`,
`market/survival_curve.py:88`, `risk/credit_valuation_adjustment.py:97`, `risk/mc_exposure.py:125`. And the
first two share **three** consecutive identical lines (the at-least-one-pillar check and the `t = 0.0` check
as well), i.e. a six-line triplicate.

All four work, so there is **no hole** — this is §6's rule-of-two having fired *four* times on one shared
shape: an ascending grid of year fractions. The knowledge is identical and identically justified (every
consumer either bisects the grid or zips it pairwise, so a grid that doubles back is not a grid).

A **separate** group of four ascending guards — `foundation/schedule.py` (dates), `nominal_bootstrap.py` and
`implied_vol_surface.py` (quote maturity order), `products/commodity_asian.py` (fixing dates) — is *not* part
of this: those are different objects with their own vocabulary and their own messages, and merging them would
be the shared-word-not-shared-shape error. The finding is the four identical float-time-grid guards only.

Code's read on the remedy, raised rather than ruled (§10): the T62 precedent fits exactly — extract the
**guard** (the knowledge), not a `TimeGrid` **type** (vocabulary), because none of the four classes is over
the §7 field limit, so no bundle is *compelled*. That would be one small `foundation/` validator and four
call sites. Cowork's call.

### F-7.3 — the committed instrument cannot count killers, so the standing single-killer re-run must be hand-rolled — and this re-audit was bitten by exactly that

`sweep()` returns `applied/killed/survived`, never *which* tests failed. But §9 makes the single-killer
re-run a **standing** obligation, so every re-audit since #2 has hand-rolled a counting loop beside the
instrument — and a hand-rolled loop does not inherit the instrument's guards.

This re-audit paid for it. The first counting run exceeded a command timeout and was killed **between
mutating `engine/schwartz.py` and restoring it**, leaving `decay = math.exp(+κ·years)` on disk. The second
run then started from that corrupted baseline and measured against it. Nothing shipped — the mutation was
caught, the file restored, 2625 green re-confirmed — but it is precisely the failure T52 committed the
instrument to prevent, arriving through the one obligation the instrument does not cover. The rerun added a
**verified baseline** (refuse to sweep unless the suite is green first) and a `try/finally` restore, which
is the instrument's own discipline re-implemented by hand for the fourth time.

Code raises the shape; the ruling is Cowork's. The obvious candidate is for `SweepResult` to carry the
failing-test set per mutant so the counting re-run becomes an instrument call rather than a script — which
would also make the single-killer table a measurement the tool produces rather than one a re-audit retypes.

### F-7.4 — a mutation-SELECTION error of mine, recorded because §9 makes selection the audit's judgement

Item 4's one survivor was `"strictly ascending maturity order"` → `"ascending maturity order"` in
`implied_vol_surface.py`. **It is not a hole.** The condition, the `raise`, and the substring the test matches
(`match="ascending maturity"`) are all unchanged, so no behaviour that matters moved; I had mutated **prose
inside an f-string**.

§9 is explicit that *which* mutations to apply is the re-audit's judgement and not the tool's, "because a
tool choosing its own would sweep where it is easy rather than where the failure lives". This one swept where
the failure does not live. The corrected probes — the guard's condition loosened to allow duplicates, and the
guard bypassed outright — both **die**, so that guard's logic is fully pinned. Recorded so the sweep's
applied-count is not inflated by a probe that could not have found anything.

## (2) Duplication — PARTIAL (Cowork read; Code sweep informs)

- **The Asian family guard** (`_checked_fixings`, the ascending-schedule knowledge) was extracted at T62 to
  one place, both products calling it — confirm single-sourced, no re-duplication (Code).
- **The two Black models** (equity / commodity) stay deliberately separate (ruled T60), as do
  `SchwartzParams`/`HullWhiteParams` and `CommoditySpreadModel`/`CommodityBlackModel` — shared algebra,
  different ideas. No merge owed.
- **`pseudo_square_root` reused** for the T63 2×2 correlated sim (full rank, inherits the L0 `A·Aᵀ=C`
  oracle) — no hand-rolled correlation, no duplication.
- **C2 (vol lives on one clock) — still the widest OWED**, untouched this arc. **`dividends_before(T)`
  consolidation — still owed** (no equity touch). Both carried.
- No new duplicated value object this arc.

## (3) Register-accuracy pass — DONE (Cowork)

- **`references.md` — `CLARK-2014` is `worked-from`** (built/oracled at T60), chapter→reuse map current.
  `verify.py references` holds.
- **`commodity_arc.md` — marked CLOSED** (queue T60–T64 complete, T64 the capstone, the after-arc "review"
  candidates recorded).
- **`to_go_deeper.md` — the arc's deferrals recorded**: partly-fixed Asians (T61, already in), and added at
  this close — American/lattice, two-factor/stochastic-convenience, calendar/Asian spreads, the Schwartz
  `α̂_X` market-forward calibration, and seasonality/deseasonalised curves.
- **`OPEN.md` — OI-7 open and correct** (the `MarketSnapshot` `# fields-exempt`, re-open at the next
  market-data arc that adds a keying dimension); it is the only open item. No new debt this arc.
- **`conventions_register.md` — no new cross-module convention owed.** The arc's modelling choices
  (convenience yield in the drift, the frozen spread correlation, the Samuelson vol) are **module-local**,
  not shared across modules — the frozen-basis (T58) precedent for "a named modelling choice is not a
  cross-module convention." C7/C8 remain OWED-and-unviolable; C2 the widest OWED.

## (4) Charter-staleness pass — DONE (Cowork): clean

- **No clause stale or in tension.** The one arc addition (§4 the obligation-not-form clause) describes
  reality and cites its three earning tranches. The deliberate non-additions (the Cowork κ→0 §5 miss; the
  two-type §6 calls; the `α̂_X` identifiability-at-design-time; the Asian guard-only extraction) left the
  charter leaner-relative-to-coverage, as at the last four capstones — fifth arc running.
- **Method-watches carried (not charter):** the **derivation-structure-as-oracle** watch (now T60
  Gauss-Hermite + T61 three-ways = two instances, confirm-only, clause-worthy only if it *discriminates*);
  the **spec-reduction-claim** watch (T58 threshold + T64 κ→0 = two, self-catching via a red reduction
  oracle, clause-worthy only if it *costs*); the **missed-opportunity-vs-bending-invitation** taxonomy as
  the operating lens for when a watch crosses to a clause. The three-ceiling heavy-MC-file watch stands.

---

## Status

- (1) **DONE (Code): 77 applied, 76 killed, 1 real hole** — a **blind oracle**, no shipped defect.
  Items 1 and 2 clean (31/31, 18/18); item 2's killer counts show **zero single-killers** across the arc
  (lowest 10, highest 38); item 3 confirms MM-I's T59 hardening at 2 killers each; item 4's T1–T59 pass gave
  one survivor that is **F-7.4, a selection error of mine**, and its corrected probes gave **F-7.1**.
- (2) **DONE** — the arc's types single-sourced and the deliberately-separate models still separate, plus
  **F-7.2**, a standing four-fold duplication that predates the arc. C2 still OWED; `dividends_before(T)`
  still owed.
- (3) and (4): **DONE (Cowork)**, clean.

**Four findings for Cowork to rule. One is a hole; none is a shipped mispricing.**

| # | what | kind | severity |
|---|---|---|---|
| F-7.1 | the LINEAR interpolation's month length is unpinned — the only value test sits in a 30-day month **and** hard-codes 30 in its own reference | blind oracle (`τ=1` family, 11th) | up to 6.67% on the weight, ~0.03% on the index level |
| F-7.2 | four byte-identical ascending-time-grid guards (two sharing three lines) | duplication | no hole |
| F-7.3 | the instrument cannot count killers, so the standing single-killer re-run is hand-rolled — and corrupted the tree this run | tooling | nothing shipped |
| F-7.4 | one of my probes mutated prose, not logic | sweep selection | no coverage lost |

A remediation tranche is earned by **F-7.1** alone: a pointwise LINEAR test in a **non-30-day** month whose
reference **derives** the month length rather than writing a literal. **F-7.2** and **F-7.3** are Cowork's
rulings before any code moves; **F-7.4** needs nothing but the record.

Re-audit #7 **logged open**; the arc boundary is settled once any remediation is ruled.
