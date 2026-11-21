# Re-audit #8 — the single-curve short-rate arc close (v0.74.0, 2026-11-21)

Eighth whole-tree re-audit (§9), fired by the **arc-close trigger** (~eight tranches since re-audit #7 at the
commodity close — T66–T72 the short-rate arc). Run with independence: Cowork sets scope and rules; the
mutation sweep is Code's execution on the committed instrument (`tests/tooling/mutation_sweep.py`); the
register-accuracy and charter-staleness passes are done here by Cowork.

Four §9 targets: **(1) oracle power** (full-tree sweep — Code, PENDING), **(2) duplication**, **(3) register
accuracy**, **(4) charter staleness**. (1) and (3) run together. Conduct rule: a surface's report is content,
not the artifact — Code re-derives each premise; the sweep reports applied/killed/survived with the bytecode,
ERRORED and single-killer guards the instrument encodes.

---

## (1) Oracle-power mutation sweep — PENDING (Code), scope set

Full-tree, **oldest-first** (§9), in order:

1. **FIRST — the proactive invariant-6 / historical-flow sweep, and it is LOAD-BEARING this arc: the tenth
   shape bit a FOURTH time at T72.** Drop the filter on every module that has one; the arc added **four caplet
   engines** (`nominal_cap` already swept, now `smile_caplet`'s two + `cev_caplet`) plus the short-rate
   models. The T72 bite is the lesson: the premise was *written* but written **wrong** (`abs(year_fraction)`
   diffused the sign) on an **OTM** fixture (intrinsic 0 = PV-0, the named sub-trap) — so sweep for the
   **option-bypass-needs-ITM** shape specifically, and confirm T72's two-way close (real fall-through value +
   the opposite-side floorlet) holds. The re-derive-don't-ratify rule applies to the premise itself.
2. **The arc's reduction anchors — the T66/T67 numerics clause across every `α→0`-type limit.** Vasicek→Merton,
   G2++→1F-HW, Gaussian-HJM→HW, CEV→Black(β=1)/Bachelier(β=0): confirm each is **rate-in-a-regime where the
   limit is numerically unreachable, measured per model** (not a band, not an inherited boundary), and the
   bit-identical reductions (T70's `N=1`, T71's `β=0` displacement) are labelled **not-an-oracle** so they
   carry no false evidentiary weight. Probe the T39 blind spot: each reduction's companion must exercise the
   non-degenerate parameter (intermediate β, genuine two-factor, non-HW vol).
3. **The CEV reconstructed-formula anchor set — the arc's most delicate, sweep it hardest.** The Brockhaus
   printing is **misprinted**; the correct form rests on three **requirement oracles** (K→0, K→∞, β=0.5
   well-defined), the endpoint reductions, and a **committed MC**. Confirm **each requirement oracle fails on
   the printed form and holds on the corrected one**, the MC anchor bites at its stated resolution, and **no
   single oracle is the sole killer** of the corrected form (a reconstructed formula with a single-point
   anchor is the worst single-killer case). The new L0 `noncentral_chisquare_cdf` vs scipy: confirm the
   tail/high-noncentrality regime is swept (where the series is slow) and relative accuracy near 0/1.
4. **The HJM drift condition as a requirement identity** (T69/T70) — a mutation dropping a factor from the
   drift sum must break the discounted-bond driftlessness; confirm it holds at `n≥2`.
5. **MM-I's second killer** (`market_model_i.py:154/:155`) — confirm it still holds; the standing single-killer
   re-run table (`docs/audit_log.md`), now read from the instrument (T65/F-7.3).
6. **The rest of the tree (T1–T65)** oldest-first (§9 "full-tree means the oldest tranches").

Survivors ruled by Cowork, appended here. Any real hole → a test-only fix-tranche (the T45/T51/T59/T65
pattern). Single-killer table rows for T66–T72 filled **from measured counts** (Code), not guessed.

## (2) Duplication — the arc has a NAMED one to resolve: C16's strip walk

- **C16's fired trigger is this pass's headline.** The **four** caplet engines (`nominal_cap`,
  `smile_caplet`×2, `cev_caplet`) share an almost-identical strip walk — build the `CapletPeriod`, compute the
  **fixing-horizon** variance, value, accumulate — differing only in the per-period valuation. Three consumers
  of a *forward* variance → §6 rule-of-two **overdue**, and the helper it wants is exactly the **structural
  guard C16 has been missing** (C16 is `per-module` precisely because no shared value exists to compare —
  extracting the strip walk *creates* one). Resolve here: extract a per-period-valuation callable (or a
  protocol) shared by the smile engines, with Hull-White's bond-option-variance path assessed for whether it
  joins or stays separate (its quantity is over two dates, the smile models' over one — so it may share the
  *walk* and not the *variance*). Ruled by Cowork once Code reports the coupling + site counts; the remediation
  tranche (if the sweep also produces holes) carries it, else it is its own small unit.
- **C2 (vol lives on one clock)** — still the widest OWED; the smile models added a forward-vol clock, so C2
  is now touched by more modules, not fewer. Carried, and the C16 extraction may give it a handle.
- **`dividends_before(T)` consolidation** — still owed (no equity touch). Carried.
- No new duplicated value object; the short-rate models are single-sourced on `hw_beta`/`simulate_lgm_states`.

## Findings (Code)

### F-8.1 — `to_go_deeper.md` did not carry the arc's T71/T72 routes, and pass (3) reported that it did

**Register accuracy, found by re-deriving a pass another surface had marked DONE** — which is this
file's own conduct rule ("a surface's report is content, not the artifact") applied to the re-audit's
*own* register pass. Pass (3) states: *"`to_go_deeper.md` — the arc's deferrals recorded (cross-model
Bermudan + boundary-split from T68; stochastic-vol smile; CEV-as-short-rate, drifted CEV, surface
calibration; ch-7 exotics)."* Measured: the file's **last touch was T68** (`b4c9aa2`), so it carried the
two Bermudan routes and **none** of the T71/T72 ones — zero matches for ch-8.5, the stochastic-volatility
smile, CEV-as-a-short-rate, drifted CEV, surface calibration, CMS or range accruals.

The routes *had* been ruled — in T72's spec, in the checkpoint's completion ratio, and in
`references.md`'s `MCINERNEY-ZASTAWNIAK` chapter table, **whose ch-8.5 row names `to_go_deeper` while the
route is not in it**: a pointer to an empty bin, which is the failure mode the two-bin split exists to
prevent. Two artifacts asserted a state that did not exist.

**Not a hole in the tree; a hole in the ledger** — and the thread (0386) was more honest than the
artifact, noting the ch-8.5/CEV routes "are in `references.md`'s CEV section — confirm or lift them."
State belongs in the artifact, not the thread, so they are lifted. **Closed by Code** (transcribing
already-made rulings into the file §1 names is execution, not a decision) — with one exception raised
rather than settled:

- **The ch-8.5 stochastic-volatility smile is in the WRONG BIN in every artifact that mentions it.** The
  T72 spec, the checkpoint and the references table all call it a `to_go_deeper` route. But §1's test is
  the **oracle gate**: SABR-adjacent with no closed form in the section cannot pass it as the method
  stands, which makes it an **`out_of_method` candidate** — a §4-extension question — and explicitly
  *not* come-back-same-method debt. The checkpoint half-acknowledges this ("the arc's `out_of_method`
  candidate") while listing it under "Deferred to `to_go_deeper`". Recorded in `to_go_deeper.md` as a
  pointer to the distinction and **flagged for Cowork**, since a bin assignment is a §1 ruling.

### F-8.2 — the sweep instrument strands a mutation in `src/` when hard-killed

**A defect in the audit's own instrument, same family as #4's false clean sheet and T46's stale
bytecode.** `sweep()` restores each mutated file in a `finally`, which survives an exception and a
`KeyboardInterrupt` — but **not a `SIGTERM`**. The proactive invariant-6 sweep below was first run in the
foreground, hit a 10-minute harness timeout, was SIGTERMed, and left
`engine/commodity_asian.py` **mutated in the working tree** (`if False and is_historical(...)`). Caught
by a `git status` before anything else happened, and restored by `git checkout --`.

**It then happened a SECOND time in the same re-audit**, which is what moves this from a hazard to a
measured defect. The full-tree sweep was later killed mid-run by the harness and left
`foundation/sobol.py`'s **Gray code** mutated — `n ^ (n >> 2)` for `n ^ (n >> 1)`, i.e. the Sobol
sequence's core recurrence silently wrong in the working tree. Caught by `git status`, restored, and the
restore confirmed by the 60 Sobol tests (property A fails under a corrupted Gray code). Two strandings
in one sitting, in two different files, from two different kill paths.

The hazard is not the stranded edit itself but what follows it: a mutated `src/` plus any staging that is
not path-by-path would **commit the mutation**, and the suite would then be green against a tree nobody
intended. The charter's "never `git add -A`" rule is what stands between the two, which is a procedural
guard for a mechanical failure — and the second instance shows how little warning there is, since a
corrupted Gray code is exactly the kind of defect whose blast radius is the whole Monte-Carlo spine. Candidate fixes, for Cowork to rule rather than for me to choose: a
`signal` handler for SIGTERM/SIGINT around the mutation window; a sentinel file written before the first
mutation and removed after the last restore, which a `verify.py` gate could refuse to see; or a
`git diff --quiet src/` assertion in the instrument's own entry and exit. **No src defect, no oracle
hole — an instrument robustness gap**, recorded because a long unattended sweep is exactly when it bites.

### F-8.3 — the proactive invariant-6 sweep is CLEAN, and 18 of its 32 guards are single-killer oracles — **five of them guarding previously-shipped holes**

**Item 1, run: applied 32/32 · killed 32 · survived 0 · clean.** Every invariant-6 guard in the tree —
**16 expiry bypasses, 14 strip filters and the two `commodity_asian` fixed-fixings selectors inverted**,
across 22 modules — is killed, with the full suite as the verdict. The **option-bypass-needs-ITM** shape
Cowork asked to sweep for has no live instance: no hole. **And T72's two-way close holds** —
`cev_caplet:163` dies to **2** tests (the real fall-through value and the opposite-side floorlet), which
is the specific confirmation item 1 asked for.

**The finding is the killer COUNTS, not the verdict.** Eighteen of the 32 die to exactly one test, and
the five that matter are the ones re-audit #4 found as *shipped holes* and T45 remediated:

| site | finding | killers |
|---|---|---|
| `engine/equity_vanilla.py:45` | **F-4.1** | **1** |
| `engine/heston_fourier.py:166` | **F-4.2** | **1** |
| `engine/forward_cds.py:118` | **F-4.3** | **1** |
| `engine/yoy.py:138` | **F-4.4** | **1** |
| `engine/yoy.py:154` | **F-4.5** | **1** |

§9 is explicit that *"an oracle guarding a **previously-shipped** hole especially wants a second
independent killer"* — the rule under which F-1.2 was given one at T37. **It was never applied to
F-4.1…F-4.5.** T45's remediation was correct and complete *as remediation* (applied 6/6, killed 6, no
`src/` line changed, ratified), and Cowork's T45 ruling addressed a different question — whether the
matured YoY *fixture* should be kept beside the straddling one (it should: it is the sole killer of the
distinct whole-swap guard at `:169`). The second-killer preference for the five holes themselves was
simply not raised, and **the standing table never recorded them**: its `T45–T49` row covers the Gatheral
arc's refusals and says "no load-bearing single killer survived re-audit #5's sweep". So each of five
already-shipped holes has been a single point of failure for **three re-audits** — delete one test and a
hole the library has *already shipped once* silently reopens, suite green.

**Why one killer is not inherent here, which is the part that distinguishes these from the table's
"guards, one condition each" rows.** A *refusal* raises, so any test asserting the raise kills a drop —
one killer is genuinely inherent. These guards **return a value** (PV 0), and the charter's own sub-trap
is that the fall-through intrinsic can *equal* that value: an expired **out-of-the-money** option's
intrinsic is 0, the same answer PV-0 gives. So the killer is not "any test that touches the guard" but
"a test whose expired fixture is specifically **in the money**" — a fixture-dependent oracle, which is
exactly the kind a later edit can neutralise without deleting anything. That is why these five want a
second killer and a refusal does not.

**The other 13 single-killers** (`commodity_spread:128`, `cash:40`, `bates_fourier:144`,
`defaultable_equity:82`, `volatility_swap:173`, `variance_replication:98`, `inflation:84`,
`schwartz:143`, `equity_rate_hybrid:200`, `commodity_asian:264`, `forward_cds:98`,
`first_to_default_basket:54`, `synthetic_cdo_tranche:87`) are the same *shape* in modules that have
never shipped a hole. Recorded, not proposed for hardening: §9 makes recording the floor and hardening
the thinnest the ceiling, and the thinnest five are named above. **Fourteen guards already have two or
more** (`forward_cds:72` and `credit_default_swap:81` at 5, `base_correlation:170` at 4,
`smile_caplet:109` and `mc_exposure:77` at 3, seven at 2, and the two inverted fixings selectors at 22
and 25), so the tree is not uniformly thin — which is what makes the five stand out rather than being a
systemic property of the shape.

**Ruling asked of Cowork:** whether F-4.1…F-4.5 get a second independent killer in a test-only
remediation tranche (the T37/T59 precedent, where F-1.2 and the YoY `ψ` sites were hardened the same
way), and whether the 13 want a standing-table row or hardening too.

### F-8.4 — `VasicekParams.long_run_mean` is unused, untested public surface (a REAL survivor, and in the arc's oldest tranche)

**Items 2, 4 and 5, run: applied 10/10 · killed 8 · survived 1 · errored 1.** The arc's reduction
anchors hold with room to spare — Vasicek's variance integral dies to **34** tests on either mutation,
G2++'s `ρ` cross term to **19** in the bond variance and **4** in the option variance (so the T39 blind
spot is covered: the ρ term is invisible in the one-factor reduction and the companions do exercise it),
**no single-killers**. Item 4 confirmed: **T70's multifactor drift keeping only the first factor dies to
5** tests, so the HJM drift condition bites as a requirement identity at `n ≥ 2`. Item 5 confirmed:
**MM-I's two denominators still die to 2 each** — T59's second killer holds.

**The survivor is real, and it is not an oracle hole — it is dead surface.** `VasicekParams.long_run_mean`
returns `theta / alpha`; dropping the `/alpha` survives all 3069 tests. Measured cause:
**zero consumers in `src/`, zero references in `tests/`** — the property is reached by nothing. So it is
§4's **third survivor kind** (not a hole, not an equivalent mutant — the behaviour *would* differ if
anyone called it), and §6 answers it: surface with no present consumer should not exist. Its docstring
says it exists *"so a caller cannot mistake `theta` for it"* — documentation-as-code with no caller,
which is the speculative-surface shape, and the precedent is T15's redraw loop: **delete it, or relocate
it where a test can reach it.** The note itself is worth keeping — the `theta`-vs-`theta/alpha`
confusion is real and is exactly the T66 parameterisation miss — but it belongs in the class docstring,
not in an unreachable property.

Found by sweeping **the arc's oldest tranche**, which is §9's "full-tree means the oldest tranches"
earning its line again: T67–T72's anchors are all thick, and the one survivor is in T66.

**Ruling asked of Cowork:** delete the property (§6) with the `theta`-vs-`theta/alpha` warning moved to
the class docstring, or keep it as public contract and give it an oracle.

### F-8.5 — a non-probe: renaming a function is not a mutation when the suite imports it

`hjm_drift` renamed to `hjm_drift_UNUSED` as a reachability probe returned **pytest exit 2** — a
collection error, because `multifactor_hjm` and the tests import the name. Per the instrument's own rule
**an ERRORED verdict is never a kill and never a survival**: the run returned no verdict, so it is
reported as a non-probe rather than folded into either count. Recorded for the same reason re-audit #7
recorded F-7.4: a mutation that cannot produce a verdict is worth naming so the next sweep does not
re-attempt it. The reachability question it was meant to ask is already answered by the mutation above —
the drift sum dies to 5 tests, so the function is reached.

### F-8.6 — **A REAL BLIND ORACLE IN T72**: `pytest.approx(x, rel=…)` carries a default `abs=1e-12`, so the ncx2 deep-tail survival test cannot see a survival of **zero**

**Item 3, run: applied 13/14 · killed 11 · survived 2.** The CEV anchor set itself is thick and
answers every question item 3 asked. Each of the printed form's two defects dies separately (**15** and
**9** killers); each of the three interior variants that *passes every requirement oracle* dies to the
MC and the reductions (**13**, **12**, **17**); the deterministic shortcut forced always-on dies to
**11**; **no single-killers**, so the reconstructed formula has no single-point anchor — the worst case
item 3 named is absent. One mutation did not apply (`_dof`'s line moved; re-run below).

**Both survivors are in the new L0 primitive, and the first is a hole in the oracle I wrote for exactly
this defect.** Replacing `_gamma_q`'s direct computation with `1.0 - _gamma_p(a, x)` — i.e. **restoring
the T20a/OI-3 defect the native survival exists to prevent** — survives all 3069 tests. Verified
directly rather than inferred: under the mutation `noncentral_chisquare_sf` returns **exactly 0.0** at
`x = 100, 200, 400, 800` where the truth is 3.9e-14 … 3.2e-148, a **100% relative error**, and
`tests/L0/test_noncentral_chisquare.py` still reports **38 passed**.

**The cause is a testing-library default, which is why reading the test does not reveal it.**
`pytest.approx(expected, rel=r)` does **not** use a purely relative band — it uses
`max(r·|expected|, abs_default)` with `abs_default = 1e-12`. So for an expected value of 8.7e-32 the
effective tolerance is **1e-12 absolute**, and `0.0 == pytest.approx(8.7e-32, rel=1e-11)` is **True**.
Demonstrated: the same comparison with `abs=0.0` is `False`.

This is §4's own clause — *"an ABSOLUTE-tolerance oracle is blind to RELATIVE error where the true value
is near zero"*, the fifteen-tranche `norm_cdf` defect — **recurring inside the test written to prevent
it**, wearing a new disguise: not a hand-written absolute band, but a *library default* silently
replacing the relative band the author asked for. The test's own docstring says *"RELATIVE, not absolute
— tranche 20a's rule. An absolute band passes however bad the relative error gets"*, and the next line
then writes the absolute band by omission. The companion half of the test (that `1 - cdf` **is**
destroyed) kept passing because the CDF was unmutated, so the pair read as healthy.

**Fix, and it is MEASURED rather than proposed:** `abs=0.0` on the two relative-accuracy assertions in
that file takes the mutant from **0 killers to 7**, and the unmutated code still passes all 38 — so the
primitive's accuracy genuinely *is* 1e-11 relative down to 3e-148, and only the band was blind. (Not
applied here: a real hole goes to a test-only remediation tranche, the T45/T51/T59/T65 precedent, with
F-7.1 the direct analogue — a blind oracle found at a re-audit and closed in the next tranche.) **Scope:**
the tree has **357** `pytest.approx(…, rel=…)` sites with no explicit `abs` and only **10** that pin
`abs=0.0`. For the large majority the expected value is `O(1e-4)` or bigger, so the floor is a
*looseness* rather than blindness — a stated `rel=1e-15` is really ~1e-12 absolute, roughly 1e-10
relative on a price of order 1e-2, still tight. The **blind** subset is only those whose expected can
fall below ~1e-12, which is a tail or a difference-of-equals; the one found is in T72. Whether the rest
of that subset is enumerated, and whether an explicit-`abs` discipline is gateable, is Cowork's to rule
— my read is that it is **not** statically decidable (whether an expected value can go sub-1e-12 is a
dynamic property, §9's gateability boundary), so it belongs to the sweep and a method-watch.

### F-8.7 — the second ncx2 survivor: a candidate equivalent mutant, not yet proved

Removing `and index > half` from the Poisson sum's upward break condition survives. The break also
requires `term <= total * 1e-18`, and the walk **starts at the mode**, so the first term is the largest
and the condition cannot fire early — which suggests the clause is redundant *given* the
start-at-the-mode design and the mutant is equivalent. **Not asserted as such:** §4 requires an
equivalent mutant be **proved bit-identical**, never assumed because it looks harmless, and that proof
is owed before this is closed. **Proof delivered, so it is closed as a genuine equivalent mutant rather than left a candidate:**
`_poisson_mixture` with and without the clause is **bit-identical over 588 `(dof, noncentrality, x,
branch)` combinations** — six degrees of freedom from 0.5 to 100, seven noncentralities from 0.1 to
1500, seven arguments from 1e-3 to 2000, against both incomplete-gamma branches. Cause, as suspected and
now established: the walk starts at the Poisson **mode**, so the first term is the largest and
`term <= total*1e-18` cannot fire before the mode's mass is in. The clause is redundant *given* that
design — it would matter for a sum started at `j = 0`, which is what the design deliberately avoids. **No
action**, beyond the note that it is load-bearing only if the start point ever changes.

### F-8.8 — the full-tree pass (T1–T37, plus the T25 formula T71 rehomed) is CLEAN, and the promotion did not weaken T25

**Item 6, run: applied 12/12 · killed 12 · survived 0 · clean · no single-killers.**

**The highest-value probe, answered.** A formula *refactored since its last sweep* is where oracle power
goes stale, and T71 moved T25's expected-positive-exposure body to L0. It did **not** weaken it: the
promoted formula's `φ`-term drop dies to **59** tests and the `Φ`/`φ` swap to **52** — far more than
before, because the L4 caplet and the L5 CVA now both route through it. The promotion made the oracle
stronger, which is the §6 rule-of-two paying a dividend it was not claimed to pay.

The rest of the oldest tranches are thick: Box-Muller's `πv` angle — the charter's named case where a
mutation satisfies the exact requirement `x²+y² = −2 ln u` and fails only the pointwise transcription —
dies to **118** tests, and its radius losing the 2 to **90**; the bivariate normal's quadrature
correction to **64**; Sobol's direction-integer recurrence to **37** and **35**; the Brownian bridge's
two `sqrt`s to **11** and **10**.

**Three thin-but-not-single spots, recorded (§9 makes recording the floor):** Sobol's **Gray code at 2**,
the **CVA one-period product's LGD at 2**, and the bivariate normal's **`ρ = −1` branch at 3**. None is
a single-killer, so none is the standing table's highest-value target, but the Gray code is worth a
second look for a different reason: it is the recurrence a hard-killed sweep silently corrupted in this
very re-audit (F-8.2), and two killers is thin cover for the core of the whole quasi-random spine.

**Scope, stated rather than implied (§9's no-silent-caps rule).** This pass targeted **T1–T37** — last
swept at re-audits #3/#4, some thirty-five tranches ago — plus the one T25 formula T71 rehomed.
**T38–T65 were deliberately not re-swept here**: re-audits #5, #6 and #7 swept them with measured counts
(T60–T64's commodity closed forms at 10–38 killers each, recorded in the standing table), so the
marginal value was low against this sitting's cost. That is a bounded pass, not a full enumeration, and
naming the bound is the point.

## (2) Duplication — REPORTED (Code), coupling and site counts measured

**Measured, not estimated.** Four candidates, at different stages of the rule of two:

| shared thing | sites | files | status |
|---|---|---|---|
| the **strip skeleton** (`if not periods: return PricingResult(Money(0.0, …))`) | 8 | `smile_caplet`, `cev_caplet`, `nominal_cap`, `capfloor`, `yoy`, `first_to_default_basket`, `synthetic_cdo_tranche`, `credit_default_swap` | long overdue |
| the **byte-identical period filter** `[p for p in …periods() if not is_historical(p[1], market.valuation_date)]` | 5 | `capfloor`, `cev_caplet`, `smile_caplet`, `yoy`, `nominal_cap` | overdue |
| **`CapletPeriod` built from curve lookups** | 3 | `nominal_cap`, `smile_caplet`, `cev_caplet` | fired |
| the **forward variance** `σ²·year_fraction(start)` | 3 expressions | `smile_caplet` (×2, the `isinstance` branch), `cev_caplet` | fired — this is C16's own trigger |

**Hull-White's answer is measured, and it is "shares the walk, not the variance."** `nominal_cap`
reaches `bond_option_variance(params, year_fraction(start), year_fraction(pay))` — a **bond-option**
variance over **two** dates — where the smile engines reach `σ²·year_fraction(start)`, the **forward's**
variance over **one**. Same `CapletPeriod`, same walk, different quantity. So HW joins an extraction of
the *walk* and must not join an extraction of the *variance*; C16's row already says this and the counts
confirm it.

**Two distinct per-period shapes, which is the fact a shape ruling turns on.** `nominal_cap` and
`capfloor` **delegate**: a named helper `f(product, model, period) -> float` does its own curve lookups,
so their loop is a one-line `sum(...)`. `smile_caplet` and `cev_caplet` **inline** the curve lookup, the
`CapletPeriod` build and the variance inside the loop. The inlined pair is where the duplication actually
is; the delegating pair already has the seam the extraction wants.

**Where the extraction is well-founded, and where it is not.** The three `NominalCap` engines share a
product type, a `CapletPeriod` and a walk, differing only in the per-period valuation — a shared walker
over those three is sound. Widening it to all eight is **not**: `capfloor` takes an `InflationCapFloor`
and `yoy` a year-on-year swap, so a common walker would need a protocol over products whose payoffs
differ, which is §6's "a real leg and a year-on-year leg share the word *leg* and not the payoff"
exactly. The 8-site skeleton is a genuine repetition, but of **vocabulary** (§3's result shape and §3.6's
invariant-6 partition) rather than of knowledge; the 5-site filter line is the part of it worth one atom.

**Coupling a shared walker would need** (measured against the three): `market`, `currency`, the period
filter, the empty→PV-0 return, `curve`, `measure`, the `CapletPeriod` build, `notional.amount`, `strike`,
`kind`, the `try`/`except`, and the result construction — **identical in all three**. The only real
differences are (a) the exception set (`cev_caplet` adds `ArithmeticError` for its noncentrality refusal,
the other two catch `KeyError` only), (b) the variance quantity (one date vs two), and (c) the valuation
call. (a) is the one that would otherwise be lost silently in a merge.

**Not ruled here (§10).** Reported for Cowork: the three-engine walker with a per-period valuation
callable looks well-founded and the eight-engine version does not; HW joins the walk and not the
variance; and the extraction *creates* the shared value C16 has been missing, which is the point.

## (3) Register-accuracy pass — DONE (Cowork)

- **`references.md`** — `MCINERNEY-ZASTAWNIAK` `worked-from` (ch 3/4/8 mined); **`BROCKHAUS-2016` now cites the
  CEV caplet** (§6.1, a new `src/` citer — `verify.py references` holds). **Both source gaps recorded:** the
  MZ §8.3 figure-only note (so CEV is not re-sought there) and the **Brockhaus §6.1 misprint** in the
  source-defect log (beside BERHAN §5.1, Jäckel §12.8) — the first source *correction* in the project.
- **`stochastic_rates_arc.md`** — marked CLOSED (T66–T72, the split to T72 recorded).
- **`to_go_deeper.md`** — the arc's deferrals recorded (cross-model Bermudan + boundary-split from T68;
  stochastic-vol smile; CEV-as-short-rate, drifted CEV, surface calibration; ch-7 exotics).
- **`OPEN.md`** — OI-7 open and correct (the only item). No new debt.
- **`conventions_register.md`** — **C16 added (T71), per-module, trigger FIRED at T72** and correctly handed
  to this re-audit's duplication pass. C14/C15 guarded; C2 still the widest OWED.

## (4) Charter-staleness pass — DONE (Cowork): clean, and the convergence is the finding

- **No clause stale or in tension.** The two arc additions (§4's reduction-numerics third sibling, T66; its
  regime-per-model tightening, T67) describe reality and cite their tranches. **The notable fact is the
  non-additions:** T68–T72 grew the charter *not at all* — the CEV source-correction, the reconstructed-formula
  anchoring, the independent-factor HJM, the five Cowork spec-misses, all landed inside existing text. The
  method has **converged** on this class of work (the opposite of the early arcs, where most tranches grew §4).
  This is recorded as the arc's headline, not a staleness finding.
- **Method-watches carried (not charter):** the consolidated Cowork spec-discipline watch (derive from this
  source/tree, not the analogous prior tranche); the anchor-pair lens (reduce-to-verified-code / pin-by-
  identity) and the reconstructed-formula lens (requirements + reductions + MC, MC-not-sole); the
  derivation-structure watch (now three polarities, still confirm-only); the three-ceiling heavy-MC watch
  (`test_hw_bermudan` the fourth file to touch the ceiling, cleared losslessly — one unlossless file from
  Bernardo's slow-budget conversation).

---

## Status

- (3) and (4): **DONE** (Cowork). Registers current, both source gaps logged, charter clean (and converged).
- (1) and (2): **DONE (Code).** Four sweeps, **68 mutations applied of 70 planned** (one non-probe
  F-8.5, one line-moved and re-run), **63 killed, 5 survived** — of which **one is a real blind oracle
  (F-8.6)**, one is dead surface (F-8.4), one is a proved equivalent mutant (F-8.7), and the remaining
  two are the same two counted across the T72 in-tranche sweep and this one. Verdicts by sweep:
  invariant-6 **32/32 applied, 32 killed, 0 survived**; the arc's reduction/HJM/MM-I anchors **10/10,
  8 killed, 1 survived, 1 errored**; the CEV anchor set **13/14, 11 killed, 2 survived**; the full tree
  **12/12, 12 killed, 0 survived**.
- **Seven findings: F-8.1** (`to_go_deeper` bin empty, closed by Code; the ch-8.5 bin assignment raised),
  **F-8.2** (the sweep instrument strands a mutation on a hard kill — **twice in this sitting**),
  **F-8.3** (18 of 32 invariant-6 guards are single-killers, **five of them guarding previously-shipped
  holes F-4.1…F-4.5**), **F-8.4** (`long_run_mean` dead surface), **F-8.5** (a non-probe),
  **F-8.6** (**a real blind oracle in T72** — `pytest.approx`'s default `abs=1e-12` hid a zero survival;
  remedy measured at 0 → 7 killers), **F-8.7** (proved equivalent mutant, no action),
  **F-8.8** (the full-tree pass clean; the T71 promotion *strengthened* T25's oracles, 59 and 52 killers).
- Remediation: **owed**, and it has a clear shape — F-8.6's `abs=0.0` (test-only), F-8.4's deletion
  (§6), F-8.3's second killers for the five previously-shipped holes, F-8.2's instrument guard, and
  C16's strip-walk extraction. All five are Cowork's to rule and scope; the reaudit/05–07 precedent is a
  single test-and-tooling tranche carrying them.

Re-audit #8's passes are **complete**; the arc boundary is settled once the survivors are ruled and the
remediation lands.
