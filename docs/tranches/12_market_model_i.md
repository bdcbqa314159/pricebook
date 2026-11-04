# Tranche 12 — Market Model I (ch 3), the second model on the same instruments

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.14.0`.
**Source:** `BERHAN-2012` ch 3 (`worked-from`) — **verified it publishes the formulas in full**:
forward-rate dynamics (3.1.2–3.1.4), forward-index dynamics (3.1.5–3.1.6), the bivariate-lognormal
pair with explicit `μ`/`σ`/`ρ`, and Mercurio's freezing procedure. McInerney ch 5–6 is LMM
background, not required for the YoY/cap formulas. **Chosen by Bernardo**: finish the two market
models, MM-I first.

**Status:** **BOTH BUILT.** (a) v0.14.0 — the capability seam. (b) v0.15.0 — MM-I. **Results: §8, §10.**
**Answer to §0's open question: ONE protocol, two methods. No second dispatch key needed.** See §8.

## 0. The crux — a SECOND model prices the SAME products, and the registry is product-keyed

The YoY swap and the cap/floor already have L4 pricers, and those take a **concrete
`JarrowYildirim`** today (`model.convexity(...)`). MM-I prices the *same* products with a *different*
formula (a freezing-procedure bivariate-lognormal expectation, not JY's `e^C`). The registry
dispatches on **product type**, so there is one pricer per product — and a second model must not be
served by an `isinstance`-on-model branch inside it (§1's no-ladder law).

**Ruled: introduce a capability protocol — rule-of-two fires here (JY + MM-I are its two
consumers), exactly as Black and the HW core were extracted at their second consumer.** The YoY and
cap/floor pricers depend on a **protocol** a model implements, and both JY and MM-I implement it;
the pricer composes the model's capability structurally, never switches on its type. This is the
doc-22 structural-dispatch design, earning its place now rather than speculatively in T4.

**The open question I need Code to settle by building, not by my assertion:** can both models'
YoY-period value be written behind **one** protocol method (e.g. `yoy_period_value(period, market)`
or a `yoy_convexity`-shaped call), or do their formulas differ in shape so much that the honest
seam is a **two-key `(product, model)` registry** instead? Build the protocol first; **if MM-I's
formula will not fit the JY pricer's shape without contortion, that is an `Ask:` back to me — a
forced-fit is worse than a second dispatch key, and the charter would rather gain a ruling than hide
a bent abstraction.**

## 1. Pieces and layers (subject to the §0 question)

```
L3  models/market_model_i.py — MM-I: the forward-rate/index vols + correlations, and the
                               freezing-procedure YoY and cap/floor adjustments (BERHAN 3.1–3.2)
L3  a capability protocol the YoY/cap pricers depend on; JY and MM-I both implement it
L4  engine/yoy.py, engine/capfloor.py — REFACTORED to the protocol (pure refactor, JY's numbers
                               bit-identical, guarded by T4/T5 oracles), then MM-I flows through
```

## 2. Split (§6) — build the seam first
- **(a)** the capability protocol + refactor the JY pricers onto it, JY numbers **bit-identical**
  (pure refactor, T4/T5 oracles green, no new RED). This proves the seam on the model we trust.
- **(b)** MM-I itself — the model + its freezing-procedure formulas — flowing through the seam, with
  its own oracle.
If (a)'s refactor can be shown bit-identical in one pass it may ride with (b); keep them separable.

## 3. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **cross-MODEL reduction — the strongest anchor here:** at zero vol every model collapses to the **same deterministic forward**, so MM-I's YoY and cap/floor at zero vol equal JY's at zero vol equal the model-free forward value | **exact, cross-model** | a wrong MM-I formula diverges from a limit three independent code paths now share |
| 2 | **companion:** MM-I's convexity/adjustment MOVES the price away from that limit as its vols rise (non-vacuous) | structural | a no-op MM-I would pass oracle 1 alone |
| 3 | **independent transcription of BERHAN 3.1.x** (the bivariate `μ`/`σ`/`ρ` and the freezing form), `-raw`+`-layout`, parameter-swept — after §5.1 assume the source can misprint, so a requirement guards it | **1** | every sign and factor in the freezing formula |
| 4 | **put-call parity: MM-I cap − floor = the MM-I YoY swap** (same model, different code each side) | structural | the sign/leg error |
| 5 | **the freezing procedure is an APPROXIMATION — bound it, don't hide it:** assert MM-I agrees with a Monte-Carlo (or the exact bivariate integral) of 3.1.5–3.1.6 to the approximation's stated tolerance, not to 1e-12 | 2 — trusted/convergence | a freezing error masquerading as "close enough"; the tolerance is the honest one, named |
| 6 | failure-is-a-value / raises per convention | structural | the contracts carried |

Oracle 1 is the cross-model anchor; oracle 5 is the one genuinely new shape — MM-I is *approximate*
by construction (freezing), so its oracle is a bounded agreement with the un-frozen quantity, stated
as a tolerance, never dressed as exactness. No number fitted to a figure (ch 3 has none anyway).

## 4. Design decisions (ruled)
- **Capability protocol, not a type switch** (§0) — subject to Code's build confirming one method
  fits both; a two-key registry is the fallback and a ruling, not a workaround.
- **The JY-pricer refactor is pure** — JY's prices bit-identical, measured as the Black and HW
  extractions were.
- **MM-I is a market model: its observables are the forward rates/index, lognormal** — a different
  model from JY, so (shared-shape-not-word) it does not subclass JY; the shared thing is the
  *protocol*, not the model.
- **Cross-check dispositions:** `models/market_model_i.py` → `crosscheck: quantlib (inflation market
  model / LMM)`; the protocol module, if separate, → `crosscheck: none — a protocol, no numerics`.

## 5. Stock
Mine `_stock/.../options/inflation_*` for the freezing/market-model shape; read the formulas from
**BERHAN ch 3** (the held source, which publishes them), not the stock's code.

## 6. Process
Branch `tranche/12a-model-protocol` then `tranche/12b-market-model-i`. `test` (RED pre-registered) →
`feat` → `docs` → `chore(release)`. Mutation on MM-I's freezing formula; the zero-vol cross-model
reduction and the MC-bounded approximation are the anchors. `source:` cites `BERHAN-2012` ch 3.


---

## 8. Result — §0's question answered by building: ONE protocol, and the source supplies its shape

375 tests, eight gates green, ruff and pyright clean, zero suppressions. Six mutations on the seam,
all caught.

### The answer, and the evidence
**One protocol with two methods fits both models. No two-key `(product, model)` registry is needed,
and no formula was contorted to fit.** The evidence is in BERHAN's own text — factor the two
year-on-year leg formulas side by side and the later-period term of eq 2.5.12 (JY) and of eq 3.1.11
(MM-I) are both

```
psi_i · Pn(t,Ti) · [ E − 1 ]
```

with `E` the `Ti`-forward expectation of `I(Ti)/I(Ti−1)` — for JY
`(Pn(Ti−1)/Pn(Ti))·(Pr(Ti)/Pr(Ti−1))·e^C`, for MM-I the bivariate-lognormal integral of 3.1.11. The
cap/floors agree too: JY's eq 2.6.5 and MM-I's §3.2 inner expectation are both **Black on `(E, V²)`**,
differing only in what `V²` is.

So the seam is **the source's own decomposition**, not an abstraction invented to hold two things —
and both quantities are ones BERHAN names (`m`, eq 2.6.2; `V²`, eq 2.6.4). That is why this is not a
forced fit: had the protocol been JY's `convexity`, MM-I would have had to manufacture a fake `C` by
taking the log of its integral, which *is* the contortion the spec warned about.

### The refactor is NOT bit-identical, and that is reported rather than claimed otherwise
The spec required JY's numbers bit-identical as the proof the refactor is pure. **It cannot be
met**, and the reason is the factoring itself: `psi·(Pn(Ti−1)·PrRatio·e^C − Pn(Ti))` and
`psi·Pn(Ti)·(m − 1)` are the same quantity in exact arithmetic but reorder a division, so the last
digits move. Measured on a 4-period swap:

| | value |
|---|---|
| pre-refactor arithmetic | 36119.025432954419 |
| refactored through the protocol | 36119.025432954477 |
| relative difference | **1.6e-15** (5.8e-13 bp of notional) |

Every tranche 4 and tranche 5 oracle passes unchanged at its own stated tolerance. The substitute
for a bit-identical claim is therefore: *agreement to 1.6e-15, with the cause named (a reordered
division), and the full pre-existing oracle set green.* Saying "bit-identical" here would have been
false, and the alternative — keeping JY's grouping — would have required the protocol to expose
`convexity`, which is what makes MM-I impossible.

### The oracle that a type annotation cannot be
`_StubRatioModel` implements only the two protocol methods and is not a `JarrowYildirim`. Both
pricers price it correctly, and the cap/floor's output is *exactly* Black on the stub's two numbers —
so the pricers demonstrably read **only** `m` and `V²` from a model. If either still reached for
`model.convexity` or `model.params`, that test would raise. A protocol annotation alone proves
nothing at runtime; this does.

## 9b. Cowork's ruling on (a) — PR #17 (§10 step 4, 2026-11-03)

**Reviewed the built seam, not only the narrative.** `engine/yoy.py` and `engine/capfloor.py` read a
model only through `expected_index_ratio`, `index_ratio_variance` and `market` — no `model.convexity`
anywhere in a pricer — and `models/index_ratio.py` is a two-method protocol over the quantities the
source names. The seam is the one specced.

**Ruling 1 — accept the 1.6e-15; the two-key registry is rejected.** The purity proof is the
*exact-arithmetic identity* `Pn(Tᵢ)·m = Pn(Tᵢ₋₁)·PrRatio·e^C`, which makes the two groupings the same
quantity in ℝ; the 1.6e-15 is one division round-trip (~7 ULP), the right size for exactly that
reordering, and every T4/T5 oracle stays green at its own tolerance. "Bit-identical" was the wrong
requirement here, not an unmet one: it is available only when op order is preserved, and preserving
it means keeping JY's grouping — which forces MM-I to manufacture a fake `C` from the log of its
integral, the §6 contortion. Reporting the ULP gap with its cause is the honest substitute, and the
`_StubRatioModel` oracle (a non-JY model priced correctly, cap/floor exactly Black on its two
numbers) is the runtime proof the pricers read only `(m, V²)`. **Charter §4 gains the general rule:
a pure refactor that legitimately reorders floating-point ops is proved by the exact-arithmetic
identity plus bounded agreement, not by bit-identity.**

**Ruling 2 — adopt the draft-PR convention as §10 step 6's mechanism.** Opening the PR as a draft and
marking it ready only at step 6 makes "merge when Code says, never when Cowork finishes" enforced by
the tooling rather than by a line in a message; readiness *is* the "merge now" signal. Written into
§10 (steps 3, 6, and a precedent note). Flagged for Bernardo as §10's owner — it implements the
existing step-6-before-step-7 law rather than changing who does what, so it is adopted, not merely
proposed, subject to his veto.

**Findings 9.1–9.3 — all accepted as Code reported them.** 9.3 (the `m > 1` requirement that ignored
the convexity, replaced by the `σr → 0` decomposition that pins the sign of `C`) is the stronger
requirement and the right fix; it is the fourth wrong-requirement caught by execution, and belongs in
the checkpoint tally. No change requested; **build (b).**

## 9. Findings for Cowork (§10 — raised, not decided)

### 9.1 `IndexRatioModel` carries `market` as a property, so `issubclass` does not work on it
Only `isinstance` does (`TypeError: Protocols with non-method members don't support issubclass()`).
Harmless here, and noted so a future test does not reach for the wrong one. The oracles check
behaviour regardless — tranche 5's lesson that a runtime protocol probe checks *names*, never
signatures, and that pyright is the authority.

### 9.2 `JarrowYildirim` gained a `_currency()` helper that infers its single currency
JY is single-currency by construction — nominal and real curves of one economy under the
foreign-currency analogy — so inference is not a guess, and it **refuses when ambiguous** rather than
picking. Same stance as `CurveSet.discount` refusing a missing key, and as tranche 10(b)'s
calibrator. Flagged because it is new behaviour on a model, not just a move.

### 9.3 A wrong requirement of mine, caught by its own test
I asserted `m > 1` on an inflationary curve pair. False at 3y→4y, where `m = 0.9914`: a negative
convexity overcomes the 2.1% carry. **A requirement that ignores the convexity is not a requirement
of this model** — the convexity is the whole point. Replaced with the honest decomposition: at
`σr = 0` the convexity vanishes exactly and then `m > 1` *is* forced by the curve pair, and the
convexity is asserted to move it. Stronger than what it replaced, because it also pins the sign of
`C`'s contribution. Fourth instance of a wrong-requirement caught by execution rather than review.


---

## 10. Result — (b) built, and the thesis's "approximately constant" does not hold

408 tests, eight gates green, ruff and pyright clean, zero suppressions.

### The seam worked: MM-I flows through both pricers with no change to either
That was (a)'s whole claim and it is now demonstrated rather than argued. `MarketModelI` implements
`IndexRatioModel` and the year-on-year and cap/floor engines price it untouched — including
tranche 5's cross-product identity `cap − floor = the YoY leg`, which now carries a **second model**
through the same seam. No `isinstance`, no second dispatch key.

### The anchor: three independently-coded paths, one limit
At zero volatility MM-I's integral collapses to the deterministic forward ratio, JY's `e^C`
collapses to 1, and the curves give the same number directly — all three agree to 1e-13, sharing
nothing but the curves. The strongest oracle in the tranche, and it is exact rather than a numerical
limit. Its mandatory companion requires the vols to move MM-I's ratio, and a further oracle requires
MM-I and JY to **disagree** away from the limit, since two different models silently agreeing would
mean one is not doing what it claims.

### The transcription is corroborated by a second path INSIDE the source
`pdftotext -layout` mangles eq 3.1.11 into broken columns, so §4's dual-rendering rule could not
settle it. But page 25 gives the same integral in its **pre-substitution** form — over `y` against
the conditional density, before `z = (y − µy)/σy`. The two agree to **2.2e-16**, and a misprint
would have to appear identically in both forms to survive that. Both are oracled. That is a stronger
guard than two renderings of one printed line, and worth reaching for whenever a source derives a
formula in steps.

### Where the approximation actually lives — the spec had it in the wrong place
§3's oracle 5 expected the freezing procedure to make the *leg value* approximate. **It does not.**
Given its per-period inputs, eq 3.1.11 is an exact expectation. Mercurio's freezing governs the
**recursion between periods**:

    σ_I,i−1 ≈ σ_I,i + σr·τFr(0)/(1+τFr(0)) − σn·τFn(0)/(1+τFn(0))

where the exact relation carries the *stochastic* `F(t)`. So the approximation is a statement about
the index-vol **term structure**, not about any single period's price.

### And the headline negative: the recursion drives `σ_I` negative in two steps
Walking it backwards from `σ_I = 0.012` with `σn = 25%`, `σr = 20%` on this library's own sloped
test curves:

| step | `σ_I` |
|---|---|
| i=4 | +0.012000 |
| i=3 | +0.005605 |
| **i=2** | **−0.000542 — not a volatility** |

The adjustment is negative whenever `σn·τFn/(1+τFn) > σr·τFr/(1+τFr)` — the ordinary case, since
nominal vol and nominal forwards both exceed real. So the source's *"we can still assume that the
volatilities σ_I,i are all constant"* **does not survive a realistic parameter set**, and the
relation cannot be iterated far. `index_vol_step` now refuses a non-positive result rather than
letting it reach a `√` and surface as a `nan` three layers away.

This is the **third honest negative** the inflation arc has produced about its own source's models,
after JY's flat smile and §5.3.2's structural underdetermination.

### The error bar, and why the first version of it was dishonest
A 200bp forward move shifts `σ_I,i−1` by **71% of `σ_I`** — measured, and near-linear in the shock
(35% at 100bp, 18% at 50bp). My first oracle asserted `worst < 0.01`, which *reads* as "the error is
small" while the measurement says it is large. An arbitrary smallness threshold would have dressed a
material approximation error as a bounded one, so the test now asserts the **magnitude** and its
linearity instead.

It is a **sensitivity** bound, not a distributional one: a probability-weighted bound needs an LMM
simulation of the forward rates, which this tranche does not build and does not pretend to.

## 11. Findings for Cowork (§10 — raised, not decided)

### 11.1 Mutation found a real hole that properties alone left open
Seven mutations on MM-I and the quadrature; six caught. The survivor — dropping `/(1+τFr)` from the
freezing coefficient — was **not** an equivalent mutant: at `τF ≈ 0.03` the denominator is worth ~3%.
My freezing oracles asserted only *properties* (vanishes at zero vol, moves with the vols, bounded
under a shock) and never transcribed the formula pointwise. Fixed with a pointwise oracle plus a
companion requiring the un-divided form to differ measurably; both denominator mutations now fail.
**Properties are not a substitute for transcribing the formula once** — a candidate §4 note.

### 11.2 `MMICorrelations` is a second correlation type, deliberately
Two fields, not JY's three: neither eq 3.1.11 nor §3.2 reads a nominal-index correlation, so sharing
`JYCorrelations` would force the carried-unread field §6 forbids. Rule-of-two fires on a shared
*shape*, and these shapes differ.

### 11.0 Cowork's ruling on (b) — PR #18 (§10 step 4, 2026-11-04)

**Reviewed the numerics, not only the narrative.** The 3.1.11 integrand matches the stated form
term for term; the three-path zero-vol reduction is exact by construction (at zero vol the integrand
→ `(1+τFn)/(1+τFr)`, which is JY's `e^C→1` limit and the curve ratio directly); `index_ratio_variance`
is correctly on the period `[Tᵢ₋₁,Tᵢ]`; and the negative-σ_I finding is sound — the sign condition
`σn·τFn/(1+τFn) > σr·τFr/(1+τFr)` is the ordinary case, so `index_vol_step`'s refusal is a real guard,
not padding. Accepted: v0.15.0, the third honest negative of the arc.

**11.1 — accepted, and it is the most reusable thing in the tranche.** "Properties are not a
substitute for transcribing the formula once" is now charter §4: requirement/property oracles and a
pointwise transcription catch *different* failures (a property is blind to a missing term it still
tolerates), so every load-bearing formula carries both. The dropped-`/(1+τF)` mutant surviving 404
property-only tests is the precedent.

**The derivation-step corroboration (raised in the Ask) — accepted, charter §4.** An earlier
derivation step is an independent check on the final formula, stronger than two renderings of one
line and available when `-layout` mangles the final equation. Folded into the Independence bullet
with the p.25 pre-substitution integral (2.2e-16) as the precedent.

**Oracle 5 re-placement — accepted as the correct finding.** The spec put the approximation in the
leg value; it is in the between-period σ_I recursion, and 3.1.11 is exact given its inputs. The
error bar asserts magnitude and linearity (the 71%-of-σ_I shock), not an arbitrary smallness
threshold — exactly the "tolerance is a finding, never a knob" instruction, and Code's catch of its
own first `worst < 0.01` version is the discipline working.

**11.2 — accepted, no change.** `MMICorrelations` two fields not three is a clean §6 application:
ρ_nI is unread by 3.1.11 and §3.2, so carrying it (by sharing `JYCorrelations`) is the forbidden
unread field. Rule-of-two fires on shared *shape*; the shapes differ. This recurs in MM-II (a third
correlation set) — see T13.

**11.3 — accepted.** `foundation/quadrature.py` at L0 is right: Gauss-Hermite is finance-free pure
math, the fixed rule has no tolerance knob (fits §9's reproducibility contract), and cross-checking
against `scipy.integrate.quad` puts a different integrator on the other side of the oracle. Good.

**My §10 citation slip, corrected by Code — acknowledged.** I wrote "four of PRs #1–#5 merged ahead
of review"; the actual PRs were #7/#14/#15/#16 (#6 caught in time). Code fixed the always-loaded tier
and was right to — a false citation in the charter is exactly the kind of unverified claim the record
should not carry. **Build T13 (Market Model II).**

### 11.3 `foundation/quadrature.py` is a new L0 adapter
Gauss-Hermite rather than `scipy.integrate.quad`, because an adaptive integrator chooses its own
nodes per integrand, carries an internal error estimate and can shift between scipy versions — none
of which belongs under a pinned reproducibility contract. The fixed rule has **no tolerance knob**;
its accuracy is instead measured against moments known in closed form. The oracle cross-checks MM-I's
integral against `scipy.integrate.quad`, so a different integrator sits on the other side of the
check.
