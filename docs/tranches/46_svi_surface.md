# Tranche 46 — the SVI surface parameterization

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.53.0`. **Source:**
`GATHERAL-2006` ch 3 (eq 3.20, "Another Digression: The SVI Parameterization") + the Lee (2005) wing
bound it cites. **Pre-registered RED.** The first build of the vol-surface arc (project six); its arc
plan is `docs/gatheral_vol_surface_arc.md`. Registers `GATHERAL-2006` → `worked-from`.

## Rulings (§10 step 1 continued, 2026-11-16 — from Code's build-time findings 0266–0268). These govern §§0–4 below.
1. **The fitted quantity is TOTAL variance `w(k) = σ²_BS(k)·T`, not annualised `σ²_BS(k)`** (finding 2 —
   it blocked, rightly). The formula shape is eq 3.20 with `w(k)` on the left (the modern raw-SVI form, and
   the book's own preferred variable `w := σ²T`, p.25, which is also what it *interpolates* between slices).
   In total variance Lee's bound and the calendar condition are **clean**: the wings are `w(k) → a + b(ρ±1)(k−m)`
   with slopes `b(1±ρ) ≤ 2` **directly**, and the calendar condition is monotonicity of `w` in `T`. My
   spec's annualised `b(1+|ρ|) ≤ 2` was **dimensionally wrong except at `T = 1`** — the `τ=1` degeneracy
   living in a *formula* rather than a fixture, which a one-year fixture would have passed silently. Corrected.
   The one unavoidable conversion (annualised quote → `w`) lives at the fit's **input** boundary; the output
   `w` feeds `models/black.py` (total-variance-parameterised) with no further conversion.
2. **Layer: L3** (Ask b) — settled by §2's own test, not by analogy: SVI's fit *produces parameters*
   `(a,b,ρ,m,σ)` and the formula is closed-form in them, so the params **are** the surface → L3 model-free
   parameter calibration. No L1 piece needed. (And `market/vol_surface.py`'s own note anticipates this
   consumer by name — "a smile model or a vega".)
3. **Butterfly: the first-principles Breeden–Litzenberger density `g(k) ≥ 0`** (Ask a) — the 2006 text
   carries **no** butterfly condition (only calendar), confirmed; the density is derivable from ch 2's
   held material and is a *requirement* oracle. **G–J 2014's closed-form constraints → `to_go_deeper`
   acquire-source note**, not a blocker.
4. **T46 cites ch 7 for the wing bound** (finding 1, made non-optional by finding 3's correction): ch 7's
   model-independent Lee bound `β*, α* ∈ [0,2]` is **what pins eq 3.20's radical** — the extraction confirms
   the vinculum *scope* (it covers `(k−m)²+σ²`, excludes `ρ(k−m)` — the v0.4.0 mis-nesting guard), and the
   radical's *identity* is forced because a non-radical 3.20 is **quadratic** in `k`, which violates ch 7's
   bound unboundedly. §4's derivation-step independence, across chapters. T47 keeps ch 7's *quantitative*
   content (`g(q*)`, Benaim–Friz, eqs 7.13–7.14).
5. **The cross-module convention is `C14`** (Code confirmed the next free id; `verify.py conventions`
   enforces the gap-free run). It records "the SVI surface and `models/black.py` agree the quantity is
   **total variance** `w = σ²T`". **Born `per-module`**, honestly — total-variance SVI reads the clock
   (×`T`), so C14 sits directly on **C2** (the vol-clock convention, itself OWED); it is **not** claimed
   *guarded* unless a T46 oracle structurally routes `w` end-to-end through `black.py` such that a dimension
   mismatch fails a test. Code adds the row during the build (it names the real modules/test).
6. **Reuse-target correction** (Code's §5/T22 catch): total variance is `models/black.py`'s parameterisation
   (the JY era — "parameterised by TOTAL VARIANCE … and that is the point"), **not** T31. T31 is variance
   *replication* (`K_var`, the log contract) and is T48's reuse, not T46's. §1 below is corrected.

**No charter edit.** The `τ=1`-in-a-formula (ruling 1) is an instance of the τ=1 trap caught by a dimension
check at spec review — a one-year fixture would not have caught it, which is why Code catching it *before*
RED is the system working; if a unit-degenerate spec formula recurs it earns a line. The cross-chapter
radical pin (ruling 4) is an instance of §4's derivation-step independence + requirement-oracle discipline.
Both are the charter covering new cases rather than growing — the arc opens lean.

## 0. The product worth having
**The SVI (stochastic-volatility-inspired) parameterization of the implied-volatility smile** — the
market-standard closed-form fit of one expiration's total-implied-variance slice, with the no-arbitrage
conditions that make the fitted surface admissible. For each expiration, in log-strike `k`:

```
σ²_BS(k) = a + b [ ρ (k − m) + √( (k − m)² + σ² ) ]            (Gatheral eq 3.20)
```
five parameters per slice: `a` (overall level), `b` (angle/spread of the wings), `ρ` (skew /
left-right asymmetry), `m` (horizontal shift of the minimum), `σ` (smoothness of the vertex). The
value worth having is a smile a desk can **fit to quotes and trust**: evaluate `σ²_BS(k)` anywhere,
with calendar and butterfly arbitrage ruled out.

## 1. The pieces and their layers
- **The SVI slice — a model-free parameter calibration.** Fitted to **quoted** implied vols by closed-form
  evaluation + a nonlinear least-squares solve; it reads only quotes and L0 (`norm_cdf` for the density
  check), inverts **no engine**. **Layer ruling at build (Cowork leaning: L3):** SVI yields parameters
  evaluated directly as a surface-model, parallel to `models/implied_vol_surface.py` (L3) — so L3
  model-free parameter calibration, not an L1 market bootstrap (`market/vol_surface.py`) and not an L4
  engine-inverting calibrator (§2). Confirm at build by reading what the two existing surfaces produce and
  how they are consumed; rule L1 only if SVI's output *is* the market-data surface others read.
- **The no-arbitrage conditions** — requirement predicates on the fitted parameters / surface
  (calendar, butterfly, wings), tested, and surfaced as a `PricingFailure`/refusal where a fit is
  inadmissible (§3.4 — a silently arbitrageable surface is a plausible wrong number).
- **Reuse:** the implied-vol-surface representation (T9/T32) and the **total-variance parameterisation of
  `models/black.py`** (ruling 6 — *not* T31; T31 is variance replication, T48's reuse). The SVI output `w`
  feeds `black.py` directly. No new L0 primitive expected.

## 2. The oracle (RULED)
The **synthetic round-trip** is the correctness anchor; the **no-arbitrage conditions** are
transcription-proof requirements; the **Heston↔SVI** agreement is the second-source cross-check.

- **The synthetic round-trip — the correctness anchor (§4, the T8/T32/T11 pattern).** Generate a smile
  from **known** SVI parameters, fit SVI back, and recover the parameters **and** reprice the smile.
  Non-telescoping (generation and fit are different code). Carries the **T11 identifiability** discipline:
  SVI's five parameters trade off, so assert a **multi-start parameter spread** (does the fit land on one
  point?) **and distance-from-truth** with the synthetic optimum among the starts — spread shows
  non-uniqueness, distance shows wrongness. A noise-dominated condition number is **not** the oracle (T11).
- **The `b → 0` reduction — the exact anchor (§4).** `b=0` collapses eq 3.20 to a flat variance `a`;
  exact by construction. **Companion (required, and non-degenerate in `b`):** a fit with `b ≠ 0`, `ρ ≠ 0`,
  `m ≠ 0` exercising **both** wings — a reduction a flat-smile fixture also satisfies is vacuous (T39: the
  companion must move the very parameter the reduction zeroes).
- **The wings + Lee's bound — a requirement oracle (§4), in TOTAL variance (ruling 1), citing ch 7
  (ruling 4).** As `k → ±∞`, `w(k) → a + b(ρ ± 1)(k − m)`: two linear asymptotes with slopes `b(1+ρ)` and
  `b(1−ρ)`. **Lee's moment formula (ch 7, pp. 97–98, `β*, α* ∈ [0,2]`) bounds each total-variance wing
  slope by 2** — `b(1+|ρ|) ≤ 2`, clean because it is total variance (the annualised form would carry a `T`
  — the ruling-1 correction). This same linear-wing requirement is what **pins eq 3.20's radical** (a
  non-radical 3.20 is quadratic, which the bound refuses unboundedly). Transcribe the asymptote pointwise,
  and assert the bound as a requirement a mutant dropping `b` or flipping `ρ`'s sign breaks.
- **Calendar-arbitrage condition — a requirement oracle.** Total implied variance `w(k,T) = σ²_BS(k,T)·T`
  must be **non-decreasing in `T`** at every `k` (slices do not cross). Assert on a two-slice fixture that
  genuinely differs (the §4 degenerate-fixture premise: non-flat, non-nested-trivially).
- **No-butterfly condition — a requirement oracle, with a source-boundary Ask.** The risk-neutral density
  implied by the smile must be **non-negative**, `g(k) ≥ 0` (the Durrleman / Gatheral–Jacquier condition).
  **Ask (a):** the 2006 text gives the calendar condition and says butterfly is "relatively easy to
  eliminate" but may not carry the rigorous `g(k)≥0` form — that is **Gatheral–Jacquier (2014)**. Rule at
  build whether deriving `g(k) ≥ 0` from first principles (the Breeden–Litzenberger density of the fitted
  smile, oracle-able with our discipline) suffices, or whether G–J 2014 must be obtained (a `to_go_deeper`
  acquire-source note if the closed-form G–J constraints are wanted rather than the density check).
- **The scale/level guard (§4, T43).** The oracle set must not be all scale-invariant — the round-trip
  reprices (telescopes on shared convention) and the wings/reduction are shape. **Add one absolute-level
  oracle:** the vertex total variance `w(m) = a + bσ` (total-variance params, ruling 1) against an
  independent number, so a uniform rescaling of the slice cannot survive the whole set.
- **Heston↔SVI cross-check (second source).** ch 3 fits SVI to a **Heston** surface; our Heston (T33)
  generates a smile, SVI fits it, and the two agree to a **measured** tolerance (they are different objects
  — an SVI fit of a Heston smile is an approximation, so assert the gap is small **and** report it, never
  treat it as exact cross-source identity — §4's same-formula caveat).

**Mutation (premises asserted up front — the arc's hard lesson).** Each must die: drop `m` (→ the
minimum-location; needs `m ≠ 0`); flip `ρ`'s sign (→ the skew; needs `ρ ≠ 0` and an **asymmetric** smile,
else the two wings are mirror-images and the mutant survives — the T17 symmetry-degeneracy); drop `b`
(→ the whole smile flattens; the reduction companion); bypass the butterfly/calendar refusal (→ a
fixture that is actually arbitrageable, supplied so the guard's reject branch is exercised — the T33b
input-population premise). **Premises:** a **non-flat, asymmetric** smile spanning enough strikes to
exercise **both** wings; ≥ 2 expirations for the calendar check; an arbitrageable fixture for each refusal.

## 3. Process
Branch `tranche/46-svi-surface`. `test` (RED: the synthetic round-trip + identifiability + the `b→0`
reduction & companion + the wing/Lee/calendar/butterfly requirements + the absolute-level + the
Heston↔SVI cross-check + the mutation table) → `feat` → `docs` → `chore(release): v0.53.0`. **Transcribe
eq 3.20 and the wing limit in both renderings (`-raw` + `-layout`, §4).** `source:` `GATHERAL-2006`;
`crosscheck: quantlib` — QuantLib has an SVI interpolation, a real target. **Open the draft PR early;
pre-register the round-trip + the arbitrage requirements first.** **Asks to raise:** (a) the butterfly
source boundary (G–J 2014 vs the density derivation, above); (b) the **layer ruling** (L3 vs L1), with
your reading of how `market/vol_surface.py` and `models/implied_vol_surface.py` are produced/consumed.

## 4. What this is NOT (scope fence)
No **local vol / Heston rebuild** (ch 1–2, held — T32/T33). No **SVI calibration to a live market feed**
(a downstream concern, §0 out of scope). No **surface interpolation zoo** (one monotone total-variance
interpolation across slices if the calendar check needs it, ruled at build — not Stineman-plus-alternatives).
No **asymptotics beyond the wing bound** (that is T47, provisional). No **SVI *surface* arbitrage-free
calibration across all slices simultaneously** beyond the pairwise calendar condition — the full
surface-SVI (eSSVI) is `to_go_deeper`. T46 is one admissible slice parameterization + the conditions that
make it trustworthy.

---

## 5. Cowork's ruling on T46 (§10 step 4, 2026-11-16) — RATIFIED

Built as ruled, 1852 tests, eleven gates green both runners. **C14 born `guarded`, not per-module** — the
ruling-5 nudge paid off: routing the Heston↔SVI cross-check through `black.py` on *prices* reprices at
0.0030% vs an 0.05% band, and the `×T`-omitted companion reads **81.1%**, so the guard is *proved* to
discriminate, not assumed. That is the "prove the guard discriminates" discipline on a convention oracle.
Ratified.

**Item 2 — the stale-bytecode false "killed" → a §9 charter line (added).** A first sweep read 26/27
killed; three verdicts were false because CPython ran stale `.pyc` (mutation written within one mtime tick,
`(mtime,size)` unchanged, no recompile). The existing applied-count clause is **blind** to it — applied was
27/27 and correct, the mutation *did* reach the source — so this is a genuine gap, not an instance, and the
dangerous direction (a false *killed* hides a hole). §9 gains "**the mutated code is the code that ran** —
invalidate the bytecode cache per mutant". Earned.

**Item 3 — the resolution-parameter survivor is NOT a fourth kind.** Coarsening the butterfly scan 401→3
survives because SVI density breaches are structurally **wide** (≥0.05 in log-strike across a 324-slice
sweep, ~200× the step), so a missed breach is **unreachable given `g`'s own smoothness** — the charter's
*unreachable-code* kind (the mutated behaviour would misbehave *if reached*, and the criterion is
reachability-from-a-test, not improbability). The one twist — it is a continuous *resolution* parameter, not
a dead branch — is handled by the existing **assert-the-resolution** discipline (T17): the remedy is not
deletion but a resolution oracle asserting step ≪ minimum breach width, which is what you built. So it is
the composition of two clauses already in §4, not a new one — **no charter edit.**

**Items 1 and 4 — affirmed, no action.** (1) Replacing the butterfly margin with the **analytic**
asymptotic condition `g(±∞) = ¼ − (b(1±ρ))²/16 ≥ 0 ⟺ b(1±ρ) ≤ 2` is excellent — the tails *are* Lee's
bound, unifying rulings 3 and 4, and it makes the G–J acquire-source note buy a *named* gap rather than
"more rigour". (4) Promoting `norm_pdf` to L0 (rule-of-two fired at the second copy — the stale
"has not been promoted" docstring was a comment asserting a now-false state, the CREPEY-header shape one
level down) and making the start heuristic *earn its keep* (beats a constant on exactly one slice, now its
test) are both the right §6 calls. §5 paid off: the stock carried the exact dimensional defect the spec
nearly shipped — content and regression, not authority.

The arc's second tranche (T47, surface asymptotics) is next; its standalone content is now clearer — T46
took the wing bound, so T47 owns ch 7's quantitative refinements (`g(q*)`, Benaim–Friz), confirmed thin
and possibly foldable (§1 pull gate). Spec follows on the handoff.
