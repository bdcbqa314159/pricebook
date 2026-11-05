# Tranche 18 — the Bermudan swaption under the LMM: the headline, sandwiched without a tree

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.21.0`.
**Source:** `JAECKEL-2002` §12.6–12.8 (the parametric exercise boundary) on top of §12.1–12.5
(already built as T16). **Read directly, not inferred:** §12.7/Fig 12.11 state the two coordinates
(the forward rate `f_i` and the next coterminal swap rate `s_{i+1}`); §12.8 states the algorithm
(optimise a parametric exercise-decision function on *training* paths by maximising the pathwise
exercise-dependent value, eq 12.41, with a BFGS optimiser; price on *fresh* paths). **Chosen by
Bernardo** as the BGM/J headline. Third and final unit of the arc (T16 → T17 → T18).

## 0. Scope — build Jäckel's parametric policy; do NOT build the tree

Jäckel §12.6 uses the non-recombining tree (ch 13) to *discover* the coordinate pair in which the
exercise domain separates. **He states the pair he settles on (`f_i`, `s_{i+1}`), so we take it and
skip the tree.** The tree is his tool for finding and visualising the coordinates, not a dependency of
the pricing method once the coordinates are fixed — and our **sandwich oracle self-diagnoses** whether
the chosen policy is good (a loose sandwich says the coordinates/parametrisation are inadequate for a
given trade), which is the job the tree did for him. Building ch 13 is a whole second numerical method
(§6: not the real problem here); it is recorded as a `to_go_deeper` route (the independent
cross-check Jäckel himself uses), not built.

## 1. The method (from Jäckel) vs the anchors (ours)

- **Transcribed (§4):** the parametric exercise-decision function in the (`f_i`, `s_{i+1}`) coordinates
  and the training objective (eq 12.41) — read `-raw`+`-layout`. The two coordinates and the
  "optimise on training, price on fresh" split are the load-bearing structure.
- **Ours:** the T16 path engine (now widened — §3 ruling below); the T17 European swaption as the
  exact degenerate anchor; the perfect-foresight estimator on the simulated paths as the upper bound.

## 2. Pieces and layers

```
L2  products/bermudan_swaption.py — the Bermudan: the underlying coterminal swap, the strike, and the
                                   set of exercise dates. Frozen (§6 ≤5 fields; bundle the swap).
L3  models/exercise_boundary.py (or engine-local) — the parametric exercise-decision function
                                   E_i(f_i, s_{i+1}; p_i) and its training objective. Model-free given
                                   the simulated states. Placement by the black_mc precedent — raise if
                                   unobvious.
L4  engine/bermudan_swaption_mc.py — the pricer: train the boundary on one path set, price on a fresh
                                   set, return a PricingResult WITH its standard error (invariant 7).
```

## 3. Rulings

- **The path-return widening lands HERE — this is the second consumer T17 deferred to.** T16/T17's
  engine returns terminal forwards; the Bermudan needs the forwards at **each exercise date** to
  evaluate `E_i(f_i, s_{i+1})`, so the engine widens to return the **path** (states at every exercise
  date). Rule-of-two fires now (the swaption read the endpoint, the Bermudan reads the path), exactly
  as T17's ruling anticipated. The T17 swaption pricer adapts to read the last slice — a small,
  principled refactor, bit-identical if it merely indexes the final state (§4; if not, note the ULP).
- **Return type: the Bermudan engine returns a `PricingResult` with SE** (the §3 invariant-7 ruling),
  wrapping an internal `MonteCarloEstimate`. No third shape (charter §3).
- **Train on one path set, price on a FRESH one.** Optimising the policy and pricing it on the *same*
  paths is in-sample bias — it over-states the value by letting the policy peek at the realisations it
  is priced on. The split is a correctness requirement, asserted by oracle 5.
- **The tree is not built** (§0); its absence is a `to_go_deeper` route, not debt.

## 4. The oracle — a sandwich, because there is no closed form and no tree

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **exact reduction: a ONE-exercise-date Bermudan = the T17 European swaption** (same final maturity), within the combined MC band | **reduction (ours) — the anchor** | an exercise-machinery error that survives into the single-date case; reuses T17's own exact caplet anchor beneath it |
| 2 | **companion (non-vacuity): the early-exercise premium ≥ 0** — adding exercise dates never lowers the value, and for a non-trivial trade *strictly* raises it | **structural** | a Bermudan that collapses to the European (policy never exercises early) — the no-op the reduction alone would not catch |
| 3 | **lower bound: the trained policy priced on fresh paths ≤ the true Bermudan**, and it *rises* toward the true price as training grows (a suboptimal policy under-prices — Jäckel's own point that the price is the optimum over boundary locations) | structural / monotone | a policy that over-prices (in-sample leak, oracle 5) or does not improve with training |
| 4 | **upper bound: the perfect-foresight (look-ahead) price on the SAME paths** — exercise at the ex-post best date, a biased-high estimator and a genuine upper bound. The **sandwich [policy, foresight] brackets the true price; its width IS the oracle's resolution and the policy-quality measure** — stated, per §4's banded-resolution rule, not assumed tight | 2 — bounded | a policy whose sandwich is so wide it rules out nothing; a foresight estimator that is not actually ≥ the policy |
| 5 | **train/price separation** — pricing on the training paths is asserted to over-state vs pricing on fresh paths; the production price uses fresh paths | structural | the in-sample-bias foot-gun |
| 6 | **the European lower bound is itself MC — band it at the resolution it can assert** (Code's T17 carry-in): the `bermudan ≥ european` check uses the combined SE and states the smallest premium it can actually resolve at the committed path count, with a higher-path power test | 1 — banded resolution | a bound that reads as tight while resolving only a gross error (the T17 flat-band trap) |
| 7 | **determinism; failure-is-a-value; −½ rate; PricingResult carries SE** | structural | §3 invariants |

Oracle 1 is the exact anchor (collapse to T17); oracle 4's sandwich is the honest statement of a price
with no closed form — we report the bracket and its width, never a single number dressed as exact.
**Degenerate-fixture guard (seventh watch):** do NOT oracle with a trade where early exercise is never
or always optimal (the premium and the boundary go invisible — the recurring trap in Bermudan dress);
use a trade with a genuine, interior exercise region, and a non-translation-invariant correlation and
non-uniform accruals (the T17 lesson).

## 5. Stock
`parked/ng/.../swaption_mc.py` and the legacy `options/bermudan_*` are structural hints only; the
exercise-boundary coordinates, the objective and the algorithm are read from **Jäckel §12.6–12.8**,
and the sandwich from the standard American-MC construction, not from the stock. Nothing imported.

## 6. Process
Branch `tranche/18-bermudan-swaption`. `test` (RED **pre-registered**) → `feat` → `docs` →
`chore(release): v0.21.0`. **Mutation on the exercise-decision function and the training objective**,
and on the train/price split (a mutant pricing on training paths must fail oracle 5). The engine-widen
refactor keeps T17/T16 oracles green (bit-identical if it only indexes the final state; else note the
ULP, §4). `crosscheck: quantlib (Bermudan swaption)` on the engine. `source:` cites `JAECKEL-2002`
§12.6–12.8. **Open as a DRAFT PR** (§10 step 3).

## 7. What closes, and what defers
This closes the BGM/J arc and Jäckel's headline instrument. Deferred to `to_go_deeper` (Jäckel): the
**non-recombining tree** (ch 13) as the independent cross-check Jäckel benchmarks against, and a
**tight dual upper bound** (Andersen–Broadie / Rogers) to replace the loose perfect-foresight bracket.
With T18 landed, a BGM/J checkpoint is written (the three-unit arc: LMM → European swaption →
Bermudan, each anchored on the one before, the caplet→Black exact anchor underneath all of it).

---

# BUILT — results (appended at build; §1–7 above are Cowork's spec, unedited)

`v0.21.0`. 657 tests, all eight gates green, ruff and pyright clean. `models/exercise_boundary.py`
(L3), `products/bermudan_swaption.py` (L2), `engine/bermudan_swaption_mc.py` (L4), and the path
widening in `engine/mc_path_engine.py`. **Eighteen mutations, all eighteen killed.**

## 8. THE SOURCE IS INTERNALLY INCONSISTENT — eq 12.38 contradicts eq 12.39

Both renderings of eq 12.38 print `+ p_i3` (`-layout` places the `±1` brace after that term; `-raw`
disambiguates that the brace multiplies the whole bracket, and on *that* the two agree). But eq 12.39
pins the final slice to `(0, 0, K)`, and §12.7 states exactly what that must mean: *"at the very last
exercise opportunity we have exact knowledge if exercise is optimal, namely when the residual swap is
in the money."*

At `t_(n−1)` the residual swap is a single period, so its swap rate **is** `f_(n−1)`, and
in-the-money for a payer is `f_(n−1) > K`. The two equations together give:

| | `E_(n−1)` | payer exercises when | correct? |
|---|---|---|---|
| as printed (`+ p_i3`) | `f + K` | `f + K > 0` — **always**, for any positive rate | no |
| ruled (`− p_i3`) | `f − K` | `f > K` | yes |

Measured across seven forwards and both directions: the printed sign misclassifies **every
out-of-the-money payer**. So §4's discrepancy rule applies — two steps of one source disagree, and
the reader rules by the finance rather than deferring. 12.39 wins because it carries *exact financial
content*; 12.38's sign convention is arbitrary.

**And that arbitrariness is the interesting half, because it explains how such an error survives
publication.** At every slice except the last, `p_i3` is a free parameter the optimiser fits, so it
simply converges to `−p_i3` instead: the two conventions describe the *same family* of boundaries and
produce identical decisions. The inconsistency is invisible everywhere the parameters are trained and
decisive only at the one slice where they are pinned. Both facts are oracled — the final slice
reproduces `ω(f − K) > 0` exactly, the printed variant is asserted to **fail** it (so the ruling is
proved necessary, not preferred), and the trained-slice relabelling is asserted to be exact.

Recorded in `docs/references.md` alongside the other source defects this project has found — five in
`BERHAN-2012`, one in `MCINERNEY-ZASTAWNIAK`, and now this. **No ordinal is claimed**: an earlier draft
of this section called it "the second genuine internal inconsistency", which does not survive checking
— `BERHAN-2012`'s 4.1.3 (a denominator contradicting `Di` three lines on) and 4.2.2 (`Vi` defined twice
for two different quantities) are both internal inconsistencies too. A precise tally is useful; a rank
is a claim that quietly rots.

## 9. Jäckel's prescribed optimiser cannot train Jäckel's objective

§12.8: *"one can use the Broyden-Fletcher-Goldfarb-Shanno multi-dimensional variable metric method
for this optimisation."* **BFGS makes no progress whatsoever on eq 12.41.**

Eq 12.41 averages an *indicator-selected* quantity, so it is **piecewise constant** in `p`: the value
changes only when some path's exercise decision flips. Measured at the first training slice, with the
attainable ceiling `mean(max(I_j, v_j))` for reference:

| optimiser | `U_0` reached | iterations | share of the available gain |
|---|---|---|---|
| start `(0, 0, K)` — the European policy | 0.00290435 | — | 0% |
| **BFGS** | **0.00290435** | **0** | **0%** |
| L-BFGS-B | 0.00290435 | 0 | 0% |
| Powell | 0.00363794 | 3 | 57% |
| **Nelder-Mead** (adopted) | **0.00374813** | 17 | **65%** |
| attainable ceiling | 0.00420242 | — | 100% |

The finite-difference gradient at the start is **exactly zero** at `h = 1e-8` and `h = 1e-6`, and
non-zero only at `h = 1e-4` — so what little a gradient method could see is an artefact of the step
size, not a derivative. `train_slice` therefore uses Nelder-Mead, derivative-free, and the deviation
is **oracled**: a test asserts BFGS leaves the objective at its starting value while Nelder-Mead
improves it. That is §4's "assert the wrong variant fails" pattern applied to a numerical method
rather than a formula.

Two further guards the measurement forced: the start is the **European boundary** `(0, 0, K)` — a
real, achievable policy, so training can only improve on it — and `train_slice` never returns
something worse than its start, because Nelder-Mead on a piecewise-constant objective can terminate
on a simplex that has drifted off a plateau, and a lower bound that can degrade is not a lower bound.

## 10. The sandwich, and what it actually resolves

A 6-period coterminal span, exercisable from index 2 (a 6-non-call-2), struck at the forward swap
rate, 4,000 training and 8,000 pricing paths:

| | value | ± SE |
|---|---|---|
| exercisable only at its last date | 1,484.64 | 20.72 |
| **trained policy on fresh paths — LOWER bound** | **5,860.27** | 74.26 |
| **perfect foresight, same paths — UPPER bound** | **6,783.50** | 79.56 |

- **Early-exercise premium 4,375.63**, against a resolution (4× combined SE) of 308.40 — resolved at
  14× the noise, so oracle 2 is a statement about early exercise and not about sampling.
- **Sandwich width 15.75%** of the lower bound. That *is* this tranche's resolution, reported rather
  than assumed tight: the true price lies in the bracket and nothing here narrows it further. A
  tighter dual bound (Andersen–Broadie / Rogers) is the `to_go_deeper` route the spec names.
- **In-sample bias +1.12%** when pricing on the training paths — Jäckel's *"slight upward bias"*,
  measured rather than taken on faith, and the reason oracle 5 exists.

**One honest negative about training-path count.** The lower bound does **not** improve monotonically
with training: 500 → 5,890.09, 1,000 → 5,862.98, 2,000 → 5,849.68, 4,000 → 5,860.27, 8,000 →
5,880.16 — a ~0.7% wobble with no trend. The three-parameter boundary is already near its best for
this trade, so extra training paths move the *fitted* boundary around within a plateau rather than
improving it. Oracle 3 therefore asserts non-decrease **within Monte Carlo error** rather than a
strict rise, and the spec's expectation that the bound "rises toward the true price as training
grows" is not observable at this parametrisation's capacity. The place to spend effort is a richer
boundary, not more training paths.

## 11. Five mutations survived the first pass; one of them was my own oracle repeating T17's lesson

Eighteen mutations, five survivors initially — and the most instructive was **a 1.2% error hiding
inside my own anchor**:

1. **The annuity's numéraire division dropped.** Oracle 1 compared the single-exercise Bermudan to the
   T17 European inside `4 × combined SE`, which at these path counts resolves only **8–11%** of the
   price, so a 1.2% error sat comfortably inside. This is precisely the trap T17 ratified into §4 —
   *a band wide enough to absorb Monte Carlo noise is wide enough to absorb a real defect* — and I
   walked into it one tranche later. The fix was not a wider path count but recognising the comparison
   can be **exact**: after this tranche's widening, a one-date Bermudan's schedule *is*
   `PathGrid(T, steps)`, so both engines consume the same stream and produce the same payoff path by
   path. Oracle 1 now asserts **bit-identical** PV *and* standard error.
2. **The `intrinsic > 0` guard in the policy payoff.** Training makes firing out of the money rare
   (it lowers eq 12.41), so no fixture reached the branch — §4's third survivor kind, dead defensive
   code rather than an equivalent mutant. Fixed §4's way: exercised directly with a hand-built
   boundary that fires everywhere and a negative intrinsic.
3. **The induction direction reversed.** Both directions produce a *valid lower bound*, so no bound in
   the file could see it. A first fix — asserting the earliest slice's continuations are non-zero —
   **also failed to catch it**, because running forwards still rolls the values, just through
   successors whose boundaries are not yet fitted. What distinguishes the directions is the whole
   chain, so the oracle now **reconstructs §12.8's recursion independently** from `roll_continuations`
   and `train_slice` (both pinned pointwise at L3) and requires the engine's policy to match slice for
   slice.
4. **The continuation roll made a no-op** — same blind spot; fixed by extracting Jäckel's step 1 as a
   named function and transcribing it against a hand-built three-path slice.
5. **The second coordinate switched from `s_{i+1}` to `s_i`.** §12.6's entire finding is *which*
   projection separates the exercise domain, and the two rates are highly correlated, so the boundary
   retrains and every bound still holds. Fixed by transcribing the coordinate pointwise against an
   independent swap-rate computation, with a non-vacuity assertion that the two rates differ.

The pattern across 2–5 is one thing: **a price-level bound cannot see the algorithm that produced it.**
Four of the five survivors were pieces of §12.8's *procedure*, and every oracle that caught them in
the end was a pointwise check on a named step — which is why the extraction of `roll_continuations`,
`train_exercise_policy` and `ExerciseSlices` from private helpers into named, oracled surface was
part of the fix rather than cosmetic.

## 12. The engine widening, and its purity proof

`simulate_forward_paths(model, schedule, config)` is now the single definition of eq 13.2, with
`simulate_terminal_forwards` a last-slice read of it — the rule-of-two consolidation T17 deferred to
this tranche's real second consumer. **Proved bit-identically**, which is the right bar because the
refactor reorders nothing: recording the intermediate states does not change how the final one is
computed. Asserted bit-for-bit across every forward and path, in two forms — a one-segment schedule
reproducing the old terminal result, and a *two*-segment schedule reaching the same endpoint, which is
the sharper of the two (it proves segmenting does not disturb the random stream). All T16 and T17
oracles stay green unchanged.

## 13. Findings for Cowork (§10 — raised, not decided)

1. **The eq 12.38 / 12.39 sign ruling (§8).** I ruled it by the finance per §4 rather than blocking,
   since §4 prescribes exactly that procedure and the ruling is pinned by an oracle that the printed
   variant fails. Confirm the ruling, and confirm it belongs in `docs/references.md` as a recorded
   source defect alongside `BERHAN-2012` §5.1's.
2. **The BFGS deviation (§9).** `train_slice` uses Nelder-Mead against the source's explicit BFGS,
   because the objective is piecewise constant and BFGS measurably achieves nothing. Rule whether the
   deviation stands as built and whether the measurement belongs in the references entry too — this
   is a limitation of the published method, not of our reading of it.
3. **Candidate §4 rule, from §11:** *a bound on a price cannot see the algorithm that produced it.*
   Where a method is a multi-step procedure (a backward induction, a training loop), oracle the
   **steps pointwise** — extracting them into named surface if they are buried in a private helper —
   because any valid-but-wrong variant of the procedure still satisfies the output bounds. Four of
   five first-pass survivors were procedure steps, and no price-level oracle caught any of them.
4. **The spec's oracle 3 is not observable at this parametrisation (§10).** The lower bound does not
   rise with training paths; it wobbles ~0.7% with no trend, because the three-parameter boundary is
   already at its capacity for this trade. I built it as non-decrease within MC error. Rule whether
   that stands, and whether a richer boundary is a unit worth pulling.
5. **The sandwich is 15.75% wide**, which is the honest resolution of this tranche and is wide. The
   spec names Andersen–Broadie / Rogers as the `to_go_deeper` tightening; worth ruling whether that
   becomes a real unit now that the arc's headline is landed, or stays parked.

---

## 14. Cowork's ruling on T18 — PR #25 (§10 step 4, 2026-11-05)

Reviewed at the reasoning. The exact one-date reduction (now bit-identical in PV and SE after the
widening — the schedule *is* `PathGrid(T, steps)`) is the anchor, the sandwich is the honest statement
of a price with no closed form, and eighteen mutations all die. Accepted — v0.21.0, the BGM/J arc
complete. Two of my spec premises were corrected by the build; both corrections are right.

**Finding 1 — the 12.38/12.39 sign: ratified.** 12.39's `(0,0,K)` pin carries the exact financial
content (exercise iff the one-period residual swap is ITM); 12.38's printed sign exercises a deep-OTM
payer and is the arbitrary convention. Ruling by the finance and pinning it with an oracle the printed
variant fails is exactly §4's discrepancy procedure (the T13 precedent). Recorded in
`docs/references.md` as a Jäckel source defect — seventh overall, second internal inconsistency.

**Finding 2 — BFGS → Nelder-Mead: stands.** A gradient optimiser cannot train a piecewise-constant
objective; that is arithmetic, not taste. Nelder-Mead is the correct substitution, the deviation is
oracled, and it is recorded in the references entry as a limitation of the *published method*.

**Finding 3 — charter-bound (§4).** *A bound on a price cannot see the algorithm that produced it* —
oracle a multi-step procedure's steps pointwise, extracting a buried step into named surface so a test
can reach it. Four of five first-pass survivors were §12.8 procedure steps no price bound caught; the
fifth hid in oracle 1's SE band — the T17 banded-resolution trap one tranche on, and your fix (make
the one-date reduction bit-identical, not widen the band) is the right instinct. This is the domain
rule extended from formulas to procedures, and it is the capstone lesson of the arc.

**Finding 4 — oracle 3 corrected: non-decrease within MC error, as you built it.** My spec said the
lower bound *rises* with training; it wobbles ~0.7% because the three-parameter boundary is at
capacity for this trade. My premise was over-specified (the MC arc's recurring shape — I assert an
empirical behaviour the build then corrects; owned). A richer boundary is a `to_go_deeper` route, not
pulled now — nothing needs it, and the sandwich is honestly reported.

**Finding 5 — the 15.75% sandwich stays as the arc's honest resolution; Andersen–Broadie stays
parked.** The lower bound is the quotable price (a valid sub-optimal-policy value); the upper bound is
a quality diagnostic. No current interest pulls a tight dual bound, so it remains a `to_go_deeper`
route (§1 product-pull), to become a unit if proving near-optimality is ever needed.

**Next: the BGM/J checkpoint** (spec §7) — the three-unit arc closed, LMM → European → Bermudan, each
anchored on the one before, the caplet→Black exact anchor underneath all of it.
