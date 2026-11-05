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
