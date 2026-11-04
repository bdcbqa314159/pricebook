# Tranche 16 — the LIBOR market model evolved by Monte Carlo, anchored on our Black caplet

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.19.0`.
**Source:** `JAECKEL-2002` ch 12 (`worked-from` on landing) for the MC method — the lognormal LMM SDE
(12.1), the covariance factorisation (12.2–12.7), the drift (eq 12.8, §12.3), and the evolution scheme
(12.33–12.34, predictor-corrector §12.5); `MCINERNEY-ZASTAWNIAK` ch 5–6 (`in-hand`) for the LMM model
and swaption background. Both held — **no acquisition**. **Chosen by Bernardo:** BGM/J (ch 12).

## 0. This is the FIRST of a three-unit BGM/J arc — and the only one with a closed-form anchor

Bernardo picked BGM/J, whose headline is the Bermudan swaption. That is too big for one tranche and,
more importantly, the Bermudan has **no closed form** — so it cannot be the first unit under our
discipline. The arc decomposes so that each unit is anchored by what precedes it:

- **T16 (this) — the LMM forward-rate evolution + the engine.** Anchored on a **closed form**: each
  forward's caplet reprices to our `black_price` *by construction* (eq 12.6–12.7 build the
  factorisation to retain caplet calibration), and each forward is a **martingale** under its own
  forward measure. This is the foundation, and it is the **second consumer of the MC machinery**, so
  the L4 engine is extracted here (rule-of-two: `black_mc` + the LMM share the simulate-a-path shape).
- **T17 — the European swaption under the LMM.** Anchored by convergence + the swaption *approximation*
  (§12.3–12.4) as a bounded cross-check, stated as a bound, not dressed as exact (the MM-I freezing
  lesson).
- **T18 — the Bermudan swaption**, the headline. No closed form; oracled by a **sandwich** — the
  European price as a lower bound, and the non-recombining tree (Jäckel ch 13) as the cross-check
  Jäckel himself uses — never a single number.

So the exciting Bermudan arrives only once the LMM is MC-verified against Black. T16 is specced here;
T17/T18 follow as their own units.

## 1. The method (from Jäckel) vs the anchor (our Black)

- **Method, transcribed (§4):** the lognormal SDE `df_i/f_i = µ_i(f,t)dt + Σ_j a_ij dW_j` (12.1/12.5);
  the covariance `c_ij = σ_i σ_j ρ_ij` (12.3) and its pseudo-square-root `C = ÃÃᵀ` (12.4), with the
  m-factor reduction `Σ_j a_ij² = c_ii` that **preserves caplet calibration** (12.6–12.7); the
  **state-dependent drift µ_i (eq 12.8, §12.3)** — the load-bearing formula, read `-raw`+`-layout` at
  build, NOT transcribed into this spec (the inverse-CDF lesson: I name where it is, Code reads it);
  and the constant-drift log-Euler / predictor-corrector step (12.34, §12.5).
- **Anchor, ours:** a caplet on `f_i` must reprice to `black_price(f_i(0), K, c_ii·T_i, kind)` — the
  T9 closed form — within standard error. Eq 12.6 is exactly the guarantee that the simulated variance
  is `c_ii·T_i`, so this is a closed-form anchor, not a self-check.

## 2. Pieces and layers

```
L0  foundation/linalg.py (or extend) — the pseudo-square-root of a covariance matrix (12.4) and the
                               m-factor rescaling (12.7). PURE linear algebra, finance-free → L0
                               (beside quadrature.py). The finance is which matrix; the decomposition is not.
L3  models/libor_market_model.py — the LMM: the forward tenor structure, the per-forward vols σ_i,
                               the correlation ρ_ij, and the drift µ_i (eq 12.8). Carries its
                               MarketSnapshot. A nominal-rate market model — NOT MM-I/MM-II (those are
                               inflation index models); concrete, no shared base (§6).
L4  engine/mc_path_engine.py — EXTRACTED here as the second MC consumer: evolve a process over a time
                               grid by the predictor-corrector step, estimate a payoff mean + SE.
                               Reuses the L0 rng/variates/estimator. `black_mc` is NOT retrofitted onto
                               it in this tranche unless the refactor is bit-identical (§4); if not,
                               leave black_mc and note it.
```

The factorisation is finance-free linear algebra → L0. The LMM (forwards, vols, correlations, drift)
is a model → L3. The path engine (evolve + estimate) is the stateless simulating pricer → L4, and it
earns its place now because there are **two** consumers (black_mc's GBM and the LMM) sharing the
evolve-and-average shape — rule-of-two, exactly as Black and the HW core were extracted at their
second consumer.

## 3. Rulings

- **The engine is extracted now, not retrofitted speculatively.** Two real consumers exist; the shape
  they share (evolve a process on a grid, average a payoff, report SE) is the abstraction. If folding
  `black_mc` onto the engine is bit-identical, do it (§4 pure-refactor); if it reorders floats, leave
  `black_mc` as-is and note the ULP gap — do not contort for it.
- **Measure is explicit.** The drift µ_i depends on the chosen measure (terminal vs spot LIBOR);
  the model states which, and the martingale oracle is asserted under the measure where the relevant
  forward is driftless. No ambient choice.
- **Seed is explicit config (§3.5), PCG64 + Box-Muller from T15** — the determinism and §9 checks carry
  over unchanged. The path now has multiple steps, so path construction enters; for T16 the simple
  incremental construction suffices (the Brownian bridge is a later variance-reduction unit, not here).

## 4. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **a caplet on `f_i` reprices to our Black:** `|MC − black_price(f_i(0),K,c_ii·T_i,kind)| < k·SE` | **closed form (ours) — the anchor** | a wrong drift, a wrong factorisation (variance ≠ c_ii), a discretisation bias |
| 2 | **the martingale property — a requirement oracle, measure-explicit:** under `f_i`'s own forward measure `E[f_i(T_i)] = f_i(0)` within SE | **structural** | a wrong drift sign/term (eq 12.8); this is what the drift transcription must satisfy for ALL params |
| 3 | **the −½ convergence rate:** `log SE` vs `log N`, slope ≈ −½ (the T15 oracle — a biased scheme passes 1 at one N by luck, not the rate) | structural | a discretisation bias from too few/large steps |
| 4 | **factorisation reconstructs the covariance:** `ÃÃᵀ = C` to tolerance, and the m-factor rescaling preserves the diagonal `Σ_j a_ij² = c_ii` exactly (eq 12.6 — the caplet-calibration guarantee, pointwise) | 1 — pointwise + requirement | a wrong pseudo-square-root, or a rescaling that breaks caplet calibration |
| 5 | **drift reduces exactly at the degenerate point:** the terminal forward has zero drift under the terminal measure by construction — an exact reduction, with a companion that a non-terminal forward's drift is non-zero and moves the price (non-vacuity) | exact + companion | a drift that is a no-op (passes 2 trivially) — the reduction-needs-a-companion rule |
| 6 | **determinism / failure-is-a-value** — same seed ⇒ byte-identical paths (§9); a non-positive forward or `n_paths ≤ 0` → `PricingFailure`, never NaN | structural | §3 invariants |

Oracle 1 is the anchor; oracle 2 is the one that pins the drift transcription (the load-bearing
formula) against the finance for all parameters, so it survives a transcription error oracle 1 at a
single strike might not. Mine the stock's LMM/swaption MC (`parked/ng/.../swaption_mc.py`,
`models/mc_processes.py`) for **shape and edge cases**, never structure (§5, `mc_base_inventory.md`).

## 5. Stock
`parked/ng/pricebook_ng/engine/swaption_mc.py` is the closest structural hint (an LMM-ish MC in a
clean engine layout) — read for shape. The drift, factorisation and caplet anchor are read from
**Jäckel ch 12** and **McInerney ch 5–6**, not from the stock's code. Nothing imported from `_stock/`.

## 6. Process
Branch `tranche/16-lmm-mc`. `test` (RED **pre-registered**) → `feat` → `docs` →
`chore(release): v0.19.0`. **Mutation on the drift (eq 12.8) and on the factorisation rescaling** —
both load-bearing; the martingale oracle (2) and the caplet anchor (1) must each fail under a drift
sign-flip, and oracle 4 under a rescaling that breaks `Σa² = c_ii`. Degenerate-fixture guard: do NOT
oracle with a single forward or a flat/zero correlation (the horizon and the off-diagonals go
invisible — the recurring trap); use a multi-forward tenor with non-trivial ρ_ij. `crosscheck: quantlib
(LMM / caplet)` on the model and the engine; the L0 factorisation carries `crosscheck: none —
finance-free linear algebra`. `source:` cites `JAECKEL-2002` ch 12 and `MCINERNEY-ZASTAWNIAK` ch 5–6;
names `black_price` as the anchor. **Open as a DRAFT PR** (§10 step 3).

## 7. What this unlocks (not built here)
T17 (European swaption under the LMM, anchored by convergence + the swaption approximation) and T18
(the Bermudan — the exercise boundary, §12.6–12.8, oracled by the European-lower-bound / tree
sandwich). The path engine built here is what both consume. Variance reduction (control variates,
the Brownian bridge) and Sobol remain their own later units, pulled when a product needs the speed.

---

# BUILT — results (appended at build; §1–7 above are Cowork's spec, unedited)

`v0.19.0`. 554 tests, all eight gates green. `foundation/linalg.py` (L0), `models/libor_market_model.py`
(L3), `engine/mc_path_engine.py` (L4), as specced. Oracles 1–6 all green; the anchor reprices a caplet
to `black_price` at three strikes inside 4 SE, and the step count does not move the martingale forward.

## 8. The drift's MAGNITUDE was unconstrained — found by mutation, not by a failing test

**Thirteen oracles passed, and three mutations of eq 12.8 survived all of them.** Dropping the drift's
`dt` factor (an 8× drift), feeding the drift the raw forwards instead of eq 12.8's
`(f τ)/(1 + f τ)·σ` ratios (5×), and dropping that denominator altogether (1.04×) each passed the full
550-test suite.

The cause was **structural, not a shortage of paths**, and it is the more interesting kind:

- every drift oracle asserted the drift was **present** (the paired-numeraire test, at 297–372 SE) or
  correctly **signed** (either side of the numeraire) — neither constrains its size;
- the closed-form anchor runs under the measure where the target forward's drift is **exactly zero**
  (eq 12.8's middle branch). That exactness is what makes the anchor sharp, and it is also why the
  anchor **cannot see the drift at all**.

So the suite's two strongest drift oracles were blind to magnitude by construction. This is the
charter's own named defect class — a check that cannot fail — wearing a new disguise: not a vacuous
assertion, but a *complete* set of assertions about the wrong quantity.

**The fix removes Monte Carlo error rather than fighting it.** Two models differing only in numeraire
index, one step, one seed: both consume the same normals from the same state, so the diffusion term
and the `−c_ii/2` correction cancel to the last bit, and

    log f_i^A(dt) − log f_i^B(dt)  =  (µ_i^A − µ_i^B) · dt          exactly, every path

That is an identity, not an estimate. Measured agreement with a fresh transcription of eq 12.8 is
**1e-13 relative** with a cross-path spread of 7e-16 — precision no path count could buy, and the
reason it can see M7's 3.8%. The grid is a **half**-year single step deliberately: at `dt = 1.0` the
`dt` scaling is invisible, which is exactly how M3 survived. All eight mutations now die.

## 9. The new oracle caught an error in its own reference first

The transcription of eq 12.8 written for §8's oracle used the off-diagonal `ρ = 0.80` for every pair,
including `ρ_ii`, which is 1. On first run forward 0 agreed to 1e-13 and forwards 1–3 disagreed in the
third digit — and **the asymmetry is what located the bug**: forward 0's drift sums never reach
`k = 0`, so it is the only index whose reference never touches the diagonal. The model was right; the
new reference was wrong.

Worth recording because it is the dual of the usual failure. The standing worry about an independent
transcription is that it repeats the implementation's error (the shared-rendering rule). Here it
introduced a *fresh* one — and a disagreement pattern that pointed at the reference rather than the
code is what distinguished the two. A uniform disagreement across all four forwards would have
implicated the model.

## 10. Findings for Cowork (§10 — raised, not decided)

1. **`black_mc` is NOT folded onto the new engine, and it cannot be bit-identically.** §3's ruling says
   do it if the refactor is bit-identical, else leave it and note the gap. It is not: `black_mc` takes a
   **single** step to expiry, while the engine steps a grid, so the two consume different numbers of
   draws in a different order. Even at `steps=1` the engine's drift path and `−c/2` handling reach the
   exponent by a different summation order. So `black_mc` stays as written, per the spec's own
   instruction. The shared machinery (`rng`, `variates`, `mc_estimator`) *is* common; only the stepping
   loop is duplicated, and it is six lines.
2. **Candidate §4 rule, from §8:** *a set of oracles can be complete about presence and sign and still
   be silent about magnitude.* The generalisation: when a quantity enters a formula as a **product**,
   assert the product's **value** against an independent transcription, not merely that each factor is
   there and correctly signed. The sub-lesson for anchors: an oracle made sharp by a quantity being
   exactly zero is, for that same reason, blind to that quantity.
3. **Candidate §4 rule, from §9:** when an independent transcription disagrees with the code, the
   **pattern across indices/parameters says which side is wrong** — a disagreement confined to the
   cases that exercise one special term implicates the term, not the implementation. Worth a sentence
   because the instinct on a red oracle is to debug the code first.
