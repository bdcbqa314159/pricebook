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
