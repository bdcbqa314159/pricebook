# Tranche 21 — path construction / the Brownian bridge, and Sobol wired into the LMM path

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.26.0`.
**Source:** `JAECKEL-2002` §10.8 (path construction: incremental §10.8.1, spectral §10.8.2, the
**Brownian bridge §10.8.3**, the comparison §10.8.4, multivariate §10.8.5) and appendix §10.9.2 (the
bridge's conditional distribution — the load-bearing formula). Third of four in the Jäckel arc;
**finishes ch 10** and makes T20b's Sobol pay on the LMM path. Queue after T20c, before T22 (Greeks).

## 0. The point — recover Sobol's advantage on a high-dimensional path
T20b measured Sobol's edge washing out by d≈24 and gone at d=32; the LMM path is ~48 dimensions, which
is why 15.1 did **not** wire Sobol into it. The Brownian bridge is the fix: it is a *reordering* of
which low-discrepancy coordinates control which time points, so the first few (best-equidistributed)
Sobol dimensions set the coarse shape of the path and the later (worse) dimensions only add fine
detail. The variance that matters loads onto the dimensions Sobol is good at. This is the standard
QMC+bridge pairing, and it is the reason the bridge is built now rather than for its own sake.

## 1. The method (from Jäckel) vs the anchor (ours)
- **Transcribed (§4):** the bridge construction §10.8.3 and its conditional mean/variance §10.9.2
  (`mean = linear interpolation between the two bracketing fixed points`, `var = (t−t_a)(t_b−t)/(t_b−t_a)`)
  — the load-bearing formula, read `-raw`+`-layout`, pointwise.
- **Anchor, ours:** the bridge is a **linear reordering of the same Gaussian increments**, so the path
  it builds has the **identical covariance** to the incremental construction (T15/T16). That identity
  is the exact anchor — a wrong bridge weight changes the covariance and breaks it.

## 2. Pieces and layers
```
L0  foundation/brownian_bridge.py — the bridge: given N standard normals and a time grid, place them
                                    by the bridge order (midpoint-first) with the §10.9.2 weights.
                                    Finance-free path construction. Composes with Sobol (T20b) and the
                                    inversion (T20a): Sobol → uniforms → Φ⁻¹ → normals → bridge → path.
L4  the LMM path engine (T16) gains a path-construction choice — incremental (current) or bridge —
                                    so a Sobol-driven LMM path uses the bridge. The engine's shape is
                                    unchanged; the construction is a strategy it is handed.
```

## 3. The oracle (RULED unless TO MEASURE)
- **RULED — exact: the bridge reproduces the incremental path's covariance.** Build a path both ways
  from the same normals; the realised covariance matrices agree to tolerance (they are the same
  Gaussian, reordered). An exact requirement anchor; a wrong weight in §10.9.2 breaks it. Companion
  (non-vacuity): the per-step *order* in which dimensions are consumed differs (the bridge is not the
  identity reordering) — assert it is genuinely midpoint-first, not a no-op.
- **RULED — §10.9.2 pointwise:** the conditional variance `(t−t_a)(t_b−t)/(t_b−t_a)` transcribed and
  checked at interior points (max at the midpoint, → 0 at the brackets).
- **RULED — Sobol+bridge beats Sobol+incremental on the LMM path, and both beat pseudo-random**
  (measured fitted exponent, T20b's honest shape — a slope, not an asserted rate). This is the whole
  reason for the tranche; a bridge that did not improve the high-dimensional convergence would be
  pointless (non-vacuity on the *point*, not just correctness).
- **TO MEASURE — how much the bridge recovers:** report the convergence exponent for Sobol+bridge on
  the LMM caplet/swaption vs Sobol+incremental vs pseudo-random, and the effective dimension at which
  it still holds. (The honest reading, per T20b, is a statement about feasible N, not about the bridge
  in the abstract.)
- **RULED — determinism; failure-is-a-value.** Same normals + same grid ⇒ same path; a grid the bridge
  cannot place (fewer than 2 points) → a value, not a wrong path.
- **Degenerate-fixture guard:** do not oracle the covariance identity on a 2-point grid (the bridge is
  trivial there); use a grid with interior points so the midpoint-first weights actually act.

## 4. Stock & process
Mine `_stock/.../models/brownian.py` and `numerical/_mc.py` for the bridge *shape* only; the weights
are read from **Jäckel §10.8.3 / §10.9.2**, not the stock. Branch `tranche/21-brownian-bridge`.
`test` (RED pre-registered: the covariance identity) → `feat` → `docs` → `chore(release): v0.26.0`.
Mutation on the §10.9.2 weight and on the bridge ordering (a mutant that reorders to incremental must
fail the "midpoint-first" companion; a wrong weight must fail the covariance identity).
`crosscheck: quantlib (BrownianBridge)` on `foundation/brownian_bridge.py`. `source:` cites
`JAECKEL-2002` §10.8/§10.9. **Open as a DRAFT PR** (§10 step 3).

## 5. What this closes
Ch 10 (variance reduction) is complete: antithetic + control variates (T19) + path construction (T21).
Only ch 11 (Greeks, T22) remains to close the nine-chapter spine and trigger the first whole-tree
re-audit (§9) at the Jäckel arc checkpoint.
