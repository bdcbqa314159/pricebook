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


---

# BUILT — results (appended at build; §0–5 above are Cowork's spec, unedited)

`v0.26.0`. 787 tests, all eight gates green, ruff and pyright clean. `foundation/brownian_bridge.py`
(L0) and the engine's `Sampling` bundle. **Thirteen mutations, all thirteen killed.** Chapter 10 is
complete.

## 6. The anchor is exact, with no Monte Carlo in it

The spec asked for the covariance identity *"to tolerance"*. It does not need one. The bridge is a
**linear** map of its input variates, so its matrix `A` is recovered exactly by feeding unit basis
vectors, and "same Gaussian, reordered" becomes an algebraic identity:

    A Aᵀ = M     where   M_ij = min(t_i, t_j)   — the Wiener covariance

checked to machine precision at 2, 3, 4, 5, 8, **14** and 16 steps. A realised covariance over sampled
paths would have had to be banded by the estimator's own error; there is no estimator here, so there
was no tolerance to choose. The same holds one level down: the increment map's Gram matrix is exactly
the identity, which is what "the increments stay i.i.d." means.

14 steps is in the sweep because it is the source's own worked example (figure 10.4) and is **not** a
power of two — the case §10.8.3 singles out as needing the general *"halve into two parts containing
approximately the same number of points"* rule.

## 7. A source misprint, faithful in both renderings

Appendix §10.9.2's eq **10.76** prints `W_tk = W_tj + √(t_k − t_i) z`. It contradicts eq 10.77 three
lines later — `z := (W_tk − W_ti)/√(t_k − t_i)` forces `W_ti` — and 10.76 is introduced as a rewrite
of 10.75, whose expansion begins at `W_ti`. With `W_tj` the first increment is counted twice. **Both
`-raw` and `-layout` print it identically**, so transcription independence is powerless (§4) and only
the internal inconsistency catches it.

It does not reach the build: the formulas implemented are 10.33/10.34, which are consistent and which
I re-derived independently (`E[W_tj] = W_ti + (t_j−t_i)/(t_k−t_i)·(W_tk − W_ti)` rearranges to 10.33)
rather than reading from the typo'd sketch. Worth recording because §4 treats a derivation as an
independent check on its result — and here the check itself has a defect, so it had to be repaired
before it could be used.

## 8. The claim holds, but only one half of it is assertable

**Measured through the engine, LMM caplet, 32 Sobol dimensions (8 steps × 4 forwards):**

| N | pseudo-random | Sobol + incremental | Sobol + bridge |
|---|---|---|---|
| 2,048 | 1.00e-04 | 2.22e-05 | 1.07e-05 |
| 4,096 | 8.10e-05 | 1.83e-05 | 5.54e-06 |
| 8,192 | 6.33e-05 | 6.40e-06 | 1.26e-06 |
| 16,384 | 1.68e-05 | 2.49e-06 | **8.34e-07** |
| **fitted exponent** | **−0.255** | **−0.616** | **−0.940** |

So the bridge nearly reaches the theoretical −1 and the ordering the spec predicts is real: at 16,384
paths it is **20× better than pseudo-random** and **3× better than Sobol alone**.

**But the spec asked for the comparison against pseudo-random as a fitted exponent, and that is not
assertable.** A single seed gives one noisy realisation, and its fitted slope swings from **−0.637** to
**−1.056** depending only on where the range ends — on the shorter range it *beats the bridge*, by
luck. So the oracle makes the bridge-vs-incremental comparison on the **exponent** (both are Sobol,
hence deterministic, so the only wobble is the sequence's own non-monotonicity) and the comparison
against pseudo-random on the **RMS error over the range**, which aggregates instead of fitting:
bridge/incremental **0.35**, bridge/pseudo **0.17**, identical to two decimals across both ranges.
Third tranche running in which a headline claim had to be restated in the form that is measurable
above its own noise.

## 9. The null result needed averaging too, and that is the subtler half

Under pseudo-random draws the bridge must change **nothing** — it is a reordering of the same
Gaussian, so it produces a different *sample*, not a better one. A first version of that oracle
compared one seed and measured the bridge **6× better**. That looked like a result and was sampling
luck: two draws from one distribution differ by that much routinely.

Averaged over twelve seeds at two path counts the RMS ratio is **1.31** — and 0.92 / 1.26 / 1.29 under
three other seed-and-count choices. Neutral, as the algebra says it must be. **A null measured on one
sample is not a measurement**, which is the same rule as §8 pointing the other way: there, noise
manufactured an effect that was not there; here it manufactured one in a place where the expected
answer was "none".

## 10. The ordering is the purpose, and no output property could see it

Twelve of thirteen mutations died at once. The survivor was **depth-first instead of breadth-first
subdivision** — and it is a real hole, not an equivalent mutant. Depth-first still places every point
exactly once, still brackets each by already-placed points, and still yields a valid Wiener path, so
**the exact covariance identity cannot see it**. What it breaks is the entire purpose: descending one
branch to the bottom means the second variate shapes a small early region instead of the global
midpoint, so Sobol's best dimensions would carry fine detail in one corner.

Tranche 18's lesson in a new place — *a property of the output cannot see the algorithm that produced
it*. The distinguishing property is the bracket **width**: coarsest-first means widths are
non-increasing through the plan. A first attempt demanded that strictly and failed at 14 steps, where
unequal halving legitimately puts a width-3 bracket before a width-4 one; **non-increasing up to a
slack of one** is the source's own *"approximately the same number of points"*, and it still rejects
depth-first, which returns to a width-7 gap after width-2 ones.

## 11. `Sampling` — the bundle §7's limit forced, and it was the right answer

Three sampling choices now have real consumers: Sobol (T20b), the bridge (T21), antithetic mirroring
(T19). Four engine arguments plus three knobs breaches §7's five-argument limit, whose prescribed
response is to *"group into a frozen value object"* — so `Sampling(source, construction, mirrored)`,
defaulting to exactly tranche 16's behaviour, which is why all 784 existing tests passed unchanged.
Nothing in it is speculative; each field is the switch some landed tranche needed.

Jäckel's own *"all of the weighting coefficients can be precalculated"* became
`BrownianBridge.increment_matrix()`, so the engine reorders every path and factor in one `tensordot`
rather than a Python loop per path — the cost pattern the `slow` gate's docstring names and that I
walked into in T19. The oracle points at the **shipped** matrix and additionally checks it reproduces
the step-by-step construction, so the precalculation is a factoring-out rather than a second
implementation.

## 12. Findings for Cowork (§10 — raised, not decided)

1. **The Bermudan's own path still cannot use Sobol.** Its tenor has six forwards; eight steps makes
   **48** dimensions, past the 32 Table 8.3 covers. The bridge is therefore demonstrated on a
   four-forward caplet path (32 dimensions exactly — the table's full reach), stated in the fixture
   rather than quietly chosen. So T20b's `to_go_deeper` acquisition is now load-bearing for a real
   consumer, not hypothetical: extending the direction numbers is what would let the headline
   instrument use QMC.
2. **Candidate §4 note, from §9:** *a null result needs averaging as much as an effect does.* The
   existing rule says an oracle must be measurable above its own noise; the case not covered is the
   oracle whose expected answer is "no difference", where a single sample can manufacture a 6×
   difference and look like a discovery. Worth a clause only if you think §4 can carry one more.
3. **Eq 10.76's misprint** (§7) — recorded here; rule whether it belongs in `references.md`'s Jäckel
   defect list alongside the 12.38/12.39 sign inconsistency. It is a *derivation* defect that does not
   reach any implemented formula, so it is a weaker entry than that one.
