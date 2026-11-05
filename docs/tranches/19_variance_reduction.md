# Tranche 19 — Variance reduction I: antithetic sampling and control variates

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.22.0`.
**Source:** `JAECKEL-2002` §10.1 (antithetic sampling) and §10.3 (control variates). **Chosen by
Bernardo** as the first unit of the Jäckel completion arc (`docs/jaeckel_completion_arc.md`). First of
four: T19 → T20 (Sobol+inversion) → T21 (Brownian bridge) → T22 (Greeks).

## 0. The point — make the MC we built SHARP, and oracle that it stays HONEST

T18 measured the cost of raw MC: the Bermudan sandwich is 15.75% wide, mostly sampling error. Variance
reduction shrinks that error at the same path count. The one thing it must never do is **introduce
bias** — a "reduction" that moves the price is worse than none. So every oracle here is a pair: the
reduced estimator is **unbiased** (same expectation as plain MC, anchored on our closed forms), AND it
**reduces variance** (measured). Neither half alone is the technique.

Two foundational unbiased reducers, separable:
- **(a) antithetic (§10.1):** for each path driven by normals `z`, run its mirror `−z`; average. A
  sampling-side technique (L0).
- **(b) control variates (§10.3):** price the target `X` by MC but subtract a correlated control `Y`
  whose mean `E[Y]` we know in closed form: `X̄ − β(Ȳ − E[Y])`. We already own the controls — Black
  for the caplet, the Rebonato swaption for the European/Bermudan. This is the one that tightens the
  sandwich.

## 1. The method (from Jäckel) — load-bearing formulas transcribed (§4)
- **Antithetic:** the estimator `½(f(z) + f(−z))` averaged over paths; its variance identity (§10.1).
- **Control variate (§10.3):** `θ̂ = X̄ − β(Ȳ − E[Y])`, with the **optimal** `β* = Cov(X,Y)/Var(Y)`
  and the resulting variance `Var(X̄)(1 − ρ²_{XY})`. Transcribe `β*` and the variance-reduction factor
  `-raw`+`-layout`; they are the load-bearing formulas and oracle 4 pins them pointwise.

## 2. Pieces and layers
```
L0  foundation/mc_estimator.py — EXTEND: an antithetic sampling mode (pair z with −z), and a
                                 control-variate estimator taking X samples, Y samples and the known
                                 E[Y], returning the adjusted mean + its reduced SE and β*. Pure
                                 statistics — finance-free (it does not know what X or Y are).
L4  the MC engines opt in — a pricer supplies its control's analytic mean (Black for the caplet,
                                 the Rebonato swaption for the swaption/Bermudan) and routes its paths
                                 through the reducer. The CHOICE of control is finance → L4; the
                                 reduction math is the L0 primitive above.
```
The split mirrors §3's `MonteCarloEstimate`/`PricingResult` ruling: the variance-reduction *math* is
an L0 primitive; the *choice of control* (a finance object) is the engine's.

## 3. Rulings
- **Unbiasedness is the correctness anchor, variance reduction the companion** — in that order. A
  reduced estimator that fails to reproduce the closed form (oracle 1) is rejected however much
  variance it saves.
- **`β` is estimated from the same paths** (the sample `Cov/Var`), and that introduces a tiny `O(1/N)`
  bias in principle — assert it is below the SE at the committed path count (measured), or use `β=1`
  for the highly-correlated controls and report the reduction (the simpler, provably-unbiased choice;
  rule `β=1` vs `β*` by what the build measures). Raise as an `Ask:` if `β*`'s bias is not negligible.
- **The SE a reduced pricer reports (invariant 7) is the REDUCED SE** — the whole point; a banded
  oracle reading it then resolves a smaller defect (the §4 banded-resolution rule, now improving).

## 4. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **unbiased — the anchor:** the antithetic and control-variate estimators converge to the SAME closed form as plain MC (Black for the caplet; the exact one-period reduction for the swaption) within SE | **closed form (ours)** | a reduction that moves the price — the one failure that matters |
| 2 | **variance actually reduced (measured, non-vacuous):** reduced SE < plain SE at the same paths, and the factor reported (≈ 1−ρ² for the control; the antithetic reduction for a monotone payoff) | structural | a "reduction" that doesn't reduce — pointless, and a sign the control is uncorrelated or mis-wired |
| 3 | **antithetic is EXACT on a linear payoff:** for a payoff linear in `z`, `½(f(z)+f(−z))` has zero variance — an exact reduction; companion: a convex payoff (a call) reduces but not to zero (non-vacuous) | exact + companion | an antithetic that mis-pairs the draws (the mirror is not `−z`) |
| 4 | **the control-variate algebra, pointwise:** `β* = Cov/Var` and the variance `Var(1−ρ²)` transcribed and checked on synthetic X,Y with a known correlation | 1 — pointwise + requirement | a wrong `β*` or variance formula (the magnitude/transcription rule) |
| 5 | **the payoff: the Bermudan/swaption sandwich tightens** with the Rebonato control — the concrete win, the new sandwich width reported against T18's 15.75% | 2 — measured | the control (Rebonato) uncorrelated with the target, or wrongly centred on `E[Y]` |
| 6 | **determinism; −½ rate preserved** (reduction changes the constant, not the exponent); failure-is-a-value | structural | §3 invariants; a reduction that secretly changes the convergence order |

Oracle 1 is the anchor (unbiasedness against our closed forms); oracle 3's exact linear-payoff
cancellation is the sharpest antithetic check. **Degenerate-fixture guard:** do not oracle the control
variate with a control perfectly correlated with the target (ρ=1 zeroes the variance trivially and
hides a centring error) nor uncorrelated (ρ=0 reduces nothing); use a realistically correlated control
(the Rebonato swaption vs the Bermudan, ρ high but < 1).

## 5. Stock
`_stock/.../models/mc_variance_reduction.py` (207 lines) is the shape hint — antithetic and control
variates were built once; mine for the interface shape and its migration edge cases
(`mc_base_inventory.md`), never the structure. The `β*` and variance formulas are read from **Jäckel
§10.1/§10.3**, not from the stock. Nothing imported.

## 6. Process
Branch `tranche/19-variance-reduction` (or `19a`/`19b` if the two techniques land separately —
antithetic first, control variates second; keep them separable). `test` (RED **pre-registered**) →
`feat` → `docs` → `chore(release): v0.22.0`. **Mutation on `β*` (Cov/Var) and the antithetic pairing**
(a mutant mirroring to `+z` or dropping the `½` must fail oracle 3); and on the control's centring
(`E[Y]`), which oracle 5 catches. `crosscheck: quantlib (variance-reduced MC)` on the engines; the L0
estimator carries `crosscheck: none — finance-free statistics`. `source:` cites `JAECKEL-2002`
§10.1/§10.3; names Black and the Rebonato swaption as the controls. **Open as a DRAFT PR** (§10 step 3).

## 7. What this unlocks
T20 (inversion + Sobol — a different error-reduction axis, composable with these), T21 (the Brownian
bridge), T22 (Greeks). Control variates recur in T22 (a pathwise-Greek control) and T20 (Sobol +
control variate compound).

---

# BUILT — results (appended at build; §1–7 above are Cowork's spec, unedited)

`v0.22.0`. 674 tests, ruff and pyright clean. Both reducers at L0
(`foundation/mc_estimator.py`), both wired at L4: antithetic as a mirrored-draw mode on the path
engine, the control variate into the Bermudan pricer.

## 8. The spec's headline is REFUTED — the sandwich is structural, not sampling error

§0 states the Bermudan sandwich is 15.75% *"because of sampling error"*, and oracle 5 expects the
control variate to *"visibly narrow that sandwich"*. **It cannot, and tranche 18's own numbers already
said so** — the gap there was 12 standard errors wide, not one. Measured over a **64× path increase**:

| pricing paths | policy (lower) | foresight (upper) | width | SE(policy) | width in SEs |
|---|---|---|---|---|---|
| 2,000 | 5,847.77 | 6,736.07 | 15.19% | 151.25 | 5.9 |
| 8,000 | 5,860.27 | 6,783.50 | 15.75% | 74.26 | 12.4 |
| 32,000 | 5,976.53 | 6,877.75 | 15.08% | 37.78 | 23.9 |
| 128,000 | 5,941.69 | 6,856.34 | 15.39% | 18.75 | **48.8** |

The standard error falls by exactly a factor of 8 across 64× paths — the `−½` rate, precisely. **The
width does not move at all.** It is the perfect-foresight bias plus the policy's suboptimality, both
structural; no variance reduction touches either.

So the tranche's deliverable is not a narrower bracket but **sharper error bars**, which is real and
measured (§10). The oracle was rewritten to assert the honest statement: the reduction shrinks the SE
and **leaves the bracket where it is** — and that invariance is itself a check, because a reduction
that *moved* the bracket would be introducing bias.

## 9. The spec's chosen control is BIASED — and an exactly-centred one exists

§2 and oracle 5 name the **Rebonato swaption** as the Bermudan's control. Eq 10.8 is unbiased *only*
because the added term `β(g* − g)` has mean zero, which requires `g*` to be the **exact** `E[g]`.
Black(Rebonato) is an approximation — §5.11 proves the LMM swap rate is not lognormal — so centring on
it injects the ~0.5% freezing error as a **permanent bias**, at every path count. That is exactly what
the spec's own oracle 1 (unbiasedness, "the one failure that matters") forbids. The spec's §3 ruling
and its §2 piece list are in conflict, and §3 is the one that is right.

**The exactly-centred alternative needs no model result at all.** Under the `P_N`-forward measure every
asset-over-numéraire ratio is a **martingale**, so the numéraire-relative annuity `A/B_N` has
expectation `A(0)/B_N(0)` by construction — exactly, with no approximation anywhere. Measured against
the Bermudan payoff, with each candidate's martingale property verified:

| control (all exactly centred) | ρ | 1 − ρ² | SE factor | martingale check |
|---|---|---|---|---|
| `f_2(T_first)` | +0.6133 | 0.6239 | 0.790 | 0.80 SE from exact |
| `f_3(T_first)` | +0.6977 | 0.5133 | 0.716 | 0.63 SE |
| `f_4(T_first)` | +0.6999 | 0.5101 | 0.714 | 0.51 SE |
| `f_5(T_first)` | +0.6264 | 0.6077 | 0.780 | 1.12 SE |
| **`A/B_N` over the span (adopted)** | **+0.7368** | **0.4571** | **0.676** | **0.66 SE** |

So the adopted control gives a **54% variance reduction** (SE × 0.676) with **zero** bias — better
than any approximate control could honestly deliver. The intrinsic at the first exercise date
correlates marginally better (ρ = 0.7444) but has **no closed-form mean**, so it is not a control.

## 10. The constant-control fallback, which is the finance rather than a safety net

For a single-period residual swap, `A/B_N` collapses to exactly `τ` on every path — the same identity
that makes tranche 17's one-period reduction exact — so the control has zero variance and §10.3's
`β* = Cov/V[g]` is undefined. A constant control is not a degraded control; it is the absence of one,
and the engine falls back to the plain estimator.

**This also preserved tranche 18's sharpest oracle, and that is how it was found.** T18's oracle 1
requires a one-exercise-date Bermudan to agree with the T17 European **bit-identically**. Reducing one
side and not the other broke it immediately — four T18 oracles went red on the first wiring. Widening
that comparison back into a standard-error band would have undone the power T18 gained (it was
precisely the 8–11%-resolving band that hid a 1.2% defect there). The fallback means both sides are
plain in exactly the case where the comparison is exact, so T18's bit-identity survives untouched.

## 11. Jäckel does not print the variance factor — so it is derived, not transcribed

§1 of the spec says to transcribe *"`β*` and the variance-reduction factor `-raw`+`-layout`"`. Eqs
10.7/10.8/10.9 are printed and the two renderings **agree exactly**. The factor `V[v](1 − ρ²)` is
**not in the source at all** — second time this book has sent us to a formula it does not print (T15's
`Φ⁻¹`). It is instead *derived* from 10.9 in two lines (`V[v] + β²V[g] − 2βCov` at `β = β*` collapses
to `V[v] − Cov²/V[g]`) and asserted as a **theorem**, verified to 2e-17 on a synthetic pair with ρ
fixed by construction. §4 blesses this directly: a derivation step is an independent check on its
result, and a theorem is stronger than a second reading of one line.

Also worth recording: §10.3 offers a **third** route for `β` that the spec's "β=1 vs β*" framing omits
— *"a remedy to the problem of bias due to a correlated estimate of the control parameter β is to use
an initial simulation, possibly with fewer iterates than the main run, to estimate β* in isolation."*
That dominates both options (optimal β *and* provably unbiased). `control_variate_estimate` takes
`beta` as an optional argument precisely so a caller can supply a pilot-fitted one; the Bermudan uses
the same-sample `β*`, whose `O(1/N)` bias is below the reported SE at the committed path count.

## 12. Two of my own test errors, both caught by execution

- **"Antithetic has exactly zero standard error on a linear payoff."** It is zero in ℝ and **1.5e-18**
  in floating point: `3z + 1.5` and `−3z + 1.5` each round, so their sum is not bit-exactly `3.0`. The
  residual is machine noise in the *summands*, not variance in the estimator. Asserted now as a
  relative bound at the floating-point floor, with the reason stated.
- **The centring-bias sign.** I asserted `wrong.mean == honest.mean − β·Δ`; eq 10.8 *adds*
  `β(g* − g)`, so raising `g*` **raises** the estimate. The direction is now part of what the oracle
  pins rather than an incidental.

And one process error worth recording because it cost a false gate failure: I backgrounded
`verify.py slow` and then ran the full suite and pyright on top of it. The gate reported
`test_swaption_mc.py` at **690.6s** against a true 1.75s — a 400× artefact of CPU contention I
created. **A timing gate's reading is only valid when nothing else is competing**, which is obvious in
retrospect and was not obvious while doing it.

## 13. Findings for Cowork (§10 — raised, not decided)

1. **Oracle 5 is not achievable and was rewritten (§8).** The sandwich is structural; variance
   reduction cannot narrow it, and T18's numbers implied that before this tranche started. Rule
   whether the rewritten oracle (SE shrinks, bracket invariant) stands.
2. **The control was substituted (§9).** Rebonato is not a valid control because its mean is
   approximate; the numéraire-relative annuity is exactly centred by the martingale property and
   gives a 54% variance reduction. Rule the substitution. The general lesson may be worth binding:
   **a control variate needs an EXACT mean, so an approximation is not a control however well it
   correlates** — and in a martingale measure the exactly-centred controls are free (any
   asset/numéraire ratio).
3. **Antithetic is wired into the path engine but NOT into the Bermudan pricer.** Compounding two
   reductions would entangle the oracles, and the control variate is the larger, exactly-centred win.
   Antithetic is oracled end-to-end on the caplet instead. Rule whether that scope split stands or
   whether the Bermudan should compound both.
4. **Candidate §4 note, from §12:** a timing gate's measurement is invalid under concurrent load.
   Worth one line somewhere, because the failure looks exactly like a real regression.
