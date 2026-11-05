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
