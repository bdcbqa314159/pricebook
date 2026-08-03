# Checkpoint — Topic 1 Slice 4: global solve + Jacobian (v0.88.0)

New vocabulary (`CalibrationMethod`, `SolveConfig`, `Jacobian`) + the N-D solver adapter, and C1
advances. Four review inputs + named next checkpoint.

## 1. Oracle-quality audit
- **Degeneracy (sequential == simultaneous)** matches to ~1e-15 (discount 5.6e-16, projection 1.8e-15).
  This is the strongest oracle in the slice: it is **anchored to the already-green sequential path** and
  cross-checks **two independent orchestrations** (brent per-pillar vs scipy LM on the whole system) —
  not self-consistency. If the global solve had a bug, it would disagree with the proven bootstrap.
- **Reprice-to-par through the L4 engine** under the globally-solved curves (worst |PV| ~3.5e-16) — the
  backstop, unchanged from prior slices.
- **Jacobian is oracle-able, not asserted**: the tangent prediction `residual(x+dx) − residual(x) ≈
  J·dx` holds to < 1e-6 (finite-difference bump-and-rebuild). It validates the *meaning* of the matrix.
- **Non-convergence is a value** (`max_iterations=1` → `CalibrationFailure`); global reprice
  byte-identical.
- **Weak spot (carried):** still no external (QuantLib / analytic multi-curve) cross-check — the rates
  oracles remain self-consistent-to-par + the sequential anchor. A QuantLib dual-curve benchmark stays
  the standing hardening item.

## 2. Drawdown reconciliation (N / 793) — retire-read of the quarry global solvers
Retire-read (end-to-end) of the three global-solve homes. **All superseded by ng's simultaneous solve;
0 ng consumers of any shed/deferred bit** (grep: the 3 apparent hits are docstrings/provenance):
| quarry module | LOC · fan-in | disposition |
|---|---|---|
| `curves/multicurve_solver.py` | 486 · 0 | **superseded** — joint OIS+projection Newton = ng's exact 2-curve global solve |
| `curves/ncurve_solver.py` | 279 · 0 | **superseded** — the mined `InstrumentPricer` state-vector pattern; ng does the 2-curve instance |
| `curves/global_solver.py` | 417 · 2 (uncrossed) | **superseded** — simultaneous Newton over all pillar DFs |

Shed / deferred (all 0 ng consumers): hand-rolled **damped Newton** + **finite-difference Jacobian** →
`shed` (ng uses scipy per §7bb — implementation choice, not a missing capability); **analytic Jacobian**
(`curve_analytical_jacobian`) → `deferred→C3-risk / AAD` (doc 18 §6); **N-curve > 2** generality →
`deferred→3rd-projection-curve` (rule of two); `validate_curve` robustness checks → `deferred` (no
consumer). Forward-links filed on those destination rows.

**Two flags on the count:**
1. **Blocked on #119.** The bootstrap/discount retire tick (13→15) is **not yet merged to main**, so
   this slice branched off the 13 base. The physical tracker count reconciles once #119 lands: 15 → **18**
   (three global solvers). Recorded here; the MANIFEST/tracker count edit is deferred to avoid a
   conflicting write against #119's rows.
2. **Exceeds the tasking's "16 or 17".** The task named `ncurve_solver` (+ `global_solver`); the read
   found **`multicurve_solver` is also directly superseded** (it *is* the joint OIS+projection case). So
   three cross, not two — flagged for Cowork's spot-check (may hold one partial; reversal is cheap).

## 3. Challenge-me list
1. **Ticking three ~400-LOC hand-rolled solvers with one ~50-LOC scipy wrapper.** Defensible under
   §7bb (scipy replaces hand-rolled) + phantom-residual (deferred bits have no ng consumer) — but it is
   the largest supersede-vs-clone ratio yet. Attack it.
2. **The Jacobian is scipy's numerical `result.jac`, not analytic.** Correct and oracle-validated, but
   is a numerical Jacobian the right thing to *ratify* as `CalibrationResult.jacobian`, or should the
   type carry a provenance flag (numerical vs analytic vs AAD) now?
3. **Seed = flat 3% DFs.** Mined from `ncurve_solver`'s default. Robust here; is it robust for inverted
   / stressed curves, or should the seed be the sequential result (at the cost of the independence the
   degeneracy oracle relies on)?
4. **`SolveConfig` shares knobs across both methods.** Fine now (Brent xtol / LM xtol·ftol); the moment
   a method-specific knob appears (damping, trust-region), it must decompose by method family (§3b).

## 4. Smell + debt scan
- **Debt:** 0 suppressions; `verify.py debt` green. No `# noqa` / `# type: ignore`.
- **`root_nd` catches `(ValueError, FloatingPointError, ZeroDivisionError)` → `converged=False`** — a
  *typed* catch returning a failure value (invariant 4), not a silenced/empty except; not a ledger item.
- **Union return** `tuple[…] | CalibrationFailure` — pyright clean (0 errors); existing callers
  unaffected.
- **Fields:** `CalibrationSpec` 5, `SolveConfig`/`Jacobian`/`CalibrationResult` 3, `CalibrationFailure`
  1 — all ≤5; `verify.py fields` green. ruff (src) + pyright clean; 176 tests.
- No `isinstance` ladders; the orchestration fork is a single `spec.solve.method is SIMULTANEOUS` check.

## Named next checkpoint
**C1 cluster close** — after the two remaining C1 capabilities: **Hagan–West monotone-convex
interpolation** (genuinely new work, absent from both trees) and **CSA / collateral-keyed discounting +
xccy** (activates the `discount(ccy, collateral)` signature already landed). C1 closes when *every
pillar reprices to par* holds across the full construction set under both orchestrations. Backstop: the
§6 ≤6-slice cadence (this is slice 4 of ≤6 since Topic-0 close). **Immediate housekeeping:** merge #119,
then reconcile the drawdown count to 18/793 (or ratify the multicurve_solver flag).
