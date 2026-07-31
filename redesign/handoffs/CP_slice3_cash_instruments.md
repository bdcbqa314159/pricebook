# Checkpoint — Topic 1 Slice 3: cash instruments (v0.87.0)

Written per §6 (this slice lands **two** cross-cutting abstractions — the L4 engine registry and the
L3 `CalibrationInstrument` protocol — and completes the C1 cash front). Four review inputs + the named
next checkpoint.

## 1. Oracle-quality audit
- **What ran:** every one of the 15 calibrating instruments reprices to **zero NPV through the full L4
  engine** (registry-dispatched), not merely through the calibrator residual — discount worst |PV| =
  6.7e-16, projection worst |PV| = 2.0e-15. Future reproduces `1 − price` at its IMM segment (worst |Δ|
  = 5.4e-16). Deposit `df = 1/(1+r·τ)` closed form. Regression: slice-1/2 par swaps still zero.
  Statelessness byte-identical; beyond-curve / unresolved key → `PricingFailure`.
- **Strength / weakness (honest):** reprice-to-par is **self-consistent** and, by F1 §6, blind to
  mis-resolution. Two things offset it: (a) the deposit `df` is a **closed-form** anchor (not
  self-consistent); (b) the §3d shared-atom construction makes the L3 residual and L4 price compose the
  *same* `deposit_df`/`forward`, so the through-engine reprice is a genuine backstop, and index→curve
  resolution is structural (`CurveSet.projection(index)` fails loud → `PricingFailure`, not a wrong
  number). **Weak spot to record:** no external cross-check (QuantLib / analytic dual-curve) yet — the
  rates oracles are self-consistent-to-par. A QuantLib cross-check would strengthen; deferred (no
  oracle-infra). The futures oracle deliberately tests the *approximation applied*, never a market price
  (doc 18 §2).

## 2. Quarry-drawdown reconciliation (13 / 793)
- Parked at Topic-0 close: **13**. This slice **ticks 0**.
- §4 consumer analysis (evidence, in the align commit): "full G10 conventions" gate has **0** ng
  consumers ⇒ `deferred→G10`; `discount_curve.py`'s `zero_rate`/`bump`/`roll_down` consumed only by
  uncrossed quarry modules ⇒ `deferred→(risk/construction)`. **No needed-now blocker.**
- **Verdict:** deletable-bar rigor (§4) forbids ticking from "looks covered" — `bootstrap.py` (713 LOC)
  + `discount_curve.py` (300 LOC) require the formal end-to-end retire read (classify every omission
  with grep evidence). That read is the **immediate next step** (below), not rushed into this build.
- **FLAG (reconcile):** the drawdown denominator is inconsistent — `/768` (CLAUDE.md and slices 1–2
  CHANGELOG/MANIFEST) vs `/793` (this checkpoint per Cowork, and the ng-migration memory). Substance is
  unaffected (13 parked, 0 new ticks), but the denominator must be pinned once.

## 3. Challenge-me list (design choices to attack)
1. **`residual(discount, projection)` naming** — `projection` is overloaded to mean "the trial curve
   being solved" (in the discount build it *is* the discount curve). Correct but confusing; should the
   signature name the trial explicitly?
2. **Three type-keyed dicts** — engine product→pricer, calibration quote→instrument, quote→product.
   Right amount of structure, or is one collapsible? (Each avoids `isinstance`; all three are one-liners.)
3. **`FRAInstrument` vs `FutureInstrument`** are near-identical (`forward − target`); kept separate per
   the ratified spec. Collapse to one `ForwardInstrument`, or does the `1 − price` convention earn a type?
4. **Futures pillar placement** — IMM `df(start)` interpolates between the 12M pillar and the trial end.
   Acceptable, or should placement be explicit (doc 18 §3 hints at it for C3)?
5. **Projection front = a spot FRA (`start=None`)**; deposits anchor only the discount curve. Is
   `FRAQuote.start: Tenor | None` clean, or should a spot fixing be its own instrument?

## 4. Smell + debt scan
- **Debt:** 0 suppressions, `verify.py debt` green. No `# noqa` / `# type: ignore`.
- **No `isinstance` ladders** — dispatch is type-keyed dict lookups (§1).
- **Fields:** all ≤5 (`CurveBuild` 5, `FRA`/`Future` 5, rest fewer); `verify.py fields` green.
- **`Any`** appears only in the registry `Pricer` alias and the two factory-dict annotations — required
  for heterogeneous dispatch typing; pyright 0 errors.
- **Watch:** parallel `quote`/`product`/`instrument`/`pricer` per type (4 kinds × 3 new instruments) is
  a lot of small types. It is the ratified separation (doc 18 §1: quotes ≠ products), but if a future
  instrument type adds no distinct behaviour, revisit for consolidation.

## Named next checkpoint
- **Immediate next action (a retire step, not a checkpoint):** the end-to-end retire read of
  `bootstrap.py` + `discount_curve.py`, ruling their tick with evidence (first Topic-1 drawdown > 0).
  The align-commit consumer analysis is its head start.
- **Next CHECKPOINT: C1 cluster CLOSE** — after the remaining C1 slices (Hagan–West monotone-convex
  interpolation · global/simultaneous solve + par→zero Jacobian · CSA/xccy discounting), when *every
  pillar reprices to par* holds across the full construction set. Backstop: the §6 ≤6-slice cadence
  (this is slice 3 of ≤6 since Topic-0 close).
