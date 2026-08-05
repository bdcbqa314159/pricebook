# Checkpoint — Topic 1 Slice 5: Hagan–West monotone-convex interpolation (v0.89.0)

New L0 vocabulary (a primitive + an `Interpolation` value) and the first external-oracle slice. Four
review inputs + named next checkpoint.

## 1. Oracle-quality audit — the figures→equations reframe (on file, as instructed)
The tasking expected the paper's worked example as the external value oracle. **On sourcing the AMF
2006 paper (dropped locally), both it and the 2008 Wilmott paper present the monotone-convex example
ONLY as figures (Fig 2/3/7/8/9/10) — no value table exists.** Ruling (Cowork): anchor on the paper's
**equations** (closed-form), the top of §4's hierarchy, which is *stronger* than reading points off a
graph. The oracle set:
- **eq-33 interval-average reproduction to 1e-14** — the method's defining integral identity, machine
  precision. Necessary but NOT sufficient (validates averages, not amelioration shape).
- **Per-region pointwise checks (i–iv)** vs values **hand-derived from eq 47/49–56** and hard-coded in
  the test (independent of the implementation) to 1e-12 — this is the sufficiency check that catches a
  wrong-region-formula bug; plus the boundary conditions `g(0)=g₀, g(1)=g₁` in every region.
- Positivity (clamp keeps forwards ≥ 0, eq-33 preserved), degeneracy (HW == log-linear on constant
  forwards, df diff 0.0), reprice-to-par through the L4 engine on HW curves (1.4e-16).
- **tf-quant-finance cross-check NOT run** — it is a heavy TensorFlow dependency; per the ruling ("if
  tf-qf unusable, fall back to a minimal independent scratch re-implementation, §7bb"), the
  hand-derived per-region values ARE that independent scratch check. Recorded as the substitution.
- **Weak spot:** no third-party numeric confirmation of the *ameliorated* values (only the paper
  equations + hand calc). The equation anchor is rigorous, but a tf-qf/QuantLib corroboration would
  add coverage — deferred to an oracle-infra pass.

## 2. Drawdown reconciliation (N / 793) — premise correction
The tasking said HW is "missing from both trees." **Wrong** — the quarry HAS it:
`core/forward_interpolation.py::monotone_convex_forwards` (fan-in 0) and
`curves/curve_advanced.py::smooth_forward_curve` (fan-in 0). ng's HW supersedes the monotone-convex
forward *construction*. **Verdict: ticks 0** — `forward_interpolation.py` is a **multi-method** module
(`ForwardInterpolationMethod` enum), so ng's single-method HW is at most a **partial** cross; deletable-
bar rigor forbids ticking a 256-LOC multi-method module from one method without the full retire-read.
Flagged as a partial-cross candidate for the next align / Cowork spot-check. Drawdown stays 18/793.

## 3. Challenge-me list
1. **L0/L1 split.** The falsification gate held (the L0 primitive is pure math — knots/averages/flag),
   but is `MonotoneConvex.integral` at L0 finance-free, or is "integrate the reconstruction" already a
   curve concept? I judged it math (antiderivative of an abstract function). Attack it.
2. **Sequential-HW deferred.** HW is non-local; I scoped it to the simultaneous solve and deferred the
   sequential terminal-interval convention. Is deferring right, or does a consumer need sequential-HW?
3. **The l=0.20 smoothness amelioration (eq 63–80) is NOT built** — only the four-region monotone-convex
   construction. Correct scope, or is the less-local ameliorated variant needed for real curves?
4. **`_forward_reconstruction` rebuilt per `df()` call** (O(n) each). Fine for the correctness env; a
   hot HW curve wants caching (Topic-1 curve-caching, already deferred).

## 4. Smell + debt scan
- **Debt:** 0 suppressions; `verify.py debt` green. No `# noqa`/`# type: ignore`.
- **Fields:** `MonotoneConvex` 4 fields ≤5; `verify.py fields` green. Enum value added, one field on
  `CurveBuild` unchanged.
- **`layers` gate green** — `foundation/hagan_west.py` is finance-free (the gate would catch a forward/
  DF/vol leak; it passed, corroborating the falsification gate).
- ruff (src) + pyright clean; 185 tests. The only ruff hit is the pre-existing `test_rate_index.py`
  helper (out of CI's `src`-only scope).

## Named next checkpoint
**C1 cluster close** — one capability remains: **CSA / collateral-keyed discounting + xccy** (activates
the `discount(ccy, collateral)` signature already landed). C1 closes when every pillar reprices to par
across the full construction set under both orchestrations and interpolation modes. Backstop: §6
≤6-slice cadence (this is slice 5 of ≤6). **Immediate housekeeping:** the `forward_interpolation.py`
partial-cross retire-read (does ng's HW + the deferred methods fully cross it?).
