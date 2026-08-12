# Checkpoint — Cluster C1 CLOSE (Topic 1, slices 1–6c)

**Version:** v0.92.0 · **Date:** 2026-08-08 · **Branch:** `slice/06c-xccy-basis`

C1 = the linear-rates curve world: single-curve → dual-curve → cash instruments → global solve →
Hagan–West interpolation → CSA/xccy. This is the planned cluster-boundary checkpoint (CLAUDE.md §6).
It doubles as the deferred-inventory ("what are we carrying") round.

The four ratified review inputs + the named next checkpoint follow.

---

## 1. Oracle-quality audit (across ALL C1 oracles)

| Slice | Oracle | Class | Verdict |
|---|---|---|---|
| 1 single-curve | `PV = N·exp(−r·t)` to 1e-12; DV01 analytic vs FD to 1e-6 | **closed-form** | strongest |
| 2 dual-curve | EURIBOR par swap → zero NPV dual-curve to 1e-9 | self-consistent (par) | adequate¹ |
| 3 cash | deposit/FRA/future each reprice to par; future = 1−price at IMM | self-consistent | adequate¹ |
| 4 global solve | SEQUENTIAL == SIMULTANEOUS to 1e-10; every instrument reprices | self-consistent (degeneracy) | strong² |
| 5 Hagan–West | eq-33 interval-average identity to 1e-14; per-region hand-derived pointwise to 1e-12 | **external (paper equations)** | strong³ |
| 6a FX spot | declared-pair resolution + inversion; raises on undeclared cross | closed-form (identity) | strong |
| 6b collateral | own-ccy collateral == domestic OIS to 1e-12; basis recorded | closed-form (degeneracy) | strong |
| 6c xccy | reprice-to-zero (0 + 10bp); **CIP F = S·df_f/df_d at zero basis to 1e-10** | **closed-form (CIP)** | strong⁴ |

¹ Reprice-to-par is self-consistent; par swaps telescope, so it can mask a shared-atom bug. Mitigated
structurally by §3d (one `rpv01`/`float_leg_pv` both calibrator and engine compose) — the class of bug
reprice-to-par is blind to is *designed out*, not tested out.
² The two orchestrations agreeing is a genuine cross-check (independent solve paths). Does **not** cover
xccy (its pillars are a sequential post-step, not in the global vector — stated, not claimed).
³ tf-quant-finance cross-check not run (heavy TF dep); the paper equations + hand-derived per-region
values are the independent anchor. No value table exists in the source (figures only).
⁴ CIP is a true external closed form (covered interest parity), stronger than the reprice-to-par self-
consistency that pins the rest of the curve. This is the strongest oracle in the cluster after slice 0.

**Verdict:** every C1 slice has ≥ self-consistent, and the two structural risk points (par-telescoping
blindness; xccy not in the degeneracy) are each closed by design (§3d) or explicitly scoped, not left
implicit.

**Audit-response amendment (slice 6d, v0.92.2).** A third-party audit of v0.92.0 (`analysis/AUDIT_FINDINGS.md`)
confirmed the numerical core clean but found the C1 oracles tested only positive-rate / in-range / well-formed
inputs — a **coverage blind spot at input/range boundaries**, not a wrongness in the priced numbers. Eight
findings, all now closed red→green: **negative-rate** sequential calibration (DF>1 bracket + invariant-4
escape — #1); **Hagan–West extrapolation** past the last pillar (silent, now raises — #2); **bad-input**
guards (ICMA freq>12 hang / non-divisor #3; RegularPeriod coincident-or-reversed anchors #4 and short-stub
flag #5; WeekendSchedule ordering #6); **config-tolerance** decoupling (convergence graded at a literal
1e-10, now at `spec.solve.tolerance` — #7); and the HW O(n²) reval (now cached — #8). The oracle set is
extended with boundary coverage; the in-range oracles above are unchanged and still green.

## 2. Drawdown reconciliation

**19 / 793** (6a/6b/6c add capability, 0 ticks; +1 from the slice-5 `forward_interpolation` full cross,
housekeeping 2026-08). = 13 Topic-0 parked + 6 Topic-1 crossed (bootstrap, discount_curve,
ncurve_solver, global_solver, multicurve_solver, forward_interpolation). 6a/6b/6c added **capability**, superseded **no** quarry
module: FX spot / collateral keying / a minimal foreign-collateral curve do not delete the quarry's
full FX/XVA suite (xccy_swaption, PRDC, fx_smile_cube, NDFs, bilateral_csa, collateral_optimisation), which
retires with the **FX / XVA topics**. 0 ticks is correct and ratified (doc 18 §4/§9). The slice-4
`multicurve_solver` spot-check is now **ruled: full cross (Cowork), 3 ticks stand, 19/793** — §4 grep
found 0 production/dynamic consumers (only quarry-internal tests, which retire with the quarry).

## 3. Challenge-me — the C1 design as a whole

- **`CurveSet` = closed shape × open keys.** A new asset adds keys (collateral, entity), not fields —
  held across 6 slices (6b grew `CurveKey` by one field for collateral, then stopped). *Challenge:* is
  `CurveKey.collateral` the last field growth, or will entity/underlying force more? (Doc 19 says keys;
  6b's field-growth is the one concession — watch it.)
- **Two orchestrations (sequential/simultaneous).** Justified by the rule of two (both have real
  consumers: pillar bootstrap + global solve + Jacobian). *Challenge:* xccy breaks the symmetry (post-
  step only) — is a unified state vector worth building before a 2nd xccy consumer? Ruled: no (deferred).
- **Collateral keying.** `discount(ccy, collateral)` normalizes own-ccy → domestic. *Challenge:* the
  product carries `collateral` (L2), but the relocation trigger says it belongs at L6 (trade). Carrying a
  known-misplaced field until L6 — acceptable debt, logged.
- **FX directionality (doc 19 §2.1).** `fx_rate` inverts around a declared canonical pair, raises on
  undeclared. *Challenge:* triangulation (EUR/JPY via USD) is deferred — the raise is the guard.
- **Unified multi-currency front.** One `calibrate`, per-currency loop + xccy post-step. *Challenge:* the
  `.currency`/`.discount`/`.projection` convenience accessors on a multi-currency spec raise — a mild
  smell (a spec that sometimes has a `.currency`). Justified: zero churn on slices 1–5. Re-open if a 2nd
  multi-currency consumer wants them gone.
- **The xccy S-cancellation finding.** FX spot + USD-DF drop out of the constant-notional bootstrap. This
  is correct for the minimal scope but means `spec.fx` is *carried but unused* by the xccy solve (used by
  the CIP oracle + downstream FX). *Challenge:* is carrying `fx` on the spec premature? No — the snapshot
  needs it regardless (6a), and MtM-notional resets (deferred) will make it load-bearing in the bootstrap.

## 4. Smell + debt scan

- **`OPEN.md` / debt ledger:** `verify.py debt` green (suppressions − ledger = 0). No new suppressions in
  C1. No `# type: ignore`, no skipped tests, no empty `except` added.
- **Pre-existing smell (not C1):** `tests_ng/L0/test_rate_index.py:_rfr` trips ruff `PLR0913` (8 args) —
  a test helper, pre-dates this cluster, tests are outside the §3b `src/**` scope. Flagged, not fixed here.
- **Exception-count (§3d gauge):** zero `isinstance`/type-switch on product-or-model inside engine or
  calibrator across C1. Dispatch is registry-by-type; the xccy instrument composes the shared atoms. No
  cluster of exceptions → no missing building block signalled.
- **Field/arg discipline:** every new type ≤5 fields (`CurrencyCurves` 3, `CalibrationSpec` 5, `XccyBuild`
  3, `XccyBasisQuote` 2, `XccyBasisSwap` 4). `verify.py fields` green.
- **Carried debt (deferred, all named-triggered; now in `OPEN.md` §5 ledger):** collateral→L6 relocation;
  simultaneous xccy solve; triangulation; FX vol/exotics/NDF/tenor-basis/MtM-resets → FX/XVA topics;
  `forward()` subtract-first hardening (trigger: RFR daily compounding). *(`multicurve_solver` spot-check
  now ruled full cross; `forward_interpolation.py` retire-read now resolved — see §2/§5.)*

## Named next checkpoint

**Audit-fix slice first, THEN C2 opening.** An audit-fix slice (the reconciliation findings that need
*code*, not just docs) lands before C2. Only then the models/calibration cluster opens — the F2 capability
model (doc 22): first dynamics model (`CalibratedModel` + capability protocols) with its own numerical-
config value (invariant 5 returns shaped by a real consumer). Checkpoint fires at the first of ≤6 slices
or the C2 capability boundary. **Do not open C2 until the audit-fix slice lands AND this checkpoint is
amended** (in addition to Cowork ruling this C1 checkpoint).
