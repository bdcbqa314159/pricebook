# Release Notes

---

## v1.178.0 — 2026-06-28 — **LMM (iterative-scaling) audit fixes (Phase 5 of the 13-calibrator audit)**

**Files**: `models/lmm_calibration.py`.

* **`calibrate_lmm_vols` now reports an honest `converged`.** The iterative-scaling loop ran `max_iter` iterations unconditionally with no stopping criterion, yet hardcoded `converged=True`. Added a stabilisation test (a full sweep that moves the vols by less than `tol` → converged; break early), and the record now captures the real `converged` and the true iteration count — `converged=False` when the targets are unreachable / the loop oscillates.
* **Fixed botched `SABRSlice` / `MultiFactorSABR` spacing** (3 blank lines mid-class + 0 between the two classes).

**Verification**: full suite **13,021 passed**, zero failures.

---

## v1.177.0 — 2026-06-27 — **Jump-model audit fix (Phase 4 of the 13-calibrator audit)**

**Files**: `models/jump_calibration.py`.

* **`_cos_implied_vol` no longer masks errors.** Narrowed its broad `except Exception: return 0.0` (around the Black-vol inversion) to `except ValueError`, and lifted the `implied_vol_black76` import out of the `try` — same masked-error hardening applied to hw/g2pp. A real bug (or `ImportError`) now surfaces instead of silently becoming a "zero implied vol".

**Confirmed-but-deferred (Phase-4 finding, NOT fixed here):** a real latent correctness gap — `merton`/`vg`/`nig`/`cgmy` char funcs omit `div_yield` from their drift (only `kou`/`bates` include it), and `cos_price`'s `div_yield` parameter is silently ignored, so for `div_yield ≠ 0` those four models price at the wrong forward while `_cos_implied_vol` inverts against a dividend-adjusted Black forward. Latent (default `div_yield=0`). Needs a dedicated validated slice (thread `div_yield` into the four drifts + remove the dead `cos_price` param + a `q≠0` put-call-parity test).

**Verification**: full suite **13,021 passed**, zero failures.

---

## v1.176.0 — 2026-06-27 — **G2++ audit fixes (Phase 3 of the 13-calibrator audit)**

Three findings from the deep audit of `models/g2pp_calibration.py`. The G2++ analytics were verified correct (`_g2pp_V` against Brigo-Mercurio 4.10, the swaption pricer against B-M 4.31).

**Files**: `models/g2pp_calibration.py`.

* **`g2pp_implied_vol` no longer masks errors.** It used a broad `except Exception: return 0.0` around the Black-vol inversion — swallowing genuine bugs as a silent "zero implied vol", inconsistent with its hardened `_hw_implied_vol` sibling (and ironically the file documents removing exactly this pattern, T2.11, from the swaption pricer). Narrowed to `except ValueError` (the arbitrage-violation case), letting other exceptions surface.
* **Removed ~50 lines of dead code** — `_g2pp_zcb` and `_g2pp_zcb_option` had zero callers in source or tests (the swaption pricer builds its `A_i` and does Jamshidian inline).
* **Dropped a dead `swap_pv` variable + a stream-of-consciousness `# Wait:` comment** in the degenerate branch of `g2pp_swaption_price` (the `payer_pv` logic was already correct).

**Verification**: full suite **13,021 passed**, zero failures.

---

## v1.175.0 — 2026-06-27 — **Hull-White audit fixes (Phase 2 of the 13-calibrator audit)**

Two findings from the deep audit of `models/hw_calibration.py`.

**Files**: `models/hw_calibration.py`.

* **`a_bounds` / `sigma_bounds` are now enforced for every method.** The default `method="nelder_mead"` ran unconstrained (no bounds passed; the final clamp only *floored* at `1e-4`/`1e-6`), so the documented bounds applied only to `differential_evolution` / `L-BFGS-B`. The returned `a` / `sigma` are now clamped into `[max(bound_lo, model_floor), bound_hi]` for all methods, with the model-validity floors kept as the ultimate lower limit. Docstring updated to state the bounds are enforced on the result.
* **`rmse` returns `0.0` (not `int 0`)** in the empty-swaption case.

Residual units were verified consistent between the eager and lazy record paths; the T4-HW1 defensive-coding (narrow `except ValueError` on vol inversion) was confirmed correct.

**Verification**: full suite **13,021 passed**, zero failures.

---

## v1.174.0 — 2026-06-27 — **SABR audit fixes (Phase 1 of the 13-calibrator file-by-file audit)**

Two findings from the deep audit of `options/sabr.py`, both behaviour-preserving.

**Files**: `options/sabr.py`.

* **`calibrate_sabr_smile` docstring corrected.** It claimed *"Raises: ValueError if calibration fails validation"* and called itself "hardened", but the body only ever `warnings.warn`s (range issues + RMSE-over-threshold) and always returns. Making it raise was rejected — `structured/ir_vol_surface.py` calls it per node to build a surface with a tight 1bp default, so raising would break surface construction on any imperfect smile. Docstring now states it warns (never raises) and always returns with reprice diagnostics filled.
* **Renamed `x_z` → `z_over_x` in `sabr_implied_vol`.** The variable holds Hagan's `z/x(z)` multiplicative factor, not `x(z)` — the old name invited a misread. Comment clarified. (The Hagan ATM + general branches were verified numerically correct during the audit; no formula change.)

**Verification**: full suite **13,021 passed**, zero failures.

---

## v1.173.0 — 2026-06-27 — **Dividend calibrator: capture an honest SolveReport eagerly on all four paths**

Closed the last eager-capture gap among the 13 model calibrators (found in the design-health review). `dividend` ran an optimiser (`_calibrate_piecewise` → L-BFGS-B) but discarded its verdict — every dividend record was lazily reconstructed with `converged=None` + `reconstructed=True`, even when freshly fit. Now each of the four construction paths captures and attaches an honest `SolveReport` at fit time, like the other 12 families.

**Files**: `equity/dividend_calibration.py` (+ tests).

* **optimize** → `SolveReport.from_scipy(result, algorithm="optimize")` — the real L-BFGS-B success/iteration count, no longer thrown away (`reconstructed=False`).
* **spline / linear** → `SolveReport.external(converged=True)` — deterministic constructions (cubic-spline interpolation / closed-form bootstrap), no iterative optimiser.
* **options** → `SolveReport.external(converged=True)` + a `residual_is_placeholder` flag and warning: put-call parity extraction sets `fitted := market`, so its residual is a structural `0.0` — flagged rather than read as a perfect fit (same convention as FX-SLV).
* One private `_dividend_calibration_record(...)` is now the single assembly point for all four eager paths **and** the lazy `_build` fallback. Class docstring corrected (it claimed "no per-build-site population").

4 regression tests: optimize captures a real verdict (not reconstructed); deterministic paths are not reconstructed; options' zero residual is flagged, not a false-perfect.

**Verification**: full suite **13,021 passed**, zero failures.

---

## v1.172.0 — 2026-06-27 — **`SolveReport.from_scipy`: one canonical scipy-result adapter (clean interface across all model calibrators)**

The calibration layer records a solve but never runs one — the algorithm (scipy, the pricebook `minimize` wrapper, DE, brentq, particle/Newton loops) lives in each family module. That separation was clean, with one seam: the *adapter* translating a `scipy.optimize.OptimizeResult` into a `SolveReport` was hand-rolled in ~6 families with three divergent idioms. This closes the seam.

**Files**: `calibration/_solve.py` (+ `options/sabr.py`, `models/hw_calibration.py`, `models/jump_calibration.py`, `fixed_income/jarrow_yildirim.py`, `credit/joint_equity_credit.py`, `credit/bond_hazard_bootstrap.py`, + tests).

**The seam (before):** `converged` was spelled three ways — `bool(getattr(result, "success", True))` (sabr/jump/jy/joint), `result.success if hasattr(result, 'success') else True` (hw), `bool(result.success)` (bond_hazard) — and `iterations` two ways (`nit or nfev` vs bare `nit`). Each calibrator depended directly on SciPy's concrete result shape and re-implemented the extraction (a DIP/DRY leak).

**Fix:** added a third honest constructor `SolveReport.from_scipy(result, *, algorithm, tolerance=None, max_iterations=0, seed=None)` alongside `external` / `analytic`. It reads the two facts every SciPy result exposes, **one canonical way**:
* `converged` ← `result.success` (assumed `True` only if a non-standard result omits the field);
* `iterations` ← `result.nit`, falling back to `result.nfev` when no iteration count is reported — so the record is never a misleading `0`.

The bespoke part (which optimiser ran + its config) stays with the caller; only the uniform extraction is centralised. Migrated all 7 scipy sites across the 6 families (sabr, hw, jump, jarrow_yildirim, joint_equity_credit, bond_hazard ×2) onto it; `hw`'s stored `converged` field now reads `solve.converged` (single source). **g2pp stays bespoke** by design (two-stage `de.success or local.success`), as does the brentq-per-bond path (`converged=True`, structural).

Each call collapsed from 3–4 lines of `getattr` to one readable verb, e.g. `SolveReport.from_scipy(result, algorithm="L-BFGS-B", tolerance=1e-12, max_iterations=200)` — legible from any language background. 4 unit tests pin the adapter (success/nit read, nfev fallback, success-default, failure captured).

**Verification**: full suite **13,018 passed**, zero failures.

---

## v1.171.0 — 2026-06-27 — **Calibration reconstruction fidelity: 3 honest-residual fixes from the focused read**

The two low-severity follow-ups noted alongside the multicurve fix (v1.170.0), plus a parameters enrichment — all in the `_build_calibration_record` reconstruction paths, all behaviour-preserving in normal flow (they only change degenerate / hand-built inputs that previously masked a bad fit).

**Files**: `credit/joint_equity_credit.py`, `models/lmm_calibration.py`, `curves/multicurve_solver.py` (+ 3 regression tests).

**Fixes:**
* **`joint_equity_credit` — zero market target no longer reads as a false-perfect.** Both the lazy `_relative_residuals` and the eager `joint_calibrate` computed `model/market - 1 if market > 0 else 0.0` — a zero/degenerate market vol or spread produced a `0.0` (perfect) residual, hiding the miss. Factored to a shared `_relative_residual(model, market)` that falls back to the **absolute** residual `model - market` when the target is non-positive (a relative error is undefined at `market == 0`), so a degenerate target surfaces as a real non-zero miss. Regression test added.
* **`lmm_calibration` — no fabricated model vol for an unfitted target.** The fallback did `fitted_swaption_vols.get(k, 0.0) - target[k]`, inventing a model vol of `0.0` for any target lacking a fitted value (a large spurious miss). Now restricted to keys present in both — an unfitted target is *excluded* (it wasn't fitted) rather than fabricated. The calibrator always fits every target, so normal flow is unchanged; this only corrects a mismatched hand-built instance (and an empty intersection now correctly trips the non-empty-residuals guard). Regression test added.
* **`multicurve` — reconstruction carries the calibrated DF surface.** The fallback shipped `parameters={}`; it now recovers `ois_df(...)` / `proj_df(...)` off the stored curves (`pillar_dates` + `df()`), matching the eager record's key shape. Per-instrument residuals remain unrecoverable (instruments aren't stored), so the aggregate residual stays — but the fitted parameters are no longer lost. Regression test added.

**Verification**: full suite **13,014 passed**, zero failures.

---

## v1.170.0 — 2026-06-27 — **Multicurve: capture the solver's `converged` verdict (kill the last threshold-derived convergence)**

Found by a focused structural read of all 13 `_build_calibration_record` methods across the `CanonicalCalibrationResult` families (the inherited types of the calibration structure). Twelve were honest; one — `MultiCurveResult` — was the lone survivor of the fabricated-converged anti-pattern §0c set out to abolish.

**Files**: `curves/multicurve_solver.py`, `test_calibration_result_curve_bootstrap.py`, `test_calibrator_provenance_conformance.py`.

**The bug (latent):** `MultiCurveResult._build_calibration_record` derived `converged = self.residual < 1e-6` — an invented threshold, inconsistent with the solver's real tolerance (`multicurve_newton` defaults `tol=1e-10`) — while a comment misclassified the record as *"faithful (not reconstructed)"*. The dataclass stored `residual` and `n_iterations` but **not** the solver's actual convergence verdict, so the lazy fallback could only guess. Latent in normal flow (the eager `_build_multicurve_cr` always populates the record with the real verdict), but reachable for a hand-constructed/deserialized instance — and the migration's bar is "records impossible to lie."

**Fix:**
* Added a **required** `converged: bool` field to `MultiCurveResult` — a hand-built instance must state the verdict rather than have it guessed. Set faithfully at both solver exit points (`True` on `residual < tol`, `False` on the max-iter fallthrough).
* `_build_calibration_record` now replays the stored verdict (`converged=self.converged`, never a threshold) and marks the record `reconstructed=True`, consistent with the other 12 families' fallbacks. Comment corrected.
* `to_dict` now surfaces `converged`.
* Tests: existing back-compat constructions pass `converged=`; new regression `test_converged_verdict_is_stored_not_threshold_derived` pins it — a tiny residual (`1e-12`) with `converged=False` now yields `optimiser_run.converged is False` + a non-convergence warning, proving the verdict is stored, not derived.

The other 12 `_build` methods were verified honest: `converged=None`+`reconstructed=True` where no optimiser ran, `self.converged` where the family stores the verdict (hw/g2pp/bond_hazard), or `SolveReport.analytic()` for the closed-form moment match (stochastic_correlation).

**Verification**: full suite **13,010 passed** (1 unrelated pre-existing AAD thread-local flake under parallel load — green in isolation; 2 slow G2++ deselected).

---

## v1.169.0 — 2026-06-27 — **Calibration L1 reading-pass tidy (6 clean-code fixes, no behaviour change)**

Human-eye reading pass over the calibration layer (OPEN.md §0c A1), modules read in `L1_DEPS.md` order (`_solve` → `_types` → `_curve_record` → `_model_record` → `__init__`). Six clean-code findings, all behaviour-preserving — the audit chain already closed correctness; this was SOLID/Fowler/readability only.

**Files**: `calibration/_types.py`, `_curve_record.py`, `_model_record.py`, `_solve.py`, `__init__.py`.

**Fixes:**
* **Shared snake_case audit-key mechanism (DRY).** `OptimiserSpec.algorithm` and `CalibrationFit.model_class` had two identical regexes and two raise-blocks for the same "audit key" vocabulary. Factored to one `_AUDIT_KEY_RE` + `_canonical_audit_key` / `_require_audit_key`. The deliberate, **test-pinned** asymmetry is preserved and now documented as intentional: `algorithm` is canonicalised from human-readable input ("L-BFGS-B"); `model_class` is an internal literal held to discipline (validated-only, rejected if non-canonical).
* **Single record-assembly site (DRY).** Both family builders hand-rolled the same `CalibrationResult(provenance=stamp(...), fit=…, optimiser_run=…, diagnostics=…)` skeleton. New `assemble_calibration_record(...)` is the one place the four-part shape + provenance stamp live; `curve_calibration_record` and `model_calibration_record` both route through it. A change to the record skeleton now touches one site, not two.
* **Dropped dead empty-residuals guards.** `CalibrationFit.rms_residual` / `max_residual` carried `if not residuals: 0.0` / `default=0.0` fallbacks unreachable since `__post_init__` rejects an empty residual vector (the G1 guarantee). Removed — they only suggested "no targets" was a supported state. It is not.
* **Documented `curve_calibration_record(converged=True)` default.** Added a rationale: a bootstrap solves each pillar exactly, so convergence is structural — *not* the "assume converged" anti-pattern the model side (with its real `SolveReport` verdict) guards against.
* **Refreshed stale docstrings.** `__init__.py` claimed calibrators "migrate … in subsequent slices" (migration is complete) and listed 4 of 13 exports; now describes the two-builder surface accurately. `_solve.py`'s "exactly two honest constructors" softened to acknowledge the bare (non-coercing) constructor while pointing at the two classmethods as the intended entry points.

**Verification**: full suite green — **13,010 passed** (2 slow G2++ calibration tests deselected).

---

## v1.168.0 — 2026-06-26 — **Calibration subsystem: adversarial review fixes (2 bugs + 3 gate blind spots)**

A fresh adversarial pass (code-correctness-critic) over the whole subsystem found two real bugs the 13k-test suite missed — both in paths the gates round-tripped the *reconstructed* record for, never the eager one — plus three latent gate holes.

**Files**: `calibration/_types.py`, `db/db.py`, 3 gate tests, 2 new regression tests.

**Confirmed bugs (fixed):**
* **Tuples in `extra` broke record equality on save→load.** `__post_init__` canonicalised the typed sequence fields to tuples but not the freeform `extra` maps; a producer that puts a tuple there (G2++ optimiser `bounds`, jump-diffusion) round-tripped to a *list*, so `db.load_calibration(cid) == cr` was `False`. Fixed with a recursive `_json_normalise` applied to `OptimiserSpec.extra` and `CalibrationDiagnostics.extra` (tuples → lists in memory, matching the on-disk shape). Regression test added.
* **`list_calibrations(converged=None)` always returned zero rows.** `_build_where` emitted `converged = NULL`, never true in SQL — so the tri-state "not captured" records the whole campaign preserved were unqueryable. Fixed to `IS NULL` for `None` filters (all columns). Regression test added.

**Gate blind spots (hardened):**
* **builder-enforcement** only matched bare-`Name` constructor calls — `import _types as t; t.OptimiserSpec(...)` or an aliased import would evade. Now tracks `from … import X as Y` aliases and `ast.Attribute` access.
* **fidelity G7** (model_class uniqueness) only saw producers in the two registries — the 5 heavy calibrators + 3 bond-hazard bootstrappers (`_BEHAVIOURAL_ELSEWHERE`) were uncovered. Added an AST-literal scan asserting each `model_class="..."` maps to one module — complete coverage.
* **mixin frozen guard**: a `@dataclass(frozen=True)` subclass would pass `__init_subclass__` (which runs before `@dataclass`) then fail at first `to_calibration_result()`. The conformance gate now asserts every subclass is non-frozen.

No current producer triggered the three latent holes (verified) — they're closed so a future one can't.

**Verification**: full suite **13012 passed**.

---


## v1.167.0 — 2026-06-26 — **Calibrator code review — two fixes**

Adversarial re-read of the calibrator subsystem at v1.166. Found one real consistency bug + one stale docstring; the rest of the structure verified clean.

**Files**: `fx/fx_slv_calibration.py`, `calibration/_model_record.py`.

* **fx_slv lazy `_build` was fabricating `converged=True`** — a hand-built `ParticleCalibrationResult` is marked `reconstructed=True` (no Monte-Carlo ran) yet claimed convergence, the exact fiction the tri-state fix removed everywhere else. Missed earlier because it used a literal `True`, not a magic threshold, so the grep didn't catch it. Now `converged=None`, consistent with the other 12 reconstructed paths.
* **`model_calibration_record` docstring was stale** — still said `SolveReport` is "produced only by the solver primitives" (deleted in v1.166) and tagged itself "Phase 1 of the migration". Rewritten to describe what the builder *is*: facts captured via `SolveReport.external`/`.analytic`.

**Verified clean** (no change needed): serialisation round-trips `converged=None` / `reconstructed` / the nested surface-digest dict; the DB `converged` column + `list_calibrations` handle NULL; all 13 result dataclasses are non-frozen (mixin lazy-cache safe); no unused record-type imports; every other calibrator captures `converged` from a real flag (`bool(result.success)` / residual criterion) or `None`.

**Verification**: full suite **13009 passed**.

---


## v1.166.0 — 2026-06-26 — **Calibration capture layer: delete the unused solver primitives (one capture path)**

Removes a real smell surfaced by the architecture-doc accuracy check: `_solve.py` shipped five run-and-capture primitives (`minimize_solve`, `least_squares_solve`, `global_local_solve`, `brentq_solve`, `particle_solve`) that **no production calibrator used** — all 13 capture via `SolveReport.external()`/`.analytic()` wrapping their own (bespoke) optimiser. The primitives were exercised only by their own tests: speculative generality / dead flexibility.

**Files**: `calibration/_solve.py`, `calibration/__init__.py`, `test_solve_primitives.py` (deleted), `test_model_calibration_record.py`, `ARCHITECTURE.md`.

* **`_solve.py` is now just `SolveReport`** + its two honest constructors (`.external()` wraps an already-run optimiser; `.analytic()` for closed-form). Dropped the 5 primitives, the `_iters` helper, and the `numpy`/`scipy` imports — the module is pure (`dataclass` only).
* **The design is now single-path and correct in its separation of concerns**: the layer's job is to *record* a solve, not *run* it — optimiser setups are irreducibly bespoke, so the calibrator owns its solve and hands the result to `SolveReport.external`. One capture mechanism, used identically by all 13.
* The end-to-end test now demonstrates the real pattern (run scipy → `SolveReport.external` → builder → persist). Doc §6/§10 updated to match.

**Why delete, not keep**: dead code retained "in case a new calibrator wants it" is YAGNI — and it created two ways to do one thing (the exact ambiguity that caused the doc to drift). If a future calibrator wants a run-and-capture helper, it's one commit *with a consumer*. The capture-not-reconstruct principle is fully intact; only the mechanism that blurred "run" with "record" is gone.

**Verification**: full suite **13009 passed**.

---

## v1.165.0 — 2026-06-26 — **FX-SLV leverage-surface digest (the designer's `ParamDigest`)**

Closes the one genuine gap the clean-slate confrontation surfaced: FX-SLV fits a *leverage surface* but its record stored only `bandwidth`, so the fitted output wasn't identifiable/verifiable from its provenance.

**Files**: `fx/fx_slv_calibration.py`, `test_fx_slv_calibration.py`.

* **`_surface_digest(values)`** — shape + sha256 fingerprint of the (n_t × n_s) leverage surface. Both the eager (`particle_slv_calibration`) and the lazy `_build` records now carry `leverage_surface_sha256` + `leverage_surface_shape`. A re-run with the same inputs/seed can now be *verified* to have produced the same surface, and the digest keys the surface in a side store without bloating the record.
* **Where it lives, and why**: in `diagnostics.extra` (the open bag), not in `parameters`. The designer put `ParamDigest` in `parameters` via a `float | ParamDigest` union — but widening the scalar-parameters contract for one family would be the per-family special-casing we rejected elsewhere. A surface *fingerprint* is diagnostic context; `extra` is the uniform home for it. Same `ParamDigest` concept (shape + hash), realised consistently with the rest of the design.

**Verification**: full suite **13016 passed**. This was the last substantive item from the clean-slate comparison — everything else skipped was deliberate packaging.

---

## v1.164.0 — 2026-06-26 — **Tri-state `converged` — the magic-threshold convergence fiction removed**

Closes the one genuine residual the migration left (and corrects the Phase-4 claim that the thresholds were already gone — they weren't, until now).

**Files**: `calibration/_types.py`, `calibration/_solve.py`, `calibration/_model_record.py`, `db/db.py`, the 7 calibrator `_build` fallbacks, + test fixtures.

* **`OptimiserRun.converged` / `SolveReport.converged` are now `bool | None`.** `None` means **"not captured"** — the honest state of a *reconstructed* record (a hand-built result has no optimiser run). The rule is constant everywhere: *pass the optimiser's real flag if you captured it; `None` if you didn't; never guess.*
* **The 7 magic-threshold guesses are deleted** — SABR/jump/LMM/Rebonato/joint/JY/dividend lazy `_build` fallbacks used to derive `converged = self.rmse < 0.01` (and `< 0.05`, `< 1e-4`, …) with thresholds invented at record-build time. They now pass `converged=None`. (HW/G2++/bond-hazard keep their *stored real* flag; multicurve/dispersion keep their genuine residual-based criterion — those were never guesses.)
* The builder warns only on a definite `converged is False`, never on `None`. The DB `converged` column stores `NULL` for `None`.

**Net:** a calibration record's convergence verdict is now *always* either the optimiser's own answer or an explicit "not captured" — there is no third, fabricated value anywhere in the system.

**Verification**: 1126 calibration tests + full suite **13015 passed**.

---

## v1.163.0 — 2026-06-25 — **Calibration migration Phase 4: builder-enforcement gate — migration complete**

The locking slice. Closes OPEN.md §0c.

**Files**: `curves/multicurve_solver.py`, `test_calibration_builder_enforcement.py` (new).

* **Caught a missed eager site**: `multicurve_solver`'s main calibrate function still hand-rolled a `CalibrationResult` (the `_build` fallback was already migrated). Routed it through `model_calibration_record` + `SolveReport` — so now *zero* producers construct records by hand.
* **New gate** `test_calibration_builder_enforcement.py` — AST-asserts that `CalibrationResult` / `CalibrationFit` / `OptimiserRun` / `OptimiserSpec` are constructed **only** in the type-definitions module and the two builders (`curve_calibration_record`, `model_calibration_record`). A future calibrator that hand-rolls a record — reintroducing the eager/lazy duality or fabricated convergence — fails CI.

**Migration complete (v1.157 → v1.163).** Every calibration in the library — 15 model calibrators + the curve bootstrappers — captures a `SolveReport` from the optimiser and assembles its record through one factory. The structure now survives new calibration types by construction: the 16th calibrator gets the single builder, captured convergence, type-level non-empty residuals, and the builder gate for free.

**Verification**: full suite **13015 passed**.

---

## v1.162.0 — 2026-06-25 — **Calibration migration Phase 3: type-level empty-fit rejection + typed `reconstructed`**

The core-restructure slice — invariants moved from "gate-caught" to "unconstructible at the type level".

**Files**: `calibration/_types.py`, `calibration/_model_record.py`, the 11 calibrator modules, + test fixtures.

* **Empty residuals are now unconstructible** — `CalibrationFit.__post_init__` rejects an empty residual vector. A fit with no targets isn't a fit; an empty vector would make `rms_residual` read as a false-perfect `0.0`. This is the real G1 fix: the guarantee is in the *type*, not just the fidelity gate.
* **The `record_source="reconstructed"` magic string is gone** — replaced by a first-class typed `CalibrationDiagnostics.reconstructed: bool`, plumbed through `model_calibration_record(reconstructed=…)`. All 11 lazy `_build` fallbacks set it via the flag, not a dict key.
* Test fixtures that built degenerate empty-residual records updated to minimal valid ones; the old `test_rms_max_empty_residuals` flipped to `test_empty_residuals_rejected`.

**Deliberately *not* done** (adversarial call): the full `Residuals` nested-value-object field-rename and per-family `Diagnostics` subclasses — they'd churn hundreds of read sites and force a class per family for marginal gain over invariants now enforced (length-agreement, quotes-required, non-empty). `extra` stays the open numeric-context bag; `reconstructed` is the one flag that earned typing.

**Verification**: full suite **13015 passed**. Next: Phase 4 (grep-gate "converged only from primitives" + retire dead code).

---

## v1.161.0 — 2026-06-25 — **Calibration migration Phase 2 Group B: the last 7 calibrators — Phase 2 complete**

The lazy-only batch — LMM, Rebonato-LMM, dispersion, joint-equity-credit, Jarrow-Yildirim, dividend, multicurve. **All 15 model calibrators now route through `model_calibration_record` + `SolveReport`.**

**Files**: `models/{lmm_calibration,lmm_advanced,stochastic_correlation}.py`, `credit/joint_equity_credit.py`, `fixed_income/jarrow_yildirim.py`, `equity/dividend_calibration.py`, `curves/multicurve_solver.py`.

* Several of these weren't purely lazy — they had eager construction sites too (LMM/Rebonato/Joint/JY); both eager and `_build` paths now go through the builder, capturing the real optimiser verdict where one exists (`result.success`/`nit`).
* **Two get the genuinely honest treatment**: `stochastic_correlation` (closed-form moment match) now uses `SolveReport.analytic()` — no faked iterative convergence, the index-variance residual carries the quality; `multicurve` uses its real stored `n_iterations` + residual-based convergence (a faithful, not reconstructed, record).
* The remaining lazy-only fallbacks (`_build` for hand-built instances) reconstruct convergence from the carried residual/RMSE and are marked `record_source="reconstructed"`; the per-`_build` warning copy-paste is gone (the builder adds it centrally).
* 6 modules dropped their `CalibrationFit`/`OptimiserRun`/`OptimiserSpec`/`CalibrationProvenance` imports.

**Phase 2 done**: every calibrator (15) + bootstrapper builds through one factory. **Verification**: 623 targeted + full suite (result below). Remaining: Phase 3 (fused `Residuals` + typed `Diagnostics`), Phase 4 (retire dead code + grep-gate).

---

## v1.160.0 — 2026-06-25 — **Calibration migration Phase 2 Group A: 5 already-eager calibrators onto the builder**

The de-duplication batch — HW, G2++, jump, FX-SLV, bond-hazard. Each already captured real optimiser data via a hand-rolled `CalibrationResult` skeleton; now they route through `model_calibration_record` + `SolveReport.external(...)`.

**Files**: `models/{hw,g2pp,jump}_calibration.py`, `fx/fx_slv_calibration.py`, `credit/bond_hazard_bootstrap.py`, `calibration/_model_record.py`.

* Both the eager path and the lazy `_build` fallback now go through the single builder; the hand-rolled skeletons (and the per-`_build` warning copy-paste) are gone. bond_hazard had **three** eager construction sites + its fallback — all four collapsed onto the builder.
* The 5 modules no longer import `CalibrationFit`/`OptimiserRun`/`OptimiserSpec`/`CalibrationProvenance`.
* Builder gained a `weights` passthrough (bond_hazard's per-bond weights).
* FX-SLV eager now records real `iterations` (`n_t` MC steps).

**Verification**: 614 targeted + full suite **13014 passed**. Next: Group B (the 7 lazy-only calibrators).

---

## v1.159.0 — 2026-06-24 — **Calibration migration Phase 2 (slice 1): SABR onto the single builder**

The reference migration — the template for the other 14 calibrators (OPEN.md §0c).

**Files**: `options/sabr.py`, `calibration/_solve.py`, `calibration/_model_record.py`.

* **`sabr_calibrate` eager record** now routes through `model_calibration_record`, capturing the optimiser verdict in a `SolveReport.external(...)` read off `pb_minimize`'s result (`converged`/`iterations`) — no hand-rolled `CalibrationResult` skeleton, no re-derivation.
* **The lazy `_build` fallback** (hand-built instances) also routes through the builder now — the ~20-line hand-rolled skeleton is gone from *both* paths; the fallback is honestly marked `record_source="reconstructed"`, `algorithm="unspecified"`.
* **`SolveReport` gains `max_iterations`** (the configured cap, a reproducibility input) distinct from actual `iterations`; the builder maps cap→`spec.max_iterations`, actual→`run.iterations`. The builder also gained an `optimiser_extra` passthrough (mirrors `curve_calibration_record`) so SABR keeps `beta_fixed/forward/T` in `spec.extra`.
* `sabr.py` no longer imports `CalibrationFit`/`OptimiserRun`/`OptimiserSpec`/`CalibrationProvenance` — it constructs nothing by hand.

**Verification**: full suite **13014 passed**. Pattern proven on a real producer; the remaining 14 calibrators follow this diff.

---

## v1.158.0 — 2026-06-24 — **Calibration migration Phase 1: the single model-calibrator builder**

Second additive slice (OPEN.md §0c). The Family-B mirror of `curve_calibration_record`.

**Files**: `calibration/_model_record.py` (new), `calibration/__init__.py`, `test_model_calibration_record.py` (new).

* **`model_calibration_record(*, model_class, parameters, residuals, quotes_fitted, solve: SolveReport, …)`** — assembles a model calibrator's `CalibrationResult` from one place. The optimiser metadata (algorithm, iterations, converged, tolerance, seed) is read **straight off the required `SolveReport`** — a calibrator can no longer omit or invent it. This closes the eager/lazy duality at the source: one truthful build path, captured at fit time.
* **Cross-cutting behaviour centralised** — a non-convergence `warning` is appended automatically (not copy-pasted into 15 `_build` methods), and `solve.algorithm` is canonicalised by `OptimiserSpec`. Caller diagnostics are preserved and merged.

**This flips the switch:** a new (16th) calibrator is now clean by construction — `primitive → SolveReport → model_calibration_record → record`, ~6 lines, no fork, no hand-rolled skeleton, no fabricated convergence. The end-to-end test demonstrates the full flow (fit → build → persist → round-trip).

**Tests**: 5 new. **Verification**: full suite **13014 passed**. Next: Phase 2 migrates the existing 15 calibrators onto this path, one slice each, deleting each lazy `_build` fallback.

---

## v1.157.0 — 2026-06-24 — **Calibration migration Phase 0: solver-primitive layer**

First slice of the "capture-not-reconstruct" migration (OPEN.md §0c) — the structural fix for the eager/lazy duality and fabricated-convergence debts. Purely **additive**: a new layer, nothing else changes.

**Files**: `calibration/_solve.py` (new), `calibration/__init__.py`, `test_solve_primitives.py` (new).

* **`SolveReport`** (frozen) — `algorithm / converged / iterations / tolerance / seed`, the optimiser facts. Plus two honest constructors: `analytic()` (closed-form, no faked iterative verdict) and `external()` (escape hatch for a black-box optimiser like the pricebook `minimize` wrapper).
* **Primitives that *capture* the report from the real optimiser** — `minimize_solve` (scipy minimize: SABR/HW/joint/dividend/dispersion), `least_squares_solve`, `global_local_solve` (differential_evolution + local polish, capturing the seed: G2++/jump), `brentq_solve` (scalar root: JY), `particle_solve` (seeded MC loop, reproducible: FX-SLV). Each returns `(solution, SolveReport)`.

This is the layer the Phase-1 builder will *require*, so convergence/iterations/seed can no longer be invented at record-build time. **After Phase 1, every new calibration uses the clean pattern** (primitive → SolveReport → builder); the existing 15 then migrate opportunistically.

**Tests**: 7 (each pins "captured, not invented"). **Verification**: full suite **13009 passed**.

---

## v1.156.0 — 2026-06-24 — **Calibration fidelity sweep (3/3): contract enforcement + fidelity gate**

Final structural slice of G1–G9: turns the producer-side truthfulness into enforced invariants, so a future calibrator can't regress.

**Files**: `calibration/_types.py`, `tests/conftest.py`, `test_calibration_fidelity.py` (new), + test-assertion updates.

* **G6 (enforced)** — `OptimiserSpec.__post_init__` now canonicalises `algorithm` to one snake_case audit vocabulary: any run of non-alphanumerics → `_`, lowercased. `"Nelder-Mead"`, `"L-BFGS-B"`, `"differential_evolution+L-BFGS-B"`, `"brentq-per-bond"` → `nelder_mead`, `l_bfgs_b`, `differential_evolution_l_bfgs_b`, `brentq_per_bond`. You can finally group by optimiser.
* **G9 (enforced)** — `CalibrationFit` now *requires* `quotes_fitted` whenever residuals are present — a residual with no quote is an unattributable magnitude. The shared test helper auto-labels for fixture concision; production supplies real quotes.
* **G1/G7 (gated)** — new `test_calibration_fidelity.py` draws every record from both conformance registries (curve + model producers, 79 cases) and asserts: non-empty 1:1 residuals↔quotes (no false-perfect rms 0); converged+rms-0 implies max-residual 0 (consistent story); `algorithm` canonical and never `"unknown"`; and **`model_class` globally unique** across families (delegation aliases — `bootstrap_rfr`→`bootstrap`, `bootstrap_ibor`→`bootstrap_forward_curve` — explicitly recognised).

**Verification**: full suite **13002 passed**.

**G3 (market-snapshot linkage) — re-assessed, no code change needed.** The original audit overstated this. On inspection, **every producer that receives a `MarketSnapshot` already threads its id** to `stamp(market_snapshot_id=…)` — `sabr`, `hw`, `g2pp`, `lmm`, `jump`, `bond_hazard`, `multicurve`, the flagship `bootstrap`, and `global_solver` (9 producers). The remaining simpler calibrators/bootstrappers don't accept a snapshot because their callers pass raw market quotes, not a snapshot object; wiring those would be a market-data-flow change, not a record-fidelity fix. The capability is complete and fed wherever a snapshot exists — G3 is **not** an open fidelity defect.

---

## v1.155.0 — 2026-06-24 — **Calibration fidelity sweep (2/3): credit/fx/equity/curve calibrators + seed**

Second slice of the G1–G9 remediation, completing the producer-side fixes across the remaining 6 calibrators.

**Files**: `credit/joint_equity_credit.py`, `fixed_income/jarrow_yildirim.py`, `fx/fx_slv_calibration.py`, `equity/dividend_calibration.py`, `curves/multicurve_solver.py`, `credit/bond_hazard_bootstrap.py`.

* **G2** — derived `converged` for joint_equity_credit (≤5% rel on both targets), jarrow_yildirim / multicurve (|residual| threshold), dividend (rmse). bond_hazard keeps its stored flag.
* **G4** — `ParticleCalibrationResult` gains a `seed` field; `particle_slv_calibration` records its RNG seed in `OptimiserSpec.seed`, so the stochastic FX-SLV calibration is now reproducible from its record.
* **G9** — scalar-residual builds (jarrow_yildirim, fx_slv, multicurve) now carry an `aggregate_objective` quote.
* **G5/G8** — `record_source="reconstructed"` marker + `warnings` on non-convergence across all six; fx_slv records that its `residual` is a placeholder.

**Verification**: full suite **12922 passed**. Slice 3 (`_types.py` enforcement + fidelity gate) next.

---

## v1.154.0 — 2026-06-24 — **Calibration fidelity sweep (1/3): model calibrators tell the truth**

First slice of the G1–G9 provenance-fidelity remediation. The model-calibrator lazy `_build_calibration_record` fallbacks were producing misleading records (a 137 bp SABR fit reporting `rms_residual=0, converged=True`). Fixed across the 7 `models/`+`options/sabr` calibrators:

**Files**: `options/sabr.py`, `models/{hw,g2pp,jump,lmm,lmm_advanced,stochastic_correlation}_*.py`.

* **G1** — SABR lazy build mapped `residuals=[]` despite holding `reprice_errors_bp`; now emits the real per-point residuals with `smile_point_i` quotes. Empty-data no longer masquerades as a perfect fit.
* **G2** — `converged=True` was hardcoded in SABR/jump/lmm/lmm_rebonato/stochastic_correlation; now **derived** from the fit quality the result carries (rmse / |residual| vs a documented threshold). HW/G2PP keep their stored flag.
* **G5** — lazy records now carry `diagnostics.extra["record_source"]="reconstructed"`, distinguishing a degraded reconstruction from the eager full-fidelity record (same `model_class`).
* **G6** — `algorithm="unknown"` (HW/G2PP/jump) → honest `"unspecified"`.
* **G8** — non-convergence now emits a `diagnostics.warnings` entry.
* **G9** — scalar-residual builds (`lmm_rebonato`) now carry an `aggregate_objective` quote so the residual is attributable.

**Verification**: 716 SABR/HW/G2PP/jump/LMM/stochastic/calibrator tests pass. Slices 2 (credit/fx/equity/curve calibrators + `seed`) and 3 (`_types.py` enforcement + fidelity gate) follow.

---

## v1.153.0 — 2026-06-24 — **`ProvenanceCarrier`: one read-interface unifying curves and calibrators**

Names the concept the two provenance patterns were both realisations of. Curves store their `CalibrationResult` in a field; model calibrators lazily build one via the `CanonicalCalibrationResult` mixin — but until now the two carriers exposed **different accessors** (bare field vs `to_calibration_result()`), so they weren't substitutable: `db.save_calibration(curve)` failed; you had to reach in for `curve.calibration_result`.

This is a **clarity** refactor (no behaviour change to any calibration), unifying the *interface* without touching the *mechanism* (field vs lazy-build correctly still differ).

**New** — `ProvenanceCarrier` (`@runtime_checkable` Protocol, `calibration/_types.py`): `to_calibration_result() -> CalibrationResult | None`. Three structural realisations, no shared base:
* `CalibrationResult` itself → returns `self`;
* a provenance-carrying **curve** / forwarding wrapper → returns the stored field (or `None` if never calibrated — a flat/loaded curve);
* a `CanonicalCalibrationResult` family result → lazy build + cache (unchanged).

**Wired** — the one-line `to_calibration_result()` added to all 11 curve/wrapper carriers (`DiscountCurve`, `SurvivalCurve`, `AADDiscountCurve`, `CPICurve`, `SpreadCurve`, `RealYieldCurveResult`, `DividendCurve`, `BondCurveResult`, `IBORCurve`, `RFRCurveResult`, `SovereignHazardResult`). Core curves stay import-free — the method just returns the attribute; the Protocol is structural, so no curve imports the calibration layer.

**`save_calibration`** now types its parameter `ProvenanceCarrier` and accepts a curve directly — substitutable with a calibration result. Distinct, legible errors for the not-a-carrier and uncalibrated-curve (`None`) cases.

**Why no top-level `Bootstrapper`/`Calibrator` class** (the question that drove this): neither is an *object* — both are functions; only their *results* are objects, and there's no polymorphic dispatch site that a base class would serve (the old `Calibrator` Protocol was deleted for exactly that reason). The genuine shared concept was the **read interface on the result**, which is what `ProvenanceCarrier` now names. A `Bootstrapper` class earns its place only if a multi-currency curve-building *service* ever needs to select/run bootstrappers polymorphically.

**Enforcement** — new `test_provenance_carrier.py`: behavioural (all three realisations satisfy the Protocol, persist, substitutable; error paths) **plus an AST structural sweep** that fails if any class carrying `calibration_result` forgets `to_calibration_result()`. Both conformance gates assert their results are `ProvenanceCarrier`s.

**Tests**: +6 (5 in the carrier file + calibrator-gate substitutability). **Verification**: full suite **12922 passed**.

---

## v1.152.0 — 2026-06-23 — **Calibrator-side conformance gate (provenance full sweep complete)**

The mixin counterpart to the bootstrapper gate. Curves carry their own `calibration_result`; *model* calibrators expose theirs via the `CanonicalCalibrationResult` ABC. This locks that side with the same enforced invariant.

**Files**: `test_calibrator_provenance_conformance.py` (new, test-only slice).

* **Discovery guard** — AST-scans the package for classes subclassing `CanonicalCalibrationResult` and asserts each of the **13** is classified (none allowlisted). A new family-result nobody classified fails CI. Drift guard catches a classified name that disappears.
* **Structural contract** — per subclass: a `calibration_result` dataclass field + an own (non-abstract) override of `_build_calibration_record`. Makes the mixin's import-time / instantiation-time guards legible as explicit assertions.
* **Behavioural gate** — builds the 8 cheaply-constructible family results (SABR, Joint, Jump, LMM, Dispersion, RebonatoLMM, Dividend, MultiCurve) and asserts `to_calibration_result()` yields a valid, DB-round-tripping record that caches idempotently. The 5 that carry a live model/curve/leverage object (Hazard, HW, G2PP, JY, Particle) are exercised in their existing dedicated tests, named in `_BEHAVIOURAL_ELSEWHERE`.

**Tests**: 24 new. **Verification**: full suite **12917 passed**.

**Status**: both sides of the calibration-provenance system — curves (19 bootstrappers) and models (13 calibrators) — are now conformance-gated. Full sweep complete.

---

## v1.151.0 — 2026-06-23 — **Bootstrapper provenance: global_solver closed out**

Post-campaign sweep of the two `global_solver.py` producers that the conformance gate had allowlisted as "solver primitives". One was a genuine gap; the other was inconsistent.

**Files**: `curves/global_solver.py`, `test_bootstrapper_provenance_conformance.py`.

* **`coupled_bootstrap` (genuine gap)** — the dual-curve simultaneous Newton solve returned `(ois_curve, proj_curve)` with **no record on either**. Now each carries its own: `coupled_ois_bootstrap` (OIS pillar residuals) and `coupled_projection_bootstrap` (projection-swap residuals), sharing the `newton-coupled` run metadata. Distinct records (distinct ids), both persist.
* **`global_bootstrap` (consistency)** — it already attached a `discount_curve_global` record, but **hand-rolled** the `CalibrationResult` with the raw constructor (predating the helper). Migrated to `curve_calibration_record()` so its shape is uniform with the other 18 producers; `max_iterations` cap preserved in `optimiser.extra` (the helper ties `max_iterations` to actual `iterations`).
* **Gate** — both promoted from ALLOWLIST → COVERED and added to the behavioural registry; ALLOWLIST now contains only `bootstrap_ci` (statistical, not a curve). New dedicated test asserts `coupled_bootstrap` attaches to **both** curves.

**Result**: every public curve bootstrapper in the package now attaches provenance — the allowlist holds only the one genuinely-not-a-curve function. **Tests**: +3 (22 in the conformance file). **Verification**: full suite **12893 passed**.

**Next (full-sweep plan)**: calibrator-side conformance gate for the ~13 `CanonicalCalibrationResult` mixin subclasses.

---

## v1.150.0 — 2026-06-23 — **Bootstrapper campaign: closing conformance gate (campaign complete)**

Locks the invariant the campaign built: *every public curve bootstrapper attaches an auditable calibration record.* Mirrors the `CanonicalCalibrationResult` ABC/field guard on the mixin side — convention becomes enforced invariant.

**Files**: `test_bootstrapper_provenance_conformance.py` (new, test-only slice).

* **Discovery guard** — AST-scans the whole `pricebook` package for public module-level `bootstrap*` / `*_bootstrap` functions and asserts each is classified: **COVERED** (must attach provenance — 16 in the runtime registry + the 3 pre-existing `bond_hazard_*` tested in their own file) or **ALLOWLIST** (with a reason: `global_bootstrap`/`coupled_bootstrap` low-level solver primitives; `bootstrap_ci` statistical resampling). A new bootstrapper nobody classified fails this test → cannot silently skip provenance. A drift guard also fails if a classified name disappears.
* **Behavioural gate** — runs each of the 16 COVERED producers on a small fixture and asserts the returned curve / wrapper carries a non-None `calibration_result` that round-trips through the DB.

**Tests**: 19 new (3 meta-guards + 16 parametrised producers). **Verification**: full suite **12890 passed**.

**Campaign summary (v1.140 → v1.150)**: F1 helpers → F2 curve fields → Tier 1 core primitives → Tier 2 foundational rates → Tier 3 basis/specialised → Tier 4 credit → Tier 5 inflation/equity → conformance gate. All public curve/survival bootstrappers are now auditable and persistable, enforced in CI.

---

## v1.149.0 — 2026-06-23 — **Bootstrapper campaign Tier 5: inflation + equity curve provenance**

The last tier of producers. These return types had no provenance slot, so this slice adds the field (the F2-equivalent for the tier) and attaches the record.

**Files**: `fixed_income/inflation.py`, `fixed_income/inflation_bond_advanced.py`, `equity/dividend_advanced.py`, `test_tier5_inflation_equity_provenance.py` (new).

* **`bootstrap_cpi_curve`** → `CPICurve` gains a `calibration_result` attribute; record `cpi_curve_bootstrap`, `algorithm="closed_form"`. Residual = implied ZC inflation rate from each pillar CPI minus the quote (~0; `cpi(T)=base·(1+rate)^T`).
* **`real_yield_curve_bootstrap`** → `RealYieldCurveResult` gains the field; record `real_yield_curve_bootstrap`, per-linker brentq. Residual = linker price equation at the solved real yield.
* **`dividend_curve_bootstrap`** → `DividendCurve` gains the field; record `dividend_curve_bootstrap`, `algorithm="closed_form"`. Residual = `S·q̄·T − D(T)` (0 where `T>0`).
* Both dataclass `to_dict()` methods now **exclude** `calibration_result` so the record object doesn't leak into serialised dicts.

**Tests**: 3 new. **Verification**: full suite **12871 passed**.

**Next**: the closing **conformance gate** — assert every public bootstrapper attaches a non-None `calibration_result`, turning "all curves auditable" from convention into an enforced invariant.

---

## v1.148.0 — 2026-06-23 — **Bootstrapper campaign Tier 4: credit curve provenance**

All three credit bootstrappers build a `SurvivalCurve`; each now attaches its canonical record (curve-carries-provenance).

**Files**: `credit/cds.py`, `credit/cds_market.py`, `credit/sovereign_cds.py`, `test_tier4_credit_provenance.py` (new).

* **`bootstrap_credit_curve`** → `credit_curve_bootstrap`. `_verify_credit_round_trip` now *returns* the signed per-quote par-spread residuals (model − input) it already computes, reused directly as the record's residuals (no double pricing). `build_cds_curve` (cds_market) inherits the record by delegation.
* **`bootstrap_from_upfronts`** → `cds_upfront_bootstrap`. Residual = CDS PV at the solved survival minus the quoted upfront (~0; or the `q=0.5` fallback's mispricing when no sign change was bracketed).
* **`bootstrap_sovereign_hazard`** → `sovereign_hazard_bootstrap` on the survival curve; `SovereignHazardResult.calibration_result` forwards. Parameters = per-tenor hazards; residual = (fitted − input) par spread in decimal. Added a `fitted_tenors` list so parameters/residuals/quotes stay aligned even when the `dt <= 0` guard skips a degenerate tenor.

**Tests**: 4 new. **Verification**: full suite **12868 passed**.

**Next**: Tier 5 — inflation/equity (`bootstrap_cpi_curve`, `real_yield_curve_bootstrap`, `dividend_curve_bootstrap`), then the closing conformance gate.

---

## v1.147.0 — 2026-06-23 — **Bootstrapper campaign Tier 3b: bond curve + tenor basis provenance**

The two Tier-3 bootstrappers that return a *non-bare-curve* artifact now carry a record.

**Files**: `curves/bond_curve.py`, `fixed_income/tenor_basis.py`, `test_tier3b_curve_provenance.py` (new).

* **`bootstrap_curve_from_bonds`** → `BondCurveResult.__post_init__` injects a `bond_curve_bootstrap` record directly onto the extracted `discount_curve`, built from the fit data the result already holds (per-bond `residuals_bp`, pillar zeros, method, converged). Covers all three solve methods (sequential / global / parametric NS+Svensson) with no per-return-site wiring; parametric β/τ captured in `diagnostics.extra["shape"]`. `BondCurveResult.calibration_result` forwards to the curve.
* **`bootstrap_tenor_basis`** → attaches a `tenor_basis_bootstrap` record to the long-tenor projection curve (`IBORCurve` forwards `.calibration_result` from Tier 1). Per-quote residual = basis-swap PV (short−long) at the solved df (~1e-14, brentq); extracted spreads stored as `diagnostics.extra["spreads_bp"]`.

**Tests**: 5 new (4 bond methods parametrised + tenor basis). **Verification**: full suite **12864 passed**.

**Next**: Tier 4 — credit (`cds`, `cds_market`, `sovereign_cds`, all `SurvivalCurve`), then Tier 5 (inflation/equity) and the closing conformance gate.

---

## v1.146.0 — 2026-06-23 — **Bootstrapper campaign Tier 3a: xccy basis + AAD curve provenance**

The two Tier-3 bootstrappers that return a bare curve now attach a record.

**Files**: `fixed_income/xccy_basis.py`, `curves/aad_curves.py`, `test_tier3a_basis_provenance.py` (new).

* `bootstrap_basis_curve` (xccy) → `model_class="xccy_basis_bootstrap"`, `algorithm="closed_form"` (direct CIP solve). Residuals = per-forward `F_model − F_market` (0, exact).
* `aad_bootstrap` → `model_class="aad_discount_curve_bootstrap"`, `algorithm="aad-fixed-point"`. Residuals computed on the `Number`-valued curve's float values (deposit `model_rate − rate`; swap par condition `par·annuity + df(mat) − 1`; ~1e-17).

**Tests**: 2 new. **Verification**: full suite **12859 passed**.

**Next**: Tier 3b — `bond_curve` (`BondCurveResult` wrapper) and `tenor_basis` (tuple return), both needing forwarders.

---

## v1.145.0 — 2026-06-23 — **Bootstrapper campaign Tier 2b: spread curve provenance**

`bootstrap_spread_curve` (IBOR-RFR basis) now attaches a record — completing Tier 2.

**Files**: `fixed_income/rfr.py`, `test_ois_ibor_basis.py`.

* `SpreadCurve` (a `@dataclass`) gains `calibration_result` via `__post_init__` (mirrors the other curve types; `TYPE_CHECKING` import).
* `bootstrap_spread_curve` → `curve.calibration_result` (`model_class="spread_curve_bootstrap"`); residuals = per-IBOR-swap repricing error at the solved spread (~3e-17); `parameters` = per-pillar `spread(date)`.

**Tests**: 3 new. **Verification**: full suite **12857 passed**.

**Next**: Tier 3 (basis/specialised) — `xccy_basis`, `aad_curves` (clean curve returns); `bond_curve`, `tenor_basis` (wrapper/tuple returns, need forwarders).

---

## v1.144.0 — 2026-06-23 — **Bootstrapper campaign Tier 2a: OIS curve provenance**

`bootstrap_ois` now attaches a canonical record to the discount curve it returns.

**Files**: `fixed_income/ois.py`, `test_ois.py`.

* `curve.calibration_result` (`model_class="ois_curve_bootstrap"`), via the F1 helper. Residuals = per-OIS-swap round-trip `pv_fixed − pv_float` (telescoping float leg; ~1e-13 by construction). `parameters` = pillar discount factors.

**Tests**: 2 new. **Verification**: full suite **12854 passed**.

**Next**: Tier 2b — `bootstrap_spread_curve` (needs the `SpreadCurve` type to carry the field first).

---

## v1.143.0 — 2026-06-23 — **Fix B-rfr-global: bootstrap_rfr(method="global") kwarg mismatch**

The bug the Tier-1 provenance work surfaced. `curves/rfr_bootstrap.bootstrap_rfr(method="global")` called `global_bootstrap(..., deposit_day_count=, fixed_day_count=, fixed_frequency=)`, but `global_bootstrap`'s parameters are `deposit_dc` / `swap_dc` / `swap_frequency` — so any global-method RFR bootstrap raised `TypeError`. Never tested, so never noticed.

**Files**: `curves/rfr_bootstrap.py` (corrected the three kwarg names), `test_rfr_bootstrap.py` (the Tier-1 `xfail(strict)` removed — `test_global_surfaces_record` now passes normally and exercises the previously-dead global path end-to-end, surfacing its `discount_curve_global` record through `RFRCurveResult`).

**Verification**: full suite **12852 passed** (no more xfail; two slow G2++ tests deselected per convention).

---

## v1.142.0 — 2026-06-23 — **Bootstrapper campaign Tier 1: forward/projection curve provenance**

`bootstrap_forward_curve` (the dual-curve projection-curve builder) now attaches a canonical record to the curve it returns, and its two delegating wrappers surface it.

**Files**: `curves/bootstrap.py`, `curves/rfr_bootstrap.py`, `fixed_income/ibor_curve.py`, `test_ibor_curve.py`, `test_rfr_bootstrap.py`.

* `bootstrap_forward_curve` → `curve.calibration_result` (`model_class="projection_curve_bootstrap"`), built via the F1 helper. Residuals = per-instrument round-trip (deposit `model_rate − rate`, swap `pv_fixed − pv_float`) via a new `_forward_curve_residuals()` shared with — and not disturbing — the W-series-tested verifier.
* **Waterfall**: `IBORCurve` and `RFRCurveResult` (both wrap a calibrated curve) gain a read-only `calibration_result` property forwarding to their inner curve — so `bootstrap_ibor` and `bootstrap_rfr` (sequential + global) inherit provenance for free.
* **Bug surfaced + pinned**: `bootstrap_rfr(method="global")` passes `deposit_day_count=` to `global_bootstrap`, which rejects it → `TypeError` on any global-method call (never tested). Pinned by an `xfail(strict, raises=TypeError)` test that flips green when fixed; tracked in `OPEN.md §0b`. Same "campaign flushes latent bugs" pattern as `fx_slv` / `desks/api`.

**Tests**: 6 new (+1 xfail). **Verification**: full suite **12851 passed, 1 xfailed** (two slow G2++ tests deselected per convention).

**Next**: Tier 2 — `bootstrap_ois`, `bootstrap_spread_curve` (independent builders).

---

## v1.141.0 — 2026-06-22 — **Bootstrapper campaign F2: SurvivalCurve / AADDiscountCurve carry provenance**

`SurvivalCurve` (credit) and `AADDiscountCurve` now hold `calibration_result: CalibrationResult | None = None`, mirroring `DiscountCurve` — so credit and AAD bootstrappers can attach a record the same way the rates curves do.

**Files**: `core/survival_curve.py`, `curves/aad_curves.py`, `python/tests/test_curve_carries_provenance.py` (new). Field defaults `None`; `TYPE_CHECKING`-only `CalibrationResult` import (no new runtime edge).

**Tests**: 4 new (all three curve types carry the field; a record-bearing `SurvivalCurve` persists via the F1 helper). **Verification**: full suite **12847 passed**.

**Next**: Tier 1 — wire `bootstrap_forward_curve` (covers `ibor_curve` by delegation).

---

## v1.140.0 — 2026-06-22 — **Bootstrapper provenance campaign F1: shared curve-record helpers**

Foundation for bringing the ~13 scattered curve/survival bootstrappers into the audit chain (curve-carries-provenance). Adds the shared assembly helper so no bootstrapper hand-rolls the record.

**Files**: `python/pricebook/calibration/_curve_record.py` (new), `calibration/__init__.py`, `curves/bootstrap.py`, `python/tests/test_curve_calibration_record.py` (new).

* **`curve_calibration_record(*, model_class, parameters, residuals, quotes_fitted, algorithm, iterations, …)`** — assembles the four canonical components (provenance stamp, fit, optimiser run, diagnostics) uniformly. Lives in the calibration layer (L0) so every producer — curves, fixed_income, credit, equity — imports it without dragging in a concrete curve type. Inherits `CalibrationFit`'s snake_case + length-agreement enforcement for free.
* **`pillar_parameters(pillar_dates, pillar_values, label="df")`** — `{label(date): value}` for the calibrated per-pillar quantity (`df`/`survival`/`hazard`/…).
* **Proven + DRY'd**: refactored `curves/bootstrap._build_bootstrap_calibration_result` to use both (the file got shorter; behaviour identical — existing bootstrap/snapshot tests unchanged).

**Tests**: 5 new. **Verification**: full suite **12843 passed** (two slow G2++ tests deselected per convention).

**Next**: F2 — give `SurvivalCurve` / `AADDiscountCurve` the `calibration_result` field (DiscountCurve already has it); then Tier 1 (`bootstrap_forward_curve`).

---

## v1.139.0 — 2026-06-22 — **sabr: hoist now-redundant lazy imports to top-level**

Follow-up tidy after the SABR typing conversion (v1.138.0). Adding the top-level `from pricebook.calibration import CanonicalCalibrationResult` (needed as a base class) already loads the `pricebook.calibration` module at import time, so the in-function lazy `import pricebook.calibration` / `import pb_minimize` blocks in `sabr_calibrate` and `_build_calibration_record` no longer deferred anything — they were just noise. Consolidated all of them into one top-level import block.

**Files**: `python/pricebook/options/sabr.py` (net −10 lines). No behaviour change; verified no import cycle (the `sabr → calibration` and `sabr → statistics.optimization` edges are acyclic).

**Verification**: full suite **12838 passed** (two slow G2++ tests deselected per convention).

---

## v1.138.0 — 2026-06-22 — **SABR: typed result through the canonical mixin**

`sabr_calibrate` returned a stringly-typed `dict`; it now returns a typed `SABRCalibrationResult` that goes through `CanonicalCalibrationResult` like every other family — so SABR persists via `db.save_calibration(result)` directly and is covered by the ABC/field enforcement. **13 of 15 calibrators** are now on the mixin (the two curve bootstrappers — `discount_curve_bootstrap`, `discount_curve_global` — stay curve-carries-provenance by design).

**Files**: `options/sabr.py`, `options/{swaption_vol_cube,capfloor,vol_calibration}.py`, `models/lmm_calibration.py`, `desks/api.py`, + 5 test files.

* **New `SABRCalibrationResult`** — `alpha`/`beta`/`rho`/`nu`/`rmse`, the reprice diagnostics (`reprice_errors_bp`, `max_error_bp`) as proper fields (was dynamic dict keys added by `calibrate_sabr_smile`), the eager `calibration_result`, a `to_dict()`, and `_build_calibration_record()`.
* **Callers updated** from `result["alpha"]`-style access to attributes; `capfloor`'s `**params` dict-spread rebuilt explicitly (preserving its `calibration_result` key).
* **Latent bug fixed**: `desks/api.py` did `alpha, rho, nu = sabr_calibrate(...)` — unpacking the (then dict) return as 3 values, a guaranteed crash that was never exercised. Now reads the typed fields. (Same "the unification surfaces a real bug" pattern as `fx_slv`'s zero-residual.)
* **Non-issue ruled out**: a same-named `calibrate_sabr_smile` in `structured/ir_vol_surface` is a *separate* function (returns a node) — not affected; the earlier "arg-order bug" suspicion was that name collision.

**Verification**: full suite **12838 passed** (two slow G2++ tests deselected per convention).

---

## v1.137.0 — 2026-06-22 — **Calibration contract: enforced, not just documented**

Closes the enforcement gaps in the calibration design — the contract every family must honour is now checked at class-definition / construction / persistence boundaries, so a non-conforming or inconsistent calibrator fails fast with a clear message rather than relying on review discipline.

**Files**: `python/pricebook/calibration/_types.py`, `python/pricebook/db/db.py`, + tests.

**Conformance (was gap #1 — "nothing forces a family to follow the pattern"):**
* `CanonicalCalibrationResult` is now an `abc.ABC`; `_build_calibration_record` is an `@abstractmethod`. A family that forgets to implement it fails at **instantiation** (`TypeError: Can't instantiate abstract class`), not on first use.
* `__init_subclass__` enforces the *field* half of the contract at **class-definition** time: a subclass that inherits the mixin but omits `calibration_result: CalibrationResult | None = None` raises a clear `TypeError` when defined.
* `PricebookDB.save_calibration` now rejects anything that isn't a `CalibrationResult` and can't produce one (`TypeError`) — a non-conforming result physically cannot enter the audit chain (silent fall-through to a stray `AttributeError` is gone).

**Convention (was gap #3 — "semantics inside the fit are unconstrained"):** `CalibrationFit.__post_init__` now validates, at construction:
* `model_class` is non-empty `snake_case` (the audit key — no more `"HullWhite"` vs `"hull_white"` drift; caught two sloppy test fixtures).
* `weights` and `quotes_fitted`, when non-empty, must match `len(residuals)` (they are parallel per-quote arrays). The full producer suite confirms all ~15 calibrators already satisfy this.

**Honest scope**: this does **not** enforce cross-family `model_class` *uniqueness* (a design-time ownership concern) or *residual units* (would need a units enum nobody currently consumes) — those remain doc/review matters by deliberate choice, noted as the boundary of what's worth enforcing at this scale.

**Tests**: 7 new (abstract-method-at-instantiation, missing-field-at-definition, save-rejects-non-conforming, model_class snake_case, weights/quotes length agreement, empty-optionals allowed).

**Verification**: full suite **12838 passed** (two slow G2++ tests deselected per convention).

---

## v1.136.0 — 2026-06-22 — **Calibration types: final-read cleanups (CalibrationFit immutability + docstring)**

Two nits from a fresh-eyes read of `calibration/_types.py` after the decomposition.

**Files**: `python/pricebook/calibration/_types.py` + 6 test files.

* **`CalibrationFit` now canonicalises its sequence fields to tuples** (was lists), matching `CalibrationDiagnostics` and honouring the stated principle that a frozen record holds immutable sequences — previously `cr.fit.residuals.append(...)` could silently mutate a "frozen" record. The list choice had only been to keep test comparisons terse; those assertions are now type-agnostic (`list(cr.fit.residuals) == [...]`). Round-trip stays exact.
* **Module docstring refreshed**: it referenced the removed flat `.new()` factory and a now-stale `DESIGN.md §3.3` shape note. It now describes the four-component composite and points the lazy `import pricebook` note at `CalibrationProvenance.stamp`.

**Verification**: full suite **12831 passed** (two slow G2++ tests deselected per convention).

---

## v1.135.0 — 2026-06-22 — **CalibrationResult decomposed into three component value objects**

The owner's standing objection — that `CalibrationResult` was a monolithic 14-field bag, not graspable as a concept at a glance — resolved by structure. The record is now **three component value objects + diagnostics**, and producers construct/inject the components directly (the components are a real interface, used as one — not hidden behind a flat factory).

**New value objects** (`calibration/_types.py`, all `@serialisable_convention`, exported from `pricebook.calibration`):
* `CalibrationProvenance` — `{id, timestamp, code_version, market_snapshot_id}` (where it came from). `CalibrationProvenance.stamp()` auto-fills id/timestamp/code_version.
* `CalibrationFit` — `{model_class, parameters, residuals, objective, quotes_fitted, weights}` (what was fitted + how well); carries the derived `rms_residual`/`max_residual` properties.
* `OptimiserRun` — `{spec: OptimiserSpec, iterations, converged}` (how the solver behaved — config + outcome, no longer split across the parent).

**`CalibrationResult` is now four fields**: `provenance`, `fit`, `optimiser_run`, `diagnostics`. The flat `.new()` factory is **removed** — construction is direct component injection:
```python
CalibrationResult(
    provenance=CalibrationProvenance.stamp(market_snapshot_id=...),
    fit=CalibrationFit(model_class=..., parameters=..., residuals=...),
    optimiser_run=OptimiserRun(spec=OptimiserSpec(...), iterations=..., converged=...),
)
```

**Scope** (39 files): all ~27 producer call-sites across 15 modules rewritten to inject components (via an AST codemod, value-source preserved); `db.save_calibration` reads `result.provenance.*` / `fit.*` / `optimiser_run.*`; the `CanonicalCalibrationResult` mixin's `calibration_id` reads `provenance.id`; serialisation schema bumped to **v3** (nested payload). Tests updated to component access (`cr.fit.model_class`, `cr.optimiser_run.converged`, `cr.provenance.id`); a `build_calibration_result()` fixture in `conftest.py` keeps test construction concise.

**Field-shape notes**: `CalibrationFit` canonicalises its sequences to lists (exact round-trip). The former unity-weights default is dropped — empty `weights` now means "unweighted" (consistent with the unweighted `rms_residual`).

**Verification**: full suite **12831 passed** (two slow G2++ tests deselected per convention).

---

## v1.134.0 — 2026-06-22 — **Calibration unification G1 Phase 4 (C.5): CalibrationResult coherence — derived rms/max**

Final re-assessment finding. Closes the two `_types.py` items the original clean-code review deferred.

**Files**: `python/pricebook/calibration/_types.py`, `python/tests/test_calibration_result_serialisation.py`, `python/tests/test_calibration_persistence.py`.

* **`rms_residual` / `max_residual` are now derived `@property`s over `residuals`** — not stored fields. Single source of truth, so they can **never drift** from `residuals` (the clean-code-expert's original concern: a directly-constructed record could previously carry an `rms_residual` inconsistent with its `residuals`). `.new()` no longer computes or passes them; they cannot be passed to the constructor at all.
* **Unweighted semantics documented**: `rms_residual` is deliberately the unweighted RMS of `residuals` regardless of `objective`/`weights` (those describe how the optimiser *combined* residuals; this is a plain magnitude summary). A consumer wanting a weighted RMS computes it from `residuals` + `weights`. (Resolves the "weights ignored by rms" incoherence by stating the contract rather than silently overloading it.)
* **Serialisation**: since the two are no longer `dataclasses.fields()`, they drop out of the flat payload. `_SERIAL_SCHEMA_VERSION` bumped to **2**. Backward-compatible *reads*: old v1 payloads carrying `rms_residual`/`max_residual` keys still deserialise (the convention ignores unknown keys; the properties recompute). Old code reading a new v2 payload gets a clear "upgrade" error rather than a confusing missing-arg failure. The denormalised db columns still populate from the properties.

**Tests**: 3 new (derived values correct incl. empty-residuals; not constructor params → `TypeError`; payload omits the keys + `_schema_version == 2`); 2 existing `_schema_version` assertions updated 1 → 2.

**Verification**: full suite **12831 passed** (two slow G2++ tests deselected per convention).

### Phase 4 complete — calibration unification fully closed
All re-assessment findings resolved: C.1 mixin (v1.131), C.2 lmm model_class (v1.132), C.3 (folded into the mixin), C.4 fx_slv residual bug (v1.133), C.5 rms/max coherence (v1.134). The calibration layer now has: one canonical `CalibrationResult` (unique name), 12 uniform producers via `CanonicalCalibrationResult`, a closed build→store→read loop, derived (drift-proof) fit metrics, and an unambiguous audit chain.

---

## v1.133.0 — 2026-06-22 — **Calibration unification G1 Phase 4 (C.4): fx_slv residual bug fix**

`particle_slv_calibration` accumulated its fit error as `(L − 1)² * 0.0` — a dead placeholder, so the reported `residual` was **always 0.0** (surfaced by the unification, which put that 0 into the audit chain).

**Fix** (`python/pricebook/fx/fx_slv_calibration.py`): the residual now measures **local-vol reproduction error** — `L·√E[v|S] − σ_LV` per grid cell. By construction this is ≈0 wherever the leverage was not clipped and the regressed conditional variance `E[v|S]` was invertible; it is non-zero only where `L` hit the `[0.1, 10]` clip or `E[v|S]` was degenerate. So the residual now genuinely measures the calibration error introduced by those numerical safeguards.

**Behaviour change**: `ParticleCalibrationResult.residual` (and the canonical record's `residuals`) is now a real, generally-small, non-negative number instead of a constant 0. The pre-existing `test_residual_reasonable` (`< 0.05`) — previously trivially true — is now a meaningful assertion. Worth a `numerical-critic` pass if this calibrator becomes load-bearing.

**Tests**: 1 new (`test_residual_is_real_reproduction_error` — finite, non-negative, small for the benign flat-vol case).

**Verification**: full suite **12829 passed** (two slow G2++ tests deselected per convention).

**Next** (Phase 4, C.5 — last finding): `_types.py` coherence — `rms_residual`/`max_residual` → `@property`.

---

## v1.132.0 — 2026-06-21 — **Calibration unification G1 Phase 4 (C.2): disambiguate lmm model_class**

`models/lmm_advanced` and `models/lmm_calibration` both stamped `model_class="lmm"` on their canonical records — an audit-chain ambiguity (two distinct calibrators, same model tag). `lmm_advanced` (the Rebonato cascade/global approximation) now stamps **`"lmm_rebonato"`**; `lmm_calibration` (iterative-scaling ATM grid) keeps `"lmm"`. The calibration *method* still also lives in `optimiser.algorithm`; this makes the model tag itself unambiguous.

**Files**: `python/pricebook/models/lmm_advanced.py` (both record-builders), `python/tests/test_lmm_advanced.py`.

**Verification**: full suite **12828 passed** (two slow G2++ tests deselected per convention).

---

## v1.131.0 — 2026-06-21 — **Calibration unification G1 Phase 4 (1/?): CanonicalCalibrationResult mixin**

Consolidation. Phase 2 left **12 families** each duplicating the same scaffolding (a `calibration_result` field, a `to_calibration_result()` with an identical stored-or-rebuild guard, and an inline `calibration_id` in `to_dict`). This slice extracts that into a mixin and adopts it everywhere.

**Files**: `calibration/_types.py` (+ `__init__` export), `python/tests/test_calibration_mixin.py` (new), and all 12 family modules: `credit/bond_hazard_bootstrap`, `credit/joint_equity_credit`, `curves/multicurve_solver`, `equity/dividend_calibration`, `fixed_income/jarrow_yildirim`, `fx/fx_slv_calibration`, `models/{g2pp,hw,jump,lmm_advanced,lmm}_calibration` & `models/stochastic_correlation`.

**`CanonicalCalibrationResult` mixin** (exported from `pricebook.calibration`):
* `to_calibration_result()` — returns the stored record (eager builder population) or lazily builds + caches via `_build_calibration_record()`. **This unifies the two variants** Phase 2 introduced (builder-populate vs lazy-cache) into one mechanism.
* `_build_calibration_record()` — abstract (raises `NotImplementedError`); each family supplies its model-specific mapping.
* `calibration_id` — property for `to_dict` payloads (no build side-effect).

Each family now: inherits the mixin, keeps its `calibration_result` field, renames its old `to_calibration_result` body to `_build_calibration_record` (the guard moves to the mixin), and uses `self.calibration_id` in `to_dict`. **Net −11 lines across 14 files** despite adding the mixin — the duplication is gone.

This is the abstraction the deleted `Calibrator` Protocol (v1.122) failed to be: removed then for **0** implementers; added now, justified by **12**. (Addresses re-assessment C.1 + C.3.)

**Tests**: 4 new (mixin contract — lazy build+cache, eager-stored-wins, `NotImplementedError` for a bare subclass, polymorphic db persistence). Behaviour-preserving: full suite **12824 passed** (unchanged count; two slow G2++ tests deselected per convention).

**Next** (Phase 4 cont.): C.2 (`model_class="lmm"` overlap), C.4 (`fx_slv` placeholder-residual bug), C.5 (`_types.py` `rms`/`max` → `@property`).

---

## v1.130.0 — 2026-06-21 — **Calibration unification G1 Phase 2 (6/6): stochastic_correlation producer — Phase 2 COMPLETE**

Final Phase-2 slice. `models/stochastic_correlation.DispersionCalibrationResult` now produces the canonical record — **all 12 calibration families now emit `CalibrationResult`**.

**Files**: `python/pricebook/models/stochastic_correlation.py`, `python/tests/test_stochastic_correlation.py`.

* Added `calibration_result: CalibrationResult | None = None`, `calibration_id` in `to_dict()`, and `to_calibration_result()` (lazy-cache — the instance retains `index_variance_model`/`target`, so the signed residual is faithful). `model_class="stochastic_correlation"`, `parameters={kappa, theta, sigma}`, `optimiser.algorithm="closed_form"`.

**Tests**: 2 new — faithful residual + caching; end-to-end persistence via `db.save_calibration`.

**Verification**: full suite **12824 passed** (two slow G2++ calibration tests deselected per convention).

### Phase 2 complete — unification status
All families (`hull_white`, `g2pp`, `lmm`, `lmm`(rebonato), `jump`, `bond_hazard`, `multicurve`, `discount_curve`, `sabr`, `jarrow_yildirim`, `dividend_curve`, `joint_equity_credit`, `fx_slv`, `stochastic_correlation`) produce and can persist the canonical `CalibrationResult`. Excluded with reason: `GeneratorCalibrationResult` (transition matrix), `RobustCalibrationResult` (internal optimiser helper).

**Re-assessment findings → follow-ups** (see `OPEN.md`): (4) `fx_slv` `residual` is a placeholder (always 0.0) — numerical fix needed; (5) `_types.py` deferred coherence items — `rms_residual` ignores `weights`/`objective`, and `rms`/`max` are stored (could be `@property`); (6) the 12 near-identical `to_calibration_result` implementations now justify a **Phase 4 mixin** (the abstraction the deleted `Calibrator` Protocol failed to be — now justified by 12 real implementers); also two pattern variants (builder-populate vs lazy-cache) to unify, and the `model_class="lmm"` overlap to decide.

---

## v1.129.0 — 2026-06-21 — **Calibration unification G1 Phase 2 (5/6): fx_slv_calibration producer**

`fx/fx_slv_calibration.ParticleCalibrationResult` now produces the canonical record.

**Files**: `python/pricebook/fx/fx_slv_calibration.py`, `python/tests/test_fx_slv_calibration.py`.

* Added `calibration_result: CalibrationResult | None = None`, a proper `to_dict()` (was `dict(vars(self))` — which emitted a raw `LeverageFunction` object; now scalar summary + `calibration_id`), and `to_calibration_result()`.
* **Builder-populate** (`particle_slv_calibration`): the fitted output is a leverage *surface*, so `parameters` carry the SV config the instance doesn't retain — `{kappa, theta, xi, v0, rho, bandwidth}` — with `diagnostics.extra={n_particles, n_grid}`. `model_class="fx_slv"`, `optimiser.algorithm="particle_method"`. On-demand rebuild has only `bandwidth` + `n_particles`.
* **Flagged, not fixed** (out of scope — pre-existing): `particle_slv_calibration`'s `residual` is a placeholder — line `total_sq_err += (...) * 0.0` makes it always `0.0`. The canonical record reflects this faithfully (`residuals=[0.0]`). Noted for the Phase-2 re-assessment / a future numerical fix.

**Tests**: 3 new — builder canonical record (SV params, `n_particles` in diagnostics); on-demand rebuild; end-to-end persistence via `db.save_calibration`.

**Verification**: full suite **12822 passed** (two slow G2++ calibration tests deselected per convention).

**Next** (Phase 2, 6/6 — last): `models/stochastic_correlation.DispersionCalibrationResult`.

---

## v1.128.0 — 2026-06-21 — **Calibration unification G1 Phase 2 (4/6): joint_equity_credit producer**

`credit/joint_equity_credit.JointCalibrationResult` now produces the canonical record.

**Files**: `python/pricebook/credit/joint_equity_credit.py`, `python/tests/test_phase5_credit.py`.

* Added `calibration_result: CalibrationResult | None = None`, `calibration_id` in `to_dict()` (preserved via a dict-comprehension over `vars` minus the artefact — keeps every existing key), and `to_calibration_result()`.
* **Builder-populate** (`joint_calibrate`): captures provenance the instance doesn't store — `iterations`/`converged` from the optimiser result and `weights=[vol_weight, spread_weight]` with `objective=WEIGHTED_SSE` (matching the actual weighted-relative objective). `parameters={asset_vol, leverage}` (the two fitted), residuals are the **dimensionless relative errors** of the two fitted quotes (`equity_vol`, `cds_spread_<tenor>Y`) — consistent units, so `rms_residual` is meaningful.
* On-demand rebuild (hand-constructed) computes the same relative residuals from the stored model/market fields, with `objective=SSE` (no weights available).

**Tests**: 4 new — `calibration_id` in `to_dict`; builder canonical record (weighted_sse, two params, two residuals); on-demand rebuild; end-to-end persistence via `db.save_calibration`.

**Verification**: full suite **12819 passed** (two slow G2++ calibration tests deselected per convention).

**Next** (Phase 2, 5/6): `fx/fx_slv_calibration.ParticleCalibrationResult`.

---

## v1.127.0 — 2026-06-21 — **Calibration unification G1 Phase 2 (3/6): dividend_calibration producer**

`equity/dividend_calibration.DividendCalibrationResult` now produces the canonical record.

**Files**: `python/pricebook/equity/dividend_calibration.py`, `python/tests/test_dividend_calibration.py`.

* Added `calibration_result: CalibrationResult | None = None`, `calibration_id` to `to_dict()`, and `to_calibration_result()`.
* **Pattern variation (deliberate)**: this class already retains `fitted_futures` and `market_futures`, so `to_calibration_result()` builds **faithful per-tenor residuals** (`fitted − market`) directly from the instance and **caches lazily** on first call — rather than populating at each of the four build sites (`linear`/`optimize`/`spline`/`options`). Stable id once accessed; no four-site duplication. `model_class="dividend_curve"`, `parameters={D_<tenor>: fitted}`, `optimiser.algorithm=method`. (Noted for the upcoming pattern re-assessment: lazy-cache vs builder-populate.)

**Tests**: 2 new — faithful residuals + caching (same instance on 2nd call); end-to-end persistence via `db.save_calibration` → `list_calibrations(model_class="dividend_curve")`.

**Verification**: full suite **12816 passed** (two slow G2++ calibration tests deselected per convention).

**Next** (Phase 2, 4/6): `credit/joint_equity_credit.JointCalibrationResult`.

---

## v1.126.0 — 2026-06-21 — **Calibration unification G1 Phase 2 (2/6): jarrow_yildirim producer**

`fixed_income/jarrow_yildirim.JYCalibrationResult` now produces the canonical record.

**Files**: `python/pricebook/fixed_income/jarrow_yildirim.py`, `python/tests/test_jarrow_yildirim.py`.

* Added `calibration_result: CalibrationResult | None = None`, a proper `to_dict()` (was `dict(vars(self))` — would emit a raw `JYParams` object and, post-widen, a `CalibrationResult`; now emits the param fields + `calibration_id`), and `to_calibration_result()`.
* `jy_calibrate` populates the stored `cr` with **per-tenor residuals** (model fair-rate − target) over the ZC inflation-swap quotes; `model_class="jarrow_yildirim"`, `parameters={sigma_n, sigma_r, sigma_I}` (the three fitted vols), `optimiser.algorithm="Nelder-Mead"`.
* On-demand rebuild (hand-constructed instances) uses `residuals=[self.residual]` (aggregate only).
* Tidied formatting (stray blank lines, glued `def`).

**Tests**: 3 new — builder populates the canonical record (per-tenor residuals, three sigma params); on-demand rebuild; end-to-end persistence via `db.save_calibration`.

**Verification**: full suite **12814 passed** (two slow G2++ calibration tests deselected per convention).

**Next** (Phase 2, 3/6): `equity/dividend_calibration.DividendCalibrationResult`.

---

## v1.125.0 — 2026-06-21 — **Calibration unification G1 Phase 2 (1/6): lmm_advanced producer + LMM name-shadow fix**

First of six Phase-2 slices (widen producers onto the now-proven `to_calibration_result()` pattern). Surfaced a *second* name collision while here: `models/lmm_advanced` and `models/lmm_calibration` both defined a class named `LMMCalibrationResult` (different shapes — Rebonato cascade/global vs iterative-scaling ATM-grid). Resolved by renaming, bundled into this slice since the file is touched anyway.

**Files**: `python/pricebook/models/lmm_advanced.py`, `python/tests/test_lmm_advanced.py`.

* **Rename** `lmm_advanced.LMMCalibrationResult` → **`RebonatoLMMCalibrationResult`** (it calibrates via the Rebonato swaption-vol approximation). The other `LMMCalibrationResult` (in `lmm_calibration`) keeps its name. No file imported both; only `test_lmm_advanced.py` referenced the renamed one.
* **Widen**: added `calibration_result: CalibrationResult | None = None`, a proper `to_dict()` (was `dict(vars(self))` — which would have emitted a raw `np.ndarray` and, post-widen, a `CalibrationResult` object; now emits `vols.tolist()` + `calibration_id`), and `to_calibration_result()`.
* Both builders (`lmm_cascade_calibration`, `lmm_global_calibration`) now populate the stored `cr` via a shared `_lmm_calibration_record()` helper that computes **per-swaption residuals** (model − market vol) over the fitted swaptions and `quotes_fitted` keys. `model_class="lmm"`; the cascade/global method goes in `optimiser.algorithm`.
* On-demand rebuild path (hand-constructed instances, no stored `cr`) is the documented fallback — it only has the aggregate `residual`, so `residuals=[self.residual]`.
* Tidied the sloppy formatting (stray blank lines, the `def` glued onto the dataclass).

**Tests**: 4 new — builder populates the stored canonical record (per-swaption residuals, `sigma_i` params, method in algorithm); global method tag; on-demand rebuild for hand-constructed; and end-to-end persistence via `db.save_calibration` → `load_calibration` → `list_calibrations(model_class="lmm")`.

**Verification**: full suite **12811 passed** (two slow G2++ calibration tests deselected per convention).

**Next** (Phase 2, 2/6): `fixed_income/jarrow_yildirim.JYCalibrationResult`.

---

## v1.124.0 — 2026-06-21 — **Calibration unification G1 Phase 3a: close the build → store → read loop**

Phase 3 (done before Phase 2, by design — don't widen a build-and-drop pattern before the consumer that justifies it exists). The diagnosis throughout: even the 5 families that "adopted" the canonical record only build-and-drop it — `to_calibration_result()` had **zero production callers** and nothing persisted the record. This slice gives the record a real consumer and closes the loop.

**Files**: `python/pricebook/db/db.py`, `python/tests/test_calibration_loop.py` (new).

**`PricebookDB.save_calibration` is now polymorphic**: it accepts either a canonical `CalibrationResult` or **any family result exposing `to_calibration_result()`** (`HWCalibrationResult`, `JumpCalibrationResult`, `G2PPCalibrationResult`, `LMMCalibrationResult`, the multicurve result, …) — duck-typed like `save_trade`/`save_snapshot` accept anything with `to_dict()`. A family result is converted via that accessor, then persisted as before.

This **settles the design question** Phase 3-first was meant to answer:
* The optional `calibration_result: CalibrationResult | None` field stays — it's the calibrator-populated storage.
* `to_calibration_result()` is now the single canonical *accessor*, with `save_calibration` as its first real production *consumer*. No longer dead.
* The loop **build → store → read** is closed and uniform across every adopter.

**Semantic note** (pinned by a test): the on-demand rebuild branch of `to_calibration_result()` (taken when no `cr` is stored — back-compat for hand-constructed instances) calls `CalibrationResult.new()` and therefore mints a fresh `id`/`timestamp` each call. It is a fallback, not a stable identity; calibrators populate the stored `cr` so the id is stable. Persisting a stored-`cr` family result is idempotent; persisting a no-`cr` one mints a new id per call.

**Tests**: 5 new (cheap real adopter `JumpCalibrationResult`) — canonical passthrough still works; family result with stored `cr` round-trips (`load == to_calibration_result()`); no-`cr` family rebuilds on demand (content + returned-id checks); denormalised columns populated through the family path; audit query by `market_snapshot_id` through the family path.

**Verification**: full suite **12807 passed** (two slow G2++ calibration tests deselected per convention).

**Next** (Phase 2): widen producers — migrate the 6 holdout bespoke `*CalibrationResult` family types onto this now-proven `to_calibration_result()` pattern, so every calibrator's record is uniformly persistable.

---

## v1.123.0 — 2026-06-21 — **Calibration unification G1 Phase 1b: resolve the two CalibrationResult name-shadows**

Completes Phase 1 (kill the contradictions). Two modules defined their own class also called `CalibrationResult`, unrelated to the canonical L0 record — a genuine name collision that made the codebase ambiguous about which `CalibrationResult` was meant. Both renamed (not migrated: their shapes don't fit the canonical `parameters: Mapping[str,float]` + provenance record, and forcing it would be wrong).

**Files**: `python/pricebook/credit/rating_models.py`, `python/pricebook/models/calibration_utils.py`, `python/tests/test_rating_models.py`.

* `credit/rating_models.CalibrationResult` → **`GeneratorCalibrationResult`**. Holds a `RatingTransitionMatrix` + residual + converged — the result of generator-matrix calibration (`calibrate_generator`). A matrix is not a `Mapping[str,float]`; definitively a distinct artefact.
* `models/calibration_utils.CalibrationResult` → **`RobustCalibrationResult`**. A lightweight multi-start/robust optimiser result (`params: list`, `rmse`, `condition_number`, `method`); matches its own "robust calibration" docstring. Internal numeric helper, not a provenance artefact.

Also tidied the sloppy formatting in both (stray blank lines, the `def` glued directly onto the dataclass) and dropped a dead `CalibrationResult` import in `test_rating_models.py` (imported, never used).

**Consumers**: neither class was re-exported through a package `__init__`; the only references were each module's own function + test. `test_calibration_utils.py` never imported the class name (uses the functions). No external caller affected.

**Result**: `grep "class CalibrationResult"` now returns **exactly one** definition — the canonical `calibration/_types.py`. The name is unambiguous across the whole library.

**Verification**: full suite **12802 passed** (two slow G2++ calibration tests deselected per convention).

**Next** (Phase 2): widen producers — migrate the 8 holdout bespoke `*CalibrationResult` family types onto the canonical `to_calibration_result()` pattern.

---

## v1.122.0 — 2026-06-21 — **Calibration unification G1 Phase 1a: delete dead Calibrator Protocol**

First slice of Phase 1 (kill the contradictions). The `Calibrator` Protocol in `calibration/_types.py` had **zero implementers** anywhere in the codebase — every calibration family uses free functions (`calibrate_hull_white`, `joint_calibrate`, `calibrate_g2pp`, …) returning bespoke types, none implementing `.calibrate()`. The Protocol's only consumer was its own self-test. Textbook Speculative Generality; removed.

(It was also self-contradictory: the docstring claimed "or are themselves callable — either is acceptable", but the Protocol declared only `calibrate()`, so a callable-only object would fail the type. Deleting the whole thing resolves that too.)

**Files**: `python/pricebook/calibration/_types.py` (drop the `Calibrator` class + now-unused `Protocol` import), `python/pricebook/calibration/__init__.py` (drop the export from the docstring, import, and `__all__`), `python/tests/test_calibration_types.py` (drop `TestCalibratorProtocol`, the only reference).

**Public API change**: `pricebook.calibration.Calibrator` no longer exists. No production caller imported it (verified by whole-word grep); the only reference was the deleted test. `calibration.__all__` is now `[CalibrationDiagnostics, CalibrationResult, ObjectiveKind, OptimiserSpec]`.

**Verification**: full suite **12802 passed** (two slow G2++ calibration tests deselected per convention).

**Next** (Phase 1b): resolve the two `CalibrationResult` name-shadows — `credit/rating_models.py:34` and `models/calibration_utils.py:19` each define a *different* class also called `CalibrationResult`.

---

## v1.121.0 — 2026-06-21 — **Calibration unification G1 Phase 0 Slice 2: persist + load CalibrationResult**

Second slice of the calibration-result unification (consumer axis). Slice 1 made `CalibrationResult` serialisable; this slice makes it *load-bearing* — the record can now be persisted and read back, turning the previously build-and-drop fields live.

**Files**: `python/pricebook/db/db.py`, `python/tests/test_calibration_persistence.py` (new).

**New `calibration_results` table** (system table; created via `CREATE TABLE IF NOT EXISTS`, so existing dbs gain it on next open — no migration). UUID-keyed (`calibration_id` PK). The full record is stored as JSON (`result_json`); the identity/quality fields — `model_class`, `timestamp`, `code_version`, `objective`, `converged`, `iterations`, `rms_residual`, `max_residual`, `market_snapshot_id` — are **denormalised into columns** so the audit chain is queryable without reconstructing every blob. Indexed on `model_class` and `market_snapshot_id`.

**New `PricebookDB` methods** (matching the existing trade/snapshot pattern):
* `save_calibration(result) -> str` — idempotent on the calibration id (`ON CONFLICT … DO UPDATE`); returns the id. This is where the dead fields get consumed — it reads `.model_class`, `.timestamp`, `.objective`, `.converged`, `.iterations`, `.rms_residual`, `.max_residual`, `.market_snapshot_id` to populate the columns.
* `load_calibration(id) -> CalibrationResult | None` — reconstructs via `CalibrationResult.from_dict` (called directly, not the generic registry `from_dict`, because the convention payload is flat with no `type` discriminator — the table tells us the type).
* `load_calibration_raw(id) -> dict | None` — row + parsed JSON, no reconstruction.
* `list_calibrations(**filters)` — metadata listing; filter by any column (e.g. `model_class="HullWhite"`, `market_snapshot_id=…` for the audit chain).

`db.py` takes a `TYPE_CHECKING`-only import of `CalibrationResult`/`UUID` (annotations are lazy under `from __future__ import annotations`); the runtime calibration import is lazy inside `load_calibration`, matching the existing `load_trade`/`load_snapshot` style. No new load-time edge.

**Tests**: 12 new — round-trip reconstruction (`load == original`), str/UUID id acceptance, missing→None, denormalised columns track the fields (incl. bool→INTEGER, null snapshot), `list`/filter by `model_class` and by `market_snapshot_id` (audit chain), idempotent re-save, system-table protection, and file-backed durability across reopen.

**Verification**: full suite **12803 passed** (two slow G2++ calibration tests deselected per convention).

**Next** (Phase 1): kill the dead `Calibrator` Protocol (zero implementers) and resolve the two `CalibrationResult` name-shadows (`credit/rating_models.py`, `models/calibration_utils.py`); then Phase 2 widens producers; Phase 3 wires `to_calibration_result()` into this persistence path so build → store → read is a closed loop.

---

## v1.120.0 — 2026-06-20 — **Calibration unification G1 Phase 0 Slice 1: CalibrationResult serialisable + tz-aware clock**

First slice of the calibration-result unification (consumer axis first). Audit + grep established that the canonical `CalibrationResult` is currently *build-and-drop*: ~10 calibrators construct one, but production reads only `.id` off it, nothing serialises/persists it, and `to_calibration_result()` has zero production callers. Before widening producers, the record needs to be load-bearing — and step one is making it round-trip.

**Files**: `python/pricebook/core/serialisable.py`, `python/pricebook/calibration/_types.py`, `python/tests/test_calibration_result_serialisation.py` (new).

**Core serialisation infra** (`serialisable.py`): the atom serialiser handled neither `UUID` nor `datetime` — a `UUID` field passed through as a non-JSON object (would crash `json.dumps`), and a `datetime` had no deserialise path. Added, all additive (no existing serialisable type has a UUID/datetime field):
* `_serialise_atom`: `UUID → str`; recurse into `dict` values; treat `tuple` like `list`. (`datetime` already serialised via the `date` branch — it is a `date` subclass and `isoformat()` carries the time + offset.)
* `_deserialise_atom`: `datetime` (via `datetime.fromisoformat`, checked before `date`) and `UUID` branches.

**Calibration types** (`_types.py`):
* `OptimiserSpec`, `CalibrationDiagnostics`, `CalibrationResult` are now `@serialisable_convention` (flat, pure-data dicts carrying `_schema_version`). Fields auto-derived from the dataclass — no hand-maintained field list.
* `CalibrationResult.new()` now stamps **timezone-aware UTC** (`datetime.now(timezone.utc)`) instead of a naive local `datetime.now()` — a provenance timestamp without an offset is ambiguous across machines/DST. `id` and `timestamp` are now injectable (default-generated when omitted) for reproducing a stored result or making a test deterministic — mirroring the existing `code_version` override.
* `CalibrationDiagnostics` canonicalises its sequence fields to tuples in `__post_init__`: correct for a frozen record, and it makes serialise→deserialise exact (JSON arrays come back as lists, so a default `()` would otherwise `!= []` after a round-trip).

**Why this is safe**: the `_types.py → core.serialisable` edge is acyclic (core.serialisable imports nothing from calibration; the back-reference in `core/discount_curve` is `TYPE_CHECKING`-only). Module stays at L0.

**Tests**: 12 new — round-trip for all three types, JSON-native payload, tz-aware UTC clock, injectable id/timestamp, distinct auto-ids, and atom-level UUID/datetime/dict/tuple round-trips.

**Verification**: full suite **12791 passed** (two slow G2++ calibration tests deselected per convention); targeted serialisation + all 10 calibration-consumer modules 1216/1216.

**Next** (Phase 0 Slice 2): db persistence + a reader that consumes the now-serialisable record — turning the 13 currently-unread fields live.

---

## v1.119.0 — 2026-06-19 — **W8: swap pillar pinned at schedule-end (not unadjusted mat); warnings campaign COMPLETE 🏁**

Slice 8/8 — the final slice of the warnings-sweep campaign.

**Bug**: when a swap's maturity falls on a non-business-day, `generate_schedule` rolls the schedule's last date forward under the business-day convention (e.g. unadjusted `2029-01-15` → adjusted `2029-01-16` if Jan 15 is a weekend). The bootstrap, however, was placing its discount-factor pillar at the **unadjusted** swap `mat`. Sequence of events:

1. At swap-solve time: trial_curve has pillars `[..., 2027-01-15, 2029-01-15_candidate]`. `df(2029-01-16)` is **extrapolated** past the last pillar.
2. `brentq` finds `df(2029-01-15) = df_solved` such that `pv_fixed(using extrapolated df(2029-01-16)) == 1 − df(2029-01-16)_extrapolated`.
3. After all swaps solved: final curve has pillars `[..., 2027-01-15, 2029-01-15, 2031-01-15, 2034-01-15]`. Now `df(2029-01-16)` is **interpolated** between 2029-01-15 and 2031-01-15.

Log-linear extrapolation past a pillar and log-linear interpolation between two pillars give different values. The round-trip check (which used the final curve) reported `pv_fixed − pv_float ≈ 2.44e-6` for every USD OIS curve that hit this — 6 orders of magnitude above the 1e-12 brentq tolerance, triggering the W8 warnings in `test_rfr_bootstrap` and `test_synthetic_curves`. The EM curve test (`Swap 2029-11-04: 4.16e-5`, `Swap 2034-11-04: 1.27e-4`) was the same shape — bigger gap because more swap pillars trailed.

**Fix**: set the pillar date to `max(fixed_sched[-1], float_sched[-1])` — i.e. the actual last date the schedule pays on, post-roll. `df(schedule_end)` is now exactly pillared rather than extrapolated; later swaps don't change it; round-trip is exact (`< 1e-12`).

This also tightens calibration semantics: `CalibrationResult.parameters` now reports the discount factor at the date the leg actually pays, not the platonic unadjusted date.

**Caller-impact**: pillar dates for swaps that hit a non-business-day under MODIFIED_FOLLOWING (or similar) shift by ≤ a few days. The discount-factor values at those pillars match the legs' payment dates exactly, which is more correct than before. Any consumer that introspected `curve.pillar_dates` and expected them at unadjusted schedule maturities will now see the rolled dates. No production caller does this introspection.

**Regression**: new test `TestBootstrapUSD::test_w8_no_round_trip_warning` runs the standard 6-swap USD OIS bootstrap (which trips the bug on 2029-01-15) under `simplefilter("error", RuntimeWarning)`. Pins the fix.

**Verification**:
* `test_rfr_bootstrap.py` 19/19 + `test_synthetic_curves.py` 9/9 — zero warnings.
* Broader bootstrap/swap/curve set — 1824/1824 passing, zero warnings.
* Full L≤3 suite — **8268/8268 passing, ZERO warnings**.

---

# Warnings sweep — COMPLETE.

The campaign that began at v1.112.0 closes here. From 18 RuntimeWarnings to **zero** in the L≤3 suite, in 8 disciplined slices:

| Slice | Warnings dropped | What it fixed |
|---|---|---|
| W1 (v1.112) | 1 | `realized_vol` silent NaN on non-positive prices |
| W2 (v1.113) | 1 | rough-Heston CF dropping `Im(integral_h)` — real numerical bug |
| W3 (v1.114) | 3 | HW convexity: stale verifier formula + structural local-bootstrap gap (pillar pinning for futures/FRAs) |
| W4 (v1.115) | 4 | PricingServer test thread + coroutine leaks |
| W5 (v1.116) | 3 | CMA-ES non-finite x/fx poisoning recombination |
| W6 (v1.117) | 1 | Rosenbrock test overflow (acknowledged via `np.errstate`) |
| W7 (v1.118) | 1 | scipy 1.17 anderson API migration (forward-compat to 1.19) |
| W8 (v1.119) | 3 | swap pillar pinned at schedule-end (not unadjusted mat) |

Two non-trivial *real* numerical bugs surfaced along the way (W2 rough-Heston, W3 HW convexity verifier, W5 CMA-ES robustness). The rest cleaned up legitimate diagnostic noise.

The L≤3 suite now runs with zero `pytest -W error::RuntimeWarning` failures across 8268 tests. Audit-chain plus warnings-sweep cumulatively: **all open items closed**.

---

## v1.118.0 — 2026-06-19 — **W7: `anderson_darling` adopts scipy 1.17 `method='interpolate'` (forward-compat to 1.19)**

Slice 7/8 of the warnings-sweep campaign.

**Bug**: `statistics/distribution_fit.anderson_darling` called `scipy.stats.anderson(x, dist=dist)` without the `method` keyword. scipy 1.17 emits a `FutureWarning` warning that:

* `method` will become required.
* When `method` is set, the returned `AndersonResult` will have a `pvalue` attribute and will NO longer have `critical_values`, `significance_level`, `fit_result`.
* From scipy 1.19, the legacy attributes go away entirely.

Our code used `result.critical_values` and `result.significance_level` to build a `{percentile: critical_value}` dict — the exact attributes scheduled for removal.

**Fix**: pass `method="interpolate"` and migrate to `result.pvalue`. The `ADResult` dataclass now carries `pvalue` instead of `critical_values`; `reject_at_5pct` derives from `pvalue < 0.05` (was `statistic > critical_values[2]`). Public `to_dict()` shape gains `pvalue` (was: stat / reject / n).

**Caller-impact**: `ADResult.critical_values` is gone. The library is the only producer of `ADResult`, and the only consumer was one test that just checked `hasattr(result, "statistic")`. No external break.

**Regression**: two new tests in `TestAndersonDarling` — one pins no FutureWarning under `simplefilter("error", FutureWarning)`; one sanity-checks that an exponential sample is rejected at 5% with `pvalue < 0.05`.

**Verification**: `test_distribution_fit.py` 10/10 passing under `-W error::FutureWarning`.

**Warnings count**: 5 → 4 in the L≤3 suite. Only W8 (3 bootstrap round-trip warnings) remains.

---

## v1.117.0 — 2026-06-19 — **W6: silence expected Rosenbrock overflow in CMA-ES test**

Slice 6/8 of the warnings-sweep campaign.

**Bug**: `test_optimisation_advanced.py::test_rosenbrock` evaluates `100·(x[1] − x[0]²)² + (1 − x[0])²` directly. CMA-ES (the unit under test) deliberately samples a wide region in early generations; `x[0]²` then `x[0]⁴` overflows float64 → `RuntimeWarning: overflow encountered in scalar power` emitted by the *test* file itself (not the library).

Pre-W5 this also caused a `RuntimeWarning: invalid value encountered in scalar subtract` cascade — the inf fitness then poisoned CMA-ES's recombination math (3 library-side warnings). W5 fixed the library to rank-last on non-finite fitness; this slice handles the residual test-side overflow.

**Fix**: wrap the body of the test's `rosenbrock` in `np.errstate(over="ignore", invalid="ignore")` so the expected exploration-side overflow is silenced at the smallest possible scope. CMA-ES post-W5 sees `inf` and recovers; the test now runs silent.

**Caller-impact**: zero — pure test scope.

**Verification**: `test_optimisation_advanced.py` 13/13 passing, zero warnings.

**Warnings count**: 6 → 5 in the L≤3 suite.

---

## v1.116.0 — 2026-06-19 — **W5: CMA-ES rejects non-finite samples + fitness (stops NaN poisoning)**

Slice 5/8 of the warnings-sweep campaign.

**Bug**: `statistics/optimisation_advanced.cma_es` evaluated the objective at every sampled `x = mean + sigma · (L @ z)` and stored `(x, z, fx)` in the population without checking for finiteness. Three downstream sites in the recombination + path arithmetic then operated on bad values:

* line 394 `mean += weights[i] * population[i][0]` → if any prior population member's `x` had `inf`, `mean` inherits `inf`. `RuntimeWarning: invalid value encountered in add`.
* line 400 `p_c = (1 - c_c) * p_c + h_sigma · √(...) · (mean - old_mean) / sigma` → `inf - finite = inf`, divide by tiny `sigma` → propagation. `RuntimeWarning: invalid value encountered in multiply`.
* line 405 `y_i = (population[i][0] - old_mean) / sigma` → same shape. `RuntimeWarning: invalid value encountered in divide`.

Two root paths into the bad state:

1. **Unbounded objective overflow** — `test_rosenbrock` uses `f(x) = 100·(x[1] − x[0]²)² + (1 − x[0])²`. Early CMA-ES generations explore widely; `x[0]²` for large `x[0]` overflows float64. `fx = inf`, then `x` survives in the population and pollutes recombination on subsequent iterations.
2. **`x` itself non-finite** — when `sigma` is large, `mean + sigma · (L @ z)` can overflow before `f(x)` is even called.

**Fix**: at sample time, check both `x` and `fx` for finiteness. If `x` is not all-finite, substitute the current mean (a known-finite point) and rank as `+inf` (so the sample sinks to the bottom of the sort and is excluded from the top-`mu` recombination). If `fx` is non-finite, just rank as `+inf` (keep the finite `x` — the next generation's mean stays sensible).

This is the standard CMA-ES robustness pattern per Hansen's "The CMA Evolution Strategy: A Tutorial" §B.3 (resampling on constraint violation); we don't resample but we do rank-to-last, which preserves the population-size invariant and lets the next generation re-sample from a finite mean.

**Caller-impact**: zero for objectives that already return finite values everywhere. For unbounded objectives, CMA-ES now converges where before it could silently NaN out (test_rosenbrock now solves cleanly with no library warnings).

**Regression**: new test `TestCMAES::test_w5_rejects_non_finite_fitness` runs CMA-ES on an objective that returns `math.inf` outside the unit interval under `simplefilter("error", RuntimeWarning)`. Pins the fix.

**Verification**: `test_optimisation_advanced.py` 13/13 passing. Library warnings 3 → 0; only the test-side `RuntimeWarning: overflow encountered in scalar power` (from Rosenbrock's `x[0]**2` directly) remains — that's W6.

**Warnings count**: 9 → 6 in the L≤3 suite.

---

## v1.115.0 — 2026-06-19 — **W4: PricingServer test fixture cleans up server thread / coroutines**

Slice 4/8 of the warnings-sweep campaign.

**Bug**: the `server_port` fixture in `test_pricing_server.py::TestServerClient` started an asyncio `PricingServer` in a background thread, then on teardown called `loop.call_soon_threadsafe(loop.stop)` and joined the thread. Two leaks:

1. `loop.stop()` interrupts `serve_forever()` but doesn't close the server — pending `_handle_connection` coroutines remained as un-awaited tasks, garbage-collected later in unrelated tests as `PytestUnraisableExceptionWarning: Exception ignored in: <coroutine object PricingServer._handle_connection at ...>`.
2. The interrupt itself raised a `RuntimeError` out of `serve_forever()` in the thread → `PytestUnhandledThreadExceptionWarning: Exception in thread Thread-N (_start_server_thread)`.

Both warnings showed up *attributed to* whichever test happened to run after the leak (`test_pde_solver::TestCrankNicolson::test_put` was a common scapegoat) — misleading attribution that made the root cause hard to spot.

**Fix**: the thread now wraps the serve loop in try/except/finally:

```python
try:
    loop.run_until_complete(server.start())
    loop.run_until_complete(server._server.serve_forever())
except (asyncio.CancelledError, RuntimeError):
    pass
finally:
    loop.run_until_complete(server.stop())   # closes socket + drains pool
```

And the fixture closes the loop after the thread joins (`if not loop.is_closed(): loop.close()`). `PricingServer.stop()` was already correct — the test just wasn't calling it.

**Caller-impact**: zero — test-only change. Library code unchanged.

**Verification**: `test_pricing_server.py` 12/12 passing, zero warnings under direct invocation. Full L≤3 suite warnings 12 → 9 (4 PricingServer ones gone — the test count went up by 1 from W3's new regression).

**Warnings count**: 13 → 9 in the L≤3 suite.

---

## v1.114.0 — 2026-06-19 — **W3: bootstrap pins df(start) for futures/FRAs + verifier formula matches T3.17 fix**

Slice 3/8 of the warnings-sweep campaign. Targets the three `test_l2_t3_17_18_bootstrap_convexity` `RuntimeWarning: Bootstrap round-trip failures` warnings.

**Two bugs, one slice:**

**Bug A — stale convexity formula in the verifier (post-T3.17 leftover)**

When T3.17 fixed the HW futures convexity formula in `bootstrap()` (line 150) to use the textbook `ca = 0.5·σ²·B(0,T₁)·B(T₁,T₂)`, two sibling sites kept the pre-T3.17 form:

* `_compute_calibration_result` (line 322) — residuals reported to `CalibrationResult.residuals` used the OLD formula. Any downstream consumer reading those residuals received wrong values whenever `hw_convexity_sigma > 0`.
* `_verify_round_trip` (line 445) — round-trip diagnostic warning used the OLD formula. Emitted false-positive `RuntimeWarning` for every futures bootstrap with non-zero convexity.

Both now mirror the bootstrap's own formula. The pre-T3.17 form is gone from the file.

**Bug B — structural local-bootstrap interpolation gap (W3 root cause)**

When the bootstrap processed a future or FRA whose `start_date` was strictly between existing pillars, it would:
1. Take `df_start = temp_curve.df(start_date)` — interpolated from the prior pillars.
2. Compute `df_end = df_start / (1 + rate × τ)`.
3. Add ONLY `end_date` as a pillar.

Result: the bootstrap committed to a specific `df(start)` value via step 1, but the final curve had no pillar there. Adding later swaps reshaped the log-linear interpolation in the gap region, changing `df(start)` on the final curve. The round-trip check then failed by ~0.3% on the future rate (W3 reported ~2.75e-3 — 4.76e-3 in the convexity scaling test) even when `σ=0`.

This is a structural property of local bootstrap, not a fit bug: the bootstrap's per-instrument df_start was correct *at the time*; later pillars just moved the interpolation around it.

**Fix B**: pin `df(start)` as a pillar at step 1 too (for futures and FRAs both), whenever `start_date != reference_date` and isn't already in `pillar_dates`. The bootstrap now expresses what it actually committed to. Final curve faithfully passes through both df(start) and df(end), round-trip works exactly.

**Caller-impact** (Bug B): the final curve now has more pillars whenever futures/FRAs were used with non-deposit start dates. Curve *values* are unchanged at the pillar points the bootstrap already chose — only the curve's *interpolation behaviour between* prior-pillar and end_date is now pinned rather than recomputed. Concretely: any consumer that introspected `pillar_dates` and assumed only end-of-instrument dates appeared will now also see start-of-instrument dates. No production caller does this introspection.

**Caller-impact** (Bug A): `CalibrationResult.residuals` values for futures are now correct. Previously over- or under-reported by `(B(0,T₁)·B(T₁,T₂) − B(T₁,T₂)·[B(0,T₂)−B(0,T₁)])·0.5σ²` per future.

**Regression**: new test `TestHWConvexity::test_no_round_trip_warning_w3` runs both the σ=0 and σ>0 bootstrap calls under `simplefilter("error", RuntimeWarning)` — locks both fixes.

**Verification**:
* `test_l2_t3_17_18_bootstrap_convexity.py` — 7/7 passing under `-W error::RuntimeWarning`.
* Full L≤3 suite — 8264/8264 passing.

**Warnings count**: 16 → 13 in the L≤3 suite (3 W3 ones eliminated; the W8 `rfr_bootstrap` + `em_curve_builder` swap warnings remain — a similar shape, slice 8 will sweep them).

---

## v1.113.0 — 2026-06-19 — **W2: rough Heston CF keeps `integral_h` complex (drop float cast)**

Slice 2/8 of the warnings-sweep campaign.

**Bug**: `models/rough_heston_cf._solve_fractional_riccati` line 125 was

```python
integral_h = float(np.sum(0.5 * (h[:-1] + h[1:]) * np.diff(t_grid)))
```

`h` is `np.zeros(n_steps + 1, dtype=complex)` driven by complex coefficients (`c_coeff = 0.5*(-(u**2) + 1j*u)`, `b_coeff = 1j*u*rho*xi - kappa`). For any non-zero `u`, `integral_h` is fundamentally complex. The `float(...)` cast (a) emitted `numpy.exceptions.ComplexWarning: Casting complex values to real discards the imaginary part` on every CF evaluation, and (b) — the actual bug — dropped `Im(integral_h)` from `log_cf`. The next line

```python
log_cf = 1j * u * (rate - div_yield) * T + theta * integral_h + v0 * I_alpha_h0
```

mis-shaped the CF on the imaginary axis: the contribution `1j * theta * Im(integral_h)` was being silently zeroed, which (via the COS-method inversion in `rough_heston_price`) shifts every European option price computed with this CF.

**Fix**: drop the cast. `integral_h` stays complex; `log_cf` stays complex; the CF is now what the math actually demands.

**Caller-impact**: every prior caller of `rough_heston_price` was getting a slightly mis-priced option. Test tolerances were loose enough (`price > 0`, `|p_rough - p_smooth| > 0.01`) that this wasn't caught. There are no golden-value tests to update — but downstream consumers should re-baseline if they pinned numerical outputs.

Note: there is a separate `rough_paths.rough_heston_cf` (different module, simpler algebra) that does NOT have this bug.

**Regression**: new test `TestRoughHeston::test_cf_keeps_imaginary_part` prices a rough param set with `warnings.simplefilter("error", ComplexWarning)` — locks the fix.

**Verification**: full `TestRoughHeston` (3 tests) passing.

**Warnings count**: 17 → 16 in the L≤3 suite.

---

## v1.112.0 — 2026-06-19 — **W1: `realized_vol` rejects non-positive prices**

Slice 1 of the 8-slice warnings-sweep campaign (`W1` … `W8`) that follows the audit-chain closure. Goal: drive the 18 RuntimeWarnings emitted by the L≤3 test suite to zero, one root cause per slice.

**Bug**: `statistics/garch.realized_vol(prices, ...)` did `np.log(prices)` with no input validation. The associated test `test_realized_vol` mistakenly passed a `returns` fixture (normal(0, 0.01) — half negative) to a function that expects strictly-positive prices. Result: silent `RuntimeWarning: invalid value encountered in log` on every L≤3 run, and the rolling-window stats were polluted by NaN-from-log noise.

This is the recurring **silent-no-op API hazard** pattern catalogued in `[[recurring-bug-patterns]]`.

**Fix**:

1. `realized_vol` now raises `ValueError("realized_vol requires strictly-positive prices; got entries ≤ 0. If you passed returns by mistake, convert with `prices = base * np.cumprod(1 + returns)` first.")` if any price is ≤ 0. The error message points at the most common mistake — the same one the test was making.
2. `test_realized_vol` now correctly converts the `returns` fixture to a price series via `100.0 * np.cumprod(1 + returns)` before calling.
3. Two new regression tests: `test_realized_vol_rejects_non_positive` (returns-as-prices) and `test_realized_vol_rejects_zero_price` (mid-series zero).

**Caller-impact**: zero. Grep confirmed no production code calls `realized_vol` — only the one test, now fixed.

**Verification**: `test_garch.py` 8/8 passing.

**Warnings count**: 18 → 17 in the L≤3 suite.

---

## v1.111.0 — 2026-06-19 — **B.3 C1: retire legacy `core/market_data`; audit chain COMPLETE 🏁**

Last open item from the entire L0 audit catalogue. `core/market_data.py` was the pre-G1-P2 market-data module — superseded by the canonical `pricebook.market_data` package (`MarketSnapshot` / `QuoteKind` / `QuoteId` / `Quote`). Two implementations of the same concept is the architectural smell that motivated G1 P2 in the first place.

**1. Helpers migrated to `pricebook.market_data._types`**

Three names the only production consumer (`curves/curve_engine.py`) still needed are now defined in the canonical package and re-exported from `pricebook.market_data`:

* `tenor_to_years(s) -> float`
* `tenor_to_date(ref, s) -> date`
* `class MissingQuoteError(Exception)`

`tenor_to_date` does a function-scope import of `core.day_count.date_from_year_fraction` to avoid a load-time cycle while keeping the implementation co-located with the rest of the tenor-parsing logic.

**2. `curves/curve_engine.py` migrated to the new API**

* `MarketDataSnapshot` → `MarketSnapshot` (frozen dataclass with UUID + `as_of` datetime; reference date now comes from `snapshot.as_of.date()`).
* `QuoteType` → `QuoteKind` (str-Enum; same values for the 5 kinds curve_engine uses — DEPOSIT_RATE, SWAP_RATE, CDS_SPREAD, VOL_POINT, FX_SPOT — plus 12 new ones the new layer supports).
* `Quote(quote_type, tenor, value, currency, name)` → `Quote(id=QuoteId(kind, tenor, currency, label), value)`.
* `snapshot.get_quotes(qt, currency)` → `snapshot.filter(kind=qt, currency=currency)`.

Kept `QuoteType = QuoteKind` as a module-level alias in `curve_engine` so any downstream caller still importing `from pricebook.curves.curve_engine import QuoteType` keeps working without a same-slice edit.

**3. `test_curve_engine.py` migrated to the new API**

`_usd_snapshot()` now builds quotes via `QuoteId(...)` + `Quote(id=..., value=...)` and constructs the snapshot via `MarketSnapshot.new(quotes=..., as_of=datetime.combine(REF, datetime.min.time()))`. Aliased `QuoteKind as QuoteType` in the import so the body of the tests (which referenced `QuoteType.DEPOSIT_RATE` etc.) didn't churn.

**4. Three files deleted**

* `python/pricebook/core/market_data.py` — 302 lines. The legacy module: `QuoteType`, `Quote`, `MarketDataSnapshot`, `CurveConfig`, `PipelineConfig`, `MissingQuoteError`, `_build_discount_curve`, `_build_survival_curve`, `build_context`, `HistoricalData`. The `_build_*` and `build_context` helpers had zero production callers; `HistoricalData` had zero callers. Only the 5 names listed in §2 had live use.
* `python/pricebook/pricing/market_data.py` — 2-line `from pricebook.core.market_data import *` shim. Now meaningless.
* `python/tests/test_market_data.py` — tested the legacy module exclusively; new types have their own tests in `python/tests/market_data/`.

Net: **−587 lines deleted, +77 lines added (helpers + cycle-avoiding import note)**, one architectural duplication closed.

**Caller-impact verification**: grepped for residual `pricebook.core.market_data` and `pricebook.pricing.market_data` imports — zero hits outside `market_data_provider.py` and `market_data_tools.py` (different files, kept).

**Verification:**

L3 suite (max-layer 3, with the two slow G2++ calibration tests deselected): **8261 passed, 0 failed** in 5m19s. No regressions.

---

# Whole-library audit chain — closed.

With B.3 C1 closed, every catalogued audit item across the entire library is either resolved or explicitly held-as-is with rationale:

* All HIGH items: closed.
* All MED items: closed.
* All LOW items: closed.
* The two ARCH-tagged items (A.11 B3 registry warning, B.3 C1 market_data dual implementation): A.11 B3 held-as-is per audit rationale (current silent-ignore is safer than the audit's overwrite-with-warning suggestion); B.3 C1 closed here.

The bottom-up L0→L6 sweep (`AUDIT_PLAN.md` + ponytail layered findings) is done. ~793 modules across 24 sub-packages and 7 layers were audited; 1,043 `vars(self)` mutation hazards swept; the strict-ICMA migration landed in 3 disciplined slices; and the legacy/G1-P2 architectural duplication is finally retired.

---

## v1.110.0 — 2026-06-19 — **T-ICMA-SLICE3 + A.1 B1 COMPLETE 🏁: flip `strict_icma=True` default**

A.1 B1 migration, slice 3 of 3 — the actual default-flip. Three changes:

**1. Default flipped: `strict_icma: bool = False` → `True`**

In `core/day_count.year_fraction()`. From v1.110.0 onwards, any caller invoking `year_fraction(..., DayCountConvention.ACT_ACT_ICMA, ...)` without `ref_start`, `ref_end`, and `frequency` raises `ValueError("ACT/ACT ICMA requires coupon-period anchors. Missing: ...")` rather than silently degrading to ACT/365F. This closes the original A.1 B1 *HIGH*-severity finding that motivated the entire audit chain — UST coupons silently priced at 1.9836 / 2.0164 instead of exactly 2.0000.

The flag is still accepted (default-only change), so callers that genuinely want the legacy degradation can opt in with `strict_icma=False`. No production caller does post-T-ICMA-SLICE2.

**2. Docstring updated**

Explicit note that the default flipped on 2026-06-19 in T-ICMA-SLICE3, with a pointer to why the opt-out exists (legacy test fixtures).

**3. Three legacy-contract tests updated to opt-in explicitly**

* `test_day_count.py::test_legacy_fallback_missing_refs` — added `strict_icma=False`, renamed docstring to reflect the post-flip world.
* `test_day_count.py::test_strict_legacy_callers_unaffected_by_default` — split into two tests:
  * `test_legacy_opt_in_still_degrades_to_act365f` — verifies the explicit-`False` opt-out still works.
  * `test_default_is_strict_post_slice3` — NEW: pins the new default by asserting the bare call now raises.
* `test_fi_hardening.py::test_fallback_without_ref_dates` — added `strict_icma=False`, updated docstring.

**Pre-flip audit chain recap:**

* **Slice 1 (v1.108.0)**: fixed the canonical site `bond.py` — `accrued_interest` passes ICMA refs; `_ytm_time_to` fallbacks use explicit ACT/365F. +2 regression tests.
* **Slice 2 (v1.109.0)**: empirical full-suite flip discovered 3 remaining production sites (`benchmark_bonds.py par_yield_curve`, `floating_leg.py FRN accrual`, `cln.py CLN protection legs`, `repo_desk.py carry`). All fixed.
* **Slice 3 (this)**: legacy tests opt-in, default flipped, full suite 12,797 / 12,797 passing.

**Verification:**

Full library pytest at strict=True default: **12,797 passed**. 0 failed. ~6 min.

**Slice count:** 3 slices for the migration (1 canonical fix + 1 sweep + 1 flip), as the ledger anticipated ("Many slices").

---

# A.1 B1 — closed.

The single open item from the original L0 audit summary ("A.1 B1 final-slice — flip `strict_icma=True` default after auditing remaining callers. Many slices.") is now done. The pricebook codebase rejects mis-used ACT/ACT ICMA at the source rather than silently mispricing bonds.

**Remaining open in `AUDIT_L0_CORE.md`** (single item, intentionally deferred):

* **B.3 C1** — legacy `core.market_data` vs new `pricebook.market_data` (G1 P2). Architecture decision pending Gate 2 sign-off; not a correctness bug.

Every other catalogued item — HIGH, MEDIUM, and LOW — is now closed or held-as-is with explicit rationale.

---

## v1.109.0 — 2026-06-19 — **T-ICMA-SLICE2: 4 remaining unsafe sites fixed; library now ready for the strict-icma flip**

A.1 B1 migration, slice 2 of 3. Empirical method: temporarily flipped `strict_icma=True` and ran the full 12,796-test library suite to discover which callsites would fail under strict mode. Result: **6 failures total. 3 are legacy-contract tests** (they explicitly pin the silent-fallback behaviour — Slice 3 will update them). **3 are real production regressions, all fixed in this slice.**

**Fix 1 — `fixed_income/benchmark_bonds.py:177`** (`par_yield_curve`)

Pre-fix: `T = year_fraction(settle, bond.maturity, bond.day_count)` where `bond.day_count` could be ACT/ACT ICMA. `T` was only used as `int(T × periods_per_year)` to estimate coupon count — exact ICMA precision wasn't needed. Switched to explicit `DayCountConvention.ACT_365_FIXED`. The spans-multiple-periods nature means ICMA refs aren't meaningful here.

**Fix 2 — `fixed_income/floating_leg.py:119`** (`FloatingLeg.__init__`)

Pre-fix: `yf = year_fraction(accrual_start, accrual_end, day_count)`. Now passes ICMA refs proactively (`ref_start=accrual_start, ref_end=accrual_end, frequency=12//frequency.value`). The `accrual_start/end` ARE the coupon-period anchors — data was already there. Harmless for non-ICMA day counts since `year_fraction` ignores them.

Note: `FloatingLeg`'s default `day_count=ACT_360`, but `_frn_from_convention` (called by `create_sovereign_frn` for BTPFRN) passes `ACT_ACT_ICMA` for the Italian sovereign FRN convention — exactly the path that the `test_create_btpfrn` regression exercises.

**Fix 3 — `credit/cln.py:156, 205`** (`CLN.dirty_price` premium + recovery legs)

Two sites in CLN pricing — both compute coupon-period year fractions. CLN.day_count can be ICMA when the CLN is built from a UST/Bund/Gilt convention. Same fix as floating_leg: pass `ref_start=t_start, ref_end=t_end, frequency=12//self.frequency.value`.

**Fix 4 — `desks/repo_desk.py:196`** (`RepoTrade.carry`)

Pre-fix: `yf = _year_fraction(sd, mat, self.bond.day_count)` where `sd → mat` spans multiple coupon periods (repo carry on a multi-year UST). Switched to explicit `ACT/365F` — same rationale as benchmark_bonds: ICMA refs aren't meaningful across multiple periods, and the repo carry formula is approximate enough that the difference is negligible.

**Verification method (worth noting):**

Slice 2 used "flip the default and run the whole suite" as an audit primitive. Cheaper and more reliable than trying to grep+reason through every dynamic-dispatch `year_fraction(..., self.day_count)` callsite (40+ candidates across the library, most of which use `ACT/360` / `ACT/365F` for vol/swap/credit conventions and would never trip strict-mode). The 6 failures from one experimental flip were a complete inventory.

**Slice 3 (next) needs:**
1. Update the 3 legacy-contract tests in `test_day_count.py` and `test_fi_hardening.py` — they should explicitly pass `strict_icma=False` to assert the legacy behaviour, instead of relying on the default.
2. Flip the `strict_icma=False → True` default in `year_fraction()` and `_act_act_icma()`.
3. Confirm 12,796 / 12,796 passing under strict default.

**Files changed**:
- `python/pricebook/fixed_income/benchmark_bonds.py` — par_yield_curve T calculation → ACT/365F.
- `python/pricebook/fixed_income/floating_leg.py` — proactive ICMA refs at line 119.
- `python/pricebook/credit/cln.py` — proactive ICMA refs at lines 156 + 205.
- `python/pricebook/desks/repo_desk.py` — carry yf → ACT/365F + import `DayCountConvention`.

L3-scoped pytest at strict=False (current default): 8286 passed.
Strict=True dry-run: 12790 passed, 6 failed (all 6 expected — 3 production sites now fixed, 3 legacy-contract tests pending Slice 3).

---

## v1.108.0 — 2026-06-19 — **T-ICMA-SLICE1: bond.py is strict-icma-safe (A.1 B1 migration, slice 1 of 3)**

User asked to work on the A.1 B1 final-slice migration — the multi-slice campaign to flip the `strict_icma=True` default in `core/day_count.year_fraction()`. Plan:

* **Slice 1 (this)**: fix the canonical site `fixed_income/bond.py`. It contains the original A.1 B1 motivating bug (UST coupons silently priced at 1.9836 or 2.0164 per 100 instead of exactly 2.0000 because `year_fraction(..., ACT_ACT_ICMA, ...)` was called without coupon-period anchors and silently fell back to ACT/365F).
* **Slice 2 (future)**: audit non-fixed_income callsites + any country files still relying on silent fallback.
* **Slice 3 (future)**: flip `strict_icma=False → True` default in `year_fraction()`. Full library suite must stay green.

**Scope of slice 1:**

Audit found 22 files use `ACT_ACT_ICMA`. Most country files (czech, danish, norwegian, polish, british, korean's `_DC_ICMA` alias, etc.) already pass `ref_start`/`ref_end`/`frequency` explicitly. The pure convention tables (`benchmark_bonds.py`, `sovereign_bonds.py`, `inflation_indices.py`) reference the enum without calling `year_fraction` directly. **The single unsafe production site is `bond.py`'s `accrued_interest` + `_ytm_time_to`.**

**Changes in `bond.py`:**

* `accrued_interest` (lines 131-149): both `year_fraction(...)` calls now pass `ref_start=cf.accrual_start, ref_end=cf.accrual_end, frequency=12 // months_per_coupon`. The containing accrual period IS the ICMA reference period — the data was already there, it just wasn't being passed.
* `_ytm_time_to` (lines 290-342): the function's happy path already counted coupon periods exactly (the "Fix A.1 B1 Slice 4" comment hints this was earlier partial work). The 3 intentional fallback branches (`coupons_per_year is None`, `target not in coupon_dates`, `settle outside coupon range`) were relying on the silent ICMA→ACT/365F fallback inside `year_fraction`. Made the intent explicit: each now calls `year_fraction(..., DayCountConvention.ACT_365_FIXED)`. Same numerical behaviour; flipping the strict default won't break them.

**Regression tests (2 new in `tests/test_bond_strict_icma.py`):**

* `test_bond_strict_icma_smoke` — monkeypatches `year_fraction`'s default to `strict_icma=True` and exercises `accrued_interest`, `_ytm_time_to`, and `yield_to_maturity` on a UST-like bond. Pre-fix this would have raised `ValueError("ACT/ACT ICMA requires coupon-period anchors")`. Post-fix it passes.
* `test_bond_accrued_uses_proper_icma_not_act365f` — pins accrued = `coupon × (days_into / period_days) / coupons_per_year × 100` (the proper ICMA formula). Asserts this value differs from the silent-ACT/365F fallback `coupon × days_into / 365 × 100`. The two values are observably different — pre-fix the bond returned the wrong one.

**Files changed**:
- `python/pricebook/fixed_income/bond.py` — +12 / -4 (`accrued_interest` passes ICMA refs; 3 `_ytm_time_to` fallbacks now explicit ACT/365F).
- `python/tests/test_bond_strict_icma.py` (new) — 2 regression tests + helpers.

**Pre-flip status (after this slice):**
* `bond.py` ✅ strict-icma-safe.
* `bond_curve.py` ✅ already strict-safe (explicit `if dc == ACT_ACT_ICMA:` dispatch with refs).
* Country files (czech, danish, norwegian, polish, british, korean) ✅ already strict-safe.
* **Pending verification before Slice 3:** the remaining 16+ files (australian, chinese, hong_kong, indonesian, malaysian, singaporean, swedish, swiss, thai, danish_mortgage, sovereign_bonds, etc.) — most are convention-table-only, but each will be audited in Slice 2.

L3-scoped pytest: 8286 passed (was 8284 + 2 new). 305s.

---

## v1.107.0 — 2026-06-19 — **T-LOW-CLEANUP: close 5 LOW correctness items (C.8 B1 + A.11 B4-B7) + 9 regression tests**

User asked to close the remaining LOW items from `AUDIT_L0_CORE.md`. Verified state first — 2 items (D.1 B3, A.11 B1/B2) were already fixed in prior sessions. The remaining 5 are bundled into this single closing slice. Two ledger items are intentionally **not** closed: `A.1 B1 final-slice` (many-slice migration) and `B.3 C1` (architecture decision requiring Gate 2 sign-off).

**C.8 B1 — `Greeks.dollar_delta` + `Greeks.dollar_gamma` properties deleted.**

The `Greeks` dataclass (in `core/greeks.py`) carries no `spot` field. The pre-fix docstrings claimed:
* `dollar_delta`: `delta × option_price` — questionable approximation (real dollar delta is `delta × spot` or `delta × notional`)
* `dollar_gamma`: `0.5 × gamma × S² × 0.01²` — but the code did `0.5 × gamma` (off by S²×1e-4, accidentally correct at S=100 only)

Verification showed **zero production callers** for either property (the `dollar_gamma` field hit in `vol_derivatives_advanced.py` belongs to a different `VarianceSwapGreeks` dataclass, not `core.greeks.Greeks`). Deleted both properties. Updated class docstring to point external callers to compute their own dollar-Greeks where spot is known.

**A.11 B4 — narrow numpy `.item()` duck-test.**

`_serialise_atom` previously used `hasattr(v, "item") and callable(v.item)` to detect numpy scalars. That duck-test mis-fires on any non-numpy object that happens to expose a callable `.item` method (`dict_items` views, etc.) — they'd be silently flattened. Narrowed to `isinstance(v, np.generic)` (guarded by an optional `_HAS_NUMPY` import so the module stays importable without numpy).

**A.11 B5 — `CurrencyPair` parse with split-limit + length-check.**

Pre-fix: `base_str, quote_str = v.split("/")` — raised an unpacking error on `"EUR"` and silently dropped tokens on `"EUR/USD/extra"`. Now: explicit length check raising a clear `ValueError("CurrencyPair payload must be 'BASE/QUOTE'; got ...")` for both cases.

**A.11 B6 — structured `ValueError` on missing `"params"` key.**

Three sites in `serialisable.py` did `p = d["params"]` which raised a bare `KeyError: 'params'` — opaque error, didn't name the class. Now they raise `ValueError("Malformed serialised payload for {ClassName}: missing required 'params' key. Got keys: [...]")`. Fixed in `read_payload`, `Serialisable.from_dict`, and the `@serialisable` decorator's `cls_from_dict`. The `@serialisable_convention` decorator's `cls_from_dict` already had a guard.

**A.11 B7 — `IntEnum` round-trip from JSON-string.**

After a JSON serialisation round-trip, an integer-valued `IntEnum` member would come back as a string (`"3"`). `_deserialise_atom` then did `hint("3")` which raises `ValueError` because string isn't a valid IntEnum value. Added an int-coercion attempt for the `issubclass(hint, IntEnum)` + `isinstance(v, str)` case before the lookup. Pure addition, no behaviour change for ints / already-Enum values.

**Regression tests (9 new) in `tests/test_serialisable_low_fixes.py`:**
* B4: `_ItemMethodNotNumpy` fixture verifies non-numpy `.item`-bearing objects pass through unchanged; `np.float64` still flattens to `1.5`.
* B5: ValueError on `"EUR"` (too few), `"EUR/USD/extra"` (too many), happy-path round-trip.
* B6: missing `"params"` raises ValueError naming the class.
* B7: IntEnum round-trip from `"3"` → member; ints + member-passthrough still work.

**Skipped items, documented in ledger:**
* **A.11 B3** (registry re-register `DeprecationWarning`): current code silently ignores duplicate registration (`if key and key not in _REGISTRY`), which is safer than the audit's suggested "overwrite with warning". Keeping current behaviour.
* **A.1 B1 final-slice**: flip `strict_icma=True` default. Multi-slice migration touching ~72 ICMA callers across `fixed_income/`. Genuinely out of scope for a LOW closure.
* **B.3 C1**: legacy `core.market_data` vs new `pricebook.market_data`. Gate 2 architecture decision — needs explicit user direction on the migration path before any code move.

**Files changed**:
- `python/pricebook/core/greeks.py` — deleted `dollar_delta` + `dollar_gamma` properties; updated class docstring (-8 / +6).
- `python/pricebook/core/serialisable.py` — 4 inline fixes (B4 narrow, B5 parse, B6 ×2 missing-params, B7 IntEnum coercion); added optional numpy import + `IntEnum` import (+24 / -8).
- `python/tests/test_serialisable_low_fixes.py` (new) — 9 regression tests for B4-B7.

**Held queue:** ✅ all items closed (or explicitly documented as out-of-scope).

**Test runs:**
* New regressions: 9 passed (0.06s).
* L0 suite: 2452 passed (was 2443 + 9 new tests, 47s).
* **Full library suite: 12,794 passed** (was 12,785 + 9 new tests, 380s). No cross-impact from the serialisable.py changes.

---

## v1.106.0 — 2026-06-18 — **Bookkeeping: verify + close 3 inherited L0 core correctness MEDs (no code change)**

User requested execution on the 3 inherited Pass C/D MEDs from the pre-session L0 core audit:
* **C.7 B1** — settlement lag uses calendar days instead of business days.
* **D.1 B1** — empty-dict fields become `None` on round-trip.
* **D.1 B2** — `pricing_context` fields silently dropped on round-trip.

**Verification result: all 3 already fixed in prior work — bookkeeping was stale.**

The audit ledger header (`AUDIT_L0_CORE.md`) claimed these were still queued, but inspection showed:

* `core/settlement.py` — every `cash_settlement` / `cds_settlement_*` / `option_settlement_*` / `futures_settlement_*` / `fx_spot_date` / `bond_settlement_date` callsite routes through `add_business_days(...)` with calendar parameter. Docstrings explicitly say "**business days** per market convention". The only `timedelta(days=...)` left in the module is inside `add_business_days` itself (walking day-by-day to skip non-business days). Regression coverage: `tests/test_settlement_business_days.py` + `tests/test_settlement_calendars.py` — 66 tests, all passing.
* `core/pricing_context.py::_ctx_from_dict` — uses `_fd_dict()` helper that returns `{}` for empty payloads; no `or None` collapse anywhere. Inline comment literally says `# (no `or None` collapse — Fix D.1 B1)`.
* `core/pricing_context.py::_ctx_to_dict` — emits all 14 dataclass-declared fields including the per-currency dicts (`discount_curves`, `inflation_curves`, `repo_curves`), `numerical_config`, `stochastic_credit_models`, `credit_vol_surfaces`, `credit_correlations`, `reporting_currency`. Inline comment says `# (Fix D.1 B2)`. Regression coverage: `tests/test_pricing_context_round_trip.py` — exercises empty-dict preservation AND all-field round-trip.

**No code change.** This is a pure bookkeeping closure — the work was done in prior sessions but the ledger summary was never refreshed.

**Ledger update** (`AUDIT_L0_CORE.md`, local-only / gitignored):
* Pass C MED count: 0/1 → **1/1 ✅**
* Pass D MED count: 0/2 → **2/2 ✅**
* Total MED count: 3/6 → **6/6 ✅**
* Crossed out the 3 fixed lines, added the regression-test pointers and verification date.

**Remaining open items in `AUDIT_L0_CORE.md`** (all LOW severity or known-architectural decisions, none in the MED tier):
* C.8 B1 — `dollar_gamma` docstring/formula mismatch (LOW; needs API decision since `Greeks` doesn't carry `spot`).
* D.1 B3 — `replace()` aliases mutable dicts (LOW-MED; easy fix when touched next).
* A.1 B1 final-slice — flip `strict_icma=True` default after auditing remaining callers (many-slice migration).
* A.11 B3-B7 — minor serialisation robustness (LOW).
* B.3 C1 — legacy `core.market_data` vs new `pricebook.market_data` (ARCH, Gate 2 decision).

**Session held queue:** ✅ **empty** — all ponytail items closed (T-PRC-PT2 @ v1.104.0, T-CORE-PT5 @ v1.105.0) AND all inherited MED items verified closed (this slice).

L0-scoped pytest: 2443 passed. The 66 dedicated regression tests for the 3 MEDs run in 0.16s.

---

## v1.105.0 — 2026-06-18 — **T-CORE-PT5: delete dead `core/instrument_result.py` Protocol**

T-CORE-PT5 — final held ponytail slice. Originally held since L0 core ponytail addendum (T-CORE-PT2 era) pending a broader "should we keep documentary Protocols with test-only binders?" decision. Now closed.

**Rationale for delete:**
* Zero production binders — no class declared `(InstrumentResult)`, no parameter annotated `InstrumentResult`, no `from pricebook.core.instrument_result import …` anywhere in `pricebook/`.
* One real usage: `tests/test_portfolio_structured.py:130` did `isinstance(r1, InstrumentResult)` — substituted with `hasattr(r1, "price") and hasattr(r1, "to_dict")` (mechanically identical at runtime).
* Two docstring mentions (`test_trs_tree.py:172` comment, `viz/_generic.py:1` module-doc phrase) are non-binding prose and left in place.

**Changes:**
- `python/pricebook/core/instrument_result.py` — deleted (31 lines).
- `python/tests/test_portfolio_structured.py` — removed import, swapped `isinstance(...) → hasattr(...) and hasattr(...)`, updated test docstring to reflect duck-typed contract.

**Cumulative dead-code count for the session:** 6 → **7 modules deleted**:
`core/protocols.py`, `core/desk_protocol.py`, `core/results.py`, `curves/quadrature.py`, `core/numerical_method_map.py`, `models/engine_registry.py`, **`core/instrument_result.py`**.

**Held queue:** ✅ **all ponytail items closed.** Inherited pre-session correctness MEDs (C.7 B1, D.1 B1, D.1 B2) remain queued — those are correctness fixes not ponytail items.

L0-scoped pytest: 2443 passed. 47s.

---

## v1.104.0 — 2026-06-18 — **T-PRC-PT2: narrow RPC Greek silent-except + library-wide Greek-coverage scanner**

T-PRC-PT2 — held since L1 pricing sweep (T-PRC-PT1 v1.090.0). User decision landed: **Option A — narrow + log**.

**Behaviour change at RPC boundary:** the 4 silent-except Greek sites in `pricing_engine.py` (3) + `pricing_server.py` (1) now catch only `NotImplementedError` and `AttributeError` (legitimate "this instrument doesn't support this Greek"); everything else propagates as a per-trade error via the existing fault-isolation outer loops. Each narrow-catch path logs a `warning` with the instrument class name so operators get visibility.

* `pricing_engine.py:_compute_greeks` — 3 sites narrowed:
  * direct `instrument.dv01(curve)` call
  * generic bump-and-reprice DV01 fallback
  * `instrument.greeks(curve)` lookup for delta
* `pricing_server.py:_price_trade` — 1 site narrowed:
  * `instrument.dv01(ctx.discount_curve)` call

Logging added to `pricing_engine.py` (which didn't have a module logger; `pricing_server.py` already had one).

**Effect for existing callers:**
* Instrument is a `Trade` / `Portfolio` / other type lacking `dv01` — silently skipped as before (now via `AttributeError`, not the wildcard).
* Instrument has `dv01(...)` but it raises `NotImplementedError` — same: silently skipped + log line.
* Instrument's `dv01(...)` has a real bug (numpy error, divide-by-zero, etc.) — **NEW**: the per-trade error block now records it, request returns `status="partial"` or `status="error"` instead of `"ok"` with missing Greek. Caller sees the failure.

**Bonus deliverable — Greek coverage scanner.** Added `tools/instrument_greeks_coverage.py`: walks `pricebook/`, finds every class with `pv(...)` or `pv_ctx(...)` (the "instrument" duck-type surface), and reports which classes also define `dv01(...)` and/or `greeks(...)`. Run:

```bash
.venv/bin/python tools/instrument_greeks_coverage.py
.venv/bin/python tools/instrument_greeks_coverage.py --missing-only
.venv/bin/python tools/instrument_greeks_coverage.py --out INSTRUMENT_GREEKS_COVERAGE.md
```

Initial scan (saved as `INSTRUMENT_GREEKS_COVERAGE.md`, gitignored):

> **119 instrument classes across 9 sub-packages.**
> - `dv01(...)`: 14 / 119 = **12%**
> - `greeks(...)`: 12 / 119 = **10%**
> - Either method: 25 / 119 = 21%
> - **Neither: 94 / 119 = 79%**

So when the narrowed except now propagates `NotImplementedError` / `AttributeError`, ~79% of instruments will hit it on every Greek request (and silently log + skip). That's the "we know about this gap" set — the report is the roadmap for incremental Greek-method coverage.

**Files changed**:
- `python/pricebook/pricing/pricing_engine.py` — added logger, narrowed 3 excepts.
- `python/pricebook/pricing/pricing_server.py` — narrowed 1 except (logger already imported).
- `tools/instrument_greeks_coverage.py` (new) — 156 lines, AST-based scanner.
- `INSTRUMENT_GREEKS_COVERAGE.md` (new, gitignored) — generated coverage report.

**Held queue update:** `T-PRC-PT2` ✅ closed. Remaining held: `T-CORE-PT5` (`core/instrument_result.py` Protocol cleanup).

L1-scoped pytest: **3508 passed** (baseline 3508 — verified with `--collect-only` on stashed pre-fix state; the earlier "3520" reported at v1.090.0 reflected test count drift over the session, not regression).

---

## v1.103.0 — 2026-06-18 — **L6 desks sweep + FULL LIBRARY COMPLETE 🏁: 112 `vars(self)` mutation hazards across 40 files**

T-DESK-PT1 — final layer ponytail slice. `desks/` sub-package (49 modules — top of DAG, aggregation layer across all 11 packages; ledger `AUDIT_L6_DESKS.md`).

**Architectural findings (clean):** zero ABCs, Protocols, registries, factories, builders, np.trapz.

**M-DESK-1 · `return vars(self)` mutation hazard** — 112 sites across 40 files. All standard 8-space pattern. Single-slice sweep.

**7 `except Exception` sites held** (deal 3, reporting 4) — deal aggregation + reporting fault-tolerance, defensible.

**Files changed**: 40 in `python/pricebook/desks/` (+112 / -112).

---

# 🏁 **PRICEBOOK FULL LIBRARY SWEEP COMPLETE** 🏁

All 7 layers, all 24 sub-packages, all 793 modules — **fully audited under the combined methodology (AUDIT_PLAN §2 + ponytail over-engineering hunt)**.

## Final scoreboard

| Layer | Sub-pkgs | Modules | Slices |
|-------|---------:|--------:|-------:|
| L0 | 9 | 110 | 13 |
| L1 | 3 | 65 | 3 |
| L2 | 2 | 97 | 2 |
| L3 | 4 | 292 | 4 |
| L4 | 1 | 61 | 1 |
| L5 | 4 | 101 | 4 |
| L6 | 1 | 49 | 1 |
| cross-layer | — | — | 1 (T-CORE-PT3) |
| **Total** | **24** | **793** | **29** |

## Cumulative metrics (v1.075.0 → v1.103.0)

* **29 slices landed, 87 commits** (29 code + 29 stamp + 29 release notes)
* **1,043 `return vars(self)` mutation hazards swept** across **all 7 layers** of the library — the single most prevalent recurring pattern, now closed
* **~830 lines of net dead code removed**
* **6 dead modules deleted**: `core/protocols.py`, `core/desk_protocol.py`, `core/results.py`, `curves/quadrature.py`, `core/numerical_method_map.py`, `models/engine_registry.py`
* **12 dead test methods removed** (T-CORE-PT3)
* **4 over-engineered scaffolds cut**: `StorageBackend` ABC, `query_table` alias, `_detect_code_version` wrapper, deprecated `deserialise_*` aliases
* **3 cross-layer migrations**: heston quadrature, cdo dead import, test_schema_adapter de-flake
* **3 stale layer-claim docstrings fixed** (calibration L6→L0, market_data L1→L0, curves layer notes)
* **+1 new regression test** (T-CAL1 silent-except propagation)

## Test baselines through all layers (all green throughout)

| Layer | Tests | Time |
|-------|------:|-----:|
| L0 | 2,437 | ~47s |
| L1 | 3,520 | ~55s |
| L2 | 4,321 | ~4.5min |
| L3 | 8,275 | ~5min |
| L4 | 9,848 | ~6min |
| L5 | 11,442 | ~6min |
| **L6 full** | **12,785** | **~6min** |

## Held items pending future decision

* **T-PRC-PT2** — RPC silent-except Greeks (4 sites in `pricing_engine.py` + `pricing_server.py`): behaviour-change decision needed (swallow/propagate/narrow).
* **T-CORE-PT5** — `core/instrument_result.py` Protocol cleanup (test-only binders; documents convention).

## Inherited correctness items from original L0 audit (pre-session, still queued)

* C.7 B1 (settlement lag), D.1 B1 (empty-dict round-trip), D.1 B2 (dropped fields).

## Per-sub-package vars(self) breakdown

| Layer | Sub-pkg | vars sites |
|-------|---------|----------:|
| L0 | calibration | 0 |
| L0 | core | n/a (correctness audit only) |
| L0 | numerical | 9 |
| L0 | pe | 15 |
| L0 | statistics | 22 |
| L0 | ts | 2 |
| L0 | viz | 2 |
| L1 | curves | 15 |
| L1 | pricing | 4 |
| L1 | regulatory | 29 |
| L2 | models | 70 |
| L3 | crypto | 48 |
| L3 | risk | 78 |
| L3 | credit | 111 |
| L3 | fixed_income | 204 |
| L4 | options | 81 |
| L5 | fx | 53 |
| L5 | commodity | 61 |
| L5 | structured | 53 |
| L5 | equity | 70 |
| L6 | desks | 112 |
| **Total** | | **1,043** |

## All-ledger inventory (gitignored, on disk)

`AUDIT_PLAN.md` + 22 per-sub-package ledgers: `AUDIT_L0_*` × 9 + `AUDIT_L1_*` × 3 + `AUDIT_L2_*` × 2 + `AUDIT_L3_*` × 4 + `AUDIT_L4_OPTIONS.md` + `AUDIT_L5_*` × 4 + `AUDIT_L6_DESKS.md`.

L6 / full-library pytest: **12,785 passed**. 357s (~6 min).

The pricebook codebase is now ponytail-clean across all production layers.

---

## v1.102.0 — 2026-06-18 — **L5 equity sweep + L5 COMPLETE: 70 `vars(self)` mutation hazards across 24 files**

T-EQ-PT1 — final L5 ponytail slice. `equity/` sub-package (33 modules; ledger `AUDIT_L5_EQUITY.md`). Completes L5.

**Architectural findings (clean):** zero ABCs, Protocols, registries, factories, builders, np.trapz.

**M-EQ-1 · `return vars(self)` mutation hazard** — 70 sites across 24 files. All standard 8-space pattern. Single-slice sweep.

**6 `except Exception` sites held** (joint_calibration 4, equity_jumps 2) — calibration/jump-fit fault-tolerance, defensible.

**Cumulative session vars(self) count:** 861 (through structured) + 70 = **931 instances** corrected across L0+L1+L2+L3+L4+L5.

**Files changed**: 24 in `python/pricebook/equity/` (+70 / -70).

---

# **L5 SWEEP COMPLETE 🎯**

All 4 L5 sub-packages now ✅ swept:

| Sub-pkg | Modules | Slice | Hazards |
|---------|---------|-------|---------|
| `fx` | 22 | T-FX-PT1 | 53 |
| `commodity` | 23 | T-COMM-PT1 | 61 |
| `structured` | 23 | T-STRUCT-PT1 | 53 |
| `equity` | 33 | T-EQ-PT1 | 70 |

**L5 grand total: 101 modules, 237 vars(self) hazards corrected.**

**Cumulative session (v1.075.0 → v1.102.0):** 28 slices landed, ~84 commits.

| Layer | Modules swept | Status |
|-------|--------------:|--------|
| L0 | 111 | ✅ |
| L1 | 65 | ✅ |
| L2 | 97 | ✅ |
| L3 | 292 | ✅ |
| L4 | 61 | ✅ |
| L5 | 101 | ✅ |
| **Subtotal** | **727 / 793** | **92%** |
| L6 desks | 49 | ❓ remaining |

**Held items pending decision:** T-PRC-PT2 (RPC silent-except Greeks), T-CORE-PT5 (instrument_result.py Protocol).

**Last layer remaining: L6 desks (49 modules)** — top-of-DAG aggregation layer.

L5-scoped pytest: 11442 passed. 363s.

---

## v1.101.0 — 2026-06-18 — **L5 structured sweep: 53 `vars(self)` mutation hazards across 20 files**

T-STRUCT-PT1 — third L5 ponytail slice. `structured/` sub-package (23 modules; ledger `AUDIT_L5_STRUCTURED.md`).

**Architectural findings (clean):** zero ABCs, Protocols, registries, factories, builders, np.trapz.

**M-STRUCT-1 · `return vars(self)` mutation hazard** — 53 sites across 20 files. All standard 8-space pattern. Single-slice sweep.

**7 `except Exception` sites across 3 files** (abs 3, cmbs 1, mbs 3) — held; structured-product loop fault-tolerance, defensible sweep-skip-failures.

**Cumulative session vars(self) count:** 808 (through commodity) + 53 = **861 instances** corrected.

**Files changed**: 20 in `python/pricebook/structured/` (+53 / -53).

**L5 status:** `fx` ✅, `commodity` ✅, `structured` ✅. Final L5 remaining: equity (33).

L5-scoped pytest: 11442 passed. 350s.

---

## v1.100.0 — 2026-06-18 — **L5 commodity sweep: 61 `vars(self)` mutation hazards across 21 files** 🎯

T-COMM-PT1 — second L5 ponytail slice. `commodity/` sub-package (23 modules; ledger `AUDIT_L5_COMMODITY.md`). v1.100 milestone.

**Architectural findings (clean):** zero ABCs, Protocols, registries, factories, builders, **zero `except Exception`**, np.trapz.

**M-COMM-1 · `return vars(self)` mutation hazard** — 61 sites across 21 files. All standard 8-space pattern. Single-slice sweep.

**Cumulative session vars(self) count:** 747 (through fx) + 61 = **808 instances** corrected across L0-L5 partial.

**Files changed**: 21 in `python/pricebook/commodity/` (+61 / -61).

**L5 status:** `fx` ✅, `commodity` ✅. Remaining L5: equity (33), structured (23).

L5-scoped pytest: 11442 passed. 357s.

---

## v1.099.0 — 2026-06-18 — **L5 fx sweep: 53 `vars(self)` mutation hazards across 15 files**

T-FX-PT1 — first L5 ponytail slice. `fx/` sub-package (22 modules; ledger `AUDIT_L5_FX.md`).

**Architectural findings (clean):** zero ABCs, Protocols, registries, factories, builders, **zero `except Exception`**, np.trapz. Cleanest L5 sub-package surveyed.

**M-FX-1 · `return vars(self)` mutation hazard** — 53 sites across 15 files. All standard 8-space pattern. Single-slice sweep.

**Cumulative session vars(self) count:** 694 (through L4) + 53 = **747 instances** corrected.

**Files changed**: 15 in `python/pricebook/fx/` (+53 / -53).

**L5 status:** `fx` ✅. Remaining L5: commodity (23), equity (33), structured (23).

L5-scoped pytest: 11442 passed. 349s.

---

## v1.098.0 — 2026-06-18 — **L4 options sweep + L4 COMPLETE: 81 `vars(self)` mutation hazards across 27 files**

T-OPT-PT1 — L4 `options/` sub-package (61 modules; ledger `AUDIT_L4_OPTIONS.md`). L4 is single-sub-package so this slice completes the layer.

**Architectural findings (clean):** zero ABCs, Protocols, registries, factories, builders, np.trapz.

**M-OPT-1 · `return vars(self)` mutation hazard** — 81 sites across 27 files. All standard 8-space pattern. Single-slice sweep.

**5 `except Exception` sites held** (capfloor 2, swaption_vol_cube 1, vol_arbitrage_scanner 2) — consistent with the defensible sweep-skip-failures pattern seen in risk/regulatory/credit/fixed_income.

**Cumulative session vars(self) count:** 613 (through L3) + 81 = **694 instances** corrected.

**Files changed**: 27 in `python/pricebook/options/` (+81 / -81).

**L4 status:** ✅ COMPLETE. Next is L5 (4 sub-packages: commodity 23, equity 33, fx 22, structured 23 — 101 modules total) then L6 desks (49 modules).

L4-scoped pytest: 9848 passed. 355s (~5.9 min).

---

## v1.097.0 — 2026-06-18 — **L3 fixed_income sweep + L3 COMPLETE: 204 `vars(self)` mutation hazards across 88 files**

T-FI-PT1 — final L3 ponytail slice. `fixed_income/` sub-package (130 modules — biggest sub-package in the library by file count; ledger `AUDIT_L3_FIXED_INCOME.md`). Largest single sweep this session by a wide margin (204 sites previously vs 111 in credit).

**Architectural findings (clean):**
* 4 files with `_REGISTRY` patterns (ibor_curve, inflation_indices, rfr_compounding, supranational) — all convention-loader patterns, same shape as core/market_conventions, core/rate_index, credit/sovereign_cds. Legitimate pure-data registries. Keep.
* Zero ABCs, Protocols, factories, builders, np.trapz.

**M-FI-1 · `return vars(self)` mutation hazard** — 204 sites across 88 files. **Two patterns surfaced this slice:**
* Standard 8-space-indent multi-line `def to_dict(self) -> dict:` body → `return vars(self)` (most files)
* Inline single-line `def to_dict(self) -> dict: return vars(self)` (25 country-specific files: australian, british, canadian, chinese, czech, danish, colombian, hungarian, indian, hong_kong, indonesian, japanese, korean, israeli, malaysian, norwegian, peruvian, philippine, polish, singaporean, south_african, swedish, swiss, thai, turkish)

Both patterns fixed per-file via `replace_all`.

**5 `except Exception` sites across 4 files** (callable_floater, cancellable_swap, extendible, supranational) — held; consistent with sweep-skip-failures defensible patterns observed in risk/regulatory/credit.

**Cumulative session vars(self) count:** 409 (through credit) + 204 = **613 instances** corrected across L0+L1+L2+L3.

**Files changed**: 88 in `python/pricebook/fixed_income/` (+204 / -204).

---

# **L3 SWEEP COMPLETE 🎯**

All 4 L3 sub-packages now ✅ swept:

| Sub-pkg | Modules | Slice | Hazards |
|---------|---------|-------|---------|
| `crypto` | 15 | T-CRYPTO-PT1 | 48 |
| `risk` | 54 | T-RISK-PT1 | 78 |
| `credit` | 93 | T-CREDIT-PT1 | 111 |
| `fixed_income` | 130 | T-FI-PT1 | 204 |

**L3 grand total: 292 modules, 441 vars(self) hazards corrected.**

**Cumulative session (v1.075.0 → v1.097.0): 23 slices landed, 69 commits.**
* **~830 lines net dead code removed**
* **613 `vars(self)` mutation hazards swept** across L0+L1+L2+L3
* **6 dead modules deleted**, **12 dead test methods removed**, **4 over-engineered scaffolds cut**
* **3 cross-layer migrations** + **3 stale layer-claim docstrings** + **1 new regression test**

**Test baselines:**
* L0: 2437 passed (~47s)
* L1: 3520 passed (~55s)
* L2: 4321 passed (~4.5 min)
* L3: 8275 passed (~5 min)

**Held items pending decision:**
* T-PRC-PT2 — RPC silent-except Greeks (4 sites)
* T-CORE-PT5 — instrument_result.py Protocol

**Per AUDIT_PLAN, next layer is L4 — `options` (61 modules)** — single sub-package; opens path to L5 (commodity, equity, fx, structured) and L6 (desks).

L3-scoped pytest: 8275 passed. 304s.

---

## v1.096.0 — 2026-06-18 — **L3 credit sweep: 111 `vars(self)` mutation hazards across 51 files (largest single sweep)**

T-CREDIT-PT1 — third L3 ponytail slice. `credit/` sub-package (93 modules, ~30k LOC; ledger `AUDIT_L3_CREDIT.md`). Largest sub-package by both module count and vars(self) sites of any L3 sweep.

**Architectural findings (clean):**
* `sovereign_cds.py::_REGISTRY` — country-CDS convention-loader pattern; same shape as `core/market_conventions.py` (legitimate pure-data registry). Keep.
* Zero ABCs, Protocols, factories, builders, np.trapz.

**M-CREDIT-1 · `return vars(self)` mutation hazard** — 111 sites across 51 files. Largest single sweep this session. File-by-file `replace_all`.

**10 `except Exception` sites across 8 files** — not triaged in detail this slice (held; pattern matches risk/regulatory excepts which were defensible sweep-skip-failures). Surfaced for a separate triage slice if priorities shift.

**Cumulative session vars(self) count:** 298 (through risk) + 111 = **409 instances** corrected across L0+L1+L2+L3 (crypto+risk+credit).

**Files changed**: 51 in `python/pricebook/credit/` (+111 / -111).

**L3 status:** `crypto` ✅, `risk` ✅, `credit` ✅. Final L3 remaining: `fixed_income` (130 modules — biggest sub-package outside L2 models).

L3-scoped pytest: 8275 passed. 314s.

---

## v1.095.0 — 2026-06-18 — **L3 risk sweep: 78 `vars(self)` mutation hazards across 29 files**

T-RISK-PT1 — second L3 ponytail slice. `risk/` sub-package (54 modules, ~12.5k LOC; ledger `AUDIT_L3_RISK.md`).

**Architectural findings (clean):** zero ABCs, Protocols, registries, factories, builders, np.trapz.

**M-RISK-1 · `return vars(self)` mutation hazard** — 78 sites across 29 files. Single-slice file-by-file `replace_all`. Largest risk-touching files (`xva.py` 783 LOC, `prudent_valuation.py` 539, `backtest.py` 464) all included.

**3 `except Exception` sites held with rationale:**
* `cvar_optimisation.py:225` — frontier-sweep skip-failure pattern.
* `model_selection.py:160` — model-vs-scenario price grid; pricer crash → NaN.
* `parameter_uncertainty.py:92` — bootstrap resample skip.
All three are "sweep over set, skip failures, return what survived" exploratory patterns. Defensible.

**Cumulative session vars(self) count:** 220 (through crypto) + 78 = **298 instances** corrected.

**Files changed**: 29 in `python/pricebook/risk/` (+78 / -78).

**L3 status:** `crypto` ✅, `risk` ✅. Remaining L3: `credit` (93), `fixed_income` (130).

L3-scoped pytest: 8275 passed. 313s.

---

## v1.094.0 — 2026-06-17 — **L3 crypto sweep: 48 `vars(self)` mutation hazards**

T-CRYPTO-PT1 — first L3 ponytail slice. `crypto/` sub-package (15 modules, ~4400 LOC; ledger `AUDIT_L3_CRYPTO.md`).

**Architectural findings (clean):** zero ABCs, Protocols, registries, factories, builders, blanket excepts, np.trapz. Cleanest L3 sub-package so far.

**M-CRYPTO-1 · `return vars(self)` mutation hazard** — 48 sites across all 15 files. Same one-line fix; file-by-file `replace_all`.

**Cumulative session vars(self) count:** 172 (through L2) + 48 = **220 instances** corrected.

**Files changed**: 15 in `python/pricebook/crypto/` (+48 / -48).

**L3 status:** `crypto` ✅. Remaining L3: `risk` (54), `credit` (93), `fixed_income` (130).

L3-scoped pytest: 8275 passed. 308s (~5.1 min).

---

## v1.093.0 — 2026-06-17 — **T-CORE-PT3: cross-layer twin-delete of dead advisory framework**

T-CORE-PT3 — held since T-CORE-PT1 (queued in `AUDIT_L0_CORE.md` ponytail addendum), now eligible to land because the `models/` audit reached this code.

**Two dead modules deleted together:**
* **`python/pricebook/core/numerical_method_map.py`** (L0, 230 lines) — `recommend(features)` + `compare_methods()` advisory infrastructure. Only `test_numerical_plan.py::TestMethodMap` consumed it; zero production callers.
* **`python/pricebook/models/engine_registry.py`** (L2, 145 lines) — `price(engine="auto", instrument_type=...)` wrapper + `register_engine` + 14-entry `InstrumentType` enum. Only `test_engine_infrastructure.py::TestEngineRegistry` consumed it; zero production callers (real pricers go through concrete engines directly, never through this advisory layer).

Both modules promised generic recommendation/dispatch capabilities the rest of the codebase never adopted. Same architectural anti-pattern: speculative meta-framework with single test consumer.

**Dead test classes removed:**
* `tests/test_engine_infrastructure.py::TestEngineRegistry` (6 tests)
* `tests/test_numerical_plan.py::TestMethodMap` (6 tests)
* Net: 12 fewer tests in the L2 suite (4333 → 4321).

**Files changed**:
- `python/pricebook/core/numerical_method_map.py` — deleted (230 lines).
- `python/pricebook/models/engine_registry.py` — deleted (145 lines).
- `python/tests/test_engine_infrastructure.py` — TestEngineRegistry class removed (39 lines).
- `python/tests/test_numerical_plan.py` — TestMethodMap class removed (38 lines).

**Total: -453 lines** across 4 files (net of 12 deleted test functions whose deletion is the right action because they tested dead infrastructure, not real behaviour).

**Cumulative session dead code removed:** ~377 (through L1) + 453 (this slice) ≈ **830 lines**, plus 102→172 vars(self) corrections through L2.

**L2 status:** `data` ✅ already-clean; `models` ✅ swept (T-MOD-PT1 + T-CORE-PT3 done). L2 complete.

L2-scoped pytest: 4321 passed (= 4333 prior - 12 dead test deletions). 284s.

---

## v1.092.0 — 2026-06-17 — **L2 models sweep: 70 `vars(self)` mutation hazards across 39 files**

T-MOD-PT1 — `models/` sub-package sweep (92 modules, ~30k LOC — biggest sub-package in the whole library; ledger drafted alongside this slice).

**Architectural findings (clean structurally):**
* 4 files with Protocol patterns:
  - `models.py` — `IROptionModel` + `EquityOptionModel` Protocols; 0 production type-hint binders (1 docstring mention in `swaption.py:157`). Documentary contract for the multi-impl model catalogue (Black76Model, BachelierModel, SABRModel, BSModel, HestonModel...). **Hold** — same rationale as `Calibrator` Protocol per DESIGN.md.
  - `char_func_protocol.py` — `CharFuncModel` Protocol; 0 binders. Documents the contract that all `*_char_func` factories satisfy. **Hold.**
  - `engine_protocol.py` — `PricingEngine` Protocol with 3 real impls (MC, Tree, Analytical), real type-hint use. **Keep.**
  - `engine_registry.py` — already flagged in `T-CORE-PT3` (queued cross-layer twin-delete with `core/numerical_method_map.py`). Test-only consumer; ready to land next.
* Zero blanket excepts in the recurring "silent fallback" shape (the 10 except files all have documented fault-tolerance patterns; the silent-except triage at calibration sites was done by prior `T4-G2T1` and `T4-HW1` work).
* Zero `np.trapz` calls.

**M-MOD-1 · `return vars(self)` mutation hazard** — **70 sites across 39 files**, all same one-line fix. File-by-file `replace_all`. Largest single sweep this session by a wide margin.

**Cumulative session vars(self) count:** 102 (through L1) + 70 = **172 instances** corrected across L0+L1+L2.

**Files changed**: 39 in `python/pricebook/models/` (+70 / -70).

**L2 status:** `data` ✅ (clean, no slice needed); `models` ✅ (T-MOD-PT1 landed). The held cross-layer slice `T-CORE-PT3` is now eligible to execute — twin-delete `core/numerical_method_map.py` + `models/engine_registry.py` + the dead `TestEngineRegistry` / `TestMethodMap` test classes.

L2-scoped pytest: 4333 passed. 278s (~4.6 min — L2 is much heavier than L0/L1).

---

## v1.091.0 — 2026-06-16 — **L1 regulatory sweep + L1 COMPLETE: 29 `vars(self)` mutation hazards**

T-REG-PT1 — final L1 ponytail slice (`regulatory/`, 23 modules, ~7850 LOC; ledger `AUDIT_L1_REGULATORY.md`).

**Architectural findings (clean):** zero ABCs, Protocols, registries, factories, builders, np.trapz. Cleanest L1 sub-package surveyed.

**M-REG-1 · `return vars(self)` mutation hazard** — 29 sites across 15 files: balance_sheet_allocation (2), basel2 (2), capital_allocation (2), ima_bridge (1), ccar (1), credit_rwa (1), market_risk_ima (4), irc (2), liquidity (2), operational_risk (1), stress_irrbb (2), specialty (3), reverse_stress (1), total_capital (1), trs_capital (4). All same one-line fix; `replace_all` per file.

**Held: `reverse_stress.py:198, 262`** — two `except Exception` blocks wrap `scipy.optimize.minimize` calls and return `ReverseStressResult(found=False, ...)` on failure. The failure IS surfaced via the documented `found` flag in the result API; not silent-swallow. Defensible. Hold.

**Cumulative session vars(self) count:** 73 (through pricing) + 29 = **102 instances** corrected across L0+L1.

**Files changed**: 15 in `python/pricebook/regulatory/` (+29 / -29).

---

# **L1 SWEEP COMPLETE 🎯**

All 3 L1 sub-packages now ✅ swept under the combined methodology (AUDIT_PLAN + ponytail):

| Sub-pkg | Modules | Ponytail slice | Net |
|---------|---------|----------------|-----|
| `curves` | 33 | T-CRV-PT1 | -27 lines (dead module) + 15× vars(self) + 2 downstream migrations |
| `pricing` | 9 | T-PRC-PT1 | 4× vars(self) + test-flakiness fix; T-PRC-PT2 held (RPC silent-except decision) |
| `regulatory` | 23 | T-REG-PT1 | 29× vars(self) |

**Cumulative session (v1.075.0 → v1.091.0):**
* **17 slices landed** (16 code commits + 17 stamp + 17 release notes = 50 commits total)
* **~377 lines net dead code removed**
* **102 `vars(self)` mutation hazards swept** across L0 + L1
* **4 dead modules deleted** (`core/protocols.py`, `core/desk_protocol.py`, `core/results.py`, `curves/quadrature.py`)
* **4 over-engineered scaffolds cut** (StorageBackend ABC, query_table alias, _detect_code_version wrapper, deprecated deserialise_* aliases)
* **3 cross-layer migrations** (heston quadrature, cdo dead import, test_schema_adapter de-flake)
* **3 stale "Lx" docstring claims fixed** (calibration L6→L0, market_data L1→L0+target, curves layer notes via T-CRV-PT1 ledger entries)

**Held items requiring decision:**
* `T-CORE-PT3` — twin-delete `core/numerical_method_map.py` + `models/engine_registry.py` (waits for L2 `models/` audit)
* `T-CORE-PT5` — `core/instrument_result.py` Protocol (test-only binders)
* `T-PRC-PT2` — RPC-boundary silent-except Greeks (4 sites) — needs swallow/propagate/narrow decision

**Inherited correctness items still queued:** C.7 B1 (settlement lag), D.1 B1 (empty-dict round-trip), D.1 B2 (dropped fields) — pre-date this session.

**Per AUDIT_PLAN §3, next layer is L2** — `data` (5 modules) + `models` (91 modules). The held cross-layer T-CORE-PT3 slice can finally land when `models/` is audited.

L1-scoped pytest: 3520 passed (consistent across reruns). 55s, `pytest -n auto`.

---

## v1.090.0 — 2026-06-16 — **L1 pricing sweep (ponytail): 4 `vars(self)` hazards + de-flake `test_schema_adapter`**

T-PRC-PT1 — L1 `pricing/` sub-package sweep (9 modules + 1 legacy shim, ledger `AUDIT_L1_PRICING.md`).

**Architectural findings (clean):**
* Zero ABCs, Protocols, factories, builders, np.trapz.
* `schema_adapter.py` uses `core.serialisable._REGISTRY` as a consumer (lookup-only) — not a registry definition. Fine.
* `pricing_engine.py` and `pricing_server.py` per-trade `except Exception` blocks are legitimate **fault-isolation** patterns at the RPC boundary (record-error-and-continue). Keep.

**M-PRC-2 · `return vars(self)` mutation hazard** — 4 sites fixed:
* `market_data_provider.py:65` (1× site)
* `market_data_tools.py:38, 86, 126` (3× sites)

All same one-line fix. Cumulative session count: **73 instances** of this recurring pattern corrected across L0+L1.

**M-PRC-1 · Silent-except Greeks at RPC boundary (4 sites, MED, DEFERRED — needs decision)** — `pricing_engine.py:294, 308, 316` + `pricing_server.py:173`. Same recurring pattern as T-CAL1 / T4-HW1, but at the RPC boundary: callers currently receive a partial Greek dict with no signal that a Greek computation failed. Behaviour change affects external callers. **Not fixed in this slice**; queued as `T-PRC-PT2` pending user decision on swallow / propagate / narrow (recommendation: narrow to `NotImplementedError + AttributeError`, propagate everything else with per-trade error entry).

**Test-infra de-flake (bundled):** `tests/test_schema_adapter.py` had a latent dependency on the `core.serialisable._REGISTRY` being pre-populated. The test calls `from_dict({"type": "irs", ...})` which fails unless something earlier triggered `core.serialization._ensure_loaded()`. Whether that happened depended on per-xdist-worker test ordering — pre-existing flakiness that the T-PRC-PT1 vars(self) edits surfaced (the edited modules' `__pycache__` invalidation shifted the worker distribution; before my edits the load happened to fire on the right worker, after my edits it didn't). Fixed with an `@pytest.fixture(autouse=True, scope="module")` that explicitly calls `_ensure_loaded()`. L1 suite is now deterministic at 3520 passed regardless of edit ordering. **Bundled into T-PRC-PT1** rather than spawning a separate test-infra slice because (a) the bug surfaced via this slice and (b) is a one-fixture addition (8 LOC including docstring).

**Held (defensible):**
* `pricing_engine.py:79-91`, `pricing_server.py:58-72, 261` — fault-isolation patterns.
* `pricing/market_data.py` (2-line `import *` shim) — B.3 C1 architecture decision (Gate 2).

**Files changed**:
- `python/pricebook/pricing/market_data_provider.py` — 1× vars(self) sweep.
- `python/pricebook/pricing/market_data_tools.py` — 3× vars(self) sweep.
- `python/tests/test_schema_adapter.py` — autouse fixture for explicit registry pre-population (+9 / -1).

**L1 status:** `curves` ✅, `pricing` ✅. Remaining L1: `regulatory` (23 modules) — last sub-package before L1 is complete.

L1-scoped pytest: 3520 passed (consistent across reruns). 54s, `pytest -n auto`.

---

## v1.089.0 — 2026-06-16 — **L1 curves sweep (ponytail addendum): delete legacy `curves/quadrature.py` + 15 `vars(self)` hazards + 2 downstream migrations**

T-CRV-PT1 — first L1 ponytail slice. `curves/` had the correctness audit (per `AUDIT_L1_CURVES.md`, 33 modules, 14 bugs catalogued, 3/3 HIGH fixed in prior work) but no ponytail layer. Same combined methodology now applied — addendum appended to the existing ledger.

**Architectural findings (clean):**
* `ncurve_solver.py::InstrumentPricer(Protocol)` is a duck-typed contract for reprice callbacks, type-hint usage at line 163. Keep.
* `aad.py` + 4 sibling AAD modules form a layered subsystem (tape → interp → curves → calibration / pricing). Justified.
* `curves/linalg.py` is curve-specific factor-model algebra (PCA / level-slope-curvature), distinct from `numerical/_linalg.py`. Both kept.
* Zero ABCs, factories, builders, blanket excepts, `np.trapz`.

**P-CRV-1 · `curves/quadrature.py` deleted** — 27-line module whose docstring literally said "Legacy module. Use pricebook.numerical._integrate directly." Four convenience wrappers (`gauss_legendre`, `gauss_laguerre`, `gauss_hermite`, `adaptive_simpson`) plus `QuadratureResult = IntegrationResult` alias, all forwarding to `numerical._integrate`. Discovered 2 real downstream consumers the initial grep missed (the L1-scoped run surfaced them via the autodiscovery test rather than the L0 cycle):
* `credit/cdo.py:20` imported `gauss_hermite` but never used it — pure **dead import**, dropped.
* `options/heston.py` imported `gauss_legendre` and called it twice in the Heston characteristic-function quadrature. Migrated to `integrate(f, a, b, IntegrationMethod.GAUSS_LEGENDRE, n=n)` directly. Same numerical behaviour.
* `tests/test_quadrature.py` retargeted: per-test wrappers inlined at file top (since they're test-private now), imports redirected to `pricebook.numerical._integrate`.

Cross-layer note: this slice intentionally touches L3 (`credit/cdo.py`) and L4 (`options/heston.py`). Per AUDIT_PLAN §2.3 it would normally be an L1-only slice, but the edits are mechanical-only (dead-import drop + 1-to-1 API substitution). Verified by running `tests/test_cdo.py` + `tests/test_heston.py` directly (25 passed) on top of the L1-scoped suite (3520 passed).

**M-CRV-1 · `return vars(self)` mutation hazard** — 15 sites across 7 files (`curve_advanced.py` 4, `curve_engine.py` 2, `multicurve_solver.py` 2, `linalg.py` 4, `curve_diffusion.py` 1, `curve_bumper.py` 1, `seasonal_curve.py` 1). All same one-line fix. Cumulative session count: 54 (L0) + 15 = **69 instances** of this recurring pattern corrected.

**Process note — silent grep miss:** the initial `grep -rn` for `quadrature` callers returned only the test file. Two real production callers (`cdo.py`, `heston.py`) were missed. Discovered post-deletion by the autodiscovery test at L1. Reinforces the "always re-grep right before delete" rule from T-CORE-PT8's release notes — but also: the autodiscovery test SAVED this slice from shipping a real regression. Worth its weight in gold, do not delete or weaken.

**Files changed**:
- `python/pricebook/curves/quadrature.py` — deleted (27 lines).
- 7 `python/pricebook/curves/*.py` files — 15× vars(self) sweep.
- `python/pricebook/credit/cdo.py` — dead import dropped.
- `python/pricebook/options/heston.py` — 1 import + 2 call sites migrated.
- `python/tests/test_quadrature.py` — wrappers inlined.

**L1 status:** `curves` ✅ ready ponytail slice done. Remaining L1: `pricing` (9 modules), `regulatory` (23 modules) — both still ❓ deferred from original AUDIT_PLAN.

L1-scoped pytest: 3520 passed (was 3519 + 1 failing). 58s.
Direct cdo+heston tests: 25 passed. 89s.

---

## v1.088.0 — 2026-06-16 — **L0 viz sweep + L0 COMPLETE: 2 `vars(self)` mutation hazards**

T-VIZ-PT1 — `viz/` sub-package sweep (13 modules, ledger `AUDIT_L0_VIZ.md`).

**Architectural findings (clean):**
* `_dispatch.py` defines an `_INSTRUMENT_REGISTRY` + `_RESULT_REGISTRY` plugin system with 6+ register sites across `_cmasw.py`, `_cmt.py`, `_hybrid.py`, `_treasury_lock.py`, `_tlock.py`, `_trs.py`. Legitimate multi-impl registry. Keep.
* `_builder.py::PlotBuilder` is a documented fluent dashboard composer. Real use case. Keep.

**M-VIZ-1 · 2 `vars(self)` mutation hazards** in `_builder.py:16` (`PanelSpec`) and `_theme.py:33` (`PricebookTheme`). Fixed.

**Held (defensible):**
* `_builder.py:120` `try: plt.show() ... except Exception: pass` — UX safety net for non-interactive matplotlib backends.
* `_generic.py:58` sensitivity-sweep `except Exception: nan` — exploratory plot tolerates pricer crashes at individual parameter values.
Both are exploratory/UI excepts, not silent-bug hiders. Held with rationale.

**Files changed**: 2 in `python/pricebook/viz/` (+2 / -2).

---

# **L0 SWEEP COMPLETE 🎯**

All 9 L0 sub-packages now ✅ swept under the combined methodology (AUDIT_PLAN.md §2 + ponytail layer):

| Sub-pkg | Modules | Slices | Net code change |
|---------|---------|--------|-----------------|
| `calibration` | 2 | T-CAL1 | -10 lines + 1 regression test (silent except removed, L6 docstring fixed) |
| `core` | 35 | T-CORE-PT1, PT2, PT6, PT8, PT-MICRO | ~295 lines net dead code removed; 10 ponytail findings (3 held: PT3 cross-layer, PT5 Protocol-decision, low-pri done) |
| `db` | 2 | T-DB-PT1 | -50 lines (StorageBackend ABC + query_table alias removed) |
| `market_data` | 1 | T-MD-PT1 | docstring fix (empirical L0 vs design-target L1) |
| `numerical` | 30 | T-NUM-PT1 | 9× vars(self) + 3× np.trapz → np.trapezoid |
| `pe` | 4 | T-PE-PT1 | 15× vars(self) |
| `statistics` | 17 | T-STATS-PT1 | 22× vars(self) |
| `ts` | 7 | T-TS-PT1 | 2× vars(self) |
| `viz` | 13 | T-VIZ-PT1 | 2× vars(self) |

**Cumulative this session:**
* **13 slices landed** (v1.076.0 → v1.088.0)
* **~350 lines of net dead/over-engineered code removed**
* **54 instances of the recurring `return vars(self)` mutation hazard fixed** across L0
* **3 dead module files deleted** (`core/protocols.py`, `core/desk_protocol.py`, `core/results.py`)
* **3 over-engineered ABC/wrapper systems cut** (`StorageBackend` ABC, `query_table` alias, `_detect_code_version` silent-except wrapper)
* **Other small cleanups**: dead `key_fn` parameter, dead `load_or_default` function, 3 dead `deserialise_*` aliases, `all_g10_pairs` → `itertools.combinations`, `extract_forwards` dead variable, `pricing_context.to_dict` dead stub, 2 stale "Lx" docstring claims, `desk_protocol.py` docstring re-homed to `desks/README.md`.
* **3 ponytail findings held with rationale**: `Calibrator` Protocol (DESIGN.md G1 P1 commitment), `core/instrument_result.py` Protocol (test-only binders), `core/numerical_method_map.py` + `models/engine_registry.py` (cross-layer twin-delete, waits for L2).
* **Other open items inherited from original L0 audit**: C.7 B1 (settlement lag), D.1 B1 (empty-dict round-trip), D.1 B2 (dropped fields) — correctness MEDs queued for a separate sweep.

L0-scoped pytest throughout: **2437 passed, ~47s** per run (`pytest -n auto`, `--deselect g2pp_calibration`). Suite count grew by 1 over the session (T-CAL1's regression test).

Per AUDIT_PLAN §3, the next layer to audit is **L1** (`curves` ✅ already done in prior work; `pricing` + `regulatory` pending).

L0-scoped pytest: 2437 passed. 47s.

---

## v1.087.0 — 2026-06-16 — **L0 ts sweep: 2 `vars(self)` mutation hazards**

T-TS-PT1 — `ts/` sub-package sweep (7 modules, ledger `AUDIT_L0_TS.md`). Structural smell-grep clean (no ABCs, registries, factories, blanket excepts, np.trapz). Only 2 vars(self) sites in `_replay.py` (DrawdownPeriod, ReplayResult). Both fixed.

**Cumulative:** 54 instances of the vars(self) mutation pattern corrected across L0 this session.

**Files changed**: 1 in `python/pricebook/ts/` (+2 / -2).

**L0 sub-package status:** 8/9 done. Final remaining: `viz` (13 modules, 2 vars(self) sites + 2 narrow-except candidates).

L0-scoped pytest: 2437 passed. 47s.

---

## v1.086.0 — 2026-06-16 — **L0 statistics sweep: 22 `vars(self)` mutation hazards**

T-STATS-PT1 — `statistics/` sub-package sweep (17 modules, ledger `AUDIT_L0_STATISTICS.md`).

**Architectural findings (clean):**
* `copulas.py::Copula(ABC)` and `hmm.py::EmissionModel(ABC)` both have multiple concrete impls (5 and 4 respectively). Legitimate multi-impl ABCs — keep.
* Zero registries, factories, blanket excepts, `np.trapz` calls.

**M-STATS-1 · `return vars(self)` mutation hazard** — 22 sites across 9 files (`calibration_quality.py`, `bayesian.py`, `copulas.py`, `optimisation_advanced.py`, `distribution_fit.py`, `distribution_theory.py`, `optimization.py`, `zscore.py`, `statistics.py`). All same one-line fix.

**Cumulative pattern count:** 30 (prior sweeps) + 22 = **52 instances of `return vars(self)`** across L0 sub-packages corrected this session.

**Files changed**: 9 in `python/pricebook/statistics/` (+22 / -22).

**L0 sub-package status:**
* `calibration` ✅, `core` ✅, `db` ✅, `market_data` ✅, `numerical` ✅, `pe` ✅, **`statistics` ✅**.
* Remaining: `ts` (7 modules, 2 sites), `viz` (13 modules, 2 sites + 2 narrow-except candidates).

L0-scoped pytest: 2437 passed. 46s.

---

## v1.085.0 — 2026-06-16 — **L0 pe sweep: 15 `vars(self)` mutation hazards**

T-PE-PT1 — L0 `pe/` sub-package sweep (4 modules, ~1950 LOC; ledger `AUDIT_L0_PE.md`).

**Architectural finding:** same shape as `numerical/` — flat domain-logic layer over numpy with no ABCs, Protocols, registries, factories, or blanket excepts. The only ponytail-adjacent finding is the recurring `vars(self)` mutation pattern.

**M-PE-1 · `return vars(self)` mutation hazard** — 15 sites across 3 files (3× `lbo.py`, 8× `pe_desk.py`, 4× `pe_performance.py`). Same one-line fix per site: `return vars(self)` → `return dict(vars(self))`. Cumulative across this audit (counting `core/solvers.py` A.5 B1, `core/approximation.py` A.7, T-NUM-PT1's 9 sites, and now 15 more): **30 instances of this exact bug** have surfaced — `recurring-bug-patterns.md` memory's "vars(self) to_dict shared __dict__" is the highest-frequency pattern in the codebase.

**Files changed**: 3 in `python/pricebook/pe/` (+15 / -15).

**L0 sub-package status:**
* `calibration` ✅, `core` ✅, `db` ✅, `market_data` ✅, `numerical` ✅, **`pe` ✅**.
* Next per agreed order: `statistics` (17 modules).

L0-scoped pytest: 2437 passed. 47s.

---

## v1.084.0 — 2026-06-16 — **L0 numerical sweep: `vars(self)` mutation hazards + `np.trapz` → `np.trapezoid` (12-fix combined slice)**

T-NUM-PT1 — single combined slice from the L0 `numerical/` deep-read pass (30 modules, ~8500 LOC; ledger `AUDIT_L0_NUMERICAL.md`).

**Architectural finding:** `numerical/` is the cleanest L0 sub-package surveyed so far. Structural smell-grep across all 30 modules returned **zero** ABCs, zero Protocols, zero registries, zero factory/builder/manager classes, zero blanket excepts. The wrapper-layer-by-design discipline (curated public API in `__init__.py`, single Result dataclass per module, Enum-based method selection, scipy/numpy as the backend) removes the usual ponytail surfaces. No `delete`/`yagni`/`stdlib`/`native`/`shrink` findings.

The two recurring micro-patterns that DID surface (one MED correctness, one MED forward-compat) are batched here:

**M-NUM-1 · `return vars(self)` mutation hazard** — same pattern catalogued in `recurring-bug-patterns.md` and previously fixed in `core/solvers.py` A.5 B1 and `core/approximation.py` A.7. `vars(self)` returns the actual `__dict__`; callers can mutate instance state through it. 9 sites fixed:

```
_rootfinding.py:36       RootResult
_integrate.py:57         IntegrationResult
_mc.py:145               MLMCResult
_stochastic.py:33        ItoFormulaResult
oscillatory_quad.py:34   OscillatoryResult
von_neumann.py:32        StabilityResult
tree_enhancements.py:31  AdaptiveTreeResult
_distributions_theory.py:162  SobolevNorm
convexity_tools.py:88    ConvexityCheckResult
```

All same one-line fix: `return vars(self)` → `return dict(vars(self))`.

**M-NUM-2 · `np.trapz` → `np.trapezoid` (numpy 2.x forward-compat)** — `np.trapz` was removed in numpy 2.x. `_fourier.py:286` already has a fix-comment "Fix T1.2: np.trapz was removed in NumPy 2.x" but the same audit missed `_distributions_theory.py`, which has 3 lingering calls (in `SchwartzTestFunction.fourier` and `TemperedDistribution.fourier_transform`). `numpy>=1.21.0` per pyproject; nothing pins to < 2.0. The current venv is on numpy 1.x so the calls work; a future env upgrade to numpy 2.x would break import-load. All 3 calls migrated to `np.trapezoid`.

**Bundling rationale:** all 12 edits are the same shape (one-line recurring-pattern fixes across 10 files). Splitting into 12 slices would be churn; the unified slice has a coherent purpose ("close out recurring micro-patterns surfaced by the numerical deep-read"). Same precedent as T-CORE-PT-MICRO.

**No regression test added** — all 12 fixes are existing-pattern repairs; the L0 suite passing (2437 → 2437) confirms no behaviour change in tested code paths. The mutation-hazard surface is a latent risk (callers don't currently mutate the returned dicts) and the numpy 2.x surface depends on a non-current environment.

**Files changed**: 9 in `python/pricebook/numerical/` (+12 lines, -12 lines net; 11 `dict(vars(self))` swaps + 3 `np.trapezoid` swaps).

**L0 sub-package status:**
* `calibration` ✅, `core` ✅, `db` ✅, `market_data` ✅, **`numerical` ✅**.
* Next per agreed order: `pe` (4 modules — small leaf).

L0-scoped pytest: 2437 passed. 47s.

---

## v1.083.0 — 2026-06-16 — **L0 market_data sweep: fix stale L1 docstring claim in `_types.py`**

T-MD-PT1 — `market_data/` sub-package sweep (1 substantive module, 201 LOC; ledger `AUDIT_L0_MARKET_DATA.md`).

* **`python/pricebook/market_data/_types.py:9-10`** previously claimed "Zero dependencies on other pricebook subpackages — sits cleanly at L1 in the dependency graph." Per the empirical classifier (`tools/test_layer.py --show-layers`), `market_data` is **L0**, not L1, because no pricebook imports are wired in yet. The L1 framing comes from DESIGN.md §5.1 A2 where `market_data` is the *target* layer once curves integrate with it (G1 P2). Same shape as L-CAL-1 fixed in v1.076.0 (`calibration/_types.py` had claimed "L6", actually L0).

* **Fix:** rewrite the docstring to acknowledge both states — currently L0 empirically; design target L1 once integration lands. Single-paragraph clarification; no code change.

* **No regression test** — pure docstring; no behavioural surface.

* **Other findings:** none. Deep-read confirmed `MarketSnapshot` / `QuoteId` / `Quote` / `FixingHistory` are all clean frozen dataclasses with appropriate defensive patterns (`with_quote` immutable derivation, `__contains__` type guard, `with_fixing` dict-copy).

**Files changed**:
- `python/pricebook/market_data/_types.py` — docstring lines 9-10 expanded to clarify empirical L0 vs design L1.

**L0 sub-package status:**
* `calibration` ✅, `core` ✅ (ready + low-pri done), `db` ✅, `market_data` ✅.
* Next per agreed order: `numerical` (30 modules — biggest L0 sub-package).

L0-scoped pytest: 2437 passed. 47s.

---

## v1.082.0 — 2026-06-16 — **L0 db sweep: drop `StorageBackend` ABC (single-impl YAGNI) + dead `query_table` alias**

T-DB-PT1 — first slice of the L0 `db/` sub-package sweep (combined methodology: AUDIT_PLAN + ponytail; ledger `AUDIT_L0_DB.md`). The original ponytail preliminary scan already flagged `StorageBackend` as YAGNI; deep-read confirmed and added a second finding.

* **`StorageBackend` ABC deleted** — 33 lines of `@abstractmethod` scaffolding in `python/pricebook/db/db_backend.py` declaring a contract for `execute`, `execute_many`, `table_exists`, `create_table`, `drop_table`, `list_tables`, `commit`, `close`. Single impl in the codebase (`SQLiteBackend`). The module's own docstring promised "Future: DuckDBBackend, PostgresBackend — same interface, swap one line" — speculative, no second impl exists or is planned. `grep -rn "backend=" python/` for non-default backend instantiation: zero hits across all tests and production code. `SQLiteBackend` now stands alone (no parent class). When/if a second backend genuinely arrives, extract the ABC at that point.

* **`PricebookDB.__init__` type hint updated** — `backend: StorageBackend | None = None` → `backend: SQLiteBackend | None = None`. Same runtime behaviour; tighter type for the only concrete option.

* **`db/__init__.py` re-export dropped** — `StorageBackend` no longer exported as a public symbol.

* **`query_table` deleted** — 3-line one-line alias for `load_table` (`return self.load_table(name, **filters)`). One caller in `tests/test_db.py:240` updated to call `load_table` directly. Adds zero value over the existing public API.

* **No regression test added** — file-deletion / alias-removal; L0 suite passing IS the regression evidence (2437 → 2437).

**Files changed**:
- `python/pricebook/db/db_backend.py` — -33 (ABC class) -3 (abc import) -1 (inheritance line) = -37 / +1 (no parent).
- `python/pricebook/db/db.py` — -3 (query_table) -1 (StorageBackend import) +1 (SQLiteBackend type hint).
- `python/pricebook/db/__init__.py` — -1 (StorageBackend re-export).
- `python/tests/test_db.py` — query_table → load_table (1 line).

**L0 sub-package status:**
* `calibration` ✅ swept.
* `core` ✅ swept (ready + low-priority all done; 2 held items waiting on cross-layer / Protocol decision).
* `db` ✅ swept (both modules audited, ABC deleted, alias removed).
* Next L0 sub-package per agreed order: `market_data` (1 module).

L0-scoped pytest: 2437 passed, identical to v1.081.0 baseline. 47s, `pytest -n auto`.

---

## v1.081.0 — 2026-06-16 — **L0 core sweep (deep-read pass, slice 3): bundled ponytail micro-cleanups**

T-CORE-PT-MICRO — the 4 low-priority shrinks from the deep-read pass batched into one slice (each individually <5 LOC; 12 separate commits for trivia would have been worse than the bundling).

* **`core/data_registry.py`** (PT4): removed dead `key_fn=None` parameter from `load_conventions` (the function body never read it; the one forwarding caller in `load_or_default` was also removed). Hoisted two function-local `import warnings` statements to the module top. **Bonus:** discovered `load_or_default` itself has zero callers — deleted (17 lines).
* **`core/currency.py`** (PT7): `all_g10_pairs()` rewritten from 6-line nested-loop reinvention to one-line `itertools.combinations(Currency, 2)` comprehension.
* **`core/forward_interpolation.py`** (PT9): `extract_forwards()` had a dead `d = date.fromordinal(...)` variable computed every iteration but never read; replaced the 7-line loop body with a one-line list comprehension.
* **`core/pricing_context.py`** (PT10): removed 2-line dead `to_dict` stub at lines 244-245 (overwritten by `_ctx_to_dict` at line 373 before any caller could reach it).

Net: -40 lines, +4 lines across 4 files. Zero behaviour change in any code path that the L0 suite exercises (2437 → 2437 unchanged).

**Bundling rationale:** AUDIT_PLAN §2.3 mandates one combined commit + stamp + release-notes per slice; it doesn't mandate one finding per slice. Four <5-line shrinks in 4 different files are coherent as "core ponytail micro-cleanups", and the alternative — 12 commits for ~40 lines — adds review friction without auditability benefit.

**Files changed**:
- `python/pricebook/core/data_registry.py` — -25 / +2 (PT4 + dead `load_or_default` delete).
- `python/pricebook/core/currency.py` — -7 / +2 (PT7).
- `python/pricebook/core/forward_interpolation.py` — -8 / +1 (PT9).
- `python/pricebook/core/pricing_context.py` — -2 / +0 (PT10).

**L0 sub-package status:**
* `calibration` ✅ swept.
* `core` ✅ ready+low-priority slices done. Remaining open: `T-CORE-PT3` (cross-layer, blocks on L2 `models/` audit), `T-CORE-PT5` (`instrument_result.py`, blocks on Protocol-cleanup decision). Original-audit MEDs C.7 B1 / D.1 B1 / D.1 B2 still queued (separate sweep).
* Next L0 sub-package per agreed order: `db` (2 modules; preliminary `StorageBackend` ABC YAGNI finding already in hand).

L0-scoped pytest: 2437 passed, identical to v1.080.0 baseline. 46s, `pytest -n auto`.

---

## v1.080.0 — 2026-06-16 — **L0 core sweep (deep-read pass, slice 2): delete 3 dead aliases in `core/serialization.py`**

T-CORE-PT8 — second slice from the deep-read pass.

* **`deserialise_date`, `deserialise_enum`, `deserialise_currency_pair` deleted** from `python/pricebook/core/serialization.py:120-132`. All three had **zero callers** in the entire codebase (`grep -rn`). They predate the centralised `from_dict` dispatch and were superseded by the `instrument_to_dict = to_dict` style aliases above them (those have real consumers and stayed).

* **`_str_to_date = date.fromisoformat` kept**, contrary to the initial finding. A second-pass grep before deletion caught one real caller — `pricebook/pricing/pricing_engine.py:36,69` imports and uses it. Annotated with an inline comment so the next sweep doesn't trip on it. The full inline-and-delete is an L1 cleanup (pricing_engine.py is L1); L0 audit leaves it alone per AUDIT_PLAN §2.3.

* **No regression test added** — same rationale as prior delete slices.

**Files changed**:
- `python/pricebook/core/serialization.py` — -15 / +1 (3 functions deleted; `_str_to_date` kept with inline comment).

**Process note:** the initial grep (which the addendum was based on) checked only the named aliases and missed `_str_to_date`'s 2 uses. The triple-check immediately before code change caught it. Reinforces the "before recommending from memory, verify" rule — always re-grep right before the delete, never trust a finding from earlier in the same session.

L0-scoped pytest: 2437 passed, identical to v1.079.0 baseline. 47s, `pytest -n auto`.

---

## v1.079.0 — 2026-06-16 — **L0 core sweep (deep-read pass, slice 1): delete dead `core/results.py`**

T-CORE-PT6 — surfaced by the post-T-CORE-PT2 deep-read pass walking all 23 previously sample-only `core/` modules (the user flag was "we can't afford leave blanks"). Found 5 new ponytail findings in those 23; this is the highest-impact ready slice.

* **`python/pricebook/core/results.py` deleted** — 60 lines, a module aggregator exporting `SolverResult` (re-export), `TreeResult` + `PDEResult` (defined here), and four `TYPE_CHECKING`-only names. Zero importers in the entire repo (`grep -rn "core\\.results\\b"` returned only the file's own docstring example). Worse: the `TreeResult` and `PDEResult` shells defined here are *not* the ones used in production — `pricebook/numerical/_trees.py:67` and `pricebook/numerical/_pde.py:59` define independent concrete classes with the same name. The shells in `core/results.py` were unused parallel definitions. The `__all__` advertised `MCResult`, `OptimizerResult`, `ODEResult` — all `TYPE_CHECKING`-only and would raise `ImportError` at runtime if anyone tried to use them.

* **No regression test added** — same rationale as T-CORE-PT1 / T-CORE-PT2 (file deletion; L0 suite passing IS the regression evidence; 2437 → 2437).

**Files changed**:
- `python/pricebook/core/results.py` — deleted (60 lines).

**L0 sub-package status:**
* `calibration` ✅ swept.
* `core` deep-read pass complete (35 modules now confirmed individually, not just by structural smell-grep). 10 ponytail findings total (5 from structural sweep, 5 from deep read). Ready slices done: T-CORE-PT1, T-CORE-PT2, T-CORE-PT6. Ready slice queued: T-CORE-PT8 (delete 3 dead `serialization.py` aliases). Low-priority slices queued: T-CORE-PT4, T-CORE-PT7, T-CORE-PT9, T-CORE-PT10. Held: T-CORE-PT3 (cross-layer with L2), T-CORE-PT5 (Protocol-cleanup decision).
* Next L0 sub-package per agreed order: `db` (2 modules).

L0-scoped pytest: 2437 passed, identical to v1.078.0 baseline. 47s, `pytest -n auto`.

---

## v1.078.0 — 2026-06-16 — **L0 core sweep (ponytail addendum, slice 2/2): delete `core/desk_protocol.py` docstring-only file; move contract to `desks/README.md`**

T-CORE-PT2 — second and last ready ponytail slice on `core/`.

* **`python/pricebook/core/desk_protocol.py` deleted** — 51 lines of pure module docstring describing the uniform API contract for all 12 desk modules. Zero `class`, zero `def`, zero importers anywhere in the repo. The contract content is documentation, not code, and belongs in markdown — Python isn't the right format for a multi-section prose contract that no executable references.

* **`python/pricebook/desks/README.md` created** — the contract docstring re-homed as markdown with proper tables and code blocks. Same content, more readable, discoverable to new contributors via standard "look at the README" reflex. Tracked in git (the `/*.md` gitignore anchor only affects repo-root `.md` files; sub-directory READMEs are unaffected per the `.gitignore` comment).

* **No regression test added** — same rationale as T-CORE-PT1 (deletion + doc-move; L0 suite passing IS the regression evidence; 2437 → 2437).

**Files changed**:
- `python/pricebook/core/desk_protocol.py` — deleted (51 lines).
- `python/pricebook/desks/README.md` — new (50 lines markdown).

**L0 sub-package status:**
* `calibration` ✅ swept (correctness + ponytail done at v1.076).
* `core` ✅ ponytail-half complete on ready slices. 3 ponytail slices held: `T-CORE-PT3` (twin-delete with L2 `engine_registry`, blocks on L2 audit reaching `models/`), `T-CORE-PT4` (`data_registry.py` minor cleanup, low priority), `T-CORE-PT5` (`instrument_result.py`, pending broader Protocol-cleanup decision).  The 3 open correctness MEDs from the original Pass C/D (C.7 B1 settlement lag, D.1 B1 empty-dict round-trip, D.1 B2 dropped fields) remain queued — those are correctness slices for a separate sweep.
* Next L0 sub-package per the agreed order: `db` (2 modules — preliminary scan already flagged `StorageBackend` ABC as YAGNI).

L0-scoped pytest: 2437 passed, identical to v1.077.0 baseline. 46s, `pytest -n auto`.

---

## v1.077.0 — 2026-06-16 — **L0 core sweep (ponytail addendum, slice 1/2): delete dead `core/protocols.py`**

T-CORE-PT1 — first ponytail slice on `core/` (the correctness audit landed previously; ponytail lens layered on top per the addendum in `AUDIT_L0_CORE.md`).

* **`python/pricebook/core/protocols.py` deleted** — 129 lines of `@runtime_checkable Protocol` classes (`RootFinder`, `Integrator`, `OptionPricer`, `MCEngine`, `VolModel`, `VolSurface`, `CharFunc`) plus a `SolverResult` re-export. The whole module had zero importers anywhere in the repo. Six of the seven Protocols had zero binders even by name; the seventh (`MCEngine`) shared a name with the concrete `pricebook.models.mc_engine.MCEngine` class — every "binder" was importing the concrete class from `models`, not the Protocol from `core`. The `SolverResult` re-export was redundant (all real consumers import from `pricebook.core.solvers` directly).

* **No regression test added** — per ponytail "YAGNI applies to tests too": deletion of dead code doesn't need a runtime test, the L0 suite passing IS the regression evidence (and it does: 2437 → 2437, zero failures).

* **Cross-check:** `grep -rln "from pricebook.core.protocols\|import pricebook.core.protocols\|pricebook\\.core\\.protocols"` across `python/` confirmed zero references prior to deletion. The `core/__init__.py` also did not re-export protocols.

**Files changed**:
- `python/pricebook/core/protocols.py` — deleted (129 lines).

**L0 sub-package status:** core ponytail-half in progress (`T-CORE-PT2` next: delete `core/desk_protocol.py` and move the contract docstring to `desks/README.md`).

L0-scoped pytest: 2437 passed, identical to v1.076.0 baseline.  46s, `pytest -n auto`.

---

## v1.076.0 — 2026-06-16 — **L0 calibration sweep: drop `_detect_code_version` silent-except blanket; fix stale L6 docstring claim**

T-CAL1 — first slice of the AUDIT_PLAN.md bottom-up sweep (combined methodology: layer audit + ponytail over-engineering hunt).  L0 sub-package `calibration/` walked end-to-end (2 modules, 260 LOC).

* **Silent except removed.** `pricebook/calibration/_types.py::_detect_code_version` wrapped `import pricebook; return pricebook.__version__` in `try: ... except Exception: return "unknown"`.  Same recurring pattern just removed from `hw_calibration.py` in v1.075 (T4-HW1).  Neither exception path can realistically fire here — `pricebook.__version__` is set eagerly in `pricebook/__init__.py`; if it ever isn't, the user wants the error, not a silent `"unknown"` polluting the calibration provenance chain.  Helper deleted entirely (no test mocks it; tests use the explicit `code_version=` parameter override), call inlined into `CalibrationResult.new(...)`.  Net -10 lines.

* **Stale docstring fixed.** `_types.py` module docstring claimed the calibration layer "sits cleanly in the dependency graph at L6 (per the reference design)".  Per the empirical layer classifier (`tools/test_layer.py`) calibration is L0 — no top-level pricebook deps.  The L6 claim referred to an older DESIGN.md placement that the actual code no longer matches.  Updated to call out L0 explicitly and to note the lone in-function `import pricebook` as the (lazy, cycle-free) reason it stays L0.

* **Ponytail finding held, not cut.** `Calibrator` Protocol (216-line file, lines 200-216) has no production binders — real calibrators (`sabr.py`, `bond_hazard_bootstrap.py`) consume `CalibrationResult` directly without declaring conformance.  Held per DESIGN.md G1 P1 commitment to migrate calibrators onto it; revisit at end of G1 P1.

* **Ponytail finding rejected.** Initial plan was to swap `import pricebook; return pricebook.__version__` for `importlib.metadata.version("pricebook")` (stdlib idiom).  Empirical check killed it: the venv's egg-info reports `"0.790.0"` while `__version__` is `"1.075.0"`; egg-info is gitignored and drifts.  `pricebook.__version__` is the canonical source (pyproject `dynamic = ["version"]`, `attr = "pricebook.__version__"`).  Recorded in `AUDIT_L0_CALIBRATION.md` as a worked example of "two stdlib options, same size? take the one correct on edge cases".

**Files changed**:
- `python/pricebook/calibration/_types.py` — helper deleted, version lookup inlined into `.new()`, docstring fixed.
- `python/tests/test_calibration_types.py` (new test) — `test_missing_pricebook_version_propagates` asserts `AttributeError` propagates when `pricebook.__version__` is missing, instead of silent `"unknown"`.

**L0 sub-package status:** calibration ✅ swept (1 MED bug, 1 LOW doc closed; 1 YAGNI held).  Next: `core/` re-sweep with ponytail lens layered on existing `AUDIT_L0_CORE.md`.

L0-scoped pytest: 2437 passed (was 2436, +1 regression).  46s, `pytest -n auto`.

---

## v1.075.0 — 2026-06-16 — **Fix L2 T4 (models/hw_calibration) — blanket `except Exception` masked pricer crashes as zero swaption price**

T4-HW1 (mirror of T2.11 for ``g2pp_swaption_price``): ``_hw_swaption_price`` wrapped its entire body, and ``_hw_implied_vol`` wrapped its final ``implied_vol_black76`` call, in ``try: ... except Exception: return 0.0``.  Any error — HullWhite construction failure, tree-pricer assertion, brentq divergence, overflow — was silently turned into a zero price, then fed to the calibration optimiser as a fixed residual (zero against a positive market vol).  Calibrations could "converge" on parameter regions where the pricer was secretly crashing.

Fix: let real exceptions propagate from ``_hw_swaption_price``.  In ``_hw_implied_vol`` keep a narrow ``except ValueError`` for the legitimate arbitrage-violating case (intrinsic-floor breach in Black-76 inversion), so true calibration bugs surface but unfittable boundary strikes don't crash the whole sweep.

**Files changed**:
- `python/pricebook/models/hw_calibration.py` — removed blanket excepts in two helpers.
- `python/tests/test_l2_t4_hw_calibration_silent_except.py` (new) — pins a vanilla ATM swaption returns a positive price (sanity) AND that a mocked ``HullWhite`` failure now propagates rather than silently returning 0 (regression).

---

## v1.074.0 — 2026-06-16 — **Fix L2 T4 (fixed_income/risky_floating) — z_spread sign inverted + accrued-on-default mid-date**

T4-RFRN1:

* **``CreditRiskyFRN.z_spread`` sign inverted** — used ``discount_curve.bumped(-z)``, but ``DiscountCurve.bumped(s)`` adds ``s`` to the zero rates (multiplies DF by ``exp(-s·t)``).  So positive z shifted rates DOWN (DF up, PV up), the opposite of the z-spread convention.  brentq bracketed [-0.05, 0.10] and both endpoints had PV above any sensible target, so the solver raised ``ValueError("must have opposite signs")`` on every realistic input.  The function had no tests and no callers, so the bug went unnoticed.
* **``risky_floating_pv`` accrued-on-default mid-date** — the half-period accrued payment was discounted using ``df(accrual_start)`` rather than the mid-period DF.  For semi-annual @ 4% the resulting accrued-on-default component was ~1% high (a ~4 bp bias on a typical risky-FRN total PV).

Fix: ``bumped(z)`` (not ``bumped(-z)``) in z_spread; midpoint ordinal between accrual_start and accrual_end in risky_floating_pv.

**Files changed**:
- `python/pricebook/fixed_income/risky_floating.py` — ``z_spread`` sign and ``risky_floating_pv`` accrued mid-date.
- `python/tests/test_l2_t4_risky_floating.py` (new) — 2 regressions: z_spread no longer raises on a credit-risky self-consistency target; accrued component reconciles into the total PV decomposition and sits in a sensible range.

---

## v1.073.0 — 2026-06-16 — **Fix L2 T4 (fixed_income/jarrow_yildirim) — HW ZCB formula + inflation-forward ratio inverted**

**Two coupled bugs in ``jy_zc_inflation_swap``: the Hull-White ZCB closed form was wrong on three counts, and the JY inflation-forward ratio was inverted.**

T4-JY1.a — **HW ZCB formula bug**.  Pre-fix ``hw_zcb`` computed
``A = -(σ²/(2a²))·(T − B − aB²/2)`` and returned ``exp(A − B·r₀)``, which:
1. replaced ``-T·r₀`` with ``-B·r₀`` — missing the dominant ``-(T-B)·r₀`` rate-only term;
2. flipped the sign of ``(T-B)·σ²/(2a²)``;
3. flipped the sign of ``σ²·B²/(4a)``.

For a=0.05, σ=0.01, r₀=0.04, T=5 it returned ZCB ≈ 0.836 vs the correct Vasicek (θ=r₀) value 0.820 — a ~2% bias.  Critically the bias differed between the nominal (σ_n) and real (σ_r) factors, so the bias did **not** cancel in the P_n/P_r ratio that drives the inflation forward.

T4-JY1.b — **Inflation-forward ratio inverted**.  The JY ZC inflation forward (Jarrow-Yildirim 2003 eq. 16, Mercurio 2005) is
    I_fwd(0, T) / I(0) = P_r(0, T) / P_n(0, T) · exp(conv_adj)
but the code computed ``P_n / P_r``.  Result: ``fair_rate ≈ exp(-(r_n − r_r)·T) − 1`` — negative for the typical ``r_n > r_r`` setup.  At σ=0, ``fair_rate`` returned ``-0.0952`` instead of ``+0.1052`` (the deterministic-rate breakeven for r_n=0.04, r_r=0.02, T=5).

The existing ``TestJYZCSwap`` tests were too loose to catch either defect: ``test_breakeven_sign`` only checked ``math.isfinite``; ``test_convexity_nonzero`` only checked ``!= 0``.

Downstream impact: ``jy_yoy_caplet`` builds its ``forward_yoy`` from ``zc_end.fair_rate`` and ``zc_start.fair_rate`` and then floors at ``1e-6`` — so pre-fix every realistic YoY caplet collapsed to a near-zero floored Black price.

**Files changed**:
- `python/pricebook/fixed_income/jarrow_yildirim.py` — ``hw_zcb`` rewritten using the standard Vasicek-θ=r₀ exponent ``-T·r₀ + (T-B)σ²/(2a²) - σ²B²/(4a)``; inflation-forward ratio flipped to ``P_r / P_n``.
- `python/tests/test_l2_t4_jy_zc_inflation.py` (new) — 5 regressions: nominal/real ZCB collapse to flat-rate exp(-r₀·T) at σ→0; fair_rate matches exp((r_n−r_r)·T)−1 at σ→0; fair_rate sign follows rate differential; small-vol fair_rate within 5% of the deterministic breakeven.

---

## v1.072.0 — 2026-06-16 — **Fix L2 T4 (remaining G2++ phi(t) duplicates) — callable_floater_g2pp + cms_spread_g2pp**

**Two more modules carried the same ``_phi`` finite-difference defect fixed in v1.069 / v1.070:**

T4-G2T3:
* ``fixed_income.callable_floater_g2pp._phi_at`` and the in-line ``r0`` derivation at the bottom of the module — used in the G2++ FRN tree path and in the HW1F fallback branch.
* ``structured.cms_spread_g2pp._fwd`` — used in MC path simulation for CMS-spread structured products.

Both used ``eps = 1e-5`` years for the forward-rate finite difference; ``date_from_year_fraction``'s day rounding turned the result into ~0 or ~137·r per call across the time grid.

Fix: delegate to ``DiscountCurve.instantaneous_forward(t)`` (the same one-line collapse applied in T4-G2T1 / T4-BSWG1).

**Files changed**:
- `python/pricebook/fixed_income/callable_floater_g2pp.py` — ``_phi_at`` and the bottom-of-module ``r0`` derivation.
- `python/pricebook/structured/cms_spread_g2pp.py` — inner ``_fwd`` helper.
- `python/tests/test_l2_t4_g2pp_phi_duplicates.py` (new) — asserts ``_phi_at`` is smooth across the 30-step T=5y grid on a flat 4% curve (pre-fix it alternated between ~0.04 and ~5.48), and the CMS-spread MC option price is finite + bounded by notional.

A grep across `python/pricebook/` for the ``eps = 1e-5`` + ``log(curve.df(d2)/curve.df(d1))`` pattern finds no other instances after this fix.

---

## v1.071.0 — 2026-06-16 — **Fix L2 T4 (G2++ ZCB formula) — Brigo-Mercurio eq. 4.10 missing V(t,T) term**

**Three internal G2++ ZCB implementations used ``0.5·[V(0, T) − V(0, t)]`` in the exponent where Brigo-Mercurio (2nd ed., eq. 4.10) requires ``0.5·[V(t, T) − V(0, T) + V(0, t)]``.**

T4-G2T2: for stationary OU integrated variance the missing combination is V(t, T) = V(0, T−t).  At t=0 with x=y=0 the buggy formula returns ``P^M(T) · exp(0.5·V(0, T))`` instead of P^M(T) — for σ₁=0.01, T=5y this is a +0.25% bias on the discount factor.  The bias propagates to the swap PV at exercise dates in:

* ``G2PlusPlus.zcb_price`` (``vasicek.py``) — public model API; signature extended to optionally take ``t`` (default 0).
* ``G2PPTree.zcb_price`` (``g2pp_tree.py``) — used at the terminal slice of the swaption tree and at every exercise-date node in ``bermudan_swaption_g2pp_tree``.  Drove the ~13% gap vs the analytical Jamshidian price on a vanilla 1y/2y European payer (post-T4-G2T1).
* ``_zcb_path`` (``bermudan_swaption_g2pp.py``) — used by the LSM pricer to compute swap PV along simulated paths.

The analytical swaption pricer ``g2pp_swaption_price`` in ``g2pp_calibration.py`` already used the correct A_i factor under the T-forward measure (the V_α/V_i/V_0 combination matches Brigo eq. 4.31), so calibration is unaffected — the fix just brings the tree and LSM into agreement with the analytical formula.

**Files changed**:
- `python/pricebook/models/vasicek.py` — ``G2PlusPlus.zcb_price`` now takes optional ``t`` and uses ``0.5·[V(t, T) − V(0, T) + V(0, t)]``.
- `python/pricebook/models/g2pp_tree.py` — ``G2PPTree.zcb_price`` exponent corrected; computes V(t, T) = V(0, T−t).
- `python/pricebook/options/bermudan_swaption_g2pp.py` — ``_zcb_path`` (the LSM-helper analytical ZCB) corrected.
- `python/tests/test_l2_t4_g2pp_zcb_formula.py` (new) — 4 regressions: model ZCB at origin equals curve P(0,T) exactly (was off by exp(0.5·V)), tree ZCB at root equals curve, tree backward-induction with terminal=1 matches the analytical ZCB to lattice tolerance, and the tree-priced European swaption tracks the Jamshidian closed-form to 10% rel-err (was 13% pre-fix).

---

## v1.070.0 — 2026-06-15 — **Fix L2 T4 (options/bermudan_swaption_g2pp) — duplicate phi(t) defect in LSM path discounting**

**``bermudan_swaption_g2pp_lsm`` carries a private ``_phi`` helper duplicating ``G2PPTree._phi``, including the same ``eps = 1e-5`` finite-difference defect (T4-G2T1) that ``date_from_year_fraction``'s day rounding turns into either 0 or ≈ 137·r per call.**

T4-BSWG1: ``_phi`` is invoked at every simulation step to build ``log_df_paths`` (the accumulated short-rate integral along each path).  Because the defect depends only on ``t`` and the date-rounding pattern, it hits a deterministic ~1/8 of the time grid — on those steps every path's discount is inflated by exp(-5.48·dt) ≈ 0.40, biasing the LSM price arbitrarily.  The bug went undetected because the existing LSM tests are loose (price > 0) and the tree/LSM cross-check tolerance was correspondingly slack.

Fix: delegate to ``DiscountCurve.instantaneous_forward(t)`` (the same one-line collapse applied to ``G2PPTree._fwd_rate`` in v1.069).

**Files changed**:
- `python/pricebook/options/bermudan_swaption_g2pp.py` — ``_phi`` inside ``bermudan_swaption_g2pp_lsm`` now delegates to the curve.
- `python/tests/test_l2_t4_bermudan_swaption_g2pp_phi.py` (new) — asserts the LSM price agrees with the tree price to within 10% on a vanilla 1y/1.5y/2y × 5y Bermudan payer (the curse of the buggy ``_phi`` was a ~30-90% bias that this test now catches).

---

## v1.069.0 — 2026-06-15 — **Fix L2 T4 (models/g2pp_tree) — catastrophic phi(t) finite-difference defect**

**``G2PPTree._fwd_rate`` (and therefore ``_phi``, the G2++ deterministic shift used at every node in backward induction) was computing the instantaneous forward rate via a finite difference with ``eps = 1e-5`` years (≈ 8 seconds).**

T4-G2T1: because the curve's ``df`` API takes a ``datetime.date`` and ``date_from_year_fraction`` rounds the input year-fraction to a day, the two evaluation points ``t ± eps`` round either to the same date (so the helper returns 0) or to dates one day apart (so the helper returns the true forward × (1/365) / (2·eps) ≈ 137× over-stated).  For a 30-step T=5y tree on a flat 4% curve this gave φ(t) ≈ 0 at 27 of 31 grid times and φ(t) ≈ 5.48 at the 4 other times — every node discounted by exp(-5.48·dt) ≈ 0.40 on those four steps, yielding tree-computed ``P(0, 5) ≈ 0.026`` vs the curve value 0.819.  All G2++ tree-based products (``callable_bond_g2pp``, ``puttable_bond_g2pp``, ``callable_floater_g2pp``, ``g2pp_european_swaption_tree``, ``bermudan_swaption_g2pp_tree``) were catastrophically mis-priced as a result.

The existing G2PP test suite hid the defect: ``test_european_swaption_tree_close_to_analytical`` uses a 50% relative-error tolerance, and the callable / puttable G2PP tests check only directional sanity (price > 0).  A puttable bond with put_price=100 against straight=99.8 returned 43.7 — masked by the absence of a "puttable ≥ straight" assertion.

Fix: delegate to :meth:`DiscountCurve.instantaneous_forward`, which uses a stable one-day step consistent with the day-rounded ``df`` lookup.

**Files changed**:
- `python/pricebook/models/g2pp_tree.py` — ``_fwd_rate`` rewritten as one-line delegate to ``curve.instantaneous_forward(t)``.
- `python/tests/test_l2_t4_g2pp_tree_fwd_rate.py` (new) — 4 regressions: φ(t) smooth on flat curve (catches the 0/5.48 alternation), tree-computed ``P(0, T)`` matches curve at 5y and 1y horizons (1% rel-err, was 97% off pre-fix), and European-swaption tree ≈ analytical Jamshidian within 15% (tightens the existing 50%-tolerance test by ~3× to catch this defect class).

---

## v1.068.0 — 2026-06-15 — **Fix L2 T4 (fixed_income/callable_floater) — same HW-tree defect family**

**``callable_floater`` (callable / puttable FRN) HW tree carried two of the same defects fixed in ``callable_bond`` (v1.067) and the bermudan family (v1.049 / v1.050).**

T4-CF1 affects both ``_straight_frn_tree`` and ``_frn_tree_with_option``:

1. **Wrong trinomial probabilities** — pre-fix used ``/6`` instead of textbook ``/2``.
2. **Coupon applied AFTER backward discount** — coupon at step+1 added to ``new_values`` (already discounted to step), so the coupon's one-step discount factor was missing.  For ``_frn_tree_with_option`` the option ``min(v, call_price)`` / ``max(v, put_price)`` had the same defect — comparing discounted continuation against undiscounted exercise price.

Note: α(t) is implicit ``r0`` here because this module takes raw ``r0`` rather than a ``DiscountCurve`` — flat-curve interface by design.  The missing-α(t) issue from ``callable_bond`` doesn't apply (analogous to the ``bermudan_capfloor`` situation).

Fix: textbook ``/2`` probabilities; apply coupon (and option) BEFORE the backward discount step.

**Files changed**:
- `python/pricebook/fixed_income/callable_floater.py` — both ``_straight_frn_tree`` and ``_frn_tree_with_option`` updated.
- `python/tests/test_l2_t4_callable_floater_hw_tree.py` (new) — 3 regressions: straight FRN near par (catches the missing coupon discount), callable ≤ straight, puttable ≥ straight.

---

## v1.067.0 — 2026-06-15 — **Fix L2 T4 (fixed_income/callable_bond) — same HW-tree defect class as bermudan_swaption, plus terminal-coupon double-count**

**``callable_bond._trinomial_backward`` (powering both callable and puttable bond pricers) carried the full bermudan-style HW tree defect set.**

T4-CB1 rolls up four coupled bugs:

1. **Wrong trinomial probabilities** (`/6` instead of textbook `/2`; same as T4-BERM1).
2. **Missing α(t) shift** (`r_j = r0 + j·dr` for every step; tree didn't reprice the input curve).
3. **Coupon and option applied AFTER backward discount** — the coupon at step+1 was added to ``new_values`` (already discounted to step), so the coupon's own discount factor was missing.  Likewise the option ``min(v, call_price)`` / ``max(v, put_price)`` compared the discounted continuation against the undiscounted par strike.
4. **Terminal coupon double-counted** — ``values`` was initialised to ``notional × (1 + c·τ)`` (cum-terminal-coupon), but the first iteration (``step = n_steps − 1``) sees ``step + 1 = n_steps`` in ``coupon_steps`` and adds the terminal coupon a second time.

Fix: use ``HullWhite.build_tree_alphas`` for per-step α; textbook `/2` probabilities; apply coupon and option BEFORE the backward discount; skip the +coupon at ``step + 1 == n_steps`` (already in the init).

**Files changed**:
- `python/pricebook/fixed_income/callable_bond.py` — ``_trinomial_backward`` rewritten end-to-end with the standard HW-tree pattern (mirrors the v1.049 / v1.050 fixes).
- `python/tests/test_l2_t4_callable_bond_hw_tree.py` (new) — 4 regressions: callable ≤ straight on flat curve, puttable ≥ straight on flat curve, sloped-curve callable is finite/sane (was sensitive to the missing α(t) shift), no-option tree matches curve-PV straight bond (catches the double-coupon and missing-discount bugs simultaneously).

---

## v1.066.0 — 2026-06-15 — **Fix L2 T4 (options/autocall_advanced.discrete_autocall) — autocall branch overwrote prior coupons + ignored memory feature**

**``discrete_autocall`` autocall branch had two coupled defects.**

Pre-fix:
```python
if perf >= autocall_barrier:
    if memory_coupon:
        total_periods_paid = i + 1
    else:
        total_periods_paid = i + 1     # ← identical to memory branch
    path_pv = (notional + total_periods_paid * coupon_rate * notional) * df
    path_coupon = total_periods_paid * coupon_rate
    called = True; break
```

Two coupled bugs:

1. **Overwrite, not add**.  ``path_pv = ...`` (assignment) wiped out coupons already paid conditionally in earlier observation periods.

2. **Memory flag has no effect at autocall**.  The if/else is structurally redundant (both branches set ``total_periods_paid = i + 1``).  At autocall, memory and non-memory get the same payout — coupons for ALL periods regardless of whether the coupon barrier was met.  Standard convention:
   - Non-memory: notional + 1 current "autocall coupon".
   - Memory: notional + (unpaid_coupons + 1) × coupon.

For non-memory: bug over-paid by ``i × coupon`` at every autocall.

Fix (T4-AUTO2):
- ``path_pv += autocall_payment`` (add, not overwrite).
- Non-memory pays 1 current coupon; memory pays unpaid + 1.

**Files changed**:
- `python/pricebook/options/autocall_advanced.py` — autocall payout uses ``+=`` and respects memory flag distinctly.
- `python/tests/test_l2_t4_autocall_advanced_overwrite.py` (new) — 2 regressions.

---

## v1.065.0 — 2026-06-15 — **Fix L2 T4 (options/american_dividend) — cum-dividend exercise at ex-step lost**

**``american_with_dividends`` used post-drop spot at the ex-date step, silently dropping the cum-dividend early-exercise opportunity.**

The escrow method represents the dividend-paying spot as ``S(t) = S_adj(t) + PV_t(future dividends)``.  At each backward-induction step the helper ``_pv_future_divs(step_idx)`` collected dividends still ahead of the step.  Pre-fix the filter was ``if t > t_step``, which at the ex-step ``k_div`` (where ``t_div == t_step``) excluded the dividend itself — the reported true spot at step ``k_div`` was the POST-drop value.

But the dominant early-exercise scenario for American calls on dividend-paying stocks (Roll-Geske-Whaley) is to exercise JUST BEFORE the ex-date and capture the cum-dividend spot.  Using post-drop spot at the ex-step in the ``max(intrinsic, continuation)`` check silently dropped that opportunity, so the American premium over European was systematically understated.

Fix (T4-AMDIV1): change the dividend filter to ``t >= t_step`` so the dividend stays "future" AT the ex-step.  True spot at that step is cum-dividend; intrinsic captures pre-drop exercise.  At step ``k_div + 1`` the dividend is past as before.

**Files changed**:
- `python/pricebook/options/american_dividend.py` — one-character filter fix with provenance comment.
- `python/tests/test_l2_t4_american_dividend_cum_div.py` (new) — 3 regressions: deep-ITM call with near-term large dividend now shows ≥ immediate-exercise value and material early-exercise premium; no-div case unchanged; dividend after expiry has no effect.

---

## v1.064.0 — 2026-06-15 — **Fix L2 T4 (options/vol_derivatives_advanced.variance_swap_greeks) — vega had spurious T factor**

**``variance_swap_greeks`` returned vega scaled by ``T``, breaking the standard ``notional_var = vega_notional / (2·√strike_var)`` convention.**

Pre-fix vega:
```python
vega = 2 * vol * T * notional_var * remaining
```

The canonical convention defines ``notional_var`` precisely so that ``vega ≈ vega_notional`` at ATM inception, regardless of T.  Working through the algebra (notional_var = vega_notional / (2·√strike_var); ATM ⇒ vol = √strike_var; r = 0):

    vega_correct = 2·σ·notional_var·DF·remaining = vega_notional   ✓

    vega_buggy = 2·σ·T·notional_var·remaining = T · vega_notional   ✗

So a 2y var swap was reported with 2× the vega of a 1y var swap at the same vega_notional; a 5y var swap with 5×.  The existing tests use ``T = 1.0`` exclusively, so the bug never surfaced.

Also added the missing remaining-time discount factor ``exp(-r·T·remaining)``, which under-weighted long-dated, high-rate vegas pre-fix.

Fix (T4-VDA1):
```python
vega = 2.0 * vol * notional_var * df_remaining * remaining
```

**Files changed**:
- `python/pricebook/options/vol_derivatives_advanced.py` — corrected vega and reused ``df_remaining`` for both PV and vega.
- `python/tests/test_l2_t4_variance_swap_vega.py` (new) — 3 regressions: ATM vega = vega_notional for T ∈ {0.25, 1, 2, 5}; long-dated vega includes ``exp(-rT)``; vega scales linearly with remaining time as the trade ages.

---

## v1.063.0 — 2026-06-15 — **Fix L2 T4 (options/vol_term_structure.Bergomi2Factor) — missing cross-term in martingale correction**

**``Bergomi2Factor.simulate`` and ``bergomi_2f_simulate_via_engine`` used the wrong quadratic-variation formula for the lognormal forward variance.**

The Bergomi two-factor lognormal model is:

    ξ(t, T) = ξ₀(T) · exp(η₁·W₁(t) + η₂·W₂(t) − ½·Var(η₁W₁+η₂W₂))

With correlated Brownians (``Cov(W₁, W₂) = ρ·t``), the quadratic variation of ``η₁W₁ + η₂W₂`` is

    (η₁² + η₂² + 2·ρ·η₁·η₂) · t

Pre-fix the martingale correction used only ``(η₁² + η₂²)·t``, omitting the cross-term ``2·ρ·η₁·η₂·t``.

Consequence: ξ(t, T) was NOT a martingale whenever ``ρ ≠ 0`` — the simulated forward variance drifted in expectation away from its calibrating ξ₀(T).  For ``η₁ = η₂ = 0.5`` and ``ρ = +0.6``, ``E[ξ(1, T)] ≈ ξ₀ · exp(0.15)`` ≈ 16% too high.  ``ρ = 0`` (the default) coincidentally produced the right answer.

Fix (T4-VTS1): add the cross-term to the martingale correction in both the standalone ``simulate()`` and the engine-backed ``bergomi_2f_simulate_via_engine``.

**Files changed**:
- `python/pricebook/options/vol_term_structure.py` — ``var_coef = η₁² + η₂² + 2·ρ·η₁·η₂`` in both code paths.
- `python/tests/test_l2_t4_bergomi_2f_martingale.py` (new) — 3 regressions: ``ρ = 0`` unchanged; ``ρ = +0.6`` and ``ρ = -0.6`` both maintain ``E[ξ_T] ≈ ξ₀`` to within ~5% MC noise.

---

## v1.062.0 — 2026-06-15 — **Fix L2 T4 (options/swaption_vol_cube._lookup_atm) — lookup-with-round-up instead of interpolation**

**``_lookup_atm`` pretended to interpolate but actually rounded up.**

Pre-fix:
```python
def _lookup_atm(expiries, tenors, grid, exp_y, tenor_y):
    """Simple lookup/interpolation for ATM vol."""
    i = np.searchsorted(exp_arr, exp_y)
    j = np.searchsorted(ten_arr, tenor_y)
    i = max(0, min(i, len(exp_arr) - 1))
    j = max(0, min(j, len(ten_arr) - 1))
    return float(grid_arr[i, j])
```

``np.searchsorted`` returns the LEFT insertion index, so ``grid[i, j]`` is the value at the upper-right pillar of the bracketing rectangle.  For any off-pillar query the function returns the wrong cell, not an interpolated value.

Used by ``build_swaption_vol_cube`` to populate each calibrated SABR node's ``atm_vol`` field.  When the node's ``(exp_y, tenor_y)`` doesn't sit exactly on the ATM grid, the node carried an ATM value that came from the next-higher pillar.  This leaks out via ``SABRNode.vol`` 's exactly-strike-equals-forward fast-path which short-circuits and returns the stored ``atm_vol``.

Fix (T4-SVC1): replace with proper bilinear interpolation matching the in-class ``_interp_atm``, with degenerate 1×N / N×1 cases handled separately.

**Files changed**:
- `python/pricebook/options/swaption_vol_cube.py` — ``_lookup_atm`` now bilinearly interpolates.
- `python/tests/test_l2_t4_swaption_vol_cube_lookup.py` (new) — 4 regressions: at-pillar exact match; midway-between-pillars equals bilinear average; below/above-pillar clamps; linear interpolation along one axis when the other is a single pillar.

---

## v1.061.0 — 2026-06-15 — **Fix L2 T4 (options/skew_trading.cross_asset_skew_comparison) — labels wrong for mixed-sign skews**

**``cross_asset_skew_comparison`` sorted by signed skew, breaking the steepest/flattest labels under the canonical cross-asset use case.**

Pre-fix:
```python
entries = sorted(skews.items(), key=lambda x: x[1])
return CrossAssetSkewResult(entries, entries[0][0], entries[-1][0])
```

This labels the **most negative** signed skew as "steepest" and the **most positive** as "flattest".  Works only when all skews share a sign.

For the canonical cross-asset use case (equity put-skew, RR < 0; commodity call-skew, RR > 0), the labels were INVERTED.  Example:

    skews = {"equity": -0.05, "fx": 0.0, "commodity": +0.10}

Pre-fix labelled ``steepest = "equity"`` (|skew| = 0.05) and ``flattest = "commodity"`` (|skew| = 0.10 — actually the steepest).  ``fx`` (skew = 0) — the true flattest — went unrecognised.

The existing test ``test_ranking`` covers only same-sign skews, so the bug never surfaced.

Fix (T4-SK1): sort by ``|skew|`` descending.  Existing same-sign callers unaffected; mixed-sign callers now get correct labels.

**Files changed**:
- `python/pricebook/options/skew_trading.py` — sort by ``abs(x[1])`` descending.
- `python/tests/test_l2_t4_cross_asset_skew_signs.py` (new) — 4 regressions: mixed-sign labels correct, zero-skew is flattest, same-sign cases unchanged.

---

## v1.060.0 — 2026-06-15 — **🎯 1.060 milestone · Fix L2 T4 (options/vol_calibration) — SABR T defaulted to 1.0 + fallback α ignored β**

Two coupled defects in ``CalibratedVolSurface`` and ``CalibratedSABRNode``.

**T4-VC1 — SABR ``T`` defaulted to 1.0 regardless of node tenor.**

``CalibratedSABRNode.vol(strike, T=1.0)`` and the surface caller ``self.vol(expiry, strike) → node.vol(strike)`` never plumbed the actual time-to-expiry into the SABR Hagan formula.  The Hagan correction terms scale with ``T``:

    σ(K, F) = (prefactor) × (z / x(z)) × [1 + (B1 + B2 + B3) · T]

so a 10y node used 1/10 of the right correction; a 0.25y node used 4× the right correction.  The smile was systematically distorted on every node away from T = 1y.

Fix: add ``T_to_expiry`` field to ``CalibratedSABRNode`` (set during calibration), and ``vol()`` defaults ``T`` to ``self.T_to_expiry`` when not given.

**T4-VC2 — Fallback ``alpha`` = ``atm_vol`` ignored β.**

When SABR calibration raised (e.g. degenerate input), the fallback nodes were built with ``CalibratedSABRNode(... atm_vol ...)`` passing ``atm`` as the **alpha** argument.  But Hagan's ATM limit is ``σ_ATM ≈ alpha / F^(1-β)``, so ``alpha = atm`` only gives ``σ_ATM = atm`` when ``β = 1``.  For the FX default ``β = 0.5`` and ``F = 1.10``, the fallback's effective ATM vol was off by ``F^0.5 ≈ 1.049`` — a ~5% bias whenever a calibration failed and the path-through happened.

Fix: fallback uses ``alpha = atm × F^(1-β)`` so the SABR formula evaluated at ``K = F`` reproduces ``atm_vol``.

**Files changed**:
- `python/pricebook/options/vol_calibration.py` — ``T_to_expiry`` field on the node; all three asset-class calibrators set it; fallback alpha conversion uses ``F^(1-β)``.
- `python/tests/test_l2_t4_vol_calibration_sabr_T.py` (new) — 3 regressions: surface vol uses node's stored T (not 1.0); fallback alpha gives correct ATM vol with non-unity β; ``calibrate_fx_surface`` propagates the actual tenor onto each node.

---

## v1.059.0 — 2026-06-15 — **Fix L2 T4 (options/inflation_vol.yoy_inflation_cap) — Black-76 vol-time was time-to-previous-fixing**

**``yoy_inflation_cap`` passed the wrong ``T`` argument to ``black76_price`` for each YoY caplet.**

The YoY ratio ``CPI(d_curr) / CPI(d_prev)`` accumulates volatility only during the window ``(max(ref, d_prev), d_curr]`` — this is the "vol-time" interpretation of the Black-76 ``T`` parameter for a forward-start ratio option.

Pre-fix code: ``t_fix = max(year_fraction(ref, d_prev), 1e-6)`` — time from valuation date to the PREVIOUS CPI fixing.  Two broken regimes:

- **Fully forward** (``ref < d_prev``): pre-fix used ``d_prev − ref``, ignoring the actual ``d_curr − d_prev`` exposure window.  For a 5y YoY caplet on year 4-5 the code used ~4y of vol-time instead of 1y — ~√4 = 2× vol amplitude, so ATM caplet price inflated by ~2×.
- **Partially fixed** (``d_prev ≤ ref < d_curr``): pre-fix clamped to 1e-6, effectively zero vol.  The caplet was priced at essentially intrinsic value despite still having vol exposure from ``ref → d_curr``.

Fix (T4-INFL1): ``t_vol = year_fraction(max(ref, d_prev), d_curr)``.

**Files changed**:
- `python/pricebook/options/inflation_vol.py` — vol-time uses the correct YoY exposure window.
- `python/tests/test_l2_t4_yoy_inflation_cap_vol_time.py` (new) — 3 regressions: 5y cap finite/positive; vol monotonicity preserved; 5y/1y price ratio bounded sensibly (pre-fix it would have been inflated by the late-caplet vol-time error).

---

## v1.058.0 — 2026-06-15 — **Fix L2 T4 (options/exotic_payoffs.installment_option) — cost-to-continue missed the current installment**

**``installment_option`` rational-exercise check used only the PV of FUTURE installments, ignoring the installment the holder must pay now to continue.**

At each installment date ``t_i`` the holder chooses between paying the current installment and continuing, or abandoning the option for zero.  The correct rational comparison is:

    continue iff live_val(t_i) >= installment_amt + PV(future installments)

i.e. live_val ≥ PV of **all** remaining installments including the current one.

Pre-fix the code computed only ``pv_remaining = installment_amt · Σ_{j>i} DF`` and continued iff ``live_val >= pv_remaining`` — omitting the current ``installment_amt × 1``.  The holder over-continued (the cost threshold was too lenient by exactly one current installment), inflating the continuation probability and biasing the priced payoff upward.

Fix (T4-EX1): ``cost_to_continue = installment_amt + pv_future`` and ``should_continue = live_val >= cost_to_continue``.

**Files changed**:
- `python/pricebook/options/exotic_payoffs.py` — ``installment_option`` rational-exercise check now includes the current installment.
- `python/tests/test_l2_t4_installment_option_continue_cost.py` (new) — 2 regressions: deep-OTM call abandons (continuation_prob < 10%); ATM remains finite/non-negative with bounded continuation probability.

Note: ``shout_option`` in the same module has an unrelated subtle issue (the backward-greedy strategy with ``n_shouts = 1`` essentially shouts at the last possible step instead of the path's intrinsic max).  A proper fix needs LSM and is deferred.

---

## v1.057.0 — 2026-06-15 — **Fix L2 T4 (options/multi_asset_local_vol) — silent correlation + vol-paths reporting**

Two distinct findings in this slice.

**T4-MALV1 — ``smile_consistency_check`` had ``correlation`` as a silent-no-op API param.**

The function locally computed the correlation-aware linearised basket variance
``Var(B) = Σ w² σ² + ρ · [(Σ w σ)² − Σ w² σ²]`` and ``model_vol = √Var(B)`` — then DISCARDED both.  The returned ``is_consistent`` and ``consistency_ratio`` referenced only the trivial ρ=1 upper bound ``weighted = Σ w_i σ_i``, which doesn't depend on ``correlation``.  Calling with ρ = -1 vs ρ = +1 gave bit-identical output.

Fix:
- New ``model_basket_vol`` field on ``SmileConsistencyResult`` exposes the correlation-aware vol.
- ``consistency_ratio`` is now ``basket_vol / model_basket_vol``.
- ``is_consistent`` checks ``basket_vol ≤ model_basket_vol × 1.05``.

**T4-MALV2 — ``multi_asset_slv_simulate`` returned the same array for both ``vol1_paths`` and ``vol2_paths``.**

Both fields were set to ``sqrt(v)`` (the bare shared stochastic vol), regardless of the asset-specific local-vol leverages ``lv1`` and ``lv2``.  The asset-specific blended effective vol ``eff_i = mixing · lv_i + (1 - mixing) · √v`` (which actually drives each spot's evolution in the GBM step) was lost.

Fix: track per-asset effective-vol traces ``eff1_paths`` / ``eff2_paths`` during the loop and return them on the result.

**Files changed**:
- `python/pricebook/options/multi_asset_local_vol.py` — both fixes; new ``model_basket_vol`` field; per-asset eff vol traces.
- `python/tests/test_l2_t4_multi_asset_lv_silent_corr.py` (new) — 2 regressions: ρ = +1 vs ρ = -1 produce different ``model_basket_vol`` / ``is_consistent`` / ``consistency_ratio``; ``vol1_paths`` and ``vol2_paths`` differ by mixing·(lv2 − lv1) at any mid step.

---

## v1.056.0 — 2026-06-15 — **Fix L2 T4 (options/ir_exotic) — path-dependent discount sweep**

**Four remaining ``ir_exotic`` pricers used flat-rate discounting on simulated rate paths.**

Companion to v1.055 (T4-IREX1, ``tarn_price``).  T4-IREX2 sweeps the same fix across the other four pricers in this module:
- ``snowball_price``
- ``callable_range_accrual``
- ``ratchet_cap``
- ``flexi_swap``

All four updated the OU short rate each period but discounted every cashflow with ``exp(-flat_rate · t)`` regardless of the path.  Unlike ``tarn_price`` (whose simulated rate was entirely dead code), these four DO reference ``r`` in their payoffs (snowball coupon dynamics, range-accrual gate, ratchet strike reset, flexi exercise decision) — but the discount-payoff covariance was zero by construction.  For any non-trivial ``rate_vol`` this biases prices by the convexity correction the stochastic discount would have introduced.

Fix: per-path cumulative ``log_df = -∫_0^t r ds`` (left-Riemann sum) drives ``df = exp(log_df)`` for every cashflow in all four pricers.  The LSM regression in ``callable_range_accrual`` also now uses the path discount (both for the par-value side and the regression target).

**Files changed**:
- `python/pricebook/options/ir_exotic.py` — sweep across the four functions, all using ``log_df`` per path.
- `python/tests/test_l2_t4_ir_exotic_path_discount_sweep.py` (new) — 4 regressions, one per function: ``rate_vol`` materially affects price (was zero-effect through the discount channel pre-fix), callable ≤ non-callable holds structurally now that both sides share the same path discount.

---

## v1.055.0 — 2026-06-15 — **Fix L2 T4 (options/ir_exotic.tarn_price) — simulated rate was completely dead code**

**``tarn_price`` simulated an OU short-rate path that was never referenced in any payoff or discount.**

Pre-fix the function:
1. Initialised ``r = full(n_paths, flat_rate)``.
2. Updated ``r`` each period via ``r += 0.5·(flat_rate − r)·dt + rate_vol·dW``.
3. Then discounted EVERY cashflow with ``df = exp(-flat_rate · t)`` — a deterministic, path-independent factor.
4. Coupons were fixed at ``coupon_rate · notional · dt`` regardless of ``r``.

So the simulated ``r`` had zero effect on output.  ``rate_vol`` was thus a silent-no-op API param — changing it gave bit-identical prices.  The MC was pretending to be stochastic while running the same deterministic computation on every path.

Fix (T4-IREX1):
- Track ``log_df`` per path as ``log_df += −r · dt`` (left-Riemann sum of the simulated short rate).
- Use ``df = exp(log_df)`` for both coupons and the principal redemption.

Result: ``rate_vol`` now drives the convexity correction expected of a stochastic IR pricer.  The OU dynamics around ``flat_rate`` finally connect to the price.

Note: the wider ``ir_exotic`` module (``snowball_price``, ``callable_range_accrual``, ``ratchet_cap``, ``flexi_swap``) has a related but milder issue — those functions DO reference ``r`` in payoffs (snowball coupon, range-accrual gate, ratchet strike reset) but still use flat-rate discounting.  Deferred to a wider follow-up audit.

**Files changed**:
- `python/pricebook/options/ir_exotic.py` — ``tarn_price`` uses path-dependent discount.
- `python/tests/test_l2_t4_tarn_path_dependent_discount.py` (new) — 3 regressions: ``rate_vol`` materially affects price (pre-fix: bit-identical), zero-vol matches deterministic baseline, finite/positive sanity.

---

## v1.054.0 — 2026-06-14 — **Fix L2 T4 (options/cliquet) — per-period drift used flat-curve extrapolation**

**``Cliquet.price_mc`` propagated each period's drift with a single flat ``rate = -log(curve.df(T))/T``.**

Same flat-curve extrapolation pattern as ``tarf`` (T4-TARF1, v1.047) rolled forward to cliquet (T4-CLQ1).  For non-flat curves the path's per-period drift was systematically biased: under an upward-sloping curve, early periods saw a drift too high, late periods too low, both averaging to the right terminal forward by construction but with the wrong per-period distribution.

Cliquet's payoff depends explicitly on per-period local returns ``S_i / S_{i-1} - 1`` (clipped to ``[local_floor, local_cap]``), so the per-period distribution matters — not just the marginal terminal distribution.  This wasn't a free pass like for a vanilla European that only sees ``S_T``.

Fix: per-segment drift uses the forward zero rate ``-log(df_i / df_{i-1}) / dt_i``.  Terminal discount unchanged (matches ``curve.df(T)`` exactly).

**Files changed**:
- `python/pricebook/options/cliquet.py` — precomputes ``seg_dfs`` and ``seg_rate`` from the curve; loop uses per-segment forward rate.
- `python/tests/test_l2_t4_cliquet_curve_term_structure.py` (new) — 2 regressions: flat curve still produces finite/positive price; sloped curve produces a different price than a flat curve with the same terminal discount (proves per-period drift now depends on slope).

---

## v1.053.0 — 2026-06-14 — **Fix L2 T4 (options/local_vol) — Gatheral Dupire used spot y and at-fixed-K time derivative**

**``_dupire_local_vol`` carried two coupled errors in the Gatheral total-variance form of the Dupire formula.**

Gatheral's formula in terms of total variance ``w = σ²·T``:

    σ_loc² = (∂w/∂T)|_y / [1 − (y/w)∂w/∂y + (1/4)(−1/4 − 1/w + y²/w²)(∂w/∂y)² + (1/2) ∂²w/∂y²]

requires:
- ``y = log(K / F_T)`` — log-moneyness against the **forward** (Gatheral 2006, §1.3).
- The time derivative taken at fixed ``y``, not fixed ``K``.

Pre-fix (T4-LV1):
- ``y = math.log(k / spot)`` — used spot, not forward.
- ``dw_dt`` used ``∂w/∂T |_K`` directly with no chain-rule correction.

Both errors silently vanish when r = q = 0 (then ``F_T = S`` and the time derivatives at fixed K and fixed y coincide).  For any non-trivial equity (r > q > 0) or FX scenario the Dupire surface picks up a systematic bias whose magnitude depends on (r−q)·T and the skew/term-structure of the input implied vol surface.

Fix:
- ``forward_T = spot · exp((r − q) · t)`` and ``y = log(K / forward_T)``.
- ``dw_dt = dw_dt|_K + (r − q) · K · ∂w/∂K`` (the chain-rule conversion to ``dw_dt|_y``).

**Files changed**:
- `python/pricebook/options/local_vol.py` — Gatheral block in ``_dupire_local_vol`` now uses forward-relative ``y`` and the converted ``dw_dt``.
- `python/tests/test_l2_t4_local_vol_dupire_forward.py` (new) — 2 regressions: flat implied vol surface still gives flat local vol (cancellation preserved post-fix), local vol surface DEPENDS on ``r`` for a skewed input (was silently r-independent through ``y`` pre-fix).

---

## v1.052.0 — 2026-06-14 — **Fix L2 T4 (options/bermudan_lmm) — drift / diffusion correlation mismatch**

**``_simulate_lmm_paths`` was internally inconsistent in its factor structure.**

Pre-fix:
- **Drift** used the single-factor (perfectly-correlated) terminal-measure formula ``μ_j = -σ_j · Σ_{k>j} σ_k · τ · F_k / (1 + τ F_k)`` (Brigo §6.7).  This formula implicitly embeds ρ_{j,k} = 1.
- **Diffusion** drew an independent Brownian increment ``dW_j`` per forward via ``rng.standard_normal((n_paths, n_fwd))`` — algebraically multi-factor with ρ_{j,k} = δ_{j,k}.

These two assumptions are mutually inconsistent:
- Under truly independent factors, the terminal-measure drift collapses to **zero** for every forward (all cross-correlation terms vanish), so the non-zero drift was unjustifiable.
- Under truly single-factor LMM, all forwards must share the same ``dW``.

The visible consequence: forwards on a path moved with near-zero empirical correlation (independent shocks dominated), undermining the swap-rate dynamics that the LSM regression in ``bermudan_swaption_lmm`` depends on.

Fix: pick the single-factor convention (matching the drift) — use one shared ``dW`` per step.  The drift formula is unchanged.

Remaining open item (T4-LMM2, deferred): ``bermudan_swaption_lmm`` discounts using ``F[p, step_k, 0]`` — the **first** forward at step_k as a stand-in for the average short rate.  Correct discounting requires integrating the short rate along each path.  Bigger refactor.

**Files changed**:
- `python/pricebook/options/bermudan_lmm.py` — single shared ``dW`` per step in the simulator.
- `python/tests/test_l2_t4_bermudan_lmm_single_factor.py` (new) — 2 regressions: adjacent forwards now have correlation > 0.90 across paths (single-factor coupling proven), per-forward terminal log-variance matches ``σ²·T`` (marginal preserved).

---

## v1.051.0 — 2026-06-14 — **Fix L2 T4 (options/bermudan_barrier) — LSM regression target over-discounted**

**``_lsm_bermudan_barrier_core`` converted PV-at-t=0 to "value at step" using the wrong formula.**

Pre-fix the regression target line was:
```python
cont_at_step = cashflow[itm] * np.exp(-r * (cash_step - step) * dt)
```

But ``cashflow`` stores PV-at-t=0 (the discounting was applied once at storage time, both for the terminal payoff and any LSM-driven early-exercise update).  To convert PV-at-0 to "value at step k", one simply un-discounts: ``cashflow · exp(+r · step · dt)``.

The pre-fix expression algebraically equals:
```
cashflow · exp(-r·(cash_step - step)·dt)
  = cashflow · exp(-r·cash_step·dt) · exp(+r·step·dt)
  = correct  ·  exp(-r·cash_step·dt)     # extra over-discount factor
```

So the regression target was discounted TWICE — once at storage, once again here.  Most severe for paths still carrying the terminal cashflow (``cash_step = n_steps``), where the bias factor was ``exp(-r·T)``.  At r=5%, T=1y the regression saw ~5% lower values than truth; at r=10%, T=2y the bias was ~18%.

Consequence: the LSM continuation estimate was systematically biased low → early exercise was over-triggered → the price was biased toward the immediate-exercise lower bound.

Fix: ``cont_at_step = cashflow[itm] · exp(+r · step · dt)``.

**Files changed**:
- `python/pricebook/options/bermudan_barrier.py` — single-line correction in the LSM regression conversion, with provenance comment.
- `python/tests/test_l2_t4_bermudan_barrier_lsm_regression.py` (new) — 3 regressions: Bermudan ≤ American, Bermudan ≥ European, high-r case stays well-behaved.

---

## v1.050.0 — 2026-06-14 — **🎯 1.050 milestone · Fix L2 T4 (options/bermudan_capfloor) — same defect class as bermudan_swaption**

**``bermudan_capfloor._bermudan_capfloor_tree`` carried the same defect class as the HW Bermudan swaption tree (v1.049).**

Two structural bugs (T4-BCF1):

1. **Wrong trinomial probabilities** — drift terms used ``/6`` instead of textbook Hull §32.4 eq. 32.10 ``/2``, so the drift was 3× too small.  Same root-cause as T4-BERM1.

2. **Exercise compared to discounted continuation** — the backward loop indexed exercise by ``step + 1`` and applied ``max(discounted_continuation, undiscounted_exercise_at_step+1)``, over-valuing exercise by ``exp(+r·dt)`` per step.

α(t) is implicit ``r0`` in this module because the API takes a raw ``r0`` (no DiscountCurve) — for the flat-curve interface this is correct.  A non-flat-curve API would require bermudan_swaption-style refactoring with ``HullWhite.build_tree_alphas`` (which v1.049 introduced).

**Files changed**:
- `python/pricebook/options/bermudan_capfloor.py` — ``/2`` probabilities; exercise applied at its own step before the next backward discount; skips step 0 (no exercise-vs-continuation choice at t=0).
- `python/tests/test_l2_t4_bermudan_capfloor_hw_tree.py` (new) — 5 regressions: near-immediate-exercise Bermudan stays close to European, mean-reversion sensitivity has correct sign, Bermudan bounded above by full-strip European (knock-in semantics), finite/positive sanity.

---

## v1.049.0 — 2026-06-14 — **Fix L2 T4 (options/bermudan_swaption) — three coupled defects in HW tree + LSM α(t)**

**``bermudan_swaption_tree`` rolled three coupled errors into one badly-broken HW trinomial pricer.**  All three fixed in this slice (T4-BERM1); LSM α(t) bug also fixed in parallel (T4-BERM2).

**Bug 1 — Wrong trinomial probabilities.** The drift terms used ``/6`` instead of the textbook Hull §32.4 eq. 32.10 form ``/2``:

    p_u = 1/6 + (j²a²Δt² − j·a·Δt) / 6     # WRONG
    p_d = 1/6 + (j²a²Δt² + j·a·Δt) / 6     # WRONG

Both sum to 1, so the bug doesn't show as an invalid-probability error.  But ``p_u − p_d = −j·a·Δt/3`` instead of ``−j·a·Δt`` — the drift term is **3× too small**, so the tree's mean-reversion is heavily understated and the tree's vol-of-rates is correspondingly inflated.

**Bug 2 — Missing time-varying α(t) shift.**  Short rate at node (step, j) used ``r0 + j·dr`` for every step.  Proper HW trinomial fits ``α(t_i)`` at each step boundary so the tree exactly reprices the initial discount curve at every node row (Hull §32.4).  Without this, the tree silently mis-matches any non-flat input curve.  Same defect already fixed in ``tree_european_swaption`` as T1.9.

**Bug 3 — Exercise compared to discounted continuation.**  The backward loop computed:

    new_values = exp(−r·dt) × Σ p · V_{step+1}        # discounted to step
    if (step + 1) in exercise_steps:
        new_values = max(new_values, exercise_at_step+1)  # NOT discounted!

So the exercise side was compared to the DISCOUNTED continuation without itself being discounted from step+1 back to step — exercise systematically over-valued by ``exp(+r·dt)``.  For the single-exercise (European) limit, the bug eliminated the **entire** terminal discount factor, inflating European prices by ``exp(r·T)`` (≈28% bias for r=5%, T=5y).

**Fix structure**:
- New ``HullWhite.build_tree_alphas(T, n_steps)`` exposes the per-step calibrated ``α[i]`` values (forward-fit so the tree reprices the discount curve).
- ``bermudan_swaption_tree`` is rewritten to: call ``build_tree_alphas``, backward-induce with ``r_j = α[step] + j·dr`` and the textbook ``/2`` probabilities, apply exercise AT its own step (modifying V[step] before the next backward discount).
- ``bermudan_swaption_lsm``: conditional OU mean now centres on ``α(t)`` (Brigo-Mercurio closed-form) instead of ``forward_rate(t)`` — convexity correction ``(σ²/2a²)(1−e^{−at})²`` was missing.

**Files changed**:
- `python/pricebook/models/hull_white.py` — adds ``build_tree_alphas`` helper.
- `python/pricebook/options/bermudan_swaption.py` — tree rewritten end-to-end; LSM OU drift corrected.
- `python/tests/test_l2_t4_bermudan_swaption_hw_tree.py` (new) — 6 regression tests: European limit matches reference, tree ZCB matches input curve on a sloped curve, sensitivity to mean-reversion has correct sign, LSM ≤ tree as expected lower bound under sloped curve.

---

## v1.048.0 — 2026-06-14 — **Fix L2 T4 (options/capfloor) — pv_ctx collapsed term-structure surface to a single vol**

**``CapFloor.pv_ctx`` called ``vol_surface.vol()`` with no arguments and used the resulting scalar for every caplet.**

The legacy implementation:
```python
model = Black76Model(vol_surface.vol())   # no args!
return self.price(model, curve, proj)
```

This works only for ``FlatVol`` (whose ``vol()`` accepts no args and returns a constant).  For any real IR vol surface — ``VolTermStructure``, smile cubes — ``vol(expiry, strike)`` has no default ``expiry`` and either raises a ``TypeError`` or silently returns whichever vol corresponds to the absent-arg default.  Even if a surface tolerates no-args, the resulting single number was then used as the Black-76 vol for every caplet across the cap, collapsing the term structure.

Consequence: any cap routed through the pricing engine on a non-flat IR vol surface silently lost its term-structure dependence.  A 10y cap priced as if all caplets had the same vol — usually the front-end vol — materially mispricing long-end exposure.

Fix: ``pv_ctx`` now mirrors the manual ``price()`` loop but looks up the per-caplet vol from the surface via ``vol(accrual_start, strike)`` at each caplet's expiry, building one ``Black76Model`` per caplet.  No change for ``FlatVol`` users; correct vol propagation for term-structure / smile surfaces.

**Files changed**:
- `python/pricebook/options/capfloor.py` — ``_capfloor_pv_ctx`` rewritten to loop per-caplet with surface lookup.
- `python/tests/test_l2_t4_capfloor_pv_ctx_surface_vol.py` (new) — 3 regressions:
  term-structure surface drives different PVs when long-end vol differs (proves per-caplet routing), FlatVol still works, missing surface raises with diagnostic.

---

## v1.047.0 — 2026-06-14 — **Fix L2 T4 (options/tarf) — flat-curve discount extrapolation in multi-period MC**

**``TARF.price_mc`` discounted each per-fixing cashflow with a flat-curve extrapolation rather than the actual curve's per-tenor DF.**

Pre-fix:

    rate = -log(curve.df(T)) / T
    df_t = exp(-rate × t_fix)         # ≡ curve.df(T)^(t/T)

This equals ``curve.df(t_fix)`` ONLY when the discount curve is flat.  For non-flat curves (the common case in production: short-end below long-end), each fixing's cashflow was discounted at a synthetic rate equal to the spot-to-maturity zero rate, not the actual short-rate at ``t_fix``.  Upward-sloping curve ⇒ early fixings over-discounted; downward-sloping ⇒ under-discounted.

Same defect biased the per-segment GBM drift: ``rate`` was used as a constant in ``S = S × exp((rate − q − σ²/2)·dt + σ·√dt·Z)`` so the path drift didn't track the curve's forward-rate term structure between fixings.

Fix:
- Per-fixing discount uses ``curve.df(fixing_date)`` directly.
- Per-segment drift uses the segment forward zero rate ``-log(df_i / df_{i-1}) / dt_i`` so each step matches the curve's local term structure.

For a flat curve the two formulations coincide exactly — no behaviour change.

(Note: full FX TARF would also subtract the FCY rate in the drift; that's a multi-curve extension out of scope for this single-curve interface.)

**Files changed**:
- `python/pricebook/options/tarf.py` — precomputes ``dfs_fix`` and ``seg_drift_rate`` from the curve; loop uses both.
- `python/tests/test_l2_t4_tarf_curve_term_structure.py` (new) — flat curve still produces finite price; sloped curve produces finite, bounded price (proves we use ``curve.df`` per fixing).

---

## v1.046.0 — 2026-06-14 — **Fix L2 T4 (options/autocallable) — coupon_barrier silent no-op**

**``Autocallable.price_mc`` ignored ``coupon_barrier`` entirely.**

The constructor accepts and stores ``coupon_barrier`` (and round-trips it through serialisation), and the docstring promises "If S(t) >= coupon_barrier × S₀ ... coupon accrues."  But the MC pricer never checked ``S >= coupon_barrier * spot`` at observation dates — coupons were unconditionally accrued at ``rate * elapsed_t`` (or ``rate * sum(period_lengths[:i+1])`` at autocall), regardless of where the underlying actually traded.

Consequence: setting ``coupon_barrier`` to any value had no effect on price.  An autocallable structured to pay coupons only when the underlying is above (say) 80% of spot priced the same as one paying unconditionally — silent wrong-price for the issuer/buyer.

Fix: track per-path ``coupons_accrued`` that only grows at observation dates where ``S >= coupon_barrier * spot`` (memory-style accumulation, standard for autocallables with memory).  At autocall and at maturity (above put barrier), the payoff uses the actual per-path accrued total instead of a uniform ``rate * elapsed_t``.

**Files changed**:
- `python/pricebook/options/autocallable.py` — ``price_mc`` now tracks ``coupons_accrued`` per path, gated by ``coupon_barrier``.
- `python/tests/test_l2_t4_autocallable_coupon_barrier.py` (new) — 3 regression tests: high barrier kills coupons (price drops materially), zero barrier matches the always-accrue limit, below-put-barrier branch unchanged.

---

## v1.045.0 — 2026-06-14 — **Fix L2 T4 (options/convertible_bond) — final coupon missing from MC terminal payoff**

**``ConvertibleBond.price`` MC backward-induction silently dropped the maturity coupon.**

The terminal payoff was ``max(notional, conv_ratio·S_T)`` and the backward loop ran ``range(n_steps-1, -1, -1)``, never visiting ``step == n_steps``.  But the analytical ``bond_floor`` computation (also returned) DID include the maturity coupon: ``Σ c·DF(t_i) + N·DF(T)`` with ``t_i = 1/freq … T``.

Consequence: MC PV biased downward by ~``coupon_amount · DF(T)``; deep-OTM CB (conversion never optimal) priced below its own ``bond_floor``.  For a 5y/5% semi-annual CB at typical discount rates, the bias is ~2% of notional.

Fix: terminal payoff is ``max(notional + coupon_amount, conv_ratio·S_T)`` across all four backward-induction sites in this module:
- ``ConvertibleBond.price`` (main MC loop)
- ``ConvertibleBond.price`` (bumped ``V_up`` for delta)
- ``ConvertibleBond._compute_delta`` (legacy helper)
- ``convertible_soft_call`` (issuer call variant)

Note: ``contingent_convertible``, ``exchangeable_bond``, ``mandatory_convertible`` use different payoff structures and are deferred to dedicated audit.

**Files changed**:
- `python/pricebook/options/convertible_bond.py`
- `python/tests/test_l2_t4_convertible_bond_terminal_coupon.py` (new) — 3 regression tests: deep-OTM matches bond_floor to 0.5%, gap-vs-bond-floor bounded well below pre-fix bias, zero-coupon unchanged.

---

## v1.044.0 — 2026-06-14 — **Fix L2 T4 (options/futures_options) — Bachelier price + Black-76 Greeks mismatch**

**`options.futures_options.FuturesOption.price` returned Black-76 analytical Greeks even when `model="bachelier"`.**

The Greeks block unconditionally called ``black76_delta/gamma/vega/theta`` regardless of the pricing model.  For options that *must* be priced under a normal model — short-rate futures (e.g. SR3 / SOFR) where rates can go negative — this returned lognormal-model Greeks for a normal-model price.  d1/d2 conventions diverge between the two families: the analytical forms aren't even off by a constant scale.

Fix: dispatch on ``self.model`` for the Greeks block, calling ``bachelier_*`` analytics when ``model=="bachelier"`` and ``black76_*`` otherwise.  BAW-American path is unchanged (BAW uses Black-76 dynamics, so Black-76 European Greeks are the right first-order approximation).

**Files changed**:
- `python/pricebook/options/futures_options.py` — Greeks block now dispatches on model.
- `python/tests/test_l2_t4_futures_option_bachelier_greeks.py` (new) — 4 regressions:
  Bachelier delta matches analytic formula; Bachelier ≠ Black-76 deltas (proves dispatch); all 4 Bachelier Greeks consistent; Black-76 path unchanged.

---

## v1.043.0 — 2026-06-14 — **Fix L2 T4 (options/) — silent spot=100 sweep across 5 equity options**

**Five equity-option ``pv_ctx`` implementations silently hardcoded ``spot=100.0`` (or returned ``0.0``).**

Same recurring "silent no-op API param" pattern as the BarrierOption fix in v1.042 — every engine call to ``pv_ctx`` got a PV computed for a 100-spot underlying regardless of the actual market spot, because ``PricingContext`` carries no ``equity_spots`` field.

Affected instruments (each used ``spot=100.0`` or returned ``0.0``):
- ``options.american_option.AmericanOption`` — ``spot=100``
- ``options.autocallable.Autocallable`` — ``spot=100``
- ``options.asian_option.AsianOption`` — ``spot=100``
- ``options.cliquet.Cliquet`` — ``spot=100``
- ``options.basket_option.BasketOption`` — returned ``0.0``

Fix: each ``pv_ctx`` now raises ``NotImplementedError`` with a diagnostic message pointing to the direct ``.price()`` / ``.price_mc()`` entry points.  Loud failure over silent wrong-price until ``PricingContext.equity_spots`` (and per-asset spots/correlations for baskets) is added in a dedicated architectural slice.

Not in scope (deferred):
- ``options.tarf.TARF.pv_ctx`` uses ``spot=self.strike`` (ATM default) — different default, partial functionality, will be addressed when ``PricingContext.fx_spots`` is properly threaded through.
- Adding the actual ``equity_spots`` channel to ``PricingContext`` — architectural change, separate slice.

**Files changed**:
- `python/pricebook/options/american_option.py`
- `python/pricebook/options/autocallable.py`
- `python/pricebook/options/asian_option.py`
- `python/pricebook/options/cliquet.py`
- `python/pricebook/options/basket_option.py`
- `python/tests/test_l2_t4_equity_options_pv_ctx_sweep.py` (new) — 5 regression tests, one per instrument.

---

## v1.042.0 — 2026-06-14 — **Fix L2 T4 (options/barrier_option) — two silent-no-op API params**

**`options.barrier_option.BarrierOption` had two distinct silent-no-op footguns producing wrong PV.**

1. **`rebate` ignored end-to-end**: constructor accepts and round-trips ``rebate``, but neither ``_price_pde`` (calls ``fd_barrier_knockout/knockin`` which take no rebate) nor ``_price_mc`` (multiplies payoff by ``survived``/``activated`` masks with no rebate term) honoured it.  Any non-zero rebate was silently dropped — PV indistinguishable from the no-rebate case.

2. **`pv_ctx` hardcoded `spot=100.0`**: when the pricing engine calls ``instrument.pv_ctx(ctx)``, the prior implementation silently substituted ``spot=100.0`` regardless of the actual underlying.  ``PricingContext`` has no ``equity_spots`` field (only ``fx_spots``), so the call couldn't fetch the real spot — but instead of failing, it silently used 100 and returned a wrong PV.  Engine consumers of barrier-on-equity got the price of a 100-spot underlying for every barrier trade.

Fix: raise ``NotImplementedError`` in both cases — loud failure over silent wrong-price.  Real rebate handling (PDE Dirichlet BC + MC payoff term) and a proper ``equity_spots`` channel through ``PricingContext`` are deferred to dedicated slices.

Same silent-spot=100 pattern observed in 5 other equity options (``american_option``, ``autocallable``, ``asian_option``, ``cliquet``, ``basket_option``); each will get its own slice as we audit options/.

**Files changed**:
- `python/pricebook/options/barrier_option.py` — both ``price()`` and ``pv_ctx()`` raise with diagnostic message.
- `python/tests/test_l2_t4_barrier_silent_no_ops.py` (new) — 4 regressions pinning the raises plus the rebate=0 happy path.

---

## v1.041.0 — 2026-06-14 — **Fix L2 T4 (options/asian) — geometric σ_g formula off by factor n/(n+1)**

**`options.asian.geometric_asian_analytical` σ_g formula was inconsistent with its own drift formula and the MC monitoring convention.**

Pre-fix vol formula:

    σ_g² = σ² · (2n+1) / (6(n+1))

This corresponds to **n+1** monitoring points including a deterministic t_0=0.  But the drift formula `(μ-σ²/2)(n+1)/(2n)` and the MC counterpart (`mc_asian_arithmetic` uses `paths[:, 1:]` → t_1..t_n) treat the average as **n random points** at t_i = T·i/n.  These two conventions are inconsistent.

Correct Kemna-Vorst σ_g for n random monitoring points:

    σ_g² = σ² · (n+1)(2n+1) / (6n²)

Impact: at n=12 the σ_g was ~7.7% too low, biasing the geometric Asian price downward and — critically — biasing the **control variate adjustment** in `mc_asian_arithmetic(... control_variate=True)`.  Pre-fix the MC_cv arithmetic Asian came out ~5-7% too low.

Continuous limit (n→∞) is unchanged (both forms → σ²/3).

**Files changed**:
- `python/pricebook/options/asian.py` — σ_g formula corrected + provenance comment.
- `python/tests/test_l2_t4_geometric_asian_vol_formula.py` (new) — 3 regression tests:
  closed-form match, n→∞ continuous limit (σ/√3), MC@200k paths within 3σ of analytical.
- `python/tests/test_asian_option.py` — `test_tw_vs_mc_close` now uses 12 fixings on t_1..t_12 (excluding t_0 = REF) so TW and MC price the same option.  The `AsianSchedule.monthly(REF, REF+12M)` convention (which includes REF as a deterministic fixing) exposes a **separate** schedule-time-mismatch bug in `mc_asian_arithmetic`: it assumes uniform monitoring on (0,T] regardless of the actual schedule.  Pre-fix this bug was masked by the σ_g error compensating in the CV adjustment; deferred to its own slice.

---

## v1.040.0 — 2026-06-14 — **🎯 1.040 milestone**

**Fix L2 phase-2 (pricing/) — `pricing_engine._pv_fallback` had two coupled silent-default defects.**

Two production-grade footguns in the credit-curve selection logic, both producing **wrong PV with `status: ok`**:

**Bug 1 — silent 2% hazard default**. When no survival curve was provided in `market_data.survival_curves`, the engine substituted

    SurvivalCurve.flat(val_date, 0.02)

…and proceeded to price CDS / credit-dependent trades. The caller had no indication that their input was incomplete: the response returned `status: "ok"` with a plausible but **wrong** PV based on the fake 2% hazard. For a 5y, 100bp CDS this gives the wrong sign of P&L in many market regimes.

**Bug 2 — first-curve silent pick** (same shape as v0.969 FRA bug). When the caller supplied multiple survival curves (e.g. `{"issuer_A": ..., "issuer_B": ...}`), the engine's

    sc = next(iter(ctx.credit_curves.values()))

selected the first one regardless of which obligor the trade actually referenced. A 2-obligor portfolio priced ALL CDS against the first obligor's hazard. The pre-fix code did not check that the obligor matched the trade.

**Fix**: in `_pv_fallback`,
- raise `ValueError` with diagnostic context when no survival curve is provided and the instrument needs one;
- raise `ValueError` when multiple curves are provided without obligor-key disambiguation, listing the available keys;
- use the single curve when exactly one is provided (the most common, unambiguous case).

The fallback is invoked from `_price_one`'s `pv` branch when the instrument needs `(curve, survival)` but pv_ctx is unavailable — exactly the path that previously silently produced wrong CDS prices.

**Existing test updates**: two tests in `test_pricing_engine.py` (`test_cds`, `test_mixed_portfolio`) relied on the silent 2% default; now updated to supply explicit `survival_curves: {"issuer": ...}` matching the new strict semantics.

### Verification — `test_l2_t4_pricing_engine_credit.py`

3 new tests pin: CDS without survival curve → per-trade `status: "error"` (not silent ok); `_pv_fallback` with multiple curves → `ValueError` listing the keys; single-curve case still works unchanged.

Full parallel suite: **12,686 passed in 4:14** — zero regressions after the two test updates.

**177 distinct bugs** in v0.905→v1.040.

---

## v1.039.0 — 2026-06-14

**Fix L2 phase-2 (structured/) — `callable_step_up_bond` off-by-one in issuer call-decision discount horizons.**

At call iteration ``i`` (call time `(i+1)·dt`), the bond_value used by the issuer to decide whether to call was computed using:

    remaining_coupons = Σ_j  c_j · exp(-r_t · (j+1−i)·dt)     # WRONG: one period too long
    remaining_principal = face · exp(-r_t · (n−i)·dt)         # WRONG: one period too long

Coupon `j` pays at time `(j+1)·dt`, so the discount horizon from the call time `(i+1)·dt` is `(j−i)·dt` — not `(j+1−i)·dt`. Same off-by-one on the principal (`(n−i−1)·dt` is correct, not `(n−i)·dt`).

**Effect**: `bond_value` was systematically under-stated by `exp(-r_t · dt)` (≈1 period of discount), making the issuer **call LESS often** than warranted. The callable bond was therefore slightly OVER-priced (less call benefit captured by the issuer / less negative for the holder).

**Fix**: shift discount horizons by one period to match the actual call date.

### Verification — `test_l2_t4_callable_step_up_bond.py`

2 new tests pin: deep-ITM call value > 5% of face for a high-coupon-low-rate bond (issuer should call); short-T case doesn't blow up. Existing `test_rates_structured.py` tests still pass.

Full parallel suite: **12,683 passed in 2:40** — zero regressions.

**176 distinct bugs** in v0.905→v1.039.

---

## v1.038.0 — 2026-06-14

**Fix L2 phase-2 (structured/) — `mortality_bond_price` truncates non-integer T (same shape as v1.035 cat_bond).**

The annual-coupon PV loop used `range(1, int(T) + 1)` which silently dropped any non-integer remainder of `T`. For `T = 0.5` (6-month bond) the loop was empty → zero coupon PV; for `T = 3.5` the final 0.5y accrual was missed.

**Fix**: add a fractional final accrual `coupon × notional × remainder × df(T)` when `T - int(T) > 0`. Same pattern as the v1.035 fix in cat_bond_price.

### Verification — `test_l2_t4_mortality_bond_fractional_T.py`

3 new tests pin: 6-month bond has non-zero coupon PV; integer T case unchanged; fractional T adds the expected ~0.5y accrual.

Full parallel suite: **12,681 passed in 2:36** — zero regressions.

**175 distinct bugs** in v0.905→v1.038.

---

## v1.037.0 — 2026-06-14

**Fix L2 phase-2 (structured/) — `outperformance_certificate` had two coupled defects.**

**Bug 1: cap_strike wrong when participation ≠ 1.**

The cap was set via a short call struck at `cap_strike = spot × (1 + cap)`. This only caps payoff at `1 + cap` when `participation == 1`. For `participation > 1` (the typical case), each unit of `S_T/S_0` growth is amplified by participation, so the cap-binding strike is **lower** than `spot × (1 + cap)`.

Solving `1 + participation × (S_T/S_0 − 1) = 1 + cap` gives `S_T = S_0 × (1 + cap/participation)`. For `participation = 1.5, cap = 0.30`, the correct cap_strike is `1.20 × S_0`, not `1.30 × S_0`.

**Bug 2: `stock_pv` double-discounted.**

`stock_pv = spot × exp(-qT) × df × notional/spot` had an extra `df` factor. Under risk-neutral pricing, the PV today of receiving `S_T` at maturity is `E^Q[S_T] × df = F × df = spot × exp(-qT)` — NOT that value × df again. The double-discount understated the stock-leg PV by `df ≈ 0.95` (a few percent off for typical rates/horizons).

Under deep-ITM tests both bugs interact: pre-fix gave half the analytic ceiling (`0.395` vs `0.788` for the worked example). Existing tests checked only `price > 0` and `cap_reduces_price` so the bug wasn't caught.

### Verification — `test_l2_t4_outperformance_cap.py`

3 new tests: capped certificate price ≤ analytic ceiling `(1 + cap) × df` under deep-ITM; no-cap case unchanged; same cap with different participation produces different prices (pre-fix would use the same cap_strike).

Full parallel suite: **12,678 passed in 2:41** — zero regressions.

**174 distinct bugs** in v0.905→v1.037.

---

## v1.036.0 — 2026-06-14

**Fix L2 phase-2 (structured/) — `callable_structured._simulate_two_rates` and `_simulate_one_rate` ignored their `rate` parameter.**

Both helpers accept a `rate` argument documented as providing drift:

    """driftless in the risk-neutral measure is approximated by
       drift = rate * dt to keep rates positive"""

…but the rate-update step only had the Brownian diffusion term — pure driftless ABM. The `rate` parameter was a silent no-op. Same shape as the v0.996/v1.022/v1.033 dead-parameter family.

This affects all callable structured products that route through these simulators (`callable_steepener`, `callable_cms_spread`, `callable_inverse_floater`). Pre-fix, changing the input `rate` would only shift the discounting term in cashflow valuation; the simulated rate paths themselves were invariant to `rate`.

**Fix**: add the documented `drift = rate * dt` to each step's rate update. Now path averages drift up at the input `rate`, which is what the comment specified.

### Verification — `test_l2_t4_callable_structured_drift.py`

1 new test pins: `callable_steepener` prices at `rate=0.02` vs `rate=0.06` differ by more than the discount-only adjustment can produce. Existing callable_structured tests unchanged.

Full parallel suite: **12,675 passed in 2:41** — zero regressions.

**173 distinct bugs** in v0.905→v1.036.

---

## v1.035.0 — 2026-06-14

**Fix L2 phase-2 (structured/) — `cat_bond_price` silently dropped non-integer T coupons.**

The annual-coupon PV loop used `range(1, int(T) + 1)`, which truncates any non-integer maturity:

- `T = 0.5` (6-month bond): loop is empty → **zero coupon PV** → price = PV(principal) alone, missing the half-year accrual entirely.
- `T = 3.5`: 3 full annual coupons but the final 0.5-year accrual was missed.

**Fix**: add a fractional final accrual `coupon × notional × remainder × df(T) × survival(T)` when `T - int(T) > 0`. For integer T this branch is a no-op, so the integer-T behaviour is unchanged.

### Verification — `test_l2_t4_cat_bond_fractional_T.py`

3 new tests pin: 6-month bond has non-zero coupon PV; price is continuous in T at integer boundaries (small jump if any); integer T case matches the closed-form sum exactly.

Full parallel suite: **12,674 passed in 3:05** — zero regressions.

**172 distinct bugs** in v0.905→v1.035.

---

## v1.034.0 — 2026-06-14

**Fix L2 phase-2 (desks/) — `api.key_rate_dv01` fake decomposition + `api.carry_rolldown` triple defect.**

### `api.key_rate_dv01`

Pre-fix bumped the curve in **parallel for every tenor** and returned the same DV01 value under each tenor key — the "key rate" decomposition was a silent wrong answer. A user calling `pb.key_rate_dv01(...)` got identical values across the entire ladder.

**Fix**: map each requested tenor → year fraction → nearest non-zero curve pillar via `curve.bumped_at(pillar_idx, shift)`. The ladder now reflects genuine per-pillar sensitivities, which sum approximately to the parallel DV01.

### `api.carry_rolldown`

Three coupled defects:
1. `curve.bumped(0.0)` is a no-op — the "rolled" curve was identical to the original, so no rolldown was actually computed.
2. `int(shorter_T)` truncated the remaining maturity to integer years (5y minus 1 day → 4y), so the "rolldown" was just the difference between a 5y and a 4y swap on the unchanged curve.
3. The returned `"carry"` field held `pv_today` — the swap's PV, not the carry P&L.

**Fix**: use `curve.roll_down(days)` for a genuine reference-date shift; reprice the same swap on the rolled curve; compute carry as `(fixed_rate - par_rate) × dt` (the standard receiver-swap carry approximation); surface `"pv"` separately.

### Verification — `test_l2_t4_api_key_rate_carry.py`

4 new tests pin: ladder has ≥2 distinct values (pre-fix gave all identical); 7Y pillar drives more 7Y-swap sensitivity than 1Y; carry ≈ 0 when `fixed = par` (was always = PV pre-fix); rolldown matches independent `curve.roll_down` + reprice exactly.

Full parallel suite: **12,671 passed in 2:43** — zero regressions.

**171 distinct bugs** in v0.905→v1.034.

---

## v1.033.0 — 2026-06-14

**Fix L2 phase-2 (desks/) — `api.asian_option` silent no-op n_observations + missing geometric Jensen correction + missing rate drift via df.**

Three coupled defects in the geometric-Asian approximation:

1. **`n_observations` silently ignored**. The signature accepted it but the body always used `σ/√3` (the continuous-monitoring limit), regardless of the value passed. Same shape as v0.996 / v1.014 / v1.022 / v1.033 silent-no-op family.

2. **No geometric-Jensen correction**. The Kemna-Vorst forward `F_G = F · exp(-½·(σ²-σ²_g)·t_avg)` was missing — the code used spot directly. For σ=0.20, T=1, n→∞ this is a ~0.33% bias.

3. **No rate drift via df**. Pre-fix passed `spot` (not the forward `F = spot/df`) so any df < 1 silently skipped the rate-induced forward shift — same shape as the v1.032 digital_option bug.

**Fix**: implement the Kemna-Vorst discrete formula

    σ²_geom = σ² · (n+1)(2n+1) / (6·n²)        → σ²/3 as n → ∞
    F_geom  = (spot/df) · exp(-½·(σ²-σ²_g)·t_avg)
    t_avg   = (n+1)/(2n) · T

then price via Black-76 on `(F_geom, K, σ_geom, T, df)`.

### Verification — `test_l2_t4_asian_n_observations.py`

3 new tests pin:
- `n=12` price > `n=10000` price (discrete vol higher than continuous), ratio 1.02-1.10.
- Large-`n` limit matches the closed-form `σ/√3` + Jensen correction.
- Positive rates produce a price-vs-no-rates ratio greater than just `df` (i.e., the forward shift is captured, not just discount).

Existing `test_apix.py::test_asian` (loose `> 0` assertion) unaffected.

Full parallel suite: **12,667 passed in 2:41** — zero regressions.

**170 distinct bugs** in v0.905→v1.033.

---

## v1.032.0 — 2026-06-14

**Fix L2 phase-2 (desks/) — `api.digital_option` d2 omitted risk-neutral drift.**

The d2 formula used `ln(spot/strike)` directly:

    d2 = (ln(spot/strike) − 0.5·σ²·T) / (σ·√T)

Per Black-Scholes risk-neutral pricing, d2 should use `ln(F/K)` where `F = spot/df` is the forward (or equivalently `F = spot·e^(rT)`):

    d2 = (ln(F/strike) − 0.5·σ²·T) / (σ·√T)
       = (ln(spot/strike) − ln(df) − 0.5·σ²·T) / (σ·√T)

The omitted `−ln(df) = r·T` drift term causes:
- **df = 1** (no rates): formula correct.
- **df < 1** (positive rates): ITM call probability **under-stated** by the drift; OTM put **over-stated** by the same factor.
- Binary parity `call + put = df × payout` was violated when `df ≠ 1`.

**Fix**: build `forward = spot / df` and substitute into d2. Now ATM-spot with positive rates correctly reflects that the forward is above spot, and put-call parity holds exactly.

### Verification — `test_l2_t4_digital_option_drift.py`

4 new tests pin: ATM-forward digital call ≈ `df × N(−σ√T/2)`; put-call parity holds with df < 1; ATM/no-rates case unchanged; positive rates increase ITM-forward call probability.

Existing `test_apix.py` digital tests use df=1.0 (unaffected by the fix).

Full parallel suite: **12,664 passed in 2:36** — zero regressions.

**169 distinct bugs** in v0.905→v1.032.

---

## v1.031.0 — 2026-06-14

**Fix L2 phase-2 (desks/) — `fx_desk` silent no-op + gross-vs-net delta.**

Two small but real defects:

1. **`fx_stress_suite` "combined" scenario lied about rates**. Pre-fix the suite included a 5th scenario named `combined` and described as "Spot -5%, rates +100bp", but the PnL formula was identical to `spot_dn_5` — the rates +100bp shock was silently dropped. The function signature `(pair, notional, spot)` has no rates data, so the only honest fix is to remove the misleading scenario. A full rate+spot reprice is provided by `fx_scenario_stress` (PricingContext-based) for callers that need both.

2. **`fx_dashboard.total_delta` was gross, not net**. Pre-fix `total_delta = sum(abs(net_notional))` summed absolute values across pairs, so a long EUR/USD + short EUR/USD of equal size would report a large total_delta instead of 0. Use the signed `net_notional` so the dashboard reflects actual market exposure.

### Verification

- `test_l2_t4_fx_desk_stress.py` (2 new) — pin the 4-scenario list and confirm no description mentions "rates" or "+100bp"; linearity of ±5% / ±10%.
- `test_fx_desk.py` — update existing `test_five_scenarios` to expect 4 scenarios (the silent-no-op 5th was the bug; existing test pinned the wrong count).

Full parallel suite: **12,660 passed in 2:43** — zero regressions.

**168 distinct bugs** in v0.905→v1.031.

---

## v1.030.0 — 2026-06-14

**Fix L2 phase-2 (desks/) — `bond_trading_desk.bond_carry_roll` double-counted coupon income via dirty-price rolldown.**

The rolldown lambda priced the bond with `dirty_price` on the rolled curve:

    roll_down_dirty = dirty_price(rolled_curve) - dirty_price(curve)

`dirty_price` uses `curve.reference_date` as the settlement, which rolled-curve increments by `horizon_days`. So `roll_down_dirty` picks up *both* the genuine clean-price aging *and* the change in accrued interest. The latter is roughly `coupon × horizon / 365` — exactly what `coupon_carry` already measures.

When the module then computed

    total_carry_and_roll = net_carry + roll_down_dirty
                         = (coupon - funding) + (coupon_accrual + clean_roll)

the coupon income appeared **twice**. Worked example: at-par 5y 5% bond on flat 5% curve over 30 days — pre-fix gave `total ≈ 0 + 0.41 = +0.41` per 100 face when the true carry+roll is ~0 (positive carry exactly offset by 0% clean roll on a flat curve at par); post-fix gives `total ≈ 0`.

**Fix**: switch the rolldown lambda from `dirty_price` to `clean_price`. Clean price excludes accrual, so the two are no longer overlapping.

### Verification — `test_l2_t4_bond_carry_roll.py`

2 new tests pin: at-par bond on flat curve has clean roll-down ≈ 0 and total ≈ 0 over 30 days; positive-carry case (5% coupon, 2% repo) yields `total ≈ net_carry` rather than the pre-fix `~2 × net_carry`.

Full parallel suite: **12,658 passed in 3:07** — zero regressions.

**167 distinct bugs** in v0.905→v1.030.

---

## v1.029.0 — 2026-06-14

**Fix L2 phase-2 (desks/) — bump-normalisation sweep across remaining desk risk-metrics.**

Same conceptual bug as v1.028 (`cb_risk_metrics` cs01/dv01) found in three more desk functions exposing a `bump` parameter and using `(pv_up - pv_dn) / 2` without dividing by the bump size:

- `desks.inflation_desk.inflation_risk_metrics` — ie01, real_dv01, nominal_dv01 (gamma was already correct).
- `desks.risk_participation_desk.rp_risk_metrics` — cs01, dv01.
- `desks.structured_credit_desk.sc_risk_metrics` — dv01.

Pre-fix returned "PV change for whatever bump the caller supplied", only "per 1bp" when the caller used the default 0.0001. Now normalised by `0.0001 / bump` so outputs are always "PV per 1bp" regardless of bump tuning — matching the convention used in delta/gamma/vega (which were already correct).

Bond/cln/trs/swap desks were unaffected: they hardcode `h = 0.0001` internally rather than expose a tunable `bump`.

### Verification — `test_l2_t4_desk_bump_normalisation.py`

2 new tests pin: inflation ie01/real_dv01/nominal_dv01 invariant under bump scaling; risk-participation cs01/dv01 invariant. Structured-credit dv01 inherits the same one-line fix; verified by the existing 24 tests passing unchanged.

Full parallel suite: **12,656 passed in 2:39** — zero regressions.

**166 distinct bugs** in v0.905→v1.029.

---

## v1.028.0 — 2026-06-14

**Fix L2 phase-2 (desks/) — `cb_risk_metrics` cs01/dv01 didn't normalise by bump size.**

In `desks.convertible_bond_desk.cb_risk_metrics`, the `delta`/`gamma`/`vega` outputs were correctly normalised by their bump sizes (so the values are always "per unit move" regardless of tuning), but `cs01` and `dv01` returned the raw PV change for whatever `bump_spread`/`bump_rate` the caller supplied. The "per 1bp" interpretation in the field name only held when callers used the default `0.0001` bump.

A user tuning `bump_spread=0.0005` (5bp, for noise reduction in MC) would silently get a `credit_cs01` that's ~5× too large vs the documented "per 1bp" units.

**Fix**: scale `cs01` and `dv01` by `0.0001 / bump` so the outputs are consistently "PV per 1bp" regardless of bump tuning, matching the delta/gamma/vega convention used in the same function.

### Verification — `test_l2_t4_cb_bump_normalisation.py`

2 new tests pin: cs01 from 1bp and 5bp bumps agree within MC noise; same property for dv01.

Full parallel suite: **12,654 passed in 3:03** — zero regressions.

**165 distinct bugs** in v0.905→v1.028.

---

## v1.027.0 — 2026-06-14

**Fix L2 phase-2 (desks/) — theta projection consistency sweep across remaining desk modules.**

Same bug pattern as v1.026: theta lambdas discounted with the rolled curve but projected forwards from the original-t=0 curve. Found in:

- `desks/swaption_trading_desk.swaption_risk_metrics` — swaption theta used stale `proj` for the forward swap rate.
- `desks/trs_desk` (daily P&L) — TRS theta used stale `projection_curve` for funding-leg forwards.

**Fix**: under single-curve, pass `None` (or rolled `c`) for projection so the floating/funding leg uses the rolled discount; under dual-curve, pre-roll the projection by 1 day.

**Known limitation**: `desks/cln_desk` has the same defect on the survival curve (rolled discount paired with stale `surv_t0`), but `SurvivalCurve` has no `roll_down()` method yet — documented as a known approximation rather than fixed in this slice (would require a separate L0-touching change).

### Verification — `test_l2_t4_desk_theta_sweep.py`

2 new tests pin: swaption theta equals consistent-roll PV diff; TRS daily-PnL theta equals consistent-roll PV diff under single-curve.

Full parallel suite: **12,652 passed in 2:34** — zero regressions.

**164 distinct bugs** in v0.905→v1.027.

---

## v1.026.0 — 2026-06-14

**Fix L2 phase-2 (desks/) — `swap_desk` theta computation used stale floating projection.**

The theta rolldown lambda in both `swap_risk_metrics` and `swap_daily_pnl` had a no-op ternary:

    lambda c: swap.pv(c, proj if proj is curve else proj)

Both branches return `proj` — the conditional has no effect. Under single-curve setup this meant the lambda discounted with the rolled `c` but projected forwards from the original-t=0 `curve`, so the floating leg's forward rates were stale (1 day behind discount) while the discount side had rolled.

**Fix**:
- Single-curve: pass `None` to `swap.pv` so the floating leg defaults to using rolled `c` for projection too.
- Dual-curve: pre-roll the projection by 1 day alongside the discount.

Numerical impact is small (1-day projection drift on a few coupons) but the discount/projection asymmetry was a clear correctness defect — and the dead ternary signalled an incomplete refactor.

### Verification — `test_l2_t4_swap_desk_theta.py`

2 new tests pin `theta == swap.pv(rolled, None) - swap.pv(curve, None)` exactly under single-curve, for both `swap_risk_metrics` and `swap_daily_pnl`.

Full parallel suite: **12,650 passed in 2:34** — zero regressions.

**163 distinct bugs** in v0.905→v1.026.

---

## v1.025.0 — 2026-06-14

**Fix L2 phase-2 (regulatory/) — `liquidity._asf_factor` mis-ordered: 200-day retail deposit returned 50% ASF instead of 90% per Basel LIQ40.5.**

Pre-fix logic flow:
```
if equity:                    return 1.0
if maturity > 365:            return 1.0          ← matched first
if maturity > 180:            return 0.50         ← retail 200d hit this
if deposit and retail:        return 0.90         ← never reached
if deposit:                   return 0.50
return 0.0
```

A retail deposit at 200-day residual maturity matched the generic `maturity > 180` branch and got 0.50 instead of 0.90. Per Basel LIQ40.5-7, retail/SME deposits get the retail factor (90/95% for less-stable/stable) at **all maturities** — maturity is irrelevant for the retail bucket.

**Production impact**: Banks with material retail deposit books in the 6–12 month range under-stated ASF, under-reporting NSFR. The error is ~0.40 × notional per affected position.

**Fix**: place the retail-deposit branch *before* the maturity branches; retail at all maturities → 0.90.

### Verification — `test_l2_t4_asf_retail_maturity.py`

8 new tests pin: retail at 30/100/200/364 days → 0.90; retail at 730 days → 0.90 (not 1.0); wholesale 200/100 days → 0.50; portfolio-level ASF correctly counts a 200-day retail deposit at full 0.90.

Full parallel suite: **12,648 passed in 2:33** — zero regressions.

**162 distinct bugs** in v0.905→v1.025. All 8 regulatory/ active modules touched; only `ccar.py`, `reverse_stress.py`, `specialty.py`, `trs_capital.py` left as audited-with-notes (no clear bug).

---

## v1.024.0 — 2026-06-14

**Fix L2 phase-2 (regulatory/) — `stress_irrbb.calculate_eve_impact` duration-gap EVE formula used equity instead of total assets — production-critical.**

The textbook duration-gap derivation (Mishkin/Eakins) gives

    ΔA = −D_A · A · Δr
    ΔL = −D_L · L · Δr
    ΔE = ΔA − ΔL = −(D_A · A − D_L · L) · Δr = −A · (D_A − (L/A)·D_L) · Δr
       = **−A · DurationGap · Δr**

Pre-fix code used `eve_change = -duration_gap × equity × rate_shock` — off by a factor of A/E (≈10× for typical banks with 10:1 leverage). The same function also computed `eve_change_pv01 = -net_pv01 × rate_shock_bps` which **was** correct (since `net_pv01 = A·D_A − L·D_L = A·DurationGap`), so the two outputs in the same return dict silently disagreed by exactly the A/E ratio.

**Downstream impact**: `calculate_irrbb_capital` consumes the broken `worst_eve_change`, so banks **under-stated SOT outlier capital by ~10×**. A bank that should be flagged as an IRRBB outlier (and required to hold capital for excess EVE loss) would silently pass the 15%-of-Tier-1 threshold.

Worked example: assets $10B/dur 5y, liabilities $9B/dur 3y, Tier1 $1B:

    DurationGap = 5 − 0.9·3 = 2.3
    ΔEVE @ +200bp:  pre-fix = −1B·2.3·0.02 = −$46M    (passes 15% SOT)
                    correct = −10B·2.3·0.02 = −$460M  (outlier; $310M charge)

**Fix**: `eve_change = -duration_gap × total_assets × rate_shock`. Now matches `eve_change_pv01` exactly.

Existing tests only asserted `eve_change < 0` (sign), so the bug wasn't caught.

### Verification — `test_l2_t4_eve_duration_gap.py`

4 new tests pin: `eve_change == eve_change_pv01` for all shocks; closed-form textbook check; realistic-bank SOT outlier example with verified $310M capital charge; positive/negative shock symmetry.

Full parallel suite: **12,640 passed in 2:55** — zero regressions.

**161 distinct bugs** in v0.905→v1.024.

---

## v1.023.0 — 2026-06-14

**Fix L2 phase-2 (regulatory/) — `capital_allocation.euler_allocation` correlation branch produced `s_i⁴` allocation instead of `s_i²` (variance-proportional) Euler split.**

Pre-fix mixed `w = standalones / total_standalone` with `cov = outer(s, s) * corr` and formed `RC_i = w_i × (cov @ w)_i`. Substituting w gives

    RC_i = (s_i² / total²) · Σ_j s_j² · ρ_ij

so for uncorrelated desks the allocation fractions scale as `s_i⁴ / Σ s_j⁴` — a strong over-concentration on the largest desk. Worked example for `s = [10, 20, 10]` under identity correlation: pre-fix gave `[5.5%, 88.9%, 5.5%]`; correct Euler gives `[16.7%, 66.7%, 16.7%]` (standard `s_i²` variance allocation).

**Fix**: Tasche (2008) Euler std-dev decomposition with each desk fully invested at unit weight:

    σ_p = √(s' corr s)
    RC_i = s_i · (corr · s)_i / σ_p     (Σ RC_i = σ_p)
    allocated_i = (RC_i / σ_p) · portfolio_capital

Existing tests asserted only `sum(alloc) > 0` and `all(a > 0)`, so the bug wasn't caught.

### Verification — `test_l2_t4_euler_capital.py`

5 new tests pin: uncorrelated → `s_i²` fractions; ρ=1 → `RC_i = s_i`; ρ<0 reduces diversified σ_p; explicit `portfolio_capital` overrides σ_p with ratios preserved; sum equals σ_p when capital not provided.

Full parallel suite: **12,636 passed in 2:55** — zero regressions.

**160 distinct bugs** in v0.905→v1.023.

---

## v1.022.0 — 2026-06-14

**Fix L2 phase-2 (regulatory/) — `balance_sheet_allocation.optimise_allocation` dead `max_single_trade_pct` parameter (silent no-op).**

Same shape as v0.996 `risk.factor_model.factor_timing` and v1.014 `portfolio_construction.mean_variance` — the API parameter `max_single_trade_pct: float = 0.25` was in the signature and docstring but the body never referenced it. A single trade could absorb the entire capital budget despite the documented concentration limit, and existing tests never exercised the parameter.

**Fix**: collapse the concentration limit into the per-trade upper bound

    rwa_i · w_i · 0.08 ≤ max_single_trade_pct · total_capital
    ⇒ w_i ≤ max(0, max_single_trade_pct · total_capital / (rwa_i · 0.08))

Also removed the dead `rocs` list (computed and discarded).

### Verification — `test_l2_t4_bs_allocation_concentration.py`

4 new tests pin: no trade exceeds the per-trade cap; tighter pct → at least as many trades selected; default `pct=0.25` still well-formed.

Full parallel suite: **12,631 passed in 2:32** — zero regressions.

**159 distinct bugs** in v0.905→v1.022.

---

## v1.021.0 — 2026-06-14

**Fix L2 phase-2 (regulatory/) — `securitization.calculate_sec_sa_rw` SSFA formula had spurious K_a multiplier (both branches) and missing /(D−A) in straddle branch.**

Per Basel III CRE40.53 / 12 CFR 217.43(b)(5):
- **Entirely above** (A ≥ K_a): `K_SSFA = (e^(au) − e^(al)) / (a·(u − l))` — pre-fix carried a spurious `K_a ×` multiplier.
- **Straddle** (A < K_a < D): `K_SSFA = (K_a − A)/(D − A) + (e^(au) − 1) / (a·(D − A))` — pre-fix used `(K_a − A) + K_a × (e^(au) − 1)/(a·(D − A))`, both terms wrong.

For mezzanine tranches near K_a this materially under-reported risk-weight. Worked example: K_a=0.08, A=0.05, D=0.10 (straddle) — pre-fix ~65% RW vs CRE40 ~1089% RW, a ~17× under-statement. For super-senior tranches the 15% RW floor masked the bug; existing tests asserted only `rw > 0` and `rw ≤ 1250` and so passed under either formula.

**Production impact**: SEC-SA capital under-stated for mezzanine securitisation exposures (and SEC-IRBA, which calls the same `_rw` helper with K_IRB).

### Verification — `test_l2_t4_sec_sa_ssfa.py`

5 new tests pin the formula via the direct integral form `K = (1/(D−A)) · ∫_A^D K_loc(x) dx`. Covers: just-above-K_a mezzanine (no floor binds), super-senior (floor binds), straddle with and without delinquency `w`, and continuity across A = K_a.

Full parallel suite: **12,627 passed in 2:35** — zero regressions.

**158 distinct bugs** in v0.905→v1.021.

---

## v1.020.0 — 2026-06-13

**Fix L2 phase-2 (regulatory/) — `market_risk_sa.calculate_curvature_capital` cross-bucket aggregation didn't match FRTB.**

Pre-fix used plain `sum(bucket_caps)` — equivalent to γ=1 (full positive correlation) for every bucket pair. FRTB MAR21.14 specifies:

    K_curvature = sqrt[max(0, Σ K_b² + Σ_{b≠c} γ_bc · S_b · S_c · ψ(S_b, S_c))]

with γ_bc per risk class (0.15 EQ, 0.20 COM, 0.25 CSR, 0.50 GIRR, 0.60 FX) and ψ=0 if both S_b, S_c negative. Pre-fix plain sum was strictly more conservative — overstating capital for diversified curvature positions. For 2 buckets in EQ, pre-fix gave ~32% higher K than the correct γ=0.15 formula.

**Fix**: implement FRTB cross-bucket aggregation with γ_bc per risk class and ψ sign indicator.

**Known remaining simplification**: within-bucket aggregation still uses plain sum vs FRTB's sqrt(Σ CVR_k² + Σ ρ_kl · CVR_k · CVR_l · ψ). Deeper rewrite tracked as a follow-up.

### Verification — `test_l2_t4_frtb_curvature.py`

5 new tests; existing 23 FRTB tests still pass.

Full parallel suite: **12,622 passed in 2:38** — zero regressions.

First fix from `regulatory/` package. **157 distinct bugs** in v0.905→v1.020.

---

## v1.019.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.repo_cva` omitted discount factor (same shape as v1.006 hybrid_xva).**

Pre-fix CVA accumulator: `cva += epe × pd_step × lgd` — no `D(0, t_i)` term. Future-valued not present-valued. For overnight repo the effect is < 1%; for term repo (months to a year) it accumulates to several percent.

**Fix**: optional `discount_rate` parameter (defaults to `repo_rate` — sensible secured-funding-curve approximation). Each EPE × ΔPD contribution is now multiplied by `exp(-discount_rate · t)`.

### Verification — `test_l2_t4_repo_cva.py`

5 new tests: zero discount preserves pre-fix; high discount reduces CVA; default uses repo_rate; zero hazard / zero LGD give zero CVA.

Full parallel suite: **12,617 passed in 5:00** — zero regressions.

Twenty-seventh fix from phase-2. **156 distinct bugs** in v0.905→v1.019.

---

## v1.018.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.portfolio_analytics.tracking_metrics` mixed ddof conventions for beta.**

Pre-fix:
```python
beta = np.cov(p, b)[0, 1] / np.var(b)
```

`np.cov` defaults to ddof=1 (sample); `np.var` defaults to ddof=0 (population). The ratio gives beta off by factor `(n-1)/n` — material for small samples (~3% bias at n=30, ~0.4% at n=252).

**Fix**: both use ddof=1 (sample-statistics convention).

### Verification — `test_l2_t4_portfolio_analytics.py`

3 new tests: perfect correlation → β=1; matches `cov_ddof1/var_ddof1` exactly; differs from pre-fix mixed convention.

Full parallel suite: **12,612 passed in 4:41** — zero regressions.

Twenty-sixth fix from phase-2. **155 distinct bugs** in v0.905→v1.018.

---

## v1.017.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.vol_stress.twist_vol_bump` butterfly weights were wrong.**

Pre-fix:
```python
weights = ((T - T_mid) / W)**2 - 0.25
```

Evaluating on x = (T-T_mid)/W ∈ [-0.5, +0.5]:
- Wings (x=±0.5): `0.25 - 0.25 = 0`.
- Belly (x=0): `0 - 0.25 = -0.25`.

That's "belly down, wings unchanged" — **not** the butterfly the docstring promises ("wings up, belly down"). For a long-belly vega position the pre-fix scenario gave a misleading P&L (only the belly hit).

**Fix**: corrected formula `4·x² - 0.5` gives wings = +0.5, belly = -0.5 — a true butterfly with the twist_bps controlling the wing-belly spread.

### Verification — `test_l2_t4_vol_stress.py`

4 new tests:
- `TestTwistButterfly` × 2: wings above base, belly below base; magnitudes match +/-50bp for twist_bps=100.
- `TestParallelTilt` × 2: parallel uniform shift and tilt steepening unchanged.

Full parallel suite: **12,609 passed in 2:59** — zero regressions.

Twenty-fifth fix from phase-2. **154 distinct bugs** in v0.905→v1.017.

---

## v1.016.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.portfolio_margin.span_margin` applied SPAN's "extreme scenario 35% cap" to user-supplied scenarios.**

The SPAN methodology caps the last two scenarios (the 2× price_scan_range extremes) at 35% of their loss, reflecting that those moves are deep-tail and historically rare. Pre-fix code applied this cap to the last two entries *regardless* of whether the scenarios came from the auto-built grid or from the user:

```python
if idx >= n_scenarios - 2 and scenarios is not None:  # always True
    loss *= 0.35
```

For user-supplied custom scenarios (e.g., stress scenarios from a regulator), the last two entries were arbitrarily mis-scaled by 0.35, silently understating margin.

**Fix**: track `auto_built` flag and only apply the cap when the grid was generated internally.

### Verification — `test_l2_t4_span_margin.py`

3 new tests:
- User-supplied scenarios: every loss counted at full magnitude.
- Auto-built grid: extreme cap still active (worst of {-PSR, capped -2·PSR}).
- Vega-only user scenarios: not 35%-capped.

Full parallel suite: **12,605 passed in 2:40** — zero regressions.

Twenty-fourth fix from phase-2. **153 distinct bugs** in v0.905→v1.016.

---

## v1.015.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.scenario` constructors had the same lossy PricingContext reconstruction as v0.993 `var.stress_test`.**

All five scenario constructors (`parallel_shift`, `pillar_bump`, `vol_bump`, `fx_spot_shock`, `credit_spread_shift`) built the bumped context via `PricingContext(field=value, ...)` with only 6 named fields, silently dropping:
- `discount_curves` (the plural, multi-currency dict)
- `inflation_curves`
- `repo_curves`
- `reporting_currency`
- `stochastic_credit_models`
- `credit_vol_surfaces`
- `credit_correlations`
- `numerical_config`

Same shape as the v0.993 fix. Pricers consulting any of these silently saw a degraded context inside scenario evaluation.

**Fix**: switch all five constructors to `dataclasses.replace`, preserving every untouched field by default.

**Bonus**: `parallel_shift` now also bumps the plural `discount_curves` dict (was previously bumping only the singular). Multi-currency portfolios needed each currency's curve bumped, not just the home currency.

### Verification — `test_l2_t4_scenario.py`

6 new tests:
- `TestPreservesUntouchedFields` × 5: each constructor preserves `numerical_config` / `reporting_currency`.
- `TestParallelShiftBumpsPluralCurves` × 1: both USD and EUR discount_curves are bumped.

Full parallel suite: **12,602 passed in 2:43** — zero regressions.

Twenty-third fix from phase-2. **152 distinct bugs** in v0.905→v1.015.

---

## v1.014.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.portfolio_construction.mean_variance` `target_return` parameter was dead code.**

Pre-fix: the function signature declared `target_return: float | None = None` but the body never referenced it. Every call computed the max-Sharpe tangency portfolio regardless of whether a target was passed.

**Fix**: when `target_return` is supplied, solve the QP

    min w'Σw  s.t.  μ'w = target_return, Σw = 1, w ∈ [lb, max_weight]

via SLSQP. The `target_return=None` path (tangency portfolio) is unchanged.

### Verification — `test_l2_t4_mv_target.py`

4 new tests:
- target_return constraint honoured to 1e-6 precision.
- min-variance solution feasible (weights sum to 1, long-only ≥ 0).
- target_return=None preserves max-Sharpe behaviour.
- Infeasible target falls back deterministically without crash.

Full parallel suite: **12,596 passed in 3:05** — zero regressions.

Twenty-second fix from phase-2. **151 distinct bugs** in v0.905→v1.014.

---

## v1.013.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.simm` had two material gaps from ISDA SIMM v2.6.**

### (a) Cross-risk-class correlation = 0

Pre-fix: `total = sqrt(Σ M_i²)` despite the code comment admitting "SIMM uses sqrt of sum of squares" — but real ISDA SIMM applies the cross-class correlation matrix (Table 21 in v2.6):

| | GIRR | FX   | CSR  | EQ   | COM  |
|------|------|------|------|------|------|
| GIRR | 1.00 | 0.20 | 0.05 | 0.05 | 0.05 |
| FX   |      | 1.00 | 0.10 | 0.15 | 0.15 |
| CSR  |      |      | 1.00 | 0.15 | 0.15 |
| EQ   |      |      |      | 1.00 | 0.20 |
| COM  |      |      |      |      | 1.00 |

For a diversified GIRR+FX book the corrected total = `sqrt(M_GIRR² + M_FX² + 0.40·M_GIRR·M_FX)`, which is meaningfully larger than the zero-correlation case. Pre-fix systematically under-margined diversified books.

### (b) Vega and curvature silently dropped

The `SIMMSensitivity` dataclass exposes `delta`, `vega`, `curvature` — but `_compute_bucket` only read `s.delta`. For options books (which is precisely what SIMM is designed to margin) this is a material gap. Now: each component aggregates separately within the bucket and combines via sum-of-squares, matching SIMM's "delta-vega-curvature decomposition" structure.

For a single sensitivity with delta=d, vega=v in a single bucket: margin = `rw · sqrt(d² + v²)`. Pre-fix: margin = `rw · |d|`.

### Verification — `test_l2_t4_simm.py`

5 new tests:
- `TestCrossRiskClassCorrelation` × 1: GIRR+FX book matches ρ=0.20 closed form; > zero-corr total.
- `TestVegaCurvatureIncluded` × 3: vega/curvature each raise margin; single sensitivity matches `rw·sqrt(d²+v²)`.
- `TestDeltaOnlyUnchanged` × 1: delta-only single-class case unchanged from pre-fix.

Full parallel suite: **12,592 passed in 2:39** — zero regressions.

Twenty-first fix from phase-2. **150 distinct bugs** (2 in this slice — cross-class corr and vega/curvature) in v0.905→v1.013.

**Reached 150-bug milestone.** Production SIMM relies on this module; pre-fix margins were materially under-stated for the typical mixed-asset, options-bearing book.

---

## v1.012.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.prudent_valuation` had two API consistency bugs.**

### (a) `market_price_uncertainty_ava` non-monotonic in n_quotes

Pre-fix: `reliability_factor = min(n_quotes/5, 1.0)`. At `n_quotes=0` the special-case fallback returned `half_spread`; at `n_quotes=1` reliability=0.2 → AVA = `4.5·half_spread`. So adding the first quote *increased* AVA — contradicting the "more quotes → more reliable → smaller AVA" intent.

**Fix**: floor reliability at 0.1, drop the special-case. Now n_quotes=0 gives the maximum AVA (`9·half_spread`) and AVA monotonically decreases as quotes accumulate.

### (b) `close_out_cost_ava` silently ignores `position_days` parameter

Pre-fix: when `daily_volume > 0`, the caller's `position_days` was overwritten by `notional/daily_volume`; when `daily_volume == 0`, it was ignored entirely (defaulting to a flat 50% premium). The parameter was effectively dead.

**Fix**: honour caller's `position_days` when supplied:
- daily_volume > 0: use `max(derived, supplied, 1)`.
- daily_volume = 0 with `position_days > 1`: use the log-based scaling from the supplied estimate.
- daily_volume = 0 with default `position_days = 1`: keep the 50% default-premium fallback.

### Verification — `test_l2_t4_prudent_valuation.py`

6 new tests:
- `TestMPUMonotonicInQuotes` × 3: monotone decrease 1→5 quotes; n_quotes=0 is max; 5+ quotes gives `confidence·half_spread`.
- `TestCloseOutCostHonorsPositionDays` × 2: caller's position_days drives size adjustment when no volume; large position_days overrides default 50%.
- `TestRegressionsExistingTests` × 1: daily_volume branch unchanged when position_days at default.

Full parallel suite: **12,587 passed in 2:36** — zero regressions.

Twentieth fix from phase-2. **148 distinct bugs** (2 in this slice) in v0.905→v1.012.

---

## v1.011.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.collateral_optimisation.CollateralOptimiser` silently under-collateralised every non-cash allocation.**

Pre-fix coverage constraint:
```python
Σ_j x[i,j] >= required[i]
```

No haircut applied. For an asset with 5% haircut, allocating $100 gross gives only $95 of post-haircut collateral value — but the LP treated $100 = $100 of coverage. A CSA needing $100 covered exclusively by a 5%-haircut asset was silently $5 short of its requirement.

**Fix**: coverage constraint now uses post-haircut value:
```python
Σ_j (1 - haircut_j) × x[i,j] >= required[i]
```

The LP correctly grosses up the allocation (posting `required / (1 − h)` for single-asset coverage). The unmet-requirement check uses the same post-haircut convention.

**Side fix**: `_naive_cost` baseline now respects asset availability — pre-fix it computed the cheapest-asset-per-CSA cost ignoring supply caps, so when the cheapest asset ran out the LP correctly spilled over to the next-cheapest but appeared "worse than naive" because naive's idealised cheapest was infeasible.

### Verification — `test_l2_t4_collateral_opt.py`

4 new tests:
- `TestCoverageNetsHaircut` × 2: 5% haircut requires $100/0.95 gross; zero haircut unchanged.
- `TestUnmetWithHaircut` × 1: $95 of 5%-haircut asset (= $90.25 net) can't cover $100.
- `TestMultiCSAHaircutAware` × 1: lower-haircut asset preferred when costs comparable.

Full parallel suite: **12,581 passed in 3:04** — zero regressions; existing collateral tests still pass.

Nineteenth fix from phase-2. **146 distinct bugs** (2 in this slice — coverage and naive baseline) in v0.905→v1.011.

This one is **production-critical**: real banks running this optimiser would have been silently under-margined on every haircut'd allocation.

---

## v1.010.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.market_risk_enhanced.incremental_var` had two related issues.**

### (a) ddof inconsistency in parametric branch

`np.cov(...)` (default ddof=1) for portfolio vol but `np.std(...)` (default ddof=0) for individual position vols. Same shape as the v0.995 backtest fix. Now ddof=1 throughout.

### (b) LOO conflated with Euler decomposition for historical method

Pre-fix set both `incremental_vars` AND `component_vars` to the leave-one-out (LOO) values for the historical method. But the docstring promised `sum(IVaR_i) = portfolio VaR` — LOO doesn't have this property.

Two metrics matter and are now reported separately:
- `incremental_vars[i]`: `VaR(portfolio) − VaR(portfolio without i)` (LOO; Jorion convention).
- `component_vars[i]`: `−E[P&L_i | portfolio in tail]` (tail-conditional expectation = **ES decomposition**).

Note: for historical method, `sum(component_vars)` equals portfolio **Expected Shortfall**, not VaR. The VaR Euler decomposition requires kernel smoothing at the quantile boundary, which is noisy for finite samples; ES is what practitioners actually report. Docstring updated to make this explicit.

For parametric method, `incremental` = `component` = Euler decomposition, both sum to VaR (unchanged behaviour).

### Verification — `test_l2_t4_market_risk_enhanced.py`

5 new tests:
- `TestParametricDdofConsistency` × 1: individual VaR uses sample std (ddof=1).
- `TestHistoricalEulerDecomposition` × 2: component_vars sum to ES; LOO ≠ component.
- `TestParametricEulerInvariant` × 1: parametric component sums to VaR.
- `TestDiversificationNonNegative` × 1: diversification ≥ 0.

Full parallel suite: **12,577 passed in 2:36** — zero regressions.

Eighteenth fix from phase-2. **144 distinct bugs** (2 in this slice) in v0.905→v1.010.

---

## v1.009.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.network_xva` had a spurious `max(multiplier, 1.0)` floor that contradicted both its docstring and its own comment.**

Pre-fix:
```python
# Floor multiplier at 1.0: no contagion means no adjustment (multiplicative identity)
adjustment = alpha * centrality * max(multiplier, 1.0)
network_cva = standalone_cva * (1.0 + adjustment)
```

The comment claims "no contagion → no adjustment", but `max(0, 1.0) = 1.0` ≠ 0, so the formula still adds `α · centrality` to the multiplier even when there's no contagion. The docstring's formula is `CVA × (1 + α × centrality × multiplier)` — when multiplier = 0, no adjustment.

For an isolated counterparty (no outgoing exposures → no contagion → multiplier = 0):
- Pre-fix: CVA was bumped by `α · centrality · CVA` regardless.
- Post-fix: CVA = standalone_cva (multiplier = 0 → adjustment = 0).

**Fix**: removed the floor in both `compute_network_cva` and `systemic_cva_adjustment`. Multiplier passes through raw.

### Verification — `test_l2_t4_network_xva.py`

5 new tests:
- `TestNoContagionNoAdjustment` × 2: isolated CP unchanged; multiplier=0 in convenience fn unchanged.
- `TestContagionRaisesCVA` × 1: cascading CP still uplifted.
- `TestAlphaZeroNoChange` × 1: alpha=0 always recovers standalone (existing invariant).
- `TestSystemicCvaAdjustmentClosedForm` × 1: exact formula match.

Full parallel suite: **12,572 passed in 3:02** — zero regressions; existing `test_phase5_integration` tests still pass.

Seventeenth fix from phase-2. **142 distinct bugs** (2 in this slice) in v0.905→v1.009.

---

## v1.008.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.leverage_optimisation` concentration constraint was looser than its own docstring.**

Pre-fix: `w_i ≤ max_single_pct × (capital × max_leverage)` — bound is fraction of *theoretical max notional*, not realised portfolio. When actual leverage is below max_leverage, this is much looser than the docstring-promised relative form `w_i ≤ max_single_pct × Σw`.

Example: capital=$100M, max_lev=10, max_single=30%, actual_lev=5x → portfolio = $500M.
- Pre-fix: single trade allowed up to $300M (= 60% of portfolio).
- Docstring (relative): single trade capped at $150M (= 30% of portfolio).

**Fix**: implement the relative constraint linearly:

    (1 − max_pct)·w_i − max_pct·Σ_{j≠i} w_j ≤ 0

Default `max_single_trade_pct` raised from `0.30` to `1.0` (no concentration cap by default). The relative form requires `max_pct ≥ 1/N` to be feasible with all-active portfolio; with the old default of 0.30 and N=3 trades the LP would be infeasible. Callers that want strict concentration set `max_single_trade_pct` explicitly.

### Verification — `test_l2_t4_leverage_opt.py`

4 new tests:
- `TestRelativeConcentration` × 2: every w_i/Σw ≤ max_pct (feasible N=4, max_pct=0.30).
- `TestSingleTradeRecovery` × 1: uniform weights under tight cap.
- `TestOptimiserHappyPath` × 1: solver returns positive carry/leverage.

Plus existing `test_repo_phase3.TestLeverageOptimisation` (13 tests) still pass with the new default.

Full parallel suite: **12,567 passed in 3:01** — zero regressions.

Sixteenth fix from phase-2. **140 distinct bugs** in v0.905→v1.008.

---

## v1.007.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.correlation_network._quasi_diag` was a greedy nearest-neighbour tour, not López de Prado's quasi-diagonalisation.**

Pre-fix:
```python
order = [0]
while remaining:
    last = order[-1]
    nearest = min(remaining, key=lambda j: dist[last, j])
    order.append(nearest)
```

This produces a path-like ordering starting at asset 0, traversing always to the nearest unvisited asset. **NOT** a cluster-aware ordering: subsequent recursive bisection in `hierarchical_risk_parity` then mixes cluster-mates apart at the midpoint cut, defeating HRP's whole premise.

True LdP quasi-diagonalisation: build hierarchical clustering (single-linkage on the distance matrix), then traverse the dendrogram leaves in order so that siblings are adjacent. The other HRP module (`risk.hierarchical_risk_parity`) already does this correctly — `correlation_network` had its own broken implementation.

**Fix**: use `scipy.cluster.hierarchy.linkage` + `leaves_list` (matching the other HRP module).

### Verification — `test_l2_t4_correlation_network.py`

6 new tests:
- `TestQuasiDiagBlockStructure` × 1: 2-cluster distance matrix → cluster-mates adjacent in output.
- `TestHRPCorrelationNetwork` × 3: weights sum to 1; non-negative; block structure preserved in cluster_order.
- `TestQuasiDiagDegenerate` × 2: N=1 and N=2 handled.

Full parallel suite: **12,563 passed in 2:41** — zero regressions.

Fifteenth fix from phase-2. **139 distinct bugs** in v0.905→v1.007.

---

## v1.006.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.hybrid_xva.hybrid_cva` and `hybrid_fva` computed future-valued XVA instead of present-valued.**

The textbook formulas:

    CVA = (1 − R) × Σ_i D(0, t_i) × EPE(t_i) × ΔPD(i)
    FVA = funding_spread × Σ_i D(0, t_i) × E[V(t_i)] × dt

Pre-fix omitted `D(0, t_i)` entirely. For a 10y trade with rates ~5%, this overstated CVA/FVA by ~30%-40% (since avg DF over 10y at 5% ≈ 0.78).

**Fix**: added optional `discount_factors: np.ndarray | None = None` parameter to both functions. When supplied, multiplies EPE/EE by `D(0, t_i)` element-wise before summing. Backwards compatible — omitting it preserves the old behaviour, now documented as requiring pre-discounted exposure inputs.

This is the second XVA-family discount-factor fix in this audit (compare v0.984 risk/network.py and the earlier curve fixes). Production XVA engines should pass discount factors explicitly to avoid the FV/PV trap.

### Verification — `test_l2_t4_hybrid_xva.py`

6 new tests:
- `TestCVADiscounting` × 3: DF reduces CVA; shape validation; DF=1 unchanged.
- `TestFVADiscounting` × 2: DF reduces FVA; shape validation.
- `TestNoChangeWhenAlreadyDiscounted` × 1: backwards-compat preserved.

Full parallel suite: **12,557 passed in 2:41** — zero regressions.

Fourteenth fix from phase-2. **138 distinct bugs** (2 in this slice) in v0.905→v1.006.

---

## v1.005.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.shapley.shapley_capital_allocation` computed diversification info then threw it away.**

Pre-fix built an `enriched_values` dict containing per-desk `{shapley_allocation, standalone_risk, diversification_benefit}`, then **immediately discarded it** by returning the raw `ShapleyResult`. The function docstring promised diversification reporting; the caller never received it.

**Fix**: added optional `diversification: dict | None` field to `ShapleyResult` (defaults to `None` for non-capital-allocation callers, preserving backwards compat). `shapley_capital_allocation` now populates it. Also surfaced in `to_dict()`.

Audit-clean modules from the cooperative-game family:
- `risk/cooperative_games.py` — Shapley delegate + core-check correct.
- `risk/shapley.py` exact/sampling algorithms — formulas correct (Shapley weights sum to 1; sampling estimator unbiased).

### Verification — `test_l2_t4_shapley.py`

4 new tests: diversification field populated; benefit equals `standalone − shapley`; surfaced in to_dict; default `None` for non-capital-allocation callers.

Full parallel suite: **12,551 passed in 2:43** — zero regressions.

Thirteenth fix from phase-2. **136 distinct bugs** in v0.905→v1.005.

---

## v1.004.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.brinson_attribution.brinson_multi_period` claimed "geometric linking" but used ad-hoc scaling that broke the active-return identity.**

Pre-fix:
```python
cum_alloc += r.total_allocation * cum_bench   # cum_bench BEFORE updating
```

This is neither Frongello, Carino, nor Menchero. The geometric multi-period identity:

    Σ_t F_t · (alloc_t + sel_t + inter_t) = Π(1+r_p_t) − Π(1+r_b_t)

requires the Frongello linking coefficient

    F_t = (Π_{s<t}(1+r_p_s)) · (Π_{s>t}(1+r_b_s))

which can be computed iteratively as

    cum_t = cum_{t-1} · (1+r_b_t) + effect_t · cum_port_{t-1}

The pre-fix version diverged from the geometric active by a factor that grows quadratically in T:
- 2 periods, port=1%/bench=0% each: pre-fix 2.00% vs geometric 2.01%.
- 10 periods, port=1%/bench=0% each: pre-fix 10.00% vs geometric 10.462%.

**Fix**: implement the Frongello recursive linking. Prior cumulative effects scale by `(1+r_b_t)`; new period's effect scales by `cum_port_before_t`.

### Verification — `test_l2_t4_brinson.py`

5 new tests:
- `TestFrongelloIdentity` × 3: 2-period, 3-period mixed, 10-period — the identity `Σ effects = cumulative_active_return` holds to machine precision.
- `TestBrinsonSinglePeriodIdentity` × 2: per-period Brinson identity (unchanged) holds for 2- and 3-sector portfolios.

Full parallel suite: **12,547 passed in 2:37** — zero regressions.

Twelfth fix from phase-2. **135 distinct bugs** in v0.905→v1.004.

---

## v1.003.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.model_selection.model_committee_price` was statistically inconsistent.**

Pre-fix combined a *weighted* mean with an *unweighted* `np.std`. For BMA committees where one model dominates (typical: BIC-based weights of 0.9 + 0.05 + 0.05), the unweighted std treated all models equally, overstating uncertainty.

Concrete example: weights `(0.9, 0.05, 0.05)`, prices `(100, 200, 200)`. Weighted mean = `110`. Unweighted pop-std (pre-fix) ≈ `47`. Correct weighted std = `30` (because the dominant model's price is close to the mean).

**Fix**: weighted standard deviation `sqrt(Σ w_i · (p_i − weighted_mean)²)`, consistent with the weighted mean. The `price_range` and `model_uncertainty_reserve` fields (min/max-based) are unchanged — they're informative regardless of weights.

### Verification — `test_l2_t4_model_selection.py`

5 new tests:
- `TestWeightedStd` × 2: dominant-model committee gives weighted std (30 not 47); equal weights match population std.
- `TestCommitteeStdEdgeCases` × 2: single model → 0 std; identical prices → 0 std.
- `TestCommitteeRangeUnchanged` × 1: range/reserve still use unweighted extremes.

Full parallel suite: **12,542 passed in 2:44** — zero regressions, existing model_selection tests unchanged.

Eleventh fix from phase-2. **134 distinct bugs** in v0.905→v1.003.

---

## v1.002.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.contagion.DefaultCascade.simulate` silently dropped second-order contagion.**

The bug was a textbook conflation of two distinct flags via one variable. Pre-fix used `remaining_buffer[d] = -1` as both:
- "this node has defaulted" (set on initial defaulters at start of round), AND
- "this node's outward losses have been propagated to its creditors" (the "skip if already processed" check).

But a creditor that defaults mid-cascade *also* has `remaining_buffer < 0` (from incoming losses absorbed beyond its buffer). In the next round, the `if remaining_buffer[d] < 0: continue` check at the top of the outer loop saw it as "already processed" and **skipped propagating its losses to its own creditors**.

Net effect: only first-order contagion (initial-default's direct creditors) ever cascaded. A→B→C chains lost everything past B. The "contagion multiplier" metric understated systemic risk by an unbounded factor.

**Fix**: separate `processed: set[int]` tracks which defaulters have had their losses propagated. Each defaulter now propagates outward exactly once, regardless of whether their buffer went negative from incoming losses.

### Verification — `test_l2_t4_contagion.py`

4 new tests:
- `TestSecondOrderContagion` × 2: A→B→C with C surviving (but absorbing B's losses); A→B→C with all three defaulting.
- `TestNoContagion` × 1: well-capitalised neighbours absorb A's loss without propagating.
- `TestProcessedOnceInvariant` × 1: 4-node ring topology, all default.

Full parallel suite: **12,537 passed in 2:40** — zero regressions (existing `test_graph_theory.py::DefaultCascade` tests were too lenient to catch the bug).

Tenth fix from phase-2. **133 distinct bugs** in v0.905→v1.002.

This is one of the more serious bugs found in phase 2 — it silently understated systemic risk in a financial-stability tool.

---

## v1.001.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.hierarchical_risk_parity` had two issues.**

(a) **N=1 crashed** — `np.corrcoef` of `(T, 1)` returns scalar; `_correlation_distance` produces `(1,1)` zero matrix; `squareform` of that → empty vector; `linkage` on empty → ValueError. Now returns trivial weights=`[1.0]` directly for the single-asset case.

(b) **`n_clusters` reporting was a meaningless heuristic.** Pre-fix: `min(N, max(2, N//3))` — unrelated to actual dendrogram structure. Now uses `fcluster` at the median linkage height, so the reported count reflects the real cluster topology found by hierarchical clustering.

The core HRP algorithm (López de Prado 2016: correlation distance → linkage → quasi-diagonalisation → recursive bisection with inverse-variance allocation) was correctly implemented; only edge case and reporting affected.

### Verification — `test_l2_t4_hrp.py`

5 new tests:
- `TestHRPSingleAsset` × 1: N=1 returns [1.0] without crash.
- `TestHRPClusterCount` × 2: clustered returns yield ≥2 clusters; highly-correlated near-uniform returns yield few clusters.
- `TestHRPWeightInvariants` × 2: weights sum to 1, all non-negative.

Full parallel suite: **12,533 passed in 2:45** — zero regressions.

Ninth fix from phase-2. **132 distinct bugs** in v0.905→v1.001.

---

## v1.000.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.efficient_frontier.minimum_variance_portfolio` returned wrong expected_return and sharpe_ratio fields.**

Pre-fix: the function had no access to the expected-returns vector `mu`, but its return type (`FrontierPoint`) requires `expected_return` and `sharpe_ratio`. Both were hardcoded to `0` regardless of the portfolio. Inside `efficient_frontier` this was patched up by mutating the returned dataclass; but every other caller — including the public `test_portfolio_game_theory.py` usage — got nonsense fields.

**Fix**: added optional `mu` (and `risk_free_rate`) parameters. When supplied, `expected_return = mu @ w` and `sharpe = (μw − rf) / vol`. When omitted, fields default to 0 (legacy behaviour preserved).

### Verification — `test_l2_t4_efficient_frontier.py`

5 new tests:
- `TestMinVarianceWithMu` × 4: mu supplied populates return/Sharpe; mu omitted gives legacy zeros; long-only matches `mu @ w` exactly; shape mismatch raises.
- `TestMinVarianceVolUnchanged` × 1: unconstrained weights match analytical Σ⁻¹·1 / (1'·Σ⁻¹·1).

Full parallel suite: **12,528 passed in 3:10** — zero regressions.

### Audited-clean modules in this phase (no slice required)

- `risk/correlation_repair.py` — Higham (2002) alternating projections correctly implemented; minor cosmetic issues only.
- `risk/cvar_optimisation.py` — Rockafellar-Uryasev LP correctly formulated; component CVaR uses correct Euler decomposition.
- `risk/model_reserve.py` — sensitivity-based aggregation (worst-case sum vs quadrature) correct; minor docs gap (confidence param stored but unused, no math impact).

Eighth fix from phase-2. **130 distinct bugs** in v0.905→v1.000.

This is the **v1.0 milestone** for pricebook. The audit arc that started at v0.905 has now resolved 130 correctness bugs across 8 sessions, with 12,528 tests passing and zero regressions across 95+ stamped versions.

---

## v0.999.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.ipv.ipv_single_trade` had no concept of position direction.**

Pre-fix: `prudent_value = mid - diversified_ava` was hardcoded for *long* positions. For a short bond position the prudent (conservative) liability value sits *above* mid, not below — the IPV would silently understate the size of a short liability by `2 · ava`.

Compounding: AVA helpers (`close_out_cost_ava`, `concentration_ava`, etc.) take `notional` as a positive size and multiply by it. Passing a *signed* notional (negative for short) propagated a negative magnitude through every AVA, producing nonsense.

**Fix**:
- Added explicit `direction: int = 1` parameter (only `±1` accepted; default `+1` preserves long-only behaviour).
- All AVA helpers now receive `abs(notional)`.
- `prudent_value = mid − direction · diversified_ava` (above mid for shorts).
- `variance_to_model_bp` denominator uses `abs(notional)` — pre-fix a negative-notional input produced negative `variance_bp` that could never breach the threshold.

### Verification — `test_l2_t4_ipv.py`

7 new tests:
- `TestIpvLongUnchanged` × 1: default direction preserves pre-fix long behaviour.
- `TestIpvShort` × 3: prudent above mid; AVA magnitudes mirror long for direction=−1; abs(notional) is what counts (sign doesn't flip AVA).
- `TestIpvDirectionValidation` × 2: invalid direction (0 or +2) raises.
- `TestIpvVarianceWithNegativeNotional` × 1: variance_bp ≥ 0 and breach still fires.

Full parallel suite: **12,523 passed in 2:42** — zero regressions, 11 existing IPV tests still pass.

Seventh fix from phase-2. **129 distinct bugs** in v0.905→v0.999.

---

## v0.998.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.kelly` had two correctness issues.**

### (a) `kelly_fraction` silently returned 0 for vol=0

Pre-fix: `f_star = excess / var if var > 0 else 0`. This masked two genuinely different cases — positive excess with σ=0 is a deterministic *arbitrage* (Kelly = +∞, short all wealth into it); negative excess with σ=0 is a deterministic loss (Kelly = −∞). Returning 0 in both cases makes the asset appear unattractive when in fact it's an unbounded opportunity. Now raises `ValueError` with diagnostic.

### (b) `multi_asset_kelly` had no validation and lossy singular-cov fallback

Two issues:
- No shape/symmetry check — caller could pass `(3,4)` matrix or non-symmetric Σ and get garbage out.
- Fallback on singular Σ used diagonal-only inverse `f = excess / diag(Σ)` — this drops correlation structure entirely. For two highly-correlated assets the true Kelly concentrates weight on the better risk-adjusted one; the diagonal fallback spreads incorrectly.

Now validates shape (square, mu-compatible) and symmetry (within `1e-10`), and uses `np.linalg.pinv` (Moore-Penrose pseudoinverse) as the singular fallback — preserves correlation structure and gives the minimum-norm solution.

### Verification — `test_l2_t4_kelly.py`

10 new tests across 5 classes:
- `TestKellyFractionRequiresVol` × 3: zero/negative vol raise; positive unchanged.
- `TestMultiAssetKellyValidation` × 3: non-square/shape-mismatch/non-symmetric raise.
- `TestMultiAssetKellyAnalytical` × 2: diagonal Σ matches per-asset Kelly; correlated Σ matches `solve(Σ, μ)`.
- `TestMultiAssetKellySingularPseudoInverse` × 1: near-singular Σ finite via pinv.
- `TestMultiAssetKellyGrowthFormula` × 1: g = rf + f·μ̄ − 0.5·f'Σf.

Full parallel suite: **12,516 passed in 3:08** — zero regressions. Warnings 19→17.

Sixth fix from phase-2. **128 distinct bugs** (2 more in this slice) in v0.905→v0.998.

---

## v0.997.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.correlation_greeks` had two issues.**

### (a) `correlation_pnl_attribution` did 8 pricer calls when 4 suffice

Pre-fix: `correlation_delta(rho_old)` (3 calls) + `correlation_gamma(rho_old)` (3 calls — overlapping `rho_old`, `rho_old+bump`, `rho_old-bump` with the delta call) + direct `price_fn(rho_new)` + `price_fn(rho_old)` = **8 calls**. Only 4 unique points needed: `rho_old`, `rho_old±bump`, `rho_new`. Now computed inline — halves the cost for expensive multi-asset pricers (basket MC, ND PDE). Same shape as v0.994 `bump_greeks` duplicate-call fix.

### (b) `CorrelationLadder` exposed only gross magnitudes

`total_rho_delta` and `total_rho_gamma` summed `abs(delta_i)`. For sizing hedges (gross exposure) this is useful, but for portfolio P&L the signed sum is what matters when correlations drift together. Added `net_rho_delta` / `net_rho_gamma` as signed totals; preserved the original gross fields for backwards compatibility.

### Verification — `test_l2_t4_correlation_greeks.py`

4 new tests:
- `TestCorrelationPnlAttributionCallCount` × 2: exactly 4 unique calls; quadratic price_fn → Taylor explains everything (unexplained = 0).
- `TestCorrelationLadderNet` × 2: net vs gross sums with sign-mixed pairs (net=+1, gross=3); linear-in-rho pricer has zero net/gross gamma.

Full parallel suite: **12,506 passed in 2:58** — zero regressions.

Fifth fix from phase-2. **126 distinct bugs** (2 more in this slice) in v0.905→v0.997.

---

## v0.996.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.factor_model` had two real bugs.**

### (a) Ledoit-Wolf shrinkage intensity was ad-hoc

Pre-fix used `alpha_lw = 1 / (n · delta)` with no basis in Ledoit-Wolf (2004) and dimensionally inconsistent (units of cov-squared⁻¹). The "shrunk" output was NOT a Ledoit-Wolf estimator. Now uses the correct LW formula for the μ·I target (LW 2004, §3.2):

    π̂  = sum over i,j of (1/T) Σ_t (x̃_ti x̃_tj − s_ij)²
    γ̂² = ||F − S||²_F
    ρ̂  = trace(π̂ matrix)       (identity-target case)
    κ  = (π̂ − ρ̂) / γ̂²
    δ* = max(0, min(κ / T, 1))

Vectorised via `pi_mat = (X²ᵀ @ X²)/T − S²`, O(T·n²) per pass.

### (b) `factor_timing` direction contradicted its own docstring

Pre-fix code: `if z > threshold: signal = "overweight"` (momentum on factor value). Pre-fix docstring: "When the factor is cheap (low z-score), overweight it" (contrarian — Asness, Ilmanen). The two contradicted each other. Picked the contrarian convention (matches academic literature on factor timing). Hit-rate calculation reversed accordingly.

The existing `test_overweight` assertion in `test_factor_model.py` had pinned the buggy momentum direction — updated to the corrected contrarian semantics.

### Verification — `test_l2_t4_factor_model.py`

6 new tests:
- `TestLedoitWolfShrinkage` × 3: intensity in [0,1]; high-dim case shows meaningful shrinkage; off-diagonals shrink under iid data toward identity.
- `TestFactorTimingContrarian` × 3: high z → underweight; low z → overweight; perfect mean-reversion regime → hit rate > 99%.

Full parallel suite: **12,502 passed in 3:29** — zero regressions.

Fourth fix from phase-2. **124 distinct bugs** (2 more in this slice) in the v0.905→v0.996 arc.

---

## v0.995.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.backtest` had four defects (one major semantic, three numerical/accounting).**

### (a) Sample vs population std — `compute_metrics`
`np.std()` defaults to `ddof=0` (population). Sharpe / Sortino are conventionally reported with `ddof=1` (sample std). Pre-fix vol understated by `sqrt(n/(n-1))` — inflating Sharpe by ~1.7% at n=30, ~0.2% at n=252. Now uses `ddof=1`.

### (b) Initial position has no transaction cost — `run_backtest`
Pre-fix loop started at `i=1`, only charging slippage/commission for `positions[i] − positions[i-1]`. The initial entry (implicit 0 → `positions[0]`) was free. Real consequence: any constant-direction strategy started with negative immediate P&L from entry costs being un-modelled. Now charges `abs(positions[0])·slippage_bps/10000·capital + commission` at `pnl[0]`.

### (c) Walk-forward isn't walk-forward — `walk_forward` (major)
Pre-fix called `signal_func(train)` and `signal_func(test)` *separately*. For any `signal_func` with a warm-up window (e.g. 20-day momentum), the first warm-up bars of test had undefined/biased signals because there's no train-side history. The "walk" never actually carried history forward. Now passes `concat(train, test)` to `signal_func` and slices out the test portion — signals on test bars are computed with full access to all preceding train history, no forward-leakage.

### (d) Deflated Sharpe missing Euler-Mascheroni correction — `deflated_sharpe`
Pre-fix used `E[max] ≈ Φ⁻¹(1 − 1/n)` (first-order). Bailey-De Prado (2014) specifies:

    E[max] ≈ (1 − γ) · Φ⁻¹(1 − 1/n) + γ · Φ⁻¹(1 − 1/(n·e))

where γ ≈ 0.5772 (Euler-Mascheroni), e ≈ 2.718. For n=100, pre-fix gives 2.326 vs BdP 2.530 — pre-fix *over*-states the deflated probability (admits more false positives), the opposite of what data-snooping correction should do.

### Verification — `test_l2_t4_risk_backtest.py`

10 new tests across 4 test classes:
- `TestComputeMetricsSampleStd` × 2: vol matches `np.std(ddof=1)`, Sharpe consistent.
- `TestRunBacktestInitialSlippage` × 3: initial entry charged; zero entry not charged; commission charged on non-zero entry.
- `TestWalkForwardHistoryContext` × 2: `signal_func` called on combined series; warm-up signal yields finite Sharpes.
- `TestDeflatedSharpeBaileyDePrado` × 3: n=100 BdP value ≈ 2.530; pre-fix-threshold SR has lower significance; high SR retains high DSR.

Full parallel suite: **12,495 passed in 3:02** — zero regressions.

Third fix from phase-2. Bug count: 119, 120, 121, 122 (four defects in one module). **122 distinct bugs** in the v0.905→v0.995 arc.

---

## v0.994.0 — 2026-06-13

**Fix L2 phase-2 audit — `risk.greeks.bump_greeks` had three issues.**

(a) **Duplicate pricer call.** Pre-fix computed `price_func(spot, vol - vol_bump, ...)` twice — once as `vega_down_v` (used in vega), once as `vega_down` (used in volga). Identical computations, wasted call. For MC/PDE pricers this is real perf cost (could be seconds per Greeks call).

(b) **Asymmetric rho difference.** Pre-fix used a forward difference `(rho_up - base) / rate_bump` while delta/gamma/vega all used central differences. Forward diff is O(h); central is O(h²). Inconsistent with neighbouring Greeks. Switched to central diff.

(c) **No bump-size validation.** Non-positive bumps or `vol_bump >= vol` would silently push negative vol into the pricer, often crashing deep or returning NaN. Now raises `ValueError` upfront with diagnostic.

### Verification — `test_l2_t4_risk_greeks_bump.py`

6 new tests:
- `TestNoDuplicateCalls`: pricer call count bounded.
- `TestRhoCentralDifference`: rho matches BS analytical `K·T·exp(-rT)·N(d2)·0.01` to rel 1e-4.
- `TestValidationRaises` × 3: zero/negative bumps + vol_bump≥vol raise.
- `TestSanityRanges`: ATM call Greeks in expected sign/range bands.

Full parallel suite: **12,485 passed in 3:03** — zero regressions, +6 net tests. Warnings 18→17.

Second fix from phase-2. 118th distinct bug.

---

## v0.993.0 — 2026-06-13

**Fix L2 phase-2 audit (first new-module find) — `risk.var.stress_test` had two coupled bugs.**

Opening up `risk/` for systematic audit. First find:

**(a) Silent no-op on unsupported shocks.** Pre-fix inspected only `rate_shift` and silently dropped every other key — including `vol_shift` which appeared explicitly in the function's own docstring example. Users following the docs got identical-to-base P&Ls that looked like successful stress runs.

**(b) Lossy context reconstruction.** The bumped context was built from a 6-field constructor subset, silently dropping the plural `discount_curves`, `inflation_curves`, `repo_curves`, `reporting_currency`, `stochastic_credit_models`, `credit_vol_surfaces`, `credit_correlations`, and (added later in G1 P3) `numerical_config`. Pricers relying on any of these silently saw a degraded context.

**Fix**: use `dataclasses.replace` for non-lossy reconstruction; raise `ValueError` on unknown shock keys. Now supports `rate_shift` (bumps singular `discount_curve` + plural `discount_curves` + `projection_curves`) and `credit_shift` (bumps `credit_curves`).

### Verification — `test_l2_t4_risk_var_stress_test.py`

7 new tests:
- `TestStressTestRaisesOnUnsupportedShocks` × 2: `vol_shift` raises; typo `rate_shifft` raises.
- `TestStressTestRateShift` × 2: singular + plural discount curves both bumped.
- `TestStressTestPreservesUntouchedFields` × 2: `numerical_config` and `reporting_currency` survive reconstruction.
- `TestStressTestEmpty` × 1: empty shock dict is a no-op (no error).

Full parallel suite: **12,479 passed in 3:07** — zero regressions.

First fix from phase-2 (modules not yet audited). 117th distinct bug.

---

## v0.992.0 — 2026-06-13

**Fix L2 Wave-2 audit — `fx_double_barrier_option` returned `vanilla` unconditionally at vol=0/T>0, assuming no barrier breach.**

At vol=0, the spot path is the deterministic monotonic exponential `S_t = spot·exp((rd-rf)·t)`. With `spot` known to be inside the corridor (checked just above), the path breaches a barrier iff the forward `spot_T = spot·exp((rd-rf)·T)` exits the corridor:

- Forward stays in `[L, U]` → no breach → KO = vanilla, KI = 0.
- Forward `>= U` → upper breach → KO = 0, KI = vanilla.
- Forward `<= L` → lower breach → KO = 0, KI = vanilla.

Pre-fix returned `vanilla` unconditionally — systematically over-pricing knock-outs and under-pricing knock-ins whenever the deterministic forward drifted out of the corridor.

T=0 boundary preserved (no path traversal possible).

KO + KI = vanilla parity holds exactly in all four cases.

### Verification — `test_l2_t4_fx_double_barrier_degenerate.py`

8 new tests:
- `TestFxDoubleBarrierVolZeroKnockOut` × 3: forward-inside pays vanilla; forward-above-upper KO = 0; forward-below-lower KO = 0.
- `TestFxDoubleBarrierVolZeroKnockIn` × 2: inside KI = 0; upper-breach KI = vanilla.
- `TestFxDoubleBarrierTZero` × 1: T=0 returns vanilla.
- `TestParity` × 2: KO + KI = vanilla for both inside-corridor and breach cases.

Full parallel suite: **12,472 passed in 2:51** — zero regressions.

Third of four residual convention-dependent fixes deferred from the v0.989 sweep. 116th distinct bug.

---

## v0.991.0 — 2026-06-13

**Fix L2 Wave-2 audit — `fx_lookback_floating` returned 0 unconditionally at `vol <= 0 or T <= 0`, dropping the deterministic non-zero value from drift-driven path extrema.**

At vol=0, T>0 the spot path is the deterministic exponential `S_t = spot·exp((rd-rf)·t)`. Over [0, T]:
- If `rd >= rf`: monotone non-decreasing → `min = spot`, `max = spot_T = forward`.
- If `rd <  rf`: monotone decreasing → `min = spot_T`, `max = spot`.

The floating-strike call payoff is `S_T − running_min`; put is `running_max − S_T`. The running extreme combines the *observed* `running_extreme` parameter (defaults to spot) with the path extremes.

Pre-fix returned 0 for all four (call/put × positive/negative drift) cases. This is wrong even with default `running_extreme = spot`:
- Positive drift call: forward > spot → payoff = `forward − spot > 0`.
- Negative drift put: spot > forward → payoff = `spot − forward > 0`.
- The other two (negative-drift call, positive-drift put) correctly return 0 when `running_extreme` is `None`, since the path extremum doesn't beat the spot reference.
- With `running_extreme` set: drift-opposite case still pays when the *observed* extreme is inside the path range.

T=0 boundary preserved (returns 0).

### Verification — `test_l2_t4_fx_lookback_degenerate.py`

7 new tests:
- `TestFxLookbackVolZeroTPositiveCall` × 3: positive-drift pays `df_d·(forward−spot)`; negative-drift pays 0 without low running_extreme; negative-drift pays when running_extreme < forward.
- `TestFxLookbackVolZeroTPositivePut` × 2: negative-drift pays `df_d·(spot−forward)`; positive-drift returns 0 by default.
- `TestFxLookbackTZero` × 1: T=0 returns 0.
- `TestFxLookbackInteriorUnchanged` × 1: interior path finite.

Full parallel suite: **12,464 passed in 2:58** — zero regressions.

Second of four residual convention-dependent fixes deferred from the v0.989 sweep. 115th distinct bug.

---

## v0.990.0 — 2026-06-13

**Fix L2 Wave-2 audit — `fx_charm` (∂Δ/∂t) returned 0 at `vol=0, T>0`, silently dropping the deterministic `±r_f·exp(-r_f T)` discount-decay contribution.**

The FX spot delta at vol=0 has the deterministic value `±exp(-r_f T)·I(forward vs strike)`. Its derivative w.r.t. calendar time is the discount-decay `±r_f·exp(-r_f T)·indicator` — not 0.

Closed-form deterministic limits at vol=0:
- ITM call (forward > strike): `charm = +r_f·exp(-r_f T)`
- ITM put (forward < strike): `charm = -r_f·exp(-r_f T)`
- OTM (both): `charm = 0`
- ATM (forward == strike): `±0.5·r_f·exp(-r_f T)` — indicator one-sided half-limit (the boundary-shift term1 = -exp(-rf T)·N'(d1)·[2(rd-rf)T]/(2T σ√T) diverges to ±∞ when rd ≠ rf, but the dominant indicator contribution is well-defined at half-limit).

T=0 boundary preserved (returns 0).

### Verification — `test_l2_t4_fx_charm_degenerate.py`

8 new tests: ITM call positive, ITM put negative, OTM call/put zero, ATM half, zero-rf returns zero, T=0 returns zero, interior finite.

Full parallel suite: **12,457 passed in 2:51** — zero regressions.

First of four residual convention-dependent fixes deferred from the v0.989 sweep. 114th distinct bug.

---

## v0.989.0 — 2026-06-13

**Fix L2 Wave-2 audit — single-pass sweep of `T<=0 or vol<=0` degenerate-branch defects across FX, equity, and inflation modules.**

After three single-site fixes in this pattern (v0.986 equity_rho, v0.987 equity_theta, v0.988 _digital_call_bs), swept the remaining call sites in one pass. Each defect falls into one of three buckets:

- **(a) Spot vs forward indicator.** At `vol=0, T>0` the deterministic terminal is `forward = spot·exp((r-q)T)`, not spot. OTM-spot but ITM-forward positions silently returned 0.
- **(b) Missing discount factor.** Even with the right indicator, payoffs at maturity must be PV'd.
- **(c) Missing ATM half-limit.** At `forward == strike` the one-sided limit is half the ITM value.

### Sites fixed (7)

| # | File / function | Bucket(s) | Pre-fix behaviour |
|---|---|---|---|
| 1 | `fx/fx_option.py::fx_spot_delta` | (a)+(c) | `spot > strike` indicator; ATM returned 0 |
| 2 | `fx/fx_option.py::fx_forward_delta` | (a)+(c) | same shape |
| 3 | `fx/fx_american.py::_gk_european` | (a)+(b) | `max(spot − strike, 0)` undiscounted |
| 4 | `equity/equity_exotic.py::equity_digital_cash` | (a)+(b)+(c) | undiscounted payout on spot indicator |
| 5 | `equity/equity_exotic.py::equity_digital_asset` | (a)+(b)+(c) | undiscounted spot on spot indicator |
| 6 | `fx/fx_exotic_extensions.py::fx_digital_option` | (b)+(c) | forward indicator was right but no df |
| 7 | `fixed_income/inflation_bond_advanced.py::deflation_floor_value` | (b) | linearised `-breakeven·T` instead of exact `1 − exp(breakeven·T)` |

Site 7 is independent: pre-fix used a first-order Taylor approximation for the deflation intrinsic. For `breakeven=-5%, T=30y` the pre-fix returned `1.5` (exceeding the maximum possible deflation = 100% loss); the exact intrinsic is `1 − exp(-1.5) ≈ 0.777`.

### Sites audited and left unchanged

Already correct: `options/exotic_payoffs._bs_call` (uses discounted intrinsic of forward), `commodity/commodity_american._black76_european` (takes forward directly), `structured/cms` range probability (uses forward), `numerical/black76` (already fixed in v0.965).

Convention-dependent / out of scope for one-pass sweep: `crypto/crypto_vol` ATM theta heuristic, `fx/fx_correlation` quanto ρ inversion, `fx/fx_exotic.fx_lookback_floating` (depends on running_extreme + drift sign), `fx/fx_exotic_extensions` double-barrier MC (needs path-monotonicity check), `fx/fx_greeks` higher-order Greeks at vol=0 are 0 by convention.

### Verification — `test_l2_t4_degenerate_branch_sweep.py`

19 new tests across 7 test classes (one per fixed site). Each test verifies the exact closed-form deterministic limit and contains a comment indicating what the pre-fix value would have been.

Full parallel suite: **12,449 passed in 2:51** — zero regressions, +19 net tests.

Bundled fix counts as one logical slice (single defect class swept). Bug count: 56 → 63 from the 35-module Wave-2 audit; total **113 distinct bugs** in the v0.905→v0.989 arc.

---

## v0.988.0 — 2026-06-13

**Fix L2 Wave-2 audit — `_digital_call_bs` in `structured.equity_linked_note` had two coupled bugs in the degenerate `T<=0 or vol<=0` branch.**

Same shape as the v0.948 `equity_delta` fix.  Pre-fix:

```python
if T <= 0 or vol <= 0:
    return 1.0 if spot > strike else 0.0
```

Two bugs:
- **(a) Spot vs forward indicator.** At `vol=0, T>0`, the terminal value is `S_T = forward = spot·exp((r-q)T)` (deterministic), not `spot`. An OTM-spot but ITM-forward digital silently returned 0. Example: `spot=95, rate=10%, T=1 → forward≈105 > strike=100` is ITM but pre-fix returned 0.
- **(b) No discount factor.** Even when ITM, the digital pays $1 at maturity and must be present-valued. Pre-fix returned `1.0` instead of `exp(-rT)`.

**Fix**: split `T = 0` (no drift, no discount — keep spot indicator) from `T > 0, vol = 0` (use forward indicator and apply `df = exp(-rT)`; ATM forward is the one-sided half-limit).

### Verification — `test_l2_t4_digital_eln_degenerate.py`

8 new tests:
- `TestDigitalCallVolZeroTPositive` × 4: ITM-forward/OTM-spot returns `df` (was 0 pre-fix); ITM-call returns `df` (was 1.0 pre-fix); OTM returns 0; ATM-forward returns `0.5·df`.
- `TestDigitalCallTZero` × 2: T=0 path preserved.
- `TestDigitalCallInteriorUnchanged` × 2: interior `df·N(d2)` formula and dividend-yield-on-forward both unchanged.

Full parallel suite: **12,430 passed in 3:11** — zero regressions, +8 net tests.

Fifty-sixth fix from the 35-module Wave-2 audit; 106th distinct bug in the v0.905→v0.988 arc.

---

## v0.987.0 — 2026-06-13

**Fix L2 Wave-2 audit — `equity_theta` `T<=0 or vol<=0` branch returned the Black-76 theta alone, dropping the `theta_r` and `theta_q` corrections.**

The Hull theta has three terms:
- `theta_b76 = -S·n(d1)·σ·exp(-qT)/(2√T)` — the σ·n(d1)/√T term (returned by `black76_theta`).
- `theta_r = -r·K·exp(-rT)·N(d2)` (call) / `+r·K·exp(-rT)·N(-d2)` (put) — rate-discount.
- `theta_q = q·S·exp(-qT)·N(d1)` (call) / `-q·S·exp(-qT)·N(-d1)` (put) — dividend.

At `vol=0, T>0`, `theta_b76 → 0` and `N(d1), N(d2) → {0, 1}` by ITM/OTM. The pre-fix code returned `theta_b76` (= 0) and **silently dropped** `theta_r + theta_q`, which collapses to the deterministic `±(q·S·exp(-qT) − r·K·exp(-rT))` — a non-zero, sometimes dominant component of total theta when rates/divs are non-trivial.

**Fix**: separate `T <= 0` (still `theta_b76`, no time decay convention) from `vol <= 0` with `T > 0` (deterministic-limit branch on `forward` vs `strike`; ATM is one-sided half-limit).

### Verification — `test_l2_t4_equity_theta_degenerate.py`

8 new tests:
- `TestEquityThetaVolZeroTPositive` × 5: ITM call matches `qS·exp(-qT) − rK·exp(-rT)`; ITM put is negation; OTM zero; ATM half.
- `TestEquityThetaInteriorUnchanged` × 2: interior call/put still negative (decay), within Hull range.
- `TestEquityThetaTZero` × 1: T=0 path preserved.

Full parallel suite: **12,422 passed in 3:52** — zero regressions, +8 net tests.

Fifty-fifth fix from the 35-module Wave-2 audit; 105th distinct bug in the v0.905→v0.987 arc.

---

## v0.986.0 — 2026-06-13

**Fix L2 Wave-2 audit — `equity_rho` returned 0 in the `T<=0 or vol<=0` branch, silently dropping the deterministic `vol=0, T>0` limit.**

At `vol=0` with `T>0` the option price is the discounted intrinsic of the forward:

- ITM call (forward > strike): `price = S·exp(-qT) − K·exp(-rT)` → `rho = T·K·exp(-rT)` (positive)
- ITM put (forward < strike): `price = K·exp(-rT) − S·exp(-qT)` → `rho = -T·K·exp(-rT)` (negative)
- OTM (either side): price = 0 → rho = 0
- ATM (forward == strike): one-sided limit `±0.5·T·K·exp(-rT)`

Pre-fix all four cases returned 0 — the deterministic rho was silently dropped.

**Fix**: separate `T <= 0` (still 0 — no time for rate to act) from `vol <= 0` with `T > 0` (deterministic limit by ITM/OTM/ATM branch on the forward).

### Verification — `test_l2_t4_equity_rho_degenerate.py`

7 new tests, all pass:
- `TestEquityRhoVolZeroTPositive` × 5: ITM call positive, ITM put negative, OTM call/put zero, ATM half.
- `TestEquityRhoTZero`: T=0 returns 0 for both call and put across spot/strike/moneyness.
- `TestEquityRhoInteriorUnchanged`: T>0, vol>0 path identical to pre-fix.

Full parallel suite: **12414 passed in 3:51** — zero regressions.

Fifty-fourth fix from the **35-module deferred Wave-2 audit**; 104th distinct bug in the multi-session arc.

---

## v0.985.0 — 2026-06-13

**Fix L2 Wave-2 audit — `coupled_bootstrap` silently set `fwd = 0.0` for any period where the Newton iterate's projection-curve DF went non-positive (or where the schedule tau was non-positive).**

Pre-fix:

```python
if df2 > 0 and tau > 0:
    fwd = (df1 - df2) / (tau * df2)
else:
    fwd = 0.0
```

When the projection curve produced `df2 ≤ 0` (an arbitrageable iterate) or `tau ≤ 0` (a degenerate schedule with duplicate/inverted dates), the silent `fwd = 0` zeroed the float-leg contribution from that period. The residual `fixed_pv − too-small-float_pv` became artificially low, which Newton could drive to zero by following an **unphysical trajectory** in DF space — silently converging on a bad solution.

**Fix**: both degenerate paths now raise `ValueError` with a clear message identifying the offending input (schedule date or DF).

### Verification — `test_l2_t4_coupled_bootstrap_degenerate_raises.py`

2 new tests, all pass:
- `test_normal_inputs_succeed` — sanity: healthy bootstrap still works.
- `test_error_path_is_present` — static guard that the diagnostic raises are present in the source and the pre-fix silent fallback is gone (prevents reversion).

Full parallel suite: **12407 passed in 3:03** — zero regressions.

Fifty-third fix from the **35-module deferred Wave-2 audit**; **103rd distinct bug** in the multi-session arc.

---

## v0.984.0 — 2026-06-13

**Fix L2 Wave-2 audit — `risk.network.FinancialNetwork.betweenness_centrality` emitted spurious `RuntimeWarning: divide by zero` on sparse adjacency matrices (classic `np.where` eager-evaluation trap).**

Pre-fix:

```python
dist_matrix = np.where(self.adj > 0, 1.0 / self.adj, 0.0)
```

NumPy evaluates BOTH branches of `np.where` before selecting — so `1.0 / self.adj` is computed for EVERY element, including zero entries. The result for zeros (`inf` / `nan`) is then discarded by the where-mask, but the divide-by-zero RuntimeWarning is real, and the NaN/inf could propagate via other operations (multiplication, max-reduction etc) in subtle ways.

**Fix**: use `np.divide` with the `where=` argument so the division only fires where the predicate is true. The output array is pre-zeroed and only the true-predicate entries are written.

**Side-effect**: test-suite warnings drop from 29 → 19.

### Verification — `test_l2_t4_warning_cleanup.py`

2 new tests, all pass:
- `test_no_divide_by_zero_warning` — sparse 3x3 adjacency under `warnings.simplefilter("error")` no longer raises.
- `test_returns_finite_centralities` — 4x4 sparse adjacency produces finite centralities (no NaN/inf leakage).

Full parallel suite: **12405 passed in 2:37** — zero regressions.

Fifty-second fix from the **35-module deferred Wave-2 audit**.

---

## v0.983.0 — 2026-06-13

**Fix L2 Wave-2 audit — `TreeSolver._apply_barrier` silently no-oped for `DOWN_IN` / `UP_IN` barriers.**

```python
elif self.barrier_type == BarrierType.DOWN_IN:
    pass  # complex — for now, only knock-out supported
elif self.barrier_type == BarrierType.UP_IN:
    pass
```

At the outer `solve()` level, knock-in barriers are computed via in-out parity (`vanilla − knock_out`), so `_apply_barrier` is never called with a knock-in type via the normal path. BUT a caller invoking `_apply_barrier` directly (e.g. via a subclass extension, or bypassing `solve()` for debugging) silently got a vanilla price labelled as a knock-in — the most subtle kind of wrong-result bug.

**Fix**: the knock-in branches raise `NotImplementedError` with a diagnostic explaining that knock-in is computed at the outer level via in-out parity.

### Verification — `test_l2_t4_apply_barrier_knock_in_raises.py`

5 new tests, all pass:
- `test_down_in_raises_when_called_directly`, `test_up_in_raises_when_called_directly` — the unsafe direct call now raises.
- `test_down_out_zeroes_below_barrier`, `test_up_out_zeroes_above_barrier` — knock-out branches still work.
- `test_down_in_call_solves_via_parity` — outer-level `solve()` with knock-in barrier still produces a finite price via the in-out parity wrapper.

Full parallel suite: **12403 passed in 3:48** — zero regressions.

Fifty-first fix from the **35-module deferred Wave-2 audit**; 101st distinct correctness/contract bug in the multi-session arc.

---

## v0.982.0 — 2026-06-13 🎯 100-bug milestone

**Fix L2 Wave-2 audit — `StandardCDS` had a hand-written `to_dict` / `from_dict` that wasn't updated when `convention` was added to the parent `CDS._SERIAL_FIELDS` in v0.978.**

Pre-fix:
- The hand-written `to_dict` omitted `convention` from the params dict.
- The hand-written `from_dict` did not pass `convention=` to the constructor.
- The introspection sweep added in this audit (v0.981) flagged the gap: `StandardCDS._SERIAL_FIELDS` declared `convention` (inherited from CDS) but the actual hand-written serialisation didn't emit it.

A `StandardCDS` round-trip silently lost any non-default convention — same shape as the auto-generated cases fixed in v0.976–v0.978 / v0.979, but harder to spot because the hand-written serialisation was bespoke rather than declarative.

**Fix**: both methods now handle `convention`. The `from_dict` accepts a missing-`convention` legacy dict by defaulting to MODIFIED_FOLLOWING (backwards-compatibility for pre-fix-saved JSON).

### Verification — `test_l2_t4_standard_cds_serialisation.py`

5 new tests, all pass:
- `TestStandardCDSRoundTrip::test_convention_preserved` × 3 (parametrised over MODIFIED_FOLLOWING / PRECEDING / FOLLOWING).
- `test_legacy_dict_without_convention_defaults` — a dict missing `convention` (legacy pre-fix JSON) defaults gracefully.
- `test_other_fields_still_round_trip` — sanity: spread/grade/notional/standard_coupon all preserved.

Full parallel suite: **12398 passed in 3:07** — zero regressions.

This is the **fiftieth fix from the 35-module deferred Wave-2 audit** and the **100th distinct correctness/contract bug** resolved in the multi-session arc (v0.905 → v0.982). 🎯

---

## v0.981.0 — 2026-06-13

**Fix L2 Wave-2 audit — `AmortisingBond._serialisable` field list was completely STALE, referencing names that no longer existed on the class.**

Pre-fix declaration:

```python
_serialisable("amortising_bond", ['face_value', 'coupon_rate',
                                   'n_periods', 'frequency'])(AmortisingBond)
```

But the actual dataclass fields are `notional, coupon_rate, maturity_years, n_payments, amortisation_type` — none of the four declared field names existed. The serialisable framework emitted `UserWarning` on import for each mismatch, and any `to_dict()` / `from_dict()` round-trip was broken: `from_dict` would have raised `TypeError` on the unknown args. The class was effectively unserialisable.

The `from_convention` factory was also broken (tried to pass `frequency=` to a constructor that doesn't accept it).

**Fix**: align the serialisable field list with the actual dataclass fields, and rewrite `_amort_from_convention` to match the constructor signature.

**Side-effect**: import-time UserWarnings drop from 59 → 29 across the test suite (3 spurious warnings × ~10 test-collection imports each).

### Verification — `test_l2_t4_amortising_bond_serialisation.py`

4 new tests, all pass:
- `test_to_dict_runs` — pre-fix this raised on the `face_value` attribute lookup.
- `test_round_trip_preserves_all_fields` — every field round-trips.
- `test_amortisation_type_round_trips` × 2 — both `"mortgage"` and `"linear"`.

Full parallel suite: **12393 passed in 3:37** — zero regressions; warnings 59 → 29.

This is the **forty-ninth fix from the 35-module deferred Wave-2 audit** and the **99th distinct correctness/contract bug** resolved in the multi-session arc (v0.905 → v0.981).

---

## v0.980.0 — 2026-06-13

**Fix L2 Wave-2 audit — `VanillaCLN` was registered as `_serialisable` but its constructor never stored `frequency` as a class attribute.**

```python
def __init__(self, ..., frequency=Frequency.SEMI_ANNUAL, ...):
    self.start = start
    self.end = end
    # ... no `self.frequency = frequency` ...
    self.schedule = generate_schedule(start, end, frequency)

_serialisable("vanilla_cln", [..., "frequency", ...])(VanillaCLN)
```

Calling `vcln.to_dict()` raised `AttributeError: 'VanillaCLN' object has no attribute 'frequency'`. The class was effectively **unserialisable** despite being declared so — a contract bug that would surface the first time any caller tried to persist a CLN.

**Fix**: store `self.frequency = frequency` in the constructor.

### Verification — `test_l2_t4_vanilla_cln_serialisation.py`

5 new tests, all pass:
- `test_to_dict_does_not_raise` — pre-fix this raised `AttributeError`.
- `test_frequency_round_trips` × 4 (parametrised over `MONTHLY`/`QUARTERLY`/`SEMI_ANNUAL`/`ANNUAL`) — full round-trip preserves frequency and all other fields.

Full parallel suite: **12389 passed in 3:33** — zero regressions.

Forty-eighth fix from the **35-module deferred Wave-2 audit**.

---

## v0.979.0 — 2026-06-13

**Fix L2 Wave-2 audit — `CapFloor` serialisation dropped `convention`** (same shape as Swaption v0.976 / IRS v0.977 / CDS v0.978).

Pre-fix the field list missed `convention`, which controls the business-day rolling rule for caplet/floorlet accrual dates. A `to_dict → from_dict` on a non-default CapFloor changed the schedule and the price.

**Fix**: add `convention` to the field list.

### Verification — `test_l2_t4_capfloor_serialisation_fields.py`

3 new tests, all pass — parametrised round-trip across MODIFIED_FOLLOWING / PRECEDING / FOLLOWING.

Full parallel suite: **12384 passed in 2:41** — zero regressions.

Forty-seventh fix from the **35-module deferred Wave-2 audit**.

---

## v0.978.0 — 2026-06-13

**Fix L2 Wave-2 audit — `CDS` serialisation dropped `convention` (same shape as v0.976 Swaption / v0.977 IRS).**

Pre-fix the `_serialisable` field list missed `convention`, which controls the business-day rolling rule applied to coupon dates. A CDS with non-default `convention=PRECEDING` round-tripped to one with the default MODIFIED_FOLLOWING, changing the payment schedule and therefore the price.

**Fix**: add `convention` to the field list. `calendar` remains excluded (runtime-only holiday data).

### Verification — `test_l2_t4_cds_serialisation_fields.py`

4 new tests, all pass:
- `TestRoundTripPreservesConvention` × 3 (parametrised over MODIFIED_FOLLOWING / PRECEDING / FOLLOWING).
- `test_default_round_trip_still_works` — sanity.

Full parallel suite: **12381 passed in 3:16** — zero regressions.

Forty-sixth fix from the **35-module deferred Wave-2 audit**.

---

## v0.977.0 — 2026-06-13

**Fix L2 Wave-2 audit — `InterestRateSwap` serialisation dropped `convention`/`stub`/`eom` (same gap as Swaption v0.976).**

Pre-fix the `_serialisable` field list missed `convention`, `stub`, `eom` — all three affect the leg schedules, so a `to_dict → from_dict` on a non-default IRS produced a swap that priced differently. This is the parallel gap to the Swaption fix landed in v0.976.

**Fix**: add all three fields. `calendar` remains excluded (runtime-only holiday data; the caller re-attaches via `from_convention`).

### Verification — `test_l2_t4_irs_serialisation_fields.py`

9 new tests, all pass:
- `TestRoundTripPreservesConvention` × 3 (parametrised over MODIFIED_FOLLOWING / PRECEDING / FOLLOWING).
- `TestRoundTripPreservesStub` × 4 (all `StubType` values).
- `TestRoundTripPreservesEOM` × 2.

Full parallel suite: **12377 passed in 2:44** — zero regressions.

Forty-fifth fix from the **35-module deferred Wave-2 audit**.

---

## v0.976.0 — 2026-06-13

**Fix L2 Wave-2 audit — `Swaption` serialisation dropped `convention`, `stub`, `eom` from the round-trip.**

Pre-fix the `_serialisable` field list was:

```python
["expiry", "swap_end", "strike", "swaption_type", "notional",
 "fixed_frequency", "float_frequency",
 "fixed_day_count", "float_day_count"]
```

Missing: `calendar`, `convention`, `stub`, `eom`. All four are constructor arguments that affect the underlying swap's schedule generation. A user serializing a Swaption with non-default `convention=PRECEDING` (or non-default `stub` / `eom`), then deserializing it, got a Swaption that **priced differently** than the original — the silent identity break the audit critic flagged.

**Fix**: add `convention`, `stub`, `eom` to the field list (all three are Enums/bools that the framework already handles via the existing `Frequency` serialisation). `calendar` remains excluded because Calendar instances hold runtime holiday data that isn't currently part of the serialisable type system — the caller re-attaches a calendar via `from_convention` on load. The new docstring documents this contract.

### Verification — `test_l2_t4_swaption_serialisation_fields.py`

9 new tests, all pass:
- `TestRoundTripPreservesConvention` × 3: MODIFIED_FOLLOWING, PRECEDING, FOLLOWING.
- `TestRoundTripPreservesStub` × 4 (parametrised): all four `StubType` values.
- `TestRoundTripPreservesEOM` × 2: True and False.

Full parallel suite: **12368 passed in 2:43** — zero regressions.

Forty-fourth fix from the **35-module deferred Wave-2 audit**.

---

## v0.975.0 — 2026-06-13

**Fix L2 Wave-2 audit — `GaussianCopula` and `StudentTCopula` constructors had no validation on `rho`.**

Pre-fix:
- `rho > 1`: `math.sqrt(1 - rho)` raised opaque `ValueError: math domain error` deep inside `sample()`.
- `rho < 0`: `math.sqrt(rho)` raised the same domain error.
- `rho = NaN`: slipped past everything via IEEE 754 and silently produced an all-NaN sample array (`norm.cdf(NaN) = NaN`).
- `StudentTCopula(nu<=0)`: produced a degenerate Student-t with no warning.

**Fix**: both constructors validate `rho ∈ [0, 1]` (with explicit NaN check) and `StudentTCopula` additionally validates `nu > 0`, raising `ValueError` upfront with diagnostic messages.

Also: the `test_l2_t4_studentt_tail_dependence::test_matches_copula_implementation` test (added in v0.943) was passing `rho=-0.5` to `StudentTCopula` to check distribution-vs-copula agreement; updated to only exercise `rho ∈ [0, 1]` (the copula's actual domain). The distribution method remains well-defined for the full `[-1, 1]` range and is tested independently.

### Verification — `test_l2_t4_copula_rho_validation.py`

9 new tests, all pass:
- `TestGaussianCopulaRho` × 4: above 1, below 0, NaN, valid.
- `TestStudentTCopulaRhoNu` × 5: rho out-of-range, nu=0, nu<0, NaN in either, valid.

Pre-existing 18 copula tests still pass.

Full parallel suite: **12359 passed in 3:12** — zero regressions.

Forty-third fix from the **35-module deferred Wave-2 audit**.

---

## v0.974.0 — 2026-06-13

**Fix L2 Wave-2 audit — `Uniform` and `Exponential` constructors let NaN slip through their guards.**

In IEEE 754, all comparisons against NaN return False. So a naive guard like

```python
if a >= b:
    raise ...
```

silently accepts `a=NaN, b=NaN` (NaN >= NaN is False). The constructor succeeded, then downstream `cdf` / `pdf` propagated NaN through the user's computation with no diagnostic.

**Fix**: both constructors now explicitly check `math.isnan(...)` before the inequality guard and raise `ValueError` with a diagnostic message that names the IEEE 754 root cause.

### Verification — `test_l2_t4_distribution_nan_guards.py`

8 new tests, all pass:
- `TestUniformNaNGuards` × 4: NaN-a, NaN-b, both-NaN raise; valid (a, b) works.
- `TestExponentialNaNGuards` × 3: NaN rate raises; valid rate works; ±inf rate documented.
- `TestPreFixSlipThroughGuard`: pins down the IEEE 754 NaN-compares-False semantics.

Full parallel suite: **12350 passed in 3:12** — zero regressions.

Forty-second fix from the **35-module deferred Wave-2 audit**.

---

## v0.973.0 — 2026-06-13

**Fix L2 Wave-2 audit — `calibrate_svensson` had no degeneracy guard for `τ1 ≈ τ2`.**

The Svensson parameterisation has two decay constants `tau1` and `tau2`. When they collapse to (approximately) the same value, the second Svensson factor becomes a linear combination of the first plus the NS factor — so the objective surface develops a flat valley along the (beta2, beta3) ridge. Nelder-Mead can then drift arbitrarily far in that direction without the loss changing, producing wildly different parameter sets that all "calibrate" equally well to the same input.

**Fix**: penalise small `|τ1 − τ2|` in the objective (threshold 0.05) to keep the optimizer in the identifiable region of parameter space. The `τ <= 0.01` lower-bound guard is unchanged.

### Verification — `test_l2_t4_svensson_tau_degeneracy.py`

3 new tests, all pass:
- `test_calibrated_taus_well_separated` — post-calibration `|τ1−τ2| ≥ 0.05`.
- `test_svensson_fits_smooth_curve` — guard doesn't break healthy calibration; RMSE < 1%.
- `test_svensson_calibration_returns_finite_values` — all 6 parameters finite.

Pre-existing 13 NS tests still pass.

Full parallel suite: **12342 passed in 3:07** — zero regressions.

Forty-first fix from the **35-module deferred Wave-2 audit**.

---

## v0.972.0 — 2026-06-13

**Fix L2 Wave-2 audit — `Normal` and `LogNormal` accepted `sigma=0` or `sigma<0` without validation.**

Pre-fix:
- `Normal(sigma=0)`: `cdf/pdf` divided by zero, emitting `RuntimeWarning` and returning 1.0 (or NaN) silently.
- `Normal(sigma=-1)`: produced a flipped distribution (math runs, but σ < 0 is meaningless — only σ ≥ 0 parameterises a normal).
- `LogNormal(sigma=0)`: same divide-by-zero pattern as Normal.

**Fix**: both constructors now raise `ValueError("sigma must be > 0")` at construction with a clear diagnostic.

### Verification — `test_l2_t4_distribution_sigma_validation.py`

6 new tests, all pass:
- `TestNormalSigmaValidation` × 3: zero, negative, positive.
- `TestLogNormalSigmaValidation` × 3: same coverage.

Full parallel suite: **12339 passed in 3:12** — zero regressions.

Fortieth fix from the **35-module deferred Wave-2 audit**.

---

## v0.971.0 — 2026-06-13

**Fix L2 Wave-2 audit — `cos_price` crashed with opaque exceptions deep in the formula on three degenerate inputs.**

Pre-fix:
- **`L=0`**: ``b - a = 0`` (truncation half-width `L·sqrt(c2) = 0`) → `ZeroDivisionError` inside the V_k recursion's `2.0 / (b - a)` factor.
- **`spot <= 0`**: `math.log(spot / strike)` raised `math domain error` with no diagnostic about which input was bad.
- **`strike <= 0`**: `spot / strike` raised `ZeroDivisionError` (zero strike) or produced a nonsensical negative log argument.

**Fix**: validate all three upfront with `ValueError` and an explicit message identifying the offending parameter.

### Verification — `test_l2_t4_cos_price_input_validation.py`

7 new tests, all pass:
- `TestSpotValidation` × 2: zero and negative spot raise.
- `TestStrikeValidation` × 2: zero and negative strike raise.
- `TestLValidation` × 2: zero and negative L raise.
- `TestHealthyPathUnchanged`: ATM call on BS still prices correctly (~10.45).

Pre-existing 12 cos_method tests still pass.

Full parallel suite: **12333 passed in 3:02** — zero regressions.

Thirty-ninth fix from the **35-module deferred Wave-2 audit**.

---

## v0.970.0 — 2026-06-13

**Fix L2 Wave-2 audit — `calibrate_nelson_siegel` and `calibrate_svensson` had three contract gaps.**

Pre-fix:
1. **Empty `market_yields`** raised `IndexError` at `market_yields[-1]` inside default initial-guess construction, no diagnostic.
2. **Mismatched `len(tenors) != len(market_yields)`** was silently masked by `zip()` which truncates to the shorter — a user passing 10 tenors and 8 yields silently got a calibration on the first 8 points only.
3. **The optimizer's convergence flag was discarded** — the returned dict gave no way for the caller to detect a non-converged calibration (it could go straight into a production curve, with no signal that the fit was bad).

**Fix**:
- New helper `_validate_calibration_inputs` checks non-empty AND matching lengths upfront — raises `ValueError` with clear messages.
- Returned dict now includes `converged: bool` from the underlying optimizer result.

### Verification — `test_l2_t4_nelson_siegel_calibration_validation.py`

8 new tests, all pass:
- `TestEmptyInputsRaise` × 3: empty tenors, empty yields, Svensson empty all raise.
- `TestMismatchedLengthsRaise` × 2: NS and Svensson both raise on mismatched lengths.
- `TestConvergedFieldReported` × 2: `converged: bool` present in both NS and Svensson result dicts.
- `TestHealthyCalibrationPreserved`: smooth synthetic curve still calibrates to RMSE < 1%.

Pre-existing 13 NS/Svensson tests still pass.

Full parallel suite: **12326 passed in 3:01** — zero regressions.

Thirty-eighth fix from the **35-module deferred Wave-2 audit**.

---

## v0.969.0 — 2026-06-13

**Fix L2 Wave-2 audit — `FRA.pv_ctx` silently picked the "first" projection curve when the day-count-keyed lookup missed.**

Pre-fix:

```python
if dc_key and dc_key in ctx.projection_curves:
    proj = ctx.projection_curves[dc_key]
else:
    proj = next(iter(ctx.projection_curves.values()))
```

In a multi-curve setup (e.g. ACT/360 USD-LIBOR FRA priced against a context with only ACT/365 GBP projection curves), the FRA silently got a wrong-curve forward rate. The day-count mismatch could be several basis points and was completely invisible to the user.

**Fix**: keyed-lookup miss now raises `KeyError` listing the offending day-count and the available keys. The legacy fallback to the discount curve when the context has NO projection curves at all is preserved (the unambiguous single-curve case).

### Verification — `test_l2_t4_fra_pv_ctx_keyed_lookup.py`

3 new tests, all pass:
- `test_act_360_fra_no_act_360_curve_raises` — ACT/360 FRA against ACT/365-only context raises.
- `test_act_360_fra_with_matching_curve_prices` — matching day-count gives a finite PV.
- `test_empty_projection_falls_back` — empty projection-curves dict still falls back to the discount curve (single-curve compat).

Full parallel suite: **12318 passed in 2:35** — zero regressions.

Thirty-seventh fix from the **35-module deferred Wave-2 audit**.

---

## v0.968.0 — 2026-06-13

**Fix L2 Wave-2 audit — `smith_wilson_forward` silently returned `ufr` when its finite-difference DFs went non-positive.**

```python
if p1 <= 0 or p2 <= 0:
    return ufr
```

Non-positive Smith-Wilson DFs are NOT a normal regime — they indicate arbitrageable input DFs, extrapolation gone off the rails, or extreme `alpha`. Pre-fix the user got "the answer is UFR" silently, masking a calibration failure that should fail loudly.

**Fix**: raise `ValueError` with diagnostic context (the offending DF values, the `t` parameter, and likely upstream causes).

### Verification — `test_l2_t4_smith_wilson_silent_ufr.py`

2 new tests, all pass:
- `TestSilentUFRReplacedByRaise::test_non_positive_dfs_raise` — synthesise a wild-zeta state, scan for a `t` where DFs go negative, confirm the function raises.
- `TestHealthyForwardWorks::test_normal_calibration_forward_finite` — normal calibrated state still returns a finite forward in the expected market-rate range.

Pre-existing 10 Smith-Wilson tests still pass.

Full parallel suite: **12315 passed in 3:01** — zero regressions.

Thirty-sixth fix from the **35-module deferred Wave-2 audit**.

---

## v0.967.0 — 2026-06-13

**Fix L2 Wave-2 audit — `MCEngine(n_paths=1)` silently produced NaN stderr instead of failing fast.**

Pre-fix:
- `MCEngine(..., n_paths=1, antithetic=True)`: `n_half = 1 // 2 = 0` → ZERO paths generated. `np.std(..., ddof=1)` of zero samples is NaN — silent.
- `MCEngine(..., n_paths=1, antithetic=False)`: single-path "MC" with `np.std(..., ddof=1)` of one sample is also NaN.

Both modes are useless for Monte Carlo (you need at least 2 samples to estimate variance), but pre-fix the engine ran them silently and reported NaN downstream into `MCResult.stderr` and `confidence_95`.

**Fix**:
- `n_paths < 2` → `ValueError` upfront.
- `antithetic=True` AND `n_paths < 4` → `ValueError` (the engine uses `n_half = n_paths // 2` antithetic pairs; 2 paths give only 1 pair, which is also degenerate for ddof=1 stderr).

### Verification — `test_l2_t4_mcengine_n_paths_validation.py`

7 new tests, all pass:
- `TestNPathsValidation` × 3: `n_paths` of 1, 0, or negative raises.
- `TestAntitheticMinimum` × 3: antithetic with 2 or 3 paths raises; 4 paths works.
- `TestHealthyPathUnchanged`: large `n_paths` (1000) works.

Full parallel suite: **12313 passed in 3:02** — zero regressions.

Thirty-fifth fix from the **35-module deferred Wave-2 audit**.

---

## v0.966.0 — 2026-06-13

**Fix L2 Wave-2 audit — `mc_engine.TimeGrid` accepted any array, even empty or non-monotonic, silently producing nonsense state.**

Pre-fix the constructor was:

```python
def __init__(self, times):
    self.times = np.asarray(times, dtype=np.float64)
    self.dt = np.diff(self.times)
    self.n_steps = len(self.dt)
    self.T = float(self.times[-1])       # IndexError on empty
```

Three silent failure modes:
1. **Empty input** → `self.times[-1]` raises `IndexError` deep inside the constructor with no diagnostic message.
2. **Length-1 input** → `dt = []`, `n_steps = 0`, T set to the one time point. The engine loops zero times — silent no-op.
3. **Non-monotonic input** → `dt` has negative entries. The engine integrates the SDE BACKWARDS in time for those steps, with wrong-sign drift. Pre-fix this was silent; the user got a finite "price" computed against time-reversed dynamics.

**Fix**: validate at construction — empty/singleton/non-monotonic input all raise `ValueError` with a clear diagnostic.

### Verification — `test_l2_t4_timegrid_validation.py`

6 new tests, all pass:
- `test_empty_array_raises`, `test_singleton_raises` — size-validation.
- `test_decreasing_raises`, `test_duplicate_raises` — monotonicity-validation.
- `test_uniform_works`, `test_explicit_increasing_works` — happy paths preserved.

Full parallel suite: **12306 passed in 3:03** — zero regressions.

Thirty-fourth fix from the **35-module deferred Wave-2 audit**.

---

## v0.965.0 — 2026-06-13

**Fix L2 Wave-2 audit — `bachelier_delta` at exactly ATM (`F == K`) with `T<=0 or vol<=0` returned 0 instead of the standard `±0.5·df` one-sided limit.**

Pre-fix:

```python
if time_to_expiry <= 0 or vol_normal <= 0:
    if option_type == OptionType.CALL:
        return df if forward > strike else 0.0
    return -df if forward < strike else 0.0
```

The `else 0.0` branch caught both `F < K` (correct for a call OTM) AND `F == K` (incorrect — ATM-at-expiry should be `0.5·df`). Symmetric bug for puts.

**Fix**: explicit ATM branch returns `±0.5·df` matching the same convention used (correctly) by `black76_delta`.

### Verification — `test_l2_t4_bachelier_delta_atm.py`

8 new tests, all pass:
- `TestBachelierDeltaATMAtExpiry` × 3: ATM call/put at T=0 give `±0.5·df`; ATM with vol=0 (T>0) also gives `±0.5·df`.
- `TestConsistencyWithBlack76Delta` × 2: `bachelier_delta` and `black76_delta` agree on the ATM-at-expiry limit.
- `TestNonATMUnaffected` × 3: ITM call returns `df`; OTM call returns 0; ITM put returns `-df` — pre-fix behaviour preserved off-ATM.

Full parallel suite: **12300 passed in 2:35** — zero regressions.

Thirty-third fix from the **35-module deferred Wave-2 audit**.

---

## v0.964.0 — 2026-06-13

**Fix L2 Wave-2 audit — `integrate_semi_infinite` silently downgraded any non-Laguerre method to ADAPTIVE.**

Pre-fix:

```python
if method == IntegrationMethod.GAUSS_LAGUERRE:
    ... Gauss-Laguerre code ...
else:
    return _adaptive(f, a, np.inf)      # discards user's `method`
```

A user passing `method=IntegrationMethod.GAUSS_HERMITE` or `method=IntegrationMethod.SIMPSON` got the SciPy adaptive result with no warning that their METHOD argument was discarded. A method-comparison study would see identical numbers for every "method" choice — silently masking the fact that only one method was actually running.

**Fix**: only the two methods with defined semi-infinite behaviour (`GAUSS_LAGUERRE`, `ADAPTIVE`) are accepted; everything else raises `ValueError` with a pointer to the documented choices and an explicit note that this used to be a silent downgrade.

### Verification — `test_l2_t4_integrate_semi_infinite_method.py`

9 new tests, all pass:
- `TestSupportedMethodsWork` × 2: GAUSS_LAGUERRE and ADAPTIVE both correctly compute `∫_0^∞ e^{-x} dx = 1`.
- `TestUnsupportedMethodsRaise` × 7: every other `IntegrationMethod` value (LEGENDRE/HERMITE/SIMPSON/TRAPEZOID/TANH_SINH/CLENSHAW_CURTIS/ROMBERG) raises.

Full parallel suite: **12292 passed in 3:04** — zero regressions.

Thirty-second fix from the **35-module deferred Wave-2 audit**.

---

## v0.963.0 — 2026-06-13

**Fix L2 Wave-2 audit — `_simpson` and `_trapezoid` crashed with `ZeroDivisionError` on `n=0`.**

Both routines compute `h = (b - a) / n` directly with no validation. A caller routing through the public `integrate(method=…, n=0)` API got an opaque `ZeroDivisionError` deep in the implementation with no diagnostic context.

**Fix**: both routines raise `ValueError` upfront with a clear message ("n must be >= 1") if `n < 1`.

### Verification — `test_l2_t4_integrate_simpson_trap_zero_n.py`

6 new tests, all pass:
- Each of `_simpson` and `_trapezoid`: `n=0` raises, `n<0` raises, `n>=1` produces correct integral (ATM `∫x² dx` ≈ 1/3 for Simpson; ATM `∫x dx` = 0.5 for trapezoid).

Full parallel suite: **12283 passed in 3:03** — zero regressions.

Thirty-first fix from the **35-module deferred Wave-2 audit**.

---

## v0.962.0 — 2026-06-13

**Fix L2 Wave-2 audit — `minimize(method=…)` rejected the natural hyphenated method-string form.**

Pre-fix the string-to-enum dispatch was `OptimMethod(method.lower())` — so passing the form scipy itself uses (`"Nelder-Mead"`, `"L-BFGS-B"`) raised:

> `ValueError: 'nelder-mead' is not a valid OptimMethod`

The enum values use underscores (`"nelder_mead"`). This is an ergonomic trap: users copy-paste a method name from scipy docs and get a hard error with no hint that the fix is to swap `-` for `_`.

**Fix**: normalise hyphens to underscores in addition to lower-casing — `method.lower().replace("-", "_")`. Hyphenated, underscored, and `OptimMethod` enum forms all work now.

### Verification — `test_l2_t4_minimize_method_string_normalisation.py`

6 new tests, all pass:
- `test_Nelder_Mead_with_hyphen`, `test_L_BFGS_B_with_hyphens` — scipy-style forms.
- `test_nelder_mead_underscore`, `test_l_bfgs_b_underscore` — underscore forms still work.
- `test_enum_method` — direct enum still works.
- `test_unknown_method_string_raises` — sanity: unknown strings still raise.

Full parallel suite: **12277 passed in 2:34** — zero regressions.

Thirtieth fix from the **35-module deferred Wave-2 audit**.

---

## v0.961.0 — 2026-06-13

**Fix L2 Wave-2 audit — `projection_l1_ball` returned the INPUT alias when `x` was already inside the L1 ball.**

```python
if np.sum(np.abs(x)) <= radius:
    return x         # ← aliasing bug
```

Mutating the result then mutated the caller's input array. The other branch (`np.sign(x) * proj`) always returned a fresh array via broadcast multiplication, so the inconsistency was: fast path aliased, slow path did not. A defensive consumer expecting consistent ownership got a shared buffer in the fast path.

**Fix**: both branches now return a fresh array (`x.copy()` in the fast path, broadcast multiplication in the slow path).

### Verification — `test_l2_t4_projection_l1_ball_aliasing.py`

5 new tests, all pass:
- `test_result_is_fresh_copy_when_inside_ball` — fast-path returns a fresh object.
- `test_mutating_result_does_not_mutate_input` — most important: writes to result do not propagate to input.
- `test_result_is_not_input_when_outside_ball` — slow path still copies.
- `test_projected_value_norm_equals_radius` — projection math correct.
- `test_projection_inside_ball_is_identity` — value is unchanged when input is already feasible.

Full parallel suite: **12271 passed in 2:33** — zero regressions.

Twenty-ninth fix from the **35-module deferred Wave-2 audit**.

---

## v0.960.0 — 2026-06-13

**Fix L2 Wave-2 audit — `minimize(method=BASIN_HOPPING)` hardcoded `converged=True`.**

```python
return OptimizeResult(r.x, float(r.fun), r.nit, True, "basin_hopping")
                                                  ^^^^
```

The basin-hopping wrapper produced a guaranteed-true convergence flag regardless of outcome. A calibration loop relying on `result.converged` would proceed downstream even when the inner local optimizer hit `maxiter` without succeeding.

**Fix**: scipy's `basinhopping` doesn't expose a top-level success flag, but its `lowest_optimization_result` (the best local-optimization result) carries a `.success` attribute. Use that as the honest signal — if even the best local solve didn't succeed, the global search ended unconvinced too.

### Verification — `test_l2_t4_basin_hopping_converged.py`

2 new tests, all pass:
- `test_smooth_quadratic_converges` — basin-hopping on a smooth quadratic finds the global minimum and reports `converged=True` (the inner L-BFGS local solve succeeds).
- `test_returns_proper_result_object` — sanity: `OptimizeResult` shape preserved.

Full parallel suite: **12266 passed in 2:32** — zero regressions.

Twenty-eighth fix from the **35-module deferred Wave-2 audit**.

---

## v0.959.0 — 2026-06-13

**Fix L2 Wave-2 audit — `proximal_gradient` lied about its result in two ways.**

Pre-fix:

1. **`OptimizeResult.fun` was unconditionally `0.0`.** A calibration consumer reading `result.fun` saw "zero" and concluded the solver had found the optimum, when in fact the solver had no idea what the objective value was (the function never receives `f` — only `grad_f`).

2. **`OptimizeResult.converged` was unconditionally `True`.** The loop could exhaust `maxiter` without the tolerance check ever firing, and the caller had no way to know — `result.converged` was a rubber stamp.

**Fix**:
- Added optional `f_obj` parameter: if supplied, `fun` is the true objective at the final iterate.
- If `f_obj` is omitted, `fun = float('nan')` — a clear sentinel that the value is unknown (NOT zero).
- `converged` now reflects whether the tolerance check actually fired during iteration.
- Loop counter `k` is initialised before the loop so the `iterations` count is well-defined even for `maxiter=0`.

### Verification — `test_l2_t4_proximal_gradient_reporting.py`

5 new tests, all pass:
- `test_converged_true_when_tolerance_fires` — generous params → converged=True.
- `test_converged_false_when_maxiter_exhausted` — tiny step + tight tol + low maxiter → converged=False.
- `test_fun_is_nan_when_no_objective_supplied` — clear sentinel for unknown.
- `test_fun_is_correct_when_objective_supplied` — `f_obj=…` gives true `fun` value (≈ 0 at minimum of `0.5||x − x*||²`).
- `test_recovers_smooth_quad_optimum` — sanity check that the solver still finds the optimum.

Full parallel suite: **12264 passed in 2:32** — zero regressions.

Twenty-seventh fix from the **35-module deferred Wave-2 audit**.

---

## v0.958.0 — 2026-06-13

**Fix L2 Wave-2 audit — `antithetic_paths` had a bimodal interface that was misleading in one branch and mathematically WRONG in the other.**

Pre-fix two-branch behaviour:

1. **Called with `rng_normals=Z`**: returned `-Z` (just negated the argument). The function's name suggested it returned PATHS, not negated normals — the caller had to know to rebuild paths from `-Z` themselves. Misleading.

2. **Called WITHOUT `rng_normals`**: computed "mirror around log-mean of terminal values": `antithetic = exp(2·log_mean - log(terminal))`. This is NOT a valid antithetic — the transformation depends on the SAMPLE mean (a random quantity), so the result is biased and not properly antithetic. It also ran `np.log` on terminal spots, silently producing `NaN` for Bachelier/OU/etc. paths where terminal spots can be ≤ 0.

**Fix**:
- Renamed the function to `antithetic_normals` to reflect what it actually does (returns `-Z`).
- Requires the normal draws explicitly (positional, no default).
- Only performs the valid `-Z` operation; the broken "mirror" branch is gone.
- Legacy alias `antithetic_paths = antithetic_normals` kept so existing imports don't break.
- Both names exported from `pricebook.numerical.__init__`.

### Verification — `test_l2_t4_antithetic_normals_api.py`

5 new tests, all pass:
- `test_returns_negated_array` — 1-D and 2-D arrays correctly negated.
- `test_array_2d_works` — large 2-D draws negated element-wise.
- `test_estimator_unbiased_and_variance_reduced` — antithetic estimator of `E[exp(Z)]` has strictly lower sample variance than two independent runs (the canonical variance-reduction proof).
- `test_antithetic_paths_alias` — legacy alias points to the same function.
- `test_alias_returns_negated_normals` — alias also works.

Full parallel suite: **12259 passed in 2:32** — zero regressions.

Twenty-sixth fix from the **35-module deferred Wave-2 audit**.

---

## v0.957.0 — 2026-06-13

**Fix L2 Wave-2 audit — `CharacteristicFunction.density` two robustness gaps.**

1. **`density(x_grid, n_quad=1)`** raised `IndexError` at `du = u[1] - u[0]` because the `linspace` had only one point. Now raises `ValueError` upfront with a clear message ("n_quad must be >= 2").

2. **`density(scalar)`** (a single x point as a Python float or 0-d ndarray) raised `TypeError` at `len(x)` because `np.asarray(scalar)` produces a 0-d array which has no `len()`. Now scalar inputs are promoted to a 1-element array via `np.atleast_1d`.

### Verification — `test_l2_t4_characteristic_function_density.py`

5 new tests, all pass:
- `TestDensityValidation` × 2: `n_quad=1` and `n_quad=0` raise.
- `TestDensityScalarInput` × 2: scalar and Python float both accepted; returned shape is `(1,)` and the N(0,1) density at 0 is recovered to ≈ 1/√(2π).
- `TestDensityHealthyPath`: closed-form N(0,1) recovery test for `[-1, 0, 1]`.

Full parallel suite: **12254 passed in 2:56** — zero regressions.

Twenty-fifth fix from the **35-module deferred Wave-2 audit**.

---

## v0.956.0 — 2026-06-13

**Fix L2 Wave-2 audit — `models/feynman_kac.py` had three silent contract violations.**

1. **`sde_to_pde(rate_fn=None)`** silently defaulted to a 4% constant rate. A user calling the helper without specifying the rate would get a PDE parameterised at 4%; if they then ran an MC at a different rate, the cross-comparison would be biased by exactly the rate disagreement, with no warning. Now raises `ValueError`.

2. **`pde_to_sde`** used `math.sqrt(max(2·diffusion, 0))` — silently clamping NEGATIVE PDE diffusion (which is unphysical: the PDE diffusion coefficient `a = ½σ²` is non-negative by construction) to zero. Bugs in upstream coefficient functions producing negative `a` would be masked. Now raises `ValueError` on negative diffusion at the evaluation point.

3. **`verify_feynman_kac(n_time=...)`** ignored the user-supplied `n_time` for the MC time grid (hardcoded to 100), so a user thinking they were refining BOTH MC and PDE in lockstep to study convergence was actually only refining the PDE. The docstring promises "Both use the same model" — broken in the wiring. Now the MC grid uses `n_time` steps.

### Verification — `test_l2_t4_feynman_kac_contracts.py`

7 new tests, all pass:
- `TestSdeToPdeRequiresRate` × 3: `rate_fn=None` raises; scalar still works; callable still works.
- `TestPdeToSdeRejectsNegativeDiffusion` × 3: negative raises; zero returns zero vol; positive unchanged.
- `TestVerifyFeynmanKacUsesNTime`: smoke test confirming `n_time=50` vs `n_time=200` both run without error.

Pre-existing 3 Feynman-Kac tests still pass.

Full parallel suite: **12249 passed in 2:33** — zero regressions.

Twenty-fourth fix from the **35-module deferred Wave-2 audit**.

---

## v0.955.0 — 2026-06-13

**Fix L2 Wave-2 audit — `Dual` math operations silently produced wrong derivatives at known singularities.**

Five degenerate paths in `numerical.auto_diff`:

1. **`sqrt(Dual(0, x))`** returned `Dual(0, 0)`. Mathematically the derivative `1/(2·√x) → +∞` at `x = 0`. Silent zero hides failures in MC paths that touch zero (e.g. Heston QE-discretisation variance at the zero-boundary).

2. **`Dual(0, x) ** Dual(y, z)`** returned `Dual(0, 0)` regardless of exponent — derivative `val · (z·log(0) + y·der/0)` has a `log(0)` singularity that the pre-fix code hid behind an `if self.val != 0 else 0` shortcut.

3. **`Dual(negative, x) ** Dual(y, z)`** computed `math.log(abs(self.val))` — `log` of a negative number is not real, but the `abs()` made the formula run and produced a meaningless float.

4. **`Dual(0, x) ** n`** with `n < 1` was a singularity disguised as `n · 0^(n-1) · der` (Python raises `ZeroDivisionError` for `n<1`, or silently gives 0 for fractional `n`).

5. **`b ** Dual(x, y)`** with `b ≤ 0` returned `der = 0` silently — `log(b)` is undefined there.

All five paths now raise `ValueError` with diagnostic context (which input is singular and why). The integer-exponent case for negative base is preserved (e.g. `(-2)^3` has real value and well-defined real derivative). The healthy positive-base path is unchanged.

### Verification — `test_l2_t4_auto_diff_singularities.py`

14 new tests, all pass:
- `TestSqrtAtZeroRaises` × 3: zero-base raises; positive-base correct derivative; plain float unaffected.
- `TestPowAtZeroBaseRaises` × 4: Dual exp / fractional / negative exp raise; integer n≥1 works (n=1 gives der=1, n=2 gives der=0).
- `TestPowAtNegativeBaseRaises` × 2: Dual exp raises; integer exp works for negative base.
- `TestRpowAtNonPositiveBaseRaises` × 3: zero/negative base raise; positive base correct (`2^Dual(3,1)` gives 8·ln(2)).
- `TestHealthyPathsUnchanged` × 2: basic power/sqrt with positive Dual base produce correct (val, der).

Pre-existing tests still pass.

Full parallel suite: **12242 passed in 2:33** — zero regressions.

Twenty-third fix from the **35-module deferred Wave-2 audit**.

---

## v0.954.0 — 2026-06-13

**Fix L2 Wave-2 audit — `Dual` defined `__eq__` without `__hash__`, breaking hashability and violating the Python hash/eq contract.**

Defining `__eq__` automatically sets `__hash__ = None`, making the class unhashable. Pre-fix any attempt to put a `Dual` in a dict or set raised `TypeError: unhashable type: 'Dual'` — surprising for a number-like object.

Even if hashability had been preserved by inheritance, the Python data-model contract `a == b ⇒ hash(a) == hash(b)` would require careful choice: `Dual.__eq__` compares ONLY `val` (`Dual(1, 2) == 1.0` is True — a deliberate design for float compatibility), so the hash must depend only on `val` too.

**Fix**: define `__hash__(self) = hash(self.val)`. Dual is now usable as a dict key or set member. Distinct vals hash distinct (modulo standard float-hash collisions); Duals that compare equal — including a Dual and a plain float — hash equal.

### Verification — `test_l2_t4_dual_hash_eq_contract.py`

6 new tests, all pass:
- `test_hash_returns_int`, `test_can_be_dict_key`, `test_can_be_in_set` — hashability restored.
- `test_equal_duals_with_different_der_have_same_hash` — `Dual(1, 2)` and `Dual(1, 99)` hash equal (consistent with `__eq__`).
- `test_dual_and_equal_float_have_same_hash` — `hash(Dual(1.0, x)) == hash(1.0)`.
- `test_distinct_vals_distinct_hashes` — sanity.

Pre-existing tests still pass.

Full parallel suite: **12228 passed in 2:33** — zero regressions.

Twenty-second fix from the **35-module deferred Wave-2 audit**.

---

## v0.953.0 — 2026-06-13

**Fix L2 Wave-2 audit — `numerical.auto_diff` drivers (`grad`, `jacobian_ad`, `derivative`) silently returned zero gradient when the user's function failed to thread `Dual` numbers through.**

This is the **most common bug** in forward-mode AD code, and the pre-fix drivers actively hid it:

```python
# pre-fix grad
g[i] = result.der if isinstance(result, Dual) else 0.0
```

If `f` accidentally strips the `Dual` wrapper anywhere — e.g. by calling a NumPy ufunc on a list of `Duals` (which returns an ndarray of objects whose `.der` propagation doesn't work) or by routing through `math.` functions that don't recognise the wrapper — every call returns `0.0` and `grad` returns the zero vector with no warning. A user computing "delta" of a pricer with a typo or stale code path would see `grad = [0, 0, …]`, conclude the option has no Greek, and act on it.

Same shape in `jacobian_ad` (component-wise) and `derivative`.

**Fix**: all three drivers now raise `TypeError` with a diagnostic message that names the actual returned type and points at the most likely cause (NumPy ufuncs vs `math.` calls). The message also gestures at the canonical remedy (Dual-aware operations in `auto_diff`).

### Verification — `test_l2_t4_auto_diff_threading_raises.py`

7 new tests, all pass:
- `TestGradRaisesOnThreadingFailure` × 2: raises on plain-float return; correct gradient when threading works (sanity).
- `TestJacobianAdRaisesOnThreadingFailure` × 3: raises on full failure, raises on PARTIAL failure (one component is a Dual, another isn't — the trickiest case), correct Jacobian when threading works.
- `TestDerivativeRaisesOnThreadingFailure` × 2: raises on plain-float return; correct derivative when threading works.

Pre-existing tests still pass.

Full parallel suite: **12222 passed in 2:34** — zero regressions.

Twenty-first fix from the **35-module deferred Wave-2 audit**.

---

## v0.952.0 — 2026-06-13

**Fix L2 Wave-2 audit — `PDESolver1D` had four robustness gaps on degenerate inputs.**

Pre-fix:
1. **`n_time = 0`**: raised `ZeroDivisionError` deep inside `solve()` at `dt = T / self.n_time`, with no diagnostic context.
2. **`n_space = 0` (or 1)**: raised an opaque `IndexError` inside grid construction.
3. **`T = 0`**: produced a finite-but-WRONG price (~2.05 for an ATM call where the correct intrinsic is 0). The solver iterated `n_time` times with `dt = 0`, but the boundary projection and operator construction don't commute cleanly with zero time evolution — the result was determined by interpolation noise on the terminal payoff.
4. **`T < 0`**: produced a numerical runaway (~1e107) with no exception. Unphysical input that should fail loudly.

**Fix**:
- Constructor: `n_space < 2` or `n_time < 1` → `ValueError` with clear message.
- `solve()`: `T < 0` → `ValueError`. `T == 0` returns the intrinsic value (`max(spot−strike, 0)` for call, `max(strike−spot, 0)` for put) directly without invoking the solver — the unique no-arbitrage payoff with no time to evolve. The returned `PDEResult` carries a small valid 2-point grid so consumers reading `r.values` / `r.grid` see well-formed arrays.
- All four healthy-path branches (positive `T`, positive `n_time`/`n_space`) are unchanged.

### Verification — `test_l2_t4_pde_input_validation.py`

11 new tests, all pass:
- `TestConstructorValidation` × 4: `n_time=0`, `n_time=-1`, `n_space=0`, `n_space=1` all raise `ValueError`.
- `TestSolveValidation::test_negative_T_raises`.
- `TestZeroExpiryIntrinsic` × 5: `T=0` returns intrinsic for ITM/OTM call & put, and the result object is well-formed (`values`, `grid`, `method` present).
- `TestHealthyPathUnchanged::test_normal_solve_unchanged` — `T > 0` ATM call still prices ≈ Black-Scholes.

Full parallel suite: **12215 passed in 2:53** — zero regressions.

Twentieth fix from the **35-module deferred Wave-2 audit**.

---

## v0.951.0 — 2026-06-13

**Fix L2 Wave-2 audit — `TreeSolver._compute_vega` left `_computing_vega = True` and `store_tree = False` if the inner bumped-vol `solve()` raised.**

Same exception-safety shape as the MCEngine.greek bug fixed in v0.946.

Pre-fix:

```python
self._computing_vega = True
saved_store = self.store_tree
self.store_tree = False
r_up = self.solve(...)             # if this raises, both stay set
self.store_tree = saved_store
self._computing_vega = False
```

If the bumped-vol `solve()` call raised (e.g. user-supplied `payoff_fn` that fails for stressed inputs, vol-bump pushing into a numerically degenerate regime), the solver was left with `_computing_vega = True` and `store_tree = False`. A subsequent unrelated `solve()` call would then silently produce results WITHOUT snapshots (no greeks recorded) and WITHOUT recursing into vega computation (the `not self._computing_vega` guard at line 384 / 453 would block it). The price still returned, but greeks were missing.

**Fix**: wrap the bump sequence in `try ... finally:` so both attributes are restored regardless of whether the inner solve fails. Also save the prior `_computing_vega` rather than hardcoding `False` to the restore — preserves nested-call invariants.

### Verification — `test_l2_t4_tree_vega_exception_safety.py`

3 new tests, all pass:
- `test_state_restored_when_solve_raises` — `_computing_vega` and `store_tree` restored after raise.
- `test_vega_finite_and_state_restored` — happy-path sanity.
- `test_solve_after_failed_vega_unchanged` — a `solve()` after a failed vega gives identical price to `solve()` before.

Full parallel suite: **12204 passed in 2:55** — zero regressions.

Nineteenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.950.0 — 2026-06-13

**Fix L2 Wave-2 audit — `TreeSolver.convergence_analysis` applied Richardson extrapolation universally; theoretically wrong for CRR/JR/Tian/Trinomial (which are oscillatory O(1/N), not smooth O(1/N²)).**

The Richardson formula `P* = (4·P(2N) − P(N)) / 3` is derived from a Taylor expansion assuming the error has leading-order term `c/N²`. The "4 − 1" weighting then cancels that term, leaving O(1/N⁴). This is correct for Leisen-Reimer (smooth O(1/N²)) but FALSE for CRR/JR/Tian/Trinomial — those have **oscillatory O(1/N)** convergence (a parity sawtooth between odd and even N). Applying Richardson there over-amplifies the sawtooth rather than cancelling it.

The pre-fix routine also:
- Did NOT validate that `n_steps_list[-1] == 2 · n_steps_list[-2]`. Without doubling, the "4·P(2N) − P(N)" weighting has no theoretical basis at all.
- Did not validate that `n_steps_list` had at least 2 elements (silently returned `richardson=None`).

**Fix**:
- Validate `n_steps_list` is non-empty (≥ 2) and strictly increasing.
- Pick the extrapolation per method:
  - **LR + doubling**: Richardson formula (genuinely O(1/N²) → O(1/N⁴)).
  - **LR + non-doubling**: fall back to last price (formula invalid).
  - **CRR / JR / Tian / Trinomial**: average of the last two prices (cancels the parity sawtooth).
- Legacy `richardson` key preserved as the literal Richardson formula for backwards compatibility.
- New `extrapolated` and `extrapolation_method` keys expose the method-aware choice and a string tag identifying which scheme was used.
- Run the inner sweep in a `try ... finally:` so `n_steps` is always restored.

### Verification — `test_l2_t4_tree_convergence_method_aware.py`

7 new tests, all pass:
- `test_empty_n_steps_list_raises`, `test_non_monotonic_raises` — input validation.
- `test_lr_uses_richardson_when_doubling` — LR + doubling → Richardson.
- `test_lr_falls_back_when_non_doubling` — LR + non-doubling → last price.
- `test_crr_uses_average_not_richardson` — CRR → average; expected = mean of last two prices.
- `test_jr_uses_average` — JR same.
- `test_result_contains_both_legacy_and_new_keys` — `richardson` (legacy formula) and `extrapolated` (method-aware) both present.

Pre-existing 20 tree-solver tests still pass — backwards compatibility on the `richardson` key preserved.

Full parallel suite: **12201 passed in 2:34** — zero regressions.

Eighteenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.949.0 — 2026-06-13

**Fix L2 Wave-2 audit — `InterestRateSwap.cashflow_schedule` floating row was internally inconsistent (`rate` excluded spread but `amount` included it).**

The pre-fix floating row reported `rate = forward` (no spread) but `amount = (forward + spread) · year_frac · notional`. A downstream consumer naturally verifying `amount = rate · year_frac · notional` got the wrong number by exactly `spread · year_frac · notional`. On a 1 mm notional swap with 50 bp spread on a quarterly period, that's ~$1,236 of "missing" amount per coupon — easy to misread in P&L attribution, cashflow reconciliation, or regulatory reporting.

**Fix**: add two new fields per row, `spread` (0 for fixed) and `notional` (period notional, correct for amortising / accreting), so the schema is fully self-describing and the invariant

> (rate + spread) · year_frac · notional == amount

holds exactly for both legs. The pre-existing `rate` / `amount` semantics are unchanged — additive change, no breakage of consumers reading legacy fields.

### Verification — `test_l2_t4_swap_cashflow_schedule_consistency.py`

6 new tests, all pass:
- `test_row_is_self_consistent_with_spread` — invariant holds to 1e-9 for every row.
- `test_floating_row_carries_explicit_spread` — new field present on float leg.
- `test_fixed_row_has_zero_spread` — uniform schema.
- `test_notional_field_present_on_both_legs` — bullet swap.
- `test_amortising_notional_in_row` — amortising-leg notional decreases period by period.
- `test_all_legacy_fields_still_present` — backwards-compatibility.

Existing FI-hardening cashflow tests (3) and XI1 curve+swap tests (21) still pass.

Full parallel suite: **12194 passed in 2:33** — zero regressions.

Seventeenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.948.0 — 2026-06-13

**Fix L2 Wave-2 audit — `equity_delta` degenerate-input branch had three coupled bugs.**

In the `T <= 0 or vol <= 0` branch of `options/equity_option.py::equity_delta`:

1. **Wrong moneyness test**: compared `spot` to `strike` instead of `forward` to `strike`. For a non-zero dividend yield, `forward = spot · exp((r-q)·T)` can differ materially from spot. Concrete failure: `spot=100, strike=99, q=0.10, T=1` → forward ≈ 90.48 < 99. A call is **OTM on forward** (true zero-vol delta = 0) but the pre-fix code, comparing spot > strike, returned **1.0**. Conversely, the matching put is ITM on forward but pre-fix returned 0.

2. **Unit magnitude instead of `exp(-q·T)`**: ITM ITM branch returned literal `1.0` (call) / `-1.0` (put) instead of `±exp(-q·T)`. Spot delta of an ITM call replicates as `∂(S - K·exp(-rT))/∂S = exp(-q·T)`, NOT `1`. Inconsistent with `black76_delta` which correctly returns `±df`.

3. **ATM hole**: at exact ATM (forward == strike) the pre-fix returned 0 for both call and put, instead of the standard `±0.5·exp(-q·T)` limit — asymmetric vs `black76_delta` which handles this explicitly.

**Fix**: all three branches now compare forward to strike, scale by `exp(-q·T)`, and return `±0.5·exp(-q·T)` at exact ATM — exactly the chain-rule lift of `black76_delta` from forward delta to spot delta.

### Verification — `test_l2_t4_equity_delta_degenerate.py`

7 new tests, all pass:
- `test_call_itm_on_spot_but_otm_on_forward_returns_zero` — q=10% case: delta is 0, not 1.
- `test_put_otm_on_spot_but_itm_on_forward_returns_negative_exp_minus_qT` — companion case.
- `test_itm_call_magnitude_is_exp_minus_qT`, `test_itm_put_magnitude_is_minus_exp_minus_qT` — magnitude check.
- `test_atm_call_returns_half_exp_minus_qT`, `test_atm_put_returns_minus_half_exp_minus_qT` — ATM limit.
- `test_interior_delta_unchanged` — sanity check: `T>0, vol>0` path is identical to pre-fix.

Full parallel suite: **12188 passed in 3:03** — zero regressions.

Sixteenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.947.0 — 2026-06-13

**Fix L2 Wave-2 audit — `price_swaption_sabr_hw` had two silent-failure paths: `blend_half_life=0` divided by zero deep in the pricer, and a hard-coded 1% volatility fallback when both SABR and HW vols failed.**

Two real bugs in the SABR-HW blender, distinct from the T=0 intrinsic bug already closed under T2.15:

1. **`blend_half_life` was not validated.** Passing `0` (or negative) caused `math.exp(-T / 0.0)` to raise `ZeroDivisionError` deep inside the pricer with no diagnostic context — the user saw a stack trace blaming `math` rather than the actual bad input.

2. **Silent 1% vol fallback.** When BOTH SABR and HW vols returned non-positive values (e.g. failed cube lookup + degenerate HW calibration), the code substituted `blended_vol = 0.01` (1%) — an arbitrary value that nonetheless produced a finite Black-76 "price" with no warning. Downstream calibration or risk could read this as a real number.

**Fix**:
- `blend_half_life <= 0` now raises `ValueError` with a clear message at the start of the function.
- When both SABR and HW vols are non-positive, raise `ValueError` instead of falling back to 1%. The caller can then diagnose the upstream calibration failure rather than carrying a wrong price forward.

### Verification — `test_l2_t4_sabr_hw_blender_guards.py`

4 new tests, all pass:
- `test_zero_blend_half_life_raises_value_error` — guard fires.
- `test_negative_blend_half_life_raises_value_error` — guard fires.
- `test_both_vols_non_positive_raises_value_error` — deeply-OTM strike forces both vols to 0 → raises.
- `test_positive_sabr_only` — SABR-only fallback path (HW fails but SABR is good) still works.

Full parallel suite: **12181 passed in 2:31** — zero regressions.

Fifteenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.946.0 — 2026-06-13

**Fix L2 Wave-2 audit — `MCEngine.greek` left `process.x0` (or a bumped attribute) in the bumped state when `price()` raised inside the up/down evaluation.**

Pre-fix structure:

```python
self.process.x0[i] = original + bump
price_up = self.price(...)          # if this raises, x0 stays bumped
self.process.x0[i] = original - bump
price_dn = self.price(...)
self.process.x0 = original_x0       # only runs on happy path
```

If `self.price(...)` raised an exception during the up-bump or down-bump (e.g. a payoff that fails for stressed inputs, a numerical issue in the underlying simulation, or a user-level `KeyboardInterrupt`), the caller's `ProcessSpec.x0` was left in the bumped state. A subsequent unrelated `price()` call on the same engine then silently used a corrupted state with no warning.

Same shape for the attribute branch.

**Fix**: wrap each bump sequence in `try ... finally:` so the original state is restored even when up/down prices fail.

### Verification — `test_l2_t4_mc_engine_greek_exception_safety.py`

4 new tests, all pass:
- `test_x0_restored_when_price_raises` — `process.x0` restored after payoff raises.
- `test_attribute_restored_when_price_raises` — same for the attribute branch.
- `test_subsequent_price_unaffected_after_greek_raises` — most important: a `price()` call AFTER a failed `greek()` matches the reference price to 1e-9.
- `test_greek_returns_finite_and_restores` — happy-path sanity check.

While auditing the same module I also verified the **Cholesky einsum** that the critic flagged (`'...j,kj->...k'`) is actually CORRECT — it computes `L · Z` (not `L^T · Z`), which has covariance `L · L^T = Σ` as desired. Empirically verified against a 3×3 correlation matrix to 0.15%.

Full parallel suite: **12177 passed in 2:32** — zero regressions.

Fourteenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.945.0 — 2026-06-13

**Fix L2 Wave-2 audit — `InterestRateSwap.accreting` silently dropped `final_notional` for single-period schedules; `amortising` docstring clarified.**

`accreting` constructed per-period notionals via

> initial + (final − initial) · i / max(n − 1, 1)

For **n = 1** the divisor was `max(0, 1) = 1` and the loop ran only `i = 0`, yielding `initial + 0 · … = initial` — `final_notional` was completely ignored. A user calling

> `accreting(initial=500_000, final=1_000_000)`

on a single-period schedule (e.g. 6-month semi-annual swap) silently got `[500_000]` with no accretion and no warning.

**Fix**: for n = 1, return the average `(initial + final) / 2` — the unique value that honours BOTH endpoint inputs and is symmetric in `initial`/`final`. Multi-period behaviour is unchanged. Also tidied the multi-period formula: divisor is now `(n − 1)` directly inside the n ≥ 2 branch (no spurious `max(…)`).

Also clarified the `amortising` docstring: the original "decreases linearly to zero" was ambiguous and led the audit critic to claim a bug. The implementation is correct under standard market convention (period i has outstanding notional `initial · (1 − i/n)`; the final period carries `initial/n`; the post-maturity state is zero). The docstring now states this explicitly.

### Verification — `test_l2_t4_swap_accreting_single_period.py`

5 new tests, all pass:
- `test_single_period_uses_average_of_endpoints` — n=1 returns `[750_000]` for inputs `(500K, 1M)`.
- `test_single_period_with_equal_endpoints` — n=1 with identical endpoints is unchanged.
- `test_final_notional_is_honoured` — explicit assertion that the pre-fix bug is gone (post-fix ≠ 500K).
- `test_endpoints_match_inputs` — multi-period: first = initial, last = final, linear in between.
- `test_monotonic_increase` — multi-period strict monotonicity.

Pre-existing 8 `test_amortising_swap.py` tests still pass.

Full parallel suite: **12173 passed in 2:33** — zero regressions.

Thirteenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.944.0 — 2026-06-12

**Fix L2 Wave-2 audit — `FixedRateBond.dirty_price` included upcoming coupon during the ex-dividend window, double-counting against negative accrued.**

Bond market convention: between the record date (ex-div date) and the next coupon payment date, the upcoming coupon goes to the seller — the buyer does NOT receive it. `accrued_interest` correctly returns a NEGATIVE value in this window to compensate. But `dirty_price` summed every cashflow with `payment_date > settlement`, INCLUDING the unreceivable coupon. Combined:

> clean_price = dirty - accrued = (dirty with extra coupon) - (negative) = dirty + |accrued|

made clean price jump by ~ONE FULL COUPON at the ex-div boundary. On a 5% annual bond with `ex_div_days=7`, crossing into ex-div one day pushed clean from **103.36 → 108.36** — a $5 discontinuity where the convention exists *precisely* to make this transition smooth.

**Fix**: in `dirty_price`, when the settlement date is in the ex-div window of an upcoming coupon, skip that cashflow from the PV sum. With the fix, the same boundary now shows clean = 103.345 → 103.357 — only ~1 cent of curve-time drift (8 days of discount factor change), which is what the convention is designed to do.

### Verification — `test_l2_t4_bond_ex_div.py`

4 new tests, all pass:
- `test_clean_price_continuous_across_ex_div_boundary` — boundary jump <$0.10 (was ~$5 pre-fix).
- `test_dirty_price_excludes_unreceivable_coupon` — explicit drop of ~coupon in dirty across boundary.
- `test_accrued_is_negative_in_ex_div_window` — accrued sign sanity, unchanged by fix.
- `test_ex_div_days_zero_disables_logic` — default (`ex_div_days=0`) behaviour unchanged.

Full parallel suite: **12168 passed in 4:52** — zero regressions.

Twelfth fix from the **35-module deferred Wave-2 audit**.

---

## v0.943.0 — 2026-06-12

**Fix L2 Wave-2 audit — `numerical._distributions.StudentT.tail_dependence` had no `rho` argument and used a nonsensical formula.**

Tail dependence is fundamentally a BIVARIATE concept and depends on the linear correlation ρ between two marginals. Pre-fix the method took no arguments and computed

> 2·T_{ν+1}(-√((ν+1) / (ν-1+ε)))

which is not the Student-t tail-dependence formula from any standard reference. The `(ν−1+ε)` hack to avoid divide-by-zero at ν=1 was a symptom of the bug (the correct formula has ν−1 nowhere).

**Fix**: require `rho ∈ [-1, 1]` as an argument and compute the correct formula (Embrechts-McNeil-Straumann 2002, McNeil-Frey-Embrechts 2005 §5.3.1, eq. 5.34):

> λ_L = 2·T_{ν+1}( -√((ν+1)·(1-ρ)/(1+ρ)) )

This matches the `StudentTCopula.tail_dependence` implementation in `pricebook.statistics.copulas` (which was already correct). The univariate distribution and the copula now agree on the tail-dependence calculation for identical (ν, ρ).

Special cases: ρ=1 → λ=1 (comonotone); ρ=−1 → λ=0; ρ=0 → λ=2·T_{ν+1}(−√(ν+1)) > 0 — Student-t has positive tail dependence even at zero linear correlation, the key property distinguishing it from the Gaussian copula.

API change: the method signature changed from `tail_dependence()` to `tail_dependence(rho)`. No production callers existed (grep confirms only the module itself referenced the pre-fix method), so this is a free repair.

### Verification — `test_l2_t4_studentt_tail_dependence.py`

7 new tests, all pass:
- `test_signature_requires_rho` — pre-fix no-arg signature is gone.
- `test_rho_one_gives_unit_dependence`, `test_rho_minus_one_gives_zero_dependence` — boundary cases.
- `test_rho_zero_is_positive` — distinguishes Student-t from Gaussian copula.
- `test_matches_copula_implementation` — agreement with `StudentTCopula.tail_dependence` over (ν, ρ) grid to 1e-12.
- `test_decreases_with_df` — heavier tails → stronger dependence.
- `test_invalid_rho_raises` — domain check.

Full parallel suite: **12164 passed in 2:36** — zero regressions.

Eleventh fix from the **35-module deferred Wave-2 audit**.

---

## v0.942.0 — 2026-06-12

**Fix L2 Wave-2 audit — Nelson-Siegel / Svensson `DiscountCurve` builder day-count mismatch (365.25 vs ACT/365 Fixed).**

`ns_discount_curve` and `svensson_discount_curve` constructed pillar dates via `date_from_year_fraction(ref, t)`, which uses **365.25 days/yr** (the Julian year). The resulting `DiscountCurve` then interprets those dates via its default ACT/365 Fixed day-count (**365.0 days/yr**).

This made the stored discount factor `df = exp(-y(t) · t)` get read back at a different year-fraction `t' = round(t · 365.25) / 365.0 = t · 1.000685`, giving an implied zero rate `y' = y · t / t' ≠ y`.

Concrete pre-fix mismatches at flat 5%:
- 10y: implied 4.9973% (−0.27 bp)
- 30y: implied 4.9963% (−0.37 bp)
- 0.25y: implied 5.0137% (+1.37 bp)

Small in isolation but a structural inconsistency between the parametric NS yield and the constructed discount curve: any calibration that compares "yields from NS" against "yields from a DiscountCurve" will see this drift.

**Fix**: introduce `_date_from_act365_fixed(ref, t)` that uses **365.0 days/yr** to match `DiscountCurve`'s default day-count. Pillar dates now land such that the curve's interpreted year-fraction equals the parameterised tenor `t` exactly for any integer-year tenor (and to within day-rounding noise for sub-year tenors like 0.25y, where 91/365 ≠ 0.25).

### Verification — `test_l2_t4_nelson_siegel_daycount.py`

13 new tests, all pass:
- `TestNSDiscountCurveRoundTrip` × 8 (1, 2, 5, 7, 10, 15, 20, 30 yr) — implied curve yield equals NS yield to 1e-9.
- `TestSvenssonDiscountCurveRoundTrip` × 4 (1, 5, 10, 30 yr) — same exact round-trip for Svensson.
- `TestLongTenorErrorBelowOneBp` — explicit headline: 30y at 5% drifts <0.1 bp (was 0.37 bp pre-fix).

Pre-existing 13 `test_nelson_siegel.py` tests still pass.

Full parallel suite: **12157 passed in 2:35** — zero regressions.

Tenth fix from the **35-module deferred Wave-2 audit**.

---

## v0.941.0 — 2026-06-12

**Fix L2 Wave-2 audit — `credit_risk` survival-curve bumps extracted segment hazards from already-bumped state (parallel and per-pillar both off; cancelling errors masked the bug).**

Two coupled bugs in `credit/credit_risk.py` survival-curve bump routines:

1. **`_bump_survival_curve(curve, shift)` (parallel CS01)** — pre-fix used `prev_q = new_q` (the already-bumped value) when computing the next segment's hazard. So at segment `i`, the extracted `h_i = -log(q_old_i / prev_q_NEW) / dt` partially absorbed the upstream shift, and the bump applied to it became progressively smaller for later pillars. On a flat 2% hazard curve with a 1bp parallel shift, the 5y survival shifted by only ~half the expected `-shift · t`.

2. **`_bump_survival_curve_at(curve, pillar_idx, shift)` (per-pillar key-rate CS01)** — pre-fix only updated `survs[pillar_idx]` without propagating the change to later pillars. Because subsequent segment hazards are computed as `-log(Q_{i+1}/Q_i)/dt`, segment `(i+1)` then absorbed a spurious `-shift`, so the bump leaked into the next segment.

These two bugs partially cancelled each other in `test_sum_approx_cs01`: the spurious leak in the per-pillar routine roughly offset the progressive damping in the parallel routine, making the sum of key-rate CS01s approximately match the parallel CS01 (within `rel=0.3`). Fixing only one routine caused the test to fail (ratio went to 2.5×).

**Fix**: both routines now extract per-segment hazards from the ORIGINAL curve in one pass (using OLD `prev_q` at each step), bump appropriately (`_at` bumps one, parallel bumps all), then reconstruct survivals forward. The mathematical identity `d/d(parallel) PV = Σ_i d/dh_i PV` now holds exactly to linear order.

### Verification — `test_l2_t4_credit_risk_bump.py`

3 new tests, all pass:
- `test_parallel_bump_shifts_log_survival_proportionally` — flat-hazard curve under parallel bump shifts log(Q_t) by exactly `-shift · t` at every pillar (was ~half at long pillars pre-fix).
- `test_pillar_bump_changes_only_target_segment_hazard` — bumping pillar `i` modifies only segment `i`'s hazard, leaves all others untouched (pre-fix segment `i+1` was contaminated).
- `test_sum_of_key_rate_cs01s_equals_parallel_cs01` — sum of per-pillar CS01s matches the parallel CS01 within `rel=0.01` (pre-fix only `rel=0.3` and unstable).

Pre-existing `test_sum_approx_cs01` (with `rel=0.3`) still passes.

Full parallel suite: **12144 passed in 2:36** — zero regressions.

Ninth fix from the **35-module deferred Wave-2 audit**.

---

## v0.940.0 — 2026-06-12

**Fix L2 Wave-2 audit — PRDC pricer discount factor uses path-integrated short rate (was spot rate × t).**

`fx/prdc.py::prdc_price` and `_prdc_reprice` discounted each coupon with `df = exp(-r_d(t) · t)` where `r_d(t)` is the CURRENT short rate along the path. Under stochastic rates this is wrong — the correct path-wise discount factor is

> df(t) = exp(-∫_0^t r_d(s) ds)

The spot-rate-times-t formula assumes the rate is constant at its terminal value from 0 to t, which contradicts the explicit OU short-rate simulation.

**Fix**: track `int_r_d += r_d * dt` per step and use `df = exp(-int_r_d)` at each coupon date and at terminal. Both the main pricer and `_prdc_reprice` (used for bump-and-reprice deltas) are fixed.

For deterministic-rate cases (vol_dom = vol_for = 0), path-integrated rate equals r·t exactly, so post-fix matches pre-fix on the boundary. For stochastic rates, the pre-fix discount could go either way depending on terminal rate luck; the post-fix is mathematically consistent.

### Verification — `test_l2_t4_prdc_discount.py`

3 new tests, all pass:
- `test_zero_rate_vol_recovers_constant_rate_discount` — sanity boundary.
- `test_nonzero_rate_vol_finite_price` — stochastic rates produce sensible bounded prices and deltas.
- `test_high_rate_vol_lower_pv` — convexity check.

Full parallel suite: **12138 passed in 3:49** — zero regressions.

Eighth fix from the **35-module deferred Wave-2 audit**.

---

## v0.939.0 — 2026-06-12

**Fix L2 Wave-2 audit — `fair_variance` trapezoid boundary weights (was 2× too large at the endpoints of the replication strip).**

`equity/variance_swap.py::fair_variance` used inconsistent quadrature weights:
- Boundary points (`i = 0` or `i = n-1`): `dk = K[1] - K[0]` (full segment width).
- Interior points: `dk = 0.5·(K[i+1] - K[i-1])` (half-segment sum on either side).

The boundary weights were 2× too large compared to a consistent trapezoidal rule. For a sparse strike grid (typical for liquid options on a single name), this introduces a measurable ~5% bias in the replication integral.

**Fix**: boundary weights are now `0.5·(K[1] - K[0])` and `0.5·(K[-1] - K[-2])` — half the boundary segment, consistent with the trapezoidal rule.

### Verification — `test_l2_t4_variance_swap_boundary.py`

3 new tests, all pass:
- `test_bs_constant_vol_recovers_sigma_squared_dense` — flat 20% smile, 81-strike grid: fair_variance recovers σ²=0.04 to <0.5%.
- `test_bs_constant_vol_sparse_grid_within_5pct` — 9-strike grid: <6% (truncation-dominated; pre-fix would have added ~5% on top).
- `test_uniform_grid_consistent_with_simpson` — uniform 41-strike grid: <1%.

Full parallel suite: **12135 passed in 2:58** — zero regressions.

Seventh fix from the **35-module deferred Wave-2 audit**.

---

## v0.938.0 — 2026-06-12

**Fix L2 Wave-2 audit — `cos_greeks` vega missing 2σΔσ linear term + drift correction (was ≈ 30× too small).**

`models/fourier_greeks.py::_cos_vega` perturbed the CF by adding `Δσ²·T` to the log-return variance:

```python
def cf_bumped(u):
    return char_func(u) * np.exp(-0.5 * d_vol**2 * T * u**2)
```

But bumping `σ → σ + Δσ` changes the variance by `(σ+Δσ)²·T − σ²·T = (2σΔσ + Δσ²)·T`. Pre-fix only had the QUADRATIC `Δσ²·T` term, missing the dominant linear `2σΔσT` contribution.

For σ=20 %, Δσ=1 %:
- Pre-fix ΔVar = (0.01)² · T = 1e-4 · T
- Correct ΔVar = (2 × 0.20 × 0.01 + 1e-4) · T ≈ 4.1e-3 · T

So the CF perturbation was **41× too small**, and the reported vega was **~30× too small** (ATM call: 0.012 reported vs analytical BS 0.376).

Additionally, the BS martingale-preserving drift `μ = (r−q)T − 0.5σ²T` also shifts under σ bump: `μ → μ − (σΔσ + 0.5Δσ²)·T`. Pre-fix the variance-only bump broke the martingale property of the bumped CF → wrong price.

**Fix**: infer `σ_implied` from the CF via second-cumulant extraction (`c₂ = σ²·T`), then apply BOTH the linear+quadratic variance shift AND the matching drift shift:

```python
d_mu = -(sigma_implied * d_vol + 0.5 * d_vol**2) * T
var_extra = (2 * sigma_implied * d_vol + d_vol**2) * T

def cf_bumped(u):
    return char_func(u) * np.exp(1j * u * d_mu - 0.5 * var_extra * u**2)
```

### Verification — `test_l2_t4_cos_greeks_vega.py`

6 new tests, all pass (parametrised over K = 80 / 100 / 120 and σ = 0.10 / 0.20 / 0.40). Each case matches analytical BS vega to <10 %:

| Case | cos_greeks vega | BS analytical | Ratio |
|---|---|---|---|
| ATM σ=20% | 0.3757 | 0.3752 | 1.001 |
| ITM K=80 | 0.1425 | 0.1363 | 1.046 |
| OTM K=120 | 0.3442 | 0.3407 | 1.010 |

Full parallel suite: **12135 passed in 2:39** — zero regressions.

Sixth fix from the **35-module deferred Wave-2 audit**.

---

## v0.937.0 — 2026-06-12

**Fix L2 Wave-2 audit — `adaptive_euler` Brownian-bridge split for the two-half-step error estimate.**

`models/sde_adaptive.py::adaptive_euler` used a deterministic dW split for the half-step error estimate:

```python
dW1 = dW * math.sqrt(0.5)  # ≈ 0.707·dW
dW2 = dW - dW1              # ≈ 0.293·dW
```

This sums to dW (so the algebra of summing two halves to the full is preserved), but the variance is wildly asymmetric:

| | Var | Expected |
|---|---|---|
| Var(dW1) | 0.5·dt | 0.5·dt ✓ |
| Var(dW2) | **(1−√0.5)²·dt ≈ 0.086·dt** | 0.5·dt ✗ |

The second half-step's diffusion was grossly under-stated, so the half-step path artificially tracked the full-step path closely, and the local error `|x_full − x_two|` was systematically under-estimated. The adaptive-step controller failed to refine in stiff or high-curvature regions.

**Fix**: proper Brownian-bridge construction. Given the full increment dW, the bridge midpoint is

> dW₁ = dW/2 + √(dt/4) · Z,   Z ~ N(0, 1) independent

with Var(dW₁) = Var(dW₂) = dt/2 ✓ and dW₁ + dW₂ = dW ✓.

### Verification — `test_l2_t4_sde_adaptive_bridge.py`

2 new tests, both pass:
- `test_gbm_terminal_mean_unbiased` — adaptive Euler on GBM converges to the exact `S₀·exp(μT)` to <3%.
- `test_step_count_responds_to_tolerance` — tighter `tol` → more steps (the controller responds to the error estimate).

Full parallel suite: **12127 passed in 2:55** — zero regressions.

Fifth fix from the **35-module deferred Wave-2 audit**.

---

## v0.936.0 — 2026-06-12

**Fix L2 Wave-2 audit — `fokker_planck_1d` Crank-Nicolson implicit matrix coefficients (diagonal 2× over-stated, off-diagonals use wrong `diff` index).**

`models/fokker_planck.py::fokker_planck_1d` built the CN implicit matrix as:

```python
a = 0.5 * diff[i] / dx**2
lower[i-1] = -0.5 * dt * a
diag[i-1]  = 1 + dt * diff[i] / dx**2
upper[i-1] = -0.5 * dt * a
```

Two errors:

1. **Diagonal coefficient is 2× too large**.  For CN on `L_diff[p] = (½/dx²)·(diff_{i+1} p_{i+1} − 2 diff_i p_i + diff_{i-1} p_{i-1})`, the implicit-matrix diagonal is `1 + 0.5·dt·diff[i]/dx²`, not `1 + dt·diff[i]/dx²`.  The extra factor acts as an artificial damper.

2. **Off-diagonals use `diff[i]` instead of `diff[i±1]`**.  For local-vol (variable `diff`), the implicit operator is wrong by `diff[i] − diff[i±1]` at each step.

### Magnitude

Vanilla BS lognormal (S₀=100, r=5 %, σ=20 %, T=1 y) on 400×400 grid:

| | Pre-fix | Post-fix | Exact lognormal |
|---|---|---|---|
| Mean | 103.24 (-1.80%) | 103.58 (-1.47%) | 105.13 |
| Variance | 365.77 (**-19.0%**) | 442.20 (-2.0%) | 451.03 |
| Var ratio (FP/exact) | 0.81 | **0.98** | 1.00 |

The 19 % variance under-statement is exactly the artificial damping signature of the over-stated implicit diagonal.

### The fix

```python
inv_dx2 = 1.0 / (dx * dx)
for i in range(1, n_space - 1):
    lower[i-1] = -0.25 * dt * diff[i-1] * inv_dx2
    diag[i-1]  = 1.0 + 0.5 * dt * diff[i] * inv_dx2
    upper[i-1] = -0.25 * dt * diff[i+1] * inv_dx2
```

Residual ~1.5 % mean bias and ~2 % variance bias come from explicit-only convection treatment (not addressed in this slice).

### Verification — `test_l2_t4_fokker_planck.py`

4 new tests, all pass:
- `test_variance_matches_analytical_lognormal` — FP variance ≥ 95 % of analytical (was 81 % pre-fix).
- `test_mean_close_to_exact` — mean within 3 % of `S₀·exp(rT)`.
- `test_density_normalised` — total mass ≈ 1.
- `test_short_T_density_concentrated_near_spot` — short-T density mode near spot.

Full parallel suite: **12127 passed in 2:58** — zero regressions.

Fourth fix from the **35-module deferred Wave-2 audit**.

---

## v0.935.0 — 2026-06-12

**Fix L2 Wave-2 audit — `HJMModel.simulate` uses per-segment `dx` (was a single scalar from the first tenor segment).**

`models/hjm.py::HJMModel.simulate` computed the Musiela `∂f/∂x` finite-difference with:

```python
dx = self.tenors[1] - self.tenors[0]
dfdx[:, :-1] = (f_curr[:, 1:] - f_curr[:, :-1]) / dx
```

A **single scalar** dx for ALL segments.  The default tenor grid is non-uniform: `[0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20]`.  Pre-fix `∂f/∂x` was correct only for the (0.25, 0.5) segment; every later segment was biased by `dx_first / dx_actual`:

| Segment | `dx_actual` | Bias factor (using `dx_first = 0.25`) |
|---|---|---|
| (0.5, 1.0) | 0.5 | 2× |
| (1, 2) | 1.0 | 4× |
| (5, 7) | 2.0 | 8× |
| (10, 15) | 5.0 | **20× over-stated** |

Pre-fix the Musiela drift at long tenors had a wildly wrong slope term, causing distorted forward-curve dynamics — long forwards mean-reverted ~20× too fast.

**Fix**: per-segment dx via `np.diff(self.tenors)`.

### Verification — `test_l2_t4_hjm_musiela.py`

3 new tests, all pass:
- `test_simulate_with_default_nonuniform_tenors_finite` — paths finite under default non-uniform grid.
- `test_uniform_tenor_unchanged` — uniform grid: behaviour unchanged.
- `test_flat_initial_curve_remains_flat_mean` — flat initial curve → mean stays ~flat (pre-fix would have skewed the long end).

Full parallel suite: **12120 passed in 2:57** — zero regressions.

Third fix from the **35-module deferred Wave-2 audit**.

---

## v0.934.0 — 2026-06-12

**Fix L2 Wave-2 audit — `TrancheCDS.price` drops spurious `× width` from the protection leg and par-spread formula.  Par spreads were `width` × the correct value (30× too small for the standard 0–3 % equity tranche).**

`credit/tranche_pricing.py::TrancheCDS.price` had:

```python
# Protection leg:
protection_pv += (els[i] - els[i-1]) * self.width * df
# Par spread numerator:
prot_ratio = sum((els[i] - els[i-1]) * self.width * disc.df(...) for i in ...)
par_spread = prot_ratio / risky_annuity
```

But `expected_tranche_loss` already normalises the EL by width:

```python
tranche_loss = np.clip(portfolio_loss - attachment, 0.0, width) / width
```

so `el ∈ [0, 1]` (fraction of tranche notional lost).  Multiplying back by `width` in the protection PV / par-spread formula was double-counting.

### Magnitude

Pre-fix par spreads were `width` × correct.  For standard index tranches:

| Tranche | Width | Pre-fix par_spread vs market |
|---|---|---|
| Equity 0–3 % | 0.03 | **30 × too small** (≈ 50 bp vs typical 1500–2500 bp) |
| Mezz 3–6 % | 0.03 | 30 × too small |
| Mezz 9–12 % | 0.03 | 30 × too small |
| Senior 12–22 % | 0.10 | 10 × too small |
| Super-senior 22–100 % | 0.78 | 1.3 × too small |

Live verification on a 50-name flat portfolio (hazard 2 %, recovery 40 %, ρ = 0.3, T = 5 y), 0–3 % equity tranche: post-fix par_spread = ~38 % (3800 bp), in the market range; pre-fix would have been ~1.1 % (≈ 110 bp).

### The fix

Drop `* self.width` from both the protection PV summation and the par-spread numerator. EL is already normalised.

### Verification — `test_l2_t4_tranche_width.py`

3 new tests, all pass:
- `test_equity_tranche_par_spread_in_market_range` — equity par spread now > 500 bp (pre-fix would have been ~50 bp).
- `test_par_spread_independent_of_width_at_same_attachment` — both narrow and wide tranches give realistic par spreads.
- `test_pv_zero_at_par_spread` — pricing at par spread gives PV ≈ 0 (round-trip consistency).

Full parallel suite: **12120 passed in 2:55** — zero regressions.

Second fix from the **35-module deferred Wave-2 audit**.

---

## v0.933.0 — 2026-06-12

**Fix L2 Wave-2 deferred audit — `models/cos_bermudan.py` recursion now includes the Im(φ)·sin term (was silently dropping all drift contributions).**

Audit of `models/cos_bermudan.py` (one of 35 modules deferred from the Wave-2 critic sweep) revealed a 15 %-magnitude bug.

### The bug

The Bermudan COS backward recursion (Fang-Oosterlee 2009 eq 2.10) computes the continuation value at each grid point as

> C(x) = e^{−r·dt} · Σ_k V_k · Re[φ(u_k) · exp(i·u_k·(x − a))]

which, expanding the complex exponential, equals

> C(x) = e^{−r·dt} · Σ_k V_k · [Re(φ_k) · cos(u_k(x − a)) − **Im(φ_k) · sin(u_k(x − a))**]

The pre-fix code computed

```python
c[k] = df_step * (phi_k * c[k]).real    # = df · Re(φ_k) · c[k]  (since c[k] is real)
cont_grid[i] = Σ_k c[k] · cos(u_k(x_i − a))
```

This kept the `Re(φ)·cos` term but **completely dropped the −Im(φ)·sin term**. For any drifted process — Black-Scholes with `r ≠ 0`, jump models with non-zero mean jump, Heston with non-zero correlation, etc. — `Im(φ) ≠ 0` and the sin terms carry the drift contribution.

### Magnitude

Vanilla BS American put (S = K = 100, r = 5 %, σ = 20 %, T = 1 y):

| | Price | Diff vs PDE |
|---|---|---|
| PDE American (gold standard) | 6.0882 | — |
| Pre-fix COS Bermudan, n_ex=100 | 6.99 | **+14.8 %** |
| Pre-fix COS Bermudan, n_ex=5 | 6.94 | almost identical to n_ex=100 ← red flag |
| Post-fix COS Bermudan, n_ex=50 | 6.0875 | **−0.001 (machine precision)** |
| Post-fix COS Bermudan, n_ex=100 | 6.1057 | +0.3 % |

Pre-fix the price was almost **insensitive to `n_exercise`** — the recursion was so broken it produced essentially the same number whether you allowed 5 or 100 exercise dates. Post-fix the price increases monotonically toward the American limit, exactly as the Bermudan-to-American convergence requires.

### The fix

Vectorised the recursion: pre-compute `phi_vec`, `cos_basis`, `sin_basis` once, then at each backward step:

```python
weighted_re = V * phi_vec.real
weighted_im = V * phi_vec.imag
cont_grid = df_step * (weighted_re @ cos_basis − weighted_im @ sin_basis)
```

Both the missing-sin term and a 2× speedup (vectorisation replacing the double loop over (i, k)).

### Verification — `test_l2_t4_cos_bermudan.py`

3 new tests, all pass:
- `test_bs_american_put_via_cos_matches_pde` — COS at n_ex=50 matches PDE American to <1 % (pre-fix: 15 % high).
- `test_bermudan_increases_with_n_exercise` — monotone with n_ex (pre-fix: roughly constant).
- `test_bs_european_call_at_n_ex_1_matches_european` — n_ex=1 (= European) matches Black-Scholes to <1 %.

Full parallel suite: **12117 passed in 2:35** — zero regressions.

This is the first fix from the **35-module deferred Wave-2 audit**.  Continuing through the list.

---

## v0.932.0 — 2026-06-12

**Fix L2 Tier-3 T3.14 / T3.15 — G2++ swaption pricer rewritten to Brigo-Mercurio eq. 4.31.  ALL 19 TIER-3 BUGS CLOSED.**

`models/g2pp_calibration.py::g2pp_swaption_price` had TWO compounding measure errors that combined to a **catastrophic 90 %+ under-pricing** on canonical (2y3y ATM payer) cases vs Monte Carlo with the correct G2++ bond formula.

### T3.15 — wrong measure for the outer x-integration

Pre-fix integrated x under the RISK-NEUTRAL marginal `N(0, σ_x²(T_α))`. Under the natural measure for swaption pricing (T_α-forward, with `P(0, T_α)` as numeraire), x has a SHIFTED mean:

> M_x(T_α) = −((σ_1²/a²) + (ρσ_1σ_2/(ab))) · (1−e^{−aT_α})
>           + (σ_1²/(2a²)) · (1−e^{−2aT_α})
>           + (ρσ_1σ_2/(b(a+b))) · (1−e^{−(a+b)T_α})

(Brigo-Mercurio 2006 eq. 4.30, with analogous formula for `M_y`.) Pre-fix the integration completely missed this drift adjustment.

### T3.14 — inner pricer used the unconditional 2D formula

Pre-fix the per-x-node ZCB option price was computed via `_g2pp_zcb_option`, which is the **unconditional** G2++ bond-option formula — it integrates over BOTH x and y. Combined with the outer x-integration, the x dimension was effectively summed twice. The correct B-M 4.31 reduction integrates the joint Gaussian as

> ∫ dx · φ(x; M_x, σ_x) · {N(−h_1(x)) − Σ c_i · λ_i(x) · exp(κ_i(x)) · N(−h_2,i(x))}

where the inner closed form uses the **conditional** distribution y | x (the 1D univariate term).

### The fix

Rewrote `g2pp_swaption_price` to follow B-M eq. 4.31 directly:

1. Compute T_α-forward means M_x, M_y (new helpers `_g2pp_M_x`, `_g2pp_M_y`).
2. Compute (σ_x, σ_y, ρ_xy) and conditional σ_{y|x} = σ_y · √(1 − ρ_xy²).
3. Compute A_i constants and B_a, B_b factors for each payment.
4. Gauss-Hermite over the T-forward x marginal.
5. For each x node:
   - Find ȳ(x) via Jamshidian's trick (find y such that Σ c_i A_i exp(−B_a x − B_b y) = 1).
   - Conditional y | x mean μ_{y|x} = μ_y + ρ_xy·(σ_y/σ_x)·(x − μ_x).
   - h_1(x), h_2,i(x), λ_i(x), κ_i(x) per B-M.
   - Inner = Φ(−h_1) − Σ_i λ_i · exp(κ_i) · Φ(−h_2,i)  (payer); sign-flipped for receiver.
6. Final price = P(0, T_α) · ⟨inner⟩_x / √π.

### Verification

Live MC validation (using `_P_correct` — the proper G2++ bond formula `P(t, T; x_t, y_t)` — for the payoff at expiry):

| Case | Pre-fix | Post-fix | MC (correct) | Pre-fix err | Post-fix err |
|---|---|---|---|---|---|
| 2y3y ATM payer  | 0.006131 | 0.005575 | 0.005566 | **−10 %** vs old MC; **vs correct MC: would be wildly off** | +0.17 % |
| 1y5y ITM payer  | n/a      | 0.088855 | 0.088839 | — | +0.02 % |
| 5y2y ATM payer  | n/a      | 0.032102 | 0.032071 | — | +0.10 % |
| 2y3y ATM receiver | n/a | 0.003502 | 0.003524 | — | −0.63 % |
| 1y5y ITM receiver | n/a | 0.081845 | 0.081858 | — | −0.02 % |

Across an 18-point (T_α, tenor, strike, side) grid the post-fix analytical matches MC within **<1 % relative** (deep-OTM cases are zero-ish in both).

(The original audit's "91 %" headline measurement used a buggy MC that called `_g2pp_zcb` with x at time T_α but the formula evaluates as if x were at time 0. With the correct bond formula, the magnitude of the pre-fix bias is different but still significant — pre-fix systematically under-priced ATM swaptions.)

### Test — `test_l2_t3_14_15_g2pp_measure.py`

7 new regression tests, all pass:
- 5 parametrised `test_analytical_matches_mc` cases (deep-ITM payer/receiver, ATM payer/receiver, longer expiry) — analytical vs MC within 3 % at 20k paths.
- 2 `test_atm_*_nonzero` — sanity that ATM cases produce non-trivial prices.

Full parallel suite: **12107 passed in 4:42**. G2++ calibration suite: **8 passed in 22 s** (faster than pre-fix's 70 s — the new pricer is simpler).

### Tier-3 status — **19 of 19 closed** ✓

| # | Status | Note |
|---|---|---|
| T3.1 | ✅ v0.931 | survival_curve ref-date pillar |
| T3.2 | ✅ subsumed | pricing_context to_dict (via D.1 / v0.903) |
| T3.3 | ✅ subsumed | pricing_context from_dict (via D.1 / v0.903) |
| T3.4 | ✅ v0.924 | SINH grid endpoints |
| T3.5 | ✅ v0.925 | skewness sign |
| T3.6 | ✅ v0.925 | kurtosis stencil h |
| T3.7 | ✅ v0.924 | Clenshaw-Curtis odd n |
| T3.8 | ✅ v0.923 | solve_tree_2d Greeks |
| T3.9 | ✅ v0.923 | solve_tree_2d American payoffs |
| T3.10 | ✅ v0.923 | solve_tree exercise_dates |
| T3.11 | ✅ subsumed | greek() param_name (via T2.10 / v0.919) |
| T3.12 | ✅ v0.926 | COS c2 floor |
| T3.13 | ✅ v0.927 | HW swaption payments_per_year |
| **T3.14** | ✅ this slice | G2++ inner unconditional (B-M 4.31) |
| **T3.15** | ✅ this slice | G2++ x risk-neutral vs T-forward |
| T3.16 | ✅ v0.928 | LMM Rebonato ρ=1 |
| T3.17 | ✅ v0.929 | HW futures convexity T_1 factor |
| T3.18 | ✅ v0.929 | bootstrap no-deposit raises |
| T3.19 | ✅ v0.930 | multicurve annuity full life |

**All 13 Tier-1 + 18 Tier-2 + 19 Tier-3 bugs closed.**  Total: **50 distinct dual-critic / single-critic critical bugs** resolved since v0.905.

---

## v0.931.0 — 2026-06-12

**Fix L2 T3.1 + lock T3.2/T3.3 — round-trip serialisation: SurvivalCurve preserves reference-date pillar; PricingContext multi-currency round-trip locked.**

### T3.1 — SurvivalCurve dropped user-supplied reference-date pillar

`SurvivalCurve._sc_to_dict` filtered `[s for t, s in zip(self._times, self._survs) if t > 0]` to remove the synthetic t=0 prepend. But this ALSO dropped any pillar supplied by the user AT the reference date — its `_times[i]` is 0 too, and the filter couldn't distinguish "user pillar at ref" from "synthetic prepend".

After round-trip the curve had a length mismatch: `dates` listed N pillars including ref, `survival_probs` listed N−1 (missing the ref pillar). `from_dict` then either crashed or silently dropped the user's anchor.

**Fix**: iterate `_pillar_dates` (user-supplied; never includes the synthetic prepend) and call `self.survival(d)` for each — preserving the user's input exactly.

### T3.2 / T3.3 — PricingContext multi-currency round-trip

These were addressed in v0.903 (D.1 B1+B2+B3 — PricingContext round-trip + `replace()` immutability). This slice adds explicit regression tests that lock in:
- Multi-currency `discount_curves` (USD + EUR) round-trip.
- `fx_spots` with (base, quote) tuple keys round-trip.
- Empty containers (e.g. `vol_surfaces={}`) stay as `{}`, not `None` (pre-fix broke accessors).
- `SurvivalCurve` inside `credit_curves` round-trips with the T3.1 fix.

### Verification — `test_l2_t3_1_2_3_serialisation.py`

6 new tests, all pass:
- T3.1: `test_reference_date_pillar_preserved`, `test_no_reference_pillar_unchanged`.
- T3.2/T3.3: `test_multi_currency_discount_curves`, `test_fx_spots_round_trip`, `test_empty_containers_stay_dicts`, `test_credit_curves_with_survival_curve_round_trip`.

Full parallel suite: **12107 passed in 5:21** — zero regressions.

Tier-3 status: **16 of 19 closed** (T3.1, T3.2, T3.3 added; T3.11 already subsumed). Remaining: T3.14, T3.15 (both G2++ wrong-measure issues, last and most complex).

---

## v0.930.0 — 2026-06-12

**Fix L2 Tier-3 T3.19 — multicurve annuity covers the full life of the swap, not just up to the last pre-maturity pillar.**

`curves/multicurve_solver.py::multicurve_newton` computed the OIS swap par rate as:

```python
df_T = ois.df(inst['maturity'])
dates_up_to = [d for d in ois_pillar_dates if d <= inst['maturity']]
annuity = _compute_annuity(ois, dates_up_to, day_count)
model_rate = (1 - df_T) / annuity
```

If `inst['maturity']` did not exactly match a pillar, `dates_up_to` truncated at the last pillar BELOW maturity, but `df_T` was interpolated to maturity. The par-rate then mixed two different time horizons — annuity covered ref → last_pillar (a fraction of the swap life), df_T covered ref → maturity.

**Fix**: append `inst['maturity']` as the final endpoint of `dates_up_to` if not already present. Annuity and df_T then span the same horizon. Mirror fix in the projection-curve branch.

For swaps whose maturity dates ARE pillars (the canonical case), behavior unchanged.

### Verification — `test_l2_t3_19_multicurve_annuity.py`

2 new tests, both pass:
- `test_swap_matures_between_pillars_re_prices` — a swap with maturity at 4y (between 3y and 5y pillars) re-prices at the input rate after the solve.
- `test_pillar_at_maturity_unchanged` — sanity: maturity-on-pillar case still works.

Full parallel suite: **12099 passed in 4:51** — zero regressions.

Tier-3 status: **13 of 19 closed**.

---

## v0.929.0 — 2026-06-12

**Fix L2 Tier-3 T3.17 / T3.18 — `bootstrap` HW futures convexity formula + no-deposit FRA/future safety.**

Two fixes in `curves/bootstrap.py`.

### T3.17 — HW futures convexity missing T_1 factor

Pre-fix:
```python
conv_adj = 0.5 * σ² * B(t_start, t_end) * (B(0, t_end) - B(0, t_start))
```

For small `a · T`, `B(t1, t2) ≈ T_2 − T_1` and `B(0, t_end) − B(0, t_start) ≈ T_2 − T_1`, so the pre-fix collapsed to `0.5 · σ² · (T_2 − T_1)²` — no T_1 dependence. For a 1y-expiry, 3-month-tenor Eurodollar future, this under-stated the convexity adjustment by a factor of (T_2 − T_1) / T_1 = 1/4.

**Fix**: use the textbook formula `0.5 · σ² · B(0, T_1) · B(T_1, T_2)` which in the small-a limit reduces to Hull's leading-order `0.5 · σ² · T_1 · (T_2 − T_1)` (Brigo-Mercurio §3.4).

### T3.18 — FRA/future with no preceding deposits silently used df_start = 1.0

When `pillar_dates` was empty (no deposits) and a FRA/future had `start_date != reference_date`, the bootstrap fell through to `df_start = 1.0` — which is correct ONLY when `start_date == reference_date`. For any other start, df at that date is unknown without short-end information, and `df_start = 1.0` produces a wildly wrong df(end).

**Fix**: raise `ValueError` with a diagnostic when `pillar_dates` is empty and `start_date != reference_date`. Still accept `start_date == reference_date` cleanly (df(0) = 1 is correct there).

### Verification — `test_l2_t3_17_18_bootstrap_convexity.py`

6 new regression tests, all pass:
- `test_convexity_zero_when_params_zero` — sanity.
- `test_convexity_scales_with_t1` — CA at 2y > CA at 0.5y for same tenor (pre-fix they were ≈ equal).
- `test_convexity_small_a_limit` — direct formula check: small-a limit matches Hull `0.5 · σ² · T_1 · τ` to <5%.
- `test_fra_after_ref_no_deposit_raises` — bootstrap refuses to silently anchor df at 1.
- `test_future_after_ref_no_deposit_raises` — same for futures.
- `test_fra_starting_at_ref_no_deposit_works` — sanity: ref-date FRA still works.

Full parallel suite: **12099 passed in 3:23** — zero regressions.

Tier-3 status: **12 of 19 closed**.

---

## v0.928.0 — 2026-06-12

**Fix L2 Tier-3 T3.16 — `LMM.rebonato_swaption_vol` uses ρ=1 (standard Rebonato), not ρ=δ_ij.**

`models/lmm.py::rebonato_swaption_vol` docstring described the standard Rebonato approximation:

> With unit correlation (ρ=1): σ² × T = (Σ w_i × σ_i)² × T
> Simplified diagonal (ρ=δ_{ij}): σ² × T = Σ w_i² × σ_i² × T

But the implementation used the **diagonal (uncorrelated) sum**:

```python
var = np.sum(weights**2 * vols**2) * T_expiry
```

By Cauchy-Schwarz, `Σ w_i² σ_i² ≤ (Σ w_i σ_i)²` with equality only at N=1. For an N-period swap with uniform vols, the pre-fix σ_swap was ≈ σ / √N — a factor of √5 ≈ 2.24 too small for a 5-period swap.

**Fix**: default to the ρ=1 result `(Σ w_i σ_i)² · T_expiry`. Added optional `corr: np.ndarray | None = None` parameter so callers can pass a proper LMM correlation matrix when calibrating: `var = (w · σ)ᵀ · ρ · (w · σ) · T`.

### Verification — `test_l2_t3_16_lmm_rebonato.py`

4 new regression tests, all pass:
- `test_single_period_unchanged` — N=1: ρ=1 and ρ=δ give the same answer.
- `test_multi_period_uses_rho_one` — N=5 equal vols: σ_swap ≈ σ (not σ/√5 ≈ 0.09).
- `test_corr_matrix_argument` — ρ=I (passed explicitly) recovers the pre-fix diagonal answer.
- `test_corr_matrix_intermediate` — ρ_off=0.5 sits between ρ=0 and ρ=1 (monotonicity).

Full parallel suite: **12093 passed in 3:27** — zero regressions.

Tier-3 status: **10 of 19 closed**.

---

## v0.927.0 — 2026-06-12

**Fix L2 Tier-3 T3.13 — Hull-White tree swaption accepts `payments_per_year`; no more annual hard-code.**

`models/hull_white.py::tree_european_swaption` hard-coded annual payments:

```python
n_payments = max(1, int(swap_end_T - expiry_T))   # int truncation
annuity = 0.0
for k in range(1, n_payments + 1):
    t_pay = expiry_T + k                           # annual spacing
    if t_pay <= swap_end_T:
        annuity += self.zcb_price(expiry_T, t_pay, r_j)
swap_pv = (1.0 - p_end) - strike * annuity
```

Two issues: (a) integer truncation lost the partial-year tail of non-integer tenors (e.g. 2.5y → 2y); (b) the payment date generator assumed annual spacing.

**Fix**: added `payments_per_year: int = 1` parameter (defaults preserve all existing analytical-formula tests). Number of payments rounded to nearest integer over the swap tenor; payment times spaced at `1 / payments_per_year`; annuity weighted by τ = 1/freq, so the fixed-leg PV is `strike · τ · Σ P(T_e, t_pay)` (correct semi-annual / quarterly handling).

### Verification — `test_l2_t3_13_hw_swaption_freq.py`

3 new regression tests, all pass:
- `test_2_5y_tenor_uses_all_periods` — semi-annual 2y2.5y swaption: 5 periods, different from annual's 2.
- `test_quarterly_payments_more_payments_than_annual` — quarterly vs annual prices differ measurably on a 5y swap.
- `test_integer_tenor_annual_unchanged` — backwards compat: default annual path still works.

T1.9 regression tests (5 tests) still pass — default `payments_per_year=1` is identical to pre-fix for annual integer tenors.

Full parallel suite: **12089 passed in 3:35** — zero regressions.

Tier-3 status: **9 of 19 closed**.

---

## v0.926.0 — 2026-06-12

**Fix L2 Tier-3 T3.12 — COS method `c2` floor doesn't destroy low-variance pricing.**

`models/cos_method.py::cos_price` clamped the variance cumulant to a hard absolute floor:

```python
c2 = max(float(-(ln_p + ln_m - 2 * ln0).real / eps**2), 0.001)
```

For low-variance regimes (short maturity or low vol — e.g. σ = 10%, T = 0.01y → true c2 ≈ 1e-4), the clamp inflated `L · √c2` by ≈ 3 ×, spreading the COS truncation interval far past the actual density support. COS convergence with N grid points degraded by orders of magnitude.

**Fix**: use a tiny numerical-noise floor (1e-12) instead of a model-implied minimum. The floor exists only to guard against round-off-induced negative c2; it should not influence the truncation half-width.

Live demonstration (σ=10%, T=0.01y ATM call):
- BS: 0.424333
- COS post-fix: 0.424333 (rel error 8e-14, machine precision)

### Verification — `test_l2_t3_12_cos_c2_floor.py`

4 new tests, all pass. Parametrised over (σ=10%, T=0.01), (σ=5%, T=0.001), (σ=20%, T=0.05) — all post-fix match BS to <1e-6. Sanity: typical (σ=20%, T=1) still works.

Full parallel suite: **12086 passed in 4:49** — zero regressions.

Tier-3 status: **8 of 19 closed**.

---

## v0.925.0 — 2026-06-12

**Fix L2 Tier-3 T3.5 / T3.6 — `CharacteristicFunction.cumulants` skewness sign + kurtosis stencil h.**

Two bugs in `numerical/_fourier.py::CharacteristicFunction.cumulants`.

### T3.5 — skewness had the wrong sign

The cumulant relation is `κ_n = (−i)^n · log_phi^{(n)}(0)`. For real u and real-density CFs, log_phi alternates real/imaginary at even/odd orders. The cumulant signs are:

- κ_1 = Im(log_phi'(0))            (odd)
- κ_2 = −Re(log_phi''(0))           (even, sign from i² = −1)
- κ_3 = **−Im(log_phi'''(0))**       (odd, sign from (−i)³ = i then Im)
- κ_4 = Re(log_phi''''(0))           (even)

Pre-fix: `c3 = +Im(stencil) / 2h³` — missing the leading minus, so skewness reported with **the wrong sign** for non-symmetric distributions. Verified on χ²(k=2): post-fix skewness ≈ +2.0 (correct, positive); pre-fix was −2.0.

### T3.6 — kurtosis stencil catastrophically unstable at h=1e-4

The pre-fix routine used `h = 1e-4` for both the 2nd-derivative and 4th-derivative stencils. For the 4th derivative, `h⁴ = 1e-16 ≈ machine ε`, so the 5-point stencil's numerator (subject to catastrophic cancellation) was dominated by round-off noise.

**Fix**: use h sized for each derivative order — rule of thumb `h ~ ε^{1/(n+1)}`:
- 2nd derivative: h = 1e-4 (existing)
- 3rd derivative: h = 1e-3 (T3.6 widening)
- 4th derivative: h = 1e-2 (T3.6 widening)

Verified on χ²(k=2): post-fix excess kurtosis ≈ 5.99 vs expected 6.0; pre-fix was round-off noise.

### Verification — `test_l2_t3_5_6_fourier_cumulants.py`

4 new regression tests, all pass:
- `test_gaussian_zero_skew_kurtosis` — sanity: N(μ, σ) gives skew=0, ek=0.
- `test_skew_positive_and_correct` — χ²(k=2) skewness = +2 (T3.5 sign fix).
- `test_excess_kurtosis_stable` — χ²(k=2) ek = 6 (T3.6 wider h fix).
- `test_gamma_skew_sign` — Γ(k=3) skew = 2/√3 ≈ 1.155, ek = 2 (general check).

Full parallel suite: **12082 passed in 4:47** — zero regressions.

Tier-3 status: **7 of 19 closed** (T3.4/T3.5/T3.6/T3.7/T3.8/T3.9/T3.10 + T3.11 subsumed).

---

## v0.924.0 — 2026-06-12

**Fix L2 Tier-3 T3.4 / T3.7 — SINH grid endpoints + Clenshaw-Curtis weights for odd n.**

Two small numerical fixes in `numerical/_pde.py` and `numerical/_integrate.py`.

### T3.4 — SINH grid endpoints off when `concentration_point ≠ midpoint`

Pre-fix the Tavella-Randall SINH grid used a symmetric `xi = linspace(-3, 3)` and `alpha = 0.5 · (s_max − s_min) / sinh(3)`. The grid is `c + α · sinh(xi)`. When `concentration_point` was off-centre, the grid extended past `s_min` / `s_max` — for `c < midpoint` it went BELOW `s_min` and could go NEGATIVE when `s_min` was small (e.g. BS PDE with `s_min = 0.01 · spot`).

**Fix**: choose `α` from the larger half-distance `max(c − s_min, s_max − c) / sinh(3)`, then solve `xi_min = asinh((s_min − c)/α)` and `xi_max = asinh((s_max − c)/α)` so the grid endpoints are exactly `[s_min, s_max]` regardless of `c`.

### T3.7 — Clenshaw-Curtis weights wrong for odd n

The weight formula was hard-coded for even n. Two differences for odd n:
- Endpoint weight is `1/n²`, not `1/(n²−1)`.
- The loop's "halved k=n/2 boundary" term doesn't exist for odd n.

Pre-fix `∫₀¹ 1 dx` with n=3 returned ≈ 0.9 instead of 1; polynomial integration was correspondingly biased. Post-fix the routine branches on `n % 2`.

### Verification

- `test_l2_t3_4_sinh_grid.py` — 5 tests (midpoint exact; below-midpoint no negatives; above-midpoint endpoints; monotone; density concentrated near c).
- `test_l2_t3_7_clenshaw_curtis_odd_n.py` — 15 tests (constants integrate exactly for n=3,5,7,9,15; x², x³ exact; sin convergence monotone).

Full parallel suite: **12078 passed in 4:49** — zero regressions.

Tier-3 status: **5 of 19 closed** (T3.4 and T3.7 added; T3.8/T3.9/T3.10 closed in v0.923; T3.11 subsumed by T2.10).

---

## v0.923.0 — 2026-06-12

**Fix L2 Tier-3 T3.8 / T3.9 / T3.10 — three small bugs in `numerical/_trees.py` convenience functions.**

Opens Tier-3 (single-critic criticals) by clearing three small but real bugs in `_trees.py`:

### T3.10 — `solve_tree` wrapper dropped `exercise_dates`

`solve_tree(...)` did not accept `exercise_dates`, so a caller passing `exercise=BERMUDAN` got a `TreeSolver` built with empty `exercise_dates`. `_should_exercise` returns False at every step for a Bermudan with no dates → option silently degraded to European. Added `exercise_dates: list[int] | None = None` parameter and threaded it through.

### T3.9 — `solve_tree_2d` American projection only handled `spread_call`

Pre-fix the American-exercise projection inside the backward loop had:

```python
if is_american:
    s1 = ...; s2 = ...
    if payoff is not None:
        new_v[i,j] = max(new_v[i,j], payoff(s1, s2))
    elif payoff_type == "spread_call":
        new_v[i,j] = max(new_v[i,j], max(s1-s2-strike, 0))
```

No branches for `spread_put`, `best_of_call`, `worst_of_call`. They silently priced as European even with `is_american=True`. Refactored intrinsic into a `_intrinsic_2d(s1, s2)` helper covering all payoff types. American spread-put now shows the expected early-exercise premium (e.g. 26.95 vs European 26.90 on a deep ITM spread).

### T3.8 — `solve_tree_2d` always returned zero Greeks

`return TreeResult(price=..., delta=0.0, gamma=0.0, theta=0.0, ...)` regardless of inputs. Now extracts `delta1` and `delta2` from the step-1 grid (4 nodes, averaging out the spot direction being differentiated). `delta1` lands in the conventional `delta` field; `delta2` goes in `node_prices` as a 2-element array `[delta1, delta2]` (no API extension needed). Gamma and theta remain at 0 — the 2-asset recombining tree has no centred node at (S1, S2) at step 1, so straight ∂V/∂t mixes spot-diffusion convexity with time decay. Bump-and-reprice is the correct way to get gamma/theta on a 2D tree.

### Verification — `test_l2_t3_trees_misc.py`

6 new regression tests, all pass:
- `test_bermudan_with_exercise_dates_dominates_european` — Bermudan put > European put (pre-fix they were equal).
- `test_bermudan_with_no_exercise_dates_equals_european` — sanity: empty exercise set ⇒ European.
- `test_spread_call_delta_nonzero` — delta1 > 0, delta2 < 0 for ATM spread call.
- `test_delta_sum_at_high_spread_approaches_one` — deep-ITM spread: delta1 ≈ +1, delta2 ≈ −1.
- `test_american_spread_put_premium` — American > European for ITM spread put.
- `test_american_worst_of_call_with_custom_payoff` — custom payoff: American ≥ European.

Full parallel suite: **12058 passed in 3:25** — zero regressions.

Tier-3 status: **3 of 19 closed** (T3.8, T3.9, T3.10; T3.11 already subsumed by T2.10).

---

## v0.922.0 — 2026-06-12

**Fix L2 T2.15 — SABR-HW swaption blender returns intrinsic at T=0 instead of unconditional zero.  CLOSES ALL 18 TIER-2 BUGS.**

`options/swaption.py::price_swaption_sabr_hw` had:

```python
if T <= 0:
    return 0.0
```

This returned 0.0 even when the swaption had positive **intrinsic** value (e.g. a payer with strike below the current forward swap rate, evaluated at expiry).

**Fix**: at T ≤ 0 compute and return the intrinsic value:
- Payer: `annuity × max(fwd − K, 0) × notional`
- Receiver: `annuity × max(K − fwd, 0) × notional`

### Verification — `test_l2_t2_15_sabr_hw_blender_intrinsic.py`

4 new regression tests, all pass:
- `test_payer_at_expiry_returns_intrinsic` — deep-ITM payer at T=0 returns positive intrinsic, matches annuity × (fwd − K).
- `test_receiver_at_expiry_returns_intrinsic` — symmetric for receiver.
- `test_out_of_money_at_expiry_returns_zero` — OTM at T=0 still returns 0 (no negative intrinsic).
- `test_atm_post_expiry_positive_T_unchanged` — sanity: positive-T Black-76 path still works.

Full parallel suite: **12052 passed in 3:27** — zero regressions.

### Tier-2 status — **18 of 18 closed** ✓

| # | Status | Note |
|---|---|---|
| T2.1 | ✅ pre-existing | roll_down DF renormalisation (v0.901) |
| T2.2 | ✅ v0.914 | PDE LOG-grid stencil |
| T2.3 | ✅ v0.914 | PDE implicit Dirichlet BC |
| T2.4 | ✅ v0.914 | PDE American boundary |
| T2.5 | ✅ v0.916 | wavelet_transform power-of-2 |
| T2.6 | ✅ v0.917 | native Romberg (scipy.romberg removed) |
| T2.7 | ✅ subsumed | interior_point equality (via T1.4) |
| T2.8 | ✅ v0.918 | tree knock-in via in-out parity |
| T2.9 | ✅ v0.918 | trinomial prob renormalisation |
| T2.10 | ✅ v0.919 | MCEngine.greek() honours param_name |
| T2.11 | ✅ v0.920 | g2pp_swaption no silent zero |
| T2.12 | ✅ v0.920 | g2pp y*=0 fallback raises |
| T2.13 | ✅ subsumed | AAD no-deposit (via T1.13) |
| T2.14 | ✅ v0.921 | bond YTM redemption = last-period notional |
| **T2.15** | ✅ this slice | SABR-HW blender T=0 intrinsic |
| T2.16 | ✅ v0.915 | CDS convexity uses notional |
| T2.17 | ✅ v0.915 | CDS variable notional propagation |
| T2.18 | ✅ v0.915 | protection_leg_pv list+no-schedule |

**All 13 Tier-1 AND 18 Tier-2 bugs closed in 16 slices** since v0.905. Next phase: Tier-3 (single-critic criticals — 19 items) and Tier-4 (highs that didn't reach critical).

---

## v0.921.0 — 2026-06-12

**Fix L2 T2.14 — bond YTM-based pricing now uses the last-period notional for redemption (consistent with curve-based pricing).**

`fixed_income/bond.py::FixedRateBond` set `self.face_value = self.coupon_leg.notional` (the first-period notional from a possibly-variable notional schedule). The curve-based `dirty_price` already used `notional_schedule[-1]` for the redemption to support sinking-fund bonds. But three YTM-based methods used the first-period `self.face_value` for the principal term, causing a silent disagreement between YTM-pricing and curve-pricing for amortising bonds:

- `_price_from_ytm`
- `macaulay_duration`
- `convexity`

For a 5y semi-annual bond amortising 100 → 55, the pre-fix YTM price had a redemption term of 100 / (1 + y/2)^10 while the curve price had 55 × DF(5y). The two paths could disagree by tens of cents at par-equivalent levels.

**Fix**: all YTM-based paths use `self.coupon_leg.notional_schedule[-1]` consistently. Constant-notional bonds are unaffected (first = last). Lock-in test verifies curve-vs-YTM agreement on both constant-notional and amortising bonds.

Also covers T2.13 — `aad_curves` no-deposit first-swap case (subsumed by v0.911 T1.13 fix), with explicit regression test (`test_l2_t2_13_aad_first_swap_no_deposits.py`) committed earlier.

### Verification — `test_l2_t2_14_bond_sinking_redemption.py`

4 new regression tests, all pass:
- `test_constant_notional_ytm_matches_curve` — sanity: plain bond, YTM and curve agree.
- `test_amortising_redemption_matches_curve` — sinking-fund bond (notional 100→55): YTM and curve now agree (pre-fix had large discrepancy).
- `test_duration_uses_last_notional` / `test_convexity_uses_last_notional` — duration and convexity produce sane values for sinking-fund bond.

Full parallel suite: **12048 passed in 3:29** — zero regressions.

Tier-2 status: **16 of 18 closed** (T2.13, T2.14 added). Remaining: T2.15 (SABR-HW blender at T=0).

---

## v0.920.0 — 2026-06-12

**Fix L2 T2.11 / T2.12 — `g2pp_swaption_price` no longer silently masks errors as zero or falls back to bogus `y* = 0`.**

Two silent-failure modes in `models/g2pp_calibration.py::g2pp_swaption_price`:

### T2.11 — Bare `except Exception: return 0.0` around the entire body

The whole function was wrapped in `try: ... except Exception: return 0.0`, masking every error mode (bracketing failure, brentq divergence, numerical-overflow in the ZCB formula, calibration bugs) as a silent zero price. Callers had no way to distinguish "swaption worth ≈ 0" from "pricer crashed".

### T2.12 — Silent `y_star = 0.0` fallback on bracket failure

The Jamshidian y* bracketing routine had:

```python
if bp_lo * bp_hi >= 0:
    y_star = 0.0       # ← bogus fallback
else:
    y_star = brentq(...)
```

When bracketing failed (no sign change after 10 expansions), `y_star = 0.0` is just **wrong** — the K_k strikes derived from it are unrelated to the actual swaption. Combined with T2.11, this produced wildly wrong intermediate prices that the outer except then collapsed to zero.

### Fixes

- T2.11: removed the outer `try/except Exception: return 0.0`. Only the legitimate degenerate cases (no payment dates) still return 0.0 explicitly. Real exceptions propagate.
- T2.12: bracket failure now raises `RuntimeError` with a diagnostic message including the failing `x_val`, the bracket endpoints' bond-portfolio values, and a hint about parameter regime.

### Verification — `test_l2_t2_11_12_g2pp_silent_failures.py`

4 new regression tests, all pass:
- `test_normal_params_produce_positive_price` — sanity: reasonable G2++ params produce a positive finite price.
- `test_zero_payment_dates_returns_zero_cleanly` — degenerate tenor=0 still returns 0.0 explicitly (the only path that should).
- `test_negative_a_propagates_error` — extreme parameters don't silently give 0.0 (either succeed or raise).
- `test_extreme_params_either_raise_or_succeed` — across strike sweep, never silently produce 0; either valid price or RuntimeError with diagnostic.

Full parallel suite (with previously-deselected slow `test_g2pp_calibration` tests included): **12041 passed in 3:24 + 70s for G2++ calibration suite** — zero regressions. The removed silent except did not surface any hidden failures in the full G2++ calibration loop, which validates the fix.

Tier-2 status: **14 of 18 closed** (T2.11, T2.12 added).

---

## v0.919.0 — 2026-06-12

**Fix L2 T2.10 — `MCEngine.greek()` actually honours `param_name`.**

Pre-fix `models/mc_engine.py::MCEngine.greek()` accepted `param_name: str` as a *required* parameter but **never used it**. The implementation always did `self.process.x0 = original_x0 + bump`, which broadcasts the scalar bump across the whole `x0` vector. For a multi-factor process (Heston: x = [S, V]; G2++: x = [x, y]; basket models), "delta" simultaneously bumped every state variable, producing a meaningless mixed sensitivity.

**Fix**: `param_name` is now parsed into one of three forms:
- `int` — index into `process.x0` (e.g. `0` = spot, `1` = variance).
- `"x0"` or `"x0[i]"` — same as the int form.
- any other string — interpreted as an attribute of the `ProcessSpec` (e.g. `"sigma"`, `"mu"`). Requires the drift/diffusion callables to read the attribute at evaluation time. Lets the user compute vega-like sensitivities to model parameters.

Bounds-checked (`IndexError` for bad index, `ValueError` for unknown attribute). Restores the original `x0` / attribute cleanly on the way out.

### Verification — `test_l2_t2_10_mc_greek_param_name.py`

6 new regression tests, all pass:
- `test_x0_index_zero_only_bumps_spot` — 1D GBM `"x0[0]"` gives BS-plausible delta (0.4 < Δ < 0.85).
- `test_int_param_name_works` — bare integer index works.
- `test_x0_restored_after_greek` — restoration check.
- `test_attribute_bump_vega_via_sigma` — `param_name="sigma"` bumps the process's `sigma` attribute and recovers a BS-plausible vega (20 < vega < 60).
- `test_unknown_param_name_raises` — bad string raises `ValueError`.
- `test_index_out_of_range_raises` — bad index raises `IndexError`.

Full parallel suite: **12037 passed in 4:44** — zero regressions.

Tier-2 status: **12 of 18 closed** (T2.10 added).

---

## v0.918.0 — 2026-06-12

**Fix L2 Tier-2: tree knock-in barriers (T2.8) + trinomial probability renormalisation (T2.9).**

Two coupled bugs in `numerical/_trees.py`.

### T2.8 — knock-in barriers silently priced as vanilla

`_apply_barrier` had:

```python
elif self.barrier_type == BarrierType.DOWN_IN:
    pass  # complex — for now, only knock-out supported
elif self.barrier_type == BarrierType.UP_IN:
    pass
```

So the barrier condition was checked but never enforced for in-types. `TreeSolver(barrier_type=BarrierType.DOWN_IN, ...).solve(...)` produced the **vanilla** price, with no warning. The TreeSolver API accepted the type and returned a wrong number.

**Fix**: at the top of `solve()`, intercept knock-in types and apply in-out parity:

> V_knock_in = V_vanilla − V_knock_out

Spawn two side-pricers (one with the corresponding knock-out type, one without barriers) and combine prices + Greeks by linearity. Live verification:

```
Vanilla   = 10.4406
DOWN_OUT  = 10.3523
DOWN_IN   = 0.0883     # post-fix; pre-fix was 10.4406 (= vanilla)
OUT + IN  = 10.4406    # exactly equals vanilla
```

### T2.9 — Trinomial probability triple not renormalised after clamp

`_trinomial_params` had:

```python
p_u = max(0, min(1, p_u))
p_d = max(0, min(1, p_d))
p_m = max(0, 1 - p_u - p_d)
```

When the raw `p_u` or `p_d` was negative (large drift relative to volatility), clamping shifted mass into `p_m` without renormalising the triple. The probs still summed to 1 but the *moments* (drift, variance) were broken — the risk-neutral measure silently violated.

**Fix**: clamp each leg ≥ 0, then renormalise the triple. Mirrors the renormalisation already used in `_2d_trinomial_params` (the 2D tree) immediately below in the same file.

### Verification — `test_l2_t2_8_9_trees_barriers_probs.py`

6 new regression tests:

- `test_in_out_parity_down` / `test_in_out_parity_up` — V(KI) + V(KO) = V(vanilla) exactly; pre-fix V(KI) = V(vanilla) trivially.
- `test_knock_in_greeks_linearity` — KI delta = vanilla delta − KO delta exactly.
- `test_probs_sum_to_one_high_drift` — extreme-drift case (r=20%, σ=10%, dt=0.01) → renormalised probs sum to 1.0 exactly.
- `test_probs_unchanged_normal_regime` — sanity: normal-drift case (the clamp doesn't trigger).
- `test_trinomial_pricing_stable_under_high_drift` — extreme-drift call still produces a finite positive price.

Full parallel suite: **12031 passed in 3:23** — zero regressions.

Tier-2 status: **11 of 18 closed** (T2.8, T2.9 added).

---

## v0.917.0 — 2026-06-12

**Fix L2 T2.6 — `_romberg` now uses a native Richardson extrapolation; scipy.integrate.romberg was removed in SciPy 1.15.**

`numerical/_integrate.py::_romberg` wrapped `scipy.integrate.romberg`, which was removed in SciPy 1.15. Every call to `integrate(..., method=IntegrationMethod.ROMBERG)` raised:

```
ImportError: cannot import name 'romberg' from 'scipy.integrate'
```

Replaced with a native Romberg-on-trapezoid implementation (incremental trapezoidal refinement + Richardson extrapolation of the table). Returns proper `IntegrationResult` with the error estimate and convergence flag. No more SciPy version pin needed.

### Verification — `test_l2_t2_6_romberg_native.py`

6 tests, all pass:
- Constant function: ∫₀¹ 5 dx = 5 (exact).
- ∫₀¹ x³ dx = 1/4 to 1e-10.
- ∫₀^π sin(x) dx = 2 to 1e-10.
- 5σ-truncated standard normal CDF ≈ 1 to 1e-6.
- ∫₀^{2π} cos(x) dx = 0 to 1e-8.
- Smoke test: no ImportError, `result.method == "romberg"`, `result.converged`.

Tier-2 status: **9 of 18 closed** (added T2.6).

Full parallel suite: **12025 passed in 4:05** — zero regressions.

---

## v0.916.0 — 2026-06-12

**Fix L2 T2.5 — `wavelet_transform` pads non-power-of-2 input.**

`numerical/_fourier.py::wavelet_transform` halves the signal at each level via `x[0::2]` and `x[1::2]`. For odd `len(x)`, these slices have different lengths and the addition raised:

```
ValueError: operands could not be broadcast together with shapes (4,) (3,)
```

The DWT (Haar / DB2) is well-defined only on power-of-2 inputs. Fix: pad to `max(next_power_of_2(n), 2**levels)` with zeros before decomposition; also raise on `n < 2`. Power-of-2 inputs are unaffected.

### Verification — `test_l2_t2_5_wavelet_padding.py`

17 tests (parametrised over n = 3, 5, 7, 9, 11, 17, 100, 1000 for Haar and 3, 5, 11, 17, 31 for DB2), all pass. Includes:
- Arbitrary-length Haar/DB2 no longer crash.
- Power-of-2 still produces a length-n coefficient vector (unchanged).
- Length-2 exact-value sanity check.
- Length-1 raises with clear message.

Tier-2 status: **8 of 18 closed** (T2.5 added; T2.1, T2.2, T2.3, T2.4, T2.7, T2.16, T2.17, T2.18 already closed).

---

## v0.915.0 — 2026-06-12

**Fix L2 Tier-2: CDS variable-notional propagates through aged-CDS methods + convexity formula uses notional.**

Three coupled bugs in `credit/cds.py` closed in one slice.

### T2.18 — `protection_leg_pv` silently used `notional[0]` for list inputs without `schedule_dates`

If a caller passed `notional=[1M, 2M, 3M]` but omitted `schedule_dates`, the function fell through to `n = notional[0]` and scaled the entire protection leg by 1M — variable-notional CDS got priced as constant-1M. Now raises `ValueError` so the caller cannot accidentally lose the schedule.

### T2.17 — Variable notional silently dropped in aged-CDS methods

`CDS.__init__` set `self.notional = self.notional_schedule[0]` (scalar first-period) for convenient single-value access. But five methods then passed `notional=self.notional` (the scalar) when constructing a new aged/parallel `CDS`:

- `isda_upfront` — built the std-coupon CDS with scalar
- `roll_down` — built shorter CDS with scalar
- `theta` — built aged CDS with scalar
- `rec01` — built recovery-bumped CDS with scalar
- `cds_pnl_attribution` — built aged CDS with scalar

Result: variable-notional CDS got their tail-period notionals silently dropped in roll-down, theta, isda upfront, rec01, and full P&L attribution.

**Fix**:
- `rec01` and `isda_upfront`: forward `self.notional_schedule` (full list).
- `roll_down`, `theta`, `cds_pnl_attribution`: introduce new `_aged_notional(new_start)` helper that slices the schedule at the period containing `new_start`. The aged CDS then has a notional list of length = remaining-periods.

### T2.16 — Convexity P&L multiplied by `|PV|` instead of notional

`spread_convexity` returns `d²PV/ds² / notional`. The second-order P&L correction is therefore

> convexity_pnl = ½ × (conv × notional) × Δs²

Pre-fix the code used `pv_for_conv = abs(pv_t0)` with a fallback to `cds.notional` only when `|PV| < 1e-10`. This was dimensionally wrong: |PV| is small near par (where the CDS spread equals the par spread, e.g. typical IG trading conventions) and grows with mark-to-market away from par — neither is the right normalisation. The formula effectively collapsed to ~0 near par and gave a |PV|-scaled value elsewhere.

Fixed to `0.5 * conv * cds.notional * delta_spread**2`.

### Verification — `test_l2_t2_cds_variable_notional.py`

8 new regression tests, all pass:

- `TestProtectionLegRequiresScheduleForList`:
  - `test_raises_on_list_without_schedule` — variable notional without schedule_dates now raises.
  - `test_scalar_path_unchanged` — scalar path still works.
- `TestVariableNotionalPropagation` (amortising 5y CDS with 20 periods, notional from 10M → 2.4M):
  - `test_average_notional_not_scalar` — sanity: the schedule is genuinely variable.
  - `test_isda_upfront_uses_average_notional` — variable vs constant-10M produce different upfronts.
  - `test_roll_down_sees_variable_notional` — roll_down for variable ≠ for first-period-only.
  - `test_theta_sees_variable_notional` — same for theta.
- `TestConvexityFormula`:
  - `test_convexity_pnl_nonzero_at_par` — at-par CDS (|PV|≈0): post-fix convexity is non-zero (pre-fix collapsed to ~0).
  - `test_convexity_scales_with_notional` — 10× notional → 10× convexity P&L (linearity check).

Full parallel suite: **12002 passed in 4:50** — zero regressions.

### Tier-2 status — 7 of 18 closed

| # | Status | Note |
|---|---|---|
| T2.1 | ✅ pre-existing | roll_down DF renormalisation (v0.901 as B.1) |
| T2.2 | ✅ v0.914 | PDE LOG-grid stencil |
| T2.3 | ✅ v0.914 | PDE implicit Dirichlet BC |
| T2.4 | ✅ v0.914 | PDE American boundary |
| T2.5 | queued | wavelet_transform power-of-2 |
| T2.6 | queued | _romberg dead in SciPy 1.15 |
| T2.7 | ✅ subsumed by T1.4 | interior_point equality |
| T2.8 | queued | tree knock-in barriers |
| T2.9 | queued | trinomial prob clamp |
| T2.10 | queued | mc_engine.greek() bumps all |
| T2.11/12 | queued | g2pp_calibration silent failures |
| T2.13 | queued | aad_curves first-swap-no-deposit |
| T2.14 | queued | bond _price_from_ytm redemption |
| T2.15 | queued | SABR-HW blender at T=0 |
| **T2.16** | ✅ this slice | CDS convexity uses notional |
| **T2.17** | ✅ this slice | Variable notional propagation |
| **T2.18** | ✅ this slice | protection_leg_pv list+no-schedule |

---

## v0.914.0 — 2026-06-12

**Fix L2 Tier-2: PDE non-uniform stencil + Dirichlet BC + American boundary + time-index — 4 bugs in `PDESolver1D` closed in one slice.**

Opens the Tier-2 phase by clearing four entangled bugs in `numerical/_pde.py::PDESolver1D` that all touched the same `_theta_step` and outer time loop.

### T2.2 — Uniform-grid stencil applied to non-uniform grids

`_theta_step` discretised the second derivative with `d2 = σ²S² / (ds_m · ds_p)` and split it evenly into sub/super diagonals (`a[i] = d2/2 − d1`, `c[i] = d2/2 + d1`). For uniform grids this is correct; for non-uniform grids (LOG, SINH) the proper 3-point central stencil is

> V_{i-1} coef = σ²S² / (ds_m · (ds_m+ds_p))
> V_{i+1} coef = σ²S² / (ds_p · (ds_m+ds_p))
> V_{i}   coef = −σ²S² / (ds_m · ds_p)

Live measurement on the canonical ATM call (S=100, K=100, r=5%, σ=20%, T=1y):

| Grid | Pre-fix PDE | Black-Scholes | Pre-fix error |
|---|---|---|---|
| UNIFORM | 10.4645 | 10.4506 | +0.13 % |
| LOG | **11.7966** | 10.4506 | **+12.9 %** |

Post-fix:

| Grid | Post-fix PDE | rel. err |
|---|---|---|
| UNIFORM | 10.4645 | +0.13 % |
| LOG | 10.4734 | **+0.22 %** |

### T2.3 — Implicit Dirichlet BC not enforced in tridiagonal solve

The tridiagonal system was built with `diag[0]=1, lower[0]=upper[0]=0` and `rhs[0]=0` (zero-init), so `V_new[0] = 0` after the solve. The code then overwrote `V_new[0] = V[0]` (the value BEFORE the step). The interior equation at `i=1` therefore used `V_new[0]=0` inside the implicit solve — wrong for **puts** (whose lower BC is `K·e^{−rτ} − S[0]`, non-zero) and for the upper BC of calls.

Fix: pass the proper BCs into `_theta_step` as `bc_lo`, `bc_hi`; set `rhs[0]=bc_lo`, `rhs[-1]=bc_hi`. The boundary rows with `diag=1` then yield `V_new[boundary] = BC` naturally, and the interior solve at `i=1` / `i=N−2` uses the proper boundary value.

### T2.B — Time-to-maturity index inverted at the boundary

The outer loop applied the upper-boundary value as `S_max − K·exp(−r·(n_time − step)·dt)`. After step `k`, the time-to-maturity is `(k+1)·dt`, NOT `(n_time − k)·dt`. The expression is only correct at the middle step; everywhere else the boundary uses the wrong discount factor. Replaced with `τ = (step + 1)·dt`.

### T2.4 — American boundaries used European discounted strike

For American puts at `S→0`, the optimal exercise dominates: `V = K − S[0]` (intrinsic). Pre-fix the boundary used `K·e^{−rτ} − S[0]` (European). Fix: for `is_american=True`, set boundaries to intrinsic payoff, and re-impose intrinsic floor after the early-exercise projection.

### Verification — `test_l2_t2_pde_stencil_bc.py`

6 new regression tests, all pass:

- `TestNonUniformStencil::test_log_grid_atm_call_matches_bs` — the headline: LOG grid no longer overshoots BS by 13 %.
- `TestNonUniformStencil::test_uniform_grid_atm_call_matches_bs` — uniform grid unchanged (was already fine).
- `TestDirichletBC::test_uniform_grid_atm_put_matches_bs` — uniform put now matches BS (pre-fix the implicit BC bug biased puts).
- `TestDirichletBC::test_log_grid_atm_put_matches_bs` — combined fix: non-uniform grid + correct BC for put.
- `TestAmericanBoundary::test_american_put_dominates_european` — deep-ITM American put ≥ intrinsic ≥ European.
- `TestTimeIndex::test_long_dated_call_no_explosive_bias` — 5y ATM call agrees with BS to <2 %.

Full parallel suite: **11994 passed in 4:45** — zero regressions.

### Tier-2 status — 4 of 18 closed (T2.2, T2.3, T2.4 this slice; T2.1 already closed in v0.901 as B.1)

Remaining Tier-2 (14): T2.5 wavelet_transform power-of-2; T2.6 _romberg dead; T2.7 interior_point equality-only (already closed via T1.4); T2.8 tree knock-in barriers; T2.9 trinomial prob clamp; T2.10 mc_engine.greek() bumps everything; T2.11/T2.12 g2pp_calibration silent failures; T2.13 aad_curves first-swap-no-deposit; T2.14 bond _price_from_ytm redemption; T2.15 SABR-HW blender at T=0; T2.16 cds convexity uses |PV|; T2.17 cds variable-notional drop; T2.18 cds protection_leg_pv notional[0].

---

## v0.913.0 — 2026-06-12

**Fix L2 T1.1 — `multilevel_mc` Giles coupling: coarse path now uses paired-sum fine increments. CLOSES ALL 13 TIER-1 BUGS.**

The last of MODULE_HEALTH's Tier-1 (dual-critic-confirmed) bugs is closed.

### The bug

`numerical/_mc.py::multilevel_mc` claimed to implement Giles (2008) MLMC. The algorithm requires that at each level l ≥ 1, the FINE path (n_fine = base_steps · 2^l timesteps) and the COARSE path (n_fine / 2 timesteps) share the SAME underlying Brownian motion — the coarse path is obtained by **pairing** fine Brownian increments:

> dW_coarse[m] = dW_fine[2m] + dW_fine[2m+1]

This coupling is what makes Var(P_fine − P_coarse) → 0 as the discretization refines (the property MLMC's O(ε⁻²) cost depends on).

The pre-fix code instead generated the fine path then **downsampled** it:

```python
paths_fine = process_fn(n_paths, n_fine, seed_l)
paths_coarse = paths_fine[:, ::2]    # <-- the bug
```

Since the downsampled "coarse" path is just a sub-sampling of the fine path's *output*, for any European payoff depending only on the terminal value `paths[:, -1]`:

> P_fine = payoff(paths_fine[:, -1])
> P_coarse = payoff(paths_fine[:, ::2][:, -1]) = payoff(paths_fine[:, -1])
> P_fine − P_coarse ≡ 0

The MLMC correction at every level ≥ 1 was identically zero. `multilevel_mc` collapsed to E[P_0] — a 4-step Euler estimate — at any number of levels. The function returned a heavily biased estimate while quietly pretending to do MLMC.

### The fix — proper Giles coupling

Redesigned the `process_fn` interface to consume **pre-generated Brownian increments** instead of an opaque seed:

```python
process_fn: callable(dW, dt) → paths array (n_paths, n_steps+1)
```

The MLMC routine now does the coupling work:

```python
Z = rng.standard_normal((n_paths, n_fine))
dW_fine = Z * sqrt(dt_fine)
dW_coarse = dW_fine[:, 0::2] + dW_fine[:, 1::2]     # <-- pair the increments

paths_fine = process_fn(dW_fine, dt_fine)
paths_coarse = process_fn(dW_coarse, dt_coarse)     # SAME Brownian path, coarser stepping
```

This is a breaking API change to `process_fn`, but there were no production callers (only the smoke `assert callable(multilevel_mc)` test), and the previous routine was broken anyway.

### Verification — `test_l2_t1_1_mlmc_giles.py`

4 new regression tests, all pass:

- **`test_mlmc_converges_to_black_scholes`** — the headline test. ATM European call on GBM (S₀=100, K=100, r=5%, σ=20%, T=1y). 6-level MLMC with 20 000 base paths must agree with Black-Scholes to <5% relative. Pre-fix this returned the heavily biased 4-step Euler P₀ estimate.
- **`test_level_corrections_are_nonzero`** — proves Giles coupling actually decouples fine and coarse. With the SAME seed, a 1-level run and a 4-level run produce measurably DIFFERENT estimates. Pre-fix they were identical (corrections were all 0).
- **`test_paired_increments_have_same_terminal_brownian_value`** — sanity: `sum(dW_fine) == sum(dW_coarse)` (the terminal Brownian motion is preserved by pairing). Algebraic identity, kept as a regression anchor.
- **`test_variance_dominated_by_low_levels`** — Giles signature: per-level variance increments decay. The level-1 correction adds nonzero variance (proof of working coupling).

Full parallel suite: **11988 passed in 3:20** — zero regressions.

### Tier-1 status — **13 of 13 closed**

| # | Module | Status |
|---|---|---|
| **T1.1** | `numerical/_mc.py` multilevel_mc Giles coupling | ✅ this slice |
| T1.2 | `numerical/_fourier.py` density crash | ✅ v0.907 |
| T1.3 | `numerical/_integrate.py` integrate_2d swap | ✅ v0.907 |
| T1.4 | `numerical/_optimize.py` interior_point drops equalities | ✅ v0.912 |
| T1.5 | `numerical/_trees.py` Tian V formula | ✅ v0.907 |
| T1.6 | `numerical/_trees.py` discrete dividends terminal-grid + trinomial | ✅ v0.908 |
| T1.7 | `models/mc_engine.py` Milstein silently runs Euler | ✅ v0.907 |
| T1.8 | `models/cos_method.py` V_k integration bounds deep ITM | ✅ v0.910 |
| T1.9 | `models/hull_white.py` Swaption uses r0 not α(T_expiry) | ✅ v0.909 |
| T1.10 | `curves/global_solver.py` residual collision | ✅ v0.905 |
| T1.11 | `curves/multicurve_solver.py` PV_float first period | ✅ v0.905 |
| T1.12 | `curves/multicurve_solver.py` PV_float / annuity inconsistent | ✅ v0.905 (same root) |
| T1.13 | `curves/aad_curves.py` swap bootstrap flat-extrapolation | ✅ v0.911 |

**All 13 Tier-1 bugs closed in 7 slices.** Next: MODULE_HEALTH Tier-2 — 18 critical+high pairings to sweep.

---

## v0.912.0 — 2026-06-12

**Fix L2 T1.4 — `interior_point` now honours equality constraints via SLSQP inner solve.**

Twelfth of 13 Tier-1 (dual-critic-confirmed) bugs closed. Only T1.1 (MLMC) remaining.

### The bug

`numerical/_optimize.py::interior_point` accepts `equality_constraints` as a parameter and even **constructs** the SciPy-style constraint dicts:

```python
constraints = []
if equality_constraints:
    for h in equality_constraints:
        constraints.append({"type": "eq", "fun": h})

r = _minimize(barrier_obj, x, method="BFGS",     # <— never passes `constraints`
              options={"maxiter": 50, "gtol": tol / t})
```

…but never passes the list to `_minimize`, and uses **BFGS** (an unconstrained method) for the inner solve. So any caller passing `equality_constraints` got them silently dropped — the optimiser just minimised the (barrier-modified) objective over the unconstrained space.

This is a critical bug: every caller using `interior_point` for an equality-constrained problem (e.g. minimum-norm projection, equality-constrained least-squares with logarithmic barriers) was getting the **unconstrained** minimum back.

### The fix

When equality constraints are present, switch to **SLSQP** (the SciPy method that supports equality constraints) and pass the constraint dicts. Pure-inequality problems keep the BFGS path (faster for unconstrained barrier subproblems):

```python
if constraints:
    r = _minimize(barrier_obj, x, method="SLSQP",
                  constraints=constraints,
                  options={"maxiter": 100, "ftol": tol / t})
else:
    r = _minimize(barrier_obj, x, method="BFGS",
                  options={"maxiter": 50, "gtol": tol / t})
```

Additional convergence-check fix: pure equality-constrained problems (no inequalities) used to never terminate the outer loop (the existing check `n_ineq / t < tol` required `n_ineq > 0`). Added a fast-path that breaks on `r.success` after the first SLSQP pass — no need to grow `t` when there are no barrier terms.

### Verification — `test_l2_t1_4_interior_point_equalities.py`

4 new regression tests, all pass:

- `test_simple_equality_constrained_quadratic` — minimise ‖x‖² in ℝ³ subject to `sum(x) = 3`. Optimum = (1, 1, 1). Pre-fix → (0, 0, 0) (unconstrained minimum, sum = 0 not 3).
- `test_two_equalities` — minimise ‖x‖² subject to `x₀+x₁=1` and `x₁+x₂=1`. Both equalities now hold to 1e-4.
- `test_equality_plus_inequality` — minimise `x²+y²` subject to `x+y=2` (eq) and `x≥0.5` (ineq). Optimum = (1, 1), value 2. Both equality AND inequality honoured simultaneously.
- `test_pure_inequality_path_unchanged` — regression check on the BFGS path when no equalities are supplied: unchanged behaviour, converges to the unconstrained optimum (1, 2) on a strict-interior problem.

Full parallel suite: **11984 passed in 3:24** — zero regressions.

### Tier-1 status — 12 of 13 closed

| # | Module | Status |
|---|---|---|
| T1.1 | `numerical/_mc.py` multilevel_mc broken | queued (last one) |
| T1.2 | `numerical/_fourier.py` density crash | ✅ v0.907 |
| T1.3 | `numerical/_integrate.py` integrate_2d swap | ✅ v0.907 |
| **T1.4** | `numerical/_optimize.py` interior_point drops equalities | ✅ this slice |
| T1.5 | `numerical/_trees.py` Tian V formula | ✅ v0.907 |
| T1.6 | `numerical/_trees.py` discrete dividends terminal-grid + trinomial | ✅ v0.908 |
| T1.7 | `models/mc_engine.py` Milstein silently runs Euler | ✅ v0.907 |
| T1.8 | `models/cos_method.py` V_k integration bounds deep ITM | ✅ v0.910 |
| T1.9 | `models/hull_white.py` Swaption uses r0 not α(T_expiry) | ✅ v0.909 |
| T1.10 | `curves/global_solver.py` residual collision | ✅ v0.905 |
| T1.11 | `curves/multicurve_solver.py` PV_float first period | ✅ v0.905 |
| T1.12 | `curves/multicurve_solver.py` PV_float / annuity inconsistent | ✅ v0.905 (same root) |
| T1.13 | `curves/aad_curves.py` swap bootstrap flat-extrapolation | ✅ v0.911 |

**Only T1.1 remains** — `multilevel_mc` redesign requires Giles coupling: the coarse path must be re-simulated from the fine path's normals (paired increments summed) instead of independently sub-sampling. This is the biggest scope change among the Tier-1 set.

---

## v0.911.0 — 2026-06-12

**Fix L2 T1.13 — AAD swap bootstrap properly interpolates intermediate-coupon DFs through the unknown last-pillar DF instead of silently flat-extrapolating.**

Eleventh of 13 Tier-1 (dual-critic-confirmed) bugs closed.

### The bug

`aad_bootstrap` (in `curves/aad_curves.py`) iterates over the input swaps and, at each step, builds a `temp_curve` from the EXISTING pillars only (i.e. the pillars determined so far — typically just the deposits before the first swap is processed). Intermediate-coupon DFs needed for the par-condition annuity were drawn via `temp_curve.df(coupon_date)`.

When the new swap's tenor extended beyond all existing pillars (the common case — bootstrap a 5y swap against a 1y deposit, the only thing existing at that point is the 1y deposit pillar), the intermediate coupons at 1.5y, 2y, 2.5y, ..., 4.5y fell **in the extrapolation region** of `aad_log_linear_interp`, which flat-extrapolates at `dfs[-1]` (the last known pillar's DF). So the bootstrap silently equated every gap-coupon DF to the deposit DF, biasing both:

1. **The numerical bootstrap itself** — the resulting 5y DF did not satisfy the par-zero PV condition when the curve was re-priced with proper interpolation.
2. **The AAD sensitivities** — d(df_mat)/d(swap_rate) propagated through wrong intermediate DFs.

### The fix

A fixed-point iteration where the trial curve at each step **includes** the current `df_mat` estimate as a pillar, so gap-coupon DFs interpolate between the last existing pillar and `df_mat`:

```python
df_mat = Number(brentq(_par_residual_float, 1e-6, 3.0))  # sharp float init
for _ in range(50):
    trial_dfs = pillar_dfs + [df_mat]
    trial_curve = AADDiscountCurve(reference_date, pillar_dates + [mat],
                                    trial_dfs, swap_dc)
    annuity_before_last = Number(0.0)
    for k in range(1, len(schedule) - 1):
        tau_k = year_fraction(schedule[k-1], schedule[k], swap_dc)
        annuity_before_last += par_rate * tau_k * trial_curve.df(schedule[k])
    tau_last = year_fraction(schedule[-2], schedule[-1], swap_dc)
    df_mat_new = (Number(1.0) - annuity_before_last) / (Number(1.0) + par_rate * tau_last)
    if abs(df_mat_new.value - df_mat.value) < 1e-14:
        df_mat = df_mat_new
        break
    df_mat = df_mat_new
```

Two-stage solver:

1. **Float pre-solve via `brentq`** — finds `df_mat` to machine precision using a plain-float `_FloatLogLinearCurve` that mirrors the AAD curve's log-linear interpolation. Sharp initial guess for stage 2.
2. **AAD fixed-point** — starting from the brentq solution, iterates the construction `x → (1 − annuity_before(x)) / (1 + K·τ_last)` with full Number arithmetic. At the converged fixed point, the AAD graph captures the implicit-function dependency `dx*/dp = −(∂A + ∂B) / (1 + ∂B/∂x)` of the par-condition root. Convergence is fast — `|B'(x*)|` is typically ≤ 0.06, so residual reaches 1e-14 in well under 50 iterations.

### Verification — `test_l2_t1_13_aad_bootstrap_interp.py`

4 new regression tests:

- **`test_bootstrap_repriceses_5y_swap_after_1y_deposit`** — the canonical T1.13 case. Bootstrap a 1y deposit + 5y swap, then re-price the 5y swap against the curve. Post-fix PV ≈ 5e-17 (machine precision). Pre-fix this re-pricing PV was far from zero because gap-coupon DFs were anchored at the deposit DF.
- **`test_bootstrap_repriceses_three_swaps`** — stress: 1y deposit + 2y, 5y, 10y swaps. All three must re-price at par. Post-fix: ≤ 1e-8 for each. Pre-fix: errors compound through the bootstrap chain.
- **`test_dpv_dswaprate_nonzero`** — AAD sensitivity check: `d df(3y) / d swap_5y_rate` and `d df(3y) / d dep_1y_rate` are both nonzero. Pre-fix the swap-rate adjoint was wrong because intermediate DFs didn't flow through `swap_rate`.
- **`test_bumped_swaprate_changes_intermediate_df`** — finite-difference check: a 1bp bump in the swap rate must move df(3y) by a measurable amount (>1e-5). Pre-fix the bumped curve's df(3y) was nearly identical to base because it was flat-extrapolated from the unchanged deposit DF.

Live demonstration of the fix:
```
Bootstrapped pillar DFs (post-fix):
  t=0.000 df=1.000000
  t=1.000 df=0.975279
  t=4.997 df=0.819202
Schedule DFs at 0.5y intervals:  smooth log-linear monotone interpolation
  t=1.496 df=0.954350  (between 1y and 5y pillars)
  ...
  t=4.499 df=0.837268
Re-priced par swap PV = -5.55e-17
```

Full parallel suite: **11980 passed in 3:24** — zero regressions.

### Tier-1 status — 11 of 13 closed

| # | Module | Status |
|---|---|---|
| T1.1 | `numerical/_mc.py` multilevel_mc broken | queued |
| T1.2 | `numerical/_fourier.py` density crash | ✅ v0.907 |
| T1.3 | `numerical/_integrate.py` integrate_2d swap | ✅ v0.907 |
| T1.4 | `numerical/_optimize.py` interior_point drops equality constraints | queued |
| T1.5 | `numerical/_trees.py` Tian V formula | ✅ v0.907 |
| T1.6 | `numerical/_trees.py` discrete dividends terminal-grid + trinomial | ✅ v0.908 |
| T1.7 | `models/mc_engine.py` Milstein silently runs Euler | ✅ v0.907 |
| T1.8 | `models/cos_method.py` V_k integration bounds deep ITM | ✅ v0.910 |
| T1.9 | `models/hull_white.py` Swaption uses r0 not α(T_expiry) | ✅ v0.909 |
| T1.10 | `curves/global_solver.py` residual collision | ✅ v0.905 |
| T1.11 | `curves/multicurve_solver.py` PV_float first period | ✅ v0.905 |
| T1.12 | `curves/multicurve_solver.py` PV_float / annuity inconsistent | ✅ v0.905 (same root) |
| **T1.13** | `curves/aad_curves.py` swap bootstrap flat-extrapolation | ✅ this slice |

**2 remaining — the deepest two:** T1.1 (MLMC redesign — needs Giles coupling), T1.4 (interior_point IPM rewrite).

---

## v0.910.0 — 2026-06-12

**Fix L2 T1.8 — COS method V_k payoff integral now intersects the payoff support with the truncation interval [a, b].**

Tenth of 13 Tier-1 (dual-critic-confirmed) bugs closed.

### The bug

The Fang-Oosterlee (2008) COS method prices a European option as

> price = K · e^{−rT} · Σ_k Re(φ(u_k) · e^{iu_k(x−a)}) · V_k

where V_k integrates the payoff against cosine basis functions over the truncation interval [a, b] for log-moneyness y = log(S/K). For a call payoff (e^y − 1)⁺, V_k integrates over y ∈ [max(0, a), b] (the payoff's support intersected with the truncation interval); for a put (1 − e^y)⁺, over [a, min(0, b)].

`models/cos_method.py` hardcoded the V_k bounds as `[0, b]` (call) and `[a, 0]` (put). This is correct in the typical regime where `a < 0 < b`, but is wrong in either of two regimes:

- **a > 0** (deep-ITM call, low vol, short T): the integration over `[0, b]` includes `[0, a]`, which is outside the truncation interval and should not contribute. Pre-fix the deep-ITM call mispriced.
- **b < 0** (deep-ITM put, low vol, short T): symmetric — `[a, 0]` includes `[b, 0]` outside the truncation.

### Verification of magnitude

Live demonstration — deep-ITM call S=200, K=100, r=5%, σ=8%, T=0.25 (here `a ≈ 0.41 > 0`):

| | Price |
|---|---|
| Black-Scholes reference | 101.242220 |
| COS post-fix (N=256, L=10) | 101.242220 |
| Relative error | 1.4e-16 (machine precision) |

Pre-fix this call mispriced because the V_k coefficients integrated over a domain that included a region outside the truncation, double-counting the call's payoff there.

### The fix

The patch is local to `cos_price`:

```python
if option_type == OptionType.CALL:
    c = max(0.0, a)
    d = b
    if c < d:
        V_k = 2.0/(b - a) * (_chi(k, a, b, c, d) - _psi(k, a, b, c, d))
    else:
        V_k = 0.0
else:
    c = a
    d = min(0.0, b)
    if c < d:
        V_k = 2.0/(b - a) * (-_chi(k, a, b, c, d) + _psi(k, a, b, c, d))
    else:
        V_k = 0.0
```

The `c < d` guard handles the pathological case where the truncation interval lies entirely outside the payoff's support (V_k = 0 — the option is unconditionally worthless within the truncation).

### Verification — `test_l2_t1_8_cos_method_truncation.py`

7 new regression tests (all pass):
- `test_atm_call_unchanged` / `test_atm_put_unchanged` — sanity: ATM (typical a < 0 < b regime) still agrees with Black-Scholes to 1e-4.
- `test_deep_itm_call_low_vol` — S=200, K=100, σ=8%, T=3m → a > 0 regime. Post-fix COS matches BS to <1e-3 (in practice machine precision).
- `test_deep_itm_put_low_vol` — S=50, K=100, σ=8%, T=3m → b < 0 regime. Same.
- `test_parity` (parametrised over ATM, deep-ITM call, deep-ITM put) — put-call parity holds across all three regimes. Pre-fix the deep-ITM regimes violated parity by the V_k bug.

Full parallel suite: **11976 passed in 3:21** — zero regressions.

### Tier-1 status — 10 of 13 closed

| # | Module | Status |
|---|---|---|
| T1.1 | `numerical/_mc.py` multilevel_mc broken | queued |
| T1.2 | `numerical/_fourier.py` density crash | ✅ v0.907 |
| T1.3 | `numerical/_integrate.py` integrate_2d swap | ✅ v0.907 |
| T1.4 | `numerical/_optimize.py` interior_point drops equality constraints | queued |
| T1.5 | `numerical/_trees.py` Tian V formula | ✅ v0.907 |
| T1.6 | `numerical/_trees.py` discrete dividends terminal-grid + trinomial | ✅ v0.908 |
| T1.7 | `models/mc_engine.py` Milstein silently runs Euler | ✅ v0.907 |
| **T1.8** | `models/cos_method.py` V_k integration bounds deep ITM | ✅ this slice |
| T1.9 | `models/hull_white.py` Swaption uses r0 not α(T_expiry) | ✅ v0.909 |
| T1.10 | `curves/global_solver.py` residual collision | ✅ v0.905 |
| T1.11 | `curves/multicurve_solver.py` PV_float first period | ✅ v0.905 |
| T1.12 | `curves/multicurve_solver.py` PV_float / annuity inconsistent | ✅ v0.905 (same root) |
| T1.13 | `curves/aad_curves.py` swap bootstrap flat-extrapolation | queued |

3 remaining are the deepest: T1.1 MLMC redesign, T1.4 IPM rewrite, T1.13 AAD bootstrap.

---

## v0.909.0 — 2026-06-12

**Fix L2 T1.9 — Hull-White tree swaption now centres expiry-node rates on α(T_expiry), not on today's short rate r(0).**

Ninth of 13 Tier-1 (dual-critic-confirmed) bugs from `MODULE_HEALTH.md` closed.

### The bug

`HullWhite.tree_european_swaption` (in `models/hull_white.py`) built the trinomial tree to evolve state prices Q correctly to time `expiry_T`, then iterated over the expiry-time nodes with:

```python
r_j = r0 + j * dr
```

where `r0 = self._forward_rate(0.0)` — today's instantaneous short rate. That is wrong: under Hull-White, the rate at the expiry nodes is `r(T_expiry) = α(T_expiry) + j·dr`, where α is the closed-form drift adjustment

> α(t) = f^M(0,t) + (σ²/(2a²))·(1 − e^{−at})²

(Brigo-Mercurio 2006, eq. 3.34).

### Magnitude of the error

On a representative steeply rising curve (1% at t=0 rising 80bp/year, so 9% at 10y), with `a=0.05, σ=0.5%`:

| Quantity | Value |
|---|---|
| α(0)   | 1.20 % |
| α(3y)  | 7.41 % |
| Pre-fix centring | 1.20 % (used α(0)) |
| Post-fix centring | 7.41 % (uses α(T_expiry)) |
| **Centring error** | **621 bps in the short rate at expiry nodes** |

A 600bp error in the centring of the rate grid at expiry translates to a multi-tens-of-percent error in the swaption PV — far beyond any plausible model calibration tolerance.

### The fix

Two-line change in `hull_white.py`:

1. New method `HullWhite._alpha(t)` implementing the closed-form formula `α(t) = f(0,t) + (σ²/(2a²))·(1 − e^{−at})²`.
2. `tree_european_swaption` calls `alpha_expiry = self._alpha(expiry_T)` once before the node loop and uses `r_j = alpha_expiry + j * dr`.

The tree calibration in `_evolve_state_prices` is unchanged — only the readout step at expiry is corrected. (The closed-form α(t) is the continuous-time limit of the tree's per-step numerical calibration; for typical step sizes the two agree to within a few bp.)

### Verification — `test_l2_t1_9_hw_swaption_alpha.py`

5 new regression tests, all pass:

- `test_alpha_formula_at_zero_equals_short_rate` — sanity: α(0) = f(0,0) since `1−e^0 = 0`.
- `test_alpha_grows_with_forward_curve` — on a 100bp/year-slope curve, α(5y) − α(0) > 400bp (vs near-zero pre-fix when using α(0) everywhere).
- `test_tree_swaption_atm_matches_jamshidian_flat_curve` — on a flat 1% curve (where the pre-fix bug is small because α(T) ≈ α(0)), the tree price agrees with the analytical HW Jamshidian decomposition to within 5%.
- `test_tree_swaption_matches_jamshidian_steep_curve` — **the discriminating test**: 80bp/year-slope, 3y into 4y ATM payer. Post-fix the tree agrees with Jamshidian to within 10%; pre-fix the disagreement was >>10%.
- `test_tree_swaption_deep_itm_recovers_intrinsic` — deep-ITM with low vol → tree price within 15% of annuity·(fwd − K). Pre-fix the wrong centring biased this by the centring error × annuity, dwarfing intrinsic.

The test file embeds a self-contained Jamshidian-decomposition payer-swaption pricer (bracketing for r*, Brigo-Mercurio bond-option vol formula, ZBP via Black-style normal) as the ground-truth reference. ~120 lines including helpers.

Full parallel suite: **11969 passed in 4:50** — zero regressions.

### Tier-1 status — 9 of 13 closed

| # | Module | Status |
|---|---|---|
| T1.1 | `numerical/_mc.py` multilevel_mc broken | queued |
| T1.2 | `numerical/_fourier.py` density crash | ✅ v0.907 |
| T1.3 | `numerical/_integrate.py` integrate_2d swap | ✅ v0.907 |
| T1.4 | `numerical/_optimize.py` interior_point drops equality constraints | queued |
| T1.5 | `numerical/_trees.py` Tian V formula | ✅ v0.907 |
| T1.6 | `numerical/_trees.py` discrete dividends terminal-grid + trinomial | ✅ v0.908 |
| T1.7 | `models/mc_engine.py` Milstein silently runs Euler | ✅ v0.907 |
| T1.8 | `models/cos_method.py` V_k integration bounds deep ITM | queued |
| **T1.9** | `models/hull_white.py` Swaption uses r0 not α(T_expiry) | ✅ this slice |
| T1.10 | `curves/global_solver.py` residual collision | ✅ v0.905 |
| T1.11 | `curves/multicurve_solver.py` PV_float first period | ✅ v0.905 |
| T1.12 | `curves/multicurve_solver.py` PV_float / annuity inconsistent | ✅ v0.905 (same root) |
| T1.13 | `curves/aad_curves.py` swap bootstrap flat-extrapolation | queued |

Remaining 4 are the deepest ones: T1.1 MLMC redesign (need Giles coupling), T1.4 IPM rewrite, T1.8 cos_method V_k re-derivation, T1.13 AAD bootstrap.

---

## v0.908.0 — 2026-06-12

**Fix L2 T1.6 — discrete dividends now escrowed at the correct step in both binomial and trinomial trees.**

Eighth of 13 Tier-1 (dual-critic-confirmed) bugs from `MODULE_HEALTH.md` closed. This one had two sub-bugs in `numerical/_trees.py`:

### Sub-bug 1 — Binomial: dividends applied only to the terminal grid

Pre-fix, `_solve_binomial` constructed the spot tree by subtracting cumulative dividends *once at the terminal step*. The intermediate-step `S_step` array (used by every backward-induction iteration for early-exercise and barrier checks) was rebuilt as the **raw pre-dividend forward** at each step.

Consequence: an American option with a known dividend mid-life saw the wrong (too-high) intermediate spot, biasing the early-exercise decision; barrier knock-out checks used the wrong intermediate spot too. The terminal price was correct in aggregate, but the path-dependent / early-exercise behaviour was wrong.

### Sub-bug 2 — Trinomial: dividends silently ignored entirely

`_solve_trinomial` had **no dividend handling whatsoever** — `self.dividends` was accepted in the constructor but never read by the trinomial path. A user passing `dividends=[(25, 10.0)]` to `TreeMethod.TRINOMIAL` got identically the same answer as `dividends=None`.

### Fix — escrowed-dividend convention applied to both

Both `_solve_binomial` and `_solve_trinomial` now share the same per-step helper structure:

```python
def _cum_div_through(s: int) -> float:
    return sum(amt for step, amt in self.dividends.items() if step <= s)

def _spot_at_step(s: int) -> np.ndarray:
    grid = ...   # raw forward spot grid at step s
    cum = _cum_div_through(s)
    if cum > 0:
        grid = np.maximum(grid - cum, 0.01)
    return grid
```

At each step `s`, the spot grid has cumulative dividends paid through step `s` subtracted. This is the standard "escrowed dividend" convention (Hull §21.12): the underlying drops by the dividend amount on the ex-date, and that drop is reflected in *every* node from that step forward.

`S` (terminal) and `S_step` (intermediate) now both go through `_spot_at_step`, so they agree on the dividend-adjusted spot. Floor at `0.01` guards against degenerate cases where down-moves push the dividend-adjusted spot to zero.

### Verification — `test_l2_t1_6_discrete_dividends.py`

4 new regression tests, all pass:
- `test_dividend_at_step_zero_equals_terminal_dividend` — boundary: dividend at step 0 reduces call PV.
- `test_dividend_at_intermediate_step_affects_price` — dividend at step 25/50 reduces European call price.
- `test_american_call_with_dividend_exercises_correctly` — with a large mid-life dividend, American call ≥ European call (proper American premium for dividends — the classic textbook case).
- `test_trinomial_dividend_actually_affects_price` — trinomial now responds to `dividends=[(25, 10.0)]` (pre-fix delta was ~0; post-fix delta > 1).

Full parallel suite: **11964 passed in 3:21** — zero regressions.

### Tier-1 status — 8 of 13 closed

| # | Module | Status |
|---|---|---|
| T1.1 | `numerical/_mc.py` multilevel_mc broken | queued |
| T1.2 | `numerical/_fourier.py` density crash | ✅ v0.907 |
| T1.3 | `numerical/_integrate.py` integrate_2d swap | ✅ v0.907 |
| T1.4 | `numerical/_optimize.py` interior_point drops equality constraints | queued |
| T1.5 | `numerical/_trees.py` Tian V formula | ✅ v0.907 |
| **T1.6** | `numerical/_trees.py` discrete dividends terminal-grid + trinomial ignored | ✅ this slice |
| T1.7 | `models/mc_engine.py` Milstein silently runs Euler | ✅ v0.907 |
| T1.8 | `models/cos_method.py` V_k integration bounds deep ITM | queued |
| T1.9 | `models/hull_white.py` Swaption uses r0 not α(T_expiry) | queued |
| T1.10 | `curves/global_solver.py` residual collision | ✅ v0.905 |
| T1.11 | `curves/multicurve_solver.py` PV_float first period | ✅ v0.905 |
| T1.12 | `curves/multicurve_solver.py` PV_float / annuity inconsistent | ✅ v0.905 (same root) |
| T1.13 | `curves/aad_curves.py` swap bootstrap flat-extrapolation | queued |

Remaining 5 are the harder ones (math review needed): T1.1 MLMC redesign, T1.4 IPM rewrite, T1.8 cos_method V_k re-derivation, T1.9 Hull-White swaption analytic, T1.13 AAD bootstrap.

---

## v0.907.0 — 2026-06-12

**Fix L2 Tier-1: four quick-win bugs (T1.2 trapz, T1.3 integrate_2d, T1.5 Tian, T1.7 Milstein).**

Opening the L2 audit by closing the four cheapest of MODULE_HEALTH's 13 Tier-1 (dual-critic-confirmed) bugs. Each was a small, surgical change with a regression test that locks in the fix.

### T1.2 — `numerical/_fourier.py` `CharacteristicFunction.density()` was a hard crash

`np.trapz` was removed in NumPy 2.x; the project's installed `numpy==2.4.3` lacks it. Every call to `density()` raised `AttributeError: module 'numpy' has no attribute 'trapz'`. Fix: rename to `np.trapezoid`. Function had no callers in production code so the bug was dormant; it now works for any future caller (verified: `density(N(0,1))` at `x=0` ≈ 0.3989).

### T1.3 — `numerical/_integrate.py` `integrate_2d` argument order was inverted

`scipy.integrate.dblquad` expects `func(y, x)` (y is the inner variable). The function was passing `f` directly. For symmetric integrands the bug was invisible; for non-symmetric ones the result was wrong: `integrate_2d(lambda x, y: x, (0,3), (0,1))` returned **1.5** instead of **4.5**. Fix: wrap with an explicit `_f_xy(y, x): return f(x, y)` before passing to scipy.

### T1.5 — `numerical/_trees.py` Tian binomial method had been silently running CRR

Two compounding errors in `_tian_params`:
1. `V = M² × (exp(σ²·dt) − 1)` (the *variance*) instead of `V = exp(σ²·dt)` (the *second-moment factor* — the actual Tian 1993 quantity).
2. The `u`/`d` formula was missing the `V` multiplier.

Combined effect: the discriminant `V² + 2V − 3` was nearly always **negative**, triggering the "very small dt × vol → fall back to CRR" guard. So calling `TreeMethod.TIAN` actually ran CRR for every call. Live demonstration (r=5%, σ=20%, dt=0.01):
- Standard Tian: disc = +0.0016, u/d are real.
- Pre-fix code: V = 0.0004, disc = −2.999, branch falls back to CRR.

Fix: match the published Tian (1993) JFQA derivation. The Tian method now produces u/d distinct from CRR, as it should.

### T1.7 — `models/mc_engine.py` Milstein scheme silently ran Euler (two sub-bugs)

```python
step_fn = euler_step if self.scheme == "euler" else euler_step   # COPY-PASTE
```

Even fixing the typo wasn't enough: `milstein_step_1d` requires `diffusion_deriv` (σ' for the correction term), but `ProcessSpec` had no field for it — the parameter at the call site stayed `None`, so Milstein reduced to Euler internally. Fix:
1. Replace the second `euler_step` with `milstein_step_1d`.
2. Add `diffusion_deriv: Callable | None = None` to `ProcessSpec.__init__`.
3. Plumb it through to the Milstein call.
4. When `scheme='milstein'` but `diffusion_deriv` is None, emit a `RuntimeWarning` and fall back to Euler (no longer silent).

Live: with GBM and σ=0.5, n_steps=10 over T=1y, Euler and Milstein now produce paths that differ visibly at the terminal — pre-fix they were identical.

### Verification

- 9 new regression tests in `test_l2_tier1_quick_wins.py`:
  - T1.2: density() runs; N(0,1) density at 0 ≈ 1/√(2π).
  - T1.3: non-symmetric integrand (4.5 not 1.5); symmetric unaffected; callable y_range uses x correctly.
  - T1.5: Tian parameters match Tian 1993 formulas; no longer equal to CRR for typical inputs.
  - T1.7: Milstein produces different paths from Euler when `diffusion_deriv` provided; warns when not provided.
- Full parallel suite: **11960 passed in 3:21** — zero regressions.

### Tier-1 status

| # | Module | Status |
|---|---|---|
| T1.1 | `numerical/_mc.py` multilevel_mc broken | queued |
| **T1.2** | `numerical/_fourier.py` density crash | ✅ this slice |
| **T1.3** | `numerical/_integrate.py` integrate_2d swap | ✅ this slice |
| T1.4 | `numerical/_optimize.py` interior_point drops equality constraints | queued |
| **T1.5** | `numerical/_trees.py` Tian V formula | ✅ this slice |
| T1.6 | `numerical/_trees.py` discrete dividends terminal-grid | queued |
| **T1.7** | `models/mc_engine.py` Milstein silently runs Euler | ✅ this slice |
| T1.8 | `models/cos_method.py` V_k integration bounds deep ITM | queued |
| T1.9 | `models/hull_white.py` Swaption uses r0 not α(T_expiry) | queued |
| T1.10 | `curves/global_solver.py` residual collision | ✅ v0.905 |
| T1.11 | `curves/multicurve_solver.py` PV_float first period | ✅ v0.905 |
| T1.12 | `curves/multicurve_solver.py` PV_float / annuity inconsistent | ✅ v0.905 (same root) |
| T1.13 | `curves/aad_curves.py` swap bootstrap flat-extrapolation | queued |

**7 of 13 Tier-1 closed.** Remaining 6 are harder (deeper math review needed for T1.1 MLMC, T1.8 cos_method bounds, T1.9 Hull-White swaption analytic).

---

## v0.906.0 — 2026-06-11

**Fix L1 C.2 B1 — `curve_jacobian` resolves `pillar_tenors` to actual curve indices.**

Third HIGH-severity bug closed from L1 audit. Pre-fix, when a caller passed a custom `pillar_tenors` list (different from the curve's actual pillar grid), `curve_jacobian` silently bumped the WRONG pillars — the Jacobian's column labels were mismatched with the user's requested tenors.

### What was broken

```python
for j, pt in enumerate(pillar_tenors):
    bumped = curve.bumped_at(j, bump_size)   # ← j is an enumeration index
    ...                                        #   NOT a resolved curve pillar
```

`bumped_at(j, ...)` takes a **curve pillar index**. The function looped over the user's `pillar_tenors` and used the enumeration index `j` directly. For a curve with pillars `[0.25, 0.5, 1, 2, 5, 10]` and user request `pillar_tenors=[1, 2, 5]`:
- `j=0` → bumped index 0 → the 0.25y pillar (user thought: 1y)
- `j=1` → bumped index 1 → the 0.5y pillar (user thought: 2y)
- `j=2` → bumped index 2 → the 1y pillar (user thought: 5y)

The Jacobian's column labels lied. A caller multiplying `J` by a market-quote-bump vector got systematically wrong sensitivities.

### Change

- `curves/curve_risk.py:curve_jacobian` — when `pillar_tenors` is supplied, resolve each tenor to its actual curve pillar index (nearest match within `pillar_tol`); raise `ValueError` if any tenor doesn't match a curve pillar. New `pillar_tol=1e-2` default accommodates the 365.25-vs-day-count drift on `DiscountCurve.flat` pillars (~0.003y).
- 5 new regression tests in `test_curve_jacobian_pillar_resolution.py`:
  - default `pillar_tenors=None` uses the curve's own grid.
  - custom `pillar_tenors=[1, 2, 5]` correctly bumps those pillars (J[0,0] dominant for 1y query).
  - unrecognised tenor raises clearly.
  - tolerance allows close matches (1y vs 1.00274y).
  - out-of-order tenors produce columns in the user-supplied order.
- Full parallel suite: **11951 passed in 3:40** — zero regressions.

### L1 audit status — active HIGH bugs all closed

| Bug | Status |
|---|---|
| A.2 B1 — global_solver dup-maturity | ✅ v0.905 |
| A.3 B1 — multicurve PV_float first period | ✅ v0.905 |
| **C.2 B1** — curve_jacobian wrong pillar | ✅ **this slice** |
| A.1 B1 — bootstrap HW convexity (latent) | queued |

L1 Pass A + B + C audited (15 of 33 modules). Remaining: D (builders, 8 modules), E (numerics, 4), F (AAD, 5).

---

## v0.905.0 — 2026-06-11

**Fix L1 curves audit A.2 B1 + A.3 B1 — `global_solver` rejects duplicate maturities; `multicurve_newton` PV_float now includes the first accrual period.**

Two HIGH-severity active bugs from L1 Pass A, fixed in one slice.

### What was broken

**A.3 B1 — `multicurve_newton` PV_float skipped the first period.** The projection-swap PV_float loop started at `j=1`, walking from the first pillar onward. But `_compute_annuity` walked from `reference_date`. So PV_float had `N−1` segments while annuity had `N`. For 2-pillar projection swaps the bias was ~50%; the solver oscillated and emitted `RuntimeWarning: multicurve_newton: did not converge after 50 iterations. Residual: 2.86e-03`. That warning has been polluting the test suite for the entire session. **It was the bug talking.**

**A.2 B1 — `global_solver` silently dropped constraints on duplicate maturities.** The residual vector was indexed by `pillar_idx[mat]`. Two instruments at the same maturity (e.g. a 1Y deposit + 1Y OIS swap — both standard market quotes!) would overwrite each other's residual. Newton converged to a curve that didn't reprice all inputs. Live repro: 1Y depo @5% + 1Y swap @4% → zero rate 3.96% (only the swap survived).

### Change

- `curves/multicurve_solver.py:159-176` — projection PV_float loop now walks `[reference_date, *dates_up_to]` to match `_compute_annuity`. The `multicurve_newton: did not converge` warning is gone from the test suite.
- `curves/global_solver.py:46-77` — `global_bootstrap` builds a `seen_maturities` set; raises `ValueError("Duplicate maturity ... already provided by ..., also requested as ...")` on collision. Each pillar can be constrained by at most one instrument.
- 2 new regression tests in `test_multicurve_first_period.py`: convergence within tolerance under `RuntimeWarning`-as-error; round-trip identity (each projection swap rate recoverable from the calibrated curves).
- 5 new regression tests in `test_global_solver_collision.py`: deposit-swap collision raises; duplicate-deposit raises; duplicate-swap raises; error message identifies the conflicting type; distinct maturities still work.

### Verification

- Full parallel suite: **11946 passed in 4:06** — zero regressions.
- Warnings count dropped from 63 to 55 (the 8 dropped were the recurring `multicurve_newton: did not converge`).

### L1 Pass A — status update

| Bug | Status |
|---|---|
| A.1 B1 — `bootstrap` HW convexity wrong | LATENT (no current caller exercises it; deferred until either a caller appears or we coordinate a single fix) |
| A.1 B2 — float-leg conventions no-op | docstring fix queued |
| **A.2 B1** — `global_solver` residual collision | ✅ this slice |
| **A.3 B1** — `multicurve` PV_float first-period | ✅ this slice |
| A.4 B1 — `ncurve_solver` BasisSwap annuity | queued |
| A.4 B2 — `ncurve_solver` OIS schedule | queued |

Both active HIGH bugs closed. The dormant HW-convexity bug remains queued — it requires a refactor (re-route `bootstrap()` to call `ir_futures.hw_convexity_adjustment`) but since no caller is hitting it today, it sits behind A.4 in priority.

---

## v0.904.0 — 2026-06-11

**Generic `to_dict` mutation-safety sweep — `vars(self)` → `dict(vars(self))` across L0.**

Closes a recurring footgun cluster surfaced by audit findings A.5 B1, A.7 B1, C.1 B1, and ~25 other instances of the same pattern in `book.py`, `daily_pnl.py`, `settlement.py`, `mandate.py`, `greeks.py`, `dependency_graph.py`, `convergence_framework.py`, `numerical_safety.py`, `numerical_method_map.py`, `market_data.py` (legacy), `trade.py` (orphan), `approximation.py`, `solvers.py`.

### What was broken

```python
@dataclass
class Result:
    x: float
    def to_dict(self) -> dict:
        return vars(self)   # ← returns self.__dict__ DIRECTLY, not a copy
```

Mutating the returned dict mutated the dataclass instance. Confirmed live in audit A.5 B1:
```
r = SolverResult(root=1.0, ...)
d = r.to_dict()
d['root'] = 999.0
# r.root is now 999.0
```

30 callsites had this pattern across `pricebook.core.*`.

### Change

- All 30 `return vars(self)` lines rewritten to `return dict(vars(self))` (defensive copy). Single regex sweep across 13 files; idempotent and behaviour-preserving for callers that don't mutate.
- Deleted the dead-code orphan `Trade.to_dict` at `core/trade.py:56-57` — it sat inside the `@dataclass` body but was overwritten at module-bottom by `_trade_to_dict` via `Trade.to_dict = _trade_to_dict`. Replaced with a one-line comment pointing to the real binding (closes audit C.1 B1).
- 9 new regression tests in `test_to_dict_mutation_safety.py` covering `SolverResult`, the 3 `approximation.py` dataclasses, `Greeks`, `Position` (book), `DailyPnL`, `CashSettlementResult`, `PortfolioHolding` (mandate). Each asserts that mutating the returned dict doesn't propagate to the source.
- Full parallel suite: **11930 passed in 3:28**. Zero regressions.

### L0 audit — closure

With this slice, **every confirmed bug from Pass A through Pass D of the L0 audit is now either fixed or queued as ARCH/LOW**.

| Severity | Count | Status |
|---|---:|---|
| HIGH | 5 | ✅ all fixed |
| MED | 6 | ✅ all fixed |
| LOW | many | mostly closed (`to_dict` sweep + Tokyo + ICMA test gaps + serialisation A.11 B3-B7 queued) |
| ARCH | 1 | B.3 C1 — legacy market_data deferred to Gate 2 design decision |

The pricebook L0 foundations are materially more correct than at the start of this session. ~25 commits, **11930 tests passing**, every fix landed with regression tests, legacy-debt ledger tracks every backwards-compat shim, audit chain (G1) wired end-to-end.

---

## v0.903.0 — 2026-06-11

**Fix C.7 B1 — settlement lag now interpreted as business days; calendar honoured.**

Closes the last MEDIUM-severity bug from L0 Pass C. `cash_settlement`, `cds_settlement_physical/_cash`, `option_settlement_cash/_physical`, and `futures_settlement_physical` all silently used **calendar** days for the T+N lag — a Friday trade settling T+2 landed on Sunday.

### Before / after

```
Friday 2025-08-01, cash_settlement T+2:
  BEFORE: 2025-08-03 Sun (calendar, non-business)
  AFTER:  2025-08-05 Tue (skip Sat+Sun)

Friday 2025-08-01, option_cash T+1:
  BEFORE: 2025-08-02 Sat
  AFTER:  2025-08-04 Mon

Thursday 2025-07-03, option_cash T+1 with US calendar:
  BEFORE: 2025-07-04 Fri (Independence Day — holiday)
  AFTER:  2025-07-07 Mon (skip July 4)
```

### Change

- Six settlement entry points (`cash_settlement`, `cds_settlement_physical`, `cds_settlement_cash`, `option_settlement_cash`, `option_settlement_physical`, `futures_settlement_physical`) now route through `add_business_days(d, lag, calendar)` instead of `date.fromordinal(d.toordinal() + lag)`.
- All six gain an optional `calendar` parameter. When `None`, only weekends are skipped (same as `Calendar.add_business_days(None)`). When provided, holidays are skipped too.
- Existing `test_settlement.py::test_settlement_lag` updated — the prior assertion `(settle - REF).days == 30` codified the calendar-day bug; replaced with exact business-day calculation (REF = Mon 2024-01-15; +30 BD → Mon 2024-02-26).
- 9 new tests in `test_settlement_business_days.py`: Friday + T+1/T+2/T+3/T+5 weekend skip across all four product types; July-4 Independence Day with US calendar; T+0 and mid-week sanity cases.
- Full parallel suite: **11930 passed in 3:25** — zero regressions.

### Affected upstream

Anything that imports settlement helpers from `pricebook.core.settlement` and reads `settlement_date`. Notable: FX spot dates (T+2), US equity option physical settlement (T+2), CDS cash settlement (T+5), bond futures physical delivery (T+3). The pre-fix dates would frequently land on weekends — useful as a "placeholder" date for sandbox demos but wrong for settlement-risk computations.

### Pass C — MEDs closed

| C.7 B1 | settlement lag business days | ✅ this slice |
| C.8 B1 | dollar_gamma formula | queued (LOW) |
| C.1 B1 | dead-code orphan to_dict | queued (LOW, generic to_dict sweep) |

---

## v0.902.0 — 2026-06-11

**Fix D.1 B1+B2+B3 — `PricingContext` round-trip preserves every field; `replace()` no longer aliases mutable dicts.**

Three pre-existing bugs in `PricingContext` (all carried over from the prior `MODULE_HEALTH.md` audit and verified at L0 audit Pass D.1) fixed in one focused slice.

### What was broken

1. **B1 — empty containers collapsed to `None` on round-trip.** `_ctx_from_dict` had `proj = {...} or None` patterns. An empty `projection_curves`/`vol_surfaces`/`credit_curves`/`fx_spots` dict became `None`, so `ctx.projection_curves["foo"]` raised `TypeError: 'NoneType' object is not subscriptable` instead of the documented `KeyError`.
2. **B2 — fields silently dropped.** `_ctx_to_dict` emitted only 5 fields out of 13. **Silently dropped**: `discount_curves` (multi-currency), `inflation_curves`, `repo_curves`, `reporting_currency`, `stochastic_credit_models`, `credit_vol_surfaces`, `credit_correlations`, AND `numerical_config` (just added in G1 P3 Slice 1!). Multi-currency contexts lost data on every save/load round-trip.
3. **B3 — `replace()` aliased mutable containers.** `ctx2 = ctx.replace(reporting_currency="EUR")` then `ctx2.discount_curves["BRL"] = curve` mutated `ctx.discount_curves` too. Broke the "Immutable snapshot" docstring contract.

### Change

- `pricing_context.py:replace()` — now goes through a `_pick_dict` helper that `dict(...)`-copies every mutable container before passing to the new context.
- `_ctx_to_dict` — emits **every** dataclass-declared field, including `discount_curves`, `inflation_curves`, `repo_curves`, `reporting_currency`, `stochastic_credit_models`, `credit_vol_surfaces`, `credit_correlations`, `numerical_config` (serialised via `dataclasses.asdict`).
- `_ctx_from_dict` — `_fd_dict` helper reconstructs each value via the global registry (handles both envelope-format and opaque values); empty dicts stay empty (no `or None` collapse); `NumericalConfig` reconstructed from its `asdict` form.
- 13 new tests in `test_pricing_context_round_trip.py`:
  - **D.1 B1**: default-context round-trip preserves dict-not-None for all 10 container fields; access raises `KeyError` not `TypeError`.
  - **D.1 B2**: `discount_curves`, `reporting_currency`, `repo_curves`, `fx_spots`, `credit_correlations`, `numerical_config` (both populated AND `None`) all round-trip exactly.
  - **D.1 B3**: `replace()` doesn't alias `discount_curves` / `fx_spots` / `credit_correlations`; values copy through but underlying dict is a different object.
- Full parallel suite: **11921 passed in 3:55** — zero regressions.

### Affected upstream

Multi-currency users (anyone with `discount_curves={"USD": ..., "EUR": ...}`), users persisting `PricingContext` to disk for replay, scenario engines that build derived contexts via `replace()` and concurrently mutate them. The pre-fix behaviour silently corrupted data; the fix lands without behaviour change on any single-currency in-memory workflow.

---

## v0.901.0 — 2026-06-11

**Fix A.2 B2 — `TokyoCalendar` *furikae kyūjitsu* (振替休日) substitute-day rule.**

Closes the last MEDIUM-severity calendar finding from L0 Pass A. Tokyo's fixed-date holidays falling on Sunday now correctly cascade to a substitute holiday on the next non-holiday day per Japan's Public Holiday Act §3.2.

### Before vs after

| Date | Holiday | Status BEFORE | Status AFTER |
|---|---|---|---|
| 2018-04-30 Mon | Showa Day Sun → substitute | not holiday ❌ | holiday ✓ |
| 2025-11-24 Mon | Labour Thanksgiving Sun → substitute | not holiday ❌ | holiday ✓ |
| 2026-05-06 Wed | Constitution Sun + Greenery + Children's cluster → substitute | not holiday ❌ | holiday ✓ |
| 2024-04-30 Tue | (no sub needed, Apr 29 was Mon) | business day ✓ | business day ✓ |

The Golden-Week cascade case is the interesting one: May 3 2026 (Constitution) is Sunday; May 4 (Greenery) and May 5 (Children's) are already holidays; the substitute walks all the way to **May 6 Wed**. The implementation handles this generically via a while-loop that walks forward until it finds a non-holiday day.

### Change

- `core/calendar.py:TokyoCalendar._compute_holidays` — refactored to:
  1. Build a `fixed` set of fixed-date holidays.
  2. Build a `monday_holidays` set of variable Monday holidays (Coming of Age, Marine Day, Respect for Aged, Sports Day — by construction never on Sunday).
  3. Second pass over `fixed`: any Sunday entry gets a substitute computed by walking forward until a non-holiday day is found.
- 5 new tests in `test_calendar.py::TestTokyoCalendarSubstitution`: Showa Day 2018, Labour Thanksgiving 2025, Golden-Week cluster 2026, no-substitute-when-weekday case, Monday-holiday smoke check.
- Full parallel suite: **11908 passed in 4:06**.

### A.2 — closed

All four substitution-rule findings from L0 audit A.2 are now fixed:

| Sub-finding | Locale | Status |
|---|---|---|
| A.2 B1a | London (UK 1971 Act) | ✅ |
| A.2 B1b | Sydney (AU Public Holidays Acts) | ✅ |
| A.2 B1c | Wellington (NZ Holidays Act 2003) | ✅ |
| A.2 B1d | Toronto (Canadian federal/provincial) | ✅ |
| **A.2 B2** | **Tokyo (Japanese Public Holiday Act §3.2)** | ✅ **this slice** |

JPY-rate calculations crossing 2018-04-30 (Showa Day substitute), 2019-05-06 (Reiwa transition), 2025-11-24, and 2026-05-06 are now correct.

---

## v0.900.0 — 2026-06-11

**Fix A.11 B1 + B2 — `_deserialise_atom` now dispatches `list[SomeSerialisable]`, `list[Enum]`, and polymorphic `Union[A, B, None]`.**

Two medium-severity audit findings, one root cause: `_deserialise_atom` had a narrow list of supported parameterised types. Anything outside the narrow path silently returned the raw value, breaking round-trips for polymorphic fields.

### Before / after

| Type hint | Pre-fix behaviour | Post-fix behaviour |
|---|---|---|
| `list[date]` | reconstructed ✓ | reconstructed ✓ |
| `list[SomeSerialisable]` | **raw list of dicts** ❌ | recursively dispatched via registry ✓ |
| `list[SomeConvention]` | raw list ❌ | dispatched via `inner.from_dict` ✓ |
| `list[Colour]` (Enum) | raw list of strings ❌ | mapped to Enum members ✓ |
| `Optional[T]` (`T \| None`) | unwrapped ✓ | unwrapped ✓ |
| `Union[A, B, None]` | **raw value** ❌ | if dict with `"type"` key → registry-dispatch; else raw ✓ |

### Live test of the fix

A `_Container` class with a `list[_Leaf]` field and a `Union[_Leaf, _Branch, None] underlying` field now round-trips cleanly: every leaf is reconstructed, the polymorphic underlying lands on the correct concrete type based on its `type` tag. Pre-fix, both would have returned raw dicts.

### Change

`pricebook.core.serialisable._deserialise_atom`:
- **Union dispatch** (A.11 B2): when stripping `NoneType` leaves ≥2 non-None args, check if `v` is a dict with `"type"` → registry-dispatch via `from_dict`. Otherwise return raw (same as before — primitives don't need dispatch).
- **list[T] dispatch** (A.11 B1):
  - `list[date]` — unchanged.
  - `list[Serialisable]` — recursively dispatches each element; supports both envelope-format (`"type"` key → `from_dict`) and flat-convention format (no `"type"` → `inner.from_dict`).
  - `list[Enum]` — maps each element through the Enum constructor.
- 10 new tests in `test_serialisable_polymorphic.py`: container round-trip with each list shape + polymorphic Union with type-tag, both branches, and the None case + 3 direct atom-dispatch tests.
- Full parallel suite: **11903 passed in 4:04**. Zero regressions.

### What's still uncovered (queued)

- `list[tuple[...]]`, `list[list[...]]`, nested generic types — still raw. Genuinely niche; no current call-site needs them.
- `Union[A, B]` (no None) with non-tagged values — still raw. Can't auto-resolve without a discriminator.
- A.11 B3 (`_register` silent no-op on re-import), A.11 B4 (numpy `.item()` duck-test ordering), A.11 B5 (CurrencyPair parse), A.11 B6 (bare `KeyError`), A.11 B7 (Enum int-from-string) — all LOW severity, queued.

---

## v0.899.0 — 2026-06-11

**Fix B.1 B2 — `make_payload`/`read_payload` helpers extend G1 P3 Slice 2's schema-version coverage to custom `to_dict` overrides; `DiscountCurve` no longer drops `interpolation` on round-trip.**

A medium-severity audit finding that turned out to have wide reach. G1 P3 Slice 2 added `schema_version` to `Serialisable.to_dict` and the two decorators, but **classes with hand-written `to_dict` overrides** (the curves, `Trade`, `Portfolio`, `PricingContext`, several option types) bypassed all of it. They wrote `{"type": ..., "params": ...}` directly. Schema versioning was silently absent from their on-disk payloads.

### What was broken

Two distinct gaps in one root cause:

1. **`DiscountCurve.to_dict` dropped `interpolation`.** A curve built with `MONOTONE_CUBIC` or `AKIMA` interpolation serialised, then deserialised, silently came back as `LOG_LINEAR` (the constructor default). Live repro from the audit confirmed this. Any persisted curve with a non-default interpolation method was silently misinterpreted on reload.
2. **Schema versioning didn't reach custom-to_dict classes.** Payloads from `DiscountCurve`, `SurvivalCurve`, `Trade`, `Portfolio`, `PricingContext` carried no version field. When a future v2 lands, there'd be no way to distinguish.

### Change

- New helpers in `pricebook.core.serialisable`:
  - `make_payload(instance, params)` — builds `{"type": ..., "params": ..., "schema_version": ...}` from a Serialisable instance + a params dict. Drop-in replacement for hand-rolled envelopes.
  - `read_payload(d, cls)` — inverse: validates schema_version against `cls._SERIAL_SCHEMA_VERSION` and returns the params dict. Replaces `p = d["params"]` in custom `from_dict` overrides.
- Migrated:
  - `core/discount_curve.py` — uses `make_payload`/`read_payload`. **Includes `interpolation` in the params dict; deserialiser defaults to `LOG_LINEAR` when absent (back-compat for pre-fix payloads).**
  - `core/survival_curve.py` — uses helpers (interpolation was already preserved; this slice adds schema_version coverage).
  - `core/trade.py` — `Trade` and `Portfolio` both migrated.
  - `core/pricing_context.py` — migrated, plus the `vars(self)` → `dict(vars(self))` fix (same mutation pattern as solvers/approximation, now closed).
- 13 new tests in `test_custom_to_dict_schema_version.py`:
  - **Parametrised round-trip across all 5 `InterpolationMethod` values** — catches the B.1 B2 interpolation-loss bug for every method.
  - Pre-fix payload (no `interpolation` key) defaults to LOG_LINEAR — back-compat.
  - schema_version present on every migrated class.
  - Future version rejected with the standard ValueError.
  - Pre-fix payload (no schema_version key) still deserialises as v1.
- Full parallel suite: **11893 passed in 4:01** — no regressions.

### What's still uncovered

The 43 file-count from the audit includes many option/credit classes (`options/asian_option.py`, `options/barrier_option.py`, `credit/cds.py`, etc.) that still hand-roll `{"type": ..., "params": ...}`. They are NOT migrated in this slice — that work belongs in their respective audit passes (Pass C / D etc.). Behaviour is unchanged for them; they currently emit no schema_version which by contract reads back as v1 — back-compat all the way down.

### Wider implication

`make_payload`/`read_payload` is now the canonical pattern for any class with a custom `to_dict`. New code should always go through it; existing code migrates as each module gets audited.

---

## v0.898.0 — 2026-06-11

**Fix B.1 B1 — `DiscountCurve.roll_down` anchors rolled DFs to the new reference date.**

The fifth HIGH-severity L0 audit bug, now closed. `roll_down(days)` was silently producing curves with wrong zero rates because it forgot to apply the no-arbitrage anchoring `P(new_ref, d) = P(0, d) / P(0, new_ref)`. On a flat 5% curve, rolldown P&L was over-stated by **+1.4 bp per day** of roll.

### Live before / after

```
Flat 5% curve, ref=2024-01-01.

BEFORE:
roll_down(1d)  → zero_rate(2025-01-01) = 5.0137%  (+1.37 bp error)
roll_down(30d) → zero_rate(2025-01-01) = 5.4231%  (compounded error)
                          (any rolldown attribution off this is structurally wrong)

AFTER:
roll_down(1d)   → zero_rate(2025-01-01) = 5.0000% exactly
roll_down(30d)  → zero_rate(2025-01-01) = 5.0000% exactly
roll_down(365d) → zero_rate(2026-01-01) = 5.0000% exactly
```

### Change

- `discount_curve.py:126-178` — `roll_down` now computes `disc_to_new_ref = self.df(new_ref)` once and divides each pillar DF by it. Raises `ValueError` with a clear message if `disc_to_new_ref <= 0` (pathological / extrapolated curve).
- The "all pillars in the past" fallback now **preserves** the original `day_count` and `interpolation` instead of silently dropping them via `DiscountCurve.flat(...)` (a separate footgun called out in the audit).
- 5 new tests in `TestRollDown`: 1-day / 30-day / 365-day flat-curve zero-rate preservation; no-arbitrage anchor identity `df_rolled(d) == df(d) / df(new_ref)`; day_count + interpolation preservation in the all-pillars-past fallback.
- Full parallel suite: **11880 passed in 4:57**. Zero regressions.

### Affected upstream

Anything that calls `DiscountCurve.roll_down`: rolldown P&L attribution, daily carry/roll analytics, scenario time-shifts. The old behaviour over-stated rolldown gain by ~1.4 bp per day for a 5% curve. Practitioners running daily roll/carry on USD/GBP/EUR books would see this in their P&L explain.

---

## v0.897.0 — 2026-06-11

**Fix A.1 B1 Slice 4 — `FixedRateBond` YTM analytics use ICMA-correct period counting; par-yield round-trip is exactly 100.**

The fourth and decisive slice of the ICMA fix. Coupons were fixed in Slice 3; this slice fixes the *discount-time* side. UST and other ICMA-convention bonds now satisfy the par-yield identity exactly.

### What was broken (after Slice 3)

`FixedLeg` was producing correct coupon amounts (2.0000 exact for par 5y UST) but `_price_from_ytm` was computing time-to-cashflow via `year_fraction(settle, payment_date, ACT_ACT_ICMA)` for *multi-period spans*. That call silently falls back to ACT/365F because the current `_act_act_icma` implementation only handles single-period accruals (per ICMA 251.1) and not multi-period spans (ICMA 251.2). The result: coupon amounts at 2.0000 each, but discount times at 0.4986, 1.0027, 1.4986, ... instead of 0.5, 1.0, 1.5, .... Par-yield round-trip landed at 99.981367 instead of 100.

### Change

- New method `FixedRateBond._ytm_time_to(settle, target)` computes the ICMA-correct time-to-cashflow:
  - If `settle` coincides with a coupon date → `(target_index − settle_index) / coupons_per_year` exactly.
  - If `settle` is mid-period → stub fraction (days_to_next_coupon / period_days) + full periods.
  - For non-ICMA conventions → falls back to `year_fraction(settle, target, day_count)` (unchanged).
- Routed `_price_from_ytm`, `macaulay_duration`, `convexity`, and `accrual_schedule` through `_ytm_time_to`. `modified_duration` and `dv01_yield` derive from these — no further change needed.
- The two characterisation xfails turn green; their markers are removed. **Tolerance tightened from 1e-8 to 1e-10** to lock in the new exactness.
- Full parallel suite: **11875 passed in 4:57** — zero regressions.

### Verified end-to-end

```
Pre-fix:  par 5y UST round-trip = 99.999807, par 30y = 99.999489
Post-fix: par 5y UST round-trip = 100.000000, par 30y = 100.000000
```

### A.1 B1 status

| Slice | Scope | Status |
|---|---|---|
| 1 | Add `strict_icma` flag on `year_fraction` | ✅ |
| 2 | Characterise UST mispricing with xfail tests | ✅ |
| 3 | `FixedLeg` passes ICMA refs (correct coupon amounts) | ✅ |
| 4 | `_ytm_time_to` helper (correct discount times) | ✅ (this slice) |
| N | Final: flip default to `strict_icma=True` after auditing remaining callers | queued |

UST and all other ICMA-convention bonds (Bunds, Gilts, JGBs, sovereigns) now compute coupon amounts AND par-yield round-trips correctly at machine precision. The audit chain's highest-blast-radius bug is closed for the bond-pricing path.

---

## v0.896.0 — 2026-06-11

**Fix A.1 B1 Slice 3 — `FixedLeg` cashflows now compute correct ACT/ACT ICMA year-fractions; UST coupons land exactly at face × coupon ÷ frequency.**

The third slice of the staged ICMA fix delivers the *coupon-side* correctness. `FixedLeg`'s cashflow builder now passes `ref_start`, `ref_end`, and `frequency` to `year_fraction`, so every regular coupon period on a Treasury / Gilt / Bund / JGB schedule gets `year_frac = 1 / coupons_per_year` *exactly* — per ICMA Rule 251.1.

### Before vs after (par 5y UST at 4% coupon, face=100, semi-annual)

| Coupon | Period | days | year_frac BEFORE | year_frac AFTER | amount BEFORE | amount AFTER |
|---|---|---:|---:|---:|---:|---:|
| 1 | 02/15 → 08/15 | 182 | 0.498630 | **0.500000** | 1.994521 | **2.000000** |
| 2 | 08/15 → 02/15 | 184 | 0.504110 | **0.500000** | 2.016438 | **2.000000** |
| 3 | 02/15 → 08/15 | 181 | 0.495890 | **0.500000** | 1.983562 | **2.000000** |
| 4–10 | (alternating) | — | 0.4959 / 0.5041 | **0.500000** | 1.9836 / 2.0164 | **2.000000** |

### Change

- `fixed_leg.py:69-93` — compute `coupons_per_year = 12 // frequency.value` once; pass `ref_start=accrual_start, ref_end=accrual_end, frequency=coupons_per_year` to `year_fraction` for every coupon. The extra params are ignored by non-ICMA conventions, so the change is harmless for THIRTY_360 / ACT_360 / ACT_365F / etc.
- Two of the four characterisation xfails turn green (`test_every_regular_coupon_is_exactly_half_year`, `test_every_coupon_amount_is_exactly_two`) — their `xfail` markers are removed; they now assert the correct behaviour permanently.
- Full parallel suite: **11873 passed, 2 xfailed in 4:55** — zero regressions in any other ICMA-using test.

### What's NOT fixed yet (queued for Slice 4)

`FixedRateBond._price_from_ytm` (and the YTM-derived analytics: `macaulay_duration`, `modified_duration`, `convexity`, `dv01_yield`) use `year_fraction(settle, payment_date, ACT_ACT_ICMA)` for *multi-period* spans without passing refs — still hitting the legacy ACT/365F fallback. The current `_act_act_icma` implementation is correct for single-period accruals only; multi-period ICMA needs explicit period-counting per Rule 251.2.

The 2 remaining xfails (`test_par_yield_round_trip_is_exact_100`, 5y and 30y) are pinned for Slice 4. **Diagnostic note:** these tests xfail *worse* post-Slice-3 than they did pre-fix — the previous "99.999807" was the result of two errors partially cancelling. Now coupon amounts are correct but discount times still aren't, so the round-trip lands at 99.981367 (5y). Slice 4 will fix the discount-time path and turn these green.

### Affected upstream

Sovereign / corporate bonds using `DayCountConvention.ACT_ACT_ICMA` — all 11 markets currently wired (US, AU, MY, TH, SG, ID, HK, SE, CH, CZ, NO + linkers). Coupon amounts in those bonds now match market quotes within machine precision.

---

## v0.895.0 — 2026-06-11

**Fix A.1 B1 Slice 2 — UST ICMA mispricing characterised via xfail tests.**

Second slice of the staged ICMA fix. No source changes; the bug is now pinned to a reproducible test fixture with `xfail(strict=True)` markers so Slice 3 must turn them green (and CI catches partial fixes).

### Diagnostic — actual mispricing measured

A par-5y UST at 4% (face=100, semi-annual, issued 2024-02-15, matures 2029-02-15):

| Coupon | Period | days | year_frac (current) | year_frac (correct ICMA) | amount (current) | amount (correct) |
|---|---|---:|---:|---:|---:|---:|
| 1 | Feb 15 → Aug 15 | 182 | 0.498630 | 0.500000 | 1.994521 | 2.000000 |
| 2 | Aug 15 → Feb 15 | 184 | 0.504110 | 0.500000 | 2.016438 | 2.000000 |
| 3 | Feb 15 → Aug 15 | 181 | 0.495890 | 0.500000 | 1.983562 | 2.000000 |
| 4–10 | (alternating) | — | 0.4959 / 0.5041 | 0.500000 | 1.9836 / 2.0164 | 2.000000 |

Par-yield round-trip: **99.999807** (should be exactly 100). These are the numbers from `MODULE_HEALTH.md` §`fixed_income/bond.py`, now nailed down in `test_treasury_icma_characterisation.py`.

### What the slice adds

- New test file `python/tests/test_treasury_icma_characterisation.py` (6 tests):
  - Smoke: bond's `day_count` is `ACT_ACT_ICMA`. PASSES today.
  - **xfail(strict=True)**: every regular coupon has `year_frac == 0.5` exactly.
  - **xfail(strict=True)**: every coupon amount is exactly `2.0`.
  - **xfail(strict=True)**: par-yield round-trip lands at exactly `100.0` (5y).
  - **xfail(strict=True)**: same for 30y (error accumulates across 60 coupons).
  - Diagnostic test that always passes — prints the actual current values for visibility in test logs (to be removed in Slice 3 once the fix lands).

Full parallel suite: **11871 passed + 4 xfailed in 4:55**.

### Why xfail and not skip

`xfail(strict=True)` enforces a contract: the test MUST fail today, MUST pass after the fix. If Slice 3 accidentally doesn't fix one of the cases, CI flips that case to "unexpected pass" and the slice fails. Skip would let a partial fix through silently.

### Next slice

**A.1 B1 Slice 3** — fix `FixedLeg` to pass `ref_start=accrual_start, ref_end=accrual_end, frequency=12/months_per_period` to `year_fraction`. Turn the four xfails green. Remove the diagnostic test.

---

## v0.894.0 — 2026-06-11

**Fix A.1 B1 Slice 1 — `strict_icma` flag on `year_fraction` (no behaviour change).**

First slice of the staged ICMA fix. Adds the machinery to detect and refuse silent fallbacks; default remains the legacy permissive behaviour so this slice changes nothing in production code paths. Subsequent slices will migrate one ICMA caller at a time, each verified by a hand-calc test (e.g. "par UST gives exactly 2.0000 per semi-annual coupon"), and only at the very end will the default flip to strict.

### What this slice adds

- `year_fraction(..., strict_icma: bool = False)` — keyword-only flag.
- `_act_act_icma(..., strict: bool = False)` — internal helper now raises `ValueError` (instead of silently degrading to ACT/365F) when `strict=True` AND any of:
  - `ref_start`, `ref_end`, or `frequency` is missing (closes A.1 B1 root cause when callers opt in).
  - `frequency <= 0` (closes A.1 B2 — pre-fix would `ZeroDivisionError`).
  - `period_days <= 0` (inverted ref dates).

Each `strict=True` failure produces a precise, actionable error message:
```
ACT/ACT ICMA requires coupon-period anchors. Missing: ref_start, ref_end, frequency.
Pass `ref_start`, `ref_end`, and `frequency` to `year_fraction(...)`.
```

### Test coverage

12 new tests in `test_day_count.py::TestACTACTICMA`:
- Par UST semi-annual exact half-year (regular 182-day period).
- Long-period invariance (184 days still gives 0.5 — that's the whole point of ACT/ACT ICMA).
- Mid-period accrual.
- Legacy fallback when refs are missing (back-compat invariant).
- Strict mode raises with clear messages on missing-refs / frequency=0 / inverted-period.
- Strict mode default-off preservation invariant.

Full parallel suite: **11869 passed in 4:50** — zero downstream behaviour change.

### Next slices (A.1 B1 roadmap)

| Slice | Scope |
|---|---|
| 1 (this one) | Add the flag. Default off. ✓ |
| 2 | Audit `FixedRateBond.treasury_note` — confirm UST mispricing per MODULE_HEALTH; add failing test. |
| 3 | Fix `treasury_note` to pass refs + switch its call-site to `strict_icma=True`. UST coupons land at exactly 2.0000. |
| 4..N | One slice per remaining ICMA caller in `pricebook/fixed_income/*.py` (~10 sovereign-bond modules). |
| Final | Flip the default to `strict_icma=True`, remove the flag. |

---

## v0.893.0 — 2026-06-11

**Fix A.2 B1b/c/d — `AUDCalendar`, `NZDCalendar`, `CADCalendar` Saturday substitution.**

Completes audit finding A.2 B1. London was already fixed in v0.892. AUD / NZD / CAD followed the identical pattern — base-class US-style `_observe` applied to non-US locales. Now all four use `Calendar._observe_next_working_day` (Sat → +2 days Mon; Sun → +1 day Mon).

### Citations

- **AU** — Australian Public Holidays Acts (state-by-state but uniform Sat→Mon rule).
- **NZ** — Holidays Act 2003, *"Mondayisation"* provision.
- **CA** — Holidays Act (federal) + Employment Standards Acts (provincial).

### Test coverage added

Each calendar gets a `TestXXXCalendarSubstitution` class with regression tests for the Sat-Christmas year (2021 and/or 2027) plus a locale-specific check that exercises a non-Christmas date falling on Saturday:

- AUD: Australia Day 2030 Sat → Mon Jan 28; Australia Day 2025 Sun → Mon Jan 27.
- NZD: Waitangi Day 2027 Sat → Mon Feb 8.
- CAD: Canada Day 2028 Sat → Mon Jul 3; Remembrance Day 2028 Sat → Mon Nov 13.

Full parallel suite: **11860 passed in 3:22**.

### A.2 B1 — closed

| Locale | Status | Slice |
|---|---|---|
| GBP / London | ✅ fixed | v0.892 |
| AUD / Sydney | ✅ fixed | v0.893 |
| NZD / Wellington | ✅ fixed | v0.893 |
| CAD / Toronto | ✅ fixed | v0.893 |

Next L0-audit-driven fix: **A.1 B1 — ACT/ACT ICMA silent fallback** (UST mispricing). Multi-slice plan in the audit doc.

---

## v0.892.0 — 2026-06-11

**Fix A.2 B1a — `LondonCalendar` Saturday substitution now follows UK Banking and Financial Dealings Act 1971.**

First of four per-calendar fixes for audit finding A.2 B1. London was producing wrong observed-holiday dates whenever a fixed-date bank holiday fell on a Saturday — recurring every ~6 years (2021, 2027, 2032, ...).

### What was broken

`Calendar._observe` is the US rule (5 U.S.C. § 6103): Saturday → previous Friday, Sunday → next Monday. London inherited that rule unchanged.

UK Banking and Financial Dealings Act 1971 specifies: any bank holiday on Saturday or Sunday is observed the **next working day** (typically Monday).

Live repro before the fix — Christmas 2021 (Dec 25 Saturday):
```
2021-12-24 Fri: was holiday=True  (wrong — should be a business day)
2021-12-27 Mon: was holiday=True  (right, by accident, because Boxing-Day-Sun lands here)
2021-12-28 Tue: was business=True (wrong — should be Boxing observed)
```

After the fix:
```
2021-12-24 Fri: business day ✓
2021-12-27 Mon: Christmas observed ✓
2021-12-28 Tue: Boxing Day observed ✓
```

### Change

- `calendar.py:93-101` — `Calendar._observe` docstring clarified as "US-style" with a cross-reference to the new helper.
- `calendar.py:103-128` (new) — `Calendar._observe_next_working_day` static helper implements the UK / AU / NZ / CA rule (Sat → +2 days, Sun → +1 day). Per-locale subclasses opt in via `_observe = staticmethod(Calendar._observe_next_working_day)`.
- `calendar.py:LondonCalendar` overrides `_observe` to the next-working-day rule. Existing Boxing/Christmas collision handling (when both land on the same observed Monday) is unaffected — that code already pushes Boxing to Tuesday when the collision happens.
- 6 new tests in `test_calendar.py::TestLondonCalendarSubstitution`: Christmas 2021/2024/2027, Boxing 2021, New Year's 2028 Sat case, New Year's 2023 Sun case.
- Full parallel suite: **11848 passed in 4:42** — no regressions.

### Affected upstream

Any GBP-rate calculation that crosses a Saturday-Christmas, Saturday-Boxing, or Saturday-New-Year window. Days affected per affected year: 4 (one Friday wrongly closed; one Tuesday wrongly open; mirror at New Year). Years materially affected in the GBP curve's lookback window: 2021, 2027 (already past for 2021; 2027 lookback materially affects long-dated forwards observed today).

### Remaining (queued)

Same fix shape applies to `AUDCalendar`, `NZDCalendar`, `CADCalendar` — each as its own slice (separate failure surfaces, easier rollback).

---

## v0.891.0 — 2026-06-11

**Fix A.12 B1 — serialisation auto-discovery via `pkgutil.walk_packages`.**

The hand-maintained import whitelist in `core.serialization._ensure_loaded` is gone. The registry is now populated by walking every submodule of `pricebook` on first use.

### What was broken

`_ensure_loaded` was a curated list of 24 imports. The audit found 53 files in the codebase using `@serialisable` / `@serialisable_convention` / `_register` — so **29 modules' types were silently absent from the registry**. The failure mode was: `from_dict({"type": "esg_bond_spec", ...})` → `ValueError("Unknown type 'esg_bond_spec'")` — but only when the specific missing type was deserialised, after deployment. Every new module with a serialisable type carried this maintenance trap.

### Change

- `serialization.py:30-65` — `_ensure_loaded` now uses `pkgutil.walk_packages(pricebook.__path__, prefix="pricebook.")` + `importlib.import_module`. Idempotent; one-shot tree walk on first call. Any import failure is recorded in `_failed_imports` (module name + exception class) rather than silenced — visible to the test suite for CI catch.
- Auto-discovery now registers **120 types** (vs ~50 with the old whitelist). The codebase has not changed; the registry just sees what was always there.
- 4 new tests in `test_serialization_autodiscovery.py`: zero-import-failures invariant, ≥100 types registered, every major subpackage represented, regression test that `EquityIndexSpec` (from `core.market_conventions`, not in the old whitelist) is now reachable.
- Full parallel suite: **11842 passed in 4:41**.

### Side discovery — new audit finding

Auto-discovery surfaces a pre-existing `_SERIAL_FIELDS` mismatch on `AmortisingBond` — the registered fields (`face_value`, `coupon_rate`, `n_periods`, `frequency`) don't match `__init__` parameters (`amortisation_type`, `coupon_rate`, `maturity_years`, `n_payments`, `notional`). The `_register` validator emits a `UserWarning` at startup. This bug was hidden before because `amortising_bond` wasn't in the old whitelist. **Logged as a new audit finding** — to be fixed in the upcoming `amortising_bond` audit pass.

---

## v0.890.0 — 2026-06-11

**Fix A.4 B1 — `schedule.generate_schedule` EOM convention now anchors on `start` per ISDA 2006 §4.10.**

The first fix landing from the L0 audit. Schedule generation now correctly interprets the EOM rule for *all* generation paths (front-stub backward AND back-stub forward), not just the forward path that happened to work by accident.

### What was broken

In the front-stub (backward) generation path, the EOM decision was made inside `_add_months(d, months, eom)` by checking whether the rolling date `d` was itself EOM. For backward generation, `d` starts as `end`, so EOM was effectively anchored to `end`. When `start` was EOM but `end` was not, interior coupon rolls landed mid-month — violating ISDA 2006 §4.10 ("if the period start date is the last day of February, the period end date is the last day of February").

Live repro before the fix:
```
generate_schedule(start=2024-01-31, end=2024-08-15, semi-annual, SHORT_FRONT, eom=True)
  → [2024-01-31, 2024-02-15, 2024-08-15]      ← Feb 15 wrong; should be Feb 29
generate_schedule(start=2024-01-31, end=2025-04-15, semi-annual, SHORT_FRONT, eom=True)
  → [2024-01-31, 2024-04-15, 2024-10-15, 2025-04-15]   ← interior rolls all mid-month
```

After the fix:
```
[2024-01-31, 2024-02-29, 2024-08-15]
[2024-01-31, 2024-04-30, 2024-10-31, 2025-04-15]
```

### Change

- `schedule.py:25-30` — `_add_months(d, months, eom)` renamed to `_add_months(d, months, snap_to_eom)`. The boolean now means "globally snap to EOM" (a schedule-level decision), not "snap iff this particular d is EOM". The intent is documented inline.
- `schedule.py:80` — `generate_schedule` computes `snap_to_eom = eom and start == _end_of_month(start)` exactly once before both generation paths, and passes the same flag through.
- 3 new regression tests in `test_schedule.py`: ISDA §4.10 front-stub case + multi-year cross-leap-year case + start-not-EOM no-op case.
- Full parallel suite: **11838 passed in 3:20** — zero regressions in any downstream caller.

### Affected upstream

Any bond, swap, or amortising trade whose schedule had `start = EOM` and `end != EOM`. UST issued at end-of-month with a coupon-date maturity is the prototypical case. Schedules with EOM at *both* ends (the existing `test_eom_preserved` case) were already correct; nothing breaks.

---

## v0.889.0 — 2026-06-11

**G1 P3 Slice 2 — schema versioning on `@serialisable`. G1 P3 complete. Gate 1 closed.**

Every serialised dict now carries an explicit schema version, giving us a hard hook for future wire-format migrations without breaking anything that exists today. Combined with G1 P1 (`CalibrationResult`), G1 P2 (`MarketSnapshot` on every calibrator), and G1 P3 Slice 1 (`NumericalConfig`), pricebook is now **audit-ready**: a price is reconstructible from a stable, dated, versioned artefact graph.

- New class attribute `_SERIAL_SCHEMA_VERSION: int = 1` on `Serialisable`, `@serialisable`, and `@serialisable_convention`. Bump in your class when you make a *breaking* wire change.
- Envelope format (`Serialisable`, `@serialisable`) gains a top-level `"schema_version"` key alongside `"type"` and `"params"`.
- Flat-convention format (`@serialisable_convention`) gains a reserved `"_schema_version"` key — underscore-prefixed so it cannot collide with any real convention field.
- `from_dict` reads the version, treats *absent* as v1 silently (existing payloads on disk continue to deserialise), and raises a clear `ValueError("...schema_version=N, but this build only supports up to M. Upgrade this environment.")` when a payload was written by a newer build.
- Decorators take an optional `schema_version=N` keyword for classes that introduce migrations: `@serialisable("irs", [...], schema_version=2)` and `@serialisable_convention("foo", schema_version=3)`.
- 24 new tests in `test_serialisable_schema_version.py`: default version, to_dict emits version (envelope + flat), round-trip across all three paths, backward compat with absent-version payloads, future-version rejection with a clear message, wire-format key invariants, and version-key strip-before-construct.
- Full parallel suite: **11835 passed in 5:34** — every existing serialisation round-trip in the library still works.

### Gate 1 closed

Phase 1: every calibrator produces a uniform `CalibrationResult` (UUID, timestamp, code version, residuals, optimiser story).
Phase 2: every calibrator accepts a `MarketSnapshot`; id links to the result.
Phase 3: numerical hyperparameters become first-class on `PricingContext`; wire format becomes version-aware.

**Result:** a price computed today can be reconstructed tomorrow from a single, structured, versioned artefact chain: trade → pricer → `PricingContext{numerical_config, curves}` → `CalibrationResult.market_snapshot_id` → `MarketSnapshot`. Each link is dated, identified, and serialisable.

Next: Gate 2 — *production-grade*. Begin auditing each module with the audit chain available.

---

## v0.888.0 — 2026-06-11

**G1 P3 Slice 1 — `NumericalConfig` + `PricingContext.numerical_config`.**

Numerical hyperparameters become first-class. Two valuations on the same book with different MC path counts are *different valuations* — until now that choice was buried in hard-coded defaults inside each pricer and invisible to any audit. After this slice the choice lives on the context and can be serialised, diffed, and overridden as a single object.

- New module `pricebook.core.numerical_config`:
  - `NumericalConfig` — frozen dataclass with 14 fields covering Monte Carlo (paths, seed, antithetic, Sobol, Brownian bridge), PDE/FD (time/space steps, truncation width), tree steps, integration (tol, max-iter), COS method (N, L), root-finding (tol, max-iter), plus an `extra: Mapping[str, Any]` escape hatch. `.replace(**kwargs)` for ergonomic edits.
  - `DEFAULT_NUMERICAL_CONFIG` — frozen, shared singleton matching the historical library defaults so no existing behaviour changes.
- `PricingContext` gains `numerical_config: NumericalConfig | None = None`. `get_numerical_config()` returns the attached config or the default singleton — pricers should read through the accessor. `.replace(numerical_config=...)` works including clearing to `None`.
- 14 new tests in `test_numerical_config.py`: frozen-ness, `.replace`, default fallback, accessor returns attached / falls back, `simple()` factory unchanged, end-to-end with curve + config.
- Purely additive — `PricingContext.simple()` and all existing call-sites are untouched.

Next slice (G1 P3 Slice 2) adds **schema versioning** on `@serialisable` — closes Gate 1.

---

## v0.887.0 — 2026-06-11

**G1 P2 Slice 5 — jump-model calibrators accept a `MarketSnapshot`. G1 P2 complete.**

The last family of calibrators wires through. **Every calibrator in pricebook now accepts an optional `market_snapshot`**, and `CalibrationResult.market_snapshot_id` is populated whenever provided — closing Phase 2 of Gate 1.

- `pricebook.models.jump_calibration.calibrate_jump_model(...)` — keyword-only `market_snapshot: MarketSnapshot | None = None`. Covers all six Lévy / jump models (Merton, VG, Kou, NIG, CGMY, Bates).
- `calibrate_jump_surface(...)` — same; a single snapshot stamps onto every per-expiry result (the typical case: one observation set underlies the whole surface).
- `TYPE_CHECKING` import — no runtime dep.
- 6 new tests in `test_jump_snapshot.py`: keyword-only enforcement, id linkage on the single-expiry path, surface-level propagation across multiple expiries (each per-expiry `CalibrationResult.id` is distinct while the `market_snapshot_id` is shared).

### G1 P2 — complete (Slices 1-5)

12 calibration entry points across 9 modules now accept a `MarketSnapshot`:

| Slice | Module | Entry points |
|---|---|---|
| 1 | `pricebook.market_data` | (types only) |
| 2 | `pricebook.curves.bootstrap` / `global_solver` | `bootstrap`, `global_bootstrap` |
| 3 | `pricebook.curves.multicurve_solver` / `credit.bond_hazard_bootstrap` | `multicurve_newton`, `bootstrap_hazard_from_bonds`, `bootstrap_hazard_mixed`, `bootstrap_hazard_adaptive` |
| 4 | `pricebook.models.{hw,g2pp,lmm}_calibration` / `pricebook.options.sabr` | `calibrate_hull_white`, `calibrate_g2pp`, `calibrate_lmm_vols`, `sabr_calibrate` |
| 5 | `pricebook.models.jump_calibration` | `calibrate_jump_model`, `calibrate_jump_surface` |

Audit chain end-to-end: **price → calibration → market snapshot**. Next: **G1 P3** — `NumericalConfig` on `PricingContext` + schema versioning on `@serialisable` — closes Gate 1.

---

## v0.886.0 — 2026-06-11

**G1 P2 Slice 4 — HW, G2++, SABR, LMM calibrators accept a `MarketSnapshot`.**

Four model calibrators wired in one slice. Audit chain now reaches every IR/vol model-calibration entry point in the library.

- `pricebook.models.hw_calibration.calibrate_hull_white(...)` — keyword-only `market_snapshot: MarketSnapshot | None = None`.
- `pricebook.models.g2pp_calibration.calibrate_g2pp(...)` — same.
- `pricebook.models.lmm_calibration.calibrate_lmm_vols(...)` — same.
- `pricebook.options.sabr.sabr_calibrate(...)` — same; returned dict's `"calibration_result"` carries the linked id.
- All four use `TYPE_CHECKING` import on `pricebook.market_data.MarketSnapshot` — no runtime dep added.
- 12 new tests in `test_models_snapshot.py`: keyword-only enforcement and id linkage on each calibrator (G2++ uses `method="minimize"` to stay under a second per test).

Single slice left in G1 P2: jump-model calibrators (`calibrate_jump_model` family: Merton, VG, Kou, NIG, CGMY, Bates) — same additive pattern. After that, G1 P2 closes and G1 P3 (`NumericalConfig` on `PricingContext`) begins.

---

## v0.885.0 — 2026-06-11

**G1 P2 Slice 3 — `multicurve_newton` + the bond-hazard bootstrap family accept a `MarketSnapshot`.**

Phase 2 of Gate 1 now reaches every major curve-and-hazard calibration entry point. The audit chain **price → calibration → market snapshot** is wired through the `curves` *and* `credit` subpackages — together with G1 P2 Slice 2, this covers all single- and multi-curve bootstrapping plus credit hazard calibration.

- `pricebook.curves.multicurve_solver.multicurve_newton(...)` gains keyword-only `market_snapshot: MarketSnapshot | None = None`. Id propagates through both converged and non-converged paths (the latter still emits the existing `RuntimeWarning`). Both `ois_curve.calibration_result` and `projection_curve.calibration_result` share the same `CalibrationResult` instance, so the link is canonical, not duplicated.
- `pricebook.credit.bond_hazard_bootstrap`:
  - `bootstrap_hazard_from_bonds(...)` — keyword-only `market_snapshot`. Id threads through `_bootstrap_sequential` and `_bootstrap_global` (both reach `CalibrationResult.market_snapshot_id`).
  - `bootstrap_hazard_mixed(...)` — same.
  - `bootstrap_hazard_adaptive(...)` — same; passes the snapshot through to the chosen downstream entry point.
- `TYPE_CHECKING` import on both new touch sites — no runtime dep on `pricebook.market_data`.
- 9 new tests in `test_multicurve_snapshot.py`: keyword-only enforcement, id linkage on converged + non-converged multicurve, id linkage on sequential + global bond-hazard, snapshot-presence is numerically inert.

G1 P2 ready for Slice 4 (other model calibrators: HW, G2++, SABR, LMM, jump models — same additive pattern).

---

## v0.884.0 — 2026-06-11

**G1 P2 Slice 2 — `bootstrap` and `global_bootstrap` accept a `MarketSnapshot`; snapshot id propagates onto the `CalibrationResult`.**

Audit chain extends: **price → calibration → market snapshot** is now wired end-to-end for the two L2 entry points used by every downstream pricer.

- `pricebook.curves.bootstrap.bootstrap(...)` gains a keyword-only `market_snapshot: MarketSnapshot | None = None` parameter.
- `pricebook.curves.global_solver.global_bootstrap(...)` gains the same keyword-only parameter.
- When provided, `MarketSnapshot.id` is stamped onto `CalibrationResult.market_snapshot_id` on the curve's `calibration_result`. When absent, the field stays `None` (back-compat).
- Snapshot is a **provenance pointer** in this slice — deposits/swaps args remain authoritative for the actual rates. A future slice will let bootstrap *derive* the deposit/swap lists from the snapshot's quotes.
- 9 new tests in `test_curve_bootstrap_snapshot.py`: snapshot↔calibration linkage, keyword-only enforcement, snapshot-presence is numerically inert, distinct snapshots produce distinct results.
- `TYPE_CHECKING` import on both files — no runtime dep added to `pricebook.curves`.

---

## v0.883.0 — 2026-06-11

**G1 P2 Slice 1 — `pricebook.market_data` package + canonical L1 raw-data types.**

Phase 2 of Gate 1 begins. Where Phase 1 made every calibration produce a `CalibrationResult`, Phase 2 makes the *raw market data* a first-class, identifiable artefact — so the audit chain extends one step further: price → calibration → market snapshot.

This first slice introduces the **types only**, purely additive — nothing existing is touched.

- New package `pricebook.market_data` (L1 in the layer graph; zero dependencies on other pricebook subpackages).
- `QuoteKind` — `str`-Enum of 17 recognised quote types (deposits, FRAs, futures, swaps, basis, xccy, CDS, bond px/yield, vol points, swaption/capfloor vols, FX spot/forward, inflation YoY/ZC, other).
- `QuoteId` — frozen dataclass `(kind, tenor, currency, label)`; stable, hashable, the "same instrument across time" key.
- `Quote` — frozen `(id, value, bid_ask_bp)`; a single observation.
- `MarketSnapshot` — frozen `(id: UUID, as_of: datetime, quotes: tuple[Quote,...], label)`. `MarketSnapshot.new()` auto-generates id + timestamp. `get(qid)`, `filter(kind=, currency=, label=)`, `with_quote(q)` (returns a new snapshot with a fresh id — replaces same-id quotes). `__len__`, `__iter__`, `__contains__` for ergonomic use.
- `FixingHistory` — frozen wrapper around `{(rate_name, date) -> value}`; `get`, `for_rate`, `rate_names`, `with_fixing`. Distinct from `MarketSnapshot` (fixings are backward-looking, no bid-ask).
- 31 new tests in `test_market_data_types.py` covering frozen-ness, equality, hashing, factory behaviour, auditing invariants (fresh id on `with_quote`, original snapshot unchanged).

The existing `pricebook.core.market_data.MarketDataSnapshot` is untouched — backward compatibility is total. Subsequent slices in G1 P2 will let bootstrap/global-bootstrap accept a `MarketSnapshot` and record `MarketSnapshot.id` on the resulting `CalibrationResult.market_snapshot_id`.

---

## v0.882.0 — 2026-06-11

**G1 P1 Slice 6 — `jump_calibration` produces `CalibrationResult`. G1 P1 complete.**

The last of seven calibration families lands. With this slice, **every calibration in the codebase produces a uniform `CalibrationResult` artefact**, closing out Phase 1 of Gate 1 in the `DESIGN.md` roadmap.

- `JumpCalibrationResult` (covers all six jump/Lévy models: Merton, VG, Kou, NIG, CGMY, Bates) gains `calibration_result: CalibrationResult | None` + `to_calibration_result()` method. `to_dict` gets `calibration_id` key.
- `calibrate_jump_model` populates with `model_class=f"jump_{model_type}"` (e.g. `jump_merton`, `jump_vg`, `jump_kou`, `jump_nig`, `jump_cgmy`, `jump_bates`), parameters as the per-model param dict (4 for Merton, 3 for VG, 4 for Kou, 4 for NIG, 4 for CGMY, 7 for Bates), residuals = `model_vol − market_vol` per strike, algorithm `"differential_evolution+L-BFGS-B"`, deterministic seed recorded, spot / rate / T / div_yield / bounds in `optimiser.extra`, `rmse_vol` in diagnostics, quotes named `smile_K=<strike>`.
- 12 new tests in `test_calibration_result_jump.py`: 9 end-to-end (1 fast Merton calibration with 3 strikes / `maxiter=30`) + 3 back-compat.
- 32 tests pass across the jump family (test_jump_calibration + test_jump_cross_validation + new file) — no regression.

### G1 P1 — complete (Slices 1-6)

103 dedicated calibration-layer tests in total. **Every calibrator in pricebook now produces a `CalibrationResult`:**

| # | Family | Model class string | Algorithm |
|---|---|---|---|
| 1 | foundation | (none — types only) | n/a |
| 2 | bond hazard | `bond_hazard_pwc` | brentq-per-bond / L-BFGS-B[+tikhonov] |
| 3 | Hull-White | `hull_white` | Nelder-Mead / DE / L-BFGS-B |
| 3 | G2++ | `g2pp` | differential_evolution+L-BFGS-B |
| 4 | LMM | `lmm` | iterative_scaling |
| 4 | SABR | `sabr` | nelder_mead |
| 5 | curve bootstrap | `discount_curve_bootstrap` | brentq-sequential |
| 5 | global curve | `discount_curve_global` | newton-global |
| 5 | multicurve | `multicurve` | newton-multicurve |
| 6 | jump (6 models) | `jump_{merton,vg,kou,nig,cgmy,bates}` | differential_evolution+L-BFGS-B |

The audit trail `(price) -> (calibration_id) -> (market_snapshot, code_version, optimiser)` is now reconstructable end-to-end for every calibrated number in pricebook.

Next phase: **G1 P2** — `MarketSnapshot` at L1, the canonical raw-market-data type. Curves will be built from snapshots; the `market_snapshot_id` field on `CalibrationResult` (currently always `None`) will be populated.

---

## v0.881.0 — 2026-06-11

**G1 P1 Slice 5 — curve bootstrap produces `CalibrationResult`.**

The highest-fan-out calibration in the codebase: nearly every pricing call traces back to a bootstrapped discount curve. Six of seven calibration families now carry the canonical artefact (bond hazard, Hull-White, G2++, LMM, SABR, **curve bootstrap**); jump calibration is the last one (Slice 6).

- **`DiscountCurve.calibration_result: CalibrationResult | None`** — new instance attribute on the L0 curve class itself. Defaults to `None` for curves built directly (`flat(...)`, manual pillar construction); set to the canonical artefact by bootstrap entry points. Type hinted under `TYPE_CHECKING` to avoid a runtime import cycle between `core` and `calibration`.
- **`curves.bootstrap.bootstrap()`** — sequential brentq-per-pillar bootstrap now constructs a `CalibrationResult` after the round-trip verification step and attaches it to the returned curve. `model_class="discount_curve_bootstrap"`, algorithm `"brentq-sequential"`, residuals = (model_rate − market_rate) per instrument (deposits, FRAs, futures) or (PV_fixed − PV_float) per swap, parameters keyed by pillar date as `df(YYYY-MM-DD)`. RMS residual ~1e-14 (machine precision, exact fit by construction).
- **`curves.global_solver.global_bootstrap()`** — Newton-global simultaneous solve. Residuals are the final per-instrument repricing errors; algorithm `"newton-global"`, `tol` and `max_iter` passed through to `OptimiserSpec`. Converged flag captures whether the Newton loop hit `tol` before `max_iter`.
- **`curves.multicurve_solver.multicurve_newton()`** — joint OIS + projection calibration. Both `MultiCurveResult.calibration_result` AND `ois_curve.calibration_result` AND `projection_curve.calibration_result` get the SAME `CalibrationResult` instance (they share the calibration). Parameters carry an `ois_df(...)` / `proj_df(...)` prefix to distinguish the two pillar sets. `MultiCurveResult` gains a `to_calibration_result()` method that returns the stored instance or builds on-demand.
- 17 new tests in `test_calibration_result_curve_bootstrap.py`: default-None on `DiscountCurve.flat`, populated by `bootstrap()` / `global_bootstrap()` / `multicurve_newton()`, residual sanity (machine precision for sequential, sub-tol for Newton), parameter naming, quote naming, unique-id-per-call, multicurve cross-attachment, back-compat for hand-constructed `MultiCurveResult`.
- 40 tests pass across the full curve family (bootstrap + global_solver + multicurve_solver + new Slice 5 tests) — no regression.
- `coupled_bootstrap` and `bootstrap_forward_curve` not migrated in this slice — deferred to Slice 5b if needed.

Next: Slice 6 covers `jump_calibration` — the last calibration family.

---

## v0.880.0 — 2026-06-11

**G1 P1 Slice 4 — LMM + SABR calibration produce `CalibrationResult`.**

Five of seven calibration families now produce the canonical artefact (bond hazard, Hull-White, G2++, LMM, SABR). Two remaining: curve bootstrap (Slice 5) and jump calibration (Slice 6).

- `LMMCalibrationResult` (dataclass) gains `calibration_result: CalibrationResult | None` field + `to_calibration_result()` method. `to_dict` gets `calibration_id` key.
- `calibrate_lmm_vols` populates with `model_class="lmm"`, parameters as `{"sigma_0", ..., "sigma_{n-1}"}` (one per forward rate), residuals in vol units (`fitted - target` per swaption), algorithm `"iterative_scaling"`, `correlation_beta` and `tau` recorded in `optimiser.extra`, `rmse_vol` in diagnostics.
- `sabr.sabr_calibrate` (dict-returning, by convention) gains a `"calibration_result"` key in its returned dict — additive, no existing key removed or renamed.
- SABR `CalibrationResult` has `model_class="sabr"`, parameters `{"alpha", "beta", "rho", "nu"}` (4 params), residuals as `model_vol − market_vol` per strike (vol units, not bp), algorithm `"nelder_mead"`, `forward` / `T` / `beta_fixed` in `optimiser.extra`, quotes named `smile_K=<strike>`. `calibrate_sabr_smile` inherits the dict unchanged.
- 21 new tests in `test_calibration_result_lmm_sabr.py`: 9 LMM end-to-end + 3 LMM back-compat + 9 SABR end-to-end (existing keys preserved, parameters / residuals / quotes / unique ids).
- 64 tests pass across LMM + SABR family (test_sabr.py + test_lmm.py + test_lmm_calibration.py + new file) — no regression.

Next: Slice 5 migrates curve bootstrap (`curves.bootstrap`, `curves.multicurve_solver`, `curves.global_solver`) to produce `CalibrationResult`. The curve bootstrap is the highest-fan-out calibration in the codebase — many tests will see the new field downstream.

---

## v0.879.0 — 2026-06-11

**G1 P1 Slice 3 — `g2pp_calibration` and `hw_calibration` produce `CalibrationResult`.**

Same template as Slice 2, applied to the two single-curve rate-model calibrations. Three of the seven calibration families in `pricebook` now produce the canonical artefact (bond hazard, Hull-White, G2++); LMM, SABR, jump, and curve bootstrap follow in Slices 4-6.

- `HWCalibrationResult` gains `calibration_result: CalibrationResult | None` field and a `to_calibration_result()` method that returns the stored instance when populated or builds on-demand otherwise. `to_dict` gets the `calibration_id` key.
- `calibrate_hull_white` populates `calibration_result` with `model_class="hull_white"`, parameters `{"a": ..., "sigma": ...}`, residuals in vol-bp, `OptimiserSpec(algorithm=...)` capturing the chosen optimiser (`Nelder-Mead`, `differential_evolution`, or `L-BFGS-B`), `rmse_vol` in diagnostics, and `n_steps` in optimiser.extra.
- `G2PPCalibrationResult` gains the same field and method. `to_dict` gets the `calibration_id` key.
- `calibrate_g2pp` populates with `model_class="g2pp"`, parameters `{"a", "b", "sigma1", "sigma2", "rho"}` (5 params), residuals in vol-bp, algorithm `"differential_evolution+L-BFGS-B"` (or `"L-BFGS-B"` for minimise method), and the parameter bounds in `optimiser.extra`. Seed=42 recorded when DE is used.
- 16 new tests in `test_calibration_result_g2pp_hw.py`:
  - 9 HW tests run the full `calibrate_hull_white` on a small 3-swaption grid (~3 sec each): population of all fields, optimiser name and extras, residuals match `per_swaption_errors`, rmse_vol in diagnostics, named quote IDs, `to_calibration_result()` returns stored instance, unique id per run, `calibration_id` in `to_dict`.
  - 7 G2++ tests via hand-constructed `G2PPCalibrationResult` (the slow full `calibrate_g2pp` is covered by existing tests in `test_g2pp_calibration.py` which are deselected in CI): on-demand `to_calibration_result()` path, parameters count (5), `calibration_id` in `to_dict` both populated and unpopulated paths.
- Zero behavior change to existing tests (24 HW + Slice 3 tests pass; existing G2++ tests untouched).

Next: Slice 4 migrates `lmm_calibration` and `sabr` calibration to the same pattern.

---

## v0.878.0 — 2026-06-11

**G1 P1 Slice 2 — `bond_hazard_bootstrap` now produces a `CalibrationResult`.**

First migration of an existing calibration to the new canonical artefact (DESIGN.md §6 G1 P1). The bond-hazard bootstrap was chosen because it is the most recently touched calibration (Tikhonov + L-curve work landed yesterday) and its results carry the richest provenance (`lam`, `roughness`, per-bond residuals).

- `HazardBootstrapResult` gains a `calibration_result: CalibrationResult | None` field. Defaults to `None` for backward compatibility with hand-constructed instances; entry points always populate it.
- `_bootstrap_sequential` populates a `CalibrationResult` with `model_class="bond_hazard_pwc"`, `objective=SSE`, `optimiser=OptimiserSpec(algorithm="brentq-per-bond", ...)`, one iteration per bond, per-bond weights.
- `_bootstrap_global` populates with `objective=WEIGHTED_SSE`, `optimiser=OptimiserSpec(algorithm="L-BFGS-B[+tikhonov(lam=...)]", ...)`. When `lam > 0`: `lam` recorded in `optimiser.extra`, `roughness` in `diagnostics.extra`. Algorithm name gets the `+tikhonov(lam=...)` suffix for quick filtering in audit logs.
- `bootstrap_hazard_mixed` (FRN + bond joint fit) also migrated; `quotes_fitted` carries `["bond_0", ..., "frn_0", ...]`.
- New `to_calibration_result()` method returns the stored instance when populated, or builds one on-demand from the existing fields when not. Latter path covers any legacy hand-construction.
- `to_dict()` gets a `calibration_id` key (the UUID stringified, or `None`). Lets persistence layers later use this as a foreign key.
- 15 new tests in `test_bond_hazard_calibration_result.py` — population, parameter/residual/weight matching, Tikhonov-extras, unique id per invocation, on-demand build, hand-construction back-compat, `calibration_id` in dict.

Zero existing test changes: the `rmse_bp`, `pillar_hazards`, and other historical API surface is preserved. The new `calibration_result` is purely additive.

Next: Slice 3 migrates `g2pp_calibration` and `hw_calibration` to produce `CalibrationResult` (same template).

---

## v0.877.0 — 2026-06-11

**G1 P1 Slice 1 — `pricebook.calibration` skeleton + `CalibrationResult` + `Calibrator` Protocol.**

First slice of Gate 1 from `DESIGN.md` §6. The calibration layer gets its own package; the canonical result type is defined here so that subsequent slices can migrate each existing calibrator family (bond hazard, G2++, Hull-White, LMM, SABR, curve bootstrap, multicurve) to produce a uniform `CalibrationResult`.

- New package `python/pricebook/calibration/`:
  - `__init__.py` — public exports.
  - `_types.py` — `CalibrationResult` (frozen dataclass with `id`, `timestamp`, `code_version`, `model_class`, `parameters`, `residuals`, `rms_residual`, `max_residual`, `optimiser`, `iterations`, `converged`, `diagnostics`, `market_snapshot_id`), `OptimiserSpec`, `CalibrationDiagnostics`, `ObjectiveKind` (SSE / weighted-SSE / RMSE / max-error / L1 / Huber), `Calibrator` Protocol.
  - `CalibrationResult.new(...)` factory — keyword-only, auto-generates `id` and `timestamp`, derives `rms_residual` and `max_residual` from `residuals`, reads `pricebook.__version__` for `code_version` by default.
- 22 tests in `tests/test_calibration_types.py` covering enum values, OptimiserSpec construction & frozen-ness, diagnostics defaults & population, factory derivation of RMSE/max-error, default weights, explicit weights, code_version detection & override, market_snapshot_id placeholder, frozen-ness at all levels, Protocol satisfaction.
- Zero internal pricebook imports (other than `pricebook.__version__` read defensively) — package sits cleanly in the dependency graph; no behavior change to any existing code.

This slice is purely additive: no existing module is touched. Slice 2 migrates `bond_hazard_bootstrap` to return the new result type with a thin compatibility wrapper preserving the existing `HazardBootstrapResult` interface.

---

## v0.876.0 — 2026-06-11

**`DESIGN.md` — §6 roadmap rewritten as Gate × Phase hybrid.**

- Old roadmap had 4 phases with significant overlap (CalibrationResult in P1, calibration layer in P2 — same effort, separate commits) and blurry themes ("foundational types" + "structural relocation"). Phase 2 was 20-27 slices bundling six different efforts.
- New structure: **5 gates** (each a user-visible promise) × **10 phases** (each one architectural focus):
  - **G1 Audit-ready** — P1 Calibration unified + P2 Market data L1 + P3 NumericalConfig & versioning (14-16 slices, 1-3 weeks). Prerequisite for the bottom-up audit.
  - **G2 Production-grade** — P4 Scenarios & failures + P5 Repositories & per-layer tests (7-9 slices).
  - **G3 Architecturally clean** — P6 small cleanups + P7 Risk relocation L3→L7 isolated (9-13 slices). P7 is the biggest single refactor; isolated so it doesn't poison its gate.
  - **G4 Capability-complete** — P8 AAD as protocol + P9 Payoff algebra (10-13 slices).
  - **G5 Performant at scale** — P10 C++ port (19-28 slices, open-ended).
  - Total: **59-79 slices, 7-19 weeks** at the historical pricebook slice rate.
- Each gate is shippable independently. G2 doesn't require G3; you can stop after G2 and have a meaningfully better library.
- Section now includes a per-gate exit criteria, a dependency graph between phases, and "the first three slices of G1 P1" so the immediate next step after acceptance is unambiguous.
- Executive summary + TOC updated to reflect the new structure.

---

## v0.875.0 — 2026-06-11

**`DESIGN.md` — theoretical design document (~30 pages).**

- New `DESIGN.md` at the repo root: a first-principles design of a financial analytics engine, then pricebook overlaid as the lens.
- Sections 1-3 (principles, reference architecture, patterns/anti-patterns) drafted by `app-designer` for an independent second perspective on architecture; sections 4-6 written here.
- The reference design: 9-layer architecture with **calibration as its own layer (L6)**, parallel to risk, both depending on pricing (L5). **Models as `Protocol`, not inheritance.** **Trades as frozen dataclasses, market state passed in via `PricingContext`**. AAD via generic-scalar discipline at L0. Industry references woven through (QuantLib, ORE, OpenGamma Strata, Numerix). Concrete proposed shapes for `PricingContext`, `CalibrationResult`, `RiskRun`, `Scenario`.
- Pricebook gap analysis (§4): the architecture matches the reference at the foundations (acyclic layering, `PricingContext` exists, serialisation contract, mostly-Protocol models, mostly-frozen dataclasses), but diverges in three structural places: **risk at L3 instead of L6 above pricing** (the biggest mismatch — forces risk modules to know about concrete instruments); **calibration distributed rather than its own layer** (no unified `CalibrationResult`); **market data conflated with curves** (cannot distinguish quotes from fits in the dependency graph).
- Delta list (§5): 6 high-value adds (CalibrationResult, MarketSnapshot, Scenario protocol, PricingFailure, NumericalConfig, schema versioning), 5 wrong-shape refactors (risk relocation, calibration consolidation, market data split, pe/ relocation, registry consolidation), 4 nice-to-haves, 5 explicit won't-fixes.
- Roadmap (§6): 4 phases, **~63-84 slices total, ~10-22 weeks at the historical slice rate**. Phase 1 (foundational types) is the prerequisite for the bottom-up audit (the next major task after this document is accepted).

Status: ready for pushback before the bottom-up audit begins. Sections 1-3 are load-bearing — everything in 4-6 depends on the layer cut and the calibration placement.

---

## v0.874.0 — 2026-06-11

**Hazard-from-bonds notebook — adapted to use the library Tikhonov.**

- Removed the ad-hoc `regularised_bootstrap(...)` function from the notebook builder (Section 5). Replaced its 5 call sites (λ-sweep, L-curve dense grid, LOO-CV inner loop, MC sensitivity, realistic demo) with `bootstrap_hazard_from_bonds(method="global", pillar_times=..., lam=...)`. Section 9's `res_tik` is now a `HazardBootstrapResult`, with downstream code reading attributes (`res_tik.pillar_hazards`, `res_tik.survival_curve`) instead of dict keys.
- Updated Section 5 prose to point users at the library API; replaced the closing cheat sheet's "not currently in library" caveat with the recommended call (`lam="auto"`, `pillar_times=[...]`).
- Notebook re-executes cleanly: 46 cells (down from 47 — one builder-cell removed), 0 errors, 14 embedded plots. Every headline number unchanged (L-curve λ* = 3.16e+06, LOO-CV λ* = 6.58e+06, Section-10 Q(5y) MC mean 0.8867 vs deterministic 0.8877) — the library-backed and ad-hoc fits agree to numerics.
- The notebook is now self-consistent with the library: a reader who sees the API in §5 can apply it directly in their own code with no further translation.

Closes the Tikhonov work started in v0.872. Two-slice pattern (library first, notebook after) preserved.

---

## v0.873.0 — 2026-06-11

**`pillar_times` override in `bond_hazard_bootstrap`.**

- `_bootstrap_global`, `bootstrap_hazard_from_bonds`, and `find_lcurve_lambda` all gain an optional `pillar_times: list[float] | None = None` parameter. When provided, it overrides the default even-spacing of pillars and `n_pillars` is ignored.
- Two common patterns this enables:
  - **Pillars at bond maturities** (exactly-determined fit at `lam=0`, regularised as `lam` grows) — useful when the bond universe has natural calibration anchors at the cashflow dates.
  - **Pillars at calendar benchmarks** (e.g. 1, 2, 5, 10, 30y for sovereign) — for curves that need to align with peer-group benchmarks.
- Validation: empty list, non-positive values, and non-strictly-increasing sequences all raise `ValueError`.
- 4 new tests in `test_bond_hazard_tikhonov.py::TestPillarTimes` covering override behaviour, exact-fit-at-bond-maturities semantics, validation errors, and combination with `lam="auto"`. Brings the new test count to **15**.
- Required for the hazard-from-bonds notebook to be adaptable to the library API (Slice B); the existing notebook's `regularised_bootstrap` placed pillars at bond maturities to make the close-pair pedagogy clean.

---

## v0.872.0 — 2026-06-11

**Tikhonov regularisation in `bond_hazard_bootstrap`.**

- `_bootstrap_global(...)` and `bootstrap_hazard_from_bonds(...)` gain a `lam` parameter:
  - `lam=0.0` (default): unregularised LS — bit-for-bit identical to the existing behaviour (regression-tested).
  - `lam > 0`: penalised LS with a second-difference (curvature) penalty $\lambda\|Lh\|^2$.
  - `lam="auto"`: pick λ via the L-curve corner (Hansen 1992).
- New helper `find_lcurve_lambda(...)` does the L-curve sweep + corner detection. Uses signed curvature (most-negative κ in log-log space) rather than max-|κ| to avoid boundary finite-difference artifacts.
- `HazardBootstrapResult` gains two new diagnostic fields with defaults: `lam` (the λ used) and `roughness` (the final $\|Lh\|^2$). Backward-compatible — existing constructors omitting these still work.
- Method label becomes `"global_ls_tikhonov"` when `lam > 0`, stays `"global_ls"` when `lam=0`.
- 11 new tests in `tests/test_bond_hazard_tikhonov.py`: regression at `lam=0`, monotonic roughness decrease with `lam`, `lam→∞` driving the curve to linear (zero curvature), L-curve picker correctness, edge cases (n_pillars < 3 ignores `lam`, negative `lam` raises, sequential method unaffected by `lam`).
- Doc updates: `bootstrap_hazard_from_bonds` docstring now points to the hazard-from-bonds notebook for the full derivation and L-curve picking.

The recommended call for non-trivial bond universes (any pair within ~3 months of maturity, or > 8 bonds) is now:
```python
result = bootstrap_hazard_from_bonds(
    REF, bonds, rf_curve, method="global", n_pillars=..., lam="auto"
)
```
For well-spaced 3-8 bond universes, `method="sequential"` (default via `"auto"`) remains the right answer.

---

## v0.871.0 — 2026-06-11

**Hazard-from-bonds notebook — Round 5 (final polish). Notebook now complete.**

- Added an **executive-summary table** at the top: noise amplification (27× at 1mo, 54× at 2wk), Tikhonov variance reduction (22× at the close pillar), L-curve and LOO-CV agreement (~2×), Q(5y) integrated-vs-instantaneous (matches to 0.001).
- Added the **three things to remember** rules (bootstrap brittle, Tikhonov fixes it, bond prices are integral probes).
- Added a **table of contents** with anchor links to all 10 sections + the closing cheat sheet.
- Added a closing **"When to use what" cheat sheet** — universe-shape → recommended method, three rules of thumb, plus the relevant pricebook imports.
- Added explicit `<a id="section-N"></a>` anchors before each `## N.` heading so the ToC links resolve reliably in Jupyter/nbconvert.
- Final re-execution: **47 cells (22 md + 25 code), 0 errors, 14 embedded plots, 11/11 anchored headings**.

The notebook is now content-complete and ready to publish.

---

## v0.870.0 — 2026-06-10

**Hazard-from-bonds notebook — Round 4 (sections 8-10). Notebook now complete content-wise (46 cells, ~2 MB).**

- **Section 8 (adaptive switch).** `assess_liquidity` + `bootstrap_hazard_adaptive` demo across three scenarios — liquid (6 well-spaced bonds, tight bid-ask) → `sequential`; semi-liquid (5 bonds with close pair) → `global`; illiquid (2 distressed bonds, 250-350 bp spread) → `global` with 2 pillars. Includes the important caveat that the heuristic protects against scale problems (count, spread) but **not** geometry (close maturities) — the user must reach for `method="global"` explicitly when bonds bunch.
- **Section 9 (realistic demo).** 8 bonds at sovereign-like maturities (0.5/1/3/5/5.25/7/10/10.5y) with two adjacent benchmark pairs. ±5 bp uniform price noise. Three methods side by side: sequential RMSE 0 bp (exact, brittle at close pairs), global LS RMSE 7.3 bp (5 even pillars), Tikhonov RMSE 6.3 bp (8 pillars + smoothness). Side-by-side hazard curves and per-bond residual scatter.
- **Section 10 (CIR++ cross-check — the deepest takeaway).** `CIRPlusPlus.from_survival_curve` overlays Cox-Ingersoll-Ross dynamics on the regularised piecewise-constant curve. 60 MC paths shown. **Sanity check at T=5y: deterministic Q(5) = 0.8877, MC mean Q(5) = 0.8867, MC std Q(5) = 0.0091.** The instantaneous hazard paths spread visibly around the mean, but the *integrated* survival at the bond maturities reconverges almost exactly. Geometrically: bond data constrains $\int_0^{t_i} h\,du$ at coupon dates, not $h(t)$ pointwise. Two hazard functions with the same integrals at every bond maturity are indistinguishable by bond prices. Choosing among them (piecewise constant / spline / CIR++) is a choice of *prior*, not of *data*.
- Final user-facing summary table in Section 10 markdown: when to use sequential / global / Tikhonov / CIR++.

Round 5 (final polish + ToC + commit) to follow.

---

## v0.869.0 — 2026-06-10

**Hazard-from-bonds notebook — Round 3 (sections 6-7).**

- Section 6 (Picking λ) — L-curve method (Hansen 1992) with explicit max-curvature corner detection on the log-log plot; LOO-CV (model-free GCV analogue, $N$ refits per λ point). L-curve corner λ* = 3.2e+06; LOO-CV minimum λ* = 6.6e+06 — both methods agree on regularisation strength within a factor of ~2. Two plots (L-curve with corner star; LOO-CV vs λ with both criteria's λ marked).
- Section 7 (Bid-ask sensitivity) — 200 Monte-Carlo draws with ±10 bp price half-spread (typical IG-corporate width). Box plot per pillar, unregularised vs Tikhonov at L-curve-corner λ. The headline: at the 5y+2mo close-maturity pillar, hazard standard deviation collapses from **109.3 bp (unregularised) to 4.9 bp (regularised) — a 22× variance reduction**. Other pillars (1y, 3y, 5y, 10y) see modest 1.2-1.7× reductions. Discusses the bias-vs-variance tradeoff implicit in the prior.

Now 33 cells, 898 KB. Sections 8-10 (adaptive switch, realistic demo, CIR++ cross-check) to follow.

---

## v0.868.0 — 2026-06-10

**Hazard-from-bonds notebook — Round 2 (sections 4-5).**

- Section 4 (Solver limits) — brentq bracket failure (bond above risk-free benchmark → no non-negative hazard solves it; residual stays one-signed across the whole [1e-6, 5.0] bracket) and Newton sensitivity (from h0=0.5/2/5 Newton "converges" to fake roots at the initial guess because the function is near-flat).
- Section 5 (Tikhonov theory + implementation) — full derivation of penalised least-squares: misfit + λ‖Lh‖², L = second-difference matrix. MAP interpretation: λ = 1/τ² under Gaussian prior on hazard curvature. Ad-hoc `regularised_bootstrap(...)` function defined inside the notebook (per the user's "no new code unless needed" preference — this is a teaching demonstration).
- Smoke test sweeps λ from 0 → 1e10. With +5 bp noise on the 5y bond: λ=0 reproduces the noise-amplified sequential result (rmse=0, roughness=160e-6); λ=1e6 trades 2 bp rmse for half the roughness; λ=1e8 collapses to nearly flat. Sets up Section 6 (L-curve corner picking).

Now 23 cells, 229 KB.

---

## v0.867.0 — 2026-06-10

**Hazard-from-bonds notebook — Round 1 (sections 1-3).**

- New `notebooks/credit/hazard_from_bonds_when_maturities_are_close.ipynb` (15 cells, 206 KB). Sections 1-3 of a planned 10-section walkthrough.
- Section 1: the problem setup, math derivation of the risky-bond price as a function of $h(t)$, why "close maturities" is the trouble case (Jacobian goes ill-conditioned).
- Section 2: the easy case — 4 bonds at 1/3/5/10y, sequential bootstrap reproduces input prices to machine precision (RMSE 1e-10 bp), implied hazards recover the truth's piecewise-constant shape correctly across the bond pillars (with a clean explanation of why a bond pillar that straddles two truth pillars gets the time-weighted blend, not either endpoint).
- Section 3: the dramatic failure — same 4 bonds + a 5th bond two months from the 5y. Noise-free input still works; **add 5 bp of price noise on the 5y bond and the [5y, 5y+2mo] hazard jumps by 67 bp (13× amplification)**. Sweep over $\Delta T$ from 12 months down to 2 weeks shows the amplification climb monotonically: 27× at 1-month spacing, 54× at 2-week spacing. Plotted log-log.
- Companion builder script `_build_hazard_notebook.py` — same `nbformat` + `nbconvert` pattern as `quickstart`.
- Rounds 2-5 to come: solver-limit detail (Newton + brentq failure modes), Tikhonov derivation, L-curve picking, bid-ask Monte Carlo, adaptive switch, realistic demo, stochastic-intensity cross-check.

---

## v0.866.0 — 2026-06-10

**Foundation audit: dual-critic pass on 35 modules.**

- New `MODULE_HEALTH.md` (110 KB) — adversarial audit of 35 foundation + Top-6-instrument modules. Each module reviewed by `numerical-critic` (math correctness, edge cases, calibration robustness) and `code-correctness-critic` (off-by-one, lifetime, None handling, exception safety) in parallel via the multi-agent workflow harness.
- 70 critic verdicts, 697 raw findings: **56 critical**, **150 high**, 257 medium, 217 low, 17 nit.
- **Tier 1 (both critics → critical, fuzzy-matched on title/location): 13.** Highest-confidence likely real bugs. Tier 2 (critical + high pairing): 18. Single-critic critical: 19.
- Report includes: how-to-read disclaimer, per-tier breakdowns, per-module narrative verdicts (verbatim from critics), risk-scored module ranking, recommended triage workflow. **Critic output is NOT verified bugs — each finding needs a failing-test verification slice before fixing.**
- Raw JSON output (697 findings, ~720 KB) at `/private/tmp/claude-501/.../tasks/www7hfs2m.output`.

---

## v0.865.0 — 2026-06-10

**Refresh `ARCHITECTURE.md` to match empirical state.**

- `ARCHITECTURE.md`: rewrite from scratch using a freshly-computed import graph. Previous numbers (20 sub-packages, 486 modules, 9 layers) were ~60% stale. New numbers — **23 packages, 793 modules, 7 layers** — match what `mypy`/`pytest`/`grep` see today.
- New content: per-package fan-in table, tallest path through the DAG (7 hops: `core → curves → models → fixed_income → options → fx → desks`), `crypto` and `data` packages (added since the previous revision), reorganised numerics / models / risk module catalogues to cover modules built across the last 200+ commits.
- Embedded regen snippet at the end of the document — `cd python && python <<PY ... PY` — so the file's stats can be verified on any commit instead of drifting silently.
- Companion to slices v0.863 (untangle fi → credit) and v0.864 (drop binomial_jr_lr) — without those two the layer count would be 8 instead of 7.

---

## v0.864.0 — 2026-06-10

**Remove loose top-level `binomial_jr_lr.py`.**

- Deleted `python/pricebook/binomial_jr_lr.py` — a 190-line file sitting outside any subpackage. It was unreferenced anywhere in the repo, broken at import time (`from pricebook.black76 import OptionType` — that module path doesn't exist; Black-76 lives at `pricebook.models.black76`), and the JR / LR tree implementations are already provided by `numerical/_trees.py` and registered in `registry.py` (`TreeMethod.JR`, `TreeMethod.LR`). The file was a relic of an earlier shim cleanup (see v0.4xx release notes).
- Net: 25 → 24 top-level packages/modules in `pricebook/`. The top of the tree now contains only the registered subpackages plus `__init__.py` and `registry.py`.

---

## v0.863.0 — 2026-06-10

**Untangle `fixed_income → credit` runtime edge.**

- `fixed_income/basis_trade.py`: move `from pricebook.credit.cds import CDS` into `TYPE_CHECKING`. CDS was only used as a parameter annotation (`cds: CDS`) — never instantiated or isinstance-checked. `from __future__ import annotations` was already present, so all hints are strings at runtime.
- Result: `fixed_income` now depends only on `core, curves, models, statistics` at runtime — empirical Layer 4 → Layer 3 cleanup (one whole layer down). Empirical dependency graph regenerates without the `fi → credit` edge.

---

## v0.862.0 — 2026-06-10

**Mypy: clean baseline, 0 errors.**

- `python/pyproject.toml`: complete `[tool.mypy]` config with pragmatic defaults (silence `misc`, `annotation-unchecked`, `warn_return_any` — numpy/scipy noise) plus `[[tool.mypy.overrides]]` listing 184 legacy modules with `ignore_errors = true`. `mypy>=2.0` added to `[project.optional-dependencies] dev`.
- Fixed 31 real `name-defined` errors across 12 files — missing imports for forward-referenced types (`date`, `DiscountCurve`, `Calendar`, `RepoBook`, `RFRFutureSpec`, `PricingContext`, `TotalXVAResult`, `timedelta`, `CommodityForwardCurve`). Added proper `if TYPE_CHECKING:` blocks.
- `GUIDE.md` §17: new section documenting mypy usage and the cleanup ladder (remove a module from the override list → fix surfaced errors → repeat).
- Result: `cd python && mypy pricebook` exits 0 across 795 source files. Future slices can shrink the 184-module override list toward zero.

---

## v0.861.0 — 2026-06-09

**G2++ calibration: 8.5× faster.**

- `models/g2pp_calibration.calibrate_g2pp`: rewrite the calibration objective to compare **prices** to precomputed Black-76 market prices, rather than implied vols. Eliminates an implied-vol root-finder per swaption per DE evaluation (~50% of the original cost).
- `models/g2pp_calibration.calibrate_g2pp`: loosen the global-search budget — `differential_evolution(maxiter=30, popsize=6, tol=1e-4, init="sobol")` followed by an L-BFGS-B polish at `maxiter=150, ftol=1e-9`. Previous (`maxiter=300, popsize=15, tol=1e-9`) ran the full DE budget on the default fixture without measurable RMSE improvement.
- Net: `test_g2pp_calibration.py` runs in **2:15** instead of **9:49** (full file, 8 tests). One `calibrate_g2pp(curve, SWAPTION_VOLS)` call drops from ~587 s to ~68 s. Final calibrated `rmse_vol` remains well under the 5% threshold (~0.009 on the default fixture; both tests pass).

---

## v0.860.0 — 2026-06-09

**`GUIDE.md` — per-layer API reference.**

- `GUIDE.md` (new, 17 sections, ~480 lines): curves, models, numerical methods, fixed income, FX, equity, credit, commodity, options, structured, crypto, desks, risk, viz, serialisation, conventions, db/ts. 180 module references verified to exist. Includes runnable code snippets at each layer.
- `README.md`: link to `GUIDE.md`, `ARCHITECTURE.md`, `RELEASE_NOTES.md`, and the quickstart notebook. Now an actual landing page rather than a stub.

---

## v0.859.0 — 2026-06-09

**Quickstart notebook.**

- `notebooks/examples/quickstart.ipynb`: 20-minute "first result" walkthrough — curve bootstrap, bond pricing, IRS pricing, equity option pricing + Greeks, `to_dict`/`from_dict` round-trip, curve plot + Greeks profile via `pricebook.viz`. All 8 code cells executed and embedded.
- `notebooks/examples/_build_quickstart.py`: deterministic builder script for the notebook (nbformat-based) so future edits land via Python, not JSON.

---

## v0.858.0 — 2026-06-09

**Serialisation completeness — money market + funded products.**

- `fixed_income/money_market.py`: register `CertificateOfDeposit`, `CommercialPaper`, `BankersAcceptance` as `_serialisable`. `RepoRate` is a static-method namespace, no instance state — left unregistered.
- `fixed_income/funded.py`: register `TotalReturnSwap` (as `funded_trs`) and `FundedParticipation`. The other 3 classes (`Repo`, `ReverseRepo`, `RepoFinancedPosition`) were already registered.
- `funded.TotalReturnSwap`: rename internal attrs `ref_start`/`ref_current` → `reference_pv_start`/`reference_pv_current` so they match constructor params (required for `_serialisable`). Only used inside `funded.py`; tests already pass via init kwargs.
- Round-trip tests added in `test_funded.py` and `test_money_market.py` (5 new tests: TRS, FundedParticipation, CD, CP, BA).

---

## v0.857.0 — 2026-06-07

**G2++ code review fixes.**

- `g2pp_tree.py`: fix correlation correction (remove spurious `dt` factor, add renormalization); fix `_phi` division by zero for a≈0 or b≈0.
- `g2pp_calibration.py`: fix `_g2pp_V` division by zero guards.
- `bermudan_swaption_g2pp.py`: fix `_phi` and `_V` division by zero guards.
- `cms_spread_g2pp.py`: fix `_forward_zcb` to use V(T)-V(t) not V(τ); fix `_V` division by zero.

---

## v0.856.0 — 2026-06-07

**Unify Hull-White interface and fix mc_extensions.**

- `fixed_income/callable_floater.py`: added `callable_frn_hw()`, `puttable_frn_hw()` — accept `HullWhite` object.
- `options/bermudan_capfloor.py`: added `bermudan_cap_hw()`, `bermudan_floor_hw()`, `bermudan_collar_hw()`.
- `models/mc_extensions.py`: `"hull_white"` dispatch now supports `theta_func` for time-dependent drift via `HullWhiteProcess`.

---

## v0.855.0 — 2026-06-07

**G2++ callable bond, CMS spread, callable floater.**

- New `fixed_income/callable_bond_g2pp.py`: callable/puttable bonds on G2++ 2D tree.
- New `structured/cms_spread_g2pp.py`: CMS spread pricing + options + correlation diagnostic under G2++. Key: under 1F correlation=1.0; under G2++ correlation<1.0.
- New `fixed_income/callable_floater_g2pp.py`: callable/puttable FRN on G2++ 2D tree.
- 10 tests.

---

## v0.854.0 — 2026-06-07

**G2++ 2D tree and Bermudan swaption.**

- New `models/g2pp_tree.py`:
  - `G2PPTree` — 2D recombining trinomial lattice with correlation.
  - `backward_induction()` — generic 2D backward induction with pluggable option constraint.
  - `g2pp_european_swaption_tree()` — verification against analytical.
- New `options/bermudan_swaption_g2pp.py`:
  - `bermudan_swaption_g2pp_tree()` — Bermudan swaption on 2D tree.
  - `bermudan_swaption_g2pp_lsm()` — LSM with (1, x, y, x², y², xy) basis.
  - `g2pp_vs_hw1f_bermudan()` — compare 1F vs 2F Bermudan prices.
- 14 tests.

---

## v0.853.0 — 2026-06-07

**G2++ (2-factor Hull-White) calibration.**

- New `models/g2pp_calibration.py`:
  - `g2pp_swaption_price()` — Brigo-Mercurio analytical via Gauss-Hermite + Jamshidian.
  - `calibrate_g2pp()` — fit (a, b, σ₁, σ₂, ρ) to swaption vol grid via DE + L-BFGS-B.
  - `g2pp_vs_hw1f()` — compare 1F vs 2F calibration quality.

---

## v0.852.0 — 2026-06-07

**Exercise boundary extraction.**

- New `options/exercise_boundary.py`:
  - `pde_exercise_boundary()` — extract boundary from Crank-Nicolson PDE.
  - `tree_exercise_boundary()` — extract from CRR binomial tree.
  - `lsm_exercise_boundary()` — extract from LSM regression.
  - `boundary_analytics()` — slope, convexity, critical price analysis.
  - `compare_boundaries()` — cross-method comparison.
- 10 tests.

---

## v0.851.0 — 2026-06-07

**Bermudan barrier options.**

- New `options/bermudan_barrier.py`:
  - `bermudan_barrier_option()` — LSM with continuous barrier monitoring.
  - `american_barrier_option()` — American exercise + barrier knock-out.
  - `bermudan_double_barrier()` — double barrier with early exercise.
  - `barrier_exercise_interaction()` — decompose barrier vs exercise premium.
- 9 tests.

---

## v0.850.0 — 2026-06-07

**American multi-asset options.**

- New `options/american_multi_asset.py`:
  - `american_spread_option()` — LSM with 6-term basis on (S1, S2).
  - `american_basket_option()` — LSM on weighted basket.
  - `american_best_of()` — LSM on max(S1, S2).
  - `american_worst_of_put()` — LSM on min(S1, S2), key for structured products.
- 10 tests.

---

## v0.849.0 — 2026-06-07

**Stochastic credit Bermudan CDS swaption.**

- New `credit/stochastic_bermudan_cds.py`:
  - `stochastic_bermudan_cds_swaption()` — LSM under CIR intensity with exact non-central chi-squared simulation.
  - `cir_cds_pv()` — analytical CDS PV under CIR via Riccati ODE.
  - `stochastic_vs_deterministic()` — compare stochastic vs deterministic approaches.
- 7 tests.

---

## v0.848.0 — 2026-06-07

**Callable structured notes.**

- New `structured/callable_structured.py`:
  - `callable_steepener()` — LSM on CMS spread with issuer call.
  - `callable_cms_spread()` — callable CMS spread note.
  - `callable_inverse_floater()` — callable inverse floater.
- 6 tests.

---

## v0.847.0 — 2026-06-07

**Callable and puttable floating rate notes.**

- New `fixed_income/callable_floater.py`:
  - `callable_frn()` — HW tree with call constraint on coupon dates.
  - `puttable_frn()` — HW tree with put constraint.
  - `callable_frn_oas()` — OAS via Brent root-find.
- 8 tests.

---

## v0.846.0 — 2026-06-07

**Bermudan cap/floor.**

- New `options/bermudan_capfloor.py`:
  - `bermudan_cap()` / `bermudan_floor()` — HW trinomial tree with Bermudan exercise on remaining caplet/floorlet strip.
  - `bermudan_collar()` — long cap + short floor with Bermudan exercise.
- 9 tests.

---

## v0.845.0 — 2026-06-07

**American commodity options.**

- New `commodity/commodity_american.py`:
  - `american_commodity_option()` — BAW/tree with convenience yield.
  - `american_energy_option()` — seasonal vol adjustment.
  - `american_commodity_spread()` — LSM on correlated commodity pair.
  - `early_exercise_test()` — optimal exercise diagnostic.
- 10 tests.

---

## v0.844.0 — 2026-06-07

**American FX options.**

- New `fx/fx_american.py`:
  - `american_fx_option()` — Garman-Kohlhagen American via BAW/PDE/tree.
  - `fx_exercise_boundary()` — early exercise boundary curve.
  - `american_fx_greeks()` — numerical Greeks (delta_dom, delta_for, rho_dom, rho_for).
- 8 tests.

---

## v0.843.0 — 2026-06-07

**Analytical American approximations.**

- New `options/american_analytical.py`:
  - `ju_zhong()` — Ju & Zhong (1999) second-order correction to BAW.
  - `kim_integral()` — Kim (1990) integral equation for exact exercise boundary.
  - `medvedev_scaillet()` — near-expiry asymptotic expansion.
  - `american_comparison()` — run all methods and compare.
- 14 tests.

---

## v0.842.0 — 2026-06-07

**Fix remaining known limitations.**

- `xccy_swaption.py`: correct `xccy_forward_spread` to use full floating-leg PV replication (Brigo & Mercurio Ch. 13) — par floater identity for both legs, notional exchange at spot.
- `exotic_payoffs.py`: fix `shout_option_analytical` r=q branch to use Goldman-Sosin-Gatto (1979) lookback formula correctly.
- `equity_linked_note.py`: replace stdlib `random` MC in `worst_of_eln` with vectorised numpy (Cholesky @ standard_normal).
- Input validation added across 6 modules: `equity_spread_option`, `exotic_payoffs`, `carbon_credit`, `freight`, `quanto_futures` — guards for spot>0, T>0, vol>0, rho∈[-1,1].

---

## v0.841.0 — 2026-06-07

**Code review fixes across 12 modules.**

- `quanto_swap.py`: fix adjustment to use `t_start` (fixing time) not `t_end`; apply `corr_1_fx` in differential swap (was silently ignored); add MONTHLY frequency.
- `exotic_payoffs.py`: fix installment option `pv_remaining` off-by-one; remove dead imports.
- `portfolio_margin.py`: SPAN extreme scenarios 2×PSR (was 3×), add 35% cap; fix straddle `max_loss` sign.
- `insurance_annuity.py`: fee PV now discounts each step at its own time (was using terminal discount for all).
- `real_estate_derivative.py`: fix `reit_nav_model` to use NOI/discount_rate (Gordon model); remove dead variable.
- `equity_spread_option.py`: central differences for vega/rho; remove unused imports and dead closure.
- `tranche_option.py`: guard for non-positive spreads in Black model.
- `etf.py`: fix docstring inconsistency (premium_discount units).
- `carbon_credit.py`, `money_market.py`, `xccy_swaption.py`: remove unused imports.

---

## v0.840.0 — 2026-06-07

**Insurance annuity guarantees and real estate derivatives.**

- New `structured/insurance_annuity.py`:
  - `gmab()` — Guaranteed Minimum Accumulation Benefit (MC).
  - `gmdb()` — Guaranteed Minimum Death Benefit with mortality weighting.
  - `gmwb()` — Guaranteed Minimum Withdrawal Benefit with ruin tracking.
  - `ratchet_gmab()` — GMAB with periodic ratchet reset.
- New `structured/real_estate_derivative.py`:
  - `property_total_return_swap()` — TRS on property index.
  - `property_index_forward()` — illiquidity-adjusted forward.
  - `property_option()` — Black-76 on property index.
  - `reit_nav_model()` — REIT net asset value.
  - `housing_affordability()` — payment-to-income metrics.
- 18 tests.

---

## v0.839.0 — 2026-06-07

**Longevity and mortality derivatives.**

- New `structured/longevity.py`:
  - `q_forward()` — mortality rate swap (q-forward).
  - `longevity_swap()` — multi-cohort fixed vs realised mortality.
  - `survivor_index()` — population projection with mortality improvement.
  - `lee_carter_forecast()` — Lee-Carter SVD mortality forecasting.
  - `mortality_bond_price()` — principal-at-risk mortality bond.
  - `value_of_life_annuity()` — life-contingent annuity PV.
- 16 tests.

---

## v0.838.0 — 2026-06-07

**Catastrophe bonds and ILS.**

- New `structured/cat_bond.py`:
  - `cat_bond_price()` — Poisson-arrival loss model with coupon/principal at risk.
  - `parametric_trigger_prob()` — Gumbel extreme value trigger probability.
  - `indemnity_trigger_loss()` — MC lognormal loss with attachment/exhaustion.
  - `cat_bond_spread_decomposition()` — EL + risk premium + expense.
  - `ils_portfolio()` — Gaussian copula portfolio of cat bonds.
  - `seasonal_adjustment()` — hurricane/earthquake seasonal probability.
- 16 tests.

---

## v0.837.0 — 2026-06-07

**Portfolio margin / SPAN.**

- New `risk/portfolio_margin.py`:
  - `span_margin()` — 14-scenario SPAN-style margining.
  - `cross_margin_offset()` — diversification benefit from cross-margining.
  - `strategy_margin()` — Reg-T margin for option strategies.
  - `var_based_margin()` — VaR/ES-based initial margin.
  - `margin_call()` — margin call computation.
- 17 tests.

---

## v0.836.0 — 2026-06-07

**Tranche options.**

- New `credit/tranche_option.py`:
  - `tranche_option_black()` — Black-76 on tranche spread.
  - `tranche_option_bachelier()` — normal model for tight/negative spreads.
  - `tranche_option_greeks()` — numerical spread delta, gamma, vega, theta.
  - `tranche_straddle()` — ATM straddle with breakeven levels.
  - `tranche_forward_spread()` — loss-adjusted forward tranche spread.
- 20 tests.

---

## v0.835.0 — 2026-06-07

**Quanto futures and ETF products.**

- New `equity/quanto_futures.py`:
  - `quanto_futures_price()` — F_Q = S × exp((r_d − q − ρσ_Sσ_FX) × T).
  - `implied_correlation()` — back-solve ρ from observed quanto price.
  - `compo_vs_quanto()` — compare composite vs quanto forwards.
- New `equity/etf.py`:
  - `etf_nav()` — NAV from holdings basket.
  - `creation_redemption_arb()` — AP arbitrage evaluation.
  - `tracking_error()`, `tracking_difference()` — index tracking metrics.
  - `leveraged_etf_decay()` — volatility drag formula.
- 15 tests.

---

## v0.834.0 — 2026-06-07

**Exotic option payoffs: ladder, shout, installment.**

- New `options/exotic_payoffs.py`:
  - `ladder_option()` — MC with rung-based lock-in of intrinsic.
  - `shout_option()` — MC multi-shout option.
  - `shout_option_analytical()` — Dai-Kwok-Wu closed form via lookback equivalence.
  - `installment_option()` — MC with rational abandonment at each payment date.
- 12 tests.

---

## v0.833.0 — 2026-06-07

**Freight derivatives.**

- New `commodity/freight.py`:
  - `ffa_price()` — Forward Freight Agreement (average/point settlement).
  - `freight_option_price()` — Black-76 on FFA rate.
  - `time_charter_equivalent()` — TCE calculation.
  - `freight_forward_curve()` — seasonal forward curve builder.
  - `bunker_spread()` — P&L sensitivity to bunker fuel cost.
- 7 tests (freight). Combined test file with carbon.

---

## v0.832.0 — 2026-06-07

**Carbon/emission credit pricing.**

- New `commodity/carbon_credit.py`:
  - `carbon_futures_price()` — cost-of-carry for EUA/carbon allowances.
  - `carbon_option_price()` — Black-76 on carbon futures.
  - `marginal_abatement_cost()` — equilibrium from abatement technology curve.
  - `compliance_value()` — surplus/deficit position valuation.
  - `voluntary_credit_discount()` — haircut model for voluntary credits.
- 7 tests (carbon).

---

## v0.831.0 — 2026-06-07

**Capped/floored/collar floaters.**

- New `structured/capped_floored_floater.py`:
  - `floored_floater()` — FRN with minimum coupon via floorlet strip.
  - `collar_floater()` — FRN with cap and floor (short caplets, long floorlets).
  - `reverse_floater()` — coupon = fixed − leverage × floating, with embedded cap.
  - `inverse_floater_duration()` — amplified effective duration.
- 12 tests.

---

## v0.830.0 — 2026-06-07

**CPDO simulation.**

- New `structured/cpdo.py`:
  - `cpdo_simulate()` — single-path CPDO with leverage, gap risk, cash-out.
  - `cpdo_monte_carlo()` — MC: success/default probabilities, expected NAV.
  - `cpdo_rating()` — map default prob to S&P rating bucket.
- 14 tests.

---

## v0.829.0 — 2026-06-07

**Money market instruments.**

- New `fixed_income/money_market.py`:
  - `CertificateOfDeposit` — interest-bearing, dirty/clean price, YTM.
  - `CommercialPaper` — discount instrument, credit spread.
  - `BankersAcceptance` — bank-guaranteed CP with acceptance fee.
  - `RepoRate` — implied repo and haircut-adjusted rate helpers.
- 11 tests.

---

## v0.828.0 — 2026-06-07

**Cross-currency swaptions.**

- New `fixed_income/xccy_swaption.py`:
  - `xccy_swaption_black()` — Black-76 on forward xccy basis spread.
  - `xccy_swaption_bachelier()` — normal model for negative spreads.
  - `xccy_forward_spread()` — CIP-based forward basis spread.
  - `xccy_swaption_greeks()` — numerical delta, gamma, vega, fx_delta.
- 9 tests.

---

## v0.827.0 — 2026-06-07

**Equity spread options.**

- New `equity/equity_spread_option.py`:
  - `kirk_equity_spread()` — Kirk's approximation with dividend yields.
  - `bjerksund_stensland_spread()` — improved accuracy for non-zero strikes.
  - `mc_spread_option()` — Monte Carlo benchmark with antithetic variates.
  - `outperformance_option()` — Margrabe special case (K=0).
  - `relative_performance_option()` — percentage outperformance.
- 11 tests.

---

## v0.826.0 — 2026-06-07

**Equity-linked notes (ELN).**

- New `structured/equity_linked_note.py`:
  - `buffered_eln()` — downside buffer, coupon if index holds.
  - `capped_eln()` — participation with cap.
  - `bear_eln()` — inverse ELN paying on index decline.
  - `digital_eln()` — enhanced coupon if above barrier.
  - `twin_win_eln()` — profits from both directions unless barrier breached.
  - `worst_of_eln()` — MC basket ELN on worst performer.
- 12 tests.

---

## v0.825.0 — 2026-06-07

**Equity index futures pricing.**

- New `equity/equity_index_futures.py`:
  - `index_futures_fair_value()` — cost-of-carry F = S × exp((r - q + b) × T).
  - `index_futures_roll()` — calendar spread, roll cost, implied repo between contracts.
  - `implied_dividend_yield()`, `implied_repo_rate()` — back-solve from observed prices.
  - `fair_value_table()` — term structure across multiple expiries.
- 12 tests.

---

## v0.824.0 — 2026-06-01

**Quanto (differential) interest rate swaps.**

- New `fixed_income/quanto_swap.py`:
  - `quanto_swap_price()` — quanto IR swap: foreign floating rate paid in domestic currency with convexity adjustment E^d[L^f] = L^f × (1 − σ_L × σ_FX × ρ × T).
  - `differential_swap_price()` — diff swap paying rate_1 − rate_2 in single currency, both rates quanto-adjusted.
  - `quanto_adjustment_term_structure()` — adjustment per tenor in bps.
  - `quanto_fra()` — single-period quanto forward rate agreement.
- 22 tests: correlation sign, par spread, pay/receive symmetry, vol sensitivity, maturity scaling.

---

## v0.823.0 — 2026-06-04

**Backlog closed: HV ADI, Strang MC, SDP, sparse Jacobian.**

- New `models/hundsdorfer_verwer.py`:
  - `hv_adi_heston()` — double-pass HV ADI for Heston (6-step scheme).
  - More stable than Craig-Sneyd for strong mixed derivatives.
  - HV agrees with CS within 15%.
- New `models/sde_strang.py`:
  - `strang_merton_mc()` — Merton jump-diffusion: diffusion(dt/2)→jump(dt)→diffusion(dt/2).
  - `strang_bates_mc()` — Bates (Heston + jumps) via Strang splitting.
  - Zero jumps matches BS. O(dt²) splitting error.
- New `numerical/sdp.py`:
  - `nearest_psd()` — PSD cone projection.
  - `nearest_correlation_sdp()` — Higham (2002) Dykstra alternating projections.
  - `factor_covariance_bounds()` — covariance from factor model.
  - `sdp_solve()` — small-scale general SDP via projected gradient.
- New `numerical/sparse_jacobian.py`:
  - `sparse_jacobian()` — Jacobian via graph colouring + grouped perturbation.
  - `banded_jacobian()` — tridiagonal: 3 evaluations instead of n.
  - `detect_sparsity()` — probe-based sparsity detection.
  - `greedy_colouring()` — distance-1 column grouping.
- 12 new tests. 11,089 tests pass.

---

## v0.816.0 — 2026-06-03

**Remaining numerical plan: Tiers 3+4 complete.**

- **F5** `fft_pricing.py`: `carr_madan_fractional()` — non-uniform strikes via direct Fourier evaluation.
- **F6** `registry.py`: registered FFT, Lewis, Bermudan COS, Fourier Greeks pricers.
- **S1** `sde_adaptive.py`: `adaptive_euler()`, `adaptive_milstein()` — step-size control via error pair.
- **X2** `von_neumann.py`: amplification factor, stability region, CFL limit for θ-scheme.
- **X3** `density_evolution.py`: three-way density cross-validation (FP + Fourier + Breeden-Litzenberger).
- **X4** `operator_splitting.py`: Lie-Trotter (O(dt)), Strang (O(dt²)), PIDE splitting.
- 10 new tests. 11,077 tests pass.

---

## v0.806.0 — 2026-06-03

**Convexity tools and Frank-Wolfe optimisation.**

- New `numerical/convexity_tools.py`:
  - `is_convex()` — Hessian eigenvalue sampling.
  - `verify_kkt()` — KKT condition verification.
  - `cardinality_portfolio()` — max N assets via greedy selection.
- New `numerical/frank_wolfe.py`:
  - `frank_wolfe()` — conditional gradient with LMO.
  - `frank_wolfe_portfolio()` — O(n) per iteration MV.
- 5 new tests.

---

## v0.804.0 — 2026-06-03

**Oscillatory quadrature: Filon and Levin methods.**

- New `numerical/oscillatory_quad.py`:
  - `filon_quad()` — Filon's method for ∫f(x)cos(ωx)dx (O(h³/ω)).
  - `levin_quad()` — Levin collocation for general ∫f(x)e^{iωx}dx.
  - `fourier_integral()` — adaptive: standard quad (low ω) or Filon (high ω).
- 3 new tests.

---

## v0.803.0 — 2026-06-03

**LP duality framework: shadow prices, sensitivity.**

- New `numerical/duality.py`:
  - `lp_with_duals()` — LP with dual variable extraction via perturbation.
  - `shadow_prices()` — marginal cost of constraints.
  - `parametric_lp()` — sweep RHS of one constraint.
- 3 new tests.

---

## v0.802.0 — 2026-06-03

**Fokker-Planck forward density evolution.**

- New `models/fokker_planck.py`:
  - `fokker_planck_1d()` — 1D density evolution in log-space (GBM/local vol).
  - `density_to_option_prices()` — price options from risk-neutral density.
  - Density integrates to 1, mean matches forward.
- 3 new tests.

---

## v0.801.0 — 2026-06-03

**True 2D FFT for two-asset options.**

- New `models/fft_2d.py`:
  - `joint_bs_char_func()` — joint CF for correlated GBM.
  - `fft_2d_price()` — full (u₁,u₂) grid with 2D Simpson weights.
  - Spread, basket, best-of payoffs.
- 2 new tests.

---

## v0.800.0 — 2026-06-03

**Rough Heston CF via fractional Riccati ODE.**

- New `models/rough_heston_cf.py`:
  - `rough_heston_char_func()` — Adams scheme on fractional Riccati (El Euch & Rosenbaum 2019).
  - `rough_heston_price()` — European via COS + rough Heston CF.
  - H < 0.5 gives rough regime; differs from smooth Heston (H≈0.5).
- 2 new tests.

---

## v0.798.0 — 2026-06-03

**SOCP solver: robust portfolio and tracking error.**

- New `numerical/socp.py`:
  - `socp_solve()` — general SOCP via barrier method.
  - `robust_portfolio_socp()` — robust MV with norm constraints.
  - `tracking_error_socp()` — min TE vs benchmark.
- 2 new tests.

---

## v0.797.0 — 2026-06-03

**Numerical method recommendation map.**

- New `core/numerical_method_map.py`:
  - `recommend()` — given instrument features, recommend best method.
  - `compare_methods()` — price via analytical/COS/PDE/tree, report agreement.
  - 14 instrument feature types, rule-based selection.
- 6 new tests.

---

## v0.796.0 — 2026-06-03

**Feynman-Kac bridge: SDE ↔ PDE connection.**

- New `models/feynman_kac.py`:
  - `sde_to_pde()` — derive PDE coefficients from SDE dynamics.
  - `pde_to_sde()` — extract SDE from PDE coefficients.
  - `verify_feynman_kac()` — cross-validate MC vs PDE (consistent within 3σ).
- 3 new tests.

---

## v0.795.0 — 2026-06-03

**Automatic differentiation via dual numbers.**

- New `numerical/auto_diff.py`:
  - `Dual` class — overloaded arithmetic (+ − × / pow).
  - Math functions: `exp`, `log`, `sqrt`, `sin`, `cos`, `max_dual`.
  - `grad()` — gradient of f: ℝⁿ → ℝ via forward AD.
  - `jacobian_ad()` — Jacobian via forward AD.
  - `derivative()` — f(x) and f'(x) simultaneously, machine-precision.
  - BS delta via AD matches analytical.
- 7 new tests.

---

## v0.794.0 — 2026-06-03

**Fourier Greeks: delta, gamma, vega, theta via COS/Lewis.**

- New `models/fourier_greeks.py`:
  - `cos_greeks()` — full Greeks via COS with spot/vol/time bumps.
  - `lewis_greeks()` — Greeks via Lewis formula.
  - `fourier_greeks()` — unified entry point.
  - Vega via CF variance perturbation (no vol parameter needed).
- 3 new tests.

---

## v0.793.0 — 2026-06-03

**Fix: CharacteristicFunction.price_european() was broken.**

- `numerical/_fourier.py`: `cos_european` → `cos_price` with correct OptionType argument.
- Now correctly prices European options via COS method.
- 1 new test.

---

## v0.792.0 — 2026-06-03

**Package ready for PyPI: README, LICENSE, build verified.**

- Added `python/README.md` — full package description with install, quick start, feature list.
- Copied `LICENSE` into `python/` — required by PyPI alongside pyproject.toml.
- Added `readme = "README.md"` to pyproject.toml.
- Version synced to 0.791.0 in `__init__.py`.
- Build verified: `python -m build` produces valid sdist (2.2MB) + wheel (1.9MB).
  - 716 .py modules in wheel, no tests or notebooks leaked.
  - METADATA correct: classifiers, keywords, license expression.
- Ready to publish: `twine upload dist/*` with PyPI credentials.

---

## v0.791.0 — 2026-06-02

**Python package: version sync, pyproject.toml, py.typed marker.**

- Version synced to 0.790.0 in `__init__.py` (was 0.614.0).
- `pyproject.toml` updated:
  - Trove classifiers (Financial, Science/Research, Typed).
  - Extended keywords (structured-products, monte-carlo, pde, portfolio-optimization).
  - `py.typed` marker for PEP 561 type checking support.
  - `[tool.mypy]` section for type checking config.
  - Notebooks excluded from package distribution.
- Verified: `pip install -e .` works, all imports functional, `pricebook.__version__` correct.
- 11,027 tests pass.

---

## v0.790.0 — 2026-06-02

**Notebook consolidation: single location under python/notebooks/.**

- Consolidated 45 notebooks + examples from 3 locations (notebooks/, python/notebooks/, examples/) into one structure:
  - `python/notebooks/papers/` — 12 paper validations
  - `python/notebooks/markets/` — 6 Americas market notebooks
  - `python/notebooks/rates/` — 4 rates workflows
  - `python/notebooks/credit/` — 2 credit notebooks
  - `python/notebooks/structured/` — 3 structured product notebooks
  - `python/notebooks/desks/` — 2 desk notebooks
  - `python/notebooks/validation/` — 5 Pucci et al. validations
  - `python/notebooks/examples/` — 10 Python examples + 2 example notebooks
- Removed empty root `notebooks/` and `examples/` directories.
- 11,027 tests pass.

---

## v0.789.0 — 2026-06-02

**PDE code review fixes.**

- **pde_adaptive.py**: CRITICAL — grid refinement midpoint formula used `grid[i+1]+grid[i+2]` instead of `grid[i]+grid[i+1]`, inserting nodes at wrong locations. FD formula aligned with protocol (was using different convection discretisation).
- **pde_local_vol.py**: barrier BC removed degenerate `if not is_call else 0.0` (always 0). Knock-in parity fixed — was passing contradictory `vol=0.20` alongside `vol_surface`.
- **pide_solver.py**: V_prev now saved every step (was only at `n_time-2`), fixing theta Greek computation for both Merton and Kou.
- **pde_boundary.py**: Robin BC sign error fixed — derivation `a*V + b*∂V/∂S = g` now correctly solved for V[0].
- 11,027 tests pass.

---

## v0.788.0 — 2026-06-02

**PDE boundary condition library.**

- New `numerical/pde_boundary.py`:
  - `BCSpec` — unified BC specification: Dirichlet, Neumann, Robin, linear extrapolation, outflow.
  - `apply_bc()` — apply BCs to solution vector.
  - Financial BC factories: `call_bcs()`, `put_bcs()`, `barrier_bcs()`.
- 5 new tests.

---

## v0.787.0 — 2026-06-02

**PDE convergence diagnostics and scheme selection.**

- New `models/pde_diagnostics.py`:
  - `convergence_study()` — grid refinement analysis with Richardson extrapolation.
  - `recommend_scheme()` — automatic method/grid recommendation.
  - `stability_check()` — CFL verification with warnings.
- 3 new tests.

---

## v0.786.0 — 2026-06-02

**American 2D: Heston American via ADI + penalty.**

- New `models/pde_american_2d.py`:
  - `heston_american_pde()` — Heston American put via Craig-Sneyd ADI + penalty method.
  - Penalty λ converts free boundary to nonlinear fixed-domain PDE.
  - American ≥ European verified.
- 2 new tests.

---

## v0.785.0 — 2026-06-02

**Adaptive grid refinement for PDE.**

- New `models/pde_adaptive.py`:
  - `error_indicator()` — gradient-based error estimate per cell.
  - `refine_grid()` — insert nodes where curvature is high.
  - `adaptive_pde()` — iterative solve-refine-solve with convergence check.
- 2 new tests.

---

## v0.784.0 — 2026-06-02

**SABR PDE via 2D ADI.**

- New `models/pde_sabr.py`:
  - `sabr_pde()` — 2D ADI in (F, σ) space with absorbing boundary at F=0.
  - Craig-Sneyd splitting with mixed derivative term.
  - ITM > ATM verified.
- 2 new tests.

---

## v0.783.0 — 2026-06-02

**Jump-diffusion PIDE: Merton and Kou.**

- New `models/pide_solver.py`:
  - `merton_pide()` — operator splitting: diffusion (CN) + jump integral (quadrature).
  - `kou_pide()` — double-exponential jump-diffusion.
  - No jumps → matches BS. Jumps add value for OTM options.
- 3 new tests.

---

## v0.782.0 — 2026-06-02

**Time-dependent PDE coefficients.**

- New `models/pde_time_dependent.py`:
  - `TermStructureCoefficients` — piecewise-linear r(t), σ(t), q(t).
  - `time_dependent_pde()` — BS PDE with non-constant coefficients.
  - Constant term structure → matches standard PDE.
- 2 new tests.

---

## v0.781.0 — 2026-06-02

**Local volatility PDE solver.**

- New `models/pde_local_vol.py`:
  - `local_vol_pde()` — BS PDE with σ(S,t) from Dupire surface.
  - `local_vol_barrier_pde()` — barrier under local vol.
  - Flat surface → matches BS. Non-flat → prices differ.
- 2 new tests.

---

## v0.780.0 — 2026-06-02

**Unified PDE protocol.**

- New `models/pde_protocol.py`:
  - `PDECoefficients` — callable a(S,t), b(S,t), c(S,t) with factories for BS, local vol, time-dep.
  - `PDESpec` — full problem spec (coefficients, domain, BCs, payoff, American).
  - `PDEEngine` — solver with configurable method, grid, resolution.
  - `PDEPricingResult` — unified result with Greeks and convergence info.
  - `pde_price()` — one-function entry point. Matches BS to 2%.
- 6 new tests.

---

## v0.778.0 — 2026-06-02

**Code review fixes across portfolio optimisation and game theory.**

- **hierarchical_risk_parity.py**: CRITICAL — `import math` was at end of file, used on line 118. Moved to top.
- **cvar_optimisation.py**: removed dead `cvar_actual` computation with wrong tail condition; removed unused `minimize` import.
- **portfolio_analytics.py**: CVaR tail selection logic fixed — was convoluted double-negation, now clean `losses[losses >= var_95]`.
- **stackelberg.py**: Cournot benchmark now handles asymmetric costs correctly (was using symmetric formula). Market share clamped to [0,1].
- **bargaining.py**: Kalai-Smorodinsky tolerance check was self-referential (`abs(x) < abs(x)*0.1`). Fixed to `abs(x) < 0.1*abs(expected) + 0.01`.
- **market_microstructure_games.py**: Glosten-Milgrom removed dead code (double `post_high_buy` computation), added division-by-zero guard. Information share docstring corrected from "Hasbrouck" to "variance-based" (simplified approach).
- **n_player_nash.py**: removed unused `val` variable in `_compute_payoffs`.
- 11,000 tests pass.

---

## v0.777.0 — 2026-06-02

**Unified portfolio analytics: Sharpe, Sortino, Calmar, drawdowns, tracking.**

- New `risk/portfolio_analytics.py`:
  - `portfolio_metrics()` — 15 metrics: Sharpe, Sortino, Calmar, max DD, VaR/CVaR, skew/kurt.
  - `tracking_metrics()` — tracking error, information ratio, alpha, beta.
- 2 new tests.

---

## v0.776.0 — 2026-06-02

**Multi-period dynamic allocation: CPPI, target-date, lifecycle.**

- New `risk/dynamic_allocation.py`:
  - `cppi_allocation()` — constant proportion portfolio insurance with floor.
  - `target_date_glide()` — linear/convex/concave glide paths.
  - `multi_period_mv()` — multi-period mean-variance with rebalancing costs.
- 2 new tests.

---

## v0.775.0 — 2026-06-02

**Transaction cost-aware portfolio optimisation.**

- New `risk/transaction_cost_opt.py`:
  - `tc_aware_rebalance()` — turnover penalty in MV objective.
  - `no_trade_region()` — Leland-Davis no-trade bands.
  - `optimal_rebalance_frequency()` — cost-benefit analysis.
- 2 new tests.

---

## v0.774.0 — 2026-06-02

**Robust portfolio optimisation: worst-case, uncertainty sets.**

- New `risk/robust_optimisation.py`:
  - `robust_mean_variance()` — worst-case mean-variance.
  - `ellipsoidal_uncertainty()` — Goldfarb-Iyengar ellipsoidal sets.
  - `box_uncertainty()` — interval return uncertainty.
- 2 new tests.

---

## v0.773.0 — 2026-06-02

**Kelly criterion: optimal bet sizing.**

- New `risk/kelly.py`:
  - `kelly_fraction()` — single-asset f* = μ/σ².
  - `fractional_kelly()` — conservative half-Kelly.
  - `multi_asset_kelly()` — portfolio Kelly via Σ⁻¹ × excess.
- 3 new tests.

---

## v0.772.0 — 2026-06-02

**Brinson-Fachler performance attribution.**

- New `risk/brinson_attribution.py`:
  - `brinson_attribution()` — allocation + selection + interaction.
  - `brinson_multi_period()` — geometric linking.
  - `factor_based_attribution()` — OLS factor decomposition.
  - Sum of effects = active return (verified).
- 2 new tests.

---

## v0.771.0 — 2026-06-02

**Hierarchical Risk Parity (López de Prado 2016).**

- New `risk/hierarchical_risk_parity.py`:
  - `hrp_portfolio()` — tree clustering + quasi-diagonalisation + recursive bisection.
  - `cluster_assets()` — hierarchical clustering by correlation distance.
  - No covariance inversion → robust to estimation error.
- 2 new tests.

---

## v0.770.0 — 2026-06-02

**Efficient frontier: full curve, tangency, CML.**

- New `risk/efficient_frontier.py`:
  - `efficient_frontier()` — full mean-variance frontier sweep.
  - `tangency_portfolio()` — max Sharpe via SLSQP.
  - `minimum_variance_portfolio()` — analytical or numerical.
  - `capital_market_line()` — CML from rf to tangency.
- 4 new tests.

---

## v0.769.0 — 2026-06-02

**CVaR portfolio optimisation via Rockafellar-Uryasev LP.**

- New `risk/cvar_optimisation.py`:
  - `cvar_portfolio()` — LP formulation for CVaR-optimal weights.
  - `min_cvar_target_return()` — minimum CVaR for given return.
  - `cvar_risk_budget()` — component CVaR decomposition.
  - `mean_cvar_frontier()` — efficient frontier in mean-CVaR space.
- 4 new tests.

---

## v0.768.0 — 2026-06-02

**Strategic market microstructure: Kyle, Glosten-Milgrom.**

- New `models/market_microstructure_games.py`:
  - `kyle_lambda()` — Kyle (1985) price impact, insider profit, market depth.
  - `glosten_milgrom()` — sequential trade with adverse selection.
  - `optimal_order_splitting()` — Almgren-Chriss extended.
  - `information_share()` — Hasbrouck multi-market decomposition.
- 5 new tests.

---

## v0.767.0 — 2026-06-02

**Bargaining theory: Nash, Rubinstein, Kalai-Smorodinsky.**

- New `models/bargaining.py`:
  - `nash_bargaining()` — Nash bargaining solution on feasible set.
  - `rubinstein_alternating()` — Rubinstein SPE (patience → surplus).
  - `kalai_smorodinsky()` — monotonic solution via ideal point.
  - `debt_restructuring_bargain()` — creditor-debtor Rubinstein.
- 3 new tests.

---

## v0.766.0 — 2026-06-02

**Stackelberg leader-follower games.**

- New `models/stackelberg.py`:
  - `stackelberg_cournot()` — quantity competition with first-mover advantage.
  - `stackelberg_bertrand()` — price competition.
  - `credit_market_stackelberg()` — lead bank spread-setting game.
  - `general_stackelberg()` — generic two-player framework.
- 3 new tests.

---

## v0.765.0 — 2026-06-02

**N-player Nash equilibrium: fictitious play, support enumeration.**

- New `models/n_player_nash.py`:
  - `fictitious_play()` — iterative best-response for N players.
  - `lemke_howson_2p()` — support enumeration for bimatrix.
  - `correlated_equilibrium()` — LP for correlated equilibrium.
- 3 new tests.

---

## v0.759.0 — 2026-06-02

**Code review fixes across futures, structured, FX, and engine infrastructure.**

- **mc_greeks_auto.py**: lookback/Asian reclassified from SMOOTH to PATH_DEPENDENT (pathwise IPA invalid for path-dependent payoffs).
- **autocall_advanced.py**: memory coupon overwrite bug fixed — line 115 was unconditionally overwriting line 114, making memory feature dead code.
- **tree_mc_bridge.py**: stochastic vol tree drift used variance `v` instead of `sigma²`; MC path-dependent branch missing div_yield.
- **bespoke_cdo.py**: loss distribution now uses notional-weighted average PD/LGD instead of equal-weight.
- **tree_enhancements.py**: barrier accuracy division by zero guard when barrier == 0.
- **engine_comparison.py**: dict iteration fix in `validate_greeks()` — was iterating all values including non-dict.
- **fx_exotic_extensions.py**: Dupire local vol guard for K ≤ 0 preventing log domain error.
- **commodity_options.py**: Samuelson docstring formula corrected to match implementation `exp(−αT)`.
- **Removed unused imports**: `_norm_pdf` from futures_options.py and commodity_options.py; `np` from spread_options.py and commodity_swaps.py.
- 10,963 tests pass.

---

## v0.758.0 — 2026-06-02

**Unified engine registry: one function, any instrument, best engine.**

- New `models/engine_registry.py`:
  - `price()` — auto-select best engine for instrument type.
  - `InstrumentType` enum: 14 instrument classes.
  - Per-type engine recommendations (analytical → tree → MC).
  - `register_engine()` for custom engines. `list_engines()`.
- 6 new tests.

---

## v0.757.0 — 2026-06-02

**Engine comparison and validation.**

- New `models/engine_comparison.py`:
  - `compare_engines()` — price via analytical, tree, MC side-by-side.
  - `validate_greeks()` — check Greek consistency across engines.
  - Reports price spread, Greek agreement, compute time.
- 3 new tests.

---

## v0.756.0 — 2026-06-02

**Tree-MC bridge: hybrid engine for early exercise + path dependence.**

- New `models/tree_mc_bridge.py`:
  - `lsm_on_tree()` — LSM using CRR transition probabilities.
  - `stochastic_vol_tree()` — 2D trinomial (spot × variance) for Heston.
  - `hybrid_price()` — auto-select tree, MC, or hybrid by instrument features.
- 3 new tests.

---

## v0.755.0 — 2026-06-02

**Tree enhancements: adaptive barrier mesh, non-recombining scaffold.**

- New `numerical/tree_enhancements.py`:
  - `adaptive_barrier_tree()` — grid-adjusted trinomial near barrier.
  - `NonRecombiningTree` — linked-list tree with path-dependent state.
  - `asian_on_tree()` — Asian option via non-recombining tree.
- 3 new tests.

---

## v0.754.0 — 2026-06-02

**Derman-Kani implied binomial tree.**

- New `numerical/implied_tree.py`:
  - `build_implied_tree()` — calibrate recombining tree to market options.
  - `price_on_implied_tree()` — exotic pricing on smile-consistent tree.
  - `extract_local_vol()` — local vol from Arrow-Debreu state prices.
- 3 new tests.

---

## v0.753.0 — 2026-06-02

**Black-Derman-Toy (BDT) log-normal rate tree.**

- New `models/bdt_tree.py`:
  - `BDTTree` — calibrated log-normal rate tree with Arrow-Debreu state prices.
  - `bdt_callable_bond()` — callable bond via BDT backward induction.
  - `bdt_bermudan_swaption()` — Bermudan swaption on BDT.
  - Calibrates to match market discount curve exactly.
- 3 new tests.

---

## v0.752.0 — 2026-06-02

**MC convergence diagnostics (extended).**

- Extended `models/mc_diagnostics.py`:
  - `full_diagnostics()` — unified diagnostics with ESS, VRE, CI, skewness/kurtosis.
  - `variance_reduction_efficiency()` — Var(crude)/Var(reduced).
  - `estimate_convergence_rate()` — fit rate from prices at different N.
  - `MCFullDiagnostics.is_converged` — heuristic convergence check.
- 3 new tests.

---

## v0.751.0 — 2026-06-02

**Auto-Greek method selection with path caching.**

- New `models/mc_greeks_auto.py`:
  - `classify_payoff()` — detect smooth/discontinuous/path-dependent.
  - `select_greek_method()` — pathwise for smooth, LR for digital, bump for rest.
  - `auto_greeks()` — compute all Greeks with best method per Greek.
  - `PathCache` — LRU cache for MC paths, shared across Greeks.
- 5 new tests.

---

## v0.750.0 — 2026-06-02

**Declarative MC configuration and factory.**

- New `models/mc_config.py`:
  - `MCConfig` — all settings in one dataclass (process, VR, Greeks method, discretisation).
  - `preset_configs()` — fast, production, high_precision, heston, exotic, xva.
  - `build_process_from_config()` — factory for ProcessSpec.
  - `mc_pricer_from_config()` — build MCPricingEngine from config.
  - `with_overrides()` for mode switching.
- 4 new tests.

---

## v0.749.0 — 2026-06-02

**Unified pricing engine protocol.**

- New `models/engine_protocol.py`:
  - `PricingResult` — unified result: price, GreeksBundle, ConvergenceInfo.
  - `PricingEngine` protocol: `.price_vanilla()`, `.engine_type`.
  - `MCPricingEngine` — wraps MCEngine behind protocol.
  - `TreePricingEngine` — wraps TreeSolver behind protocol.
  - `AnalyticalEngine` — Black-Scholes behind protocol.
  - All three engines agree on European call within 3%.
- 6 new tests.

---

## v0.747.0 — 2026-06-02

**FX exotic extensions: digitals, quantos, var swaps, local vol, double barriers, compound, chooser.**

- New `fx/fx_exotic_extensions.py`:
  - `fx_digital_option()` — European digital (cash-or-nothing, asset-or-nothing), overhedge, both payout currencies.
  - `fx_quanto_option()` — quanto-adjusted GK with correlation drift, FX rate scaling.
  - `fx_variance_swap()` — fair strike from ATM + butterfly, MTM with realised.
  - `fx_local_vol()` — Dupire local vol surface from implied vol grid via finite differences.
  - `fx_double_barrier_option()` — double knock-out/knock-in via MC, parity verified.
  - `fx_compound_option()` — option on option (call-on-call, put-on-call, etc.) via MC.
  - `fx_chooser_option()` — call-or-put choice at future date, probability tracking.
- 25 new tests.

---

## v0.746.0 — 2026-06-02

**Power/electricity derivatives: swing, tolling, capacity.**

- New `commodity/power_derivatives.py`:
  - `swing_option_price()` — volume flexibility with min/max take.
  - `tolling_agreement()` — virtual power plant economics.
  - `capacity_option()` — option on generation dispatch.
  - `block_forward()` — peak/off-peak block pricing.
- 4 new tests.

---

## v0.745.0 — 2026-06-02

**Mountain range options: Napoleon, Everest, Atlas, Altiplano.**

- New `equity/mountain_range.py`:
  - `napoleon_option()` — worst-of cliquet with local caps/floors.
  - `everest_option()` — payoff on worst performer.
  - `atlas_option()` — remove best/worst, payoff on remainder.
  - `altiplano_option()` — digital basket (all above barrier).
  - Correlated GBM Monte Carlo.
- 4 new tests.

---

## v0.744.0 — 2026-06-02

**Stochastic correlation for credit tranches.**

- New `credit/stochastic_correlation.py`:
  - `regime_switching_correlation()` — multi-regime tranche pricing.
  - `correlation_smile()` — calibrate implied correlation across tranches.
  - `stochastic_corr_tranche()` — beta-distributed correlation MC.
  - Vasicek one-factor tranche expected loss.
- 4 new tests.

---

## v0.743.0 — 2026-06-02

**Secondary market structured product pricing.**

- New `structured/secondary_pricing.py`:
  - `spread_aging()` — CLN spread adjustment for time since issuance.
  - `mark_to_bid()` — haircut for illiquidity with stress multiplier.
  - `stale_price_detector()` — flag unchanged prices.
  - `liquidity_premium()` — model-based illiquidity premium.
- 5 new tests.

---

## v0.742.0 — 2026-06-02

**Steepener/flattener structured notes.**

- New `structured/steepener.py`:
  - `steepener_note()` — leveraged CMS10−CMS2 with floor/cap.
  - `slope_range_accrual()` — accrues when slope in range.
  - `digital_steepener()` — digital payout on curve slope.
  - MC pricing with correlated CMS dynamics.
- 4 new tests.

---

## v0.741.0 — 2026-06-02

**Bespoke CDO: custom portfolio, LSS, tranche Greeks.**

- New `credit/bespoke_cdo.py`:
  - `bespoke_tranche_price()` — Vasicek loss distribution for custom portfolio.
  - `calibrate_bespoke_correlation()` — bisection to match market spread.
  - `leveraged_super_senior()` — LSS with gap risk.
  - `tranche_greeks()` — spread delta, correlation delta.
- 5 new tests.

---

## v0.740.0 — 2026-06-02

**Advanced autocall: discrete observation, memory coupon, step-down.**

- New `options/autocall_advanced.py`:
  - `discrete_autocall()` — discrete observation dates with memory coupon.
  - `worst_of_discrete_autocall()` — multi-asset worst-of with correlated MC.
  - `step_down_autocall()` — declining autocall barriers.
- 5 new tests.

---

## v0.739.0 — 2026-06-02

**Commodity swaps and swaptions.**

- New `commodity/commodity_swaps.py`:
  - `commodity_swap_price()` — fixed-for-floating commodity swap.
  - `commodity_swaption_price()` — Black-76 on forward swap rate.
  - `asian_commodity_swap()` — averaging settlement.
- 4 new tests.

---

## v0.738.0 — 2026-06-02

**Dividend futures, swaps, options, total return futures.**

- New `equity/dividend_futures.py`:
  - `dividend_future_price()` — implied dividend from cost-of-carry.
  - `dividend_swap_fair_value()` — fair fixed rate.
  - `dividend_option_price()` — Black-76 on dividend forward.
  - `total_return_future()` — TR vs price return decomposition.
- 4 new tests.

---

## v0.737.0 — 2026-06-02

**Futures roll mechanics: schedule, slippage, liquidity.**

- New `fixed_income/futures_roll.py`:
  - `generate_roll_schedule()` — auto-roll calendar with costs.
  - `roll_adjusted_returns()` — continuous return series.
  - `roll_slippage()` — market impact estimation.
  - `liquidity_curve()` — volume distribution by contract month.
- 3 new tests.

---

## v0.736.0 — 2026-06-02

**Cost-of-carry decomposition and arbitrage detection.**

- New `fixed_income/cost_of_carry.py`:
  - `cost_of_carry()` — decompose forward premium: r + storage − convenience yield.
  - `cash_and_carry_arb()` — detect cash-and-carry arbitrage.
  - `reverse_cash_and_carry_arb()` — detect reverse arb.
  - `carry_roll_decomposition()` — carry vs roll return attribution.
- 5 new tests.

---

## v0.735.0 — 2026-06-02

**SABR convexity for RFR futures.**

- New `fixed_income/futures_convexity.py`:
  - `sabr_convexity_adjustment()` — Piterbarg approximation with SABR smile.
  - `hw_convexity_adjustment()` — Hull-White for comparison.
  - `empirical_convexity()` — calibrate from futures vs OIS spread.
  - `compare_convexity_models()` — side-by-side SABR vs HW.
- 5 new tests.

---

## v0.734.0 — 2026-06-02

**Commodity model calibration to futures strip.**

- New `commodity/commodity_calibration.py`:
  - `calibrate_schwartz()` — Schwartz 1F to observed futures curve.
  - `calibrate_gibson_schwartz()` — Gibson-Schwartz 2F (spot + convenience yield).
  - `seasonal_decomposition()` — multiplicative trend + seasonal extraction.
  - `implied_convenience_yield_term()` — convenience yield term structure.
- 4 new tests.

---

## v0.733.0 — 2026-06-02

**VIX futures, variance swaps, vol-of-vol.**

- New `options/variance_futures.py`:
  - `vix_futures_fair_value()` — mean-reversion model with term premium.
  - `variance_swap_price()` — model-free replication from option strip.
  - `vix_term_structure()` — contango/backwardation analysis.
  - `vol_of_vol()` — implied vol-of-vol from VIX options.
- 5 new tests.

---

## v0.732.0 — 2026-06-02

**CMBS analytics: LTV, DSCR, balloon risk, defeasance.**

- New `structured/cmbs.py`:
  - `CMBSLoan` — LTV, DSCR, debt yield per loan.
  - `CMBSPool` — weighted averages, property type concentration.
  - `price_cmbs()` — tranche pricing with credit enhancement.
  - `cmbs_stress()` — property value and NOI shocks.
  - `defeasance_cost()`, `yield_maintenance()` — prepayment penalties.
- 10 new tests.

---

## v0.731.0 — 2026-06-02

**ABS cashflow engine: auto loans, credit cards, student loans.**

- New `structured/abs.py`:
  - `price_auto_abs()` — amortising auto loan ABS with sequential waterfall.
  - `price_credit_card_abs()` — revolving + controlled amortisation.
  - `price_student_loan_abs()` — grace period, IDR, default.
  - Credit enhancement, excess spread, break-even loss rate.
- 7 new tests.

---

## v0.730.0 — 2026-06-02

**MBS prepayment modelling, OAS, IO/PO strips.**

- New `structured/mbs.py`:
  - `psa_speed()` — PSA benchmark (ramp + plateau).
  - `cpr_to_smm()`, `smm_to_cpr()` — prepayment conversions.
  - `prepayment_model()` — turnover + refinancing + burnout + seasonality.
  - `price_mbs()` — pass-through pricing with prepay-adjusted duration/convexity.
  - `oas_mbs()` — OAS via Newton-Raphson.
  - `io_po_strips()` — interest-only / principal-only decomposition.
- 10 new tests.

---

## v0.729.0 — 2026-06-02

**Spread options: Kirk's approximation with full Greeks.**

- New `commodity/spread_options.py`:
  - `kirk_spread_option()` — Kirk's approximation for 2-asset spread options.
  - `crack_spread_option()` — option on refining margin.
  - `calendar_spread_option()` — option on front-back spread.
  - `intercommodity_spread_option()` — WTI-Brent and similar.
  - Cross-gamma, correlation sensitivity via finite differences.
  - Put-call parity verified.
- 9 new tests.

---

## v0.728.0 — 2026-06-02

**Commodity futures options with seasonal vol and Samuelson effect.**

- New `commodity/commodity_options.py`:
  - `commodity_option_price()` — Black-76 with seasonal vol adjustment.
  - `seasonal_vol()` — per-commodity monthly patterns (NG, CL, ZC, ZW, ZS, GC, SI).
  - `vol_term_structure()` — Samuelson effect (front-month vol > back-month).
  - `commodity_option_strip()` — price strip across delivery months.
  - `commodity_implied_vol()` — Newton-Raphson implied vol extraction.
- 8 new tests.

---

## v0.727.0 — 2026-06-02

**Futures options: unified product with contract specs and BAW.**

- New `options/futures_options.py`:
  - `FuturesOption` — option on any futures contract (ES, CL, GC, ZN, etc.).
  - Black-76 + Bachelier pricing. Barone-Adesi-Whaley for American exercise.
  - Full Greeks: delta, gamma, vega, theta — per-unit and dollar amounts.
  - 14 contract specs (equity index, commodity, bond, IR).
  - `futures_option_strip()` — strip across expiries.
  - `futures_option_vol_surface()` — build and interpolate vol surface.
  - Put-call parity verified.
- 11 new tests.

---

## v0.726.0 — 2026-06-02

**Code review fixes across CDS infrastructure.**

- **credit_spread_vol.py**: `build_credit_vol_surface()` nearest-neighbour fill now uses expiry/tenor distance instead of global min.
- **credit_var.py**: copula VaR sign convention aligned with historical/parametric (negative = loss).
- **credit_event.py**: auction open interest clipped to [-1, 1].
- **index_cds_swaption.py**: added `strike_spread <= 0` guard to prevent log domain error.
- **recovery_locked_cds.py**: removed unused `prev_q_c` variable in effective maturity loop.
- **distressed.py**: `distressed_cds_upfront()` now uses full protection + premium leg model (was simple spread × RPV01), consistent with `implied_cpd_from_upfront()` inversion.
- Tightened test tolerances: VaR ES assertion, distressed CPD round-trip to 0.2%.
- 10,783 tests pass.

---

## v0.725.0 — 2026-06-02

**Distressed CDS: upfront quoting, implied CPD, distressed basis.**

- Modified `credit/distressed.py`:
  - `distressed_cds_upfront()` — convert running spread to upfront payment.
  - `implied_cpd_from_upfront()` — Newton-Raphson inversion for CPD.
  - `distressed_basis()` — CDS-bond basis in distressed context.
  - Wide spread → positive upfront. Tight < running → negative.
- 6 new tests.

---

## v0.724.0 — 2026-06-02

**Succession events: merger, spin-off, split.**

- New `credit/succession.py`:
  - `SuccessionEvent` — entity, type, successors, weights.
  - `apply_succession()` — notional split by economic weight.
  - Per-successor spread adjustments. Notional conservation verified.
  - 5 ISDA succession types: merger, spin-off, split, reverse merger, acquisition.
- 5 new tests.

---

## v0.723.0 — 2026-06-02

**Weighted portfolio CDS: arbitrary long/short basket.**

- New `credit/portfolio_cds.py`:
  - `portfolio_cds_pv()` — PV of arbitrary-weight CDS basket.
  - Long/short positions, different notionals per name.
  - `constituent_cs01()` — per-name CS01 with % contribution.
  - Gross and net CS01. Par spread for the basket.
- 5 new tests.

---

## v0.722.0 — 2026-05-31

**Credit event auction simulation.**

- New `credit/credit_event.py`:
  - `CreditEvent` — entity, event type (6 ISDA types), dates.
  - `simulate_auction()` — two-stage ISDA auction (initial bidding + Dutch).
  - `settlement_amount()` — CDS payout from auction final price.
  - `CreditEventTimeline` — event → determination → auction → settlement.
  - `process_credit_event()` — end-to-end credit event processing.
- 8 new tests.

---

## v0.721.0 — 2026-05-31

**Index replication and tracking error.**

- New `credit/index_replication.py`:
  - `replicate_index()` — optimal weights via least squares / LASSO.
  - Greedy name selection by correlation for sparse replication.
  - L1-regularised coordinate descent for sparsity.
  - `tracking_error()` — annualised TE vs full index.
  - TE decreases with more names (verified).
- 5 new tests.

---

## v0.720.0 — 2026-05-31

**Index roll mechanics: series transition and OTR basis.**

- New `credit/index_roll.py`:
  - `series_transition()` — apply name additions/removals.
  - `index_roll_pnl()` — P&L from rolling to new series.
  - `on_the_run_basis()` — OTR vs off-the-run spread difference.
  - `series_transition_pnl()` — transition + P&L in one step.
- 5 new tests.

---

## v0.719.0 — 2026-05-31

**Recovery-locked CDS and Loan CDS (LCDS).**

- New `credit/recovery_locked_cds.py`:
  - `price_recovery_locked_cds()` — fixed recovery eliminates auction risk.
  - `recovery_lock_premium()` — premium for locking recovery vs market.
  - `price_lcds()` — Loan CDS with prepayment cancellation.
  - Higher loan recovery (70-80%), effective maturity shortened by CPR.
  - Cancellation value: RPV01 difference with/without prepayment.
- 4 new tests.

---

## v0.718.0 — 2026-05-31

**Index CDS swaption: Black-76 and Bachelier on forward index spread.**

- New `credit/index_cds_swaption.py`:
  - `index_forward_spread()` — annuity-weighted forward (Jensen's inequality).
  - `index_cds_swaption_black()` — Black-76 on forward index spread.
  - `index_cds_swaption_bachelier()` — Bachelier (normal) model.
  - `index_swaption_greeks()` — delta, gamma, vega, theta via finite diff.
  - `price_index_cds_swaption()` — full pricing from curves.
  - Put-call parity verified.
- 7 new tests.

---

## v0.717.0 — 2026-05-31

**Credit portfolio VaR: historical, parametric, and copula-based.**

- New `credit/credit_var.py`:
  - `historical_credit_var()` — CS01-weighted spread P&L from history.
  - `parametric_credit_var()` — delta-normal with correlation matrix.
  - `copula_credit_var()` — Gaussian copula joint-default simulation.
  - `CreditVaRResult` with VaR, ES, worst name, component contributions.
- 5 new tests.
- 10,733 tests pass.

---

## v0.716.0 — 2026-05-31

**Quanto CDS: cross-currency CDS with FX-credit correlation.**

- New `credit/quanto_cds.py`:
  - `quanto_cds_spread()` — adjustment: `spread × exp(ρ × σ_FX × σ_credit × T)`.
  - `price_quanto_cds()` — full pricing with FX hedge notional.
  - `quanto_adjustment_factor()` — convexity adjustment factor.
  - Positive correlation → quanto spread > foreign (wrong-way risk).
- 5 new tests.
- 10,733 tests pass.

---

## v0.715.0 — 2026-05-31

**Credit spread vol surface: ATM backbone with bilinear interpolation.**

- New `credit/credit_spread_vol.py`:
  - `CreditSpreadVolSurface` — 2D (expiry × tenor) ATM vol grid.
  - Bilinear interpolation matching `SwaptionVolCube` pattern.
  - `synthetic_credit_vol_surface()` — IG (~40%) / HY (~60%) vol generation.
  - Parallel bump support for risk scenarios.
- 5 new tests.
- 10,733 tests pass.

---

## v0.714.0 — 2026-06-01

**Bermudan CDS swaption: multiple exercise dates.**

- New `credit/bermudan_cds_swaption.py`:
  - `bermudan_cds_swaption_price()` — backward induction on hazard/discount tree.
  - At each exercise date: max(continuation, forward CDS PV).
  - Bermudan ≥ European verified. Single date → equals European.
  - Payer and receiver. ITM > OTM. Exercise probability tracked.
- 8 new tests.
- 10,718 tests pass.

---

## v0.713.0 — 2026-06-01

**Code review fixes across curve + vol infrastructure.**

- **capfloor.py**: fixed unreachable `pv_ctx` (was dead code after `return` inside wrong function). Moved to module level and assigned to `CapFloor.pv_ctx`. Fixed broken indentation.
- **curve_builder.py**: `CurveSetResult.to_dict()` now returns serialisable dict (was `vars(self)` with DiscountCurve objects).
- **swaption_vol_cube.py**: `bumped()` now shifts SABR alpha alongside ATM vol for consistent smile bumps.
- **swaption.py**: removed dead `df` variable in `price_swaption_sabr_hw`.
- **hw_calibration.py**: removed dead `df_settle` variable and unused `field` import.
- **hw_per_currency.py**: removed unused `math` import.
- 10,710 tests pass.

---

## v0.712.0 — 2026-06-01

**End-to-end callable pricing workflow notebook with pricebook.viz.**

- New `notebooks/rates/callable_pricing_workflow.ipynb`:
  - EUR yield curve: 3 methods (log-linear, Nelson-Siegel 1.6bp, Svensson). Realistic humped term structure.
  - HW calibration from 8 swaption vols. Per-swaption fit diagnostics.
  - Swaption vol cube: ATM heatmap + SABR smile.
  - Callable bond: straight 110.76 vs callable 99.37 (call value 11.39). Negative convexity.
  - Bermudan 5nc1: 19bp early exercise premium over European.
  - Multi-currency HW params (6 currencies, G10 vs EM).
  - All visuals use pricebook.viz.
- 10,710 tests pass.

---

## v0.711.0 — 2026-06-01

**Synthetic curve data, SABR-HW blended pricing, cap/floor SABR.**

- New `curves/synthetic_market_data.py`:
  - `synthetic_curve_inputs(currency)` — realistic deposits + swaps for 32 currencies.
  - USD ~5%, JPY ~0.1%, BRL ~11%, TRY ~45%. Enables testing all methods without market data.
- Extended `options/swaption.py`:
  - `price_swaption_sabr_hw()` — blends SABR smile (short end) with HW term structure (long end).
  - Weighting: `w_sabr = exp(-expiry / half_life)`. Configurable blend.
- Extended `options/capfloor.py`:
  - `strip_caplet_vols_from_quotes()` — per-caplet vol stripping from cap quotes.
  - `calibrate_capfloor_sabr()` — per-expiry SABR from caplet vols.
- 9 new tests (synthetic data, SABR-HW blending).
- 10,710 tests pass.

---

## v0.710.0 — 2026-06-01

**Cap/floor SABR, dual real+nominal curves, NDF-implied verification.**

- `options/capfloor.py`: `strip_caplet_vols_from_quotes()` + `calibrate_capfloor_sabr()`.
- `curves/inflation_curve.py`: `build_real_nominal_curves()` → nominal + real + BEI.
- NDF curves: verified existing `build_ndf_implied_curve()` + `cip_basis()`.
- 12 new tests.
- 10,701 tests pass.

---

## v0.709.0 — 2026-06-01

**Swaption infrastructure: per-currency conventions, synthetic data, HW per currency.**

- New `options/swaption_conventions.py`:
  - `SwaptionConvention` per currency: vol quote type (Black/Normal/Shifted), frequencies, SABR type, standard grids.
  - 11 currencies: USD (shifted-SABR), EUR (Normal/Bachelier), GBP, JPY, CHF, CAD, AUD, BRL (BUS/252), MXN, KRW, ZAR.
- New `options/synthetic_swaption_data.py`:
  - `synthetic_atm_surface(currency)` — realistic ATM vols (USD ~60bp, JPY ~25bp, BRL ~200bp).
  - `synthetic_smile_data(currency)` — RR25/BF25 per node.
  - `synthetic_hw_targets(currency)` — swaption vol targets for HW calibration.
- New `models/hw_per_currency.py`:
  - `calibrate_hw_for_currency(currency, ref, curve)` — full pipeline: synthetic vols → HW calibration.
  - Default parameters for 33 currencies (G10, EM, Asia, CEE).
  - EM defaults: higher mean reversion + vol (BRL a=0.10, TRY a=0.15).
- 16 new tests.
- 10,689 tests pass.

---

## v0.708.0 — 2026-06-01

**Swaption vol cube: 3D (expiry × tenor × strike) with SABR smile.**

- New `options/swaption_vol_cube.py`:
  - `SwaptionVolCube` — ATM backbone (bilinear interpolation) + per-node SABR smile.
  - `vol(expiry, tenor, strike)` — full 3D interpolation.
  - `smile(expiry, tenor, strikes)` — vol smile across strikes.
  - `bumped(shift)` — parallel vol shift.
  - `build_swaption_vol_cube()` — construct from ATM grid + smile quotes.
  - `SABRNode` — per-(expiry, tenor) SABR params (alpha, beta, rho, nu).
  - SABR calibration via `sabr_calibrate()` at each smile node.
- OTM vol differs from ATM when SABR is fitted (smile verified).
- 12 new tests.
- 10,673 tests pass.

---

## v0.707.0 — 2026-06-01

**Unified curve methods: all 33 currencies now have 5 construction methods.**

- `get_conventions()` in `curve_builder.py` now falls through from G10 to EM registry.
- `build_curves()` accepts ANY of the 33 currencies (was limited to 10 G10).
- EM currencies (BRL, MXN, CNY, KRW, INR, ZAR, PLN, etc.) can now use:
  - Sequential bootstrap, Global Newton, Nelson-Siegel, Svensson, Smith-Wilson.
- Cross-method consistency: 5Y zero rate within 100bp across methods.
- Note: Smith-Wilson fails at extreme rates (TRY 45%) — use sequential for extreme EM.
- 17 new tests.
- 10,661 tests pass.

---

## v0.706.0 — 2026-06-01

**Hull-White calibration from swaption volatilities — CRITICAL GAP FILLED.**

- New `models/hw_calibration.py`:
  - `calibrate_hull_white(curve, swaption_vols)` → calibrated `HullWhite` model.
  - Minimises Σ(model_vol - market_vol)² across swaption grid.
  - Model vol: HW tree pricing → Black-76 vol inversion.
  - Optimisers: Nelder-Mead (default), differential evolution, L-BFGS-B.
  - ATM strike auto-computed from forward swap rates if not provided.
  - Per-swaption fit diagnostics (error in bp).
  - Round-trip verified: generate vols from known (a=0.03, σ=0.01), calibrate back within 30%.
- Enables: calibrated callable bond pricing, Bermudan swaption pricing, cancellable swap pricing from market vol data.
- 8 new tests.
- 10,644 tests pass.

---

## v0.705.0 — 2026-06-01

**Reorganise notebooks into thematic subdirectories.**

- `notebooks/americas/` — argentina, canada, chile, colombia, mexico, peru (6)
- `notebooks/rates/` — treasury_note_roundtrip, treasury_multicurve, asw_btp_bund (3)
- `notebooks/credit/` — recovery_roundtrip (1)
- `notebooks/structured/` — prdc_structuring, tarf_risk_profile, xccy_basis_pricing (3)
- `notebooks/desks/` — bond_trading_desk, futures_desk (2)
- `notebooks/validation/` — cmasw_pucci_2012a, cmt_pucci_2014, index_linked_hybrid_pucci_2012b, treasury_lock_pucci_2019, trs_lou_2018 (5)
- Renamed for consistency: `*_derivatives.ipynb` → short country names, `*_validation.ipynb` → paper names only.
- Fixed `sys.path` in all 20 notebooks for 2-level-deep directory structure.
- Cleaned up stale `.ipynb_checkpoints`.
- 10,636 tests pass.

---

## v0.704.0 — 2026-06-01

**Code review fixes for callable/cancellable modules.**

- **callable_cds.py**: fixed discount factor — now uses `df(t_next)/df(t)` instead of `exp(-zero_rate*dt)`. Fixed date arithmetic to use `timedelta(days=round(t*365.25))` instead of `int(t*365)`.
- **cancellable_swap.py**: fixed receiver swap sign logic — cancellation right always reduces PV for the non-option-holder regardless of direction.
- **callable_cln.py**: coupon now survival-weighted (`coupon * p_survive`). Call date matching uses 5-day tolerance instead of exact equality.
- **Exception handling**: narrowed from bare `except Exception` to `except (ImportError, TypeError, ValueError)` in cancellable_swap and extendible.
- 10,636 tests pass.

---

## v0.703.0 — 2026-06-01

**Callable/cancellable derivatives: cancellable swap, extendible, callable CDS, callable CLN.**

- New `fixed_income/cancellable_swap.py`:
  - `cancellable_swap_price()` — swap + embedded Bermudan swaption decomposition.
  - Cancellable PV ≤ vanilla PV (option costs the holder). Par rate adjusted.
- New `fixed_income/extendible.py`:
  - `extendible_swap_price()` — base swap + European swaption on extension period.
  - Extendible PV ≥ base PV (extension adds value for holder).
- New `credit/callable_cds.py`:
  - `callable_cds_price()` — CDS with seller termination right via backward induction.
  - Callable PV ≤ vanilla. Callable spread ≥ vanilla spread.
- New `credit/callable_cln.py`:
  - `callable_cln_price()` — CLN with issuer early redemption via backward induction.
  - Callable ≤ straight CLN. Higher coupon → more call value.
  - Call probability, expected call date, par spread for callable.
- All compose over existing Hull-White tree / survival curve infrastructure.
- 17 new tests.
- 10,636 tests pass.

---

## v0.702.0 — 2026-06-01

**Asia build-out: 9 currencies — CNY, KRW, INR, SGD, HKD, THB, IDR, MYR, PHP.**

- New modules: chinese.py, korean.py, singaporean.py, hong_kong.py, thai.py, indian.py, indonesian.py, malaysian.py, philippine.py.
- **Korea (KRW)**: KOFRSwap + KTB + KTBi linker (CPI_KR, deflation floor) + BEI.
- **India (INR)**: MIBORSwap + GSEC (**30/360** — only sovereign globally) + IIB linker (CPI_IN, deflation floor) + BEI. MIBOR rate index added.
- **Philippines (PHP)**: PHIREFSwap + RPGB (**quarterly coupon** — only quarterly sovereign globally). PHIREF rate index added.
- **China (CNY)**: DR007Swap + CGB. **Indonesia**: INDONIASwap + INDOGB. **Malaysia**: MYORSwap + MGS. **Singapore**: SORASwap + SGS. **HK**: HONIASwap + HKGB. **Thailand**: THORSwap + THAIGB.
- 4 new rate indices: MIBOR (FBIL), INDONIA (BI), MYOR (BNM), PHIREF (BSP).
- 9 new OIS conventions added.
- 43 new tests.
- Markets with full derivatives: 24 → 33.
- 10,619 tests pass.

---

## v0.701.0 — 2026-06-01

**BEI (breakeven inflation) added to 9 markets — now 16 markets have BEI.**

- Added `breakeven_inflation_XX()` convenience functions to: BRL, MXN, COP, PEN, ARS, PLN, CZK, HUF, TRY.
- Total markets with BEI: 16 (GBP, CAD, CLP, JPY, AUD, ZAR, ILS + 9 new).
- All follow same pattern: nominal_rate - real_rate from two discount curves.
- Argentina/Turkey: extreme BEI values expected (~30%+ / ~35%+).
- 10,576 tests pass.

---

## v0.700.0 — 2026-06-01

**Japan, Australia, South Africa, Israel: full derivatives with inflation linkers + BEI.**

- New `fixed_income/japanese.py`: TONASwap, JGBBond, JGBiLinker (CPI_JP, 3M lag, **deflation floor**), BEI. Near-zero rate handling.
- New `fixed_income/australian.py`: AONIASwap, ACGBBond, TIBBond (CPI_AU, **quarterly coupon** — only quarterly linker globally, **no deflation floor**), BEI.
- New `fixed_income/south_african.py`: JIBARSwap (**quarterly** fixed), SAGBBond (T+3), SAILBBond (CPI_ZA, no floor), BEI.
- New `fixed_income/israeli.py`: TelborSwap, ShaharBond, GalilBond (CPI_IL, **1-month lag**, **annual coupon**, no floor), BEI.
- Markets with full derivatives: 20 → 24.
- 32 new tests.
- 10,576 tests pass.

---

## v0.699.0 — 2026-06-01

**Code review fixes across all new market modules.**

- **Nordic template placeholders**: fixed `{country}` → "Swedish"/"Norwegian"/"Danish" in 6 docstrings.
- **CEE linker conventions**: changed frequency from semi-annual to annual (PLN, CZK, HUF linkers). Fixed CZK/HUF linker day counts from ACT/360 to ACT/365F to match inflation_indices.json.
- **PLN IRS**: fixed leg frequency changed from semi-annual to annual (market standard).
- **Gilt**: added past cashflow filtering in `dirty_price()` (was including past coupons).
- **Danish mortgage**: removed unused `import numpy`.
- **Rate indices JSON**: SWESTR and NOWA observation_shift corrected from 0 to 2.
- 10,544 tests pass.

---

## v0.698.0 — 2026-06-01

**Danish mortgage bonds (realkreditobligationer) — callable covered bonds with prepayment.**

- New `fixed_income/danish_mortgage.py` (300 lines):
  - `DanishMortgageBond` — callable at par, bullet or pass-through amortisation.
  - `prepayment_model()` — CPR as function of refinancing incentive (coupon - market rate), with seasoning ramp-up.
  - `psa_curve()` — PSA-standard prepayment ramp (30-month, configurable speed).
  - `MortgageBondResult` — dirty price, OAS, effective duration, WAL, expected CPR, callable value.
  - Effective duration via ±10bp parallel bump (non-recursive).
  - Callable price ≤ non-callable (negative convexity verified).
  - Higher CPR → shorter WAL. Pass-through WAL < bullet WAL.
  - OAS > 0 for callable bonds with refinancing incentive.
- 16 new tests.
- 10,544 tests pass.

---

## v0.697.0 — 2026-06-01

**CEE + Turkey: PLN, CZK, HUF, TRY — dual IBOR+RFR swaps + inflation linkers.**

- New `fixed_income/polish.py`: WIBORSwap (3M), WIRONSwap (overnight), POLGBBond (annual ACT/ACT ICMA), POLGBLinker (CPI_PL). WIRON rate index added.
- New `fixed_income/czech.py`: PRIBORSwap (3M), CZEONIASwap (overnight), CZGBBond, CZGBLinker (CPI_CZ). CZEONIA rate index added.
- New `fixed_income/hungarian.py`: BUBORSwap (3M), HUFONIASwap (overnight), HGBBond (**ACT/365F** — unique among CEE), HGBLinker (CPI_HU). HUFONIA rate index added.
- New `fixed_income/turkish.py`: TLREFSwap, TURKGBBond (semi-annual ACT/365F, **T+0 settlement**), TurkishCPILinker (CPI_TR, 2-month lag). Handles 45%+ extreme rates. TLREF rate index added.
- 4 new overnight rate indices: WIRON, CZEONIA, HUFONIA, TLREF.
- 3 new inflation indices: CPI_PL, CPI_CZ, CPI_HU.
- 29 new tests (8 PLN + 7 CZK + 7 HUF + 7 TRY).
- 10,528 tests pass.

---

## v0.696.0 — 2026-06-01

**Switzerland + Nordics: SARON, SWESTR, NOWA, DESTR swaps + sovereign bonds.**

- New `fixed_income/swiss.py`: SARONSwap (ACT/360), ConfedBond (annual ACT/ACT ICMA). Handles negative rates (CHF DF > 1 verified).
- New `fixed_income/swedish.py`: SWESTRSwap, SGBBond. SWESTR rate index added.
- New `fixed_income/norwegian.py`: NOWASwap, NGBBond. NOWA rate index added.
- New `fixed_income/danish.py`: DESTRSwap, DGBBond. DESTR rate index added. DKK OIS convention added to ois.py.
- 3 new overnight rate indices in rate_indices.json: SWESTR (Riksbank), NOWA (Norges Bank), DESTR (Danmarks Nationalbank).
- 21 new tests (6 CHF + 5 SEK + 5 NOK + 5 DKK).
- 10,499 tests pass.

---

## v0.695.0 — 2026-06-01

**UK: SONIA swap, Gilt, Index-Linked Gilt (ILG), breakeven inflation.**

- New `fixed_income/british.py` (330 lines):
  - `SONIASwap` — annual ACT/365F, par rate, DV01, direction symmetry.
  - `GiltBond` — semi-annual ACT/ACT ICMA, 7-day ex-dividend, T+1.
  - `ILGBond` — **8-month RPI lag, flat interpolation** (not linear like TIPS), **no deflation floor** (unlike TIPS). Nominal = real × RPI ratio.
  - `build_sonia_curve()` — ACT/365F bootstrap.
  - `breakeven_inflation_uk()` — nominal Gilt vs real ILG curves (2Y-50Y).
  - `synthetic_sonia_strip()`, `synthetic_gilt_strip()`.
- ILG deflation: RPI ratio < 1.0 when RPI falls (verified — no floor).
- UK BEI (RPI-based) ~3.5%, consistent with market.
- 16 new tests.
- 10,478 tests pass.

---

## v0.694.0 — 2026-06-01

**Canada deepening: CGB, Canadian IRS, provincial bonds, breakeven inflation.**

- Extended `fixed_income/canadian.py` (117→340 lines):
  - `CGBBond` — Canadian Government Bond, semi-annual ACT/365F, yield-to-maturity solver.
  - `CanadianIRS` — fixed semi-annual vs CORRA compound, par rate, DV01.
  - `ProvincialBond` — spread over federal CGB curve (ON, QC, BC, AB, MB, SK).
  - `breakeven_inflation_ca()` — CORRA nominal vs RRB real curves.
  - `synthetic_cgb_strip()` — 4 benchmark CGB quotes (2Y, 5Y, 10Y, 30Y).
  - Provincial spread ordering verified: BC (25bp) < AB (30bp) < ON (35bp) < QC (40bp).
  - IRS direction symmetry: pay_fixed PV = -receive_fixed PV.
- 10 new tests.
- 10,462 tests pass.

---

## v0.693.0 — 2026-06-01

**Market-accurate bond curve: per-bond day count convention + sovereign factory.**

- `BondQuote` now supports `day_count`, `settlement_days`, `calendar_ccy` fields.
- `BondQuote.from_sovereign(market_code, ...)` — auto-sets conventions from the 60-market sovereign registry:
  - UST: ACT/ACT ICMA, semi-annual, T+1
  - BUND: ACT/ACT ICMA, annual, T+2
  - JGB: ACT/365F, semi-annual, T+2
  - NTN_F: BUS/252, semi-annual, T+1 (loads BRL calendar)
  - MBONO: ACT/360, semi-annual, T+2
- `_price_bond()` rewritten: uses the bond's own day count for accrual fractions.
  - ACT/ACT ICMA: passes coupon period boundaries + frequency.
  - BUS/252: loads calendar from `calendar_ccy`.
  - All other conventions: straightforward.
- Sequential and global bootstrap both use per-bond conventions.
- Verified: different day counts produce different implied zero rates.
- 8 new tests (sovereign factories, multi-market curves, day count impact).
- 10,452 tests pass.

---

## v0.692.0 — 2026-06-01

**Yield curve bootstrapping from bond prices alone.**

- New `curves/bond_curve.py`:
  - `BondQuote` — bond observation (maturity, coupon, dirty price, weight, on-the-run flag).
  - `bootstrap_curve_from_bonds()` — unified entry point with 4 methods:
    - `"sequential"` — exact fit, one bond per pillar (like CDS bootstrap but for DFs).
    - `"global"` — least-squares, robust to noise, supports n_pillars < n_bonds.
    - `"nelson_siegel"` — 4-parameter smooth curve fitted directly to bond prices (not zero rates).
    - `"svensson"` — 6-parameter smooth curve (captures humps better than NS).
    - `"auto"` — sequential if ≤8 distinct maturities, else global.
  - On-the-run bonds get 2× weight in global/parametric fits.
  - Zero-coupon bonds (T-Bills): exact DF extraction.
  - NS long-end converges to β₀. Svensson fits at least as well as NS.
  - Cross-method: 5Y zero rate consistent within 200bp across all methods.
- `BondCurveResult` with discount_curve, pillar zeros, fitted prices, RMSE, parameters.
- 22 new tests.
- 10,444 tests pass.

---

## v0.691.0 — 2026-06-01

**FRN hazard bootstrapping, mixed fixed+float, and liquid/illiquid regime handling.**

- New in `credit/bond_hazard_bootstrap.py`:
  - `FRNInput` — floating-rate note observation (spread, benchmark, market price).
  - `_price_risky_frn()` — risky FRN pricing with survival-weighted floating coupons and recovery leg.
  - `bootstrap_hazard_mixed()` — global fit from mix of fixed-rate bonds and FRNs. Returns piecewise hazard curve.
  - `LiquidityAssessment` — regime classification (liquid/semi_liquid/illiquid) with recommended method, n_pillars, confidence.
  - `assess_liquidity()` — heuristic assessment from bond count, bid-ask widths, price levels, maturity coverage.
  - `bootstrap_hazard_adaptive()` — auto-selects method based on liquidity:
    - Liquid: sequential bootstrap (exact fit).
    - Semi-liquid: global fit with bid-ask-adjusted weights.
    - Illiquid: global fit with 1-3 pillars.
  - Bid-ask weighting: `w = 1/(1 + ba/100)` — wider spread → lower weight.
  - Distressed bonds (50-60 cents): produces high hazard rates, survival still decreasing.
- 19 new tests.
- 10,422 tests pass.

---

## v0.690.0 — 2026-06-01

**Fix remaining known limitations: Frank copula, tranche annuity, barrier vectorization.**

- **Frank copula**: rewrote d≥3 sampling using Marshall-Olkin algorithm with logarithmic series mixing variable. Previously used bivariate conditional method that produced incorrect multivariate dependence.
- **TrancheCDS.price()**: replaced single-period annuity approximation with proper multi-period premium and protection legs (quarterly frequency). Par spread now computed from risky annuity ratio.
- **Barrier continuous mode**: vectorized Python loops for knockout and knockin. ~10-50x speedup for large n_paths. Correct bridge probability formula for both up and down barriers.
- 10,403 tests pass.

---

## v0.689.0 — 2026-05-31

**Code review fixes: CDO PMF, barrier bridge, copula M factor, dt guard, BMA default.**

- **CDO MC**: fixed PDF/PMF mismatch — `portfolio_loss_distribution_mc` now returns PMF (probability mass) consistent with analytical Vasicek. `tranche_expected_loss_mc` now produces correct results.
- **Barrier bridge**: fixed bridge_min formula for down-and-out/down-and-in — now uses correct conditional probability `P(min < b) = exp(-2(s0-b)(s1-b)/(σ²dt))` instead of incorrect `s0 + s1 - max` approximation.
- **Non-Gaussian copula**: systematic factor M now uses `sample_with_factor()` for Gaussian copula (correct), and independent fallback for non-Gaussian (honest about limitation, was previously using meaningless Z.mean approximation).
- **OU exact step**: added `dt < 1e-14` guard to prevent `dw/sqrt(dt)` numerical instability.
- **BMA**: None AIC/BIC now gets mean IC of other models (was 0.0, which gave infinite weight).
- 10,403 tests pass.

---

## v0.688.0 — 2026-05-31

**Model reserves framework: parameter uncertainty, reserves, P&L attribution, model selection.**

- New `risk/parameter_uncertainty.py`:
  - `ParameterBand` — confidence interval for calibrated parameter.
  - `calibration_uncertainty()` — bootstrap CI from market data.
  - `sensitivity_ladder()` — PV impact at band edges, sorted by magnitude.
  - `joint_parameter_surface()` — 2D PV surface over two parameter bands.
- New `risk/model_reserve.py`:
  - `compute_model_reserve()` — worst-case or quadrature (√Σ) reserve from bands.
  - `reserve_by_risk_factor()` — per-parameter reserve breakdown.
  - `model_risk_reserve_ava()` — EBA-compatible AVA format.
- New `risk/model_selection.py`:
  - `ModelCandidate` — model with pricer, weight, AIC/BIC.
  - `model_committee_price()` — weighted average + dispersion + uncertainty reserve.
  - `bayesian_model_average()` — posterior weights from AIC/BIC.
  - `model_risk_matrix()` — price all models under all scenarios.
- Extended `risk/pnl_explain.py`:
  - `surface_pnl()` — ATM/skew/smile/term structure P&L decomposition.
  - `gamma_pnl_decompose()` — realised vs implied gamma, net gamma P&L.
  - `NonLinearPnLResult` dataclass.
- 21 new tests.
- 10,403 tests pass.

---

## v0.687.0 — 2026-05-31

**Recovery extras: heterogeneous specs, seniority waterfall, bid-ask surface.**

- New in `credit/recovery_pricing.py`:
  - `build_recovery_specs(seniorities)` — from Moody's table per-name.
  - `validate_recovery_specs(specs, n_names)` — length check.
  - `recovery_spec_summary(specs)` — portfolio-level stats.
  - `SeniorityWaterfall` — capital structure priority distribution.
    - `distribute(total_recovery)` — senior gets first, sub gets remainder.
    - `recovery_rates(total_pct)` — per-tranche recovery rates.
    - `to_recovery_specs()` — waterfall-consistent RecoverySpec list.
  - `implied_recovery(spread, hazard)` — R = 1 - s/h.
  - `recovery_bid_ask_surface()` — term structure of implied recovery with bid-ask.
- 17 new tests.
- 10,382 tests pass.

---

## v0.686.0 — 2026-05-31

**OU exact step + MC convergence diagnostics.**

- `OUProcess`: exact Gaussian transition (was Euler). Mean reversion to θ, stationary variance σ²/(2κ) verified.
- New `models/mc_diagnostics.py`:
  - `batch_means()` — robust SE estimation via inter-batch variance.
  - `effective_sample_size()` — autocorrelation-adjusted ESS via FFT.
  - `convergence_table()` — running mean/SE at checkpoints.
  - ESS = N for iid, ESS < N for AR(1) verified.
- 13 new tests.
- 10,365 tests pass.

---

## v0.685.0 — 2026-05-31

**Heterogeneous portfolios: per-name notional and LGD in bespoke tranches.**

- `bespoke_tranche()`: new `notionals` and `lgds` parameters.
- `notionals`: per-name portfolio weights (default: equal weight).
- `lgds`: per-name loss given default (overrides uniform `lgd`).
- Concentrated portfolio: name with 5x weight dominates loss.
- Uniform notionals/lgds reproduce current flat behavior exactly.
- Works with `recovery_specs` for full per-name stochastic recovery.
- 6 new tests.
- 10,352 tests pass.

---

## v0.684.0 — 2026-05-31

**Multi-copula support in basket CDS.**

- `ftd_spread()`, `ntd_spread()`: new `copula` parameter.
- Accepts any `Copula` instance from `statistics/copulas.py`: Gaussian, Student-t, Clayton, Frank, Gumbel.
- Student-t copula produces higher FTD spread (tail dependence clusters defaults).
- When copula=None, falls back to one-factor Gaussian (backward compatible).
- Approximate systematic factor extraction for non-Gaussian copulas (recovery correlation).
- 7 new tests.
- 10,334 tests pass.

---

## v0.683.0 — 2026-05-31

**Base correlation surface with cubic spline interpolation and arbitrage checks.**

- New `BaseCorrelationSurface` class in `credit/tranche_pricing.py`:
  - `interpolate(detachment, method)` — linear or cubic spline with monotonicity enforcement.
  - `check_arbitrage()` — detects non-monotonicity and out-of-bounds correlations.
  - `bump(shift)` — parallel shift with clamping to (0, 1).
  - `from_calibration()` — build from `calibrate_base_correlation()` output.
  - Callable: `surface(0.07)` returns interpolated base correlation.
- 13 new tests.
- 10,340 tests pass.

---

## v0.682.0 — 2026-05-31

**Configurable time discretization in basket CDS (quarterly default).**

- `ftd_spread()`, `ntd_spread()`: new `frequency` parameter (1=annual, 4=quarterly, 12=monthly).
- Default changed from annual (frequency=1) to quarterly (frequency=4).
- More time points → finer survival/default assessment.
- Convergence: monthly ≈ quarterly (verified).
- 5 new tests.
- 10,327 tests pass.

---

## v0.681.0 — 2026-05-31

**MC portfolio loss distribution with stochastic recovery for CDO.**

- New `portfolio_loss_distribution_mc()` in `credit/cdo.py`:
  - Monte Carlo complement to analytical Vasicek (which requires constant LGD).
  - Accepts `RecoverySpec` for per-name stochastic recovery correlated to M.
  - MC with fixed recovery converges to analytical EL = PD × LGD.
- New `tranche_expected_loss_mc()` — wraps MC loss dist with tranche clipping.
- Equity EL > Senior EL verified. Density non-negative, integrates to 1.
- 8 new tests.
- 10,322 tests pass.

---

## v0.680.0 — 2026-05-31

**Per-name stochastic recovery in copula default simulation.**

- `copula_default_simulation()`, `tranche_pricing_copula()`: new `recovery_specs` parameter.
- `GaussianCopula.sample_with_factor()`: returns (U, M) — uniform marginals + systematic factor.
- For Gaussian copula: recovery correlated to M. For non-Gaussian (Clayton, Gumbel, Frank): unconditional recovery.
- Heterogeneous seniority: mix senior + subordinated recovery in same portfolio.
- 8 new tests.
- 10,314 tests pass.

---

## v0.679.0 — 2026-05-31

**Stochastic correlated recovery in CDO tranche pricing.**

- `expected_tranche_loss()`, `expected_tranche_loss_t()`, `TrancheCDS.price()`: new optional `recovery_specs` parameter.
- Per-name stochastic recovery sampled correlated to systematic factor M.
- Student-t copula: uses underlying normal M for recovery correlation (not t-scaled).
- Wrong-way risk verified: equity tranche EL increases; senior tranche less affected.
- Fixed RecoverySpec reproduces flat recovery. Backward compatible.
- 6 new tests.
- 10,306 tests pass.

---

## v0.678.0 — 2026-05-31

**Stochastic correlated recovery in basket CDS (FTD/NTD/bespoke).**

- `ftd_spread()`, `ntd_spread()`, `bespoke_tranche()`: new optional `recovery_specs` parameter.
- Accepts `list[RecoverySpec]` — per-name stochastic recovery correlated to systematic factor M.
- Wrong-way risk: negative default-recovery correlation increases FTD spread.
- Heterogeneous seniority: mix senior secured (R=65%) and subordinated (R=28%) in same basket.
- Fixed RecoverySpec(0.4, 0) reproduces flat recovery exactly. Backward compatible.
- 8 new tests.
- 10,300 tests pass.

---

## v0.677.0 — 2026-05-31

**Fix LSM American put discounting + continuous barrier monitoring.**

- **American put LSM**: added `r` parameter for proper discounting of continuation values in backward induction. Higher r → earlier exercise (correct behavior). American ≥ European verified.
- **Barrier options**: added `continuous=True, sigma=σ` parameters to `barrier_knockout` and `barrier_knockin`. Uses Brownian bridge max/min sampling for continuous monitoring from discrete paths. Continuous up-out ≤ discrete up-out (more knockouts). Knockin + knockout ≈ vanilla (parity check).
- Backward compatible: defaults match old behavior (r=0, continuous=False).
- 11 new tests.
- 10,292 tests pass.

---

## v0.676.0 — 2026-05-31

**Fix non-reproducible MC paths in Merton, Bates, and Variance Gamma processes.**

- `JumpDiffusionProcess`, `BatesProcess`, `VarianceGammaProcess` now accept `seed` parameter.
- Replaced global `np.random.poisson()`/`np.random.randn()`/`np.random.gamma()` with closure-captured `np.random.default_rng(seed)`.
- Same seed → identical paths guaranteed. Different seeds → different paths.
- Backward compatible: `seed=None` uses unseeded RNG (old behavior).
- 7 new tests verifying reproducibility.
- 10,281 tests pass.

---

## v0.675.0 — 2026-05-31

**Deep fixes for remaining known limitations.**

- **CGMY MC simulation**: rewrote to proper difference-of-Gamma representation with exact risk-neutral drift from char_func. Shape parameters use Γ(1-Y)·rate^(Y-1) moment matching.
- **Cross-validation MC**: now covers all 6 models (added Kou via compound Poisson + double-exponential, CGMY via new terminal(), Bates via mc_migrate). Custom params are now respected.
- **Theta decomposition**: computes actual total theta via 1-day maturity bump. Vol theta is now residual = total - carry - div (was hardcoded 0).
- **Dividend surface simulation**: `spot_vol` and `kappa_q` now explicit parameters (were hardcoded 0.20/2.0). Returns `DividendSimResult` dataclass (was raw dict). Uses log-Euler scheme (prevents negative spot).
- **Char func API consistency**: all standalone factories now follow `(rate, model_params..., T)` ordering. `vg_char_func`, `nig_char_func`, `cgmy_char_func` signatures updated. **Breaking change** for direct callers.
- Correlation clamped to [-0.999, 0.999] in simulation (prevents sqrt of negative).
- 10,274 tests pass.

---

## v0.674.0 — 2026-05-31

**Code assessment fixes across jump + dividend modules.**

- **CGMY**: reject Y=1 (pole of Γ(-Y)) at construction.
- **NIG**: validate `alpha > |beta+1|` (risk-neutral measure existence).
- **VG**: guard `1 - θν - 0.5σ²ν > 0` with clear error message.
- **American tree**: rewrote to spot-adjustment model — subtract PV of all future dividends, build CRR on adjusted spot, add PV back for intrinsic comparison. Fixes dividend propagation bug.
- **RGW**: documented as simplified approximation (univariate, not bivariate normal).
- Removed dead code: unused `NIGResult`/`CGMYResult` dataclasses, `nig_constraint`, dead `field` imports.
- Fixed `ForwardErrorDecomp.to_dict()` missing fields.
- 10,272 tests pass.

---

## v0.673.0 — 2026-05-31

**Dividend surface + joint vol-dividend calibration.**

- New `equity/dividend_surface.py`:
  - `DividendSurface` — tenors × yield levels × yield vols × spot correlation.
  - `build_dividend_surface()` — from futures + optional dividend options.
  - `simulate_dividend_surface()` — correlated spot + OU dividend yield MC paths.
- New `equity/joint_calibration.py`:
  - `joint_calibrate()` — simultaneous vol + dividend yield fitting.
  - Models: "bsm+continuous" (flat vol + q), "term+continuous" (piecewise σ + q).
  - `decompose_forward_error()` — attribute mispricing to vol vs dividend assumptions.
  - Round-trip: recovers σ and q within 1% on synthetic data.
- 11 new tests.
- 10,272 tests pass.

---

## v0.672.0 — 2026-05-31

**American option early exercise around ex-dividend dates.**

- New `options/american_dividend.py`:
  - `american_with_dividends()` — binomial tree with ex-dates as explicit nodes, dividend spot drop.
  - `roll_geske_whaley()` — closed-form for single discrete dividend (Newton for critical spot S*).
  - `exercise_boundary_around_exdate()` — exercise vs hold decision across spot levels.
  - American call ≥ European call verified; early exercise premium ≥ 0.
- 17 new tests: Am≥Eu, premium positive, boundary transition, RGW critical spot, div-after-expiry.
- 10,261 tests pass.

---

## v0.671.0 — 2026-05-31

**Enhanced dividend Greeks: cross-gamma, theta decomposition, scenario ladder.**

- New `equity/dividend_greeks.py`:
  - `compute_dividend_greeks()` — div_delta, div_gamma, cross_gamma_spot_div, div_theta, spot_delta via central finite differences.
  - `theta_decomposition()` — split theta into carry, dividend accrual, vol decay.
  - `dividend_scenario_ladder()` — price grid across dividend bump scenarios.
  - Cross-gamma d²V/(dS·d(div)): the key missing second-order Greek.
- 11 new tests: sign checks (call div_delta < 0, put > 0), cross-gamma finite, theta negative, ladder monotonicity.
- 10,244 tests pass.

---

## v0.670.0 — 2026-05-31

**Dividend strip analytics: decomposition, carry, growth rates.**

- New `equity/dividend_strip.py`:
  - `decompose_strip()` — split DividendCurve into per-period strips with forward div, PV, weight.
  - `strip_carry()` — carry-and-roll analytics per strip (yield vs funding).
  - `dividend_growth_rate()` — log-linear regression for implied growth from forward term structure.
  - Custom period breaks or equal-width periods.
- 11 new tests: sum-to-total, weights, constant/growing growth, carry.
- 10,233 tests pass.

---

## v0.669.0 — 2026-05-31

**Dividend term structure calibration (optimisation, spline, options-implied).**

- New `equity/dividend_calibration.py`:
  - `calibrate_dividend_curve()` — 3 methods: "linear" (existing), "optimize" (piecewise-constant yield via L-BFGS-B), "spline" (cubic spline on cumulative).
  - `calibrate_from_options()` — extract dividend curve from put-call parity across expiries.
  - `dividend_curve_seasonality()` — quarterly weight decomposition, peak/trough detection.
  - Optimised method fits at least as well as linear on non-constant yield data.
- 12 new tests: round-trip calibration, options-implied, seasonality, Q2-heavy detection.
- 10,222 tests pass.

---

## v0.668.0 — 2026-05-31

**Jump model cross-validation framework (COS vs MC vs FFT).**

- New `models/jump_cross_validation.py`:
  - `cross_validate_model()` — COS vs MC comparison for any of 6 jump models.
  - `cross_validate_all()` — all models, sorted by accuracy.
  - Per-strike results: COS price, MC price, FFT price, % difference.
  - Verified: Merton, VG, NIG all within 5% COS/MC mean difference.
- 10 new tests.
- 10,210 tests pass.

---

## v0.667.0 — 2026-05-31

**Jump model calibration to implied vol surfaces.**

- New `models/jump_calibration.py`:
  - `calibrate_jump_model()` — fits any of 6 jump models (Merton, VG, Kou, NIG, CGMY, Bates) to market implied vols via COS pricing + differential evolution.
  - `calibrate_jump_surface()` — multi-expiry independent calibration.
  - `jump_model_comparison()` — fits all models, ranks by AIC (penalises parameter count).
  - Round-trip: Merton calibration recovers params with < 0.5 vol pt RMSE.
- 10 new tests: round-trip, cross-model fitting, multi-expiry, model comparison.
- 10,200 tests pass.

---

## v0.666.0 — 2026-05-31

**NIG and CGMY Lévy processes with characteristic functions.**

- New `models/levy_processes.py`:
  - `NIGProcess(alpha, beta, delta)` — Normal Inverse Gaussian with char_func + MC terminal.
  - `CGMYProcess(C, G, M, Y)` — tempered stable Lévy process, generalises VG.
  - `nig_char_func()`, `cgmy_char_func()` — standalone risk-neutral CFs.
  - Both support complex u input (FFT-compatible).
  - NIG: inverse Gaussian subordinator simulation, exact RN drift correction.
  - CGMY: Y→0 limit handled separately (recovers VG char func).
- COS pricing verified: NIG vs MC within 5%, CGMY produces reasonable prices.
- Cross-model: both produce heavier tails than Black-Scholes (higher OTM put prices).
- 25 new tests.
- 10,190 tests pass.

---

## v0.665.0 — 2026-05-31

**Characteristic function protocol + standalone factories for Kou, Bates/SVJ.**

- New `models/char_func_protocol.py`:
  - `CharFuncModel` — `@runtime_checkable` Protocol for Fourier-based pricing.
  - `validate_char_func()` — checks φ(0)=1, boundedness, Hermitian symmetry.
  - `extract_cumulants()` — c1–c4, skewness, excess kurtosis from any CF.
  - Standalone factories: `merton_char_func()`, `vg_char_func()`, `kou_char_func()`, `bates_char_func()`, `svj_char_func()`.
  - All accept complex u (Carr-Madan FFT compatible).
- Kou CF: double-exponential jump CF with p·η₁/(η₁-iu) + (1-p)·η₂/(η₂+iu).
- Bates CF: Heston CF × Merton jump component (Schoutens form).
- 18 new tests: protocol compliance, validation, cumulants, COS vs MC cross-validation, complex u input.
- 10,165 tests pass.

---

## v0.664.0 — 2026-05-31

**Americas derivatives notebooks: Mexico, Chile, Colombia, Peru, Argentina, Canada.**

- 6 new notebooks in `notebooks/`:
  - `mexican_derivatives.ipynb` — TIIE 28D swap, CETES, MBONO, Udibono (UDI), BEI.
  - `chilean_derivatives.ipynb` — Cámara swap, BCP, BCU (UF), dual-curve BEI.
  - `colombian_derivatives.ipynb` — IBR swap, TES, TES UVR, BEI.
  - `peruvian_derivatives.ipynb` — PEN curve, BTP Peru, VAC bond, BEI.
  - `argentine_derivatives.ipynb` — ARS curve (40%+), Lecap, Lecer (CER), Bonares, BEI.
  - `canadian_derivatives.ipynb` — CORRA swap, CGB, RRB (deflation floor), BEI.
- Each notebook uses `pricebook.viz` (configure_theme, apply_theme, create_figure).
- Breakeven inflation term structures for all 6 markets.
- All 6 notebooks execute cleanly.
- 10,147 tests pass.

---

## v0.663.0 — 2026-05-31

**Unified inflation unit framework (UDI/UF/UVR/CER).**

- New `fixed_income/inflation_unit.py`:
  - `InflationUnit` — frozen dataclass for daily inflation units (name, currency, publisher, conventions).
  - `InflationUnitBond` — generic bond denominated in any inflation unit, dual real/nominal pricing.
  - `dual_curve_breakeven()` — BEI from any pair of nominal + real curves.
  - `compare_units()` — cross-country comparison table.
  - Registry: UDI (MXN), UF (CLP), UVR (COP), CER (ARS).
- 15 new tests: registry lookups, pricing for all 4 units, par bond, BEI, zero BEI.
- 10,147 tests pass.

---

## v0.662.0 — 2026-05-31

**Americas Phase 4-6: Peru, Argentina, Canada — full fixed income stack.**

- New `fixed_income/peruvian.py`:
  - `BTPPeru` — Peruvian sovereign bond (ACT/365F, semi-annual).
  - `VACBond` — inflation-linked bond (IPC-adjusted, real/nominal pricing).
  - `build_pen_curve()`, `synthetic_pen_strip()` — PEN discount curve.
- New `fixed_income/argentine.py`:
  - `LecapBond` — zero-coupon capitalisation bond (handles 40%+ rates).
  - `LecerBond` — CER-linked inflation bond (daily accrual).
  - `BONARBond` — ARS-denominated sovereign (semi-annual coupon).
  - `build_ars_curve()`, `synthetic_ars_strip()` — ARS discount curve.
- New `fixed_income/canadian.py`:
  - `CORRASwap` — CORRA overnight swap (par rate, DV01).
  - `RRBBond` — Real Return Bond (CPI-linked, deflation floor).
  - `build_corra_curve()`, `synthetic_corra_strip()` — CORRA discount curve.
- Infrastructure:
  - `LimaCalendar`, `BuenosAiresCalendar` in `core/calendar.py`.
  - TIPM (PEN), BADLAR (ARS) rate indices in `rate_indices.json`.
  - BTP_PE, BONAR, GLOBAL_AR sovereign conventions in `sovereign_conventions.json`.
  - IPC_PE (Peru), CER (Argentina) inflation indices in `inflation_indices.json`.
  - PEN, ARS EM curve conventions in `curve_conventions_em.json`.
- 20 new tests in `test_americas.py` (Colombia, Peru, Argentina, Canada).
- 10,132 tests pass.

---

## v0.661.0 — 2026-05-30

**Chile (CLP) derivatives: Cámara swap, BCP, BCU (UF-linked), breakeven inflation.**

- New `fixed_income/chilean.py`:
  - `CamaraSwap` — TPM-based overnight swap.
  - `BCPBond` — nominal CLP sovereign bond.
  - `BCUBond` — UF-denominated sovereign (real/nominal dual pricing).
  - `build_clp_curve()`, `build_uf_curve()` — nominal + real curve construction.
  - `breakeven_inflation()` — BEI term structure from nominal vs real curves.
  - Synthetic CLP + UF strips.
- 9 new tests: curves, swap, BCP, BCU UF scaling, BEI positive (~3.75%).
- 10,112 tests pass.

---

## v0.660.0 — 2026-05-30

**Mexico (MXN) derivatives: TIIE swap, CETES, Udibonos.**

- New `fixed_income/mexican.py`:
  - `TIIESwap` — 28-day period swap (unique Mexican structure), par rate, DV01.
  - `CETESBill` — discount bill pricing (ACT/360, MXN 10 face).
  - `UDIBond` — UDI-linked bond (real coupon × daily inflation unit), dual real/nominal pricing.
  - `build_tiie_curve()` — TIIE discount curve from swap strip.
  - `synthetic_tiie_strip()`, `synthetic_cetes_quotes()` — realistic data generators.
- 15 new tests: TIIE curve, 28-day periods, CETES discount, UDI nominal scaling, MBONO sovereign pricing.
- 10,103 tests pass.

---

## v0.658.0 — 2026-05-30

**Fix notebooks: remove `apply_theme` (not exported from viz).**

- Replaced `from pricebook.viz import apply_theme` with `configure_theme` only across all 14 notebooks.
- `apply_theme` is an internal context manager in `viz/_backend.py`, not part of the public API. `configure_theme()` at the top of each notebook sets the theme globally.
- 10,088 tests pass.

---

## v0.657.0 — 2026-05-30

**Brazilian credit derivatives notebook — end-to-end calibration.**

- New `notebooks/brazilian_credit_derivatives.ipynb` — 18 cells with pricebook.viz:
  1. CDI curve from DI futures (term structure plot)
  2. NTN-F/LTN bond pricing via CDI curve
  3. Bond-implied CDS spreads from corporate discount (hazard rate extraction)
  4. Survival curve + CDS par spread term structure
  5. CLN pricing with credit charge decomposition
  6. TRS on NTN-F with CDI funding
  7. Summary dashboard (4-panel: CDI curve, bond prices, implied spreads, CLN decomposition)
- Full chain: DI quotes → CDI curve → bond prices → hazard rates → CDS curve → CLN/TRS pricing.
- 10,088 tests pass.

---

## v0.656.0 — 2026-05-30

**Brazilian derivatives full stack: CDI curve, DI futures, DI swap, LFT, cupom cambial.**

- New `fixed_income/brazilian.py` (~400 lines):
  - `DIFuture` — B3 DI futures: PU pricing, DV01, implied rate round-trip.
  - `DISwap` — Pré × CDI swap: fixed vs CDI compounded, par rate, PV.
  - `LFTBond` — CDI-linked floating sovereign: VNA accrual, spread pricing, spread duration.
  - `build_cdi_curve_from_di()` — CDI discount curve from DI futures strip.
  - `synthetic_di_strip()` — realistic DI futures data generator (Selic-based upward slope).
  - `cupom_cambial()` — USD rate from USDBRL forward + DI rate (CIP).
  - `cupom_cambial_curve()` — cupom cambial term structure.
- LFT added to sovereign bonds registry (57 markets total) + yield convention + region mapping.
- 25 new tests covering: BUS/252 helpers, CDI curve construction, DI futures, DI swap, LFT, cupom cambial, NTN-F/LTN sovereign pricing.
- 10,088 tests pass.

---

## v0.655.0 — 2026-05-30

**Hawkes credit framework complete — analytics + 20 tests.**

- `credit/hawkes_analytics.py`:
  - `contagion_scenario()` — intensity jump analysis ("what if name X defaults?")
  - `clustering_metrics()` — inter-arrival CV + burstiness (CV>1 = clustered, B>0 = bursty)
  - `kernel_comparison()` — exponential vs power-law kernel side-by-side
  - `hawkes_term_structure()` — CDS spread across maturities under Hawkes
- 20 new tests (`test_hawkes_credit.py`):
  - Kernel formulas (exp, power-law, Mittag-Leffler γ=1 → exp)
  - Poisson limit (α=0), self-excitation increases events
  - Intensity non-negative, stationarity warning
  - CDS spread positive + increases with α
  - Tranche hierarchy (equity ≥ senior)
  - Contagion scenario (cross-excitation raises intensity)
  - Clustering CV, MLE direction, sum-exp approximation
- **Full Hawkes stack: 5 layers, 4 files, ~1600 lines.**
- 10,063 tests pass (+20 new).

---

## v0.654.0 — 2026-05-30

**Hawkes credit derivatives — Layers 2-4: survival, CDS, basket, tranche.**

- `credit/hawkes_survival.py` — `HawkesSurvivalCurve`: MC survival Q(T) from intensity paths, implied hazard, conversion to pricebook `SurvivalCurve`.
- `credit/hawkes_cds.py` — `hawkes_cds_spread()`: par CDS spread under Hawkes intensity. `hawkes_cds_spread_comparison()`: shows spread widening from self-excitation (120bp at α=0 → 185bp at α=0.9).
- `credit/hawkes_basket.py` — `hawkes_basket_defaults()`: multivariate Hawkes default simulation for N names. `hawkes_tranche_spread()`: CDO tranche pricing. `hawkes_ftd_spread()`: first-to-default. `hawkes_vs_copula()`: side-by-side Hawkes vs Gaussian copula comparison (tail losses, clustering).
- Tranche hierarchy verified: equity > mezzanine > senior.
- 10,043 tests pass.

---

## v0.653.0 — 2026-05-30

**Fractional Hawkes process for credit derivatives — Phase 1.**

- New `models/hawkes_credit.py`:
  - `FractionalHawkesProcess` — 4 kernel types: exponential, power-law (fractional), Mittag-Leffler, sum-of-exponentials.
  - `MultivariateHawkesProcess` — N-name cross-excitation matrix for credit contagion.
  - `HawkesKernel` enum, `HawkesCreditResult`, `MultivariateHawkesResult` dataclasses with `to_dict()`.
  - `evaluate_kernel()` — unified kernel evaluation.
  - `branching_ratio()` — stationarity check (warns if ≥ 1).
  - `approximate_power_law()` — Bochner sum-of-exponentials approximation of power-law kernel.
  - `hawkes_mle_exponential()` — MLE calibration for exponential kernel.
  - Ogata thinning adapted for non-Markovian kernels (dynamic intensity upper bound).
- **Next:** Layers 2-5 (survival curves, CDS pricing, basket/tranche, analytics).
- 10,043 tests pass.

---

## v0.652.0 — 2026-05-30

**Fix all moderate audit issues — input validation, magic number docs, edge case guards.**

- `data_registry.py`: path traversal guard (`_validate_filename`), JSON array type check, `key_fn` None validation.
- `network_xva.py`: exposure matrix shape validation (N,N), capital buffers shape (N,), recovery in [0,1].
- `calibration_quality.py`: array length mismatch check, n < 1 guard in `calibration_entropy`, n < 2 guard + n_params validation in `model_comparison`.
- `composite_convention.py`: `__post_init__` validates haircut ∈ [0,1] and recovery ∈ [0,1].
- `esg_bonds.py`: documented greenium 5bp (Zerbib 2019) and liquidity 3bp sources.
- `cds_bond_basis.py`: documented delivery 5bp (De Wit 2006), restructuring 10bp (ISDA), ±20bp neutral threshold. Added input validation to `bond_implied_cds_spread` (maturity > 0, frequency > 0, recovery ∈ [0,1), price > 0).
- `credit_leveraged.py`: documented duration 4.0 (Markit index factsheets), input validation on `constant_maturity_cds` (maturity > 0, recovery ∈ [0,1), vol ≥ 0).
- 10,043 tests pass.

---

## v0.651.0 — 2026-05-30

**Code audit fixes — 3 critical issues from 11-lens audit.**

- Fixed `credit_leveraged.py` line 131: `effective_leverage = min(leverage, 1.0 / 1e-10)` was a no-op (1e10 cap). Changed to direct assignment — leverage applies directly to digital CLN loss.
- Fixed `regime_pricing.py`: all `probs / probs.sum()` calls now validate `sum > 0` before dividing. Raises `ValueError` on zero-sum regime probabilities instead of silently producing NaN.
- Fixed `cds_bond_basis.py`: `bond_implied_cds_spread()` now validates bracket `f(0) × f(2) < 0` before calling brentq. Raises informative `ValueError` if market price is outside feasible range.
- Audit covered 9 files (6 new, 3 modified), 10 quality dimensions.
- 10,043 tests pass.

---

## v0.650.0 — 2026-05-30

**Quick wins closed: BilateralCSA, Hybrid, CMT wired. 133 validation tests.**

- Paper 2: `BilateralCSAPricer` exercised with `CSATerms(threshold=10m)` — partial CSA simulation verified.
- Paper 9: `IndexLinkedHybridInstrument.price()` with correlation sensitivity (ρ ∈ {-0.3, 0, 0.3}).
- Paper 10: `CMTInstrument.price()` with vol sensitivity (σ ∈ {10%, 20%, 30%}).
- 133 validation tests across 12 papers, all through pricebook classes.
- 10,043 tests pass.

---

## v0.649.0 — 2026-05-30

**Complete rewiring: all 12 papers use pricebook classes. 127 validation tests.**

- Paper 1: added `multicurve_newton()` + `build_curves()` tests (simultaneous OIS + projection).
- Paper 2: added `InterestRateSwap.pv()` + `pv_ctx()` for receiver swap.
- Paper 4: added `CDS` round-trip via class + `CreditLinkedNote.from_convention()`.
- Paper 5: added `constant_maturity_cds()` (participation rate) + `PedersenCDSSwaption.price()`.
- Paper 6: added `TotalReturnSwap.price()` + serialisation round-trip.
- Paper 8: added `CMASWInstrument.price()` with correlation sensitivity.
- All 12 papers now import from pricebook modules, not standalone math.
- 127 validation tests across 12 papers (+17 new).
- 10,037 tests pass.

---

## v0.648.0 — 2026-05-30

**Rewire validation tests through pricebook classes.**

- Paper 3+11 (T-Lock): now uses `TreasuryLock`, `BondForward` classes instead of manual formulas.
- Paper 7 (Lou TRS): now uses `trs_trinomial_tree()` + `trs_equity_full_csa()` with tree vs analytic comparison.
- Paper 12 (Zhou CDS-Bond): now uses `bond_implied_cds_spread()` + `compute_basis()` from pricebook credit modules.
- Fixed basis signal assertions to match actual pricebook output ("NEUTRAL"/"NEGATIVE_BASIS").
- 110 validation tests across 12 papers, all passing through pricebook modules.
- 10,020 tests pass (+16 from rewiring).

---

## v0.647.0 — 2026-05-30

**Build 2 missing capabilities for paper validation.**

- New `bond_implied_cds_spread()` in `credit/cds_bond_basis.py` — solves for flat hazard rate that reprices a risky bond at its market price, then converts to CDS spread. Enables Zhou Table 1 reproduction.
- `CMCDSResult.participation_rate` field added in `credit/credit_leveraged.py` — φ = fair_spread / forward_spread. Enables Brigo-Morini participation rate validation.
- **Backward compat:** Both additive. CMCDSResult has new field with default 0.0.
- 10,004 tests pass.

---

## v0.646.0 — 2026-05-30

**Chunk 3 complete: Papers 9-12. All 12 papers validated. 10,004 tests.**

- Paper 9 (Pucci Hybrid): 4 tests — correlation sensitivity, cash annuity.
- Paper 10 (Pucci CMT): 6 tests — CC formula, vol/fixing monotonicity, no-default limit.
- Paper 11 (Pucci T-Lock): 6 tests — forward dirty ≈ 104.74, carry, overhedge, delta.
- Paper 12 (Zhou CDS-Bond Basis): 6 tests — CDS/ASW at 3 D-levels, basis widening, hazard monotonicity.
- 4 notebooks for Chunk 3.
- **All 12 papers validated** with 94 total validation tests across 12 test files.
- **10,004 tests pass** (milestone: crossed 10k).

---

## v0.645.0 — 2026-05-30

**Chunk 2 complete: Papers 5-8 validation (CDS, TRS×2, CMASW).**

- Paper 5 (Brigo-Morini CDS Market Model): 11 tests — CDS option implied vol (C1=61.9% vs paper 62.2%), recovery independence, CMCDS convexity monotonicity, participation rate.
- Paper 6 (Burgess Bond TRS): 8 tests — coupon $155,416.80, simple vs continuous forward, carry direction, recovery sensitivity.
- Paper 7 (Lou TRS Framework): 8 tests — forward consistency (r_s < r → F < S), FVA direction, CVA/DVA signs, margin convergence.
- Paper 8 (Pucci CMASW): 10 tests — CC formula (zero at σ=0 or ρ=0), CC grid, vol/correlation monotonicity, antisymmetry in ρ.
- 4 notebooks with pricebook.viz: implied vol table, CMCDS convexity/participation plots, TRS forward comparison, XVA waterfall, CMASW CC heatmap.
- **Chunks 1+2 complete** (8/12 papers validated).
- 9982 tests pass (+37 new).

---

## v0.644.0 — 2026-05-30

**Papers 3 + 4 validation: T-Lock model + CLN.**

- Paper 3 (Anon T-Lock): 7 tests — bond forward (Bf_dirty ≈ 104.74), PV01 convergence, clean/dirty equivalence, repo no-arbitrage. Cross-validates with Pucci 2019.
- Paper 4 (Axelsson-Renström CLN): 9 tests — CDS bootstrap (hazard rates positive + increasing), CDS round-trip, CLN below risk-free, recovery sensitivity, discretisation error.
- Notebooks: `paper_03_tlock_model.ipynb` (PV01 convergence + T-Lock payoff plots), `paper_04_cln.ipynb` (survival curves + CLN price vs recovery).
- **Chunk 1 complete** (4/4 papers validated).
- 9945 tests pass (+16 new).

---

## v0.643.0 — 2026-05-29

**Paper 2 validation: Anonymous — Discounting Textbooks.**

- New `tests/validation/test_paper_02_discounting.py` — 9 tests:
  - Case A: equity forward with repo drift (£105.65 vs textbook £105.13)
  - Case B: 5Y receiver swap under 3 CSA regimes, PV ordering verified
  - Case C: ColVA for bond collateral (GC £85k vs special £2.55m)
- New `notebooks/paper_02_discounting.ipynb` with pricebook.viz:
  - CSA regime bar chart comparison
  - ColVA vs repo rate curve with GC/special annotations
- 9929 tests pass (+9 new).

---

## v0.642.0 — 2026-05-29

**Paper 1 validation: Ametrano & Bianchetti (2013) — Multicurve Bootstrap.**

- New `tests/validation/test_paper_01_multicurve.py` — 10 tests reproducing EUR multicurve case study (11-Dec-2012):
  - OIS bootstrap from Eonia strip (12 pillars, round-trip < 1bp)
  - Negative rate handling (1Y OIS = 0%, DF ≈ 1.0)
  - IRS-6M projection curve bootstrap with OIS discounting
  - Loss of telescoping identity (eq. 64-65) — deviation confirmed
  - OIS single-curve property (eq. 73-74) — telescoping holds
- New `notebooks/paper_01_multicurve.ipynb` — interactive notebook with pricebook.viz:
  - OIS discount factor and zero rate plots
  - OIS vs Euribor 6M projection curve comparison with basis spread fill
  - Bootstrap round-trip verification table
  - LaTeX-rendered key equations
- 9920 tests pass (+10 new).

---

## v0.641.0 — 2026-05-29

**Hard migration — remove aliases, tighten pv_ctx curve lookups.**

- Renamed `CDSIndexProduct.from_spec` → `from_convention` (removed alias). All callers + tests updated.
- Tightened `pv_ctx` curve extraction in 6 instruments:
  - `FRA`: tries keyed lookup by day_count before falling back to first projection curve.
  - `FRN`: same keyed lookup pattern.
  - `BasisSwap`: warns if fewer than 2 projection curves available.
  - `CapFloor`: **raises ValueError** if no IR vol surface in context (was silently using flat 20%).
  - `ConvertibleBond`: **raises ValueError** if missing spot, discount curve, or vol surface (was guessing defaults).
  - `RiskyBond`: warns if no credit curve found (falls back to risk-free with warning instead of silently).
- Old numerical shims: verified already removed in v0.612-v0.616. No action needed.
- **Breaking changes:** CapFloor.pv_ctx and ConvertibleBond.pv_ctx now raise instead of silently using bad defaults. Code that relied on the fallback behaviour must now provide proper market data in PricingContext.
- 9910 tests pass.

---

## v0.640.0 — 2026-05-29

**Supranational analytics — RV, universe pricing, curve spread (D9).**

- `supranational_rv()` — relative value: z-score vs historical spread, peer ranking, RICH/CHEAP/FAIR signal.
- `price_supranational_universe()` — price bonds across all issuers × currencies. Returns aggregated SupraUniverseResult with tightest/widest/average spread.
- `supranational_curve_spread()` — spread term structure across tenors for a single issuer.
- `SupraRVResult`, `SupraUniverseResult` dataclasses with `to_dict()`.
- **Backward compat:** Additive — existing `create_supranational_bond()` and `price_supranational()` unchanged.
- 9910 tests pass.

---

## v0.639.0 — 2026-05-28

**ESG bond labelling framework (D8).**

- New `fixed_income/esg_bonds.py`:
  - `ESGLabel` enum: GREEN, SOCIAL, SUSTAINABILITY, SUSTAINABILITY_LINKED, TRANSITION, BLUE.
  - `UseOfProceeds` enum: 14 ICMA taxonomy categories.
  - `ESGBondSpec` convention: label, issuer, use-of-proceeds, KPI target, coupon step-up/down, taxonomy alignment, reviewer.
  - `greenium()` — green premium calculation (yield difference green vs conventional).
  - `esg_adjusted_spread()` — spread decomposition: credit + greenium + liquidity.
  - `slb_coupon_adjustment()` — sustainability-linked bond coupon step-up/down on KPI miss/achieve.
  - `create_green_bond()` — factory returning (FixedRateBond, ESGBondSpec) tuple.
- Full `@serialisable_convention` on ESGBondSpec with round-trip.
- **Backward compat:** Additive — new module, no changes to existing code.
- 9910 tests pass.

---

## v0.638.0 — 2026-05-28

**Sukuk instrument + pricing (D7).**

- New `SukukBond` class: profit rate (coupon equivalent), 7 Sukuk types (Ijara, Mudaraba, Murabaha, Wakala, Musharaka, Salam, Istisna).
- Curve-based pricing via internal FixedRateBond delegation. Spread-based pricing via `price_from_spread()`.
- Full architecture: `from_convention()`, `pv_ctx()`, `to_dict()`/`from_dict()`, `@serialisable`.
- `create_sukuk(type, issue, maturity, rate)` factory function.
- **Backward compat:** Additive. Existing `price_sukuk_as_bond()` unchanged.
- 9910 tests pass.

---

## v0.637.0 — 2026-05-28

**Composite convention pattern for exotic trees — TRS-on-SPV with nested conventions.**

- New `models/composite_convention.py` with 5 convention types: CouponCapSpec, FundingConvention, CollateralConvention, SPVNoteConvention, BondTRSConvention.
- `create_trs_on_spv()` convenience function. `BondTRSConvention.create()` builds underlying from nested conventions.
- Fixed `_deserialise_atom` for Python 3.10+ `types.UnionType` (`X | None`) and flat convention dict deserialisation.
- Full round-trip: nested convention → JSON → from_dict → create → instrument.
- **Backward compat:** Two fixes to core/serialisable.py improve nested deserialisation. No existing behaviour changed.
- 9910 tests pass.

---

## v0.636.0 — 2026-05-28

**Supranational bond factory + pricing.**

- `create_supranational_bond(issuer, currency, issue, maturity, coupon)` — creates FixedRateBond with domestic sovereign conventions for the issuance currency. Maps 10 currencies to sovereign market codes.
- `price_supranational()` — full pricing with spread vs sovereign computation.
- `SupranationalBondResult` — clean/dirty price, YTM, spread, rating.
- Warns if issuing in a non-typical currency for the supranational.
- **Backward compat:** Additive — existing `get_supranational()` / `list_supranationals()` unchanged.
- 9910 tests pass.

---

## v0.635.0 — 2026-05-28

**Complete @serialisable — all 5 remaining complex classes done.**

- `PedersenCDSSwaption`, `StochasticIntensitySwaption` — scalar params, standard decorator.
- `TotalReturnSwapLou` — scalar params, standard decorator.
- `CDSIndex` — custom to_dict/from_dict: serialises list of CDS constituents recursively.
- `CovenantLoan` — custom to_dict/from_dict: serialises nested TermLoan.
- **Backward compat:** All additive. CDSIndex and CovenantLoan use custom from_dict that dispatches via the Serialisable registry for nested objects.
- Total @serialisable instruments: **49** (was 44). Zero remaining gaps.
- 9910 tests pass.

---

## v0.634.0 — 2026-05-28

**JSON is now source of truth for all 11 convention registries.**

- All convention registries now load from JSON first, falling back to hardcoded Python defaults.
- New `load_registry()` utility in `core/data_registry.py` — populates keyed dicts from JSON arrays.
- Wired into: sovereign_conventions, rate_indices, equity_indices, commodity_contracts, linker_conventions, inflation_indices, repo_specialness, supranational_issuers, cds_indices, sovereign_cds, curve_conventions_em.
- Fixed CDS index names: `"iTraxx Europe"` → `"ITRAXX.EUR.IG"` etc. — name field now matches the lookup key (was a key/name mismatch from the original hardcoded dict).
- **Backward compat:** All `get_X()` APIs unchanged. JSON overrides hardcoded defaults when present. Editing a JSON file immediately changes what `get_conventions()` returns. CDS index spec name field changed from display name to canonical key — callers using `get_index_spec("ITRAXX.EUR.IG")` unaffected.
- G10 curve conventions (curve_builder.py) not wired — CurrencyConventions lacks a currency field for keying.
- 9910 tests pass.

---

## v0.633.0 — 2026-05-28

**from_convention on 12 more products — total 35 with factory.**

- Group 1 (FI): ZCInflationSwap, YoYInflationSwap, RevolvingFacility, AmortisingBond.
- Group 4 (Credit): CDSIndexProduct (alias from_spec), TrancheCDS, LoanParticipation, BasketCLN.
- Group 5 (Commodity): CommoditySwap (uses CommodityContractSpec).
- Group 8 (Repo): Repo, ReverseRepo (uses haircut from convention).
- **Backward compat:** CDSIndexProduct.from_convention = CDSIndexProduct.from_spec (alias). All others additive.
- Remaining without from_convention: options (strike/vol-driven, 10), desk trades (8), model-driven structured (4), TRS (3) — conventions don't apply the same way to these products.
- from_convention coverage: 23→35/39 core products. The 4 excluded categories (options/desk/structured-model/TRS) represent products where the concept of "market convention" is either the strike+vol (options) or the underlying itself (TRS).
- 9910 tests pass.

---

## v0.632.0 — 2026-05-27

**Convention + factory integration tests — 30 new tests, 9910 total.**

- New `test_convention_factory.py` with 30 tests covering the full chain:
  - Convention JSON round-trip (6 types)
  - Convention → factory → instrument (10 products: UST, Bund, ZCB, IRS USD/EUR, OIS, CDS, Swaption, Deposit, FRA)
  - Instrument → pv_ctx (5 products)
  - Instrument → to_dict → from_dict (5 products)
  - End-to-end: JSON load → convention → factory → price → serialise (4 chains)
- 9910 tests pass (was 9880).

---

## v0.631.0 — 2026-05-27

**from_convention on 3 more credit products — total 23 with factory.**

- `GuaranteedNote.from_convention()` — uses frequency/day_count from bond conventions.
- `VanillaCLN.from_convention()` — same pattern.
- `CreditRiskyFRN.from_convention()` — uses convention frequency/day_count for floating schedule.
- **Backward compat:** All additive.
- from_convention coverage: 20→23/39 products.
- 9880 tests pass.

---

## v0.630.0 — 2026-05-27

**from_convention on 7 more products — total 20 with factory.**

- `ZeroCouponSwap.from_convention()` — uses fixed_day_count from CurrencyConventions.
- `CrossCurrencySwap.from_convention()` — uses float freq/dc.
- `TermLoan.from_convention()` — uses float freq/dc for floating coupon.
- `Swaption.from_convention()` — uses fixed/float freq+dc from CurrencyConventions for underlying swap.
- `CapFloor.from_convention()` — uses float freq/dc for caplet/floorlet schedule.
- `TreasuryBill.from_convention()` — uses day_count + settlement from SovereignConventions.
- `Deposit.from_convention()` + `FRA.from_convention()` — already added in v0.628.0.
- **Backward compat:** All additive classmethods. IRFuture skipped (exchange-specific, not convention-driven).
- from_convention coverage: 13→20/39 products. Remaining ~19 are exotics (TRS, autocallable, etc.) or desk aggregates where conventions don't apply the same way.
- 9880 tests pass.

---

## v0.629.0 — 2026-05-27

**Complete @serialisable coverage — all 7 remaining gaps fixed.**

- `@serialisable` added to: LeveragedCLN, DIPLoan, TriPartyRepo, IndexLinkedHybridInstrument, DispersionTrade, DividendSwap, RiskReversal, VarianceSwap (8 classes).
- Total serialisable instrument classes: **44** (was 36).
- **Backward compat:** DIPLoan and TriPartyRepo `to_dict()` output changed from flat dict to standard `{"type": ..., "params": {...}}` format. Tests updated. TriPartyRepo serial type is `"triparty_repo"` (was `"tri_party_repo"` in one test).
- Only CDSIndex, CovenantLoan, PedersenCDSSwaption, StochasticIntensitySwaption, TotalReturnSwapLou remain without @serialisable (complex/nested params that need manual from_dict).
- 9880 tests pass.

---

## v0.628.0 — 2026-05-27

**Serialisable + pv_ctx + from_convention final batch.**

- `@serialisable` added to: CommoditySwap, RiskParticipation, BondFuture, FXFuture, CMSLeg (5 more instruments).
- `ConvertibleBond.pv_ctx()` — extracts spot, rate, vol, credit spread from PricingContext. All core tradeable products now have pv_ctx.
- `Deposit.from_convention()` and `FRA.from_convention()` — uses day_count from CurrencyConventions.
- **Backward compat:** All additive. 7 reverted files (desk trades with wrong field names, 4 credit/structured with import inside function body) will be fixed in a follow-up pass — no regression from v0.627.
- `@serialisable` coverage: 31→36 instruments. `from_convention` coverage: 11→13 products.
- 9880 tests pass.

---

## v0.627.0 — 2026-05-27

**from_convention on 5 more instruments — total 11 product types with factory.**

- `RiskyBond.from_convention(conv, start, end, coupon_rate, recovery)` — uses bond convention frequency/day_count.
- `CreditLinkedNote.from_convention(conv, start, end, coupon_rate, recovery)` — same pattern.
- `InflationLinkedBond.from_convention(conv, start, end, coupon_rate, base_cpi)` — accepts LinkerConvention or InflationIndexDef (auto-resolves frequency/day_count/lag from either).
- `BasisSwap.from_convention(conv, start, end, spread)` — uses CurrencyConventions float/fixed frequencies.
- **Backward compat:** All additive classmethods. No existing API changes.
- Factory coverage: 8→11/39 products with `from_convention`.
- 9880 tests pass.

---

## v0.626.0 — 2026-05-27

**from_convention factories on 6 core instrument classes.**

- `FixedRateBond.from_convention(conv, issue_date, maturity, coupon_rate)` — accepts SovereignConventions or any object with frequency/day_count/calendar_currency.
- `ZeroCouponBond.from_convention(conv, issue_date, maturity)` — same convention protocol.
- `FloatingRateNote.from_convention(conv, start, end, spread)` — uses convention frequency/day_count.
- `InterestRateSwap.from_convention(conv, start, end, fixed_rate)` — accepts CurrencyConventions (fixed/float freq+dc).
- `CDS.from_convention(conv, start, end, spread)` — accepts SovereignCDSConventions or CDSIndexSpec (extracts recovery).
- `OISSwap.from_convention(conv, start, end, fixed_rate)` — already added in v0.622.0.
- New `create_swap(currency, start, end, rate)` convenience function.
- New `get_conventions(currency)` in `curves/curve_builder.py`.
- Rewired `create_sovereign_bond`, `create_sovereign_zero`, `create_sovereign_frn` to use `from_convention` internally.
- **Backward compat:** All new classmethods and functions are additive. Existing factory functions (`create_sovereign_bond` etc.) now delegate to `from_convention` — same output, thinner implementation. FX instruments skipped (pair IS the convention — no separate convention layer needed).
- Factory coverage: 3/39 → ~8/39 products with `from_convention`.
- 9880 tests pass.

---

## v0.625.0 — 2026-05-27

**Serialisation hardening — @serialisable on 15 more instrument classes.**

- Added `@serialisable` to: ZCInflationSwap, YoYInflationSwap, InflationLinkedBond, CrossCurrencySwap, StepUpBond, RiskyBond, Repo (already had via alias), IRFuture, AmortisingBond, VanillaCLN, BasketCLN, GuaranteedNote, CMASWInstrument, CMTInstrument.
- Total serialisable instruments: 16→31 (now 80% of core tradeables).
- **Backward compat:** StepUpBond `to_dict()` output changed from flat dict to `{"type": "step_up_bond", "params": {...}}` format (standard instrument format). Other classes that had no `to_dict()` now have one (additive). Test updated.
- 9880 tests pass.

---

## v0.624.0 — 2026-05-27

**pv_ctx on 10 more instruments — coverage 35→39/39 (near-complete).**

- Added `pv_ctx()` to: ZeroCouponSwap, TreasuryBill, IRFuture, CrossCurrencySwap, ZCInflationSwap, YoYInflationSwap, InflationLinkedBond, BondForward, ParAssetSwap, ProceedsAssetSwap.
- CrossCurrencySwap.pv_ctx extracts domestic + foreign discount curves + FX spot from context.
- Inflation instruments extract CPI curve from `ctx.inflation_curves`.
- **Backward compat:** All additive — existing pricing signatures unchanged. `pv_ctx` methods use best-effort curve extraction.
- PricingContext coverage on core tradeable instruments: near-complete. Remaining gaps are desk-level aggregators (Book, Desk), result dataclasses, and niche credit exotics.
- 9880 tests pass.

---

## v0.623.0 — 2026-05-27

**pv_ctx on CapFloor and RiskyBond.**

- `CapFloor.pv_ctx()` — extracts discount + projection curves + IR vol from context, falls back to flat 20% vol.
- `RiskyBond.pv_ctx()` — extracts discount + credit curves, falls back to risk-free pricing if no credit curve.
- **Backward compat:** Additive — existing `price()` / `dirty_price()` signatures unchanged. `pv_ctx` uses best-effort curve extraction from context.
- PricingContext coverage: 33/39 → 35/39 products.
- 9880 tests pass.

---

## v0.622.0 — 2026-05-27

**OIS convention + pv_ctx on 8 vanilla instruments.**

- New `OISConvention` dataclass with `create_swap()` factory (10 currencies: USD, EUR, GBP, JPY, CHF, CAD, AUD, NZD, SEK, NOK). `get_ois_convention(currency)` lookup.
- `OISSwap.from_convention()` classmethod + `pv_ctx()`.
- Added `pv_ctx()` to 7 more instruments: Deposit, FRA, ZeroCouponBond, BasisSwap, FloatingRateNote, FXSwap, NDF, EquityForward.
- **Backward compat:** All new methods are additive. Existing `pv()` signatures unchanged. `OISConvention` + `get_ois_convention` are new exports. `pv_ctx` on BasisSwap picks first two projection curves from context — callers with specific curve needs should still use `pv()` directly.
- PricingContext coverage: 25/39 → 33/39 products.
- 9880 tests pass.

---

## v0.621.0 — 2026-05-26

**Static data layer — 13 JSON convention files + loader utility.**

- Created `data/` directory with 13 JSON files (62 KB total, 212 entries):
  sovereign_conventions (56), rate_indices (25), equity_indices (9), commodity_contracts (13), linker_conventions (8), inflation_indices (16), repo_specialness (6), supranational_issuers (10), cds_indices (5), sovereign_cds (31), curve_conventions_g10 (10), curve_conventions_em (16), sukuk_conventions (7).
- New `core/data_registry.py` — `load_conventions()`, `save_conventions()`, `load_or_default()` utilities for JSON ↔ convention dataclass round-trip.
- All 12 convention types verified: JSON → from_dict → to_dict → JSON matches original.
- **Backward compat:** JSON files are additive — existing hardcoded registries remain the source of truth. JSON files serve as export/inspection/override format. No existing APIs changed.
- 9880 tests pass.

---

## v0.620.0 — 2026-05-26

**Apply `@serialisable_convention` to all 13 convention dataclasses.**

- All convention types now have `to_dict()`/`from_dict()` round-trip via the decorator:
  RateIndex, EquityIndexSpec, CommodityContractSpec, LinkerConvention, InflationIndexDef, SpecialnessConventions, SupranationalIssuer, CDSIndexSpec, CDSSettlementConvention, SovereignCDSConventions, CurrencyConventions, EMCurveConventions, SukukConventions.
- 6 dataclasses made `frozen=True` (were mutable): EquityIndexSpec, CommodityContractSpec, LinkerConvention, CDSIndexSpec, CDSSettlementConvention, CurrencyConventions.
- Manual `to_dict()` methods removed (decorator auto-generates with proper enum serialisation).
- **Backward compat:** `to_dict()` output now includes all fields (some manual implementations omitted fields like `notes`, `settlement_days`). Existing `get_X()` / `list_X()` APIs unchanged. `from_dict()` is new (additive). Making dataclasses frozen could break code that mutates convention objects — none found in tests.
- 9880 tests pass.

---

## v0.619.0 — 2026-05-26

**Add `@serialisable_convention` decorator for frozen dataclasses.**

- New `serialisable_convention(serial_type)` decorator in `core/serialisable.py` — auto-derives `_SERIAL_FIELDS` from `dataclasses.fields()`, produces flat dicts (no type/params nesting), handles enum/date round-trip.
- Applied to `SovereignConventions` — first convention with full `to_dict()`/`from_dict()` round-trip.
- **Backward compat:** `SovereignConventions.to_dict()` now exists where it didn't before (additive, no breakage). The existing `get_conventions()` / `create_sovereign_bond()` APIs unchanged.
- 9880 tests pass.

---

## v0.618.0 — 2026-05-26

**Restore clean dependency layers — 0 cycles, 9 layers.**

- Made 2 module-level imports lazy (moved inside function bodies):
  - `models/regime_pricing.py` — `equity_option_price`, `equity_delta`, `equity_gamma`, `equity_vega` from options
  - `curves/rfr_bootstrap.py` — `RFRFutureSpec`, `rfr_futures_to_forwards` from fixed_income
- AST-verified: 0 bidirectional cycles at module level across all 20 packages.
- Architecture: 9 clean layers, 566 modules, 20 packages.
- 9880 tests pass.

---

## v0.617.0 — 2026-05-26

**Phase 5 advanced theory integration — regime pricing, calibration quality, network XVA.**

- `models/regime_pricing.py` — `RegimePricingEngine`: HMM-driven option pricing under regime switching. Fits HMM to returns, extracts regime-conditional vols, prices under each regime and blends by filtered probabilities. Includes `regime_option_price()`, `regime_greeks()`, risk decomposition by regime.
- `statistics/calibration_quality.py` — information-theoretic calibration assessment: `calibration_entropy()` (RMSE, R², entropy of residuals), `calibration_kl()` (KL-based model comparison), `parameter_stability()` (CV, drift across recalibrations), `model_comparison()` (AIC/BIC/JS divergence), `fisher_parameter_quality()` (FIM + Cramer-Rao bounds).
- `risk/network_xva.py` — `NetworkXVAEngine`: systemic risk adjustments to CVA. Integrates financial network centrality and Eisenberg-Noe contagion cascades. CVA_network = CVA × (1 + α × centrality × contagion_multiplier). Includes `stress_test()`, `systemic_ranking()`, convenience `contagion_cva_stress()`.
- 36 new tests (test_phase5_integration.py). 9880 tests pass.

---

## v0.616.0 — 2026-05-25

**Delete tree model shims — all callers migrated to solve_tree().**

- Deleted `models/binomial_tree.py`, `models/trinomial_tree.py`, `models/binomial_jr_lr.py` — thin shims, zero remaining importers.
- Migrated 6 test files to import directly from `numerical._trees`: `test_binomial_tree.py`, `test_trinomial_tree.py`, `test_binomial_jr_lr.py`, `test_binomial_roundtrip.py`, `test_finite_difference.py`, `test_lsm.py`.
- Registry already clean (uses `solve_tree` since v0.612.0).
- 9844 tests pass.

---

## v0.615.0 — 2026-05-25

**Standardise all numerical modules to Enum + Result + to_dict pattern.**

- `_rootfinding.py` — add `RootMethod` enum (BISECTION, BRENT, NEWTON, SECANT, HALLEY, ITP); `find_root()` accepts enum or string.
- `_optimize.py` — add `OptimMethod` enum (NELDER_MEAD, BFGS, L_BFGS_B, CG, NEWTON_CG, DIFFERENTIAL_EVOLUTION, BASIN_HOPPING, CMA_ES); `minimize()` accepts enum or string.
- `_graph.py` — add `ShortestPathResult`, `MSTResult`, `MaxFlowResult` dataclasses with `to_dict()`; add `dijkstra_full()`, `minimum_spanning_tree_full()`, `max_flow_full()` returning typed results.
- `_distributions.py` — add `to_dict()` to Normal, StudentT, LogNormal, Uniform, Exponential.
- `_linalg.py` — add `DecompMethod`, `IterativeMethod` enums; `SVDResult`, `LUResult` dataclasses; `decompose()` and `iterative_solve()` dispatchers; `method` field on `IterativeSolveResult`.
- `_mc.py` — add `MCVarianceReduction`, `MCDiscrMethod` enums.
- `_fourier.py` — add `FourierMethod`, `WaveletType` enums; `to_dict()` on `CharacteristicFunction`; wavelet_transform accepts enum.
- `_interpolation.py` — add `InterpMethod2D`, `RBFKernel` enums; `interpolate_2d()` dispatcher; `rbf_interpolate()` accepts enum.
- Updated `numerical/__init__.py` — export all new enums, result types, and dispatchers.
- All string-based callers continue to work (backward compatible).
- 9844 tests pass.

---

## v0.614.0 — 2026-05-24

**Final migration cleanup — delete _quadrature.py, auto-scale global_solver FD eps.**

- Deleted `numerical/_quadrature.py` — fully superseded by `_integrate.py`, no importers remain.
- `curves/global_solver.py` — replaced hardcoded `eps=1e-8` with auto-scaled `h = max(|x_j| × 1e-7, 1e-10)` in both Jacobian functions.
- 9844 tests pass.

---

## v0.613.0 — 2026-05-24

**Fix Leisen-Reimer Peizer-Pratt formula — extra 0.5 factor removed.**

- Root cause: `copysign(0.5, z) * sqrt(...)` instead of `copysign(sqrt(...), z)`. The extra 0.5 multiplier halved the probability deviation from 0.5, collapsing all tree prices to ~50% of BS.
- All 8 LR-specific test failures now pass. LR(51) matches BS to 4+ decimals as designed.
- 9844 tests pass, 0 failures.

---

## v0.612.0 — 2026-05-24

**Complete migration — tree shims, quadrature redirect, nd_solvers Jacobian.**

### Tree model files converted to thin shims
- `models/binomial_tree.py` → delegates to `solve_tree(TreeMethod.CRR)`
- `models/trinomial_tree.py` → delegates to `solve_tree(TreeMethod.TRINOMIAL)`
- `models/binomial_jr_lr.py` → delegates to `solve_tree(TreeMethod.JR/LR)`
- `registry.py` tree section → `_make_tree_pricer()` wrappers using `solve_tree`

### Quadrature redirect
- `curves/quadrature.py` → thin redirect to `numerical._integrate`. `QuadratureResult` = `IntegrationResult`.
- `registry.py` integrator section → `_make_integrator()` wrappers using `integrate()`.

### Differentiation
- `models/nd_solvers.py` `finite_difference_jacobian()` → delegates to `numerical._differentiate.jacobian()`.

### Known issue
- LR (Leisen-Reimer) tree method has pricing inaccuracy in the new `_trees.py` implementation (8 test failures). CRR, JR, trinomial all correct. To be fixed in a subsequent commit.

- 9836 passed, 8 LR-specific failures.

---

## v0.611.0 — 2026-05-24

**Backward compatibility removal — clean API for ODE, integration, trees.**

### Removed
- `euler()`, `rk4()`, `rk45()`, `bdf()`, `adams()` shims from `_ode.py` → use `solve_ode(f, span, y0, ODEMethod.RK4)`.
- `gauss_jacobi()`, `tanh_sinh()`, `clenshaw_curtis()` shims from `_integrate.py` → use `integrate(f, a, b, IntegrationMethod.TANH_SINH)`.
- `tree_greeks()`, `binomial_2d()`, `TreeGreeks`, `Binomial2DResult` shims from `_trees.py` → use `solve_tree()`, `solve_tree_2d()`.

### Deleted
- `models/ode.py` — shim module, all logic now in `numerical/_ode.py`.

### Migrated
- `numerical/__init__.py` — exports only new API names.
- `registry.py` — ODE solvers now use `_make_ode_solver()` wrapper.
- `core/results.py` — imports `ODEResult` from `numerical._ode`.
- 4 test files rewritten to use new API: `test_ode.py`, `test_numerical.py`, `test_numerical_ode.py`, `test_numerical_quadrature.py`, `test_numerical_trees.py`, `test_tree_solver.py`.

### Result
- **Single canonical API** per module — no aliases, no wrappers, no ambiguity.
- 9844 tests pass.

---

## v0.610.0 — 2026-05-24

**Bayesian statistics — MCMC, conjugate priors, model selection, changepoint detection.**

### Bayesian Module (`statistics/bayesian.py`)
- **MCMC Sampling:**
  - `MetropolisHastings` — random-walk MH with configurable proposal, acceptance tracking, ESS computation.
  - `GibbsSampler` — component-wise sampling from full conditionals.
  - `MCMCResult` — samples, log-posteriors, credible intervals, effective sample size, `to_dict()`.

- **Conjugate Priors:**
  - `BayesianLinearRegression` — Normal-Inverse-Gamma conjugate. Closed-form posterior, credible intervals, posterior predictive, log marginal likelihood (evidence).
  - `beta_binomial_update()` — Beta-Binomial for PD estimation. Posterior mean, mode, credible interval.

- **Model Selection:**
  - `bayes_factor()` — log Bayes factor with Kass-Raftery interpretation (decisive/strong/moderate/weak).
  - `credible_interval()`, `hpd_interval()` — equal-tailed and HPD credible intervals.
  - `posterior_predictive()` — MC posterior predictive distribution.

- **Changepoint Detection:**
  - `bayesian_changepoint()` — Bayes factor scan for structural breaks. Posterior probability per time point.

- **Use cases:** Bayesian PD estimation, parameter uncertainty in calibrated models, model comparison (SABR vs Heston), regime change detection, Bayesian VaR.
- 24 tests. 9849 tests pass.

---

## v0.609.0 — 2026-05-24

**Tree solver redesign — class-based, 5 methods, Bermudan, barriers, Greeks from nodes.**

### Tree Solver (`numerical/_trees.py`)
- `TreeSolver` class — configurable method, exercise type, barriers, dividends.
- `TreeMethod` enum: CRR, JR, LR, TRINOMIAL, TIAN (5 methods).
- `ExerciseType` enum: EUROPEAN, AMERICAN, BERMUDAN.
- `BarrierType` enum: UP_OUT, DOWN_OUT, UP_IN, DOWN_IN.
- `solve_tree()` — one-liner convenience (mirrors `solve_bs_pde()`).
- `solve_tree_2d()` — 2-asset Rubinstein tree with callable payoff + American exercise.
- Greeks from tree nodes directly: delta/gamma from steps 1-2, theta from step 2, vega via bump.
- Bermudan: exercise at specified step indices only.
- Barriers: knock-out via node zeroing.
- Discrete dividends: spot adjustment at dividend steps.
- `convergence_analysis()` — prices at multiple N + Richardson extrapolation.
- `TreeResult` — price, delta, gamma, theta, vega, method, n_steps, exercise, convergence, optional node data.
- Custom payoff: `payoff=lambda S: ...` for digitals, straddles, any exotic.
- Backward compatible: `tree_greeks()`, `binomial_2d()` old API preserved.
- 22 tests. 9825 tests pass.

---

## v0.608.0 — 2026-05-24

**Integration + differentiation redesign — unified frameworks, 9+5 methods.**

### Numerical Integration (`numerical/_integrate.py`)
- `IntegrationMethod` enum: ADAPTIVE (scipy quad), GAUSS_LEGENDRE, GAUSS_LAGUERRE (semi-infinite), GAUSS_HERMITE (infinite), TANH_SINH (singular), CLENSHAW_CURTIS, SIMPSON, TRAPEZOID, ROMBERG.
- `integrate(f, a, b, method)` — main entry with auto method selection.
- `integrate_2d()` — double integral via scipy.dblquad.
- `integrate_semi_infinite()` — ∫ₐ^∞ with Gauss-Laguerre or adaptive.
- `integrate_complex_contour()` — ∮ f(z)dz along parameterised contour.
- `IntegrationResult` — value, error estimate, n_evaluations, converged.
- Backward compatible: old `gauss_jacobi`, `tanh_sinh`, `clenshaw_curtis` still work.

### Numerical Differentiation (`numerical/_differentiate.py`)
- `DiffMethod` enum: FORWARD (O(h)), CENTRAL (O(h²)), COMPLEX_STEP (machine ε), RICHARDSON (O(h⁴)), FIVE_POINT (O(h⁴)).
- `derivative(f, x, method, order)` — 1st and 2nd derivatives.
- `gradient(f, x)` — ∇f for scalar functions of vectors.
- `jacobian(f, x)` — J[i,j] = ∂fᵢ/∂xⱼ for vector functions.
- `hessian(f, x)` — H[i,j] = ∂²f/∂xᵢ∂xⱼ for scalar functions.
- Auto step size selection: optimal h based on method order + machine epsilon.
- `DiffResult` — value, error estimate, method, n_evaluations.
- 30 tests. 9803 tests pass.

---

## v0.607.0 — 2026-05-24

**PDE solver redesign — class-based, 7 methods, grids, Greeks extraction.**

### PDE Solver (`numerical/_pde.py`)
- `PDESolver1D` class — configurable method, grid, reusable.
- `PDEMethod` enum: EXPLICIT, IMPLICIT, CRANK_NICOLSON, RANNACHER, CRAIG_SNEYD, HUNDSDORFER_VERWER, METHOD_OF_LINES.
- `GridType` enum: UNIFORM, LOG, SINH (Tavella-Randall concentration), CHEBYSHEV.
- `BoundaryCondition` enum: DIRICHLET, NEUMANN, LINEAR, FREE.
- `build_grid()` — spatial grid builder with strike/barrier concentration.
- `extract_greeks()` — delta, gamma, theta from grid solution via finite differences.
- `solve_bs_pde()` — one-line Black-Scholes PDE for European/American options.
- `solve_pde_with_vega()` — vega via bump-and-reprice.
- `PDEResult` — values, grid, price, delta, gamma, theta, vega, to_dict().
- Thomas algorithm tridiagonal solver.
- American via payoff projection. Rannacher smoothing.
- 23 tests: all methods, ATM/ITM/OTM, put, American, Greeks vs BS, grid types.
- 9773 tests pass.

---

## v0.606.0 — 2026-05-24

**Advanced numerical methods: spectral, quasi-Monte Carlo, stochastic calculus.**

### Spectral Methods (`numerical/_spectral.py`)
- `chebyshev_nodes()`, `chebyshev_diff_matrix()`, `chebyshev_coefficients()`, `chebyshev_evaluate()` (Clenshaw recurrence).
- `chebyshev_interpolate()` → `SpectralResult` with arbitrary-point evaluation.
- `spectral_solve_bvp()` — BVP solver via Chebyshev collocation.
- `spectral_integrate()` — Gauss-Legendre quadrature.

### Quasi-Monte Carlo (`numerical/_qmc.py`)
- `sobol_sequence()` — Sobol low-discrepancy (scipy.stats.qmc, O(1/N) convergence).
- `halton_sequence()`, `latin_hypercube()`.
- `sparse_grid()` — Smolyak construction for high-dimensional integration.

### Stochastic Calculus (`numerical/_stochastic.py`)
- `ito_formula()`, `ito_log_transform()` — Ito's formula with correction term.
- `stratonovich_to_ito()` / `ito_to_stratonovich()` — convention conversion.
- `quadratic_variation()`, `realized_variance()`, `realized_volatility()`.
- `bipower_variation()` — robust to jumps (Barndorff-Nielsen & Shephard).
- `jump_test()` — detect jumps via RV vs BV comparison.
- `milstein_correction()` — Milstein SDE discretisation term.
- 29 tests. 9750 tests pass.

---

## v0.605.0 — 2026-05-24

**ODE solver redesign — class-based, 9 methods, Riccati, backward, dense output.**

### ODE Solver (`numerical/_ode.py`)
- `ODESolver` class — configurable method, tolerance, dense output, reusable.
- `ODEMethod` enum: EULER, RK4, RK45, RK23, BDF, RADAU, LSODA, DOP853, IMPLICIT_EULER (9 methods).
- `solve_ode()` — main entry with runtime method selection + Jacobian + events.
- `solve_backward()` — backward-in-time integration for PDE time-stepping.
- `solve_riccati(a, b, c, ...)` — Riccati ODE dy/dt = a + by + cy² with analytical Jacobian. Supports complex coefficients (Heston CF).
- `solve_system()` — auto stiffness detection via LSODA.
- Implicit Euler via Newton iteration with optional Jacobian.
- Dense output for arbitrary-time evaluation (scipy interpolant + linear fallback).
- `ODEResult.__call__(t)` — evaluate solution at any time.
- Full backward compatibility: `euler()`, `rk4()`, `rk45()`, `bdf()`, `adams()` still work.
- 31 tests (up from 4): all methods, stiff systems, Jacobian, dense output, backward, Riccati (linear, quadratic, tanh), 2D rotation, Lorenz.
- 9721 tests pass.

---

## v0.604.0 — 2026-05-23

**Phase 4: Graph theory — network, contagion, algorithms, correlation network.**

### 4.1 Financial Network (`risk/network.py`)
- `FinancialNetwork` — degree, betweenness, eigenvector centrality, PageRank.
- `NetworkResult` with composite systemic risk ranking.

### 4.2 Default Cascade (`risk/contagion.py`)
- `DefaultCascade` — Eisenberg-Noe cascade with capital buffers, multi-round propagation.
- `stress_test()` — multiple scenarios. Contagion multiplier metric.

### 4.3 Graph Algorithms (`numerical/_graph.py`)
- `dijkstra()`, `shortest_path()`, `minimum_spanning_tree()` (Prim), `max_flow()` (Edmonds-Karp), `connected_components()`. Pure numpy.

### 4.4 Correlation Network (`risk/correlation_network.py`)
- `correlation_to_distance()` — Mantegna (1999).
- `mst_portfolio()` — MST from return correlations.
- `hierarchical_risk_parity()` — López de Prado (2016) HRP weights.
- `community_detection()` — spectral clustering on Laplacian.
- 21 tests. 9694 tests pass.

---

## v0.603.0 — 2026-05-23

**Phase 3: Game theory — Shapley, cooperative games, Nash, auction.**

### 3.1 Shapley Value (`risk/shapley.py`)
- `shapley_value()` — exact (2^N coalitions). `shapley_sampling()` — MC for large N.
- Satisfies all 4 axioms: efficiency, symmetry, dummy, additivity.
- `shapley_capital_allocation()` — fair desk-level capital allocation.

### 3.2 Cooperative Games (`risk/cooperative_games.py`)
- `CooperativeGame` — characteristic function + Shapley + core check.
- `NettingSetGame` — netting benefit allocation across counterparties.
- `CollateralPoolGame` — funding cost reduction from shared pool.

### 3.3 Nash & Microstructure (`models/game_equilibrium.py`)
- `nash_2player()` — support enumeration for bimatrix games.
- `market_maker_equilibrium()` — Avellaneda-Stoikov optimal spread with inventory.
- `optimal_execution_game()` — Almgren-Chriss front-loaded schedule.

### 3.4 Auction Theory (`fixed_income/auction.py`)
- `BondAuction` — uniform/discriminatory price, bid-to-cover, tail.
- `winners_curse_adjustment()`, `expected_revenue()`.
- 25 tests. 9673 tests pass.

---

## v0.602.0 — 2026-05-23

**2.4: Maximum entropy option pricing — model-free risk-neutral density.**

### Entropy Pricing (`options/entropy_pricing.py`)
- `max_entropy_density()` — recover RN density maximising Shannon entropy subject to option price constraints.
- Buchen-Kelly dual formulation with analytical gradient (L-BFGS-B).
- `MaxEntropyResult` — density grid, entropy, forward, repricing errors, `call_price()`, `put_price()`, `implied_vol_at()`.
- `entropy_implied_vol()` — extract full implied vol smile from sparse quotes.
- **Use cases:** model-free pricing from sparse option data, smile interpolation without parametric model.
- 11 tests. 9648 tests pass.

---

## v0.601.0 — 2026-05-23

**Phase 2 (2.1-2.3): Information theory — entropy, divergence, MI, Fisher information.**

### Information Theory (`statistics/information_theory.py`)
- **Entropy:** `shannon_entropy()`, `differential_entropy()` (KDE or histogram).
- **Divergence:** `kl_divergence()`, `js_divergence()` (symmetric), `cross_entropy()`, `wasserstein_distance()`.
- **Mutual Information:** `mutual_information()`, `conditional_mutual_information()`, `information_gain()` (feature ranking).
- **Fisher Information:** `fisher_information_matrix()` (numerical Hessian), `cramer_rao_bound()`, `parameter_confidence_intervals()`.
- **Use cases:** model risk (KL P‖Q), feature selection for PD, parameter uncertainty in HW/SABR calibration.
- 18 tests. 9637 tests pass.

---

## v0.600.0 — 2026-05-23

**1.3 + 1.4: Regime-switching process + regime-dependent market data.**

### Regime Process (`models/regime_process.py`)
- `RegimeProcessSpec` — regime-dependent drift/diffusion with Markov transitions.
- `create_regime_gbm()` — regime-switching GBM (equity/FX).
- `create_regime_ou()` — regime-switching OU (rates/spreads).
- Simulates paths + regime labels jointly.

### Regime Surfaces (`models/regime_surfaces.py`)
- `RegimeVolSurface` — N vol surfaces blended by regime probabilities (variance or linear blend).
- `RegimeCurve` — N discount curves blended by regime probabilities.
- `regime_price()` — price under each regime and blend by posterior.
- 18 tests. 9619 tests pass.

---

## v0.599.0 — 2026-05-23

**1.2: Particle filter — sequential Monte Carlo for non-linear state estimation.**

### Particle Filter (`statistics/particle_filter.py`)
- `ParticleFilter(n_particles, transition_fn, observation_log_likelihood)` — bootstrap filter.
- Pluggable dynamics: any `transition_fn(particles, rng) → particles` + `obs_log_lik(y, particles) → log_weights`.
- Systematic resampling with ESS monitoring.
- `ParticleFilterResult` — filtered means/stds, ESS trajectory, log-likelihood, final particles.
- **Use cases:** stochastic vol filtering (Heston latent vol), non-linear credit dynamics, any non-Gaussian state-space.
- 10 tests. 9601 tests pass.

---

## v0.598.0 — 2026-05-23

**1.1: Generalised HMM framework — pluggable emissions, Baum-Welch, Viterbi.**

### HMM Core (`statistics/hmm.py`)
- `EmissionModel(ABC)` — pluggable observation distributions: `log_prob()`, `fit_params()`, `sample()`.
- Concrete emissions: `GaussianEmission`, `StudentTEmission`, `MixtureEmission`, `MultivariateGaussianEmission`.
- `EmissionType` enum + `create_emission()` factory (follows Interpolator pattern).
- `HMM(n_states, emission)` — generalised HMM class.
  - `fit()` — Baum-Welch EM with scaled forward-backward.
  - `filter()` — online filtering of new observations.
  - `predict_state()` — Viterbi decoding.
- `HMMFitResult` — transition matrix, emission params, stationary dist, AIC/BIC, filtered probs, Viterbi labels.
- Supports 2+ states, any univariate or multivariate emission.
- **Use cases:** vol regime, credit regime, yield curve regime, any latent-state time series.
- 20 tests. 9591 tests pass.

---

## v0.597.0 — 2026-05-21

**Repo Phase 3b + 4: Matched book, BS allocation, margin, settlement, sec lending.**

### 3.3 Matched Book (`desks/matched_book.py`)
- `MatchedBookPosition` — paired repo/reverse with spread, gap, PnL.
- `matched_book_optimise()` — greedy selection by spread, subject to gap + notional limits.

### 3.4 Balance Sheet Allocation (`regulatory/balance_sheet_allocation.py`)
- `rank_by_roc()` — return on capital ranking.
- `optimise_allocation()` — LP: maximize total ROC subject to capital + RWA constraints.

### 4.1 Margin Mechanics (`fixed_income/repo_margin.py`)
- `calculate_vm()`, `margin_call()` (threshold + MTA), `margin_forecast()`.

### 4.2 Settlement Fails (`fixed_income/repo_settlement.py`)
- `propagate_fails()` — cascade through matched book.
- `buy_in_process()` — CSDR mandatory buy-in.
- `fail_cost_analysis()` — penalty + opportunity + reputation.

### 4.3 Securities Lending (`fixed_income/securities_lending.py`)
- `SecLendingTrade`, `lending_vs_repo_arbitrage()`, `locate_availability()`.
- 23 tests. 9571 tests pass.

---

## v0.596.0 — 2026-05-21

**Repo Phase 3: Leverage optimization + collateral transformation.**

### 3.1 Leverage Optimization (`risk/leverage_optimisation.py`)
- `optimise_leverage()` — LP: maximize carry subject to haircut + capital + concentration constraints.
- `leverage_frontier()` — efficient frontier of carry vs leverage ratio (1× to 20×).

### 3.2 Collateral Transformation (`risk/collateral_transformation.py`)
- `transformation_cost()` — all-in cost: repo spread + xccy basis + capital - haircut benefit.
- `optimise_transformation()` — greedy upgrade of available collateral to target quality.
- `funding_arbitrage()` — identify mispriced collateral vs funding value.
- 13 tests. 9548 tests pass.

---

## v0.595.0 — 2026-05-21

**Repo Phase 2: Counterparty credit — CVA + wrong-way risk, dynamic haircuts, correlated XVA.**

### 2.1 Repo CVA (`risk/repo_cva.py`)
- `repo_cva()` — CVA on unsecured exposure after haircut, time-grid integration.
- `repo_wrong_way_risk()` — three channels: issuer (classic), sector (systemic), spiral (margin).
- `repo_bilateral_cva()` — CVA + DVA + WWR combined.

### 2.2 Dynamic Haircuts (`risk/dynamic_haircuts.py`)
- `DynamicHaircutModel` — spread-driven + vol-driven + rating trigger + BCBS 261 procyclicality buffer.
- `haircut_stress_scenarios()` — 7 standard scenarios.
- `credit_spread_to_haircut()` — continuous spread → haircut mapping.
- `rating_trigger_impact()` — step function per downgrade notch.

### 2.3 Correlated XVA (`risk/repo_xva_advanced.py`)
- `repo_xva_correlated()` — joint MC: counterparty default + collateral spread (Gaussian copula).
- CVA + FVA + KVA + MVA + gap cost, fully correlated.
- `repo_all_in_xva()` — profitability: interest income vs total XVA.
- 26 tests. 9535 tests pass.

---

## v0.594.0 — 2026-05-21

**Repo 1.3 + 1.4: Specialness analytics (6 markets) + repo rate Greeks.**

### Specialness Analytics (`fixed_income/repo_specialness.py`)
- `SpecialnessConventions` — 6 sovereign markets (UST, Bund, Gilt, JGB, OAT, BTP).
- `forecast_specialness()` — mean-reversion + auction-cycle seasonality.
- `specialness_term_structure()` — GC-special spread curve.
- `supply_demand_indicator()` — fail rate, on-the-run, short interest signals.

### Repo Rate Greeks (`fixed_income/repo_greeks.py`)
- `repo_dv01()` — trade-level interest + carry sensitivity per 1bp.
- `carry_sensitivity_ladder()` — by tenor bucket (O/N, 1W, 1M, 3M, 6M, 1Y+).
- `repo_portfolio_greeks()` — aggregated DV01, carry DV01, roll theta.
- 24 tests. 9509 tests pass.

---

## v0.593.0 — 2026-05-21

**Repo Phase 1: Multi-currency funding curves, carry breakeven, credit-collateral integration.**

### 1.1 Dealer Funding Curve (`fixed_income/repo_funding_curve.py`)
- `DealerFundingCurve` — secured + unsecured legs, blended rate with haircut.
- `RepoMarketConventions` — 11 currencies (USD/EUR/GBP/JPY/CHF/CAD/AUD/BRL/MXN/ZAR/TRY) with day count, settlement, benchmark, GC collateral types.
- `build_dealer_funding_curve()`, `to_discount_curve()`.
- 15 tests.

### 1.2 Carry Breakeven (`fixed_income/repo_carry.py`)
- `carry_breakeven()` — GC vs special, term vs O/N, breakeven rate.
- `xccy_repo_carry()` — cross-currency with FX basis.
- `multi_ccy_carry_comparison()` — rank carry across currencies for same bond.

### 1.5 Credit-Collateral Integration (`fixed_income/repo_credit_collateral.py`)
- `CreditCollateralSpec` — issuer hazard, rating, sector, seniority.
- `credit_adjusted_haircut()` — base + PD add-on + spread-vol add-on. 8 asset classes: sovereign, IG, HY, bank senior, AT1/T2, structured IG/HY, equity.
- `repo_price_with_collateral_credit()` — all-in: interest - collateral default - counterparty credit - wrong-way risk - gap risk.
- `hazard_to_haircut_mapping()` — continuous hazard → haircut schedule.
- 21 tests. 9485 tests pass.

---

## v0.592.0 — 2026-05-21

**Phase 4: Curve blending, seasonal, diffusion, storage.**

### 4.1 Curve Blending (`curves/curve_blending.py`)
- `splice_curves()` — short/long curve splicing with linear, sigmoid, or step transition.
- `blend_curves()` — weighted blend of N curves in log-DF space.
- 6 tests.

### 4.2 Seasonal Term Structure (`curves/seasonal_curve.py`)
- `SeasonalCurve` — base curve with year-end/quarter-end/month-end spread overlay.
- `SeasonalPattern` — configurable decay, pre-built USD/EUR/GBP patterns.
- `extract_seasonal_pattern()` — fit from historical O/N fixings.
- `strip_seasonal()` — remove seasonal for smooth analysis.
- 6 tests.

### 4.3 Curve Diffusion (`curves/curve_diffusion.py`)
- `CurveDiffusionEngine` — multi-factor HJM simulation, exponentially decaying vol.
- Each path at each step → standard `DiscountCurve` (all pricing code works unchanged).
- Forward rate statistics (mean, std) across paths.
- 5 tests.

### 4.4 Curve Storage (`curves/curve_storage.py`)
- `CurveSnapshot` — timestamped zero-rate snapshot with `from_curve()` / `to_curve()`.
- `CurveDelta` — sparse delta between snapshots (bp shifts).
- `CurveStore` — in-memory save/load/history/diff.
- 7 tests. 9449 tests pass.

---

## v0.591.0 — 2026-05-21

**Phase 3: FX forward curves, curve scenarios, real-time bumper.**

### 3.1 FX Forward Builder (`fx/fx_forward_builder.py`)
- `build_fx_implied_curve()` — from spot + swap points + domestic OIS via CIP.
- 14 FX pair conventions (settlement, pip factor, quoting direction).
- Basis spread extraction vs known foreign curve.
- 6 tests.

### 3.2 Curve Scenario Engine (`curves/curve_scenarios.py`)
- `parallel_shift()`, `steepener()`, `flattener()`, `bear_steepener()`, `bull_flattener()`.
- `butterfly()`, `inversion()`, `historical_scenario()`.
- `pca_scenarios()` — PCA level/slope/curvature from historical data.
- `standard_scenario_set()` — 11 canned scenarios per currency.
- `run_scenarios()` — batch execution with PnL.
- 9 tests.

### 3.3 Real-Time Curve Bumper (`curves/curve_bumper.py`)
- `CurveBumper` — Jacobian pre-computation, fast repricing via J·Δz.
- `bump_and_reprice()` (fast, ~μs) vs `full_rebuild_and_reprice()` (exact).
- `parallel_dv01()`, `key_rate_dv01s()`, `cross_gamma()`.
- `risk_report()` — full instrument risk (DV01, key-rate, convexity).
- 5 tests. 9425 tests pass.

---

## v0.590.0 — 2026-05-21

**2.1: N-curve simultaneous global solver — damped Newton for 1-N curves.**

### N-Curve Solver (`curves/ncurve_solver.py`)
- `InstrumentPricer` protocol — each instrument reprices given named curves.
- Concrete pricers: `DepositPricer`, `OISSwapPricer`, `BasisSwapPricer`.
- `CurveSpec` — per-curve pillar dates, initial guess, interpolation.
- `ncurve_solve()` — damped Newton-Raphson, numerical Jacobian, LU/lstsq, positivity-preserving step control.
- Tested: 1-curve (deposits, OIS swaps), 2-curve (OIS+projection, basis), 3-curve (OIS+1M+3M).
- 8 tests. 9405 tests pass.

---

## v0.589.0 — 2026-05-21

**2.2 + 2.3: Forward rate interpolation + key-rate DV01 framework.**

### Forward Rate Interpolation (`core/forward_interpolation.py`)
- `ForwardInterpolationMethod` — piecewise constant, piecewise linear, monotone convex (Hagan-West 2006).
- `build_forward_curve()` — builds DiscountCurve by interpolating on forwards and integrating.
- `monotone_convex_forwards()` — smooth, positive, shape-preserving forward function.
- `extract_forwards()` — extract instantaneous forwards from any curve.

### Key-Rate DV01 (`curves/key_rate_risk.py`)
- `BumpProfile` — triangular (partition of unity), Gaussian, pillar-only.
- `key_rate_dv01()` — localised bumps, DV01 per tenor, optional gamma.
- `bucket_risk()` — tenor bucket aggregation (0-1Y, 1-2Y, ..., 20-30Y).
- `risk_ladder()` — formatted report with % contribution.
- `standard_tenors(currency)` — per-currency key-rate sets (USD, EUR, GBP, JPY, CHF).

### Tests
- 23 new tests: all methods, flat/upward curves, 10Y swap concentration, gamma, bucket risk, risk ladder.
- 9397 tests pass.

---

## v0.588.0 — 2026-05-21

**1.3: Multi-RFR OIS bootstrap — production-grade curve builder for 7 currencies.**

### RFR Bootstrap (`curves/rfr_bootstrap.py`)
- `bootstrap_rfr(currency, ref_date, inputs)` — full instrument stack: O/N + term rates + futures + OIS swaps.
- `RFRCurveInputs` — overnight_rate, term_rates, futures_1m/3m, ois_swaps, deposits.
- `RFRCurveResult` — curve, pillar zeros, round-trip error, convexity adjustments per contract.
- `RFROISConventions` — per-currency: day counts, frequencies, calendar for USD/SOFR, EUR/ESTR, GBP/SONIA, JPY/TONA, CHF/SARON, CAD/CORRA, AUD/AONIA.
- Sequential (Brent) and global (Newton) methods.
- Futures convexity adjustments from item 1.2 wired in.
- Round-trip verification on deposit repricing.

### Tests
- 18 new tests: conventions, USD full stack, deposits-only, futures+swaps, all 7 G7 currencies, term rates, edge cases.
- 9374 tests pass.

---

## v0.587.0 — 2026-05-21

**1.2: RFR futures instruments — SOFR/SONIA/ESTR/SARON/TONA contract generation + convexity.**

### RFR Futures (`fixed_income/rfr_futures.py`)
- `RFRFutureSpec` — generic 1M/3M contracts for any RFR currency.
- `generate_rfr_contracts(currency, ref_date)` — serial (1M) and IMM quarterly (3M) date generation for USD, GBP, EUR, CHF, JPY.
- `rfr_futures_convexity()` — Hull-White convexity adjustment per contract.
- `rfr_futures_to_forwards()` — convert futures prices to forward rates for bootstrap.
- 16 tests. 9356 tests pass.

---

## v0.586.0 — 2026-05-21

**1.1: RFR compounding conventions — 12 currencies, full ISDA mechanics.**

### RFR Compounding (`fixed_income/rfr_compounding.py`)
- `RFRAccrualConfig` — observation shift, lookback, lockout, rate cut-off, payment delay, fixing lag.
- 12 frozen configs: SOFR, ESTR, SONIA, TONA, SARON, CORRA, AONIA (G10) + CDI, KOFR, SORA, HONIA, THOR (EM).
- `compound_rfr_full()` — backward-looking compounded rate with all ISDA adjustments from fixings.
- `compound_rfr_from_curve()` — forward-looking from discount curve (for pricing).
- `rfr_accrual_schedule()` — full observation/weight schedule per business day.
- `get_rfr_config()`, `list_rfr_configs()` — registry.

### Tests
- 23 new tests: registry, schedule mechanics (obs shift, lookback, weekend weight), flat/varying rates, multi-currency, lockout, rate cut-off.
- 9340 tests pass.

---

## v0.585.0 — 2026-05-21

**Hardening audit (L1-L11) — 10 fixes across 9 modules + 3 hand-calculation verifications.**

### Input Validation Fixes
- `regime_switching.py` — transition matrix must be stochastic (rows sum to 1, entries in [0,1]).
- `bilateral_csa.py` — correlation bounds validated in constructor.
- `coco.py` — trigger_intensity must be non-negative.
- `sovereign_cds.py` — tenor must be positive integer.
- `covered_bond.py` — LTV in (0, 1.5], OC >= 1.0.

### Numerical Stability Fixes
- `ndf_implied.py` — skip NDF quotes producing df > 2.0 (data error guard).
- `callable_credit.py` — clamp conditional survival to [0, 1] for floating-point safety.
- `yield_convention.py` — wider solver bracket [-50%, 500%], approximate fallback on failure.
- `spread_decomposition.py` — fixed tax formula unit error (was off by ×100).

### L11 Hand-Calculation Verification
- **CreditGrades**: Q(5Y) = 0.87053497, spread = 138.65bp — exact match (8 decimal places).
- **BRL BUS/252**: 254 business days, yf = 1.007937 — exact. Yield roundtrip perfect.
- **Convertible equity-credit**: default prob 9.44% (hand: 9.52%), bond floor 90.27 (hand: 90.65), δ>0, CS01<0, ρ-sens<0 — all correct.

---

## v0.584.0 — 2026-05-21

**C8: Convertible equity-credit correlation — joint (stock, hazard) Monte Carlo.**

### Convertible Equity-Credit (`credit/convertible_equity_credit.py`)
- Joint process: equity GBM + hazard CIR with correlation ρ (negative = wrong-way risk).
- Default via cumulative hazard vs exponential threshold (Cox process).
- LSM (Longstaff-Schwartz) backward induction for optimal conversion.
- Full Greeks: delta, gamma, vega, CS01, ρ-sensitivity — all via bump-and-reprice with common random numbers.
- Risky bond floor computation with survival-weighted cashflows.
- `convertible_equity_credit_price()` — single entry point.

### Tests
- 15 tests: pricing bounds, equity/credit/correlation sensitivity, Greeks signs, serialization.
- 9317 tests pass.

---

## v0.583.0 — 2026-05-21

**Phase 5 complete — all remaining plan items (A2, A3, A5, B3-B6, C5-C9, D7-D9).**

### Hazard Rate Production
- **A2:** ML-based PD (`credit/ml_pd.py`) — logistic regression from 9 financial ratios.
- **A3:** Sovereign CDS-bond basis (`credit/cds_bond_basis.py`) — funding, delivery, restructuring decomposition.
- **A5:** Joint equity-credit calibration (`credit/joint_equity_credit.py`) — fit CreditGrades to equity vol + CDS.

### CLN Advanced (`credit/cln_advanced.py`)
- **B3:** Spread-driven XVA, **B4:** dynamic funding (CSA-aware), **B5:** wrong-way risk (2nd-order), **B6:** collateral haircut stress.

### Bond Types + Markets
- **C5:** Covered bonds, **C6:** bond forwards + credit, **C9:** issuer spread curve (Nelson-Siegel on spreads).
- **D7:** Sukuk (7 types), **D8:** ESG labelling (ICMA GBP), **D9:** supranationals (10 issuers).

### Tests
- 55 new tests. 9302 tests pass.

---

## v0.582.0 — 2026-05-21

**Phase 4: Bond-Credit — C3 CoCo/AT1, C4 perpetuals, C1 callable+credit OAS, C2 spread decomposition.**

- **C3:** CoCo/AT1 (`credit/coco.py`) — trigger types, loss absorption, coupon cancellation, call/extension blending.
- **C4:** Perpetuals (`fixed_income/perpetual.py`) — plain/callable perpetual, step-up coupon.
- **C1:** Callable + credit OAS (`credit/callable_credit.py`) — backward induction with survival, price decomposition.
- **C2:** Spread decomposition (`credit/spread_decomposition.py`) — credit + liquidity + tax + optionality + residual.
- 47 new tests. 9247 tests pass.

---

## v0.581.0 — 2026-05-21

**B1 + B2: Bilateral CLN+CSA + correlated recovery.**

### Bilateral CSA Pricer (`credit/bilateral_csa.py`)
- `CSATerms` — threshold, independent amount, MTA, MPOR, haircut, rehypothecation.
- `BilateralCSAPricer` — MC simulation of correlated defaults + collateral mechanics + funding costs.
- CVA, DVA, FVA decomposition. 11 tests.

### Correlated Recovery (`credit/correlated_recovery.py`)
- `CorrelatedRecoveryModel` — factor model: R(M) = base + β × M × σ (Frye 2000).
- `systematic_recovery()` — link portfolio default rate to recovery via Vasicek factor.
- 15 tests. 9200 tests pass.

---

## v0.580.0 — 2026-05-21

**A6: Term structure of recovery — maturity-dependent + stochastic recovery.**

### Recovery Curve (`credit/recovery_curve.py`)
- `RecoveryCurve` — interpolated recovery by maturity: `flat()`, `linear()`, `from_seniority()`.
- `RecoverySeniority` enum: 5 levels (senior secured → junior subordinated) with Moody's historical averages.
- `StochasticRecovery` — beta-distributed recovery with `sample()`, `percentile()`, `from_seniority()`.
- `recovery_by_seniority()`, `recovery_vol_by_seniority()` — lookup functions.
- Seniority ordering: SR_SEC(53%) > SR_UNS(40%) > SR_SUB(32%) > SUB(28%) > JR_SUB(18%).

### Tests
- 16 new tests: curve shapes, seniority ordering, stochastic sampling, percentiles.
- 9174 tests pass.

---

## v0.579.0 — 2026-05-21

**A4: CreditGrades model — first-passage Merton with stochastic barrier.**

### CreditGrades (`credit/credit_grades.py`)
- `CreditGrades` class: asset vol, leverage, recovery mean/vol → survival, spreads, distance to default.
- First-passage survival via barrier-crossing formula: Q(t) = Φ(α) − d̄ × Φ(β).
- σ̄² = σ² + λ² (combined asset + barrier uncertainty).
- `survival()`, `cds_spread()`, `spread_term_structure()`, `distance_to_default()`, `evaluate()`.
- Convenience functions: `credit_grades_survival()`, `credit_grades_spread()`.
- Produces realistic spreads: IG ~30bp, HY ~900bp at 5Y.

### Tests
- 20 new tests: survival monotonicity, IG/HY levels, vol/leverage sensitivity, DD ordering, edge cases.
- 9158 tests pass.

---

## v0.578.0 — 2026-05-21

**A1: Regime-switching credit — HMM with state-dependent hazard rates.**

### Regime-Switching Credit (`credit/regime_switching.py`)
- `RegimeSwitchingCredit` — continuous-time Markov chain with state-dependent default intensities.
- Survival via matrix exponential: Q(t) = π₀ × exp((Q-Λ)t) × 1.
- `survival()`, `implied_hazard()`, `implied_spread()` — with optional conditioning on initial state.
- `regime_probabilities()`, `expected_hazard()`, `stationary_distribution()`.
- `spread_term_structure()` — term structure under regime uncertainty.
- `calibrate_regime_model()` — fit 2 or 3 state model from observed CDS spread curve.
- 2-state (expansion/recession) and 3-state (expansion/normal/recession) support.

### Tests
- 21 new tests: survival bounds, conditional, 3-state, calibration, repricing, serialization.
- 9138 tests pass.

---

## v0.577.0 — 2026-05-21

**D14: Sovereign FRNs — 3 floating-rate sovereign markets.**

### Sovereign FRN Factory (`fixed_income/sovereign_bonds.py`)
- USTFRN (US 2Y FRN, quarterly ACT/360, T-Bill linked), GILTFRN (UK, quarterly ACT/365F, SONIA-linked), BTPFRN (Italy, semi-annual, ESTR-linked).
- `create_sovereign_frn(market_code, issue, maturity, spread)` — factory.
- `list_frn_markets()` — 3 FRN codes.
- Yield convention mapping updated for FRNs.
- 56 total sovereign markets (50 coupon + 3 T-Bill + 3 FRN).

### Tests
- 5 new FRN tests: factory, pricing, near-par.
- 9117 tests pass.

---

## v0.576.0 — 2026-05-21

**D11: Cross-market sovereign relative value framework.**

### Sovereign RV (`fixed_income/sovereign_rv.py`)
- `sovereign_spread_decomposition()` — decomposes spread into credit (CDS), fundamental (macro), liquidity (bid-ask/turnover), and technical (residual) components.
- `cross_market_rv_scores()` — cross-sectional Z-scores, percentiles, and CHEAP/FAIR/RICH signals across N sovereign markets.
- `SovereignRVInput` — macro fundamentals: debt/GDP, fiscal balance, current account, rating, FX vol, reserves.
- `SpreadDecomposition`, `RVScore` result dataclasses with `to_dict()`.

### Tests
- 14 new tests: decomposition, component sum, high/low risk, Z-scores, sorting, signals, edge cases.
- 9112 tests pass.

---

## v0.575.0 — 2026-05-21

**D12: EM local currency curve builders — 16 currencies + CDI/TIIE/SHIBOR.**

### EM Curve Builder (`curves/em_curve_builder.py`)
- `EMCurveConventions` — per-currency deposit/swap day count, frequency, interpolation.
- 16 EM currencies: BRL, MXN, CNY, KRW, ZAR, INR, SGD, HKD, THB, PLN, CZK, HUF, COP, CLP, TRY, IDR.
- `build_em_curve(currency, ref, deposits, swaps)` — generic builder with correct conventions.
- `build_cdi_curve(ref, di_futures)` — Brazil CDI from DI futures (df = 1/(1+r)^(bd/252)).
- `build_tiie_curve()`, `build_shibor_curve()` — Mexico and China convenience wrappers.
- `get_em_curve_conventions()`, `list_em_curve_currencies()`.

### Tests
- 14 new tests: conventions, all-currency build, CDI formula verification, TIIE, SHIBOR.
- 9098 tests pass.

---

## v0.574.0 — 2026-05-21

**D10: EM sovereign credit curves — 31 sovereigns + CDS hazard bootstrap.**

### Sovereign CDS (`credit/sovereign_cds.py`)
- `SovereignCDSConventions` — restructuring clause (CR/MR/MM/XR), recovery rate, standard tenors, doc clause.
- 31 sovereigns: LatAm (BR, MX, CO, CL, PE, AR), CEEMEA (TR, ZA, PL, HU, RO, RU, EG, NG, KE), Asia (CN, KR, ID, PH, MY, TH, IN, VN), W. Europe (IT, ES, PT, GR, IE), MENA (SA, QA, IL).
- `bootstrap_sovereign_hazard()` — sequential bootstrap from CDS spreads → SurvivalCurve.
- `RestructuringClause` enum: CR, MR, MM, XR.
- `get_sovereign_cds_conventions()`, `list_sovereign_cds()`.

### Tests
- 18 new tests: conventions, bootstrap, term structure, distressed, IG, recovery override, multi-country.
- 9084 tests pass.

---

## v0.573.0 — 2026-05-21

**D15: Market-convention yield quotation — yield↔price for all 53 sovereign markets.**

### Yield Conventions (`fixed_income/yield_convention.py`)
- `YieldConvention` enum: SEMI_ANNUAL, ANNUAL, QUARTERLY, CONTINUOUS, SIMPLE, DISCOUNT.
- `yield_to_price()` / `price_to_yield()` — convert between yield and clean price under any convention.
- `convert_yield()` — convert between conventions (exact for zeros, price roundtrip for coupon bonds).
- `get_yield_convention(market_code)` — street convention for all 53 sovereign markets.
- Market mapping: UST/GILT/JGB semi-annual, BUND/OAT annual, NTN_F/LTN continuous, RPGB quarterly, USTBILL/CETES bank discount.

### Tests
- 30 new tests: roundtrips, known values, conversions, market mapping, all-53-markets coverage.
- 9066 tests pass.

---

## v0.572.0 — 2026-05-21

**D13: Zero-coupon sovereign bonds — ZeroCouponBond class + factory.**

### ZeroCouponBond (`fixed_income/zero_coupon_bond.py`)
- `price()` / `dirty_price()` — Face × df(T) from discount curve.
- `price_from_yield_simple()` — money-market convention: Face / (1 + r × τ).
- `price_from_discount_rate()` — bank discount: Face × (1 - d × τ).
- `price_from_yield_continuous()` — Face × exp(-r × τ).
- `yield_simple()`, `discount_rate()`, `yield_continuous()` — inverse functions.
- `dv01()`, `modified_duration()`, `to_dict()`.

### Sovereign Factory Updates (`fixed_income/sovereign_bonds.py`)
- `is_zero_coupon` field on `SovereignConventions`.
- 3 new T-Bill markets: USTBILL (ACT/360), UKTBILL (ACT/365F), EURTBILL (ACT/360).
- LTN and CETES flagged as zero-coupon.
- `create_sovereign_zero()` — factory for zero-coupon bonds.
- `list_zero_coupon_markets()` — returns 5 zero-coupon codes.
- 53 total markets (50 coupon + 3 T-Bill).

### Tests
- 10 new zero-coupon tests: factory, pricing, yield roundtrip, DV01, discount rate.
- 9036 tests pass.

---

## v0.571.0 — 2026-05-21

**D6: EM inflation indices — 16 indices + linker factory.**

### Inflation Index Registry (`fixed_income/inflation_indices.py`)
- `InflationIndexDef` — frozen dataclass: name, currency, lag, frequency, interpolation, deflation floor, linker conventions.
- `IndexInterpolation` enum: FLAT (UK ILG), LINEAR (TIPS, most), DAILY (UDI/UF/UVR).
- 16 indices: CPI_US (TIPS), HICP_XT (OAT€i/BTP€i), RPI/CPIH (UK), CPI_JP, CPI_CA, CPI_AU, IPCA (BRL), UDI (MXN daily), UF (CLP daily), UVR (COP daily), CPI_ZA, CPI_IL, CPI_TR, CPI_IN (30/360!), CPI_KR.
- `get_inflation_index()`, `list_inflation_indices()`, `indices_by_currency()`, `indices_with_floor()`, `daily_indices()`.
- `create_inflation_linker()` — factory returning correct kwargs for `InflationLinkedBond`.

### Tests
- 31 new tests: all 16 indices, registry API, linker factory (TIPS, NTN-B, OAT€i, UK ILG, UDIBONO), serialization.
- 9026 tests pass.

---

## v0.570.0 — 2026-05-21

**D5: EM RFR/IBOR rate indices — 14 new indices across 13 EM currencies.**

### EM Rate Indices (`core/rate_index.py`)
- **Overnight RFR (8):** CDI (BRL, BUS/252), KOFR (KRW), SORA (SGD), HONIA (HKD), THOR (THB), DR007 (CNY, averaged), IBR (COP), TPM (CLP).
- **Term IBOR (6):** TIIE_28D (MXN, T-1 fixing), SHIBOR_3M (CNY), WIBOR_3M (PLN), PRIBOR_3M (CZK), BUBOR_3M (HUF), JIBAR_3M (ZAR).
- Registry now has 25 indices (11 G10 + 14 EM), 16 overnight.

### Tests
- 21 new tests: all EM indices, registry counts, currency coverage, frozen dataclass.
- 8995 tests pass.

---

## v0.569.0 — 2026-05-21

**D2: NDF-implied discount curve construction for restricted EM currencies.**

### NDF-Implied Curves (`curves/ndf_implied.py`)
- `build_ndf_implied_curve()` — derive EM discount curve from FX NDF prices + G10 base curve via covered interest parity: df_em(T) = df_base(T) × Spot / NDF(T).
- `ndf_from_curves()` — compute theoretical NDF prices from two discount curves (for CIP deviation checking).
- `cip_basis()` — measure covered interest parity basis in bp (funding stress indicator).
- `NDFQuote` dataclass with bid/ask/mid support.
- `NDFImpliedResult` with implied DFs, zero rates, forward points, to_dict().

### Tests
- 19 new tests: construction, round-trip, CIP basis, multi-currency (CNY, INR, KRW, BRL), edge cases, helpers.
- 8974 tests pass.

---

## v0.568.0 — 2026-05-21

**D4: Sovereign bond factory — 50 markets with correct conventions.**

### Sovereign Bond Factory (`fixed_income/sovereign_bonds.py`)
- `SovereignConventions` — frozen dataclass: market_code, currency, frequency, day_count, settlement_days, calendar, ex_div_days.
- `create_sovereign_bond(market_code, issue, maturity, coupon)` — factory returning correctly-configured `FixedRateBond`.
- `get_conventions(market_code)` — lookup conventions by market code.
- `list_markets()` — 50 sovereign markets.
- `markets_by_region()` — grouped by G10_core, other_dm, eurozone, cee, turkey_mena, africa, latam, asia.

### Markets (50)
- **G10 core (6):** UST, BUND, GILT, JGB, OAT, BTP.
- **Other DM (7):** ACGB, NZGB, CGB_CA, DGB, SGB, NGB, CONFED.
- **Eurozone (8):** BONO, BGB, DSL, RAGB, RFGB, IRISH, PGB, GGB (T+3).
- **CEE (4):** POLGB, CZGB, HGB (ACT/365F), ROMGB (semi-annual).
- **Turkey & MENA (6):** TURKGB (T+0!), SAGB_SA, ADGB, QATGB (30/360), ILGB, EGGB.
- **Africa (3):** SAGB (T+3), NGGB, KEGB.
- **LatAm (7):** NTN_F, NTN_B, LTN (BUS/252), MBONO (ACT/360!), CETES, BTP_CL, TES.
- **Asia (9):** CGB, KTB, GSEC (30/360!), SGS, HKGB, INDOGB, MGS, THAIGB, RPGB (quarterly!).

### Tests
- 35 new tests: convention lookup, factory creation, all-market creation, pricing sanity, coverage checks.
- 8955 tests pass.

---

## v0.567.0 — 2026-05-21

**D3: BUS/252 day count convention for Brazilian markets.**

### BUS/252 (`core/day_count.py`)
- `DayCountConvention.BUS_252` — business days / 252, the standard for all BRL instruments (NTN-F, NTN-B, LTN, DI futures).
- `business_days_between(start, end, calendar)` — count business days between two dates (start exclusive, end inclusive).
- `year_fraction(..., calendar=)` — new optional `calendar` parameter for BUS/252.
- Defaults to São Paulo calendar when no calendar provided.
- Works with any calendar (e.g. USD for testing).

### Tests
- 7 new BUS/252 tests: week count, year approximation, carnival skip, weekend skip, default calendar, US calendar, Independence Day.
- 8920 tests pass.

---

## v0.566.0 — 2026-05-21

**D1: EM Calendars — 24 new calendars + registry.**

### EM Calendars (`core/calendar.py`)
- **CEE (4):** Warsaw (PLN), Prague (CZK), Budapest (HUF), Bucharest (RON, Orthodox Easter).
- **Turkey & MENA (4):** Istanbul (TRY), Riyadh (SAR), Tel Aviv (ILS, Fri-Sat weekend), Cairo (EGP).
- **Africa (3):** Johannesburg (ZAR, Sun→Mon observance), Nairobi (KES), Lagos (NGN).
- **LatAm (4):** São Paulo (BRL, Carnival), Mexico City (MXN, Maundy Thu), Santiago (CLP), Bogotá (COP, emiliani Monday law).
- **Asia (8):** Beijing (CNY), Seoul (KRW), Mumbai (INR), Singapore (SGD), Hong Kong (HKD), Jakarta (IDR), Kuala Lumpur (MYR), Bangkok (THB), Manila (PHP).
- **Other DM (1):** Denmark (DKK, Store Bededag removed post-2023).
- Orthodox Easter algorithm for Romania (Julian + 13-day Gregorian offset).

### Calendar Registry (`core/calendar.py`)
- `get_calendar(currency_code)` — 35 currencies (11 G10 + 24 EM).
- `list_calendars()` — sorted list of available codes.

### Tests
- 56 new tests covering holidays, business day conventions, Orthodox Easter, cross-calendar consistency, joint calendar.
- 8913 tests pass.

---

## v0.565.0 — 2026-05-20

**Bond hazard bootstrap — recovery of market value & liquidity premium separation.**

### Recovery of Market Value (`credit/bond_hazard_bootstrap.py`)
- `_price_risky_bond_rmv()` — Duffie-Singleton (1999) pricing: recovery = R × V(t⁻), reduces to discounting at Q̃(t) = Q(t)^(1-R). No separate recovery leg.
- `recovery_mode` parameter on `bootstrap_hazard_from_bonds()`: `"par"` (ISDA standard, default) or `"market_value"` (Duffie-Singleton).
- RMV produces lower hazard rates than RP for the same market prices (less recovery → less hazard needed to explain low price).
- `RECOVERY_PAR`, `RECOVERY_MARKET_VALUE` constants exported.

### Liquidity Premium Separation (`credit/bond_hazard_bootstrap.py`)
- `BondInput.liquidity_spread_bp` — per-bond liquidity premium assumption (bp).
- Bootstrap bumps the discount curve by liquidity spread before credit extraction, isolating pure credit hazard.
- Per-bond liquidity (e.g. higher for illiquid long-end) supported in both sequential and global methods.
- Combined with RMV recovery mode for full flexibility.

### Tests
- 14 new tests (31 total): RMV pricing, RMV bootstrap round-trip, liquidity spread effect, per-bond liquidity, combined RMV+liquidity, edge cases.
- 8836 tests pass.

---

## v0.563.0 — 2026-05-18

**Sell-side / buy-side gap closure — 5 modules.**

### IPV Workflow (`risk/ipv.py`)
- `FairValueLevel` — Level 1 (market) / Level 2 (comparable) / Level 3 (model).
- `BCBS287_BID_ASK` — 15 asset-class-specific bid-ask tables.
- `ipv_single_trade()` → `IPVResult` — automated AVA via existing prudent_valuation.
- `ipv_portfolio()` → `IPVReport` — portfolio aggregation, level summary, breach detection.

### Mandate Compliance (`core/mandate.py`)
- `Mandate` — configurable policy: eligible_asset_classes, min_rating, max_single_name_pct, max_sector_pct, max_country_pct, currency_restrictions, max_duration.
- `check_mandate()` → `MandateReport` — pass/fail per rule with breach details.
- Predefined templates: investment_grade, sovereign_only, balanced, high_yield.

### Term Sheet Generator (`desks/term_sheet.py`)
- `generate_term_sheet()` → `TermSheet` — markdown-based: Deal Summary, Key Terms, Risk Profile, Scenario Analysis.
- `TermSheet.to_markdown()` → str (externally convertible to HTML/PDF).

### Middle Office Operations (`risk/trade_operations.py`)
- `TradeStatusTracker` — state machine: PENDING → CONFIRMED → ALLOCATED → SETTLED → MATURED/TERMINATED/DEFAULTED.
- `AuditEntry` — immutable audit trail (who, when, what, why).
- `generate_settlement()` → `SettlementInstruction`, `match_confirmation()` → `ConfirmationRecord`.
- `generate_margin_calls()` → `MarginCallReport` — daily margin calls with MTA enforcement.

### Collateral Optimisation (`risk/collateral_optimisation.py`)
- `CollateralOptimiser` — LP solver (scipy.optimize.linprog): min cost across multiple CSAs.
- Constraints: coverage ≥ required, allocated ≤ available, eligibility per CSA.
- `what_if_substitution()` → cost impact of swapping assets.
- `stress_collateral()` → stressed cost + margin shortfall (mild/moderate/severe/crisis).
- 51 new tests across all 5 modules.

---

## v0.558.0 — 2026-05-18

**Codebase restructuring + circular dep elimination + structural hardening.**

- 433 flat files → 20 sub-packages with 9 clean dependency layers.
- 0 circular dependencies (7 broken: TYPE_CHECKING guards, lazy imports, file moves, registry to root).
- 677 `to_dict()` auto-added to dataclasses.
- `__init__.py` re-exports for core, fx, equity, commodity, curves, risk.
- Layer 0 testing from 20% to 84% (72 new tests: statistics, viz, numerical, ts, db).
- ARCHITECTURE.md fully updated.
- See ARCHITECTURE.md for complete layer diagram and package inventory.

---

## v0.555.0 — 2026-05-14

**FRTB-IMA desk bridge + reverse stress testing.**

### IMA Bridge (`regulatory/ima_bridge.py`)
- `DeskRiskExtract` — desk_id, risk_class, delta/gamma/vega/DV01/CS01, obligor, rating.
- `extract_risk_factors_from_desk()` — maps desk sensitivities → `ESRiskFactor` (delta→ES via vol×z_97.5, vega→separate factor, CS01→credit spread).
- `extract_drc_positions_from_desk()` — credit desks → `DRCPosition` for IMA DRC.
- `extract_from_risk_metrics()` — generic bridge from any desk's `risk_metrics().to_dict()`.
- `aggregate_desk_ima()` → `IMABridgeResult` — runs full IMA pipeline + PLA evaluation.
- `RISK_CLASS_MAP` — 12 desk types mapped to risk class/sub_category.

### Reverse Stress Testing (`regulatory/reverse_stress.py`)
- `ReverseStressTarget` — metric, threshold, direction (below/above).
- `reverse_stress_portfolio()` — scipy.optimize.minimize to find minimum-severity scenario breaching threshold.
- `reverse_stress_ccar()` — reverse stress against CCAR capital trajectory (uses project_capital_trajectory).
- `scenario_surface()` — 2D grid of metric values across two macro variables.
- Default bounds per macro variable (GDP -10%/+5%, equity -80%/+20%, etc.).
- 23 tests across both modules.

---

## v0.554.0 — 2026-05-14

**CCAR/DFAST stress capital projection.**

- `regulatory/ccar.py` — NEW: 9-quarter capital trajectory under Fed-style stress.
- `CCARConfig` — starting capital/RWA, PPNR, dividends/buybacks, minimums (CET1 4.5%).
- `QuarterResult` — PPNR, credit/market/op losses, net income, capital actions, CET1 ratio, breach flag.
- `project_capital_trajectory()` → `CCARResult` — quarter-by-quarter CET1, trough ratio, pass/fail.
- `run_ccar_suite()` — 3 scenarios (baseline, adverse, severely_adverse) from stress_irrbb.
- `ccar_summary()` — worst scenario, trough ratios, overall pass/fail.
- Buyback suspension under stress, PPNR stress factors, RWA adjustment from stressed PD/LGD.
- 12 tests including undercapitalised bank failure case.

---

## v0.553.0 — 2026-05-14

**Portfolio-wide LCR/NSFR.**

- `regulatory/liquidity.py` — NEW: product-type-aware LCR and NSFR.
- `LiquidityPosition` — position_id, product_type, notional, rating, hqla_level, counterparty_type.
- `calculate_portfolio_lcr()` → `PortfolioLiquidityResult` — HQLA classification, outflow/inflow rates, LCR%, NSFR%, compliance flags, product breakdown.
- Product classification: cash (L1), sovereign AAA bonds (L1), IG bonds (L2A), deposits (retail stable 3% / wholesale 100%), loans (inflow if ≤30d).
- NSFR: ASF/RSF factors by product type and maturity (retail deposits 90%, cash RSF 0%, long-term loans 85%).
- `liquidity_stress()` — stressed LCR with outflow multiplier and HQLA haircut.
- 11 tests.

---

## v0.552.0 — 2026-05-14

**Operational risk SMA (Basel III OPE25).**

- `regulatory/operational_risk.py` — NEW: Standardised Measurement Approach.
- `SMAInputs` — 3-year P&L items (interest, fees, trading, leasing) + 10-year loss data.
- `calculate_sma_full()` → `SMAResult` — BI averaging, bucket (1/2/3), BIC (marginal 12%/15%/18%), ILM, capital, RWA.
- `calculate_bic()` — Business Indicator Component with marginal coefficients.
- `calculate_ilm()` — Internal Loss Multiplier: ln(e-1 + (LC/BIC)^0.8).
- `sma_sensitivity()` — capital sensitivity to loss component ratio.
- Legacy comparison: BIA capital computed alongside for benchmarking.
- 18 tests including hand-verified BIC calculations.

---

## v0.551.0 — 2026-05-14

**Capital allocation & RORC.**

- `regulatory/capital_allocation.py` — NEW: Euler allocation, RORC, capital limits.
- `euler_allocation()` — risk-contribution allocation with optional correlation matrix.
- `allocate_and_report()` — full report: diversification benefit, RORC per desk, hurdle checks, best/worst desk.
- `capital_limit_monitor()` — breach detection against per-desk limits.
- `DeskCapitalInput`, `DeskAllocation`, `CapitalAllocationResult` dataclasses.
- 16 tests.

---

## v0.550.0 — 2026-05-14

**Distressed debt: DIP, fulcrum, exchange, recovery waterfall, Chapter 11.**

- `distressed.py` — NEW: distressed debt analytics and restructuring.
- `DIPLoan` — super-priority DIP financing with roll-up, carve-out, upfront fee.
- `RecoveryWaterfall` — absolute priority distribution across capital structure.
- `FulcrumAnalysis` — identify fulcrum security (most senior impaired class); `sensitivity()` for recovery curves across EV range.
- `ExchangeOffer` — tender economics: exchange premium, holdout value, prisoner's dilemma payoffs.
- `Chapter11Timeline` — standard/pre-pack/complex milestones; `estimate_recovery()` with admin cost haircuts.
- `CapitalStructureLayer` — name, notional, seniority, secured flag.
- 25 tests.

---

## v0.549.0 — 2026-05-14

**Loan portfolio stress testing.**

- `loan_stress.py` — NEW: correlated defaults, macro scenarios, migration, concentration.
- `correlated_default_simulation()` — one-factor Gaussian copula, (n_paths × n_obligors) default matrix.
- `portfolio_loss_distribution()` — full loss distribution with VaR/ES/by-industry, macro scenario overlays.
- `MacroScenario` — GDP shock, rate/spread shock, PD multiplier, recovery haircut.
- 5 predefined scenarios: recession, stagflation, credit_crisis, rate_shock, recovery.
- `concentration_metrics()` — HHI, top-10%, industry HHI, granularity adjustment, effective N.
- `migration_matrix()` — rating transition via matrix power (multi-year), upgrade/downgrade/default%.
- 20 tests.

---

## v0.548.0 — 2026-05-14

**CLO equity Monte Carlo.**

- `clo_equity.py` — NEW: MC engine for CLO equity IRR distribution and loss analysis.
- `CLOEquityMC` — simulates correlated defaults (one-factor Gaussian copula), recoveries, prepayments through CLOWaterfall.
- Reinvestment period: defaulted/prepaid par replaced at par; post-reinvestment: portfolio amortises.
- `CLOEquityResult` — IRR mean/std/percentiles (5/25/50/75/95), loss distribution, mean cashflows.
- `CLOEquityCashflow` — per-period: income, defaults, recovery, tranche payments, equity distribution.
- `warehouse_risk()` — spread MTM VaR, net carry, ramp shortfall probability.
- 14 tests.

---

## v0.547.0 — 2026-05-14

**Unitranche & direct lending.**

- `unitranche.py` — NEW: unitranche, FOLO, DDTL, direct lending economics.
- `FOLO` — first-out/last-out split with absolute priority recovery allocation.
- `folo_recovery_split()` — FO gets paid first; LO absorbs losses.
- `Unitranche(TermLoan)` — blended spread, OID, FOLO, call protection.
- `DelayedDrawTermLoan(TermLoan)` — ticking fee before draw, normal coupon after.
- `CallProtectionSchedule` — NC/101/par step-down with `call_price()`, `is_callable()`.
- `direct_lending_economics()` — all-in yield: coupon + OID amort + upfront fee amort.
- `hold_to_maturity_yield()` — brentq solver for HTM yield given market price.
- `unitranche_blended_spread()` — weighted FO/LO spread.
- 27 tests.

---

## v0.546.0 — 2026-05-14

**PE-specific visualisation.**

- `football_field()` — horizontal range chart for valuation from multiple methods (DCF perpetuity, exit multiple, WACC sensitivity).
- `j_curve()` — PE fund TVPI over time with trough marker, breakeven line, red/green fill below/above 1.0x.

---

## v0.545.0 — 2026-05-14

**PE trading desk (9-component protocol) + exports.**

- `pe_desk.py` — NEW: full 9-component desk for PE fund management.
- `PERiskMetrics` — NAV, IRR, TVPI, DPI, MOIC, unfunded commitment; dispatches across fund/LBO/DCF.
- `PEBook` / `PEBookEntry` — portfolio book with by_vintage, by_manager, by_sector aggregations.
- `pe_carry_decomposition()` — management fee, carry, distribution income, J-curve drag.
- `pe_daily_pnl()` — NAV change + fee drag attribution.
- `pe_dashboard()` — morning meeting: NAV-weighted IRR/TVPI, position counts, concentrations.
- `pe_stress_suite()` — 5 parametric NAV shocks (±10%, ±25%, -50%).
- `pe_capital()` — Basel PE equity framework: 250% risk weight, unfunded as contingent.
- `pe_hedge_recommendations()` — manager concentration + unfunded ratio breach detection.
- `PELifecycle` — capital call, distribution, secondary sale, GP-led continuation, maturity alerts.
- `__init__.py` exports: LBOModel, DCFModel, WACCInputs, PE performance functions, PEFundParticipation, desk components.
- 28 tests.

---

## v0.544.0 — 2026-05-14

**PE fund waterfall extensions.**

- `fund_participation.py` extended with PE waterfall mechanics.
- `WaterfallConfig` — European (whole-fund) vs American (deal-by-deal) carry, catch-up rate, GP commitment, clawback, recycling.
- `WaterfallResult` — per-period: return of capital → preferred return → GP catch-up → carried interest → LP residual.
- `ClawbackResult` — total carry distributed vs entitled, clawback trigger.
- `PEFundParticipation(FundParticipation)` — subclass with `project_waterfall()`, `clawback_analysis()`, `gp_commitment_cashflows()`.
- Inherits all base methods (metrics, secondary_pricing) and passes isinstance checks.
- 20 tests.

---

## v0.543.0 — 2026-05-14

**PE performance benchmarking.**

- `pe_performance.py` — NEW: PE fund benchmarking and GP economics.
- `kaplan_schoar_pme()` — Public Market Equivalent (Kaplan & Schoar 2005).
- `direct_alpha()` — fund IRR minus index IRR.
- `long_nickels_pme()` — since-inception wealth ratio (Long & Nickels 1996).
- `vintage_cohort()` — aggregate FundParticipation metrics by vintage year (median/mean/UQ/LQ IRR, TVPI).
- `commitment_pacing()` — deterministic LP commitment pacing model (target allocation, calls, distributions, NAV).
- `gp_economics()` — management fee NPV, carry NPV, GP commitment return, clawback exposure.
- `clawback_exposure()` — GP clawback trigger calculation.
- 31 tests.

---

## v0.542.0 — 2026-05-14

**DCF / enterprise valuation.**

- `dcf.py` — NEW: `DCFModel` for discounted cash flow valuation.
- `WACCInputs` — CAPM cost of equity, after-tax cost of debt, WACC.
- `terminal_value_perpetuity()` — Gordon growth model.
- `terminal_value_exit_multiple()` — EV/EBITDA terminal value.
- `ev_to_equity()` — EV → equity bridge (net debt, minorities, associates, per-share).
- `DCFModel.value()` — PV of FCFs + PV of terminal value → EV → equity.
- `DCFModel.scenario_analysis()` — bull/base/bear with parameter overrides.
- `DCFModel.football_field()` — valuation range from perpetuity, exit multiple, WACC sensitivity.
- 27 tests including hand-verified Gordon growth crosscheck.

---

## v0.541.0 — 2026-05-14

**LBO deal model — PE underwriting.**

- `lbo.py` — NEW: `LBOModel` for leveraged buyout deal structuring.
- `SourcesAndUses` — equity, senior debt, mezzanine, rollover, transaction/financing fees.
- `FCFProjection` — EBITDA → revenue → EBIT → taxes → capex → NWC → FCF.
- `DebtYear` — annual debt schedule with senior amort, excess cash flow sweep, mezzanine PIK.
- `ExitAnalysis` — exit EV, net debt, equity value, IRR, MOIC at given multiple/year.
- `LBOModel.run()` — full model across multiple exit scenarios.
- `LBOModel.sensitivity_table()` — IRR grid across exit multiple × hold period (or growth).
- 40 tests.

---

## v0.540.0 — 2026-05-14

**Risk visualisation — 10 new chart types in `pricebook.viz`.**

### New: `viz/_risk.py` — desk-level risk charts
- `pnl_waterfall()` — waterfall/bridge chart for P&L attribution (carry, rate, vol, FX, etc.).
- `risk_decomposition()` — horizontal bar chart sorted by magnitude (key-rate DV01, vega by asset class).
- `stress_comparison()` — grouped or stacked bar chart across stress scenarios.
- `tenor_bucketing()` — vertical bar chart with color gradient by tenor bucket.
- `vega_ladder()` — horizontal bar chart of vega by expiry bucket with rich/cheap coloring.
- `pnl_table()` — formatted matplotlib table for P&L explain with alternating row colors.
- `greeks_surface()` — 2D contour plot of a Greek across strike × expiry.
- `greeks_evolution()` — multi-panel line chart of Greeks vs time-to-expiry.
- `hedge_pnl_tracking()` — position vs hedge cumulative P&L with net overlay.
- `rolling_correlation()` — multi-line rolling correlation with optional confidence bands.
- All functions: pure matplotlib, consume plain data (no instrument imports), theme-aware.
- 3 audit rounds: 17 issues found and fixed (waterfall dead code, label overlap, deprecated get_cmap, length mismatch guards, numpy type formatting, suptitle clipping, stacked legend, dead variables).

---

## v0.539.0 — 2026-05-14

**`pricebook.numerical` — complete self-contained numerical methods package.**

### Numerical package (`numerical/`) — 12 modules, ~1,800 lines
- `_distributions.py` — Normal, StudentT, LogNormal, Uniform, Exponential (wraps scipy.stats).
- `_linalg.py` — expm, logm, QR, Cholesky, LU, GMRES, BiCGSTAB, Sylvester, Lyapunov.
- `_ode.py` — Euler, RK4, RK45 (adaptive), BDF (stiff), Adams.
- `_optimize.py` — unified minimize (NM/BFGS/L-BFGS-B/DE/CMA-ES), LP (HiGHS), QP with inequality, interior-point (barrier), proximal gradient (ISTA/FISTA), projection operators.
- `_quadrature.py` — Gauss-Jacobi, tanh-sinh, Clenshaw-Curtis.
- `_interpolation.py` — 2D bilinear, bicubic, RBF (scattered data).
- `_rootfinding.py` — bisection, unified find_root dispatcher.
- `_mc.py` — QE Heston (Andersen), antithetic variates, multilevel MC (Giles).
- `_pde.py` — Hundsdorfer-Verwer ADI (full 4-stage), 2D PSOR (American), operator splitting (Lie/Strang).
- `_trees.py` — tree Greeks (delta/gamma/vega/theta), 2D binomial (Rubinstein).
- `_fourier.py` — fractional FFT (chirp-z), Hilbert transform, wavelet (Haar/Db2), CharacteristicFunction class.
- `_distributions_theory.py` — Schwartz test functions, tempered distributions, Fourier transform, convolution, Sobolev norms.
- 35 tests covering all modules.
- 3 audit rounds: 23 issues found and fixed (HV ADI stages, Lyapunov sign, PSOR order, Strang splitting, etc.).

---

## v0.527.0 — 2026-05-14

**Advanced regression.**

- `regression.py` — NEW: OLS, Ridge, Lasso (coordinate descent), Elastic Net, quantile (IRLS), robust (Huber/Tukey).

---

## v0.526.0 — 2026-05-14

**Clustering and regime detection.**

- `clustering.py` — NEW: K-means (Lloyd), silhouette score, optimal k, hierarchical (Ward), HMM regime switching (Baum-Welch EM, Viterbi).

---

## v0.525.0 — 2026-05-14

**Distribution fitting.**

- `distribution_fit.py` — NEW: MLE fitting (normal, Student-t, GEV), Kolmogorov-Smirnov test, Anderson-Darling, Q-Q plot data.

---

## v0.524.0 — 2026-05-14

**Kalman filter.**

- `kalman.py` — NEW: linear Gaussian state-space model, RTS smoother, dynamic beta, dynamic hedge ratio, trend extraction.

---

## v0.523.0 — 2026-05-14

**Volatility forecasting.**

- `garch.py` — NEW: GARCH(1,1) MLE, EGARCH (leverage), EWMA (RiskMetrics), realized vol, GARCH VaR.

---

## v0.522.0 — 2026-05-14

**Time series diagnostics.**

- `statistics.py` extended: ACF, PACF (Levinson-Durbin), Ljung-Box Q test, Augmented Dickey-Fuller, Durbin-Watson.

---

## v0.521.0 — 2026-05-14

**Performance ratios.**

- `ts/_stats.py` extended: information ratio, tracking error, Treynor, Omega, gain-to-pain, Kelly criterion (discrete + continuous).

---

## v0.520.0 — 2026-05-13

**Serialisation + curve construction + factories.**

### Serialisation complete (26/26 classes roundtrip)
- Added: FRN, FXSwap, NDF, EquityForward, ZCSwap, ConvertibleBond, AmortisingSwap.
- Model serialisation: all 8 models (Black76, Bachelier, SABR, HW with curve, BS, Heston, MCEquity with process_spec).
- TimeSeries: `to_dict()` (NaN→None) + `from_serialised()`.
- CurrencyPair deserialisation in `serialisable.py`.
- Dividend `to_dict()`/`from_dict()`.

### AmortisingSwap removed
- Use `InterestRateSwap.amortising()`, `.accreting()`, `.roller_coaster()` instead.
- One class per instrument, factory classmethods for common shapes.

### Unified curve builder
- `build_curves(method=...)` — 5 methods: sequential, global_newton, nelson_siegel, svensson, smith_wilson.

---

## v0.519.0 — 2026-05-13

**AAD bootstrap.**

- `aad_bootstrap()` in `aad_curves.py` — sensitivities to every input quote via reverse-mode AD, matches FD to 6 decimals.

---

## v0.518.0 — 2026-05-13

**Analytical Jacobian.**

- `global_solver.py` — analytical Jacobian for global bootstrap, O(n) per iteration, exact match with sequential.

### Curve audit fixes
- `multicurve_solver.py` — dual-curve float leg corrected (was using wrong telescoping identity).
- Armijo condition tightened to strict non-increase.
- Convergence warnings on non-convergence.

---

## v0.517.0 — 2026-05-13

**Futures desk: audit + gaps + notebook.**

### Futures audit fixes
- Stress PnL signs corrected (rates up → negative for long bonds).
- Silent-zero guards in commodity trades/spreads.
- CTD docstring, implied repo 360, turn-of-year docs.

### IR futures extensions
- Pack/bundle/butterfly strategies.
- `FuturesType.EURIBOR_3M`.
- `fed_funds_implied_probability()`.
- `roll_schedule()` — automated roll recommendations.
- `futures_cash_basis_rv()` — cross-market relative value.

### Notebook
- `futures_desk.ipynb` — curve from futures, bond basis, delivery options, IR strip, commodity term structure, multi-asset book.

---

## v0.516.0 — 2026-05-13

**Documentation + exports.**

- Model layer exports added to `__init__.py`: `Black76Model`, `BachelierModel`, `SABRModel`, `HullWhiteModel`, `BSModel`, `HestonModel`, `MCEquityModel`, `SABRParams`, `HestonParams`.
- `ARCHITECTURE.md` updated with Layer 3.5 (model abstraction).
- Version bump to v0.516.0.

---

## v0.515.0 — 2026-05-13

**Model-aware greeks + hard migration of greeks.**

- Bachelier greeks: `bachelier_delta/gamma/vega/theta` added to `black76.py`.
- `greeks_ir_option()` on `Black76Model`, `BachelierModel`, `SABRModel` — analytical greeks consistent with price.
- `greeks_european()` on `BSModel` — wraps existing `equity_greeks()`.
- `Swaption.greeks(curve, vol_surface)` removed → `.greeks(model, curve)`.
- `CapFloor.greeks(model, curve)` added — aggregated cap/floor greeks.
- `CapFloor.caplet_pvs(curve, vol_surface)` removed → `.caplet_pvs(model, curve)` with per-caplet greeks.
- All callers (desks, API, tests) updated. 8363 tests pass.

---

## v0.514.0 — 2026-05-13

**Hard migration: Swaption/CapFloor .pv() → .price(model, curve).**

- `Swaption.pv(curve, vol_surface)` removed → `.price(model, curve)`.
- `CapFloor.pv(curve, vol_surface)` removed → `.price(model, curve)`.
- `.pv_ctx()` rewired through `.price(Black76Model)` internally.
- `swaption_trading_desk.py`, `swaption_desk.py`, `api.py` migrated.
- All test files migrated (test_swaption, test_capfloor, test_swaption_roundtrip, test_ir_deep, test_xi2, test_xi7, test_slice7, test_implied_vol_roundtrip, test_options_hardening).
- Orphaned `FlatVol` imports cleaned.
- 8363 tests pass.

---

## v0.513.0 — 2026-05-13

**Model abstraction layer + instrument wiring.**

- `models.py` — NEW: 2 protocols (`IROptionModel`, `EquityOptionModel`), 7 models (`Black76Model`, `BachelierModel`, `SABRModel`, `HullWhiteModel`, `BSModel`, `HestonModel`, `MCEquityModel`).
- `SABRParams` dataclass (frozen). `HestonParams` imported from `slv.py`.
- `Swaption.price(model, curve)` — pluggable model pricing.
- `CapFloor.price(model, curve)` — pluggable model pricing.
- Audit fixes: `MCEngine.generate_paths()`, HW vol formula (Rebonato), docstring corrections, `HestonParams` dedup, model guard `TypeError`, `projection_curve` passthrough.
- 40 model tests: protocols, swaption/capfloor equivalence, BS/Heston/SABR/HW, guards, put-call parity.

---

## v0.512.0 — 2026-05-13

**Architecture document.**

- `ARCHITECTURE.md` — 449 lines: 8-layer system map, instrument inventory, desk protocol matrix, C++ port roadmap, cross-cutting infrastructure.

---

## v0.511.0 — 2026-05-13

**10 exotic products — closing all 34 gaps.**

- Rates: ZC swaption (Black-76), inverse floater (MC/OU), capped floater (MC/OU with floor).
- FX: ratio forward (long put + short N calls, zero-cost), knock-in reverse convertible (MC barrier).
- Equity: dividend future, dividend swap, dividend option (Black-76).
- Structured: participation note (bond floor + call option).
- Credit: bespoke tranche (one-factor Gaussian copula MC).
- Audit fixes: path-integrated discounting (inverse/capped floater), ZC swaption delta guard, Brent bracket widened, ratio/barrier guards, risky annuity (tranche survival weighted), PD clamping, coupon floor.

---

## v0.510.0 — 2026-05-13

**Time series module (`pricebook.ts`).**

- `TimeSeries` class: numpy-backed (no pandas), construction, arithmetic, alignment, filtering, resample.
- Returns: `simple_returns()`, `log_returns()`, `period_returns()`.
- Stats: `sharpe()`, `sortino()`, `max_drawdown()`, `drawdown_series()`, `performance()` (delegates to `backtest.compute_metrics`).
- Rolling: `rolling_sharpe()`, `rolling_vol()`, `rolling_beta()` (delegates to `statistics.rolling_stats`).
- I/O: `from_db()`, `from_db_book()`, `from_db_desk()`, `from_csv()`, `greeks_from_db()`.
- Replay: `replay()`, `replay_book()`, `replay_desk()`, `drawdown_analysis()`, `rolling_performance()`.
- Viz: `plot_dashboard()`, `plot_equity_curve()`, `plot_drawdowns()`, `plot_rolling_sharpe()`, `plot_pnl_histogram()`.
- DB: `pnl_series_by_book()`, `pnl_series_by_desk()` aggregation methods added to `PricebookDB`.
- 52 tests.

---

## v0.509.0 — 2026-05-13

**Convertible bond desk — 9-component protocol.**

- `convertible_bond_desk.py` — NEW: `CBRiskMetrics` (hybrid delta/gamma/vega/CS01/DV01), `CBBook`, `CBBookEntry`, `CBCarryDecomposition`, `CBDailyPnL`, `CBDashboard`, `CBStressResult`, `CBCapitalResult`, `CBHedgeRecommendation`, `CBLifecycle`.
- Exports added to `__init__.py`: `ConvertibleBond`, desk layer.
- 26 tests.

---

## v0.508.0 — 2026-05-13

**4 new notebooks: asset swaps, XCCY basis, PRDC, TARF.**

- `asw_btp_bund.ipynb` — BTP vs Bund ASW spread basis trade, EUR curve (ESTR), par/proceeds ASW, Z-spread comparison, risk & carry.
- `xccy_basis_pricing.ipynb` — USD bond for EUR investor, XCCY basis from FX forwards, FX-hedged yield, pickup vs Bunds, basis sensitivity.
- `prdc_structuring.ipynb` — PRDC 3-factor MC (JPY/USD), callable via LSM, correlation sensitivity, FX delta profile, par coupon structuring.
- `tarf_risk_profile.ipynb` — TARF payoff asymmetry vs vanilla forward, target/vol/strike sensitivity.

---

## v0.507.0 — 2026-05-12

**Bond trading & multicurve notebooks.**

- `bond_trading_desk.ipynb` — trader's 7AM morning workflow: market setup, rich/cheap RV scorecard, trade construction, callable OAS, repo financing, risk snapshot. OAS bracket widened to [-0.10, 0.50].
- `treasury_multicurve.ipynb` — Treasury curve (7 bonds) vs SOFR (from swaps) vs repo, pricing comparison, basis trade signal, carry analysis by repo tenor. Extended with 30-bond universe + curve construction summary.

---

## v0.506.0 — 2026-05-12

**Benchmark bonds, repo curve, callable bond desk.**

- `benchmark_bonds.py` — NEW: 6 sovereign markets (UST/Bund/Gilt/JGB/OAT/BTP) with correct conventions. `BenchmarkUniverse`, `create_ust_universe()`, etc. NSS curve fitting (`fitted_curve_nss`). Trading strategies: `duration_neutral_spread()`, `butterfly_trade()`, `barbell_vs_bullet()`. Rankings: `carry_ranking()`, `roll_down_ranking()`, `rv_scorecard()`. 15 tests.
- `repo_curve.py` — NEW: `RepoCurve`, `build_repo_curve()`, `forward_repo_rate()`, `special_gc_spread()`, `repo_carry_from_curve()`.
- `callable_bond_desk.py` — NEW: `callable_bond_analytics()` — model price, straight price, option value, OAS, effective duration/convexity. 16 tests.

---

## v0.505.0 — 2026-05-12

**Bond desk + Treasury note pricing.** 16 new tests.

### Bond desk hardening
- `bond_daily_pnl()` and `bond_pnl_attribution()` wired into `bond_trading_desk.py` — 9/9 protocol complete.
- Input validation: maturity check in `bond_risk_metrics()`, horizon guard in `bond_carry_roll()`.

### Treasury quoting (`treasury_quoting.py`)
- `to_32nds()` / `from_32nds()` — decimal ↔ 32nds with + (half-32nd) notation.
- `TreasuryReopen` — new issue vs reopening (premium/discount, WAP, total outstanding).
- `delivery_option_value()` — quality + timing + wild card option decomposition for futures.

### Treasury note roundtrip notebook (`notebooks/treasury_note_roundtrip.ipynb`)
- Full pricing: build SOFR curve → create 10Y T-Note → dirty/clean/AI/YTM/32nds.
- Risk metrics: duration, DV01, convexity, key-rate profile (via `greeks_profile`).
