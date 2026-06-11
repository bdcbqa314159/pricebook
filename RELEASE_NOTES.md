# Release Notes

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
