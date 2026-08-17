# Changelog

All notable changes to the new tree (`pricebook_ng`) are recorded here, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. `0.x` = migration
in progress; `1.0.0` is reached exactly when the quarry (`python/pricebook/`) is empty.

## [Unreleased]

## [0.94.0] - 2026-08-18

**C2 slice 2 — Black European swaption (annuity numeraire).** The clean §3d payoff: numeraire = the
annuity `rpv01`, underlying = the forward swap rate `S = float_leg_pv/rpv01` — the SAME shared atoms the
swap calibrator/engine compose. `black()` reused verbatim; only the vol's meaning changes. Second capability
on one model — the capability-model rule-of-two now has its multi-capability case. Red→green; ticks 0.

### Added
- **`SwaptionVol`** capability (L3) — a SIBLING of `BlackVol` (not a reuse) with its Q3′(a) contract: the
  lognormal vol of the forward SWAP RATE under the **annuity (swap) measure** (numeraire `rpv01`), distinct
  from `BlackVol`'s index-forward/T-forward measure. `BlackModel` now satisfies `CalibratedModel` + `BlackVol`
  + `SwaptionVol` — **`BlackVol`/`BlackModel` meaning unchanged** (additive, doc 22 Q1 opt-in capabilities).
- **`SwaptionSurfaceKey(index, swap_tenor)`** (L1) — keyed by index AND swap tenor, never colliding with the
  optionlet `SurfaceKey(index)`. Reuses the flat-minimal `Surface`; `MarketSnapshot.surfaces` key type widened
  to `SurfaceKey | SwaptionSurfaceKey` (a union over one shape, not a new field).
- **`Swaption(swap, expiry, option_type)`** (L2) — reuses `VanillaSwap` (strike = its fixed rate; payer=CALL,
  receiver=PUT). **`price_swaption`** (L4) composes the shared `rpv01`/`float_leg_pv` atoms + `black()`,
  validates `SwaptionVol` structurally, and **never touches the calibrator's private `_par_rate`**.

### Oracle
- Reprices to `N·rpv01·black(S,K,vol,t)` vs an independent inline Black to **<1e-12**; payer−receiver parity
  `N·rpv01·(S−K)`; vol→0 intrinsic; and the **§3d identity** — swaption(payer−receiver) equals the swap
  engine's PV of the same swap (<1e-9), proving engine and calibrator share the `S`/annuity composition.

### Deferred (named triggers)
Cash-settled/IRR swaptions · swaption vol cube + 2D smile · vol calibration/stripping · Bermudan ((A)-fork) ·
SABR/HW swaption vols (B4) · midcurve swaptions · swaption greeks (C3).

Drawdown 19/793 (partial cross of `options/swaption.py`, tick 0).

## [0.93.0] - 2026-08-14

**C2 slice 1 — Black-76 European caplet (opens the models & calibration cluster).** The first dynamics
model, the first capability beyond `Discounting`, the first `surfaces` snapshot shape — the first real
rule-of-two test of doc 22's capability model. Red→green; ticks 0 quarry modules (partial crosses).

### Changed
- **Engine rebased onto the `CalibratedModel` protocol** (refactor under green). `CalibratedModel` + `BlackVol`
  were designed in doc 22 but code-deferred to the second model; they land now. The registry and every pricer
  depend on the capability (`model.market`), not concrete `DiscountingModel` — no `isinstance`, no change to
  any ratified type's meaning. The slice-1..6c linear suite stayed green through the rebase.
- **`MarketSnapshot` gains the `surfaces` shape** (`SurfaceKey`/`Surface`, keyed by index — doc 19 §2, first
  vol consumer). Flat-minimal surface; grid + 2D smile interpolation deferred.

### Added
- **`BlackModel`** (L3) — second `CalibratedModel`, built-not-solved (reads vol from `market.surfaces`); adds
  the **`BlackVol`** capability with its Q3′(a) semantic contract (lognormal, T-forward measure, annualized).
- **`black()`** (L3) — undiscounted Black-76 closed form (df kept out, ≤5 args; at L3 not L0 per the
  `black.py` precedent; uses the `norm_cdf` foundation adapter).
- **`Caplet`** + **`OptionType`** (L2) and **`price_caplet`** (L4) — `PV = df(pay)·N·τ·black(F,K,vol,t,CALL)`;
  the model's `BlackVol` capability is validated structurally (runtime-checkable protocol), never `isinstance`
  on a concrete type.

### Oracle
- Caplet reprices to an **independent** Black-76 evaluation (inline `erf`, distinct from the scipy adapter) to
  **<1e-12**; put-call parity `caplet − floorlet = df·N·τ·(F−K)` to <1e-12; vol→0 = discounted intrinsic.

### Deferred (named triggers)
Bachelier/normal vol (2nd vol consumer) · vol calibration / surface stripping · swaption + annuity numeraire
(B5) · 2D smile interpolation · engine numerics-config (invariant 5 — name it distinctly from `SolveConfig`) ·
Bermudan ((A)-fork trigger) · SABR/HW (B4). Doc fix: rename doc 22's `SolverConfig` → `SolveConfig`.

Drawdown 19/793 (partial crosses of `black76.py` / `capfloor.py`, tick 0).

## [0.92.2] - 2026-08-12

**Slice 6d — audit response.** Fixes all 8 findings from a third-party audit of v0.92.0
(`analysis/AUDIT_FINDINGS.md`). The numerical core was verified clean; exposure was at input/range
boundaries (oracles had tested positive-rate / in-range / well-formed inputs only). Each fix is
red→green (the oracle that exposes the finding, then the fix). Ticks 0 quarry modules.

### Fixed — correctness (crash/hang/wrong)
- **#1 [HIGH] Negative-rate sequential calibration.** `_bootstrap`/`_bootstrap_xccy` used a fixed
  `[1e-6, 1.0]` bracket that baked in positive rates (a negative-rate pillar has DF > 1) and let
  `brentq`'s "different signs" `ValueError` escape. New `_solve_pillar_df` expands the upper bound
  until the residual changes sign (capped); a non-bracketable/failed solve returns
  **`CalibrationFailure`**, never raises — **invariant 4 restored** across the sequential path.
- **#2 [MED] Hagan–West extrapolation.** `MonotoneConvex.value`/`integral` now guard their domain
  `[knots[0], knots[-1]]`; the curve's HAGAN_WEST `df()` **raises** past the last pillar, consistent
  with the log-linear RAISE policy (was a silent flat DF).
- **#3 [MED] ICMA frequency guard.** `_act_act_icma` now requires `1 ≤ frequency ≤ 12 and 12 % f == 0`
  — closes a `frequency > 12` **infinite loop** and a non-divisor silently-wrong DCF.
- **#4 [MED] RegularPeriod anchors.** `_unadjusted` validates `first_regular < last_regular` up front —
  coincident/reversed anchors raise a clear schedule error instead of a silent zero-length period.

### Fixed — robustness (mispriced/degraded quietly)
- **#5 [LOW]** RegularPeriod's short final period now reports `is_stub=True` (non-dividing tenor).
- **#6 [LOW]** `WeekendSchedule` sorts its transitions in `__post_init__` — `on(year)` is order-independent.
- **#7 [LOW]** Convergence is graded at `spec.solve.tolerance`, not a literal `1e-10`.
- **#8 [LOW/perf]** The Hagan–West reconstruction is a `cached_property` — a full-curve reval is O(n)
  not O(n²); the frozen curve makes the cache honest (`cached == uncached`).

### Tests
- Boundary oracles added at L0/L1/L3 (`tests_ng/L*/test_audit_response_l*.py`). Full suite green (207).

## [0.92.1] - 2026-08-09

**Docs/tracker reconciliation — honest baseline before the audit-fix slice.** No `src` behaviour change.

### Migration tracker
- **`multicurve_solver.py` full-cross ruled (Cowork):** the slice-4 spot-check is resolved — §4 grep found
  0 production/dynamic consumers of `multicurve_newton`/`validate_curve`/`curve_analytical_jacobian` (only
  quarry-internal tests, which retire with the quarry). 3 ticks stand. `validate_curve`/`curve_analytical_jacobian`
  forward-linked → C3 risk.
- **`forward_interpolation.py` retire-read done → full cross (+1 → 19/793):** `MONOTONE_CONVEX` → ng
  `HAGAN_WEST`, `PIECEWISE_CONSTANT` → ng `LOG_LINEAR`, `PIECEWISE_LINEAR` → shed (`deferred→` future
  interp-method consumer); architecture superseded by ng's DF-interp + `forward()` atom. **Drawdown now
  19/793** (13 Topic-0 parked + 6 Topic-1 crossed).

### Design artifacts
- 12 design docs flipped **Draft → Ratified** (redesign/01,02,05,06,07,08,09,10,11,12,13,20); stale
  "(pending ratification)" cleared at 05.

### Ledger (`OPEN.md`, §5)
- Added the Topic-1/C1 carried-debt section (5 non-balance entries, each with a named re-open trigger):
  `forward()` subtract-first hardening; collateral→L6 relocation; simultaneous xccy solve; FX triangulation;
  analytic/AAD Jacobian.
- **AC-3.6b** updated: the FX pair-conventions registry **landed** at v0.90.0 (slice 6a); only the
  asymmetric-ACI intermediate-day settlement rule remains deferred.

### Pointers
- C1 checkpoint next-step: an **audit-fix slice lands before C2 opens**. Fixed 768→793 (doc 11) and the
  moved `redesign/archive/16_topic0_foundation.md` link (topic-00 MANIFEST).

## [0.92.0] - 2026-08-08

**Topic 1, slice 6c (C1 CLOSE) — xccy basis curve via the unified calibration front.** Two steps:
a refactor-under-green to a multi-currency `CalibrationSpec`, then the xccy basis curve on top.
Closes cluster C1 (single-curve → dual-curve → cash → global solve → Hagan–West → CSA/xccy).

### Changed
- **`CalibrationSpec` reshaped to the unified multi-currency front** (§8 refactor-under-green):
  `(valuation_date, curves: tuple[CurrencyCurves], solve, xccy, fx)` — 5 fields. `single_currency()`
  classmethod + `.currency`/`.discount`/`.projection` accessors keep slices 1–5 unchanged (1-tuple =
  degenerate). `calibrate()` loops currencies into ONE `CurveSet` on ONE model. No new behaviour; the
  full existing oracle set stayed green through the new shape.

### Added
- **`XccyBasisSwap`** (L2) — constant-notional cross-currency basis swap: domestic OIS-flat leg vs a
  foreign leg + basis, notional exchanged at spot, under a `collateral` CSA.
- **`XccyBasisQuote`/`XccyBuild`** and **`XccyBasisInstrument`** — the foreign-collateral (xccy-basis)
  curve is bootstrapped as a SEQUENTIAL post-step (domestic OIS → foreign OIS → foreign-in-collateral
  last), keyed `(DISCOUNT, foreign, collateral)`. The instrument and the new **`price_xccy`** engine
  pricer compose the SAME `float_leg_pv`/`rpv01` atoms (§3d).
- `test_collateral`'s foreign case now uses a **real calibration**, not a manual curve injection.

### Oracle
- **Reprice-to-zero:** every calibrating xccy basis swap prices to ~0 through the L4 engine (zero and
  10bp basis).
- **CIP closed-form anchor:** at zero basis the curve reproduces `df_foreign` (foreign OIS), so the FX
  forward `F = S·df_foreign^coll/df_domestic` equals the textbook `S·df_foreign/df_domestic` to **1e-10**;
  a real basis moves the curve below CIP.

### Note (design finding)
The constant-notional both-OIS-flat xccy basis swap has the **FX spot and USD discount cancel out of the
reprice-to-zero condition** (the domestic leg telescopes to par; `N_foreign = N_domestic/S`). They enter
only the CIP FX-forward oracle, not the bootstrap — the curve is pinned by `df_foreign` + the basis alone.

### Deferred (named triggers)
Simultaneous/joint xccy solve (xccy pillars in the global state vector) → 2nd xccy consumer; consequently
the sequential==simultaneous degeneracy does **not** cover xccy. Triangulation/3rd ccy, FX vol/options,
tenor-basis, MtM-notional resets, NDFs, collateral optionality → their asset topics. Collateral/CSA
relocates from the L2 product to the L6 trade layer when L6 lands.

Drawdown unchanged (18/793): the quarry xccy/CSA/FX suite is full FX/XVA breadth, not superseded by a
minimal foreign-collateral curve — it retires with the FX/XVA topics (0 ticks, ratified).

## [0.91.0] - 2026-08-08

**Topic 1, slice 6b (C1 close, 2/3) — CSA collateral-keyed discounting.** Activates the
`discount(ccy, collateral)` hook stubbed since slice 2. Red→green.

### Changed
- **`CurveKey` grows `collateral: Currency | None`** (3 fields) — the discount curve is now keyed by
  currency AND CSA collateral. Existing keys default `collateral=None`, so slices 1–5 are unchanged.
- **`CurveSet.discount(ccy, collateral)`** normalizes own-currency collateral (`None` or `== ccy`) to the
  domestic OIS curve; a foreign collateral selects its keyed curve (the xccy curve lands in 6c).
- **`VanillaSwap.collateral: Currency | None`** (5 fields). The engine resolves `discount(ccy, collateral)`
  through `model.market` (A1) and records **`PricingResult.basis`** (`None` for own-currency, the
  collateral currency otherwise). *(Relocation trigger recorded: collateral/CSA moves to the L6 trade
  layer when L6 lands.)*

### Oracle
- **Degeneracy:** a EUR swap collateralized in EUR prices identically to `collateral=None` (domestic OIS)
  to 1e-12; `basis` is `None` for both. All slices 1–5 reprice unchanged through the new key path.
- A foreign-collateral (USD) swap resolves its keyed curve, records `basis == USD`, and differs from the
  domestic price.

Drawdown unchanged (18/793).

## [0.90.0] - 2026-08-05

**Topic 1, slice 6a (C1 close, 1/3) — FX spot in the snapshot.** First step of CSA/xccy: the directional
FX spot. Red→green.

### Added
- **`MarketSnapshot.scalars`** — the first non-curve shape (doc 19 closed-shapes × open-keys), keyed by
  `ScalarKey(pair)`; FX spots stored in the declared canonical quote order. Existing snapshots default
  `scalars` empty (slices 1–5 unaffected); `MarketSnapshot` stays 3 fields.
- **`MarketSnapshot.fx_rate(base, quote)`** — resolves the declared canonical pair, reads the spot,
  inverts on the reverse direction; **raises on an undeclared cross** — never a bare pair-scalar (doc 19
  §2.1, fail loud).
- **`foundation.register_fx_pair` / `fx_pair`** — the FX pair-conventions registry (EUR/USD declared, EUR
  base). Completes **AC-3.6b's registry half**; the asymmetric-ACI FX-date rule stays deferred.

### Oracle
- `fx_rate(A,B)·fx_rate(B,A) == 1` to 1e-15; canonical direction returns the stored value, the reverse
  its reciprocal; an undeclared cross (GBP/JPY) raises; duplicate / same-currency registration raise.

Drawdown unchanged (18/793) — FX-conventions plumbing, no quarry module crossed (the FX suite is
reassigned to the FX/XVA topic).

## [0.89.0] - 2026-08-05

**Topic 1, slice 5 — Hagan–West monotone-convex forward interpolation.** Owns what scipy lacks
(§7bb); the first EXTERNAL-oracle slice in the topic (the paper's equations, since it publishes the
example only as figures). Red→green throughout.

### Added
- **`foundation/hagan_west.py` (L0, finance-free)** — the monotone-convex reconstruction from
  interval averages (Hagan & West AMF 2006): knot estimates eq 30–32, four-region construction eq
  47/49–56, optional positivity clamp eq 60–62. Parameters are `knots` + `averages` + a `positive`
  flag — **no forwards/DFs/rates/curves** (the falsification gate held: pure math). It is a distinct
  primitive, not a mode of the point-based `interpolate()`. `monotone_convex(...)` → `MonotoneConvex`
  with `value(x)` and `integral(x)`.
- **`Interpolation.HAGAN_WEST`** — a MODE TAG; the point-based `interpolate()` raises
  `NotImplementedError` on it (it reconstructs from interval integrals, not point values).

### Changed
- **`DiscountCurve.df` dispatches `HAGAN_WEST`** to the forward path: discrete forwards from the
  pillar DFs → the L0 primitive → `df(t) = exp(−∫₀ᵗ f)`, reproducing the pillar DFs exactly. HW is
  **non-local**, so its home is the SIMULTANEOUS solve (slice 4); sequential-HW is deferred (the
  paper's terminal-interval convention) with a named trigger.

### Oracle (equation-anchored — the paper has no value table)
- **eq-33 interval-average reproduction to 1e-14** (the method's defining invariant).
- **Per-region pointwise checks (regions i–iv)** vs values hand-derived from eq 47/49–56 (independent
  of the code) to 1e-12 — catches a wrong-region transcription; boundary conditions `g(0)=g₀,g(1)=g₁`.
- Positivity clamps knots ≥ 0 and keeps interpolated forwards ≥ 0; eq-33 still holds under it.
- **Reprice-to-par** through the L4 engine on HW curves (via the simultaneous solve), worst |PV|
  1.4e-16; **degeneracy** HW == log-linear on constant forwards (df diff 0.0).

### Drawdown — premise correction, ticks 0 (partial-cross candidate flagged)
The tasking said HW was "missing from both trees" — **incorrect**: the quarry has it in
`core/forward_interpolation.py::monotone_convex_forwards` and `curves/curve_advanced.py::
smooth_forward_curve`. ng's HW supersedes the monotone-convex forward *construction*, but
`forward_interpolation.py` is a **multi-method** module (a `ForwardInterpolationMethod` enum), so this
is at most a **partial** cross — not ticked without a full retire-read. Recorded as a candidate for the
next align/Cowork spot-check; drawdown stays 18/793.

### Drawdown — retire read (docs/tracker only, no code)
The slice-3 checkpoint's named next action is done: the end-to-end retire read of `curves/bootstrap.py`
(713 LOC) + `core/discount_curve.py` (300 LOC). **Both cross → deletable 13 → 15 / 793 (first Topic-1
tick).** Evidence: convention/calendar is *imported* from already-parked Topic-0 modules (not resident);
the only `bootstrap.py`-resident deferred bits are futures convexity (`:151-158`) and turn-of-year
(`:161-162`); `discount_curve.py` residents are curve analytics (`zero_rate`/`bumped`/`roll_down`/… ) —
`forward_rate` already crossed by ng's `forward()` atom. 0 ng consumers of any deferred capability (bare-
name grep across src + tests). Forward-links filed on destination rows (models ← convexity, seasonality
← turn-of-year, C3-risk ← curve analytics, persistence ← curve serialisation); per-leg conventions
recorded as an ng-side note (convention code already parked — nothing to migrate). Tracker
(`quarry_reconciliation.md`) + Topic-1 MANIFEST carry the full record. Deletable = *superseded*; physical
park at Topic-1 close.

### Drawdown — slice-4 reconciliation (docs/tracker only, no code)
Closes the slice-4 (v0.88.0) deferred tick: the three quarry global solvers — `curves/ncurve_solver.py`,
`curves/global_solver.py`, `curves/multicurve_solver.py` — cross, superseded by ng's simultaneous solve.
**Deletable 15 → 18 / 793** (Topic-1: 2 → 5). Hand-rolled damped Newton + FD Jacobian §7bb-shed (scipy);
analytic Jacobian → `deferred→C3/AAD`; N-curve > 2 → `deferred→3rd-projection-curve`; 0 ng consumers.
`multicurve_solver` **flagged for Cowork spot-check** (read found 3 crossings, task expected 2). Evidence:
`CP_slice4_global_solve.md`; tracker + MANIFEST updated.

## [0.88.0] - 2026-08-03

**Topic 1, slice 4 — the simultaneous (global) calibration orchestration + par→zero Jacobian.** The
second orchestration (doc 22 Q4): one `CalibrationSpec`, one residual definition, two orchestrations —
the existing sequential bootstrap and a new global N-D solve of the whole `CurveSet` at once, proven
equal on the degenerate case. Red→green throughout.

### Added
- **`CalibrationMethod` {SEQUENTIAL, SIMULTANEOUS}** — earned now (the second orchestration, rule of two).
- **`SolveConfig`** — bundles `method` + solver knobs (tolerance, max_iterations) so `CalibrationSpec`
  stays **5 fields** (§3b); the first tuned-knob consumer earns the numerics config (invariant 5).
- **`Jacobian`** value type — `∂residualᵢ/∂dfⱼ` + pillar/instrument labels; the SIMULTANEOUS solve
  populates `CalibrationResult.jacobian`, the SEQUENTIAL path leaves it `None` (resolving doc 22 Q4's
  deferred branch — the sequential post-hoc Jacobian arrives with its C3-risk consumer).
- **`foundation/solvers.root_nd`** — the N-D root-find (scipy Levenberg-Marquardt) returning
  `(solution, jacobian, converged)`; `least_squares` now delegates to it. §7bb: scipy wrapped behind
  the foundation adapter, no hand-rolled damped Newton, never called from `calibrate`.
- **`CalibrationFailure`** — non-convergence as a value (invariant 4); `calibrate` returns it instead
  of a silent bad curve.
- **`curve_set_residuals(spec, curves)`** — the residual vector both orchestrations drive to zero.

### Changed
- **`calibrate(spec)` forks on `spec.solve.method`** — sequential (unchanged) or the global solve.
  Both compose the SAME `CalibrationInstrument` residuals (§3d); only the orchestration + Jacobian
  provenance differ. The global path assembles one flat state vector of all pillar DFs across both
  curves and solves them jointly (seed: flat 3% DFs, the `ncurve_solver` default).

### Oracle
- **sequential == simultaneous degenerate** (doc 18 §8): pillar DFs match to **~1e-15** (discount
  5.6e-16, projection 1.8e-15).
- Every calibrating instrument reprices to zero NPV through the L4 engine under the globally-solved
  curves (worst |PV| ~3.5e-16).
- **Jacobian validated vs finite-difference** (tangent prediction) to < 1e-6.
- Non-convergence (`max_iterations=1`) → `CalibrationFailure`; global reprice byte-identical.

### Drawdown
Retire-read of the quarry global solvers (see the slice checkpoint): `curves/ncurve_solver.py`,
`curves/global_solver.py`, `curves/multicurve_solver.py` are all superseded by ng's simultaneous solve
(hand-rolled damped Newton + FD Jacobian are §7bb-shed → scipy; analytic Jacobian → `deferred→C3/AAD`;
N-curve > 2 → `deferred→3rd-curve`; 0 ng consumers of any shed bit). **Tick pending #119 merge** (the
bootstrap/discount retire tick is not yet on main — the tracker count reconciles once it lands).

## [0.87.0] - 2026-07-31

**Topic 1, slice 3 — cash instruments: deposits + FRAs + IMM futures on both curves.** Completes the
"cash front" of the dual-curve build and lands two ratified dispatch abstractions, machinery-first as a
green-guarded refactor (Step A) then the instruments (Step B). Red→green throughout.

### Added (Step A — machinery, pure refactor)
- **Engine registry inside L4** (`engine/registry.py`) — structural product→pricer dispatch, no
  `isinstance` (CLAUDE.md §1). The `VanillaSwap` pricer moved behind it unchanged; `price` is the
  registry `dispatch`.
- **`CalibrationInstrument` protocol (L3)** — `residual(discount, projection)`, 0 at solution;
  `SwapInstrument` wraps the existing `_par_rate`. `_bootstrap` now Brent-solves each pillar via the
  instrument's residual. `calibrate` still imports no L4.

### Added (Step B — instruments)
- **L1 quotes** (`market/quotes.py`): `DepositQuote` · `FRAQuote` (spot-anchored when `start=None`) ·
  `FutureQuote` (price; forward = `1 − price`) · `ParSwapQuote` (moved from L3).
- **L2 products** (`products/cash.py`): `Deposit` · `FRA` · `Future` (IMM-dated).
- **`deposit_df` atom** (`1/(1+r·τ)`) composed by BOTH the L3 `DepositInstrument` and the L4 deposit
  pricer (§3d); FRA/Future compose the existing `forward()`. Engine pricers registered per type.
- **Heterogeneous bootstrap**: `CurveBuild.quotes` is now the deposit/FRA/future/swap union; discount
  (ESTR) = OIS deposits + swaps, projection (EURIBOR_3M) = spot-anchored FRAs + IMM futures + par
  swaps, chained from spot (an unpinned `df(start)` surfaces as `PricingFailure`).

### Oracle
- Every calibrating instrument reprices to **zero NPV through the L4 engine**: discount curve worst
  |PV| = 6.7e-16; projection curve worst |PV| = 2.0e-15 (15 instruments, converged).
- **Future reproduces `1 − price` at its IMM segment** (worst |Δ| = 5.4e-16) — the forward
  approximation is *applied* correctly (doc 18 §2), not a claim the futures price is right.
- Regression: slice-1/2 par swaps still price to zero through the registry-dispatched engine.

### Deferred (named triggers)
Futures **convexity** (`forward = futures_rate − convexity(a,σ,t)`, models topic) · Hagan–West ·
global/simultaneous solve + Jacobian · basis/xccy · full G10 conventions.

### Drawdown
Ticks 0 (13/793). §4 consumer analysis (align commit): the "full G10 conventions" gate has 0 ng
consumers (`deferred→G10`), and `discount_curve.py`'s `zero_rate`/`bump`/`roll_down` are consumed only
by uncrossed quarry modules (`deferred→risk/construction`) — no needed-now blocker, but a tick requires
the formal end-to-end retire read of `bootstrap.py`/`discount_curve.py`, named as the immediate next
step at this slice's checkpoint.

### Docs — factual-counter correction (denominator pinned)
The drawdown denominator had two conventions in flight — `/768` (CLAUDE.md §4/§6 + slices 1–2) vs
`/793` (the live tracker). Verified by direct count: **793 = every `.py` in the original quarry**
(`python/pricebook/` remaining, 780, + `parked/`, 13); `768 = 793 − 25 __init__.py` markers — the same
universe, a different convention, not an error. **Pinned to 793**, and `redesign/handoffs/
quarry_reconciliation.md` is now the single source (its convention stated there). CLAUDE.md §4/§6 no
longer hardcode a literal — they point to the tracker, so the count can't re-drift. The tracker's frozen
`PRE-REPLAN RECORD` (`7 / 768`) is left untouched as history. No ruling reopened.

## [0.86.0] - 2026-07-30

**Topic 1, slice 2 — dual-curve: a EURIBOR swap prices to ZERO off ESTR discounting.** The first
genuine second curve: a swap projects off EURIBOR_3M and discounts off ESTR. Proves the multicurve
machinery and that single-curve is its degenerate case. Drawdown unchanged (0 quarry modules ticked;
the align commit records the named C1 slices where `bootstrap.py`/`discount_curve.py` actually cross).
Red→green throughout.

### Added
- **`market/curve_set.py` (L1)** — `CurveSet` with typed accessors `discount(currency, collateral=None)`
  and `projection(index)` over one `CurveKey`-keyed store (doc 19 §2-§3, closed shapes × open keys).
  Ratified signature lands now; `collateral` non-None raises (CSA/xccy deferred); `survival`/`inflation`
  and a multi-projection map arrive with their consumers.
- **`forward()` projection atom (L1, §3d)** — the single definition of the projection forward rate
  `(df(start)/df(end) − 1)/τ`, which `float_leg_pv` now **composes** (it never inlines the df ratio).

### Changed
- **`float_leg_pv` generalised in place** to dual-curve `Σ df_disc·τ·forward(proj, ·)` — the slice-1
  telescoping `df(start₀) − df(endₙ)` is now its degenerate case (`projection is discount`), not a
  separate code path. `rpv01` unchanged (discount only).
- **`FloatLeg` gained `index: RateIndex`** (→3 fields); it selects the projection curve.
- **`MarketSnapshot.discount_curve` → `curves: CurveSet`.**
- **`calibrate(spec)`** — one entry, still sequential; `CalibrationSpec` now a curve-**set** spec
  (`discount` + `projection` builds) bootstrapped in dependency order (discount self-discounting →
  projection discounted on it). `SingleCurveSpec` replaced by `CurveBuild`; `single_curve_swap` →
  `par_swap`. No `method`/`numerics`/Jacobian (rule of two).

### Oracle
- **Dual-curve EURIBOR par swap → zero NPV: worst |PV| = 1.9e-15** (unit notional).
- **OIS degeneracy (slice-1 par swap through the generalised path): worst |PV| = 6.7e-16** — slice 1
  survives.
- **Degeneracy Δ (general float leg vs telescoping identity): 2.8e-17, rel 2.0e-16** (tight tolerance,
  not bit-identity — the df ratio is not bit-exact).
- DV01 analytic vs finite-difference to 1e-6; repricing byte-identical; beyond-curve → `PricingFailure`.

## [0.85.0] - 2026-07-29

**Topic 1, slice 1 — the first pricing vertical: a par swap prices to ZERO NPV, L0→L4.** First
consumer of F1 (redesign/19) and F2 (redesign/22 rev 3), proving both foundations by pricing a real
swap. Drawdown is unchanged (0 quarry modules ticked deletable yet — parking is one event at Topic-1
close; this slice builds the ng counterparts the tick will later require). Red→green throughout.

### Added
- **`TimeMeasure` (L0)** — the one sanctioned `date → year-fraction` map (anchor + day count, ruling
  A1); nothing re-derives it.
- **`market/` (L1)** — `DiscountCurve` (log-linear discount factors, exact `exp(-r·t)` on a flat
  curve) behind the `CurveHandle` capability; `MarketSnapshot` (valuation date + discount curve,
  frozen, A1); and the **shared `rpv01` / `float_leg_pv` building-block atoms (CLAUDE.md §3d)** the
  calibrator and the engine both compose — one annuity definition, no drift.
- **`products/swap.py` (L2)** — `VanillaSwap` = `FixedLeg` + `FloatLeg`, pure data (≤5 fields each).
- **`models/discounting_model.py` (L3)** — `DiscountingModel`, the degenerate model built (not solved)
  from a curve, carrying its `MarketSnapshot` (A1).
- **`calibration/calibrate.py` (L3)** — `calibrate(spec) → (model, result)`, the sequential
  single-curve bootstrap; the model is the output, the residual is formed through the L1 building
  blocks (never L4, doc 22 Q2). `CalibrationSpec`/`CalibrationResult`/`ParSwapQuote`/`SingleCurveSpec`.
- **`engine/linear.py` (L4)** — `price(swap, model)`, stateless, reaching the market only through
  `model.market` (A1); a cashflow beyond the curve returns a `PricingFailure` (failure is a value).

### Oracle
- **Par swap reprices to zero NPV through the L4 engine: worst |PV| = 5.6e-16** (unit notional; doc 18
  C2 / doc 22 §Q2 backstop). Bootstrap converged, max|residual| = 5.8e-16.
- Flat-curve `df(t) = exp(-r·t)` to 1e-12 (the single-cashflow discounting anchor).
- DV01 analytic (−N·annuity) vs central finite-difference: |diff| = 1.7e-12 (< 1e-6).
- Repricing byte-identical (statelessness).

### Docs
- Ratified F1 (redesign/19 rev 3), F2 (redesign/22 rev 3), **CLAUDE.md §3d — Building-block
  discipline**; 8 superseded design docs moved to `redesign/archive/`.

## [0.84.0] - 2026-07-20

Standing rule (Bernardo): **no deferred item before market-data (F1) unless a downstream topic genuinely
owns and will shape it — "waits on an event" is not a topic.** Applied to the `AC-*` ledger: **22 → 15**
deferrals. Red→green throughout.

### Fixed
- **AC-T4.7 — `is_holiday` year−1 spill.** A December holiday observed *forward* into January (e.g. 31 Dec
  Sunday → 1 Jan) was missed; `is_holiday` now checks `year−1` as well as `year`/`year+1`.
- **AC-T4.8 — `observe()` parameterised off the weekend rule.** Was hardcoded Sat/Sun, so a FRI_SAT
  market under a mondayising regime left a Friday holiday unshifted; substitution now derives from the
  calendar's own weekend days (no change for SAT_SUN calendars).
- **AC-T4.9 — `NEAREST` tie-break RULED: roll forward** (QuantLib / common market practice), documented
  in the convention and tested.
- **AC-T4.11 — `growth_factor` / `convert_rate` reject a ≥100% loss.** A rate below −100% gave a bare
  `math domain error` (or a silent complex on a periodic basis); now raises naming the rate, basis and `t`.

### Added
- **`WeekendSchedule` (AC-T4.17)** — a frozen `((since_year, Weekend), …)` transition list a `Calendar`
  may carry in place of a single `Weekend`, so a weekend-rule change (Saudi 2013; a future ILS change) is
  *expressible*. Shape only — no market's future rule is guessed, and `Calendar` stays ≤5 fields (the
  `weekend` field's type widens, no field added).

### Changed
- **Ledger tightening.** Closed AC-T4.7/8/9/11/17 (above). **AC-3.6b** (FX-spot completion) **promoted out
  of the ledger into F1's scope** (`redesign/README.md`) — its trigger *is* F1, so it is scoped work, not
  a deferral. **AC-T4.18** (month-arith triplication) recorded as a **considered exception, not a
  deferral** (rule of three; consolidating now adds indirection with no present consumer). Ledger count
  **22 → 15**; `AC-C1` (half-day concept) stays deferred (its trigger is a real shaping consumer).

## [0.83.0] - 2026-07-20

Post-closure seam residue (`closed_POST_CLOSURE{,_FINDINGS}.md`): the gaps *between* findings each
marked FIXED — which the disposition census structurally cannot see. Nothing reopened a closed
disposition. Red→green throughout.

### Fixed
- **B1 — `accrued_rate` lockout underflow (silent wrongness).** A `lockout >= len(window)` drove
  `frozen` negative, and Python negative-indexing then silently froze the rate series to *early*
  dates (reachable from a short stub with a standard lockout). Now raises, naming the window. Test
  `test_b1_lockout_longer_than_window_raises`.
- **B2 — `PricingResult.clean` currency guard.** `clean` subtracted raw amounts, so a cross-currency
  `accrued` produced a silently wrong `clean`. Now `return self.pv - self.accrued`, delegating to
  `Money.__sub__`'s existing guard (a fix that deletes code). Test
  `test_b2_clean_rejects_cross_currency_accrued`.
- **A3 — `JointCalendar` by-name round trip.** A composite `"A+B"` identity could not rehydrate
  (`get_calendar("A+B")` raised), so the first serialized XCCY trade could not round-trip — a seam
  between audit 3.2 and 3.4, both FIXED. Added `JointCalendar.to_dict`; `get_calendar` splits on
  `"+"` into a `JointCalendar` (its return type widens to `CalendarProtocol`). Test
  `test_a3_jointcalendar_round_trips_by_name`.

### Ruled (recorded, code lands with first consumer)
- **A1 — `TimeMeasure(anchor, day_count)` is the only sanctioned `date→t` map** (`redesign/20`
  Part A addendum); build as an L0 module in Topic 1's first slice. Ledger `AC-T4.5` promoted to
  "before Topic 1".
- **A2 — `Frequency.per_year()` raises for non-integer tenors** (28D/daily/bullet); BUS-period
  products (TIIE/CDI) do not enter ICMA contexts. Ledger `AC-T4.15`.
- **B3 — `fx_spot_date` keeps joint intermediate-day counting.** The asymmetric ACI rule (a USD
  holiday on an intermediate day does not pause a USD pair's count) is **not** implemented — no
  citable source with a verifiable worked example, and the green-oracle gate forbids coding an
  unverifiable convention. Scope pinned in `OPEN.md` **AC-3.6b** + the `fx_spot_date` docstring.

### Removed
- **POST_CLOSURE C residue (~41 lines, most predating the closure).** Deleted the unread half-day
  machinery — `DayType`, `Calendar.day_type`, `HolidaySet.half_days`, `_half_days_of`,
  `day_after_thanksgiving` + the 3 US half-day rules + their exports and S5 test (`CalendarProtocol`
  deliberately excludes `day_type`; **S5 redirected** from classify-now to defer-to-first-fixing-
  cutoff-consumer — the *concept* is ledgered as `OPEN.md` **AC-C1**, trigger "first fixing-cutoff /
  early-close consumer", so the deferral carries an id like every other). Deleted `_SPOT_LAGS["USDRUB"]`
  (RUB unregistered, pair unconstructible). Dropped
  the unused `name` param of `register_unit`. `_boundary_slope` now returns the exact end-segment
  slope for LINEAR (mirrors `_boundary_slope_log`); the finite step is kept for splines.

### Changed
- `redesign/11` gains a 6th standing checkpoint review input (**V3**): re-read the whole `AC-*`
  deferred ledger and check each trigger, since no automated gate watches those items and six
  triggers are condition-driven rather than roadmap-driven. (`CLOSURE_VERIFICATION.md` V1/V2 were
  already fixed on the closure branch; `closed_POST_CLOSURE.md` carries the D2 falsifier-pointer.)

## [0.82.0] - 2026-07-19

### Fixed
- **Ledger tightening — reclassified the 19 audit-closure ledger items against "wrong answer / invalid
  input today?" Only one was a latent bug wearing a Tier-4 badge:**
  - **AC-T4.2 — `Tenor` accepted non-positive counts** (invalid input). `Tenor.parse("0D")`/`"-3M"` and
    `Tenor(0, …)` now raise (`__post_init__` guard) — a zero step never advances a schedule loop.
    Removed from `OPEN.md`. Test: `test_t4_2_non_positive_tenor_is_rejected`.

### Added
- **EURIBOR_6M** — the standard EUR 6M vanilla floating leg (EUR, TARGET, ACT/360). Its absence was a
  gap in an index set presented as complete (`EURIBOR_3M` already shipped). Test:
  `test_t4_1_euribor_6m_is_registered`. The other missing indices (EFFR, BBSW, AONIA, CORRA, TIIE, …)
  stay ledgered — absent capabilities, deferred to their currency/product topics.

### Notes
- The remaining 17 ledger items confirmed **NO** on the wrong-answer/invalid-input test (absent
  capability or clean rejection — building them now would reintroduce the speculative infra Phase 4
  deleted, §6b) and stay in `OPEN.md`. Six of their triggers are condition/event/usage-driven rather
  than roadmap topics (AC-T4.7/8/9/11/17/18) — flagged as such in the ledger.

## [0.81.0] - 2026-07-19

### Removed
- **PONYTAIL_AUDIT micro-cuts (findings 4/5/8).** `Currency.value`/`Unit.value` alias properties
  (zero surviving call sites); `RfrConvention.none()` (byte-identical to the default — the `_term`
  factory already names the "term rate, no RFR mechanics" intent); the one-entry `_ANNUAL_BASIS` dict
  (inlined as a direct `BUS_252` conditional). Kept with reasons: `exponential_growth` (verified
  primitive + oracle, no schema hazard, near-term EM-bond consumer) and `Calendar._rule_set` (its
  union-narrowing is load-bearing for pyright).

### Changed
- **Foundation independent audit — CLOSED.** The three reports under `redesign/independent_audits/`
  are renamed `closed_AUDIT.md` / `closed_PONYTAIL_AUDIT.md` / `closed_PONYTAIL-DEBT.md`, each with a
  per-finding **disposition block** (fixed-with-a-test → CHANGELOG v0.75.0–v0.81.0 / ledgered →
  `OPEN.md` / rejected+reason). All Tier-4 items and the deferred sub-parts (NYSE+Fed-bank calendars,
  Tokyo Silver-Week/Olympic, FX quote-order registry, the interpolator-rebuild ponytail marker) are
  ledgered in `OPEN.md` with named re-open triggers. `redesign/README.md` row 0 now points at the
  closed record instead of "active work".

## [0.80.0] - 2026-07-19

### Removed
- **Foundation audit closure — Phase 4: A3 speculative-field cuts.** The design SHAPE was ratified;
  the guessed FIELDS were not (ruling A3 — a guessed field frozen into a schema is a costlier retrofit
  than the file-touch to reintroduce it correctly). Cut, each with zero consumers:
  - `underlying.py` sibling identities `ReferenceEntity` / `InflationIndex` / `FxFixing` /
    `EquityUnderlying` / `CommodityUnderlying` (guessed `fixing_time`, a required `grade`) — the file
    now holds only the `Underlying` protocol + `AssetClass`; each sibling returns *validated* when its
    asset class actually ships (§3c, one identity at a time).
  - `numerical_config.py` (`NumericalConfig` + sub-configs) — ships with the first engine that reads it;
    no engine exists yet, so no knob has a present consumer.
  - `PricingResult.cashflow_breakdown` / `sensitivities` / `diagnostics` and the `DiscountBasis`
    wrapper — `basis` is inlined to `Currency | None` (the collateral currency); greeks/breakdowns/
    diagnostics return with the L4/L5 layer that produces them. `results.py` drops its `fields-exempt`.

### Changed
- **A1 doc correction** (`interpolation.py`): documented that `CONTINUE_SLOPE` extends the end slope in
  the interpolation's *own* space (log space for `LOG_LINEAR`) — the fix that keeps extrapolated DFs
  positive.
- **A2 doc correction** (`AUDIT_topic0_foundation.md`, `HANDOFF_topic0_gate.md`,
  `handoff_topic1_conventions.md`): the S16 "record the invariant" ruling is **withdrawn** — business-day
  counting is one half-open `[start, end)` primitive, not a recorded invariant (the CDI/BUS-252 consumer
  needed the other convention, so the premise was false).

## [0.79.0] - 2026-07-19

### Changed
- **Foundation audit closure — Phase 3b: Schedule provenance (finding 3.5 discharged, red→green).**
  `Schedule.periods` emits a per-period **`SchedulePeriod`** record (accrual start/end, `is_stub`,
  payment date). **ACT/ACT ICMA Rule 251.2**: long coupons now sum the day count over their notional
  (quasi-coupon) periods — pinned to the ISDA 2006 §4.16 published long-first-coupon example
  (0.9157608696) — which **deletes the long-stub raise** (NG-DEFER-1 discharged). Explicit stub anchors
  via **`RegularPeriod(first_regular_date, last_regular_date)`** (the ISDA concept, folded into
  `ScheduleTerms.stub` as a union to stay ≤5 fields, no `fields-exempt`). **`PaymentRule(calendar, lag)`**
  places `payment_delay` — a payment settles `lag` business days after the adjusted accrual end on its
  own (often different, for XCCY) calendar; it was dead code until a schedule gave it a payment column.

## [0.78.0] - 2026-07-19

### Changed
- **Foundation audit closure — Phase 3: Tier-3 structural (red→green).** **3.2** `CalendarProtocol`
  (`is_business_day`/`adjust`/`add_business_days`/`identity`); `JointCalendar` implements it via a shared
  arithmetic mixin, and every `Calendar` consumer slot (`RollRule.calendar`, `year_fraction`,
  `settlement_date`, `accrued_rate`) is retyped to it — cross-currency schedules now type-check and work.
  **3.3** open registries: public `register_calendar`/`register_rate_index`, all `register_*` raise on
  conflicting re-registration (was a silent overwrite), `temporary_*` context managers for test isolation.
  **Q1** the USD calendar is renamed `NEW_YORK_SIFMA → US_GOVERNMENT_SECURITIES` (it is the govvies
  calendar, not a generic NY one) and SOFR binds to it explicitly. **3.5(partial)** ACT/ACT ICMA long
  stub RAISES rather than mis-accruing (multi-period support lands with 3b). **3.6** `fx_spot_date` (T+2
  joint-calendar; T+1 USD/CAD; a cross cannot settle on a US holiday); `spot_lag` moved out of
  `CurrencyPair` identity into a pair-conventions registry. **3.7** `_denominator` raises on unsupported
  day counts (no silent 360); `FixingSource` protocol on `accrued_rate` (`FixingHistory` is the impl).

## [0.77.0] - 2026-07-19

### Fixed
- **Foundation audit closure — Phase 2: four Tier-2 convention bugs (red→green).**
  **2.1** SOFR was declared with a non-standard `observation_shift=2`; ISDA SOFR OIS is plain
  compounded-in-arrears (`RfrConvention(payment_delay=2)`) — the LIBOR-fallback index correctly keeps
  the shift. **2.2** the `NEW_YORK_SIFMA` calendar was missing **Good Friday** (SOFR does not publish
  → `FixingHistory` raised on real data annually) and used federal Sat→Fri observance (wrongly closing
  2021-12-31); it now includes Good Friday and uses `SUNDAY_ONLY` (SIFMA does not shift Saturday
  holidays). **2.3** LONDON was missing the 2022–23 one-off bank holidays (Platinum Jubilee, state
  funeral, coronation, 2020 VE-Day move) — added a `dates()` one-off DSL combinator and `since/until`
  gating on `nth`. **2.4** Tokyo equinoxes were hardcoded (wrong in 2024–26); now computed
  astronomically (`equinox()`), plus the Emperor's-Birthday move (`since=2020`). Three existing
  calendar tests that encoded the buggy SIFMA Sat→Fri observance were corrected.

## [0.76.0] - 2026-07-19

### Fixed
- **Foundation audit closure — Phase 1: five Tier-1 computational bugs (red→green).**
  **1.1** ACT/ACT AFB undercounted leap-to-leap spans (2004-02-29→2008-02-29 gave 3.9973, now 4.0) —
  whole years are counted from `end` directly (`end − k·years`), not by accumulating single-year clamps.
  **1.2** `LOG_LINEAR` + `CONTINUE_SLOPE` returned a **negative** discount factor (DF(30) = −0.275);
  per ruling **A1** it now extrapolates in the interpolation's own (log) space → +0.209.
  **1.3** BUS/252 and CDI counted business days over different intervals; per ruling **A2** there is now
  one primitive, `business_days_between` counting `[start, end)`, and the S16 invariant is withdrawn.
  **1.4** backward schedule generation drifted the roll day through short-month clamps (May 31 quarterly
  → Nov 29) and keyed EOM on `start`; now generated as `anchor ± k·tenor` with EOM anchored on the
  generation seed (maturity for backward). **1.5** furikae substitution iterates `sorted(holidays)`
  (deterministic; the substitute union was already order-invariant, so no answer changed).

## [0.75.0] - 2026-07-19

### Added
- **Foundation audit closure — Phase 0: the foundation is now a real, importable package.** Added
  `src/pricebook_ng/foundation/__init__.py` with an explicit `__all__` public-API surface (the contract
  upper layers may depend on), a `py.typed` marker (typed code no longer reads as `Any` downstream), and
  a root `pyproject.toml` (src layout, dynamic version, numpy/scipy deps) — `pip install -e .` works.
  **Renamed `foundation/calendar.py` → `calendars.py`** so it no longer shadows stdlib `calendar` once
  `foundation/` is on `sys.path`. No behaviour change (136 tests, all gates + pyright green).

## [0.74.3] - 2026-07-19

### Fixed
- **L0 is now static-type clean (pyright 22 → 0 errors), no behaviour change (136 tests green).**
  (1) `Currency` (37), `Unit` (9) and `Frequency` (7) named constants are declared as `ClassVar`, so
  `Currency.USD` / `Unit.BARREL` / `Frequency.MONTHLY` type-check **and autocomplete** — the static half
  of the open registry (undoes A3's accepted-cost note; the registry stays open via `register_*`).
  (2) the closed registries use a mutable backing + frozen `MappingProxyType` **view** (`_registry`→
  `_REGISTRY`, `_calendars`→`_CALENDARS`) instead of rebinding a `dict`-typed name. (3) `Calendar`
  narrows the `HolidaySet | tuple[Rule,...]` union via a `_rule_set` property. (4) scipy `brentq`'s
  union return is `cast` to `float`. `verify.py fields` now excludes `ClassVar` (definitionally not a
  dataclass field) — a real gate correction.

## [0.74.2] - 2026-07-19

### Fixed
- **Foundation audit (fix/foundation-audit) — three RFR/calendar correctness fixes, red→green.**
  **F2 (silent-wrongness):** `accrued_rate` under `observation_shift` compounded the shifted
  observation window but normalised by the *interest-period* day-count fraction; the two windows can
  differ by a day (a shift crossing a holiday), silently scaling the rate (e.g. 5.72% for a flat 5%
  series). Now normalises by the numerator window's own days (`total/basis`) — consistent for both
  observation-shift and lookback (SOFR 1e-12 oracle unchanged). **F1 (silent-wrongness + crash):** a
  valid accrual containing no business day returned `0.0` (COMPOUNDED) or raised `ZeroDivisionError`
  (AVERAGED/EXPONENTIAL); now raises `ValueError` cleanly (S14). **F3 (wrong-answer, latent):**
  `Calendar.add_business_days(d, 0)` returned `d` even when `d` was not a business day (a `fixing_lag=0`
  lookup could land on a non-business date); `n==0` now requires a business day and raises otherwise —
  callers snap via `adjust()`. Systematic checks (Easter, observance regimes, EOM, IMM/CDS, solvers)
  verified clean against published references.

## [0.74.1] - 2026-07-19

### Changed
- **L0 final cleanup (last work before Topic 1).** (A1) `Accrual` moved `cashflow.py` → `day_count.py`
  — it is an *applied day count*, not a cashflow concept; the move **drops the semantically-wrong
  `rate_index → cashflow` import edge** (rate_index 8→7 imports), no new edges. (A2) the calendar and
  rate-index registries are frozen with `types.MappingProxyType` (the quarry's bug was registry
  mutation at import); the **open** currency/unit registries keep a mutable backing behind the
  sanctioned `register_*` path but expose read-only `CURRENCIES`/`UNITS` views — direct mutation raises,
  runtime registration still works (S1). (A3) `money.py` records *why* `Currency` is an open registry
  and what reverting to an `Enum` would cost (silently dropping BRL / the LatAm scope). Pure/hardening
  change; no parking change (the 13 remain parked).

## [0.74.0] - 2026-07-19

### Changed
- **Topic 0 GATE CLOSE — foundation parked (drawdown 13/793).** All 8 gate slices (S1–S17, F1–F4)
  landed and green (125 L0 tests; `fields` on merit; `layers`/`acyclic`/`debt`/`version`/`provenance`/
  ruff; both regression oracles). Parked the Topic 0 set to `parked/topic-00-foundation/`: the 11
  covered files + `data_registry` (**dead** — import-time JSON registry ruled away) + `notional`
  (**absorbed** into `Money`/`Leg`). `core/fixings` is **reassigned→market-data** (immutable
  `FixingHistory` read model is the L0 type; the mutable store + file I/O is not L0). Roll-up refreshed;
  gate-close report `CP_topic0_gate_close.md`. Topic 1 (multicurve + linear rates) opens once Cowork
  ratifies the gate.

## [0.73.0] - 2026-07-19

### Added
- **Topic 0 gate Slice 8 — serialisation pattern demonstrated on a hard case.** Per-class
  `to_dict`/`from_dict` (+ `schema_version` at the serialised root) on `Leg` and its constituents
  `Cashflow`/`Delivery`/`Accrual`/`Money`/`Quantity` — one case exercising all three dimensions at
  once: a **collection** (flows), an **enum** (`DayCountConvention`), **nested value objects**, and a
  `Cashflow | Delivery` **union** via a `kind` discriminator. The round-trip is JSON-clean;
  `Leg.from_dict` refuses an unknown `schema_version`. No framework — the quarry's 831-line machinery
  is not carried (the pattern is the deliverable).

## [0.72.0] - 2026-07-19

### Changed
- **Topic 0 gate Slice 5 — L0 numerics are thin scipy adapters (S17).** `distributions.py`
  (`scipy.stats.norm`), `solvers.py` (`scipy.optimize`: `brent` · `newton` · `secant` · LM
  `least_squares`), `interpolation.py` (`scipy.interpolate` `CubicSpline`/`PchipInterpolator`/
  `Akima1DInterpolator`; linear/log-linear ours). **Removes** the hand-rolled `bisect_root`/
  `nelder_mead` (no duplicates). Interpolation now states its **extrapolation policy per end**
  (`FLAT | CONTINUE_SLOPE | RAISE`, default RAISE both) — closes C4's silent divergence. scipy is
  the single C++-port swap point (never called from engines/models); `numpy`/`scipy` pinned in CI
  for convergence reproducibility. Hagan-West monotone-convex stays in Topic 1 (curve construction).

## [0.71.0] - 2026-07-19

### Changed
- **Topic 0 gate — foundation parked (first physical drawdown).** The gate audit (F1–F4 + S1–S16) is
  landed and green; `git mv` the **11 covered quarry `core/` files** →
  `parked/topic-00-foundation/` with a per-file coverage manifest (each classified `covered`/`dead`/
  `reassigned` with consumer-analysis evidence; all shed features are forward-linked deferred
  *capabilities*, none blocking). `core/data_registry` + `core/notional` are **reassigned→topic-01**
  (no L0 counterpart). `quarry_reconciliation.md` is now a thin topic-method roll-up (pre-replan CP
  record frozen as history). **Drawdown 11/793.** Topic 1 (yield curves) opens once Cowork ratifies
  the gate.

### Added
- **Topic 0 gate Slice 1 — `date + Tenor` (S7) + rule-based schedule anchors (S8).** `Tenor.__radd__`
  gives the raw shifted date (day/week exact; month/year clamp to the target month's length) — the
  most-used curve-building op; business-day rolling stays a separate `RollRule` concern (finance-free).
  `RollConvention.IMM`/`CDS` on `ScheduleTerms.roll_day` anchors interior periods on the 3rd Wednesday /
  20th regardless of the effective date (IMM-dated FRAs/futures, standard CDS). Oracles: `date+3M` month
  clamps; IMM-anchored schedule lands on 3rd Wednesdays of Mar/Jun/Sep/Dec; CDS on the 20th.

## [0.70.0] - 2026-07-19

### Changed
- **Topic 0 gate rework (6b, final) — open `Currency`/`Unit` registries (audit S1/S4 + the meta-rule).**
  `Currency` is now an ISO code + **`minor_units`** value with an open **registry** (JPY/KRW/CLP are
  zero-decimal); `Unit` is a symbol + registry. A new market/commodity is `register_currency(...)` /
  `register_unit(...)` — a *declaration*, never an L0 enum edit (open-ended domain → registry). The
  standard members stay class constants (`Currency.USD`, `Unit.BARREL`), interned so `is`/`==` both hold;
  the `Money`/`Quantity` guards compare by value. **BRL-end-to-end gate oracle** (Currency · BUS/252 ·
  São Paulo) green — the in-scope LatAm claim is real.

## [0.69.0] - 2026-07-19

### Changed
- **Topic 0 gate rework (6a) — `Calendar` `day_type` / half-days (audit S5).** `Calendar.day_type(d)` →
  `BUSINESS | HALF | HOLIDAY | WEEKEND`: a **half-day** is an early close (a business day with a
  shortened session — affects fixing cut-offs/settlement), e.g. US July 3, Christmas Eve, the day after
  Thanksgiving. To carry half-days and stay ≤5 fields, the holiday rules bundle into a
  **`HolidaySet(holidays, half_days)`**; a bare rule tuple auto-wraps, so only the half-day calendar
  changed. `is_business_day` is unchanged (a half-day is still a business day).

## [0.68.0] - 2026-07-19

### Changed
- **Topic 0 gate rework (5/6) — convention completeness (audit S11/S8/S9/S16).**
  `DayCountConvention` gains **`1/1`** (always 1) and **`ACT/ACT AFB`** (whole years + leap-aware stub) —
  ISDA 2006 completeness (S11). `ScheduleTerms` gains an optional **`roll_day`** — the day-of-month the
  interior periods roll on (a bond the 15th, CDS the 20th), else anchored on `start` (S8). Recorded
  invariants: **time-of-day never enters `Calendar`** (fixing time is index metadata, expiry cut is L2 —
  S9); business-day counting is **start-exclusive/end-inclusive** (S16).

## [0.67.0] - 2026-07-19

### Changed
- **Topic 0 gate rework (4/5) — rate-quotation basis (audit S12, the biggest missing concept).**
  `foundation/rate_basis.py`: **`Compounding`(SIMPLE·ANNUAL·SEMI_ANNUAL·QUARTERLY·MONTHLY·CONTINUOUS)** —
  the basis a *quoted* rate compounds on — and **`convert_rate(rate, t, from, to)`** (via `growth_factor`).
  A rate is meaningless without its basis; treating semi-annual as continuous is silently wrong by ~r²/2.
  **Recorded invariant:** internal curve rates are continuous on the curve day-count; quotes carry their
  own basis and convert at the boundary (keeps `Rate` a plain float). The index enum
  **`CompoundingMethod` is renamed `AccrualMethod`** to kill the name collision (it is a different
  concept — index averaging, not quotation basis).

## [0.66.0] - 2026-07-19

### Changed
- **Topic 0 gate rework (3/5) — the index rework (audit F2/F3/F4/S10).**
  - **F2 (the critical fix): `RateIndex` carries its own calendar.** `accrued_rate` now reads
    `index.accrual.roll.calendar` — **no currency inference** (the reverted quarry flaw #14/#16). SOFR
    fixes on **SIFMA**; two same-currency indices on different calendars now differ. Regression oracle:
    SOFR-on-SIFMA ≠ a same-currency index on TARGET over a period spanning Columbus Day.
  - **F3: `RateIndex` decomposed** → `IndexId`(name·currency·`Tenor`) · `AccrualConvention`(day_count·
    `RollRule`) · `FixingRule`(observation_style·compounding·fixing_lag) · `RfrConvention`(shift·lookback·
    lockout·payment_delay, with `.none()` for IBORs) · `spread_adjustment` — 5 parts, **marker removed**.
  - **F4/S10: `foundation/underlying.py`** — the general `Underlying` protocol (`name`, `asset_class`)
    that `RateIndex` satisfies, plus sibling identities **defined now, populated later**:
    `ReferenceEntity`, `InflationIndex` (lag·interpolation·base), `FxFixing` (pair·source·time),
    `EquityUnderlying`, `CommodityUnderlying` (unit·**delivery location**·**grade** — Brent ≠ WTI).

## [0.65.0] - 2026-07-19

### Changed
- **Topic 0 gate rework (2/5) — config & result shapes (audit F1/S15/S6/S2/S13/S14).**
  - **`NumericalConfig` decomposed** by method family — `MonteCarloConfig`/`LatticeConfig`(tree folded
    in)/`IntegrationConfig`/`SolverConfig`, each ≤5 fields; the `fields-exempt` marker is **removed**
    (`verify.py fields` passes on merit; reads as `numerics.monte_carlo.paths`). RNG family **pinned**
    as an invariant `RNG_FAMILY` (S15 — not a field, which would push MC to 6). Nested serialisation.
  - **`PricingResult` full decomposition vocabulary** fixed once (pv · accrued · clean · cashflow
    breakdown · sensitivities · diagnostics), fields optional/unpopulated — a legitimate output-record
    exemption per §3b, so no asset class retouches L0 to add a field (S6).
  - **`Leg` holds `flows: tuple[Cashflow | Delivery, ...]`** — a physical/commodity leg is now
    expressible (S2); **pay/receive is the sign of the amount, no direction field** (S13).
  - **Degenerate periods raise** — a reversed/zero-length `Accrual`, and `year_fraction(end < start)`
    (S14).

## [0.64.0] - 2026-07-18

### Changed
- **Topic 0 gate rework (1/5) — `Tenor` value type + `Frequency` reshape (audit S7/S3).**
  `foundation/tenor.py`: **`Tenor(count, unit: D/W/M/Y)`** with `parse("28D")`/`__str__`/`months()` — the
  one primitive behind index tenors, schedule steps and curve pillars (overturns "Tenor stays a string").
  **`Frequency` is now a `Tenor`-step** (or `BULLET`, a single period), so it expresses **28-day (TIIE),
  daily, and single-period** — a month-int enum could not. Behaviour-preserving for the named
  frequencies (MONTHLY/QUARTERLY/… are class constants). Oracles: parse/str round-trip; 28D/daily/bullet
  schedules; standard frequencies unchanged.

## [0.63.0] - 2026-07-18

### Added
- **Topic 0, Slice 6 — `numerics-config`: the finite-difference/numeric floor + engine I/O (L0).**
  The last slice before the Topic 0 gate.
  - **Complete `NumericalConfig`** — the full reproducibility knob set designed up front (MC
    paths/seed/antithetic/sobol/bridge · PDE time/space/n-std-devs · tree steps · quadrature tol/max-iter
    · COS n/L · root-finder tol/max-iter · fd bump), so it never retrofits a foundational value type (the
    12 knobs deferred at CP-3 retire #1). `# fields-exempt: config aggregate`; positive-knob validation;
    the **serialisation pattern** (`to_dict`/`from_dict` + `schema_version`, no framework).
  - **`distributions`** (norm_cdf/pdf/ppf) · **`solvers`** (bisect_root, Nelder-Mead) · **`interpolation`**
    (linear + log-linear — *mechanism only*; extrapolation is an L1 curve policy, so out-of-range raises).
  - **`PricingResult`** (A2 decomposition: dirty PV + accrued ⇒ clean) now records its **basis** — value
    currency (`pv`) **and** the collateral/discounting `DiscountBasis` — so a PV is never ambiguous
    (settlement ruling §1; the numeraire *choice* stays L3). **`PricingFailure`** — failure as a value.
  - Deferred with their first consumer (no shape risk): the MC/PDE/Fourier-COS/tree/quadrature *engines*.
  - slice: `numerics-config` (Topic 0 S6)

## [0.62.0] - 2026-07-18

### Added
- **Topic 0, Slice 5 — `index-identity` (widened): `RateIndex` + `FixingHistory` + `accrued_rate` (L0).**
  `foundation/rate_index.py`. A new index is a **declaration, never a code change.**
  - **`RateIndex` covers all rate kinds:** the RFR set (`fixing_lag`/`observation_shift`/`lookback`/
    `lockout`/`payment_delay`/`compounding`) **plus the widening** — **`observation_style`**
    (BACKWARD_LOOKING RFR vs FORWARD_LOOKING term/IBOR) and **`spread_adjustment`** (ISDA fallback
    credit spread). It's an identity aggregate (`# fields-exempt`).
  - **`FixingHistory` is generic over index** (name→date→value), so it will also hold inflation levels
    / FX fixings / equity observations — the sibling identities land later under the same pattern.
  - **One generic `accrued_rate(index, accrual, fixings)`**, branching only on `CompoundingMethod`:
    FLAT/forward → the single fixing at the start; backward → compounded/averaged over the observation
    window, with **`lookback` (shift the rate only) vs `observation_shift` (shift the whole window)**
    distinct, plus `lockout`; `spread_adjustment` added (a fallback = base RFR + spread, not absorbed).
    `CompoundingMethod` also carries **`EXPONENTIAL`** — the Brazilian BUS/252 `(1+r)^(bd/basis)` used by
    CDI/SELIC (and LTN/NTN-F/DI), where a flat rate reprices to itself exactly (vs money-market
    `∏(1+r·δ)`); `CDI` (BRL) is declared. The **basis is derived from the day-count** (`BUS/252 → 252`),
    not hardcoded — another basis is a day-count entry. The daily-series (`rᵢ`, floating CDI in arrears)
    path is `accrued_rate`; the single-fixed-rate (`r`, LTN/NTN-F) growth factor is the separate
    `exponential_growth(rate, business_days, day_count)` primitive.
  - **Registry by explicit construction** (SOFR, SONIA, ESTR, TONA, SARON, EURIBOR_3M, TERM_SOFR_3M,
    a USD-LIBOR fallback) — **no import-time JSON reload** (the quarry rebound the whole `_REGISTRY`
    from a file, where one bad row dropped the other 27).
  - Oracles: compounded RFR vs a hand-computed series; lookback ≠ observation-shift; 0/0 ≠ 2/2;
    forward-looking term ≠ backward-looking compounded; fallback = base + spread. `verify.py layers` green.
  - slice: `index-identity` (Topic 0 S5)

## [0.61.0] - 2026-07-18

### Added
- **Topic 0, Slice 4b — `settlement`: the flow-shape vocabulary (L0).** `foundation/settlement.py`
  (new slice, ruled after S4 — `rulings_topic0_settlement_and_index.md`).
  - **`Delivery`(date, `Quantity`)** — the physical-world counterpart to `Cashflow(date, Money)`, giving
    the S4 `Quantity` a home in a flow. **`SettlementType`** (CASH / PHYSICAL / **AUCTION** — a marker
    only; CDS credit-event mechanics stay in the credit topic). **`SettlementTerms`**(type, currency,
    lag) — the settlement currency **may differ from the contract currency** (quanto / NDF) and is
    `None` for physical; cash/auction require one. **`settlement_date`** = trade + `lag` business days
    under a calendar (FX T+2).
  - Scoped strictly L0: no PV (the quarry's `cash_settlement` returned one — that's L4), no
    product→convention table (L2), no collateral/CSA discounting (numeraire is an **L3** model choice
    per §1). Content mined from the 398-LOC zero-fan-in orphan `core/settlement.py`.
  - Oracles: T+2 settlement date skips the weekend; cash pays `Money` vs physical delivers `Quantity`;
    settlement currency ≠ contract currency; physical carries no currency. `verify.py layers` green.
  - slice: `settlement` (Topic 0 S4b)

## [0.60.0] - 2026-07-18

### Added
- **Topic 0, Slice 4 — `money-quantity`: value types + instrument atoms (L0).**
  `foundation/money.py` + `foundation/cashflow.py`.
  - **`Money`(amount, currency)** with currency-guarded arithmetic — **mixing currencies is a
    `TypeError`**; `+`/`-`/unary `-`/scalar `*`. **`Currency`** covers the **37** markets (matches the
    Slice-1 calendars). **`Quantity`(amount, `Unit`)** — commodities settle in barrels/MWh/tonnes/troy
    ounces/…; closed under **same-unit** arithmetic only (barrels and MWh do not add).
    **`CurrencyPair`**(base, quote, spot_lag) with `.name` and T+2 default (USD/CAD T+1).
  - **`Accrual`(start, end, day_count)** with **`Accrual.year_fraction(*, coupon_period=None,
    calendar=None)`** — the ergonomic entry point over the Slice-2 primitive (bundles start+end+
    day_count; threads ICMA anchors / BUS-252 calendar). **`Cashflow`**(date, `Money`, accrual) ·
    **`Leg`**(cashflows, day_count).
  - Serialisation deferred to Slice 6 (per the topic plan). Oracles: currency-mixing type error;
    same-unit-only `Quantity`; `Accrual.year_fraction` == the S2 primitive (incl. strict-ICMA anchors).
    `verify.py layers` green (L0 finance-free).
  - slice: `money-quantity` (Topic 0 S4)

## [0.59.0] - 2026-07-18

### Fixed
- **Topic 0 S3-checkpoint corrections (Cowork ruling `rulings_topic0_s3.md`).** Three fixes to the
  merged calendars/daycounts slices, done before Slice 4.
  - **ANZAC-Day observance (§3.1 — the `observe`-lift correction).** The lift of the per-rule `observe`
    flag to a calendar-level `Observance` was *not* behaviour-preserving: AU/NZ **ANZAC Day (25 Apr)** is
    commemorated on the actual date and **not** mondayised, while New Year/Christmas in the same
    calendars are. `fixed()` regains a per-rule `observed=` override (default follows the regime);
    `SYDNEY`/`WELLINGTON` ANZAC set `observed=False`. Regression oracle: ANZAC on a weekend does not
    shift while New Year in the same calendar does.
  - **ACT/365L is frequency-dependent (§3.3, ISDA §4.16(i)).** Annual (or no context) → 366 iff a 29 Feb
    lies in the period; more frequent than annual → 366 iff the period **end** is in a leap year. Uses
    `CouponPeriod.frequency` (now serving three conventions — ICMA, 30E/360-ISDA, ACT/365L).
  - **`Coverage` marker (§4).** Eight EM calendars (Riyadh, Cairo, Istanbul, Tel Aviv, Beijing, Seoul,
    Mumbai, Bangkok) are **secular-only** — they omit lunar/religious holidays. Now marked
    `Coverage.SECULAR_ONLY` (not silently wrong); lunar data forward-linked to the EM-rates topic.
  - slice: `topic0-s3-corrections`

## [0.58.0] - 2026-07-18

### Added
- **Topic 0, Slice 3 — `schedules`: `RollRule`/`ScheduleTerms`/`Schedule` + IMM/CDS rolls (L0).**
  `foundation/schedule.py`.
  - `Frequency` · `StubType` (all four) · **`RollRule`**(calendar, convention, eom) ·
    **`ScheduleTerms` → `build_schedule` → `Schedule`**. A `Schedule` carries **both** the unadjusted
    period boundaries (accrual) **and** the business-day-adjusted dates (payment) — C2, not the same
    list. **EOM anchored once from `start`** (ISDA §4.10); stdlib month arithmetic (day clamped to
    month length), no `dateutil`.
  - **Long stubs are explicit** — the stub period merges with its neighbour by construction; the
    quarry's `first_gap < months*30*0.5` merge heuristic is **shed**.
  - **New: IMM roll dates** (`imm_date`/`next_imm` — 3rd Wednesday) **and CDS roll dates**
    (`cds_roll_date`/`next_cds_roll` — 20th of Mar/Jun/Sep/Dec). Neither existed in the quarry; futures
    and credit both need them.
  - Oracles: ISDA §4.10 EOM anchoring (month-end start → month-ends), all four stubs, adjusted ≠
    unadjusted under a holiday, published IMM/CDS tables. `verify.py layers` green (L0 finance-free).
  - slice: `schedules` (Topic 0 S3)

## [0.57.0] - 2026-07-18

### Added
- **Topic 0, Slice 2 — `daycounts`: 10 conventions + `CouponPeriod` (L0).** `foundation/day_count.py`
  — `year_fraction(start, end, convention, *, coupon_period=None, calendar=None)`.
  - The seven market-standard conventions (ACT/360, ACT/365F, **30U/360 US-SIA with the February
    rules**, 30E/360, ACT/ACT ISDA, ACT/ACT ICMA, BUS/252) **plus the three gaps: ACT/365L, 30E/360
    ISDA (with the termination-date rule), NL/365.**
  - **No hidden context (content mined, defaults dropped):** ICMA anchors ride on a `CouponPeriod`
    (reference_start/end, frequency, is_final) — **strict**: missing anchors *raise* (the deleted
    `strict_icma=False` silently fell back to ACT/365F and priced a UST coupon at 1.9836); BUS/252
    *requires* a `Calendar` — no São Paulo default; passing none raises.
  - Oracles: ISDA 2006 §4.16 (ACT/ACT ISDA worked example, 30U/360 Feb edges), ICMA Rule 251
    (**UST semi-annual coupon = exactly 2.0000**), ACT/365L & NL/365 leap-day handling, 30E/360-ISDA
    last-day-of-month + February-termination exception. `verify.py layers` green (L0 finance-free).
  - slice: `daycounts` (Topic 0 S2)

## [0.56.0] - 2026-07-18

### Added
- **Topic 0, Slice 1 — `calendars`: the holiday-rule DSL + all 37 markets, identity-keyed (L0).**
  `foundation/calendar.py` (engine) + `foundation/market_calendars.py` (declarations).
  - **DSL** (`fixed`/`easter`/`orthodox`/`nth`/`monday` + the `christmas_boxing`/`victoria_day`/
    `midsummer_eve`/`mexico_inauguration` cascades, `since`/`until` year-gating), Gregorian + Orthodox
    Easter, and the **three observance regimes** — US 5 U.S.C. §6103 (Sat→Fri, Sun→Mon), Commonwealth
    next-working-day (Sat/Sun→Mon), Johannesburg Sunday-only — plus Tokyo **furikae** and the
    Israel/MENA **Fri–Sat weekend**. `BusinessDayConvention` gains `NEAREST`. `JointCalendar` = union.
  - **Structure cleaned (content mined, not structure):** the quarry's ~38 `Calendar` subclasses +
    currency-keyed registry collapse into **one frozen `Calendar` value** (identity · rules · weekend ·
    observance — 4 fields) and **37 data declarations keyed by identity** (`NEW_YORK_SIFMA`, `TARGET`,
    …); currency → calendar is a lookup (`calendar_for_currency`), C1. The per-rule `observe` flag is
    lifted to the calendar's `Observance` regime.
  - Oracles: published holiday dates; US-vs-UK Saturday divergence; Juneteenth `since=2021`; Danish
    Store Bededag `until=2023`; Christmas/Boxing collision cascade; Tokyo furikae; Fri–Sat weekend;
    business-day adjustment (following/modified/nearest); joint-calendar union. `verify.py layers`
    green (L0 finance-free).
  - slice: `calendars` (Topic 0 S1)

## [0.55.0] - 2026-07-18

### Changed
- **Topic 0, Slice 0 — `ng-parking`: the whole ng tree parked, rebuild starts clean.** The pre-topic
  ng tree (54 modules + tests) is moved wholesale to `ng_parked/` and `src/pricebook_ng/` is reset to a
  bare package seed. **No behaviour claim** — this is a structural reset, not a feature change.
  - **Why:** the parked tree was built without `RollRule`, without identity-keyed calendars, without
    dual adjusted/unadjusted schedules. Editing it forward would inherit those decisions — the exact
    error we forbid with the quarry. `ng_parked/` is now a **content source only** (conventions,
    formulas, edge cases, oracle reference values); its *organisation carries no authority*
    (CLAUDE.md: mine for content, never for structure). `ng_parked/MANIFEST.md` maps each parked
    module → the topic that rebuilds it → its re-base oracle.
  - **New gate `verify.py layers`** — semantic layer conformance: L0 (`foundation/`) must be
    finance-free (no strikes/vols/payoffs/discounting). `acyclic` proves dependency *direction*; this
    catches a module on the *wrong layer* — the drift `foundation/black.py` (Black-76 at L0) slipped
    through. Wired into CI.
  - **CI repointed:** dropped the `--layer 6` tier (it guarded the now-parked tree); added the `layers`
    gate; the tree climbs from L0 as Topic 0 populates it.
  - Two quarry modules retired before the reset stay ticked (`core/numerical_config`,
    `fixed_income/fixed_leg`). Prior drawdown detail moves to per-topic manifests (redesign/13 §3.4).
  - slice: `ng-parking` (Topic 0)

## [0.54.0] - 2026-07-18

### Added
- **ZCIS serialisation (CP-3 tail, build-early per §4.5) — no drawdown change.** `ZeroCouponInflationSwap`
  gains `to_dict`/`from_dict` (+ `schema_version`), added to the property-based round-trip sweep.
  - **Retire-read: `fixed_income/inflation.py` is held a PARTIAL cross, NOT ticked** (drawdown stays
    7/768). It is a **multi-product module** (`CPICurve`, `ZCInflationSwap`, `YoYInflationSwap`,
    `InflationLinkedBond`); ng supersedes only the ZCIS (`ZeroCouponInflationSwap` + `ZCISEngine`, real
    curve via A5 `MarketKey(INFLATION, index)` + Fisher, which also supersedes `CPICurve` — 0 external
    consumers). But **`YoYInflationSwap` is `dead`** (0 production consumers, 2 test-refs) and
    **`InflationLinkedBond` has 2 production consumers** — `desks/api` + `inflation_indices` (both
    un-crossed) → `deferred→` those. Ticking would delete a module that defines a whole product ng never
    built — the first such case, so it is **routed to Cowork at CP-4** (multi-product-module retire
    pattern) rather than ticked unilaterally.
  - Oracles: dict round-trip; schema version present / absent-reads-v1 / future-rejected.
  - quarry: `python/pricebook/fixed_income/inflation.py` (ZCIS only) · slice: `serialisation-zcis`

## [0.53.0] - 2026-07-18

### Removed
- **`fixed_income/fixed_leg.py` retired → seventh quarry retire — CP-3 tail (drawdown 6 → 7/768).**
  A **no-code tick** (§4.5 "several may tick immediately"): ng already fully supersedes it, so no runtime
  change — only the reconciliation-map + version ledger advance.
  - **Superseded:** the `Cashflow` atom was **promoted** to `foundation/cashflow.py` (ng stores amount
    as `Money` + an `Accrual` period; the quarry's decomposed notional/rate/year_frac is shed —
    year_frac derives from the accrual); `FixedLeg` construction → `products/leg.py`
    `fixed_coupon_cashflows`; the container → `products/swap.py FixedLeg`; `FixedLeg.pv` → the engine.
  - **Consumer analysis (§4):** the 3 external quarry `FixedLeg(` sites are `swap.py` (un-crossed →
    `deferred→swap`) and `ois.py`/`bond.py` (already deletable). `annuity`/`weighted_annuity` (RPV01)
    are consumed only by un-crossed modules (swaption, CMS, desks) ⇒ `deferred→swap`, to be exposed as
    an **engine/curve building block** when swap crosses (CLAUDE.md §3), not a product method.
  - **Oracle:** no new code ⇒ no new red; the tick rests on ng's existing green leg/cashflow/swap/bond
    oracles + the consumer analysis. `floating_leg.py` is a separate retire (projection/multi-curve
    `pv`) that travels with the swap / multi-curve slice.
  - quarry: `python/pricebook/fixed_income/fixed_leg.py` · slice: `retire-fixed-leg`

## [0.52.0] - 2026-07-18

### Added
- **FixedRateBond serialisation → sixth quarry retire — CP-3 tail (drawdown 5 → 6/768).** `FixedRateBond`
  gains `to_dict`/`from_dict` (+ `schema_version`); its quarry counterpart `fixed_income/bond.py` is
  now **deletable**. *The heaviest retire so far — flagged for Cowork spot-check at CP-4.*
  - **Consumer-analysis retire-read (§4/§4.5):** ng supersedes the bond **product** (coupon+redemption
    cashflows) + curve pricing (`DiscountingEngine`) + **accrued / clean-vs-dirty** (the engine's A2
    decomposition, `PricingResult.accrued`/`clean`). All 8 production instantiations of the quarry
    `FixedRateBond` are in **un-crossed** modules (`desks/api` ×6, `benchmark_bonds`, `sukuk`).
  - **`deferred→` (whole yield-analytics surface, no crossed consumer):** `yield_to_maturity`,
    `macaulay/modified_duration`, `convexity`, `dv01_yield`, `price_from_yield_sc`, `irr_sc`,
    `risk_factor_sc` → `desks/api` + `benchmark_bonds` + `sukuk`. Per the spine these are **L4/L5 engine
    analytics, not product methods** — built when a consumer crosses, never cloned onto the product
    (§6b). `from_convention` → `composite_convention` + `esg_bonds` + `supranational` + `sovereign_bonds`.
    Forward-linked on those backlog rows.
  - **`deferred→persistence`:** serialisation itself (added early per §4.5 build-as-you-go — cheap,
    already in the module; never blocks the tick).
  - Serialisation reuses the shared `Cashflow`/`Money` encoders; `FixedRateBond` added to the
    property-based round-trip sweep.
  - Oracles: dict round-trip; schema version present / absent-reads-v1 / future-rejected.
  - quarry: `python/pricebook/fixed_income/bond.py` · slice: `serialisation-bond`

### Changed
- **Property-based serialisation oracle (CP-3 ruling §4.2).** A hypothesis sweep
  (`tests_ng/L4/test_serialisation_property.py`) round-trips *generated* instances of every
  serialisable type — `NumericalConfig`, `FixedCashflow`, `Deposit`, `ForwardRateAgreement`,
  `OvernightIndexSwap`, `FixedRateBond`, and the shared `Cashflow`/`Accrual`/`Money` atoms — asserting
  `T.from_dict(x.to_dict()) == x` for all valid `x`, strengthening the per-type one-example checks to
  "round-trips any instance". (The property run also confirmed the vanilla builders correctly fail-fast
  on day-counts needing extra context — `ACT_ACT_ICMA` coupon anchors, `BUS_252` a calendar — which
  are excluded from builder-based generation.)

## [0.51.0] - 2026-07-18

### Added
- **FixedCashflow serialisation → fifth quarry retire — CP-3 #5 (drawdown 4 → 5/768).** `FixedCashflow`
  gains `to_dict`/`from_dict` (+ `schema_version`) — the genuine residual that lets it **supersede the
  quarry `fixed_income/zero_coupon_bond.py`**: a zero-coupon bond *is* a single fixed cashflow
  (`Face·DF(T)`), which the `DiscountingEngine` already prices.
  - **Consumer-analysis retire-read (§4):** the quarry `ZeroCouponBond` has **no external production
    instantiation** (the only `ZeroCouponBond(` is its own docstring; production builds flow through
    `sovereign_bonds.py` via `from_convention`). Its money-market analytics
    (`price_from_yield_simple/discount_rate/continuous`, `yield_simple/continuous`, `modified_duration`,
    `dv01`) have **zero production consumers** — only `tests/test_sovereign_bonds.py` — and the quarry
    already has a dedicated `fixed_income/tbill.py` (`TreasuryBill`) as the real home for T-Bill
    conventions, so ZCB's copies are **`dead` duplicates**.
  - **`shed:` `ZeroCouponBond.from_convention` = `deferred→sovereign_bonds`** — the one production
    consumer is `sovereign_bonds.py:479` (un-crossed quarry); the obligation forward-links onto the
    sovereign-bonds crossing (its row in `quarry_reconciliation.md`), it is not owed by this retire.
  - Serialisation reuses the shared `Cashflow`/`Money` encoders (no new lifting needed).
  - Oracles: dict round-trip; schema version present / absent-reads-v1 / future-rejected.
  - quarry: `python/pricebook/fixed_income/zero_coupon_bond.py` · slice: `serialisation-zcb`

## [0.50.0] - 2026-07-18

### Added
- **OIS serialisation → fourth quarry retire — CP-3 #4 (drawdown 3 → 4/768).**
  `OvernightIndexSwap` gains `to_dict`/`from_dict` (+ `schema_version`), the genuine residual that
  supersedes the quarry `fixed_income/ois.py`, now deletable.
  - **Consumer-analysis retire-read (§4):** the quarry `OISSwap` has 2 production instantiations —
    `desks/api.py:641` (un-crossed ⇒ `deferred→desks/api`) and `OISConvention.create_swap` (inside the
    quarry ois module itself ⇒ retires with it). ng supersedes single-curve pricing via `engine/ois.py`
    (`OIS == vanilla IRS`). The `pv_ctx` multi-currency `discount_curves` branch is a multi-curve role
    ⇒ deferred. Serialisation (DB `from_dict` dispatcher) is the real residual.
  - **`shed:` `OISSwap.from_convention` = `dead`** — sole caller is
    `tests/test_convention_factory.py` (a quarry test); no production consumer.
  - **Rule of two fired twice:** OIS's `FixedLeg` coupons carry `Accrual` and `Cashflow`, so
    `Accrual.to_dict`/`from_dict` (consumers: FRA + coupon cashflows) and `Cashflow.to_dict`/`from_dict`
    (consumers: deposit + OIS legs) are lifted to `foundation/cashflow.py`; FRA and deposit refactored
    to use them under their green oracles. `FixedLeg`/`FloatLeg` encoding stays inlined in OIS (its only
    serialising consumer) — lift a shared leg encoder when the vanilla swap serialises.
  - Oracles: dict round-trip; schema version present / absent-reads-v1 / future-rejected.
  - quarry: `python/pricebook/fixed_income/ois.py` · slice: `serialisation-ois`

## [0.49.0] - 2026-07-18

### Added
- **FRA serialisation → third quarry retire — CP-3 #3 (drawdown 2 → 3/768).** `ForwardRateAgreement`
  gains `to_dict`/`from_dict` (+ `schema_version`), the genuine residual that supersedes the quarry
  `fixed_income/fra.py`, now deletable.
  - **Consumer-analysis retire-read (per the new §4 phantom-residual rule):** the quarry `FRA` class
    has a single production instantiation — `desks/api.py:273` `FRA(...).pv(curve, projection_curve)`,
    a **multi-curve** desk path. `desks/` is un-crossed quarry ⇒ that role is `deferred→desks/api`
    (travels with the desks crossing / the multi-curve slice), **not a residual now**. ng already
    supersedes single-curve FRA pricing (`engine/fra.py`, incl. seasoned fixings). The only
    production-reachable reconstruction path is the DB dispatcher (`db.py from_dict`) ⇒ serialisation
    is the real residual, exactly as for deposit.
  - **`shed:` `FRA.from_convention` = `dead`** — sole caller is `tests/test_convention_factory.py::`
    `test_fra_from_convention` (a quarry test, retires with the quarry); no production consumer.
  - **Rule of two fired:** FRA is the second serialising product, so `Money.to_dict`/`from_dict` is
    lifted to `foundation/money.py` and both deposit + FRA use it. The `Accrual` (FRA) and `Cashflow`
    date (deposit) encodings stay inlined — one consumer each — to be lifted at their own second use.
  - Oracles: dict round-trip; schema version present / absent-reads-v1 / future-rejected.
  - quarry: `python/pricebook/fixed_income/fra.py` · slice: `serialisation-fra`

## [0.48.0] - 2026-07-18

### Added
- **Deposit serialisation → second quarry retire — CP-3 #2 (drawdown 1 → 2/768).** `Deposit` gains
  `to_dict`/`from_dict` (round-trippable wire form + `schema_version`), the genuine residual that lets
  it **supersede the quarry `fixed_income/deposit.py`**, now marked deletable.
  - **Finding (refines the CP-3 #2 ruling):** the ruling named *conventions/RateIndex* as deposit's
    residual, but the evidence says otherwise. The quarry `Deposit` class has **zero production
    consumers** (`grep 'Deposit(' python/pricebook` = 0 instantiations); it is exercised only by
    quarry tests. ng already supersedes every role: the product cashflows + `DiscountingEngine`
    pricing, and the **curve-pillar role via `DepositQuote`** in `bootstrap_discount_curve`
    (`1/(1+rτ)` closed form). The only reconstruction path that is production-reachable is the DB
    dispatcher (`db.py` `from_dict`) → **serialisation** is the real residual (same as CP-3 #1),
    deferred-consumed by the not-yet-crossed persistence/data-spine slice.
  - **`shed:` `Deposit.from_convention` = `dead`** — `grep 'Deposit.from_convention' python/` finds a
    single caller, `tests/test_convention_factory.py::test_deposit_from_convention` (a quarry test that
    retires with the quarry); no production consumer. `discount_factor`/`pv`/`pv_ctx`/`year_fraction`
    properties: no production consumer (0 `Deposit(` sites), superseded by ng.
  - **No conventions/RateIndex built** — it is not deposit's residual and ng has no present consumer
    for it (§6b / "watch sprawl": a cross-cutting slice must retire a module, not add speculative
    infra). Conventions re-aims at its genuine consumer (per-currency curve construction) when crossed.
  - Oracles: dict round-trip; schema version present / absent-reads-v1 / future-rejected.
  - Serialisation stays per-class (rule of two — `Money`/`Cashflow` encoding inlined; lift a shared
    helper at the second product, FRA/swap, CP-3 #3).
  - quarry: `python/pricebook/fixed_income/deposit.py` · slice: `serialisation-deposit`

## [0.47.0] - 2026-07-18

### Added
- **Serialisation → first quarry retire — CP-3 #1 (drawdown 0 → 1/768).** `NumericalConfig` gains
  `to_dict`/`from_dict` (round-trippable wire form + `schema_version`: absent = legacy v1, a newer
  version is refused not misread) and a `replace` convenience — the genuine residual that let it
  **supersede the quarry `core/numerical_config.py`**, now marked deletable.
  - Per the ratified "deletable = supersede-with-evidence" bar: the quarry's 12 extra knobs
    (`pde_*`, `cos_*`, `tree_steps`, `mc_antithetic/sobol/bridge`, `integration_*`, `rootfinder_*`,
    `extra`) are shed **`dead`** — `grep '\.<knob>' python/` finds **zero production consumers** (only
    the quarry module's own unit test + a same-named local kwarg). The quarry config was a
    half-adopted abstraction (its docstring: adoption "incremental"); ng correctly omitted them (§6b).
    Full `shed:` evidence in `quarry_reconciliation.md` (flagged for Cowork's close review).
  - Oracles: dict round-trip; schema version present / absent-reads-v1 / future-rejected; `replace`.
  - Serialisation is per-class (rule of two — no shared framework at one serialisable type yet).
  - quarry: `python/pricebook/core/numerical_config.py` · slice: `serialisation-numerical-config`

## [0.46.0] - 2026-07-17

### Added
- **OIS + `RateCurve` protocol + `curve.forward_rate` (L1/L2/L4) — CP-2c #4, closes the CP-2b
  ruling §4.1/§4.2.** `OvernightIndexSwap` + `OISEngine`: fixed vs compounded-overnight float.
  Single-curve, the compounded rate equals the curve forward, so the OIS **reprices identically to
  the vanilla IRS**.
  - **`curve.forward_rate(d1, d2, day_count)`** (§4.1) — the simply-compounded forward
    `(P(0,d1)/P(0,d2)-1)/τ` as a curve building block, now that OIS is the 2nd forward consumer.
  - **`RateCurve` protocol** (§4.2, `df`+`zero_rate`+`instantaneous_forward`+`forward_rate`) —
    distinct from `CurveHandle` (df-only / survival); consumers are Hull-White (now typed `RateCurve`,
    duck-type removed) and OIS.
  - Extracted a shared **`float_leg_pv`** used by both `SwapEngine` and `OISEngine` (forward via
    `curve.forward_rate`, seasoned via fixings); `FRAEngine` also moved onto `curve.forward_rate`.
    All byte-identical (255 green).
  - Oracles: `forward_rate` == DF-ratio forward (flat + bootstrapped); par OIS → 0; OIS == vanilla IRS.
  - **Deletable-bar read** (`fixed_income/ois.py`): residual before deletable — currency conventions
    (SOFR/SONIA/ESTR), `bootstrap_ois`, par_rate/annuity/dv01, daily-fixing compounding, multi-curve
    basis. Logged in the map. Drawdown 0/768.
  - quarry: `python/pricebook/fixed_income/ois.py` · slice: `ois-spine`

## [0.45.0] - 2026-07-17

### Added
- **Deposit product (L2) — CP-2c #3, fixed_income spine.** A money-market deposit modelled as its
  two cashflows (`−N` at start, `N·(1+rate·τ)` at maturity) priced by the existing `DiscountingEngine`
  — **no bespoke engine**. A2 reconciles the views: a forward deposit reprices to zero at par; a spot
  deposit's principal-today is realized cash (excluded from the mark), so the mark is the redemption
  value `N` at par. `Deposit(face, rate, cashflows)` — 3 fields (under the new `fields` gate).
  - Oracles: forward par → 0; spot par → principal; off-par closed form; par → 0 on a bootstrapped curve.
  - **Deletable-bar read** (`fixed_income/deposit.py`): ng covers pricing + forward/temporal (the quarry
    values the redemption only). Residual before deletable: convention builder, implied-DF-as-method
    (ng has it via `DepositQuote`), `pv_ctx`, serialisation — logged in `quarry_reconciliation.md`.
    Drawdown 0/768.
  - quarry: `python/pricebook/fixed_income/deposit.py` · slice: `deposit-spine`

## [0.44.0] - 2026-07-17

### Added
- **Seasoned FRA via `FixingHistory` (L4) — CP-2c #2, fixings / seasoned-float.** The `FRAEngine`
  now handles the temporal cases (A2): a forward-starting/spot period uses the curve forward; a
  **seasoned** period (`accrual.start < valuation`) uses the realized reset looked up in the
  snapshot's `FixingHistory`; a fully-paid period (`end <= valuation`) settles to PV 0 (the shell
  remembers the realized cash). First `FixingHistory`-consuming engine — the swap float leg and the
  L6 float realized P&L follow the same pattern.
  - Oracles: seasoned FRA prices to `face·τ·(fixing−K)·DF(end)`; zero at `K = fixing`; missing fixing
    → `PricingFailure`; fully-paid → 0; forward FRA unchanged.
  - **Deletable-bar read** (`fixed_income/fra.py`): quarry ISDA-settle-at-start equals ng
    end-settle for single-curve (`DF(start)/(1+fwd·τ)=DF(end)`), and the quarry FRA lacks fixings
    (ng now ahead). Residual before deletable: multi-curve `forward_rate(projection_curve)`,
    `par_rate`/`pv_ctx`, convention builder — logged in `quarry_reconciliation.md`. Drawdown 0/768.
  - quarry: `python/pricebook/fixed_income/fra.py` · slice: `fra-seasoned-fixings`

## [0.43.0] - 2026-07-17

### Added
- **`verify.py fields` merge gate (CP-2c #1).** The dataclass-field analogue of `PLR0913` (which
  only sees function args): a value dataclass has **≤5 fields** unless it carries an explicit
  `# fields-exempt: <reason>` marker. AST-based, scans all `src/pricebook_ng`; added to `verify all`
  and `redesign/09` / `CLAUDE.md §3b`.

### Changed
- **Product field-bundling (behaviour-preserving).** Closes the FRA smell flagged at the CP-2b
  checkpoint. Bundled loose primitives into value objects that already exist —
  `Money` (notional/strike + currency) and `Accrual` (start + end + day_count):
  - `ForwardRateAgreement` 7→4 (`face: Money`, `accrual: Accrual`); `CDS`, `ZeroCouponInflationSwap`
    6→5 (`face: Money`); `EquityOption`, `CommodityOption` 6→5 (`strike: Money`, the strike price).
  - Engines read `.face.amount`/`.face.currency`/`.strike.amount`/`.accrual.*`; **all PVs byte-identical**
    (243 green), guarded by every product oracle.
  - Legit-wide aggregates carry `# fields-exempt:` markers: `MarketSnapshot` (A5 shape), `XvaReport`
    (output record), `XvaReportConfig` (config).
  - Not a parity slice (no quarry module retired) — code-quality; drawdown unchanged 0/768.
  - slice: `product-field-bundling`

## [0.42.0] - 2026-07-17

### Added
- **FRA — forward rate agreement (L2 product + L4 engine) — CP-2b #3, fixed_income spine.**
  `ForwardRateAgreement` (pure data) + `FRAEngine`: pay fixed, receive the simply-compounded
  forward `L(T1,T2) = (P(0,T1)/P(0,T2)-1)/τ` over one period, settled at T2 —
  `PV(pay-fixed) = notional·τ·(L-K)·P(0,T2)`, composed from the curve's discount factors so it
  prices on any curve (flat **or** bootstrapped — the general-curve payoff for the spine).
  - Oracles: a par FRA (K = L) reprices to zero; off-par matches the closed form; receiving fixed
    flips the sign; and the implied forward reprices to par on a bootstrapped curve.
  - Scope: forward-starting / spot (`accrual_start ≥ valuation`); a seasoned FRA needs a fixing
    (`FixingHistory`, a later slice, like the seasoned float leg).
  - **Parity:** first new fixed_income vanilla under the parity-depth mode; steps `fixed_income/fra`
    toward deletable. bond/swap already price on general curves; deposit / OIS / seasoned-float remain.
  - quarry: `python/pricebook/fixed_income/fra.py` · slice: `fra-spine`

## [0.41.0] - 2026-07-17

### Changed
- **General-curve Hull-White (L3) — CP-2b #2, the biggest single parity gap under the XVA stack.**
  HW no longer asserts `FlatDiscountCurve`: it reads `df` **and** `instantaneous_forward` from any
  curve, replacing the flat `r0` with the market forward `f(0,t)` in `alpha(t)` and the `zero_bond`
  reconstitution, and taking its time axis from `market.valuation_date`. `forward_short_rate` and
  the risk-neutral path simulator are now **date-based** (they need `f(0,t)`), and their callers
  (`swaption_mc`, the exposure/MPOR engines) pass dates.
  - **Byte-identical on a flat curve** (`f(0,t)=r0`), so every existing swaption / exposure / XVA /
    measure-consistency / MPOR oracle stays green (239 total).
  - New oracles on a **bootstrapped** curve: the model refits the initial curve
    (`zero_bond(0,S,f(0,0)) == P^M(0,S)`); ZCB-option put-call parity; a flat-pillars curve matches
    `FlatDiscountCurve` exactly; and the analytic swaption == the MC swaption (`rel=2%`).
  - **Parity:** the whole XVA/exposure stack can now run on a real bootstrapped curve, not just the
    flat skeleton — steps `models/hull_white` toward deletable (term-structure vol + per-currency +
    trees remain the gap). Drawdown still 0/768 (narrowed).
  - quarry: `python/pricebook/models/hull_white.py` · slice: `general-curve-hw`

## [0.40.0] - 2026-07-17

### Added
- **General-curve rate accessors (L1) — CP-2b #1, the first parity-depth slice.** Both discount
  curves gain `zero_rate` (continuously-compounded `-ln P(0,t)/t`) and `instantaneous_forward`
  (`f(0,t) = -d/dt ln P(0,t)`) — the capability a general-curve Hull-White needs where the flat
  curve used a constant `r0`. `FlatDiscountCurve` returns the constant `rate`; the log-linear
  `DiscountCurve` returns a piecewise-**constant** forward per segment (exact `-slope`, not
  finite-difference), via a shared `_bracket_slope` that also backs `df`.
  - Oracles: flat curve → constant rate; constant-rate pillars → that same constant forward
    everywhere; a rising curve's segment forward equals the analytic log-DF slope and its running
    integral reconstructs `-ln df` at each pillar; `zero_rate = -ln df/t`.
  - **Parity:** steps `core/discount_curve` toward deletable (adds 2 of its 3 rate accessors);
    `forward_rate` (simply-compounded) and pluggable interpolation remain the recorded gap.
    Unblocks CP-2b #2 (general-curve HW). Drawdown still 0/768 (partial cross, gap narrowed).
  - quarry: `python/pricebook/core/discount_curve.py` + `curves/` · slice: `general-curve-rates`

## [0.39.0] - 2026-07-16

### Added
- **Consolidated XVA report (L6, A6.2)** — completes the CP-1 cluster. `xva_report(swaps, model,
  numerics, config)` in `shell/xva_report.py` simulates a netting set's exposure **once** and
  returns every adjustment (CVA/DVA/BCVA/FVA/KVA/MVA) + the EE/PFE/EAD profiles, consolidating the
  six separate L5 calls. Lives at L6 because a per-counterparty netting set is a book of trades.
  - Netting-set exposure: new `netting_set_exposure(swaps, model, numerics, pfe_quantile)` (L5)
    sums each swap's value on **shared paths** so offsetting trades net; KVA uses the netting-set
    SA-CCR EAD runoff (`netting_set_ead` as-of each grid date). The economy (discount + both
    parties' survival curves) rides on `model.market` (A5), read directly by the integrators.
  - Oracles: a single-trade netting set reproduces each standalone L5 value exactly (one pass, same
    draws); a payer + mirror receiver nets the portfolio exposure to zero, collapsing the netted CVA.
  - Refactor: `_simulate_swap_values` is now the single-swap case of `_simulate_netting_set`
    (byte-identical, guarded by the measure/BCVA/stochastic reproducibility oracles).
  - quarry: `python/pricebook/risk/` + `desks/` · slice: `xva-report`

## [0.38.0] - 2026-07-16

### Added
- **L6 trade lifecycle — the first vertical up into the shell (A6.2).** Extends the A3
  Trade/Book/benefit-table stub to the full realized-vs-mark split. New `Trade.mark(market,
  numerics, engine)` (sum of the products' PVs + accrued as of the snapshot) and `Book.value(...)`
  (the book's mark = Σ trade marks, linearity), sharing a `_combine` helper; `BookedTrade.value`
  now delegates to `Trade.mark` and remembers the observed mark.
  - Oracle over a bond's life: at issue the mark equals the full discounted PV (realized = 0);
    mid-life the mark prices **only future flows** (the engine excludes the paid ones, A2) while
    the benefit table holds the realized cash; **dirty = clean + accrued** at a mid-period date;
    at maturity mark = 0 and realized = the total nominal; **book mark & realized are linear**
    across trades. So `realized + mark` is the trade's total economics (A3).
  - Deferred: SQLite persistence of the benefit table (`pnl_history` behind the persistence
    interface) is its own data-spine slice; float-leg realized needs fixings (seasoned-float slice).
  - quarry: `python/pricebook/core/book.py` + `pnl_history` · slice: `l6-trade-lifecycle`

## [0.37.1] - 2026-07-16

### Added
- **Measure-consistency binding oracle (L5, Amendment A6.1).** The exposure stack runs two
  simulators — the per-date **forward-measure** engine (EE/PFE profiles) and the **risk-neutral
  joint-path** engine (MPOR). A6.1 rules them one model under a change of numeraire
  (`E^Q[D·max(V,0)] = P(0,t)·E^{T_t}[max(V,0)]`), so they may never silently diverge. The new
  test binds them: the risk-neutral joint-path marginal, shifted by the analytic forward-measure
  drift `m(t)`, reproduces the forward-measure EE **and** PFE per date (two independent MC
  estimates of the same discounted exposure, agreeing to `rel=4%`). No behaviour change — the
  `_simulate_rate_paths` docstring/provenance now document the ratified relationship.
- Ratified **Amendment A6** in `redesign/02_spine.md` (exposure measure + first-L6 rulings) and
  the **checkpoint-&-review cadence** (`redesign/11`, `CLAUDE.md §6`): stop at ≤6 slices or a
  capability-cluster boundary; every checkpoint carries an oracle audit, quarry drawdown `N/768`,
  a challenge-me list, and a smell/debt scan.
  - slice: `measure-consistency-oracle` (CP-1, first of the A6 cluster)

## [0.37.0] - 2026-07-16

### Added
- **MPOR path-simulated exposure (L5)** — the residual exposure that survives full
  collateralisation. At a counterparty default the collateral reflects the value one margin
  period of risk ago, so `E_mpor(t) = mean max(V(t) - V(t - MPOR), 0)`. This needs the joint
  distribution of `V(t)` and `V(t-MPOR)`, so the slice adds `_simulate_rate_paths` — a
  **risk-neutral joint-path simulator** (exact Ornstein-Uhlenbeck steps on the HW state,
  `r = x + α(t)`) — and `mpor_exposure(swap, model, numerics, mpor_days)`.
  - Oracles: the path simulator reproduces the analytic HW/OU moments — marginal mean `α(t)`,
    variance `σ²(1-e^{-2at})/2a`, and cross-date covariance `e^{-a(t-s)}·var(s)`; a zero gap
    gives exactly zero exposure; exposure grows with the gap; it feeds a positive CVA.
  - **Provisional measure choice (flagged for review):** the EE/PFE profiles use per-date
    forward measure; this path simulator uses the risk-neutral measure. For the local MPOR
    *difference* the choice is second-order, but unifying the exposure stack onto one measure
    is a design question for the L5 brainstorm — see the handoff.
  - Scope: single swap, zero threshold; float valued via `notional - couponbond` at the
    pre-gap date too (float ≈ par, O(MPOR) error). Netting-set / threshold MPOR is a refinement.
  - quarry: `python/pricebook/risk/` · slice: `mpor-paths`

## [0.36.0] - 2026-07-16

### Added
- **Margined (collateralised) exposure (L5)**. `collateralized_exposure(swap, model, numerics,
  threshold)` models a two-way CSA with variation margin and an uncollateralised threshold `H`:
  collateral posts the mark beyond `H`, so exposure is capped — `E_coll = min(max(±V, 0), H)`.
  `H = 0` is fully collateralised (exposure 0); a huge `H` recovers the uncollateralised
  `exposure_profiles`. Feeds a collateralised CVA/DVA below the uncollateralised one.
  - Oracles: a huge threshold reproduces `exposure_profiles` exactly (same draws); zero
    threshold gives zero exposure; `σ=0` caps the deterministic exposure exactly (and the cap
    bites); collateralised CVA < uncollateralised CVA.
  - Fourth consumer of the extracted `_simulate_swap_values` (means, PFE quantiles, collateral).
  - Scope: marginal (per-date) model — the margin-period-of-risk close-out gap that leaves
    residual exposure under full collateralisation needs joint-path simulation, a later slice.
  - quarry: `python/pricebook/risk/` · slice: `margined-exposure`

## [0.35.0] - 2026-07-15

### Added
- **PFE-quantile / dynamic-IM engine (L5)**. `pfe_profile(swap, model, numerics, quantile)`
  returns the potential future exposure at confidence `q`: `PFE_q(t_j)` = q-quantile of
  positive exposure `max(V(t_j), 0)` across the simulated paths — the exposure tail the EPE
  averages over, and a high-quantile PFE doubles as a dynamic initial-margin proxy that feeds
  `mva`.
  - Oracle: because the remaining swap value is monotonic in the Gaussian short rate, the
    q-quantile of V equals V at the q-quantile of r — so `PFE_q(t_j) = max(V(t_j; r_q), 0)`
    with `r_q = forward_short_rate(t_j, Φ⁻¹(q))`, matched by MC (`rel=2%`); `σ=0` collapses it
    to the deterministic exposure; PFE rises with `q` and funds a positive MVA.
  - Refactor: extracted `_simulate_swap_values` (per-path V by date), now shared by the EE
    means (`exposure_profiles`) and the PFE quantiles — byte-identical MC (same seed/draws), so
    the CVA/BCVA/stochastic-EAD oracles stay exact.
  - Scope: PFE-as-IM proxy; a margin-period-of-risk IM on ΔV is the noted refinement.
  - quarry: `python/pricebook/risk/` · slice: `pfe-quantile`

## [0.34.0] - 2026-07-15

### Added
- **MVA — margin valuation adjustment (L5)**, completing the XVA family. `mva(im, snapshot,
  key, funding_spread)`: `MVA = s_F·Σ IM(t_i)·DF(t_i)·S(t_i)·τ_i` — the funding cost of posting
  initial margin over the trade's life, the **same survival-weighted funding annuity** as FVA
  and KVA (now three consumers of the shared `_annuity_adjustment`), with the IM profile in
  place of net exposure / capital.
  - Oracles: MVA on unit IM equals `s_F · RPV01`; linearity in spread and IM; a vertical where
    an IM profile taken as the SA-CCR PFE (AddOn) runoff feeds MVA to a positive charge matching
    its annuity.
  - `IM(t)` is an input — generating it (SIMM, or a dynamic MC-quantile IM over the margin period
    of risk) is the upstream slice, as the exposure engine is upstream of CVA.
  - quarry: `python/pricebook/risk/` · slice: `mva`

## [0.33.0] - 2026-07-15

### Added
- **Netting-set SA-CCR (L5)**. `netting_set_ead(trades, valuation_date)` aggregates a
  netting set of IR swaps (each `(swap, mark)`) the Basel way: signed effective notionals
  `D = δ·notional·SD·MF` (δ = +1 payer / −1 receiver) net within maturity buckets (<1y,
  1–5y, >5y) and combine across buckets with the supervisory correlations
  (`√(ΣD² + 1.4(D₁D₂+D₂D₃) + 0.6·D₁D₃)`), while the marks net into one replacement cost.
  - Oracles: a one-trade set equals single-trade `saccr_ead`; a payer + mirror receiver
    perfectly nets (signed notionals and marks cancel → EAD 0); a two-bucket set matches
    the hand-computed correlation aggregation; netting is sub-additive vs standalone EADs.
  - Refactor: extracted `_effective_notional_magnitude` and `_ead_from_addon` (the
    RC/multiplier/EAD assembly), now shared by the single-trade and netting-set paths.
  - Scope: single-currency IR hedging set, unmargined, no collateral — margined MF, other
    asset classes, and collateral haircuts remain later refinements.
  - quarry: `python/pricebook/risk/` · slice: `netting-saccr`

## [0.32.0] - 2026-07-15

### Added
- **Stochastic-mark SA-CCR EAD profile (L5)** — unifies the two halves of the capital
  stack. `stochastic_ead_profile(swap, model, numerics)` sets SA-CCR's replacement cost to
  the MC exposure engine's expected positive exposure instead of the ATM zero:
  `EAD(t_j) = α·(EPE(t_j) + AddOn_remaining(t_j))`. Because `EPE ≥ 0` pins the multiplier at
  1, this is `forward_ead_profile` with `mark = EPE(t_j)`, and decomposes exactly as
  `forward_ead(t_j) + α·EPE(t_j)` — both pieces already oracle-checked.
  - Oracles: exact decomposition into the deterministic PFE profile plus `α·EPE`; it dominates
    the ATM profile; and its KVA charge exceeds the ATM one (expected exposure adds capital).
  - Closes the loop end-to-end: MC exposure → SA-CCR RC → capital profile → KVA.
  - quarry: `python/pricebook/risk/` · slice: `stochastic-ead`

## [0.31.0] - 2026-07-15

### Added
- **Forward SA-CCR EAD profile → KVA (L5)** — closes the SA-CCR → capital → KVA loop.
  `forward_ead_profile(swap, valuation_date)` reprices SA-CCR at each future coupon date on
  the shrinking remaining trade (an EAD **runoff**), and `capital_profile(ead, risk_weight)`
  scales it to `8%·RWA` — the capital `K(t)` that `kva` charges the cost of capital on. Under
  the ATM assumption (expected mark = 0 → RC = 0, multiplier = 1) the runoff is the deterministic
  supervisory PFE, so the whole chain has a closed-form oracle.
  - Oracles: each `EAD(t_j)` equals the closed-form `α·SF·notional·SD·MF` on the remaining
    maturity (first point == single-date `saccr_ead`); the profile runs off monotonically;
    `capital = 8%·EAD·RW`; KVA on it equals the cost-of-capital annuity.
  - Refactor: `saccr_ead` now delegates to a param-level `_ead_ir(notional, S, E, mark)` shared
    with the runoff (guarded by the SA-CCR oracle, no behaviour change).
  - Scope: ATM-mark runoff; a stochastic-mark RC (expected positive exposure from the MC engine)
    is the noted refinement.
  - quarry: `python/pricebook/risk/` · slice: `forward-ead-kva`

## [0.30.0] - 2026-07-15

### Added
- **SA-CCR — Basel standardised counterparty EAD & RWA (L5)**. New `risk/saccr.py`:
  `saccr_ead(swap, mark, valuation_date) = α·(RC + PFE)` for a single-trade interest-rate
  netting set (unmargined, uncollateralised), plus `risk_weighted_assets(ead, risk_weight)`
  and `saccr_capital(rwa) = 8%·RWA`. RC = max(V,0); PFE = multiplier·AddOn with the IR
  supervisory factor (0.5%), supervisory duration (5% decay), unmargined maturity factor,
  and the 5%-floored multiplier; α = 1.4.
  - Oracles: a 10y ATM $100mm IRS has EAD ≈ 5.5% of notional (the published SA-CCR anchor);
    RC adds `α·mark` in the money; deep out-of-the-money the multiplier hits its 0.05 floor
    (EAD → `0.05·EAD_atm`); RWA/capital identities.
  - This is the regulatory EAD generator that both counterparty RWA and (extended to a
    forward profile) the KVA capital input build on.
  - Scope: single IR trade, one hedging set, no collateral/margin — netting-set buckets +
    correlations, margined MF, other asset classes, and collateral haircuts are later slices.
  - quarry: `python/pricebook/risk/` · slice: `saccr`

## [0.29.0] - 2026-07-15

### Added
- **KVA — capital valuation adjustment (L5)**. `kva(capital, snapshot, key, cost_of_capital)`
  in `risk/xva.py`: `KVA = γ_K·Σ K(t_i)·DF(t_i)·S(t_i)·τ_i` — the cost of capital charged on
  the capital profile `K(t)`, discounted and survival-weighted. The **same funding annuity as
  FVA** (the CDS RPV01 structure), with capital in place of net exposure and the hurdle rate in
  place of the funding spread.
  - Oracles: KVA on unit capital equals `γ_K · RPV01`; capital proportional to EPE
    (`K = k·EPE`) integrates as expected; linearity in the cost of capital.
  - `K(t)` is an input — generating it from a regulatory model (SA-CCR EAD → RWA → capital)
    is the upstream RWA slice, exactly as the exposure engine is upstream of CVA.
  - quarry: `python/pricebook/risk/` · slice: `kva`

### Changed
- Extracted the shared **`_annuity_adjustment`** (`rate·Σ profile·DF·S·τ`) now backing both
  FVA and KVA — a survival-weighted annuity, the CDS RPV01 structure. `fva` refactored onto it
  under its green oracle, no behaviour change.

## [0.28.0] - 2026-07-15

### Added
- **FVA — funding valuation adjustment (L5)**. `fva(exposure, snapshot, key, funding_spread)`
  in `risk/xva.py`: `FVA = FCA - FBA = s_F·Σ (EPE_i - ENE_i)·DF(t_i)·S(t_i)·τ_i` — the
  funding spread carried over each interval on the **net** exposure, discounted and
  survival-weighted (funding stops on default). Reuses the same `ExposurePair` as CVA/DVA.
  - Where CVA/DVA weight exposure by a *protection leg* (default increments × `(1-R)`), FVA
    weights it by a *funding annuity* `S·τ` — the CDS RPV01 structure. So the oracle: FVA on
    unit positive exposure equals `s_F · RPV01` (the survival annuity) exactly; plus a
    symmetric-exposure zero (cost cancels benefit) and linearity in spread and exposure.
  - Scope: symmetric funding spread, single survival curve, discounting-approach FVA
    (FVA/DVA overlap and own-vs-joint survival are known modelling debates, out of scope).
  - quarry: `python/pricebook/risk/` · slice: `fva`

## [0.27.0] - 2026-07-15

### Added
- **DVA + bilateral BCVA (L5)**. `dva`, `bcva`, and the `CreditParty` bundle in
  `risk/xva.py`. DVA is the mirror of CVA — expected gain from *our own* default while
  out of the money — which is exactly the CVA integral on the **negative** exposure
  profile (ENE) against our own survival curve, so `dva` reuses `cva` and
  `bcva(exposure, snapshot, counterparty, self_party) = CVA - DVA` (net credit charge;
  value adjustment is `-BCVA`). `CreditParty(key, recovery)` keeps `bcva` under the
  5-arg ceiling.
  - Oracles: ENE of a payer swap equals EPE of the mirror receiver swap (exact, same
    simulated rates); BCVA decomposes into `CVA - DVA`; a default-free self (Q≡1) zeroes
    DVA so BCVA collapses to unilateral CVA.
  - Scope: unilateral pair (exposure ⟂ default, no first-to-default survival weighting —
    a later refinement multiplies each term by the other party's Q(t)).
  - quarry: `python/pricebook/risk/` · slice: `bcva`

### Changed
- **`expected_exposure` → `exposure_profiles`**, now returning an `ExposurePair`
  (EPE **and** ENE) from a single MC pass — ENE is free from the same paths. `ExposurePair`
  and `ExposureProfile` (docstring generalised to `E[(±V)^+]`) live in `risk/xva.py`.

## [0.26.0] - 2026-07-15

### Added
- **Monte-Carlo expected-exposure engine (L5)** — generates the real `EE(t)` profile
  that CVA consumes. `expected_exposure(swap, model, numerics)` in `risk/exposure.py`
  simulates the Hull-White short rate to each grid date under that date's t_j-forward
  measure (one exact Gaussian draw) and reprices the remaining swap analytically via
  `zero_bond`, returning an `ExposureProfile`. Closes the exposure-generation gap left
  open by the CVA slice.
  - Oracles: (1) `sigma = 0` -> `EE(t_j)` equals the deterministic forward swap value's
    positive part, exact to `1e-8`; (2) the forward-measure identity — `P(0,t_j)·EE(t_j)`
    equals the analytic co-terminal swaption expiring at `t_j` (Jamshidian), matched by
    MC within `rel=2%` at 120k paths; (3) end-to-end into `cva` (positive, finite).
  - Consequence: feeding this `EE(t)` to `cva` (which multiplies by `DF(t_j)`) yields the
    correct discounted expected exposure `Σ_j swaption(t_j)·ΔQ_j` — CVA as a swaption strip.
  - quarry: `python/pricebook/risk/` · slice: `mc-exposure`

### Changed
- **`HullWhite.forward_short_rate(t, z)`** extracted as a model capability — the exact
  t-forward-measure short-rate draw, now shared by the MC swaption and MC exposure engines
  (rule of two). `coupon_bond_cashflows` now takes a `VanillaSwap` (three consumers), and
  `SwaptionMCEngine` reuses `forward_short_rate` — pure refactors under the green MC/analytic
  swaption oracles, no behaviour change.

## [0.25.0] - 2026-07-15

### Added
- **Unilateral CVA (L5 risk & capital)** — the first XVA. New `risk/xva.py` with an
  `ExposureProfile` (`EE(t) = E[(V(t))^+]` on a time grid) and
  `cva(profile, snapshot, key, recovery)` = `(1-R)·Σ EE(t_i)·DF(t_i)·(Q(t_{i-1})-Q(t_i))`.
  Structurally a CDS **protection leg** with the unit notional replaced by the exposure
  profile — CVA is protection bought on your own counterparty exposure.
  - Keyed to the counterparty survival curve in the snapshot (A5), reached through the
    `CurveHandle` `df` capability — so a credit bump (`bump_curve`/`credit01`) already
    yields CVA sensitivity, no new machinery.
  - Oracles: unit-exposure CVA **equals** the CDS protection leg (`cds_pv` at zero
    spread, already oracle-checked) to `1e-14`; linearity in exposure; a default-free
    (Q≡1) counterparty gives zero CVA.
  - Scope (unilateral): exposure ⟂ default (no wrong-way risk), own default ignored
    (that is DVA). **Exposure generation is upstream/out of scope** — `EE(t)` is an
    input (analytic for deterministic trades, MC for optional ones — a later slice).
  - quarry: `python/pricebook/risk/` · slice: `cva`

## [0.24.0] - 2026-07-15

### Added
- **Joint HW `(a, sigma)` calibration from a cap strip** — the calibration front's
  first multi-instrument least-squares fit. `calibrate_hull_white_cap(snapshot, quotes)`
  fits both mean reversion and vol to a strip of caplet (ZCB-option) quotes by
  minimising the repricing SSE. A single caplet can't separate `a` from `sigma`; a
  strip spanning expiries can (needs ≥2 quotes). `sigma` is fitted via its magnitude
  (price depends only on `sigma^2`), so the model carries `sigma >= 0`.
  - Oracles: round-trip recovers `(a*, sigma*)` from a self-priced strip
    (`a` to `1e-4`, `sigma` to `1e-5`); fitted model reprices the strip (SSE `< 1e-14`).
  - quarry: `python/pricebook/calibration/` · slice: `hw-cap-strip`
- **`nelder_mead` (L0 numerical toolkit)** — derivative-free downhill-simplex minimiser
  (Nelder & Mead 1965), the stdlib least-squares engine behind multi-parameter
  calibration (no scipy — the ng tree stays stdlib-pure, like `bisect_root`). Converges
  on **both** a flat objective and a small simplex, so a weakly identified direction
  (HW `a`) still pins down instead of drifting. Oracle: quadratic bowl + Rosenbrock.

## [0.23.0] - 2026-07-15

### Changed
- **Rate & credit bootstraps migrated under the L3 calibration front.**
  `bootstrap_discount_curve` and `bootstrap_survival_curve` move from `market/` (L1)
  to `calibration/` (L3), joining `calibrate_hull_white` as the per-family solvers of
  the unified front (`market -> calibrate -> model`). The curve *types*
  (`DiscountCurve`, `SurvivalCurve`) and market observables (`DepositQuote`,
  `ParSwapQuote`, `CDSQuote`) and CDS leg math (`cds_pv`, …) stay at L1 — quotes and
  curves are market data, the *solvers* are calibration. Each bootstrap still reprices
  with L1 closed forms (curve `df` / `cds_pv`), never the L4 engine, so `acyclic` stays
  green with `calibration` at rank 3.
  - Pure relocation guarded by the existing reprice-to-par / reprice-to-zero oracles
    (their tests move `tests_ng/L1 -> L3` accordingly). No behaviour change; 174 green.
  - Import path change: `from pricebook_ng.calibration.discount_curve import
    bootstrap_discount_curve` (was `...market.discount_curve`); likewise survival.

## [0.22.0] - 2026-07-15

### Added
- **Unified calibration front (L3), first tenant: Hull-White vol.** New
  `calibration/` package establishing `market -> calibrate -> model -> price` (A1):
  `calibrate_hull_white(snapshot, quote, a)` fits the HW `sigma` (with mean reversion
  `a` fixed) so the model reprices a `ZCBOptionQuote` — a caplet, i.e. a European
  option on a zero-coupon bond (Brigo & Mercurio s.3.3, the textbook HW vol instrument).
  - Correctly layered: the fit reprices with the model's own analytic `zero_bond_option`,
    **not** the L4 engine — calibration depends only on L0/L1/L3 (verify `acyclic` green
    with `calibration` at rank 3). The ZCB-option price is monotone in `sigma`, so one
    bracketed root pins it; an unreachable quote (below intrinsic) raises `ValueError`.
  - Oracles: round-trip sigma recovery (`abs=1e-9`); calibrated model reprices the quote
    (`abs=1e-10`); unreachable-quote raise.
  - Establishes the front's shape (`calibrate_*(snapshot, quotes, …) -> CalibratedModel`);
    the rate/credit curve bootstraps are the sibling solvers that migrate under it next.
  - quarry: `python/pricebook/calibration/` · slice: `calibration-front`

## [0.21.0] - 2026-07-15

### Added
- **Key-rate (bucketed) dv01.** `key_rate_dv01(priceable, snapshot, numerics)` in
  `risk/greeks.py` returns per-pillar KR01 for the home bootstrapped curve — bump one
  pillar's zero at a time (`DiscountCurve.bump_pillar(i, shift)`) and reprice. Log-linear
  interpolation tents each bump between neighbours, so the buckets **partition** the
  parallel `dv01` (Σ buckets = dv01). This turns the single parallel number into a
  per-tenor risk vector a hedger can actually neutralise.
  - Oracles: (1) buckets sum to `dv01` (partition of unity); (2) a cashflow landing
    exactly on pillar j is carried only by bucket j (`KR01_j = -N·t_j·DF(t_j)·1bp`,
    neighbours ~0, since the tent vanishes at the node).
  - quarry: `python/pricebook/core/discount_curve.py` · slice: `key-rate-buckets`

## [0.20.0] - 2026-07-14

### Added
- **Bootstrapped-curve dv01.** `DiscountCurve.bumped(shift)` — a parallel zero-rate
  shift (`DF -> DF·exp(-shift·t)` at every pillar), so log-linear interpolation keeps
  the shift uniform between pillars. `dv01`/`curve01` now compute rate risk on a real
  **bootstrapped** discount curve, not only the flat curve — closing the gap noted when
  generic curve greeks landed. Same shape as `SurvivalCurve.bumped`; `bump_rate` /
  `bump_curve` dispatch through it polymorphically (no `isinstance`).
  - Oracle: `dv01 = -1bp·Σ cf_i·t_i·DF_boot(t_i)` (analytic vs central-difference on the
    actual bumped curve); plus a check that every pillar's zero rose by exactly `shift`.
  - quarry: `python/pricebook/core/discount_curve.py` · slice: `bootstrapped-dv01`

## [0.19.0] - 2026-07-14

### Added
- **Generic curve greeks** (completes the A5 unification for curves). `curve01` /
  `bump_curve` in `risk/greeks.py` — parallel-shift the curve at a `MarketKey` and
  reprice, whatever the curve type. This gives rate risk on the **FX foreign curve**,
  the **inflation real/breakeven curve** (and dividend/carry) for free.
  - Curves gained a polymorphic `bumped(shift)`: `FlatDiscountCurve` shifts its rate,
    `SurvivalCurve` shifts its hazard (`Q -> Q·exp(-shift·t)`). `bump_curve` dispatches
    through it (no `isinstance`).
  - `credit01` is now a named alias of `curve01` on a survival key; `bump_hazard` is
    folded into `SurvivalCurve.bumped`. `dv01`/`bump_rate` use the same `bumped`.
  - Oracle: `curve01` on the FX foreign curve matches `-base·spot·T·DF_base·1e-4`; on
    the inflation real curve matches `notional·(-T·DF_real)·1e-4`; `credit01 == curve01`
    on a survival key; `bump_curve` shifts only the keyed curve.

### Deferred
- Pillar-wise / key-rate bumps for a **bootstrapped** curve (today `bumped` is a flat
  rate shift); home-discount `dv01` still assumes the flat curve.

## [0.18.0] - 2026-07-14

### Added
- **Inflation — zero-coupon inflation swap (first inflation slice).**
  - `products/inflation.py`: `ZeroCouponInflationSwap` (index, notional, fixed rate,
    maturity, currency, receive/pay inflation), pure data.
  - `engine/inflation.py`: `ZCISEngine` — the inflation forward index ratio
    `I(T) = DF_real(T)/DF_r(T)` (Fisher), with the real curve keyed at
    `MarketKey(INFLATION, index)`; receiver PV = `notional·DF_r·(I(T) − (1+K)^T)`.
  - `AssetClass.INFLATION` (a real curve is the only new market data — another cheap
    keyed asset class, A5).
  - Oracle: par ZCIS (K = breakeven) reprices to zero; PV matches the formula;
    receiver = −payer; below-breakeven fixed is valuable to the inflation receiver;
    missing real curve fails.

### Deferred
- Year-on-year inflation swaps and inflation-linked bonds; a breakeven/inflation01
  greek (bump the real curve, keyed — like `credit01`); seasonality; index lag.

## [0.17.0] - 2026-07-14

### Added
- **Commodity — European commodity option (first commodity slice).** The A5 keyed
  registry made it cheap: `AssetClass.CMDTY` + a `CommodityOption` product + a thin
  `CommodityOptionEngine`, and the **greeks are free** (the generic `spot_delta` /
  `vol_vega` work with no new code — the whole point of A5).
  - `engine/spot_option.py`: shared `price_spot_option(option, asset, model, numerics)`
    — Black-Scholes on the forward `F = spot·DF_carry/DF_r`, keyed by
    `MarketKey(asset, ticker)`. Behind both the equity (carry = dividends) and
    commodity (carry = convenience yield net of storage) engines.
  - **`EquityOptionEngine` refactored** onto the shared engine (rule of two;
    behaviour-preserving — the equity oracles stay green).
  - Oracle: put-call parity ties to the commodity forward; independent Black recompute;
    `sigma -> 0` intrinsic; missing market fails; and a demonstration that `spot_delta`/
    `vol_vega` price commodity greeks with no commodity-specific code.

### Deferred
- Commodity forward, seasonality/term-structure carry, futures vs spot, a vol surface.

## [0.16.0] - 2026-07-14

### Changed
- **Amendment A5 — `MarketSnapshot` keyed market-data registry.** All market data
  except the home `discount_curve` moves into `curves` / `spots` / `vols` maps keyed
  by `MarketKey(asset: AssetClass, id: str)` (new `market/keys.py`).
  - `survival_curve` / `fx_*` / `equity_*` fields are removed; folded into the maps.
    Folding survival **adds multi-issuer** — a `CDS` now names its `issuer`, and the
    engine looks up `curves[MarketKey(CREDIT, issuer)]`. `SurvivalCurve` gains a `df`
    alias (a hazard curve is the credit-risky discount-factor curve), so it lives in
    the same `curves` map as discount/dividend/foreign curves.
  - **Greeks collapse to one generic each**: `bump_spot`/`bump_vol` +
    `spot_delta`/`vol_vega` keyed by `MarketKey` — the per-asset `fx_delta`/`equity_delta`,
    `fx_vega`/`equity_vega`, `bump_fx_*`/`bump_equity_*` are deleted. `credit01`/`bump_hazard`
    are keyed by issuer. A new asset class now adds **keys, not fields, and no new greeks**.
  - Behaviour-preserving: every FX/equity/credit/rates PV and greek is unchanged
    (all prior oracles reused). New: a `MarketKey` namespacing test (FX "EUR" ≠ equity "EUR").
  - `cds(...)` builder drops the schedule-terms arg (CDS premiums are annual ACT/360 by
    convention) and takes `issuer` — stays within the 5-arg ceiling.

## [0.15.0] - 2026-07-14

### Added
- **Equity greeks** — `bump_equity_spot` / `bump_equity_vol` + `equity_delta` /
  `equity_vega` in `risk/greeks.py`, keyed by ticker, on the same `Priceable`
  protocol as FX and rate greeks. Oracle: the bumps move only their field;
  `equity_delta` matches the analytic BS delta `quantity·DF_div·N(d1)`; a put's
  delta is negative; `equity_vega` matches the analytic Black vega
  `quantity·DF_r·F·φ(d1)·√T`.

## [0.14.0] - 2026-07-14

### Added
- **Equity — European equity option (Black-Scholes with dividends), first equity slice.**
  - `products/equity_option.py`: `EquityOption` (frozen pure data — ticker, quantity,
    strike, maturity, currency, call/put).
  - `engine/equity_option.py`: `EquityOptionEngine` — Black-Scholes as Black-76 on the
    equity forward `F = spot·DF_div/DF_r`, discounted by `DF_r`; `PricingFailure` if the
    equity market is absent; expired → 0.
  - `MarketSnapshot`: `equity_spots` / `equity_div_curves` / `equity_vols`, keyed by ticker.
  - `foundation/black.py`: shared **`black_76`** primitive (option on a forward), used by
    both the equity (BS) and FX (GK) engines.
  - Oracle: put-call parity ties to the equity forward value; matches an independent Black
    recompute; `sigma -> 0` intrinsic; ATM-forward call == put.

### Changed
- `FXOptionEngine` refactored onto the shared `black_76` (behaviour-preserving; GK oracle
  stays green).

### Deferred
- Equity greeks (delta/vega, bump `equity_spots`/`equity_vols` through the `Priceable`);
  foreign-listed equity (own currency curve); a real dividend schedule vs the flat repo
  curve; a vol surface.

## [0.13.0] - 2026-07-14

### Added
- **FX vega** — `bump_fx_vol` + `fx_vega` (∂PV per unit FX vol) in `risk/greeks.py`,
  on the same `Priceable` protocol as `fx_delta`/`dv01`. Oracle: `bump_fx_vol` moves
  only that vol; `fx_vega` matches the analytic Black vega
  `notional·DF_quote·F·φ(d1)·√T`; a long option is long vol (vega > 0).

### Changed
- Removed `fx_forward_priceable` — it was byte-identical to `discounting_priceable`;
  all FX products (forward + option) now bind via `discounting_priceable`.

## [0.12.0] - 2026-07-14

### Added
- **FX option — Garman-Kohlhagen.**
  - `products/fx_option.py`: `FXOption` (frozen pure data — base/quote legs,
    maturity, call/put) + `fx_option(...)` builder.
  - `engine/fx_option.py`: `FXOptionEngine` — GK as Black-76 on the FX forward
    `F = spot·DF_base/DF_quote`, discounted by the quote curve; `sigma -> 0`
    collapses to discounted intrinsic; `PricingFailure` if the FX market is absent.
  - `MarketSnapshot.fx_vols`: flat FX vol per pair (market data, §5.1).
  - Oracle: put-call parity ties to the FX forward PV (cross-slice); matches an
    independent Black recompute; `sigma -> 0` intrinsic; ATM-forward call == put.

### Deferred
- FX vega (bump `fx_vols` through the `Priceable`, like `fx_delta`) — trivial
  follow-up; a vol surface (strike/tenor) replacing the flat vol; American/barrier.

## [0.11.0] - 2026-07-13

### Changed
- **FX market data promoted into `MarketSnapshot`** (ruling §5.1, closing the FX
  loop as survival-in-snapshot did for credit). The snapshot now carries
  `fx_curves` (foreign-currency curves) and `fx_spots` (home units per foreign
  unit), keyed by currency. `FXForwardModel` is removed — the FX forward is a
  linear product priced with a `DiscountingModel` over the snapshot; the engine
  looks up the base curve/spot by the product's base currency and returns
  `PricingFailure` if the FX market is absent.

### Added
- **FX greeks on the `Priceable` protocol.** `bump_fx_spot` + `fx_delta` (∂PV per
  unit spot) in `risk/greeks.py`; `fx_forward_priceable` factory. The same FX
  Priceable feeds both `fx_delta` (spot bump) and the generic `dv01` (quote-curve
  bump).
  - Oracle: FX data lives in the snapshot; `bump_fx_spot` moves only that spot;
    `fx_delta` matches the analytic `base_notional·DF_base(T)`; sell = -buy; the
    same Priceable gives a non-zero `dv01`.

### Deferred
- Base-currency rate risk (bump `fx_curves`), a currency→curve map replacing the
  home/foreign split, FX options (Garman-Kohlhagen), and a `CurrencyPair` type.

## [0.10.0] - 2026-07-13

### Added
- **FX — FX forward (first FX slice).**
  - `products/fx_forward.py`: `FXForward` (frozen pure data — base leg, quote leg,
    maturity, buy/sell) and an `fx_forward(...)` builder taking a strike.
  - `models/fx_model.py`: `FXForwardModel(market, base_curve, spot)` — the quote-
    currency market + the base-currency curve + spot (quote per base).
  - `engine/fx_forward.py`: `FXForwardEngine` — values both legs in the quote
    currency by covered interest parity; buyer PV = base leg − quote leg; a matured
    forward settles to 0 (A2).
  - Oracle: a par forward (struck at `F = spot·DF_base/DF_quote`) prices to zero;
    PV matches CIP; sell = −buy; below-forward strike is valuable to the buyer;
    matured forward settles.

### Deferred
- Promote FX market data (base curve + spot) into the `MarketSnapshot` and unify
  FX greeks (delta, per-currency rate risk) on the `Priceable` protocol — the
  §5.1 follow-up, exactly as the CDS survival curve was promoted (multi-currency
  snapshot: curves keyed by currency + FX spots).
- FX options, NDFs, and a real `CurrencyPair`/quoting-convention type.

## [0.9.1] - 2026-07-13

### Changed
- Spelling fix: **`Pricable` → `Priceable`** everywhere — the `Priceable` protocol,
  the `discounting_priceable` / `hull_white_priceable` / `credit_priceable`
  factories, the module (`risk/priceable.py`), and the design docs (`CLAUDE.md`,
  `redesign/`). Pure rename, behaviour-preserving (all oracles green).

## [0.9.0] - 2026-07-13

### Changed
- **SurvivalCurve promoted into `MarketSnapshot`** (Cowork ruling §5.1). The
  credit/hazard curve is market data, so it now lives on the snapshot
  (`survival_curve`, reached through a new `SurvivalHandle` protocol, mirroring
  `CurveHandle`). `CreditModel(market, recovery)` reads it via `market.survival_curve`
  (its `survival` field is now a property).
- **Credit risk unified on the `Pricable` protocol.** `credit01` (CS01) moved into
  `risk/greeks.py` alongside `dv01`, both central-differencing a `Pricable` under a
  snapshot bump (`bump_rate` / `bump_hazard`) — one finite-difference core.
  `risk/credit_greeks.py` is removed.
  - `credit_pricable(product, recovery, engine, numerics)` builds the credit
    `Pricable`; the same pricable feeds both `credit01` (hazard bump) and `dv01`
    (rate bump) — a CDS now has rate risk *and* credit risk through one interface.
  - CDS pricing is behaviour-preserving (all prior oracles green).
  - Oracle: survival lives on the snapshot; `bump_hazard` moves only the credit
    curve; buyer credit01 > 0, seller = -buyer; matches an independent hazard FD;
    the CDS pricable also yields a non-zero rate dv01.

## [0.8.0] - 2026-07-13

### Added
- **Float-leg fixings — seasoned swaps** (consumes the `FixingHistory` A1 added).
  - The `SwapEngine` float leg is now temporality-aware: a period that already
    paid (`b <= valuation`) is settled; a period whose reset is strictly past
    (`a < valuation`) uses the realized fixing `market.fixings.get(a)`; a future
    period projects the curve forward. Missing fixing ⇒ `PricingFailure`.
  - `FloatLeg` gains `day_count` (needed to accrue a fixed current period from its
    reset); `vanilla_swap` sets it from the float schedule.
  - The old "float leg starts before valuation" guard is retired.
  - Oracle: a seasoned swap's current coupon uses the fixing and matches the
    independent per-period sum; an already-paid period is excluded; a missing
    fixing fails; a spot swap needs no fixings (behaviour-preserving).

### Deferred
- Fixing lag (reset a few days before accrual start), and swap-level accrued/clean
  decomposition on the float leg — add when needed.

## [0.7.0] - 2026-07-13

### Added
- **Monte-Carlo engine — HW swaption, `analytic vs MC convergence`** (closes the
  named oracle for the whole Hull-White arc).
  - `engine/swaption_mc.py`: `SwaptionMCEngine` prices the European swaption under
    the T0-forward measure — one exact Gaussian draw of `x(T0)` (mean `M`, variance
    `V`), reconstitutes the coupon bond via the model's `zero_bond`, averages the
    payer/receiver payoff, discounts by `P(0,T0)`. Stdlib `random`, no numpy.
  - `NumericalConfig` gains `mc_paths` and `mc_seed` (fixed seed ⇒ reproducible,
    referential transparency preserved).
  - `engine/swaption.py`: extracted the shared `coupon_bond_cashflows` helper (used
    by both the analytic and MC engines; analytic oracle preserved).
  - Oracle: MC converges to the S08 Jamshidian analytic within ~2% at 200k seeded
    paths (payer and receiver); exact at `sigma=0` (deterministic); reproducible
    under a fixed seed.

### Deferred
- Variance reduction (antithetics, Sobol), a general MC path engine for other
  products, and MC greeks — added when a product needs them.

## [0.6.0] - 2026-07-13

### Added
- **CDS credit01 / CS01** (`risk/credit_greeks.py`) — CDS PV sensitivity to a 1bp
  parallel credit-spread (hazard) shift, by central finite difference: `bump_hazard`
  scales each survival pillar by `exp(-dh*t)`, rebuilds the `CreditModel`, reprices.
  The hazard analogue of `dv01` (which bumps the discount rate).
  - Oracle: `bump_hazard` shifts each pillar by exactly `exp(-dh*t)` (anchor
    unchanged); buyer credit01 > 0 (protection gains as credit worsens); seller =
    -buyer; matches an independent hazard-bump central difference.

### Deferred
- Routing credit01 through a `survival -> PV` closure (like the `Pricable`
  factories) — added when a second hazard-sensitive product exists; today CDS is
  the only one, so it calls `CDSEngine` directly.

## [0.5.0] - 2026-07-13

### Added
- **CDS as an engine-priced product.**
  - `products/cds.py`: `CDS` (frozen pure data — premium schedule, spread,
    notional, buyer/seller) and a `cds(...)` builder.
  - `models/credit_model.py`: `CreditModel(market, survival, recovery)` — carries
    the discounting market + the bootstrapped hazard curve + recovery (A1).
  - `engine/cds.py`: `CDSEngine` — values the protection buyer via the L1 CDS leg
    math; the seller is the negative.
  - Oracle: a par CDS reprices to zero through the engine; the engine matches the
    L1 `cds_pv`; seller = -buyer; buyer value falls as the contract spread rises.

### Deferred
- CDS greeks — a `credit_pricable` factory drops the CDS onto the L5 `Pricable`
  protocol (rate dv01 today, credit01 once a hazard bump exists); add when wanted.
- Seasoned CDS (segment-and-settle on the premium leg) and quarterly premiums.

## [0.4.0] - 2026-07-12

### Added
- **Credit — hazard/survival curve + CDS bootstrap (first credit slice, L1).**
  - `market/survival_curve.py`: `SurvivalCurve` (piecewise-hazard, log-linear in
    `ln Q`, behind `survival(date)`), the single-name CDS leg math (`RPV01`,
    protection PV, `cds_par_spread`, `cds_pv` for the protection buyer), and
    `bootstrap_survival_curve` — sequential solve so each CDS reprices to zero at
    its par spread.
  - Oracle: `survival(valuation)=1`; each input CDS reprices to zero (`< 1e-10`);
    the curve-implied par spread equals each quote; survival strictly decreasing in
    (0,1]; log-linear between pillars. Mirrors the S03 discount-curve bootstrap.
  - Quarry: `core/survival_curve.py`.

### Deferred
- CDS as an **engine-priced product** (L2 `CDS` product + L3 `CreditModel` +
  L4 `CDSEngine` + greeks via `Pricable`) — the immediate follow-up, exactly as the
  discount curve preceded the bond/swap.
- Quarterly premiums, accrual-on-default, and a finer protection integral (the
  reprice-to-zero oracle is exact on the shared discretisation regardless).

## [0.3.0] - 2026-07-12

### Added
- **Risk relocated to L5 on the `Pricable` protocol** (spine structural fix).
  - `risk/pricable.py`: `Pricable` — a `snapshot -> PV` closure that risk consumes;
    factories `discounting_pricable` (linear products, any engine over a
    `DiscountingModel`) and `hull_white_pricable` (rebuilds HW under the snapshot).
  - `risk/greeks.py`: generic `dv01` (central-difference bump-and-reprice) and
    `bump_rate` (parallel shift). One `dv01` prices rate delta for a cashflow, a
    bond, a swap, and an HW swaption — the swaption rebuilds the model under the
    bumped snapshot (Amendment A1). **No `isinstance`-on-product ladders.**
  - Oracle: `dv01` matches the analytic sensitivity for a single cashflow and a
    bond (`< 1e-6`); is generic over any `Pricable` (a raw closure); and for the HW
    swaption equals a manual rebuild-and-reprice bump (the model-rebuild path).

### Changed
- The Slice-0 `risk/dv01.py` (specific to `DiscountingModel`) is replaced by the
  generic `risk/greeks.dv01` on the `Pricable` protocol; its analytic-vs-FD oracle
  is preserved through the new API.

### Deferred
- Higher greeks (gamma, vega), pillar-wise bumps of a bootstrapped curve, and
  XVA/RWA — all land on the same `Pricable` protocol in later slices.

## [0.2.0] - 2026-07-12

### Added
- **Amendment A3 — Product / Trade / Book + the benefit table (L6 shell).**
  - `shell/booking.py`: `Trade` (a collection of products + a start date), `Book`
    (a collection of trades), and `BookedTrade` with the **benefit table** —
    `realized(as_of)` sums cashflows that have already paid as actual cash, **never
    discounted**. `value(...)` aggregates the products' marks (dirty PV + accrued).
  - Oracle: realized at issue is 0; realized sums paid cashflows undiscounted;
    realized + remaining nominal = total nominal; at end of life realized = total
    and the mark is 0; a `Book` aggregates realized across trades.

### Changed
- **Renamed the L2 atom `instrument -> product`** (Amendment A3): the
  `instruments/` package is now `products/`; `FixedCashflowTrade -> FixedCashflow`;
  the engine protocol `CashflowInstrument -> CashflowProduct`. "Trade" is now an
  L6 concept (a collection of products), freeing the name. Behaviour-preserving.
- CI layer tier bumped to `--layer 6` (the slice reaches the L6 shell).

### Deferred
- Realized P&L for float legs (needs fixings) — wired with a seasoned-float slice.
- Per-product model dispatch in `Trade.value` (all products are linear today, so
  one `DiscountingModel` suffices); a registry/facade arrives when a trade mixes
  model families.

## [0.1.0] - 2026-07-12

### Added
- **Amendment A2 — valuation is temporality-aware.** The engine partitions a
  product's cashflows by `model.market.valuation_date`:
  - cashflows on or before valuation are **historical** — excluded from PV (the
    shell settles them), never discounted; the "fail on past cashflow" guard is
    retired in favour of **segment-and-settle**.
  - future cashflows discount from valuation.
  - `PricingResult` is now a **decomposition**: `pv` (dirty), `accrued`
    (earned-but-unpaid, nominal), and `clean = pv - accrued`.
  - `Cashflow` gains an optional `Accrual(start, end, day_count)`; fixed-leg
    coupons carry it, so a **seasoned bond** accrues the current period on its own
    day count.
  - Oracle (closed-form, exact): seasoned bond excludes paid coupons and matches
    the sum over remaining flows; forward-starting prices only future flows;
    accrued matches the day-count fraction; `dirty = clean + accrued`; a cashflow
    exactly on the valuation date is historical.
  - Behaviour-preserving for at-issue/forward pricing (accrued = 0): every prior
    oracle stays green.

### Deferred
- Fixing resolution for seasoned **float** legs (reset ≤ valuation → realized
  `FixingHistory`) — no such instrument present yet; wired when one arrives (A3+).

## [0.0.11] - 2026-07-12

### Changed
- **Amendment A1 — the engine depends on the model, not a market argument.**
  `price(product, model, numerics)`: the model carries the `MarketSnapshot` it was
  calibrated to (`model.market`); the engine reaches curves/valuation-date through
  it. Market/model mismatch is now structurally impossible (there is no second
  market to pass). Behaviour-preserving — every S00–S08 oracle stays green.
  - New L3 `DiscountingModel(market)` (thin model for linear products) and a
    `CalibratedModel` protocol.
  - `HullWhite` now carries a `MarketSnapshot` instead of a bare curve.
  - `DiscountingEngine` / `SwapEngine` / `SwaptionEngine` drop the `market` arg;
    `book.value` builds the model for the date's snapshot; `dv01` bumps the
    snapshot and rebuilds the model (risk flows through the model).
  - `FixingHistory` is now first-class on `MarketSnapshot` (empty default; the
    economy = curves + fixings). Its seasoned-period consumer lands with A2.
  - Oracle: engine-model binding test (no `market` param; a model only prices
    against its own snapshot) + unchanged PVs.

### Ratified
- `CLAUDE.md` §0/§2/§3 and `redesign/02_spine.md` Amendments A1/A2/A3 (Cowork).

## [0.0.10] - 2026-07-12

### Added
- **S08 — Hull-White European swaption (Jamshidian).**
  - `instruments/swaption.py`: `Swaption(expiry, swap)` — a European option on a
    forward-starting `VanillaSwap` (payer/receiver via `swap.pay_fixed`), pure data.
  - `engine/swaption.py`: `SwaptionEngine` — Jamshidian decomposition into a
    portfolio of HW ZCB options (S07), with a bisection solve for the critical
    rate `r*` that prices the coupon bond at par.
  - `models/hull_white.py`: `zero_bond` reconstitution `P(T,S) = A e^{-B r}` (the
    state-dependent bond price the Jamshidian solve needs).
  - `foundation/solvers.py`: `bisect_root` — bracketed bisection (first root-finder
    of the L0 toolkit, migrated on demand).
  - Oracle (closed-form, exact): put-call parity `payer - receiver ==
    P(0,T0)*notional - sum(c_i P(0,t_i))`; ATM symmetry (payer == receiver at the
    forward par rate); `sigma -> 0` collapses to the discounted intrinsic
    `max(forward swap PV, 0)`, cross-checked against the S06 `SwapEngine`.
  - Quarry: `fixed_income/` (swaption), `pricing/`, `core/solvers.py`.

### Deferred
- MC engine + analytic-vs-MC convergence oracle (the **next slice, S09**), where
  `NumericalConfig` gains the MC knobs (`mc_paths`, `mc_seed`).

## [0.0.9] - 2026-07-12

### Added
- **S07 — Hull-White 1F analytic core (first L3 model).**
  - `models/hull_white.py`: `HullWhite(a, sigma, curve)` fitted to a flat curve
    (reprices it by construction); the `B(t,T)` factor and the closed-form
    European option on a zero-coupon bond (Brigo & Mercurio 3.40-3.41).
  - `foundation/distributions.py`: `norm_cdf` via `math.erf` — first piece of the
    L0 numerical toolkit, migrated on demand (dependency-free).
  - Oracle (all closed-form, exact < 1e-12): the model refits the initial curve;
    `B(t,T) -> (T-t)` as `a -> 0`; ZCB-option put-call parity
    `call - put == P(0,S) - K*P(0,T)`; `sigma -> 0` collapses to discounted
    intrinsic; and a match against an independent recompute of the ZBC formula.
  - Quarry: `models/` (hull_white), `numerical/_distributions.py` (norm_cdf).

### Deferred
- HW swaption engine (Jamshidian decomposition) + analytic-vs-MC convergence —
  the **next slice (S08)**, where the MC engine and `NumericalConfig` MC knobs land.
- General (bootstrapped) curve fit, time-dependent `a`/`sigma`, and the L3/L4
  boundary for analytic-model option formulas (flagged for the L3 report).

## [0.0.8] - 2026-07-12

### Added
- **S06 — vanilla single-curve interest-rate swap.**
  - `instruments/swap.py`: `FixedLeg`, `FloatLeg` (structural — schedule + face
    only), `VanillaSwap`, `SwapTerms`, and a `vanilla_swap(...)` builder.
  - `engine/swap.py`: `SwapEngine` — discounts the fixed leg (reusing
    `DiscountingEngine`) and computes the float leg's coupons as the curve's
    forwards (`DF(a)/DF(b) - 1`) at pricing time; NPV is payer/receiver aware.
  - `instruments/leg.py`: shared `fixed_coupon_cashflows` — one definition of a
    fixed leg's coupons, now used by both the bond and the swap (rule of two).
  - Oracle: par swap reprices to zero NPV; float leg telescopes to
    `notional*(DF(start)-DF(maturity))`; off-par NPV `= notional*annuity*(par-rate)`;
    receiver `= -payer`; and a swap matching an S03 bootstrap input reprices to ~0.
  - Quarry: `fixed_income/` (swap / fixed + float legs).

### Changed
- `fixed_rate_bond` builds its coupons via the shared `fixed_coupon_cashflows`
  (behaviour identical; the S04 bond oracle stays green).

### Deferred
- Multi-curve (OIS discount / IBOR projection), basis spread on the float leg,
  and an engine registry/facade selecting engine per instrument — no consumer yet.

## [0.0.7] - 2026-07-12

### Changed
- Signature discipline (CLAUDE.md §3b) is now enforced by the **CI ruff step**
  (`ruff check src/pricebook_ng`, rule `PLR0913`/`max-args=5`) rather than a
  bespoke `verify.py signatures` check — aligning with `redesign/09` ("same CI
  ruff step, not a bespoke checker"). `ruff.toml` is unchanged (it's the config
  that step reads); `verify.py signatures` is removed.
- Ratified the `CLAUDE.md §3b` and `redesign/09` guardrail text (previously
  uncommitted).

## [0.0.6] - 2026-07-12

### Added
- **Signature discipline (CLAUDE.md §3b).** New `verify.py signatures` check —
  the 5-argument ceiling (ruff `PLR0913`/`max-args=5`), enforced in the merge
  gate and CI; `self`/`cls` and `*args`/`**kwargs` are not counted. Root
  `ruff.toml` carries the matching `PLR0913` rule for editor/dev feedback
  (quarry `python/` exempt).
- Frozen grouping value objects to collapse wide signatures:
  `CouponPeriod` (ICMA anchors), `RollRule` (calendar / business-day / eom),
  `ScheduleTerms` (frequency / day-count / roll).

### Changed
- `year_fraction(start, end, convention, *, period=None, calendar=None)` — ICMA
  anchors bundled into `CouponPeriod` (7→5 args).
- `generate_schedule(start, end, frequency, roll=None)` — roll conventions
  bundled into `RollRule` (6→4 args).
- `fixed_rate_bond(face, coupon_rate, start, maturity, terms)` — notional+currency
  as `Money` face, schedule/accrual conventions as `ScheduleTerms` (9→5 args).
- No behaviour change: every S00–S04 oracle stays green (same expected values).

## [0.0.5] - 2026-07-11

### Added
- **S04 — fixed-rate bond (first L2/L4-pricing slice).**
  - `instruments/fixed_rate_bond.py`: `FixedRateBond` (frozen pure data — coupon
    + redemption cashflows, no `pv` method) and a `fixed_rate_bond(...)` builder
    that expands schedule + day-count into explicit `Cashflow`s.
  - Oracle: closed-form discounted-cashflow PV on a flat curve and on the S03
    bootstrapped curve (independent sum), exact < 1e-12; plus cashflow-structure
    checks and the zero-coupon tie-back to the Slice 0 pure-discount result.
  - Quarry: `fixed_income/` (fixed-rate bond / fixed leg).

### Changed
- `engine/discounting.py` generalised from a single cashflow to a **cashflow
  leg**: it now prices any instrument satisfying the structural
  `CashflowInstrument` protocol (`.cashflows`) — no `isinstance`, no import of
  concrete instrument classes. `FixedCashflowTrade` gained a `.cashflows` view;
  the Slice 0 oracle stays green.
- CI layer tier bumped to `--layer 4` (the slice now prices through the engine).

### Deferred
- Seasoned-bond pricing (dropping already-paid coupons), accrued interest / clean
  vs dirty price, and business-day-adjusted coupon dates — no consumer yet.

## [0.0.4] - 2026-07-11

### Added
- **S03 — bootstrapped discount curve (first L1 slice).**
  - `market/discount_curve.py`: `DepositQuote`, `ParSwapQuote`, a log-linear
    interpolated `DiscountCurve` (behind the existing `CurveHandle`), and
    `bootstrap_discount_curve` — deposits give short-end DFs in closed form,
    par swaps extend the curve by a sequential closed-form solve (single-curve).
  - Oracle: every input reprices to par — deposits to their closed-form DF, swaps
    to zero NPV via the single-curve telescoping identity — exact < 1e-12; plus
    df(valuation)=1, strictly-decreasing DFs, and the log-linear interpolation law.
  - Quarry: `core/discount_curve.py`.
- CI layer tier bumped to `--layer 1` (the slice now reaches L1).

### Deferred
- Business-day-adjusted curve pillars, multi-curve (OIS discount / IBOR
  projection), non-pillar swap coupons (interpolated bootstrap), and QuantLib
  cross-check — the closed-form self-consistency oracle is stronger here; these
  arrive with the slices that need them.

## [0.0.3] - 2026-07-11

### Added
- **S02 — schedule & business-day calendar.**
  - `foundation/calendar.py`: `BusinessDayConvention` (UNADJUSTED / FOLLOWING /
    MODIFIED_FOLLOWING / PRECEDING / MODIFIED_PRECEDING) and a minimal
    data-driven `Calendar` (weekend + explicit holiday set) with `is_business_day`,
    `adjust`, `business_days_between`.
  - `foundation/schedule.py`: `Frequency` and `generate_schedule` — regular
    periods, short front stub (backward generation), EOM roll, optional
    business-day adjustment. No third-party dependency (stdlib month arithmetic).
  - `foundation/time.py`: BUS/252 completed now that calendars exist.
  - Oracle: hand-computed reference dates/counts (adjustments, coupon schedules
    incl. EOM + short front stub, BUS/252 day counts) — exact.
  - Quarry: `core/calendar.py`, `core/schedule.py`, `core/day_count.py` (BUS/252).

### Deferred
- Concrete national calendars (TARGET, US, London, Sao Paulo, ...) and long/back
  schedule stubs — no current consumer; they land with the instrument slice that
  first needs them (avoids the quarry's approximate long-stub heuristic).

## [0.0.2] - 2026-07-11

### Added
- **S1 — day-count conventions.** Extended `foundation/time.py` beyond the
  Slice 0 ACT/365F stub to the calendar-free conventions: ACT/360, 30/360
  (US bond basis), 30E/360 (Eurobond basis), ACT/ACT ISDA, ACT/ACT ICMA.
  - Oracle: published ISDA 2006 s.4.16 / ICMA Rule 251.1 year-fraction vectors,
    each expected value written as the convention's defining arithmetic, exact
    < 1e-12.
  - Quarry: `core/day_count.py`.

### Changed
- ACT/ACT ICMA now **requires** its coupon-period anchors (`ref_start`,
  `ref_end`, `frequency`) and raises when they are missing or invalid.

### Removed
- Debt shed (CLAUDE.md §5): the quarry's `strict_icma` flag and its silent
  fallback to ACT/365F on missing anchors (audit finding A.1 B1 — hidden
  wrongness) does not cross into the new tree.

### Deferred
- BUS/252 day-count, business-day `Calendar`, and `Schedule` generation move to
  their own slice (they need the calendar); not part of S1's named oracle.

## [0.0.1] - 2026-07-11

### Added
- **Slice 0 — walking skeleton.** A single fixed cashflow discounted on a flat,
  continuously-compounded curve, priced end-to-end L0->L6 through the stateless
  engine. Proves the spine holds before any further migration.
  - L0 `Money`/`Currency`, `Cashflow` (promoted from `fixed_income`),
    `year_fraction` (ACT/365F), `NumericalConfig`, `PricingResult`/`PricingFailure`.
  - L1 `MarketSnapshot` + `FlatDiscountCurve` behind a `CurveHandle` (df = exp(-r t)).
  - L2 `FixedCashflowTrade` (frozen, no `pv` method).
  - L4 `DiscountingEngine.price(...)` (null model; failure-as-value).
  - L5 `dv01` by bumping the snapshot; L6 `book(trade).value(...)`.
  - Oracle: PV = notional·exp(-r·t) closed form < 1e-12; analytic vs
    central-difference DV01 < 1e-6; repricing byte-identical (statelessness).
  - Quarry: `core/currency.py`, `core/day_count.py`, `core/numerical_config.py`,
    `core/discount_curve.py`, `fixed_income/fixed_leg.py` (Cashflow).

## [0.0.0] - 2026-07-11

### Added
- Bootstrap of the new tree: layer packages (`foundation`, `market`,
  `instruments`, `models`, `engine`, `risk`, `shell`), `verify.py`
  (`acyclic`/`tests`/`debt`/`provenance`/`version`/`all`), CI matrix
  (Ubuntu + Windows, Python 3.12), `.gitattributes` (LF), root `conftest.py`.
