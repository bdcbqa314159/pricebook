# Topic 1 — Yield-Curve World · MANIFEST

Living tracker (#13 §3.3). Status: `target` → `covered` | `dead` | `reassigned→X` → `parked`.
**MINE** = read for domain content before parking, even if production-dead.
Fan-in = production consumers in `python/pricebook/` (excl. own tests, verified incl. `__init__` re-exports).

---

## Slice log (ng build progress) — drawdown steering is §4, not this table

- **Slice 1 (v0.85.0)** — single-curve vertical L0→L4; par swap → zero NPV. Built `foundation/time_measure`,
  `market/{curve,snapshot,building_blocks}`, `products/swap`, `models/discounting_model`,
  `calibration/calibrate` (sequential single-curve bootstrap), `engine/linear`. **Ticked 0 deletable.**
- **Slice 2 (this slice)** — dual-curve: ESTR discount + EURIBOR_3M projection; a EURIBOR swap prices to
  zero off ESTR discounting. Built `market/curve_set` (`CurveSet`/`CurveKey`); added the `forward()`
  projection atom and generalised `float_leg_pv` to compose it (§3d, no telescoping duplicate);
  `FloatLeg` gained `index`; `calibrate` now builds a curve **set** in dependency order (discount →
  projection). Mines `curves/bootstrap.py` + `curves/ncurve_solver.py` **for concept only** (sequential,
  not the global N-curve solver). **Ticks 0 deletable.**

- **Slice 3 (this slice)** — cash instruments: deposits + FRAs + IMM futures (as forwards, rate = 1−price)
  on both curves. Landed two ratified abstractions FIRST as a green-guarded refactor (STEP A): the L4
  **engine registry** (product→pricer, structural, no isinstance) and the L3 **`CalibrationInstrument`**
  protocol (`residual(discount, projection)`); then the instruments (STEP B). New shared atom `deposit_df`
  (`1/(1+r·τ)`); FRA/future compose the existing `forward()`.
  **Drawdown verdict — retire read COMPLETE (§4 retire flow: read end-to-end → classify → tick).** The
  formal end-to-end read of `curves/bootstrap.py` (713 LOC) and `core/discount_curve.py` (300 LOC) is done;
  **both cross at THIS slice → Topic-1 deletable 0 → 2 (overall 13 → 15 / 793 deletable; physical park at
  Topic-1 close).**
  - **Convention/calendar is IMPORTED, not resident** in `bootstrap.py` (`core/day_count`, `core/schedule`,
    `core/calendar` — lines 11/14/16, already parked to `topic-00-foundation/`, covered by ng L0 supersets),
    so deleting it orphans no convention code. The genuine `bootstrap.py`-resident deferred bits are
    **futures convexity** (inline `:151-158`) and **turn-of-year** (inline `:161-162`); the global solve is
    **not** resident (it is `ncurve_solver.py`).
  - **`discount_curve.py` residents** are analytics on a curve ng HAS: `zero_rate:207` · `bumped:120` ·
    `bumped_at:184` · `instantaneous_forward:226` · `pillar_dates:99` · `roll_down:131` · `forward_rate:246`
    · `to_dict`/`from_dict:270-299`. `forward_rate` is **crossed** (ng `forward()` atom, same
    `(df₁−df₂)/(τ·df₂)`); the rest deferred (below). `calibration_result`-on-curve is **shed by design**
    (ng: `CalibrationResult` beside the model, doc 19 §5).
  - **ng consumer evidence (0 needed-now, bare-name across src + tests):** `fixed/float/deposit_day_count`
    → 0 · calendar/BDC in the curve build → 0 (`RollRule(calendar=None)`, `calibrate.py:176`) · `convexity`
    → 0 (2 hits, both docstrings) · `turn_of_year` → 0. Supersede basis: ng reproduces
    **deposits→FRAs→futures→swaps dual-curve sequential**, MINUS the deferred capabilities (SUPERSEDE, not clone).

### Named crossings — status after the slice-3 & slice-4 retire reads (§4)

| quarry module | status | detail |
|---|---|---|
| `core/discount_curve.py` | **✓ CROSSED (deletable) — slice 3** | ng `DiscountCurve` (df/flat/log-linear) + the `forward()` atom supersede; analytics deferred (forward-links below) |
| `curves/bootstrap.py` | **✓ CROSSED (deletable) — slice 3** | ng `calibrate` dual-curve sequential bootstrap supersedes; convexity/turn-of-year deferred (forward-links below) |
| `curves/ncurve_solver.py` | **✓ CROSSED (deletable) — slice 4** | ng's simultaneous solve did the 2-curve instance of the state-vector pattern; N-curve > 2 deferred |
| `curves/global_solver.py` | **✓ CROSSED (deletable) — slice 4** | simultaneous Newton over all pillar DFs → ng's global solve (scipy LM) |
| `curves/multicurve_solver.py` | **✓ CROSSED (deletable) — slice 4** ⚠ | joint OIS+projection = ng's exact case; **flagged for Cowork spot-check** (read found it also-superseded; task expected 2 ticks, not 3) |
| `curves/curve_risk.py`, `key_rate_risk.py` | target → **C3 · curve-risk slice** | no risk consumer until C3 |

**Forward-links filed on DESTINATION rows (deferred capability travels with X's crossing slice, §4):**
- **Models slice** ← futures convexity: `curves/bootstrap.py:151-158` (inline) **AND**
  `fixed_income/ir_futures.py::hw_convexity_adjustment` — both must cross there or one orphans.
  Form: `forward = futures_rate − convexity(a,σ,t)`.
- **Seasonality slice** ← turn-of-year: `curves/bootstrap.py:161-162` (inline) **AND** `curves/seasonal_curve.py`.
- **C3 · curve-risk slice** ← `discount_curve.py` analytics: `zero_rate` · `bumped` · `bumped_at` ·
  `instantaneous_forward` · `pillar_dates` · `roll_down` (add to ng `DiscountCurve`).
  *(Deviation from the tasking's Step-3 list: `forward_rate` is NOT forward-linked — ng already has it as
  the `forward()` atom, so it is crossed, not deferred.)*
- **Persistence slice** ← `discount_curve.py` `to_dict`/`from_dict`.
- **C3 · curve-risk slice / AAD** ← the solvers' **analytic Jacobian**
  (`multicurve_solver.curve_analytical_jacobian`, `global_solver._jacobian_analytical`): ng's Jacobian is
  scipy's numerical `result.jac`; the analytic/adjoint form arrives with its C3/AAD consumer (doc 18 §6).
- **3rd-projection-curve slice** ← **N-curve > 2** generality (`ncurve_solver`): ng does 2 curves (rule of
  two); the N>2 state-vector generalization travels with the second projection curve.
- **ng-side note (NOT a quarry migration):** grow `CurveBuild` to per-leg conventions (fixed vs float
  day-count/frequency) + wire a real calendar when a multi-convention consumer arrives (rule of two). The
  convention *code* is already parked (Topic-0 supersets); nothing to migrate.
- **ng-side hardening note (surfaced by this retire read):** `discount_curve.forward_rate` is *crossed*
  (ng `forward()` atom, same value), but the quarry uses the numerically-stable **subtract-first** form
  `(df₁−df₂)/(τ·df₂)` (its docstring: "critical for overnight and short-period forwards"), while ng's
  `market/building_blocks.forward()` uses the **divide-first** form `(df₁/df₂−1)/τ`, which loses precision
  when `df₁ ≈ df₂`. Algebraically identical; agree to ~1e-16 on the current annual oracles, so **not a
  needed-now residual**. **Trigger:** switch `forward()` to the subtract-first form when an RFR
  daily-compounding / overnight-forward consumer arrives (1-line change; rides that curves slice, not this
  docs commit).

---

## IN TOPIC 1

### Conventions (core/) — **OWNED BY TOPIC 0 (parked); mine the parked copies for content**
> Superseded by #16 (Topic 0). These are **not** Topic-1 targets — they parked to
> `parked/topic-00-foundation/` at the Topic 0 gate. Listed here only so Topic-1 slices know **where to
> mine** (curve schemes / query helpers deferred out of L0 travel here). Fan-in was never a Topic-1
> concern (ng does not import the quarry).

| file | now at | Topic-1 mines it for |
|---|---|---|
| `day_count.py` | `parked/topic-00-foundation/` | `date_from_year_fraction` (t→date) if a curve needs it; else ng `foundation/day_count.py` is a superset |
| `calendar.py` | `parked/topic-00-foundation/` | — (ng superset) |
| `schedule.py` | `parked/topic-00-foundation/` | — (ng superset) |
| `rate_index.py` | `parked/topic-00-foundation/` | `overnight_indices` / `indices_for_currency` query helpers (curve-build) |
| `currency.py` | `parked/topic-00-foundation/` | `CurrencyPair.forward_rate`/`forward_points` + `all_g10_pairs` → **reassigned→FX**, not Topic 1 |

### Still Topic-1 conventions/data (in the quarry)
| file | LOC | fan-in | role | MINE | status |
|---|---|---|---|---|---|
| `core/notional.py` | 56 | 4 | notional scalar/list expansion (amortising) | ★ | target *(reassigned in from Topic 0 — an L2 product concern)* |
| `core/fixings.py` | 253 | 2 | fixing **store** + persistence (`get_with_lag`, CSV/JSON) | ★★ | target — ng read-model `FixingHistory` covers lookup; the mutable store + persistence is Topic-1/market-data |
| `core/data_registry.py` | 156 | 9 | JSON ↔ **curve-convention** dataclasses (G10) | ★★ | target *(reassigned in from Topic 0)* |

### Curves & interpolation (core/)
| file | LOC | fan-in | role | MINE | status |
|---|---|---|---|---|---|
| `core/discount_curve.py` | 300 | 229 | df/zero/forward/bump/roll_down | ★★★ | target |
| `parked/topic-00-foundation/interpolation.py` | 298 | 56 | 5 schemes incl. Hyman-filtered | ★★★ | **mine (parked under Topic 0)** — L0 has the mechanism; Topic 1 builds curve schemes (Monotone/Akima/Hyman, Hagan-West) from the parked copy |
| `parked/topic-00-foundation/solvers.py` | 261 | 41 | root-find variants (Newton/Brent/ITP) | ★★ | **mine (parked under Topic 0)** — L0 has bisect+Nelder-Mead; mine for Brent/Newton when curve-solve convergence bites |
| `core/forward_interpolation.py` | 256 | 0 | forward-space construction | ★ | target |

### Curve construction (curves/)
| file | LOC | fan-in | role | MINE | status |
|---|---|---|---|---|---|
| `curves/bootstrap.py` | 713 | 8 | sequential single/dual-curve bootstrap | ★★★ | target |
| `curves/curve_builder.py` | 298 | 2 | G10 convention table + method dispatch | ★★★ | target |
| `curves/global_solver.py` | 417 | 2 | simultaneous Newton, analytic Jacobian | ★★★ | **✓ crossed (slice 4)** — ng simultaneous solve |
| `curves/multicurve_solver.py` | 486 | **0** | joint OIS+projection, damped Newton | ★★★ | **✓ crossed (slice 4)** ⚠ flagged for Cowork spot-check |
| `curves/ncurve_solver.py` | 279 | **0** | **generic N-curve solve — the pattern** | ★★★ | **✓ crossed (slice 4)** — ng did the 2-curve instance; N>2 deferred |
| `curves/rfr_bootstrap.py` | 328 | **0** | post-LIBOR RFR stack + conventions | ★★★ | target |
| `curves/curve_engine.py` | 292 | **0** | declarative CurveDefinition + CurveSet | ★★★ | target |
| `curves/curve_advanced.py` | 337 | **0** | NS/Svensson post-fit, monotone forwards, TOY | ★★ | target |
| `curves/nelson_siegel.py` | 212 | 2 | NS/Svensson parametric | ★★ | target |
| `curves/smith_wilson.py` | 153 | 2 | UFR extrapolation (Solvency II) | ★★ | target |
| `curves/curve_blending.py` | 135 | **0** | splice/blend curves | ★ | target |
| `curves/seasonal_curve.py` | 188 | **0** | turn-of-year funding premium | ★★ | target |
| `curves/bond_curve.py` | 579 | **0** | strip curve from bond prices | ★★ | target |
| `curves/synthetic_market_data.py` | 109 | **0** | 32-currency synthetic quotes | ★★ | target |

### Curve risk (curves/)
| file | LOC | fan-in | role | MINE | status |
|---|---|---|---|---|---|
| `curves/curve_risk.py` | 223 | **0** | zero & **quote** Jacobians, roll-down | ★★★ | target |
| `curves/key_rate_risk.py` | 257 | **0** | key-rate DV01, buckets, ladders | ★★★ | target |
| `curves/curve_bumper.py` | 173 | **0** | cached Jacobian repricing, cross-gamma | ★★ | target |
| `curves/curve_scenarios.py` | 261 | **0** | named shocks + PCA scenarios | ★★ | target |

### Market data plumbing (data/)
| file | LOC | fan-in | role | MINE | status |
|---|---|---|---|---|---|
| `data/rate_source.py` | 129 | 3 | `RateSource` protocol, tenor vocabulary | ★★ | target |
| `data/euribor_loader.py` | 380 | 1 | scrape+cache Euribor | ★ | target |
| `data/euribor_source.py` | 84 | 1 | the only RateSource impl | ★ | target |
| `data/market_curve.py` | 384 | 1 | curve-from-source facade | ★★ | target |
| `data/rate_database.py` | 335 | **0** | SQLite fixings + curve params | ★★ | target |

---

## REASSIGNED OUT OF TOPIC 1
| file | LOC | → topic | note |
|---|---|---|---|
| `curves/aad.py` `aad_curves` `aad_interp` `aad_pricing` `aad_calibration` | 1092 | **AAD** | complete reverse-mode tape; `aad_curves` bootstrap is instructive |
| `curves/inflation_curve.py` | 90 | **inflation** | thin; BEI only |
| `curves/em_curve_builder.py` | 306 | **EM/sovereign** | ★★ the 18-currency EM convention table is the asset |
| `curves/ndf_implied.py` | 267 | **FX** | CIP + `cip_basis` — revisit for xccy |
| `curves/curve_diffusion.py` | 176 | **rate models** | multi-factor HJM evolution |
| `curves/linalg.py` `sparse.py` `sparse_grids.py` | 549 | **numerical utils** | PCA/CSR/Smolyak — no curve semantics |
| `curves/curve_storage.py` | 149 | **infrastructure** | in-memory, not really storage |
| `core/market_conventions.py` | 210 | **equity/commodity/inflation** | misleading filename; nothing rates |
| `core/currency.py` (CurrencyPair half) | — | **FX** | forward_rate/points on a convention object |

---

## CRITICAL FINDINGS (change the design — see rulings)
1. **No par delta exists anywhere.** `curve_risk.input_jacobian` gives d(zero)/d(quote) but is never composed with ∂PV/∂z. The quarry never computed par sensitivities.
2. **Key-rate buckets do NOT sum to parallel DV01.** Docstring claims partition of unity; **no normalisation is performed**. Two incompatible bucket definitions coexist (`key_rate_risk._bump_weight` vs `curve_bumper`'s fixed 1y tent).
3. **Parametric methods are post-fits, not constructions.** NS/Svensson/Smith-Wilson fit a curve that was *already bootstrapped*. They are smoothing overlays.
4. **`ncurve_solver` is the generalisation to adopt** — `CurveSpec` + `InstrumentPricer` protocol makes basis and xccy expressible. Closest thing to the ng `CurveSet` design.
5. **The bump loop is reimplemented 6+ times**, every copy dropping `day_count`/`interpolation` from the source curve.
6. **Market data is effectively empty** — one EUR 3M deposit column (774 rows); no USD or GBP source at all; `MarketCurve` is deposit-only so curves stop at 12M. Two unconnected data stacks (`data/` vs `pricing/market_data_provider.py`).

   **RULED: synthetic quote sets with real conventions.** Topic 1 builds EUR/USD/GBP from a
   checked-in synthetic quote set per currency (deposits + OIS + par swaps at standard pillars),
   using the **real G10 convention tables** (`data/curve_conventions_g10.json`). Deterministic,
   offline, CI-safe — and reprice-to-par oracles are exact regardless of whether the levels are
   real. Mine `curves/synthetic_market_data.py` for the per-currency base-rate/slope table (32
   currencies, inverted EM curves, ARS/TRY stress cases) but fix its flaws: linear `base+slope·y`
   is not a plausible shape (no front hump, no long-end flattening), and unknown currencies
   silently default instead of raising.
   **Live feeds (ECB/FRED/BOE) are NOT in Topic 1** — the two unconnected data stacks and the
   scraping/API work become their own later "market data" topic. `data/euribor_*`,
   `data/rate_database.py`, `data/market_curve.py` are therefore **reassigned→market-data** unless
   a Topic-1 slice genuinely needs them.
