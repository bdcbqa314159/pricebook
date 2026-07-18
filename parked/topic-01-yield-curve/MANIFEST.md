# Topic 1 — Yield-Curve World · MANIFEST

Living tracker (#13 §3.3). Status: `target` → `covered` | `dead` | `reassigned→X` → `parked`.
**MINE** = read for domain content before parking, even if production-dead.
Fan-in = production consumers in `python/pricebook/` (excl. own tests, verified incl. `__init__` re-exports).

---

## IN TOPIC 1

### Conventions (core/) — the richest material in the quarry
| file | LOC | fan-in | role | MINE | status |
|---|---|---|---|---|---|
| `core/day_count.py` | 274 | 227 | 7 day-count conventions | ★★★ | target |
| `core/calendar.py` | 943 | 52 | 37 calendars + holiday-rule DSL | ★★★ | target |
| `core/schedule.py` | 143 | 80 | schedules, stubs, EOM anchoring | ★★★ | target |
| `core/rate_index.py` | 329 | 2 | 28 indices (RFR + IBOR) | ★★★ | target |
| `core/currency.py` | 135 | 3 | Currency enum (+FX pair → reassign) | ★★ | target |
| `core/notional.py` | 56 | 4 | notional scalar/list expansion | ★ | target |
| `core/fixings.py` | 253 | 2 | fixing store, `get_with_lag` | ★★ | target |
| `core/data_registry.py` | 156 | 9 | JSON ↔ convention dataclasses | ★★ | target |

### Curves & interpolation (core/)
| file | LOC | fan-in | role | MINE | status |
|---|---|---|---|---|---|
| `core/discount_curve.py` | 300 | 227 | df/zero/forward/bump/roll_down | ★★★ | target |
| `core/interpolation.py` | 298 | 55 | 5 schemes incl. Hyman-filtered | ★★★ | target |
| `core/forward_interpolation.py` | 256 | 0 | forward-space construction | ★ | target |

### Curve construction (curves/)
| file | LOC | fan-in | role | MINE | status |
|---|---|---|---|---|---|
| `curves/bootstrap.py` | 713 | 8 | sequential single/dual-curve bootstrap | ★★★ | target |
| `curves/curve_builder.py` | 298 | 2 | G10 convention table + method dispatch | ★★★ | target |
| `curves/global_solver.py` | 417 | 2 | simultaneous Newton, analytic Jacobian | ★★★ | target |
| `curves/multicurve_solver.py` | 486 | **0** | joint OIS+projection, damped Newton | ★★★ | target |
| `curves/ncurve_solver.py` | 279 | **0** | **generic N-curve solve — the pattern** | ★★★ | target |
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
