# Artifact #17 — Classification of the remaining quarry "L0"

**Purpose:** label every old file the quarry classified as L0, and identify **which park when
Topic 1 (multicurve + linear rates) closes**. Written at ng v0.74.1 (L0 closed, 13 parked).

**Reminder:** the quarry's "L0" was computed *mechanically* ("imports nothing internal"), which is why
it contains portfolio analytics, plotting, time series and a database. Our classification is by
**meaning**, per `CLAUDE.md §3c`.

---

## Headline

| | count |
|---|---|
| quarry L0 originally | 111 |
| parked at Topic 0 close | **13** |
| **remaining** | **98** |
| of which **park when Topic 1 closes** | **~5 (from L0) + the Topic-1 set outside L0** |

**⚠ Loose end found:** the seven `fixed_income` modules ticked deletable during CP-3 —
`deposit · fra · ois · zero_coupon_bond · bond · fixed_leg · swap` — are **still physically in the
quarry**. They were ticked under the old per-module scheme, before topic-parking existed. They are
linear-rates products: **fold them into Topic 1's parking set.**

---

## A. `core/` — 16 remaining

| module | target | parks when |
|---|---|---|
| **`discount_curve.py`** | **TOPIC 1** — becomes the `YieldCurve` capability | Topic 1 close |
| **`pricing_context.py`** | **TOPIC 1** — becomes `MarketSnapshot` + `CurveSet` | Topic 1 close |
| **`forward_interpolation.py`** | **TOPIC 1** — forward-space construction + **Hagan–West** (the real one) | Topic 1 close |
| **`greeks.py`** (39 LOC) | **TOPIC 1** — a result container; curve risk ships with Topic 1 | Topic 1 close |
| `survival_curve.py` | credit | credit topic |
| `fixings.py` | market-data / persistence (store + file I/O ≠ L0) | that topic |
| `trade.py` · `book.py` · `daily_pnl.py` · `mandate.py` | **L6 shell / lifecycle** | shell topic |
| `market_conventions.py` | equity / commodity / inflation (misleading name — nothing rates) | those topics |
| `approximation.py` · `numerical_safety.py` · `convergence_framework.py` | numerics | numerics topic |
| `caching.py` · `dependency_graph.py` | infrastructure (production-orphans in the quarry) | infra topic |

## B. `market_data/` — 1 remaining
| module | target | parks when |
|---|---|---|
| **`_types.py`** (`Quote` · `QuoteId` · `QuoteKind` · `MarketSnapshot` · `FixingHistory`) | **TOPIC 1** — the quote/snapshot vocabulary | Topic 1 close |

## C. `numerical/` — 30 remaining. **S17 changes the picture: most are now `dead`, superseded by scipy.**

| group | modules | disposition |
|---|---|---|
| **scipy-superseded** (our thin adapters already cover, or will) | `_distributions` · `_distributions_theory` · `_rootfinding` · `_optimize` · `_integrate` · `_interpolation` · `_linalg` · `_ode` · `_spectral` · `_differentiate` · `_qmc` | **`dead`** — S17 ratified scipy; these are not migrated, they are *replaced*. Verify no finance-specific content before ticking. |
| **finance-specific numerics** (own them) | `_mc` · `_pde` · `_trees` · `_fourier` · `_stochastic` · `pde_boundary` · `tree_enhancements` · `implied_tree` · `operator_splitting` · `oscillatory_quad` | **demand-migrate** with their first consumer (options/models topics) |
| **specialised optimisation / AD** | `auto_diff` · `sparse_jacobian` · `duality` · `frank_wolfe` · `sdp` · `socp` · `convexity_tools` · `von_neumann` · `_graph` | **AAD / advanced-optimisation topic** (or `dead` — most have zero fan-in) |

**This is a large, cheap block.** With scipy ratified, ~11 modules are `dead`-by-substitution, not
migration work. Confirm each with the §4 evidence protocol before ticking.

## D. `statistics/` — 17 remaining
`bayesian · clustering · copulas · distribution_fit · distribution_theory · garch · hmm ·
information_theory · kalman · optimisation_advanced · optimization · particle_filter · regression ·
rng · statistics · zscore · calibration_quality`
→ **statistics/analytics topic** (several are scipy/statsmodels-superseded; `rng` interacts with the
pinned `RNG_FAMILY`). **None are Topic 1.**

## E. `viz/` (13) · `ts/` (7) · `pe/` (4) · `db/` (2)
→ **viz topic** · **time-series topic** · **L6 portfolio/PE** · **data-spine/persistence**.
None are Topic 1. All were only ever "L0" because they import nothing internal.

---

## Topic 1's parking set (multicurve + linear rates)

**From quarry L0 (5):** `core/discount_curve` · `core/pricing_context` · `core/forward_interpolation`
· `core/greeks` · `market_data/_types`

**From outside L0 — the bulk:**
- **`curves/` (~18 of 31):** `bootstrap` · `curve_builder` · `global_solver` · `multicurve_solver` ·
  `ncurve_solver` · `rfr_bootstrap` · `curve_engine` · `curve_advanced` · `nelson_siegel` ·
  `smith_wilson` · `curve_blending` · `seasonal_curve` · `bond_curve` · `synthetic_market_data` ·
  `curve_risk` · `key_rate_risk` · `curve_bumper` · `curve_scenarios`
  *(reassigned out: the 5 `aad*` · `inflation_curve` · `em_curve_builder` · `ndf_implied` ·
  `curve_diffusion` · `linalg` · `sparse` · `sparse_grids` · `curve_storage`)*
- **`fixed_income` linear (7 already ticked + the rest of the vanilla spine):** `deposit` · `fra` ·
  `ois` · `zero_coupon_bond` · `bond` · `fixed_leg` · `swap` (+ `floating_leg` · `basis_swap` ·
  `csa` · futures)

**Expected drawdown at Topic 1 close: 13 → ~45+.**

---

## Two things to action
1. **Fold the 7 already-ticked `fixed_income` modules into Topic 1's manifest** — they are ticked but
   unparked, an artefact of the pre-topic scheme.
2. **Fix the denominator.** The gate report used **793** (all `.py` incl. `__init__`); earlier counts
   used **768** (non-init). A progress bar whose denominator moves is not a progress bar. `__init__.py`
   is packaging, not content, and disappears when its package empties — **hold 768.**
