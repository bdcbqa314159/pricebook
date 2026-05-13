# Release Notes

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
