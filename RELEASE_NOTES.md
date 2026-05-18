# Release Notes

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
