# Release Notes

---

## v0.289.0 — 2026-04-21

FH5+FH6: Consumer API integration. 5093 tests.

`swap.py`, `frn.py`, `basis_swap.py`:
- Added `payment_delay_days` and `observation_shift_days` pass-through
- `BasisSwap.par_spread()` throwaway leg preserves all settings (calendar, convention, stub, eom, delay, shift)
- `FloatingRateNote.discount_margin()` throwaway FRN preserves all settings

`test_floating_leg_fixing.py` (+6 tests):
- Swap/FRN/basis_swap pass-through, backward compat, par_spread preserves, DM preserves

---

## v0.288.0 — 2026-04-21

FH5: Consumer API pass-through for fixing parameters. 5087 tests.

`swap.py`, `frn.py`, `basis_swap.py`:
- Added `payment_delay_days` and `observation_shift_days` parameters
- Passed through to FloatingLeg construction
- Default 0 preserves backward compatibility

`test_floating_leg_fixing.py` (+4 tests):
- Swap delay/shift, FRN shift, basis swap both legs, backward compat

---

## v0.287.0 — 2026-04-21

FH4: Forward rate projects from observation window. 5083 tests.

`floating_leg.py`:
- Added `observation_start` and `observation_end` fields to FloatingCashflow
- `forward_rate()` now uses `df(obs_start)/df(obs_end)` instead of accrual dates
- `fixing_date` derived from `observation_start` (= accrual_start - shift)
- Year fraction still based on accrual dates (correct per ISDA)

`test_floating_leg_fixing.py` (+4 tests):
- Obs equals accrual without shift, shifted obs dates, forward rate differs, fixing = obs_start

---

## v0.286.0 — 2026-04-21

FH3: Current-period fixing for term rates. 5079 tests.

`floating_leg.py`:
- Changed fixing condition from `accrual_end <= ref` to `fixing_date <= ref`
- Term rates (EURIBOR) are known at period start — now used immediately
- Shifted observation dates trigger fixing lookup earlier (correct for SOFR T-2)

`test_floating_leg_fixing.py` (+3 tests):
- Current period uses known fixing, future fixing_date ignored, shifted current period

---

## v0.285.0 — 2026-04-21

FH1+FH2: Business-day observation shift and payment delay. 5076 tests.

`floating_leg.py`:
- `fixing_date` now uses `Calendar.add_business_days(accrual_start, -shift)` when calendar provided
- `payment_date` now uses `Calendar.add_business_days(accrual_end, +delay)` when calendar provided
- Without calendar, falls back to calendar days (backward compat)
- Stores calendar, convention, stub, eom on `self` for downstream use

`test_floating_leg_fixing.py` (+6 tests):
- Obs shift skips weekends/holidays, payment delay skips weekends, backward compat without calendar

---

## v0.284.0 — 2026-04-17

FX6: Calendar-aware business day lag. **FIXING MANAGEMENT COMPLETE.** 5070 tests.

`calendar.py`:
- Added `Calendar.add_business_days(d, n)` — move forward/backward by n business days, skipping weekends and holidays

`fixings.py`:
- Added `FixingsStore.get_with_lag(rate_name, d, lag, calendar)` — retrieve fixing at lag business days before d
- Without calendar, falls back to calendar days

`test_floating_leg_fixing.py` (+7 tests):
- Business day forward/backward, holiday skip, calendar-aware lag, no-calendar fallback, missing, zero lag

---

## v0.283.0 — 2026-04-17

FX5: RateIndex registry. 5063 tests.

`rate_index.py` (NEW):
- RateIndex dataclass: name, currency, day_count, fixing_lag, compounding, observation_shift, payment_delay, tenor_months, is_overnight, administrator
- CompoundingMethod enum: COMPOUNDED, AVERAGED, FLAT
- 8 overnight RFRs: SOFR (FRBNY), ESTR (ECB), SONIA (BOE), TONA (BOJ), SARON (SIX), CORRA (BOC), AONIA (RBA), NZOCR (RBNZ)
- 3 term IBORs: EURIBOR_3M, EURIBOR_6M, TIBOR_3M
- get_rate_index(), all_rate_indices(), overnight_indices(), indices_for_currency()

`test_floating_leg_fixing.py` (+12 tests):
- Per-index validation, overnight/term distinction, frozen immutable, G10 RFR coverage

---

## v0.282.0 — 2026-04-17

FX4: Back stub types for schedule generation. 5051 tests.

`schedule.py`:
- Added `StubType.SHORT_BACK` and `StubType.LONG_BACK`
- SHORT_BACK: regular periods from start, short final period
- LONG_BACK: short final stub merged into previous period
- Forward generation from start (vs backward from end for front stubs)

`test_floating_leg_fixing.py` (+5 tests):
- Short back stub, long back stub, exact division, front-vs-back difference, FloatingLeg compat

---

## v0.281.0 — 2026-04-17

FX3: Observation shift for RFR fixings. 5046 tests.

`floating_leg.py`:
- Added `observation_shift_days` parameter (default 0)
- New `fixing_date` field on FloatingCashflow: `accrual_start - shift`
- SOFR uses T-2: fixing observed 2 days before accrual start
- FixingsStore lookup now uses `fixing_date` instead of `accrual_start`

`test_floating_leg_fixing.py` (+5 tests):
- Default zero shift, T-2 shift, fixing lookup at shifted date, wrong-date fallback, negative raises

---

## v0.280.0 — 2026-04-17

FX2: FixingsStore integration with FloatingLeg. 5041 tests.

`floating_leg.py`:
- `pv()` now accepts `fixings: FixingsStore` and `rate_name: str`
- Past accrual periods (accrual_end <= reference_date) use stored fixing rate
- Missing fixings fall back to forward curve projection
- Future periods always use forward curve (fixings ignored)

`test_floating_leg_fixing.py` (+5 tests):
- Past periods use fixings, future ignore, backward compat, missing fallback, exact amount

---

## v0.279.0 — 2026-04-17

FX1: Payment delay on FloatingLeg. 5036 tests.

`floating_leg.py`:
- Added `payment_delay_days` parameter (default 0 for backward compat)
- `payment_date = accrual_end + timedelta(days=payment_delay_days)`
- Standard T+2 for SOFR/EURIBOR, T+0 for OIS

`test_floating_leg_fixing.py` (NEW, 6 tests):
- Default zero delay, T+2 delay, PV impact, monotone in delay, negative raises, backward compat

---

## v0.278.0 — 2026-04-21

PR5: Cross-Asset Validation. **PRODUCTION READINESS COMPLETE.** 5030 tests.

`test_cross_asset_validation.py` (NEW, 16 tests):
- IR: swap PV = 0 at par rate; receiver DV01 positive
- Bond: dirty = clean + accrued; YTM reasonable
- FX: CIP holds; triangular consistency (EUR/JPY = EUR/USD × USD/JPY)
- Credit: upfront round-trip; IG/HY sign convention
- Options: put-call parity at ATM + OTM
- Curves: all 10 G10 OIS valid (monotone DFs, DF[0]=1)
- Inflation: index ratio identity

---

## v0.277.0 — 2026-04-21

PR4: Equity, Commodity & Inflation Conventions. **5000+ tests milestone.**

`market_conventions.py` (NEW):
- EquityIndexSpec: SPX, NDX, SX5E, DAX, UKX, NKY, HSI, AS51, SPTSX (9 indices)
- CommodityContractSpec: CL, BRN, NG, GC, SI, HG, ZC, ZW, ZS, CO (10 contracts)
- LME metals: LCU, LAH, LZS with prompt date system (3 metals)
- LinkerConvention: US (CPI-U, deflation floor), UK (RPI, 8M lag), FR/IT/DE (HICP), CA, AU, JP
- index_ratio(): daily linear interpolation + simple monthly
- 26 new tests, 5014 total

---

## v0.276.0 — 2026-04-21

PR3: Credit Conventions.

`cds_conventions.py` (NEW):
- next_imm_date: next 20th of Mar/Jun/Sep/Dec
- standard_cds_dates: IMM-based quarterly premium dates, snaps to previous IMM
- STANDARD_COUPONS_BPS: 100bp (IG), 500bp (HY) per ISDA Big Bang
- STANDARD_RECOVERY: 40% IG, 25% HY
- upfront_from_par_spread / par_spread_from_upfront: standard coupon conversion with round-trip
- CDSIndexSpec: CDX.NA.IG (125), CDX.NA.HY (100), iTraxx Europe (125), iTraxx Crossover (75)
- cds_index_roll_date: next Mar/Sep roll
- CDS_SETTLEMENT: auction, accrued on default (ISDA 2009)
- 24 new tests

---

## v0.275.0 — 2026-04-21

PR2: G10 FX Completion.

`currency.py`:
- All 10 G10 currencies defined (EUR, GBP, AUD, NZD, USD, CAD, CHF, NOK, SEK, JPY)
- ACI base priority: EUR > GBP > AUD > NZD > USD > CAD > CHF > NOK > SEK > JPY
- all_g10_pairs(): generates all 45 unique G10 cross pairs in market convention
- CurrencyPair.settlement_lag: T+1 for USD/CAD, T+2 for all others
- CurrencyPair.is_ndf: all G10 pairs are deliverable
- CurrencyPair.forward_rate(): CIP-based forward F = S × exp((r_q − r_b) × T)
- CurrencyPair.forward_points(): F − S

21 new tests: 45 pairs count, base/quote conventions (EUR/USD, GBP/USD, USD/JPY, AUD/USD,
USD/NOK, USD/SEK, USD/CHF), settlement lags, NDF flag, CIP forward pricing, inversion

---

## v0.274.0 — 2026-04-21

PR1: G10 Fixed Income Completion — Production Readiness begins.

`calendar.py`:
- SEKCalendar: Midsummer Eve, National Day, Epiphany, Ascension, Christmas Eve/NYE
- NOKCalendar: Constitution Day (17 May), Maundy Thursday, Ascension, Whit Monday
- NZDCalendar: Waitangi Day, Anzac Day, Queen's Birthday, Labour Day (4th Mon Oct)
- All 10 G10 currencies now have calendar implementations

`day_count.py`:
- THIRTY_E_360 (30E/360 Eurobond Basis): both d1 and d2 capped at 30 unconditionally
- Used for Bunds, Eurobonds, EUR corporate bonds
- Differs from US 30/360 when d2=31 and d1<30

`curve_builder.py`:
- Convention definitions for CHF, CAD, AUD, NZD, SEK, NOK
- All 10 G10 currencies now have explicit day count, frequency, interpolation conventions
- build_curves() works for all 10 currencies with correct market conventions

18 new tests: calendars (Midsummer, Constitution Day, Waitangi, etc.), 30E/360 vs 30/360, all G10 build_curves()

---

## v0.273.0 — 2026-04-21

P0: Round-Trip Repricing + Negative Rate Testing.

`bootstrap.py`:
- _verify_round_trip(): automatic verification after every bootstrap
  - Reprices all deposits (simple interest), swaps (par rate), FRAs (forward rate)
  - Emits RuntimeWarning if any instrument reprices with error > 1e-6
  - Catches bootstrap bugs before they propagate to pricing

`test_curve_robustness.py` (NEW, 13 tests):
- Round-trip: deposits, swaps, FRAs, full pipeline all reprice exactly
- Negative rates: EUR-style deposits (-50bp), swaps (-30bp), mixed regime
  - DFs > 1 for negative rates verified
  - Zero-crossing (negative→positive) curve works
- build_curves: USD normal + EUR negative via unified entry point
- Date precision: exact dates preserved through bootstrap

---

## v0.272.0 — 2026-04-21

CH4: Multi-Curve Newton + Validation — Curve Hardening Phase 4. CURVE HARDENING COMPLETE.

`multicurve_solver.py` (NEW):
- multicurve_newton: simultaneous OIS + projection bootstrap via Newton-Raphson
  - Numerical Jacobian with damped step control
  - Positive DF constraint enforced
  - Convergence warning if max_iter reached
- validate_curve: production checks (monotone DFs, forward rate bounds, gap detection, duplicates)
  - CurveValidationResult: is_valid, warnings list, forward rate range, n_pillars
- curve_analytical_jacobian: ∂zero_rate(t_i)/∂zero_rate(t_j) via finite difference
  - Returns full Jacobian matrix for real-time risk

---

## v0.271.0 — 2026-04-21

CH3: Pipeline Integration — Curve Hardening Phase 3.

`bootstrap.py`:
- New `futures` parameter: list of (start_date, end_date, futures_rate)
- Hull-White convexity adjustment integrated: `hw_convexity_a`, `hw_convexity_sigma`
- Turn-of-year spread: `turn_of_year_spread` adds basis for year-end crossing periods
- `bootstrap_forward_curve()` also accepts fras, futures, convexity, TOY params

`curve_builder.py` (NEW):
- build_curves(): unified entry point for curve construction
- Currency-specific conventions (USD, EUR, GBP, JPY day counts + frequencies)
- CurveSetResult: returns OIS + optional projection curve
- Handles full pipeline: deposits → FRAs → futures (with convexity + TOY) → swaps

---

## v0.270.0 — 2026-04-21

CH2: Date Precision + FRA Bootstrap — Curve Hardening Phase 2.

`discount_curve.py`:
- Store original pillar dates in `_pillar_dates_original` (eliminates `int(t*365)` drift)
- `pillar_dates` property now returns exact dates, not approximations

`bootstrap.py`:
- New `fras` parameter: list of (start_date, end_date, rate) for FRA bootstrap
- FRA relationship: df(end) = df(start) / (1 + rate × τ)
- Interpolates df(start) from existing pillars to chain FRA-implied discount factors
- FRAs bootstrapped between deposits (short end) and swaps (long end)

---

## Code & Numerical Review Fixes — 2026-04-20/21

Code lens review: 14 CRITICAL bugs fixed across 14 files.
Numerical lens review: 4 CRITICAL bugs fixed across 3 files.

Key fixes:
- convexity.py: CMS replication vol/T argument swap
- lmm_advanced.py: Rebonato swaption vol missing /T
- structural_credit.py: Merton/Black-Cox div-by-zero for T=0/vol=0
- fx_exotic.py + commodity_exotic.py: Asian geometric drift formula
- stochastic_correlation.py: Wishart covariance double √dt
- convertible_bond.py: coupon_step_interval zero guard
- commodity_real_options.py: mine DF off-by-one
- amortising_bond.py: PSA ramp m/30 → (m+1)/30
- bond_futures_options.py: quality option double-counted DV01
- inflation_bond_advanced.py: deflation floor deterministic units
- cross_asset_structured.py: fusion note payoff formula
- tail_risk.py: mixture model drift + EVT VaR formula
- rough_equity.py: fBM full kernel convolution + rough Heston CF integral
- hybrid_mc.py: GBM drift now includes risk-free rate

---

## v0.269.0 — 2026-04-21

CH1: Fix curve_engine + Solver Robustness — Curve Hardening begins.

`curve_engine.py`:
- build_curve() now delegates swap bootstrap to bootstrap.py (was incorrectly treating par rates as zero rates: df=exp(-r*t))
- Proper iterative Brent root-finding on discount factors for swap maturities
- Smith-Wilson extrapolation reads DFs from bootstrapped curve (not from broken formula)

`solvers.py`:
- brentq() now emits RuntimeWarning when |f(root)| > tol*1000 after maxiter (convergence check)
- Prevents silent solver failures from producing garbage prices

`bootstrap.py`, `ois.py`, `futures_bootstrap.py`:
- Brent bracket widened from [0.001, 1.5] to [1e-6, 3.0]
- Handles negative interest rates (DF > 1) up to approximately -3%

---

## v0.268.0 — 2026-04-20

Commodity-Rates Link — ALL DEEPENING COMPLETE. 297 slices, 4905 tests — Phase HY6 complete.

`commodity_rates_link.py`:
- CrossAssetPCAResult (class): 
- inflation_commodity_factor_model: PCA factor model across rates, commodity, and inflation series.
- CommodityInflationSwapResult (class): 
- commodity_inflation_swap: Commodity-linked inflation swap: floating = w_c × commodity_return + w_i × CPI_return.

---

## v0.267.0 — 2026-04-20

Hybrid XVA — Phase HY5 complete.

`hybrid_xva.py`:
- HybridCVAResult (class): 
- hybrid_cva: CVA on multi-asset exotic from exposure profiles.
- WrongWayRiskResult (class): 
- wrong_way_risk_adjustment: Wrong-way risk: equity down → credit spread up → higher default prob.
- HybridFVAResult (class): 
- hybrid_fva: FVA for long-dated hybrid: funding cost of uncollateralised exposure.
- Ref: Gregory, *Counterparty Credit Risk and CVA*, Wiley, 2012.; Brigo, Morini & Pallavicini, *Counterparty Credit Risk, Collateral and; Funding*, Wiley, 2013.

---

## v0.266.0 — 2026-04-20

Multi-Factor Hybrid MC — Phase HY4 complete.

`hybrid_mc.py`:
- HybridFactor (class): Definition of one factor in the hybrid.
- HybridMCResult (class): Multi-factor hybrid simulation result.
- HybridMCEngine (class): N-factor correlated MC engine for hybrid products.
- HybridPayoffResult (class): Evaluated hybrid payoff.
- hybrid_payoff_evaluate: Evaluate a path-dependent payoff on hybrid MC paths.
- simulate: 
- Ref: Piterbarg, *Smiling Hybrids*, Risk, 2006.; Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2003.

---

## v0.265.0 — 2026-04-20

Equity-Rates Hybrids — Phase HY3 complete.

`equity_rates_hybrid.py`:
- CallableEquityNoteResult (class): 
- callable_equity_note: Equity-linked note with issuer call right.
- JointSimResult (class): 
- equity_ir_joint_simulate: Joint equity + Hull-White rate simulation.
- HybridAutocallResult (class): 
- hybrid_autocallable: Autocall with IR floor: only autocalls if equity above barrier AND rate above floor.
- Ref: Overhaus et al., *Equity Hybrid Derivatives*, Wiley, 2007.

---

## v0.264.0 — 2026-04-20

Structural Credit — Phase HY2 complete.

`structural_credit.py`:
- MertonResult (class): 
- merton_equity_credit: Merton (1974): equity = call on assets; debt = assets − equity.
- KMVResult (class): 
- kmv_distance_to_default: KMV distance-to-default.
- BlackCoxResult (class): 
- black_cox_first_passage: Black-Cox: default occurs first time V hits barrier H (H < V₀).
- ImpliedCreditResult (class): 
- implied_credit_from_equity: Back out credit spread from equity vol and leverage.
- Ref: Merton, *On the Pricing of Corporate Debt*, JF, 1974.; Provisions*, JF, 1976.

---

## v0.263.0 — 2026-04-20

PRDC — Hybrids Deepening begins — Phase HY1 complete.

`prdc.py`:
- PRDCResult (class): 
- prdc_price: PRDC: coupon = fixed + participation × (FX / FX_strike − 1), floored at 0.
- CallablePRDCResult (class): 
- callable_prdc: Callable PRDC: issuer can call at par on coupon dates.
- Ref: Piterbarg, *Smiling Hybrids*, Risk, 2006.; Overhaus et al., *Equity Hybrid Derivatives*, Wiley, 2007.

---

## v0.262.0 — 2026-04-20

Vol Surface Stress — VOLATILITY DEEPENING COMPLETE — Phase VL6 complete.

`vol_stress.py`:
- VolBumpResult (class): 
- parallel_vol_bump: Parallel vol bump: shift all vols by bump_bps basis points.
- tilt_vol_bump: Tilt: short-end up, long-end down (steepening).
- twist_vol_bump: Twist/butterfly: wings up, belly down.
- VolReplayResult (class): 
- vol_scenario_replay: Replay a historical vol scenario on the current book.
- CrossAssetVolStressResult (class): 
- cross_asset_vol_stress: Correlated vol bump across asset classes.
- Ref: Alexander, *Market Risk Analysis*, Vol. IV, Wiley, 2008.

---

## v0.261.0 — 2026-04-20

Tail Risk & Far-OTM — Phase VL5 complete.

`tail_risk.py`:
- RogerLeeBoundsResult (class): 
- roger_lee_bounds: Roger Lee (2004): wing slope of total variance w(k) is bounded by 2.
- SVIWingsResult (class): 
- svi_wings_fit: SVI fit with Roger Lee wing constraints.
- TailRiskResult (class): 
- tail_risk_pricing: Deep OTM put pricing via heavy-tailed distribution (Pareto tail).
- EVTVaRResult (class): 
- extreme_value_var: VaR from Generalised Pareto Distribution (Peaks over Threshold).
- Ref: Gatheral & Jacquier, *Arbitrage-Free SVI Volatility Surfaces*, QF, 2014.; McNeil, Frey & Embrechts, *Quantitative Risk Management*, Princeton, 2015.

---

## v0.260.0 — 2026-04-20

Vol Term Structure — Phase VL4 complete.

`vol_term_structure.py`:
- ForwardVolResult (class): 
- forward_vol_from_term: 
- CalendarSpreadResult (class): 
- calendar_spread_strategy: 
- VolCurveShapeResult (class): 
- vol_curve_shape: 
- Bergomi2FactorResult (class): 
- Bergomi2Factor (class): Bergomi two-factor forward variance model.
- simulate: 
- Ref: Bergomi, *Stochastic Volatility Modeling*, CRC, 2016.; Gatheral, *The Volatility Surface*, Wiley, 2006.

---

## v0.259.0 — 2026-04-20

Skew Trading — Phase VL3 complete.

`skew_trading.py`:
- RiskReversalResult (class): 
- risk_reversal_strategy: 
- SkewMeanReversionSignal (class): 
- skew_mean_reversion_signal: 
- SkewCarryResult (class): 
- skew_carry_trade: 
- CrossAssetSkewResult (class): 
- cross_asset_skew_comparison: 
- Ref: Gatheral, *The Volatility Surface*, Wiley, 2006.; Bollen & Whaley, *Does Net Buying Pressure Affect the Shape of Implied Volatility Functions?*, JF, 2004.

---

## v0.258.0 — 2026-04-20

Vol Risk Premium — Phase VL2 complete.

`vol_risk_premium.py`:
- VRPResult (class): 
- vrp_single_asset: 
- VRPTermStructureResult (class): 
- vrp_term_structure: 
- CrossAssetVRPResult (class): 
- cross_asset_vrp_comparison: 
- VRPSignalResult (class): 
- vrp_strategy_signal: 
- Ref: Carr & Wu, *Variance Risk Premiums*, RFS, 2009.; Bollerslev, Tauchen & Zhou, *Expected Stock Returns and Variance Risk Premia*, RFS, 2009.

---

## v0.257.0 — 2026-04-20

Vol Model Comparison — Volatility Deepening begins — Phase VL1 complete.

`vol_model_comparison.py`:
- ModelPriceEntry (class): Price from one model.
- ModelComparisonResult (class): Comparison across models.
- compare_models: Price the same option under up to 4 models, report dispersion.
- ModelRiskResult (class): Model risk quantification across a strike grid.
- model_risk_quantification: Model risk across a strike grid.
- ModelGuideResult (class): Model selection recommendation.
- model_selection_guide: Recommend best vol model per product type.
- Ref: Gatheral, *The Volatility Surface*, Wiley, 2006.; Rebonato, *Volatility and Correlation*, Wiley, 2004.; Bergomi, *Stochastic Volatility Modeling*, CRC Press, 2016.

---

## v0.256.0 — 2026-04-18

Inflation-Commodity Link — INFLATION DEEPENING COMPLETE — Phase IN6 complete.

`inflation_commodity_link.py`:
- OilBreakevenRegressionResult (class): 
- oil_breakeven_regression: Regress breakeven changes on oil price changes.
- CommodityInflationHybridResult (class): 
- commodity_inflation_hybrid: Commodity-linked inflation swap: payoff = w_c × commodity_return + w_i × inflation − K.
- Ref: Hobijn, *Commodity Prices and Inflation*, FRBSF, 2008.

---

## v0.255.0 — 2026-04-18

Inflation Carry — Phase IN5 complete.

`inflation_carry.py`:
- RealYieldRolldownResult (class): 
- real_yield_rolldown: Roll-down on real yield curve.
- LinkerCarryResult (class): 
- linker_carry_decomposition: Linker carry = real yield carry + breakeven carry − financing.
- InflationCarryVolResult (class): 
- inflation_carry_vs_vol: Carry / vol ratio for inflation trades.
- Ref: Barclays, *US TIPS: A Guide for Investors*, 2012.

---

## v0.254.0 — 2026-04-18

Inflation Basis — Phase IN4 complete.

`inflation_basis.py`:
- ZCYoYBasisResult (class): 
- zc_yoy_basis: ZC vs YoY inflation swap basis.
- CrossMarketBasisResult (class): 
- cross_market_inflation_basis: Cross-market basis: HICP vs CPI vs RPI.
- InflationBasisTradeResult (class): 
- inflation_basis_trade: Construct basis trade: long one leg, short another.
- Ref: Kerkhof, *Inflation Derivatives Explained*, Lehman Brothers, 2005.; Deacon et al., *Inflation-Indexed Securities*, Wiley, 2004.

---

## v0.253.0 — 2026-04-18

CPI Seasonality — Phase IN3 complete.

`cpi_seasonality.py`:
- SeasonalFactors (class): 
- estimate_seasonal_factors: Estimate monthly CPI seasonal factors from historical data.
- DeseasonalisedBreakevenResult (class): 
- deseasonalise_breakeven: Remove CPI seasonality from observed breakeven.
- SeasonalCarrySignal (class): 
- seasonal_carry_signal: Carry signal from CPI seasonality.
- Ref: Canty & Heider, *Seasonality in CPI*, BIS, 2012.; Barclays, *Inflation Seasonality and TIPS Valuation*, 2011.

---

## v0.252.0 — 2026-04-18

Inflation Smile — Phase IN2 complete.

`inflation_smile.py`:
- InflationSmileNode (class): 
- calibrate_inflation_sabr: Calibrate SABR to inflation caplet smile at one tenor.
- InflationVolCube (class): 
- ZCCapSmileResult (class): 
- zc_inflation_cap_smile: ZC inflation cap prices at multiple strikes.
- obj: 
- vol: 
- Ref: Mercurio, *Pricing Inflation-Indexed Derivatives*, QF, 2005.; Kenyon, *Inflation Is Normal*, Risk, 2008.

---

## v0.251.0 — 2026-04-18

Jarrow-Yildirim — Inflation Deepening begins — Phase IN1 complete.

`jarrow_yildirim.py`:
- JYParams (class): Jarrow-Yildirim model parameters.
- JYSimulationResult (class): JY simulation result.
- JarrowYildirim (class): Jarrow-Yildirim three-factor inflation model.
- JYZCSwapResult (class): JY ZC inflation swap result.
- jy_zc_inflation_swap: Analytical ZC inflation swap rate under JY.
- JYCapletResult (class): JY YoY inflation caplet result.
- jy_yoy_caplet: YoY inflation caplet under JY.
- JYCalibrationResult (class): JY calibration result.
- jy_calibrate: Calibrate JY (σ_n, σ_r, σ_I) to ZC inflation swap term structure.
- simulate: Simulate joint (r_n, r_r, I) paths.
- hw_zcb: 
- objective: 
- Ref: Related Derivatives Using an HJM Model*, JFE, 2003.; Mercurio, *Pricing Inflation-Indexed Derivatives*, QF, 2005.

---

## v0.250.0 — 2026-04-18

Cross-Gamma Hedging — OPTIONS DEEPENING COMPLETE — Phase OP8 complete.

`cross_gamma_hedging.py`:
- OptimalHedgeResult (class): Optimal multi-asset hedge result.
- optimal_multi_asset_hedge: Find hedge weights to neutralise target Greeks.
- VegaNettingResult (class): Cross-asset vega netting result.
- cross_asset_vega_netting: Net vega across asset classes.
- CorrelationAwareSizingResult (class): Position sizing with correlation constraints.
- correlation_aware_sizing: Size a new position accounting for correlation to existing book.
- MinVarianceExoticHedgeResult (class): Minimum-variance hedge for multi-asset exotic.
- minimum_variance_exotic_hedge: Find optimal basket of vanilla hedges for a multi-asset exotic.
- Ref: Alexander, *Market Risk Analysis*, Vol. IV, Wiley, 2008.

---

## v0.249.0 — 2026-04-18

Correlation Monitoring — Phase OP7 complete.

`correlation_monitor.py`:
- ImpliedRealisedCorrResult (class): 
- implied_vs_realised_correlation: Track implied − realised correlation spread.
- CorrTermStructureResult (class): 
- correlation_term_structure: Term structure of implied correlation.
- CorrStressResult (class): 
- correlation_stress_matrix: Apply stress to correlation matrix.
- SmileArbCheckResult (class): 
- multi_asset_smile_arb_check: Check if basket vol is consistent with constituent smiles + implied ρ.
- Ref: Bossu, *Advanced Equity Derivatives*, Wiley, 2014.; Alexander, *Market Risk Analysis*, Vol. IV, Wiley, 2008.

---

## v0.248.0 — 2026-04-18

Multi-Asset Local Vol — Phase OP6 complete.

`multi_asset_local_vol.py`:
- LocalVol2DResult (class): 2D local vol result.
- dupire_2d_local_vol: Compute marginal local vol surfaces for two assets.
- MultiAssetSLVResult (class): Multi-asset SLV simulation result.
- multi_asset_slv_simulate: 2-asset SLV: each asset has LV + shared Heston-like stochastic vol.
- SmileConsistencyResult (class): Basket smile consistency check.
- smile_consistency_check: Check if basket vol is consistent with constituent smiles.
- Ref: Guyon & Henry-Labordère, *Nonlinear Option Pricing*, CRC, 2014.

---

## v0.247.0 — 2026-04-18

Cross-Asset Structured Notes — Phase OP5 complete.

`cross_asset_structured.py`:
- FusionNoteResult (class): 
- equity_fx_fusion_note: Equity performance paid in foreign currency.
- CorrelationTriggerResult (class): 
- correlation_trigger_note: Coupon paid if realised correlation stays below threshold.
- CommodityEquityAutocallResult (class): 
- commodity_equity_autocall: Equity autocall with commodity knock-out.
- DualRangeAccrualResult (class): 
- dual_asset_range_accrual: Accrues coupon for each observation where BOTH assets are in range.
- Ref: Bouzoubaa & Osseiran, *Exotic Options and Hybrids*, Wiley, 2010.; De Weert, *Exotic Options Trading*, Wiley, 2008.

---

## v0.246.0 — 2026-04-18

Multi-Asset Exotics — Phase OP4 complete.

`multi_asset_exotic.py`:
- RainbowResult (class): Rainbow option result.
- rainbow_option: N-asset rainbow option.
- KnockoutBasketResult (class): Knockout basket result.
- knockout_basket: Knockout basket: barrier on one asset, payoff on another.
- ConditionalBarrierResult (class): Conditional barrier result.
- conditional_barrier: Conditional barrier: knock-in on one asset, knock-out on another.
- MultiAssetDigitalRangeResult (class): Multi-asset digital range result.
- multi_asset_digital_range: Digital range on multiple assets: pays if ALL stay in their range.
- Ref: De Weert, *Exotic Options Trading*, Wiley, 2008.; Bouzoubaa & Osseiran, *Exotic Options and Hybrids*, Wiley, 2010.

---

## v0.245.0 — 2026-04-18

Vol-of-Vol Derivatives — Phase OP3 complete.

`vol_vol_derivatives.py`:
- OptionOnVarianceResult (class): Option on variance swap result.
- option_on_variance_swap: Option on realised variance (vol-of-vol product).
- GammaSwapResult (class): Gamma swap result.
- gamma_swap_price: Gamma swap: variance swap weighted by S(t) / S(0).
- CorridorVarianceResult (class): Corridor variance swap result.
- corridor_variance_swap: Corridor variance swap: accumulates variance only when spot in range.
- VIXOptionResult (class): VIX-like option result.
- vix_option_price: VIX-like option via Heston variance simulation.

---

## v0.244.0 — 2026-04-18

Stochastic Correlation — Phase OP2 complete.

`stochastic_correlation.py`:
- CIRCorrelationResult (class): CIR correlation simulation result.
- CIRCorrelation (class): Mean-reverting correlation via transformed CIR process.
- StochCorrPricingResult (class): Multi-asset pricing with stochastic correlation.
- simulate_two_asset_stoch_corr: Simulate two assets with CIR stochastic correlation.
- WishartResult (class): Wishart covariance simulation result.
- WishartCovariance (class): Wishart covariance matrix process.
- DispersionCalibrationResult (class): Stochastic correlation calibration to dispersion result.
- calibrate_stoch_corr_to_dispersion: Calibrate CIR correlation to match index variance.
- simulate: 
- simulate: Simplified Wishart: simulate diagonal (variances) via CIR,
- model_index_var: 
- Ref: J. Math. in Industry, 2016.; Stochastic: An Analytical Framework*, RFS, 2007.

---

## v0.243.0 — 2026-04-18

Correlation Greeks — Options Deepening begins — Phase OP1 complete.

`correlation_greeks.py`:
- CorrelationDeltaResult (class): Correlation delta result.
- correlation_delta: Correlation delta: ∂V/∂ρ via central difference.
- CorrelationGammaResult (class): Correlation gamma result.
- correlation_gamma: Correlation gamma: ∂²V/∂ρ².
- CrossGammaResult (class): Cross-gamma: ∂²V/∂S₁∂S₂.
- cross_gamma: Cross-gamma: ∂²V/∂S₁∂S₂ via finite difference.
- CorrelationPnLAttribution (class): P&L attribution from correlation changes.
- correlation_pnl_attribution: Attribute P&L to correlation changes via Taylor expansion.
- CorrelationLadderEntry (class): One entry in the correlation sensitivity ladder.
- CorrelationLadder (class): Full correlation sensitivity ladder.
- correlation_sensitivity_ladder: Compute correlation sensitivity for each pair in the ρ matrix.
- Ref: Bossu, *Advanced Equity Derivatives: Volatility and Correlation*, Wiley, 2014.; Alexander, *Market Risk Analysis*, Vol. IV, Wiley, 2008.; De Weert, *Exotic Options Trading*, Wiley, 2008.

---

## v0.242.0 — 2026-04-17

Inflation Bonds — BOND DEEPENING COMPLETE — Phase BN8 complete.

`inflation_bond_advanced.py`:
- RealYieldCurveResult (class): Real yield curve bootstrap result.
- real_yield_curve_bootstrap: Bootstrap real yield curve from TIPS/linker prices.
- BreakevenTradeResult (class): Breakeven trade decomposition.
- breakeven_trade: Breakeven = nominal yield − real yield.
- SeasonalBreakevenResult (class): Seasonality-adjusted breakeven.
- seasonality_adjusted_breakeven: Adjust breakeven for CPI seasonality.
- LinkerASWResult (class): Linker (real) asset swap spread.
- linker_asw: Real asset swap spread = linker real yield − real swap rate.
- DeflationFloorResult (class): Deflation floor option value.
- deflation_floor_value: TIPS deflation floor: guaranteed return of par at maturity.
- Ref: Deacon, Derry & Mirfendereski, *Inflation-Indexed Securities*, Wiley, 2004.; Barclays, *US TIPS: A Guide for Investors*, 2012.; Kerkhof, *Inflation Derivatives Explained*, Lehman Brothers, 2005.

---

## v0.241.0 — 2026-04-17

Duration Hedging Deepening — Phase BN7 complete.

`duration_advanced.py`:
- ImmunisationResult (class): Key rate immunisation result.
- key_rate_immunise: Solve for hedge weights to immunise all key rate durations.
- BarbellBulletResult (class): Barbell vs bullet comparison.
- barbell_bullet_analysis: Barbell (short + long) vs bullet comparison.
- LDIMatchResult (class): LDI cash-flow matching result.
- ldi_cashflow_match: Liability-driven investment cash-flow matching.
- CrossCurrencyHedgeResult (class): Cross-currency duration hedge.
- cross_currency_duration_hedge: Cross-currency duration hedge ratio.
- Ref: Martellini, Priaulet & Priaulet, *Fixed-Income Securities*, Wiley, 2003.

---

## v0.240.0 — 2026-04-17

Advanced Bond RV — Phase BN6 complete.

`bond_rv_advanced.py`:
- IssuerCurveFitResult (class): Issuer curve fit result.
- issuer_curve_fit: Fit Nelson-Siegel to a single issuer's bonds. Residuals = rich/cheap.
- InvoiceSpreadResult (class): Invoice spread result.
- invoice_spread: Invoice spread: bond ASW − futures-implied ASW.
- OISASWResult (class): OIS-asset swap decomposition.
- ois_asw_decomposition: IBOR-OIS asset swap decomposition.
- PCAResult (class): PCA on bond yield changes.
- bond_yield_pca: PCA on bond yield changes (Litterman-Scheinkman).
- objective: 
- Ref: Tuckman & Serrat, *Fixed Income Securities*, Wiley, 2012.

---

## v0.239.0 — 2026-04-17

Bond Futures Optionality — Phase BN5 complete.

`bond_futures_options.py`:
- EOMOptionResult (class): End-of-month (wild card) option value.
- end_of_month_option: End-of-month wild card option.
- QualityOptionResult (class): Quality option (CTD switch) result.
- quality_option: Quality option: option to deliver cheapest bond.
- TimingOptionResult (class): Timing option value.
- timing_option: Timing option: short chooses when in the delivery month to deliver.
- NetBasisResult (class): Net basis decomposition.
- net_basis_decomposition: Net basis = gross basis − carry.
- JointDeliveryResult (class): Joint delivery option valuation.
- joint_delivery_option_value: Combined delivery option value (with correlation discount).
- Ref: Burghardt & Belton, *The Treasury Bond Basis*, McGraw-Hill, 2005.; Tuckman & Serrat, *Fixed Income Securities*, Wiley, 2012, Ch. 14.

---

## v0.238.0 — 2026-04-17

Advanced Repo & Financing — Phase BN4 complete.

`repo_advanced.py`:
- RepoCurve (class): Repo curve: tenor → rate (simple).
- build_repo_curve: Construct a repo curve from observed tenor points.
- RepoOISSpread (class): Repo spread vs OIS.
- repo_spread_to_ois: GC repo vs OIS spread.
- SpecialBond (class): Bond trading special in repo.
- identify_specials: Identify bonds trading special (lower repo rate than GC).
- RepoCounterparty (class): A counterparty for repo financing.
- FinancingPlan (class): Result of multi-counterparty financing optimisation.
- optimise_financing: Greedy optimisation: allocate to cheapest counterparties first.
- HaircutCurve (class): Repo haircut term structure by asset type.
- repo_haircut_curve: Typical haircut curve by asset type.
- rate_at: Rate at a given tenor (days).
- financing_cost: Financing cost for borrowing at this repo rate.
- haircut_at: 

---

## v0.237.0 — 2026-04-17

Sovereign Bond Trading — Phase BN3 complete.

`sovereign_bond.py`:
- SovereignSpreadCurve (class): Term structure of sovereign spreads over a benchmark.
- build_spread_curve: Build sovereign spread curve from yields.
- SovereignBasisResult (class): Sovereign basis: bond vs CDS vs futures.
- sovereign_basis: Decompose sovereign basis.
- CrossCountryRV (class): Cross-country sovereign RV analysis.
- cross_country_rv: Current cross-country yield spread vs historical distribution.
- AuctionResult (class): Auction analytics result.
- auction_analytics: Analyse a sovereign auction.
- OTROFRResult (class): On-the-run vs off-the-run comparison.
- otr_ofr_analysis: Compare OTR vs OFR yields; detect squeezes.
- spread_at: Spread (in bps) at tenor T, interpolated.
- z_score: Z-score of current spread vs historical distribution.
- Ref: RFS, 2009.; Mgmt. Sci., 2014.

---

## v0.236.0 — 2026-04-17

Amortising & Sinker Bonds — Phase BN2 complete.

`amortising_bond.py`:
- AmortisingBondResult (class): Amortising bond pricing result.
- AmortisingBond (class): Bond with amortising principal schedule.
- cpr_to_smm: Convert Conditional Prepayment Rate (annual) to Single Monthly Mortality.
- psa_schedule: PSA (Public Securities Association) standard prepayment schedule.
- PrepaymentBondResult (class): Prepayment bond pricing result.
- prepayment_bond_price: Price a bond with PSA prepayment model.
- average_life: Principal-weighted average life.
- weighted_average_maturity: Weighted average maturity across a portfolio of bonds.
- SinkerComparisonResult (class): Sinker vs bullet comparison.
- sinker_vs_bullet: Compare amortising sinker vs bullet bond with same coupon/maturity.
- schedule: Return payment schedule: [(time, principal_paid, interest_paid), ...].
- price: Price, duration, DV01 at a flat discount rate.
- Ref: Fabozzi, *Handbook of Mortgage-Backed Securities*, 7th ed., McGraw-Hill, 2016.; Tuckman & Serrat, *Fixed Income Securities*, Wiley, 2012.; Securities*, Wiley, 2001.

---

## v0.235.0 — 2026-04-17

Convertible Bonds — Bond Deepening begins — Phase BN1 complete.

`convertible_bond.py`:
- ConvertibleResult (class): Convertible bond pricing result.
- ConvertibleBond (class): Convertible bond specification.
- DeltaHedgeResult (class): Convertible delta hedge result.
- convertible_delta_hedge: Compute shares to short for delta-neutral CB hedge.
- SoftCallResult (class): Soft call convertible result.
- convertible_soft_call: Soft call convertible: issuer can call at par (or call price) if stock
- CoCoResult (class): Contingent convertible result.
- contingent_convertible: Contingent convertible: mandatorily converts when equity crosses trigger.
- ExchangeableResult (class): Exchangeable bond result (convertible into different issuer's stock).
- exchangeable_bond: Exchangeable bond: convertible into another issuer's stock.
- MandatoryConvertibleResult (class): Mandatory convertible bond result.
- mandatory_convertible: Mandatory convertible (MC): forced conversion at maturity.
- parity: Conversion value / notional.
- conversion_price: Stock price at which conversion_value = notional.
- price: MC pricing with optimal holder conversion.
- Ref: Tsiveriotis & Fernandes, *Valuing Convertible Bonds with Credit Risk*, JF, 1998.; Calamos, *Convertible Securities*, McGraw-Hill, 2003.

---

## v0.234.0 — 2026-04-17

Commodity Structured — COMMODITY DEEPENING COMPLETE — Phase CM8 complete.

`commodity_structured.py`:
- CommodityAutocallResult (class): Commodity autocallable result.
- commodity_autocallable: Commodity autocallable (gold/oil/wheat).
- CommodityLinkedBondResult (class): Commodity-linked bond result.
- commodity_linked_bond: Bond with coupons tied to commodity performance.
- CommodityTARFResult (class): Commodity TARF result.
- commodity_tarf: Commodity TARF: accumulates P&L until target reached, then terminates.
- CommodityRangeAccrualResult (class): Commodity range accrual result.
- commodity_range_accrual: Commodity range accrual: pays coupon × fraction_of_days_in_range.
- DualCommodityResult (class): Dual commodity note result.
- dual_commodity_note: Dual commodity note: long one, short another.
- Ref: Clewlow & Strickland, *Energy Derivatives*, Wiley, 2000, Ch. 9.; Geman, *Commodities and Commodity Derivatives*, Wiley, 2005.; Bouzoubaa & Osseiran, *Exotic Options and Hybrids*, Wiley, 2010.

---

## v0.233.0 — 2026-04-17

Commodity Real Options — Phase CM7 complete.

`commodity_real_options.py`:
- MineValuationResult (class): Mine valuation result.
- MineValuation (class): Brennan-Schwartz (1985) mine valuation with output flexibility.
- PowerPlantResult (class): Power plant dispatch valuation.
- power_plant_dispatch_value: Simplified power plant dispatch value (spark spread).
- UnitCommitmentResult (class): Unit commitment with startup/shutdown costs.
- unit_commitment_value: Simplified unit commitment with startup/shutdown costs & min up/down times.
- FTRResult (class): Financial Transmission Rights valuation.
- pipeline_ftr_value: Financial Transmission Rights (FTR) valuation.
- value: Value the mine with optimal operation (naive: operate when S > c).
- Ref: Dixit & Pindyck, *Investment Under Uncertainty*, Princeton UP, 1994.; Electric Power Plants in Competitive Markets*, Ops. Res., 2004.

---

## v0.232.0 — 2026-04-17

Weather Derivatives — Phase CM6 complete.

`weather_derivatives.py`:
- DegreeDayIndex (class): Degree days index result.
- hdd_index: Heating Degree Days: Σ max(reference − T_mean, 0).
- cdd_index: Cooling Degree Days: Σ max(T_mean − reference, 0).
- WeatherFutureResult (class): HDD/CDD futures pricing result.
- hdd_future_price: CME HDD futures price.
- WeatherOptionResult (class): HDD/CDD option pricing result.
- hdd_option_price: HDD call/put option (with optional cap).
- TemperaturePaths (class): Simulated daily temperature paths.
- SeasonalOUTemperature (class): Alaton-Djehiche-Stillberger seasonal OU temperature model.
- RainfallResult (class): Rainfall derivative pricing result.
- rainfall_derivative_price: Simple rainfall derivative via Poisson-Gamma model.
- WindOptionResult (class): Wind index option result.
- wind_index_option: Wind index option (e.g. average wind speed over a period).
- seasonal_mean: Expected temperature at day t_days.
- simulate: 
- Ref: Derivatives*, Applied Math. Finance, 2002.; Brody, Syroka & Zervos, *Dynamical Pricing of Weather Derivatives*, QF, 2002.; Jewson & Brix, *Weather Derivative Valuation*, Cambridge UP, 2005.

---

## v0.231.0 — 2026-04-17

Commodity Basis & Locational Spreads — Phase CM5 complete.

`commodity_basis.py`:
- BasisCurve (class): Term structure of basis differentials.
- basis_curve_from_futures: Construct basis curve from aligned forward prices.
- WTIBrentResult (class): WTI/Brent basis analysis.
- wti_brent_basis: Analyse WTI/Brent basis (typically Brent > WTI since ~2011).
- PowerLocationalBasis (class): Power locational basis result (e.g. PJM hub vs zone).
- power_locational_basis: Locational basis for power (hub vs zonal or nodal pricing).
- GasBasisResult (class): Gas basis result (HH vs regional hub).
- gas_basis_curve: Gas basis between Henry Hub and a regional hub.
- QualityBasisResult (class): Quality basis adjustment result.
- quality_basis: Quality-adjusted commodity price.
- basis: Interpolated basis at tenor T.
- forward_price: Derivative forward price at tenor T.
- Ref: Clewlow & Strickland, *Energy Derivatives*, Wiley, 2000, Ch. 8.; Eydeland & Wolyniec, *Energy and Power Risk Management*, Wiley, 2003.

---

## v0.230.0 — 2026-04-17

Commodity Vol Surface — Phase CM4 complete.

`commodity_vol_surface.py`:
- CommoditySmileNode (class): SABR calibration at one commodity tenor.
- calibrate_commodity_sabr: Calibrate SABR (α, ρ, ν) to market smile at one tenor.
- CommodityVolCube (class): Commodity vol cube: SABR smile per tenor.
- build_commodity_cube: Build cube from per-tenor smiles.
- KirkResult (class): Kirk spread option result.
- kirk_spread_smile: Kirk (1995) spread option with optional smile adjustment.
- objective: 
- vol: Implied vol at (T, K) via parameter interpolation.
- atm_vol: 
- forward: Forward price at tenor T (interpolated).
- Ref: Clewlow & Strickland, *Energy Derivatives*, Wiley, 2000.

---

## v0.229.0 — 2026-04-17

Commodity Exotic Options — Phase CM3 complete.

`commodity_exotic.py`:
- CommodityBarrierResult (class): Commodity barrier option result.
- commodity_barrier_smile: Commodity barrier option with Vanna-Volga smile adjustment.
- CommodityLookbackResult (class): Commodity lookback option result.
- commodity_lookback: Commodity lookback option.
- CommodityAsianResult (class): Commodity Asian option result.
- commodity_asian_monthly: Commodity Asian option — monthly averaging (standard oil/gas settlement).
- QuantoCommodityResult (class): Quanto commodity option result.
- quanto_commodity_option: Quanto commodity option: payoff in different currency than commodity.
- Ref: Clewlow & Strickland, *Energy Derivatives*, Wiley, 2000, Ch. 5-7.; JBF, 1990.

---

## v0.228.0 — 2026-04-17

Commodity Swing Options — Phase CM2 complete.

`commodity_swing.py`:
- SwingOptionResult (class): Swing option pricing result.
- swing_option_lsm: Swing option via Longstaff-Schwartz with exercise-count state.
- VirtualStorageResult (class): Virtual gas storage optimisation result.
- VirtualGasStorage (class): Virtual gas storage facility valuation.
- NominationResult (class): Nomination rights contract value.
- nomination_rights_value: Value of daily nomination rights in a commodity supply contract.
- intrinsic_value: Intrinsic value: optimal deterministic schedule on forward curve.
- value: Full stochastic valuation via LSM-DP.
- Ref: Longstaff & Schwartz, *Valuing American Options by Simulation*, RFS, 2001.; Exercise Options*, Math. Finance, 2004.; Mgmt. Sci., 2004.

---

## v0.227.0 — 2026-04-17

Commodity Stochastic Models — Commodity Deepening begins — Phase CM1 complete.

`commodity_models.py`:
- SchwartzOneFactorResult (class): Schwartz one-factor simulation result.
- SchwartzOneFactor (class): Schwartz (1997) Model 1: mean-reverting log-spot.
- GibsonSchwartzResult (class): Gibson-Schwartz two-factor simulation result.
- GibsonSchwartz (class): Gibson-Schwartz (1990) two-factor commodity model.
- SchwartzSmithResult (class): Schwartz-Smith long-short decomposition result.
- SchwartzSmith (class): Schwartz-Smith (2000) long-short two-factor decomposition.
- CommodityJumpResult (class): Commodity jump-diffusion simulation result.
- CommodityJumpDiffusion (class): Merton-style jump-diffusion for commodities.
- forward_price: Analytical forward (futures) price F(S, T).
- simulate: 
- forward_price: Analytical forward price.
- simulate: 
- forward_price: Analytical forward price.
- simulate: 
- simulate: 
- Ref: Commodity Prices*, Mgmt. Sci., 2000.; Clewlow & Strickland, *Energy Derivatives*, Wiley, 2000, Ch. 2-3.

---

## v0.226.0 — 2026-04-17

Equity Jumps & Hybrids — EQUITY DEEPENING COMPLETE — Phase EQ8 complete.

`equity_jumps.py`:
- KouResult (class): Kou jump-diffusion pricing result.
- kou_equity_price: Kou double-exponential jump-diffusion for equity.
- SVJResult (class): SVJ pricing result.
- SVJEquityModel (class): SVJ model: Heston + Merton jumps for equity.
- RegimeResult (class): Regime-switching equity result.
- RegimeSwitchingEquity (class): Bull/bear/crisis regime-switching equity model.
- MertonHybridResult (class): Merton jump-diffusion equity result.
- merton_equity_hybrid: Merton jump-diffusion for equity (crash risk premium).
- simulate_option: 
- simulate: 
- Ref: Kou, *A Jump-Diffusion Model for Option Pricing*, Mgmt Sci, 2002.; Duffie-Pan-Singleton, *Transform Analysis and Option Pricing*, Econometrica, 2000.; Series and the Business Cycle*, Econometrica, 1989.

---

## v0.225.0 — 2026-04-17

Rough Vol for Equity — Phase EQ7 complete.

`rough_equity.py`:
- rBergomiResult (class): rBergomi simulation result.
- rBergomiEquity (class): rBergomi model for equity.
- RoughHestonParams (class): Rough Heston parameters.
- rough_heston_cf: Characteristic function for rough Heston.
- rough_heston_price: European option price under rough Heston via Fourier inversion.
- ForwardVarianceCurve (class): Forward variance curve ξ(T).
- forward_variance_curve: Bootstrap forward variance curve from ATM variance term structure.
- simulate: Simulate rBergomi spot and variance paths.
- forward_variance: Forward variance ξ(T). Constant in this simplified model.
- implied_vol: Approximate implied vol from MC price.
- Ref: Bayer, Friz & Gatheral, *Pricing Under Rough Volatility*, QF, 2016.; Math. Finance, 2019.; Gatheral, Jaisson & Rosenbaum, *Volatility is Rough*, QF, 2018.

---

## v0.224.0 — 2026-04-17

Equity Basket & Correlation — Phase EQ6 complete.

`equity_basket.py`:
- MargrabeEquityResult (class): Margrabe exchange option result.
- margrabe_equity: Margrabe exchange: max(q₁ S₁ − q₂ S₂, 0) at T.
- MaxMinResult (class): Max/min of 2 assets option result.
- johnson_max_call: Call on max(S₁, S₂) via Johnson (1987) / Stulz (1982).
- johnson_min_call: Call on min(S₁, S₂).
- EquityBasketResult (class): Equity basket option result.
- equity_basket_mc: Multi-asset equity basket option.
- CorrelationSwapResult (class): Correlation swap result.
- correlation_swap_price: Correlation swap: pays (realised − strike) × notional.
- implied_correlation_from_dispersion: Implied correlation from index variance and constituent variances.
- DispersionTradeResult (class): Dispersion trade value.
- dispersion_trade_value: Classic dispersion trade: long single-name variance, short index variance.
- Ref: Margrabe, *The Value of an Option to Exchange One Asset for Another*, JF, 1978.; Stulz, *Options on the Minimum or Maximum of Two Risky Assets*, JFE, 1982.; Bossu, *Advanced Equity Derivatives: Volatility and Correlation*, Wiley, 2014.

---

## v0.223.0 — 2026-04-17

Dividend Deepening — Phase EQ5 complete.

`dividend_advanced.py`:
- BuhlerResult (class): Bühler stochastic dividend simulation result.
- BuhlerStochasticDividend (class): Bühler stochastic dividend yield model.
- DividendCurve (class): Bootstrapped dividend curve from futures.
- dividend_curve_bootstrap: Bootstrap dividend curve from index / single-stock dividend futures.
- ImpliedDividendResult (class): Implied dividend yield from put-call parity.
- implied_dividend_yield: Extract implied dividend yield from put-call parity.
- DividendBasisResult (class): Dividend basis (cash vs futures) analytics.
- dividend_basis_trade: Dividend basis: cash dividend PV vs dividend future.
- DividendHedgeResult (class): Dividend hedge ratio result.
- dividend_hedge_ratio: Hedge ratio for dividend risk using dividend futures.
- simulate: 
- implied_forward: Model-implied forward from MC.
- Ref: Kragt, *Managing Dividend Risk*, Risk, 2015.

---

## v0.222.0 — 2026-04-17

Equity Smile & Vol Cube — Phase EQ4 complete.

`equity_smile.py`:
- SSVIParams (class): Surface SVI parameters.
- ssvi_fit: Fit SSVI to a grid of observed smiles.
- ssvi_vol: Implied vol from SSVI at log-moneyness k, tenor T.
- EquityCubeNode (class): SABR parameters at one expiry for equity.
- EquityVolCube (class): Equity vol cube with SABR smile per expiry.
- calibrate_equity_sabr_tenor: Calibrate SABR (α, ρ, ν) at one equity tenor to ATM + 25D smile.
- build_equity_vol_cube: Build equity vol cube from market quotes per tenor.
- ForwardVolResult (class): Forward-starting volatility result.
- forward_vol: Forward vol from cube: vol over [T1, T2].
- SmileRegimeResult (class): Smile dynamics regime (sticky strike vs delta).
- sticky_strike_dynamics: Vol at fixed strike, bumped spot (sticky strike convention).
- sticky_delta_dynamics: Vol at fixed delta under sticky delta regime (vol follows spot).
- objective: 
- vol: Implied vol at (T, K) with SABR parameter interpolation.
- atm_vol: ATM vol at tenor T.
- Ref: Gatheral & Jacquier, *Arbitrage-Free SVI Volatility Surfaces*, QF, 2014.; Gatheral, *The Volatility Surface*, Wiley, 2006.; Bergomi, *Stochastic Volatility Modeling*, CRC Press, 2016.

---

## v0.221.0 — 2026-04-17

Equity Structured Products — Phase EQ3 complete.

`equity_structured.py`:
- EquityAutocallableResult (class): Equity autocallable pricing result.
- equity_autocallable: Phoenix autocallable with memory coupon.
- WorstOfAutocallResult (class): Worst-of autocallable result.
- worst_of_autocallable: Autocallable on worst-of basket: autocall when min(S_i / S_i^0) ≥ barrier.
- ReverseConvertibleResult (class): Reverse convertible note result.
- reverse_convertible: Reverse convertible: bond with enhanced coupon + embedded short put.
- SharkFinResult (class): Shark-fin note result.
- shark_fin_note: Shark-fin: capped call with knock-out + rebate.
- AirbagResult (class): Airbag note (capped-floored) result.
- airbag_note: Airbag / capped-floored note.
- Ref: Bouzoubaa & Osseiran, *Exotic Options and Hybrids*, Wiley, 2010.

---

## v0.220.0 — 2026-04-17

Variance Derivatives — Phase EQ2 complete.

`variance_derivatives.py`:
- VarianceSwapResult (class): Variance swap pricing result.
- variance_swap_replication: Fair variance via Demeterfi-Derman-Kamal-Zou static replication.
- VolatilitySwapResult (class): Volatility swap result.
- volatility_swap_heston: Fair vol swap strike under Heston via expected realised variance.
- brockhaus_long_approx: Brockhaus-Long approximation for vol swap.
- VarianceFuturesResult (class): Variance futures / VIX-like index result.
- variance_future_price: VIX-like variance index from option strip (CBOE methodology).
- VRPResult (class): Variance risk premium decomposition.
- variance_risk_premium: Variance risk premium = implied − realised variance.
- Ref: Variance Swaps*, Goldman Sachs Quantitative Strategies Research Notes, 1999.; Brockhaus & Long, *Volatility Swaps Made Simple*, Risk, 2000.

---

## v0.219.0 — 2026-04-17

Equity Exotic Options — Equity Deepening begins — Phase EQ1 complete.

`equity_exotic.py`:
- EquityBarrierResult (class): Equity barrier option result.
- equity_barrier_smile: Equity barrier option with Vanna-Volga smile adjustment.
- DigitalResult (class): Digital option result.
- equity_digital_cash: Cash-or-nothing digital: pays `payout` if S_T > K (call) or S_T < K (put).
- equity_digital_asset: Asset-or-nothing digital: pays S_T if S_T > K (call) or S_T < K (put).
- EquityLookbackResult (class): Equity lookback option result.
- equity_lookback_floating: Floating-strike lookback for equity: payoff S_T − min(S) (call).
- equity_lookback_fixed: Fixed-strike lookback: payoff max(max(S) − K, 0) (call).
- CompoundResult (class): Compound option result (Geske 1979).
- equity_compound_option: Compound option: option on an option.
- inner_call: 
- Ref: * :func:`equity_lookback_floating` — Goldman-Sosin-Gatto floating strike.; Geske, *The Valuation of Compound Options*, JFE, 1979.; Goldman, Sosin & Gatto, *Path Dependent Options*, JF, 1979.

---

## v0.218.0 — 2026-04-17

FX Jumps & Regime — FX DEEPENING COMPLETE — Phase FX8 complete.

`fx_jumps.py`:
- MertonFXResult (class): Merton jump-diffusion FX pricing result.
- merton_fx_price: Merton jump-diffusion for FX options.
- BatesFXResult (class): Bates (Heston + jumps) FX simulation result.
- BatesFXModel (class): Bates model for FX: Heston + Merton jumps.
- RegimeSwitchingResult (class): Regime-switching simulation result.
- RegimeSwitchingVol (class): Markov regime-switching FX vol model.
- InterventionResult (class): FX peg break / intervention adjustment result.
- fx_intervention_adjustment: Intervention/peg-break risk adjustment for FX option.
- simulate_option: 
- simulate: 
- Ref: JFE, 1976.; Bates, *Jumps and Stochastic Volatility: Exchange Rate Processes*, RFS, 1996.; Econometrica, 1989.

---

## v0.217.0 — 2026-04-17

FX Correlation & Baskets — Phase FX7 complete.

`fx_correlation.py`:
- TriangularResult (class): Triangular FX correlation consistency check.
- triangular_correlation: Triangular FX vol consistency.
- implied_correlation_from_triangular: Invert triangular relation for correlation.
- BasketResult (class): FX basket option result.
- fx_basket_option: Multi-asset FX basket option.
- fx_worst_of: Worst-of FX option: payoff on min(S₁, S₂, ...).
- fx_best_of: Best-of FX option: payoff on max(S₁, S₂, ...).
- MargrabeResult (class): Margrabe exchange option result.
- margrabe_fx_exchange: Margrabe formula for exchange option on two FX assets.
- ImpliedCorrelationResult (class): Implied correlation from market prices.
- implied_correlation_quanto: Invert quanto adjustment for implied correlation.
- Ref: Wystup, *FX Options and Structured Products*, 2nd ed., Wiley, 2017, Ch. 6.; Margrabe, *The Value of an Option to Exchange One Asset for Another*, JF, 1978.

---

## v0.216.0 — 2026-04-17

FX Greeks Deepening — Phase FX6 complete.

`fx_greeks.py`:
- fx_vega: Vega = ∂V/∂σ. Same for call and put.
- fx_vanna: Vanna = ∂²V/∂S∂σ = ∂Δ/∂σ.
- fx_volga: Volga (vomma) = ∂²V/∂σ² = ∂vega/∂σ.
- fx_charm: Charm = ∂Δ/∂t (time derivative of delta).
- fx_dvega_dspot: DvegaDspot = ∂vega/∂S via finite difference.
- fx_dvega_dvol: DvegaDvol = volga.
- VegaBucket (class): Vega in one (tenor, delta) bucket.
- VegaLadder (class): FX vega ladder per (expiry, delta) bucket.
- fx_vega_ladder: Compute vega ladder across (tenor, delta) buckets.
- SmileGreeksResult (class): Smile-consistent Greeks via VV.
- fx_smile_consistent_greeks: Smile-consistent Greeks via Vanna-Volga.
- vv_price: 
- Ref: Wystup, *FX Options and Structured Products*, 2nd ed., Wiley, 2017, Ch. 2.; Clark, *FX Option Pricing*, Wiley, 2011.

---

## v0.215.0 — 2026-04-17

FX Structured Products — Phase FX5 complete.

`fx_structured.py`:
- TARFResult (class): Target Redemption Forward pricing result.
- fx_tarf_price: FX TARF: accumulate profit until target hit, then terminate.
- AutocallableResult (class): FX autocallable result.
- fx_autocallable_price: FX autocallable: autocalls at barrier, with optional memory coupon.
- DCDResult (class): Dual-Currency Deposit result.
- fx_dual_currency_deposit: Dual-Currency Deposit: higher yield with FX conversion risk.
- PivotResult (class): Pivot/digital range option result.
- fx_pivot_option: Pivot / digital range option.
- Ref: Wystup, *FX Options and Structured Products*, 2nd ed., Wiley, 2017, Ch. 5.; Clark, *FX Option Pricing*, Wiley, 2011.

---

## v0.214.0 — 2026-04-17

FX Smile Cube — Phase FX4 complete.

`fx_smile_cube.py`:
- FXSmileNode (class): Calibrated SABR smile at one FX tenor.
- calibrate_sabr_fx_tenor: Calibrate SABR (α, ρ, ν) at one FX tenor to market quotes.
- FXVolCube (class): FX vol cube with SABR smile at each tenor.
- build_fx_vol_cube: Build FX vol cube from market quotes.
- SVIParams (class): SVI (Stochastic Vol Inspired) raw parameters.
- svi_fit: Fit SVI raw parameters to observed smile.
- svi_vol: Implied vol from SVI at log-moneyness k.
- ArbitrageCheckResult (class): Arbitrage check result.
- check_butterfly_arbitrage: Check butterfly (no-arb) condition: ∂²C/∂K² ≥ 0 for all K.
- check_calendar_arbitrage: Check calendar arbitrage: total variance T × σ²(T, K) is non-decreasing in T.
- objective: 
- vol: Implied vol at (T, K) via time-interpolated SABR parameters.
- vol_at_delta: Implied vol at (T, δ).
- objective: 
- Ref: Clark, *FX Option Pricing*, Wiley, 2011, Ch. 3-4.; Gatheral, *The Volatility Surface*, Wiley, 2006.; Gatheral & Jacquier, *Arbitrage-Free SVI Volatility Surfaces*, QF, 2014.

---

## v0.213.0 — 2026-04-17

Vanna-Volga Deepening — Phase FX3 complete.

`vanna_volga.py`:
- VVWeights (class): Vanna-Volga hedge weights.
- vv_weights: Solve 3×3 system for smile-implied VV hedge weights.
- VVResult (class): Vanna-Volga adjusted price result.
- vv_adjust_vanilla: Vanna-Volga adjusted vanilla price (market-consistent smile).
- vv_adjust_digital: VV-adjusted cash-or-nothing digital price.
- vv_adjust_touch: VV-adjusted one-touch price.
- vv_adjust_quanto: VV-adjusted quanto FX option.
- greeks: 
- Ref: Wystup, *FX Options and Structured Products*, 2nd ed., Wiley, 2017.; Clark, *FX Option Pricing*, Wiley, 2011, Ch. 5.

---

## v0.212.0 — 2026-04-17

FX SLV Calibration — Phase FX2 complete.

`fx_slv_calibration.py`:
- LeverageFunction (class): Calibrated leverage function L(S, t).
- calibrate_leverage_function: Calibrate leverage function L(S, t) via forward-Kolmogorov density.
- ParticleCalibrationResult (class): Particle method calibration result.
- particle_slv_calibration: Guyon-Henry-Labordère particle method for SLV calibration.
- MixingResult (class): SLV mixing fraction calibration result.
- slv_mixing_calibration: Calibrate the mixing fraction η by bisection on an exotic price.
- SLVBarrierResult (class): Barrier option price under calibrated SLV.
- slv_barrier_price: Price an FX barrier option under calibrated SLV via MC.
- Ref: Models*, Risk, 2007.

---

## v0.211.0 — 2026-04-17

FX Exotic Options — FX Deepening begins — Phase FX1 complete.

`fx_exotic.py`:
- TouchResult (class): One-touch / no-touch option result.
- fx_one_touch: One-touch: pays `payout` at expiry if spot touches barrier before T.
- fx_no_touch: No-touch: pays `payout` if spot never touches barrier.
- fx_double_no_touch: Double no-touch: pays if spot stays in [barrier_low, barrier_high].
- fx_double_touch: Double touch: pays if spot touches either barrier.
- LookbackResult (class): Lookback option result.
- fx_lookback_floating: Floating-strike lookback: payoff = S_T − m (call) or M − S_T (put).
- fx_lookback_fixed: Fixed-strike lookback: max(max(S) − K, 0) call or max(K − min(S), 0) put.
- AsianResult (class): Asian option result.
- fx_asian_geometric: Geometric Asian FX option — closed form.
- fx_asian_arithmetic: Arithmetic Asian FX — MC with geometric control variate.
- RangeAccrualResult (class): Range accrual result.
- fx_range_accrual: FX range accrual: pays coupon × fraction_of_days_in_range.
- AccumulatorResult (class): KODA (knock-out discount accumulator) result.
- fx_accumulator: FX accumulator (KODA): buy FX daily at discount, knock out on barrier.
- Ref: * :func:`fx_lookback_floating` / :func:`fx_lookback_fixed` — Goldman-Sosin-Gatto.; Wystup, *FX Options and Structured Products*, Wiley, 2nd ed., 2017.; Clark, *FX Option Pricing*, Wiley, 2011.

---

## v0.210.0 — 2026-04-16

Advanced Inflation — IR DEEPENING COMPLETE — Phase IR8 complete.

`inflation_advanced.py`:
- YoYConvexityResult (class): Year-on-year convexity adjustment result.
- yoy_convexity_adjustment: Convexity adjustment from ZC inflation rate to YoY forward rate.
- LPIResult (class): Limited Price Index swap pricing result.
- lpi_swap_price: Price a Limited Price Index (LPI) swap.
- InflationSwaptionResult (class): Inflation swaption pricing result.
- inflation_swaption_price: Price a swaption on a breakeven inflation swap.
- RealRateSwaptionResult (class): Real rate swaption pricing result.
- real_rate_swaption_price: Price a swaption on a real rate swap.
- Ref: Mercurio, *Pricing Inflation-Indexed Derivatives*, QF, 2005.; Kerkhof, *Inflation Derivatives Explained*, Lehman Brothers, 2005.

---

## v0.209.0 — 2026-04-16

IR Vol Surface — Phase IR7 complete.

`ir_vol_surface.py`:
- SABRSmileNode (class): SABR parameters at one (expiry, tenor) point.
- SmileDynamicsResult (class): Backbone / smile dynamics analysis.
- calibrate_sabr_smile: Calibrate SABR (α, ρ, ν) to market smile at one node.
- VolCubeNode (class): Internal node in the vol cube.
- SwaptionVolCube (class): Swaption vol cube: vol(expiry, tenor, strike) via SABR smiles.
- build_vol_cube: Build full vol cube by calibrating SABR at each node.
- smile_dynamics: Analyse smile dynamics: sticky strike vs sticky delta.
- objective: 
- vol: Implied vol at (expiry, tenor, strike).
- atm_vol: ATM implied vol at (expiry, tenor).
- smile: Return (strikes, vols) for the smile at (expiry, tenor).
- get_node: 
- interp: 
- Ref: Rebonato, *Volatility and Correlation*, Wiley, Ch. 16-17.

---

## v0.208.0 — 2026-04-16

Advanced Curves — Phase IR6 complete.

`curve_advanced.py`:
- NSFitResult (class): Nelson-Siegel fit result.
- nelson_siegel_fit: Fit Nelson-Siegel model to observed yield curve.
- ns_yield_curve: Evaluate Nelson-Siegel yield at given maturities.
- SvenssonFitResult (class): Svensson fit result.
- svensson_fit: Fit Svensson (extended Nelson-Siegel) to yield curve.
- svensson_yield_curve: Evaluate Svensson yield at given maturities.
- SmoothForwardResult (class): Smooth forward curve result.
- smooth_forward_curve: Monotone-preserving forward rate interpolation.
- TOYResult (class): Turn-of-year adjustment result.
- turn_of_year_adjustment: Apply turn-of-year funding premium to yield curve.
- objective: 
- objective: 
- Ref: Hagan & West, *Interpolation Methods for Curve Construction*, AMF, 2006.

---

## v0.207.0 — 2026-04-16

Convexity Adjustments — Phase IR5 complete.

`convexity.py`:
- CMSConvexityResult (class): CMS convexity adjustment result.
- cms_convexity_adjustment: CMS convexity adjustment (Hagan linear swap rate model).
- cms_rate_replication: CMS rate via static replication with payer/receiver swaptions.
- ArrearsResult (class): LIBOR-in-arrears adjustment result.
- arrears_adjustment: LIBOR-in-arrears convexity adjustment.
- TimingResult (class): Payment timing adjustment result.
- timing_adjustment: Timing/payment delay adjustment.
- QuantoIRResult (class): Cross-currency rate adjustment result.
- quanto_ir_adjustment: Quanto adjustment for IR rate paid in different currency.
- annuity_func: PV of unit annuity at rate K.
- annuity_deriv2: Second derivative of annuity w.r.t. K (numerical).
- Ref: Pelsser, *Mathematical Foundation of Convexity Correction*, QF, 2003.

---

## v0.206.0 — 2026-04-16

Bermudan under LMM — Phase IR4 complete.

`bermudan_lmm.py`:
- BermudanLMMResult (class): Bermudan swaption pricing result under LMM.
- ExerciseBoundary (class): Exercise boundary: swap rate threshold at each exercise date.
- BermudanBoundsResult (class): Lower and upper bounds for Bermudan swaption.
- bermudan_swaption_lmm: Bermudan swaption priced via LSM under LMM dynamics.
- bermudan_exercise_boundary: Extract exercise boundary from LSM regression.
- bermudan_upper_bound: Andersen-Broadie dual upper bound for Bermudan swaption.
- Ref: Longstaff & Schwartz, *Valuing American Options by Simulation*, RFS, 2001.; Andersen & Broadie, *Primal-Dual Simulation for Upper Bounds*, Ops. Res., 2004.

---

## v0.205.0 — 2026-04-16

LMM Deepening — Phase IR3 complete.

`lmm_advanced.py`:
- LMMCalibrationResult (class): Result of LMM calibration to swaption matrix.
- lmm_cascade_calibration: Cascade (column-by-column) calibration of LMM to swaption matrix.
- lmm_global_calibration: Global calibration: fit all instantaneous vols simultaneously.
- SABRLMMResult (class): SABR-LMM simulation result.
- SABRLMM (class): SABR-LMM: stochastic vol on each forward rate.
- PredictorCorrectorResult (class): LMM with predictor-corrector drift.
- lmm_predictor_corrector: LMM simulation with predictor-corrector drift.
- LMMGreeksResult (class): LMM pathwise sensitivities.
- lmm_pathwise_greeks: Pathwise (IPA) Greeks for a caplet under LMM.
- objective: 
- simulate: Simulate SABR-LMM forward rate + vol paths.
- drift: LMM drift for forward j under terminal measure.
- objective: 
- Ref: Rebonato, *Modern Pricing of Interest-Rate Derivatives*, Princeton, 2002.; Rebonato, McKay & White, *The SABR/LIBOR Market Model*, Wiley, 2009.

---

## v0.204.0 — 2026-04-15

Short Rate Model Deepening — Phase IR2 complete.

Slices 237-239 merged (share short_rate_models.py):
- BKRateModel: d(ln r) = (θ(t)−a ln r)dt + σdW, always positive, MC + ZCB
- CIRPPRateModel: r(t) = x(t) + φ(t), analytical ZCB matches MC to 15%
- CheyetteModel: Markovian HJM (x, y), r = f(0,t) + x, analytical ZCB
- AffineModel: Dai-Singleton A_m(n), unified Riccati ODE for ZCB
  - A_0(1) = Vasicek, A_1(1) = CIR, A_0(2) = G2++ verified
- Ref: BK 1991, Brigo-Mercurio Ch.3-4, Cheyette 1992, Dai-Singleton 2000

---

## v0.203.0 — 2026-04-15

Exotic IR Products — Phase IR1 complete. IR Deepening begins.

Slices 234-236 merged (share ir_exotic.py):
- tarn_price: Target Redemption Note (early redemption when cumulative coupon hits target)
- snowball_price: coupon_n = max(coupon_{n-1} + spread − r_n, floor)
- callable_range_accrual: range accrual + issuer call provision
- ratchet_cap: strike resets to min(previous fixing, strike)
- flexi_swap: holder chooses which periods to exercise (max_exercises cap)
- TARN early redemption verified; snowball accumulation verified
- Ref: Brigo-Mercurio Ch.15-16, Andersen-Piterbarg Vol.III Ch.19

---

## v0.202.0 — 2026-04-15

Leveraged Credit Structures — Phase C9 complete. ALL CREDIT DEEPENING DONE.

Slices 231-233 merged (share credit_leveraged.py):
- leveraged_cds: first-loss on single name, spread scales with leverage
- digital_cln_leveraged: binary payout CLN with leverage
- constant_maturity_cds: CMCDS with convexity adjustment (fair > forward)
- cds_straddle: payer + receiver swaption, breakeven move
- credit_trs: total return swap on credit index (carry + price change − funding)
- Ref: O'Kane Ch.13-16, Schönbucher Ch.10-11

---

## v0.201.0 — 2026-04-15

Advanced Recovery — Phase C8 complete.

Slices 229-230 merged (share recovery_advanced.py):
- seniority_waterfall: senior → sub → equity priority allocation
- waterfall_recovery_rates: recovery rate per seniority class
- lgd_cycle: pro-cyclical LGD (recovery drops when defaults spike)
- stochastic_recovery_cds: CDS spread with recovery-intensity correlation
- wrong_way_recovery_cva: CVA with recovery falling in stress
- Senior recovery > sub > equity verified across all recovery levels
- Downturn LGD > normal LGD verified; wrong-way premium > 0
- Ref: Altman 2006, Schönbucher Ch.6, Andersen-Sidenius 2004

---

## v0.200.0 — 2026-04-15

Copulas — Phase C7 complete. v0.200.0 milestone.

Slices 226-228 merged (share copulas.py):
- Copula ABC with sample() and default_indicators()
- GaussianCopula: one-factor equi-correlation (baseline)
- StudentTCopula: tail dependence via ν, converges to Gaussian at ν→∞
- ClaytonCopula: lower tail dependence (Marshall-Olkin sampling)
- FrankCopula: symmetric, no tail dependence (conditional method)
- GumbelCopula: upper tail dependence (stable frailty)
- copula_default_simulation: portfolio defaults under any copula
- tranche_pricing_copula: CDO tranche EL and spread under any copula
- Ref: Li 2000, McNeil-Frey-Embrechts 2005, Cherubini-Luciano-Vecchiato 2004

---

## v0.199.0 — 2026-04-15

Credit Hybrids — Phase C6 complete.

Slices 223-225 merged (share credit_hybrid.py):
- callable_risky_bond: binomial rate tree + survival overlay, call decision, OAS
- floating_cln: floating coupon CLN, deterministic + MC with stochastic hazard
- convertible_bond: three-factor MC (equity + credit + rates), bond floor + conversion
- Callable ≤ non-callable verified; convertible ≥ max(floor, conversion) verified
- Higher equity vol → higher convertible price verified
- Ref: Schönbucher Ch.9, Tsiveriotis-Fernandes 1998, Brigo-Mercurio Ch.22

---

## v0.198.0 — 2026-04-14

CDS Swaptions — Phase C5 complete.

Slices 221-222 merged (share cds_swaption.py):
- forward_cds_spread: par spread of forward-starting CDS, F ≈ λ(1−R) verified
- cds_swaption_black: Black-76 on forward spread, survival knockout factor
- PedersenCDSSwaption: analytical + MC (MC matches Black to 5%)
- Payer/receiver, ATM symmetry, higher vol → higher premium
- cds_swaption_put_call_parity: payer − receiver = Q×A×(F−K), verified at all strikes
- Ref: Pedersen 2003, Schönbucher Ch.11, O'Kane Ch.15

---

## v0.197.0 — 2026-04-14

Stochastic Hazard Rate Models — Phase C4 complete.

Slices 218-220 merged (share hazard_rate_models.py):
- HWHazardRate: dλ = (θ(t)−aλ)dt + σdW, analytical survival, MC simulation
- BKHazardRate: d(ln λ) = (θ(t)−a ln λ)dt + σdW, always positive, MC + trinomial tree
- CIRPlusPlus: λ(t) = x(t) + φ(t), deterministic shift for exact calibration
- TwoFactorIntensity: λ = x₁ + x₂ + φ, level + slope with correlation
- BK positivity verified; CIR++ shift calibrates to market; two-factor richer dynamics
- Ref: Hull-White 1990, Black-Karasinski 1991, Brigo-Mercurio Ch.3-4, Schönbucher Ch.7-8

---

## v0.196.0 — 2026-04-14

ECL Provisioning — Phase C3 complete.

- stage_classification: IFRS 9 three-stage (SICR via relative PD, absolute PD, DPD)
- ecl_12_month: stage 1 ECL = PD × LGD × EAD × df
- ecl_lifetime: stage 2/3 ECL = Σ marginal_PD(t) × LGD × EAD × df(t)
- marginal_pds_from_cumulative / cumulative_pds_from_hazard: PD conversion
- ecl_portfolio: probability-weighted macro scenarios (base/stress/severe)
- Lifetime ECL ≥ 12-month ECL verified; stress ECL > base ECL
- Ref: IFRS 9 §5.5, CECL ASC 326, EBA 2017

---

## v0.195.0 — 2026-04-14

Exotic Credit Payoffs — Phase C2 complete.

Slices 215-216 merged (share credit_exotic.py):
- capped_coupon_bond: min(floating + spread, cap) with default risk + recovery
- digital_cds: fixed payout on default, par spread, digital ≈ standard when payout = (1−R)
- credit_range_accrual: coupon accrues when spread ∈ [L, U] (normal approximation)
- credit_linked_loan: margin grid by leverage ratio, covenant triggers, expected loss
- Ref: Schönbucher 2003 Ch.10, O'Kane 2008 Ch.12

---

## v0.194.0 — 2026-04-14

Rating Models — Credit Deepening Phase C1 begins.

- calibrate_generator: fit Q from observed P via matrix log + projection + L-BFGS-B
- Round-trip verified: exp(Q_calibrated × t) ≈ P_observed
- ttc_to_pit / pit_to_ttc: Vasicek/Merton cycle adjustment Φ(Φ⁻¹(PD) ± factor)
- MomentumTransitionMatrix: downgrades beget downgrades (momentum_factor × intensities)
- Momentum PD > memoryless PD verified; factor=1 matches base
- time_varying_generator: economic cycle scaling (stress ↑ downgrades, expansion ↑ upgrades)
- Ref: JLT 1997, Lando 2004 Ch.7, Israel-Rosenthal-Wei 2001

---

## v0.193.0 — 2026-04-14

Advanced Fourier — Phase M13 complete. ALL PLANNED WORK DONE.

Slices 211-212 merged (share fourier_advanced.py):
- cumulants_from_cf: extract mean/variance/skewness/kurtosis from characteristic function
- edgeworth_expansion: density approximation from cumulants (Gram-Charlier)
- hilbert_implied_vol_slope: Lee (2004) wing slope from critical moment
- fft_2d_basket: two-asset basket via moment-matched vol + BS
- mellin_power_option: power option S^p via adjusted BS formula
- BS cumulants verified: mean = (r−0.5σ²)T, variance = σ²T
- Edgeworth integrates to 1, skew shifts mass correctly
- Ref: Lee 2004, Hurd & Zhou 2010, Panini & Srivastav 2004

---

## v0.192.0 — 2026-04-14

Rough Paths — Phase M12 complete.

Slices 208-210 merged (share rough_paths.py):
- fbm_circulant: O(N log N) fBM simulation via circulant embedding (vs O(N³) Cholesky)
- path_signature: truncated iterated integrals up to level 3 (Chen identity)
- log_signature: compact representation via S₂ − 0.5 S₁⊗S₁
- rough_heston_cf: fractional Riccati ODE via forward Euler, CF at u=0 = 1 verified
- fBM H=0.5 reproduces BM statistics, lower H → rougher paths verified
- Ref: Lyons 1998, Chevyrev & Kormilitzin 2016, El Euch & Rosenbaum 2019

---

## v0.191.0 — 2026-04-13

Optimal Transport — Phase M11 complete.

Slices 205-207 merged (share optimal_transport.py):
- wasserstein_1d: closed-form W_p via quantile functions
- wasserstein_gaussian: analytical W_2 between Gaussians
- wasserstein_discrete: LP-based OT between discrete distributions
- sinkhorn: entropic regularisation, converges to true OT as ε → 0
- martingale_ot_bounds: model-free exotic option bounds from vanilla prices
- Sinkhorn plan marginals verified, approaches true OT cost
- Ref: Villani 2009, Cuturi 2013, Beiglböck et al. 2013

---

## v0.190.0 — 2026-04-13

Distribution Theory + Feynman-Kac — Phase M10 complete.

Slices 202-204 merged (share distribution_theory.py):
- Distribution class: distributions as functionals on test functions
- dirac_delta / heaviside_dist / regular_distribution
- Distributional derivative: ⟨T', φ⟩ = −⟨T, φ'⟩ verified (⟨δ', x⟩ = −1)
- sobolev_norm: H⁰/H¹/H^s norms via FFT
- greens_function_heat: heat kernel (integrates to 1)
- greens_function_bs: BS transition density (lognormal, integrates to 1)
- feynman_kac_pde: SDE → PDE coefficient extraction
- feynman_kac_verify: PDE vs MC consistency check (FD matches BS verified)
- Ref: Schwartz 1966, Evans 2010, Shreve 2004

---

## v0.189.0 — 2026-04-13

Advanced Optimisation — Phase M9 complete.

Slices 199-201 merged (share optimisation_advanced.py):
- quadratic_program: KKT system for equality-constrained QP
- markowitz_portfolio: mean-variance with weights-sum-to-1, optional target return
- constrained_minimize: augmented Lagrangian with equality + inequality constraints
- admm_lasso: ADMM for L1-regularised least squares (sparse signal recovery)
- cma_es: Covariance Matrix Adaptation ES (derivative-free, sphere + Rosenbrock verified)
- Ref: Boyd & Vandenberghe 2004, Hansen 2016

---

## v0.188.0 — 2026-04-13

Approximation Theory — Phase M8 complete.

Slices 197-198 merged (share approximation.py):
- chebyshev_interpolate: near-minimax on [a,b] via Lobatto points + DCT
- Clenshaw evaluation, convergence diagnostic (tail coefficient magnitude)
- exp(x) on [-1,1] with n=20 accurate to 1e-10 verified
- pade_approximant: [L/M] rational from Taylor coefficients
- Padé [1/1] exactly reproduces 1/(1+x) verified
- richardson_table: full extrapolation table, diagonal estimates
- bspline_basis: Cox-de Boor recursion, partition of unity verified
- Ref: Trefethen ATAP 2013, Baker & Graves-Morris 1996

---

## v0.187.0 — 2026-04-13

Advanced PDE Methods — Phase M7 complete.

Slices 194-196 merged (share pde_advanced.py):
- psor_american: Projected SOR for linear complementarity (American free boundary)
- Exercise boundary extraction for American put
- chebyshev_bs: Chebyshev collocation for BS PDE (exponential convergence, N=32 matches BS)
- method_of_lines: FD in space + RK4 in time (matches BS to 2%)
- richardson_extrapolation: combine O(h^p) + O((h/2)^p) → O(h^{p+1})
- Ref: Wilmott et al. 1995, Trefethen 2000, Duffy 2006

---

## v0.186.0 — 2026-04-13

MC Greeks + Optimal MLMC + LSM Improvements — Phase M6 complete.

Slices 191-193 merged (share mc_greeks.py):
- pathwise_delta / pathwise_vega: IPA for smooth payoffs (matches BS to 5%)
- likelihood_ratio_delta / likelihood_ratio_vega: LR for discontinuous payoffs
- LR has higher variance than IPA (verified)
- optimal_mlmc: Giles 2008 allocation N_l ∝ √(V_l/C_l), automatic level selection
- lsm_with_basis: Laguerre, Chebyshev, polynomial bases for American LSM
- dual_upper_bound: simplified Andersen-Broadie upper bound
- Bracket: LSM ≤ true ≤ dual upper bound verified
- Ref: Glasserman Ch. 7, Giles 2008, Andersen-Broadie 2004

---

## v0.185.0 — 2026-04-13

Extended Stochastic Processes — Phase M5 complete. 100th slice milestone.

Slices 188-190 merged (share processes_extended.py):
- CEV: dS = μS dt + σS^β dW (β=1 → GBM verified)
- 3/2 model: dv = κv(θ−v) dt + εv^{3/2} dW (vol-of-vol explodes with v)
- Kou double-exponential jumps: asymmetric up/down with compensator
- Bates = Heston + Merton jumps (reduces to Heston at λ=0, verified)
- Hawkes self-exciting process: Ogata thinning, intensity spikes after events
- VG full paths: Gamma-subordinated BM (previously terminal-only)
- Ref: Cox 1975, Kou 2002, Bates 1996, Hawkes 1971

---

## v0.184.0 — 2026-04-13

Convergence Testing Framework — Phase M4 (SDE Methods) complete.

- strong_convergence_study: run scheme at multiple dt, compute E[|X−X_ref|]
- weak_convergence_study: run scheme at multiple dt, compute |E[X]−ref|
- scheme_comparison: rank schemes by estimated convergence order
- ConvergenceStudyResult: step_sizes, errors, estimated order, consistency check
- Log-log regression for empirical order estimation
- Ref: Kloeden & Platen Ch. 9-10, Higham 2001

---

## v0.183.0 — 2026-04-13

Exact CIR + Implicit Euler — Phase M4 slice 2.

- exact_cir: non-central chi-squared sampling (zero discretisation bias)
- exact_cir_zcb: analytical CIR ZCB price (CIR 1985, Eq. 23)
- MC ZCB matches analytical to 2% verified
- Exact CIR always non-negative (by construction)
- implicit_euler_step: drift-implicit fixed-point iteration for stiff SDEs
- implicit_euler_paths: full path simulation
- Stable for κΔt < 1 (fixed-point convergence condition documented)
- Ref: Broadie & Kaya 2006, Cox-Ingersoll-Ross 1985

---

## v0.182.0 — 2026-04-13

Milstein Scheme — Phase M4 (SDE Methods) begins.

- milstein_step: generic one-step with Itô-Taylor correction 0.5σσ'(ΔW²−Δt)
- milstein_paths: full path simulation for any SDE
- milstein_gbm: GBM specialisation (strong order 1.0, passes martingale test)
- milstein_cev: CEV model (σ'=βσS^{β-1}), reduces to GBM at β=1
- milstein_cir: CIR variance (σ'=ξ/(2√v)), absorption at zero
- Empirical strong convergence order ~1.0 verified (vs Euler ~0.5)
- Ref: Kloeden & Platen, Ch. 10

---

## v0.181.0 — 2026-04-13

COS Bermudan — Phase M3 (Transforms) complete.

- cos_bermudan: backward COS iteration with physical-space early exercise
- cos_american: Bermudan approximation with many exercise dates
- Grid-based continuation value via COS coefficient → evaluate → max with payoff
- Bermudan put ≥ European put verified (early exercise premium)
- Ref: Fang & Oosterlee, Numer. Math. 114, 2009

---

## v0.180.0 — 2026-04-12

Laplace Inversion: Talbot, Euler acceleration, Gaver-Stehfest.

- talbot_inversion: deformed Bromwich contour, exponential convergence (Weideman 2006)
- euler_inversion: Abate-Whitt method with Richardson acceleration
- gaver_stehfest: real-valued inversion (no complex arithmetic needed)
- invert: unified interface selecting method by name
- All three methods agree on exp(-at) to within 5%
- Talbot most accurate (rel error < 1e-6 for smooth transforms)
- Ref: Talbot 1979, Abate & Whitt 1995, Stehfest 1970

---

## v0.179.0 — 2026-04-12

FFT Pricing: Carr-Madan, density recovery — Phase M3 begins.

- carr_madan_fft: O(N log N) call prices at N strikes in one FFT pass
- Simpson weights for numerical integration, damping parameter α
- FFT prices match Black-Scholes within 1-2% across strike range
- FFT matches COS method verified
- lewis_price: contour integral (positive price, CF convention fix pending)
- density_from_calls: Breeden-Litzenberger d²C/dK² = RN density
- density_from_cf: inverse Fourier of characteristic function
- Density non-negativity and integration-to-1 verified
- Ref: Carr & Madan, J. Comp. Finance, 1999

---

## v0.178.0 — 2026-04-12

Numerical Safety — Phase M2 complete.

Slices 180-181 merged (share numerical_safety.py):
- check_cfl: CFL condition for explicit FD (dt ≤ dx²/(σ² + |μ|dx))
- check_feller: Feller condition for CIR/Heston (2κθ ≥ ξ²)
- martingale_test: E[e^{-rT}S_T] = S_0 — catches drift/measure errors in MC
- convergence_rate: empirical order from log-log regression at multiple resolutions
- strong_convergence_test: E[|X_fine - X_coarse|] for SDE schemes
- weak_convergence_test: |E[f(X_T)] - reference| for terminal distributions
- GBM passes martingale test; wrong drift fails (verified)
- Ref: Lax-Richtmyer 1956, Feller 1951, Glasserman Ch. 6

---

## v0.177.0 — 2026-04-12

ND Root Finding — Phase M1 (Math Foundations I) complete.

- newton_nd: multi-dimensional Newton-Raphson with analytical or FD Jacobian
- broyden: quasi-Newton with rank-1 Jacobian updates (avoids full recomputation)
- damped_newton: Armijo backtracking line search for robustness with bad initial guesses
- finite_difference_jacobian: central differences, O(ε²) accuracy
- NDSolverResult: x, residual, iterations, converged, residual_norm
- Ref: Nocedal & Wright, Numerical Optimization, Ch. 11

---

## v0.176.0 — 2026-04-12

Correlation Matrix Repair: Higham nearest PD, eigenvalue floor, interpolation.

- nearest_correlation_matrix: Higham (2002) alternating projections with Dykstra correction
- eigenvalue_floor: simple clip + rescale to unit diagonal
- is_positive_definite / is_valid_correlation: Cholesky-based checks
- correlation_interpolation: linear blend + nearest PD projection
- Higham proven closest in Frobenius norm vs eigenvalue floor
- Ref: Higham, IMA J. Numer. Anal., 2002

---

## v0.175.0 — 2026-04-12

Linear Algebra: PCA, eigendecomposition, SVD, condition numbers — Phase M1 begins.

- pca: eigendecomposition of covariance, sorted components, explained variance ratios
- PCAResult: project/reconstruct, cumulative variance, orthonormal components
- eigendecomposition: sorted by |λ|, positive-definiteness check
- svd_decomposition: thin SVD with numerical rank computation
- condition_number: κ₂ = σ_max/σ_min, well/ill-conditioned diagnosis with recommendation
- explained_variance: minimum components for target variance threshold
- Ref: Litterman & Scheinkman 1991 (yield curve PCA)

---

## v0.174.0 — 2026-04-12

Multi-Asset Hedging — Pillar 10 (Options Desk) complete. All 10 pillars done.

- optimal_hedge: N instruments × M targets via least-squares (+ Tikhonov cost penalty)
- HedgeTarget / HedgeInstrument / HedgeAllocation / HedgeResult
- hedge_residual: compute residual risk after applying allocations
- what_if_analysis: impact of adding/removing a single hedge instrument
- hedge_recommendation: optimal hedge + risk reduction %, largest residual Greek
- Hedged book verified near-zero for all targeted Greeks

---

## v0.173.0 — 2026-04-12

Vol Surface Arbitrage: calendar/butterfly detection, arb-free enforcement.

- detect_calendar_arb: total variance must be non-decreasing in time
- detect_butterfly_arb: call prices must be convex in strike
- check_surface_arbitrage: combined report with is_arb_free flag
- enforce_no_calendar_arb: floor each σ²T at previous level
- enforce_no_butterfly_arb: cap mid-price at linear interpolation of neighbours

---

## v0.172.0 — 2026-04-12

Exotic Options Book: barrier/digital/asian/autocall position management.

- ExoticBook: aggregate by exotic type and by underlying
- ExoticEntry: trade with Greeks + model tag (black_scholes, local_vol, slv)
- model_risk_comparison: max delta/vega diff across models for a position
- hedge_exotic_book: flatten delta + vega using vanilla instruments

---

## v0.171.0 — 2026-04-12

Vol Correlation: cross-asset matrix, monitor, correlation trade.

- vol_correlation_matrix: pairwise correlation from vol time series (symmetric, diagonal=1)
- is_valid_correlation_matrix: symmetry + diagonal check
- correlation_monitor: z-score pairwise correlation vs history (high/low/fair)
- build_correlation_trade: vega-neutral long/short vol across asset classes
- correlation_sensitivity: ∂PV/∂ρ proxy from cross-vega terms

---

## v0.170.0 — 2026-04-12

Cross-Asset Greeks: unified attribution, carry vs convexity, multi-factor stress.

- greek_attribution: delta/gamma/vega/theta/rho P&L by asset class
- BookGreekAttribution: carry (theta) and convexity (gamma) properties
- StressScenario: named multi-factor scenario (spot + vol + rate shocks)
- multi_factor_stress: apply scenario to options book, get per-Greek P&L
- Attribution sums to total across all asset classes verified

---

## v0.169.0 — 2026-04-11

Cross-Asset Options Book — Pillar 10 (Options Desk) begins.

- OptionsBook: unified container for option positions across all asset classes
- OptionEntry: trade_id, asset_class, underlying, expiry, Greeks (delta/gamma/vega/theta/rho)
- by_asset_class: aggregate vega/gamma/theta/delta per asset class
- by_expiry: aggregate vega by expiry bucket
- vol_pnl_attribution: daily vega/gamma/theta P&L by asset class
- Total vega = sum of per-asset vegas verified

---

## v0.168.0 — 2026-04-11

Inflation Trading — Pillar 9 (Inflation Desk) complete.

Slices 167-170 merged (share inflation_trading.py):
- Breakeven monitor + DV01-neutral breakeven trade construction
- Carry: real yield carry, breakeven roll-down, seasonal CPI patterns
- RV: breakeven vs CPI swap basis, cross-market linker spreads
- Risk decomposition: IE01 + real DV01 = nominal DV01
- Regulatory: GIRR inflation risk weight (1.6%), one-call capital report

---

## v0.167.0 — 2026-04-11

Inflation Book — Pillar 9 (Inflation Desk) begins.

- InflationBook: positions by issuer/product type, IE01/real DV01 aggregation
- InflationTradeEntry: linker/zc_swap/yoy_swap/cap/floor, breakeven, IE01, real DV01
- InflationLimits: per-issuer notional, max IE01, max real DV01
- check_limits with breach detection

---

## v0.166.0 — 2026-04-11

FX FRTB SA Capital — Pillar 8 (FX Desk) complete.

- FXRiskInputs: per-pair delta sensitivity + notional
- is_liquid_pair: EUR/USD, USD/JPY, GBP/USD, etc. (11.25% RW vs 15%)
- fx_to_frtb_positions: wire into FRTB SA FX risk class
- fx_frtb_capital: one-call FX capital report

---

## v0.165.0 — 2026-04-11

FX Hedging: delta/cross-hedge, triangular arb, NDF settlement.

- fx_delta_hedge: spot/forward hedge quantity
- fx_cross_hedge: proxy hedge with minimum-variance ratio (ρ × σ_target/σ_proxy)
- triangular_arb_monitor: synthetic cross vs direct rate, arb detection
- ndf_settlement: cash settlement from FX fixing, T+2 calendar-aware

---

## v0.164.0 — 2026-04-11

FX Vol Desk: straddle/strangle/RR/butterfly, vol RV, skew, vega ladder.

- FXStraddle / FXStrangle / FXRiskReversal / FXButterfly structures
- fx_vol_rv: implied vs realised vol z-score
- fx_skew_monitor: 25-delta RR level vs history
- fx_vega_ladder / total_fx_vega: aggregate by (pair, expiry)

---

## v0.163.0 — 2026-04-11

FX Basis Strategies: monitor, term structure, RV, carry, curve trades.

- basis_monitor: z-score cross-currency basis vs history
- basis_term_structure: 1Y/3Y/5Y/10Y basis pillars
- cross_market_basis_rv: rank pairs by basis z-score
- basis_carry: holding period return from basis position
- BasisCurveTrade: steepener/flattener on the basis curve

---

## v0.162.0 — 2026-04-11

FX Carry Strategies: G10 ranking, NDF carry, carry-vol ratio.

- carry_signal: annualised carry from forward-spot differential
- carry_adjusted_forward: break-even spot move
- g10_carry_ranking: rank pairs by carry (most negative = best carry)
- ndf_carry: NDF carry from domestic/foreign rate differential
- carry_volatility_ratio: carry / vol (Sharpe-like)

---

## v0.161.0 — 2026-04-11

FX Daily P&L: spot/carry/basis decomposition + per-pair/per-ccy attribution.

- compute_fx_daily_pnl: spot (Δrate × position) + carry (fwd points) + basis + new + amendments
- attribute_fx_pnl: per-pair and per-currency breakdown
- FXPairAttribution: spot/carry/basis/total per pair
- FXCurrencyAttribution: P&L allocated to each currency
- Both dimensions sum to total verified

---

## v0.160.0 — 2026-04-11

FX Book — Pillar 8 (FX Desk) begins.

- FXBook: positions by currency pair, long/short netting, trade count
- FXTradeEntry: pair (BASE/QUOTE), notional, spot_rate, forward_points, reporting_rate
- FXPairPosition: net/long/short notional, PV in reporting currency
- CurrencyExposure: net per-currency exposure (long base = +base, −quote)
- FXLimits: per-pair notional, per-currency exposure, gross notional caps
- check_limits with breach detection

---

## v0.159.0 — 2026-04-11

Bond FRTB SA Capital — Pillar 7 (Bond Desk) complete.

- BondRiskInputs: per-bond IR/CS sensitivity, notional, rating, seniority
- bond_to_frtb_positions: wire into GIRR (by currency) + CSR (by sector) + DRC
- bond_frtb_capital: one-call GIRR + CSR + DRC capital report
- BondCapitalReport: sbm_capital property, capital efficiency, total RWA
- Manual verification: GIRR 1M × 11% = 110K; DRC senior BBB 10M × 0.75 × 2% = 150K
- Cross-check against calculate_frtb_sa output

---

## v0.158.0 — 2026-04-11

Bond Rich/Cheap: fitted curve RV, cross-market, spread strategies.

- fitted_curve_rv: rich/cheap vs model curve, z-score + percentile
- cross_market_rv: UST vs Bunds/Gilts with FX hedge cost adjustment
- asw_spread_monitor / zspread_monitor: z-score vs history
- build_credit_curve_trade: DV01-neutral 2s10s credit spread curve trade
- crossover_monitor: BBB/BB boundary spread z-score
- new_issue_premium: new issue vs secondary market spread

---

## v0.157.0 — 2026-04-11

Bond Futures Basis Trading: CTD switches, delivery option, convergence.

- ctd_switch_scenarios: which bond is CTD under parallel yield shifts
- ctd_switch_probability: fraction of scenarios where CTD changes
- delivery_option_value: quality option + timing (wild-card) option
- construct_switch_trade: sell old CTD / buy new CTD with P&L
- basis_at_delivery: linear convergence to zero, monotone decreasing

---

## v0.156.0 — 2026-04-11

Duration Management: DV01 ladder, curve risk, KRD hedge, barbell vs bullet.

- DV01Ladder: per-tenor key-rate DV01, total_dv01 = sum of rungs
- curve_dv01: parallel, 2s10s steepener, 2s5s10s butterfly decomposition
- duration_target_tracking: book vs mandate, within-band check
- optimal_krd_hedge: least-squares solve to flatten DV01 ladder using N instruments
- hedged_ladder: residual DV01 after applying allocations
- barbell_vs_bullet: convexity comparison, duration-matched recommendation

---

## v0.155.0 — 2026-04-11

Credit Bond Tools: allocation, tracking error, concentration, sector rotation.

- sector_allocation: portfolio weights by sector (financials, industrials, …)
- index_tracking_error: L1 active weights vs benchmark, max overweight/underweight
- concentration_risk: Herfindahl HHI by name, sector, rating bucket
- sector_spread_monitor: z-score sector spread vs history (wide/tight/fair)
- cross_sector_rv: rank sectors by cheapness (highest z-score = widest vs history)
- rating_migration_impact: P&L estimate from one-notch upgrade/downgrade

---

## v0.154.0 — 2026-04-11

Govt Bond Trading: OTR/OFR, WI pricing, auction analytics, basis decomposition.

- otr_ofr_spread: on-the-run vs off-the-run yield spread with z-score (wide/tight/fair)
- when_issued_price: interpolate WI yield from bracketing tenors, rough price estimate
- auction_analytics: tail (bps), bid-to-cover, dealer/indirect/direct allocation %
- AuctionResult.well_received heuristic (BTC > 2.3, tail < 1bp)
- basis_decomposition: gross basis = carry + net basis (optionality), implied repo
- ctd_switch_monitor: rank deliverables by implied repo, identify CTD

---

## v0.153.0 — 2026-04-11

Repo Desk: book, GC/special tracking, financing optimisation, fails.

- RepoBook: positions by counterparty, collateral type (GC/special), term
- RepoTradeEntry: carry (coupon − financing), cash amount, direction-aware
- by_counterparty / by_collateral_type aggregation with weighted-average rates
- gc_rate / special_rate: weighted-average repo rates by collateral type
- repo_rate_monitor: z-score current rate vs history
- cheapest_to_deliver_repo: select bond that minimises financing cost
- term_vs_overnight: compare term lock-in vs rolling overnight
- FailsTracker: track settlement fails, penalty costs, by-counterparty breakdown

---

## v0.152.0 — 2026-04-11

Bond Daily P&L: MTM/accrual + carry/rolldown/curve/spread attribution.

- compute_bond_daily_pnl: MTM (dirty price change × face) + coupon accrual + new trades + amendments
- BondDailyPnL with market_move_pnl property (MTM + accrual)
- attribute_bond_pnl: carry (coupon − financing), roll-down (aging), curve (DV01 × parallel shift), spread (DV01 × spread change), unexplained
- BondTradeAttribution with explained property, per-issuer and per-tenor breakdown
- BondBookAttribution: aggregate across all positions
- Attribution sums to total verified across carry + rolldown + curve + spread + unexplained

---

## v0.151.0 — 2026-04-11

Bond Book — Pillar 7 (Bond Desk) begins.

- BondBook: per-issuer, per-sector (govt/IG/HY/EM), per-tenor aggregation
- BondTradeEntry: issuer / sector / currency / face / dirty price / coupon / maturity / DV01 / duration
- BondPosition: net/long/short face, market value, DV01, weighted duration per issuer
- BondSectorExposure: sector aggregation with issuer count
- BondTenorBucket: net face, market value, DV01 per maturity bucket (≤1Y through 30Y+)
- net_dv01(), net_market_value(), weighted_duration() book-level aggregates
- BondLimits: per-issuer face, per-sector face, total DV01, per-tenor DV01, max duration
- check_limits with breach detection

---

## v0.150.0 — 2026-04-11

Settlement + Calendars — infrastructure phase complete.

- CHFCalendar: Swiss holidays (Berchtoldstag, Ascension, Whit Monday, National Day)
- AUDCalendar: Australian holidays (Australia Day, Anzac, Queen's Birthday, Easter Saturday)
- CADCalendar: Canadian holidays (Family Day, Victoria Day, Canada Day, Thanksgiving)
- add_business_days(start, n, calendar): skip weekends + holidays, supports negative n
- fx_spot_date(trade_date, base, quote, calendar): T+2 default, T+1 for USD/CAD
- bond_settlement_date(trade_date, market, calendar): T+1 US/UK, T+2 EU/JP/AU
- All settlement dates land on business days (verified across full month)

---

## v0.149.0 — 2026-04-11

Multi-currency PricingContext — infrastructure for multi-desk aggregation.

- discount_curves: per-currency discount curves (keyed by ccy code)
- inflation_curves: per-currency CPI/inflation curves
- repo_curves: per-currency repo/funding curves
- reporting_currency: base currency for P&L aggregation (default "USD")
- get_discount_curve(ccy): checks discount_curves first, falls back to discount_curve
- get_inflation_curve(ccy), get_repo_curve(ccy): per-currency accessors
- fx_rate(from, to): direct + inverse lookup, identity for same ccy
- fx_translate(value, from_ccy, to_ccy): convert to reporting currency
- replace() updated to preserve all new fields
- Full backward compatibility: all 3108 existing tests pass unchanged

---

## v0.148.0 — 2026-04-10

Commodity FRTB SA Capital + Forward/Futures Protocol Decision — Pillar 6 (Commodity Desk) complete.

- CommodityRiskInputs / CommodityClassification: per-commodity sensitivities + bucket mapping
- map_to_com_bucket: auto-map 30+ commodity names to FRTB SA COM buckets
- commodity_to_frtb_positions: builds delta/vega/curvature/RRAO dicts for calculate_frtb_sa
- commodity_frtb_capital: one-call SbM + RRAO capital report with efficiency ratio
- Manual hand-calculation tests (single bucket, intra-bucket correlation, cross-bucket)
- Forward/Futures protocol decision: NO protocol — pricing interfaces too divergent across
  IR/FX/equity/commodity/bond futures; a thin Tradeable protocol can be added later if needed

---

## v0.147.0 — 2026-04-10

Commodity Rich/Cheap Monitors + Roll Strategies.

- spread_zscore: z-score calendar spreads vs history
- ratio_monitor: inter-commodity ratios (gold/silver, Brent/WTI) vs history
- seasonality_monitor: current price vs seasonal norm by month
- roll_pnl: P&L from rolling a position (direction-aware)
- roll_cost_or_gain: classify as contango_cost / backwardation_gain / flat
- optimal_roll_date: pick the roll date maximising spread (minimising cost)
- track_roll_pnl: cumulative roll P&L across a sequence of rolls

---

## v0.146.0 — 2026-04-10

Commodity Storage and Carry Plays.

- cash_and_carry: buy spot / sell forward / store — profit decomposition
- Implied storage cost from forward-spot spread
- Implied convenience yield extraction
- Arbitrage-free curve verified: zero cash-and-carry profit
- StorageFacility: capacity, injection/withdrawal rate limits, variable costs
- Intrinsic value: deterministic buy-low / sell-high (single-cycle greedy)
- Extrinsic value: spread-option proxy proportional to vol × √T
- Total value = intrinsic + extrinsic; positive in contango, zero on flat curve
- Inventory constraints (initial inventory, min working level)

---

## v0.145.0 — 2026-04-10

Inter-Commodity Spreads: crack, spark, dark, crush.

- GenericSpread: weighted multi-leg inter-commodity spread with residual_exposure check
- crack_spread_321 / crack_spread_532: crude → gasoline + distillate/heating oil
- SparkSpread: power − heat_rate × gas, implied generation margin
- DarkSpread: power − heat_rate × coal
- crush_spread / reverse_crush: soybean → meal + oil
- Balanced spreads verified: zero residual and PV invariant under uniform price shift

---

## v0.144.0 — 2026-04-10

Commodity Term Structure Trading: calendars, steepeners, butterflies.

- CommodityCalendarSpread: matched-notional long near / short far, zero parallel by construction
- CommoditySteepener: long far / short near, profits from curve steepening
- CommodityButterfly: 1:2:1 weights, zero parallel, zero steepener for even spacing
- dv01_neutral_quantity: far-leg quantity for flat parallel sensitivity
- curve_structure_monitor: contango / backwardation / mixed / flat classification
- CurveStructureSnapshot with sorted deliveries, forwards, consecutive spreads
- Numerical verification: parallel shift leaves all trade PVs unchanged

---

## v0.143.0 — 2026-04-10

Commodity Daily P&L: spot/carry/roll decomposition + attribution.

- compute_commodity_daily_pnl: official P&L = spot + carry + roll + new trades + amendments
- CommodityDailyPnL dataclass with market_move_pnl property
- Spot P&L from forward curve moves at constant delivery dates
- Carry P&L from convenience yield earned minus storage cost accrued
- Roll P&L from delivery-date rolls (old → new at current curve)
- attribute_commodity_pnl: per-commodity and per-tenor bucket breakdown
- Parallel-vs-shape decomposition: parallel = Σ qty × mean(Δfwd), shape = residual
- Attribution sums to total across both commodity and tenor dimensions

---

## v0.142.0 — 2026-04-10

Commodity Book — Pillar 6 (Commodity Desk) begins.

- CommodityBook: per-commodity, per-sector, per-tenor position aggregation
- CommodityTradeEntry: trade + commodity / sector / unit / quantity / reference price / delivery date
- CommodityPosition: net/long/short quantity and notional per commodity
- CommoditySectorExposure: aggregate notional per sector with commodity count
- TermStructureBucket: net notional per tenor bucket (front, ≤6M, ≤1Y, ≤2Y, >2Y)
- commodity_tenor_bucket helper
- CommodityLimits: per-commodity, per-sector, net, gross, per-tenor caps
- check_limits: detect breaches across all limit types

---

## v0.141.0 — 2026-04-10

Equity FRTB SA Capital — Pillar 5 (Equity Desk) complete.

- EquityRiskInputs: per-name delta / vega / curvature / notional
- EquityClassification: market cap, region, rating, exotic flag
- map_to_frtb_bucket: large/small cap × developed/emerging (BCBS $2bn threshold)
- equity_to_frtb_positions: builds dict format for calculate_frtb_sa
- equity_frtb_capital: one-call SbM + DRC + RRAO equity capital report
- EquityCapitalReport: components, total RWA, capital efficiency, bucket breakdown
- Manual hand-calculation tests for delta, DRC, intra-bucket correlation
- Direct cross-check against calculate_frtb_sa output

---

## v0.140.0 — 2026-04-10

Equity Rich/Cheap Monitors + Delta/Vega Hedging.

- ZScoreSignal: generic z-score / percentile / rich-cheap-fair signal
- implied_vs_historical_vol: implied vol vs realised vol distribution
- skew_monitor: 25-delta risk reversal level vs history
- calendar_monitor: front-vs-back vol spread vs history
- delta_hedge / vega_hedge: single-instrument hedge quantity
- optimal_delta_vega_hedge: 2x2 linear system to flatten both delta and vega
- hedged_exposure: combined book + allocations exposure
- Linear-dependence guard for degenerate hedge instrument pairs

---

## v0.139.0 — 2026-04-10

Dividend Strategies: futures, basis, carry trade, roll-down, backtest.

- DividendFuture: exchange-traded contract on cumulative divs over a period
- DividendFuture.settlement_value, fair_price, MTM PV with direction
- dividend_basis: traded − option_implied
- EquityCarryTrade: long stock + short matched dividend swap
- EquityCarryTrade.net_dividend_exposure: zero by construction when matched
- dividend_curve_carry: P&L from selling implied curve and receiving realised
- ImpliedVsRealisedBacktest + implied_vs_realised_backtest: bias / MAE / RMSE
- DividendSwap PV identity test: PV = PV(realised) − fixed × df

---

## v0.138.0 — 2026-04-09

Dispersion Trading: implied correlation, dispersion trades, correlation risk.

- index_variance / index_vol: σ²_idx = Σw²σ² + ρ((Σwσ)² − Σw²σ²)
- implied_correlation: round-trips against index_vol exactly
- historical_correlation: mean upper-triangle pairwise correlation
- DispersionTrade: long basket of single-name var, short index var (or reverse)
- DispersionTrade.pv: model PV given single vols and uniform correlation
- DispersionTrade.dispersion_value: pure basket-vs-index dispersion
- DispersionTrade.correlation_sensitivity: analytic ∂PV/∂ρ matches finite difference
- CorrelationTermStructure: linear interp / flat extrapolation
- Long dispersion is monotone decreasing in correlation (max at ρ=0)

---

## v0.137.0 — 2026-04-09

Equity Vol Desk: surface management, RV strategies, vega ladder, cross-Greeks.

- EquityVolSurface: strike × expiry surface anchored on ATM strike
- VolPillar: per-expiry strikes/vols
- Bumps: parallel, term (per-pillar), skew (linear tilt), curvature (quadratic)
- Bumped surface reprices vanillas to first order via vega
- CalendarSpread: same-strike two-expiry vol RV trade
- RiskReversal: long OTM call + short OTM put (or reverse) — picks up skew
- VarianceSwap: fair variance via call/put strip replication (Demeterfi-Derman-Kamal-Zou)
- vega_ladder: aggregate vega by (expiry, strike) bucket; total_vega helper
- volga: ∂²Price/∂σ² via central difference
- vanna: ∂²Price/(∂S ∂σ) via central difference

---

## v0.136.0 — 2026-04-07

Equity Daily P&L: official decomposition + Greek-based attribution.

- compute_equity_daily_pnl: official P&L = market move + new trades + amendments
- EquityDailyPnL dataclass: prior_pv, current_pv, market_move_pnl, new_trade_pnl, amendment_pnl, total_pnl
- attribute_equity_pnl: Greek-based decomposition Δ·ΔS + ½·Γ·ΔS² + ν·Δσ + θ·Δt + ρ·Δr
- TradeGreeks dataclass: delta, gamma, vega, theta, rho per trade
- GreekAttribution + EquityBookAttribution: per-trade and per-ticker breakdown
- Direction and notional_scale applied via signed multiplier
- Unexplained residual = total - explained (against re-priced PV)

---

## v0.135.0 — 2026-04-09

Equity Position Management + Books (Pillar 5 begins).

- EquityBook: per-ticker, per-sector position aggregation
- EquityTradeEntry: trade + ticker/sector/spot/beta/delta metadata
- EquityPosition: net/long/short notional, delta exposure per ticker
- SectorExposure: sector aggregation with name count
- EquityLimits: per-name, per-sector, net, gross, beta-weighted exposure caps
- check_limits: detect breaches across all limit types
- Long/short netting per ticker, separate long/short tracking

---

## v0.134.0 — 2026-04-09

Code Review Fixes (slices 107-133).

15 bugs fixed across phases A/B/C/D after thorough code review:
- BA-CVA formula: corrected with (1-ρ²) factor per MAR50.5
- SEC-SA SSFA: first-loss tranche now correctly returns K=1 (1250% RW)
- IRC modified duration: replaced annuity factor with proper Macaulay→Modified
- CMS convexity adjustment: simplified API (takes duration directly)
- CMS spread option: Bachelier fallback uses abs(spread_fwd)
- Bond futures: zero yield case returns c×n+1, implied_repo NaN sentinel
- Inflation vol: store original expiry dates, fix div-by-zero on coincident pillars
- Futures: guard against negative ratio in implied_convenience_yield
- Settlement: AUCTION/ELECT handled, configurable replacement_cost_pct
- Fixings: load_csv proper error handling with skip_invalid flag
- Local vol: removed dead code (unused Dupire numerator/denom)
- Stress IRRBB: validate year and macro_paths bounds
- Bond desk: narrow exception handling

NB: BA-CVA and SEC-SA fixes will change capital numbers vs v0.133.0.

---

## v0.133.0 — 2026-04-09

Total Capital Aggregation + Unified Regulatory Portfolio.

**Phase D: Basel Regulatory Framework COMPLETE.**

- calculate_total_rwa: aggregates credit, securitisation, CCR, CVA, market, op risk
- Output floor application across IRB and SA totals
- calculate_capital_ratios: CET1, Tier1, Total with combined buffer requirements
- Capital conservation buffer (2.5%) + countercyclical + G-SIB
- Compliance check at all three capital levels
- RegulatoryPortfolio: unified container with VaR + IRC + RWA methods
- Position management with chaining, conversion to IRC/credit exposures
- One-call risk_summary() returns full breakdown
- Removed superseded basic regulatory.py (now fully replaced by subpackage)

---

## v0.132.0 — 2026-04-09

Specialty Regulatory Modules.

- Crypto-assets (BCBS d545): Group 1a/1b/2a/2b classification + RWA
- Group 2 exposure limits: 2% (2a+2b) and 1% (2b only) of Tier 1
- Step-in risk (BCBS 398): UnconsolidatedEntity, 9 indicators, weighted scoring
- 3 risk levels (high/medium/low) with capital charge factors
- FX rates with market conventions: direct, inverse, USD/EUR triangulation
- 19 standard pairs (EUR/USD/GBP/JPY + EM)
- Default FX rates for testing/demos
- Simplified SA for market risk (small bank framework)
- Pillar 3 disclosure: KM1 (key metrics), OV1 (RWA overview)

---

## v0.131.0 — 2026-04-09

Basel II / II.5 Legacy Framework.

- Basel II SA: sovereign, bank (Option 1 & 2), corporate, retail, real estate
- Basel II BBB corporate at 100% (vs Basel III's 75%)
- Basel II IRB: same Vasicek formula but no Basel IV LGD floors
- Operational risk: BIA (15% × GI), TSA (8 business lines × β), AMA (internal models)
- AMA insurance mitigation capped at 20%, BEICF adjustments
- CEM (Current Exposure Method): EAD = max(0, MTM) + add-on by derivative type/maturity
- CEM netting: A_net = 0.4 × A_gross + 0.6 × NGR × A_gross
- Basel II.5 stressed VaR with multiplier and holding period scaling
- Basel II.5 total market risk: VaR + sVaR + IRC + CRM + specific risk

---

## v0.130.0 — 2026-04-09

Liquidity (LCR/NSFR) + Operational Risk (SMA).

- LCR (LIQ30): HQLA with L1/L2A/L2B caps and haircuts
- Cash outflows by liability type, cash inflows capped at 75% of outflows
- LCR = HQLA / Net Cash Outflows ≥ 100%
- NSFR (LIQ40): ASF / RSF with full factor tables
- Available Stable Funding by funding type, RSF by asset type + off-BS
- Operational Risk SMA (OPE25):
  - Business Indicator components: ILDC + SC + FC
  - BIC piecewise linear: 12% / 15% / 18% marginal coefficients
  - Internal Loss Multiplier from 10-year average loss
  - K_SMA = BIC × ILM, RWA = K × 12.5

---

## v0.129.0 — 2026-04-09

Stress Testing + IRRBB.

- Macro stress framework: GDP, unemployment, house prices, credit spreads, FX
- 3 standard scenarios: baseline, adverse, severely adverse (EBA/CCAR style)
- PD/LGD stress multipliers from macro shocks (with sign-corrected formula)
- Credit, market, integrated stress tests over multi-year horizon
- IRRBB (SRP31): standardised IR shock scenarios (parallel up/down, steepener, flattener)
- Duration gap analysis, PV01, EVE impact under all scenarios
- NII sensitivity by repricing bucket
- IRRBB capital charge via Supervisory Outlier Test (15% Tier1)

---

## v0.128.0 — 2026-04-09

Capital Framework + G-SIB + TLAC.

- Output floor (CAP30): max(IRB, 72.5% × SA) with transitional schedule
- Output floor by risk type with breakdown
- Leverage ratio (LEV30): Tier1 / total exposure ≥ 3% (+ G-SIB buffer)
- Large exposures (LEX30): 10% reporting, 25% limit, 15% G-SIB-to-G-SIB
- CRM: collateral haircuts, exposure with collateral
- Off-balance sheet CCF: short-term, long-term, guarantees
- G-SIB scoring: 5 categories × 12 indicators × global denominators
- G-SIB buckets 1-5 with buffer requirements (1.0% to 3.5%)
- TLAC: max(18% × RWA + buffer, 6.75% × leverage)
- MREL: loss absorption + recapitalisation with 8%/3% floor

---

## v0.127.0 — 2026-04-09

Counterparty Credit Risk: Full SA-CCR, BA-CVA, SA-CVA, CCP.

- SA-CCR (CRE52): full supervisory factors across IR, FX, CR, EQ, COM
- Maturity factor (margined and unmargined), supervisory duration
- Supervisory delta for non-options and options
- Adjusted notional with IR/CR using duration, others using MF
- Replacement Cost with collateral, threshold, MTA, NICA
- PFE multiplier with floor 5%
- Total EAD = α × (RC + PFE), α = 1.4
- BA-CVA: K_CVA = 2.33 × sqrt(Σ(s×CVA)² + ρ² × (Σ s×CVA)²)
- SA-CVA: K = m_CVA × sqrt(K_delta² + K_vega²)
- CCP exposures: 2% RW for QCCP trades, 100% non-QCCP
- CCP default fund contribution RWA per CRE54

---

## v0.126.0 — 2026-04-09

Securitisation + Trade-Specific RWA.

- SEC-SA: supervisory formula approach (SSFA) with K_SA, n, LGD, w
- SEC-IRBA: same SSFA but using K_IRB from underlying pool
- ERBA: External Ratings-Based Approach (CRR2) with CQS 1-17, RW base table
- ERBA senior vs non-senior with thickness adjustment T^(-c)
- STS securitisation lower floor (10% vs 15%)
- Re-securitisation: 1.5x multiplier with 100% floor
- pd_to_cqs: PD-to-CQS mapping for ERBA
- Trade-specific RWA: CDS, repo, TRS, loan helpers
- Each trade-RWA includes CCR + reference asset risk decomposition

---

## v0.125.0 — 2026-04-09

Credit RWA (Basel III/IV).

- SA-CR risk weights: sovereign, bank (ECRA + SCRA), corporate, retail, RE
- Residential RE risk weights by LTV bucket (general + income-producing)
- Commercial RE risk weights by LTV
- IRB Vasicek formula: K = LGD × [Φ(...) - PD] × maturity_factor
- Asset correlation R per asset class with SME firm-size adjustment
- Maturity adjustment b(PD) = (0.11852 - 0.05478 × ln(PD))²
- F-IRB calculation with default LGD 0.45 (senior unsecured)
- A-IRB with LGD floor (Basel IV reforms)
- Specialised lending slotting (PF, OF, CF, IPRE, HVCRE) — 5 categories
- compare_sa_vs_irb: side-by-side approach comparison

---

## v0.124.0 — 2026-04-09

Incremental Risk Charge (IRC).

- 7 rating transition matrices: global, Europe, EM, financials, sovereign, recession, benign
- Credit spread term structure by rating (1Y-10Y, with linear interpolation)
- LGD by seniority (senior_secured 25%, unsecured 45%, subordinated 75%, equity 100%)
- IRCPosition dataclass and IRCConfig with configurable matrix
- Vectorised MC simulation via Gaussian copula (NumPy + scipy)
- 99.9% 1-year loss percentile with full distribution stats
- Issuer-level netting (long/short within issuer), no cross-issuer netting
- quick_irc() and calculate_irc_by_issuer() convenience functions

---

## v0.123.0 — 2026-04-09

FRTB Internal Models Approach (MAR30-33).

- Liquidity-adjusted ES with 5-step cascading horizon (10/20/40/60/120 days)
- Stressed ES via ratio scaling on reduced factor set
- NMRF charge: zero-diversification sum scaled to liquidity horizon
- Internal DRC: vectorised two-factor Gaussian copula MC (50K paths default)
- IMCC formula: max(ES, m_c × ES_avg) + max(SES, m_c × SES_avg) + NMRF
- Backtesting: traffic light zones with plus factor (MAR33)
- P&L Attribution Test: Spearman + KL divergence per desk
- compare_ima_vs_sa: side-by-side IMA vs SA comparison

---

## v0.122.0 — 2026-04-09

FRTB Standardised Approach (full).

- SbM: delta, vega, curvature across GIRR, CSR, EQ, FX, COM
- Risk weights and correlations per Basel MAR21
- Within-bucket and across-bucket aggregation formulae
- DRC: Default Risk Charge with obligor netting (jump-to-default)
- RRAO: Residual Risk Add-On (exotic 1%, other 0.1%)
- Total FRTB-SA = SbM + DRC + RRAO, with RWA = capital × 12.5
- Replaces basic FRTB delta in regulatory.py with full multi-risk-class version

---

## v0.121.0 — 2026-04-09

Regulatory Subpackage: Ratings + VaR/ES Engine.

Begins the Basel regulatory framework port from my_calculations/.
- regulatory/ subpackage with unified API
- ratings.py: PD-rating tables (21 ratings), normalisation, resolve functions, IG/HY classification
- var_es.py: parametric VaR (normal + t), historical VaR, Monte Carlo VaR
- Expected Shortfall (CVaR) for all three methods
- Portfolio VaR: component/marginal decomposition, diversification benefit
- Backtesting: Basel traffic light zones (green/yellow/red), Kupiec POF test
- quick_var() and compare_var_methods() convenience functions

---

## v0.120.0 — 2026-04-09

LMM Calibration + Multi-Factor SABR.

- Rebonato swaption vol approximation from LMM parameters
- Exponential decay correlation matrix for forward rates
- LMM vol calibration: iterative scaling to match swaption grid (RMSE < 2%)
- MultiFactorSABR: SABR with term structure of (alpha, rho, nu), shared beta
- Linear interpolation of SABR params between expiries
- Joint calibration across multiple expiries via per-slice SABR calibrate

---

## v0.119.0 — 2026-04-09

Rough Volatility (rBergomi).

- Fractional Brownian motion via Cholesky on Toeplitz covariance
- Correct fBM variance scaling: Var(B^H(T)) = T^{2H}
- rBergomi MC: variance driven by fBM with Hurst H < 0.5
- European option pricing under rough vol
- Higher vol-of-vol produces fatter tails (validated)

---

## v0.118.0 — 2026-04-09

Stochastic Local Vol (SLV).

- SLVModel: combines Heston stochastic vol with Dupire local vol
- Leverage function L(S, t, v) with configurable mixing fraction
- mixing=1 recovers local vol, mixing=0 recovers pure Heston
- SLV MC: correlated Euler scheme for spot and variance
- European pricing: SLV MC matches local vol MC for mixing=1

---

## v0.117.0 — 2026-04-09

Local Volatility (Dupire).

- Dupire local vol surface from market implied vols (finite difference)
- Gatheral total variance formulation with regularisation
- LocalVolSurface: bilinear interpolation on (strike, time) grid
- Local vol MC: Euler scheme simulation under σ_loc(S, t) dynamics
- European option pricing: MC matches Black-Scholes for flat vol (within 5%)
- Handles skewed implied surfaces with strike-dependent local vol

---

## v0.116.0 — 2026-04-09

Historical Data + Fixings.

- FixingsStore: set/get/has daily rate fixings (SOFR, ESTR, CPI, etc.)
- JSON persistence: save to disk, reload on init
- CSV import: load fixings from external CSV files
- Bulk operations: bulk_set, series queries with date filtering
- create_sample_fixings: synthetic SOFR, ESTR, Fed Funds, CPI history
- Deterministic sample data for reproducible testing

---

## v0.115.0 — 2026-04-09

FRED + ECB Market Data Providers.

- MarketDataProvider base class with fetch/available_series interface
- FREDProvider: SOFR, Fed Funds, UST yields (3M-30Y), CPI via FRED API
- ECBProvider: ESTR, EURIBOR, EUR govt yields via ECB SDW API
- SampleProvider: synthetic data for testing (no network, deterministic)
- RateSeries: time series with latest(), on_date(), between() queries
- build_curve_from_yields: convert yield dict to DiscountCurve
- Session-level caching for both FRED and ECB providers

---

## v0.114.0 — 2026-04-09

Physical vs Cash Settlement Framework.

- SettlementType enum: CASH, PHYSICAL, AUCTION, ELECT
- Settlement conventions per product type (IR, CDS, equity, FX, commodity)
- CDS settlement: physical (deliver bond, receive par) vs cash/auction (receive LGD)
- Option settlement: cash (receive intrinsic) vs physical (deliver/receive shares)
- Futures settlement: cash (variation margin) vs physical (delivery + invoice)
- Settlement risk: exposure window between trade and settlement date
- Physical CDS net payout matches cash payout (validated)

---

## v0.113.0 — 2026-04-09

Dividend Modelling.

- Implied dividends from put-call parity (single expiry and term structure)
- Strip discrete dividends from cumulative PV curve
- DividendSwap: exchange realised divs for fixed, fair fixed rate, PV
- Dividend forward: forward price of cumulative dividends
- Dividend risk: dF/d(div) delta and div yield rho sensitivity
- Dividend forward matches swap fair fixed (round-trip validated)

---

## v0.112.0 — 2026-04-09

Bond Desk Tools.

- Curve fitting from bonds: bootstrap discount curve from market prices
- Rich/cheap analysis: identify bonds trading above/below fitted curve
- Repo carry: coupon income minus financing cost, breakeven repo rate
- Securities lending fee: borrow cost for short selling

---

## v0.111.0 — 2026-04-08

Bond Futures + CTD.

- Conversion factor: price at standard yield (6%), normalises deliverables
- Cheapest-to-deliver: minimises gross basis across delivery basket
- Implied repo rate: annualised return from futures delivery
- Bond futures basis: gross basis, carry (coupon - financing), net basis
- Net basis ≈ delivery option value (gross minus carry)

---

## v0.110.0 — 2026-04-08

Equity Index + Commodity Futures.

- EquityFuture: fair price (S×exp((r-q)T)), basis, convergence, daily settlement P&L
- CommodityFuture: observed price, contango/backwardation detection
- Roll yield: annualised return from rolling near to far contract
- Calendar spread: long near, short far with spread and structure analysis
- Futures strip curve: sorted forward curve from observed prices
- Implied convenience yield from futures/spot relationship

---

## v0.109.0 — 2026-04-08

Inflation Caps/Floors.

- ZC inflation cap/floor: Black-76 on CPI index ratio vs compounded strike
- YoY inflation cap/floor: strip of annual caplets on CPI(t)/CPI(t-1)
- Deep ITM cap exceeds swap PV (time value validated)
- InflationVolSurface: ATM vol grid with linear interpolation, flat extrapolation
- Parallel vol bump for sensitivity analysis

---

## v0.108.0 — 2026-04-08

Zero-Coupon Swaps + IR Digitals.

- ZeroCouponSwap: single compounded fixed vs floating at maturity
- ZC par rate: df(T)^(-1/T) - 1 (differs from standard swap par rate)
- Digital cap/floor: binary payoff via tight call-spread approximation
- Digital CMS cap: digital on CMS rate with convexity adjustment
- Payout scaling: digital PV scales linearly with payout amount

---

## v0.107.0 — 2026-04-08

IR Exotics — CMS + Range Accruals.

- CMS convexity adjustment via linear TSR model (positive, scales with vol² and time)
- CMSLeg: floating leg paying a long-dated swap rate with convexity-adjusted forwards
- CMS cap/floor: strip of CMS caplets priced via Black-76 on adjusted forward
- CMS spread option: payoff on (CMS_long - CMS_short - K) via Margrabe vol approximation
- Range accrual: coupon accrues only when rate stays in [L, U], digital decomposition
- Bachelier fallback for near-zero/negative spread forwards

---

## v0.106.0 — 2026-04-08

CVA Desk Tools.

- CVA CS01: sensitivity to 1bp credit spread shift (positive — CVA rises with spreads)
- CVA IR01: sensitivity to 1bp rate shift (negative — higher rates reduce CVA)
- CVA by trade: stand-alone CVA contribution per trade
- CVA hedge: CDS notional to offset CVA CS01, residual risk near zero
- Incremental CVA: marginal CVA from adding a trade, captures netting benefit

---

## v0.105.0 — 2026-04-08

Tranche Correlation Trading.

- Tranche delta: spread sensitivity per bp of index move (equity > mezzanine)
- Tranche CS01: PV change per bp, scales with notional
- Correlation sensitivity: d(tranche_spread)/d(rho) — equity vs senior opposite signs
- Base correlation skew: implied correlation curve from market tranche spreads
- Skew bumps: parallel and tilt correlation bumps with spread impact

---

## v0.104.0 — 2026-04-08

Credit Curve Relative Value.

- Cross-name RV: z-score and percentile ranking within peer group
- Outlier detection: rich/cheap/fair signals based on z-score threshold
- Term structure RV: CDS curve slope (long - short), steepness z-score
- Sector screening: mean spread, dispersion, cheapest/richest per sector
- Sorted output: sectors by widest, names by z-score

---

## v0.103.0 — 2026-04-08

Basis Trading (Bond vs CDS).

- CDS-bond basis: CDS spread minus Z-spread (and ASW spread variant)
- Matched credit curve produces near-zero basis (validated)
- Negative basis trade: buy bond + buy protection, positive carry when coupon > spread
- Positive basis trade: sell bond + sell protection, opposite economics
- Basis monitor: z-score, percentile, signal (negative/positive/fair)

---

## v0.102.0 — 2026-04-08

Credit Index Flow Trading.

- IndexDefinition: named, weighted constituents with sector/rating metadata
- Intrinsic spread: notional-weighted average of constituent par spreads
- Spread dispersion: weighted std of constituent spreads (zero for uniform)
- IndexSeries: on-the-run / off-the-run series tracking
- Index skew: market spread vs intrinsic basis with dispersion
- Index roll: composition change detection (added/removed names), spread change

---

## v0.101.0 — 2026-04-08

Single-Name CDS Market Making.

- build_cds_curve: bootstrap survival curve from market par spreads (round-trips)
- Upfront/running conversion: spread_to_upfront, upfront_to_spread (standard quoting)
- Pricing ladder: bid/ask spread bumps with PV and upfront at each level
- Mark-to-market: PV, par spread, upfront, RPV01, spread-to-par for seasoned CDS
- Roll P&L: unwind old contract + enter new on-the-run, captures roll cost

---

## v0.100.0 — 2026-04-08

Credit Risk Measures + Book Integration.

- CS01: PV change per 1bp parallel shift in hazard rates (bump-and-reprice)
- Spread DV01: per-pillar credit spread sensitivity (key-rate CS01)
- Jump-to-default: PV change on immediate default (LGD - current PV)
- Survival curve bumping: parallel and per-pillar hazard rate shifts
- CreditBook: credit trade container with per-name CS01, JTD, positions
- Long/short netting: offsetting trades produce zero CS01

---

## v0.99.0 — 2026-04-08

Regulatory Capital (IR).

- SA-CCR: trade-level add-on (notional × supervisory factor × maturity factor × delta)
- SA-CCR netting: hedging set aggregation across 3 maturity buckets with cross-bucket correlation
- Perfectly offsetting trades produce zero add-on
- FRTB Sensitivities-Based Approach: IR delta risk charge
- 10 tenor buckets (3M–30Y) with Basel risk weights
- Three correlation scenarios (low, medium, high)
- Spread positions get diversification benefit

---

## v0.98.0 — 2026-04-08

Rich/Cheap Analysis.

- Relative value: market rate vs model-implied, spread, z-score, percentile
- Signal generation: rich/cheap/fair based on z-score threshold
- rv_from_curve: relative value using curve-implied par rate
- Spread monitor: 2s10s, 5s30s etc with z-score signals (wide/tight)
- Butterfly monitor: 2s5s10s with belly_cheap/belly_rich signals
- Flat curve produces near-zero spread and butterfly (validated)

---

## v0.97.0 — 2026-04-07

Swaption Desk Tools.

- VolCube: ATM vol grid + per-cell SABR smile (incl shifted SABR for negative rates)
- Vol surface bumps: parallel and term structure
- Straddle, strangle, risk reversal with full Greeks (delta, vega, gamma, theta)
- Delta hedge: offset swaption book delta with a swap (auto-notional)
- Vega hedge: offset vega with another swaption (auto-ratio)
- Bump-and-reprice Greeks engine for swaptions

---

## v0.96.0 — 2026-04-07

Curve Trading Strategies.

- Spread trades: 2s10s steepener/flattener with auto DV01-neutral hedge ratios
- Butterfly trades: 2s5s10s with DV01-neutral wing sizing (equal split)
- swap_dv01: bump-and-reprice DV01 for any swap
- swap_carry: carry + roll-down via time-shifted curve (preserves forward rates)
- breakeven_rate_move: how much rates can move before carry is lost

---

## v0.95.0 — 2026-04-07

Daily P&L Workflow.

- Official P&L: market move + new trade + amendment decomposition
- Desk P&L: per-book P&L across a desk
- P&L Attribution: sequential bump-and-reprice (rates → vol → theta → unexplained)
- Per-trade attribution with trade IDs
- Per-bucket attribution by tenor bucket
- Components sum to total (explained + unexplained = total)

---

## v0.94.0 — 2026-04-07

Position Management + Books.

- Book: named trade container with PV, DV01, position aggregation
- Desk: collection of books with aggregate risk
- Position: net exposure by instrument type and tenor bucket (FRTB-style buckets)
- BookLimits: DV01 cap, per-counterparty notional cap, tenor-bucket DV01 limits
- Breach detection: check_limits returns all violations with details
- Desk risk summary: PV by book, positions, breaches in one report

---

## v0.93.0 — 2026-04-05

Waterfall + Triggers + Autocall.

- WaterfallEngine: priority allocation (senior → mezz → equity)
- Interest paid before principal, senior before junior
- Triggers: OC ratio, IC ratio — breach diverts to senior
- Autocall: early redemption when reference exceeds barrier
- Reset for multi-period simulation

---

## v0.92.0 — 2026-04-05

Deal Structuring — groups linked instruments with roles and metadata.

- Deal: container with deal_id, counterparty, book, desk
- DealComponent: instrument + role (principal, hedge, fee, option, collateral)
- Linkages: components reference each other by name
- Aggregate PV, PV by component, PV by role
- Deal DV01 and risk report (JSON-serializable)
- Full serialization: to_dict / from_dict / to_json / from_json

---

## v0.91.0 — 2026-04-05

Curve Jacobian + Roll-Down Analysis.

- curve_jacobian: d(zero_rate) / d(pillar_zero_rate) via FD per pillar
- input_jacobian: d(zero_rate) / d(market_quote) for any build function
- curve_rolldown: zero rate changes from time passing on shaped curve
- rolldown_pnl: estimated P&L for a position from curve roll-down
- Flat curve: diagonal Jacobian ≈ identity, roll-down = pure time decay
- Upward curve: positive roll-down P&L for long positions

---

## v0.90.0 — 2026-04-05

Multi-Curve Simultaneous Solver.

- global_bootstrap: Newton iteration solving all pillar DFs simultaneously
- FD Jacobian: d(instrument_PV) / d(df_pillar) for the full system
- Matches sequential bootstrap to 2% at all tenors
- coupled_bootstrap: OIS + projection solved together in one Newton system
- Dual-curve: floating forwards from projection, discounting from OIS

---

## v0.89.0 — 2026-04-05

Curve Engine — declarative curve building.

- CurveDefinition: name, currency, instruments, interpolation, extrapolation
- Presets: usd_ois(), eur_estr() with standard tenors
- CurveBuilder: definition + MarketDataSnapshot → DiscountCurve
- Smith-Wilson extrapolation policy support
- CurveSet: groups discount + projection curves, to_pricing_context()
- Definition serialization: to_dict / from_dict round-trip

---

## v0.88.0 — 2026-04-05

Calibration Stability — robust calibration utilities.

- tikhonov_regularise: penalise deviation from prior parameters
- enforce_bounds: clip parameters to valid ranges
- calibration_quality: RMSE, max error, residuals
- multi_start_calibrate: run from N random starts, pick best
- perturbation_stability: test param sensitivity to input noise
- Multi-start finds Rosenbrock minimum from any starting region

---

## v0.87.0 — 2026-04-05

FEM Non-Uniform Mesh — improved accuracy for BS pricing.

- _sinh_mesh: concentrated nodes around strike via sinh mapping
- fem_bs_supg: heat-transform FEM with non-uniform mesh
- Better accuracy than uniform mesh at same node count
- Works for ATM, ITM, and OTM options

---

## v0.86.0 — 2026-04-05

AAD Calibration Jacobian — parameter risk to market moves.

- sabr_jacobian: d(model_vol)/d(alpha, rho, nu) via AAD for all strikes
- _sabr_hagan_aad: full Hagan formula with Number arithmetic on tape
- calibration_risk: d(params)/d(market_vols) via implicit function theorem
- AAD Jacobian matches finite differences to 5%+
- Alpha sensitive to ATM, rho to skew, nu to wings

---

## v0.85.0 — 2026-04-05

AAD Swaptions + Caplets — all Greeks in one backward pass.

- aad_swaption_pv: Black-76 on forward swap rate with AAD
- aad_caplet_pv: single caplet/floorlet with AAD
- Delta, vega, annuity sensitivity all from one propagation
- AAD Greeks match finite differences to 3+ significant figures

---

## v0.84.0 — 2026-04-05

XVA Depth — wrong-way risk and collateralised XVA.

- simulate_wwr_exposures: correlated rate + hazard paths (Gaussian copula)
- cva_wrong_way: path-level EPE × default intensity, captures WWR
- cva_collateralised: margin period of risk reduces exposure → lower CVA
- fva_collateralised: only uncollateralised portion incurs funding cost
- Fully collateralised → CVA near zero, FVA = 0
- WWR CVA ≥ independent CVA (positive rate-credit correlation)

---

## v0.83.0 — 2026-04-05

Smith-Wilson Extrapolation — regulatory curve extension.

- smith_wilson_calibrate: fit coefficients to match market DFs exactly
- smith_wilson_df / forward: extrapolated DF and forward rate
- Forward rate converges to UFR asymptotically
- smith_wilson_curve: builds DiscountCurve with extrapolation tenors
- EIOPA defaults: UFR=3.45%, alpha=0.1, convergence at 60Y

---

## v0.82.0 — 2026-04-05

Futures-Based Curve Stripping.

- futures_strip: deposits + futures + swaps in a single bootstrap
- HW convexity adjustment applied to futures rates
- Turn-of-year spread for year-end funding premium
- Futures-stripped curve reprices inputs at market rates

---

## v0.81.0 — 2026-04-05

Nelson-Siegel + Svensson parametric yield curves.

- nelson_siegel_yield: 4-param (level, slope, curvature, decay)
- svensson_yield: 6-param (adds second hump)
- ns/svensson_discount_curve: build DiscountCurve from parameters
- calibrate_nelson_siegel / calibrate_svensson: fit to market yields
- NS(beta0, 0, 0) = flat curve; Svensson(beta3=0) = NS

---

## v0.80.0 — 2026-04-05

FX Barriers + Vanna-Volga.

- fx_barrier_pde: knock-in/out via 1D PDE with FX rates (r_dom, r_for)
- vanna_volga_barrier: smile-consistent pricing from ATM/25D RR/25D BF
- Flat smile VV matches PDE; non-flat smile adjusts price via vanna/volga costs
- In-out parity holds, up/down barriers, calls and puts

---

## v0.79.0 — 2026-04-05

Commodity Seasonality + Storage.

- SeasonalFactors: monthly adjustment factors (natural gas, power presets)
- SeasonalForwardCurve: base price × seasonal factor per delivery month
- StorageCostModel: convenience yield, cost of carry, contango/backwardation
- calendar_spread_option: Kirk/Margrabe approximation for inter-month spreads
- Higher correlation → lower spread vol → lower spread option price

---

## v0.78.0 — 2026-04-05

Multi-Calendar + Settlement conventions.

- TARGETCalendar (EUR): Good Friday, Easter Monday, Labour Day, Christmas
- LondonCalendar (GBP): Early May, Spring/Summer bank holidays
- TokyoCalendar (JPY): 18+ Japanese holidays
- JointCalendar: union of holidays from multiple calendars
- ACT/ACT ISDA day count: proper year-fraction across leap year boundaries
- Easter computation (anonymous Gregorian algorithm)

---

## v0.77.0 — 2026-04-05

Amortising + Accreting Swaps — per-period notional schedules.

- AmortissingSwap: arbitrary notional profile per period
- amortising() factory: linear decrease to zero
- accreting() factory: linear increase from initial to final
- Roller-coaster: any arbitrary notional schedule
- par_rate, DV01, WAL (weighted average life)
- Amortising DV01 < bullet DV01

---

## v0.76.0 — 2026-04-05

Shifted SABR + Normal Vol Swaptions.

- shifted_sabr_implied_vol: F+shift, K+shift for negative rate environments
- shifted_sabr_price: full pricing with shift parameter
- sabr_normal_vol: lognormal → normal vol conversion
- Zero shift reduces to standard SABR exactly
- Bachelier pricing with SABR-derived normal vol consistent with Black price

---

## v0.75.0 — 2026-04-04

Multi-Factor HJM + LIBOR Market Model.

- MultiFactorHJM: 2-3 volatility functions (level, slope, curvature), no-drift per factor
- Single-factor reduces to existing HJM, two factors produce more variance
- LMM (BGM): forward LIBOR rates as state variables, lognormal dynamics
- LMM caplet pricing via MC, matches Black caplet price within 15%
- Rebonato swaption vol approximation, scales linearly with forward vols

---

## v0.74.0 — 2026-04-04

SABR MC Dynamics — direct simulation of SABR SDE.

- sabr_mc_paths: Euler-Maruyama with absorbing boundary at F=0
- sabr_mc_european: European pricing, matches Hagan approximation within 10%
- sabr_mc_asian: arithmetic average Asian under SABR smile dynamics
- sabr_mc_implied_vol: MC implied vol for comparison with Hagan formula
- Forward stays non-negative, vol stays positive (log-normal process)

---

## v0.73.0 — 2026-04-04

Heston MC Simulation — full-truncation Euler and QE schemes.

- heston_euler: full-truncation scheme, variance stays non-negative
- heston_qe: Andersen (2008) quadratic exponential, accurate with few steps
- heston_mc_european: European pricing via MC, matches semi-analytical within 5%
- heston_mc_barrier: knock-in/out under stochastic vol, in-out parity holds
- Put-call parity verified under MC

---

## v0.72.0 — 2026-04-04

Two-Asset Options via ADI — spread, basket, best-of on 2D correlated GBM.

- two_asset_option: Craig-Sneyd ADI on (log S1, log S2) grid
- Spread: max(S1 - S2 - K, 0), correlation reduces spread vol
- Basket: max(w1*S1 + w2*S2 - K, 0), diversification effect
- Best-of: max(max(S1, S2) - K, 0), dispersion increases value
- Mixed derivative via Craig-Sneyd handles correlation correctly

---

## v0.71.0 — 2026-04-04

Unified Pricer Registry — completes the architecture from slice 23.

- get_pricer(): "cos_bs", "cos_heston" for spectral pricing
- get_greek_engine(): "aad" (BS, swap, CDS), "bump" (dv01, key rate)
- Heston PDE registered in get_pde_pricer("heston")
- All entries accessible via registry, interchangeable

---

## v0.70.0 — 2026-04-04

Dashboard Data Layer — structured reports for any frontend.

- portfolio_risk_report: total PV, DV01, per-trade breakdown
- scenario_grid: portfolio x scenarios → P&L matrix
- trade_blotter: table view with dates, types, PVs, counterparties
- All output is plain dicts — JSON-serializable for Plotly or any frontend
- 1737 tests, 95% coverage

---

## v0.69.0 — 2026-04-04

Convenience API + Top-Level Imports.

- `from pricebook import InterestRateSwap, PricingContext, Trade, ...` (30+ exports)
- `DiscountCurve.flat(ref, rate)` and `SurvivalCurve.flat(ref, hazard)` class methods
- `PricingContext.simple(ref, rate, vol, hazard)` for quick setup
- `PricingContext.replace(discount_curve=...)` for bump-and-reprice
- conftest helpers now delegate to `.flat()` classmethods

---

## v0.68.0 — 2026-04-04

Serialization — JSON round-trip for curves, instruments, trades, portfolios.

- Curve serialization: DiscountCurve, SurvivalCurve, PricingContext to/from dict
- Instrument serialization: IRS, Bond, CDS, FRA, Swaption, CapFloor
- Trade + Portfolio serialization with instrument type tags
- Instrument registry: get_instrument_class("irs"), list_instruments()
- Generic loaders: load_trade(dict), load_portfolio(list), from_json(string)
- to_json() for any supported object

---

## v0.67.0 — 2026-04-04

Sparse Grids (Smolyak Quadrature) — high-dimensional integration.

- clenshaw_curtis_nodes: nested 1D quadrature with exact polynomial integration
- smolyak_grid: Smolyak construction on [-1,1]^d, O(N*log(N)^{d-1}) vs O(N^d) for tensors
- sparse_grid_integrate: multi-dimensional integration with custom bounds
- Fewer points than full tensor product for comparable accuracy
- 2D and 3D polynomial integration verified

---

## v0.66.0 — 2026-04-04

Finite Element Method (1D) — Galerkin FEM for option pricing PDE.

- P1 (linear) and P2 (quadratic) elements with mass and stiffness matrices
- Sparse global assembly using SparseMatrix from v0.65
- Crank-Nicolson time stepping with Dirichlet boundary conditions
- Heat equation solver converges to analytical solution
- Black-Scholes FEM pricer via log-price heat transformation
- P2 converges faster than P1 on coarse grids

---

## v0.65.0 — 2026-04-04

Sparse Matrix Operations — foundation for FEM and large-scale risk.

- SparseMatrix: CSR wrapper with triplet construction, add, matmul, transpose
- sparse_solve: direct LU via scipy.sparse
- sparse_lu: factorisation for repeated solves
- sparse_cg: conjugate gradient for SPD systems
- tridiagonal_matrix: efficient builder for FD/FEM banded systems
- Matches dense results to machine precision

---

## v0.64.0 — 2026-04-04

Akima Spline Interpolation — local method that avoids overshoots.

- AkimaInterpolator: weighted average of neighbouring secants (Akima 1970)
- Ghost boundary treatment for edge slopes
- No wild oscillations on sharp jumps (unlike global cubic spline)
- Exact on linear data, continuous at knots
- Added to InterpolationMethod enum and create_interpolator factory
- Works with DiscountCurve

---

## v0.63.0 — 2026-04-04

JR + LR Binomial Trees — alternative parameterisations for option pricing.

- Jarrow-Rudd: equal probabilities (p=0.5), drift-adjusted up/down moves
- Leisen-Reimer: Peizer-Pratt inversion, N=51 matches BS to 4+ digits
- LR converges faster than CRR; all three converge to same Black-Scholes limit
- European and American pricing for both trees
- Registered in registry as "jr"/"jarrow_rudd" and "lr"/"leisen_reimer"

---

## v0.62.0 — 2026-04-04

CSA and Funding Framework — generic collateral and funding adjustments for any trade.

- CSA: threshold, MTA, rounding, margin frequency, eligible collateral, haircut, initial margin
- FundingModel: secured/unsecured rates, collateral rate, funding spread
- required_collateral: exposure → collateral amount respecting CSA terms
- collateral_adjusted_pv: generic — works with any trade (Trade, instrument, or pv_ctx object)
- funding_benefit_analysis: compare CSA options side by side
- Fully collateralised → zero funding cost; uncollateralised → spread × exposure × horizon

---

## v0.61.0 — 2026-04-04

Cross-Currency Swaps — full notional exchange with basis spread.

- CrossCurrencySwap: bilateral notional exchange, floating + basis spread
- MTM reset variant: foreign notional resets to market FX each period
- par_spread: analytical solver for zero-PV basis spread
- dv01_domestic / dv01_foreign: rate sensitivities per currency
- fx_delta: FX rate sensitivity
- MTM reset reduces FX delta vs standard XCCY

---

## v0.60.0 — 2026-04-03

IR Futures — SOFR futures with convexity adjustment.

- IRFuture: 1M/3M SOFR futures, price = 100 - rate, daily mark-to-market PV
- implied_forward: simply compounded forward rate from discount curve
- hw_convexity_adjustment: Hull-White analytical convexity bias (futures rate > forward rate)
- Convexity is positive, increases with vol (quadratic in σ) and maturity
- futures_strip_rates: compute rates for a strip with increasing convexity
- DV01: tick value per basis point

---

## v0.59.0 — 2026-04-03

AAD End-to-End Integration — the capstone. All Greeks in one backward pass.

- aad_black_scholes: delta, vega, rho simultaneously from one propagation
- aad_swap_pv: IR01 per pillar in one pass (receiver fixed swap)
- aad_cds_pv: IR01 + CS01 per pillar in one pass (protection buyer)
- Portfolio: sum of trades, AAD portfolio Greeks = sum of individual AAD
- Performance: AAD >2x faster than bump-and-reprice for 10 pillars
- All AAD Greeks match finite differences to 4+ significant figures
- Put-call parity verified for AAD Black-Scholes
- 1533 tests, 95% coverage

---

## v0.58.0 — 2026-04-03

AAD-aware Curves — pillar sensitivities in one backward pass.

- AADDiscountCurve: Number-valued pillar DFs, df(date) returns Number on tape
- AADSurvivalCurve: Number-valued pillar survivals, survival(date) returns Number
- d(price)/d(df[i]) and d(price)/d(surv[i]) for all pillars in one pass
- Values match float curves to machine precision
- AAD sensitivities match bump-and-reprice finite differences
- Only bracketing pillars get non-zero sensitivity (log-linear locality)
- 1519 tests, 95% coverage

---

## v0.57.0 — 2026-04-03

AAD-aware Interpolation — differentiation through curve lookups.

- aad_linear_interp: piecewise linear with Number-valued y, adjoint = interpolation weights
- aad_log_linear_interp: log-linear with Number-valued y (standard for discount factors)
- Values match float interpolation to machine precision
- AAD derivatives match finite differences
- Flat extrapolation beyond boundaries
- 1510 tests, 95% coverage

---

## v0.56.0 — 2026-04-03

MVA + KVA — margin and capital costs, completing the XVA framework.

- mva: margin valuation adjustment from IM profile and funding spread
- kva: capital valuation adjustment from regulatory capital profile and hurdle rate
- XVAResult: aggregates CVA, DVA, FVA, MVA, KVA with bcva and total properties
- Total XVA = CVA - DVA + FVA + MVA + KVA
- All components scale linearly, zero when inputs are zero
- 1496 tests, 95% coverage

---

## v0.55.0 — 2026-04-03

DVA + FVA — bilateral credit risk and funding cost.

- dva: debit valuation adjustment from own default probability and ENE
- bilateral_cva: BCVA = CVA - DVA, verified zero-sum for symmetric counterparties
- fva: funding valuation adjustment from expected exposure and funding spread
- FVA = 0 for fully collateralised trades, scales linearly with spread
- 1487 tests, 95% coverage

---

## v0.54.0 — 2026-04-03

CVA (Credit Valuation Adjustment) — the price of counterparty credit risk.

- simulate_exposures: MC simulation of portfolio PV under diffused rates
- expected_positive_exposure / expected_negative_exposure / expected_exposure
- cva: unilateral CVA = ∫ EPE * df * dPD * (1-R), discretised over time grid
- CVA increases with counterparty spread and LGD, zero with no default or no exposure
- 1478 tests, 95% coverage

---

## v0.53.0 — 2026-04-03

Trade Lifecycle — amendments, exercises, novations, and audit history.

- ManagedTrade: versioned wrapper around Trade with full lifecycle support
- Amendments: change notional, direction, counterparty, or instrument; original preserved
- Exercise: option → underlying (e.g. swaption → swap), prevents double exercise
- Novation: transfer counterparty, economics unchanged
- Audit trail: chronological LifecycleEvent history with version numbers
- EventType enum: CREATED, AMENDED, EXERCISED, NOVATED
- 1469 tests, 95% coverage

---

## v0.52.0 — 2026-04-03

Market Data Snapshots — structured market data for curve building and pricing.

- Quote: typed market observations (deposit rate, swap rate, CDS spread, vol point, FX spot)
- MarketDataSnapshot: dated collection of quotes, JSON serialisation round-trip
- Tenor parsing: "3M", "5Y", "1W", "1D" → year fractions and dates
- CurveConfig/PipelineConfig: declarative curve building configuration
- build_context: snapshot + config → PricingContext (discount, credit, vol, FX)
- HistoricalData: time series of snapshots, date-keyed lookup, JSON round-trip
- 1448 tests, 95% coverage

---

## v0.51.0 — 2026-03-29

VaR and stress testing.

- historical_var: percentile-based VaR from P&L vector
- historical_cvar: expected shortfall (conditional VaR), mean of tail losses
- parametric_var: delta-normal VaR from factor deltas and covariance matrix
- stress_test: reprice under shifted PricingContext scenarios (rate, vol, credit shifts)
- STANDARD_STRESSES: 4 predefined scenarios (parallel ±100bp, steepener, vol +5%)
- 1422 tests, 95% coverage

---

## v0.50.0 — 2026-04-02

P&L Explain — attribution of portfolio value changes.

- PnLResult: total, carry, rolldown, rate/vol/credit/FX/theta components, unexplained
- Greek-based attribution: delta*dx + 0.5*gamma*dx^2 per risk factor
- Carry: coupon income - funding cost
- Components sum to explained, total - explained = unexplained
- Extensible via `other` dict for custom components
- 1409 tests, 95% coverage

---

## v0.49.0 — 2026-04-02

Caching, memoisation, and lazy evaluation.

- CurveCache: LRU cache for df/forward lookups, per-curve invalidation, hit rate stats
- CalibrationCache: stores calibrated model params keyed by inputs hash, invalidation
- LazyValue: defers computation until .value accessed, computed once, resettable
- LRU eviction at configurable maxsize
- Lazy curve bootstrap: curve not built until first query
- 1395 tests, 95% coverage

---

## v0.48.0 — 2026-04-02

Dependency graph for incremental risk computation.

- DAG with market data → curves → instruments → portfolio topology
- Dirty-flag propagation: mark upstream → all downstream nodes dirty
- Topological sort: recompute only dirty nodes in correct order
- Isolated nodes stay clean (bump USD rate → EUR instruments unaffected)
- Cycle detection for graph integrity
- Node add/remove with automatic edge cleanup
- Portfolio integration test: realistic 9-node graph with selective bumping
- 1379 tests, 95% coverage

---

## v0.47.0 — 2026-04-02

Exotic loans — completing the exotic product suite.

- CPR/PSA prepayment models: constant and ramped prepayment rates
- Prepayment-adjusted cashflows and WAL (shorter than bullet confirmed)
- Covenant triggers: breach probability per period, acceleration on breach
- Covenant-adjusted PV and expected maturity
- Zero prepay/breach = unchanged from base loan
- 1361 tests, 95% coverage

---

## v0.46.0 — 2026-04-02

CDO tranches.

- Vasicek large homogeneous pool: conditional PD, loss distribution via Gauss-Hermite
- Tranche pricing: equity/mezzanine/senior expected loss and spread
- Equity highest spread > mezzanine > senior confirmed
- Tranche losses sum to portfolio loss
- Base correlation: flat corr that reprices [0, K] tranche, round-trip validated
- 1350 tests, 95% coverage

---

## v0.45.0 — 2026-04-02

Bermudan swaptions — two pricing methods.

- Hull-White tree: backward induction with exercise decision at each coupon date
- Longstaff-Schwartz MC: rate path simulation + polynomial regression on continuation
- Bermudan ≥ European confirmed for both methods
- Tree and LSM agree within 50% (simplified implementations, same order of magnitude)
- Payer and receiver variants, higher vol → higher price
- 1337 tests, 95% coverage

---

## v0.44.0 — 2026-04-02

Callable and puttable bonds — first exotic product.

- Callable bond: HW tree backward induction, min(continuation, call_price)
- Puttable bond: max(continuation, put_price) at put dates
- Callable ≤ straight ≤ puttable confirmed
- Low vol → optionality vanishes → approaches straight bond
- OAS: spread to risk-free that reprices with embedded option, round-trip validated
- 1329 tests, 95% coverage

---

## v0.43.0 — 2026-04-02

Recovery rate models — completing the credit model suite.

- BetaRecovery: recovery from Beta distribution (mean/std parameterised), PDF/CDF
- LGDModel: Loss Given Default = 1 - R, expected loss computation
- CorrelatedRecovery: R = f(M) systematic factor model, downturn LGD at tail percentiles
- Portfolio loss with correlated recovery: higher tail risk when recovery drops in downturns
- Zero sensitivity → deterministic recovery, floor/cap bounds respected
- 1317 tests, 95% coverage

---

## v0.42.0 — 2026-04-02

Credit rating transition models.

- Generator matrix Q with auto-computed diagonal (rows sum to 0)
- Transition probability P(t) = exp(Qt) via scipy matrix exponential
- Default probability, survival, spread term structure per rating
- Jarrow-Lando-Turnbull risky ZCB pricing
- MC rating migration simulation with absorbing default state
- Simulated default probabilities match analytical within 1%
- AAA < BBB < CCC spread ordering confirmed across tenors
- Standard 8-state generator (AAA to D) included
- 1295 tests, 95% coverage

---

## v0.41.0 — 2026-04-02

Stochastic credit intensity models.

- CIR intensity: analytical survival (Riccati/affine), MC survival matches within 3%
- Cox process: doubly stochastic Poisson default simulation from intensity paths
- Joint (rate, hazard) simulation with correlation via correlated BM
- Wrong-way risk: negative r-λ correlation → positive corr(df, default) confirmed
- Calibration: fit CIR params to CDS par spread term structure
- 1270 tests, 95% coverage

---

## v0.40.0 — 2026-04-02

HJM forward rate framework.

- Musiela parameterisation: f(t, x) where x = T-t
- No-drift condition: alpha = sigma * ∫sigma ds (risk-neutral)
- MC simulation of forward curve evolution on a tenor grid
- Discount factors from simulated short rate paths
- Average ZCB matches initial curve within 5%
- Zero vol → deterministic forward curve confirmed
- From-curve factory: extract forwards from DiscountCurve
- 1256 tests, 95% coverage

---

## v0.39.0 — 2026-04-02

Vasicek and G2++ short-rate models.

- Vasicek: analytical ZCB (A/B formulas), mean/variance, caplet pricing, exact OU simulation
- G2++: two-factor (x + y + phi), analytical ZCB, correlated BM simulation, MC discount factor
- G2++ calibrated to initial term structure via phi(t) and V(T)
- sigma2→0 collapses G2++ to one-factor Hull-White confirmed
- MC discount factors match market within 5%
- 1245 tests, 95% coverage

---

## v0.38.0 — 2026-04-02

Funded structures: repo, total return swap, funded participation.

- Repo: classic repo PV, cash lent, repurchase price, haircut, effective funding rate, implied repo from spot/forward
- Total Return Swap: total return receiver PV, fair spread, notional scaling
- Funded Participation: partial risk transfer, net carry (yield - funding - expected loss), pro-rata PV
- Cash-CDS basis: funded spread vs unfunded CDS spread comparison
- 1226 tests, 95% coverage

---

## v0.37.0 — 2026-04-02

Multi-curve framework depth: RFR compounding, IBOR fallback, stochastic basis.

- RFR compounding: backward-looking daily compounding with lockout convention
- SpreadCurve: deterministic IBOR-RFR spread term structure
- IBORProjection: IBOR forward = RFR forward + spread
- Bootstrap spread curve: extract spreads from IBOR swap rates + OIS curve
- StochasticBasis: OU-driven spread for joint (rate, spread) simulation
- IBOR fallback: compounded RFR + ISDA spread adjustment
- 1206 tests, 95% coverage

---

## v0.36.0 — 2026-04-02

Stochastic foundations complete — slices 34-36.

- **Slice 34 — Brownian Motion:** Wiener, correlated BM (Cholesky), Brownian bridge
- **Slice 35 — Jump Processes:** Poisson, compound Poisson, Merton jump-diffusion (with char func), Variance Gamma (with char func + COS pricing cross-check)
- **Slice 36 — Special Processes:** CIR/square-root (Feller, analytical mean), Ornstein-Uhlenbeck (exact simulation), Bessel (via squared Bessel), Gamma subordinator, Inverse Gaussian
- All char funcs verified: φ(0)=1, COS ≈ MC for Merton and VG
- 1187 tests, 95% coverage

---

## v0.34.0 — 2026-04-02

Brownian motion framework — stochastic foundation.

- WienerProcess: 1D standard BM, paths + increments, W(0)=0, Var[W(t)]=t
- CorrelatedBM: d-dimensional via Cholesky, simulated correlation matches input
- BrownianBridge: conditioned BM, exact mean/variance formulas, endpoint pinning
- Independent increments verified, reproducible seeds
- 1151 tests, 95% coverage

---

## v0.33.0 — 2026-04-02

Term loans — completing the product suite.

- TermLoan: amortising floating-rate with credit spread, bullet or amortising schedule
- Cashflow generation with forward rate projection and amortisation
- Weighted average life (WAL): amortising < bullet confirmed
- Discount margin via Brent solver, round-trip validated
- Dual-curve support (separate discount and projection)
- 1130 tests, 95% coverage

---

## v0.32.0 — 2026-04-01

Basket CDS, Gaussian copula, and leveraged CLN.

- Gaussian copula: one-factor model for correlated defaults, MC simulation
- First-to-default spread: higher correlation → lower FTD (clustering)
- Nth-to-default: spreads decreasing in N, 1st-TD = FTD confirmed
- LeveragedCLN: amplified credit exposure, higher leverage → lower PV
- Default clustering validated: high correlation → higher variance in default count
- 1118 tests, 95% coverage

---

## v0.31.0 — 2026-04-01

CDS index and vanilla credit-linked note.

- CDSIndex: equally-weighted portfolio of single-name CDS, PV, flat spread, intrinsic spread
- VanillaCLN: funded credit note with survival-weighted coupons + recovery on default
- CLN credit spread: implied spread from risky vs risk-free price
- Round-trip: index PV = sum of constituents, CLN → risk-free as hazard → 0
- 1104 tests, 95% coverage

---

## v0.30.0 — 2026-04-01

Risky bonds, Z-spread, and asset swap spread.

- RiskyBond: survival-weighted cashflows with recovery on default
- Z-spread: constant spread to risk-free curve via Brent solver, round-trip validated
- Asset swap spread: floating spread that equates bond PV to par
- Zero hazard → risk-free price, higher hazard → lower price, higher recovery → higher price
- ASW ≈ Z-spread for near-par bonds confirmed
- 1092 tests, 95% coverage

---

## v0.29.0 — 2026-04-01

Commodities — new asset class.

- CommodityForwardCurve: forward prices with interpolation, contango/backwardation
- Convenience yield: implied from forward/spot/discount relationship
- CommoditySwap: fixed-for-floating, PV, par price
- Commodity options: Black-76 on the forward (reuses existing Black-76)
- Round-trip validated: swap at par PV=0, put-call parity, quantity scaling
- 1081 tests, 95% coverage

---

## v0.28.0 — 2026-04-01

AAD: tape-based adjoint algorithmic differentiation — the numerics capstone.

- Number class: overloaded float with automatic tape recording
- Tape: operation graph with mark/rewind for MC Greeks pattern
- All arithmetic: +, -, *, /, **, neg with correct derivatives
- Math functions: exp, log, sqrt, norm_cdf, maximum (differentiable)
- Reverse-mode propagation: all Greeks in a single backward pass
- Black-Scholes Greeks via AAD match analytical (delta, vega, rho within 1%)
- MC mark/rewind pattern: accumulate Greeks across paths efficiently
- Translated from CompFinance C++ engine (Savine). References moved to REFERENCES.md.
- 1054 tests, 95% coverage

---

## v0.27.0 — 2026-04-01

ADI 2D PDE solver for Heston stochastic volatility.

- Craig-Sneyd ADI scheme: handles mixed derivatives (ρ*ξ*v cross-term)
- 2D grid in (log-spot, variance) space with bilinear interpolation
- Boundary conditions: call/put at S→0/∞, BS limit at v→0, linear extrapolation at v→∞
- Converges to semi-analytical Heston price (verified via grid refinement)
- Zero vol-of-vol → Black-Scholes, higher v0 → higher price
- 1027 tests, 95% coverage

---

## v0.26.0 — 2026-04-01

COS spectral method for ultra-fast European option pricing.

- COS pricer: Fourier-cosine expansion with O(N) complexity, exponential convergence
- Automatic truncation range via numerical cumulant estimation from char func
- Black-Scholes char func: matches analytical to 1e-6 with N=256
- Heston char func: matches semi-analytical Gauss-Legendre pricer within 2%
- Put-call parity verified, convergence rate exponential in N
- 1018 tests, 95% coverage

---

## v0.25.0 — 2026-04-01

ODE solvers for model dynamics.

- RK4: classical 4th-order Runge-Kutta (fixed step)
- RK45: adaptive Dormand-Prince with automatic step-size control
- BDF: backward differentiation for stiff systems (wraps scipy)
- ODEResult dataclass: t, y, n_evaluations, method
- Registered: get_ode_solver("rk4"), get_ode_solver("rk45"), get_ode_solver("bdf")
- All solvers agree on smooth problems, RK45 more efficient, BDF handles stiff
- 1006 tests, 95% coverage

---

## v0.24.0 — 2026-04-01

Optimization toolkit and calibration refactor.

- Unified minimize(): Nelder-Mead, BFGS, L-BFGS-B, differential evolution, basin hopping
- Least-squares: Levenberg-Marquardt and Trust Region Reflective
- OptimizerResult dataclass: x, fun, iterations, converged, method
- Registry: get_optimizer("nelder_mead"), list_optimizers()
- SABR calibration refactored to use pricebook optimizer (was raw scipy)
- Heston calibration refactored to use pricebook optimizer
- 995 tests, 95% coverage

---

## v0.23.0 — 2026-04-01

Architecture refactor: protocols, unified results, registry.

- Protocols: VolSurface, RootFinder, Integrator, OptionPricer, MCEngine, VolModel (runtime-checkable)
- Unified result types: SolverResult, QuadratureResult, MCResult (existing), TreeResult, PDEResult (new)
- Registry: get_solver, get_integrator, get_tree_european/american, get_pde_pricer, get_mc_pricer
- String-based method lookup for swapping implementations without changing client code
- All vol surface types (FlatVol, VolSurfaceStrike, SwaptionVolSurface, FXVolSurface) satisfy VolSurface protocol
- All tree/MC/PDE methods accessible via registry, cross-validated to agree on prices
- 978 tests, 95% coverage

---

## v0.22.0 — 2026-03-31

Advanced Monte Carlo: LSM American pricing, stratified/importance sampling, MLMC.

- Longstaff-Schwartz: American/Bermudan via polynomial regression on continuation value, matches binomial tree within 3%
- Stratified sampling: lower variance than plain MC, better uniform coverage
- Importance sampling: drift shift for OTM options, likelihood ratio correction
- Multi-level Monte Carlo: telescoping estimator with coarse/fine path coupling
- All techniques agree on European prices, cross-validated against Black-Scholes
- 952 tests, 95% coverage

---

## v0.21.0 — 2026-03-31

SABR and Heston stochastic volatility models.

- SABR: Hagan (2002) approximation for implied vol, pricing via Black-76, calibration via Nelder-Mead
- SABR smile properties: skew from rho, convexity from nu, beta=1 lognormal limit
- Heston: semi-analytical pricing via characteristic function (P1/P2 decomposition)
- Heston calibration via differential evolution (global optimizer)
- Round-trip validated: SABR calibrate→reprice, Heston zero xi→BS, put-call parity, rho/vol effects
- 930 tests, 95% coverage

---

## v0.20.0 — 2026-03-31

Barrier options via finite difference PDE.

- Knock-out barriers: down-and-out, up-and-out, double knock-out
- Knock-in via in-out parity: knock-in = vanilla - knock-out
- Rannacher smoothing: initial implicit steps for stable Greeks near barrier
- Round-trip validated: in-out parity, barrier monotonicity, far barrier ≈ vanilla, double ≤ single
- 904 tests, 95% coverage

---

## v0.19.0 — 2026-03-31

Trinomial trees, Hull-White short-rate model, and rate tree pricing.

- Trinomial tree: Kamrad-Ritchren parameterisation, European + American options
- Faster convergence than CRR binomial verified
- Hull-White one-factor model: analytical ZCB pricing P(t,T) = A(t,T)*exp(-B(t,T)*r)
- Trinomial rate tree calibrated to initial discount curve via alpha fitting
- Tree reprices input ZCBs within 2% (calibration validation)
- European swaption pricing on the rate tree
- Higher vol → higher swaption (vega positive) confirmed
- 891 tests, 95% coverage

---

## v0.18.0 — 2026-03-31

Extended solvers and numerical integration.

- Newton-Raphson: quadratic convergence with analytical derivative
- Secant method: superlinear without derivative
- Halley: cubic convergence with f, f', f''
- ITP (Interpolate-Truncate-Project): optimal worst-case bracketing
- SolverResult dataclass: root, iterations, converged, function_value
- Gauss-Legendre quadrature: exact for polynomials of degree ≤ 2n-1
- Gauss-Laguerre: semi-infinite integrals [0, ∞)
- Gauss-Hermite: Gaussian-weighted integrals (-∞, ∞)
- Adaptive Simpson: automatic refinement with error control
- Round-trip: all solvers agree on roots, quadrature reproduces Black-Scholes via risk-neutral integral
- 860 tests, 95% coverage

---

## v0.17.0 — 2026-03-31

Trade object, portfolio, and scenario risk engine — architectural capstone.

- Trade: wraps any instrument with direction (+1/-1), notional scaling, counterparty metadata
- Portfolio: aggregate PV, PV-by-trade breakdown
- Scenario engine: named perturbations applied to PricingContext
- Standard scenarios: parallel rate shift, pillar bump (DV01 ladder), vol bump, FX spot shock
- ScenarioResult: base PV, scenario PV, P&L
- Round-trip validated: zero shift = zero PnL, up/down opposite sign, linear PnL approximation, DV01 ladder non-zero, vol sensitivity correct sign
- 824 tests, 95% coverage

---

## v0.16.0 — 2026-03-31

Inflation — new asset class with CPI curve, swaps, and linkers.

- CPI curve: forward CPI index levels, breakeven rates, log-linear interpolation, from_breakevens factory
- Zero-coupon inflation swap: PV and par rate, receiver-inflation convention
- Year-on-year inflation swap: periodic CPI ratio payments
- Inflation-linked bond: indexed coupons + principal, dirty price, real yield via Brent
- CPI curve bootstrap: strip from ZC swap rates, reprices all inputs
- Round-trip validated: ZC swap at par, breakeven ≈ par rate, bootstrap repricing, linker real yield
- 801 tests, 95% coverage

---

## v0.15.0 — 2026-03-31

Finite difference PDE solver for European and American options.

- Three schemes: explicit (conditionally stable), implicit (unconditionally stable, 1st order), Crank-Nicolson (unconditionally stable, 2nd order)
- Thomas algorithm for tridiagonal systems
- Log-spot grid with configurable range and resolution
- American options via CN with early exercise check at each step
- Round-trip validated: CN matches Black-Scholes (<0.5%), American FD matches binomial tree (<1%), put-call parity, CN more accurate than implicit
- 784 tests, 95% coverage

---

## v0.14.0 — 2026-03-31

Floating-rate notes and basis swaps — completing the IR product suite.

- FloatingRateNote: dirty/clean price, accrued interest, discount margin via Brent solver
- BasisSwap: float-vs-float with dual projection curves, par spread computation
- FRN at par validation: zero spread on own curve = 100 (flat and steep curves)
- Round-trip validated: DM recovery, spread DV01, basis swap par repricing
- 770 tests, 95% coverage

---

## v0.13.0 — 2026-03-31

FX vanilla options with delta conventions and market vol surface.

- FX option pricing: Garman-Kohlhagen via Black-76, put-call parity verified
- Delta conventions: spot delta, forward delta, premium-adjusted delta
- Strike-from-delta: inverse mapping for all three conventions, round-trip tested
- FX vol surface: ATM/RR25/BF25 market quotes → 3-point smile → interpolated surface
- FXVolSurface: multi-expiry with per-expiry smiles, compatible with vol(expiry, strike) interface
- Round-trip validated: ATM-DNS zero straddle delta, RR/BF recovery, synthetic forward = CIP, all delta conventions round-trip
- 742 tests, 95% coverage

---

## v0.12.0 — 2026-03-30

CRR binomial tree for European and American options.

- CRR binomial tree: u = exp(vol*sqrt(dt)), d = 1/u, risk-neutral probability
- European options via tree: backward induction, converges to Black-Scholes
- American options via tree: early exercise check at each node
- Continuous dividend yield support for both European and American
- Round-trip validated: European convergence O(1/n), American call = European (no divs), American put > European, put-call bounds, Greeks vs analytical
- 692 tests, 95% coverage

---

## v0.11.0 — 2026-03-30

Implied volatility solvers, vol smile, and strike-dependent vol surface.

- Implied vol solver (Black-76): Newton-Raphson with vega, bisection fallback, edge case handling
- Implied vol solver (Bachelier): Newton-Raphson for normal vol, bisection fallback
- VolSmile: strike-dependent vol at a single expiry, cubic spline interpolation, flat wing extrapolation
- VolSurfaceStrike: per-expiry smiles with linear expiry interpolation, compatible with all pricing functions
- Round-trip validated: implied vol recovery across strikes/expiries/models, smile impact on OTM prices, put-call parity with smile, swaption + cap/floor integration with smile surface
- 661 tests, 95% coverage

---

## v0.10.0 — 2026-03-30

Equity forwards, options, and discrete dividends — third asset class.

- Equity forward: continuous dividend yield and discrete dividend pricing, PV
- Equity option (Black-Scholes): European call/put via Black-76 on the forward, spot Greeks (delta, gamma, vega, theta, rho)
- Discrete dividend model: PV of dividends, adjusted forward, piecewise forward with jumps, option pricing with spot adjustment
- Round-trip validated: put-call parity (continuous + discrete), MC matches analytical, dividend-adjusted forward recovery, all Greeks match bump-and-reprice
- 578 tests, 96% coverage

---

## v0.9.0 — 2026-03-30

Monte Carlo engine, GBM paths, and Asian options — first numerical engine.

- Random number generation: pseudo-random (numpy) and quasi-random (Sobol) standard normal generators with seed management
- GBM path generation: single-step and multi-step, antithetic variates, quasi-random support
- MC European pricer: call/put with antithetic variates and control variate variance reduction, cross-checked against Black-76
- Asian options: geometric average analytical (closed-form), arithmetic average MC, fixed and floating strike, geometric average as control variate
- Round-trip validated: European MC within 3σ of Black-76, variance reduction reduces SE, geometric Asian MC matches analytical (~2%), convergence rate 1/√N confirmed
- 516 tests, 96% coverage

---

## v0.8.0 — 2026-03-30

PricingContext, European swaptions, and swaption vol surface.

- PricingContext: bundles valuation date, discount/projection curves, vol surfaces, credit curves, FX spots into one object
- Swaption: European payer/receiver on a vanilla IRS, Black-76 pricing on the forward swap rate
- Swaption pv_ctx: price from a PricingContext with named curves and vol surfaces
- SwaptionVolSurface: 2D expiry×tenor grid with bilinear interpolation, flat extrapolation
- Round-trip validated: payer-receiver parity (ATM + OTM), ATM symmetry, vega/delta bump-and-reprice, PricingContext consistency, vol surface integration
- 441 tests, 96% coverage

---

## v0.7.0 — 2026-03-29

European options, Black-76, and IR caps/floors — first options slice.

- Black-76 model: call/put pricing on a forward, handles zero vol and expiry edge cases
- Bachelier (normal) model: arithmetic Brownian motion, works with negative rates
- Analytical Greeks: delta, gamma, vega, theta with bump-and-reprice cross-checks
- Vol surface: flat vol and vol term structure (strike dimension anticipated, not yet built)
- IR cap/floor: strip of caplets/floorlets priced with Black-76
- Round-trip validated: put-call parity (parametrised), ATM delta ≈ ±0.5, vega maximised ATM, cap-floor parity
- 382 tests, 97% coverage

---

## v0.6.0 — 2026-03-29

FX forwards, swaps, and cross-currency basis — second currency.

- Currency and currency pair with market quoting conventions (EUR, GBP, USD)
- FX forward: covered interest rate parity pricing, forward points, PV
- FX swap: near/far legs, swap points, fair valuation
- Cross-currency basis: implied spread from market forwards, basis curve bootstrap
- Round-trip validated: CIP holds, triangular consistency (EUR/USD + GBP/USD = EUR/GBP), basis curve reprices all forwards
- 324 tests, 98% coverage

---

## v0.5.0 — 2026-03-29

CDS and credit curve — third asset class.

- Survival curve: survival probabilities, hazard rates, default probabilities
- CDS protection leg: discretised integration with mid-point approximation, analytical cross-check
- CDS premium leg: scheduled coupons contingent on survival, accrued-on-default approximation
- CDS instrument: PV, par spread, upfront, risky annuity (RPV01)
- Credit curve bootstrap: strip survival probabilities from CDS par spreads using OIS discount
- CS01: credit spread sensitivity via bump-and-reprice
- Risky bond cross-check: risk-free price minus credit adjustment
- 273 tests, 98% coverage

---

## v0.4.0 — 2026-03-29

FRA, OIS, and dual-curve framework.

- Forward rate agreement (FRA): single-period forward rate contract
- Dual-curve floating leg, swap, and FRA: separate projection and discount curves
- OIS swap: compounded overnight rate with telescoping PV
- OIS bootstrap: strip OIS par rates into a risk-free discount curve
- Dual-curve bootstrap: forward curve from IRS par rates, discounting off OIS
- Round-trip validated: OIS reprices, IRS reprices dual-curve, FRAs consistent, single-curve recovery exact
- 220 tests, 98% coverage

---

## v0.3.0 — 2026-03-28

Fixed-rate bonds and risk sensitivities.

- Fixed-rate bond: dirty/clean price, accrued interest
- Yield to maturity: Brent solver (extracted to shared solvers module)
- Macaulay duration, modified duration, convexity, yield DV01
- Curve-based risk: parallel bump DV01, key rate durations (bump and reprice)
- Round-trip validated: YTM recovery, analytical duration matches bump risk, convexity improves approximation
- 181 tests, 97% coverage

---

## v0.2.0 — 2026-03-28

Interest rate swaps and full yield curve bootstrap.

- Schedule generation: monthly, quarterly, semi-annual, annual frequencies with stub handling and end-of-month rule
- Fixed leg: cashflow generation, present value, annuity factor
- Floating leg: forward rate projection from discount curve, spread support
- Interest rate swap: payer/receiver direction, PV, par rate
- Curve bootstrap: deposits (short end) + swap par rates (long end), Brent root finder
- Round-trip validated: all input instruments reprice, forwards positive, dfs decreasing
- 139 tests, 97% coverage

---

## v0.1.0 — 2026-03-28

Foundation layer: the building blocks for curve construction.

- Day count conventions: ACT/360, ACT/365F, 30/360
- Business day calendar: USD settlement (NYSE/SIFMA), adjustment conventions (following, modified following, preceding, modified preceding)
- Money market deposit: cashflow, discount factor, present value
- Discount curve: built from discount factors, queries for df, zero rate, forward rate
- Interpolation: linear, log-linear, cubic spline, monotone cubic (Hyman filter)
- Round-trip validated: deposits bootstrap into a curve and reprice to zero
- 79 tests, 97% coverage
