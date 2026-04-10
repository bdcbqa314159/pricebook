# Release Notes

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
