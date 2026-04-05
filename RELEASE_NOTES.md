# Release Notes

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
