# Pricebook Architecture

**486 modules | 20 sub-packages | 9 dependency layers | 0 circular deps | ~8,800 tests**

---

## Package Structure

```
pricebook/
├── core/              29 modules   Layer 0   Curves, scheduling, trade, serialisation
├── db/                 2 modules   Layer 0   SQLite database
├── numerical/         12 modules   Layer 0   Self-contained numerical methods
├── pe/                 4 modules   Layer 0   Private equity (LBO, DCF, performance)
├── statistics/        12 modules   Layer 0   GARCH, Kalman, copulas, regression
├── ts/                 7 modules   Layer 0   Time series (numpy-backed, no pandas)
├── viz/               13 modules   Layer 0   Matplotlib visualisation layer
├── pricing/            9 modules   Layer 1   Pricing engine, market data, codecs
├── regulatory/        22 modules   Layer 1   Basel III/IV capital, FRTB, SA-CCR, SIMM
├── curves/            18 modules   Layer 2   Bootstrap, curve building, AAD
├── models/            52 modules   Layer 3   MC, PDE, trees, Black-76, Hull-White
├── credit/            56 modules   Layer 4   CDS, CLN, CLO, loans, recovery, distressed
├── risk/              27 modules   Layer 4   Greeks, hedging, XVA, VaR, stress
├── fixed_income/      66 modules   Layer 5   Bonds, swaps, FRA, repos, inflation
├── options/           46 modules   Layer 6   Options, vol surfaces, SABR, Heston
├── commodity/         15 modules   Layer 7   Commodity instruments
├── equity/            22 modules   Layer 7   TRS, dividends, variance swaps
├── fx/                19 modules   Layer 7   FX forwards, swaps, exotics
├── structured/         9 modules   Layer 7   CMS, CMO, structured notes
├── desks/             46 modules   Layer 8   16+ trading desks (9-component protocol)
└── registry.py                     Top       Solver/pricer registry (wiring)
```

---

## Dependency Order (bottom-up, verified acyclic)

```
Layer 0: core, db, numerical, pe, statistics, ts, viz      ── foundations (no deps)
Layer 1: pricing, regulatory                                ── infrastructure
Layer 2: curves                                             ── curve construction
Layer 3: models                                             ── pricing engines
Layer 4: credit, risk                                       ── credit instruments + risk
Layer 5: fixed_income                                       ── bonds, swaps, repos
Layer 6: options                                            ── options + vol surfaces
Layer 7: commodity, equity, fx, structured                  ── asset classes
Layer 8: desks                                              ── trading desks (top)
```

Each layer depends only on layers below it. No circular dependencies.

---

## Data Flow

```
Market Data (quotes, fixings, indices)
    │
    ▼
Curves (discount, survival, IBOR, vol)         ── curves/
    │
    ▼
PricingContext (bundles all curves)             ── core/pricing_context
    │
    ▼
Models (Black-76, Heston, MC, PDE, trees)      ── models/
    │
    ▼
Instruments (83+ types, all implement pv_ctx)  ── fixed_income/, credit/, options/, fx/, equity/, commodity/
    │
    ├──▶ Risk (bump-and-reprice, greeks, XVA)  ── risk/
    │
    ▼
Trade → Portfolio → Book → Desk               ── core/trade, desks/
    │
    ├──▶ Daily P&L (sequential attribution)    ── desks/*_daily_pnl
    ├──▶ Regulatory Capital (FRTB, SA-CCR)     ── regulatory/
    ├──▶ Stress Testing (CCAR, reverse stress) ── regulatory/ccar, regulatory/reverse_stress
    │
    ▼
Viz (PlotBuilder, dashboards, charts)          ── viz/
TimeSeries (replay, drawdown, rolling Sharpe)  ── ts/
Database (trades, P&L history, snapshots)      ── db/
```

---

## Core Infrastructure (`core/`)

| Module | Key Classes | Role |
|--------|------------|------|
| `discount_curve` | `DiscountCurve` | `df()`, `zero_rate()`, `forward_rate()`, `bumped()` |
| `survival_curve` | `SurvivalCurve` | `survival()`, `hazard_rate()`, `default_prob()` |
| `pricing_context` | `PricingContext` | Bundles all curves for a valuation date |
| `trade` | `Trade`, `Portfolio` | Position management |
| `book` | `Book`, `Desk`, `BookLimits` | Limits, aggregation |
| `schedule` | `Frequency`, `generate_schedule` | Payment schedule generation |
| `day_count` | `DayCountConvention`, `year_fraction` | ACT/360, ACT/365, 30/360 |
| `calendar` | `Calendar`, `BusinessDayConvention` | Holiday calendars |
| `currency` | `Currency`, `CurrencyPair` | ISO 4217 |
| `interpolation` | `InterpolationMethod` | LOG_LINEAR, CUBIC, MONOTONE_CONVEX |
| `solvers` | `brentq`, `newton` | Root finding |
| `serialisable` | `@serialisable` decorator | Auto `to_dict()`/`from_dict()` |
| `serialization` | `to_json()`, `from_json()` | Full instrument serialisation |

---

## Models (`models/`)

### Pricing Model Abstraction

```python
class IROptionModel(Protocol):
    def price_ir_option(self, forward, strike, annuity, T, option_type) -> float
    def greeks_ir_option(self, forward, strike, annuity, T, option_type) -> Greeks

class EquityOptionModel(Protocol):
    def price_european(self, spot, strike, rate, T, option_type, div_yield) -> float
```

| Model | Module | Wraps |
|-------|--------|-------|
| `Black76Model` | `models/models` | Lognormal vol → `black76_price()` |
| `BachelierModel` | `models/models` | Normal vol → `bachelier_price()` |
| `SABRModel` | `models/models` | SABR smile → implied vol + Black-76 |
| `HullWhiteModel` | `models/models` | Mean-reversion → Rebonato approximation |
| `BSModel` | `models/models` | Black-Scholes → `equity_option_price()` |
| `HestonModel` | `models/models` | Stochastic vol → `heston_price()` |
| `MCEquityModel` | `models/models` | Any ProcessSpec → MCEngine |

### MC Engine

| Module | Key Classes |
|--------|------------|
| `mc_engine` | `MCEngine`, `TimeGrid`, `ProcessSpec`, `MCResult` |
| `mc_processes` | 16 processes: GBM, Heston, SABR, HW, LMM, Bates, VG, CEV, G2++ |
| `mc_payoffs` | 13 payoffs: European, American (LSM), Asian, barrier, cliquet, autocall |
| `mc_variance_reduction` | Antithetic, control variate, importance sampling |

---

## Instruments

### Fixed Income (`fixed_income/` — 66 modules)

Bonds, swaps, FRA, FRN, repos, inflation, deposits, OIS, basis swaps, xccy, callable bonds, treasury.

### Credit (`credit/` — 56 modules)

CDS, CLN, CLO, term loans (floored, PIK), covenants, recovery models, distressed (DIP, fulcrum, exchange offers), unitranche, waterfall, fund participation.

### Options (`options/` — 46 modules)

Swaptions, caps/floors, equity options, barriers, Asians, autocallables, cliquets, TARFs, convertibles, SABR, Heston, local vol, implied vol, vol surfaces.

### FX (`fx/` — 19 modules)

FX forwards, swaps, NDFs, barriers, exotics, structured (TARF, DCD), hedging strategies, PRDC.

### Equity (`equity/` — 22 modules)

TRS, equity forwards, dividends, variance swaps, RV analysis, structured products.

### Commodity (`commodity/` — 15 modules)

Forward curves, storage, swing options, spreads, seasonal patterns, exotic commodities.

---

## Trading Desks (`desks/` — 46 modules)

### 9-Component Protocol

Every desk implements:

| # | Component | Pattern |
|---|-----------|---------|
| 1 | `RiskMetrics` | Per-position Greeks (PV, DV01, CS01, delta, gamma, vega) |
| 2 | `Book` + `BookEntry` | Position management, aggregation, grouping |
| 3 | `CarryDecomposition` | Coupon, funding, roll-down, net carry |
| 4 | `DailyPnL` | Rate/vol/credit/theta/carry/unexplained attribution |
| 5 | `Dashboard` | Morning summary: positions, risk totals, top exposures |
| 6 | `StressResult` | Parametric scenarios (rates, spreads, vol shocks) |
| 7 | `CapitalResult` | SA-CCR EAD, RWA, FRTB charges, SIMM IM |
| 8 | `HedgeRecommendation` | Limit breach detection + action suggestions |
| 9 | `Lifecycle` | Event tracking: maturities, coupons, triggers, alerts |

16+ desks: swap, bond, CDS, CLN, swaption, asset swap, convertible bond, structured credit, TRS, repo, FX, equity, commodity, inflation, futures, PE.

---

## Risk (`risk/` — 27 modules)

| Module | Role |
|--------|------|
| `greeks` | `Greeks` dataclass, `bump_greeks()` |
| `var` | Historical, parametric, Monte Carlo VaR |
| `xva` | CVA, DVA, FVA, KVA, MVA |
| `scenario` | `parallel_shift()`, `pillar_bump()`, `vol_bump()` |
| `pnl_explain` | `PnLResult` decomposition |
| `cross_asset_greeks` | Multi-asset Greek attribution |
| `cross_gamma_hedging` | Correlation-aware position sizing |
| `simm` | ISDA SIMM initial margin |
| `backtest` | `run_backtest()`, walk-forward, deflated Sharpe |
| `portfolio_construction` | Mean-variance, Black-Litterman, risk parity |
| `factor_model` | Factor attribution, covariance, timing |

---

## Regulatory (`regulatory/` — 22 modules)

| Module | Framework |
|--------|-----------|
| `credit_rwa` | SA-CR, F-IRB, A-IRB, specialised lending slotting |
| `market_risk_sa` | FRTB-SA: SbM (delta/vega/curvature) + DRC + RRAO |
| `market_risk_ima` | FRTB-IMA: liquidity-adjusted ES, NMRF, MC DRC, PLA |
| `ima_bridge` | Desk sensitivities → IMA risk factors |
| `counterparty` | SA-CCR: replacement cost, PFE, add-on |
| `var_es` | Parametric, historical, MC VaR/ES, backtesting |
| `irc` | Incremental Risk Charge (MC rating migration) |
| `securitization` | SEC-SA, SEC-IRBA, ERBA |
| `capital_framework` | Output floor, leverage ratio, G-SIB, TLAC |
| `total_capital` | Total RWA aggregation across all risk types |
| `ccar` | 9-quarter capital projection (baseline/adverse/severely adverse) |
| `reverse_stress` | Minimum-severity scenario finder (scipy optimisation) |
| `capital_allocation` | Euler allocation, RORC, hurdle rates, capital limits |
| `liquidity` | Portfolio-wide LCR/NSFR with product classification |
| `operational_risk` | SMA: BI/BIC/ILM, bucket classification |
| `repo_capital` | SFT EAD, LCR outflow, NSFR RSF |

---

## Visualisation (`viz/` — 13 modules)

| Function | Chart Type |
|----------|-----------|
| `greeks_profile()` | Multi-panel Greeks vs spot |
| `pnl_waterfall()` | Waterfall/bridge P&L attribution |
| `stress_comparison()` | Grouped/stacked scenario bars |
| `greeks_surface()` | 2D contour (strike × expiry) |
| `greeks_evolution()` | Greeks vs time-to-expiry |
| `vega_ladder()` | Vega by expiry bucket |
| `tenor_bucketing()` | DV01 by tenor bucket |
| `hedge_pnl_tracking()` | Position vs hedge cumulative P&L |
| `rolling_correlation()` | Multi-line rolling correlation |
| `football_field()` | Valuation range (DCF, comps, WACC) |
| `j_curve()` | PE fund TVPI over time |
| `sensitivity_grid()` | 2D parameter sensitivity heatmap |
| `pnl_distribution()` | P&L histogram with VaR/CVaR |

---

## Numerical Methods (`numerical/` — 12 modules)

Self-contained toolkit — scipy/numpy are the backend, users import from `pricebook.numerical`.

| Module | Methods |
|--------|---------|
| `_distributions` | Normal, StudentT, LogNormal, Uniform, Exponential |
| `_linalg` | expm, logm, QR, Cholesky, LU, GMRES, Sylvester, Lyapunov |
| `_ode` | Euler, RK4, RK45, BDF, Adams |
| `_optimize` | minimize, LP, QP, interior-point, proximal (ISTA/FISTA) |
| `_quadrature` | Gauss-Jacobi, tanh-sinh, Clenshaw-Curtis |
| `_interpolation` | 2D bilinear, bicubic, RBF |
| `_rootfinding` | bisection, find_root |
| `_mc` | QE Heston, antithetic, multilevel MC |
| `_pde` | Hundsdorfer-Verwer ADI, 2D PSOR, operator splitting |
| `_trees` | tree Greeks, 2D binomial |
| `_fourier` | fractional FFT, Hilbert, wavelet, CharacteristicFunction |
| `_distributions_theory` | Schwartz distributions, Sobolev norms |

---

## Private Equity (`pe/` — 4 modules)

| Module | Key Classes |
|--------|------------|
| `lbo` | `LBOModel` — sources & uses, debt schedule, FCF, exit analysis, sensitivity |
| `dcf` | `DCFModel` — WACC, terminal value, EV bridge, scenario analysis, football field |
| `pe_performance` | Kaplan-Schoar PME, direct alpha, vintage cohort, commitment pacing, GP economics |
| `pe_desk` | 9-component PE desk protocol |

---

## Statistics (`statistics/` — 12 modules)

| Module | Methods |
|--------|---------|
| `garch` | GARCH(1,1), EGARCH, EWMA, realized vol, GARCH VaR |
| `kalman` | Kalman filter, RTS smoother, dynamic beta, trend extraction |
| `clustering` | K-means, silhouette, hierarchical, HMM regime switching |
| `regression` | OLS, Ridge, Lasso, Elastic Net, quantile, robust |
| `copulas` | Gaussian, Student-T, Clayton, Frank, Gumbel |
| `distribution_fit` | MLE fitting, KS test, Anderson-Darling, QQ |
| `statistics` | ACF, PACF, Ljung-Box, ADF, Durbin-Watson |

---

## Database (`db/` — 2 modules)

`PricebookDB`: 7 system tables (entities, ratings, trades, market_snapshots, pricing_results, pnl_history, kv_store) + custom tables + CSV export. `SQLiteBackend` (pluggable).

---

## C++ Port Roadmap

### Phase 1: Curves + Bootstrap
`core/discount_curve` → `DiscountCurve.h`, `curves/bootstrap` → `Bootstrap.h`, `core/solvers` → `Solvers.h`

### Phase 2: Core Instruments
`fixed_income/swap` → `InterestRateSwap.h`, `fixed_income/bond` → `FixedRateBond.h`, `credit/cds` → `CDS.h`

### Phase 3: MC Engine
`models/mc_engine` → `MCEngine.h` (parallelisable), `models/mc_processes` → `Processes.h`

### Phase 4: Risk + Numerics
`risk/` → `RiskEngine.h`, `models/finite_difference` → `FDSolver.h`

### Stays Python
`viz/`, `ts/`, `db/`, `regulatory/`, `desks/`, notebooks, API
