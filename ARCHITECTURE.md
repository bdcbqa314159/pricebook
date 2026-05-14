# Pricebook Architecture

**458 files | 126,648 lines | 8-layer pricing engine | ~8,000 tests**

## Data Flow

```
Market Data (quotes, fixings, indices)
    |
    v
Curves (discount, survival, IBOR, vol surfaces)
    |
    v
PricingContext (bundles all curves for a valuation date)
    |
    v
Instruments (83 types, all implement pv_ctx)
    |
    +---> Risk (bump-and-reprice, 29 desks, 9-component protocol)
    |
    v
Trade --> Portfolio --> Book (limits) --> Desk
    |
    v
Daily P&L (sequential attribution: rates, vol, credit, theta, FX, unexplained)
    |
    v
Viz (PlotBuilder, dashboards) + TimeSeries (replay, drawdown, rolling Sharpe)
    |
    v
Database (PricebookDB: trades, snapshots, P&L history, entities, ratings)
```

---

## Layer 1: Market Data

| File | Class | Role |
|------|-------|------|
| `market_data.py` | `Quote`, `QuoteType`, `MarketDataSnapshot` | Raw market observations (deposit rates, swap rates, CDS spreads, vol points, FX spots) |
| `rate_index.py` | `RateIndex` | Benchmark definitions (SOFR, ESTR, SONIA, EURIBOR, TIBOR) with compounding, fixing lag, payment delay |
| `fixings.py` | `FixingsStore` | File-backed daily fixing storage for retroactive floating leg pricing |
| `benchmark_bonds.py` | `BenchmarkUniverse` | Sovereign bond snapshots (UST, Bund, Gilt, JGB, OAT, BTP) with correct conventions |

---

## Layer 2: Curves

### Discount Curves

| File | Class/Function | Role |
|------|----------------|------|
| `discount_curve.py` | `DiscountCurve` | Core: `df(date)`, `zero_rate()`, `forward_rate()`, `bumped()`, `bumped_at()` |
| `bootstrap.py` | `bootstrap()` | Sequential bootstrapping from deposits + FRAs + futures + swaps |
| `bond_desk.py` | `fit_curve_from_bonds()` | Bootstrap discount curve from bond prices |
| `sofr_curve.py` | `build_sofr_curve()`, `build_estr_curve()`, `build_sonia_curve()` | RFR-specific curve builders |
| `repo_curve.py` | `RepoCurve`, `build_repo_curve()` | Repo term structure from market quotes |
| `funding_curve.py` | `FundingCurve` | CSA-specific discounting |
| `nelson_siegel.py` | `nelson_siegel()`, `svensson()` | Parametric yield curve fitting |

### Credit Curves

| File | Class | Role |
|------|-------|------|
| `survival_curve.py` | `SurvivalCurve` | `survival(date)`, `hazard_rate()`, `default_prob()`, `bumped()` |
| `hazard_term_structure.py` | `HazardTermStructure` | Extended hazard rate models |
| `recovery_surface.py` | `RecoverySurface` | 2D (seniority x tenor) recovery rate surface |

### Projection Curves

| File | Class | Role |
|------|-------|------|
| `ibor_curve.py` | `IBORCurve`, `IBORConventions` | IBOR projection with tenor-specific conventions (EURIBOR_3M, TIBOR_3M) |
| `multi_currency_curves.py` | `MultiCurrencyCurveSet` | OIS + IBOR + tenor basis + XCCY basis per currency |
| `xccy_basis.py` | `bootstrap_basis_curve()` | XCCY basis-adjusted curves from FX forwards |
| `cra_curve.py` | `CRACurve` | Credit-risk-adjusted curve |

### Volatility Surfaces

| File | Class | Role |
|------|-------|------|
| `vol_surface.py` | `FlatVol`, `VolTermStructure`, `VolSurface` | `vol(expiry, strike)`, `bumped()` |
| `sabr.py` | `SABRParams`, `sabr_vol()` | SABR smile calibration |
| `heston.py` | `HestonParams` | Heston stochastic vol |
| `local_vol.py` | `LocalVolSurface` | Dupire local vol |
| `ir_vol_surface.py` | `IRVolSurface` | Swaption vol cube |
| `fx_vol_surface.py` | `FXVolSurface` | FX delta-space vol surface |

### Interpolation

| File | Function | Role |
|------|----------|------|
| `interpolation.py` | `InterpolationMethod` enum | LOG_LINEAR, LINEAR, CUBIC, HERMITE, MONOTONE_CONVEX |

---

## Layer 3: PricingContext

**File:** `pricing_context.py`

```python
PricingContext(
    valuation_date,
    discount_curve,                    # single-ccy shortcut
    discount_curves: dict[str, DC],    # per-currency
    projection_curves: dict[str, DC],  # named (e.g., "USD.3M")
    vol_surfaces: dict[str, VS],       # named (e.g., "ir", "fx", "eq")
    credit_curves: dict[str, SC],      # per-name survival curves
    fx_spots: dict[tuple, float],      # (base, quote) -> rate
    inflation_curves: dict[str, obj],  # CPI curves
    repo_curves: dict[str, DC],        # funding/repo
    reporting_currency: str = "USD",
)
```

**Accessors:** `get_discount_curve(ccy)`, `get_projection_curve(name)`, `get_vol_surface(name)`, `get_credit_curve(name)`, `fx_rate(from, to)`, `replace(**kwargs)`

**Factory:** `PricingContext.simple(date, rate, vol, hazard)` for quick testing.

---

## Layer 3.5: Models

**File:** `models.py`

The model layer sits between PricingContext and instruments. It separates **what** to price (instrument) from **how** to price (model). Instruments call `model.price_ir_option()` or `model.greeks_ir_option()` instead of hardcoding Black-76.

### Two Protocols

```python
class IROptionModel(Protocol):
    def price_ir_option(self, forward, strike, annuity, T, option_type) -> float: ...
    def greeks_ir_option(self, forward, strike, annuity, T, option_type) -> Greeks: ...

class EquityOptionModel(Protocol):
    def price_european(self, spot, strike, rate, T, option_type, div_yield) -> float: ...
    def greeks_european(self, spot, strike, rate, T, option_type, div_yield) -> Greeks: ...
```

### IR Models

| Model | Params | Wraps | Greeks |
|-------|--------|-------|--------|
| `Black76Model(vol)` | Lognormal vol | `black76_price()` | Analytical |
| `BachelierModel(vol_normal)` | Normal vol | `bachelier_price()` | Analytical |
| `SABRModel(SABRParams)` | alpha, beta, rho, nu | `sabr_implied_vol()` + Black-76 | Composite |
| `HullWhiteModel(hw)` | HullWhite object | Rebonato approximation | Bump-and-reprice |

### Equity Models

| Model | Params | Wraps | Greeks |
|-------|--------|-------|--------|
| `BSModel(vol)` | Lognormal vol | `equity_option_price()` | Analytical |
| `HestonModel(HestonParams)` | v0, kappa, theta, xi, rho | `heston_price()` | Bump-and-reprice |
| `MCEquityModel(process)` | Any ProcessSpec | MCEngine | Bump-and-reprice |

### Usage

```python
from pricebook.models import Black76Model, BachelierModel

swaption.price(Black76Model(vol=0.20), curve)
swaption.greeks(BachelierModel(vol_normal=0.005), curve)
capfloor.price(SABRModel(SABRParams(...)), curve)
```

---

## Layer 4: Instruments

All modern instruments implement `pv_ctx(ctx: PricingContext) -> float`.

### Rates

| File | Class | Key Methods |
|------|-------|-------------|
| `swap.py` | `InterestRateSwap` | `pv()`, `par_rate()`, `annuity()`, `dv01()` |
| `fra.py` | `FRA` | `pv()`, `forward_rate()` |
| `ois.py` | `OISSwap` | `pv()`, `par_rate()` |
| `basis_swap.py` | `BasisSwap` | `pv()`, `par_spread()` |
| `capfloor.py` | `CapFloor` | `pv()`, `cap_pv()`, `floor_pv()` |
| `swaption.py` | `Swaption` | `pv_ctx()`, `delta()`, `gamma()`, `vega()` |
| `deposit.py` | `Deposit` | `discount_factor()`, `zero_rate()` |
| `cms.py` | `CMSLeg` | `pv()` with convexity adjustment |

### Bonds

| File | Class | Key Methods |
|------|-------|-------------|
| `bond.py` | `FixedRateBond` | `dirty_price()`, `clean_price()`, `ytm()`, `duration()`, `convexity()` |
| `frn.py` | `FloatingRateNote` | `dirty_price()`, `discount_margin()` |
| `callable_bond.py` | `callable_bond_price()` | Hull-White trinomial tree, OAS |
| `convertible_bond.py` | `ConvertibleBond` | MC + LSM, Tsiveriotis-Fernandes credit adjustment |
| `tbill.py` | `TreasuryBill` | `price()`, `ytm()` |

### Credit

| File | Class | Key Methods |
|------|-------|-------------|
| `cds.py` | `CDS` | `pv()`, `par_spread()`, `rpv01()`, `cs01()` |
| `cln.py` | `CreditLinkedNote` | `pv()`, `greeks()` |
| `basket_cds.py` | `ftd_basket_spread()`, `ntd_spread()`, `bespoke_tranche()` | Gaussian copula |
| `trs.py` | `TotalReturnSwap` | `pv()`, carry, funding |
| `risky_bond.py` | `RiskyBond` | `z_spread()`, `credit_duration()` |

### FX

| File | Class | Key Methods |
|------|-------|-------------|
| `fx_forward.py` | `FXForward` | `pv()`, `forward_rate()`, `fx_delta()` |
| `fx_option.py` | `fx_option_price()` | Black-76 FX option |
| `fx_swap.py` | `FXSwap` | `pv()`, `fair_swap_points()` |
| `ndf.py` | `NDF` | `pv()`, `forward_rate()` |
| `fx_exotic.py` | Touch, lookback, Asian, range accrual, accumulator | MC-based exotics |
| `fx_hedging.py` | Window barrier, fader, participating fwd, seagull, ratio fwd, KI reverse | Hedging structures |
| `fx_structured.py` | TARF, autocallable, DCD, pivot | Structured FX |
| `prdc.py` | `prdc_price()`, `callable_prdc()` | 3-factor MC (dom rate + for rate + FX) |

### Equity

| File | Class | Key Methods |
|------|-------|-------------|
| `equity_option.py` | `equity_option_price()` | Black-Scholes |
| `barrier_option.py` | `BarrierOption` | MC with bridge correction |
| `asian_option.py` | `AsianOption` | Arithmetic + geometric |
| `autocallable.py` | `Autocallable` | MC with memory coupons |
| `equity_exotic_extended.py` | Forward-start, chooser, quanto, Himalaya, accumulator, dividends | Extended exotics |
| `variance_swap.py` | `VarianceSwap` | Replication via log contract |
| `convertible_bond.py` | `ConvertibleBond` | Equity-credit hybrid |

### Commodity

| File | Class | Key Methods |
|------|-------|-------------|
| `commodity_swing.py` | Swing option | LSM with exercise constraints |
| `commodity_spreads.py` | Crack, spark, calendar spreads | MC simulation |
| `commodity_storage.py` | Storage valuation | Dynamic programming |

### Structured

| File | Class | Key Methods |
|------|-------|-------------|
| `structured_notes.py` | Capital-protected, dual digital, bonus cert, participation note | Black-76 based |
| `guaranteed_note.py` | `GuaranteedNote` | Joint default copula |
| `rates_structured.py` | CMS range accrual, callable step-up, inflation RA, ZC swaption, inverse floater, capped floater | MC rates |

---

## Layer 5: Risk

### Bump-and-Reprice Framework

```python
# Pattern used everywhere:
base_pv = instrument.pv_ctx(ctx)
bumped_ctx = ctx.replace(discount_curve=ctx.discount_curve.bumped(+0.0001))
dv01 = instrument.pv_ctx(bumped_ctx) - base_pv
```

**Files:** `risk.py`, `greeks.py`, `cross_asset_greeks.py`, `scenario.py`, `var.py`

### 9-Component Desk Protocol

Every trading desk implements:

| # | Component | Pattern |
|---|-----------|---------|
| 1 | `RiskMetrics` | Per-position Greeks (PV, DV01, CS01, delta, gamma, vega) |
| 2 | `Book` + `BookEntry` | Position management, aggregation, grouping |
| 3 | `CarryDecomposition` | Coupon income, funding cost, roll-down, net carry |
| 4 | `DailyPnL` | Attribution: rate/vol/credit/theta/carry/unexplained |
| 5 | `Dashboard` | Morning summary: positions, risk totals, top exposures |
| 6 | `StressResult` | Parametric scenarios (rates +/-100bp, spreads, vol shocks) |
| 7 | `CapitalResult` | SA-CCR EAD, RWA, FRTB charges, SIMM IM |
| 8 | `HedgeRecommendation` | Limit breach detection + action suggestions |
| 9 | `Lifecycle` | Event tracking: maturities, coupons, triggers, alerts |

### Desk Inventory (29 files)

| Desk | File | Full Protocol? |
|------|------|:---:|
| Swap | `swap_desk.py` | Y |
| Bond | `bond_trading_desk.py` | Y |
| CDS | `cds_desk.py` | Y |
| CLN | `cln_desk.py` | Y |
| Swaption | `swaption_trading_desk.py` | Y |
| Asset Swap | `asset_swap_desk.py` | Y |
| Convertible Bond | `convertible_bond_desk.py` | Y |
| Structured Credit | `structured_credit_desk.py` | Y |
| TRS | `trs_desk.py` | partial |
| Repo | `repo_desk.py` | partial |
| FX | `fx_desk.py` | partial |
| Equity | `equity_desk.py` | partial |
| Commodity | `commodity_desk.py` | partial |
| Inflation | `inflation_desk.py` | partial |
| Futures | `futures_desk.py` | partial |
| Bond Desk (analytics) | `bond_desk.py` | analytics only |
| Callable Bond | `callable_bond_desk.py` | analytics only |
| CVA | `cva_desk.py` | analytics only |
| Vol | `vol_desk.py` | analytics only |
| FX Vol | `fx_vol_desk.py` | analytics only |
| Equity Vol | `equity_vol_desk.py` | analytics only |

---

## Layer 6: Portfolio

```
Trade(instrument, direction=+1/-1, notional_scale, trade_id)
    |
    v
Portfolio([trades], name)       -- pv(ctx) = sum(trade.pv(ctx))
    |
    v
Book(name, limits: BookLimits)  -- add(trade), check_limits(ctx)
    |
    v
Desk(name, books: dict)         -- aggregate across books
```

**Files:** `trade.py`, `book.py`

**BookLimits:** `max_dv01`, `max_notional`, `tenor_limits`, `max_concentration`

**Position aggregation:** By `instrument_type x tenor_bucket` (12 standard buckets: <=3M through 30Y+)

---

## Layer 7: P&L Attribution

**File:** `daily_pnl.py`

```python
compute_daily_pnl(book, prior_ctx, current_ctx, new_trades, amendments)
    -> DailyPnL(market_move_pnl, new_trade_pnl, amendment_pnl, total_pnl)

attribute_pnl(book, prior_ctx, current_ctx)
    -> BookAttribution (per-trade: rate_pnl, vol_pnl, credit_pnl, theta_pnl, fx_pnl, unexplained)
```

**Sequential bumping order:**
1. Rates: bump discount + projection curves
2. Vol: bump vol surfaces
3. Credit: bump survival curves
4. Theta: advance valuation date
5. FX: bump spot rates
6. Unexplained: total - sum(attributed)

**Desk-specific:** `bond_daily_pnl.py`, `equity_daily_pnl.py`, `fx_daily_pnl.py`, `commodity_daily_pnl.py`

---

## Layer 8: Visualization & Time Series

### viz/ — Plot Abstraction

| File | What |
|------|------|
| `_backend.py` | `apply_theme()` context manager, `create_figure(n_panels)` |
| `_theme.py` | `PricebookTheme`, `LIGHT`/`DARK`, `configure_theme()` |
| `_seaborn.py` | `correlation_heatmap()`, `pnl_distribution()`, `greeks_profile()`, `sensitivity_grid()` |
| `_builder.py` | `PlotBuilder` fluent API: `.payoff().greeks().sensitivity().figure()` |
| `_dispatch.py` | `plot(instrument, curve)` auto-detect dashboard |

### ts/ — Time Series Abstraction

| File | What |
|------|------|
| `_core.py` | `TimeSeries` class (numpy-backed, no pandas) |
| `_returns.py` | `simple_returns()`, `log_returns()`, `period_returns()` |
| `_stats.py` | `sharpe()`, `sortino()`, `max_drawdown()`, `performance()` |
| `_rolling.py` | `rolling_sharpe()`, `rolling_vol()`, `rolling_beta()` |
| `_io.py` | `from_db()`, `from_csv()`, `greeks_from_db()` |
| `_replay.py` | `replay()`, `drawdown_analysis()`, `rolling_performance()` |
| `_replay_viz.py` | `plot_dashboard()`, `plot_equity_curve()`, `plot_drawdowns()` |

---

## Cross-Cutting Infrastructure

### MC Engine (22 files)

| File | What |
|------|------|
| `mc_engine.py` | `MCEngine`, `TimeGrid`, `ProcessSpec`, `MCResult` |
| `mc_processes.py` | 16 processes: GBM, Heston, SABR, HW, LMM, Bates, VG, CEV, G2++ |
| `mc_payoffs.py` | 13 payoffs: European, American (LSM), Asian, barrier, cliquet, autocall, swing |
| `mc_variance_reduction.py` | Antithetic, control variate, importance sampling |
| `mc_greeks_engine.py` | Pathwise + finite difference Greeks |
| `mc_migrate.py` | `gbm_paths()`, `ou_paths()`, `cir_paths()` — path generation helpers |

### Numerical Methods

| File | What |
|------|------|
| `solvers.py` | `brentq()`, `newton()` |
| `binomial_tree.py` | Cox-Ross-Rubinstein, Jarrow-Rudd, Leisen-Reimer |
| `trinomial_tree.py` | Hull-White trinomial |
| `finite_difference.py` | Crank-Nicolson, implicit, explicit |
| `cos_method.py` | COS Fourier pricing |
| `fft_pricing.py` | FFT option pricing |
| `quadrature.py` | Gauss-Legendre, Gauss-Hermite |

### Database

| File | What |
|------|------|
| `db.py` | `PricebookDB`: 7 system tables (entities, ratings, trades, market_snapshots, pricing_results, pnl_history, kv_store) + custom tables + CSV |
| `db_backend.py` | `SQLiteBackend` (pluggable for DuckDB/PostgreSQL) |

### AAD (Algorithmic Differentiation)

| File | What |
|------|------|
| `aad.py` | Tape-based forward/reverse mode AD |
| `aad_curves.py` | AD-enabled discount curves |
| `aad_pricing.py` | AD-enabled pricing for O(1) Greeks |

### XVA

| File | What |
|------|------|
| `xva.py` | CVA, DVA, FVA, KVA, MVA |
| `cln_xva.py`, `trs_xva.py`, `repo_xva.py` | Product-specific XVA |

### Regulatory (17 files in `regulatory/`)

Market risk (SA + IMA), credit RWA, securitization, SIMM, liquidity, counterparty risk, IRC, IRRBB stress testing.

### Serialization

| File | What |
|------|------|
| `serialisable.py` | `Serialisable` mixin: auto `to_dict()`/`from_dict()` from `_SERIAL_FIELDS` |
| `serialization.py` | `to_json()`, `from_json()`, instrument registry |

---

## C++ Port Roadmap

### Phase 1: Curves + Bootstrap (direct port, perf critical)

```
discount_curve.py    -> DiscountCurve.h/.cpp
survival_curve.py    -> SurvivalCurve.h/.cpp
interpolation.py     -> Interpolation.h (template)
day_count.py         -> DayCount.h (enum + functions)
calendar.py          -> Calendar.h/.cpp
schedule.py          -> Schedule.h/.cpp
bootstrap.py         -> Bootstrap.h/.cpp
solvers.py           -> Solvers.h (header-only)
```

### Phase 2: Core Instruments

```
swap.py              -> InterestRateSwap.h/.cpp
bond.py              -> FixedRateBond.h/.cpp
cds.py               -> CDS.h/.cpp
fra.py               -> FRA.h/.cpp
swaption.py          -> Swaption.h/.cpp
capfloor.py          -> CapFloor.h/.cpp
fixed_leg.py         -> FixedLeg.h/.cpp
floating_leg.py      -> FloatingLeg.h/.cpp
```

### Phase 3: MC Engine (biggest perf win)

```
mc_engine.py         -> MCEngine.h/.cpp (parallelizable)
mc_processes.py      -> Processes.h/.cpp (GBM, Heston, HW)
mc_payoffs.py        -> Payoffs.h/.cpp
heston.py            -> Heston.h/.cpp
hull_white.py        -> HullWhite.h/.cpp
```

### Phase 4: Risk + Numerics

```
risk.py              -> RiskEngine.h/.cpp
finite_difference.py -> FDSolver.h/.cpp
trinomial_tree.py    -> TrinomialTree.h/.cpp
```

### Stays Python

- `viz/`, `ts/` — visualization and time series (matplotlib/numpy ecosystem)
- `db.py`, `db_backend.py` — database (SQLite bindings)
- `notebooks/` — Jupyter notebooks
- `regulatory/` — regulatory capital (business logic, not perf-critical)
- All `*_desk.py` — desk protocol (business logic)
- `api.py`, `api_desk.py` — trader API

---

## Numerical Methods Package (`numerical/`)

Self-contained toolkit — scipy/numpy are the backend, users import from `pricebook.numerical`.

| Module | What |
|--------|------|
| `_distributions.py` | Normal, StudentT, LogNormal, Uniform, Exponential |
| `_linalg.py` | expm, logm, QR, Cholesky, LU, GMRES, BiCGSTAB, Sylvester, Lyapunov |
| `_ode.py` | Euler, RK4, RK45 (adaptive), BDF (stiff), Adams |
| `_optimize.py` | minimize (NM/BFGS/L-BFGS-B/DE/CMA-ES), LP, QP, interior-point, proximal (ISTA/FISTA) |
| `_quadrature.py` | Gauss-Jacobi, tanh-sinh, Clenshaw-Curtis |
| `_interpolation.py` | 2D bilinear, bicubic, RBF (scattered data) |
| `_rootfinding.py` | bisection, unified find_root dispatcher |
| `_mc.py` | QE Heston, antithetic variates, multilevel MC |
| `_pde.py` | Hundsdorfer-Verwer ADI, 2D PSOR, operator splitting (Lie/Strang) |
| `_trees.py` | tree Greeks (delta/gamma/vega/theta), 2D binomial |
| `_fourier.py` | fractional FFT, Hilbert transform, wavelet (Haar/Db2), CharacteristicFunction |
| `_distributions_theory.py` | Schwartz test functions, tempered distributions, Fourier transform, Sobolev norms |

---

### C++ ↔ Python Bridge

Use pybind11 to expose C++ curves and instruments back to Python, replacing the pure-Python implementations transparently. The `PricingContext` stays Python but holds C++ curve objects.
