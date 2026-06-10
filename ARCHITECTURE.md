# Pricebook Architecture

**23 sub-packages | 793 modules | 7 dependency layers | 0 circular deps | 11,600+ tests**

Counts above are **empirical** — produced by walking the import graph at
`v0.864.0`. Regenerate at any time with the snippet at the end of this
document.

---

## Package Structure

```
pricebook/
├── core/              35 modules   L0   Curves, scheduling, trade, serialisation
├── db/                 3 modules   L0   SQLite + JSON store (PricebookDB)
├── numerical/         31 modules   L0   Self-contained numerical methods
├── pe/                 5 modules   L0   Private equity (LBO, DCF, performance)
├── statistics/        18 modules   L0   GARCH, Kalman, copulas, regression
├── ts/                 8 modules   L0   Time series (numpy-backed, no pandas)
├── viz/               14 modules   L0   Matplotlib + seaborn visualisation
├── curves/            33 modules   L1   Bootstrap, global solver, NSS / SW, AAD
├── pricing/           10 modules   L1   Pricing engine, codecs, market data
├── regulatory/        24 modules   L1   Basel III/IV, FRTB, SA-CCR, SIMM, CCAR
├── data/               6 modules   L2   Market curve / rate data loaders
├── models/            92 modules   L2   MC, PDE, trees, Black-76, Hull-White, G2++, COS, FFT
├── credit/            94 modules   L3   CDS, CLN, CLO, loans, recovery, distressed
├── crypto/            16 modules   L3   Perps, options, AMM, DeFi, staking
├── fixed_income/     131 modules   L3   Bonds, swaps, FRA, FRN, repos, inflation
├── risk/              55 modules   L3   Greeks, hedging, XVA, VaR, stress
├── options/           62 modules   L4   Swaptions, vol surfaces, SABR, Heston, exotic options
├── registry.py         1 module    L4   Solver / pricer registry (wiring)
├── commodity/         24 modules   L5   Commodity instruments
├── equity/            34 modules   L5   TRS, dividends, variance swaps, equity-linked
├── fx/                23 modules   L5   FX forwards, swaps, NDFs, barriers, exotics
├── structured/        24 modules   L5   CMS, CMO, structured notes, hybrids
└── desks/             50 modules   L6   16+ trading desks (9-component protocol)
```

`__init__.py` at the top of `pricebook/` re-exports the public surface.

---

## Dependency Layers (bottom-up, verified acyclic)

```
L0  foundations (no pricebook deps)
    core    db    numerical    pe    statistics    ts    viz

L1  needs only Layer 0
    curves         (core, numerical, statistics)
    pricing        (core)
    regulatory     (core)

L2  needs curves
    data           (core, curves)
    models         (core, curves, numerical, statistics)

L3  needs models
    credit         (core, curves, models, statistics)
    crypto         (models)
    fixed_income   (core, curves, models, statistics)
    risk           (core, models, numerical)

L4  needs fixed_income
    options        (core, curves, fixed_income, models, statistics)
    registry       (core, curves, models, numerical, risk, statistics)

L5  asset classes — need options
    commodity      (core, fixed_income, models, options, statistics)
    equity         (core, fixed_income, models, options, statistics)
    fx             (core, fixed_income, models, options, statistics)
    structured     (core, credit, fixed_income, models, options)

L6  desks — top of the DAG
    desks          (commodity, core, credit, curves, equity, fixed_income,
                    fx, models, options, regulatory, risk, statistics)
```

Each layer depends only on layers below it. **0 circular dependencies** at any package level — verified by Tarjan SCC on every commit's `pricebook/` import graph.

---

## Fan-in — how many packages depend on each

| Pkg | Fan-in | Used by |
|---|---:|---|
| `core` | 15 | everyone |
| `models` | 11 | commodity, credit, crypto, desks, equity, fixed_income, fx, options, registry, risk, structured |
| `statistics` | 10 | commodity, credit, curves, desks, equity, fixed_income, fx, models, options, registry |
| `curves` | 7 | credit, data, desks, fixed_income, models, options, registry |
| `fixed_income` | 6 | commodity, desks, equity, fx, options, structured |
| `options` | 5 | commodity, desks, equity, fx, structured |
| `numerical` | 4 | curves, models, registry, risk |
| `credit` | 3 | desks, structured, (fixed_income via TYPE_CHECKING only) |
| `risk` | 2 | desks, registry |
| `equity`, `commodity`, `regulatory`, `fx` | 1 | desks |
| leaves (no fan-in) | 0 | `desks`, `structured`, `crypto`, `viz`, `pricing`, `ts`, `data`, `pe`, `db` |

`core` and `models` are the load-bearing centres of the system. Touch either and you ripple across the entire library.

---

## Tallest path through the DAG (worst-case build / port order)

```
core → curves → models → fixed_income → options → fx → desks
  L0      L1       L2          L3           L4    L5    L6
```

Seven hops, matching the layer count. This is the order in which packages must come online for any C++ port.

---

## Data Flow

```
Market Data (quotes, fixings, indices)
    │
    ▼
Curves (discount, survival, IBOR, vol)         ── curves/, core/
    │
    ▼
PricingContext (bundles all curves)             ── core/pricing_context
    │
    ▼
Models (Black-76, Heston, MC, PDE, trees, G2++) ── models/
    │
    ▼
Instruments (all implement pv() / pv_ctx())    ── fixed_income/, credit/, options/, fx/, equity/, commodity/, crypto/
    │
    ├──▶ Risk (bump-and-reprice, Greeks, XVA)   ── risk/
    │
    ▼
Trade → Portfolio → Book → Desk                ── core/trade, desks/
    │
    ├──▶ Daily P&L (sequential attribution)     ── desks/*_daily_pnl
    ├──▶ Regulatory Capital (FRTB, SA-CCR)      ── regulatory/
    ├──▶ Stress Testing (CCAR, reverse stress)  ── regulatory/ccar, regulatory/reverse_stress
    │
    ▼
Viz (PlotBuilder, dashboards)                  ── viz/
TimeSeries (replay, drawdown, rolling Sharpe)  ── ts/
Database (trades, P&L history, snapshots)      ── db/
```

---

## Core Infrastructure (`core/` — 35 modules)

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
| `serialisable` | `@serialisable` decorator | Auto `to_dict()` / `from_dict()` |
| `serialization` | `to_json()`, `from_json()` | Full instrument serialisation |
| `numerical_method_map` | recommender | Pick MC / PDE / Fourier per instrument |
| `numerical_safety` | guards | log / division / boundary checks |

---

## Models (`models/` — 92 modules)

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
| `G2PlusPlus` | `models/g2pp_*` | 2-factor HW, calibrated to swaption vols |
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

### Fourier / spectral

`models/cos_method`, `cos_bermudan`, `fft_pricing`, `fft_2d`, `fourier_greeks`, `rough_heston_cf` — Fang-Oosterlee COS, Carr-Madan FFT (1D + 2D), CF-differentiation Greeks, fractional Riccati for rough Heston.

### PDE / SDE

`models/adi`, `hundsdorfer_verwer`, `fokker_planck`, `density_evolution`, `feynman_kac`, `sde_adaptive` — ADI for Heston, HV ADI, forward Kolmogorov, three-way density consistency, adaptive Euler / Milstein.

### Tree

`models/bdt_tree`, `g2pp_tree`, `tree_mc_bridge` — calibrated trees, 2D recombining for G2++, bridge to MC.

---

## Instruments

### Fixed Income (`fixed_income/` — 131 modules)

Bonds (fixed-rate, FRN, risky, callable, amortising, sinker, prepayment), swaps (IRS, OIS, basis, xccy, ZCS, ASW), FRA, repos (term, reverse, financed positions, TRS-as-funding), inflation (CPI curve, ZCI / YoY swaps, linkers), money market (CD, CP, BA, deposits, T-Bills), money-market futures + convexity, treasury (32nds quoting, when-issued, benchmark RV, butterflies, barbells), sovereign curves (33 markets), funded participation, sukuk, supranational, callable / puttable structured notes.

### Credit (`credit/` — 94 modules)

CDS (single-name, index, basket, FtD / NtD), CDS swaptions (4 models), CDX / iTraxx index, CLN (vanilla, leveraged, basket, stochastic recovery, XVA), CDO / CLO tranches (Gaussian / t-copula, base correlation), term loans (floored, PIK, covenant), recovery (beta, market-implied, downturn LGD, locks, swaps, surface), distressed (DIP, fulcrum, exchange offers), unitranche, waterfall, fund participation, stochastic-credit Bermudan CDS.

### Options (`options/` — 62 modules)

Swaptions (Black, SABR, HW, G2++, Bermudan via tree / LSM / LMM), cap / floor, equity options (BS, Bachelier, binomial, PDE, autocallable), barriers (PDE + Vanna-Volga), Asians (geometric, arithmetic via MC + control variate), implied vol (Brent / Newton), vol surfaces / cubes (SABR / SVI / SLV), local vol (Dupire), Heston (CF + MC), TARFs, convertibles, weather, variance / dispersion options.

### FX (`fx/` — 23 modules)

FX forwards, swaps, NDFs, options (Garman-Kohlhagen, 3 delta conventions), barriers (PDE + Vanna-Volga), exotics (TARF, accumulator, DCD, pivot), PRDC (3-factor MC + callable LSM), SVI / SABR / VV smile cubes, hedging strategies, EM-FX restrictions, cross-currency basis.

### Equity (`equity/` — 34 modules)

Equity forwards, dividends (discrete + continuous + futures + swaps + options), TRS (unified, tree, XVA), variance / vol swaps, dispersion, mountain range (Himalaya / Everest / Atlas), autocallable structures, ELN (6 types), index futures, quanto futures.

### Commodity (`commodity/` — 24 modules)

Forwards, basis (crack / spark), seasonal factors, swing options, storage / contango, Schwartz / GibsonSchwartz / SchwartzSmith, carbon credits / EUA, freight, power derivatives, spread options + dynamics, commodity-rates link.

### Crypto (`crypto/` — 16 modules)

Perpetuals (linear / inverse / quanto), funding rate curve, crypto options (Deribit-style), 24/7 vol surface (Parkinson / Yang-Zhang), AMM (Uniswap v2, Curve stableswap), impermanent loss, DeFi rates (Aave / Compound kink), staking (validator yield, slashing, liquid staking), liquidation cascades, basis arb, crypto VaR, stablecoin, smart-contract risk, tokenomics.

### Structured (`structured/` — 24 modules)

CMS (rate, convexity, cap / floor, spreads), G2++ CMS spread, rates structured (range accruals, inverse / capped floaters, ZC swaptions, callable step-ups), ABS / MBS / CMBS / CMO (PSA / CPR / SMM, OAS, IO / PO), CAT bonds, CPDO, ELN, insurance annuity guarantees, real estate derivatives, longevity, secondary pricing, capped / floored / collar floaters, cross-asset structured.

---

## Trading Desks (`desks/` — 50 modules)

### 9-Component Protocol

Every desk implements:

| # | Component | Pattern |
|---|-----------|---------|
| 1 | `RiskMetrics` | Per-position Greeks (PV, DV01, CS01, delta, gamma, vega) |
| 2 | `Book` + `BookEntry` | Position management, aggregation, grouping |
| 3 | `CarryDecomposition` | Coupon, funding, roll-down, net carry |
| 4 | `DailyPnL` | Rate / vol / credit / theta / carry / unexplained attribution |
| 5 | `Dashboard` | Morning summary: positions, risk totals, top exposures |
| 6 | `StressResult` | Parametric scenarios (rates, spreads, vol shocks) |
| 7 | `CapitalResult` | SA-CCR EAD, RWA, FRTB charges, SIMM IM |
| 8 | `HedgeRecommendation` | Limit-breach detection + suggestions |
| 9 | `Lifecycle` | Event tracking: maturities, coupons, triggers, alerts |

16+ desks: swap, bond, futures, CDS, CLN, swaption, asset swap, convertible bond, structured credit, TRS, repo, FX, equity, commodity, inflation, crypto, dispersion, PE.

---

## Risk (`risk/` — 55 modules)

| Module | Role |
|--------|------|
| `greeks` | `Greeks` dataclass, `bump_greeks()` |
| `var` | Historical, parametric, Monte Carlo VaR |
| `xva` | CVA, DVA, FVA, KVA, MVA, ColVA |
| `scenario` | `parallel_shift()`, `pillar_bump()`, `vol_bump()` |
| `pnl_explain` | `PnLResult` decomposition |
| `cross_asset_greeks` | Multi-asset Greek attribution |
| `cross_gamma_hedging` | Correlation-aware position sizing |
| `simm` | ISDA SIMM initial margin |
| `backtest` | `run_backtest()`, walk-forward, deflated Sharpe |
| `portfolio_construction` | Mean-variance, Black-Litterman, risk parity |
| `factor_model` | Factor attribution, covariance, timing |
| `cvar_optimisation` | CVaR-constrained portfolio optimisation |
| `vol_stress` | Vol-shock scenario stress |

---

## Regulatory (`regulatory/` — 24 modules)

| Module | Framework |
|--------|-----------|
| `credit_rwa` | SA-CR, F-IRB, A-IRB, specialised lending slotting |
| `market_risk_sa` | FRTB-SA: SbM (delta / vega / curvature) + DRC + RRAO |
| `market_risk_ima` | FRTB-IMA: liquidity-adjusted ES, NMRF, MC DRC, PLA |
| `ima_bridge` | Desk sensitivities → IMA risk factors |
| `counterparty` | SA-CCR: replacement cost, PFE, add-on |
| `var_es` | Parametric, historical, MC VaR / ES, backtesting |
| `irc` | Incremental Risk Charge (MC rating migration) |
| `securitization` | SEC-SA, SEC-IRBA, ERBA |
| `capital_framework` | Output floor, leverage ratio, G-SIB, TLAC |
| `total_capital` | Total RWA aggregation across all risk types |
| `ccar` | 9-quarter capital projection |
| `reverse_stress` | Minimum-severity scenario finder |
| `capital_allocation` | Euler allocation, RORC, hurdle rates, capital limits |
| `liquidity` | Portfolio-wide LCR / NSFR |
| `operational_risk` | SMA: BI / BIC / ILM, bucket classification |
| `repo_capital` | SFT EAD, LCR outflow, NSFR RSF |

---

## Visualisation (`viz/` — 14 modules)

**Use this layer always; never `import matplotlib.pyplot` directly.**

| Function | Chart Type |
|----------|-----------|
| `plot(instrument, curve)` | Auto-detect type, 2 × 2 dashboard |
| `PlotBuilder(...).payoff().greeks().figure()` | Fluent builder |
| `configure_theme(...)` | Set seaborn style / context |
| `greeks_profile()` | Multi-panel Greeks vs spot |
| `pnl_waterfall()` | Waterfall / bridge P&L attribution |
| `stress_comparison()` | Grouped / stacked scenario bars |
| `greeks_surface()` | 2D contour (strike × expiry) |
| `greeks_evolution()` | Greeks vs time-to-expiry |
| `vega_ladder()` | Vega by expiry bucket |
| `tenor_bucketing()` | DV01 by tenor bucket |
| `hedge_pnl_tracking()` | Position vs hedge cumulative P&L |
| `rolling_correlation()` | Multi-line rolling correlation |
| `football_field()` | Valuation range (DCF, comps, WACC) |
| `j_curve()` | PE fund TVPI over time |
| `sensitivity_grid()` | 2D parameter sensitivity heatmap |
| `pnl_distribution()` | P&L histogram with VaR / CVaR |

---

## Numerical Methods (`numerical/` — 31 modules)

Self-contained toolkit. NumPy / SciPy are the backend; users import from `pricebook.numerical`.

| Module | Methods |
|--------|---------|
| `_distributions`, `_distributions_theory` | Normal, StudentT, LogNormal, Schwartz distributions, Sobolev norms |
| `_linalg` | expm, logm, QR, Cholesky, LU, GMRES, Sylvester, Lyapunov |
| `_ode` | Euler, RK4, RK45, BDF, Adams |
| `_optimize` | minimize, LP, QP, interior-point, proximal (ISTA / FISTA) |
| `_integrate` | Adaptive Gauss-Kronrod |
| `_interpolation` | 2D bilinear, bicubic, RBF |
| `_rootfinding` | bisection, find_root |
| `_mc`, `_qmc`, `_stochastic` | MC core, Sobol / Halton / lattice QMC, stochastic processes |
| `_pde` | Theta-scheme PDE, Rannacher smoothing, MoL |
| `_trees` | Tree Greeks, JR / LR / Tian, 2D binomial |
| `_fourier` | Fractional FFT, Hilbert, wavelet, CharacteristicFunction |
| `_spectral`, `_differentiate` | Spectral / Chebyshev, finite-difference |
| `oscillatory_quad` | Filon + Levin collocation |
| `socp`, `sdp`, `duality`, `frank_wolfe`, `convexity_tools` | Convex optimisation (SOCP, SDP, dual extraction, conditional gradient, PSD checks) |
| `auto_diff`, `sparse_jacobian` | Forward + reverse-mode AD, graph-colouring sparse Jacobians |
| `operator_splitting`, `von_neumann` | Lie-Trotter / Strang / PIDE splitting, stability region analysis |
| `pde_boundary`, `implied_tree`, `tree_enhancements` | PDE BCs, implied trees, tree refinements |

---

## Private Equity (`pe/` — 5 modules)

| Module | Key Classes |
|--------|------------|
| `lbo` | `LBOModel` — sources & uses, debt schedule, FCF, exit, sensitivity |
| `dcf` | `DCFModel` — WACC, terminal value, EV bridge, football field |
| `pe_performance` | Kaplan-Schoar PME, direct alpha, vintage cohort, commitment pacing |
| `pe_desk` | 9-component PE desk protocol |

---

## Statistics (`statistics/` — 18 modules)

| Module | Methods |
|--------|---------|
| `garch` | GARCH(1,1), EGARCH, EWMA, realized vol, GARCH VaR |
| `kalman`, `particle_filter` | Kalman filter, RTS smoother, dynamic beta; particle filtering |
| `clustering`, `hmm` | K-means, silhouette, hierarchical, HMM regime switching |
| `regression`, `bayesian` | OLS, Ridge, Lasso, quantile, robust; Bayesian inference |
| `copulas` | Gaussian, Student-T, Clayton, Frank, Gumbel |
| `distribution_fit` | MLE fitting, KS test, Anderson-Darling, QQ |
| `statistics` | ACF, PACF, Ljung-Box, ADF, Durbin-Watson |
| `optimisation_advanced` | Robust optimisation, multi-objective |
| `information_theory` | Entropy, mutual information, KL |
| `zscore`, `calibration_quality` | Z-score signals, calibration diagnostics |

---

## Database & Time series

`db/` (3 modules) — `PricebookDB`: 7 system tables (entities, ratings, trades, market_snapshots, pricing_results, pnl_history, kv_store) + custom tables + CSV export. `SQLiteBackend`.

`ts/` (8 modules) — Numpy-backed time series, rolling stats, replay-from-DB for backtests, returns, IO, viz.

`data/` (6 modules) — Market data loaders (Euribor, rate database, market_curve, synthetic data).

---

## Pricing layer (`pricing/` — 10 modules)

Pricing server, client, engine, schema, codecs, market data provider. Cross-process pricing service.

---

## C++ Port Roadmap

### Phase 1 — Curves + Bootstrap
`core/discount_curve` → `DiscountCurve.h`
`curves/bootstrap` → `Bootstrap.h`
`core/solvers` → `Solvers.h`

### Phase 2 — Core Instruments
`fixed_income/swap` → `InterestRateSwap.h`
`fixed_income/bond` → `FixedRateBond.h`
`credit/cds` → `CDS.h`

### Phase 3 — MC Engine
`models/mc_engine` → `MCEngine.h` (parallelisable)
`models/mc_processes` → `Processes.h`

### Phase 4 — Risk + Numerics
`risk/` → `RiskEngine.h`
`numerical/_pde` → `PDESolver.h`
`numerical/auto_diff` → `AAD.h` (the production-O(1)-Greeks endgame)

### Stays Python
`viz/`, `ts/`, `db/`, `regulatory/`, `desks/`, notebooks, API.

---

## Regenerating this document's stats

```bash
cd python
../.venv/bin/python <<'PY'
import os, re
from pathlib import Path
from collections import defaultdict
os.chdir(".")
PKG_RE = re.compile(r"^from\s+pricebook\.([a-z_][a-z_0-9]*)(?:\.[a-z_][a-z_0-9.]*)?\s+import|^import\s+pricebook\.([a-z_][a-z_0-9]*)", re.M)
deps, files = defaultdict(set), defaultdict(int)
for py in Path("pricebook").rglob("*.py"):
    if "__pycache__" in str(py): continue
    parts = py.parts
    pkg = parts[1][:-3] if parts[1].endswith(".py") else parts[1]
    files[pkg] += 1
    src = py.read_text(errors="ignore")
    for m in PKG_RE.finditer(src):
        imp = m.group(1) or m.group(2)
        if imp and imp != pkg: deps[pkg].add(imp)
nodes = {p for p, n in files.items() if n > 0 and p != "__init__"}
edges = {p: deps[p] & nodes for p in nodes}
layer = {}
def lvl(p): 
    if p in layer: return layer[p]
    layer[p] = 0 if not edges[p] else 1 + max(lvl(q) for q in edges[p])
    return layer[p]
for p in nodes: lvl(p)
print(f"{len(nodes)} packages, {sum(files[p] for p in nodes)} files, {max(layer.values())+1} layers")
for l in sorted(set(layer.values())):
    for p in sorted(p for p in nodes if layer[p] == l):
        print(f"  L{l}  {p:<15} {files[p]:>4} files  deps={sorted(edges[p])}")
PY
```
