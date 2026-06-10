# Pricebook — Usage Guide

A per-layer API reference with runnable snippets. This is the practical
counterpart to `ARCHITECTURE.md` (which shows *where* code lives) — here
the focus is *how to use it*.

Start at **`python/notebooks/examples/quickstart.ipynb`** for a 20-minute
end-to-end walkthrough. This guide goes deeper, per layer.

Convention used throughout: all snippets assume `python/` is on
`sys.path`; from the repo root:

```bash
cd python
../.venv/bin/python -c "..."        # one-shot
../.venv/bin/python -m pytest tests/ -n auto    # full suite
```

---

## Contents

1. [Curves](#1-curves) — bootstrap, NSS / Smith-Wilson, global solver, AAD
2. [Models](#2-models) — engine registry, MC / PDE / Tree / Fourier / AAD
3. [Numerical methods](#3-numerical-methods) — direct-access PDE, FFT, MC, optimisation, AD
4. [Fixed income](#4-fixed-income) — bonds, swaps, FRAs, FRNs, money market, repo, inflation
5. [FX](#5-fx) — forwards, swaps, NDFs, options, barriers, exotics
6. [Equity](#6-equity) — forwards, options, autocallables, variance / dispersion, TRS
7. [Credit](#7-credit) — CDS, CLN, CDO, basket, recovery, distressed
8. [Commodity](#8-commodity) — forwards, options, spreads, seasonal, swing
9. [Options](#9-options) — Black-Scholes, SABR, Heston, LMM, Bermudan, American
10. [Structured](#10-structured) — CMS, range accruals, snowballs, TARNs
11. [Crypto](#11-crypto) — perps, funding, options, AMM, DeFi, staking
12. [Desks](#12-desks) — 9-component protocol, bond desk, futures desk, crypto desk
13. [Risk](#13-risk) — Greeks, XVA, VaR, SIMM, hedging
14. [Visualisation](#14-visualisation) — `pricebook.viz` (never raw matplotlib)
15. [Serialisation](#15-serialisation) — `to_dict` / `from_dict` and the registry
16. [Conventions](#16-conventions) — calendars, day counts, schedules, frequencies
17. [Type checking (mypy)](#17-type-checking-mypy)
18. [Database & time series](#18-database-and-time-series)

---

## 1. Curves

Layer 2. Bootstrap from market quotes, fit functional forms, run a global
multi-curve solver, or build AAD-instrumented curves for exact risk.

| Module | Purpose |
|---|---|
| `pricebook.curves.bootstrap` | Sequential bootstrap (deposits → FRAs → futures → swaps) |
| `pricebook.curves.global_solver` | Levenberg-Marquardt simultaneous solve |
| `pricebook.curves.multicurve_solver` | OIS + IBOR / cross-currency joint bootstrap |
| `pricebook.curves.nelson_siegel` | NSS functional form |
| `pricebook.curves.smith_wilson` | Smith-Wilson (insurance / Solvency II) |
| `pricebook.curves.aad_curves` | AAD-instrumented curves |
| `pricebook.curves.curve_bumper` | Parallel / key-rate bumps for risk |
| `pricebook.curves.em_curve_builder` | EM market curve construction (CDI, TIIE, SHIBOR, …) |

```python
from datetime import date, timedelta
from pricebook.curves.bootstrap import bootstrap

REF = date(2026, 6, 1)
deposits = [(REF + timedelta(days=91), 0.0525)]
swaps = [(REF + timedelta(days=365*n), r)
         for n, r in [(1, 0.048), (5, 0.041), (10, 0.0395)]]
curve = bootstrap(REF, deposits, swaps)
print(curve.df(REF + timedelta(days=365*5)))
```

Notebooks: `notebooks/rates/treasury_multicurve.ipynb`, `notebooks/papers/paper_01_multicurve.ipynb`.

---

## 2. Models

Layer 3. The pricing engines — registry, MC, PDE, trees, Fourier, AAD.

| Module | Purpose |
|---|---|
| `pricebook.models.engine_registry` | One function, any instrument, best engine — `price(...)` |
| `pricebook.models.mc_processes` | Heston, Bates, SABR, SLV, jumps, rough vol, Bergomi |
| `pricebook.models.cos_method` | Fang-Oosterlee COS for European / Bermudan |
| `pricebook.models.fft_pricing` | Carr-Madan + fractional FFT |
| `pricebook.models.fft_2d` | 2D FFT for basket options |
| `pricebook.models.fourier_greeks` | Greeks via CF differentiation |
| `pricebook.models.rough_heston_cf` | Adams scheme on fractional Riccati |
| `pricebook.models.hull_white` | 1F Hull-White (tree, MC, analytical) |
| `pricebook.models.g2pp_*` | G2++ (2F HW) — calibration, tree, swaptions |
| `pricebook.models.lmm` / `lmm_advanced` | LMM (Brace-Gatarek-Musiela) |
| `pricebook.models.adi` / `hundsdorfer_verwer` | ADI PDE solvers |
| `pricebook.models.feynman_kac` | SDE ↔ PDE bridge, cross-validation |
| `pricebook.models.fokker_planck` | Forward Kolmogorov density evolution |
| `pricebook.models.density_evolution` | 3-way density cross-validation (FP + Fourier + B-L) |
| `pricebook.models.sde_adaptive` | Adaptive Euler / Milstein step control |
| `pricebook.numerical.auto_diff` | Forward + reverse mode AD via dual numbers |

```python
from pricebook.models.engine_registry import price
result = price(spot=100, strike=100, vol=0.20, rate=0.04, T=1.0)
print(result.price, result.greeks.delta)
```

---

## 3. Numerical methods

Layer 0. Self-contained algorithms — no model assumptions. Use when you
need a primitive (a solver, a quadrature, a convex optimiser) outside the
pricing layer.

| Module | Purpose |
|---|---|
| `pricebook.numerical._pde` | Theta-scheme PDE, Rannacher smoothing, MoL |
| `pricebook.numerical._mc` | MC core, antithetic, stratified |
| `pricebook.numerical._qmc` | Sobol / Halton / lattice QMC |
| `pricebook.numerical._fourier` | Characteristic function utilities |
| `pricebook.numerical._integrate` | Adaptive Gauss-Kronrod |
| `pricebook.numerical.oscillatory_quad` | Filon + Levin collocation |
| `pricebook.numerical._optimize` | Local solvers (Brent, Nelder-Mead, …) |
| `pricebook.numerical.socp` | Interior-point SOCP |
| `pricebook.numerical.sdp` | Projected-gradient SDP |
| `pricebook.numerical.frank_wolfe` | Conditional gradient over polytopes |
| `pricebook.numerical.duality` | LP / QP dual extraction |
| `pricebook.numerical.convexity_tools` | Hessian PSD, cardinality B&B |
| `pricebook.numerical.auto_diff` | Dual-number forward + reverse AD |
| `pricebook.numerical.sparse_jacobian` | Graph-colouring sparse Jacobian |
| `pricebook.numerical.operator_splitting` | Lie-Trotter + Strang + PIDE splitting |
| `pricebook.numerical.von_neumann` | Stability region of theta-schemes |
| `pricebook.core.numerical_method_map` | Recommend method by instrument features |

---

## 4. Fixed income

Layer 5. Bonds, swaps, money market, repo, inflation.

| Module | Purpose |
|---|---|
| `pricebook.fixed_income.bond` | `FixedRateBond` — dirty / clean / YTM / duration |
| `pricebook.fixed_income.risky_bond` | `RiskyBond` — survival-curve discounted |
| `pricebook.fixed_income.callable_bond` | Callable / puttable via HW / G2++ |
| `pricebook.fixed_income.frn` | `FloatingRateNote` |
| `pricebook.fixed_income.amortising_bond` | Amortising / sinker / prepayment |
| `pricebook.fixed_income.swap` | `InterestRateSwap` (dual-curve, amortising notional) |
| `pricebook.fixed_income.zc_swap` | `ZeroCouponSwap`, digital cap / floor |
| `pricebook.fixed_income.basis_swap` | Single-currency basis (1M v 3M, OIS v IBOR) |
| `pricebook.fixed_income.xccy_swap` | Cross-currency swap |
| `pricebook.fixed_income.fra` | `FRA` |
| `pricebook.fixed_income.deposit` | Money market deposit |
| `pricebook.fixed_income.money_market` | CD / CP / BA / RepoRate helpers |
| `pricebook.fixed_income.inflation` | CPI curve, ZCI / YoY swaps, inflation-linked bonds |
| `pricebook.fixed_income.funded` | Repo, ReverseRepo, TRS, FundedParticipation, RepoFinancedPosition |
| `pricebook.fixed_income.bond_futures` | Bond futures, CTD, basis, delivery options |
| `pricebook.fixed_income.ir_futures` | STIR / SOFR / Euribor futures |
| `pricebook.fixed_income.futures_convexity` | Convexity adjustment (HW) |
| `pricebook.fixed_income.benchmark_bonds` | UST universe, RV, butterflies, barbells |
| `pricebook.fixed_income.treasury_quoting` | 32nds, reopenings, when-issued |

```python
from pricebook.fixed_income.swap import InterestRateSwap, SwapDirection
swap = InterestRateSwap(start=REF, end=REF + timedelta(days=365*5),
                        fixed_rate=0.041, direction=SwapDirection.PAYER,
                        notional=10_000_000)
print(swap.pv(curve), swap.par_rate(curve), swap.dv01(curve))
```

---

## 5. FX

Layer 7.

| Module | Purpose |
|---|---|
| `pricebook.fx.fx_forward` | `FXForward` — CIP-priced |
| `pricebook.fx.fx_swap` | `FXSwap` |
| `pricebook.fx.ndf` | `NDF` (non-deliverable forward) |
| `pricebook.fx.fx_option` | Garman-Kohlhagen, 3 delta conventions |
| `pricebook.fx.fx_barrier` | PDE + Vanna-Volga barriers |
| `pricebook.fx.fx_structured` | TARFs, autocallables, DCD, pivots |
| `pricebook.fx.prdc` | PRDC (Power Reverse Dual Currency) |
| `pricebook.fx.fx_smile_cube` | SVI / SABR / VV smile cube |
| `pricebook.fx.fx_basis` | Cross-currency basis |

---

## 6. Equity

Layer 7.

| Module | Purpose |
|---|---|
| `pricebook.equity.equity_forward` | `EquityForward` |
| `pricebook.equity.dividend_model` | Discrete / continuous dividends |
| `pricebook.equity.trs` | Unified equity TRS (incl. funding leg spec) |
| `pricebook.equity.variance_swap` | Variance swap, replication strike |
| `pricebook.equity.variance_swap_instrument` | Variance swap as serialisable instrument |
| `pricebook.options.autocallable` | Autocallable structures |
| `pricebook.equity.mountain_range` | Himalaya / Everest / Atlas |
| `pricebook.equity.equity_index_futures` | Index futures, basis |
| `pricebook.equity.quanto_futures` | Quanto futures |
| `pricebook.equity.dividend_futures` | Dividend futures + swaps |

---

## 7. Credit

Layer 4.

| Module | Purpose |
|---|---|
| `pricebook.credit.cds` | Single-name CDS, par spread |
| `pricebook.credit.cds_swaption` | CDS swaptions (4 models) |
| `pricebook.credit.cds_index` | Index CDS (CDX, iTraxx) |
| `pricebook.credit.basket_cds` | FtD / NtD basket CDS |
| `pricebook.credit.cln` | CLN (vanilla, leveraged, basket, floating) |
| `pricebook.credit.cln_xva` | CLN XVA |
| `pricebook.credit.cdo` | CDO tranches (Gaussian / t copula) |
| `pricebook.credit.tranche_option` | Options on tranches |
| `pricebook.credit.recovery_model` | Beta / market-implied recovery |
| `pricebook.credit.recovery_locked_cds` | Recovery-locked CDS |
| `pricebook.credit.recovery_trades` | Recovery trades (lock, swap) |
| `pricebook.credit.loan` | Term loans, revolvers |
| `pricebook.credit.exotic_loan` | Covenant loans |
| `pricebook.credit.distressed` | Distressed debt analytics |
| `pricebook.credit.bond_hazard_bootstrap` | Hazard rate bootstrap from bonds |

---

## 8. Commodity

Layer 7.

| Module | Purpose |
|---|---|
| `pricebook.commodity.commodity` | Spot / forward curve |
| `pricebook.commodity.commodity_basis` | Crack / spark spreads |
| `pricebook.commodity.commodity_seasonal` | Seasonal factors |
| `pricebook.commodity.commodity_swing` | Swing options |
| `pricebook.commodity.commodity_storage` | Storage / contango carry |
| `pricebook.commodity.commodity_models` | Schwartz one- / two-factor models |
| `pricebook.commodity.carbon_credit` | Carbon credits / EUA |
| `pricebook.commodity.freight` | Freight derivatives |

---

## 9. Options

Layer 6.

| Module | Purpose |
|---|---|
| `pricebook.options.equity_option` | Black-Scholes price + Greeks |
| `pricebook.options.swaption` | `Swaption` (Black / SABR / HW) |
| `pricebook.options.bermudan_swaption` | Bermudan via tree + LSM + LMM |
| `pricebook.options.swaption_vol` | Swaption vol surface |
| `pricebook.options.sabr` | SABR analytical + MC |
| `pricebook.options.heston` | Heston (closed-form via CF) |
| `pricebook.options.heston_mc` | Heston MC |
| `pricebook.options.local_vol` | Dupire local vol |
| `pricebook.options.slv` | Stochastic-local vol |
| `pricebook.options.vol_smile` / `vol_surface_strike` | Smile / surface containers |
| `pricebook.options.asian` | Geometric + arithmetic Asian |
| `pricebook.options.bermudan_barrier` | Bermudan barriers |
| `pricebook.options.exercise_boundary` | Boundary extraction (PDE / tree / LSM) |
| `pricebook.options.implied_vol` | Brent / Newton implied-vol solver |
| `pricebook.options.inflation_vol` | Inflation cap / floor vol |
| `pricebook.options.bond_futures_options` | Options on bond futures |
| `pricebook.options.futures_options` | Options on IR / commodity futures |
| `pricebook.options.variance_futures` | Variance futures |

---

## 10. Structured

Layer 7.

| Module | Purpose |
|---|---|
| `pricebook.structured.cms` | CMS rate, convexity adjustment, cap / floor, spreads |
| `pricebook.structured.cms_spread_g2pp` | CMS spread under G2++ |
| `pricebook.structured.rates_structured` | Range accruals (single-rate, CMS-spread, inflation), inverse / capped floaters, ZC swaptions |
| `pricebook.structured.structured_notes` | Capital-protected / participation / bonus certificates |
| `pricebook.fx.fx_structured` | TARFs, FX autocallables, DCD, pivots |
| `pricebook.fx.prdc` | PRDC (3-factor MC, callable LSM) |
| `pricebook.structured.steepener` | Steepener / flattener |
| `pricebook.fixed_income.callable_floater` | Callable / puttable FRN (HW + G2++) |
| `pricebook.structured.cross_asset_structured` | Cross-asset structured products |

---

## 11. Crypto

Layer 7. Fully covered (12 phases + 4 bonus modules, ~135 tests).

| Module | Purpose |
|---|---|
| `pricebook.crypto.perpetual` | Linear / inverse / quanto perps |
| `pricebook.crypto.funding_rate` | Funding curve, carry, EWMA |
| `pricebook.crypto.crypto_options` | Linear + inverse options, DVOL |
| `pricebook.crypto.crypto_vol` | 24/7 vol surface, Parkinson / YZ |
| `pricebook.crypto.amm` | Uniswap v2, Curve stableswap |
| `pricebook.crypto.impermanent_loss` | IL, breakeven, optimal range |
| `pricebook.crypto.defi_rates` | Aave / Compound kink model |
| `pricebook.crypto.staking` | Validator yield, slashing, liquid staking |
| `pricebook.crypto.liquidation_cascade` | Margin, liquidation price, cascades |
| `pricebook.crypto.basis_arb` | Spot-perp, cash-and-carry, triangular |
| `pricebook.crypto.crypto_risk` | 24/7 VaR, tail risk |
| `pricebook.crypto.stablecoin` | Depeg, reserves |
| `pricebook.crypto.smart_contract_risk` | Protocol risk metrics |
| `pricebook.crypto.tokenomics` | Supply schedules, emission |
| `pricebook.desks.crypto_desk` | Crypto desk (9-component protocol) |

---

## 12. Desks

Layer 8. Each desk follows the **9-component protocol**: book → quote → trade
construction → risk → P&L → carry / roll → hedge → reporting → end-of-day.

| Module | Desk |
|---|---|
| `pricebook.desks.bond_trading_desk` | Bond / Treasury trading |
| `pricebook.desks.futures_desk` | Bond + IR + commodity futures |
| `pricebook.desks.swap_desk` | IRS / OIS / xccy |
| `pricebook.desks.cds_desk` | Single-name + index CDS |
| `pricebook.desks.fx_desk` | Vanilla + exotic FX |
| `pricebook.desks.equity_desk` | Cash equity + delta-one |
| `pricebook.desks.options_book` | Listed + OTC options |
| `pricebook.desks.structured_credit_desk` | Tranches, CLN-on-CDX, bespokes |
| `pricebook.desks.repo_desk` | Repo / reverse / GC / specials |
| `pricebook.desks.crypto_desk` | Crypto |
| `pricebook.desks.convertible_bond_desk` | Convertible bonds |
| `pricebook.desks.dispersion_desk` | Index vs single-name dispersion |

Notebooks: `notebooks/desks/bond_trading_desk.ipynb`, `notebooks/desks/futures_desk.ipynb`.

---

## 13. Risk

Layer 4.

| Module | Purpose |
|---|---|
| `pricebook.risk.greeks` | Bump / pathwise / LR Greeks |
| `pricebook.risk.xva` | CVA / DVA / FVA / KVA / MVA / ColVA |
| `pricebook.risk.var` | Historical / parametric / Monte Carlo VaR |
| `pricebook.risk.cvar_optimisation` | CVaR-constrained portfolio optimisation |
| `pricebook.risk.simm` | ISDA SIMM |
| `pricebook.regulatory.counterparty` | SA-CCR |
| `pricebook.regulatory.market_risk_ima` | FRTB IMA sensitivities |
| `pricebook.regulatory.market_risk_sa` | FRTB SA |
| `pricebook.regulatory.reverse_stress` | Reverse stress testing |
| `pricebook.risk.vol_stress` | Vol-shock scenario stress |
| `pricebook.risk.cross_gamma_hedging` | Cross-gamma hedge construction |
| `pricebook.curves.key_rate_risk` | Key-rate DV01 |

---

## 14. Visualisation

Layer 0. **Always use `pricebook.viz`. Never `import matplotlib.pyplot`.**

| Entry | Purpose |
|---|---|
| `plot(instrument, curve)` | Auto-detect type, 2×2 dashboard |
| `PlotBuilder(instrument, curve).payoff().greeks().figure()` | Fluent builder |
| `configure_theme(...)` | Set seaborn style / context once per session |
| `greeks_profile(spot_range, greeks_by_spot, ...)` | Multi-panel Greeks vs spot |
| `sensitivity_grid(...)` | Heat map of one Greek vs two factors |
| `correlation_heatmap(...)` | Correlation matrix |
| `pnl_distribution(...)` | P&L histogram + KDE |
| `pnl_waterfall(...)` | Greeks-explained P&L |
| `risk_decomposition(...)` | Bar chart of risk contributions |
| `vega_ladder(...)` | Vega by tenor / strike |
| `j_curve(...)` | Private equity J-curve |
| `football_field(...)` | Valuation football field |
| `exposure_profile(...)` | Expected exposure / EE+ / EE- |

```python
from pricebook.viz import configure_theme, greeks_profile
configure_theme(seaborn_style="whitegrid", seaborn_context="notebook")

spots = np.linspace(70, 130, 41)
greeks_by_spot = {"Delta": deltas, "Gamma": gammas, "Vega": vegas}
fig = greeks_profile(spots, greeks_by_spot, title="Equity call")
```

---

## 15. Serialisation

Layer 0 (`pricebook.core.serialisable`). Every traded class is registered with
`_serialisable("type_key", [fields])` and gets `to_dict()` / `from_dict()` for
free. The global registry dispatches `from_dict({"type": ...})` to the right
class.

```python
from pricebook.fixed_income.money_market import CertificateOfDeposit
from pricebook.core.serialisable import from_dict
import json

cd = CertificateOfDeposit(settlement=REF, maturity=REF + timedelta(days=180),
                          face_value=1_000_000, coupon_rate=0.048)
blob = json.dumps(cd.to_dict(), default=str)
rebuilt = from_dict(json.loads(blob))   # registry dispatch
```

For new instruments:

```python
class MyTrade:
    def __init__(self, start, end, rate, notional=1.0):
        self.start = start
        self.end = end
        self.rate = rate
        self.notional = notional

from pricebook.core.serialisable import serialisable as _serialisable
_serialisable("my_trade", ["start", "end", "rate", "notional"])(MyTrade)
```

Field names in the list **must** match constructor parameter names AND attributes
on `self`. The registry validates this at import time and warns on mismatch.
For polymorphic fields (nested instruments) or custom logic, override `from_dict`.

---

## 16. Conventions

Layer 0 (`pricebook.core.*`).

| Module | Purpose |
|---|---|
| `pricebook.core.calendar` | Holidays per market, business-day conventions |
| `pricebook.core.day_count` | ACT/360, ACT/365F, 30/360, ACT/ACT, … |
| `pricebook.core.schedule` | Schedule generation, frequency, stub types |
| `pricebook.core.currency` | Currency, CurrencyPair |
| `pricebook.core.fixings` | Historical fixings store |
| `pricebook.core.discount_curve` | `DiscountCurve` — the canonical curve API |
| `pricebook.core.survival_curve` | Survival curve for credit |
| `pricebook.fixed_income.inflation` | `CPICurve` for inflation |
| `pricebook.core.market_data` | Quote / MarketDataSnapshot / HistoricalData |
| `pricebook.core.trade` | Trade abstraction, pricing context |

---

## 17. Type checking (mypy)

Mypy is configured in `python/pyproject.toml` under `[tool.mypy]`. Pragmatic baseline — pedantic categories (`misc`, `annotation-unchecked`, numpy-return-Any) are silenced; the rest matter. 184 legacy modules are listed under a `[[tool.mypy.overrides]]` block with `ignore_errors = true` — to be cleaned up incrementally.

```bash
cd python
../.venv/bin/pip install -e ".[dev]"      # one-time
../.venv/bin/mypy pricebook --no-incremental
# → Success: no issues found in 795 source files
```

**To clean a module:** remove its entry from the `module = [...]` list in `pyproject.toml`, then `mypy pricebook` and fix the surfaced errors. Goal: shrink the override list to zero, one slice at a time.

---

## 18. Database and time series

Layer 0.

| Module | Purpose |
|---|---|
| `pricebook.db.db` | `PricebookDB` — SQLite + JSON store; 7 system tables + custom |
| `pricebook.ts._core` | Numpy-backed time series (no pandas) |
| `pricebook.ts._rolling` | Rolling statistics |
| `pricebook.ts._replay` | Replay from DB for backtests |

```python
from pricebook.db.db import PricebookDB
db = PricebookDB("trades.db")
db.save_instrument("usd-irs-5y", cd.to_dict())
```

---

## Where next

- **`ARCHITECTURE.md`** — layer structure and dependency rules
- **`RELEASE_NOTES.md`** — version history
- **`notebooks/examples/quickstart.ipynb`** — 20-min walkthrough
- **`notebooks/papers/`** — 12 paper validations
- **`notebooks/desks/`** — trader workflows
- **`notebooks/validation/`** — exact-paper-table reproductions
