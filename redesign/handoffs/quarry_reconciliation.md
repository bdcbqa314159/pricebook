# Quarry reconciliation map

Date: 2026-07-17 (refreshed at CP-2c)   ·   Version: `0.46.0`   ·   Living document.

Delivers the CP-2a ruling (`rulings_CP1.md`): the honest drawdown baseline and the ordered
parity-gap list, before parity-depth slices begin. **"Crossed" = the quarry module it supersedes
is *deletable* (realigned parity reached)** — a simplified skeleton is a *partial* cross with a
recorded gap, not a cross (`CLAUDE.md §4`).

## RETIRED (deletable) — the drawdown numerator

**Drawdown = 1 / 768 (0.13%).** First quarry module superseded (CP-3 #1).

### `core/numerical_config.py` → superseded by `foundation/numerical_config.py` (v0.47.0)
- **Genuine residual — CLOSED:** `to_dict`/`from_dict` (round-trip + `schema_version`) and `replace`.
- **Needed functionality covered:** `mc_paths`, `mc_seed` (ng engines consume these) + `fd_bump`
  (ng's DV01 knob). Validation of positive knobs preserved.
- **`shed:` — the 12 extra knobs, all `dead` (evidence: `grep '\.<knob>' python/` → no production
  consumer; only refs are the quarry module's own unit test, which retires with it, and a same-named
  local kwarg `compare_engines(tree_steps=…)`):**
  `mc_antithetic, mc_use_sobol, mc_brownian_bridge, pde_time_steps, pde_space_steps, pde_n_std_devs,
  tree_steps, integration_tol, integration_max_iter, cos_n, cos_L, rootfinder_tol, rootfinder_max_iter, extra`.
  The quarry's own docstring admits the config was *"incremental… a pricer that doesn't yet read the
  context's config simply uses its own defaults"* — i.e. the knobs were defined but never wired to any
  pricer. ng correctly omitted them (§6b); a future ng PDE/COS/tree engine adds its own knob when it lands.
- **Cowork: review this first `shed:` list** — evidence is the `\.<knob>` grep above (zero production hits).

## Headline

- **Quarry: 768 modules. New tree: 54 modules. Deletable: 1** (`core/numerical_config`).
- **Drawdown = 0 / 768 (0.0%).** The ng tree is a coherent *simplified parallel build*, not yet a
  migration. Nothing is deletable yet — but CP-2b narrowed the foundational gaps (see below).
- **CP-2b progress (parity order #1→#3):** curve rate accessors added (`zero_rate`,
  `instantaneous_forward`); **HW de-flattened — the biggest single gap, now general-curve**; `FRA`
  added (was untouched backlog). Deletability still needs a rigorous per-module parity confirmation
  (the CP-2b process gap).
- Of the 49 ng modules, **~7 are redesign spine with no direct quarry counterpart** (they *enable*
  future crossings but don't themselves retire a quarry module): `market/keys` (A5 `MarketKey`),
  `market/snapshot` (A5), `models/discounting_model` (A1), `risk/priceable` (Priceable protocol),
  `foundation/results` (A2 decomposition), `foundation/numerical_config`, `market/discount_curve`
  partially (loglinear curve type).

## Per-ng-module parity (49) — all `partial`

### L0 foundation (10) → mostly `core/` + `numerical/`
| ng module | quarry counterpart(s) | parity gap |
|---|---|---|
| time | core/day_count, schedule, rate_index | subset of conventions; no ICMA/BUS252 breadth, no RateIndex fixings |
| schedule | core/schedule | no full roll/stub/calendar-adjusted breadth |
| calendar | core/calendar, market_conventions | minimal holiday model; no market_conventions |
| money | core/currency, notional | Money only; no currency table / notional conventions |
| cashflow | core/settlement, notional | atom only |
| distributions | numerical/_distributions(_theory) | `norm_cdf` only; no inverse/other laws |
| solvers | numerical/_rootfinding, _optimize | `bisect_root` + `nelder_mead` of a 30-module toolkit |
| black | models/black76 | one formula |
| numerical_config | core/numerical_config, models/mc_config | redesign knobs; not the quarry superset |
| results | core/pricing_context (decomposition) | redesign (A2); no direct quarry module |

### L1 market (4) → `core/` + `curves/` + `credit/`
| ng module | quarry counterpart(s) | parity gap |
|---|---|---|
| discount_curve | core/discount_curve, curves/bootstrap | +zero_rate/instantaneous_forward (CP-2b) + **forward_rate + RateCurve protocol (CP-2c, closes ruling §4.1/§4.2)**; pluggable interpolation + roll_down remain; vs curves/ (31) |
| survival_curve | core/survival_curve, credit/issuer_curve, hazard_term_structure | annual grid, log-linear; no term-structure breadth |
| snapshot | core/pricing_context, market_data | redesign (A5 keyed registry); enabler |
| keys | core/data_registry | redesign (A5); enabler |

### L3 models + calibration (6) → `models/` (90) + `curves/` + `credit/`
| ng module | quarry counterpart(s) | parity gap |
|---|---|---|
| models/hull_white | models/hull_white, hw_calibration, hw_per_currency, short_rate_models | **general-curve now (CP-2b — flat gap CLOSED)**; remaining: constant vol (no term structure), per-ccy, tree. Closest to parity. |
| models/discounting_model | core/pricing_context | redesign (A1); enabler |
| models/credit_model | credit/issuer_curve, credit_risk | thin adopt-market wrapper |
| calibration/discount_curve | curves/bootstrap, rfr_bootstrap, multicurve_solver | deposits+par-swaps single-curve |
| calibration/hull_white | models/hw_calibration | caplet + cap-strip least-squares; no swaption/co-terminal |
| calibration/survival_curve | credit/bond_hazard_bootstrap, hazard_term_structure | CDS par bootstrap only |

### L2 products (11) → `fixed_income/` (130) + `credit/` + `fx/` + `equity/` + `commodity/`
| ng module | quarry counterpart(s) | parity gap |
|---|---|---|
| fixed_rate_bond | fixed_income/bond, zero_coupon_bond | vanilla fixed only; vs 130-module bond zoo |
| swap | fixed_income/swap, ois | single-curve vanilla IRS; no OIS/basis/xccy |
| fra | fixed_income/fra | +seasoned via FixingHistory (CP-2c, quarry FRA lacks fixings). Deletable-bar read: quarry ISDA-settle == ng end-settle (single-curve). **Residual: multi-curve `forward_rate(projection_curve)`, `par_rate`/`pv_ctx`, convention builder.** Not deletable. |
| deposit | fixed_income/deposit | **new (CP-2c)**; 2-cashflow trade via DiscountingEngine (fwd par→0, spot par→principal via A2). Deletable-bar read: ng covers pricing + forward/temporal (quarry values redemption-only). **Residual: convention builder, implied-DF-as-method (ng has via DepositQuote), pv_ctx, serialisation.** Not deletable. |
| ois | fixed_income/ois | **new (CP-2c)**; single-curve OIS via shared `float_leg_pv` (== vanilla IRS). Deletable-bar read: ng covers single-curve pricing + par. **Residual: currency conventions (SOFR/SONIA/ESTR), `bootstrap_ois`, par_rate/annuity/dv01, daily-fixing compounding, multi-curve basis.** Not deletable. |
| swaption | fixed_income/(swaption), options | European payer/receiver only |
| leg | fixed_income/fixed_leg, floating_leg | fixed + structural float |
| fixed_cashflow | fixed_income (cashflow) | atom |
| inflation | fixed_income/inflation(_unit), jarrow_yildirim | ZCIS only |
| cds | credit/cds, cds_conventions | single-name vanilla |
| fx_forward / fx_option | fx/ (22) | outright fwd + GK vanilla |
| equity_option | equity/ (33) | BS vanilla |
| commodity_option | commodity/ (23) | BS-on-carry vanilla |

### L4 engine (14) → `pricing/` + `models/` + asset-class pricers
| ng module | quarry counterpart(s) | parity gap |
|---|---|---|
| discounting | pricing/pricing_engine, core | linear DF engine (A1/A2) |
| swap / swaption | fixed_income, models | vanilla + Jamshidian analytic |
| swaption_mc | models/mc_engine, mc_exposure, mc_pricer | T-fwd single-factor MC; no general MC framework |
| spot_option / equity_option / commodity_option / fx_option / fx_forward | models/black76, asset-class | closed-form vanillas |
| cds / inflation | credit, fixed_income | leg math / Fisher |

### L5 risk (5) → `risk/` (54) + `regulatory/` (23) + `models/mc_exposure`
| ng module | quarry counterpart(s) | parity gap |
|---|---|---|
| exposure | models/mc_exposure, risk/xva | HW-1F swap exposure (fwd + risk-neutral paths, MPOR, collateral); no general product/model |
| xva | risk/xva, network_xva, hybrid_xva, simm | CVA/DVA/BCVA/FVA/KVA/MVA on one netting set; no WWR/FTD/SIMM-sensitivities |
| saccr | regulatory/counterparty, credit_rwa | IR single hedging set; no margined MF / other asset classes / collateral |
| greeks | core/greeks, risk/greeks, pathwise_greeks | bump-and-reprice on Priceable; no AAD/pathwise |
| priceable | risk/ (protocol) | redesign; enabler |

### L6 shell (2) → `core/book, trade, daily_pnl` + `risk/valuation_report` + `desks/`
| ng module | quarry counterpart(s) | parity gap |
|---|---|---|
| booking | core/book, trade, daily_pnl, settlement | Trade/Book/benefit table + realized-vs-mark; no persistence (data-spine), no lifecycle events, float realized needs fixings |
| xva_report | risk/valuation_report, desks/ | swap-only netting set (parity gap, ruled); no Book input / non-linear products |

## Untouched backlog — quarry subpackages with ~no ng counterpart

| subpackage | modules | note |
|---|---|---|
| fixed_income | 130 | the bulk; ng has ~7 vanillas (bond/swap/leg/cashflow/inflation/fra); deposit/OIS/fixings next; repo/futures/sovereign/xccy families untouched |
| credit | 93 | ng has vanilla CDS + hazard; CDO/tranche/CLN/loan/hawkes/recovery untouched |
| models | 90 | ng has flat HW; PDE/MC framework, Levy/rough/LMM/G2++/trees untouched |
| options | 61 | fully untouched |
| risk | 54 | ng has exposure/xva/greeks/saccr; portfolio/attribution/optimisation/simm untouched |
| desks | 49 | fully untouched (trading-desk layer) |
| equity | 33 | ~1 vanilla; rest untouched |
| curves | 31 | ng has 1 loglinear + bootstrap; NS/Smith-Wilson/multicurve/AAD untouched |
| numerical | 30 | ng has 2 solvers; PDE/MC/FFT/AD/QMC toolkit untouched |
| core | 29 | ~10 crossed-partial; approximation/caching/serialization/interpolation untouched |
| commodity / regulatory / structured | 23 each | commodity ~1 vanilla; regulatory ~saccr-partial; structured untouched |
| fx | 22 | ~2 vanillas |
| statistics | 17 | untouched |
| crypto / viz / pricing / ts / data / db / pe / calibration | 15/13/8/7/5/2/4/4 | untouched (calibration = the quarry's own record types) |

## Ordered parity-gap priority (foundation-first, per the CP-2 ruling)

Progress now = **quarry modules retired**, not features added. Bring the foundational spine to
realigned parity + oracle until each quarry counterpart is deletable, then move up:

1. **curves — general / bootstrapped multi-pillar curve** — *CP-2b: rate accessors done; forward_rate
   + interpolation + roll_down remain.* Nelson-Siegel / Smith-Wilson / multicurve later.
2. **models/hull_white — general-curve HW-1F** — *CP-2b: DONE (flat gap closed).* hw_calibration and
   term-structure vol to parity remain.
3. **fixed_income spine** — bond / swap / leg / deposit / **fra (CP-2b done)** / ois to parity, plus
   **fixings/seasoned-float** (the 130-module bulk). *In progress.*
4. **credit spine** — issuer/hazard curve + vanilla CDS to parity.
5. Then breadth (commodity, options, more XVA) resumes — only after the foundation supersedes its
   quarry modules.

## Checkpoint note

Refreshed at **CP-2c** (v0.46.0, 255 green): fixed_income spine advanced — seasoned FRA (fixings),
deposit, OIS (+ RateCurve/forward_rate). Deletable-bar reads applied per slice; **every spine
module's residual is the same cross-cutting trio — conventions, serialisation, multi-curve** — so
drawdown stays 0/768 until those layers land. **Named next checkpoint (CP-3): cross-cutting vs
credit spine** — see `CP2c_checkpoint.md` §4.1/§6 (proposal: a conventions layer + serialisation +
multi-curve to retire a *batch* of modules, vs continuing breadth). Awaiting Cowork ruling.
