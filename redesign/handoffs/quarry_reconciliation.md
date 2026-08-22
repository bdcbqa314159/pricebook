# Quarry reconciliation map

## TOPIC-METHOD ROLL-UP (current — supersedes the pre-replan record below)

Date: 2026-07-19 · Version: `0.73.0`. Under the **Topic Method** (#13) this doc is a *thin roll-up*;
the per-topic **manifest** is where files are tracked. The pre-replan drawdown record (CP-1…CP-4,
`fixed_income` retires, drawdown /768) is **frozen history** — those retire *ticks* were never
git-mv'd, and their modules (linear products / curve pillars) belong to **Topic 1**, retired when it
closes. The physical drawdown restarts here.

| topic | scope | status | manifest | parked |
|---|---|---|---|---|
| **Topic 0 — Foundation (L0)** | conventions · money/quantity · index identity · settlement · numerics-config (scipy, S17) · serialisation pattern | **PARKED** (gate green: F1–F4 + S1–S17 landed; both regression oracles pass) | `parked/topic-00-foundation/MANIFEST.md` | **13** |
| **Topic 1 — the first complete vertical (L0→L6)** | curves (C1) · Black models + vol (C2) · risk + shell (C3) | **CLOSED v0.98.0** — the spine proven L0→L6 for a full instrument lifecycle | `parked/topic-01-yield-curve/MANIFEST.md` | **6 parked** (git-mv'd at T1 close) |

**Files parked: 13 / 793** (reported, never steered — #12). **This is the single source of truth for
the drawdown denominator** (CLAUDE.md §4/§6 point here, not to a hardcoded literal). **Convention
(pinned):** the total is *every `.py` file in the original quarry* = files remaining in
`python/pricebook/` (**780**) + files already parked under `parked/` (**13**) = **793**, `__init__.py`
markers included. Verified 2026-08-01 by direct count.

**Deletable (the §4 drawdown bar): 19 / 793 — ALL PHYSICALLY PARKED** (13 at Topic-0 close **+ 6 at the
Topic-1 close**, v0.98.0). The 6 Topic-1 modules below are now git-mv'd into `parked/topic-01-yield-curve/`:
- **slice-3 retire read (2):** `curves/bootstrap.py` and `core/discount_curve.py`, superseded by ng
  `calibrate` / `DiscountCurve` (deposits→FRAs→futures→swaps dual-curve sequential, minus deferred
  capabilities).
- **slice-4 retire read (3):** `curves/ncurve_solver.py`, `curves/global_solver.py`,
  `curves/multicurve_solver.py`, all superseded by ng's **simultaneous** solve (scipy LM). Hand-rolled
  damped Newton + FD Jacobian are §7bb-shed (scipy); analytic Jacobian → `deferred→C3/AAD`; N-curve > 2
  → `deferred→3rd-projection-curve`; 0 ng consumers of any shed bit (evidence: `CP_slice4_global_solve.md`).
  **`multicurve_solver`: full cross ratified (Cowork), 3 ticks stand, 19/793.** §4 grep confirmed 0
  production/dynamic consumers of `multicurve_newton`/`validate_curve`/`curve_analytical_jacobian` (bare-
  name search across `python/**` source+tests: only quarry-internal test files, which retire with the
  quarry). `validate_curve`/`curve_analytical_jacobian` forward-linked → C3 risk.
- **slice-5 retire read (1) — `core/forward_interpolation.py` (housekeeping 2026-08):** full cross.
  Its 3 methods map — `MONOTONE_CONVEX` → ng `HAGAN_WEST` (slice 5); `PIECEWISE_CONSTANT` (flat forwards ≡
  linear log-df) → ng `LOG_LINEAR`; `PIECEWISE_LINEAR` → shed (`deferred→` future interp-method consumer).
  Interpolate-on-forwards-then-integrate architecture superseded by ng's DF-interpolation + `forward()`
  atom. §4 grep: 0 production/dynamic consumers (only its own `test_forward_interpolation.py`; the
  `test_commodity_vol_surface` hit is a method-name collision, not an import).

- **slice-6c retire read (0 ticks — ratified):** the xccy/CSA/FX quarry suite (`fixed_income/xccy_swap`,
  `xccy_basis`, `xccy_bond`, `xccy_swaption`, `csa`, `credit/bilateral_csa`, `risk/collateral_*`,
  `curves/ndf_implied`, and all of `fx/*` — `multi_currency_curves`, `fx_basis`, `fx_forward*`, `ndf`,
  `prdc`, `fx_smile_cube`, …) is **NOT superseded** by 6c. ng's 6c delivers one minimal foreign-collateral
  *discount curve* (constant-notional, single-curve-per-ccy, basis-as-spread); the quarry suite is
  full-featured FX/XVA breadth (vol surfaces, exotics, NDFs, collateral optimisation, bilateral CSA).
  A curve does not supersede a product+risk suite → **0 ticks (correct, ratified C1 domain depth, doc 18
  §4/§9)**. Named crossing: these files retire with the **FX / XVA topics**, not Topic 1. Drawdown
  unchanged at **19 / 793**.

- **C2 slice-1 retire read (0 ticks — PARTIAL crosses):** `models/black76.py` (Black-76 *price* crosses → ng
  `black()`; `black76_delta/gamma/vega/theta` → C3 risk, `bachelier_*` → 2nd vol consumer) and
  `options/capfloor.py` (single caplet *price* crosses → ng `Caplet`+`price_caplet`; floorlet → its consumer,
  `CapFloor` strip → B5, `strip_caplet_vols` → vol-calibration slice, `calibrate_capfloor_sabr` → SABR/B4).
  Both remain resident (greeks/Bachelier/strips/stripping/SABR deferred) → **tick 0, drawdown 19/793**. Full
  evidence: `CP_C2_slice1_black_caplet.md` §Drawdown.

- **C2 slice-2 retire read (0 ticks — PARTIAL cross):** `options/swaption.py` — the European
  Black×annuity swaption *price* crosses (ng `Swaption` + `price_swaption`, `PV = N·rpv01·black(S,K,vol,t)`
  on the shared atoms). Resident/deferred: `price_swaption_sabr_hw` (SABR/HW blend) → SABR slice (B4);
  greeks → C3; cash-settled/IRR → first cash-settled consumer; the separate `swaption_vol_cube.py`,
  `bermudan_swaption*.py`, `midcurve_swaption.py` → their own slices (cube/(A)-fork Bermudan/midcurve). Partial
  → **tick 0, drawdown 19/793**. Evidence: `CP_C2_slice2_black_swaption.md` §Drawdown.

- **C2 slice-3 retire read (0 ticks — PARTIAL crosses):** `options/capfloor.py` — the caplet-vol
  *term-structure strip* now crosses (ng `strip_caplet_vols`), but `calibrate_capfloor_sabr` + the smile/
  cube generation stay resident → SABR/smile slice; the `CapFloor` strip container + greeks were already
  deferred (slice 1). `options/vol_calibration.py` — a MULTI-ASSET surface calibrator; only the IR caplet
  term-structure portion of `calibrate_ir_surface` is superseded — `calibrate_fx/equity/commodity_surface`,
  `CalibratedSABRNode`, `CalibratedVolSurface` travel with their asset topics (FX/equity/commodity) and the
  SABR slice. Both remain resident → **tick 0, drawdown 19/793**. Evidence: `CP_C2_slice3_caplet_vol_strip.md`.

- **C3 opening retire read (0 ticks — PARTIAL crosses):** the generic bump-and-reprice greek *mechanism*
  now exists at L5 (`ir_delta` DV01 + `vol_vega` on the `Priceable` protocol), **discharging the accrued
  "greeks exist at L5" →C3 forward-links** (curve DV01, caplet/swaption vega — any priceable gets them for
  free). But every quarry risk file retains breadth beyond ng's parallel first-order pair: `risk/greeks.py`
  (`bump_greeks` — gamma/theta resident), `curves/key_rate_risk.py` (key-rate/bucketed DV01 → bucketed-risk
  consumer), `curves/curve_risk.py` (`curve_jacobian`/rolldown → C3-later/AAD), `risk/pathwise_greeks.py`
  (AAD), `curves/curve_bumper.py` (pillar bumps), `core/greeks.py` (`Greeks` aggregate → portfolio consumer);
  `models/black76.py` delta/gamma/theta stay partial (analytic greeks → their slice). All partial →
  **tick 0, drawdown 19/793**. Evidence: `CP_C3_opening_risk_greeks.md`.

- **L6 opening retire read (0 ticks — PARTIAL cross):** the frozen-description + mark + portfolio-DV01
  *concept* crosses (ng `Trade`/`Book` + `mark`/`mark_book`/`book_dv01`; the quarry's `Trade.pv(ctx)`
  self-pricing is **realigned** to the shell `mark()`, shed not migrated). But `core/trade.py`/`core/book.py`
  stay resident: serialisation (`to_dict`/`from_dict` → persistence slice), mutable `Book.add`, `Desk`,
  `Position`/`BookLimits`/limit-breaches, tenor-bucketed DV01, counterparty netting (→ stateful/desk/limits
  slices), **and ~10+ un-crossed desk consumers** (`desks/*`, `credit_risk` — deleting would orphan them).
  Realized-P&L/benefit-table/`BookedTrade` travel with the next (stateful) L6 slice. Partial → **tick 0,
  drawdown 19/793**. Evidence: `CP_L6_opening_shell.md`.

- **L6-realized retire read (0 new ticks — PARTIAL, but the Topic-1 PARKING event fires):** the
  realized-vs-mark / benefit-table *concept* crosses (ng `BookedTrade`/`realized`/`total` + the invariant-6
  engine exclusion; the quarry's `Trade.pv(ctx)` self-pricing is realigned to the shell `mark`). But
  `core/trade.py`/`core/book.py`/`core/fixings.py` stay resident — serialisation, `Desk`, limits/positions/
  netting, lifecycle events, mutable fixing store + file I/O, AND ~10 un-crossed desk consumers — so the
  booking modules do NOT tick. **Because this CLOSES C3 and Topic 1**, the 6 accumulated Topic-1 deletables
  were **physically parked** (one `git-mv` event into `parked/topic-01-yield-curve/`): `bootstrap`,
  `discount_curve`, `ncurve_solver`, `global_solver`, `multicurve_solver`, `forward_interpolation`. ng is
  unaffected (never imports the quarry); the merge gate never touches `python/`. Drawdown **19/793, now all
  physically parked** (13 Topic-0 + 6 Topic-1) — the numerator is unchanged (no new tick; booking partial),
  the *parking* is the physical realization. Evidence: `CP_Topic1_close.md`.

The six Topic-1 deletables are now **physically parked** (git-mv'd at the T1 close, one event, doc 18 §9).
Full retire evidence + resident inventory (file:line) + forward-links are in the Topic-1 MANIFEST retire
record + `CP_slice3`/`CP_slice4` checkpoints. *(The historical `/768` — CLAUDE.md and the
slice-1/2 records — is the same universe minus the 25 `python/pricebook/__init__.py` package markers;
`793 − 25 = 768`. Not wrong, a different convention; superseded by this all-`.py` count so the series is
consistent. The **PRE-REPLAN RECORD's** separate `7 / 768` below is frozen product-vertical-era history
and is left untouched.)* Topic 0's **13** files (11 covered —
`day_count`, `calendar`, `schedule`, `rate_index`, `currency`, `settlement`, `numerical_config`,
`interpolation`, `solvers`, `serialisable`, `serialization` — plus `data_registry` **dead** and
`notional` **absorbed** into `Money`/`Leg`) are git-mv'd to `parked/topic-00-foundation/`.
`core/fixings` is **reassigned→market-data** (its immutable read model `FixingHistory` is the L0 type;
the mutable store + file I/O is not L0 — not parked here). Topic 1 does not begin until this gate is
ratified by Cowork.

---

## PRE-REPLAN RECORD (frozen history — superseded by the roll-up above)

Date: 2026-07-17 (refreshed at CP-2c)   ·   Version: `0.46.0`.

Delivers the CP-2a ruling (`rulings_CP1.md`): the honest drawdown baseline and the ordered
parity-gap list, before parity-depth slices begin. **"Crossed" = the quarry module it supersedes
is *deletable* (realigned parity reached)** — a simplified skeleton is a *partial* cross with a
recorded gap, not a cross (`CLAUDE.md §4`).

## RETIRED (deletable) — the drawdown numerator

**Drawdown = 7 / 768 (0.91%).** Quarry modules superseded (CP-3 #1–#5 + tail bond, fixed_leg).

### `fixed_income/fixed_leg.py` → superseded by `foundation/cashflow.py` + `products/leg.py` + `products/swap.py` (v0.53.0) — CP-3 tail · **no-code tick** (§4.5 "ticks immediately")
- **Superseded (no new ng code):** the `Cashflow` atom → **promoted** to `foundation/cashflow.py`
  (ng stores amount as `Money` + `Accrual` period; the quarry's decomposed notional/rate/year_frac is
  shed — year_frac derives from the accrual); `FixedLeg` construction → `products/leg.py`
  `fixed_coupon_cashflows`; the leg container → `products/swap.py FixedLeg`; `FixedLeg.pv` → the engine
  (discounted cashflows). `Cashflow.to_dict` → ng `Cashflow.to_dict`/`from_dict` (already in the
  property sweep).
- **Consumer analysis (§4):** the 3 external quarry `FixedLeg(` sites are `swap.py` (un-crossed →
  `deferred→swap`) and `ois.py`/`bond.py` (**already deletable** → retire with them). Quarry `Cashflow`
  has 0 external production instantiations.
- **`deferred→swap` (+ options/structured/desks):** `FixedLeg.annuity` / `weighted_annuity` (RPV01
  building blocks) — consumed only by un-crossed modules (swaption, CMS, zc_swap, swap/asset desks); no
  crossed ng consumer. ng exposes annuity as an **engine/curve building block** when swap crosses
  (CLAUDE.md §3 "building blocks reused upward"), not as a product method.
- **Oracle:** no new code, so no new red — the tick rests on ng's existing green leg/cashflow/swap/bond
  oracles (which already exercise `fixed_coupon_cashflows` + `Cashflow` round-trip) + the consumer
  analysis above. `floating_leg.py` is a **separate** retire (projection/multi-curve `pv`) — travels
  with the swap / multi-curve slice.

### `fixed_income/bond.py` → superseded by `products/fixed_rate_bond.py` (v0.52.0) — CP-3 tail ⚠️ heaviest retire, flagged for Cowork CP-4 spot-check
- **Superseded:** the bond **product** (coupon+redemption cashflows) → ng `FixedRateBond`; curve
  pricing → `DiscountingEngine`; **`accrued_interest` + clean-vs-dirty → engine A2 decomposition**
  (`PricingResult.accrued`/`clean`).
- **Consumer analysis (§4/§4.5):** all 8 production instantiations of quarry `FixedRateBond` are in
  **un-crossed** modules (`desks/api` ×6, `benchmark_bonds`, `sukuk`). No **crossed** ng module consumes
  any bond yield-analytic (the only `duration`/`convexity` in ng is SA-CCR's unrelated *supervisory*
  duration). ⇒ nothing is needed-now beyond the product.
- **`deferred→` (large surface — the judgment call in this retire):** the whole yield-analytics suite
  (`yield_to_maturity`, `macaulay/modified_duration`, `convexity`, `dv01_yield`, `*_sc`) →
  `desks/api`+`benchmark_bonds`+`sukuk`; `from_convention` → `composite_convention`+`esg_bonds`+
  `supranational`+`sovereign_bonds`. Per the spine these are **L4/L5 engine analytics, not product
  methods** — built when a consumer crosses (§6b), never cloned onto the pure-data product.
- **`deferred→persistence`:** serialisation (added early, §4.5 — never blocked the tick).
- **Tick rationale:** under the ratified consumer-analysis rules the product is superseded and the
  analytics have no crossed consumer ⇒ deletable. But it rests on the largest deferred surface yet, so
  it is explicitly flagged for Cowork spot-check (may un-tick; reversal is cheap).

> **Serialisation classification (Cowork CP-3 ruling §4.5, `rulings_CP3.md`):** across all five CP-3
> retires, `to_dict`/`from_dict` is **`deferred→persistence`**, NOT a genuine residual — ng has no
> serialisation consumer yet (the DB `from_dict` dispatcher is quarry-side; the persistence/data-spine
> layer is un-crossed). It was **built early by deliberate policy** (adding it in-passing beats a mass
> retrofit after persistence lands), never to justify a tick. Binding rule (CLAUDE.md §4):
> **serialisation never blocks a tick** — where cheap and already in the module, add it; else retire
> and forward-link `→persistence`. The five ticks stand; nothing to rework.

### `fixed_income/zero_coupon_bond.py` → superseded by `products/fixed_cashflow.py` (v0.51.0) — CP-3 #5
- **Genuine residual — CLOSED:** `to_dict`/`from_dict` (reuses shared `Cashflow`/`Money` encoders). A
  ZCB *is* a single fixed cashflow (`Face·DF(T)`), priced by the existing `DiscountingEngine`.
- **Consumer analysis (§4):** quarry `ZeroCouponBond` has **no external production instantiation** (only
  its own docstring; production ZCBs flow via `sovereign_bonds.py from_convention`). The money-market
  yield analytics (`price_from_yield_*`, `yield_*`, `modified_duration`, `dv01`) have 0 production
  consumers (only `tests/test_sovereign_bonds.py`); the quarry's dedicated `fixed_income/tbill.py`
  (`TreasuryBill`) is the real home for T-Bill conventions ⇒ ZCB's copies are **`dead` duplicates**.
- **`shed:` `from_convention` = `deferred→sovereign_bonds`** — see the forward-link on that backlog row.

### `fixed_income/ois.py` → superseded by `products/ois.py` (v0.50.0) — CP-3 #4
- **Genuine residual — CLOSED:** `to_dict`/`from_dict` (round-trip + `schema_version`); DB-dispatcher path.
- **Consumer analysis (§4):** quarry `OISSwap` has 2 production instantiations — `desks/api.py:641`
  (`deferred→desks/api`) + `OISConvention.create_swap` (in-module, retires with it). ng supersedes
  single-curve pricing (`engine/ois.py`, `OIS == vanilla IRS`); `pv_ctx` multi-ccy branch = deferred.
- **`shed:` `from_convention` = `dead`** — sole caller `tests/test_convention_factory.py` (quarry test).
- **Refactor (rule of two ×2):** `Accrual.to_dict`/`from_dict` (FRA + coupon cashflows) and
  `Cashflow.to_dict`/`from_dict` (deposit + OIS legs) lifted to `foundation/cashflow.py`; FRA + deposit
  refactored under green oracles. Leg encoding stays inlined in OIS (its only serialising consumer).

### `fixed_income/fra.py` → superseded by `products/fra.py` (v0.49.0) — CP-3 #3
- **Genuine residual — CLOSED:** `to_dict`/`from_dict` (round-trip + `schema_version`). Same
  production-reachable path as #1/#2 — the DB dispatcher (`db.py from_dict`).
- **Consumer analysis (§4 phantom-residual rule):** the quarry `FRA` class has **one** production
  instantiation — `desks/api.py:273` `FRA(...).pv(curve, projection_curve)`, a **multi-curve** path.
  `desks/` is un-crossed ⇒ `deferred→desks/api` (multi-curve travels with it), **not owed now**. ng
  supersedes single-curve pricing via `engine/fra.py` (+ seasoned fixings).
- **`shed:` `from_convention` = `dead`** — sole caller `tests/test_convention_factory.py::`
  `test_fra_from_convention` (quarry test, retires with quarry); 0 production callers.
- **Refactor:** rule of two fired at the 2nd serialising product → `Money.to_dict`/`from_dict` lifted
  to `foundation/money.py` (deposit + FRA consume it); deposit refactored under its green oracle.

### `fixed_income/deposit.py` → superseded by `products/deposit.py` (v0.48.0) — CP-3 #2
- **Genuine residual — CLOSED:** `to_dict`/`from_dict` (round-trip + `schema_version`). Same residual
  as CP-3 #1: the production-reachable reconstruction path is the DB dispatcher (`db.py:310`
  `from_dict(json.loads(...))`), so a product must round-trip to be persistable.
- **Needed functionality covered:** the deposit *concept* (place `face`, receive `face·(1+rτ)`) as
  the two-cashflow `Deposit` product priced by `DiscountingEngine`; the **curve-pillar role**
  (`discount_factor = 1/(1+rτ)`) superseded by `DepositQuote` in `bootstrap_discount_curve`.
- **Evidence method** (CLAUDE.md §4): bare-name grep `Deposit` / `Deposit(` / `Deposit.from_convention`
  across `python/` source + tests; checked the DB serialisation dispatcher for dynamic reconstruction.
- **Key evidence — the quarry `Deposit` class has ZERO production instantiations:** `grep 'Deposit('
  python/pricebook` → 0 hits (bootstrap uses a loose `deposit_day_count` param + inline `year_fraction`,
  never the class). Every `Deposit(` site is a test.
- **`shed:`**
  - **`dead`** — `from_convention` (1 caller: `tests/test_convention_factory.py::test_deposit_from_convention`,
    a quarry test that retires with the quarry; 0 production callers — same interpretation as CP-3 #1's
    `mc_antithetic`); `discount_factor` / `pv` / `pv_ctx` / `year_fraction` / `cashflow` properties
    (0 production consumers; superseded by ng's product + engine + `DepositQuote`).
- **Ruling-refinement flag (for Cowork):** CP-3 #2 was scoped as *conventions/RateIndex → retire
  deposit*. The evidence shows deposit's residual was **serialisation, not conventions** — deposit has
  no production consumer of conventions at all. Conventions/RateIndex is real cross-cutting infra but
  belongs to its genuine consumer (per-currency curve construction, `curves/curve_builder`
  `get_conventions`), to be built when *that* is crossed — not speculatively here. **No conventions
  code was written.**

### `core/numerical_config.py` → superseded by `foundation/numerical_config.py` (v0.47.0) — tick confirmed (Cowork spot-check, `rulings_spotcheck_retire_1.md`)
- **Genuine residual — CLOSED:** `to_dict`/`from_dict` (round-trip + `schema_version`) and `replace`.
- **Needed functionality covered:** `mc_paths`, `mc_seed` (ng engines consume these) + `fd_bump`
  (ng's DV01 knob). Validation of positive knobs preserved.
- **Evidence method** (per CLAUDE.md §4 protocol — narrow `\.name` was too weak, missed constructor
  kwargs): bare-name grep across `python/` **source + tests**, + constructor-kwarg (`NumericalConfig(name=…)`)
  + serialisation-round-trip check. Hit counts recorded below.
- **`shed:` — the 12 knobs, re-classified (Cowork §2):**
  - **`dead`** (no consumer / no identifiable future one): `mc_antithetic` (3 hits — all the module's
    own test), `mc_use_sobol` (1), `mc_brownian_bridge` (0), `extra` (0 on `NumericalConfig.extra`;
    the 123 bare hits are a *different* `.extra` on calibration types).
  - **`deferred→X`** (identifiable future consumer in the un-crossed `numerical/` toolkit — forward-linked
    on X's backlog row):
    `cos_n`(6)/`cos_L`(0)→`numerical/_fourier`; `pde_time_steps`(4)/`pde_space_steps`(1)/`pde_n_std_devs`(0)→`numerical/_pde`;
    `tree_steps`(5, all `compare_engines` local kwarg)→`numerical/_trees`; `integration_tol`(2)/`integration_max_iter`(0)→`numerical/_integrate`;
    `rootfinder_tol`(2)/`rootfinder_max_iter`(0)→`numerical/_rootfinding`.
  - **Correction note:** my first pass labelled all 12 `dead` from an attribute-only grep; the audit
    caught `cos_n` reached via a serialisation-round-trip constructor kwarg → *not* `dead`. Deferred ≠ dead
    (deferred carries the obligation below). Tick stands: no ng module needs any of the 12 now.

## Headline

- **Quarry: 768 modules. New tree: 55 modules. Deletable: 7** (`core/numerical_config`,
  `fixed_income/deposit`, `fixed_income/fra`, `fixed_income/ois`, `fixed_income/zero_coupon_bond`,
  `fixed_income/bond`, `fixed_income/fixed_leg`).
- **Drawdown = 7 / 768 (0.91%)** — CP-3 serialisation cluster (#1 config, #2 deposit, #3 FRA, #4 OIS,
  #5 ZCB) + tail (bond, fixed_leg). The rest of the ng tree is a *simplified parallel build* still short of superseding its counterparts (below).
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
| fixed_income | 130 | the bulk; ng has ~7 vanillas (swap/leg/cashflow/inflation); **deposit + fra + ois + zero_coupon_bond + bond + fixed_leg RETIRED (v0.48–0.53)**; `floating_leg` deferred→swap/multi-curve; repo/futures/sovereign/xccy families untouched. **Deferred — on crossing `sovereign_bonds.py`:** ZCB + bond `from_convention` path (per-currency conventions), deferred from the ZCB (v0.51) + bond (v0.52) retires. **Deferred — on crossing `esg_bonds`/`supranational`/`composite_convention`:** bond `from_convention`. **Deferred — `tbill.py` carries its own T-Bill analytics** (ZCB copies shed dead). **Deferred — on crossing `swap.py`:** expose `annuity`/`weighted_annuity` (RPV01) as an engine/curve building block, deferred from the `fixed_leg` retire (v0.53); `floating_leg` retire (projection pv) also travels with swap/multi-curve. |
| credit | 93 | ng has vanilla CDS + hazard; CDO/tranche/CLN/loan/hawkes/recovery untouched |
| models | 90 | ng has flat HW; PDE/MC framework, Levy/rough/LMM/G2++/trees untouched |
| options | 61 | fully untouched |
| risk | 54 | ng has exposure/xva/greeks/saccr; portfolio/attribution/optimisation/simm untouched |
| desks | 49 | fully untouched (trading-desk layer). **Deferred obligation — on crossing `desks/api.py`:** the bond **yield-analytics** suite (`yield_to_maturity`, `macaulay/modified_duration`, `convexity`, `dv01_yield`, `*_sc`), deferred from the bond retire (v0.52.0) — build as **L4/L5 engine analytics** (spine), not product methods. Same suite deferred to `benchmark_bonds`, `sukuk`. |
| equity | 33 | ~1 vanilla; rest untouched |
| curves | 31 | ng has 1 loglinear + bootstrap; NS/Smith-Wilson/multicurve/AAD untouched |
| numerical | 30 | ng has 2 solvers; PDE/MC/FFT/AD/QMC toolkit untouched. **Deferred obligations from `core/numerical_config` retire (v0.47.0)** — on crossing, add the knob back to `NumericalConfig`: `_fourier`←`cos_n,cos_L`; `_pde`←`pde_time_steps,pde_space_steps,pde_n_std_devs`; `_trees`←`tree_steps`; `_integrate`←`integration_tol,integration_max_iter`; `_rootfinding`←`rootfinder_tol,rootfinder_max_iter`. |
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
3. **fixed_income spine** — bond / swap / leg / **deposit (RETIRED v0.48.0)** / **fra (CP-2b done)** /
   ois to parity, plus **fixings/seasoned-float** (the 130-module bulk). *In progress.*
4. **credit spine** — issuer/hazard curve + vanilla CDS to parity.
5. Then breadth (commodity, options, more XVA) resumes — only after the foundation supersedes its
   quarry modules.

## Checkpoint note

Refreshed at **CP-3 #2** (v0.48.0): **`fixed_income/deposit` retired → drawdown 2/768.** The retire
read refined the ruling: deposit's residual was **serialisation, not conventions** (it has zero
production consumers; `from_convention` is a quarry-test-only `dead` feature). No conventions/RateIndex
code was written — it re-aims at its genuine consumer (per-currency curve construction) when crossed.
Cowork ratified the correction (`rulings_CP3_correction.md`): serialisation through-line confirmed,
**§4 phantom-residual rule added** (residuals need consumer evidence; re-derive by consumer analysis at
retire time; gaps likely overstated → drawdown faster than the map claims). CP-3 #3 (FRA), #4 (OIS),
#5 (ZCB), tail (bond, fixed_leg) done via consumer-analysis retire-reads. CP-3 checkpoint written + ruled
(`rulings_CP3.md`, §4.5: serialisation deferred→persistence, built-early, never blocks a tick).
**inflation.py — held PARTIAL CROSS (v0.54, NOT ticked, drawdown stays 7):** multi-product module;
ng supersedes only ZCIS (`ZeroCouponInflationSwap`+`ZCISEngine`+A5 real curve, which also supersedes
`CPICurve`). `YoYInflationSwap` = `dead` (test-only); `InflationLinkedBond` = `deferred→desks/api`+
`inflation_indices` (un-crossed) but a whole UNBUILT product. ZCIS serialisation added (build-early).
**⚠️ CP-4 Cowork question: how to retire a multi-product module when ng has built only some of its
products** (delete-defining-whole-unbuilt-products is a new pattern beyond bond's deferred methods).
**Vanilla cluster now closed** (deposit/fra/ois/zcb/bond/fixed_leg retired; inflation partial).
**Next = the SWAP decision → CP-4.** **Watch:** swap has 29
production instantiations (curve pillars/XVA) — load-bearing, NOT a serialisation-only retire; its real
residual is larger (multi-curve/curve-build role). **⚠️ Bond (v0.52) is the heaviest tick — large
deferred yield-analytics surface; flagged for Cowork CP-4 spot-check (may un-tick).** **CP-4 checkpoint
at first of: (a) vanilla cluster retired + swap decision, (b) 6 slices since CP-3, (c) multi-curve introduced.**
