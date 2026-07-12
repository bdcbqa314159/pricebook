# Changelog

All notable changes to the new tree (`pricebook_ng`) are recorded here, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. `0.x` = migration
in progress; `1.0.0` is reached exactly when the quarry (`python/pricebook/`) is empty.

## [Unreleased]

## [0.3.0] - 2026-07-12

### Added
- **Risk relocated to L5 on the `Pricable` protocol** (spine structural fix).
  - `risk/pricable.py`: `Pricable` — a `snapshot -> PV` closure that risk consumes;
    factories `discounting_pricable` (linear products, any engine over a
    `DiscountingModel`) and `hull_white_pricable` (rebuilds HW under the snapshot).
  - `risk/greeks.py`: generic `dv01` (central-difference bump-and-reprice) and
    `bump_rate` (parallel shift). One `dv01` prices rate delta for a cashflow, a
    bond, a swap, and an HW swaption — the swaption rebuilds the model under the
    bumped snapshot (Amendment A1). **No `isinstance`-on-product ladders.**
  - Oracle: `dv01` matches the analytic sensitivity for a single cashflow and a
    bond (`< 1e-6`); is generic over any `Pricable` (a raw closure); and for the HW
    swaption equals a manual rebuild-and-reprice bump (the model-rebuild path).

### Changed
- The Slice-0 `risk/dv01.py` (specific to `DiscountingModel`) is replaced by the
  generic `risk/greeks.dv01` on the `Pricable` protocol; its analytic-vs-FD oracle
  is preserved through the new API.

### Deferred
- Higher greeks (gamma, vega), pillar-wise bumps of a bootstrapped curve, and
  XVA/RWA — all land on the same `Pricable` protocol in later slices.

## [0.2.0] - 2026-07-12

### Added
- **Amendment A3 — Product / Trade / Book + the benefit table (L6 shell).**
  - `shell/booking.py`: `Trade` (a collection of products + a start date), `Book`
    (a collection of trades), and `BookedTrade` with the **benefit table** —
    `realized(as_of)` sums cashflows that have already paid as actual cash, **never
    discounted**. `value(...)` aggregates the products' marks (dirty PV + accrued).
  - Oracle: realized at issue is 0; realized sums paid cashflows undiscounted;
    realized + remaining nominal = total nominal; at end of life realized = total
    and the mark is 0; a `Book` aggregates realized across trades.

### Changed
- **Renamed the L2 atom `instrument -> product`** (Amendment A3): the
  `instruments/` package is now `products/`; `FixedCashflowTrade -> FixedCashflow`;
  the engine protocol `CashflowInstrument -> CashflowProduct`. "Trade" is now an
  L6 concept (a collection of products), freeing the name. Behaviour-preserving.
- CI layer tier bumped to `--layer 6` (the slice reaches the L6 shell).

### Deferred
- Realized P&L for float legs (needs fixings) — wired with a seasoned-float slice.
- Per-product model dispatch in `Trade.value` (all products are linear today, so
  one `DiscountingModel` suffices); a registry/facade arrives when a trade mixes
  model families.

## [0.1.0] - 2026-07-12

### Added
- **Amendment A2 — valuation is temporality-aware.** The engine partitions a
  product's cashflows by `model.market.valuation_date`:
  - cashflows on or before valuation are **historical** — excluded from PV (the
    shell settles them), never discounted; the "fail on past cashflow" guard is
    retired in favour of **segment-and-settle**.
  - future cashflows discount from valuation.
  - `PricingResult` is now a **decomposition**: `pv` (dirty), `accrued`
    (earned-but-unpaid, nominal), and `clean = pv - accrued`.
  - `Cashflow` gains an optional `Accrual(start, end, day_count)`; fixed-leg
    coupons carry it, so a **seasoned bond** accrues the current period on its own
    day count.
  - Oracle (closed-form, exact): seasoned bond excludes paid coupons and matches
    the sum over remaining flows; forward-starting prices only future flows;
    accrued matches the day-count fraction; `dirty = clean + accrued`; a cashflow
    exactly on the valuation date is historical.
  - Behaviour-preserving for at-issue/forward pricing (accrued = 0): every prior
    oracle stays green.

### Deferred
- Fixing resolution for seasoned **float** legs (reset ≤ valuation → realized
  `FixingHistory`) — no such instrument present yet; wired when one arrives (A3+).

## [0.0.11] - 2026-07-12

### Changed
- **Amendment A1 — the engine depends on the model, not a market argument.**
  `price(product, model, numerics)`: the model carries the `MarketSnapshot` it was
  calibrated to (`model.market`); the engine reaches curves/valuation-date through
  it. Market/model mismatch is now structurally impossible (there is no second
  market to pass). Behaviour-preserving — every S00–S08 oracle stays green.
  - New L3 `DiscountingModel(market)` (thin model for linear products) and a
    `CalibratedModel` protocol.
  - `HullWhite` now carries a `MarketSnapshot` instead of a bare curve.
  - `DiscountingEngine` / `SwapEngine` / `SwaptionEngine` drop the `market` arg;
    `book.value` builds the model for the date's snapshot; `dv01` bumps the
    snapshot and rebuilds the model (risk flows through the model).
  - `FixingHistory` is now first-class on `MarketSnapshot` (empty default; the
    economy = curves + fixings). Its seasoned-period consumer lands with A2.
  - Oracle: engine-model binding test (no `market` param; a model only prices
    against its own snapshot) + unchanged PVs.

### Ratified
- `CLAUDE.md` §0/§2/§3 and `redesign/02_spine.md` Amendments A1/A2/A3 (Cowork).

## [0.0.10] - 2026-07-12

### Added
- **S08 — Hull-White European swaption (Jamshidian).**
  - `instruments/swaption.py`: `Swaption(expiry, swap)` — a European option on a
    forward-starting `VanillaSwap` (payer/receiver via `swap.pay_fixed`), pure data.
  - `engine/swaption.py`: `SwaptionEngine` — Jamshidian decomposition into a
    portfolio of HW ZCB options (S07), with a bisection solve for the critical
    rate `r*` that prices the coupon bond at par.
  - `models/hull_white.py`: `zero_bond` reconstitution `P(T,S) = A e^{-B r}` (the
    state-dependent bond price the Jamshidian solve needs).
  - `foundation/solvers.py`: `bisect_root` — bracketed bisection (first root-finder
    of the L0 toolkit, migrated on demand).
  - Oracle (closed-form, exact): put-call parity `payer - receiver ==
    P(0,T0)*notional - sum(c_i P(0,t_i))`; ATM symmetry (payer == receiver at the
    forward par rate); `sigma -> 0` collapses to the discounted intrinsic
    `max(forward swap PV, 0)`, cross-checked against the S06 `SwapEngine`.
  - Quarry: `fixed_income/` (swaption), `pricing/`, `core/solvers.py`.

### Deferred
- MC engine + analytic-vs-MC convergence oracle (the **next slice, S09**), where
  `NumericalConfig` gains the MC knobs (`mc_paths`, `mc_seed`).

## [0.0.9] - 2026-07-12

### Added
- **S07 — Hull-White 1F analytic core (first L3 model).**
  - `models/hull_white.py`: `HullWhite(a, sigma, curve)` fitted to a flat curve
    (reprices it by construction); the `B(t,T)` factor and the closed-form
    European option on a zero-coupon bond (Brigo & Mercurio 3.40-3.41).
  - `foundation/distributions.py`: `norm_cdf` via `math.erf` — first piece of the
    L0 numerical toolkit, migrated on demand (dependency-free).
  - Oracle (all closed-form, exact < 1e-12): the model refits the initial curve;
    `B(t,T) -> (T-t)` as `a -> 0`; ZCB-option put-call parity
    `call - put == P(0,S) - K*P(0,T)`; `sigma -> 0` collapses to discounted
    intrinsic; and a match against an independent recompute of the ZBC formula.
  - Quarry: `models/` (hull_white), `numerical/_distributions.py` (norm_cdf).

### Deferred
- HW swaption engine (Jamshidian decomposition) + analytic-vs-MC convergence —
  the **next slice (S08)**, where the MC engine and `NumericalConfig` MC knobs land.
- General (bootstrapped) curve fit, time-dependent `a`/`sigma`, and the L3/L4
  boundary for analytic-model option formulas (flagged for the L3 report).

## [0.0.8] - 2026-07-12

### Added
- **S06 — vanilla single-curve interest-rate swap.**
  - `instruments/swap.py`: `FixedLeg`, `FloatLeg` (structural — schedule + face
    only), `VanillaSwap`, `SwapTerms`, and a `vanilla_swap(...)` builder.
  - `engine/swap.py`: `SwapEngine` — discounts the fixed leg (reusing
    `DiscountingEngine`) and computes the float leg's coupons as the curve's
    forwards (`DF(a)/DF(b) - 1`) at pricing time; NPV is payer/receiver aware.
  - `instruments/leg.py`: shared `fixed_coupon_cashflows` — one definition of a
    fixed leg's coupons, now used by both the bond and the swap (rule of two).
  - Oracle: par swap reprices to zero NPV; float leg telescopes to
    `notional*(DF(start)-DF(maturity))`; off-par NPV `= notional*annuity*(par-rate)`;
    receiver `= -payer`; and a swap matching an S03 bootstrap input reprices to ~0.
  - Quarry: `fixed_income/` (swap / fixed + float legs).

### Changed
- `fixed_rate_bond` builds its coupons via the shared `fixed_coupon_cashflows`
  (behaviour identical; the S04 bond oracle stays green).

### Deferred
- Multi-curve (OIS discount / IBOR projection), basis spread on the float leg,
  and an engine registry/facade selecting engine per instrument — no consumer yet.

## [0.0.7] - 2026-07-12

### Changed
- Signature discipline (CLAUDE.md §3b) is now enforced by the **CI ruff step**
  (`ruff check src/pricebook_ng`, rule `PLR0913`/`max-args=5`) rather than a
  bespoke `verify.py signatures` check — aligning with `redesign/09` ("same CI
  ruff step, not a bespoke checker"). `ruff.toml` is unchanged (it's the config
  that step reads); `verify.py signatures` is removed.
- Ratified the `CLAUDE.md §3b` and `redesign/09` guardrail text (previously
  uncommitted).

## [0.0.6] - 2026-07-12

### Added
- **Signature discipline (CLAUDE.md §3b).** New `verify.py signatures` check —
  the 5-argument ceiling (ruff `PLR0913`/`max-args=5`), enforced in the merge
  gate and CI; `self`/`cls` and `*args`/`**kwargs` are not counted. Root
  `ruff.toml` carries the matching `PLR0913` rule for editor/dev feedback
  (quarry `python/` exempt).
- Frozen grouping value objects to collapse wide signatures:
  `CouponPeriod` (ICMA anchors), `RollRule` (calendar / business-day / eom),
  `ScheduleTerms` (frequency / day-count / roll).

### Changed
- `year_fraction(start, end, convention, *, period=None, calendar=None)` — ICMA
  anchors bundled into `CouponPeriod` (7→5 args).
- `generate_schedule(start, end, frequency, roll=None)` — roll conventions
  bundled into `RollRule` (6→4 args).
- `fixed_rate_bond(face, coupon_rate, start, maturity, terms)` — notional+currency
  as `Money` face, schedule/accrual conventions as `ScheduleTerms` (9→5 args).
- No behaviour change: every S00–S04 oracle stays green (same expected values).

## [0.0.5] - 2026-07-11

### Added
- **S04 — fixed-rate bond (first L2/L4-pricing slice).**
  - `instruments/fixed_rate_bond.py`: `FixedRateBond` (frozen pure data — coupon
    + redemption cashflows, no `pv` method) and a `fixed_rate_bond(...)` builder
    that expands schedule + day-count into explicit `Cashflow`s.
  - Oracle: closed-form discounted-cashflow PV on a flat curve and on the S03
    bootstrapped curve (independent sum), exact < 1e-12; plus cashflow-structure
    checks and the zero-coupon tie-back to the Slice 0 pure-discount result.
  - Quarry: `fixed_income/` (fixed-rate bond / fixed leg).

### Changed
- `engine/discounting.py` generalised from a single cashflow to a **cashflow
  leg**: it now prices any instrument satisfying the structural
  `CashflowInstrument` protocol (`.cashflows`) — no `isinstance`, no import of
  concrete instrument classes. `FixedCashflowTrade` gained a `.cashflows` view;
  the Slice 0 oracle stays green.
- CI layer tier bumped to `--layer 4` (the slice now prices through the engine).

### Deferred
- Seasoned-bond pricing (dropping already-paid coupons), accrued interest / clean
  vs dirty price, and business-day-adjusted coupon dates — no consumer yet.

## [0.0.4] - 2026-07-11

### Added
- **S03 — bootstrapped discount curve (first L1 slice).**
  - `market/discount_curve.py`: `DepositQuote`, `ParSwapQuote`, a log-linear
    interpolated `DiscountCurve` (behind the existing `CurveHandle`), and
    `bootstrap_discount_curve` — deposits give short-end DFs in closed form,
    par swaps extend the curve by a sequential closed-form solve (single-curve).
  - Oracle: every input reprices to par — deposits to their closed-form DF, swaps
    to zero NPV via the single-curve telescoping identity — exact < 1e-12; plus
    df(valuation)=1, strictly-decreasing DFs, and the log-linear interpolation law.
  - Quarry: `core/discount_curve.py`.
- CI layer tier bumped to `--layer 1` (the slice now reaches L1).

### Deferred
- Business-day-adjusted curve pillars, multi-curve (OIS discount / IBOR
  projection), non-pillar swap coupons (interpolated bootstrap), and QuantLib
  cross-check — the closed-form self-consistency oracle is stronger here; these
  arrive with the slices that need them.

## [0.0.3] - 2026-07-11

### Added
- **S02 — schedule & business-day calendar.**
  - `foundation/calendar.py`: `BusinessDayConvention` (UNADJUSTED / FOLLOWING /
    MODIFIED_FOLLOWING / PRECEDING / MODIFIED_PRECEDING) and a minimal
    data-driven `Calendar` (weekend + explicit holiday set) with `is_business_day`,
    `adjust`, `business_days_between`.
  - `foundation/schedule.py`: `Frequency` and `generate_schedule` — regular
    periods, short front stub (backward generation), EOM roll, optional
    business-day adjustment. No third-party dependency (stdlib month arithmetic).
  - `foundation/time.py`: BUS/252 completed now that calendars exist.
  - Oracle: hand-computed reference dates/counts (adjustments, coupon schedules
    incl. EOM + short front stub, BUS/252 day counts) — exact.
  - Quarry: `core/calendar.py`, `core/schedule.py`, `core/day_count.py` (BUS/252).

### Deferred
- Concrete national calendars (TARGET, US, London, Sao Paulo, ...) and long/back
  schedule stubs — no current consumer; they land with the instrument slice that
  first needs them (avoids the quarry's approximate long-stub heuristic).

## [0.0.2] - 2026-07-11

### Added
- **S1 — day-count conventions.** Extended `foundation/time.py` beyond the
  Slice 0 ACT/365F stub to the calendar-free conventions: ACT/360, 30/360
  (US bond basis), 30E/360 (Eurobond basis), ACT/ACT ISDA, ACT/ACT ICMA.
  - Oracle: published ISDA 2006 s.4.16 / ICMA Rule 251.1 year-fraction vectors,
    each expected value written as the convention's defining arithmetic, exact
    < 1e-12.
  - Quarry: `core/day_count.py`.

### Changed
- ACT/ACT ICMA now **requires** its coupon-period anchors (`ref_start`,
  `ref_end`, `frequency`) and raises when they are missing or invalid.

### Removed
- Debt shed (CLAUDE.md §5): the quarry's `strict_icma` flag and its silent
  fallback to ACT/365F on missing anchors (audit finding A.1 B1 — hidden
  wrongness) does not cross into the new tree.

### Deferred
- BUS/252 day-count, business-day `Calendar`, and `Schedule` generation move to
  their own slice (they need the calendar); not part of S1's named oracle.

## [0.0.1] - 2026-07-11

### Added
- **Slice 0 — walking skeleton.** A single fixed cashflow discounted on a flat,
  continuously-compounded curve, priced end-to-end L0->L6 through the stateless
  engine. Proves the spine holds before any further migration.
  - L0 `Money`/`Currency`, `Cashflow` (promoted from `fixed_income`),
    `year_fraction` (ACT/365F), `NumericalConfig`, `PricingResult`/`PricingFailure`.
  - L1 `MarketSnapshot` + `FlatDiscountCurve` behind a `CurveHandle` (df = exp(-r t)).
  - L2 `FixedCashflowTrade` (frozen, no `pv` method).
  - L4 `DiscountingEngine.price(...)` (null model; failure-as-value).
  - L5 `dv01` by bumping the snapshot; L6 `book(trade).value(...)`.
  - Oracle: PV = notional·exp(-r·t) closed form < 1e-12; analytic vs
    central-difference DV01 < 1e-6; repricing byte-identical (statelessness).
  - Quarry: `core/currency.py`, `core/day_count.py`, `core/numerical_config.py`,
    `core/discount_curve.py`, `fixed_income/fixed_leg.py` (Cashflow).

## [0.0.0] - 2026-07-11

### Added
- Bootstrap of the new tree: layer packages (`foundation`, `market`,
  `instruments`, `models`, `engine`, `risk`, `shell`), `verify.py`
  (`acyclic`/`tests`/`debt`/`provenance`/`version`/`all`), CI matrix
  (Ubuntu + Windows, Python 3.12), `.gitattributes` (LF), root `conftest.py`.
