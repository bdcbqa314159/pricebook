# Changelog

All notable changes to the new tree (`pricebook_ng`) are recorded here, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. `0.x` = migration
in progress; `1.0.0` is reached exactly when the quarry (`python/pricebook/`) is empty.

## [Unreleased]

## [0.19.0] - 2026-07-14

### Added
- **Generic curve greeks** (completes the A5 unification for curves). `curve01` /
  `bump_curve` in `risk/greeks.py` — parallel-shift the curve at a `MarketKey` and
  reprice, whatever the curve type. This gives rate risk on the **FX foreign curve**,
  the **inflation real/breakeven curve** (and dividend/carry) for free.
  - Curves gained a polymorphic `bumped(shift)`: `FlatDiscountCurve` shifts its rate,
    `SurvivalCurve` shifts its hazard (`Q -> Q·exp(-shift·t)`). `bump_curve` dispatches
    through it (no `isinstance`).
  - `credit01` is now a named alias of `curve01` on a survival key; `bump_hazard` is
    folded into `SurvivalCurve.bumped`. `dv01`/`bump_rate` use the same `bumped`.
  - Oracle: `curve01` on the FX foreign curve matches `-base·spot·T·DF_base·1e-4`; on
    the inflation real curve matches `notional·(-T·DF_real)·1e-4`; `credit01 == curve01`
    on a survival key; `bump_curve` shifts only the keyed curve.

### Deferred
- Pillar-wise / key-rate bumps for a **bootstrapped** curve (today `bumped` is a flat
  rate shift); home-discount `dv01` still assumes the flat curve.

## [0.18.0] - 2026-07-14

### Added
- **Inflation — zero-coupon inflation swap (first inflation slice).**
  - `products/inflation.py`: `ZeroCouponInflationSwap` (index, notional, fixed rate,
    maturity, currency, receive/pay inflation), pure data.
  - `engine/inflation.py`: `ZCISEngine` — the inflation forward index ratio
    `I(T) = DF_real(T)/DF_r(T)` (Fisher), with the real curve keyed at
    `MarketKey(INFLATION, index)`; receiver PV = `notional·DF_r·(I(T) − (1+K)^T)`.
  - `AssetClass.INFLATION` (a real curve is the only new market data — another cheap
    keyed asset class, A5).
  - Oracle: par ZCIS (K = breakeven) reprices to zero; PV matches the formula;
    receiver = −payer; below-breakeven fixed is valuable to the inflation receiver;
    missing real curve fails.

### Deferred
- Year-on-year inflation swaps and inflation-linked bonds; a breakeven/inflation01
  greek (bump the real curve, keyed — like `credit01`); seasonality; index lag.

## [0.17.0] - 2026-07-14

### Added
- **Commodity — European commodity option (first commodity slice).** The A5 keyed
  registry made it cheap: `AssetClass.CMDTY` + a `CommodityOption` product + a thin
  `CommodityOptionEngine`, and the **greeks are free** (the generic `spot_delta` /
  `vol_vega` work with no new code — the whole point of A5).
  - `engine/spot_option.py`: shared `price_spot_option(option, asset, model, numerics)`
    — Black-Scholes on the forward `F = spot·DF_carry/DF_r`, keyed by
    `MarketKey(asset, ticker)`. Behind both the equity (carry = dividends) and
    commodity (carry = convenience yield net of storage) engines.
  - **`EquityOptionEngine` refactored** onto the shared engine (rule of two;
    behaviour-preserving — the equity oracles stay green).
  - Oracle: put-call parity ties to the commodity forward; independent Black recompute;
    `sigma -> 0` intrinsic; missing market fails; and a demonstration that `spot_delta`/
    `vol_vega` price commodity greeks with no commodity-specific code.

### Deferred
- Commodity forward, seasonality/term-structure carry, futures vs spot, a vol surface.

## [0.16.0] - 2026-07-14

### Changed
- **Amendment A5 — `MarketSnapshot` keyed market-data registry.** All market data
  except the home `discount_curve` moves into `curves` / `spots` / `vols` maps keyed
  by `MarketKey(asset: AssetClass, id: str)` (new `market/keys.py`).
  - `survival_curve` / `fx_*` / `equity_*` fields are removed; folded into the maps.
    Folding survival **adds multi-issuer** — a `CDS` now names its `issuer`, and the
    engine looks up `curves[MarketKey(CREDIT, issuer)]`. `SurvivalCurve` gains a `df`
    alias (a hazard curve is the credit-risky discount-factor curve), so it lives in
    the same `curves` map as discount/dividend/foreign curves.
  - **Greeks collapse to one generic each**: `bump_spot`/`bump_vol` +
    `spot_delta`/`vol_vega` keyed by `MarketKey` — the per-asset `fx_delta`/`equity_delta`,
    `fx_vega`/`equity_vega`, `bump_fx_*`/`bump_equity_*` are deleted. `credit01`/`bump_hazard`
    are keyed by issuer. A new asset class now adds **keys, not fields, and no new greeks**.
  - Behaviour-preserving: every FX/equity/credit/rates PV and greek is unchanged
    (all prior oracles reused). New: a `MarketKey` namespacing test (FX "EUR" ≠ equity "EUR").
  - `cds(...)` builder drops the schedule-terms arg (CDS premiums are annual ACT/360 by
    convention) and takes `issuer` — stays within the 5-arg ceiling.

## [0.15.0] - 2026-07-14

### Added
- **Equity greeks** — `bump_equity_spot` / `bump_equity_vol` + `equity_delta` /
  `equity_vega` in `risk/greeks.py`, keyed by ticker, on the same `Priceable`
  protocol as FX and rate greeks. Oracle: the bumps move only their field;
  `equity_delta` matches the analytic BS delta `quantity·DF_div·N(d1)`; a put's
  delta is negative; `equity_vega` matches the analytic Black vega
  `quantity·DF_r·F·φ(d1)·√T`.

## [0.14.0] - 2026-07-14

### Added
- **Equity — European equity option (Black-Scholes with dividends), first equity slice.**
  - `products/equity_option.py`: `EquityOption` (frozen pure data — ticker, quantity,
    strike, maturity, currency, call/put).
  - `engine/equity_option.py`: `EquityOptionEngine` — Black-Scholes as Black-76 on the
    equity forward `F = spot·DF_div/DF_r`, discounted by `DF_r`; `PricingFailure` if the
    equity market is absent; expired → 0.
  - `MarketSnapshot`: `equity_spots` / `equity_div_curves` / `equity_vols`, keyed by ticker.
  - `foundation/black.py`: shared **`black_76`** primitive (option on a forward), used by
    both the equity (BS) and FX (GK) engines.
  - Oracle: put-call parity ties to the equity forward value; matches an independent Black
    recompute; `sigma -> 0` intrinsic; ATM-forward call == put.

### Changed
- `FXOptionEngine` refactored onto the shared `black_76` (behaviour-preserving; GK oracle
  stays green).

### Deferred
- Equity greeks (delta/vega, bump `equity_spots`/`equity_vols` through the `Priceable`);
  foreign-listed equity (own currency curve); a real dividend schedule vs the flat repo
  curve; a vol surface.

## [0.13.0] - 2026-07-14

### Added
- **FX vega** — `bump_fx_vol` + `fx_vega` (∂PV per unit FX vol) in `risk/greeks.py`,
  on the same `Priceable` protocol as `fx_delta`/`dv01`. Oracle: `bump_fx_vol` moves
  only that vol; `fx_vega` matches the analytic Black vega
  `notional·DF_quote·F·φ(d1)·√T`; a long option is long vol (vega > 0).

### Changed
- Removed `fx_forward_priceable` — it was byte-identical to `discounting_priceable`;
  all FX products (forward + option) now bind via `discounting_priceable`.

## [0.12.0] - 2026-07-14

### Added
- **FX option — Garman-Kohlhagen.**
  - `products/fx_option.py`: `FXOption` (frozen pure data — base/quote legs,
    maturity, call/put) + `fx_option(...)` builder.
  - `engine/fx_option.py`: `FXOptionEngine` — GK as Black-76 on the FX forward
    `F = spot·DF_base/DF_quote`, discounted by the quote curve; `sigma -> 0`
    collapses to discounted intrinsic; `PricingFailure` if the FX market is absent.
  - `MarketSnapshot.fx_vols`: flat FX vol per pair (market data, §5.1).
  - Oracle: put-call parity ties to the FX forward PV (cross-slice); matches an
    independent Black recompute; `sigma -> 0` intrinsic; ATM-forward call == put.

### Deferred
- FX vega (bump `fx_vols` through the `Priceable`, like `fx_delta`) — trivial
  follow-up; a vol surface (strike/tenor) replacing the flat vol; American/barrier.

## [0.11.0] - 2026-07-13

### Changed
- **FX market data promoted into `MarketSnapshot`** (ruling §5.1, closing the FX
  loop as survival-in-snapshot did for credit). The snapshot now carries
  `fx_curves` (foreign-currency curves) and `fx_spots` (home units per foreign
  unit), keyed by currency. `FXForwardModel` is removed — the FX forward is a
  linear product priced with a `DiscountingModel` over the snapshot; the engine
  looks up the base curve/spot by the product's base currency and returns
  `PricingFailure` if the FX market is absent.

### Added
- **FX greeks on the `Priceable` protocol.** `bump_fx_spot` + `fx_delta` (∂PV per
  unit spot) in `risk/greeks.py`; `fx_forward_priceable` factory. The same FX
  Priceable feeds both `fx_delta` (spot bump) and the generic `dv01` (quote-curve
  bump).
  - Oracle: FX data lives in the snapshot; `bump_fx_spot` moves only that spot;
    `fx_delta` matches the analytic `base_notional·DF_base(T)`; sell = -buy; the
    same Priceable gives a non-zero `dv01`.

### Deferred
- Base-currency rate risk (bump `fx_curves`), a currency→curve map replacing the
  home/foreign split, FX options (Garman-Kohlhagen), and a `CurrencyPair` type.

## [0.10.0] - 2026-07-13

### Added
- **FX — FX forward (first FX slice).**
  - `products/fx_forward.py`: `FXForward` (frozen pure data — base leg, quote leg,
    maturity, buy/sell) and an `fx_forward(...)` builder taking a strike.
  - `models/fx_model.py`: `FXForwardModel(market, base_curve, spot)` — the quote-
    currency market + the base-currency curve + spot (quote per base).
  - `engine/fx_forward.py`: `FXForwardEngine` — values both legs in the quote
    currency by covered interest parity; buyer PV = base leg − quote leg; a matured
    forward settles to 0 (A2).
  - Oracle: a par forward (struck at `F = spot·DF_base/DF_quote`) prices to zero;
    PV matches CIP; sell = −buy; below-forward strike is valuable to the buyer;
    matured forward settles.

### Deferred
- Promote FX market data (base curve + spot) into the `MarketSnapshot` and unify
  FX greeks (delta, per-currency rate risk) on the `Priceable` protocol — the
  §5.1 follow-up, exactly as the CDS survival curve was promoted (multi-currency
  snapshot: curves keyed by currency + FX spots).
- FX options, NDFs, and a real `CurrencyPair`/quoting-convention type.

## [0.9.1] - 2026-07-13

### Changed
- Spelling fix: **`Pricable` → `Priceable`** everywhere — the `Priceable` protocol,
  the `discounting_priceable` / `hull_white_priceable` / `credit_priceable`
  factories, the module (`risk/priceable.py`), and the design docs (`CLAUDE.md`,
  `redesign/`). Pure rename, behaviour-preserving (all oracles green).

## [0.9.0] - 2026-07-13

### Changed
- **SurvivalCurve promoted into `MarketSnapshot`** (Cowork ruling §5.1). The
  credit/hazard curve is market data, so it now lives on the snapshot
  (`survival_curve`, reached through a new `SurvivalHandle` protocol, mirroring
  `CurveHandle`). `CreditModel(market, recovery)` reads it via `market.survival_curve`
  (its `survival` field is now a property).
- **Credit risk unified on the `Pricable` protocol.** `credit01` (CS01) moved into
  `risk/greeks.py` alongside `dv01`, both central-differencing a `Pricable` under a
  snapshot bump (`bump_rate` / `bump_hazard`) — one finite-difference core.
  `risk/credit_greeks.py` is removed.
  - `credit_pricable(product, recovery, engine, numerics)` builds the credit
    `Pricable`; the same pricable feeds both `credit01` (hazard bump) and `dv01`
    (rate bump) — a CDS now has rate risk *and* credit risk through one interface.
  - CDS pricing is behaviour-preserving (all prior oracles green).
  - Oracle: survival lives on the snapshot; `bump_hazard` moves only the credit
    curve; buyer credit01 > 0, seller = -buyer; matches an independent hazard FD;
    the CDS pricable also yields a non-zero rate dv01.

## [0.8.0] - 2026-07-13

### Added
- **Float-leg fixings — seasoned swaps** (consumes the `FixingHistory` A1 added).
  - The `SwapEngine` float leg is now temporality-aware: a period that already
    paid (`b <= valuation`) is settled; a period whose reset is strictly past
    (`a < valuation`) uses the realized fixing `market.fixings.get(a)`; a future
    period projects the curve forward. Missing fixing ⇒ `PricingFailure`.
  - `FloatLeg` gains `day_count` (needed to accrue a fixed current period from its
    reset); `vanilla_swap` sets it from the float schedule.
  - The old "float leg starts before valuation" guard is retired.
  - Oracle: a seasoned swap's current coupon uses the fixing and matches the
    independent per-period sum; an already-paid period is excluded; a missing
    fixing fails; a spot swap needs no fixings (behaviour-preserving).

### Deferred
- Fixing lag (reset a few days before accrual start), and swap-level accrued/clean
  decomposition on the float leg — add when needed.

## [0.7.0] - 2026-07-13

### Added
- **Monte-Carlo engine — HW swaption, `analytic vs MC convergence`** (closes the
  named oracle for the whole Hull-White arc).
  - `engine/swaption_mc.py`: `SwaptionMCEngine` prices the European swaption under
    the T0-forward measure — one exact Gaussian draw of `x(T0)` (mean `M`, variance
    `V`), reconstitutes the coupon bond via the model's `zero_bond`, averages the
    payer/receiver payoff, discounts by `P(0,T0)`. Stdlib `random`, no numpy.
  - `NumericalConfig` gains `mc_paths` and `mc_seed` (fixed seed ⇒ reproducible,
    referential transparency preserved).
  - `engine/swaption.py`: extracted the shared `coupon_bond_cashflows` helper (used
    by both the analytic and MC engines; analytic oracle preserved).
  - Oracle: MC converges to the S08 Jamshidian analytic within ~2% at 200k seeded
    paths (payer and receiver); exact at `sigma=0` (deterministic); reproducible
    under a fixed seed.

### Deferred
- Variance reduction (antithetics, Sobol), a general MC path engine for other
  products, and MC greeks — added when a product needs them.

## [0.6.0] - 2026-07-13

### Added
- **CDS credit01 / CS01** (`risk/credit_greeks.py`) — CDS PV sensitivity to a 1bp
  parallel credit-spread (hazard) shift, by central finite difference: `bump_hazard`
  scales each survival pillar by `exp(-dh*t)`, rebuilds the `CreditModel`, reprices.
  The hazard analogue of `dv01` (which bumps the discount rate).
  - Oracle: `bump_hazard` shifts each pillar by exactly `exp(-dh*t)` (anchor
    unchanged); buyer credit01 > 0 (protection gains as credit worsens); seller =
    -buyer; matches an independent hazard-bump central difference.

### Deferred
- Routing credit01 through a `survival -> PV` closure (like the `Pricable`
  factories) — added when a second hazard-sensitive product exists; today CDS is
  the only one, so it calls `CDSEngine` directly.

## [0.5.0] - 2026-07-13

### Added
- **CDS as an engine-priced product.**
  - `products/cds.py`: `CDS` (frozen pure data — premium schedule, spread,
    notional, buyer/seller) and a `cds(...)` builder.
  - `models/credit_model.py`: `CreditModel(market, survival, recovery)` — carries
    the discounting market + the bootstrapped hazard curve + recovery (A1).
  - `engine/cds.py`: `CDSEngine` — values the protection buyer via the L1 CDS leg
    math; the seller is the negative.
  - Oracle: a par CDS reprices to zero through the engine; the engine matches the
    L1 `cds_pv`; seller = -buyer; buyer value falls as the contract spread rises.

### Deferred
- CDS greeks — a `credit_pricable` factory drops the CDS onto the L5 `Pricable`
  protocol (rate dv01 today, credit01 once a hazard bump exists); add when wanted.
- Seasoned CDS (segment-and-settle on the premium leg) and quarterly premiums.

## [0.4.0] - 2026-07-12

### Added
- **Credit — hazard/survival curve + CDS bootstrap (first credit slice, L1).**
  - `market/survival_curve.py`: `SurvivalCurve` (piecewise-hazard, log-linear in
    `ln Q`, behind `survival(date)`), the single-name CDS leg math (`RPV01`,
    protection PV, `cds_par_spread`, `cds_pv` for the protection buyer), and
    `bootstrap_survival_curve` — sequential solve so each CDS reprices to zero at
    its par spread.
  - Oracle: `survival(valuation)=1`; each input CDS reprices to zero (`< 1e-10`);
    the curve-implied par spread equals each quote; survival strictly decreasing in
    (0,1]; log-linear between pillars. Mirrors the S03 discount-curve bootstrap.
  - Quarry: `core/survival_curve.py`.

### Deferred
- CDS as an **engine-priced product** (L2 `CDS` product + L3 `CreditModel` +
  L4 `CDSEngine` + greeks via `Pricable`) — the immediate follow-up, exactly as the
  discount curve preceded the bond/swap.
- Quarterly premiums, accrual-on-default, and a finer protection integral (the
  reprice-to-zero oracle is exact on the shared discretisation regardless).

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
