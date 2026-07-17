# Changelog

All notable changes to the new tree (`pricebook_ng`) are recorded here, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. `0.x` = migration
in progress; `1.0.0` is reached exactly when the quarry (`python/pricebook/`) is empty.

## [Unreleased]

## [0.45.0] - 2026-07-17

### Added
- **Deposit product (L2) — CP-2c #3, fixed_income spine.** A money-market deposit modelled as its
  two cashflows (`−N` at start, `N·(1+rate·τ)` at maturity) priced by the existing `DiscountingEngine`
  — **no bespoke engine**. A2 reconciles the views: a forward deposit reprices to zero at par; a spot
  deposit's principal-today is realized cash (excluded from the mark), so the mark is the redemption
  value `N` at par. `Deposit(face, rate, cashflows)` — 3 fields (under the new `fields` gate).
  - Oracles: forward par → 0; spot par → principal; off-par closed form; par → 0 on a bootstrapped curve.
  - **Deletable-bar read** (`fixed_income/deposit.py`): ng covers pricing + forward/temporal (the quarry
    values the redemption only). Residual before deletable: convention builder, implied-DF-as-method
    (ng has it via `DepositQuote`), `pv_ctx`, serialisation — logged in `quarry_reconciliation.md`.
    Drawdown 0/768.
  - quarry: `python/pricebook/fixed_income/deposit.py` · slice: `deposit-spine`

## [0.44.0] - 2026-07-17

### Added
- **Seasoned FRA via `FixingHistory` (L4) — CP-2c #2, fixings / seasoned-float.** The `FRAEngine`
  now handles the temporal cases (A2): a forward-starting/spot period uses the curve forward; a
  **seasoned** period (`accrual.start < valuation`) uses the realized reset looked up in the
  snapshot's `FixingHistory`; a fully-paid period (`end <= valuation`) settles to PV 0 (the shell
  remembers the realized cash). First `FixingHistory`-consuming engine — the swap float leg and the
  L6 float realized P&L follow the same pattern.
  - Oracles: seasoned FRA prices to `face·τ·(fixing−K)·DF(end)`; zero at `K = fixing`; missing fixing
    → `PricingFailure`; fully-paid → 0; forward FRA unchanged.
  - **Deletable-bar read** (`fixed_income/fra.py`): quarry ISDA-settle-at-start equals ng
    end-settle for single-curve (`DF(start)/(1+fwd·τ)=DF(end)`), and the quarry FRA lacks fixings
    (ng now ahead). Residual before deletable: multi-curve `forward_rate(projection_curve)`,
    `par_rate`/`pv_ctx`, convention builder — logged in `quarry_reconciliation.md`. Drawdown 0/768.
  - quarry: `python/pricebook/fixed_income/fra.py` · slice: `fra-seasoned-fixings`

## [0.43.0] - 2026-07-17

### Added
- **`verify.py fields` merge gate (CP-2c #1).** The dataclass-field analogue of `PLR0913` (which
  only sees function args): a value dataclass has **≤5 fields** unless it carries an explicit
  `# fields-exempt: <reason>` marker. AST-based, scans all `src/pricebook_ng`; added to `verify all`
  and `redesign/09` / `CLAUDE.md §3b`.

### Changed
- **Product field-bundling (behaviour-preserving).** Closes the FRA smell flagged at the CP-2b
  checkpoint. Bundled loose primitives into value objects that already exist —
  `Money` (notional/strike + currency) and `Accrual` (start + end + day_count):
  - `ForwardRateAgreement` 7→4 (`face: Money`, `accrual: Accrual`); `CDS`, `ZeroCouponInflationSwap`
    6→5 (`face: Money`); `EquityOption`, `CommodityOption` 6→5 (`strike: Money`, the strike price).
  - Engines read `.face.amount`/`.face.currency`/`.strike.amount`/`.accrual.*`; **all PVs byte-identical**
    (243 green), guarded by every product oracle.
  - Legit-wide aggregates carry `# fields-exempt:` markers: `MarketSnapshot` (A5 shape), `XvaReport`
    (output record), `XvaReportConfig` (config).
  - Not a parity slice (no quarry module retired) — code-quality; drawdown unchanged 0/768.
  - slice: `product-field-bundling`

## [0.42.0] - 2026-07-17

### Added
- **FRA — forward rate agreement (L2 product + L4 engine) — CP-2b #3, fixed_income spine.**
  `ForwardRateAgreement` (pure data) + `FRAEngine`: pay fixed, receive the simply-compounded
  forward `L(T1,T2) = (P(0,T1)/P(0,T2)-1)/τ` over one period, settled at T2 —
  `PV(pay-fixed) = notional·τ·(L-K)·P(0,T2)`, composed from the curve's discount factors so it
  prices on any curve (flat **or** bootstrapped — the general-curve payoff for the spine).
  - Oracles: a par FRA (K = L) reprices to zero; off-par matches the closed form; receiving fixed
    flips the sign; and the implied forward reprices to par on a bootstrapped curve.
  - Scope: forward-starting / spot (`accrual_start ≥ valuation`); a seasoned FRA needs a fixing
    (`FixingHistory`, a later slice, like the seasoned float leg).
  - **Parity:** first new fixed_income vanilla under the parity-depth mode; steps `fixed_income/fra`
    toward deletable. bond/swap already price on general curves; deposit / OIS / seasoned-float remain.
  - quarry: `python/pricebook/fixed_income/fra.py` · slice: `fra-spine`

## [0.41.0] - 2026-07-17

### Changed
- **General-curve Hull-White (L3) — CP-2b #2, the biggest single parity gap under the XVA stack.**
  HW no longer asserts `FlatDiscountCurve`: it reads `df` **and** `instantaneous_forward` from any
  curve, replacing the flat `r0` with the market forward `f(0,t)` in `alpha(t)` and the `zero_bond`
  reconstitution, and taking its time axis from `market.valuation_date`. `forward_short_rate` and
  the risk-neutral path simulator are now **date-based** (they need `f(0,t)`), and their callers
  (`swaption_mc`, the exposure/MPOR engines) pass dates.
  - **Byte-identical on a flat curve** (`f(0,t)=r0`), so every existing swaption / exposure / XVA /
    measure-consistency / MPOR oracle stays green (239 total).
  - New oracles on a **bootstrapped** curve: the model refits the initial curve
    (`zero_bond(0,S,f(0,0)) == P^M(0,S)`); ZCB-option put-call parity; a flat-pillars curve matches
    `FlatDiscountCurve` exactly; and the analytic swaption == the MC swaption (`rel=2%`).
  - **Parity:** the whole XVA/exposure stack can now run on a real bootstrapped curve, not just the
    flat skeleton — steps `models/hull_white` toward deletable (term-structure vol + per-currency +
    trees remain the gap). Drawdown still 0/768 (narrowed).
  - quarry: `python/pricebook/models/hull_white.py` · slice: `general-curve-hw`

## [0.40.0] - 2026-07-17

### Added
- **General-curve rate accessors (L1) — CP-2b #1, the first parity-depth slice.** Both discount
  curves gain `zero_rate` (continuously-compounded `-ln P(0,t)/t`) and `instantaneous_forward`
  (`f(0,t) = -d/dt ln P(0,t)`) — the capability a general-curve Hull-White needs where the flat
  curve used a constant `r0`. `FlatDiscountCurve` returns the constant `rate`; the log-linear
  `DiscountCurve` returns a piecewise-**constant** forward per segment (exact `-slope`, not
  finite-difference), via a shared `_bracket_slope` that also backs `df`.
  - Oracles: flat curve → constant rate; constant-rate pillars → that same constant forward
    everywhere; a rising curve's segment forward equals the analytic log-DF slope and its running
    integral reconstructs `-ln df` at each pillar; `zero_rate = -ln df/t`.
  - **Parity:** steps `core/discount_curve` toward deletable (adds 2 of its 3 rate accessors);
    `forward_rate` (simply-compounded) and pluggable interpolation remain the recorded gap.
    Unblocks CP-2b #2 (general-curve HW). Drawdown still 0/768 (partial cross, gap narrowed).
  - quarry: `python/pricebook/core/discount_curve.py` + `curves/` · slice: `general-curve-rates`

## [0.39.0] - 2026-07-16

### Added
- **Consolidated XVA report (L6, A6.2)** — completes the CP-1 cluster. `xva_report(swaps, model,
  numerics, config)` in `shell/xva_report.py` simulates a netting set's exposure **once** and
  returns every adjustment (CVA/DVA/BCVA/FVA/KVA/MVA) + the EE/PFE/EAD profiles, consolidating the
  six separate L5 calls. Lives at L6 because a per-counterparty netting set is a book of trades.
  - Netting-set exposure: new `netting_set_exposure(swaps, model, numerics, pfe_quantile)` (L5)
    sums each swap's value on **shared paths** so offsetting trades net; KVA uses the netting-set
    SA-CCR EAD runoff (`netting_set_ead` as-of each grid date). The economy (discount + both
    parties' survival curves) rides on `model.market` (A5), read directly by the integrators.
  - Oracles: a single-trade netting set reproduces each standalone L5 value exactly (one pass, same
    draws); a payer + mirror receiver nets the portfolio exposure to zero, collapsing the netted CVA.
  - Refactor: `_simulate_swap_values` is now the single-swap case of `_simulate_netting_set`
    (byte-identical, guarded by the measure/BCVA/stochastic reproducibility oracles).
  - quarry: `python/pricebook/risk/` + `desks/` · slice: `xva-report`

## [0.38.0] - 2026-07-16

### Added
- **L6 trade lifecycle — the first vertical up into the shell (A6.2).** Extends the A3
  Trade/Book/benefit-table stub to the full realized-vs-mark split. New `Trade.mark(market,
  numerics, engine)` (sum of the products' PVs + accrued as of the snapshot) and `Book.value(...)`
  (the book's mark = Σ trade marks, linearity), sharing a `_combine` helper; `BookedTrade.value`
  now delegates to `Trade.mark` and remembers the observed mark.
  - Oracle over a bond's life: at issue the mark equals the full discounted PV (realized = 0);
    mid-life the mark prices **only future flows** (the engine excludes the paid ones, A2) while
    the benefit table holds the realized cash; **dirty = clean + accrued** at a mid-period date;
    at maturity mark = 0 and realized = the total nominal; **book mark & realized are linear**
    across trades. So `realized + mark` is the trade's total economics (A3).
  - Deferred: SQLite persistence of the benefit table (`pnl_history` behind the persistence
    interface) is its own data-spine slice; float-leg realized needs fixings (seasoned-float slice).
  - quarry: `python/pricebook/core/book.py` + `pnl_history` · slice: `l6-trade-lifecycle`

## [0.37.1] - 2026-07-16

### Added
- **Measure-consistency binding oracle (L5, Amendment A6.1).** The exposure stack runs two
  simulators — the per-date **forward-measure** engine (EE/PFE profiles) and the **risk-neutral
  joint-path** engine (MPOR). A6.1 rules them one model under a change of numeraire
  (`E^Q[D·max(V,0)] = P(0,t)·E^{T_t}[max(V,0)]`), so they may never silently diverge. The new
  test binds them: the risk-neutral joint-path marginal, shifted by the analytic forward-measure
  drift `m(t)`, reproduces the forward-measure EE **and** PFE per date (two independent MC
  estimates of the same discounted exposure, agreeing to `rel=4%`). No behaviour change — the
  `_simulate_rate_paths` docstring/provenance now document the ratified relationship.
- Ratified **Amendment A6** in `redesign/02_spine.md` (exposure measure + first-L6 rulings) and
  the **checkpoint-&-review cadence** (`redesign/11`, `CLAUDE.md §6`): stop at ≤6 slices or a
  capability-cluster boundary; every checkpoint carries an oracle audit, quarry drawdown `N/768`,
  a challenge-me list, and a smell/debt scan.
  - slice: `measure-consistency-oracle` (CP-1, first of the A6 cluster)

## [0.37.0] - 2026-07-16

### Added
- **MPOR path-simulated exposure (L5)** — the residual exposure that survives full
  collateralisation. At a counterparty default the collateral reflects the value one margin
  period of risk ago, so `E_mpor(t) = mean max(V(t) - V(t - MPOR), 0)`. This needs the joint
  distribution of `V(t)` and `V(t-MPOR)`, so the slice adds `_simulate_rate_paths` — a
  **risk-neutral joint-path simulator** (exact Ornstein-Uhlenbeck steps on the HW state,
  `r = x + α(t)`) — and `mpor_exposure(swap, model, numerics, mpor_days)`.
  - Oracles: the path simulator reproduces the analytic HW/OU moments — marginal mean `α(t)`,
    variance `σ²(1-e^{-2at})/2a`, and cross-date covariance `e^{-a(t-s)}·var(s)`; a zero gap
    gives exactly zero exposure; exposure grows with the gap; it feeds a positive CVA.
  - **Provisional measure choice (flagged for review):** the EE/PFE profiles use per-date
    forward measure; this path simulator uses the risk-neutral measure. For the local MPOR
    *difference* the choice is second-order, but unifying the exposure stack onto one measure
    is a design question for the L5 brainstorm — see the handoff.
  - Scope: single swap, zero threshold; float valued via `notional - couponbond` at the
    pre-gap date too (float ≈ par, O(MPOR) error). Netting-set / threshold MPOR is a refinement.
  - quarry: `python/pricebook/risk/` · slice: `mpor-paths`

## [0.36.0] - 2026-07-16

### Added
- **Margined (collateralised) exposure (L5)**. `collateralized_exposure(swap, model, numerics,
  threshold)` models a two-way CSA with variation margin and an uncollateralised threshold `H`:
  collateral posts the mark beyond `H`, so exposure is capped — `E_coll = min(max(±V, 0), H)`.
  `H = 0` is fully collateralised (exposure 0); a huge `H` recovers the uncollateralised
  `exposure_profiles`. Feeds a collateralised CVA/DVA below the uncollateralised one.
  - Oracles: a huge threshold reproduces `exposure_profiles` exactly (same draws); zero
    threshold gives zero exposure; `σ=0` caps the deterministic exposure exactly (and the cap
    bites); collateralised CVA < uncollateralised CVA.
  - Fourth consumer of the extracted `_simulate_swap_values` (means, PFE quantiles, collateral).
  - Scope: marginal (per-date) model — the margin-period-of-risk close-out gap that leaves
    residual exposure under full collateralisation needs joint-path simulation, a later slice.
  - quarry: `python/pricebook/risk/` · slice: `margined-exposure`

## [0.35.0] - 2026-07-15

### Added
- **PFE-quantile / dynamic-IM engine (L5)**. `pfe_profile(swap, model, numerics, quantile)`
  returns the potential future exposure at confidence `q`: `PFE_q(t_j)` = q-quantile of
  positive exposure `max(V(t_j), 0)` across the simulated paths — the exposure tail the EPE
  averages over, and a high-quantile PFE doubles as a dynamic initial-margin proxy that feeds
  `mva`.
  - Oracle: because the remaining swap value is monotonic in the Gaussian short rate, the
    q-quantile of V equals V at the q-quantile of r — so `PFE_q(t_j) = max(V(t_j; r_q), 0)`
    with `r_q = forward_short_rate(t_j, Φ⁻¹(q))`, matched by MC (`rel=2%`); `σ=0` collapses it
    to the deterministic exposure; PFE rises with `q` and funds a positive MVA.
  - Refactor: extracted `_simulate_swap_values` (per-path V by date), now shared by the EE
    means (`exposure_profiles`) and the PFE quantiles — byte-identical MC (same seed/draws), so
    the CVA/BCVA/stochastic-EAD oracles stay exact.
  - Scope: PFE-as-IM proxy; a margin-period-of-risk IM on ΔV is the noted refinement.
  - quarry: `python/pricebook/risk/` · slice: `pfe-quantile`

## [0.34.0] - 2026-07-15

### Added
- **MVA — margin valuation adjustment (L5)**, completing the XVA family. `mva(im, snapshot,
  key, funding_spread)`: `MVA = s_F·Σ IM(t_i)·DF(t_i)·S(t_i)·τ_i` — the funding cost of posting
  initial margin over the trade's life, the **same survival-weighted funding annuity** as FVA
  and KVA (now three consumers of the shared `_annuity_adjustment`), with the IM profile in
  place of net exposure / capital.
  - Oracles: MVA on unit IM equals `s_F · RPV01`; linearity in spread and IM; a vertical where
    an IM profile taken as the SA-CCR PFE (AddOn) runoff feeds MVA to a positive charge matching
    its annuity.
  - `IM(t)` is an input — generating it (SIMM, or a dynamic MC-quantile IM over the margin period
    of risk) is the upstream slice, as the exposure engine is upstream of CVA.
  - quarry: `python/pricebook/risk/` · slice: `mva`

## [0.33.0] - 2026-07-15

### Added
- **Netting-set SA-CCR (L5)**. `netting_set_ead(trades, valuation_date)` aggregates a
  netting set of IR swaps (each `(swap, mark)`) the Basel way: signed effective notionals
  `D = δ·notional·SD·MF` (δ = +1 payer / −1 receiver) net within maturity buckets (<1y,
  1–5y, >5y) and combine across buckets with the supervisory correlations
  (`√(ΣD² + 1.4(D₁D₂+D₂D₃) + 0.6·D₁D₃)`), while the marks net into one replacement cost.
  - Oracles: a one-trade set equals single-trade `saccr_ead`; a payer + mirror receiver
    perfectly nets (signed notionals and marks cancel → EAD 0); a two-bucket set matches
    the hand-computed correlation aggregation; netting is sub-additive vs standalone EADs.
  - Refactor: extracted `_effective_notional_magnitude` and `_ead_from_addon` (the
    RC/multiplier/EAD assembly), now shared by the single-trade and netting-set paths.
  - Scope: single-currency IR hedging set, unmargined, no collateral — margined MF, other
    asset classes, and collateral haircuts remain later refinements.
  - quarry: `python/pricebook/risk/` · slice: `netting-saccr`

## [0.32.0] - 2026-07-15

### Added
- **Stochastic-mark SA-CCR EAD profile (L5)** — unifies the two halves of the capital
  stack. `stochastic_ead_profile(swap, model, numerics)` sets SA-CCR's replacement cost to
  the MC exposure engine's expected positive exposure instead of the ATM zero:
  `EAD(t_j) = α·(EPE(t_j) + AddOn_remaining(t_j))`. Because `EPE ≥ 0` pins the multiplier at
  1, this is `forward_ead_profile` with `mark = EPE(t_j)`, and decomposes exactly as
  `forward_ead(t_j) + α·EPE(t_j)` — both pieces already oracle-checked.
  - Oracles: exact decomposition into the deterministic PFE profile plus `α·EPE`; it dominates
    the ATM profile; and its KVA charge exceeds the ATM one (expected exposure adds capital).
  - Closes the loop end-to-end: MC exposure → SA-CCR RC → capital profile → KVA.
  - quarry: `python/pricebook/risk/` · slice: `stochastic-ead`

## [0.31.0] - 2026-07-15

### Added
- **Forward SA-CCR EAD profile → KVA (L5)** — closes the SA-CCR → capital → KVA loop.
  `forward_ead_profile(swap, valuation_date)` reprices SA-CCR at each future coupon date on
  the shrinking remaining trade (an EAD **runoff**), and `capital_profile(ead, risk_weight)`
  scales it to `8%·RWA` — the capital `K(t)` that `kva` charges the cost of capital on. Under
  the ATM assumption (expected mark = 0 → RC = 0, multiplier = 1) the runoff is the deterministic
  supervisory PFE, so the whole chain has a closed-form oracle.
  - Oracles: each `EAD(t_j)` equals the closed-form `α·SF·notional·SD·MF` on the remaining
    maturity (first point == single-date `saccr_ead`); the profile runs off monotonically;
    `capital = 8%·EAD·RW`; KVA on it equals the cost-of-capital annuity.
  - Refactor: `saccr_ead` now delegates to a param-level `_ead_ir(notional, S, E, mark)` shared
    with the runoff (guarded by the SA-CCR oracle, no behaviour change).
  - Scope: ATM-mark runoff; a stochastic-mark RC (expected positive exposure from the MC engine)
    is the noted refinement.
  - quarry: `python/pricebook/risk/` · slice: `forward-ead-kva`

## [0.30.0] - 2026-07-15

### Added
- **SA-CCR — Basel standardised counterparty EAD & RWA (L5)**. New `risk/saccr.py`:
  `saccr_ead(swap, mark, valuation_date) = α·(RC + PFE)` for a single-trade interest-rate
  netting set (unmargined, uncollateralised), plus `risk_weighted_assets(ead, risk_weight)`
  and `saccr_capital(rwa) = 8%·RWA`. RC = max(V,0); PFE = multiplier·AddOn with the IR
  supervisory factor (0.5%), supervisory duration (5% decay), unmargined maturity factor,
  and the 5%-floored multiplier; α = 1.4.
  - Oracles: a 10y ATM $100mm IRS has EAD ≈ 5.5% of notional (the published SA-CCR anchor);
    RC adds `α·mark` in the money; deep out-of-the-money the multiplier hits its 0.05 floor
    (EAD → `0.05·EAD_atm`); RWA/capital identities.
  - This is the regulatory EAD generator that both counterparty RWA and (extended to a
    forward profile) the KVA capital input build on.
  - Scope: single IR trade, one hedging set, no collateral/margin — netting-set buckets +
    correlations, margined MF, other asset classes, and collateral haircuts are later slices.
  - quarry: `python/pricebook/risk/` · slice: `saccr`

## [0.29.0] - 2026-07-15

### Added
- **KVA — capital valuation adjustment (L5)**. `kva(capital, snapshot, key, cost_of_capital)`
  in `risk/xva.py`: `KVA = γ_K·Σ K(t_i)·DF(t_i)·S(t_i)·τ_i` — the cost of capital charged on
  the capital profile `K(t)`, discounted and survival-weighted. The **same funding annuity as
  FVA** (the CDS RPV01 structure), with capital in place of net exposure and the hurdle rate in
  place of the funding spread.
  - Oracles: KVA on unit capital equals `γ_K · RPV01`; capital proportional to EPE
    (`K = k·EPE`) integrates as expected; linearity in the cost of capital.
  - `K(t)` is an input — generating it from a regulatory model (SA-CCR EAD → RWA → capital)
    is the upstream RWA slice, exactly as the exposure engine is upstream of CVA.
  - quarry: `python/pricebook/risk/` · slice: `kva`

### Changed
- Extracted the shared **`_annuity_adjustment`** (`rate·Σ profile·DF·S·τ`) now backing both
  FVA and KVA — a survival-weighted annuity, the CDS RPV01 structure. `fva` refactored onto it
  under its green oracle, no behaviour change.

## [0.28.0] - 2026-07-15

### Added
- **FVA — funding valuation adjustment (L5)**. `fva(exposure, snapshot, key, funding_spread)`
  in `risk/xva.py`: `FVA = FCA - FBA = s_F·Σ (EPE_i - ENE_i)·DF(t_i)·S(t_i)·τ_i` — the
  funding spread carried over each interval on the **net** exposure, discounted and
  survival-weighted (funding stops on default). Reuses the same `ExposurePair` as CVA/DVA.
  - Where CVA/DVA weight exposure by a *protection leg* (default increments × `(1-R)`), FVA
    weights it by a *funding annuity* `S·τ` — the CDS RPV01 structure. So the oracle: FVA on
    unit positive exposure equals `s_F · RPV01` (the survival annuity) exactly; plus a
    symmetric-exposure zero (cost cancels benefit) and linearity in spread and exposure.
  - Scope: symmetric funding spread, single survival curve, discounting-approach FVA
    (FVA/DVA overlap and own-vs-joint survival are known modelling debates, out of scope).
  - quarry: `python/pricebook/risk/` · slice: `fva`

## [0.27.0] - 2026-07-15

### Added
- **DVA + bilateral BCVA (L5)**. `dva`, `bcva`, and the `CreditParty` bundle in
  `risk/xva.py`. DVA is the mirror of CVA — expected gain from *our own* default while
  out of the money — which is exactly the CVA integral on the **negative** exposure
  profile (ENE) against our own survival curve, so `dva` reuses `cva` and
  `bcva(exposure, snapshot, counterparty, self_party) = CVA - DVA` (net credit charge;
  value adjustment is `-BCVA`). `CreditParty(key, recovery)` keeps `bcva` under the
  5-arg ceiling.
  - Oracles: ENE of a payer swap equals EPE of the mirror receiver swap (exact, same
    simulated rates); BCVA decomposes into `CVA - DVA`; a default-free self (Q≡1) zeroes
    DVA so BCVA collapses to unilateral CVA.
  - Scope: unilateral pair (exposure ⟂ default, no first-to-default survival weighting —
    a later refinement multiplies each term by the other party's Q(t)).
  - quarry: `python/pricebook/risk/` · slice: `bcva`

### Changed
- **`expected_exposure` → `exposure_profiles`**, now returning an `ExposurePair`
  (EPE **and** ENE) from a single MC pass — ENE is free from the same paths. `ExposurePair`
  and `ExposureProfile` (docstring generalised to `E[(±V)^+]`) live in `risk/xva.py`.

## [0.26.0] - 2026-07-15

### Added
- **Monte-Carlo expected-exposure engine (L5)** — generates the real `EE(t)` profile
  that CVA consumes. `expected_exposure(swap, model, numerics)` in `risk/exposure.py`
  simulates the Hull-White short rate to each grid date under that date's t_j-forward
  measure (one exact Gaussian draw) and reprices the remaining swap analytically via
  `zero_bond`, returning an `ExposureProfile`. Closes the exposure-generation gap left
  open by the CVA slice.
  - Oracles: (1) `sigma = 0` -> `EE(t_j)` equals the deterministic forward swap value's
    positive part, exact to `1e-8`; (2) the forward-measure identity — `P(0,t_j)·EE(t_j)`
    equals the analytic co-terminal swaption expiring at `t_j` (Jamshidian), matched by
    MC within `rel=2%` at 120k paths; (3) end-to-end into `cva` (positive, finite).
  - Consequence: feeding this `EE(t)` to `cva` (which multiplies by `DF(t_j)`) yields the
    correct discounted expected exposure `Σ_j swaption(t_j)·ΔQ_j` — CVA as a swaption strip.
  - quarry: `python/pricebook/risk/` · slice: `mc-exposure`

### Changed
- **`HullWhite.forward_short_rate(t, z)`** extracted as a model capability — the exact
  t-forward-measure short-rate draw, now shared by the MC swaption and MC exposure engines
  (rule of two). `coupon_bond_cashflows` now takes a `VanillaSwap` (three consumers), and
  `SwaptionMCEngine` reuses `forward_short_rate` — pure refactors under the green MC/analytic
  swaption oracles, no behaviour change.

## [0.25.0] - 2026-07-15

### Added
- **Unilateral CVA (L5 risk & capital)** — the first XVA. New `risk/xva.py` with an
  `ExposureProfile` (`EE(t) = E[(V(t))^+]` on a time grid) and
  `cva(profile, snapshot, key, recovery)` = `(1-R)·Σ EE(t_i)·DF(t_i)·(Q(t_{i-1})-Q(t_i))`.
  Structurally a CDS **protection leg** with the unit notional replaced by the exposure
  profile — CVA is protection bought on your own counterparty exposure.
  - Keyed to the counterparty survival curve in the snapshot (A5), reached through the
    `CurveHandle` `df` capability — so a credit bump (`bump_curve`/`credit01`) already
    yields CVA sensitivity, no new machinery.
  - Oracles: unit-exposure CVA **equals** the CDS protection leg (`cds_pv` at zero
    spread, already oracle-checked) to `1e-14`; linearity in exposure; a default-free
    (Q≡1) counterparty gives zero CVA.
  - Scope (unilateral): exposure ⟂ default (no wrong-way risk), own default ignored
    (that is DVA). **Exposure generation is upstream/out of scope** — `EE(t)` is an
    input (analytic for deterministic trades, MC for optional ones — a later slice).
  - quarry: `python/pricebook/risk/` · slice: `cva`

## [0.24.0] - 2026-07-15

### Added
- **Joint HW `(a, sigma)` calibration from a cap strip** — the calibration front's
  first multi-instrument least-squares fit. `calibrate_hull_white_cap(snapshot, quotes)`
  fits both mean reversion and vol to a strip of caplet (ZCB-option) quotes by
  minimising the repricing SSE. A single caplet can't separate `a` from `sigma`; a
  strip spanning expiries can (needs ≥2 quotes). `sigma` is fitted via its magnitude
  (price depends only on `sigma^2`), so the model carries `sigma >= 0`.
  - Oracles: round-trip recovers `(a*, sigma*)` from a self-priced strip
    (`a` to `1e-4`, `sigma` to `1e-5`); fitted model reprices the strip (SSE `< 1e-14`).
  - quarry: `python/pricebook/calibration/` · slice: `hw-cap-strip`
- **`nelder_mead` (L0 numerical toolkit)** — derivative-free downhill-simplex minimiser
  (Nelder & Mead 1965), the stdlib least-squares engine behind multi-parameter
  calibration (no scipy — the ng tree stays stdlib-pure, like `bisect_root`). Converges
  on **both** a flat objective and a small simplex, so a weakly identified direction
  (HW `a`) still pins down instead of drifting. Oracle: quadratic bowl + Rosenbrock.

## [0.23.0] - 2026-07-15

### Changed
- **Rate & credit bootstraps migrated under the L3 calibration front.**
  `bootstrap_discount_curve` and `bootstrap_survival_curve` move from `market/` (L1)
  to `calibration/` (L3), joining `calibrate_hull_white` as the per-family solvers of
  the unified front (`market -> calibrate -> model`). The curve *types*
  (`DiscountCurve`, `SurvivalCurve`) and market observables (`DepositQuote`,
  `ParSwapQuote`, `CDSQuote`) and CDS leg math (`cds_pv`, …) stay at L1 — quotes and
  curves are market data, the *solvers* are calibration. Each bootstrap still reprices
  with L1 closed forms (curve `df` / `cds_pv`), never the L4 engine, so `acyclic` stays
  green with `calibration` at rank 3.
  - Pure relocation guarded by the existing reprice-to-par / reprice-to-zero oracles
    (their tests move `tests_ng/L1 -> L3` accordingly). No behaviour change; 174 green.
  - Import path change: `from pricebook_ng.calibration.discount_curve import
    bootstrap_discount_curve` (was `...market.discount_curve`); likewise survival.

## [0.22.0] - 2026-07-15

### Added
- **Unified calibration front (L3), first tenant: Hull-White vol.** New
  `calibration/` package establishing `market -> calibrate -> model -> price` (A1):
  `calibrate_hull_white(snapshot, quote, a)` fits the HW `sigma` (with mean reversion
  `a` fixed) so the model reprices a `ZCBOptionQuote` — a caplet, i.e. a European
  option on a zero-coupon bond (Brigo & Mercurio s.3.3, the textbook HW vol instrument).
  - Correctly layered: the fit reprices with the model's own analytic `zero_bond_option`,
    **not** the L4 engine — calibration depends only on L0/L1/L3 (verify `acyclic` green
    with `calibration` at rank 3). The ZCB-option price is monotone in `sigma`, so one
    bracketed root pins it; an unreachable quote (below intrinsic) raises `ValueError`.
  - Oracles: round-trip sigma recovery (`abs=1e-9`); calibrated model reprices the quote
    (`abs=1e-10`); unreachable-quote raise.
  - Establishes the front's shape (`calibrate_*(snapshot, quotes, …) -> CalibratedModel`);
    the rate/credit curve bootstraps are the sibling solvers that migrate under it next.
  - quarry: `python/pricebook/calibration/` · slice: `calibration-front`

## [0.21.0] - 2026-07-15

### Added
- **Key-rate (bucketed) dv01.** `key_rate_dv01(priceable, snapshot, numerics)` in
  `risk/greeks.py` returns per-pillar KR01 for the home bootstrapped curve — bump one
  pillar's zero at a time (`DiscountCurve.bump_pillar(i, shift)`) and reprice. Log-linear
  interpolation tents each bump between neighbours, so the buckets **partition** the
  parallel `dv01` (Σ buckets = dv01). This turns the single parallel number into a
  per-tenor risk vector a hedger can actually neutralise.
  - Oracles: (1) buckets sum to `dv01` (partition of unity); (2) a cashflow landing
    exactly on pillar j is carried only by bucket j (`KR01_j = -N·t_j·DF(t_j)·1bp`,
    neighbours ~0, since the tent vanishes at the node).
  - quarry: `python/pricebook/core/discount_curve.py` · slice: `key-rate-buckets`

## [0.20.0] - 2026-07-14

### Added
- **Bootstrapped-curve dv01.** `DiscountCurve.bumped(shift)` — a parallel zero-rate
  shift (`DF -> DF·exp(-shift·t)` at every pillar), so log-linear interpolation keeps
  the shift uniform between pillars. `dv01`/`curve01` now compute rate risk on a real
  **bootstrapped** discount curve, not only the flat curve — closing the gap noted when
  generic curve greeks landed. Same shape as `SurvivalCurve.bumped`; `bump_rate` /
  `bump_curve` dispatch through it polymorphically (no `isinstance`).
  - Oracle: `dv01 = -1bp·Σ cf_i·t_i·DF_boot(t_i)` (analytic vs central-difference on the
    actual bumped curve); plus a check that every pillar's zero rose by exactly `shift`.
  - quarry: `python/pricebook/core/discount_curve.py` · slice: `bootstrapped-dv01`

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
