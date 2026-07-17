# Checkpoint CP-2c — field-hardness + fixed_income spine (fixings, deposit, OIS)

Date: 2026-07-17   ·   Version: `0.46.0`   ·   Tests: 255 green (`verify.py all`)

Third checkpoint, second parity-depth cluster. Executes the CP-2b ruling
(`rulings_CP2b.md`): the field-hardness gate + the ordered fixed_income spine (fixings →
deposit → OIS), OIS bringing `RateCurve`/`forward_rate`. **First cluster to apply the ratified
deletable-bar step** — every parity slice ended by reading its quarry counterpart; residuals are
logged in `quarry_reconciliation.md` (refreshed). Four mandatory review inputs below.

---

## 1. Slices landed (CP-2c cluster)

| Slice | Version | Layer | What |
|---|---|---|---|
| product-field-bundling | 0.43.0 | verify/L2 | `verify.py fields` gate (≤5 dataclass fields); FRA 7→4, CDS/options/ZCIS 6→5 via `Money`/`Accrual`; aggregates exempt |
| fra-seasoned-fixings | 0.44.0 | L4 | FRAEngine consumes `FixingHistory` (seasoned reset uses the fixing; settled → 0) |
| deposit-spine | 0.45.0 | L2 | `Deposit` — 2-cashflow money-market deposit via `DiscountingEngine` (A2: fwd par→0, spot par→principal) |
| ois-spine | 0.46.0 | L1/L2/L4 | OIS + `RateCurve` protocol + `curve.forward_rate`; shared `float_leg_pv`; FRA/HW onto the building blocks |

**Ledger deltas:** `verify.py fields` (+ CLAUDE.md §3b / redesign/09). Curves gained `forward_rate`;
`RateCurve` protocol added (HW typed to it, duck-type removed). New `products/deposit`,
`products/ois`, `engine/ois`; shared `float_leg_pv` extracted in `engine/swap`. ng modules 51 → 54.

## 2. Oracle-quality audit

| Slice | Oracle | Class |
|---|---|---|
| product-field-bundling | `verify.py fields` flags exactly the 8 over-limit; all product PVs **byte-identical** | gate + regression |
| fra-seasoned-fixings | seasoned = `face·τ·(fixing−K)·DF(end)`; K=fixing → 0; missing → failure; settled → 0 | **closed-form** |
| deposit-spine | fwd par → 0; spot par → principal; off-par closed form; bootstrapped par → 0 | **closed-form** |
| ois-spine | `forward_rate` == DF-ratio; par OIS → 0; **OIS == vanilla IRS** (cross-check on the S06 swap) | **closed-form + cross-check** |

All strong; none self-consistency-only.

## 3. Quarry-drawdown reconciliation (refreshed) + deletable-bar

- **768 quarry · 54 ng · deletable: 0. Drawdown still 0.0/768.** The **fixed_income spine advanced**
  (fra + deposit + ois added; each read against its quarry module, residual logged).
- **Deletable-bar reads this cluster** (the new ratified step, working):
  - `fixed_income/fra` — quarry ISDA-settle == ng end-settle (single-curve); ng *ahead* on fixings.
    Residual: multi-curve `forward_rate(projection)`, par_rate/pv_ctx, convention builder.
  - `fixed_income/deposit` — quarry values redemption-only; ng the trade (A2-reconciled). Residual:
    convention builder, implied-DF method (ng has via `DepositQuote`), pv_ctx, serialisation.
  - `fixed_income/ois` — single-curve == vanilla IRS. Residual: SOFR/SONIA/ESTR conventions,
    `bootstrap_ois`, par_rate/annuity/dv01, daily-fixing compounding, multi-curve basis.
- **The pattern in the residuals is the real finding (see §4.1):** every spine module's residual is
  the *same three cross-cutting families* — market **conventions**, **serialisation**, and
  **multi-curve** — plus a few analytics methods. No amount of per-vanilla work retires a module
  until those cross-cutting layers exist. Drawdown will stay 0 under pure vanilla-by-vanilla work.

## 4. Design choices to challenge

1. **HEADLINE — the deletable bar is cross-cutting, not per-instrument.** Every spine residual is
   conventions + serialisation + multi-curve. **Proposal:** before more vanillas, cut the
   cross-cutting slices (a **market-conventions** layer, a **serialisation** layer, **multi-curve**)
   that let a *batch* of modules go deletable at once — otherwise drawdown stays 0 for many slices.
   Re-sequence CP-3 around this?
2. **OIS == vanilla IRS in single-curve.** I built a distinct `OvernightIndexSwap`/`OISEngine` that is
   numerically the swap (single-curve); the real deliverables were `RateCurve`/`forward_rate`. Keep
   the thin OIS type now, or defer OIS until multi-curve (where it genuinely differs)?
3. **`verify.py fields` scans all of `src`** (not just `products/`+`foundation/` as CLAUDE.md §3b's
   text says) — I broadened it so it would catch `MarketSnapshot`/`XvaReport` (which the ruling
   listed as exempt). Ratify the all-src scope, or narrow it and drop those exemptions?
4. **`curve.forward_rate(d1, d2, day_count)`** takes the accrual day-count (the forward is
   convention-dependent); the quarry's `forward_rate(d1,d2)` uses the curve's own dc. Confirm the
   day-count-as-arg signature (it composes into `float_leg_pv`/FRA cleanly).

## 5. Smell + debt scan

- **Home curve is duck-typed for `forward_rate`/`bumped`.** `MarketSnapshot.discount_curve` is typed
  `CurveHandle` (df-only) because it needs both the `RateCurve` capability (forward_rate) *and*
  `bumped` (greeks), and no single protocol declares both. `RateCurve` is used where a rate curve is
  explicit (HW). — *proposed: accept; a `DiscountCurve`-capability protocol (df+forward_rate+bumped)
  only if a 3rd consumer needs it.*
- **OIS builder duplicates the `vanilla_swap` builder** (fixed_coupon_cashflows + generate_schedule).
  Rule-of-two (swap + OIS). — *proposed: extract a shared leg-builder if a 3rd swap-shaped product lands.*
- `_bracket_slope` O(n)-per-call ceiling (carried). No suppressions; `verify debt` green.

## 6. Ready-for-next / named next checkpoint

The fixed_income vanilla spine is broad (bond/swap/leg/cashflow/inflation/fra/deposit/ois). Per §4.1
the honest next move is **cross-cutting, not more vanillas**. **Proposed CP-3 (for Cowork to rule):**
- **(a) market-conventions layer** — SOFR/SONIA/ESTR + IBOR conventions, the shared residual of
  every spine module (unblocks `from_convention` builders across fra/deposit/ois/swap);
- **(b) serialisation layer** — the schema/serialisation residual (a `verify`-gated concern);
- **(c) multi-curve** — OIS-discount / IBOR-projection, which makes OIS≠IRS and closes the biggest
  pricing residual;
- or continue to the **credit spine** (#4) if breadth is preferred over retiring modules.

**Requesting Cowork:** rule §4.1 (cross-cutting vs more vanillas) and the CP-3 sequence; ratify §4.3
(fields scope) and §4.2 (thin OIS).
