# Artifact #4 — Slice Plan (DRAFT)

**Status:** Draft. Slice 0 fully specified below; later slices sketched. Each slice is a
thin *vertical* cut through every layer it needs, and **ships with an oracle** — a
red/green comparison to a known-correct value. A slice that can't be checked against its
oracle in one pass is too big and must be split.

---

## Slice 0 — the walking skeleton

**Goal:** prove the entire spine holds by pricing the simplest possible trade end-to-end
through every layer, before *anything* else migrates. It is deliberately trivial: its
only job is to make the layers, the stateless engine, and the oracle mechanism real.

### The trade
A **single fixed cashflow**: pay `notional` in currency `C` at date `T`. No coupons, no
schedule, no optionality. (Equivalently: a zero-coupon bond.)

### What each layer contributes
| Layer | Contribution | Vocabulary exercised |
|---|---|---|
| **L0 Foundation** | `Date`, `DayCountConvention.year_fraction`, `Money(notional, C)`, `Cashflow(date=T, amount=Money)` | the promoted `Cashflow` at L0; `Money` at the boundary |
| **L1 Market data** | a flat `MarketSnapshot`: one continuously-compounded rate `r`; a `DiscountCurve` exposing `df(t)=exp(-r·t)` behind a `CurveHandle` | immutable snapshot; curve-as-handle |
| **L2 Instrument** | `FixedCashflowTrade` — a frozen dataclass holding one `Cashflow`. **No `pv` method.** | instrument = pure data |
| **L3 Model** | none (discounting needs no dynamics) — proves the engine works with a null model | — |
| **L4 Engine** | `DiscountingEngine.price(trade, model=None, market, numerics) → PricingResult(pv=Money)` where `pv = notional · df(T)` | the stateless engine contract |
| **L5 Risk** | one sensitivity: DV01 by bumping `r` in the snapshot and re-pricing | risk depends on the engine, not the trade type |
| **L6 Shell** | `book(trade)` → `BookedTrade`; `book.value(date)` runs the engine and stores a `PricingResult` | stateful lifecycle over stateless core |

### The oracle (red/green)
Flat curve, continuous compounding ⇒ **closed form, exact to machine precision**:

```
t   = year_fraction(valuation_date, T, ACT/365F)
DF  = exp(-r · t)
PV  = notional · DF
```

- **Price oracle:** `engine.price(...).pv.amount == notional * exp(-r*t)` to < 1e-12.
- **Risk oracle:** analytic DV01 `= -notional · t · exp(-r·t) · 1e-4`; finite-difference
  DV01 from the engine must match to < 1e-6.
- **Statelessness oracle:** pricing the same trade twice, and pricing after a risk bump,
  returns byte-identical results (no state leaked).

### Definition of done for Slice 0
All three oracles green; the new tree contains a runnable path
`book(trade).value(date)` that touches L0→L6; `PricingContext` used is frozen; no entry
crossed from the quarry without adaptation. When green, the spine is proven and the
bottom-up foundation migration begins.

---

## After Slice 0 — provisional order (to be detailed per layer)

Each is a thin vertical slice with its own oracle; order follows the dependency stack
bottom-up, and within a layer, simplest-oracle-first.

1. **S1 — Day-count & schedule** → oracle: published ISDA/ICMA year-fraction test vectors.
2. **S2 — Bootstrapped discount curve** (from deposit/swap quotes) → oracle: re-price the
   input instruments to par (self-consistency) + QuantLib cross-check.
3. **S3 — Fixed-rate bond** (multi-cashflow, real schedule) → oracle: closed-form PV +
   QuantLib.
4. **S4 — Vanilla IRS** (fixed vs float leg, projection curve) → oracle: par swap rate
   reprices to zero NPV; QuantLib.
5. **S5 — First model: Hull-White** + swaption engine → oracle: analytic HW swaption vs
   MC engine convergence.
6. **S6 — Credit: CDS + hazard bootstrap** → oracle: par-spread reprices to zero upfront;
   ISDA model cross-check.
7. **S7 — Risk relocated to L5** (greeks/XVA on the `Pricable` protocol) → oracle: bump-
   and-reprice greeks vs analytic where available; XVA vs simplified closed form.
8. **S8+ — Remaining asset classes** (fx, equity, commodity, structured, crypto), each as
   its own slice set with per-payoff oracles, per the full-scope decision.

The L0 ledger determines exactly which foundation entries feed S0–S2; those are the first
rows to rule on.

---

## Amendment-driven refactor slices (2026-07, after S0–S4 landed)

Surfaced during the build; each is a guarded refactor (behaviour-preserving where noted)
with its own oracle, landed before forward work resumes.

- **S05 — signature bundling** (CLAUDE.md §3b). Bundle oversized signatures into value
  objects (`RollRule`, `CouponPeriod`, `ScheduleTerms`, `Money`); enable ruff `PLR0913`
  `max-args=5`. Guarded by existing green oracles; no behaviour change.
- **S06 — engine → model-only + snapshot=curves+fixings** (Amendments A1). Change to
  `price(product, model, numerics)`; `build(snapshot) → CalibratedModel`; `DiscountingModel`
  wraps the curve; `FixingHistory` is first-class in `MarketSnapshot`. Refactor the Slice-0
  engine + call sites. Oracle: unchanged PVs (behaviour identical) + a model/market-binding
  test (a model can only price against its own snapshot).
- **S07 — temporal decomposition** (Amendment A2 + A3 mark). Retire the "fail on past
  cashflow" guard; replace with **segment-and-settle** (past → settle, current → accrue,
  future → price). `PricingResult` grows the dirty-PV + cashflow/accrual breakdown. Oracles
  (closed-form): seasoned bond excludes the paid coupon; forward-starting prices only future
  flows; dirty = clean + accrued to <1e-12; cashflow exactly on valuation_date is historical.
- **S08 — Product/Trade/Book + benefit table** (Amendment A3, L6 shell). Rename the L2 atom
  `instrument → product`; `Trade` holds a collection of products + start date; `Book` = trades;
  `BookedTrade` remembers the **benefit table** (realized-cash P&L, ties to quarry
  `pnl_history`). Oracle: realized + mark reconcile to total economics over a trade's life;
  realized P&L = Σ actually-paid cashflows (no discounting).

Order: S05 → S06 → S07 → S08, then resume the forward migration (S1 day-count onward re-slots
behind these). S06 precedes S07 (temporality reads `valuation_date` through `model.market`);
S08 last (the L6 benefit table consumes the core's segment-and-settle output).

---

## Slice discipline (the bar, restated)
- One vertical cut; touches only the layers it needs; nothing speculative.
- Ships green against a named oracle before it counts as done.
- Migrates entries by **copy-adapt** from the quarry (never copy-paste); see artifact #5.
- Any debt incurred is logged, never silenced.
