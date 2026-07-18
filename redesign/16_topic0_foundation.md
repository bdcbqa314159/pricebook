# Artifact #16 — Topic 0: the Foundation (cross-asset complete)

**Status:** Draft for ratification. **Supersedes the multicurve-scoped foundation** in #14 §0.
Topic 0 runs **before** Topic 1 (yield curves).

---

## 1. Why this exists

The earlier plan scoped L0 to what *multicurve* needed — EUR/USD/GBP calendars, rate indices, three
`NumericalConfig` knobs. That repeats, one level down, the exact error we corrected at the curve
layer: **scope the foundation to one topic and every later topic retrofits it.** Retrofitting the
foundation is the most expensive change in the system — everything stands on it.

**The rule (same as D1):** *design complete, populate incrementally.* Cross-asset-capable natively,
rates as the first population — exactly as multi-curve-capable with single-curve as the degenerate
configuration.

**Evidence base.** What the quarry's non-rates asset classes actually consume from `core/`:
`DayCountConvention`/`year_fraction` (79 uses, all six classes) · `solvers.brentq` (18) ·
`Frequency`/`generate_schedule`/`StubType` (17) · `date_from_year_fraction` (7) · interpolation (4) ·
`Calendar`/`BusinessDayConvention` (3) · `CurrencyPair` · `notional` · serialisation (universal).
**Caveat:** usage proves the floor, not the ceiling. The quarry has no `Money` type, so nothing
"uses Money" — absence of usage ≠ absence of need. The gaps below come from domain reasoning.

---

## 2. The foundation (complete)

### 2.1 Time & dates
- `Date` (stdlib) · `Tenor` (string + helpers)
- **`Calendar`** — holiday-rule DSL (`fixed`/`easter`/`orthodox`/`nth`/`monday`, `since`/`until`),
  weekend rule, **three observance regimes** (US §6103 · Commonwealth · Johannesburg),
  `JointCalendar`. **Keyed by IDENTITY** (`TARGET`, `LONDON`, `NEW_YORK_SIFMA`), currency→calendar a
  lookup (C1). **All 37 markets declared** — each is a short declaration once the DSL exists.
- **`BusinessDayConvention`** — the 5 present **plus `NEAREST` and the EOM variant** (gaps).
- **`DayCountConvention` + `year_fraction`** — the 7 present **plus ACT/365L, 30E/360 ISDA, NL/365**
  (gaps). Calendar **passed in** (no hidden default); ICMA anchors via **`CouponPeriod`**.
- `Frequency` · `StubType` · **`RollRule`**(calendar, convention, eom)
- **`ScheduleTerms` → `Schedule`** — returns **both adjusted and unadjusted** dates (C2);
  EOM anchored **once from `start`** (ISDA §4.10).
- **`RollDate` / IMM conventions — GAP.** 3rd-Wednesday IMM (futures) and **CDS 20th of
  Mar/Jun/Sep/Dec**. Neither exists in the quarry. Credit *and* futures both need them.

### 2.2 Money, quantity, identity
- `Currency` · **`CurrencyPair`** (+ spot settlement lag, quote convention)
- **`Money`**(amount, currency) — currency-mixing is a type error
- **`Quantity`(amount, unit) — GAP.** Commodities settle in barrels/MWh/tonnes; `Money` alone
  cannot express physical delivery.

### 2.3 Instrument atoms
- **`Cashflow`**(date, `Money`) · **`Leg`** (ordered cashflows + convention)
- **`Accrual`**(start, end, day_count) · **`CouponPeriod`** (ICMA anchors)

### 2.4 Index / underlying identity  — **one concept, several instances**
- A general **declarative index/underlying identity**, of which **`RateIndex`** is the first
  instance. Equity, inflation and commodity underlyings get the same treatment.
  **Rule: a new index is a DECLARATION, never a code change.**
- `RateIndex` carries currency · tenor · day_count · `RollRule` · and the **full RFR set**:
  `fixing_lag` · `observation_shift` · `lookback` · `lockout` · `payment_delay` · `compounding`.
- **`FixingHistory` generic over index — GAP.** Ours is rates-shaped; Asian options and inflation
  need fixings too.
- **Exercise/expiry conventions — GAP.** expiry→settlement lag, exercise style vocabulary.

### 2.5 Finance-free numerics — **RULED: interfaces complete, algorithms demand-migrated**
- **In Topic 0:** `Interpolator` (mechanism only — the *curve* extrapolation policy is L1, C4) ·
  root-finding & optimisation (`brentq`, Nelder-Mead) · distributions.
- **`NumericalConfig` carries the FULL knob set now** — MC (paths/seed/antithetic/sobol/bridge) ·
  PDE (time/space steps, n-std-devs) · tree steps · quadrature (tol, max-iter) · COS (n, L) ·
  root-finder (tol, max-iter). *This is the one that would otherwise retrofit:* the 12 knobs we
  deferred at retire #1 are `deferred→_fourier/_pde/_trees/_integrate/_rootfinding` and would come
  back as a change to a foundational value type. `# fields-exempt: config aggregate`.
- **Demand-migrated (no shape risk):** MC engine, PDE, Fourier/COS, trees, quadrature — they are
  independently testable and arrive with their first consumer.

### 2.6 Engine I/O & serialisation
- **`PricingResult`** (decomposition: dirty PV + cashflow/accrual breakdown ⇒ clean, accrued;
  sensitivities; diagnostics) · **`PricingFailure`** (failure as a value)
- **Serialisation pattern** — per-class `to_dict`/`from_dict` + `schema_version`. Universal, so the
  *pattern* is foundational; **no framework** (the quarry's 831-line machinery is not carried).

---

## 3. Not in the foundation
Curves · `CurveSet` · `MarketSnapshot` · quotes (**L1**) · products (**L2**) · models incl. Black-76
and Hull-White (**L3**) · engines (**L4**) · greeks/XVA (**L5**) · trade/book/P&L (**L6**).

**The membership test:** *conventions describe how time and money are counted; pricing computes
value.* Day count in. Black-76 out. No valuation in L0 — enforced by `verify.py layers`.

---

## 4. Topic 0 slice order

| # | slice | delivers |
|---|---|---|
| 0 | `ng-parking` | park **all** current ng to `ng_parked/`; rebuild clean (ng_parked is a content source, never a structural one) |
| 1 | `calendars` | DSL + 3 observance regimes + `JointCalendar` + **all 37 markets**, identity-keyed |
| 2 | `daycounts` | 10 conventions (7 + 3 gaps), `CouponPeriod`, calendar passed in, `strict_icma` deleted |
| 3 | `schedules` | `Frequency`/`StubType`/`RollRule`/`ScheduleTerms`; adjusted **and** unadjusted; **IMM + CDS roll dates** |
| 4 | `money-quantity` | `Currency`/`CurrencyPair`(+lag) · `Money` · **`Quantity`** · `Cashflow` · `Leg` · `Accrual` |
| 4b | **`settlement`** | cash / physical / auction · settlement ccy ≠ contract ccy · settlement lag · **`Delivery(date, Quantity)`** alongside `Cashflow(date, Money)`. *Mine `core/settlement.py` — a 398-LOC zero-fan-in orphan.* |
| 5 | `index-identity` | declarative index concept + `RateIndex` covering **all rate kinds** — RFR **and** forward-looking term/IBOR (`observation_style`) + **`spread_adjustment`** (ISDA fallbacks) — generic `FixingHistory`; **sibling index types** (inflation level+lag+interpolation · FX fixing source/time · equity/commodity observation) defined, populated later |
| 6 | `numerics-config` | `Interpolator` · solvers · distributions · **complete `NumericalConfig`** · `PricingResult`/`PricingFailure` · serialisation pattern |

**Oracles:** published throughout — ISDA 2006 §4.16 · ICMA Rule 251 (incl. the UST coupon =
**exactly 2.0000** regression) · known holiday and observance dates · IMM/CDS roll date tables ·
lookback vs observation-shift must give **different** rates.

**Checkpoint** after slice 3 (cadence ≤6) and at Topic 0 close. Topic 1 (yield curves) does not
begin until Topic 0's gate is green.
