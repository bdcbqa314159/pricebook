# Artifact #14 — Topic 1 object model: the yield-curve ecosystem

**Status:** Draft for ratification. The concrete objects of Topic 1 (#13), bottom-up. Written as
**API sketches** because the legibility requirement is literal: *you must be able to read the code and
understand it immediately.* If a sketch below needs explaining, the design is wrong.

---

## 0. Foundational objects (L0) — the atoms everything speaks

```python
Date                     = datetime.date                # stdlib; no custom date
DayCountConvention       # ACT/360, ACT/365F, 30/360, ACT/ACT ISDA, ACT/ACT ICMA, BUS/252
year_fraction(start, end, convention, *, period=None, calendar=None) -> float

Calendar                 # holidays + is_business_day + adjust
BusinessDayConvention    # FOLLOWING, MODIFIED_FOLLOWING, PRECEDING, ...
RollRule(calendar, convention, eom)                     # the bundled adjustment rule
Frequency                # ANNUAL, SEMI, QUARTERLY, MONTHLY
ScheduleTerms(start, maturity, frequency, roll)         # -> generate_schedule() -> list[Date]

Currency ; Money(amount, currency)                      # currency-mixing is a type error
Accrual(start, end, day_count)      .year_fraction() -> float
Cashflow(date, amount: Money)                           # the atom
Leg = ordered Cashflows + convention
```

**The bridge object to the curve world:**

```python
@dataclass(frozen=True)
class RateIndex:            # THE multi-curve key: one projection curve per index
    currency: Currency
    tenor: Tenor            # "3M", "6M", "ON"
    day_count: DayCountConvention
    roll: RollRule
    fixing_lag: int         # business days

EURIBOR_3M = RateIndex(EUR, "3M", ACT_360, TARGET_ROLL, 2)
ESTR       = RateIndex(EUR, "ON", ACT_360, TARGET_ROLL, 0)
```

**`RateIndex` is pure declarative metadata — a new index is a DECLARATION, never a code change.**
It carries the **full RFR convention set**: `compounding` (COMPOUNDED / AVERAGED / FLAT),
`fixing_lag`, **`observation_shift`** (shifts fixing *and* accrual weighting), **`lookback`**
(shifts the fixing date, accrual weighting stays on the original period — *a different number*),
**`lockout`** (rate frozen for the final N days), `payment_delay`.

```python
get_rate_index("SOFR") -> RateIndex          # name -> metadata
accrued_rate(index, accrual, fixings) -> float   # ONE generic function, driven by the fields
```
No `SOFRIndex` subclass, no `if index.name == …`. ESTR and SONIA differ *only* in their
declarations (SONIA 0/0 vs SOFR 2/2). This is "products are pure data" applied to conventions.

---

## 1. The curve ecosystem (L1) — single and multi-curve as ONE design

```python
class YieldCurve(Protocol):                 # the capability, not a class
    def df(self, d: Date) -> float: ...
    def zero_rate(self, d: Date) -> float: ...
    def forward_rate(self, start: Date, end: Date, day_count) -> float: ...

FlatCurve(rate, anchor, day_count)
InterpolatedCurve(pillars, interpolation)   # pillars = [(date, df)]
Interpolation                               # LOG_LINEAR_DF | LINEAR_ZERO | MONOTONE_CONVEX
```

**`CurveSet` — the multi-curve container, and the heart of the topic:**

```python
@dataclass(frozen=True)
class CurveSet:
    discounts:   dict[DiscountKey, YieldCurve]     # DiscountKey(currency, collateral_currency)
    projections: dict[RateIndex,   YieldCurve]

    def discount(self, ccy, collateral=None) -> YieldCurve: ...
    def projection(self, index: RateIndex) -> YieldCurve: ...
```

```python
# multi-curve:  ESTR discounting, EURIBOR 3M projection
curves.discount(EUR)                 -> estr_curve
curves.projection(EURIBOR_3M)        -> euribor3m_curve

# single-curve is the DEGENERATE CONFIGURATION, not a second code path:
CurveSet.single(curve)               # every projection(index) returns `curve`

# xccy (D3): collateral currency selects the discount curve
curves.discount(EUR, collateral=USD) -> eur_usd_collateral_curve
```

**The economy the model is built on:**

```python
@dataclass(frozen=True)
class MarketSnapshot:               # fields-exempt: aggregate
    valuation_date: Date
    curves: CurveSet
    fx_spots: dict[CurrencyPair, float]
    fixings: FixingHistory
```

---

## 2. Market quotes (the inputs) — distinct from curves and from products

```python
DepositQuote(index, maturity, rate)
FRAQuote(index, start, end, rate)
ParSwapQuote(currency, tenor, rate, fixed_freq, float_index)
OISQuote(currency, tenor, rate)
BasisSwapQuote(index_a, index_b, tenor, spread)
XccyBasisQuote(pair, tenor, spread, collateral)
```

A quote is *market data you calibrate to*. A product is *a trade you price*. Never the same object.

---

## 3. Products (L2) — pure data, multi-curve aware

```python
@dataclass(frozen=True)
class VanillaSwap:
    fixed_leg: FixedLeg          # rate + accruals + notional(Money)
    float_leg: FloatLeg          # index: RateIndex + accruals + spread
    # no pv() — pricing lives in L4
```
`Deposit` · `FRA` · `VanillaSwap` · `OIS` · `BasisSwap` · `FixedRateBond` · `FXForward`.
The float leg carries a **`RateIndex`**, which is how the engine knows *which projection curve* to use.

---

## 4. Curve construction = calibration (L3)

```python
CurveBuildSpec(target, quotes, interpolation)          # what builds what

bootstrap(spec, snapshot) -> YieldCurve                # sequential, one curve
solve(specs, snapshot)    -> CurveSet, CalibrationResult   # simultaneous, multi-curve
```
`CalibrationResult` carries residuals, iterations, convergence, provenance — diagnostics are first
class, not printed. **Oracle: every input quote reprices to par off the built curve.**

---

## 5. Pricing (L4)

```python
model  = DiscountingModel(snapshot)          # the model carries its snapshot (A1)
result = engine.price(product, model, numerics) -> PricingResult
```
For linear rates products the "model" is just the calibrated curve world. `PricingResult` is a
decomposition: dirty PV, accrued, clean, cashflow breakdown.

```python
# the whole chain, legibly:
quotes   -> solve(...)      -> CurveSet
CurveSet -> MarketSnapshot  -> DiscountingModel
engine.price(swap, model, numerics).pv
```

---

## 6. Risk (L5) — bump the market, rebuild the model, reprice

The ratified pattern (A1/A6): **risk perturbs the snapshot, the model is rebuilt from it, the product
is repriced.** Nothing bumps a model directly.

```python
class Priceable(Protocol):
    def __call__(self, snapshot: MarketSnapshot) -> Money: ...

bump_curve(snapshot, key, shift)                  # parallel shift of one curve
bump_pillar(snapshot, key, pillar, shift)         # one node  -> key-rate buckets
bump_quote(quotes, quote_id, shift)               # one market instrument -> PAR delta (rebuilds!)

dv01(priceable, snapshot, key)                    # parallel
key_rate_dv01(priceable, snapshot, key)           # per-pillar; Σ buckets == dv01
par_delta(priceable, quotes, spec)                # sensitivity to the QUOTES you hedge with
CurveScenario(name, apply: snapshot -> snapshot)  # named, composable
```

### Ratified: both bases, par via bump-and-rebuild
- **Zero / pillar (key-rate) deltas** — bump a node of the *built* curve. Cheap. Oracle: buckets sum
  to the parallel DV01; analytic vs FD agree.
- **Par deltas** — bump each **quote** and **re-run the curve build**, then reprice. These are the
  numbers a desk hedges with, so they are in scope now, not deferred. Oracle: a par-rate bump on the
  instrument's own pillar reproduces its analytic sensitivity; a self-hedge nets to zero.
- **No calibration Jacobian yet.** Bump-and-rebuild is slower but obviously correct and keeps risk
  decoupled from calibration internals. Topic 1 has few pillars, and correctness-first is the stance.
  *Re-open trigger:* if curve-build cost makes par risk impractical (a real perf complaint, not a
  hypothetical), introduce the Jacobian then — with the bump-and-rebuild result as its oracle.

This ordering matters: bump-and-rebuild **becomes the reference** that any future fast path must
reproduce. Build the slow correct one first.

---

## 6b. Corrections forced by the quarry read (amendments — see `rulings_ng_parking_and_topic1_design.md`)

- **C1 Calendars keyed by identity, not currency.** `TARGET`/`LONDON`/`NEW_YORK_SIFMA`; currency →
  calendar is a *lookup*. (Quarry keys by currency and admits it cannot express joint calendars.)
- **C2 Schedules return adjusted AND unadjusted dates** — accrual uses unadjusted, payment uses
  adjusted. (Quarry adjusts in place and loses the accrual dates.)
- **C3 `CurveBuild` owns pillar placement as an explicit rule** — default: pillar at the **rolled
  schedule end**, FRA `df(start)` **pinned as an extra pillar**. Without this, reprice-to-par fails.
  (Quarry documents both fixes and is still internally inconsistent between its two bootstraps.)
- **C4 `Interpolation` states its extrapolation policy on both ends**; flat-**forward** is the default
  right policy. (Quarry: only log-linear does flat-forward; others silently go flat-in-level.)
- **C5 Solver curve addressing** — a `CurveRef` resolving into the `CurveSet`, to give the solver
  generic addressing over its state vector. Resolve at the `CurveBuild` slice.

## 7. The legibility contract (what "understand it right away" means)

1. **Named domain objects, never nested dicts.** `curves.projection(EURIBOR_3M)` — not
   `market["curves"]["EUR"]["3M"]`.
2. **The code reads like the domain sentence.** *"Discount EUR under USD collateral"* →
   `curves.discount(EUR, collateral=USD)`.
3. **One concept, one object.** Quote ≠ product ≠ curve. Never overloaded.
4. **No indirection without a present need** (§6b) — depth of abstraction is a cost.
5. **Provenance in every module header** — the paper/convention it implements.
6. **≤5 args / ≤5 fields** — if a signature needs explaining, bundle it.
