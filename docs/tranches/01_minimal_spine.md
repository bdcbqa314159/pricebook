# Tranche 1 — the minimal spine

**Proposed:** 2026-10-31 · **By:** Code (lead, per thread 0003) · **Status:** proposed, awaiting
Cowork's independent review before landing · **Target:** `v0.1.0`

## 0. Target and why this one

**Price a single fixed cashflow on a flat curve, end to end**, through the stateless
`price(product, model) → PricingResult`.

Bernardo did not name a product (thread 0003 leaves the first target to interest), so this is the
suggested opener: the backbone every product needs and no product can avoid. It **prejudges
nothing** — a swap, a CDS and an equity option all discount cashflows — while proving the one thing
that has to be right before anything interesting pulls on it: *the engine contract*.

It is also the smallest tranche with a **rung-1 oracle that is not self-referential**: `N·e^{−rt}`
is arithmetic, written independently of the code that produces it.

## 1. The pieces, by layer

```
L0  Money(amount, currency)        · Currency        — value types, currency-mixing a type error
    Date handling + DayCountConvention.year_fraction  — ACT/365F only for now
    TimeMeasure(anchor, day_count)                    — the ONE date→t map
    Cashflow(date, amount)                            — the instrument atom
    PricingResult / PricingFailure                    — failure is a VALUE, never an exception
L1  DiscountCurve (flat) + CurveHandle protocol       — reached through `df(date)`
    CurveSet · CurveKey                               — curves keyed, not positional
    MarketSnapshot(valuation_date, curves)            — immutable; carries the valuation date
L2  FixedCashflow(cashflow, currency?)                — frozen data; does NOT price itself
L3  DiscountingModel(market)                          — carries the snapshot it was built on
L4  engine.registry (register/dispatch) + price_fixed_cashflow
```

**Deliberately NOT in this tranche:** schedules, legs, conventions beyond ACT/365F, interpolation
beyond flat, projection curves, any second product, any model with dynamics. Each arrives when a
product pulls it.

## 2. Stock → purify, or build fresh

Everything needed exists in `_stock/` and is **purify-ready** — this tranche is mostly adaptation,
which is the point of keeping the stock.

| piece | stock source | lines | call |
|---|---|---|---|
| `Money`, `Currency` | `foundation/money.py` | 289 | **purify, trimmed hard.** It carries a currency registry, `Quantity`, `Unit`, `CurrencyPair` — none pulled by this tranche. Take `Money` + `Currency` + the mixing guard; leave the rest in the stock until a consumer asks |
| day count / `year_fraction` | `foundation/day_count.py` | 332 | **purify, ACT/365F only.** The file holds the full ISDA set plus ICMA coupon machinery; one convention is all a flat-curve cashflow needs |
| `TimeMeasure` | `foundation/time_measure.py` | 33 | **purify as-is.** Already minimal, and it is the "one date→t map" convention the register's C2 is about |
| `Cashflow` | `foundation/cashflow.py` | 87 | **purify, minus `Leg` and `accrual`.** Accrual belongs with the first coupon-bearing product |
| `PricingResult` / `PricingFailure` | `foundation/results.py` | 48 | **purify as-is.** `pv`/`accrued`/`clean`/`basis` is a decomposition, not a scalar — keep it |
| `DiscountCurve`, `CurveHandle` | `market/curve.py` | 122 | **purify, flat + log-linear only.** Drop Hagan–West monotone-convex until a curve-build consumer needs it |
| `CurveSet`, `CurveKey` | `market/curve_set.py` | 87 | **purify, discount role only.** Projection/survival roles arrive with their products |
| `MarketSnapshot` | `market/snapshot.py` | 154 | **purify, curves only.** Its `scalars`/`surfaces`/key-union grew per asset class; start with `valuation_date` + `curves` |
| `DiscountingModel` | `models/discounting_model.py` | 30 | **purify as-is** — 30 lines and already exactly this |
| `engine.registry` | `engine/registry.py` | 67 | **purify as-is.** Structural dispatch is what keeps `isinstance`-on-product out of the engine |
| `FixedCashflow` product | `products/cash.py` | 52 | **build fresh.** The stock's `cash.py` is `Deposit`/`FRA`/`Future`; a bare fixed cashflow product is new |
| the pricer | `engine/cash.py` | — | **build fresh**, mining the stock's shape: discount the flow, exclude it if historical |

**Estimated: ~10 purified modules, 2 built fresh, no file over ~120 lines.**

## 3. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | `PV = N·e^{−r·t}` to **<1e-12**, across notionals, rates and horizons | **1** — closed form | the reference is arithmetic written in the test, independent of every code path under it |
| 2 | **analytic vs finite-difference DV01** to <1e-6 | **1** | an analytic sensitivity and a bump-reprice are two different computations; agreement is not tautological |
| 3 | a flow **on or before** the valuation date is excluded from PV, and a flow **after** is not | structural | invariant 6; the boundary case (`date == valuation_date`) is where off-by-one lives |
| 4 | repricing the same inputs is **byte-identical**; the snapshot is never mutated | structural | statelessness and immutability, same process |
| 5 | an unpriceable request returns `PricingFailure`, never raises and never a silent `NaN` | structural | failure-is-a-value is a contract, so it is tested as one |
| 6 | currency mixing is a **type/value error**, not a silent addition | structural | the one thing `Money` exists for |

**On oracle 1 being "just arithmetic":** that is the strength. The 5.14% bug passed because its
oracle restated the builder's own composition; `N·e^{−rt}` restates nothing — if the curve, the day
count and the engine disagree, the number moves.

## 4. Decisions I am making, flagged for review

1. **ACT/365F only, flat curve only.** Not a limitation to apologise for — the day-count and
   interpolation families in the stock are large, and pulling them in now would be building for
   products that have not asked. Each arrives with its first consumer.
2. **Serialisation (`to_dict`/`from_dict`) IS included on `Cashflow` and `FixedCashflow`.** This is
   the call I most want reviewed. Arguments for: the new scope line makes *how a product and a
   pricing request are represented* the **durable public contract**, so designing products with it
   from the start is building toward the stated aim; and the old tree explicitly ruled that adding
   serialisation per-product while already inside the module beats a mass retrofit later. Argument
   against: no consumer exists yet (the pricing server is a later tranche), so by §6b it is
   speculative. **I come down on including it** — the retrofit cost is real and the forward
   reference is explicit in the scope — but it is a 20-line reversal if you disagree.
3. **`PricingResult` keeps `accrued`/`clean`/`basis`** even though a bullet cashflow has no accrual
   and no CSA. Purified as-is rather than trimmed, because it is a *decomposition* the scope
   commits to and trimming it would mean re-widening the public result type at the first coupon.
4. **No `CLAUDE.md` charter yet.** Tranche 1 is the first real evidence about what the conventions
   should say; writing the charter from one tranche's experience beats writing it from memory of the
   retired frame. Proposed: charter after tranche 2 or 3, when the conventions have been *used*.

## 5. Process

Branch `tranche/01-minimal-spine`. Commit sequence: `test` (oracles RED) → `feat` (green) → `docs`
(this file updated with results) → `chore(release): v0.1.0`. Provenance header on every module:
`stock:` the file purified from, `source:` the reference, `oracle:`, `slice:` the tranche.

Gates: all eight, plus ruff and pyright, plus the test floor raised in the same commit.
