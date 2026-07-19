# Artifact #18 — Topic 1: the Yield-Curve World

**Status:** Ratified scope. Supersedes the Topic-1 sketch in #13/#14 (those remain for the object
model and the quarry scoping read). Begins now that L0 is closed at v0.74.1.

## The goal, in one sentence
**A solid yield-curve production line, end to end: from cash-instrument quotes to a calibrated,
risk-differentiable `CurveSet` — single- and multi-curve, CSA-aware, including xccy and basis.**

---

## 1. The pillar taxonomy — what "cash instruments" means

Each pillar type maps to a **specific curve**. That mapping is what makes this multicurve rather
than a pile of quotes.

| segment | instrument | builds |
|---|---|---|
| short end | **deposits** (ON/TN, 1W–12M) | front of the discount curve |
| | **FRAs** (3x6, 6x9, …) | forward segments of a projection curve |
| | **futures** (IMM-dated) | 3M–3y — the *liquid* front |
| mid / long | **par swaps** (fixed vs index) | the projection curve for that index |
| | **OIS** (fixed vs RFR) | the discount curve under CSA |
| basis | **tenor basis** (3M vs 6M) | the second projection curve |
| | **xccy basis** | the foreign-collateral discount curve |

Quotes are **market data**, distinct from products: `DepositQuote` · `FRAQuote` · `FutureQuote` ·
`ParSwapQuote` · `OISQuote` · `BasisSwapQuote` · `XccyBasisQuote`.

## 2. Futures — approximated by forwards (ratified, with conditions)
Convexity requires a model (Hull–White) and belongs to a later block. **Approximate futures as
forwards now.** Two honesty conditions:
- The **oracle tests that the forward approximation is applied correctly**, never that the futures
  price is right.
- **Recorded re-open trigger:** add the convexity adjustment when the models topic lands.
Note the error is genuinely small at the front (a few bp inside 2y) and grows past 5y — i.e. it is
"no harm" exactly where futures are the liquid instrument.

## 3. Curve construction — the calibration methods in scope
- **Sequential bootstrap** — pillar by pillar, each solved with Brent. The reference implementation.
- **Simultaneous / global solve** — multi-curve Newton across the state vector (discount +
  projections + basis), the `ncurve_solver` `CurveSpec` + `InstrumentPricer` pattern generalised.
- **Par→zero Jacobian** — an *artifact of the solve* (the solver already forms ∂residual/∂zero).
  Feeds par risk. **In scope.**
- **Interpolation** — the L0 point schemes, **plus Hagan–West monotone-convex** (deferred out of L0
  specifically to land here; it reconstructs a function from *interval averages* and is the
  practitioner standard for forward curves). **Missing from both trees — genuine new work.**
- **Turn-of-year / seasonality** — the year-end funding spike is a real curve feature
  (quarry: `seasonal_curve`).
- **Pillar placement is explicit** (C3): pillar at the **rolled schedule end**, FRA `df(start)`
  **pinned**. Without this, reprice-to-par fails.

## 4. CSA, discounting and xccy
- **Collateral selects the discount curve** — `curves.discount(ccy, collateral=X)` (D3). CSA is the
  numeraire choice, made explicit.
- **`DiscountBasis`** on `PricingResult` already records what a PV was discounted on (L0, done).
- **xccy basis curves** built in this topic; the foreign-collateral discount curve is the output.
- Mine `fixed_income/csa.py` for the collateral rules.

## 5. Risk (ships with the curves)
- **Zero / pillar (key-rate) deltas** — bump the built curve. **Oracle: buckets sum to parallel
  DV01** — the quarry's do *not*, and its code knows it. A real gate they never had.
- **Par deltas** — bump each **quote**, re-run the build, reprice. The numbers a desk hedges with.
  **Bump-and-rebuild is the reference**; the Jacobian path must reproduce it.
- **Curve scenarios** — named, composable snapshot transforms.

## 6. Explicitly OUT (with reasons, not deferrals-by-omission)
- **AAD** — a general differentiation *framework* (tape, operator overloading), cross-cutting across
  options and XVA, not a curve concept. **Its own topic**, and its oracle is the bump-and-rebuild /
  Jacobian result produced here. Building it inside Topic 1 would invert the ratified
  "slow-correct-version-first-as-the-reference" ordering.
- **ML curve techniques** — an ML-fitted curve has **no oracle** unless a trusted deterministic curve
  exists to judge it against. This topic *is* that reference, so it is a **prerequisite, not a
  delay**. Own topic, afterwards, where it becomes falsifiable.
- **Futures convexity** — needs a model (see §2).
- Optionality (caps/swaptions), credit, equity, commodity, inflation, XVA — later topics.

## 7. Currencies
**EUR · USD · GBP** populated, from **synthetic quote sets on real conventions** (the G10 JSON
convention tables are real; the quotes are checked-in synthetics — the quarry has essentially no
market data). Live feeds are the separate market-data topic.

## 8. Oracles
- **Every pillar reprices to par** off the built curve — after the *full* curve is built, not just at
  solve time (C3).
- **Single-curve == multi-curve degenerate case**, exactly.
- **Key-rate buckets sum to the parallel DV01.**
- **Par delta by bump-and-rebuild == Jacobian-implied par delta.**
- **Discounted EE / forward invariants** where a closed form exists.
- xccy: a cross-currency par instrument reprices to zero under its collateral discounting.

## 9. Topic 1 is a FULL VERTICAL (L1→L6), in three clusters

A curve with no consumer is unfalsifiable: *"every pillar reprices to par"* only means something when
a real swap, priced through a real engine off the built curve, returns zero NPV. So the topic spans
the spine:

```
L1 market data      quotes · CurveSet · snapshot · FX spots
L3 model+calibrator bootstrap / global solve / Jacobian → DiscountingModel
L2 products         deposit · FRA · future · swap · OIS · basis · bond
L4 engine           price(product, model, numerics)
L5 risk             zero / pillar / par deltas · scenarios
L6 trade/portfolio  Trade → Book · monitor · stress
```

**Most of the vertical is a RE-BASE, not greenfield.** `ng_parked/` already holds the linear
products, the discounting and swap engines, `Priceable`/greeks, and the `Trade`/`Book` shell with the
benefit table — built single-curve on the old foundation. Topic 1 is therefore:
- **new:** curve machinery — `CurveSet` · Hagan–West · bootstrap/global solve · Jacobian · CSA/xccy · quotes
- **re-based from `ng_parked`:** products · engines · risk protocol · trade/book shell
`ng_parked` is a **content** source exactly like the quarry — mined, never structurally copied.

**Integration is the ratified contract — no element invents its own curve access:**
1. **A1** — the model carries its snapshot; the engine takes `(product, model, numerics)`; *nothing
   reaches for curves directly.*
2. The float leg's **`RateIndex` selects the projection curve.**
3. **Collateral/CSA selects the discount curve** (D3).

### The three clusters (checkpoint each; ONE parking event at topic close)
| cluster | delivers | closes when |
|---|---|---|
| **C1 market data + construction** | quotes · `CurveSet` · interpolation incl. Hagan–West · bootstrap · global solve · Jacobian · CSA/xccy · snapshot assembly | every pillar reprices to par; single-curve == multi-curve degenerate |
| **C2 model + engine + products** | `DiscountingModel` · `price(product, model, numerics)` · deposit/FRA/future/swap/OIS/basis/bond (re-based multi-curve aware) | a par swap prices to **zero NPV** off the built curve, end to end |
| **C3 trade / portfolio / risk** | `Trade` → `Book` · monitor (realized vs mark, A3) · zero/pillar/par deltas · scenarios · stress | key-rate buckets sum to DV01; par delta by bump-and-rebuild == Jacobian-implied; a book stresses coherently |

**Re-classification this forces:** `core/trade`, `core/book`, `core/daily_pnl` move from the "shell
topic" **into Topic 1** — `Trade`/`Book` are generic and get built by whichever topic first needs
them, and C3 needs them.

## 10. Expected drawdown
Topic 1 parks **~35+**:
- **quarry L0 (8):** `discount_curve` · `pricing_context` · `forward_interpolation` · `greeks` ·
  `market_data/_types` · **`trade` · `book` · `daily_pnl`** (moved in — see §9)
- **`curves/` (~18 of 31):** bootstrap · curve_builder · global_solver · multicurve_solver ·
  ncurve_solver · rfr_bootstrap · curve_engine · curve_advanced · nelson_siegel · smith_wilson ·
  curve_blending · seasonal_curve · bond_curve · synthetic_market_data · curve_risk · key_rate_risk ·
  curve_bumper · curve_scenarios
- **`fixed_income` vanilla spine**, **including the 7 ticked in CP-3 but never physically parked**
  (`deposit` · `fra` · `ois` · `zero_coupon_bond` · `bond` · `fixed_leg` · `swap`), plus
  `floating_leg` · `basis_swap` · `csa` · futures

**13 → ~50.** One parking event, at topic close.
