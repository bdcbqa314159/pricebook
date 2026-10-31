# Tranche 6 — calibrating Jarrow–Yildirim to YoY floor quotes

**Status:** proposed, awaiting Cowork's review pass.
**Source:** BERHAN-2012 ch 5 (`worked-from`). Thesis endpoint: JY fitted to YoY floor quotes, ATM only.

---

## 0. The headline, before the design: the stock's calibration must NOT be followed

`_stock/quarry/pricebook/fixed_income/jarrow_yildirim.py::jy_calibrate` fits `(σn, σr, σI)` to a
**ZCIIS** term structure. Under correct JY a ZCIIS carries **no volatility information at all** —
its fair rate is `Pr(0,T)/Pn(0,T) − 1` (eq 1.2.13/1.2.16), a pure curve relationship. It is
*because* the ZCIIS is model-free that T2 used it to **build** the real curve.

So the stock is fitting volatilities to quotes that cannot see volatility. Its objective is non-flat
only through two defects:

1. it builds `P_n`, `P_r` from a **Vasicek closed form with `θ = r₀`** — a *model-implied flat*
   curve — instead of the market curves, which lets `σn`, `σr` leak in through a bond convexity that
   the market curve would have absorbed; and
2. it applies `conv_adj = −ρ_nI·σn·σI·B_n·T` **to the ZCIIS**, where JY puts the convexity on the
   **YoY** swap (eq 2.5.11 — T4's formula), not on the zero-coupon swap.

The spurious sensitivity is **real but small, and measured rather than guessed**: at a 5y ZCIIS,
`an=ar=0.05`, `rn=4%`, `rr=1%`, `ρnI=−0.2`, doubling `σI` (2%→4%) moves the stock's fair rate
**+10.3 bp**, and doubling `σr` (0.5%→1%) moves it **+15.1 bp**. That is enough to drive a
least-squares fit, since the quotes are quoted to the basis point — and it is entirely artefact. A
fit that converges on 10 bp of bug is worse than one that fails to converge, because it reports
success.

**This is the stock-is-not-authority rule
landing for the third time**, and the first time it has hit a *calibration* rather than a formula:
T4's wrong `ρ → 0` anchor came from the stock, T5 found the stock's first-order convexity is a
different object from 2.5.11, and now the stock's calibration target is the wrong instrument.

T6 calibrates to the **vol-dependent** instruments — the YoY cap/floor strip T5 built. That is also
what the thesis does.

---

## 1. Shape: two stages, and only the second is new

```
curve stage (DONE, T2)   nominal curve + real curve from ZCIIS breakevens   -> model-free
vol   stage (T6)         (σr, σI) fitted to a YoY floor strip               -> this tranche
```

The curve stage is already pinned and must not be re-fitted here — refitting curves against vol
quotes is how a calibration launders a curve error into a vol.

**Fixed (not fitted):** `κn`, `κr`, the three correlations, and `σn`. The thesis fixes mean
reversions and correlations from historical estimation and takes `σn` from **nominal** ATM caps —
an instrument this library has no engine for, so `σn` is an **input**, not a fit. Stated plainly
rather than hidden in a default.

**Free:** `(σr, σI)`. Both enter `V²` (2.6.4) and `σr` additionally scales all of `C` (2.5.11), so
they move the strip differently — which oracle A2 **proves** rather than assumes.

**Quotes:** a strip of ≥4 ATM YoY floors. Two free parameters, overdetermined, least squares.

---

## 2. The oracle

| id | criterion | oracle | rung |
|---|---|---|---|
| **A1** | parameter recovery | generate the floor strip from known `(σr*, σI*)` through T5's engine, calibrate back, recover both to 1e-6 | 2 — synthetic round-trip |
| **A2** | **the strip is discriminating** (A1's non-vacuity companion) | perturb each free parameter by 10%: at least one quote moves by > 100x the fit tolerance; and the strip's quotes are >50% distinct | 1 — requirement |
| **A3** | residual at the truth | objective at `(σr*, σI*)` is < 1e-18, and the fit strictly decreases it from a deliberately-wrong start | 1 — requirement |
| **A4** | **cross-stage**: the vol fit does not move the curve stage | the calibrated model reprices the ZCIIS strip that *built* the real curve back to its input breakevens, 1e-12 | 1 — cross-product |
| **A5** | failure is a value | a non-converging fit returns `PricingFailure`, never NaN and never raises | 1 — requirement |

**A2 is the load-bearing one and it is in the table because of §0.** A parameter-recovery
round-trip passes *trivially* on a flat objective — and a flat objective is exactly what the stock
built. A1 without A2 is a check that cannot fail. This is the floor-on-variance rule applied where
it was earned.

**A4 is a genuine cross-product oracle, and I rejected the weaker version of it.** The tempting
oracle is "the calibrated vols leave the ZCIIS fair rate unchanged" — but `price_zc_inflation_swap`
takes a `DiscountingModel` and *structurally cannot see vols*, so that check cannot fail and is
worth nothing. The real content is the **round trip through both stages**: breakevens → real curve →
vol fit → reprice the breakevens. (The YoY leg reduces to the ZCIIS when the first fixing is
known, which is the bridge.)

---

## 3. Pieces

- `foundation/solvers.py` — a thin `scipy.optimize.least_squares` adapter. **Flagged for Cowork,
  §4.1:** one consumer today, so the rule of two says write it concrete.
- `models/jy_calibration.py` — `calibrate_jy_to_yoy_floors(...) -> JYCalibration | PricingFailure`;
  `JYCalibration` carries ≤5 fields (`params`, `rms_residual`, `n_quotes`, `iterations`) and holds
  the fitted `JYParams`, so the output of calibration *is* a usable model input.

No new products, no new engines: T6 consumes T5's `price_inflation_capfloor` as its forward model.

---

## 4. Decisions, flagged for the review pass

### 4.1 `foundation/solvers.py` with one consumer — rule of two vs. the adapter policy
`distributions.py` set the precedent: a thin scipy adapter created at its *first* consumer, justified
as "one place to pin behaviour, one swap point for a port". A solver adapter is the same shape. But
the rule of two is explicit that one caller means write it concrete, and a solver has a wider surface
than `norm_cdf` (method, bounds, tolerances, the result object) — more room for a speculative
signature. **My call: build it, minimal — one function, takes a residual callable and a start, returns
`(x, cost, iterations, converged)`.** Rule of two fires on a shared *shape*, and "wrap a scipy
numerical primitive behind our own API" is a shape with two instances. Cowork to confirm or overrule.

### 4.2 `σn` as an input, not a fit
Fitting `σn` to nominal caps needs a nominal cap engine (Black on a nominal caplet) that does not
exist here. Building one to serve a calibration would be infrastructure ahead of its product. `σn`
is a named argument with no default.

### 4.3 ATM only, no smile
The thesis's own finding is that JY calibrates to ATM and does not produce a smile. T6 reproduces
that limit rather than papering over it; the relative-error surfaces (Fig 5.x) are plots and are not
an oracle.

### 4.4 Not building
Market Model II (ch 5's second model, its own tranche); the cap/floor *surface*; `σn` from nominal
caps (4.2); any reproduction of Figures 5.1–5.10.

---

## 5. Process

The **cluster checkpoint fires at the close of this tranche** — five tranches of inflation plus a
model and its calibration is a capability cluster. Its main input is the v0.4.0 wrong-formula
incident and §0 of this document: three of the four defects this vertical has produced came from
treating the stock as authority, and the fourth (the merged-paren transcription) came from treating
one rendering of the source as authority.
