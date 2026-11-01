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

**Fixed (not fitted):** `κn`, `σn`, `κr`, and the three correlations. Free: `(σr, σI)`.

**Corrected 2026-11-01 — this is a NARROWER fit than the thesis performs, and the first version of
this section misdescribed why.** It claimed the mean reversions and correlations come from
historical estimation. They do not. Read in both `-raw` and `-layout`:

- **§5.3.1 fits `κn` *and* `σn`** to ATM *nominal* cap volatilities, by least squares over
  `Cap_HullWhite(t,Ti,ψi,K_ATM,σn,κn) − Cap_Black(t,Ti,ψi,K_ATM,σ_ATM)`. Both are a fit, not an
  estimate — and it needs a Hull-White nominal cap engine, which this library has no reason to own
  yet. So both stay **inputs** here.
- **§5.3.2 fits SIX parameters** — `{κr, σr, σI, ρnr, ρIr, ρIn}` — to the YoY quotes: "all these
  parameters enter the YoY inflation Cap/Floor valuation formula, so that we may attempt to
  calibrate to market prices."

We fit **two of those six**, holding `κr` and the three correlations fixed. That is a deliberate
subset, now stated as one: the thesis's own result is that the six-parameter fit **does not work** —
"we know that we cannot fit to the whole surface", removing the two OTM strikes "still results in
high relative errors", excluding the long end "still results in a poor fit", and "the presence of a
strike skew makes calibration unfeasible for non ATM contracts". Its conclusion is to restrict to
near-ATM contracts and a truncated expiry range. A2b is precisely the diagnostic for why: six
parameters against a thin, nearly-collinear strip is an ill-conditioned fit, and the thesis found
that empirically without naming it.

**So A1 proves our fit recovers parameters from a strip *our own model generated* — a round trip. It
is not evidence that JY fits market quotes, and the thesis says plainly that it does not.**

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

**A2 split into two faces after Cowork's review pass (thread 0035), and the split was right.** My
A2 perturbed each free parameter individually and checked a quote moved. That proves the objective
is not **flat** — the stock's failure — but for a *two*-parameter fit it cannot prove the optimum is
**unique**: `σr` and `σI` can each move the strip a lot while moving it in nearly collinear
directions, and then per-parameter sensitivity passes while the pair is unidentified, with A1
landing near truth only by luck of its start. Flat and degenerate are different failures.

**Demonstrated, not argued.** Collapse the strip to a single quote (two parameters, one equation)
and:

| | result |
|---|---|
| A2a `σr` +10% | moves the quote **8296** — 83× its gate. **Passes.** |
| A2a `σI` +10% | moves the quote **1198** — 12× its gate. **Passes.** |
| the fit, from `(0.020, 0.004)` | `rms = 8.7e-11` — a *perfect* fit — at `σI = 0.0039` |
| truth | `σI = 0.0120` |

A1 and A2a together would have reported success with `σI` **wrong by a factor of three**, and three
different starts give three different answers. A2b's conditioning number is `inf` there and rejects
it; on the real six-quote strip it is `6.7e2`, comfortably inside the `1e4` gate. So the gate bites
on the degenerate case and does not bind on the good one — which is what a threshold has to show.

**A2 is load-bearing because of §0.** A parameter-recovery
round-trip passes *trivially* on a flat objective — and a flat objective is exactly what the stock
built. A1 without A2 is a check that cannot fail. This is the floor-on-variance rule applied where
it was earned.

**A4's premise check had the same disease, and it bit during the build.** The test asserts the
rebuilt real curve equals the direct one *before* comparing fits — otherwise it compares two
calibrations of two different markets. The first version checked that only at the quote maturities,
which is exactly where the two agree by construction. It passed while the curves differed **below
2y**, where `real_curve_from_breakeven` can only flat-extrapolate under its shortest quote; the fit
then missed `σI` by 0.14%, which is how the hole surfaced. Fixed on both sides: the premise is now
checked on a 30-day grid across the whole horizon the floors touch, and the fixture's real curve is
flat on `[0, 1y]` so the inversion can reproduce it exactly. **A premise check that only looks where
agreement is guaranteed is a check that cannot fail** — the tranche's own subject, found in its own
test.

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

### 4.5 Layer: the calibrator is **L4**, with the pricers — not L3 with the models
Not covered by the review pass, and the gate forced the question: `verify.py` ranks `calibration`
at L3 and `engine` at L4, so a calibrator importing the floor pricer would be an upward import and
`acyclic` would reject it. The resolution is not to loosen the gate but to notice the spine means
two different things by "calibration": the **model-free** kind (a curve bootstrap, T2's break-even
inversion) is L3 and imports nothing above it, while a calibrator that *inverts an engine* is a
consumer of `price()` and must sit at or above it. This one lives in `engine/jy_calibration.py`.
No gate change, no charter change — and it calls `price_inflation_capfloor` directly rather than
the registry's `dispatch`, since dispatching on a type it already knows is indirection with no
consumer.

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
