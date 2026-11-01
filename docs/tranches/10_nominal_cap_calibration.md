# Tranche 10 — the Hull-White nominal cap engine and the (κn, σn) fit (§5.3.1)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.11.0`.
**Source:** `BERHAN-2012` §5.3.1 (`worked-from`) + standard Hull-White / Jamshidian caplet.
**Why now:** the finish-the-book ladder's next rung, and it **closes the two-of-six gap** — σn has
been a hand-supplied input since T6; §5.3.1 is where the thesis sources it, by fitting `(κn, σn)` to
ATM nominal cap vols. It is the first **non-inflation** instrument (a vanilla rates cap), flagged
plainly, but it is the thesis's own calibration step, not a detour.

## 0. Split — two rungs, build (a) first

- **(a) the Hull-White nominal cap ENGINE** — a priceable rates cap under HW, closed form.
- **(b) the CALIBRATION** `(κn, σn)` to ATM nominal cap vols — inverts (a).

They land as separate PRs; (b) composes (a). If (a)'s own oracle will not check in one pass, it may
split again, but I expect it whole.

## 1. Pieces and layers

```
L3  models/hull_white.py  — EXTRACT HullWhiteParams + hw_beta from JY (§3d, rule-of-two: JY's
                            nominal factor + this cap are the two consumers), ADD the HW caplet
                            closed form (a Jamshidian option on a zero-coupon bond)
L2  products/nominal_cap.py — NominalCap: schedule + strike + notional + kind (CAP/FLOOR), frozen
L4  engine/nominal_cap.py   — price_nominal_cap: a strip of HW caplets off the nominal curve        (a)
L4  engine/hw_cap_calibration.py — fit (κn, σn) to ATM cap vols, via the solver seam                (b)
```

**The HW caplet is L3** (a model analytic — a bond-option closed form, finance, like Black and the
JY convexity). **The cap pricer is L4** (an engine: `price(NominalCap, HullWhiteModel)`). **The
calibration is L4** — it **inverts the cap engine** (reprices to the ATM market, solving `(κn,σn)`),
so by §2 it is engine-inverting and lives in `engine/`, like `jy_calibration`.

`HullWhiteParams` and `hw_beta` already exist inside `models/jarrow_yildirim.py` (T4). T10 is their
**second** consumer, so they move to `models/hull_white.py` and JY composes them — a pure refactor
guarded by T4's green oracles, numbers unmoved. The nominal HW model is JY's nominal factor
standing alone.

## 2. Oracle — part (a), the cap engine

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **the HW caplet closed form vs an independent transcription** of the Jamshidian zero-bond-option formula, read from the source in `-raw`+`-layout`, parameter-swept | **1** | every `κ`, factor-of-2 and `B(S,T)` error lives here; the independent transcription must not share the implementation's rendering (§4) |
| 2 | **anchor reduction: `σn → 0` ⇒ the caplet is its intrinsic value** on the deterministic forward, exactly — with the mandatory companion that **σn MOVES the price** (not a no-op) | exact + companion | `σn` is the vol; zero vol is the degenerate case with a known answer, and the companion stops a reduction that an empty implementation also passes |
| 3 | **put-call parity: cap − floor = the forward-rate swap value** (the fixed-vs-float identity), composed from the same caplet atom, different code each side | structural cross-product | the sign/leg error no magnitude check catches |
| 4 | **a requirement that survives a source misprint:** the caplet is positive, monotone in `σn`, and → 0 as the strike → ∞; and the HW caplet-vol term structure has the right `κn` sign (vol of the `T`-caplet rises then saturates in maturity) | structural | the §4 "formula must satisfy a property" guard — a transcription or source typo that oracle 1 might share |
| 5 | failure-is-a-value / raises per convention (an unpriceable cap, a bad schedule) | structural | the engine returns `PricingFailure`; a malformed product RAISES at construction (T8 convention) |

## 3. Oracle — part (b), the calibration

| # | oracle | rung | why it can go red |
|---|---|---|---|
| B1 | **parameter recovery:** generate ATM cap vols from a known `(κn*, σn*)` through (a), fit back, recover both to tolerance | 2 — round trip | the inverse of (a) |
| B2 | **identifiability (B1's non-vacuity companion, the T6 pattern):** `(κn, σn)` are a 2-parameter fit, so **multi-start recovery to one point** AND/OR a bounded Jacobian condition number — not single-parameter sensitivity | 1 — requirement | `κn` shapes the caplet-vol *term structure* while `σn` sets its *level*, so they SHOULD separate across a maturity strip — but that must be proven, not assumed (the T6 identifiability valley) |
| B3 | a deliberately-wrong start strictly improves; residual at truth ≈ 0 | 1 | the objective is not flat |
| B4 | failure-is-a-value: a non-converging fit returns `PricingFailure` | structural | the solver seam's `converged` flag (T6) |

The thesis reports the ATM nominal-cap fit is reasonable ("not too bad in the long end", §5.3.1) —
unlike the 6-parameter YoY fit it feeds. That is context, not an oracle; B1–B2 are self-contained.

## 4. Design decisions (ruled)

- **HW core extracted to L3, shared — rule-of-two fires (JY + this).** `models/hull_white.py` holds
  `HullWhiteParams`, `hw_beta`, and the caplet; JY and the nominal cap both compose it, so a HW
  convention cannot diverge between the inflation and rates uses. JY's refactor is pure, T4's
  oracles re-run green, numbers unmoved.
- **The NominalCap is a real L2 product priced by an L4 engine**, not a calibration-internal formula
  — a nominal cap is a first-class instrument (and §5.3.1's forward model), priced through
  `price()` like every other product.
- **This closes T6's σn gap but does NOT re-open §5.3.2.** The full six-parameter YoY fit stays as
  T6 left it; wiring σn from here into a §5.3.2 re-run is a *later* unit, not T10.
- **Cross-check dispositions (§8 gate) for the new modules:**
  - `models/hull_white.py` → `crosscheck: quantlib HullWhite / HullWhiteModel caplet`
  - `engine/nominal_cap.py` → `crosscheck: quantlib BlackCapFloorEngine / AnalyticCapFloorEngine`
  - `engine/hw_cap_calibration.py` → `crosscheck: quantlib HullWhite calibration (CapHelper)`
  - `products/nominal_cap.py` → `crosscheck: none — a frozen cap description, no numerics`

## 5. Stock
**Mine for content:** `_stock/.../jarrow_yildirim.py` already has the HW `β` and parameter bundle;
the HW caplet formula read from the **source** (standard HW/Jamshidian, §5.3.1), not from a stock
rates module, which may use a different parameterisation. Build the caplet fresh on the extracted
`models/hull_white.py`.

## 6. Process
Branch `tranche/10a-hw-nominal-cap` then `tranche/10b-hw-cap-calibration`. `test` (RED) → `feat` →
`docs` → `chore(release)`. `source:` cites `BERHAN-2012` §5.3.1 + the standard HW caplet. The HW
extraction's provenance notes it supersedes JY's inline `HullWhiteParams`/`hw_beta`, with T4's
oracles re-run green to prove the refactor moved nothing.
