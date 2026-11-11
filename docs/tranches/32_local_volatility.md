# Tranche 32 — the implied-vol surface and Dupire local volatility

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.38.0`.
**Source:** `BROCKHAUS-2016` ch 4 (implied volatility — skew, tails, term structure) + ch 6.1 (local
volatility / Dupire). **Pre-registered RED.** Third unit of the equity & hybrids arc; follows T31. The
**modelling foundation** — everything equity-exotic downstream pulls a vol model.

## 0. The product worth having
**The local-volatility model, calibrated to the implied-vol surface by Dupire's formula** — the
unique one-factor diffusion `dS = (r−q)S dt + σ_loc(S,t) S dW` that **reprices the entire vanilla
surface by construction**. The surface is the market object (ch 4); Dupire (ch 6.1) inverts it to the
local vol:

> `σ_loc²(K,T) = [ ∂C/∂T + (r−q)·K·∂C/∂K + q·C ] / [ ½·K²·∂²C/∂K² ]`

(call-price form, with repo/dividend yield `q`). This is the first equity *model*, and the base the
stochastic-vol (T33) and hybrid (T34–T36) tranches generalise.

## 1. The pieces and their layers
- **L1 `market/` — the equity implied-vol surface** (ch 4): a `(K, T) → σ_impl` parametrization —
  skew (4.1), tail behaviour (4.2), term structure (4.3). **Check whether T9's inflation vol surface
  generalises** (read it before building a second — §5/§10 read-before-asserting); if the shape is
  shared it is a rule-of-two consolidation, if not a separate equity surface. A `call_price(K,T)` view
  (Black on the dividend forward, T30) feeds Dupire.
- **L3 `models/` — the local-volatility model:** `σ_loc(S,t)` from the surface via Dupire (ch 6.1),
  carrying the `MarketSnapshot`/surface it was calibrated to. The Dupire inversion is a **model-free
  parameter calibration** reading no engine → L3 (§2). Numerical config (the finite-difference grid for
  the surface derivatives) arrives with it (§3.5) — the T31 `(market, grid)` shape.
- **L4 `engine/` — the local-vol Monte-Carlo pricer:** the Jäckel path engine (T15–T21) extended to a
  **state-dependent** vol `σ_loc(S,t)` in the Euler step — reusing the RNG, the bridge, antithetic. The
  independent repricer the consistency anchor needs. Carries its SE (invariant 7).

## 2. The oracle (RULED) — the consistency anchor must not telescope
"Local vol reprices the vanillas it was calibrated to" is the defining property, but a naïve round-trip
(surface → Dupire → reprice → surface) **telescopes** if the repricer and the Dupire derivatives share
a convention (§4). The anchor is a **synthetic truth**, so both directions are independent:

- **The synthetic-truth round-trip — the correctness anchor (§4, reduction/known-truth).** Pick a
  **known analytic** local vol `σ_loc(S,t)` (a simple deterministic form — e.g. `σ₀·(1 + β·log(S/S₀))`
  floored positive); price a grid of vanillas under it with the **local-vol MC** (forward map);
  implied-vol them into a synthetic surface; apply **Dupire** (backward map, different code) and assert
  it **recovers the known `σ_loc(K,T)`** within a band (the MC SE + the FD-derivative error, measured —
  the T17 split, not one flat percentage). Forward MC vs backward formula are independent, so a Dupire
  transcription error cannot survive, and a no-op `σ_loc = σ_impl` fails on the skewed truth.
- **Dupire pointwise transcription (§4), two renderings.** The formula read in `-raw` and `-layout` and
  reconciled; the `∂C/∂T`, `∂C/∂K`, `∂²C/∂K²` finite differences the test computes **its own** way (its
  own grid), never reading back the model's.
- **The flat reduction (exact), with the mandatory skew companion.** A **flat** implied surface ⇒
  `σ_loc ≡ σ` everywhere (local vol = implied vol with no skew/term structure) — exact, but a no-op
  satisfies it, so it is vacuous alone. The companion is a **skewed, term-structured** surface where
  `σ_loc` is visibly *steeper* than the implied skew (Dupire's "rule of two" — local skew ≈ twice
  implied skew near the money); assert that non-degeneracy, or the whole formula is untested.
- **Repricing consistency (self-consistency guard, telescopes — keep it, not the anchor).** The
  local-vol MC on the *real* input surface reprices its own vanillas within SE. Useful as a guard; it
  shares the surface with the construction, so it is not the correctness check — the synthetic truth is.
- **Requirement oracles + mutation (§4).** `σ_loc² ≥ 0` (a surface violating `∂²C/∂K² > 0` — a
  butterfly arbitrage — must `PricingFailure`, not return a negative variance, §3.4); monotonicities;
  the local-vol MC's martingale check (`E[S_T]/S₀ = F(T)/S₀`, the T19/T26 free identity) pins the
  simulated drift independently of the vol. **Mutation:** drop the `(r−q)K ∂C/∂K` drift term; drop the
  `q·C`; use `K·∂²C/∂K²` instead of `K²`; the `½`; sign-flip `∂C/∂T`. Each must die on the synthetic
  round-trip.

## 3. Process
Branch `tranche/32-local-volatility`. `test` (RED: the synthetic-truth round-trip + the Dupire
transcription + the flat/skew pair + the mutation table) → `feat` → `docs` → `chore(release): v0.38.0`.
**Open the draft PR early.** `source:` `BROCKHAUS-2016`. The surface/model/engine modules carry
`crosscheck:` markers (QuantLib has `LocalVolSurface`/`Dupire` — a real target). **Re-time `slow`** —
a local-vol MC grid is a new cost (the T19 lesson). **Raise an `Ask:`** if: T9's surface does not
generalise (then a second surface type is a design call, mine); or the FD-derivative noise on
`∂²C/∂K²` makes the synthetic band too loose to resolve the mutations (then the grid/measurement needs
a ruling, not a widened band).

## 4. What this tranche is NOT (scope fence)
No **stochastic volatility** (Heston) and no **local-stochastic vol** — T33 / `to_go_deeper` (ch 6.2–
6.3). No **finite-difference PDE** engine — the repricer is MC, reusing the Jäckel engine; a Dupire
forward-PDE is a `to_go_deeper` route (a different numerical method, pulled only if MC is too slow for
a product). No **jump diffusion** (ch 6.4), no **implied-vol dynamics** (ch 7), no **arbitrage-free
surface calibration/smoothing** beyond rejecting a butterfly-arbitrage surface with `PricingFailure`
(a full arb-free smoother is `to_go_deeper`). The point is the local-vol *model and its Dupire
calibration*, oracled by the round-trip — the modelling base the arc stands on.
