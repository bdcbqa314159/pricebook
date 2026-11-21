# T69 — the Gaussian HJM framework

**Cowork spec, 2026-11-21.** Fourth tranche of the single-curve short-rate arc (project nine,
`MCINERNEY-ZASTAWNIAK`; ch 4.1–4.3, 4.5). The framework the short-rate models are special cases of: HJM models
the **instantaneous forward rate** `f(t,T)` directly, and its no-arbitrage **drift condition** is the one
equation the whole chapter turns on. The oracle spine is unusually clean here — the drift condition is a
*requirement* oracle (forced by no-arbitrage, not a source value), and **Gaussian HJM reduces to Hull-White
exactly**, so the tree's own fifty-tranche-old HW machinery is the reduction's cross-check (the T67 beat,
again).

## Target
Price a zero-coupon bond and a caplet under one-factor **Gaussian** HJM (deterministic forward-rate
volatility) — `price(product, GaussianHjmModel) → PricingResult` — with `df(t,T) = α(t,T)dt + σ(t,T)dW`.

## Pieces, layer, reuse
- **The model** (new, L3) — `GaussianHjmModel` carrying the `MarketSnapshot` (§3) and the **forward-rate
  volatility function** `σ(t,T)` (deterministic). The initial forward curve `f(0,T)` comes from the
  snapshot's `DiscountCurve` (reuse — `f(0,T) = −∂lnP(0,T)/∂T`). The vol `σ(t,T)` is the model's one
  genuinely-new parameter; represent it so the **Hull-White case `σ(t,T) = σ·e^{−κ(T−t)}`** is expressible
  (that is the reduction anchor). Over the field limit → bundle (§7).
- **The bond / caplet** (new, L3/L4) — Gaussian forwards ⇒ the bond `P(t,T)` is lognormal, so the caplet is
  Black on the forward bond with the HJM bond variance; **reuse `zero_bond_option`** where the variance
  structure matches, else the new HJM variance (the `∫σ` integrals).
- **The drift** — `α(t,T) = σ(t,T)·∫_t^T σ(t,u)du` (the HJM no-arbitrage condition). This is the framework's
  defining equation and it is a **requirement** oracle, not a transcription target alone (below).

## The mathematics (re-transcribe from the source — NOT this spec, §5)
Read **McInerney–Zastawniak ch 4.1 (one-factor HJM), 4.2 (Gaussian models), 4.5 (forward rate under the
forward measure)** in `-raw` and `-layout` and reconcile (§4). The source may print the forward-rate **SDE**
and leave the integrated bond variance to be derived (the T68 hazard — a closed form from memory is exactly
§5's trap); if so, **pin the integrated form by an identity it must satisfy** (the bond reprices, the drift
condition holds) rather than transcribing a form the book does not print. The spec names the obligation, not
the form (§4).

## Oracles (RED-first, §4)
1. **The HJM drift condition as a REQUIREMENT oracle — the framework's spine.** `α(t,T) = σ(t,T)·∫_t^T σ du`
   is forced by no-arbitrage (the discounted bond is a martingale), so it holds for **all** `σ(t,T)` and
   survives a transcription error a source-value oracle would share (§4 requirement-oracle rule). Assert it
   directly: build the model, then check that the discounted bond `P(t,T)/B(t)` is driftless under the
   risk-neutral measure to numerical zero. **And** transcribe the drift condition pointwise at least once
   (the T12(b) rule — a requirement oracle never replaces a pointwise transcription).
2. **Reduction: `σ(t,T) = σ·e^{−κ(T−t)}` ⇒ Hull-White, exactly — the arc's self-verifying anchor.** With
   this vol the HJM bond and caplet **are** the tree's `HullWhiteModel` output (eq-for-eq). Assert agreement
   against the held HW machinery (`hw_beta`, `zero_bond_option`) — a **cross-check against already-oracled
   code** (the T67 form, the strongest anchor here: not a transcription, the tree verifying itself), flagged
   as a consistency guard, not cross-source independence (§4 same-object caveat). **Companion required
   (§4/T39):** a **non-Hull-White** `σ(t,T)` (e.g. a humped or two-factor-style vol) exercising the general
   HJM — the HW reduction is blind to everything the specific vol structure degenerates.
3. **Forward-measure martingale (self-consistency).** Under the `T`-forward measure the forward bond price
   `P(t,S)/P(t,T)` is a martingale (ch 4.5) — a self-consistency guard (telescopes, so it is a guard not the
   correctness anchor, §4; the correctness anchors are 1 and 2).
4. **The MC companion** simulates the forward curve (Musiela parameterisation or the direct `f(t,T)` SDE),
   banded per §4 (`k·SE`, resolution stated); the closed-form bond inside the band. Pin the simulated drift
   against the drift condition first (the unbiased-before-credited discipline).
5. **Watch for `α→0`-type limits in the bond variance** (the `∫σ` integrals carry `(1−e^{−κτ})/κ` forms for
   the HW vol) — if a reduction hits a removable singularity, it is **rate-in-a-regime, not banded**, with
   the regime **measured for this model** (the T66/T67 clause — this is a named recurrence).
6. **Requirement oracles:** the bond boundary `P(T,T)=1`; `f(0,T)` reproduces the initial `DiscountCurve`
   (the model is calibrated to today's curve by construction — a cross-product check against `market/curve`).

## Degeneracy premises (§4 / §9)
- The HW-reduction companion is **non-degenerate in the vol structure** (a genuinely non-HW `σ(t,T)`, T39).
- **Non-flat `σ(t,T)` and non-trivial maturities** so the `∫σ` integrals are not constant; prefer a
  **mutation to a reading** on the drift condition (drop the `∫`, swap `σ(t,T)` for `σ(t,u)` inside the
  integral, drop the outer `σ(t,T)` factor).
- The martingale check must use a **non-trivial** forward (a flat curve makes the forward-measure drift
  trivially zero).

## Provenance / gates
- Header `source: MCINERNEY-ZASTAWNIAK §4.1–4.3, §4.5`; `oracle:`; `crosscheck:` (QuantLib has a Gaussian/HJM
  short-rate engine — a real target, or the HW engine since they coincide on the reduction).
- Version → **v0.71.0**; one floor bump; one `CHANGELOG` entry. Re-time the curve-simulation MC on **both**
  CI runners (§9).

## Out of scope (carried / deferred → T70+)
- **Multi-factor HJM** (ch 4.4) — T70 (the next tranche; this one-factor Gaussian HJM is the base it
  generalises, and the one-factor reduction is T70's anchor).
- **Non-Gaussian / stochastic forward-rate vol** — `to_go_deeper` (the Gaussian/deterministic-σ case is the
  spine; a stochastic σ loses the closed form).
- **HJM calibration to a cap/swaption surface** (ch 4.3 in full) — the model carries `σ(t,T)`; fitting it to
  a surface is `to_go_deeper` (a real optimisation, the T11 identifiability discipline would apply). The
  reproduce-the-initial-curve requirement (oracle 6) is the calibration this tranche *does* carry.
