# Tranche 31 — the variance swap: a model-independent static replication

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.37.0`.
**Source:** `BROCKHAUS-2016` ch 3.7 (volatility products) + ch 5.8 (variance swap). **Pre-registered
RED.** Second unit of the equity & hybrids arc (`docs/equity_hybrids_arc.md`); follows T30.

## 0. The product worth having
**The variance swap, priced by its *exact, model-independent* static replication** — the most
beautiful oracle in equity. The fair future variance `K_var` is a strike-continuum of vanilla options
(the log-contract): it depends only on the *option prices*, not on the model that generated them. Two
equivalent forms Brockhaus gives (ch 5.8):

- **the strip:** `K_var = (2 e^{rT}/T) · [ ∫₀ᶠ P(K)/K² dK + ∫ᶠ^∞ C(K)/K² dK ]` — the `1/K²`-weighted
  portfolio of out-of-the-money puts and calls;
- **the log-contract:** `K_var = −(2/T) · E^Q[ log(S_T / F) ]` — the fair variance *is* the price of a
  `−2 log(S/F)` payoff (continuous monitoring, diffusion, no jumps).

## 1. The pieces and their layers
- **L1 `market/` — a minimal implied-vol skew** (ch 4.1 parametrization): a `strike → σ` function, the
  option-price source across strikes the replication integrates. **Pulled by the variance swap** — the
  *full* surface (term structure, calibration) and **Dupire local vol are T32**, not here; T31 needs
  only a skew to feed the strip. Keep it a small frozen parametrization.
- **L2 `products/` — the variance swap** (frozen data: the variance strike, the monitoring schedule,
  notional). Describes, does not price.
- **L4 `engine/` — the replication engine:** the `1/K²`-weighted strip integral over **T30's equity
  vanilla** (`black_price` on the forward at each strike), and/or the log-contract form. `@register`ed.
- No new model: the replication is model-free; the skew is just the price source.

## 2. The oracle (RULED)
- **The exact reduction — the anchor (§4, closed form).** Under **flat-vol** Black-Scholes the strip
  integrates to **exactly `σ²`** (`K_var = σ²`), because `E[log(S_T/F)] = −½σ²T`. Feed a flat-vol BS
  surface (our verified Black, T5/T30) into the replication and assert `K_var = σ²` to tolerance — a
  **cross-product oracle**, different code each side (the strip integrator vs Black). It fails under any
  error in the `1/K²` weight, the `2/T` factor, the OTM put/call split at `F`, or the forward.
- **The log-contract requirement identity (§4, survives a transcription error).** The two forms agree:
  the strip integral equals `−(2/T)·E[log(S_T/F)]`, computed independently (the test builds the
  log-payoff expectation under the same prices). A derived identity beside the transcription (§4's
  stronger rung) — and it is where the `−2` and the `2/T` are pinned.
- **Strike-grid convergence (banded resolution, §4 — the T25 lesson).** The continuum integral is a
  discrete strike strip in practice; as the grid **widens and refines**, the discrete `K_var → σ²` at a
  measurable rate. **Give the grid an explicit validity regime** — a truncated grid *underprices* the
  tails (the T25 "a reference is code too" trap: a too-narrow strike range returns < σ² and the
  reduction fails for the wrong reason). Assert the truncation + discretization error is below the band,
  and state the smallest defect the committed grid resolves.
- **The skew non-degenerate companion (mandatory — else the reduction is vacuous).** A **flat** vol
  surface makes `K_var = σ²` trivially, so the reduction a no-op also satisfies. The companion is a
  **skewed** surface (ch 4.1, a material skew): `K_var` then moves **away from the ATM σ²** (the
  convexity premium — `K_var > σ²_ATM` for a typical skew), which only a correctly `1/K²`-weighted strip
  produces. Assert the skew premium's sign and that it is material; this is the whole point of the
  variance swap and what the reduction alone cannot see.
- **Requirement oracles + mutation table (§4).** `K_var ≥ 0`; monotone increasing in the vol level.
  **Mutation:** drop the `1/K²` weight (→ fails the σ² reduction); drop the `2/T` or the `e^{rT}`; put
  calls below `F` / puts above `F` (wrong OTM split); flip the log-contract sign. Each must die. The
  degenerate-fixture premises to assert: **non-flat skew** (per above) and a **grid wide enough** that
  truncation is below the band.

## 3. Process
Branch `tranche/31-variance-swap`. `test` (RED: the flat-vol σ² reduction + the log-contract identity +
the skew companion + the mutation table) → `feat` → `docs` → `chore(release): v0.37.0`. **Open the
draft PR early.** `source:` `BROCKHAUS-2016`; transcribe the replication (ch 5.8) in **both renderings**
(the `2/T`, `1/K²`, `e^{rT}` factors are a reflow risk). The engine wants a `crosscheck:` (QuantLib has
a variance-swap/replication helper, or `none — <reason>`, §8). **Raise an `Ask:`** if Brockhaus's
convention (continuous vs discrete monitoring, the discrete-dividend correction to the log-contract)
needs a piece the spine lacks.

## 4. What this tranche is NOT (scope fence)
No **full implied-vol surface** (term structure, no-arbitrage calibration) and no **Dupire local vol** —
those are T32; T31 pulls only the minimal skew it needs to price the strip. No **jumps** (the
log-contract's model-independence breaks under jumps — ch 3.7's caveat; the jump correction is
`to_go_deeper`). No discrete-dividend variance-swap correction beyond what the forward already carries
(deferred). No volatility *swap* (vol, not variance — a separate convexity, `to_go_deeper`). No gamma/
vega swaps or corridor variance (the volatility-product zoo, `to_go_deeper`).
