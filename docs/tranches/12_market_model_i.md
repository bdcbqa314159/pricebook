# Tranche 12 — Market Model I (ch 3), the second model on the same instruments

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.14.0`.
**Source:** `BERHAN-2012` ch 3 (`worked-from`) — **verified it publishes the formulas in full**:
forward-rate dynamics (3.1.2–3.1.4), forward-index dynamics (3.1.5–3.1.6), the bivariate-lognormal
pair with explicit `μ`/`σ`/`ρ`, and Mercurio's freezing procedure. McInerney ch 5–6 is LMM
background, not required for the YoY/cap formulas. **Chosen by Bernardo**: finish the two market
models, MM-I first.

## 0. The crux — a SECOND model prices the SAME products, and the registry is product-keyed

The YoY swap and the cap/floor already have L4 pricers, and those take a **concrete
`JarrowYildirim`** today (`model.convexity(...)`). MM-I prices the *same* products with a *different*
formula (a freezing-procedure bivariate-lognormal expectation, not JY's `e^C`). The registry
dispatches on **product type**, so there is one pricer per product — and a second model must not be
served by an `isinstance`-on-model branch inside it (§1's no-ladder law).

**Ruled: introduce a capability protocol — rule-of-two fires here (JY + MM-I are its two
consumers), exactly as Black and the HW core were extracted at their second consumer.** The YoY and
cap/floor pricers depend on a **protocol** a model implements, and both JY and MM-I implement it;
the pricer composes the model's capability structurally, never switches on its type. This is the
doc-22 structural-dispatch design, earning its place now rather than speculatively in T4.

**The open question I need Code to settle by building, not by my assertion:** can both models'
YoY-period value be written behind **one** protocol method (e.g. `yoy_period_value(period, market)`
or a `yoy_convexity`-shaped call), or do their formulas differ in shape so much that the honest
seam is a **two-key `(product, model)` registry** instead? Build the protocol first; **if MM-I's
formula will not fit the JY pricer's shape without contortion, that is an `Ask:` back to me — a
forced-fit is worse than a second dispatch key, and the charter would rather gain a ruling than hide
a bent abstraction.**

## 1. Pieces and layers (subject to the §0 question)

```
L3  models/market_model_i.py — MM-I: the forward-rate/index vols + correlations, and the
                               freezing-procedure YoY and cap/floor adjustments (BERHAN 3.1–3.2)
L3  a capability protocol the YoY/cap pricers depend on; JY and MM-I both implement it
L4  engine/yoy.py, engine/capfloor.py — REFACTORED to the protocol (pure refactor, JY's numbers
                               bit-identical, guarded by T4/T5 oracles), then MM-I flows through
```

## 2. Split (§6) — build the seam first
- **(a)** the capability protocol + refactor the JY pricers onto it, JY numbers **bit-identical**
  (pure refactor, T4/T5 oracles green, no new RED). This proves the seam on the model we trust.
- **(b)** MM-I itself — the model + its freezing-procedure formulas — flowing through the seam, with
  its own oracle.
If (a)'s refactor can be shown bit-identical in one pass it may ride with (b); keep them separable.

## 3. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **cross-MODEL reduction — the strongest anchor here:** at zero vol every model collapses to the **same deterministic forward**, so MM-I's YoY and cap/floor at zero vol equal JY's at zero vol equal the model-free forward value | **exact, cross-model** | a wrong MM-I formula diverges from a limit three independent code paths now share |
| 2 | **companion:** MM-I's convexity/adjustment MOVES the price away from that limit as its vols rise (non-vacuous) | structural | a no-op MM-I would pass oracle 1 alone |
| 3 | **independent transcription of BERHAN 3.1.x** (the bivariate `μ`/`σ`/`ρ` and the freezing form), `-raw`+`-layout`, parameter-swept — after §5.1 assume the source can misprint, so a requirement guards it | **1** | every sign and factor in the freezing formula |
| 4 | **put-call parity: MM-I cap − floor = the MM-I YoY swap** (same model, different code each side) | structural | the sign/leg error |
| 5 | **the freezing procedure is an APPROXIMATION — bound it, don't hide it:** assert MM-I agrees with a Monte-Carlo (or the exact bivariate integral) of 3.1.5–3.1.6 to the approximation's stated tolerance, not to 1e-12 | 2 — trusted/convergence | a freezing error masquerading as "close enough"; the tolerance is the honest one, named |
| 6 | failure-is-a-value / raises per convention | structural | the contracts carried |

Oracle 1 is the cross-model anchor; oracle 5 is the one genuinely new shape — MM-I is *approximate*
by construction (freezing), so its oracle is a bounded agreement with the un-frozen quantity, stated
as a tolerance, never dressed as exactness. No number fitted to a figure (ch 3 has none anyway).

## 4. Design decisions (ruled)
- **Capability protocol, not a type switch** (§0) — subject to Code's build confirming one method
  fits both; a two-key registry is the fallback and a ruling, not a workaround.
- **The JY-pricer refactor is pure** — JY's prices bit-identical, measured as the Black and HW
  extractions were.
- **MM-I is a market model: its observables are the forward rates/index, lognormal** — a different
  model from JY, so (shared-shape-not-word) it does not subclass JY; the shared thing is the
  *protocol*, not the model.
- **Cross-check dispositions:** `models/market_model_i.py` → `crosscheck: quantlib (inflation market
  model / LMM)`; the protocol module, if separate, → `crosscheck: none — a protocol, no numerics`.

## 5. Stock
Mine `_stock/.../options/inflation_*` for the freezing/market-model shape; read the formulas from
**BERHAN ch 3** (the held source, which publishes them), not the stock's code.

## 6. Process
Branch `tranche/12a-model-protocol` then `tranche/12b-market-model-i`. `test` (RED pre-registered) →
`feat` → `docs` → `chore(release)`. Mutation on MM-I's freezing formula; the zero-vol cross-model
reduction and the MC-bounded approximation are the anchors. `source:` cites `BERHAN-2012` ch 3.
