# Tranche 12 — Market Model I (ch 3), the second model on the same instruments

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.14.0`.
**Source:** `BERHAN-2012` ch 3 (`worked-from`) — **verified it publishes the formulas in full**:
forward-rate dynamics (3.1.2–3.1.4), forward-index dynamics (3.1.5–3.1.6), the bivariate-lognormal
pair with explicit `μ`/`σ`/`ρ`, and Mercurio's freezing procedure. McInerney ch 5–6 is LMM
background, not required for the YoY/cap formulas. **Chosen by Bernardo**: finish the two market
models, MM-I first.

**Status:** **(a) BUILT** (v0.14.0) — the capability seam. **(b) pending** — MM-I itself.
**Answer to §0's open question: ONE protocol, two methods. No second dispatch key needed.** See §8.

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


---

## 8. Result — §0's question answered by building: ONE protocol, and the source supplies its shape

375 tests, eight gates green, ruff and pyright clean, zero suppressions. Six mutations on the seam,
all caught.

### The answer, and the evidence
**One protocol with two methods fits both models. No two-key `(product, model)` registry is needed,
and no formula was contorted to fit.** The evidence is in BERHAN's own text — factor the two
year-on-year leg formulas side by side and the later-period term of eq 2.5.12 (JY) and of eq 3.1.11
(MM-I) are both

```
psi_i · Pn(t,Ti) · [ E − 1 ]
```

with `E` the `Ti`-forward expectation of `I(Ti)/I(Ti−1)` — for JY
`(Pn(Ti−1)/Pn(Ti))·(Pr(Ti)/Pr(Ti−1))·e^C`, for MM-I the bivariate-lognormal integral of 3.1.11. The
cap/floors agree too: JY's eq 2.6.5 and MM-I's §3.2 inner expectation are both **Black on `(E, V²)`**,
differing only in what `V²` is.

So the seam is **the source's own decomposition**, not an abstraction invented to hold two things —
and both quantities are ones BERHAN names (`m`, eq 2.6.2; `V²`, eq 2.6.4). That is why this is not a
forced fit: had the protocol been JY's `convexity`, MM-I would have had to manufacture a fake `C` by
taking the log of its integral, which *is* the contortion the spec warned about.

### The refactor is NOT bit-identical, and that is reported rather than claimed otherwise
The spec required JY's numbers bit-identical as the proof the refactor is pure. **It cannot be
met**, and the reason is the factoring itself: `psi·(Pn(Ti−1)·PrRatio·e^C − Pn(Ti))` and
`psi·Pn(Ti)·(m − 1)` are the same quantity in exact arithmetic but reorder a division, so the last
digits move. Measured on a 4-period swap:

| | value |
|---|---|
| pre-refactor arithmetic | 36119.025432954419 |
| refactored through the protocol | 36119.025432954477 |
| relative difference | **1.6e-15** (5.8e-13 bp of notional) |

Every tranche 4 and tranche 5 oracle passes unchanged at its own stated tolerance. The substitute
for a bit-identical claim is therefore: *agreement to 1.6e-15, with the cause named (a reordered
division), and the full pre-existing oracle set green.* Saying "bit-identical" here would have been
false, and the alternative — keeping JY's grouping — would have required the protocol to expose
`convexity`, which is what makes MM-I impossible.

### The oracle that a type annotation cannot be
`_StubRatioModel` implements only the two protocol methods and is not a `JarrowYildirim`. Both
pricers price it correctly, and the cap/floor's output is *exactly* Black on the stub's two numbers —
so the pricers demonstrably read **only** `m` and `V²` from a model. If either still reached for
`model.convexity` or `model.params`, that test would raise. A protocol annotation alone proves
nothing at runtime; this does.

## 9. Findings for Cowork (§10 — raised, not decided)

### 9.1 `IndexRatioModel` carries `market` as a property, so `issubclass` does not work on it
Only `isinstance` does (`TypeError: Protocols with non-method members don't support issubclass()`).
Harmless here, and noted so a future test does not reach for the wrong one. The oracles check
behaviour regardless — tranche 5's lesson that a runtime protocol probe checks *names*, never
signatures, and that pyright is the authority.

### 9.2 `JarrowYildirim` gained a `_currency()` helper that infers its single currency
JY is single-currency by construction — nominal and real curves of one economy under the
foreign-currency analogy — so inference is not a guess, and it **refuses when ambiguous** rather than
picking. Same stance as `CurveSet.discount` refusing a missing key, and as tranche 10(b)'s
calibrator. Flagged because it is new behaviour on a model, not just a move.

### 9.3 A wrong requirement of mine, caught by its own test
I asserted `m > 1` on an inflationary curve pair. False at 3y→4y, where `m = 0.9914`: a negative
convexity overcomes the 2.1% carry. **A requirement that ignores the convexity is not a requirement
of this model** — the convexity is the whole point. Replaced with the honest decomposition: at
`σr = 0` the convexity vanishes exactly and then `m > 1` *is* forced by the curve pair, and the
convexity is asserted to move it. Stronger than what it replaced, because it also pins the sign of
`C`'s contribution. Fourth instance of a wrong-requirement caught by execution rather than review.
