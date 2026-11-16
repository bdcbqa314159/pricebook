# Tranche 48 — volatility derivatives (the volatility swap and its convexity)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.54.0`. **Source:**
`GATHERAL-2006` ch 11 ("Volatility Derivatives"). **Pre-registered RED.** The second build of the
vol-surface arc (T47 was dropped by the §1 pull gate; `docs/gatheral_vol_surface_arc.md`). Reuses T31
(variance replication) and T33 (Heston) whole.

## 0. The product worth having
**The volatility swap and its convexity adjustment** — the derivative that pays realized *volatility*
`√V_T` rather than realized *variance* `V_T`, and the model-dependent gap between the two. The variance
swap (T31) already prices `E[V_T]` by static replication; the volatility swap pays `√V_T`, which
**cannot be statically replicated** (Gatheral ch 11: "there is no replicating portfolio for a volatility
swap and the magnitude of the convexity adjustment is model dependent"). The value worth having is the
fair volatility-swap strike `K_vol = E[√V_T]` and the **convexity adjustment** `√(K_var) − K_vol ≥ 0`,
computed in a model the arc already holds (Heston, T33).

## 1. The pieces and their layers
- **L2 — `VolatilitySwap` (a frozen product).** Notional, the realized-variance window (the same
  observation convention T31's `VarianceSwap` carries). A `VarianceOption` (a call/put on `V_T`) is a
  **flagged in-scope-if-clean extension** — see §4.
- **L4 — the volatility-swap engine, `price(VolatilitySwap, HestonModel)`.** `K_vol = E[√V_T]` via the
  **Laplace-transform representation** `E[√V_T] = (1/2√π) ∫₀^∞ (1 − E[e^{−sV_T}]) · s^{−3/2} ds`, with the
  Heston transform of integrated variance `E[e^{−sV_T}]` — a closed form built from the T33 Heston
  apparatus. A 1-D quadrature of a closed-form transform, so it is a closed-form pricer (SE only if any
  piece goes to MC, §3.7).
- **Reuse:** T31's variance replication supplies `K_var = E[V_T]` (the √-argument and the convexity's
  other half — a **cross-product anchor**, different code each side); T33's Heston characteristic-function
  machinery supplies the transform. No new L0 primitive expected.

## 2. The oracle (RULED)
The **Jensen inequality** is the model-independent correctness requirement; the **Heston convexity
closed form** is the magnitude anchor; the **cross-product** reconciliation with T31 ties it to what we
hold.

- **The Jensen bound — a requirement oracle, for ALL parameters (§4).** `K_vol = E[√V_T] ≤ √E[V_T] =
  √(K_var)`, by Jensen (√ concave), so the convexity adjustment `√(K_var) − K_vol ≥ 0` **always**. Assert
  it across a parameter sweep — it is derived from the finance, so it survives a transcription error in
  the transform (the requirement-oracle discipline).
- **The `η → 0` reduction — the exact anchor, with its mandatory non-degenerate companion (§4, T39).**
  Zero vol-of-vol makes `V_T` deterministic, so `K_vol = √(K_var)` **exactly** (the convexity vanishes).
  **The companion MUST move `η`** — this reduction is exact *because* it zeroes the very parameter the
  convexity lives in, so it is structurally blind to the convexity (the T39 blind spot, verbatim). The
  companion asserts the convexity is **non-zero** and matches the Heston closed form at `η ≠ 0`.
- **The Heston convexity magnitude — the absolute-level anchor (§4, T43).** The Jensen bound and the
  reduction are both scale/shape; the *level* of the convexity is unpinned by them. Assert the convexity
  adjustment's **absolute value** against the ch 11 closed form, and its **small-`T` leading order**
  `≈ (η/λ)·√(v/T)·(const)` (ch 11 p.144 gives the leading-order behaviour `v/T · η/λ`) as a reduction in
  the short-maturity limit — two independent handles on the magnitude, not just its sign.
- **Cross-check: Heston `K_vol` vs an independent MC of `√V_T` (§4, the banded-resolution discipline).**
  Simulate the Heston variance path, average `√V_T`, and agree with the transform-integral `K_vol` to
  `k·SE + the quadrature's own measured error` (T17 — never one flat band), with the band's resolution
  stated (the smallest convexity error a green rules out at the committed path/quadrature count).
- **Cross-product: the variance leg reconciles with T31 (§4).** `K_var` inside T48 (the √-argument) must
  equal T31's independently-built replication `K_var` to tolerance — different code each side, the T3/T5
  cross-product anchor, catching a shared Heston-reading error no within-T48 check can.

**Mutation (premises asserted up front — the arc's lesson).** Each must die: drop the convexity (return
`√(K_var)` — the `η→0` degenerate state; killed only by the **`η ≠ 0` companion**, assert it); mis-sign
the Laplace integrand (killed by the magnitude anchor, not the Jensen bound, which tolerates a smaller
positive convexity — the T43 "sign/presence complete, magnitude silent" trap); swap `√V` for `V` in the
payoff (killed by the cross-product vs T31 — they would then be identical, which the convexity-non-zero
premise forbids). **Premises:** `η ≠ 0` (non-degenerate vol-of-vol), `T` spanning both a short and a
long maturity (the leading-order reduction is a *short-T* statement — a one-maturity fixture is the
degenerate case), the variance window straddling the valuation date if any historical observation is
load-bearing (the tenth-shape premise, carried before the sweep).

## 3. Process
Branch `tranche/48-volatility-derivatives`. `test` (RED: the Jensen requirement + the `η→0` reduction &
its `η≠0` companion + the Heston convexity magnitude & small-`T` order + the MC cross-check + the T31
cross-product + the mutation table) → `feat` → `docs` → `chore(release): v0.54.0`. **Transcribe the
Laplace-transform representation and the Heston integrated-variance transform in both renderings (§4).**
`source:` `GATHERAL-2006`; `crosscheck: quantlib` (QuantLib has Heston variance/vol-swap tooling). Open
the draft PR early; pre-register the Jensen requirement + the convexity magnitude first.

## 4. What this is NOT (scope fence)
No **variance-swap rebuild** — T31 is reused whole for `K_var`. No **model-independent vol-swap
replication** (there is none — that is the whole point; the convexity is Heston's). **Options on variance
`(V_T − K)⁺` — a flagged extension, ruled at build:** if the Heston **variance distribution** (the
inverse transform of `E[e^{−sV_T}]`) falls out cleanly from the same transform T48 already builds, fold
a `VarianceOption` engine in (oracle: the zero-strike variance call = the variance swap `E[V_T]`;
put-call parity `call − put = K_var − K`); if it pulls a materially new object (a full density inversion),
it is **T48b** or `to_go_deeper`, not force-fit here (§1 pull gate). Raise it as an `Ask:` with the
measured cost. No **VXB / forward-starting** convexity (ch 11's later sections) — a later pull. No **Napoleon/
cliquet** structures (ch 10, the exotic zoo).

---

## 5. Cowork's ruling on T48 (§10 step 4, 2026-11-16) — RATIFIED

Built as ruled, 1959 tests, eleven gates green both runners (the `slow` risk passed — 2.0s vs 10s),
applied 20/20 killed 20 survived 0. The volatility swap and its convexity are in. Ratified.

**The variance-option fork → `to_go_deeper`, not T48b** (`to_go_deeper` route 3). Nothing in the spine
pulls it (T49/Bates does not), and it is a materially new object (a numerical *inverse* Laplace transform,
not T48's forward quadrature). **And the decisive finding is a Cowork-oracle error, mine, caught by
measurement:** the two oracles I specced for variance options are *structurally blind*. The zero-strike
reduction passes on a **broken inverter** because `C(K)→E[V]` is carried by the exact `s·E[W]/s²` term and
the inversion contributes nothing at the limit — the **T39 reduction blind spot** verbatim (the limit is
exactly where the thing under test vanishes); put-call parity is satisfied **by construction**. A set
complete about the limit and the relation, silent about the interior — the **T43 shape at the oracle-set
level**. Folding them in would have shipped a plausible wrong number green. A future build needs an
**interior-strike absolute-level anchor** (its own MC). No charter edit — it is T39 + T43 composed, both
already in §4.

**Two spec errors of mine, corrected by Code's derivation (point 2).** (a) The convexity's leading order is
**quadratic** in `η/λ`, not linear as §2 wrote: `√E[V] − E[√V] ≈ var[V]/(8·E[V]^{3/2})`, so ch 11's
"roughly 3.39×" is `3.39² = 11.50` (measured 11.49). (b) It is a **`T→∞`** statement, not the small-`T` the
spec claimed. Both were mine, inheriting ch 11's own order-of-magnitude claim — which is **wrong by a
power** (the source as code, §4 "a reference is code too"), settled by the Taylor expansion rather than the
plot (§4 "no number is fitted to a figure"). Good catch; owned.

**Affirmed, no action.** (Point 1) The lost overbar glyph on the Heston transform — both renderings drop
it, so dual-rendering is powerless — recovered by a **requirement oracle against T33** (slope-at-zero =
`integrated_variance`, 1.5e-8), with a **non-degenerate `v₀ ≠ θ` fixture** because Gatheral's own Figs
11.2/11.3 set `v₀ = θ` (the source hands over the degenerate fixture). This is the **second** tranche where
a lost glyph was recovered by independent code rather than a second rendering (T46 by a later chapter, T48
by an earlier tranche) — an emerging pattern; a **third** instance earns a §4 line, not yet. (Point 3) The
zero-*rate* `df=1` mutation hole is the catalogue's first shape ("a zero makes a factor invisible") in a
parameter not obviously load-bearing; fixed at 3%, premise asserted — the vol arc keeps surfacing the first
shape (T46 OTM intrinsic=0, T48 zero rate).

**The miscount in 0273 was mine** — three artifacts, not "four"; you committed the three that existed and
rightly did not hunt a phantom. The new rule turned on my own report, correctly.

T49 (Bates / SV+jumps) is the queue's next; I spec it on the merge handoff.
