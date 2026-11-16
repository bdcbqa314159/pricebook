# Tranche 49 — stochastic volatility plus jumps (the SVJ / Bates model)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.55.0`. **Source:**
`GATHERAL-2006` ch 5 ("Adding Jumps"), the SVJ model. **Pre-registered RED.** The **arc's last build**;
its merge closes the vol-surface arc (`CP_gatheral_vol.md` + re-audit #5). Reuses T33 (Heston Fourier
inversion) whole.

## 0. The product worth having
**The SVJ model — Heston stochastic volatility plus Merton-style lognormal jumps in the underlying** —
the model ch 5 shows "wins" because jumps generate the steep short-dated skew that Heston alone cannot
(the very gap T46's SVI fit exposed, ch 3). The SDE (Gatheral ch 5):

```
dS = μ S dt + √v S dZ₁ + (e^{α+δε} − 1) S dq        jumps: dq Poisson(intensity m), log-jump ε ~ N(0,1)
dv = −λ(v − v̄) dt + η √v dZ₂                        Heston variance (λ = MEAN-REVERSION, not jump rate)
```

**Notation guard (built into the spec):** Gatheral's `λ` is the Heston **mean-reversion speed**; the
**jump intensity** is the Poisson rate of `dq` (call it `m` here, jumps/year). The reduction to Heston is
at **`m → 0`**, never `λ → 0`. The vocabulary object is a `BatesParams` (the Heston five + jump intensity
`m`, jump mean `α`, jump vol `δ`) — eight fields, so it bundles (§7) into the Heston block + a `JumpParams`.

The value worth having: a European option priced under SVJ by the **same Fourier inversion T33 built**,
the SVJ characteristic function being the **Heston CF × the jump CF** (independent jumps).

## 1. The pieces and their layers
- **L3 — `BatesModel`** carrying `BatesParams` + the `MarketSnapshot` (§3). The characteristic function
  `φ_SVJ(u) = φ_Heston(u) · φ_jump(u)`, with `φ_jump(u) = exp( m·T·( e^{iuα − δ²u²/2} − 1 ) − iu·m·T·(e^{α+δ²/2} − 1) )`
  (the drift compensator keeps `S` a martingale). Transcribe both factors in two renderings (§4).
- **L4 — the engine** is T33's Fourier-inversion pricer, reused whole, reading `φ_SVJ` instead of
  `φ_Heston`. A `BatesModel` registers for the European option through the T33b `(product, model)` registry
  — one product, a new model-pricer, the dispatch already built. No new inversion machinery.
- **Reuse:** T33's Heston CF and inversion (the engine and `φ_Heston`); the European option product (T30);
  `black`/`norm_cdf` for the Merton reduction's closed form. No new L0 primitive expected.

## 2. The oracle (RULED) — a DOUBLE reduction, each the other's companion
The SVJ sits at the intersection of Heston and Merton, so **two exact reductions anchor it, and each is
structurally blind to exactly what the other tests** — the T39 discipline made symmetric.

- **Reduction A — to Heston at `m → 0` (the exact anchor, reuses T33).** Zero jump intensity ⇒ pure
  Heston; `φ_SVJ → φ_Heston` exactly, so the price equals T33's to the ULP. **This reduction is blind to
  the entire jump component** (it zeroes it), the T39 blind spot — so it cannot be the only anchor.
- **Reduction B — to the Merton jump-diffusion at `η → 0`, `v₀ = v̄` (the companion that exercises the
  jumps).** Zero vol-of-vol with variance pinned at its mean ⇒ deterministic variance ⇒ the **Merton
  jump-diffusion closed form**, the Poisson-weighted Black–Scholes series
  `C = Σ_{n≥0} e^{−m'T}(m'T)ⁿ/n! · BS(S, K, σ_n, T)` with `m' = m·e^{α+δ²/2}` and `σ_n² = v̄ + nδ²/T`.
  This is an **independent closed form** (not the Heston CF), and it is non-degenerate in exactly the jump
  parameters Reduction A kills — so it is Reduction A's mandatory companion, and A is B's (B zeroes the
  stochastic vol). The two are each blind to what the other tests: **A cannot see the jumps** (it is taken
  at `m=0`, so varying `α, δ, m` leaves it unchanged), **B cannot see the stochastic vol** (taken at
  `η=0`). Assert both reductions **and** make this blindness explicit — a jump-parameter mutation must
  leave A green and kill B, a vol-of-vol mutation the reverse — so neither reduction is mistaken for
  covering the whole model.
- **Put-call parity — a requirement oracle (§4), model-independent**, for all parameters.
- **The jump CF transcription + a requirement (§4).** Transcribe `φ_jump` pointwise (two renderings), and
  carry the requirement that the **compensator makes `S/numéraire` a martingale**: `φ_SVJ(−i) = 1`
  (the forward is recovered exactly), which a dropped or mis-signed compensator breaks — the T34
  "assert the identity, not just the terms" shape.
- **MC cross-check (§4, banded per T17).** Simulate Heston variance + a compound-Poisson lognormal jump,
  price by MC, agree with the Fourier price to `k·SE + the inversion's own measured error`, the band's
  resolution stated. The SE rides on the `PricingResult` (§3.7).
- **Cross-arc connective check (ordering, not a value — "no number fitted to a figure", §4).** Fit T46's
  SVI to an SVJ-generated surface and to a Heston-generated one; the SVJ short-`T` skew is **steeper**
  (ch 5's thesis, the T46 gap). Assert the *ordering* (SVJ skew > Heston skew at short `T`), never a
  figure value — the capstone tie-back to SVI.

**Mutation (premises asserted up front).** Each must die: drop the jump CF factor (→ Reduction A's state;
killed only by **Reduction B / a non-zero `m`** — assert it); drop the compensator (→ the forward drifts;
killed by `φ_SVJ(−i)=1` and parity); mis-sign `α` (→ skew direction; killed by B and the ordering check).
**Premises:** `m ≠ 0`, `η ≠ 0`, `α ≠ 0` (a symmetric jump hides the skew — the T17 symmetry degeneracy),
both a short and a long `T` (the jump-skew effect is short-`T`), the option window straddling val-date if
any historical feature is load-bearing (the tenth-shape premise).

## 3. Process
Branch `tranche/49-bates-svj`. `test` (RED: reductions A **and** B + parity + the martingale/compensator
identity + the MC cross-check + the SVI-ordering tie-back + the mutation table) → `feat` → `docs` →
`chore(release): v0.55.0`. **Transcribe `φ_jump` and the Merton series in both renderings (§4).** `source:`
`GATHERAL-2006`; `crosscheck: quantlib` (QuantLib has a Bates engine). Open the draft PR early; pre-register
reductions A and B first — B is the one that is easy to omit and is the whole point.

## 4. What this is NOT (scope fence)
No **SVJJ** (ch 5 — simultaneous jumps in price *and* volatility; a `to_go_deeper` extension, one more
jump term in `dv`). No **jump-parameter calibration** to a market surface (T49 prices given params; fitting
is a later pull, the T46/SVI-informed route). No **Merton model of default / jump-to-ruin** (ch 6 — a
different use of jumps, partly T35's territory; `to_go_deeper`). No new **inversion machinery** — T33's is
reused whole; if it needs changing, that is a finding to raise, not a silent rewrite. T49 is the SVJ
European price and its two anchors; with it the vol-surface arc closes.

---

## 5. Cowork's ruling on T49 — (reserved for §10 step 4, after the build)
