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

## 5. Cowork's ruling on T49 (§10 step 4, 2026-11-16) — RATIFIED; the arc's last build

Built as ruled, 2089 tests, eleven gates green both runners. **The double reduction is the arc's best
oracle design** — A (→Heston at `m=0`) and B (→Merton at `η=0`) *are* each other's companion, both
blindnesses asserted (at `m=0`, `α/δ=0.9/1.2` and A still passes); the third T39 instance this arc and the
first of the mutual-companion shape. This goes in `CP_gatheral_vol.md` as a method high point. Ratified.

**Finding 1 (SVJ needs a wider truncation) — affirmed, handled in T49.** The jump tails make `φ_SVJ` decay
slower in `u`, worst at short `T`; the grid is config that belongs to *this* model (T44a's "config lives on
the model, its cost/accuracy determines it"), set to 400 and the constant oracled. Correct.

**Finding 2 (the negative Heston call price already on `main`) → `OPEN.md` + remediation at re-audit #5.
Ruled.** This is a **real defect**, not `to_go_deeper` (that bin is for unpulled *capabilities*, never for
a wrong number in shipped code) — a negative call price at `σ_v=0.9, T=30d, K=130` (−2.5e-04 vs the fine
grid's +1.7e-05), an arbitrage violation in a regime that is actually quoted, merged sixteen tranches ago.
It is the **"a reference is code too"** shape (T20a/T25): T33's oracle used a committed grid that is itself
wrong in the extreme regime, so it could not see the production grid was wrong there — the same defect
Finding 1 fixes one regime over. Disposition:
- **`OPEN.md` entry (OI-4), added by Code at step 6** — the six excluded regression points with the reason
  asserted (Code's interim call is right: exclude-with-reason, never pin a wrong number, §8), the measured
  defect, and the **re-open trigger = re-audit #5's oracle-power pass**.
- **The fix is a remediation tranche out of re-audit #5** (the T45-from-#4 precedent). Re-audit #5 fires on
  this merge; it will sweep T33 oldest-first anyway, and this is a **named** target in its scope — an
  adaptive/regime-aware truncation for the Heston (and SVJ) inversion, oracled against a high-precision
  reference in the high-vol/short-`T` regime the current one cannot resolve.

**Finding 3 (bit-identity committed as a cross-platform test) → a §4 charter line (added).** The Gil-Pelaez
extraction to `engine/fourier_inversion.py` is the right §6 rule-of-two call (second consumer; the
alternative broke §7). The lesson — **bit-identity is a property of a measurement, not a committed test**,
and an accumulated transcendental sum's committed tolerance is **relative**, not ULP — is genuinely new:
the existing op-order clause did not stop it, because a *true* bit-identity committed as the wrong kind of
test is a different failure from a *false* bit-identity claim. One §4 line, earned by two CI rounds.

On merge the vol-surface arc closes: `CP_gatheral_vol.md` + re-audit #5 (carrying the glyph sub-kind and
first-shape-twice watch items, and OI-4 as a named remediation target).
