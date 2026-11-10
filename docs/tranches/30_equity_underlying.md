# Tranche 30 — the equity underlying: forward, dividends, and the Black-Scholes vanilla

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.36.0`.
**Source:** `BROCKHAUS-2016` ch 2 (dividends, repo, vanilla), ch 5.1–5.3 (the dividend forward), ch 18.2
(Black-Scholes). **Pre-registered RED.** First unit of the equity & hybrids arc
(`docs/equity_hybrids_arc.md`) — the equity base everything else pulls. First *equity* in the tree.

## 0. The product worth having
**The European equity option**, which drags in the equity underlying it needs: a spot, a repo rate, a
**dividend** schedule, and the **equity forward** they imply. The option itself is Black-Scholes on
that forward (`models/black`, verified since T5) — so the genuinely new content is the **dividend
forward**, the equity-specific object the whole arc rests on.

The dividend forward (ch 5.1–5.3), the three conventions this tranche carries:
- **continuous yield `q`:** `F(T) = S₀ · e^{(r−q)T}`;
- **proportional dividends `{dᵢ}` at `{tᵢ}`:** `F(T) = S₀ · e^{rT} · ∏_{tᵢ≤T}(1−dᵢ)`;
- **discrete (cash) dividends `{Dᵢ}`:** `F(T) = (S₀ − ∑_{tᵢ≤T} Dᵢ·e^{−r tᵢ}) · e^{rT}` (the spot-minus-
  PV-of-dividends forward).

## 1. The pieces and their layers
- **L1 `market/` — the equity underlying as market data:** spot `S₀`, a repo/borrow rate, and a
  **dividend schedule** (a frozen value object; discrete-cash and proportional entries, or a
  continuous `q`). Quote-side equity data, the same status as a discount curve.
- **L2 `products/` — the European equity option** (frozen data: strike, expiry, call/put). Describes,
  does not price (§2).
- **L4 `engine/` — the equity vanilla engine:** compute the dividend forward, then **reuse
  `black_price`** on `(F, K, total_variance, df)`. Thin by design — the equity content is the forward,
  not a new option formula. `@register`ed.
- No new model: Black-Scholes flat vol is the T30 model; the vol *surface* and local/stochastic vol are
  T32/T33.

## 2. The oracle (RULED)
- **Reduction to the verified core (exact, the preferred anchor §4).** Zero dividends and zero repo ⇒
  `F = S₀·e^{rT}`, and the equity vanilla is **bit-identically** `black_price` on that forward (Black
  is verified since T5; this tranche must not perturb it). The non-degenerate companion: a **material**
  dividend stream where the forward is visibly below `S₀·e^{rT}`.
- **Pointwise transcription of the dividend forward (§4), two renderings.** Each of the three forms
  (continuous/proportional/discrete) transcribed from ch 5.1–5.3 in `-raw` and `-layout` and reconciled.
  The test computes its own reference (`∏(1−dᵢ)`, `∑Dᵢe^{−rtᵢ}`), never reading it back from the code.
- **Cross-form consistency (a cross-product oracle, different code each side).** A fine strip of small
  proportional dividends `dᵢ = q·Δt` converges to the continuous-yield forward `e^{−qT}` as `Δt → 0`
  (assert small *and* shrinking); a single discrete cash dividend and the equivalent proportional one
  agree at inception. Different formulas for the same forward — catches a shared transcription error.
- **Requirement oracles (the finance, survive a transcription error).** **Put-call parity**
  `C − P = df·(F − K)` on the equity option; the forward is **monotone decreasing** in every dividend
  and in `q`; `F → S₀·e^{rT}` as dividends → 0; the option is monotone in vol and satisfies the
  no-arbitrage bounds.
- **Degenerate-fixture guards (mandatory, §4).** Assert the dividend stream is **material** (else the
  forward equals the no-dividend forward and all the equity content is invisible — the trap); assert at
  least one discrete *and* the schedule has a dividend **strictly before** expiry (a dividend after `T`
  must not affect `F(T)` — a date-boundary the `tᵢ ≤ T` sum must get right, its own degeneracy if every
  dividend is pre-expiry). **Mutation table:** drop the `e^{−r tᵢ}` PV discounting on cash dividends;
  use `∑Dᵢ` (undiscounted); flip `1−dᵢ` to `1+dᵢ`; include a post-expiry dividend; sign-flip `q`. Each
  must die.

## 3. Process
Branch `tranche/30-equity-underlying`. `test` (RED: the dividend-forward transcription + the
Black-reduction + the mutation table) → `feat` → `docs` → `chore(release): v0.36.0`. **Open the draft PR
early** (the T26b CI note). `source:` `BROCKHAUS-2016` (now `worked-from`; the `references` gate will
require it once a module cites it). The equity engine/market modules carry `crosscheck:` markers
(QuantLib has equity forwards / `BlackScholesProcess` — `crosscheck: quantlib BlackScholesProcess`, or
`none — <reason>`, §8). **Raise an `Ask:`** if a dividend convention needs a piece the spine lacks.

## 4. What this tranche is NOT (scope fence)
No vol **surface** or skew (ch 4 → T32), no local/stochastic vol (ch 6 → T32/T33) — T30 is flat-vol
Black-Scholes. No **affine/stochastic dividends** (ch 5.4–5.6) and no stochastic-hazard dividend
coupling (ch 5.7) — deferred (`to_go_deeper`). No exotics (ch 3). No hybrids yet (T34–T36). The point
of T30 is the *underlying and its forward*, done right, so every later equity tranche pulls a verified
base.
