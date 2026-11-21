# T71 — the volatility-smile short-rate models: Normal/Bachelier + displaced-diffusion

**Cowork spec, 2026-11-21.** Sixth tranche of the single-curve short-rate arc (project nine,
`MCINERNEY-ZASTAWNIAK`; ch 8.1–8.4). The chapter extends the Black caplet to a **smile** three ways; this
tranche builds the two that are a **reparameterisation of Black** and reuse the tree whole. **CEV (ch 8.3) is
split to T72** — it is semi-closed (a non-central chi-squared the tree does not have), a genuinely-new
primitive that deserves its own unit rather than riding the two easy ones (the arc plan flagged the split).

## Target
Price a caplet under the **Normal/Bachelier** model (ch 8.2) and the **displaced-diffusion** model (ch 8.4),
each producing an implied-Black-vol **smile/skew** — `price(Caplet, model) → PricingResult` for each.

## Pieces, layer, reuse
- **Bachelier/Normal** (new engine, L4): the caplet `df·[(F−K)·Φ(d) + σ_N√T·φ(d)]`, `d = (F−K)/(σ_N√T)` —
  a **normal** forward, not lognormal. Reuses `foundation/distributions.norm_cdf`/`norm_pdf`. **The Bachelier
  call IS the T25 Bachelier EPE** (`m·Φ(m/s) + s·φ(m/s)`, `credit_valuation_adjustment.py`) — so this reuses /
  cross-checks **already-verified code** (the arc's signature anchor, T67/T69): different code each side, the
  same closed form.
- **Displaced-diffusion** (new engine, L4): shift the forward and strike by a constant `β`, then apply Black —
  `black_price(F+β, K+β, σ_dd²·T, ·)` — **reuses `black_price` whole**. The shift is the one new parameter.
- **The model(s)** (L3): a `BachelierModel` (a normal vol `σ_N`) and a `DisplacedDiffusionModel` (a lognormal
  vol `σ_dd` + a shift `β`), each carrying the `MarketSnapshot` (§3). Distinct types — a normal model and a
  shifted-lognormal model are different ideas (§6), and distinct from the Gaussian `BlackModel`.

## The mathematics (re-transcribe from the source — NOT this spec, §5)
Read **McInerney–Zastawniak ch 8.2 (Normal) and 8.4 (displaced-diffusion)** in `-raw` and `-layout` and
reconcile (§4). Transcribe the Bachelier caplet and the displaced-diffusion mapping pointwise. The spec names
the obligation, not the form (§4) — in particular the **smile each produces is MEASURED against the closed
form, never pre-asserted as a slope or sign** (the T62/T63 lesson: match-forced error and sign changes are
real; measure the smile, report its shape).

## Oracles (RED-first, §4)
1. **Bachelier caplet — pointwise transcription AND a cross-check against the T25 CVA Bachelier EPE.** The
   EPE and the Bachelier call are the same closed form in different code, so their agreement is the
   reduce-to-already-verified-code anchor (consistency, not independence — §4 same-object caveat, flag it);
   the pointwise transcription from ch 8.2 is the independent correctness anchor.
2. **Displaced-diffusion — reduction to Black at `β = 0`, exact** (`black_price(F, K, σ²T)` directly).
   **Companion required (§4/T39):** a genuine `β ≠ 0` case checked against the pointwise transcription — the
   `β = 0` reduction is blind to the whole shift. Expect the T61 `log`/`exp` ULP if the reduction composes
   through the shift — **band it (equal in ℝ), do not assert `==`** (the standing lesson).
3. **The implied-Black-vol smile is MEASURED, not asserted** (`black_implied_vol`, reused): invert each
   model's prices to Black vols across strikes and **report** the smile the closed form produces (Bachelier
   gives one shape, displaced-diffusion a near-linear skew). Assert only what the closed form forces (e.g.
   at-the-money the Bachelier and Black vols relate by a known factor), not a pre-committed slope — the
   obligation-not-form clause, which this arc has now paid three times.
4. **Requirement oracles:** put–call parity per model (`C − P = df·(F − K)`); the `σ → 0` intrinsic limit
   (`df·(F−K)⁺`); and **monotonicity in volatility** (price rises with `σ_N` / `σ_dd`) as a sign check.
5. **A cross-model tie at the money:** a displaced-diffusion with a specific shift and a Bachelier can be
   matched at the money (both reduce to near-linear local behaviour); if the source gives the relation, assert
   it — else leave it to T72's fuller smile comparison.

## Degeneracy premises (§4 / §9)
- The `β = 0` companion is **non-degenerate in the shift** (T39).
- **Off-the-money strikes** (the smile is invisible at the money — a one-strike fixture is `τ=1`'s cousin),
  **non-trivial `σ√T`**, and a **non-zero `F−K`** for the Bachelier `d`. Prefer a **mutation to a reading**:
  drop the `φ` term in Bachelier, swap `Φ`/`φ`, drop the shift in displaced-diffusion, shift only `F` and not
  `K` (a classic displaced-diffusion bug the parity oracle catches).
- The smile fixture must span strikes where the measured implied-vol actually **varies** (a near-ATM-only
  sweep makes every model look flat).

## Provenance / gates
- Header `source: MCINERNEY-ZASTAWNIAK §8.2, §8.4`; `oracle:`; `crosscheck:` (QuantLib has Bachelier and
  shifted-Black engines — a real target).
- Version → **v0.73.0**; one floor bump; one `CHANGELOG` entry.

## Out of scope (deferred)
- **CEV (ch 8.3) → T72**, the last spine tranche: it needs a **non-central chi-squared** CDF (new L0
  primitive, oracled against `scipy.stats.ncx2` off the production path — the T20a `distributions.py`-off-scipy
  arrangement), and reduces to **Black at β=1** and **Bachelier at β=0** (the two models this tranche builds
  are its reduction anchors). Its own unit because the primitive is real infrastructure, not a Black
  reparameterisation.
- **Stochastic volatility (ch 8.5)** — `to_go_deeper` (SABR-adjacent, no closed form; a candidate, not pulled).
