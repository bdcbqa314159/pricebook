# Arc plan — the single-curve short-rate spine (project nine, `MCINERNEY-ZASTAWNIAK`)

**Opened 2026-11-20 at v0.67.0.** The ninth vertical, and the first **revisit** arc: a source already
`worked-from` since T10 (the Hull-White cap machinery), now mined for the chapters T10 left. The target is a
solid **single-curve short-rate spine** — the deliberate **pre-crisis counterpoint** to the
collateralised/multi-curve arc (project seven). A short-rate model generates the *whole* discount curve from
one factor `r`, so discounting and forwarding come off **one curve by construction** — the single-curve mode
of the design invariant, exercised rather than asserted. This is the "parallel" the collateralised arc was
built to enable: the same tree prices in either era.

Scope ruled by Bernardo, 2026-11-20: **the bigger spine (~6 tranches)** — the short-rate models of ch 3, the
HJM framework of ch 4, and the volatility-smile short-rate models of ch 8.

## Why "big" is affordable — the reuse (step-1 greps, 2026-11-20)
- **The OU / Hull-White machinery is the spine's backbone.** `models/hull_white.py` carries
  `HullWhiteParams`, **`hw_beta(κ,t,T)`** (which **is** the affine `B(t,T) = (1−e^{−κτ})/κ` every ch-3 model
  shares), `bond_option_variance`, `zero_bond_option`, `hw_caplet`. `models/lgm.py`'s
  **`simulate_lgm_states`** is the OU simulator — every ch-3 short rate is OU (or Gaussian), so it is the MC
  companion throughout. **Vasicek is not rebuilt from scratch — it is the constant-parameter sibling of the
  Hull-White the tree holds.**
- **`foundation/linalg.py`** (Cholesky / `pseudo_square_root`, reused at T63) — the two correlated factors of
  G2++ (T67).
- **`models/exercise_boundary.py` + `engine/bermudan_swaption_mc.py`** (T18) — the Bermudan exercise policy,
  reused in a short-rate model (T68) rather than the LMM.
- **`models/black.py`** (`black_price`) and the vol-surface machinery — the ch-8 smile models (T71) extend
  the Black caplet; Normal/Bachelier and displaced-diffusion are `black_price`-adjacent.

## The spine (queue)
**T66 → T67 → T68 → T69 → T70 → T71 → checkpoint → re-audit #8.** Single-curve, single-currency throughout.
Every ch-3 model is **affine** (`P(t,T) = A(t,T)·e^{−B(t,T)·r}`), so each carries a closed-form zero-coupon
bond and a closed-form caplet/swaption, and the models **nest** — the reductions are the oracle spine.

- **T66 — the Vasicek model** (ch 3.4), with **Merton** (ch 3.3) as its reduction. The canonical
  mean-reverting Gaussian short rate `dr = κ(θ−r)dt + σ dW`; affine bond + caplet. L3. **Oracles:** the bond
  `A(t,T)`, `B(t,T)` and caplet transcribed pointwise from ch 3.4; the **reduction κ→0 ⇒ Merton** (the
  driftless Gaussian short rate, ch 3.3) — exact, with a **companion at genuine κ>0** (the reduction kills
  the mean reversion, T39); the **MC companion reusing `simulate_lgm_states`** (the short rate is OU); and a
  **cross-check against Hull-White** (Vasicek is HW with constant θ — `hw_beta` IS Vasicek's `B`, so assert
  they agree, a consistency guard not independence, §4 same-object caveat). Heavy reuse, deliberately first.
- **T67 — the two-factor Hull-White model (G2++)** (ch 3.7). Two correlated Gaussian factors summing to the
  short rate; affine bond in two state variables. L3. **Oracles:** the two-factor bond transcribed pointwise;
  the **reduction to one-factor HW** at zero second-factor vol (exact, companion at genuine two-factor);
  correlation pinned by the **realised covariance** of the simulated factors (T17) AND its **value** in the
  bond variance (T43) — the T63 correlation discipline; `linalg` Cholesky reuse.
- **T68 — Bermudan swaptions in the Hull-White model** (ch 3.6). The tree has Bermudans via the LMM (T18);
  this prices them in a short-rate model — a cleaner, lower-dimensional approach. L4. **Oracles:** reuse the
  T18 `exercise_boundary` machinery; **cross-model check** — the HW Bermudan and the LMM Bermudan on a
  matched European-swaption calibration must agree within the structural bracket (T18/T19: a foresight +
  policy-suboptimality sandwich is not sampling error); the European-swaption reduction (a one-exercise
  Bermudan = the European) exact.
- **T69 — the Gaussian HJM framework** (ch 4.1–4.3, 4.5). The forward-rate SDE `df(t,T) = α dt + σ dW` with
  the **HJM drift condition** `α = σ·∫σ` (no-arbitrage), the Gaussian (deterministic-σ) case, and its
  calibration. L3. **Oracles:** the drift condition is a **requirement oracle** (it is forced by
  no-arbitrage, survives a transcription error); the **reduction to Hull-White** — a Gaussian HJM with
  `σ(t,T) = σe^{−κ(T−t)}` **is** Hull-White exactly (the exact reduction anchoring the whole framework);
  the forward-measure martingale property as a self-consistency guard.
- **T70 — multi-factor HJM** (ch 4.4). Extends T69 to N driving factors; the drift condition becomes a sum.
  L3. **Oracles:** the reduction to one-factor HJM (T69) at a single factor; the drift-condition requirement
  oracle per factor; the factor correlation pinned as in T67 (realised covariance + value).
- **T71 — the volatility-smile short-rate models** (ch 8.1–8.4): **Normal/Bachelier** (8.2), **CEV** (8.3),
  **displaced-diffusion** (8.4) — each extends the Black caplet (8.1) to a smile. L3. **Oracles:** each
  transcribed pointwise; **reductions to Black** — displaced-diffusion → Black at zero shift, CEV → Black at
  β=1, Normal → the Bachelier closed form; put-call parity per model; the implied-Black-vol smile each
  produces **measured** against the closed form (the §4 obligation-not-form clause — measure the smile, do
  not pre-assert its slope). May split if one tranche is too large; ch 8.5 stochastic vol is `to_go_deeper`.

## Deferred at the pull gate (to `to_go_deeper`, ruled per-tranche)
- **Exotic IR products** (ch 7: LIBOR-in-arrears, in-arrears/constant-maturity swaps, ratchet floaters, range
  accruals, trigger swaps) — oracle-able (each a payoff on the models built here), unpulled unless a product
  pulls one.
- **Stochastic-volatility smile** (ch 8.5) — SABR-adjacent, harder to oracle; a candidate, not pulled.
- **LMM/SMM** (ch 5–6) — **held** (BERHAN MM-I + the Jäckel arc); McInerney is a second source, not a rebuild.

## The single-curve parallel — what this arc demonstrates
These models are single-curve **by construction** (one `r` → one `P(t,T)` → one curve for discounting and
forwarding). Built beside the collateralised arc's multi-curve machinery, they are the concrete "pre-crisis
era" half of the design invariant Bernardo asked for: the same library prices a trade in either world, and a
parallel project can show the two side by side. No new invariant — the existing one (project seven),
exercised from the other side.

## Status (updated 2026-11-21)
- **Built & merged:** T66 Vasicek (v0.68.0) · T67 G2++ (v0.69.0) · T68 Bermudan-in-HW (v0.70.0).
- **Built & merged (cont.):** T69 Gaussian HJM (v0.71.0) · T70 multi-factor HJM (v0.72.0, generalised into one
  `GaussianHjmModel`).
- **Specced, queued:** T71 Normal/Bachelier + displaced-diffusion smile models; **T72 CEV** (split out — it
  needs a non-central chi-squared primitive, so it is its own unit rather than riding the Black-reparameterise
  pair; the arc plan always flagged the ch-8 split). The queue is now **T71 → T72 → checkpoint → re-audit #8**.
- `MCINERNEY-ZASTAWNIAK` `worked-from`; the arc note + chapter map are in `docs/references.md`.
- **Method advances so far:** §4's reduction-verification third sibling (exact-but-numerically-unreachable →
  rate-in-a-regime, T66) + its regime-measured-per-model tightening (T67). Two `to_go_deeper` routes from T68
  (cross-model Bermudan validation; boundary-split quadrature).
- After T71: the checkpoint (`CP_stochastic_rates.md`) + re-audit #8.
