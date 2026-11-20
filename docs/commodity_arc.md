# Arc plan — commodity option pricing (project eight, `CLARK-2014`)

**Opened 2026-11-19 at v0.61.0.** The eighth vertical: the **commodity** layer — the forward curve built
from cost-of-carry and **convenience yield**, Black-76 on futures, and the closed-form exotics the
commodity desk actually trades (Asians, spread options) plus a mean-reverting spot model. Clark Ch 2 is the
spine; it is unusually closed-form-dense, which is why a five-tranche spine is affordable.

Scope ruled by Bernardo, 2026-11-19: **the closed-form core (~5 tranches)** — every tranche closed-form or
closed-form-plus-MC-companion, maximal reuse, balanced against `to_go_deeper` for the lattice/stochastic and
market-specific material.

## Why "big" is affordable here — the reuse (step-1 greps, 2026-11-19)
- **`models/black.py` IS Black-76 on a forward** — `black_price(forward, strike, total_variance, kind)`,
  plus `black_delta/gamma/vega`, `digital_price`, `black_implied_vol`. `engine/equity_vanilla.py` already
  imports `black_price`, the precedent for a product engine reusing it. **So Black-76 is not rebuilt** — the
  genuinely-new object of T60 is the commodity *forward* (convenience yield), and the option engine reuses
  `black_price`. *(This is the T54 "exists but misunderstood" trap caught at step 1, not build.)*
- **The MC estimator primitive** — `foundation/mc_estimator.py`, `foundation/results.py`
  (`MonteCarloEstimate`, `PricingResult` with SE) — is the companion for the arithmetic-Asian and Kirk
  approximations (bands per §4's `k·SE + measured approximation error`, resolution stated).
- **The LGM/OU simulator** — `models/lgm.py` (`simulate_lgm_states`, `lgm_H`, `lgm_zeta`) — is a driftless
  Gaussian OU process. The Schwartz one-factor commodity model is OU in **log-spot**, so T64's model
  **reduces to** this simulator (its MC companion) exactly as the LGM reduced to Hull-White.
- **Forward/discount curves** — the T8/T54 bootstrap and discount handles — carry the `(r−δ)` the
  cost-of-carry forward needs.

## The spine (queue)
**T60 → T61 → T62 → T63 → T64 → checkpoint → re-audit #7.** Single-commodity, single-currency throughout
(no FX — the LICHTERS deferral stands; a commodity priced in a foreign currency is the same FX gap).

- **T60 — the commodity forward (cost-of-carry + convenience yield) + the commodity option** (ch 2.1–2.2,
  2.5). `F(T) = S·exp((r − δ)·T)` with convenience yield δ; a `CommodityForward`/option product (L2) and an
  engine that **reuses `black_price`** (L4). **Oracles:** the forward transcribed pointwise from ch 2 in two
  renderings (§5, from source not this plan); the **reduction δ=0 ⇒ pure cost-of-carry** `S·e^{rT}` (exact);
  the **single-fixing / European reduction to `black_price`** (exact, the companion that stops the reduction
  being vacuous); put-call parity as a requirement oracle. Mostly reuse — the lightest tranche, deliberately
  first.
- **T61 — geometric Asian options, exact closed form** (ch 2.7.1 continuous, 2.7.3 Kemna–Vorst discrete).
  The geometric average of lognormals is lognormal, so the price is **exact Black-76 on an adjusted
  forward+vol**. L2 product + L4 engine. **Oracles:** the adjusted-forward and adjusted-variance formulae
  transcribed pointwise; the **reduction to a single fixing ⇒ the T60 European** (exact); an **MC companion**
  on the geometric average (the reduction degenerates the averaging, T39 — the companion must exercise a
  genuine multi-fixing average, banded per §4). Closed-form, strong.
- **T62 — arithmetic Asian, Turnbull–Wakeman** (ch 2.7.2 continuous, 2.7.4 discrete). A **moment-matching
  approximation** (match the first two moments of the arithmetic average to a lognormal, price with Black-76).
  L4 engine on the T61 product family. **Oracles:** the first two moments transcribed pointwise (these are
  exact — the *approximation* is only the lognormal assumption); an **MC companion** with the band split
  `k·SE + measured TW error` and the resolution stated (§4 — never a flat percentage); the **geometric ≤
  arithmetic** ordering as a requirement oracle (AM–GM). The approximation is **not** credited as exact —
  the MC companion measures its error.
- **T63 — spread options: Margrabe (exact) + Kirk (approx)** (ch 2.9.1–2.9.2). Margrabe exchange option is
  **exact** (zero-strike spread, 1-D Black on the combined vol `√(σ1²+σ2²−2ρσ1σ2)`); Kirk is the finite-strike
  **approximation** (same combined-vol trick around a shifted forward). **Both are 1-D on a combined vol, so
  both reuse `black_price` — no bivariate normal.** L2 two-asset product + L4 engine. **Oracles:** Margrabe
  transcribed pointwise; the **reduction Kirk → Margrabe as strike → 0** (exact); an **MC companion** for
  Kirk's finite-strike error (2-D correlated GBM, banded); correlation pinned by the realised covariance of
  the simulated increments (the T17 pin, since ρ enters the combined vol as a product — assert the product's
  value, T43/T17). The ρ=±1 degenerate reductions (perfectly correlated ⇒ deterministic spread) as
  requirement oracles.
- **T64 — Schwartz one-factor mean-reverting model** (ch 2.10.1). OU log-spot
  `d ln S = κ(θ − ln S)dt + σ dW`; closed-form futures `F(t,T)` and a European option under the model. L3
  model. **Oracles:** the closed-form futures and option-vol transcribed pointwise; the **reduction κ→0 ⇒
  the GBM/Black-76 limit** of T60 (exact, with a **companion at genuine κ>0** — the reduction kills exactly
  the mean-reversion, T39, so the companion must be mean-reverting); the **MC companion reusing
  `simulate_lgm_states`** (the OU simulator — the Schwartz state is the same driftless-Gaussian object after
  the deterministic drift is removed), banded per §4; the stationary-variance `σ²/(2κ)` limit as a
  requirement oracle. The capstone — biggest reuse, the model the earlier tranches' forwards feed.

## Deferred at the pull gate (to `to_go_deeper`, ruled per-tranche)
- **American options — Barone-Adesi-Whaley + lattice** (ch 2.6): oracle-able, but its natural anchor is a
  binomial lattice the tree would build only for it; not pulled by the closed-form spine.
- **Multi-factor and stochastic convenience-yield models** (ch 2.10.2–2.10.3): Gibson–Schwartz two-factor,
  etc. — pulled only if a product needs them.
- **Commodity swaptions** (ch 2.8), **calendar/Asian spread options** (ch 2.9.3–2.9.4): closed-form-adjacent,
  deferred unless pulled.
- **Market-specific chapters 3–9** (precious/base metals, energy, power, agriculturals, alternative): vol
  surfaces from market quotes, seasonality, electricity spike models, weather/freight. Mostly
  `to_go_deeper` (conventions) or **`out_of_method`** (quotes-and-plots, structural/statistical — the
  electricity reduced-form/structural spike models and weather derivatives have no closed-form oracle).

## After the arc — the "review" Bernardo flagged
Bernardo wants to "think what we could review after" this spine. Settled at the checkpoint, not now; the
candidates to weigh then: **re-audit #7** (the §9 cadence — ~11 tranches since #6 would land around T64/T65,
and the arc close fires it regardless); **a deep-dive into a pending `to_go_deeper` subject** (the
breadth-vs-depth question Bernardo raised — e.g. the FX layer that unblocks cross-currency collateralisation
and foreign-commodity pricing at once, or the American/lattice infrastructure); or a **cross-source review**
now that CLARK gives a second Black-76 and a second OU short rate against the tree's own. Noted here so the
boundary conversation has its options in hand.

## Status — CLOSED (v0.66.0, 2026-11-20)
- **Spine complete:** T60 forward+Black-76 (v0.62.0) → T61 geometric Asian (v0.63.0) → T62 arithmetic
  Asian/TW (v0.64.0) → T63 spread options Margrabe+Kirk (v0.65.0) → T64 Schwartz mean-reverting (v0.66.0,
  the capstone). All five merged.
- `CLARK-2014` is `worked-from` (built/oracled at T60).
- Checkpoint: `docs/checkpoints/CP_commodity_options.md`. Arc close triggered **re-audit #7**
  (`docs/audit/reaudit_07_findings.md`).
- One charter advance: §4's obligation-not-form clause (earned T61/T62/T63).
- Deferrals in `to_go_deeper.md` (commodity section): partly-fixed Asians, American/lattice, two-factor /
  stochastic-convenience models, calendar/Asian spreads, the Schwartz `α̂_X` calibration, seasonality.
- **Next: arc boundary** — Bernardo's interest (a new reference is likely) + a Cowork spec.
