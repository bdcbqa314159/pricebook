# T64 — the Schwartz one-factor mean-reverting model (the commodity arc capstone)

**Cowork spec, 2026-11-19.** Fifth and last spine tranche of the commodity arc (project eight,
`CLARK-2014`). The model the earlier tranches' forwards were waiting for: a **mean-reverting** spot
(Schwartz one-factor), which gives a commodity the term structure of volatility a flat Black vol cannot —
the **Samuelson effect**. It is the arc's biggest reuse (the OU state **is** the LGM state) and its sharpest
reduction (κ → 0 kills exactly the mean reversion the model exists for), and it is where
`CommodityBlackModel`'s own docstring prophecy — "a commodity's volatility is on a path to diverge from an
equity's within this very arc" — comes true.

## Target
Price a European option on a commodity under the Schwartz one-factor model, with the model's closed-form
**futures price** `F(t,T)` and maturity-dependent **option variance** — `price(CommodityOption,
SchwartzModel) → PricingResult`. The product is T60's `CommodityOption` (reused); the model is new.

## Pieces, layer, reuse
- **The model** (new, L3) — `SchwartzModel` carrying the `MarketSnapshot` (§3) and the three OU parameters:
  mean-reversion speed **κ**, long-run log-level **θ**, volatility **σ**. Bundle `(κ, θ, σ)` into a frozen
  value object if market + three pushes the field count (§7 — `HullWhiteParams` is the precedent, and
  `SchwartzParams` may *be* `HullWhiteParams` + θ or its own type; Code rules the shape by whether the
  grouping is the same idea). **Distinct from `CommodityBlackModel`** — mean reversion is a different idea,
  the divergence that docstring already names, so no merge (the §6/T60 call made in advance).
- **The engine** (new, L4) — assembles the Schwartz **futures** `F(t,T)` and the **option variance**
  `σ²_opt(T)`, then **reuses `black_price`** (the log-spot is Gaussian, so the option is Black on the model's
  own forward and variance). Reuses T60's discount handling. The genuinely-new closed-form code is `F(t,T)`
  and `σ²_opt(T)`.
- **The MC companion reuses `simulate_lgm_states`** (`models/lgm.py`) — the Schwartz log-spot
  `d ln S = κ(θ − ln S)dt + σ dW` is an Ornstein–Uhlenbeck process, which **is** the LGM state after its
  deterministic drift is accounted for. This is the capstone reuse: the same driftless-Gaussian simulator
  that anchored T57's LGM anchors the Schwartz MC, exactly as LGM reduced to Hull-White.

## The mathematics (re-transcribe from the source — NOT this spec, §5)
Read **Clark ch 2.10.1** in `-raw` and `-layout` and reconcile (§4). The load-bearing closed forms are the
futures price `F(t,T)` (an `e^{−κ(T−t)}`-weighted blend of the current log-spot and the risk-adjusted
long-run level, plus a variance term) and the option variance `σ²_opt(T)`, which carries the
`(1 − e^{−2κ(T−t)})/(2κ)` factor — the term a mutation will most plausibly corrupt and the one to transcribe
pointwise. The risk-neutral drift adjustment (the market price of risk / the `θ*` vs `θ` distinction) is a
modelling point to read carefully from the source and state in the docstring, as T60 did for the convenience
yield; do not fold it silently into θ.

## Oracles (RED-first, §4 — obligations, not forms)
Per the new §4 clause, each oracle below names **what to pin or measure**, not the shape the answer takes.

1. **Pointwise transcription of `F(t,T)` and `σ²_opt(T)`** — the test forms its own `e^{−κ(T−t)}` and
   `(1−e^{−2κT})/(2κ)` independently, not read back from the engine. The two new load-bearing formulae;
   §4 requires each transcribed pointwise at least once (the F-6.1 lesson).
2. **Reduction: κ → 0 ⇒ the T60 GBM / Black-76 limit — the sharpest T39 instance of the arc.** As κ → 0 the
   OU log-spot becomes arithmetic Brownian (GBM in S), `σ²_opt(T) → σ²T`, and `F(t,T) →` the cost-of-carry
   forward — so the Schwartz price → the T60 commodity-option price. **Exact limit** (expect the T61
   `log`/`exp` ULP gap, so band it — equal in ℝ, not `==`). **The companion is mandatory and must exercise
   genuine κ > 0** (§4/T39): the reduction degenerates *exactly the mean reversion the model exists for*, so
   it is structurally blind to the whole model — a κ > 0 case checked against the MC companion is what makes
   the anchor non-vacuous. State the premise that the companion's κ is materially non-zero.
3. **The MC companion (reusing `simulate_lgm_states`) prices the option under genuine mean reversion**,
   banded per §4 (`k·SE`, since the closed form is exact; resolution stated, higher-path power test). This
   is the oracle that sees what κ → 0 cannot. **Prove the simulated OU process reproduces the model's own
   mean and variance first** (the realised-moment pin, the T17/T19 unbiased-before-credited discipline) —
   the Schwartz state's conditional mean `e^{−κΔ}ln S + (1−e^{−κΔ})θ` and variance `σ²(1−e^{−2κΔ})/(2κ)` are
   both closed form, so pin both against the simulated increments before crediting the price band.
4. **Requirement oracle — the stationary variance.** `σ²_opt(T)` is **measured** across a maturity sweep
   and checked against the closed form; its **limit** as T → ∞ is `σ²/(2κ)` (the variance saturates — the
   structural fact that distinguishes a mean-reverting commodity from GBM, whose variance grows without
   bound). Assert the measured term structure matches the closed form and the finite limit holds; **do not**
   pre-assert a monotonicity shape — measure the term structure and let the closed form supply its shape
   (the new §4 clause, applied to this spec: pin the obligation, report the form). The Samuelson effect
   (near-dated futures more volatile than far-dated) is then a *reported consequence*, not an asserted form.
5. **Requirement oracle — put–call parity** on the model's futures forward `F(t,T)`, swept off the money.
6. **Cross-check — the OU state against the LGM.** Since the MC reuses `simulate_lgm_states`, assert that at
   matched parameters the simulated Schwartz state and the LGM state agree (same simulator, a consistency
   guard, **not** independence — the T17 same-object caveat; flag it as such).

## Degeneracy premises (§4 / §9)
- The κ → 0 companion is **non-degenerate in κ** (T39) — state it; the capstone's whole point is the
  parameter the reduction zeroes, so a companion with near-zero κ would be the reduction in disguise.
- **Distinct, well-spread maturities** for the vol-term-structure oracle — a single maturity cannot see a
  term structure, and the stationary limit needs large T.
- Prefer a **mutation to a reading**: drop the `e^{−κ(T−t)}` blend weight, flip the sign of κ, drop the
  `1/(2κ)` or the `(1−e^{−2κT})` in the variance, swap θ for the current spot. A survivor is a hole.
- The realised-moment pin needs enough paths that a drift/variance bias is resolvable above the MC SE (the
  T17 count discipline).

## Provenance / gates
- `source: CLARK-2014 §2.10.1`; `crosscheck:` (QuantLib has Schwartz / one-factor mean-reverting commodity
  engines — a real target); `oracle:`.
- Version → **v0.66.0**; one floor bump; one `CHANGELOG` entry (lead with the measured vol term structure and
  the stationary limit — the Samuelson effect is the beat).
- This is the arc's last spine tranche: its merge triggers the **checkpoint** (`CP_commodity_options.md`) and
  **re-audit #7** (§9 — arc close, and the cadence lands here).

## Out of scope (carried / deferred → to_go_deeper)
- **Two-factor / stochastic convenience-yield models** (Gibson–Schwartz, ch 2.10.2–2.10.3) — the one-factor
  model is the spine; the second factor is pulled only if a product needs it.
- **Seasonality / deseasonalised forward curves** (ch 5.2.1) and the market-specific calibration (ch 3–9) —
  `to_go_deeper`/`out_of_method` (quotes-and-plots).
- **American options under Schwartz** (needs the lattice already deferred at the arc plan) — `to_go_deeper`.
