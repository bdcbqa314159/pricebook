# T79 — perpetual timer options (the vol-derivatives arc capstone)

**Cowork spec, 2026-11-24.** Sixth and final tranche of the vol-derivatives arc (project ten, `KWOK-ZHENG`).
The capstone: a **perpetual timer call** — exercised when accumulated variance `∫₀^τ V_s ds` first hits a
**variance budget B** (a random exercise time `τ_B`), a *first-passage-on-variance* exotic. It closes the arc
by using the Heston transform machinery (T48) at a random, variance-triggered horizon rather than a fixed one.

## Sourcing (pinned from the TOC)
Read in `-raw` + `-layout` and reconcile (§4):
- **§1.3.3** — the timer-option payoff definition.
- **§6.2 — pricing perpetual timer options**: **§6.2.1** the conditional-expectation Black–Scholes-type
  formula (Theorem 10, **Bernard & Cui, 2011** — we cite the held book, not the un-held paper, §7), **§6.2.2**
  the integral price formulas under Heston (the Bessel-process-with-drift characterization of `(V_τB, τB)`),
  §6.2.3 the perturbation approximation (secondary — a cross-check if cheap, else a `to_go_deeper` note).
- **§6.3 finite-maturity discrete timer options is OUT OF SCOPE** (PDE/numerical — already a `to_go_deeper`
  route in the arc plan; the perpetual case is the closed form).

## The structure (why it is a clean capstone)
At the first-passage time `τ_B`, `ln(S_τB/S0) ∼ N(µ_τB, (1−ρ²)B)` — the variance is **fixed at the budget B**
(scaled by `1−ρ²`), and all the model-dependence is in the **random drift** `µ_τB` (eq 6.14, a function of
`τ_B`, `V_τB`, and `∫g(V)dt`). So the timer call is a **Black–Scholes-type formula with total variance
`(1−ρ²)B`** integrated over the joint law of `(τ_B, V_τB, …)`, which under Heston (§6.2.2) is an **integral
transform** off the Bessel-process characterization. That is the T48 machinery at a variance-triggered horizon.

## Pieces, layer, reuse
- **The product** (new, L2) — a `TimerOption` (or `PerpetualTimerCall`) frozen dataclass: underlying, strike,
  **variance budget B**, option kind. ≤5 fields (§7). Perpetual, so no fixed expiry field.
- **The engine** (new, L4) — `@register(TimerOption, HestonModel)`, the §6.2.1/6.2.2 integral formula:
  the Black–Scholes-type core (**reuse `black_price`** with total variance `(1−ρ²)B`) integrated over the
  first-passage law, which reuses **`integrated_variance_transform`** (T48) and the **`noncentral_chisquare`
  / Bessel** machinery (T72/T77). Projects to a `PricingResult` (§3).
- **The MC companion** — simulate Heston paths, accumulate `∫V dt` to the budget B, record the first-passage
  time `τ_B` and `S_τB`, price `e^{−rτ_B}(S_τB − K)⁺`. Establish the simulator's own first-passage accuracy
  first (the "a reference is code too" rule — the hitting-time discretisation has its own bias; measure it).

## The step-1 layer Ask (expect it may fire this time)
§6.2.2's Bessel-process characterization may put a **Bessel function `I_ν`** on the production path (the
first-passage density of a Bessel process). The `noncentral_chisquare` machinery already relates to `I_ν`,
so it **may** reduce to the existing ladder — **confirm at step 1**. If a genuine `I_ν` is needed in `src/`,
raise the **T20a layer `Ask:`** (scipy oracle-only; build our own `I_ν` at L0, oracled against scipy) before
building — the T77 pattern. Report which at step 1.

## Oracles (RED-first, §4) — name the obligation, measure the form (the arc's repeated spec-miss is mine; §4)
1. **The §6.2.2 integral price formula transcribed pointwise** (the load-bearing new formula; two renderings,
   §4). Where §6.2 derives it in steps (Theorem 10 → the integral form), use the earlier step as the
   independent check (T12(b)).
2. **The exact reduction — pin the limit by DERIVING it, not by guessing its value.** The clean candidate:
   at **`ρ = 0`** the variance locks to `(1−0)B = B` and the spot/variance decouple, so the timer call
   collapses toward a Black–Scholes call with total variance **exactly B** (the budget *is* the variance) —
   **derive the exact `ρ=0` (and, if the drift's `τ_B`-dependence does not vanish, `r=q=0`) limit and assert
   it**, rather than asserting a value up front. Companion with **genuine `ρ≠0`** (T39 — the reduction zeroes
   the correlation the first-passage drift lives in).
3. **The MC companion** measures the integral formula, banded `k·SE + measured error`, resolution stated
   (§4), swept where the timer structure is resolvable above the SE. **Put–call parity** for timer options.
4. **The §6.2.3 perturbation approximation as an independent cross-check** (if cheap) — its measured error
   against the integral formula and the MC (the T62 measured-error-is-the-deliverable pattern); else a
   `to_go_deeper` note.
5. **Requirement oracles:** price `> 0`; **monotone in the budget B** (more variance budget ⇒ more optional
   value, the finance forces it); the `ρ→0` BS-variance-`B` identity as a requirement, swept non-degenerate.

## Degeneracy premises (§4 / §9)
- The `ρ=0` reduction's companion is **non-degenerate in `ρ`** (a genuine correlation, T39) — state it.
- The budget-monotonicity / first-passage fixtures use a **genuine stochastic variance** (`σ_v > 0`, `v₀ ≠
  θ`) so the first-passage law is non-trivial (a deterministic variance hits the budget at a fixed time — the
  timer degenerates to a fixed-horizon option).
- The MC first-passage estimator's **own bias is measured first** (the hitting-time discretisation; "a
  reference is code too").
- Any `κ→0`-style numerics watch is scoped to **references, not production** (T77/T78 input).

## Provenance / gates
- Header `source: KWOK-ZHENG §1.3.3 + §6.2.1/§6.2.2 (Bernard–Cui 2011, via the book)`; `oracle:`;
  `crosscheck:` (check QuantLib for a Heston timer engine before asserting `none`).
- `references.md`: `KWOK-ZHENG` already `worked-from`; no flip. If an `I_ν` primitive lands, its own
  provenance/oracle follows (the T20a arrangement).
- Version → **v0.81.0**; one floor bump; one `CHANGELOG` entry. **Re-time the Heston first-passage MC on BOTH
  CI runners (§9)** — the arc's heavy-numerics family; a lossless pass before any OI-8-style exemption.

## Out of scope (arc closes after this)
- **§6.3 finite-maturity discrete timer options** — PDE/numerical, `to_go_deeper` (the arc plan's deferral).
- **The 3/2-model timer (perturbation, §6.2.3 for 3/2)** — `to_go_deeper`; T77 built the 3/2 variance swap,
  the timer under it is a separate method.
- **Timer swaps / timer out-performance options (§1.3)** — a product-pull away, `to_go_deeper`.
- After T79: the arc **checkpoint** (`CP_vol_derivatives.md`) + **re-audit #9** (its duplication pass picks
  up C17 / `implied_carry`).
