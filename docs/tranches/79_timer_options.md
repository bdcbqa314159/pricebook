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
- **The engine** (new, L4) — `@register(TimerOption, HestonModel)`, the **MC first-passage pricer**: simulate
  Heston paths, accumulate `∫V dt` to the budget `B`, record the first-passage time `τ_B` and `S_τB`, price
  `e^{−rτ_B}(S_τB − K)⁺`. Returns a `PricingResult` with SE (§3, the Bermudan shape). Establish the
  simulator's own first-passage accuracy first (the "a reference is code too" rule — the hitting-time
  discretisation has its own bias; measure it). **This is the production pricer** — the §6.2.2 closed form is
  deferred (see below).

> **SCOPE RULED (C), 0424 — the §6.2.2 closed form is NOT this tranche.** Step-1 (0422) found §6.2.2 needs
> three production primitives the single-underlying arcs never required: a **complex-argument Bessel `I_ν`**
> (the Bromwich contour makes `λ` complex, so the real-argument `noncentral_chisquare` ladder cannot supply
> it — the "may reduce to the ladder" hope is false for a structural reason), a **Bromwich/Laplace
> density-inversion** (distinct from the Gil-Pelaez CDF inverter), and a **2-D `(x,t)` quadrature**. These
> belong to the **structural/numerical track** (Pagès/Seydel), which §6.2.2 becomes the first concrete pull
> for (`to_go_deeper`, recorded 0424) — come-back-same-method debt, NOT out-of-method, and it retro-anchors
> this MC when built (the T74←T75 pattern). Eq (6.23)'s transcription is settled (the `√(2λ)`-alone reading,
> pinned by the density-integrates-to-1 requirement, 1.0000 vs 0.51–0.90 — T79 step-1), ready for then.

## Oracles (RED-first, §4) — MC-not-sole, two independent non-MC pins (the T72 in-method lens)
1. **The exact reduction at `ρ=0, r=q=0` ⇒ vanilla Black–Scholes at total variance `B`** (eq 6.10;
   **reuse `black_price`**). The `τ_B`-dependence of the drift `µ_τB = (r−q)τ_B − B/2` does **not** vanish at
   `ρ=0` (confirmed 0422), so `r=q=0` **is** required — at `ρ=0,r=q=0` the drift is the deterministic `−B/2`
   and eq (6.10) is exactly `S₀N(d₁) − KN(d₂)` at variance `B`. Companion with **genuine `ρ, r`** (T39 — the
   reduction is blind to `ρ`).
2. **The §6.2.3 perturbation as a small-vol-of-vol cross-check** (the second independent, non-degenerate-in-`ρ`
   pin — it is what lifts the tranche clearly above the out-of-method line). Band the MC against it
   `k·SE + the perturbation's MEASURED error`, in the perturbation's **validity regime** (small `σ_v`;
   **confirm the regime at build**, the T25/T17 "a reference is code too, where it is used" discipline). Used
   as an *oracle for the MC*, never as the quoted price.
3. **Self-consistency:** put–call parity for timer options; **monotone in the budget `B`** (more variance
   budget ⇒ more optional value — the finance forces it), swept non-degenerate.
4. **MC self-convergence** — the SE falls as `1/√N`; the price is stable under path count at the committed
   resolution.

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
