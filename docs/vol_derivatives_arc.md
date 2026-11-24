# Arc plan — volatility & variance derivatives (project ten, `KWOK-ZHENG`)

**Opened 2026-11-22 at v0.75.0.** The tenth vertical: the **products on** variance and volatility — the
exotic variance swaps, options on realized variance, VIX derivatives, and timer options — extending the
variance/vol-swap base and the Heston Fourier machinery project six left. Distinct from `GATHERAL-2006` (the
vol *surface*); this is `KWOK-ZHENG`, the vol *derivatives*.

Scope ruled by Bernardo, 2026-11-22: **the bigger spine (~6 tranches)** — the core four plus a new
stochastic-vol model (3/2) and the time-changed-Lévy framework with the discrete-vs-continuous convergence.

## Why the reuse makes it affordable (step-1 greps, 2026-11-22)
- **`engine/volatility_swap.py::integrated_variance_transform(HestonParams, ψ, years)`** — T48's Heston
  **Laplace transform of integrated variance** — is the single most load-bearing reuse: the fair variance/vol
  under Heston, the VIX² (a forward variance-swap rate), the conditional law of realized variance (options on
  it), and the timer option's first-passage law all come off this transform. Plus `fair_volatility`,
  `convexity_adjustment` (T48).
- **`engine/variance_replication.py::fair_variance`** — T31's `1/K²`-weighted strip / log-contract — is the
  base the **generalized** swaps re-weight (gamma = `1/K`, corridor = `1/K²` on a restricted strike range).
- **`models/heston.py` (`HestonParams`, the char function)** and the Fourier inversion (T33/T49) — the
  pricing engine for VIX options and options on realized variance.
- **The T62 moment-approximation + MC-companion discipline** — options on discrete realized variance are a
  moment match to a lognormal/normal with the MC measuring the error, exactly the Turnbull–Wakeman pattern.
- `heston_mc.py` — the MC companion for every realized-variance payoff.

## The spine (queue)
**T74 → T75 → T76 → T77 → T78 → T79 → checkpoint → re-audit #9.** Single-underlying throughout.

- **T74 — generalized variance swaps: gamma + corridor** (§1.3 payoffs, **§1.4 the static replication** —
  Prop 1.2 / eqs 1.24–1.26; NOT ch 4.3, which is the *moment-generating-function* model route T75/T76 use —
  a Cowork sourcing correction at build, recorded in `references.md`). The **gamma swap** weights realized
  variance by `S_t/S_0` (replicated by a `1/K` strip); the **corridor** variance swap accrues variance only
  while `S` is in `[L,U]` (the `1/K²` strip restricted to the corridor). L2 products + L4 engine **reusing**
  `variance_replication`. **Oracles:** the two weightings transcribed pointwise from ch 4.3; the **reduction
  corridor → vanilla variance swap** at `[0,∞)` (exact — the restricted strip becomes the full strip), with a
  finite-corridor companion (T39); `corridor ≤ vanilla` and the gamma/vanilla relation as requirement
  oracles; an MC companion (realized gamma/corridor variance from simulated paths). The lightest, first.
- **T75 — options on *continuous* realized variance under Heston** (ch 4.2.1, 5). *(Built & merged
  v0.77.0, PR #96; ruled **continuous** at 0401 — the spec's "discrete" target contradicted its own
  out-of-scope and §5.2; the discrete correction is T78's, below.)* The fair variance under Heston
  (closed form off `integrated_variance_transform`, **reuse T48**), and **options on realized variance** by
  the **normal/moment approximation** (ch 5.2) with the MC companion measuring the error (the T62 pattern —
  ~8% near the money, the honest cost of the normal assumption and the argument for §5.3). **The cross-method
  anchor T74 lacked landed**: Heston fair variance in closed form == T74's `fair_variance` strip fed Heston's
  implied vols (convergence-from-below asserted, not two cancelling errors). The Heston smile was presented to
  T31's strip by **subclassing `ImpliedVolSkew` and overriding `volatility`** — acceptable for one consumer;
  **T76 extracts the `SmileLike` protocol** (see the arc note below). **Oracles:** the Heston fair-variance
  pointwise (cross-check vs T48's transform, and the second moment derived-as-theorem per T19 since the
  printed one was jump-extended + reflow-mangled); the moment approximation banded `k·SE + measured error`,
  resolution stated; `σ√T`-swept honest-negative; put–call parity.
- **T76 — VIX derivatives** *(Built & merged target v0.78.0, PR #97; sourcing corrected at 0404.)*
  **VIX² is the variance-swap rate** — the identity is **§1.5.2 / eq (1.31)** (NOT §3.1.1, which is
  Carr–Wu's *jumps* correction showing how VIX² and the variance-swap rate **differ** under jumps —
  out of scope since Heston is pure-diffusion; the T74 §1.4-vs-ch-4.3 sourcing shape again, a Cowork
  spec miss caught at build). The futures/options are **§3.1.2** (eqs 3.6/3.7a), the affine pricing
  **§3.2.1**. Because eq (1.31) **is** T31's strip verbatim, the tranche collapses into reuse:
  `vix_squared == heston_fair_variance`, model-free VIX² `== fair_variance`, with `VIX²_t`
  **spot-starting** at `t` over `[t,t+τ]` (no forward-window arrangement — the affine
  `a=(1−e^{−κτ})/(κτ)`, `b=θ(1−a)` fall out of T75's formula at 4.4e-16). VIX future/option as 1-D
  integrals against the **non-central χ²** law of `V_T` (T72's primitive), reduced by integration-by-
  parts to one integrand against the **survival** function (F-8.6's deep-tail work paying off). L2
  products + L4 engine; **`SmileLike` protocol extracted at L1** (ruling 0401(2)). **Oracles:** the
  VIX²/variance-swap-rate identity (a requirement oracle); the cross-method strip-vs-closed-form anchor
  (T75 oracle-1 shape); VIX future `E[VIX_T]` + convexity `E[VIX] ≤ √E[VIX²]` (Jensen, requirement);
  `σ_v→0` reductions; MC companion. **First `slow-test-exempt` of the arc (OI-8, accepted 0404).**
- **T77 — the 3/2-model variance swap** (ch 4.2.2). A **new stochastic-vol model** (`dv = κv(θ−v)dt +
  σv^{3/2}dW`), under which the variance swap has a closed form. L3 model. **Oracles:** the 3/2 variance-swap
  closed form transcribed; the **reduction to a known limit** (the 3/2 model's relation to Heston/CIR under
  a parameter map — ruled at build whether exact); an MC companion; the `α→0`-type numerics watch (the T66
  rate-in-a-regime clause, if a removable singularity appears).
- **T78 — time-changed Lévy variance swaps + the discrete-monitoring correction (a PRICING tranche)**
  (§2.4 framework; §4.4.1/§4.4.2/§4.4.4–5 pricing + convergence; §1.5.3 model-free discrete cross-check —
  sourcing pinned from the TOC at T78 spec, the arc plan's earlier "4.3.3" was wrong). *(Scope set at 0401:
  T75 priced the continuous case, so T78 **carries the
  discrete-monitoring adjustment as a priced deliverable** — it prices a discretely-monitored realized-variance
  claim as "the continuous counterpart **plus** the §5.2 adjustment", not only a convergence study.)* The
  variance swap under a time-changed Lévy process (a general framework), and the **discrete-monitoring
  correction** — the discretely-sampled realized variance converges to the continuous one at a measurable
  **rate** (the T25 rate-detector, measured not plotted), which is how the adjustment is *oracled*. L3.
  **Oracles:** the Lévy variance-swap transcribed; the discrete price = continuous + adjustment transcribed
  pointwise; the discrete→continuous convergence **rate** (O(1/n)) as the adjustment's own check; the
  reduction to the continuous variance swap (T31) as `n→∞`.
- **T79 — timer options** (ch 6), the capstone. A perpetual timer call exercises when accumulated variance
  hits a budget — a **first-passage-on-variance** exotic. L4 engine off the Heston transform. **Oracles:** the
  perpetual timer closed form transcribed (ch 6.2); the **reduction** to a vanilla as the variance budget →
  the option's fixed-horizon variance (ruled at build); an MC companion (first-passage time of accumulated
  variance); put–call parity.

## Arc-wide lens — the replication grid's TRUNCATION is per-weight, measured not inherited (T74)
A replication/quadrature grid measured adequate for one weight is **truncation-blind for a slower-decaying
one.** T74's gamma strip (`1/K`) decays one power of `K` more slowly than the vanilla (`1/K²`), so cutting the
upper tail at `5·F` costs it far more — the flat-skew identity (gamma = vanilla) held only to 2.98e-05 on
T31's grid. The diagnosis is the T25/T20a technique (vary the knob to localise the error): the gap was
**invariant in node count** (2.98e-05 at 1601 *and* 6401 nodes) and **collapsed with width** (8.8e-09 on
`[0.02,10]`, 1e-10 on `[0.01,20]`) — so it is **truncation, not discretisation**, and the fix is a wider grid,
not more nodes or a band. **T76 (VIX) and T79 (timer) integrate different weights off the transform and must
measure their own grid adequacy per weight — do not inherit T31's or T74's.** (A lens, the "a reference is
code too, in the regime where it is used" clause specialised to quadrature truncation — not a new clause.)
Keep the corridor's contract bounds `[L,U]` **separate** from the grid's truncation bounds (T74): conflating
them would make a wider grid change the payoff.

## Arc note — the `SmileLike` protocol is extracted at T76 (§6 rule-of-two, ruled 0401)
T75 fed Heston's smile to T31's replication strip by **subclassing `ImpliedVolSkew`** and overriding
`volatility(strike, forward, years)` — correct for one consumer, but subclassing a log-linear closed form only
to discard its log-linearity is a mild smell. T76 (VIX) is the genuine **second** consumer of "present a
model's smile to the strip", so **rule-of-two fires at T76**: extract a minimal `@runtime_checkable SmileLike`
protocol — the single real member `volatility(strike, forward, years)` (memberful, per T33b — an empty
protocol is satisfied by everything and could not discriminate) — that `fair_variance` /
`generalized_fair_variance` consume, with `ImpliedVolSkew` and the model-smile presenters both implementing
it, collapsing T75's Heston subclass into a presenter in the same change (the C16 raise-at-review pattern).
Not built speculatively before T76; decided now so T76's spec carries it.

## Deferred at the pull gate (to `to_go_deeper`, ruled per-tranche)
- **Finite-maturity (non-perpetual) timer options** — the perpetual case (T79) is the closed form; the
  finite-maturity timer is PDE/numerical, a `to_go_deeper` route.
- **Options on VIX under direct VIX models** (ch 3.3) — the "direct modelling of VIX" school (vs the
  consistent/variance-swap-rate approach T76 takes); deferred unless pulled.
- **The partially-exact-and-bounded approximation** (ch 5.3) — a sharper options-on-variance approximation
  than the normal one; a `to_go_deeper` refinement on T75 if the normal approximation's error is too wide.

## Status
- **T74 merged** (v0.76.0); **T75 merged** (v0.77.0); **T76 merged** (v0.78.0, PR #97 — ruled 0404;
  first arc `slow-test-exempt`, OI-8).
- **T77 merged** (v0.79.0, PR #98 — ruled 0411: oracle 5's `→ θ` corrected to the stationary mean
  `θ/(1+ε²/(2κ))`, and `stationary_variance` ruled test-only under §6).
- **T78 merged target** (v0.80.0, PR #99 — ruled 0419: oracle-2 anchor corrected to `(r−q−σ²/2)²T/N`, Prop
  4.8 the arc's fourth source defect (`var(L₁)`, not the printed second moment), C17 `implied_carry` → #9).
- **T79 specced** (`docs/tranches/79_timer_options.md`, the capstone). Queue **T79**.
- `KWOK-ZHENG` is `worked-from` (flipped at T74).
- After T79: the checkpoint (`CP_vol_derivatives.md`) + re-audit #9 (its duplication pass picks up C17).
