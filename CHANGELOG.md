# Changelog

All notable changes are recorded here, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. `pricebook.__version__` is the
single source of truth and CI asserts it matches the top entry.

## [Unreleased]

## [0.80.0] - 2026-11-23

### Added — tranche 78, variance swaps under time-changed Lévy processes (the arc's jump pillar, and the discrete-monitoring adjustment as a priced deliverable)

- **`models/time_changed_levy.py` (L3)** — `LevyParams` (a diffusion part plus `Y = 0` CGMY /
  **Variance Gamma** jumps), `ActivityRateParams` (the **absolutely continuous** CIR clock), and
  `TimeChangedLevyModel`. The clock's long-run level is **1 by the `E[Z_t] = t` normalisation**, not
  a field — eq (4.81) pins it, so a `theta` would be a fifth wheel.
- **`products/discrete_variance_swap.py` (L2)** — `DiscreteVarianceSwap`, carrying the monitoring
  count. A **separate type**, not a `None`-count variant of T31's swap (§6): the count is
  constitutive of the payoff, which is the whole subject here.
- **`engine/time_changed_levy.py` (L4)** — the continuous strike (Prop 4.2), the discrete
  adjustment (Prop 4.3), the multiplier (Table 4.3), and two `@register`ed pricers. **No Fourier
  inversion and no special function:** everything is a derivative of the cumulant exponent at 0 or
  its value at 1.

#### Two readings of the source had to be ruled, both pinned by oracles

**§4.4 contradicts itself.** Prop 4.8 prints `Kv(∞) = −(1/T)φ_L⁽²⁾(0)E[Z_T]`, which under the book's
own (4.84) is the **second moment** `E[L₁²]`; Prop 4.2 with Prop 4.1 gives `var(L₁)`. They differ by
`(E[L₁])²` — measured **20%**, **5%** and **24,245%** apart on three parameter sets, and agreeing
only for a centred `L`, which is why an ordinary fixture would not have noticed. **We compute
`var(L₁)`**: a fair strike is expected quadratic variation, `[L,L]` is **drift-free**, and Prop 4.1's
own proof states `E[L,L]₁ = var(L₁)`. Props 4.9/4.10 use `ψ_L″` rather than `φ`, which makes `φ⁽²⁾`
the misprint. The §4 derivation-step-discrepancy rule, and the arc's **fourth** source defect.

**Table 4.3's CGMY row prints no `Y` and no `Γ`** while its footnote permits `Y < 2`. Derived against
the Lévy density it is the `Y = 0` case **exactly** (machine precision, three parameter sets) and
2.8% off at `Y = 0.5`. The formula is right; the row's generality claim is not.

#### The spec's structural anchor was false, and the replacement is stronger

The spec made *"the adjustment vanishes in the pure-diffusion limit"* the heart of the tranche. **It
does not vanish.** Prop 4.3's first correction term `(1/T)Σ(E[Δln S])²` survives the vanishing of the
jumps entirely, and the pure-diffusion adjustment is **exactly `(r − q − σ²/2)²·T/N`** — the classic
discrete-variance result, asserted at `rel=1e-14` rather than as a limit. The adjustment vanishes as
`N → ∞`, not as the jumps go. That is a **stronger** anchor than the spec's: an exact nonzero closed
form rather than a zero, which a no-op adjustment would also have satisfied (§4's vacuous-reduction
trap). It is simultaneously oracle 3's rate, since the same constant is §4.4.4's `O(1/N)` constant.

#### Three findings beyond that

- **The multiplier buys no independence.** `Q_L·(κ(1) − κ′(0)) = κ″(0)` is an **algebraic identity**,
  so computing the strike "both ways" is one computation written twice (T70). `Q_L` is built because
  Table 4.3 tabulates it — it is the transcription target — and the strike's only independent check
  is the simulation. Asserted as an identity so the oracle set is not credited with independence it
  does not have.
- **The clock's variance carries a cancellation defect in PRODUCTION, not in a reference.** The spec
  scoped the κ→0 watch to references (T77's landed there); here `var(ΔZ) = 2s²(κΔ − 1 + e^{−κΔ})/κ²`
  carries it. Written naively the bracket is **29% wrong at `x = 2.5e-8` and returns exactly 0.0** by
  `2.5e-10`. **Not a removable singularity** — `s² = σ_v²/(2κ)` diverges, so the variance genuinely
  blows up; the defect is the arithmetic. Fixed with `expm1` **plus a series branch**, because
  `x + expm1(−x)` still cancels (losing `2ε/x`); the crossover at `1e-3` is where the two errors
  meet, **measured for this model rather than inherited** (T67).
- **The clock must be an integrated rate, not a subordinator.** §4.4.4's convergence assumes
  `E[(ΔZ)²] ~ O(N⁻²)`, which an absolutely continuous clock gives and a gamma subordinator does not
  (`O(N⁻¹)`) — under which the discrete strike would **not** converge to the continuous one at all.
  Asserted as a measured rate (`var/Δ² → s²`), not as a remark.

#### Oracles

Table 4.3's CGMY row transcribed pointwise and its Brownian row as an **exact reduction** (`Q_L = 2`
at `rel=1e-15`), with a T39 companion where the multiplier exceeds 2; the pure-diffusion adjustment
against its exact closed form; the `O(1/N)` rate as `adjustment × N → constant` over `N = 50…1600`
while the adjustment itself moves 32× (the T25 rate-detector, measured not plotted); §4.4.2's printed
**positivity** requirement; and an **exact-increment** Lévy simulation — a `Y = 0` CGMY process is the
difference of two gamma processes, so the Lévy increments carry no discretisation error and only the
clock integral is sub-stepped. Simulation resolves the adjustment at **3.86 SE** with 250k paths at
`N = 4`; at 60k it is 1.92 SE and **not** resolvable, so the path count is part of the claim.

#### The mutation sweep found two holes with one root cause, and the second corrected the code

**19 mutations: 18 killed, 1 survived** — substituting the naive cancelling bracket for `expm1`.
Every `κ` the stability test probed put `x = κ·Δ` **below** the branch threshold, so the whole
`expm1` branch was unexercised. Chasing it, all three forms were measured against a **60-digit
`decimal` reference** (T20a's independent-high-precision move, with the standard library rather
than scipy), which overturned two of the build's own claims:

- the series-based reference used while diagnosing was itself the **weaker side** above a few
  thousandths, so readings there measured the reference's truncation, not the bracket — "a
  reference is code too", inside a diagnostic;
- **the crossover was in the wrong place.** The code had `1e-3` and the docstring claimed the two
  errors met there. They meet at **2e-3**; at `1e-3` the series is **70× more accurate** than the
  `expm1` it switched to (2.78e-15 against 1.95e-13) — accurate enough to pass every oracle, and
  wrong about where the trade-off lies. Corrected; worst case is now ~5e-14 across the domain.
  (`expm1` everywhere would not do either: it is **1.5e-7 wrong at `x = 1e-9`**.)

**A second sweep then surfaced a second survivor with the same root cause**: reverting the
threshold to `1e-3` survived, because the new crossover test compared the two *forms* to each other
and **never called production** — a true statement about the error curves that holds whatever
threshold the code uses, i.e. a tautology with respect to the code under test. Killed by the one
point where the forms are separable: at `x = 1e-3` they differ 70×, where at `1.5e-3` it is 2.8×
and at `1.9e-3` only 1.7×, so most candidate points would not have discriminated. Both holes are
the same shape — **an oracle that did not reach the thing that could be wrong**: first a branch no
fixture entered, then a constant no assertion touched. Final: **25 mutations, 25 killed**
(19 + 6 branch mutations), applied counts full on every run.

## [0.79.0] - 2026-11-23

### Added — tranche 77, the 3/2-model variance swap (two routes to one number, and `src/` stays scipy-free)

- **`models/three_halves.py` (L3)** — `ThreeHalvesParams` and `ThreeHalvesModel` for
  `dv = κv(θ−v)dt + εv^{3/2}dW` (eq 2.75d), non-affine, with the mean-reversion **speed scaling with
  the variance level**.
- **`engine/three_halves.py` (L4)** — the fair variance and
  `@register(VarianceSwap, ThreeHalvesModel)`. **No new product:** T31's `VarianceSwap` priced
  model-based, its third model after the model-free strip and Heston.
- **`foundation/noncentral_chisquare.poisson_weighted_sum` (L0)** — the Poisson ladder, extracted at
  its second consumer (§6).

#### The source defect, and why acquiring the paper resolved it

The book's **eq (3.30)** prints the 3/2 Laplace transform as `₁F₁(−2/σ²U; α, γ)` — **argument
first** — and read that way it **diverges**: `L(0)` is 4.32 or 4.19 depending on which rendering's
grouping of `α` you take, and the renderings disagree about that grouping. A Laplace transform of a
non-negative quantity must give `L(0) = 1`, so the printed form is not transcribable (reported at
step 1, thread 0407, as the arc's third source defect and the first I could not repair from the held
text). Eq (3.30) attributes itself to Zhao et al. (2018); Bernardo acquired it, and that paper's
**Proposition 3 / eq (19)** writes `M(α, γ, −2/(σ²y_T))` — **argument last** — whereupon `L(0) = 1`
**exactly**, since α's grouping makes `α(0) = 0` and then `M(0,γ,z) = 1`, `(·)^0 = 1`,
`Γ(γ)/Γ(γ) = 1`. So the requirement oracle does not merely check the transcription: it **pins the
grouping the two renderings could not settle**. §5's "acquire the source and reconcile", working.

#### Production needs no hypergeometric, which was the layer question

Eq (19) *is* hypergeometric, which would have put a new special function on the production path and
scipy inside `src/`. It is not needed: §4.2.2's **eq (4.35)** gives `Y = 1/V` as a **CIR** process, so
`E[V_t] = E[1/Y_t]` is the **inverse first moment of a non-central chi-squared** — a Poisson-weighted
sum of *elementary* terms `1/(dof + 2j − 2)`, because `χ'²_d(λ)` is a Poisson mixture of central
chi-squareds and `E[1/χ²_m] = 1/(m−2)`. So production is elementary and **eq (19) is the oracle**,
with `scipy.special.hyp1f1` in the test only (the T20a arrangement). **The two routes agree to
8.1e-10** — an elementary ladder against a hypergeometric transform differentiated at zero, different
derivations and different code. `E[1/Y]` always exists here: `dof = 4(κ+ε²)/ε² > 4` for any positive
`κ`, so the Feller-type condition §4.2.2 discusses is automatic — asserted rather than guarded for.

#### Three findings beyond the spec

- **The long-horizon limit is NOT `θ`.** The spec's oracle 5 asserts it. The stationary mean is
  **`θ/(1 + ε²/(2κ))`** — verified to machine precision at four parameter sets, and a **26% gap at
  `ε = 1.2`**: the vol-of-vol pulls the stationary mean *below* the reversion level. `θ` is the
  `ε → 0` case, now asserted as the reduction. Cowork ruled this its own error (0411) — the
  obligation-not-form lesson applied to a *requirement oracle's value*: the stationary mean of a
  **non-affine** process is not its reversion parameter, so a requirement oracle's target must be
  derived rather than read off the naive parameter. The derived `stationary_variance` lives in the
  **test**, not `engine/`: its only consumer is this oracle, so under §6 it is an oracle's reference,
  and it graduates to engine surface by product-pull. `θ`-as-long-run-variance stays correct for
  **Heston** and is wrong for **3/2** — a model-specific distinction, so it lives in the 3/2 docstring.
- **The ladder has a two-sided regime, and both edges are exact limits rather than fudges.** The
  noncentrality diverges as `t → 0` or `ε → 0` — where `Y` **concentrates**, so `E[1/Y] → 1/E[Y]`,
  converging as `1/nc` (2.5e-02 at nc=77, 1.4e-04 at 2.8e4) — and vanishes as `t → ∞`, where the law
  is **central** and `E[1/χ²_d] = 1/(d−2)` exactly. The concentration premise was checked in **both**
  directions it can be reached, which is T72's lesson: there, "a large parameter means a negligible
  dispersion" was false and returned an intrinsic for a caplet worth 0.00398.
- **The `ε → 0` reduction's T39 companion needed its own fixture.** At `ε = 0.4` the separation from
  the logistic limit is only **5.8e-04** — nearly degenerate in the very parameter the companion
  exists to exercise, the T39 trap wearing a plausible fixture's clothes. It runs at `ε = 2.0`
  (1.9e-02), with the main fixture's weakness **asserted** rather than hidden.

#### The reduction itself

`ε = 0` makes the dynamics the **logistic** ODE (the speed scales with `v`), so
`∫₀^T v dt = (1/κ)·ln[(v₀+(θ−v₀)e^{−κθT})/θ] + θT` — a genuine closed form derived in the test, not a
numerical limit, and the reduction converges at **second order in `ε`**. The κ→0 watch: the logistic
integral does carry a removable `(1/κ)ln(1+O(κ))` singularity, reported rather than left silent.

### Notes

- **Mutation sweep: applied 10/10, killed 10, survived 0, no single-killers.** One first-pass survivor
  was the new L0 ladder's intensity guard — unreachable because every present caller guards before
  calling (§4's third survivor kind, T33b's validation-guard shape), closed by supplying the input the
  guard exists to reject, plus three further oracles for the new L0 surface.
- **The ladder extraction is a pure refactor**: T72's CDF and survival are **bit-identical across 336
  `(dof × noncentrality × x)` points** (§4/T49, in-process).
- `ZHAO-CUI-FLORESCU-2018` flips **`in-hand` → `worked-from`**; eq (19) is credited to Carr–Sun
  (2007), which is **not** held, so the held paper is what is cited (§7: no reference by inference).
- **`verify.py` test floor 3187 → 3213.**

## [0.78.0] - 2026-11-22

### Added — tranche 76, VIX derivatives (and the `SmileLike` protocol extracted at its second consumer)

- **`engine/vix.py` (L4)** — `vix_squared`, `vix_affine_coefficients`, `vix_future_value`,
  `vix_option_value`, and the two registered pricers.
- **`products/vix.py` (L2)** — `VixFuture` and `VixOption`, both naming the index's own window.
- **`market/equity_skew.SmileLike` (L1)** — the protocol ruling 0401(2) asked for.

#### There is almost no new mathematics, and that is the finding

Four things the arc already had compose into the whole tranche:

1. **The identity is §1.5.2 / eq (1.31), not §3.1.1.** The spec pinned "§3.1 (and §3.1.1 for the jump
   correction) … **this is the load-bearing identity**". §3.1.1 opens by saying the opposite — *"Under
   the assumption of pure diffusion process … we have shown **in Sec. 1.5.2** that VIX² equals the
   risk neutral expected cumulative diffusion variance **[see (1.31)]**"* — so §3.1.1 is Carr–Wu's
   **jumps** correction, which Heston does not have. *(The spec's own honesty flag asked for the
   futures/options sub-section to be pinned from the TOC: **§3.1.2** defines them, eqs (3.6) and
   (3.7a), and **§3.2.1** "Affine stochastic volatility models" is the pricing — exactly where
   expected, so no `Ask:` was needed there.)*
2. **Eq (1.31) is tranche 31's strip verbatim** — same `1/K²`, same `2/T`, same OTM split at the
   forward. So `vix_squared` **is** `heston_fair_variance` and the "model-free VIX² strip" **is**
   `fair_variance`. There is no forward-window arrangement to build for the level: `VIX²_t` reads
   `[t, t+τ]`, which at `t = 0` is spot-starting.
3. **`VIX²_T` is affine in `V_T` with coefficients that fall out of (1)**: `a = (1−e^{−κτ})/(κτ)`,
   `b = θ(1−a)`, matching `heston_fair_variance` to **4.4e-16** across starting variances.
4. **`V_T`'s law is non-central chi-squared, so tranche 72's primitive prices these** —
   verified against a 400k-path simulation at the 10th/50th/90th percentiles (0.24–0.65%, the Euler
   bias). **And integrating by parts** reduces both payoffs to one integrand against the **survival**
   function: no density (so nothing new at L0, §6), nothing differentiated (so no finite-difference
   noise floor, the T11 trap), and the integrand decays with the chi-squared tail — exactly where
   `noncentral_chisquare_sf` is accurate and where `1 − cdf` collapses to zero. F-8.6's defect,
   turned into the reason this construction is the right one.

#### Two things the oracles caught in my own build

- **The first quadrature spanned 60× the *mean*** while the distribution's *spread* shrinks as the
  noncentrality grows, so the nodes stopped resolving the transition. The tell was a **non-monotone**
  error (1.3e-4, 8.6e-5, 1.9e-5, 1.1e-4) — a quadrature losing resolution, not a limit being
  approached. The span now tracks the spread, with the stretch *below* the mass integrated
  **analytically** (`∫ a/2√(ax+b) dx = √(ax+b)`, the survival being 1 there), which is what makes a
  narrow span safe rather than truncating.
- **Measuring that span exposed a coupling worth recording: wider is not safer.** With the node count
  fixed, widening past the mass **thins the nodes** and the answer drifts back — 0.2479 at 0.25
  spreads (8.1% wrong), flat to ~1e-3 from 6 through 30, drifting again by 9e-4 at 60. Holding the
  *step size* constant instead, the shipped 12-spreads/2000-nodes pair is converged to **4e-6**. The
  oracle asserts the plateau *and* that too narrow a span moves it by orders more, so the stability
  claim is not vacuous.

#### The protocol extraction (ruling 0401(2))

`SmileLike` is `@runtime_checkable` with the **single real member** `volatility` — memberful
deliberately, since an empty protocol is satisfied by everything (T33b) and could never reject a
non-smile. It lives at **L1** beside `ImpliedVolSkew`, so the L4 strip depends *down* on it. T75's
Heston subclass **collapsed into a plain presenter** in the same change, and the retype was proved a
pure refactor by **bit-identity across 108 strip evaluations** (§4/T49, in-process).

### Notes

- **Mutation sweep: applied 12/12, killed 11, survived 1.** The survivor narrows the quadrature's
  **span margin**, and for a margin that is the correct outcome — its job is to be wide enough, and a
  narrower one that still covers the mass is not a defect. Rather than pin a margin to an exact
  value (the opposite of what a margin is), the insensitivity is now **asserted**. Two first-pass
  survivors and four single-killers were closed: the below-mass analytic branch was **unreachable**
  in the main fixture (`start == lower` always — §4's third survivor kind, dead code in that regime),
  closed by a concentrated law whose mass clears zero; and the notional-scaling "killer" was
  **homogeneity**, which a *uniform* scaling error satisfies perfectly — replaced by a check through
  the engine against the simulation.
- **The `slow` gate took five passes, four lossless and one a real seam.** Sharing the simulation,
  sharing the concentrated future, measuring the node count down to 2000 (quadrature error 5.2e-05
  against bands of ~0.5–1%), skipping the future's integral for a call that never uses it, and
  finally **splitting along a subject seam** — "is the integral computed correctly" is a different
  question from "is VIX² the variance-swap rate and do the products price". 8.99s → **2.88s + 3.62s**.
  The concentrated law's ~1.4s per integral is irreducible: reaching the analytic branch *requires* a
  high noncentrality, and the mixture costs O(√noncentrality) per node.
- **An honest `slow-test-exempt`, ledgered as `OPEN.md` OI-8 — the first in the arc.** Five passes
  took the VIX tests from 8.99s to **2.97s + 3.60s** locally, and Windows CI still read **11.5s and
  11.1s** (a **4×** runner factor, worse than the 3.3× the two previous tranches showed — itself the
  F-8.9 data point). Everything left to cut is an oracle: fewer nodes, a narrower span or a smaller
  simulation all widen bands that are *measured* to resolve the defects they exist to catch, and the
  concentrated law cannot be made cheaper because reaching the quadrature's below-mass branch
  **requires** a high noncentrality whose mixture is `O(√nc)` per node. **And the obvious
  optimisation was tried and measured slower:** hoisting the Poisson ladder out of the node loop (the
  T53 shape) is bit-identical and ran 2.88s → 3.12s and 3.62s → 4.44s, because the scalar survival
  terminates its ladder early *per node* while a shared one runs until the **worst** node converges.
  That is recorded in the code so it is not retried blind, and it is the exemption's **named removal
  trigger**: a mixture with *per-argument* early exit, raised as its own unit. An honest exemption
  beats a gamed green.
- **`verify.py` test floor 3155 → 3187.**

## [0.77.0] - 2026-11-22

### Added — tranche 75, options on realized variance under Heston (and the cross-method anchor T74 lacked)

- **`engine/realized_variance_option.py` (L4)** — `integrated_variance_moments` (the first two
  moments of `∫V dt`), `heston_fair_variance` (the annualised first moment — the continuous
  variance-swap fair strike under Heston), and `realized_variance_option_value` (§5.2's normal
  approximation, pricing through L0's `normal_positive_part`).
- **`products/realized_variance_option.py` (L2)** — `RealizedVarianceOption`, five fields.

#### The cross-METHOD anchor — the independent route T74 never had

T74 built the fair variance by **static replication** (a `1/K²` strip of vanillas) and every oracle
it had was a transcription, a reduction or a requirement: no *independent route to the number*.
Heston gives one. The oracle prices Heston's European options, implies their Black vols, feeds them
through **T74's own `fair_variance` strip**, and compares to the closed form — a replication of a
model-generated skew against a moment of the variance process, different code and different
mathematics on each side. It **retroactively gives T74 the cross-method anchor**, which is why this
piece belonged in this tranche.

The skew is presented through a **subclass** of `ImpliedVolSkew` overriding `volatility`, so T74's
strip is reused *literally* and the chain stays type-safe. `ImpliedVolSkew` is a log-linear closed
form and cannot represent Heston's smile; a log-linear **fit** was the alternative and would have
made the oracle a measure of the fit's quality rather than of the two methods' agreement. **The grid
is measured here, not inherited** (T74's lens): Heston's integrand is not the skew T31 measured
against, so the width is swept and the replication is asserted to converge to the closed form
**from below** — truncation omits tail value — reaching ~0.5%.

#### The two moments: one transcribed, one derived as a theorem

The first is printed (§4.1, per monitoring interval; at the full horizon
`θT + (V₀−θ)(1−e^{−κT})/κ`). The second is **derived**, because the source's printed second moment is
per-interval *and* for the jump-extended model, and a `-layout` reflow of it is not a thing to
transcribe. T19's precedent governs — derive it and assert it as a theorem to machine precision,
which is immune to a source misprint — so `Var[∫V dt]` is derived from the printed covariance
structure and asserted **two independent ways**: against a double quadrature of that covariance
integral (3e-08) and against **tranche 48's `integrated_variance_transform`**, the CIR Laplace
transform whose derivatives at zero *are* these moments and which shares no line with the new code.
The finite differences are one-sided (the transform refuses a negative `ψ`) and therefore `O(h)`, so
that band is the FD's own error and is argued as such.

#### What this does NOT price, and why that is the spec's own reading

The spec's target says "a call/put on **discrete** realized variance" while its out-of-scope defers
"the discrete-monitoring correction" to T78 — and §5.2 (Drimus et al.) is explicitly *"the price of
its continuous counterpart plus an adjustment term due to discrete sampling"*. Both cannot hold: the
discrete price **is** the continuous one plus the correction T78 owns. This prices the **continuous**
case, the coherent half. Raised as an `Ask:` rather than settled.

#### The approximation's error is the deliverable, not the agreement

The normal match is the only approximation — the moments are exact — so the MC companion's measured
gap *is* the cost of assuming realized variance is normal. Banded `4·SE + the measured error` (§4's
two-error-source rule), with the simulation's own accuracy established against the closed-form
moments **before** it is used to judge anything, and the premise asserted that the approximation
error is **resolvable above the sampling noise** (else the test asserts only that both numbers are
noisy). Measured at ~8% of the option value near the money — a real limitation of §5.2, and the
reason §5.3's partially-exact-and-bounded approximation exists (the arc plan's recorded refinement).

### Notes

- **Mutation sweep: applied 9/9, killed 9, survived 0, no single-killers.** Two single-killers were
  closed in-tranche rather than left for a re-audit: the `omega` factor (its only killer was parity,
  a *relation* between the arms — now also transcribed directly) and the invariant-6 expiry bypass
  (hardened by the **opposite arm**, an expired in-the-money put, the construction T73 used for
  F-4.1…F-4.5).
- **The `slow` gate took three lossless passes, and CI found what the local clock could not.** The
  file began at **11.43s** locally — one Fourier inversion per strike per grid, ~1800 of them — and
  `heston_call_prices` prices a whole strike vector in one pass (T53's shared `φ(u)`), giving
  identical numbers at 2.80s. The simulation was then shared through a module fixture *before* CI
  asked, since F-8.9's fifth file was found by CI rather than foresight. **That still was not
  enough:** at 3.88s local it read **12.7s on the Windows runner** against the 10.0s ceiling — §9's
  "for a numerically-heavy file a local re-time is not the measurement, CI is", for the second
  tranche running. Two further lossless passes: the covariance double quadrature **vectorised** (the
  gate's own advice — ~2M pure-Python evaluations per parameter set, the *same rule on the same
  nodes*, so the reference's accuracy and its `rel=1e-6` band are untouched), and the two
  cross-method oracles **consolidated onto one shared skew over shared grids** (1804 implied-vol
  root finds → 1203, one vectorised inversion). **Final 1.83s local**, and nothing was traded for
  the clock: reducing the quadrature's node count or the strike counts would have been the oracle
  trimming §9 forbids, since those counts are exactly what the bands rest on.
- **`verify.py` test floor 3124 → 3155.**

## [0.76.0] - 2026-11-22

### Added — tranche 74, generalized variance swaps (the gamma swap and the corridor variance swap)

First tranche of the vol-derivatives arc (project ten, `KWOK-ZHENG`), and the lightest by design:
both products are the T31 variance swap's static replication re-weighted, so the engine reuses the
strip and the oracle spine is the reduction back to the vanilla swap.

- **`products/generalized_variance_swap.py` (L2)** — `GammaSwap` and `CorridorVarianceSwap`, frozen
  data, siblings of `VarianceSwap` and **not** subclasses: they share the words "variance swap" and
  not the accrual (§6's real-leg / year-on-year-leg line).
- **`models/variance_replication_model.VarianceStrip` (L3)** — the grid, the weighted term and the
  strike range as one frozen value object, built through three named constructors. Vocabulary, not
  abstraction (§6), and it is what keeps the walk inside §7's five arguments.
- **`engine/variance_replication.generalized_fair_variance` (L4)** — one strip walk for all three
  products, with `fair_variance` now a named case of it. Two registered pricers.

#### The source is §1.4, not ch 4.3 — finding one

The spec cited "ch 4.3 (the generalized variance swaps)" for the static replication. **Ch 4.3 is
*Moment generating function methods*** — a model-based route (the Heston MGF) that T75/T76 will want,
not a replication. The replication of both products is **Proposition 1.2 and eq (1.26) in §1.4**; the
*payoffs* are §1.3. Transcribed from §1.4 in `-raw` and `-layout` and reconciled, and recorded in
`references.md` so the next reader is not sent to ch 4.3.

#### The spec understated the gamma strip — finding two

It says "a `1/K`-weighted strip". Eq (1.24) is
`K_g = (2e^{rT}/(T·F_0))·[∫_0^{F_0} p(K)/K dK + ∫_{F_0}^∞ c(K)/K dK]` — a `1/K` weight **and a
`1/F_0` normalisation**. Building from the spec's words would have been wrong by a factor of the
forward. The `1/F_0` is folded into the weighted term rather than the prefactor, so all three
products share one `2/T` factor, the walk has exactly one varying part, and T31's vanilla path stays
**bit-identical** (one division either way, no reordering — T12(a)).

#### The gamma strip needs a WIDER grid than the vanilla one — finding three

Under a flat surface the gamma swap *is* the vanilla variance swap (`F_t` is a martingale, so
`E[∫(F_t/F_0)σ²dt]/T = σ²`). The identity is exact in the continuum and held only to **2.98e-05** on
T31's grid — and the gap is **invariant in the node count** (2.98e-05 at 1601 nodes and at 6401), so
it is **truncation, not discretisation**: the `1/K` weight decays one power of `K` more slowly, so
truncating at `5·F` costs it far more. Widening collapses it (8.8e-09 on `[0.02,10]`, 1.0e-10 on
`[0.01,20]`). Both halves are asserted, which turns "use a wider grid" into a **stated validity
regime** (§4's "a reference is code too, in the regime where it is used"): the grid T31 measured as
adequate for the vanilla strip is **not** adequate for the gamma strip at the same tolerance.

#### A §3.4 violation in the first draft, found by chasing a second killer

`VarianceStrip.restricted` refuses a corridor that does not contain the forward (§1.4 states the
assumption, and the strip splits *at* the forward, so a one-sided corridor integrates an empty half).
`_price_generalized` originally caught only `KeyError`, so that refusal propagated as a `ValueError`
**out of a registered pricer** — failure is a value, never raise-and-hope. Found not by review but by
asking what a second independent killer for the guard would look like; the answer was "reach it
through the engine", and reaching it raised.

#### Oracles

Pointwise transcriptions of both strips against an independently-formed quadrature; the corridor's
**full-range reduction** to `fair_variance`, asserted bit-identically **and labelled a tautology**
(T70) so the finite-corridor companion carries the correctness; the flat-skew identity on an
untruncated grid; requirement orderings (restricting the range only lowers accrued variance; a wider
corridor accrues more than a nested narrower one; the gamma strike sits below the vanilla under a
falling skew — a sign, not a magnitude); and the two premises this file's power rests on, asserted
rather than assumed: **a non-flat skew** (under flat vol the two strips are identical, so a
`1/K`↔`1/K²` mutation would survive — measured 9.66% apart when skewed) and a corridor that
**excludes real mass**.

**Mutation sweep: applied 9/9, killed 9, survived 0, no single-killers.** One survivor and two
single-killers were closed first: the corridor's **half-openness** `(L,U]` survived because no
fixture put a strike *on* a bound — closed by a pointwise transcription with bounds taken from the
grid's own nodes, after a first attempt that asserted the value moves when the bounds move, which
does not discriminate (both conventions move). The gamma pricer went 1 → 2 with a cross-product check
that never computes the gamma strip, and the forward guard 1 → 2 through the engine path.

### Notes

- `KWOK-ZHENG` flips **`in-hand` → `worked-from`**; `verify.py references` holds.
- **`verify.py` test floor 3090 → 3124.**
- The corridor is a **contract term** and deliberately not the grid's truncation bounds: conflating
  them would make a wider quadrature grid change the payoff.

## [0.75.0] - 2026-11-22

### Added / Changed — tranche 73, re-audit #8 remediation

One tranche for all of re-audit #8 (the T59 precedent). The sweep ran **68 applied / 63 killed / 5
survived**; this closes every finding it produced.

#### F-8.6 — the real blind oracle, and the audit found thirteen more sites than reported

`pytest.approx(x, rel=r)` uses `max(r·|x|, 1e-12)`, not a relative band — so in a tail a `rel=` claim
is **secretly absolute**, and a returned **exactly 0.0** passes. The mutation restoring the T20a/OI-3
defect (computing a survival as `1 − cdf`) therefore survived all 3069 tests, inside the very test
whose docstring cites tranche 20a. Charter clause added at §4's T20a bullet.

**The blind subset was audited mechanically, not by eye** (the F-7.1 lesson): a one-off plugin wrapped
`pytest.approx` and recorded every call whose expected value could fall at or below the floor without
an explicit `abs`. **Fourteen sites**, not one — and pinning `abs=0.0` on them showed **six were
hiding real relative error**:

| site | what the floor was covering |
|---|---|
| ncx2 native survival | a survival of **exactly 0.0** against a 8.7e-32 truth — the blind oracle |
| CVA `expected_positive_exposure` | 1.6e-12 relative at ratio −10, genuine cancellation in `m·Φ + s·φ` |
| Bachelier transcription ×2 | the test's reference priced the **nominal** forward where the engine prices the recovered one — 3.0e-14, and **5.6e-13** deep out of the money |
| displaced-diffusion transcription | the same cause, 4.9e-14 |
| Hull-White caplet ×2 | 1.7e-13 and **4.0e-12** between two independent compositions |
| Black transcription | **1.35e-11** at a price of 3.5e-29, where `F·Φ − K·Φ` cancels away every digit |

Four of those are **not tails at all** — they are `O(1e-4)` values whose true error merely exceeds
their *stated* tolerance, which the 1e-12 absolute floor silently covered. Resolved per site rather
than by one blanket loosening: the two transcriptions now price the forward the engine prices (exact,
rel 0.0); the composition and cancellation cases carry **measured** bands with the cause named; Black
carries **two** measured regimes so the tight claim survives where it is true. The five
structurally-**zero** sites were measured to return exactly 0.0, so `abs=0.0` is both honest and
tightest there; the two near-zero **parity** checks get an absolute band against the price scale
(T55 — a difference of equals has nothing to be relative to). **Re-audit after the fixes: zero blind
sites**, verified by re-running the same instrument that found them. The ncx2 survival defect goes
**0 → 8 killers**.

*(Checked and not claimed: the OI-3 `norm_cdf` defect itself was **not** re-introducible — it already
failed 9 tests before this tranche. T20c's oracle was sound; the pin adds one more killer.)*

#### F-8.3 — second killers for five previously-shipped holes (the highest-value item)

F-4.1…F-4.5 — the expired-option bypasses shipped before re-audit #4 and remediated at T45 — each had
**exactly one** killer, and §9 says an oracle guarding a previously-shipped hole especially wants a
second (the rule F-1.2 got at T37 and these never did). **Worse than first reported: `yoy.py:138` and
`:154` shared the same single test**, so deleting one test reopened *two* shipped holes.

One killer is not inherent here, which is what separates these from the standing table's "guards, one
condition each" rows: a refusal *raises*, so any test asserting the raise kills a drop — these return
PV 0, and the fall-through intrinsic can **equal** it, so the killer must be a specifically
**in-the-money** expired fixture. Each second killer is independent **by construction**, not by
parameter: the **opposite arm** for the two equity engines (an expired ITM *put*, non-degenerate
exactly where a call's fixture is zero — T72's cap/floor pattern), and for the CDS swaption the
opposite **direction** plus the `<=` **boundary** (an expiry falling exactly on the valuation date,
which a strictly-past fixture cannot discriminate). The two YoY sites get a test each, asserting a
**truncation identity** — the straddling leg equals the leg of a swap whose schedule holds only the
live periods — so neither reads the other's reference. **All five measured 1 → 2; the sweep reports
no single-killers.**

#### F-8.2 — the sweep instrument survives a kill

Its `finally` restore does not survive SIGTERM, which stranded a mutation **twice in one sitting** —
once `engine/commodity_asian.py`, once **`foundation/sobol.py`'s Gray code**, the core recurrence of
the quasi-random spine, silently wrong in the working tree. Three layers now: the per-mutation
`finally`, a **signal-handled restore** (SIGTERM/SIGINT → restore → re-raise, so the process still
dies with the signal it was sent), and a **tree-digest assertion** that refuses to return a verdict
from a tree left mutated. The end-to-end test sends a **real signal to a real child**, because that
is the path that failed and a unit test of the handler would not have caught it. It is
**platform-aware rather than skipped**: on Windows `send_signal(SIGTERM)` is `TerminateProcess` and no
handler can run, so the digest assertion is the only guard there and the test says so — the tree has
carried **zero** suppressions since v0.6.0 and a conditional skip would have both broken that and
hidden the limitation.

#### F-8.4 / F-8.1 / F-8.5 / F-8.7 / F-8.8

- **`VasicekParams.long_run_mean` deleted** — zero consumers in `src/` *or* `tests/`, §4's third
  survivor kind. The `theta`-vs-`theta/alpha` warning it carried already lives in the class docstring.
- **ch-8.5 re-binned to `out_of_method` as a candidate.** §1's test is the oracle gate, and the
  section prints no closed form — so it is a §4-extension question, **not** come-back-same-method
  debt. Three artifacts had called it a `to_go_deeper` route; `references.md`'s row pointed *at* a bin
  the route was not in. The arc's other T71/T72 routes were transcribed into `to_go_deeper.md`, which
  held **none** of them. What would change the bin is recorded: a source printing a closed-form or
  asymptotic approximation (Hagan's SABR expansion).
- **F-8.5** a non-probe, **F-8.7** a proved equivalent mutant (bit-identical over 588 combinations),
  **F-8.8** the full tree clean — and the T71 L0 promotion *strengthened* T25 (59 and 52 killers).

#### C16 — the strip walk extracted, and the convention becomes GUARDED

`engine/cap_strip.price_cap_strip` is now the one walk the three smile-family pricers share: it
partitions by the valuation date, reads the curve, builds each `CapletPeriod`, and hands a valuation
the **fixing-date horizon** — computed once. That is the structural guard C16 has lacked since T71: a
valuation is **never handed a payment date**, so it cannot read the wrong one, and C16 flips
**per-module → guarded**.

**Scope measured, not assumed.** Three consumers, not eight. Hull-White stays separate — its quantity
is a *bond-option* variance over two dates, so there is no shared value to extract — and remains
per-module, which is the honest residue. `capfloor` and `yoy` take different payoffs, so a common
walker would need a protocol over products sharing the word "leg" and not its content (§6).
**Each caller's exception set is a required parameter with no default**, because `cev_caplet`'s
`ArithmeticError` is the one a careless merge drops silently, turning a `PricingFailure` into a
traceback out of a registered pricer. **Purity: bit-identical** across all 54
`(periods × strike × kind × model)` combinations — the T12(a) proof, available because no operation
order moved.

### Notes

- **`verify.py` test floor 3069 → 3090.**
- One charter clause (§4, F-8.6's `pytest.approx` floor), ratified with re-audit #8's rulings.
- The slow-budget margin (F-8.9) is **not** in this tranche — it is Bernardo's §9 call.

## [0.74.0] - 2026-11-21

### Added — tranche 72, the CEV smile model (the single-curve short-rate arc capstone)

The arc's last spine tranche, and the one chapter-8 smile model that is **not** a reparameterisation
of Black. Its caplet needs a non-central chi-squared, which is the tranche's real new infrastructure.

- **`foundation/noncentral_chisquare.py` (L0)** — `noncentral_chisquare_cdf` and
  `noncentral_chisquare_sf`, the Poisson-weighted mixture of central chi-squareds summed **outward
  from the Poisson mode** (from `j = 0` the leading weights underflow at a noncentrality of 1500).
  The regularised incomplete gamma is private: one consumer, so §6's rule of two has not fired.
- **`models/cev.py` (L3)** — `CevModel(market, vol, beta)`, `0 <= beta < 1`.
- **`engine/cev_caplet.py` (L4)** — Cox's closed form, `cev_caplet`, `cev_absorption_probability`,
  and the strip pricer over the **existing** `NominalCap`. `NominalCap` now carries four pricers.

#### Finding 1 — the arc's own source prints no CEV closed form, so the capability was RE-SOURCED

`MCINERNEY-ZASTAWNIAK` §8.3 gives the SDE (8.5), the absorbing-boundary discussion, two **simulated
histograms** (Figures 8.1/8.2) and then pivots to the displaced-diffusion approximation as the
practical answer. No Schroder/Cox formula, no chi-squared argument triple, no worked example — and
the whole book's only other mention of the non-central chi-squared is a one-line taxonomy remark
about Cox–Ingersoll–Ross in ch 3. §4 forbids fitting a number to a figure, so every held PDF was
searched; **`BROCKHAUS-2016` §6.1 pp.79–80 prints Cox's (1996) formula** and is already
`worked-from`. Both register entries now record this so it is not re-attempted.

What `MCINERNEY-ZASTAWNIAK` *does* still anchor is structural rather than pictorial: it states the
absorbed fraction **in prose** — "over 10%" at `F(0) = 1%` and "typically low" at 5% (both
`beta = 0.5`, `sigma = 0.1`, `t = 1`). The closed form's `1 - chi2(2z; 1/(1-beta))` gives **13.53%**
and **0.0045%**. That is §4's "reproduce its structure, not its pixels", satisfied with the figures'
own stated numbers.

#### Finding 2 — the printed formula is WRONG, and both renderings print it identically

As printed: `C = S0 Q(2w; 2 + 1/(1-rho), 2z) - K [1 - Q(2w; 2 - 1/(1-rho), 2z)]`. Two defects — both
chi-squared terms share one argument/noncentrality pair where Schroder's result **swaps** them, and
the first term is missing its complement. `-raw` and `-layout` agree, so transcription independence
is powerless: this is the source's own misprint, the case §4 says only a requirement oracle can
catch. Three requirements catch it, and each is asserted **failing** on the printed form:

| requirement | correct | as printed |
|---|---|---|
| `K → 0` must give the forward (a martingale) | `F` | ≈ `−K` |
| `K → ∞` must give zero | `0` | `F` |
| defined at `beta = 0.5` (the source's own "market practice") | yes | **undefined** — `2 − 1/(1−0.5)` is *zero* degrees of freedom |

The third is the sharpest: the printed form is undefined on the entire upper half of the range it is
stated for. The corrected form was then **ruled by Monte Carlo, not by recall** — a 600k-path Euler
scheme with absorption agrees to **under 1 SE at three strikes**.

#### Finding 3 — the two reductions are ASYMMETRIC, and neither is what the spec predicted

- **`beta → 1` (Black) is unreachable**, so it earns an **O(1−beta) convergence rate over a measured
  regime** (§4's T66 clause). Errors 34.7% → 9.37% → 3.03% as `1−beta` falls 1e-1 → 3e-2 → 1e-2,
  first order over 1.5 decades. **And the regime is a property of our IMPLEMENTATION, not of the
  formula** — T67's tightening, biting harder here than there: `scipy` reaches `1−beta = 1e-4`, our
  mixture cannot, because the noncentrality grows as `1/(1-beta)²` and the sum costs
  O(√noncentrality) terms (0.003s at 1e-1, 0.34s at 1e-2, 3.6s at 3e-3, refused at 1e-3). Inheriting
  scipy's regime would have asserted a convergence we cannot compute.
- **`beta = 0` (Bachelier) is reachable exactly but is NOT an identity** — CEV absorbs at zero and
  Bachelier does not, so they agree only where the absorbed mass is negligible. Measured in
  `F/sigma_N`: 3e-10 at 5.9, **0.50% at 1.25**, **12.7% at 0.62**. Asserted as equality *in a stated
  regime* plus a companion asserting the divergence outside it, so the regime is measured.

#### The measured skew (§4 obligation-not-form — measured, then reported)

Implied Black vol at `F = 5%`, `T = 1`, with `sigma` re-scaled per `beta` (a CEV vol carries units
of rate^(1−beta), so a fixed `sigma` would confound the skew with a vol change):

| `beta` | K=4% | K=5% | K=6% | span |
|---|---|---|---|---|
| 0.2 | 24.06% | 22.03% | 20.46% | 3.60% |
| 0.5 | 23.26% | 22.01% | 21.02% | 2.24% |
| 0.9 | 22.25% | 22.00% | 21.80% | 0.45% |

Downward at every exponent — the only thing the finance forces — and flattening toward Black's flat
smile as `beta → 1`, which is the cross-model continuity the two tranche-71 models bracket.

### Notes

- **A native survival function, because `1 - cdf` is destroyed exactly where the CEV call needs it.**
  Cox's first term is an upper tail. Measured: `1 - cdf` carries **1.2% relative error at `x = 100`**
  and returns **exactly 0.0 beyond `x ≈ 200`**, where the truth is 8.7e-32 — OI-3's `norm_cdf` defect
  reproduced one model over, and a deep out-of-the-money caplet is precisely where a CEV pricer is
  asked for a number. `noncentral_chisquare_sf` sums the **upper** incomplete gammas directly and
  holds **relative** accuracy to 1.1e-14 down to 3e-148. The oracle asserts both halves — that `sf`
  is accurate there and that `1 - cdf` is not — so the two-function design is proved necessary (T15).
- **A guard I wrote was wrong in the dangerous direction, and the rate oracle caught it.** The first
  draft returned the intrinsic whenever `z` was large, reasoning that a large `z` means a negligible
  variance. **False:** `z` also diverges as `beta → 1` at a *fixed* variance, so an at-the-money
  caplet worth 0.00398 came back as **0.0** — a plausible wrong number of exactly the kind §3's
  failure-is-a-value rule exists to prevent. Replaced by a **moneyness** test (deterministic only when
  `|F−K|` exceeds 40 dispersions) plus a **refusal** above an affordable noncentrality, which the
  engine turns into a `PricingFailure`.
- **§4's tenth degenerate shape bit a FOURTH time — with its premise test already written.** The
  fixture straddled the valuation date, and the premise test asserted the historical period's leak
  was material — but computed it with `abs(year_fraction)`, i.e. a positive variance and a diffused
  price. The engine with the filter dropped sees a **negative** year fraction, collapses the
  dispersion to zero and returns the **intrinsic** — and the fixture's historical forward (2.6167%)
  sat *below* its 3% strike, so that intrinsic was 0, the same answer PV-0 gives. The premise measured
  a quantity the mutation never produces. This is the charter's own "an option bypass additionally
  needs the expired option IN THE MONEY" sub-trap, verbatim, one clause below where I was reading.
  Fixed by measuring the actual fall-through value and by a 2% strike; a **floorlet** at the 3% strike
  kills it a second way, since a cap's and a floor's intrinsics vanish on opposite sides of the forward.
- **Mutation sweep: applied 16/16, killed 15, survived 1, no single-killer oracles.** The survivor is
  a **proved** equivalent mutant — relaxing the CDF's `x <= 0` guard to `x < 0` is bit-identical over
  25 `(dof, noncentrality)` pairs, because every term of the mixture at `x = 0` is `_gamma_p(a, 0.0)`
  = 0.0; the guard is a cost guard, not a correctness one. Two first-pass survivors and two
  single-killers were closed: the invariant-6 filter went 1 → 2 killers and the fixing-date horizon
  1 → 4.
- **Convention C16's trigger has FIRED**, both halves: the CEV caplet is the fourth caplet pricer and
  the third consumer of a *forward* variance, and the three smile engines now share an almost
  identical strip walk differing only in the per-period valuation — duplicated knowledge (§6), and
  the helper it wants is the structural guard C16 has been missing. **Not extracted here**: the shape
  is a design question and this is the arc's last tranche, with the checkpoint and re-audit #8 next.
  Recorded in the register and raised for Cowork, not silently deferred.
- **The MC anchor's resolution, stated per DEFECT** (Cowork's ruling, 0385). Three plausible
  transcription slips — the first term's dof doubled, the second's shifted by 2, `z`/`w` unsquared in
  `(1−beta)` — **all satisfy every requirement oracle**, because those pin the structure and are
  silent on the interior magnitude (the T43 gap). They deviate **33.9%, 47.2% and 37.2%**, against a
  committed band of `4·SE + 0.5% ≈ 2.7%` at 120k paths (the 600k ruling run resolves ~1.5%) — so each
  sits 12–17× outside it, and the simulation is demonstrably what distinguishes the corrected form.
  Asserted in the suite, not only measured here.
- **The corrected form confirmed as the KNOWN one, not merely one that passes our checks.** The
  first term's shape follows from the section's own derivation of the forward's law (the
  squared-Bessel process is non-central chi-squared with the *forward's* level as noncentrality, so
  the exceedance probability is a survival at the *strike's* argument) — §4's earlier-step check. The
  second term is pinned at `w → 0`, where its chi-squared degenerates to the central one and equals
  `1 − P(absorbed)` **bit-identically**, a quantity independently validated against the source's
  prose claims and the simulation's own absorbed fraction. The attribution chain (`COX-1996`,
  `DELBAEN-SHIRAKAWA-2002`) is now in `references.md`, transcribed from Brockhaus's reference list;
  `SCHRODER-1989` is listed separately and flagged as an attribution from recollection, since no held
  source names it and nothing rests on it.
- **The Brockhaus misprint recorded in `references.md`'s source-defect log**, beside `BERHAN-2012`
  §5.1's and `JAECKEL-2002` §12.8's — the project's **first source correction** rather than a choice
  between two renderings. A **second** defect in the same section is recorded with it and deliberately
  *not* pinned: eq (6.10)'s drifted-case substitution carries a spurious `/T` against the space-time
  change printed on the same page, which the derivation step settles — but a caplet's forward is
  already a martingale, so eq (6.10) never arises here. Recorded in case a drifted CEV is ever pulled.
- **`verify.py` test floor 2974 → 3069.**
- **A repeat of §9's timing-gate clause, for the record:** `slow` read 16.5s for a T60 file and
  **900.4s** for `test_mutation_sweep.py` when chained behind a full suite run; alone it is `ok`. A
  gate's reading is only valid measured in isolation, and 900s is the artefact shape T19 named.

## [0.73.0] - 2026-11-21

### Added — tranche 71, the volatility-smile caplets (Normal/Bachelier + displaced-diffusion)

The sixth tranche of the single-curve short-rate arc, and the two of chapter 8's three smile models
that are a **reparameterisation of Black** rather than a new primitive. CEV (§8.3) is T72: it needs a
non-central chi-squared the tree does not have.

- **`models/smile.py` (L3)** — `BachelierModel` (a normal forward, `dF = σ_N dW`, eq 8.3) and
  `DisplacedDiffusionModel` (a shifted-lognormal forward, `d(F+α) = σ_D(F+α) dW`, eq 8.7). Two frozen
  types, no base class: they share the word "volatility" and not its units — Bachelier's is a *rate*
  per root-year, the displaced-diffusion's is dimensionless — so §6's rule of two does not fire.
  Plus eq (8.9), the at-the-money matching rule `σ_D = σ·F/(F+α)`.
- **`engine/smile_caplet.py` (L4)** — both pricers, over the **existing** `NominalCap` product: a
  smile model is a different measure on the same contract, so nothing lands at L2. `NominalCap` is
  now the second product carrying one pricer per model (three: Hull-White, Bachelier, displaced).
- **`foundation/distributions.normal_positive_part` (L0)** — the normal partial expectation
  `E[(m+sZ)⁺] = m·Φ(m/s) + s·φ(m/s)`, **promoted from tranche 25's
  `risk/credit_valuation_adjustment.expected_positive_exposure` at its second consumer.** A Bachelier
  call and an expected positive exposure are the *same closed form*; the caplet is L4 and the exposure
  L5, so §2 forbade the import and the only alternative was a third copy.
- **`CapletPeriod.forward_rate()`** — the simple forward the period implies, a derived reading of
  fields the bundle already holds, arriving with two consumers at once.

#### Oracles

- **Two pointwise transcriptions**, each read in `-raw` and `-layout` and reconciled: the Bachelier
  caplet from p.147 (written in the source's own `(s/√(2π))e^{−d₁²/2}` shape, *not* as `s·φ(d₁)`, so
  it does not restate `norm_pdf`) and the displaced-diffusion `d±` from pp.151–152.
- **The chapter's own printed pair, Example 8.1** — Black 0.003789 and the displaced-diffusion
  0.003793 at `α = 0.05` with eq (8.9)'s matched `σ_D = 0.1`. Reproduced to 4.6e-07 and 2.8e-07,
  both rounding to the printed six decimals. This is the displaced-diffusion piece's only external
  anchor, and the reason it matters is below.
- **The measured smiles** (§4 obligation-not-form — measured, then reported, never pre-asserted as a
  slope). Implied Black vol at `F = 5%`, `T = 1`:

  | strike | displaced-diffusion | Bachelier |
  |---|---|---|
  | 2% | 25.75% | 26.03% |
  | 5% | 20.03% | 17.02% |
  | 8% | 17.93% | 13.33% |

  Both monotonically decreasing, which is what the source claims for the displaced-diffusion curve
  (p.152) and what a strike-independent *absolute* vol forces for Bachelier. The Bachelier skew is the
  steeper of the two.
- **Requirements**: put–call parity per model (the oracle that catches shifting `F` and forgetting
  `K`, the classic displaced-diffusion bug — it breaks parity by exactly the shift), the
  vanishing-variance intrinsic limit, and monotonicity in each model's own vol.

#### Two ties recorded as TAUTOLOGIES, not oracles (§4, tranche 70's lesson)

- The displaced-diffusion reduction at `α = 0` is **bit-identical** to `black_price` — `F + 0.0 == F`
  exactly, so the composition never passes through a `log`/`exp` and there is no tranche-61 ULP to
  band. The spec expected one and said "band it"; banding would have been *worse* than useless, because
  a tolerance makes a tautology look like a measurement. Asserted with `==` and labelled.
- The Bachelier caplet and `expected_positive_exposure` are now **one function**, so their agreement is
  true by construction. Kept as a guard on the delegation, labelled as not independent.

### Changed

- **`expected_positive_exposure` delegates to L0** and its two private copies of `Φ` and `φ` are gone —
  a second and a fourth copy of the L0 adapters, unused by anything else since tranche 46 promoted
  `norm_pdf`. The purity proof is §4's algebraic identity plus a **measured ULP bound**, not
  bit-identity: the local `Φ` was byte-for-byte L0's, but the local `φ` multiplied by a rounded
  reciprocal where L0 divides — the same quantity in ℝ, one reordered operation, and the division
  rounds once instead of twice. Over 48 points the density moves ≤ **1 ULP** and the exposure 1 ULP at
  47 of them; the 48th cancels 26.5-fold and amplifies that bit to **16 ULP — 2.55e-15 relative**.
  Demanding bit-identity would have forbidden the promotion. All 261 L5 oracles green.
- **`verify.py` test floor 2897 → 2974.**
- **`tests/tooling/test_engine_registry.py`** — the "one pricer per product" invariant generalised to
  the one it meant: *every product registers each model at most once*. It had exempted
  `EuropeanEquityOption` **by name**, and `NominalCap` became the second multi-model product — exactly
  what its own comment warned against ("a test that has to be edited on every addition is a test that
  will eventually be edited without thought"). The hard-coded name list had also quietly stopped
  covering the tree: `CommodityOption` carries two pricers and was not on it, so nothing checked it at
  all — tranche 26b's floor-discovery defect one level down. It now walks every registered product.

### Notes

- **`math.sqrt(variance)`, not `variance ** 0.5`**, in the Bachelier caplet. A negative total variance
  means the horizon ran backwards (a period that has already fixed), and `(-x) ** 0.5` returns a
  *complex number* silently; `math.sqrt` refuses. That makes the invariant-6 filter self-enforcing —
  drop it and the engine raises rather than returning a plausible wrong number.
- **Convention C16 added** (`docs/conventions_register.md`): a caplet's volatility accumulates to the
  **fixing** date, never the payment date. Recorded **per-module** across the three caplet pricers,
  with the honest reason no structural guard exists — Hull-White's quantity is a bond-option variance
  over two dates, the smile models' is the forward's variance over one, so there is no shared value to
  compare. The trigger is a second consumer of a *forward* variance.
- **Mutation sweep: applied 10/10, killed 10, survived 0**, via the committed instrument. The first run
  was clean but flagged **three single-killer oracles** (§9) — the variance horizon moved to the payment
  date, and each model's variance made linear in vol — all dying only to the strip-sum test. The horizon
  one is the hazard this module's own docstring names in prose, so leaving it on one test was the gap §9
  asks the sweep to report. Closed with an **at-the-money closed form** that shares no line with the
  strip walk (`F = K` collapses Bachelier to `τ·df·s/√(2π)` exactly); the three now die 3, 2 and 2 times
  and the sweep reports no single-killer oracles.
- **The shift is named `displacement`, not `β`.** The source calls it `α` and reserves `β` for the CEV
  exponent it derives from (`α = (1−β)F(0)/β`); the spec called the shift `β`, which would have
  hard-wired a collision into the tree one tranche before CEV arrives. Tranche 66's `MertonParams.drift`
  is the precedent.

## [0.72.0] - 2026-11-21

### Added — tranche 70, multi-factor HJM

`models/multifactor_hjm.py` (L3) — tranche 69 with `n` driving factors.

### Two things the source settled, and the second REMOVED an oracle

The spec wrote per-factor drifts `αₖ` and an explicit factor correlation `ρₖⱼ`. §4.4 has neither:
there is **one** drift (the forward rate has a single drift and `n` diffusion terms, per eq 4.10),
and the Brownian motions are **independent** — the section opens with exactly that word.

The second matters beyond notation. The spec expected the **T17/T43 correlation discipline** to
return here as it did at G2++, and it does not: **there is no correlation parameter to pin.**
Correlation between forward rates of *different maturities* is an **output** of the volatility
vector, not an input to it — which is the framework's actual content. Where G2++ had a `ρ` needing
two independent instruments, this model has a structure that can be checked exactly.

### Which makes the reduction an identity rather than an agreement

With independent factors every quantity is a plain **sum** of T69's — `α = Σᵢ(T69's drift)`,
`ν = Σᵢ(T69's variance)`, no cross terms — so the module **calls T69's functions and sums them**.
The `N = 1` reduction is therefore **bit-identical by construction**: a one-element sum *is* its
summand. That is the arc's reduce-to-already-verified-code anchor in its strongest form, and it is
why there is no second transcription of the per-factor integrals here — rewriting them to
"generalise" them would have created the one defect the reduction could not then detect.

The chain is asserted end to end: multi-factor at `N = 1` is T69, and T69 at the Hull-White vol is
tranche 10's `bond_option_variance`. Three tranches, one number.

### Oracles

- The factor-sum drift against the **integrated condition per factor** (two forms, two paths) and
  transcribed pointwise besides (T12(b)), with each of three factors measured to matter so a
  dropped term cannot hide.
- The variance equal to a **single integral of the summed squares** — a different computation path —
  and **order-independent** in the factors, which a spurious asymmetric cross term would break.
- A mandatory three-factor companion measured **43%** away from `N = 1` (T39), with the premise that
  no two factors are scalar multiples, or the fixture would be `N = 2` in disguise.

### What the sweep caught: the T69 lesson applied to one of two guards

One survivor — bypassing the variance's inverted-interval refusal. Cause: T69's per-factor function
refuses the same interval with **the same wording**, so the guard here was bypassable without any
test noticing. The drift escaped only because its message already differed. Both now name *this*
operation, with T69's distinct message asserted alongside so the discrimination is real. A guard
whose removal still raises is pinned only by what it says (the T44b shape).

**Eight mutations applied, eight killed** after that fix.

### Changed — ONE model type for any number of factors (the §6 ruling)

`GaussianHjmModel` now holds a **tuple** of forward volatilities and there is no
`MultiFactorHjmModel`. A one-factor HJM does not *resemble* the `n`-factor one — it **is** the
`n`-factor one at `n = 1`, which eq (4.11) makes plain by collapsing to eq (4.3).

The ruling turns on the finding above. What would normally keep two such types apart is that each
one's reduction to the other is a **self-verifying oracle** — but here that reduction is
**bit-identical by construction**, so it is a *tautology*, not an independent cross-check. Two types
would have been separated to preserve nothing. The one-factor behaviour stays oracled because T69's
own transcription and Hull-White-reduction tests exercise the generalised model at `n = 1`, and a
test now pins the single-type decision so a second type cannot quietly return.

*(Code raised it as a finding with the count — one construction site, one field read, functions
untouched since they take a `ForwardVolatility` rather than a model; Cowork ruled generalise.)*

### Also — a tranche-68 file tipped the `slow` ceiling on this PR

`tests/L4/test_hw_bermudan.py` ran **10.4s against §9's 10.0s** on the Windows runner while reading
2.94s locally — a file from two tranches ago, tipped over by runner variance across a 4% margin.
That is §9's "headroom is not bankable" and its "CI is the measurement" in one event: nothing about
this tranche touched that file.

Fixed by a **lossless** reduction in §9's prescribed order: three requirement oracles each
recomputed the *same* four-point Bermudan price ladder (one, two, three and four exercise rights),
which is now a module-scoped fixture — the T50/T53 memoisation pattern, with every assertion reading
the numbers it would have computed for itself. 2.94s → **2.18s**, no test deleted and no oracle
narrowed, so no exemption was needed.

## [0.71.0] - 2026-11-21

### Added — tranche 69, the Gaussian HJM framework

`models/gaussian_hjm.py` (L3) — the framework the short-rate models are special cases of. HJM models
the **instantaneous forward rate** directly, and chapter 4 turns on one equation.

### Both anchoring forms in one tranche, which is why its oracle spine is short

Cowork named the pair at T68: anchor by **reducing to what is already proved**, or by **an identity
the answer must satisfy**. This tranche has both.

**The drift condition is a requirement, not a source value.** `α(t,T) = σ(t,T)·∫ₜᵀσ(t,u)du` is forced
by no-arbitrage, so it holds for *every* `σ(t,T)` and a transcription error cannot hide behind a
matching printed number. And the source **derives** it rather than quoting it, printing both the
integrated condition `∫α = ½(∫σ)²` and its `T`-derivative — so the two printed forms check each
other (2e-16). It is asserted at a Hull-White vol *and* at a **humped** one, where nothing about the
exponential structure can be doing the work, plus transcribed pointwise besides (T12(b): a
requirement never replaces a transcription).

**The bond-option variance reduces to already-verified code.** At `σ(t,T) = σ·e^{−κ(T−t)}`,
`ν(0,S) = ∫₀ˢ(∫ₛᵀσ(s,u)du)²ds` **is** tranche 10's `bond_option_variance` — identical to **1.3e-15**
— so the one new closed form is checked against code that has carried its own oracles for
fifty-nine tranches. The T39 companion is a humped vol, measured 2%+ away so the general double
integral is doing real work.

The integrals are computed by **quadrature of the printed integrand**, deliberately: `σ(t,T)` is an
arbitrary deterministic function — that generality *is* the framework — so there is no general closed
form to transcribe, and writing one for the Hull-White case only would build the special case the
tranche exists to generalise.

### Two findings, both a claim of mine that the mathematics refused

**The bond-option variance is NOT monotone in the option expiry.** "More time to accumulate"
suggests it rises; it does not, because moving the expiry later lengthens the outer accumulation but
**shortens the bond span**, which shrinks the inner integral. Measured at `T = 10`: 2.81e-3, 5.03e-3,
7.92e-3, **9.08e-3**, 7.74e-3, 3.95e-3, 5.78e-4 across `S = 0.5…9` — a turnover near `S = 3`. The
oracle asserts the **turnover**, which is stronger than either direction, and an earlier draft that
sampled `S = 2` as the peak passed for Hull-White while failing for the humped vol — the
fixture-chosen-by-assumption trap in miniature, so the sample now straddles the *measured* peak.

**Two guards survived the sweep — the T44b reachability shape.** Bypassing `hjm_drift`'s and
`hjm_bond_option_variance`'s inverted-interval refusals still *raises*, because the volatility
callable refuses the interval itself once `quad` reaches it — so a test matching only `"before"`
passed either way and pinned nothing. The guards are kept rather than deleted (not the §4
third-survivor case of dead code: each refuses at the level the caller asked at, naming the
operation, and before any integration is wasted), and are now pinned by their **messages**, with the
volatility's own distinct message asserted so the discrimination is real.

**Sixteen mutations applied, sixteen killed** after that addition.

## [0.70.0] - 2026-11-21

### Added — tranche 68, Bermudan swaptions in the Hull-White model

Two pieces the step-1 grep found were **not** already in the tree: the Hull-White **European
swaption** by Jamshidian's trick (`engine/hw_swaption.py`), and the **Bermudan** by exact backward
induction (`engine/hw_bermudan.py`). Plus `hw_bond`/`hw_state_variance` at L3, which Hull-White had
never carried as functions of the state.

### The design ruling: a lattice, and the counts changed the question

Cowork framed it as "generalise the Longstaff–Schwartz regression, or build a lattice". One grep
showed the first **was not available as work**: `models/exercise_boundary.py` imports only `math`
and `dataclass`, and its single "LMM" mention is the provenance line — the regression is already
model-agnostic, and the coupling lives one level up in the engine. What is LMM-specific is
`BoundaryState`'s coordinates, constructed at **one site in `src/`**.

That reframing made the lattice decisive for a stronger reason than "exact beats regression":
**Longstaff–Schwartz exists to approximate a continuation value over a high-dimensional state; under
Hull-White it is an exact deterministic function of one Markov coordinate.** Regressing would invert
§4's `closed form > cross-check` ordering and pay a regression error *plus* the
foresight/suboptimality bracket for a quantity needing neither.

### The same ordering, applied twice more

**The measure change is pinned, not transcribed.** The source prints the forward-measure *SDE*
(3.28), not the integrated drift adjustment — so writing a closed form from memory is what §5
forbids. It is integrated from the printed integrand and pinned by an identity that admits no other
answer: a bond must **reprice** through the transition, `B(t,T)·E^{P_T}[B(T,Tⱼ)|x] = B(t,Tⱼ)`.
Measured at **0 to 2.2e-16**.

**The last roll is analytic.** Gauss-Hermite converges only as about `n⁻²` on the kinked payoff
(3.9e-3 at 48 nodes, 1.1e-3 at 96, 1.1e-4 at 192; `quadrature.py` correctly refuses >370). But the
step into the last exercise date **is a European swaption**, which piece 1 has a closed form for — so
it is computed analytically at every grid point. Using quadrature there would have been the same
error the lattice was chosen to avoid, one level down. It also makes the one-exercise reduction
**exact** (0.00e+00, bit-identical) rather than grid-limited.

### The honest negative: the remaining error is measured, not claimed away

The **earlier** rolls integrate `max(exercise, continuation)`, which has no closed form, so the kink
remains: the price moves about **2e-3 relative** across 48/96/144 nodes rather than settling. That is
stated in the module, asserted by an oracle as a *bounded spread* rather than a convergence claim,
and the fix is named — **split the integral at the exercise boundary** so each piece is smooth —
rather than left vague. A test asserting `1e-6` convergence here would have asserted something false.

**And the grid is deliberately not the finest that fits.** 401 points carry 1.2e-4 against 1201 —
an order and a half *below* the 2e-3 that dominates — so refining it spends three times the work to
sharpen the negligible term. §4's resolution thinking applied to cost; the oracle asserts that
tripling the grid moves the price by less than the quadrature spread, which is the claim that
justifies the choice.

### Oracles

- `hw_bond` cross-checked against **tranche 67's `g2pp_bond` at zero second volatility** — identical
  to 1e-15 for any second speed, so new code is checked against code that carried its own anchors
  one tranche ago rather than against a second transcription.
- Jamshidian's **premise** asserted, not assumed: the bond is decreasing in the state, which is what
  makes the critical rate exist and the decomposition valid; plus the critical state's **defining
  identity** (the coupon bond is worth exactly 1 there) and the per-bond strike ordering that splits
  the payoff.
- Zero volatility reduces the swaption **exactly** to the forward swap's intrinsic, and the Bermudan
  to the best intrinsic across exercise dates.
- The Bermudan dominates **each** co-terminal European, not merely the earliest; more rights never
  hurt, with increments measured to diminish.

**Twenty-two mutations applied, twenty-two killed.**

### Not delivered, and why

**The cross-model oracle (HW Bermudan vs the T18 LMM Bermudan on a matched calibration) is not in
this tranche.** It needs a calibration bridge — solving for HW parameters that reproduce an LMM
European swaption — and the LMM swaption is a **Monte-Carlo** price, so the match would inherit
sampling noise into the premise the whole comparison rests on (the T19 lesson about centring on an
approximation). That is its own unit of work with its own oracle question, and is raised rather than
improvised.

## [0.69.0] - 2026-11-20

### Added — tranche 67, the two-factor Hull-White model (G2++)

`models/g2pp.py` (L3) — two **correlated** Gaussian factors, the first model in this tree carrying a
factor correlation in its own dynamics. So the T17/T43 correlation discipline from the spread arc
returns, and §4's newest clause gets its first real use.

Both affine coefficients **are** `hw_beta` — called once per factor, not rewritten. The
term-structure fit is structural: eq (3.42) needs only the *integral* of `φ`, which the initial
curve supplies, so the model reproduces today's curve by construction rather than carrying a fitted
`φ` field.

### Three oracle structures the source handed over

**1. `V(t,T)` is printed TWO ways — as integrals (3.40) and integrated — which beats a dual
rendering.** A dual rendering gives two typesettings of one expression; this gives two *different
computations the source itself supplies*, so quadrature of the printed integrand is an independent
check on the printed closed form. Agreement 2.8e-14, held across the whole correlation range. (T66
had the mirror image: there the book *declined* the algebra and left Exercise 3.2, so quadrature
checked the reader's working.)

**2. `ν(0,S)` at `η = 0` is EXACTLY tranche 10's `bond_option_variance`** — the first term of (3.43)
is eq 3.20 verbatim. So the one-factor reduction is a cross-check against **already-verified code**:
not a second transcription, not a numerical limit, but the same expression evaluated by code that
has had its own oracles since T10. Agreement 1e-14, with a T39 companion measuring the genuine
two-factor case 26% away.

**3. The correlation is pinned two ways, neither substituting.** `ρ` enters only through the product
`ρση`, so the cross term's **value** is transcribed at four correlations (T43 — presence and sign
are not magnitude; `ρ → ρ²` is the clean probe and dies), and the **realised correlation** of the
simulated factors is pinned against `ρ` before any price band is credited (T17/T19), improving as
`√N` over six seeds (T21).

### §4's newest clause: first use, and the regime is narrower than T66's

`V`'s single-factor terms divide by `α³` after a bracket that is itself `O(α³)` — a `0/0` three
times over, where Vasicek's was once. Measured against quadrature: 3.3e-14 at `α = 0.1`, 2.6e-11 at
`0.01`, 1.7e-8 at `1e-3`, **1.9e-6 at `3e-4`**. So the sound regime is roughly `α ≥ 0.01`, an order
narrower than T66's — **a tranche that had reused T66's boundary would have been measuring
arithmetic**, which is asserted as its own test. The limits are convergence **rates** inside the
regime, with a degradation companion stating where it fails.

The rate itself needed the right instrument: it approaches 1 **from below** (0.823, 0.899, 0.946,
0.972), because the coarse end still carries a large second-order term. A flat band would reject a
genuine O(α) approach, so what is asserted is that the measured order **rises monotonically toward
1 and reaches it** — a stronger statement than any single ratio, and one that separates O(α) from
O(α²) (ratios near 2) and from no convergence (near 0).

### What the mutation sweep found, and the lesson is the catalogue's oldest

The first sweep left **two survivors**: dropping `V(0,t)` from the fit, and dropping the `½` on the
variance correction. Cause: **every bond oracle sat at `t = 0`**, and at inception the correction
`½(V(0,t) − V(0,T) + V(t,T))` **cancels identically** — `V(0,0) = 0` and the other two terms are the
same quantity. So the correction was invisible to the inception fit, to `B(T,T) = 1` (which returns
early), and to the monotonicity companions. The §4 catalogue's oldest shape — a fixture whose key
parameter sits where it cannot be seen, here the *observation time* — closed by a pointwise bond
oracle at `t > 0` built from quadrature, plus a companion asserting the correction is material after
inception and exactly zero at it.

**Twenty-seven mutations applied, twenty-seven killed** after that addition.

### Changed

- `G2ppParams` is **one** object, not two `HullWhiteParams` plus a loose `ρ`: the correlation is
  *between these two factors*, so a loose signature would let one model's factors pair with
  another's correlation. Five cohesive fields at the §7 ceiling.
- `G2ppState(x, y)` bundles the factor pair — `g2pp_bond` would otherwise take six arguments (§7),
  and the pair is cohesive rather than co-located: the simulation produces them together.
- **`α ≠ β` is refused, because the source assumes it.** §3.7 states it after eq (3.33), and the
  structure needs it twice: (3.43)'s cross term divides by `α + β` while Exercise 3.10's
  `y = u/(α − β)` divides by the difference. Producing a number there would invent a result the
  source does not stand behind.

## [0.68.0] - 2026-11-20

### Added — tranche 66, the Vasicek short-rate model: the single-curve arc opens

`models/vasicek.py` (L3) — the canonical mean-reverting Gaussian short rate, plus Merton as its own
model. Project nine (`MCINERNEY-ZASTAWNIAK` §3.3/§3.4), the deliberate counterpoint to the
collateralised arc: a short-rate model **makes** the discount curve out of `r`, so the single-curve
mode of the design invariant is exercised **by construction** rather than selected by a snapshot's
shape.

`D(t,T) = (1−e^{−ατ})/α` **is** `hw_beta`, reused rather than re-derived — the reuse the whole arc
rests on, since every model in the book's chapter 3 is affine and shares it.

### The source leaves the closed form as an exercise, which is a better oracle than two renderings

Eq (3.6) is printed as **integrals** — `B = exp(−rD − θ∫D du + ½σ²∫D² du)` — and the explicit form is
the book's **Exercise 3.2**. So both integrals are oracled against **numerical quadrature of the
printed integrands**: a genuinely different computation path (a Gauss rule over `u` against closed
forms rational in `1/α`), not a second transcription of something the source never prints. A
companion asserts the reference would *disagree* with a plausible mis-working, so the check is
independent rather than circular.

### Two findings the source read forced

**1. The book's `α` means two different things three pages apart.** Vasicek (§3.4) is
`dr = (θ − αr)dt + σdW`, so **α is the mean reversion and the long-run mean is `θ/α`**; Merton
(§3.3) is `dr = α dt + σdW`, where α is a plain drift. The tranche spec used `dr = κ(θ−r)dt`, the
other common parameterisation, in which `θ` *is* the level — and the two readings of `θ` differ by a
factor of α, **5.6× on this fixture**. The module follows the source (§5), names Merton's parameter
`drift` rather than `α` so the collision cannot propagate, and an oracle asserts the long-run mean
is `θ/α` and not `θ`.

**2. The α→0 reduction is exact in ℝ and numerically unreachable, so it is asserted as a RATE.** The
spec called it "exact by construction". `∫D du = (τ−D)/α` is `0/0` as `α→0` and `∫D²du` is a `0/0`
over `α²`, so catastrophic cancellation sets in long before the limit: measured against quadrature
the closed form is machine-exact at `α = 0.1`, 5e-11 at `1e-3`, and **1.4% wrong at `1e-6`**. The
formula *diverges* from its own limit rather than approaching it to a few ULP, so this is **not** the
T61 "exact in ℝ, banded in float" case — banding would not help. It is a **stated-regime** case: the
reduction is asserted as the **O(α) convergence rate** over the range where the closed form is sound
(the T25 rate-detector), measured at ratios 0.949 → 0.998 and 92.7× end to end over a 100× range in
α, with a companion asserting the closed form **degrades** outside that regime so the regime is
stated rather than assumed. Merton is therefore its own closed form (3.4), which is also how the
source presents it.

### Oracles

- `I1`, `I2` and the bond each against quadrature of the printed integrand; `B(T,T) = 1` **exactly**
  as the requirement companion (every term carries a factor of τ, so equality not tolerance).
- `D` cross-checked against `hw_beta` and **labelled a consistency guard, not independence** — one
  object read twice (§4's same-formula caveat).
- The short-rate moments pointwise, with the **stationary variance measured to saturate** rather
  than pre-asserted as a shape (§4's obligation-not-form clause) — a quarter of `σ²/(2α)` inside a
  year, 99.9% inside twenty.
- Merton's (3.4) pointwise; the T39 companion measured at a 17% bond difference from Merton.

**Nineteen mutations applied, nineteen killed** — including both `α²` divisors, every factor of 2,
the `θ/α`-to-`θ` slip, and Merton's `1/6`.

### Changed

- `VasicekParams` is its own type, not `HullWhiteParams` + θ: `α` and `σ` play the same algebraic
  role and `D` *is* `hw_beta`, but Vasicek's **constant** θ against Hull-White's curve-fitted `θ(t)`
  is the difference between a model that *produces* a curve and one that *reproduces* it (§6). T57's
  LGM reuse is the contrast that draws the line.
- `short_rate` is carried as the model's **initial state**, not read from the snapshot: recovering
  `r(0)` from a curve means differentiating an interpolated curve for an instantaneous forward, a
  real piece nothing pulls yet (§6). A short-rate model's curve is its *output*.

## [0.67.0] - 2026-11-20

### Fixed — tranche 65, the re-audit #7 remediation

Re-audit #7 swept **77 applied / 76 killed / 1 real hole** and ruled four findings. This tranche
closes all of them, test- and tooling-only — **no `src/` behaviour changes** — with two charter
edits riding in the same merge so neither clause describes a practice that does not yet exist.

**F-7.1 — a new shape in the degenerate-fixture catalogue: a reference sharing a coincidental
literal.** Hard-coding `days_in_month = 30.0` in `reference_level` survived all 2625 tests. The
source was always right; the oracle was blind, for **two** reasons that had to be broken together:
the only pointwise LINEAR value test read **2026-04-15 — and April has exactly 30 days** — *and that
test's own reference wrote `/ 30` as well*. So §4's independence rule was satisfied **formally** while
the reference was independent of the **code** and not of the **coincidence**. Fixed with a February
fixture (28 days) whose reference **derives** the month length, plus a 31-day companion so the
fixture brackets 30 rather than merely avoiding it. The mutant now dies in both directions, and a
companion measures the −6.67% weight error that was invisible.

**F-7.2 — four byte-identical guards, one home.** `if any(b <= a for a, b in zip(self.times, ...))`
with the identical message existed in `market/curve.py`, `market/survival_curve.py`,
`risk/credit_valuation_adjustment.py` and `risk/mc_exposure.py` — four consumers across three
layers, two of them sharing three consecutive lines. §6's rule-of-two had fired four times over and
six prior duplication passes had not named it, because an identical two-line block in four files
reads as idiom. Extracted to `foundation/time_grid.py` as a **guard, not a `TimeGrid` type** (the T62
precedent: nothing compels a bundle, and a value object would have changed `self.times` to
`self.grid.times` at every site for nothing). Pure refactor — **bit-identical in-process** across 22
curve and survival values, committed at tolerance only (T49). The separate group of four
*domain-specific* ascending guards is deliberately excluded: merging those would be §6's
shared-word-not-shared-shape error.

**F-7.3 — the instrument now owns the single-killer count, and writing the fix exposed a second
defect.** §9 makes the single-killer re-run standing, but `SweepResult` returned only
applied/killed/survived, so every re-audit since #2 hand-rolled a counting loop *beside* the
instrument, inheriting none of its verified baseline or `finally` restore. Re-audit #7 paid: a
timeout killed the counter between mutating `schwartz.py` and restoring it, and the next run
measured against the corrupted baseline. The lesson is sharper than "the instrument had a gap" — **an
obligation that is standing but uninstrumented will be re-implemented until it fails.**

`SweepResult` now carries the per-mutant failing-test set, with `single_killers()` and
`killer_counts()`. **And the instrument's own known-answer tests caught the first draft of that
fix:** the verdict run carries `-x`, so pytest stops at the first failure and reports one `FAILED`
line however many tests would have failed — counting from it would have reported *every* kill as a
single-killer, a false-positive machine and the exact mirror of the false clean sheet this instrument
exists to prevent. So counting is a **separate mode**, and a verdict-only result **refuses** the
single-killer question rather than answering "none", since an empty answer reads as "none found"
where the truth is "nobody looked". Second time those known-answer tests have found a defect in the
thing they test.

**F-7.4 — no action, recorded.** One re-audit probe mutated prose inside an f-string and survived
correctly; the corrected probes die, so the guard is pinned. Recorded for applied-count honesty.

### Charter

- **§4** gains the degenerate-fixture catalogue's **eleventh** shape (F-7.1): a reference that
  derives independently can still be blind if it shares a coincidental literal the fixture makes
  correct — distinct from a value read back from the code and from a transcription shared with the
  source.
- **§9**'s committed-instrument clause gains the single-killer count, the standing-but-uninstrumented
  lesson, and the `-x`-incompatibility that makes counting a separate mode.

**Six mutations applied, six killed, zero single-killers** — counted *by the instrument*, which is
F-7.3's whole point.

### Also — the §9 slow gate did its job, and CI was the measurement

The eight new tooling tests put `test_mutation_sweep.py` at **13.2s on Windows** against the 10.0s
ceiling, while reading **3.45s locally** — §9's "for a numerically-heavy file a local re-time is not
the measurement, CI is", third instance. The cost is structural: every `sweep()` call verifies the
baseline by running the suite *before* mutating, so each sweeping test pays two subprocess runs, and
a Windows process spawn is expensive. The file was already at ~73% of the ceiling before this tranche.

Both reductions taken were **lossless**, in the order §9 prescribes: the assertions reading identical
inputs now share **one** module-scoped counting sweep (the T50/T53 memoisation pattern — nothing any
of them examines changed), and the file was then **split along a real subject seam** into
`test_mutation_sweep_counting.py`. The seam is a capability, not a clock: counting is its own opt-in
mode with its own pytest arguments and its own refusal semantics, and its tests share fixtures the
mechanics tests never touch. No test was deleted and no oracle narrowed — §9 declines trading the
oracle for the stopwatch, and no exemption was needed. Measured after: **1.99s** and **1.03s**.

*(A `# noqa: E402` briefly appeared in the new file's imports and the `debt` gate rejected it within
the minute. `tests/tooling/conftest.py` already puts the path in place, and its own docstring warns
against precisely the `sys.path` insert that had made the suppression look necessary — §7's "fix the
cause, never silence it", enforced by the gate rather than by memory.)*

## [0.66.0] - 2026-11-20

### Added — tranche 64, the Schwartz one-factor model: the commodity arc's capstone

Mean reversion in **log-spot** (`CLARK-2014` ch 2.10.1.1), which gives a commodity the *term
structure of volatility* a flat Black vol cannot. `models/schwartz.py` (L3) and
`engine/schwartz.py` (L4); the product is T60's `CommodityOption`, unchanged — which is the point of
a stateless engine keyed on `(product, model)`: the trade is the same and the model is what was
swapped.

**The variance saturates, and that is the whole qualitative difference from GBM.**
`Var[X_T] = (σ²/2κ)(1 − e^{−2κT})` is bounded by `σ²/(2κ)`, where a GBM's `σ²T` grows without
bound. Measured term structure of annualised variance at `κ = 0.55`, `σ = 31%`:

| maturity | annualised variance |
|---|---|
| 3 months | 0.0840 |
| 1 year | 0.0583 |
| 2.5 years | 0.0327 |
| 5 years | 0.0174 |
| 20 years | 0.0044 |

a factor of **19.2** from the near end to the far — the **Samuelson effect**, reported as a
measured consequence rather than asserted as a shape (§4's new obligation-not-form clause, applied).
And this is where `CommodityBlackModel`'s own docstring prophecy comes true: it said a commodity's
volatility was "on a path to diverge from an equity's within this very arc", which is why the two
models stayed separate types at T60 rather than being merged to save six lines.

### The spec's κ → 0 reduction was wrong, and the corrected one is stronger

The spec had `F(t,T) →` the cost-of-carry forward, so the price would reduce to T60's. **It does
not**: there is no `r − δ` anywhere in (2.147) — the Schwartz drift is *pure mean reversion* — so as
`κ → 0` the log-spot becomes driftless arithmetic Brownian motion and `F → S₀·exp(σ²T/2)`. Measured:
**88.5136** against T60's carry forward of **83.9785**. Different models, not a discrepancy, and
Clark says so on p.82 — matching market forwards means *calibrating* `α̂_X`, not something the model
supplies for free.

The cross-engine anchor survives with the true condition attached: the two agree exactly when T60's
carry reproduces the κ → 0 Schwartz drift, i.e. at **`δ = r − σ²/2`**. The oracle asserts that *and*
asserts the two **disagree by >20% at a genuine κ**, so the anchor is not vacuous — which the
spec's version could not have done, since it was comparing numbers that never matched.

### The capstone reuse, and it is exact

The MC companion reuses **T57's `simulate_lgm_states`**. The LGM state is `z_T ~ N(0, ζ(T))` with
`ζ(T) = σ²(e^{2κT} − 1)/(2κ)`, so `Var(e^{−κT}·z_T) = σ²(1 − e^{−2κT})/(2κ)` is **identically**
(2.150b) — the same `X̂_t = e^{κt}X_t` integrating factor Clark uses to integrate the SDE. The
Schwartz state is the LGM state scaled by `e^{−κT}` plus a deterministic mean. Asserted as an
identity on the two **closed forms**, not as a tolerance on a measurement, and labelled a
**consistency guard rather than independence** (one simulator, two readings — the T17 caveat).

### Oracles

- (2.150b) and (2.156) transcribed **pointwise** over a `(κ, horizon)` grid, both named surface
  (F-6.1); `F(0,0) = S₀` and zero variance at zero horizon as requirement companions.
- **The stationary limit `σ²/(2κ)`** asserted, plus the measured term structure against the closed
  form, plus monotonicity in `κ` as a requirement the algebra forces.
- **The simulated OU mean AND variance pinned before the price band is credited** (T19), each
  against its own closed form — a price would see them only through one scalar (T17) — with the
  error improving as `√N` over **six seeds** (T21: a null needs averaging as much as an effect).
- An MC price band at `4·SE` with **no approximation term** (the closed form is exact), its
  resolution named against the **mean-reversion factor**: replacing `(1−e^{−2κT})/(2κ)` by `T` falls
  far outside.
- Put–call parity on the model's own futures price; invariant 6 with an in-the-money expired fixture.

**Sixteen mutations applied, sixteen killed** — including the convexity half, the blend weight
inverted, θ swapped for the log-spot, the spot's `e^{−κT}` power dropped, and the invariant-6 bypass.

### Changed

- **`SchwartzParams` is its own type, not `HullWhiteParams` plus a level** (the ruling the spec left
  to Code). The algebra is shared — both are OU — but a *rate's* mean reversion and a
  *log-commodity-price's* are different ideas, and the DRY rule is about knowledge. T57's LGM reuse
  was the opposite case and is the contrast that makes the line clear: the LGM's numbers literally
  *are* Hull-White's, reparameterised.
- **The model carries the risk-neutral level `α̂_X`, and says so.** Clark's decomposition is
  `α̂_X = α_S − σ²/(2κ) − λ/κ`, so the physical level and the market price of risk reach every
  observable *only* through that composite — no set of futures prices can separate them. That is
  T11's identifiability lesson arriving at **design time** rather than at a calibration that fails
  to converge, so what the type carries is what is observable and the decomposition is documented
  rather than folded in silently.

## [0.65.0] - 2026-11-19

### Added — tranche 63, commodity spread options: Margrabe (exact) and Kirk (approximate)

An option on `S⁽¹⁾ − S⁽²⁾ − K` — the crack, spark and calendar spreads a commodity desk trades.
`products/commodity_spread.py` (L2), `models/commodity_spread.py` (L3),
`engine/commodity_spread.py` (L4).

**One function serves both methods, and the reduction is structural.** `S₀e^{−r^fT}` is `F·df(T)`
under T60's carry, so Margrabe (2.138) is exactly `df(T)·black_price(F₁, F₂, σ_M²T, call)` —
Black-76 with the *second forward as the strike* — and Kirk is the same with strike `F₂ + K` and the
short-leg vol scaled by `w = F₂/(F₂+K)`. At `K = 0` the weight is exactly 1, so **the same code is
exact at zero strike and an approximation away from it**, rather than two near-duplicate functions
kept in step. No bivariate normal is needed.

### The finding the measurement forced: ρ, not the strike, is where Kirk degrades

Kirk's stated assumption is `K ≪ S⁽²⁾`, so the strike looks like the axis to sweep. On this fixture
it is the **weaker axis by an order of magnitude**. Measured at `K = 40`:

| `ρ` | Kirk error |
|---|---|
| −0.9 | +2.32% |
| −0.45 | +1.50% |
| 0.0 | +0.58% |
| +0.45 | −0.19% |
| +0.9 | **+8.37%** |

against **under 1% across the entire strike range** `K = 4…90` at `ρ = 0.45`. The mechanism is in
the combined vol: as `ρ → 1` it tends to `|σ₁ − wσ₂|`, which is small, so the spread approaches
deterministic and a modest *absolute* vol error becomes a large *relative* price error. `|ρ| → 1` is
therefore where Kirk should be distrusted — which matters precisely because a crack spread is a
highly correlated pair. In the strike direction the error also **changes sign** (between `K = 60` and
`K = 90`), the same shape T62 found along its own axis and for the same reason: an approximation
pinned by matching something exactly cannot be one-signed everywhere. So monotonicity in the strike
is *not* asserted — it is not true.

**Measuring any of this needed the T16 one-seed differencing instrument.** The raw `Kirk − MC` gap is
about `−0.047` at *every* strike — a shared sampling offset some twenty times the real effect at
`K = 4`. Because `K = 0` is Margrabe and therefore **exact**, the gap there is pure noise, so
subtracting it at the same seed cancels everything the two runs share and leaves Kirk's own error.
A test asserts the instrument is necessary rather than decorative.

### Oracles

- **The combined variance transcribed pointwise BY VALUE** over a `(ρ, strike)` grid. `ρ` enters only
  through the product `σ₁σ₂ρ`, so an oracle set establishing it is present and correctly signed would
  be *complete about presence and sign and silent about magnitude* (T43) — the magnitude is the thing
  pinned.
- **The weight asserted to multiply BOTH the squared and the cross term** — the half-right mutation
  this most attracts, and both halves are measured to differ.
- **The `ρ = ±1` limits as requirement oracles**: `σ_M = |σ₁ − σ₂|` and `σ₁ + σ₂`, which pin the
  cross term's **sign** (the two are a factor of twenty apart here), plus monotonicity in `ρ` and
  non-negativity across the whole range.
- **The correlation proved unbiased BEFORE the companion is credited** (T19), by the T17 pin: the
  realised covariance of the simulated log-increments reproduces `σ₁σ₂ρ·t` — the quantity `ρ` *is*,
  not a contraction of it — with the error halving as `√N` **over six seeds**, because a null needs
  averaging as much as an effect does (T21).
- **Margrabe exact against the simulation at `4·SE` with no approximation term**, which is what
  "exact" means operationally and what separates it from every Kirk point.
- **Zero short-leg vol reduces to T60's vanilla** — a cross-product check against a different engine.
- Put–call parity (the vol cancels, so it survives a transcription error outright); invariant 6 with
  an **in-the-money** expired fixture.

**Eighteen mutations applied, eighteen killed** — the cross term dropped, sign-flipped, de-doubled,
`ρ → ρ²`, each half of the weighting, the legs swapped in Black, and both engine guards.

### Changed

- The two legs are bundled into an **ordered** `SpreadLegs` pair — six loose fields is over the §7
  ceiling, and the fix is to group, never to raise the limit. Ordered because `WTI − BRENT` is not
  `BRENT − WTI`, and equality sees it.
- **A spread option's strike may be zero or negative**, unlike every other option in the tree: zero
  is Margrabe, and negative just means the holder is paid to enter. So there is no positive-strike
  construction guard. The real constraint is `F₂ + K > 0` — which needs market data, and is therefore
  a `PricingFailure` rather than a `ValueError` (§3.4).
- `foundation/linalg.py`'s `pseudo_square_root` reused for the correlated simulation: at two factors
  on a 2×2 correlation it is full rank, so nothing is truncated and the reuse inherits the L0
  `A·Aᵀ = C` oracle rather than hand-rolling `ρZ₁ + √(1−ρ²)Z₂`.

## [0.64.0] - 2026-11-19

### Added — tranche 62, arithmetic Asian options (Turnbull–Wakeman)

The average-rate option that is actually traded — Clark: *"The overwhelming majority of Asian options
in actual traded markets are with arithmetic averaging not geometric averaging."* A sum of lognormals
is **not** lognormal, so unlike T61 there is no exact closed form: Turnbull–Wakeman matches the
arithmetic average's first two moments to a lognormal and prices with Black-76.

**This is the arc's first approximation, and the deliverable is its measured error.** The moments are
*exact*; the only approximation is the lognormal assumption applied afterwards. Measured at the
money against a 400,000-path simulation of the real arithmetic average:

| `σ√T` | TW relative error |
|---|---|
| 0.079 | 0.41% |
| 0.158 | 0.58% |
| 0.316 | 0.98% |
| 0.490 | 1.66% |
| 0.948 | **4.81%** |

So it crosses **0.5% at about `σ√T ≈ 0.12`** and is a four-to-five-percent effect by `σ√T ≈ 1`. The
honest negative is the material.

### Two findings the measurement forced, neither anticipated by the spec

**1. The TW error changes sign across strikes, so a universal per-strike resolution claim is
impossible.** Matching two moments *exactly* means the error cannot be one-signed: TW over-prices
the low strikes (+5.9 SE at `K = 65`) and under-prices the high ones (−4.9 SE at `K = 115`), passing
through **zero** near `K ≈ 98`. At that strike no path count resolves the error because there is
none to resolve. The first draft of the oracle asserted the error must be resolvable everywhere and
failed at `K = 95` — the band now treats the measured error as a **bound**, and the sign change is
asserted as the structural property that explains why.

**2. The band resolves a 25% second-moment defect, not 10% — and that is the right answer, not a
weakness.** The limiting term is TW's own approximation error (a third of the band), not sampling, so
more paths would not sharpen it. §4's T51 clause applies exactly: when a band cannot resolve a
defect, ask whether that quantity needs a band at all. `M₁` and `M₂` are **closed form** and are
pinned pointwise at `rel=1e-14` — twelve orders tighter than any path count buys — so the moments are
not the band's business. What the band *is* for is the one thing no exact check can see: **the
lognormal assumption**, to which a moment-built oracle is structurally blind.

### Oracles

- **`M₂`'s two independent printed forms** required to agree (2.120's symmetric double sum against
  2.122's diagonal-plus-twice-upper-triangle), and a companion showing they **part company on a
  descending schedule** — which is what makes the ascending guard load-bearing for *this* engine too.
- **`M₁` and `M₂` transcribed pointwise**, as named surface (F-6.1), plus `M₁` asserted **independent
  of the volatility** (it is the average of today's forwards, a martingale statement) and
  `M₂ ≥ M₁²` by Jensen with equality at zero vol, so (2.125)'s logarithm is defined.
- **Geometric ≤ arithmetic (AM–GM), the free cross-product oracle** — T61's exact closed form against
  T62's moment match, different code each side — with the ordering **reversed for puts** as the
  companion, and equality at a single fixing.
- **The `n = 1` reduction banded, not `==`** — the spec pre-paid T61's lesson, and the
  vol-invariance proof carried over.

**Fifteen mutations applied, fifteen killed** — `min → max` in the kernel, the kernel dropped, cross
terms dropped, `1/n² → 1/n`, the log dropped from (2.125), `M₂` fed to Black as the forward, and both
engines' invariant-6 and partly-fixed guards.

### Changed

- **The shared schedule guard, extracted once** (`_checked_fixings`). §6's rule-of-two fired on the
  one piece of knowledge the two Asian products genuinely share: an averaging schedule must be
  non-empty and strictly ascending, because the source's sorted evaluations of *both* the geometric
  variance (p.64) and the arithmetic second moment (2.122) are valid only for an ascending schedule.
  *(This corrects what Code reported at T61's handoff — that the guard was a geometric-only
  constraint. Re-derived from the print rather than carried forward from its own report.)* The two
  products stay **distinct types**: field-identical but priced by different methods, so a flag would
  select a whole pricing method and the dispatch could no longer tell exact from approximate.

## [0.63.0] - 2026-11-19

### Added — tranche 61, geometric Asian options (exact closed form)

The commodity arc's first exotic: a European-style option on the **geometric average** over a fixing
schedule. `products/commodity_asian.py` (L2) and `engine/commodity_asian.py` (L4).

**Exact, not approximate, and that is the whole reason it is a closed form.** The geometric average
of lognormals is itself lognormal, so the option is Black-76 on an **adjusted forward** and an
**adjusted variance**. The only new code is those two adjustments; `black_price` is reused whole and
T60's `commodity_forward` supplies each fixing's forward, so the carry stays defined in one place.

### The source prints the load-bearing quantity three ways

`CLARK-2014` p.64 gives `Var[ln A_G]` as the `min(tᵢ,tⱼ)` **double sum**, as its sorted single-sum
evaluation `Σ tᵢ[2(n−i)+1]`, and — for a uniform grid — as the closed form
`σ²T[1 − (1/6)(4n+1)(n−1)/n²]` (2.114b), whose `n → ∞` limit is `σ²T/3` (2.115b). **Three different
computation paths for one number**, which is far stronger than two renderings of one line: a
misprint would have to corrupt all three identically. They agree to 1e-13 across `n = 1…60`, and the
same treatment pins the drift (2.111 general against 2.114a).

The implementation is built on the **general** forms — real schedules are not equally spaced, and
Clark says the uniform grid is "purely to simplify the algebra" — which is what makes the
specialisations independent checks rather than restatements of the code.

### Oracles

- **`F_G` and `σ²_G` transcribed pointwise**, both as *named surface* rather than locals inside the
  pricer — the F-6.1 lesson, that a formula living only inside a composite is never actually pinned.
- **The `n = 1` reduction to T60's European**, with the exact/bit-identity distinction stated
  honestly (§4, T12a): the `±½σ²t₁` pair cancels **exactly in ℝ**, but the composition evaluates
  `exp(log F − a + a)` where T60 evaluates `F` directly, so a `log`/`exp` round trip leaves **1 ULP,
  2.22e-16 relative**. Claiming bit-identity would be false — the ops are legitimately reordered.
  **The gap's invariance in vol is what proves the cancellation is algebraic**: a defect in either
  half would scale with `σ²`, and the oracle asserts the invariance (identical at every vol from 0
  to 1.2), not merely the smallness.
- **The companion the reduction requires** (T39): `n = 1` collapses the averaging and is blind to
  the entire double sum, so a six-fixing non-uniform schedule is checked pointwise — measured at 46%
  of the single-fixing variance over a 2.2-year span.
- **An MC companion on the real geometric average**, banded at `4·SE` with **no approximation term**
  because the closed form is exact — and the resolution measured rather than asserted.
- **Averaging reduces variance**, with equality *iff* the fixings coincide — the boundary an
  implementation that merely scaled the variance down would fail — plus monotonicity in `n`.
- **Put–call parity** on the adjusted forward, swept off the money.

Premises asserted: the fixings are distinct and **non-uniformly** spaced (a uniform grid is where a
mis-indexed `min(tᵢ,tⱼ)` hides — the T17 lesson), `T ≈ 2.5` not 1, and the averaging moves the price
>20% off the European. **Fifteen mutations applied, fifteen killed**, including `min → max`, the
cross terms dropped, `1/n² → 1/n`, both halves of the convexity, and the invariant-6 bypass.

### Scope, and the source scopes it the same way

A **partly fixed** schedule is refused in domain vocabulary: its price depends on realised prints the
snapshot does not carry for a commodity, and pricing it as unfixed would be a plausible wrong
number. Clark draws the same line — he prices "the case where no historical fixings have taken place
yet". A **fully** settled schedule is invariant 6's business and is PV 0, with the in-the-money
fixture F-4.1's sub-shape requires.

## [0.62.0] - 2026-11-19

### Added — tranche 60, the commodity arc opens (project eight, `CLARK-2014`)

The commodity **forward** — cost of carry plus **convenience yield** — and the European commodity
option on it. `market/commodity.py` (L1), `products/commodity_option.py` (L2),
`models/commodity_black.py` (L3) and `engine/commodity_vanilla.py` (L4).

**The new object is the forward, not the option formula.** `models/black.py` has been
Black-76-on-a-forward since tranche 5, and `engine/equity_vanilla.py` set the precedent of a product
engine reusing it — so the engine assembles `(F, K, σ²T, ω)` and reuses `black_price`. A second
transcription of Black would only have been another chance to be wrong; the reduction to the
verified function is both the simpler design and the stronger oracle. **The step-1 grep is what made
this the arc's lightest tranche** rather than a rebuild — third arc running that it has reshaped a
spec before a line was written.

**The convenience yield enters the drift, not the discounting**, and the source is explicit about
why: Clark derives it by analogy with FX (§2.2.3), where the tradeable asset is the spot times the
return on holding the commodity, so the PDE (2.19) carries `(rᵈ − rᶠ)S ∂V/∂S` in the convection term
and `rᵈ` alone in the forcing term. That asymmetry is the whole of it, and it is why this tranche
produces a forward and leaves discounting untouched.

### Oracles

- **Eq (2.56) transcribed pointwise** over a `(δ, T)` grid, from both renderings and reconciled —
  the one genuinely-new formula, so it carries the §4 pointwise obligation.
- **The source's own earlier derivation step as an independent check.** Clark p.37 prints
  `F = S₀·E[exp(σW_T + (r−δ−½σ²)T)]` *before* applying the martingale. That expectation is evaluated
  by **Gauss-Hermite quadrature** — a different numerical path, not a second transcription — and must
  reproduce the closed form. A misprint would have to corrupt both steps identically to survive.
- **The `δ = 0` cost-of-carry reduction with its mandatory companion** (§4, T39): the reduction is
  exact *because* it zeroes the very parameter a convenience-yield defect lives in, so a non-zero-δ
  case is checked pointwise and measured material (3.8%).
- **The engine is exactly `df(T)·N·black_price`** on the same forward, to the last bit — the
  reduction proving this tranche adds carry and does not perturb Black.
- **Put–call parity**, model-free, swept **off the money** (at `K = F` the identity degenerates to
  `0 = 0`, which every wrong implementation also satisfies).
- **The Clark-vs-`black.py` agreement is kept as a CONSISTENCY GUARD and labelled as one** — both
  render the same Black-76, so it is guaranteed by construction and is not the cross-source
  independence it resembles (the T17 lesson, named in the test class).

Premises asserted rather than believed: `T ≈ 2.5` (not 1, where the time factor hides), `r ≠ δ` with
both non-zero and of different magnitude (at `r = δ` the forward collapses to the spot), a sloped-curve
companion proving the constructor reads the curve rather than a scalar rate, contango and
backwardation both reachable, and an **in-the-money** expired fixture for invariant 6 (F-4.1's
sub-shape: an out-of-the-money one returns `0` either way). Ten mutations applied, ten killed.

### Changed

- `MarketSnapshot` gains a `commodities` store, keyed by `CommodityAsset` — its own store rather than
  a widened one, the `survivals`/`projections` precedent for a new keying dimension.
- `CLARK-2014` flips `in-hand` → `worked-from`, with the chapters built against recorded.

### Debt

- **OI-7 opened**: `MarketSnapshot` now carries a `# fields-exempt` marker at six fields. It is the
  gate's own named category (*"a legitimately-wide aggregate"*) and nothing in it is a loose
  primitive wanting grouping — but bundling the four quote stores behind one `quotes` object is a
  real alternative, measured at 11 read sites and 26 construction sites across 12 test files. That
  changes the durable public contract, which is a design ruling and not this tranche's; raised to
  Cowork, with the ruling as the entry's removal trigger.

## [0.61.0] - 2026-11-19

### Fixed — tranche 59, the re-audit #6 remediation

Re-audit #6 swept **76 applied / 74 killed / 2 survived** and ruled six findings. This tranche
closes all of them plus `OPEN.md`'s last item, under one stamp (the reaudit/05 precedent).

**Two blind oracles closed — the code was right, the tests could not see a mutation.**

**F-6.1 — the LGM numéraire's `½` (eq 11.1) died to nothing.** `lgm_numeraire` appeared in its test
file only ever as a **divisor** inside the martingale ratio, so §4's "every load-bearing formula is
transcribed pointwise at least once" was simply unmet. The file's resolution test is a *correct*
measurement — the `4·SE` band does resolve the eq-(11.4) convexity defect at 7.5 SE — and is silent
about a different defect in the same family: the `½` shifts the mean 0.197%, **1.60 SE**, inside the
band. F-5.3's per-defect rule one arc on, in its sharper form where the satisfied clause is what
hides the sibling. Fixed by transcribing eq (11.1) pointwise from both renderings (0 → **21**
killers), and by stating the band's limit as its own test rather than letting it look covered.
Per §4's T51 clause the answer is not more paths: `lgm_numeraire` is closed form, so it is pinned
exactly, for free.

**F-6.2 — T4's YoY module was `ψ`-blind, and the fixed leg's `ψ` survived the whole suite.** Every
fixture was `VAL + 365·k` on ACT/365F, so **every `ψ` was exactly `1.0`** — the `τ=1` degeneracy, in
the oldest corner of the tree (§9's "full-tree means the oldest tranches", earning itself again).
`yoy_fixed_leg_value`'s `ψ` had **zero** killers: dropping it overstates the leg +100.5% semi-annual,
+301% quarterly, +1117% monthly. Both inflation-leg sites had exactly one, and it lived in *another
product's* file (T13's cross-product oracle, doing precisely what §4 credits it with). One non-unit,
non-uniform semi-annual fixture closes all three: 0 → 2, 1 → 3, 1 → 3.

### Changed

- **MM-I's owed second killer, since re-audit #5** (`market_model_i.py:154/:155`). The existing
  companion compares against the form with **both** denominators dropped, so a **single-leg** drop
  landed on a third value and slipped through — which is why two load-bearing holes sat behind one
  test. The new killer is a *requirement*, not a second transcription: `τF/(1+τF)` is a weight, so it
  must **saturate at 1**, asserted per leg. Each denominator now dies to two tests.
- **F-6.6 — the era selection consolidated into `CurveSet.curves_for_era` (L1).** It was written out
  twice (the swap pricer at L4, the capstone at L5), and the swap helper's "the one place the mode is
  selected" docstring was false. §6's rule-of-two fired on a shared **shape**. The argument is not
  tidiness: the arc's single-curve design invariant *is realised through this lookup*, so one home
  gives the invariant one structural place to be guarded instead of a per-consumer re-derivation.
  Pure refactor, no floating-point work reordered — proved **bit-identical in-process** across 18
  swap PVs in both eras and 6 capstone exposure grids with their CVAs; committed at tolerance, never
  as bit patterns (§9/T49).
- **F-6.5 — the T29 narrowing finished.** The T58 ruling narrowed `collateral.py`'s `oracle:` line
  from "collateral never increases the adjustment" to a plausibility check, but left
  `test_collateral_can_never_INCREASE_the_adjustment` inside
  `class TestTheRequirementOraclesTheFinanceForces` — name and class both still asserting the
  retracted theorem. The check is right and stays; it is now `TestThePlausibilityChecks`.
- **F-6.4 — the single-killer re-run table brought current**, and its conflation corrected: the
  "second killer at T37" credit belongs to T3's **linker** (re-measured at 2), not to the YoY module.
  The T45–T58 rows, owed for two re-audits, are added.

### Removed — F-6.3

Pyflakes had never been selected (`ruff.toml` carried only `PLR0913` and `SLF001`). Removed the two
dead locals it finds — including `collateralised_capstone.py`'s `states = simulate_lgm_states(...)`,
assigned and never read, so **T58 simulated a whole path set on every call and discarded it** — and
ten genuinely unused imports. **`F841` is now gated**: statically decidable with no legitimate
exception, which is exactly §9's test. **`F401` is not** — the registry's `@register` imports are
unused by design, so a wholesale select needs the exemption list §9 declines; routed to
`to_go_deeper` behind an `__all__` prerequisite.

### Fixed — OI-6 closed, the ledger empty again

`nominal_bootstrap._validate` admitted a quote set that **skips** a pillar (a 3y schedule does extend
a 1y one), leaving two unknowns in one par condition, and died with a bare `KeyError` — a
precondition surfacing as a crash rather than a refusal (§3.4). It now counts the pillars a quote
introduces and refuses unless it is exactly one, naming the unsolved dates. Two killers: dropping the
guard still *raises*, so only a test asserting the message is ours discriminates (the T44b shape).
A `ValueError` beside the prefix check — matching every other guard in `_validate` and T54's identical
refusal — not the spec's shorthand `PricingFailure`, which is §3's *engine* return and would change an
L1 curve constructor's type for every caller; raised to Cowork rather than settled.

## [0.60.0] - 2026-11-18

### Added — tranche 58, the collateralised-discounting arc's capstone

`risk/collateralised_capstone.py` (L5) wires the whole arc: the **LGM (T57)** simulates rate paths →
the **collateralised IRS (T55)** marks `V(t)` on each → **`collateral.py` (T29)** takes the
margin-period gap → **`mc_exposure_cva` (T26)** prices the CVA — in **both discounting eras**.
Plus the one CSA mechanic T29 deferred that most changes the economics: a **threshold**.

**The step-1 grep reshaped the tranche before it was written.** The spec Cowork almost wrote —
"build collateralised exposure under a CSA" — already existed as T29. So T58 is the *integration*
plus the threshold, not a rebuild. Fourth time the grep has changed a scope this arc.

### Two findings from the spec, both changing what was built

**1. The threshold formula was one-way and could not satisfy the spec's own reduction.** The spec
gave posted collateral as `max(V(t−δ) − H, 0)` and asked that `H = 0` reproduce T29's gap. It
cannot: T29 posts `V(t−δ)` **unfloored**, so the one-way form differs on every path whose previous
mark is negative — about half of them for a swap. The form that reduces is the **two-way** CSA
threshold, `max(V − H, 0) − max(−V − H, 0)`, which is `V⁺ − V⁻ = V` exactly at `H = 0` and `0` as
`H → ∞`. Both named reductions hold for the two-way form; only one holds for the one-way. The same
shape as T57's eq (11.4): correct in one degenerate corner, caught by the spec's own reduction.

**2. The LGM→IRS wiring needed a surface none of the four modules exposed.**
`swap_values_on_paths` (T26) is LMM-specific — forward-curve paths, an `LmmTenor`, a `SwapSpan` — so
it cannot value a T55 swap on LGM states.

**And that surface needs a modelling assumption, stated rather than buried: the tenor basis is
frozen** at today's forward basis, so a projection bond on a path is the LGM-evolved discount bond
times today's deterministic basis factor. A *choice*, not a transcription. It has the property worth
having: in single-curve mode the factor is identically `1`, so **one code path serves both eras** and
the arc invariant holds by construction rather than by branch — as in T55's pricer.

### Fixed — a real bug, caught by a requirement oracle that is not actually a law

The basis factor was built **reciprocal**, which under-stated the floating leg and flipped the
simulated mark from **+122,861 to −204,826**. What caught it was T29's *collateral never increases
the adjustment* oracle, reading **982.63 against an uncollateralised 650.41**.

Worth recording *why* it could: pathwise the property is **not** forced — with `V(t−δ) = −100` and
`V(t) = +10` the gap is `110` against an uncollateralised `10`. What makes the aggregate hold is a
*well-behaved* mark moving little over one margin period. So a violation does not mean the collateral
logic is wrong; it means the **mark** is. The oracle has real detection power precisely because it is
a property of plausible paths rather than a theorem — and whether T29's universal phrasing should be
narrowed is raised, not restated.

### Verification

49 new tests (2300 → **2322**, net of the capstone's own). Mutation: **applied 14/14, killed 14,
survived 0** via T52's instrument — after three survivors, each a known shape:

- **the `H = 0` shortcut hid the two-way form.** Both reductions the spec named (`H = 0` and
  `H → ∞`) are blind to it, because `H = 0` returns the mark directly. Only a **negative mark below
  `−H`** discriminates — where we must *post*, so the balance is negative and the one-way form floors
  it at zero. A set of oracles complete about the limits and silent about the middle.
- **the mark band was too loose.** A 25% band let a mutation that stopped discounting the floating
  leg survive; the measured drift is 2.2%, so the band is 10% — four times tighter than the defect.
- **every fixture was a PAYER**, so `omega = +1` throughout and dropping the direction changed
  nothing. T33b's input-population degeneracy; a receiver fixture now pins it.

## [0.59.0] - 2026-11-18

### Added — tranche 57, the Linear Gauss-Markov (LGM) short-rate model

`models/lgm.py` (L3). LGM is Hull-White in the parameterisation whose Monte-Carlo is **exact**: the
state `z(t)` is driftless with deterministic variance `ζ(t)`, so a simulation reaches any horizon in
**one draw** — no SDE to discretise, no stepping error to band against — and reads bond prices off it
through a deterministic `H(t)`. That is the clean rate engine T58's collateralised exposure needs,
which is the pull.

Built **from** a `HullWhiteParams` pair (`H(t) = hw_beta(κ,0,t)`,
`ζ(t) = σ²(e^{2κt}−1)/(2κ)`), so "LGM is Hull-White" holds by construction: the state's bond variance
reproduces Hull-White's own `bond_option_variance` to **machine precision**. No new parameter type —
the numbers *are* Hull-White's two, and what differs is the parameterisation, which is functions of
them rather than data. (The tranche's flagged §6 question, answered by what the type would hold.)

### The spec mis-transcribed eq (11.4), and it is wrong only where it cannot be seen

The spec wrote the bond's convexity term as `½(H_T − H_t)²ζ_t`. The source has
**`½(H_T² − H_t²)ζ_t`** — confirmed in `-raw` and `-layout`, which agree. The two coincide exactly
when `H_t = 0`, i.e. at **`t = 0`** — the one date at which nothing has been simulated, and
therefore the one date an LGM is never used at. Measured on this tranche's parameters: identical at
`t = 0`, **2.27× apart at `t = 3`**.

**Both oracles the spec itself named catch it**, which is the part worth keeping:

| oracle | source (11.4) | the spec's version |
|---|---|---|
| numéraire martingale `E[P(t,T)/N(t)] = P(0,T)` | 0.1 SE | **7.5 SE** |
| MC zero-bond option vs analytic Hull-White | 0.10 SE | **20.8 SE** |

A requirement oracle (a tradeable over the numéraire is a martingale **by construction**, no model
result involved) and a simulation that the closed-form reduction is blind to — each independently
sufficient. This is the §4 pointwise-transcription rule earning its place: the reduction to
Hull-White would **not** have caught it, because the analytic reduction never evaluates the bond at
`t > 0`.

### Verification

27 new tests (2273 → **2300**). Mutation: **applied 15/15, killed 15, survived 0** via T52's
instrument — the spec's convexity variant, the convexity dropped and sign-flipped, the state sign,
the `P(0,t)` denominator, three ways of breaking `ζ`, `H(t) = t` (the κ→0 degenerate form), and the
state scaled by the variance rather than its root.

One survivor on the first run: a **maturity-order guard** I had written and never reached. It is a
real precondition rather than dead code — `P(3,1)` asks for a bond that matured two years before it
is observed, and eq (11.4) would return a number above 1 rather than refuse — so it got a test
rather than deletion.

The MC companion is **mandatory, not decorative** (§4, T39): the analytic reduction is exact
*because* LGM ≡ Hull-White in closed form, so it cannot see the simulation, which is the entire point
of the model. `κ ≠ 0` needs no assertion — `HullWhiteParams` already refuses a non-positive mean
reversion, so the premise is structural.

The arc's era invariant holds: the curve is passed in, so the model evolves rates off whatever the
snapshot carries — a single discount curve or a `CurveSet`'s OIS curve. Nothing here makes a
`CurveSet` mandatory, which T58 must preserve.

## [0.58.0] - 2026-11-18

### Added — tranche 55, CSA discounting (the collateralised swap)

The first pricer that reads **both** of T54's curves: a swap whose floating coupons project off their
**tenor** curve and whose every cashflow discounts at **OIS**, because a cash-CSA trade earns the
collateral rate on posted collateral (`LICHTERS-2015` ch 5).

Four pieces: **`Tenor`** (L0 — a whole number of months, finance-free, the dimension a projection
curve is keyed by), the **`projections` store** on `CurveSet` keyed by `(currency, Tenor)` with a
typed accessor (L1 — its own store rather than a widened `CurveKey`, which would hang a `None` tenor
on every `DISCOUNT`/`REAL` key; the `survivals` precedent), the **`InterestRateSwap`** product (L2),
and the pricer (L4). Convention row **C15**, opened **guarded**.

### The scope finding: the spec's primary reuse anchor did not exist

The spec named "an existing swap/leg product and the single-curve swap pricer" as reuse anchors.
**Neither existed.** The tree priced inflation swaps, caps, swaptions, equity options and credit, and
had never carried a vanilla fixed-vs-float IRS — the swap *pieces* were there (`swap_annuity`,
`ParSwapQuote`) but a quote is market data, not a trade. So three sentences of the spec described
something absent, and the product had to come with the tranche.

**The spec's design survived intact, which is why this proceeded rather than stalling.** Its §1
already described *one* pricer discounting off "a source that is either a single discount curve or
the `CurveSet`'s OIS curve, chosen by what the snapshot holds" — which needs no second pricer.
Genuine single-curve mode is a snapshot with **no projections**, where the pricer projects off the
discount curve itself. Second read-before-asserting finding in two tranches, and the second time the
anchor *existed differently* rather than being missing (T54: the OIS bootstrap already existed; here:
the swap pricer never did).

### The arc invariant has its first real guard, and it is a pair

Both eras price the same swap, and the oracle asserts **both** halves: they **agree** where the eras
coincide (OIS = projection) and **differ** where a real basis exists. The second half is the
load-bearing one — a pricer that quietly fell back to one curve satisfies *agree* and fails *differ*
— and the mutation sweep confirms it: forcing the engine to always take the single-curve branch
**dies**. That is the T33b input-population degeneracy applied to a **mode** rather than a value.

Because the mode is a property of the market rather than a flag, there is **no second code path to
atrophy**, which is a stronger form of the invariant than two pricers would have been.

### Fixed — my own premise, and it was the same error as T54's, one layer up

The *differ* oracle first asserted the inter-era gap **vanishes at one period**, carrying over T54's
measurement that the two bootstraps agree on the 1y pillar. That is a statement about the **curves**;
prices also differ in their **discounting**, so a one-period swap already differs — measured
**−59.93** on a 10M notional. A curve-level identity carried into a price-level claim, which is
exactly T54's finding-2 slip repeated one level up, and it is named in the test rather than quietly
corrected.

What is true, measured: the gap is **non-zero at every maturity and grows** — 6.0e-06 of notional at
1y, 5.3e-06 at 2y, 1.3e-05 at 3y, 5.2e-05 at 4y, 1.2e-04 at 5y, changing sign as the swap crosses
par. The oracle asserts the non-vanishing and the growth, which is what a basis does.

**And the reduction's band was the wrong instrument.** It asserted `rel=1e-11` against a PV that is
**zero by construction** at the par fixed rate, and failed with `rel=1.25` on a difference of
2.1e-09 — §4's near-zero trap wearing its other face (the charter states it for an *absolute*
tolerance blind to relative error near zero; here a *relative* tolerance has nothing to be relative
to). Measured, the difference is a flat 2.1e-09 at every fixed rate — the leg arithmetic's absolute
scale, 2e-16 of notional — so the band is absolute against the notional, at ~500x the measurement.

### Fixed — `test_bates_svj.py` was sitting on the `slow` ceiling

Found by CI on a **docs-only** commit: 10.3s on Windows against the 10.0s ceiling, having passed the
run before. A file whose verdict is decided by timing variance is worse than one that simply fails,
so the cause was fixed rather than nudged — and the cause is **tranche 53's insight one layer over**.

Three tests called the MC helper with the same `(expiry, seed)` and different strikes, and **the
simulation does not depend on the strike** — only the payoff does. So a 60,000-path, 400-step
simulation ran three times for one set of paths. Caching it is lossless, exactly as sharing `φ(u)`
across a maturity's strikes was. The grid-width test also priced three strikes against a
24,001-node reference one call at a time, where `bates_call_prices` gives it one `φ` sweep per grid —
T53's entry point used in the case that motivated it.

**2.9s → 1.75s local**, no exemption, no oracle trimmed.

### Verification

61 new tests (2212 → **2273**). Mutation: **applied 14/14, killed 14, survived 0** via T52's
instrument — including the silent single-curve fallback, the roles swapped (discount at the tenor
curve, project at OIS), projecting off OIS, discounting the float leg at the projection curve, the
invariant-6 filter dropped, and the payer/receiver `omega` reversed.

The tool's own **no-op refusal** fired during the build, on a placeholder mutation of mine that
changed nothing — T52's guard working in real use rather than only in its own tests.

The invariant-6 fixture **straddles** the valuation date (the tenth-shape premise in its sharpened
form), so the filter has something to filter and a dropped filter would discount a settled coupon at
negative `t`.

## [0.57.0] - 2026-11-18

### Added — tranche 54, the OIS / multi-curve bootstrap (project seven opens)

`market/multi_curve.py` (L1): the forward-projection curve bootstrapped **against** a known discount
curve. The tree discounts every trade on one curve per currency — the pre-crisis world; a
collateralised trade discounts at the **collateral (OIS) rate** while each floating leg projects off
a **tenor-specific** curve. First build of the collateralised-discounting arc;
`LICHTERS-2015` → `worked-from`.

**Half of this tranche turned out to need no new code, and establishing that was the first result.**
An OIS par swap discounts its own cashflows, so its floating leg telescopes and the bootstrap
equation is BERHAN 5.1.2 — which is `nominal_curve_from_par_swaps` (T8). The OIS curve is therefore
that function applied to OIS quotes, **reused verbatim**. The genuinely new content is the
projection bootstrap, where the discounting curve is a *different, already-known* curve and nothing
telescopes:

    c · A^ON = Σ f^δ(t_{i−1}, t_i) · δ_i · df^ON_i      (LICHTERS §3.5)
    df^δ_{s+δ} = df^δ_s / (1 + δ f^δ(s, s+δ))

Closed form per pillar, no root-find — T8's reason exactly: given the earlier pillars this is an
exact rearrangement, and solving it numerically would be solving a closed form.

### The telescoping identity separates the two eras, which is what makes the reduction non-vacuous

Eq (1.2) — `f(t,t+δ)·df_{t+δ}·δ = df_t − df_{t+δ}` — means a floating leg sums to `1 − df_n`
**only when one curve both projects and discounts**. Measured: single-curve agrees exactly (0.0),
multi-curve differs by **1.2e-03**. A structural signature rather than a tolerance, and the
companion the exact reduction requires (T39), since a projection bootstrap that ignored the discount
curve and discounted on itself would pass the reduction and be wrong everywhere else.

**On the reduction's precision, measured rather than claimed.** The spec asked for the single-curve
case to reproduce T8 *"to the ULP"*. It does — worst **1 ULP** — and that is the honest claim rather
than bit-identity: the two routes are the same quantity in ℝ computed in a **different operation
order**, so §4's op-order condition fails. The committed assertion is a *relative* tolerance with
the ULP distance recorded beside it (the T49 lesson).

### Single-curve stays a first-class mode, not a limit

The arc's design invariant: multi-curve is a sibling, not a replacement. The reduction proves the
eras *coincide* at the boundary; a separate oracle proves single-curve is *reachable and exercised*
in genuine pre-crisis mode, with nothing from this module involved — because the way that path rots
is the T33b input-population degeneracy, where every new fixture is multi-curve and the old path
leaves the test population.

The two modes are asserted to agree where they coincide and **differ where a basis exists** — and
the size of that difference was measured after my first guess at it was an order of magnitude wrong:
**0 at 1y, rising to 7.8e-05 at 5y.** The 1y pillar being *identical* explains the shape: with a
single period the discount factor cancels whichever curve carries it, so the effect is
multi-period and grows with maturity. (Not to be confused with the basis itself — OIS against the 6M
projection curve is ~160bp, two orders larger.)

### Changed

`swap_annuity` is typed on the **`CurveHandle` protocol** rather than concrete `DiscountCurve`.
pyright refused the OIS curve a multi-curve bootstrap discounts on; the function reads nothing but
`df`, which is exactly what the protocol promises, so the concrete annotation was denying a correct
reuse rather than protecting anything. **The fourth time a concrete annotation has done that here**
(T42 twice, T44a). Annotation only — no behaviour change.

### Verification

27 new tests (2185 → **2212**). Mutation: **applied 14/14, killed 14, survived 0** — run with T52's
committed instrument. After two survivors, each a different lesson:

- **an unreachable guard.** A contiguity check (each period starting where the last paid) can never
  fire: `CouponSchedule.periods()` is built from `zip(dates, dates[1:])`, so periods are contiguous
  **by construction**, and the first period's start is covered by the anchor check. §4's third
  survivor kind, where the remedy is to delete the branch rather than keep a check no test can
  reach.
- **a guard test a crash satisfied.** `pytest.raises(ValueError)` passed with the
  non-positive-discount-factor guard removed, because the negative factor then reached
  `math.log(negative)` and raised `ValueError("math domain error")`. Matching the refusal's own
  domain vocabulary is what makes the assertion about *that* guard.

**A finding raised rather than patched:** T8's `_validate` admits a quote set that skips a pillar —
its prefix check passes, since a 3y schedule does extend a 1y one — and then fails with a `KeyError`
on the pillar that was never solved. A precondition surfacing as a crash instead of a refusal. This
module states the real condition (**one equation, one unknown**: a par quote determines exactly one
new pillar) and refuses in domain vocabulary; T8's own roughness is reported, not reached into.

## [0.56.1] - 2026-11-18

### Changed — tranche 53: `φ(u)` shared across a maturity's strikes (OI-5 retired)

`φ(u)` does not depend on the strike — only the Gil-Pelaez phase `e^{−iuk}` does — yet the inversion
re-evaluated it at every node for every strike, so a smile cost `N ×` the `φ` calls it needs. `φ` is
the dear half of the integrand: a complex `sqrt`, `log` and two `exp` per node for Heston, strictly
more for SVJ, against one `exp` for the phase.

`InversionKernel` now holds `φ` at a grid's nodes and `probabilities_at` inverts it at any strike, so
`heston_call_prices` / `bates_call_prices` price a whole ladder on one sweep. **Measured: a
five-strike smile costs the same `φ` calls as a single strike, where pricing them separately costs
5×** — asserted, not described, because T50's first threshold showed that *a rationale that is not a
test erodes*.

**The purity argument is structural rather than numerical.** `heston_call_price` delegates to
`heston_call_prices` with one strike and returns its first element, so the two paths cannot drift:
there is one implementation. On top of that, `probabilities_at` runs the same operations in the same
order the per-strike version did — the same `φ` values, the same phase, the same sequential trapezoid
— so a price is bit-identical in-process (§4's op-order condition holds for a loop hoist), while the
committed regression is a **relative** tolerance. That split is the T49 lesson applied deliberately:
bit-identity is a property of a measurement, not of a cross-platform test.

Also asserted, because batching is exactly where a guard gets quietly bypassed: the batch path
applies T50's **arithmetic floor** (a sub-floor price in a batch still returns exactly zero, a
materially negative one is still refused) and selects **one** adaptive grid for the maturity — which
is the same property that makes the kernel shareable at all.

### Fixed — OI-5, the tree's only `slow-test-exempt`, is CLOSED

The named removal trigger fired as intended. The marker was taken off **before** pushing, so the gate
measured the file rather than skipping it, and both runners pass: **ubuntu 3m49s, windows 6m55s**,
`slow` green. Locally the OI-4 oracle file went **2.8s → 2.13s** wall (test work 2.54s → 1.62s). The
closure was explicitly conditional on the CI reading, per §9, and that is the reading.

**An honest correction to OI-5's own ledger entry.** It predicted 4–10×; the file improved by ~25%.
Its cost was never dominated by the region sweep alone — the high-precision reference grid (64,001
nodes, 16× the committed one) is the dearest thing in it, with the SVJ sweep next — and batching all
three is what cleared the ceiling. The 4–10× figure is the saving on **one maturity's `φ`
evaluations**, which is what a smile-pricing caller sees; it was never the whole file's runtime, and
the entry should have said so.

**And what this does not claim.** No production *product* prices a smile today — `price(product,
model)` prices one option per call — so the saving is realised by the four test files that fit or
sweep a smile at one maturity, and prospectively by any smile consumer. The waste was in `src/` and
the fix is in `src/`; claiming a realised production speedup would be the prospective claim §6
refuses.

### Fixed — the sweep tool's own test was sitting on the `slow` ceiling

Unrelated to the hoist, and found by CI on a **docs-only** commit: `test_mutation_sweep.py` read
**10.2s on Windows against the 10.0s ceiling**, having passed the run before. A file whose verdict is
decided by timing variance is worse than one that simply fails, so it was fixed rather than nudged.

Its cost is subprocess `pytest` spawns — ~1s each on the Windows runner — and two of them were being
spent on the identical-lines regression, which **tests the wrong layer**: positional addressing is a
property of the *apply* step (given a line, mutate that line and no other), and it happens entirely
before any test runs. Asserting it at `_apply` is both free and better targeted; the end-to-end
counts were already proved by the mixed sweep. Local 2.9s → **2.1s**, with the two spawns gone.

### Verification

26 new tests (2159 → **2185**). The performance property is an oracle in its own right — `φ` counted
directly, with the premise asserted (`φ` really is evaluated per node) and the saving measured
(separate pricing costs > 4.5× the shared path), so the win cannot silently erode at the next
refactor. The SVJ path carries the same assertions, because T50's lesson one engine over is that a
shared rule being correct is not evidence both callers route through it.

## [0.56.0] - 2026-11-17

### Added — tranche 52: the mutation sweep as a committed instrument

`tests/tooling/mutation_sweep.py`. §9 makes a whole-tree mutation sweep a standing obligation — every
arc close and every eleven tranches — and for five re-audits it was an ad-hoc script rewritten from
scratch each time: **the one part of the method with no provenance**, and the one that re-introduced
the same tooling bugs twice. The engine is now versioned, reviewed, and tested against known answers.

**It is the engine, never the audit.** Mutation *selection* stays the re-audit's judgement — where a
transcription error would land, which blind spot to probe (the T43 uniform rescale, the T39 reduction
blindness), oldest-first scoping. A tool that chose its own mutations would sweep where it is easy
rather than where the failure lives, which is §9's can't-fail gate wearing an instrument's clothes.
It is not a gate and nothing runs it on a push.

**What it encodes as code rather than as things to remember:**

- **Positional addressing.** A mutation names a *line*, and the fragment is matched within that line
  only, so two textually identical load-bearing lines in different functions are two real sites. A
  string-replacement sweep refuses them as non-unique and silently leaves them uncounted — which bit
  twice in re-audit #5 alone (`yoy.py:136/:152`, `credit_legs.py:140/:250`). Carrying both the line
  and the fragment is deliberate: the line makes identical sites distinct, the fragment keeps the
  intent readable and turns a stale line number into a loud *not applied* instead of a silent
  mutation of whatever moved there.
- **Per-mutant bytecode invalidation** (T46). A mutation written inside one filesystem mtime tick
  leaves the `.pyc`'s `(mtime, size)` unchanged, so CPython declines to recompile and the run tests
  the *original* while printing a confident verdict. At T46 that produced a false **killed** and
  flipped three verdicts once caches were cleared.
- **`is_clean` requires all three conditions** — every mutation applied, nothing survived, nothing
  errored — so re-audit #4's "survivors: 0" from 21 runs that all errored is **not expressible**.

### Found by the instrument's own known-answer test: an errored run is not a kill

The clause that did not exist before this tranche, and the argument for having built the tool. By
return code alone an import error, a usage error or a collection failure is **indistinguishable from
a test failure**, so every errored run reads as *killed* — the dangerous direction. The applied-count
guard cannot catch it, because those mutations genuinely do reach the source; that is precisely the
shape re-audit #4 suffered.

So a kill is specifically pytest exit **1** (tests ran, one failed). Exits 2–5 are `errored`, counted
apart and never as a kill. The tool's planted-hole fixture surfaced this immediately: the known
survivor came back *killed*, because the inner run was erroring for an unrelated reason.

**And the baseline is verified before anything is mutated.** Tests that do not pass on the unmutated
tree make every mutant "fail" for a reason that is not the mutation — a false clean sheet. The
verdict is memoised on a **content digest** of the tree rather than re-run per sweep, which is sound
rather than a shortcut: a sweep restores every file it touches, so consecutive sweeps face a
byte-identical tree. Deliberately *not* keyed on mtimes — that is the very signal the T46 defect
exploited, and keying the guard's own memo on it would repeat the mistake one level up.

### Verification

14 new tests (2145 → **2159**), all against a **known answer** rather than trusting the verdict: a
tiny subject module whose oracle is deliberately weak in one place, so it is known in advance which
mutation must be killed and which must survive. A sweep that mis-scores is worse than none, because
it launders false confidence.

One mixed sweep demonstrates all four verdict categories at once — killed, survived, not applied,
errored — which is both the cheaper shape (each inner `pytest` costs ~0.27s of startup whatever it
runs) and the clearer one, since keeping those four apart is the instrument's whole job. The subject
is rewritten immediately before that sweep, so every mutation in it is written inside one mtime tick
of a fresh `.pyc`: the T46 window, making the `killed` verdict a regression test for the bytecode
guard rather than a separate run.

Also asserted: the subject is restored afterwards (a sweep that left the tree mutated would be a
worse failure than any verdict), an empty mutation set is refused (it would report `applied 0/0,
survived 0` and be trivially "clean" — the can't-fail shape), a no-op mutation is refused at
construction (it would be scored as a mutant while testing the original), and a red baseline is
refused rather than swept.

## [0.55.2] - 2026-11-17

### Fixed — tranche 51, re-audit #5's three blind oracles (test-only, no `src/` change)

Re-audit #5's discount-factor sweep found three sites where dropping the discounting survived the
whole suite. All three are **blind oracles** — the production code is correct and untouched; the
oracle's domain excluded the failure. `src/` has no change, which is this tranche's own test of
whether the diagnosis was right.

| # | site | why it survived | the killer |
|---|---|---|---|
| **F-5.1** | `engine/heston_fourier.py` put arm | **no test priced a put through this engine**, so the registered pricer's put *arm* was never taken — the branch *condition* is covered (reversing it kills, because a call then takes the put arm) while the arm's own arithmetic was not | a put priced **through `price(product, model)`**, anchored on the `σ_v = 0` reduction to **Black's put** (computed in the test, not the engine's call rearranged), plus parity with **both** legs priced by the engine |
| **F-5.2** | `engine/bates_fourier.py` put arm | the same, inherited by building T49 in T33's shape | a put priced through the registered Bates engine |
| **F-5.3** | `engine/local_vol_mc.py` `surface_call_price` | the MC repricing band **absorbs** it: `4·SE` measured at **11.4% / 17.6% / 30.0%** of the price against a **6.18%** defect | the discounting pinned **pointwise** against this file's independent `_price`, resolution ~1e-12 — and a second, independent killer |

**F-5.2 is the one worth naming.** `test_bates_svj.py` already asserted put-call parity, and it did
not protect the engine: it computes the put **test-side** from `bates_call_price`, so the registered
pricer's put arm is never entered. *Asserting a relation in a test does not exercise the engine's
code for that relation* — the reachability rule (T18) applied to a branch arm rather than a step.

**F-5.3 shows why the charter line this re-audit earned was needed.** The same file carries
`test_the_repricing_band_is_tight_enough_to_RESOLVE_a_dupire_defect`, a correct and measured
resolution statement — about the Dupire drift, at 8.9 SE. It reads as "resolution asserted" while
being silent about a defect of smaller signal. So the discounting got its own oracle, and the right
instrument was **not a band at all**: `surface_call_price` is a closed form, so it is pinned
pointwise, with ~1e-12 resolution rather than tens of percent and no path count involved.

### A second independent killer for F-5.3

§9 asks that an oracle guarding a **previously-shipped** hole have a second independent killer, so
deleting one test cannot silently reopen it. The pointwise check leans on the file's own `_price`;
the second shares no code with it.

Two curves chosen to hold the **carry** fixed — `r − q = 0.05` as `(6%, 1%)` and `(11%, 6%)` — give
the *same* forward, so the surface's vol and the undiscounted Black value are identical by
construction and the only difference is the discount factor. The ratio of the two prices must then be
exactly the ratio of the discount factors: measured agreement **2e-16**, and a dropped `df` collapses
the ratio to `1.0`, 4.88% away. The alternative the spec offered — a higher-path band — needs
**40,000 paths** to resolve 6.18% and costs 2.8s of simulation; this is exact and free.

### Verification

12 new tests (2133 → **2145**). Mutation: **applied 5/5, killed 5, survived 0** — the three sweep
survivors, plus a parity sign flip and a strike/forward swap to show the new tests are specific
rather than broadly sensitive. Every site now has **at least two** independent killers (F-5.1 three,
F-5.2 two, F-5.3 two).

Premises asserted in each case, because every one of these holes is the catalogue's first shape one
step away: at a zero rate a put arm's discounting is a multiplication by one and no number of puts
would reveal it, so `df < 0.97` is a stated premise in both engines; and F-5.3 asserts both that the
defect is material **and** that the MC band genuinely cannot resolve it, so the second oracle is
justified rather than merely added.

## [0.55.1] - 2026-11-17

### Fixed — tranche 50: OI-4, the Fourier-inversion truncation defect

The Heston/SVJ inversion returned **negative call prices** — arbitrage violations — in the
short-dated, high-vol-of-vol, out-of-the-money wing. Found at T49, mapped at re-audit #5, and shipped
since T33: **113 materially negative prices** across a 600-point `(σ_v, T, K/F)` sweep, including
**4.0%/4.4% error at a one-month 115% strike**, which is a quoted instrument rather than a tail
curiosity.

**The mechanism.** Heston's characteristic function decays *exponentially* in `u` (not Gaussian, as
Black's does), at a rate that collapses as `T → 0` and the vol-of-vol grows. Measured at `u = 200`,
the integrand `|φ(u)|/u` is **4.2e-47** for a one-year benign slice and **1.7e-04** in the seven-day
high-vol wing — so the committed `upper = 200` truncated while the remaining tail was larger than the
price it was computing. One constant cannot serve both ends of that range.

**The fix is regime-aware, as ruled.** `adaptive_grid` doubles the range from T33's `upper = 200`
until the integrand is genuinely negligible, at **fixed node spacing** so only the truncation error
moves (discretisation and truncation are separate error sources and this is about the second). It
measures the decay of the actual characteristic function rather than keying off a formula in the
model's parameters, so it serves Heston and SVJ — whose jump tails decay slower still — without a
second constant. Result: **zero materially negative prices**, and the mapped region reprices to a
high-precision reference at 1e-6 relative or better.

**The threshold was wrong on the first attempt, in the direction the ruling cared about.** It tests
the integrand *at* the cutoff, while what a price feels is the integral of the remaining tail —
larger by the decay length. At `1e-14` a standard tree fixture sat at `3.7e-14`, so it extended to
`upper = 400` and **doubled the cost of every benign pricing call** to change the price by 9.6e-15
relative. That is precisely the "tax every call to repair a corner" the ruling refused, re-introduced
by a constant chosen for safety rather than measured. At `1e-13` the corner reprices exactly as well
(worst relative error 2.0e-03 at both thresholds, and that residual is the arithmetic floor below,
not truncation), ordinary fixtures stay at `upper = 200` and cost what they did — measured at **0.0%
overhead** — and what still extends does so because its price genuinely moves. **The cost property is
now an oracle**, not a rationale: nothing had been guarding it, which is why the regression was
invisible.

### Added — the arithmetic floor, and why a price can now be refused

The severity sweep found a second thing the ledger had not predicted: truncation error falls under
refinement only until it reaches an *arithmetic* floor, then stalls. Measured at a point whose true
price is zero to double precision, the computed value runs 1.3e-02 → −1.4e-04 → −4.0e-08 → −1.5e-11
and then stops improving, because `P1 = ½ + I/π` cancels two `O(½)` quantities and carries an absolute
noise floor of ~2e-13 whatever the grid. A price inherits `forward × 2e-13`.

So `checked_call_price` reads a raw inversion output in three branches: a **price** above the floor;
**`0.0`** within it, because zero is the honest rendering of a number smaller than the arithmetic that
produced it; and a **refusal** below it, because a materially negative call price means the method
failed in that regime, not that the option is cheap. The third branch is what makes the second
trustworthy — a bare clamp would have converted this very defect into a clean-looking `0.0` and hidden
it — and the two cases are separated by seven orders of magnitude, not by a margin.

`heston_call_price` and `bates_call_price` now accept either a curve (choose the grid) or a
`(curve, grid)` pair (use that grid), which keeps §7's five-argument limit while letting a test pin
the old truncation and show the defect.

### Verification

44 new tests (2089 → **2133**). Mutation: **applied 17/17, killed 17, survived 0** — after two gaps
the sweep found and the oracles did not:

- **the rule could walk to its 25,600 cap with every price still correct.** Probing the decay at the
  *base* rather than at the current upper leaves correctness untouched and silently takes sixteen
  times the work the corner needs. Every oracle passed, because cost was the casualty and nothing
  tested cost. The corner's grid is now bounded above as well as below.
- **the SVJ engine never reached the floor rule at all** — bypassing it there survived the whole
  suite, because every SVJ fixture priced above the floor. The T33b input-population degeneracy one
  engine over: the shared rule being correct is not evidence that both callers route through it. A
  `K/F = 2.2` seven-day fixture now exercises both floor branches in that engine.

**OI-4 is CLOSED.** What remains is recorded rather than closed silently: prices below
`forward × 1e-12` return zero because the Gil-Pelaez *parameterisation* cannot resolve them. A
formulation that integrates the option value directly (Carr–Madan, or a COS expansion) never forms
`½ + I/π` and so never cancels that way — a new object, not a fix, and a `to_go_deeper` route.

### The `slow` gate, and the tree's first timing exemption (OI-5)

The oracle file exceeded the 10s budget on CI — 2.8s locally, ~11s on Windows, §9's "a local re-time
is not the measurement" for the fourth time. Reduced as far as it goes without trading the oracle for
the stopwatch: the pricer memoised (lossless, §3.1), two redundant region sweeps merged into one
(prices that fall with the strike and are non-negative at the *highest* strike are non-negative
throughout), the reference grid halved to twice the widest range the fix selects, and the
arithmetic-floor oracles split into their own file along a real subject seam. 5.9s → 2.8s, and no
further.

The remainder is taken as a declared **`slow-test-exempt`**, ledgered as **OI-5** — the tree's first
timing exemption, which is why it is argued rather than asserted. The region *is* the oracle's domain
(§4: an oracle must contain the failure, and this region caught a defect that had shipped sixteen
tranches), so cutting cells to fit a budget would trade the thing that works for the thing that
measures it.

**And the real fix is named in the ledger with the trigger to remove the exemption:**
`gil_pelaez_probabilities` re-evaluates `φ(u)` at every node **for each strike**, although `φ` does
not depend on the strike — only the phase does. A smile costs 4–10× what it needs to, in production
as much as in this file. Sharing one `φ` across a maturity's strikes would both retire the exemption
and make real smile pricing several times faster; it is out of this tranche's scope by ruling ("no
reshaping of the inversion where it was already correct"), so it is raised rather than folded in.

## [0.55.0] - 2026-11-16

### Added — tranche 49, the SVJ / Bates model (the vol-surface arc's last build)

`models/bates.py` (L3) and `engine/bates_fourier.py` (L4): Heston plus Merton-style lognormal jumps
in the underlying, priced by **tranche 33's Fourier inversion reused whole**. Ch 5's argument for the
model is the one T46 met from the other side — jumps supply the steep short-dated skew stochastic
volatility alone cannot, which is the gap the SVI fit exposed.

`φ_SVJ = φ_Heston · φ_jump`, with
`ψ(u) = m(e^{iuα − u²δ²/2} − 1) − iu·m(e^{α+δ²/2} − 1)` transcribed from ch 5 p.65 in both
renderings and reconciled. The eight SVJ numbers bundle as `HestonParams` + a three-field
`JumpParams`, which keeps `BatesParams` inside §7's limit without inventing a nine-field object.

**A notation trap the spec defused before RED:** Gatheral writes `λ` for Heston's **mean-reversion
speed** and `λ_J` for the **jump intensity**, in the same formula. The reduction to Heston is at zero
*intensity*; `λ → 0` would destroy the variance process instead.

### The oracle is a double reduction, and each one's blindness is asserted

* **Reduction A — zero intensity ⇒ Heston.** Exact, reuses T33 — and **blind to every jump
  parameter**, since it is taken where the jumps vanish. Asserted directly: at `m = 0`, `α` and `δ`
  can be anything (`-0.5/0.4`, `0/0.01`, `0.9/1.2`) and A still passes.
* **Reduction B — zero vol-of-vol with `v₀ = v̄` ⇒ the Merton jump diffusion.** An *independent closed
  form* (the Poisson-weighted Black series, not the Heston CF), non-degenerate in exactly what A
  kills, and symmetrically blind to the stochastic volatility. The series is the test's own
  derivation — conditional on `n` jumps, `log S_T` is normal with forward `F·e^{−mkT}·e^{n(α+δ²/2)}`
  and total variance `V + nδ²` — checked against an MC of the jump diffusion before being committed
  (0.07 to 1.39 SE over five cases) and equal to plain Black **bit-identically** at `m = 0`.

Third tranche running where the T39 blind spot bit (T46's `b→0`, T48's `η→0`, here), and the first
where the two reductions are each other's companion rather than one needing a companion bolted on.

**One spec correction from the tree's own convention:** §2 asked for `φ_SVJ(−i) = 1`, which holds
only at zero carry — this tree's Heston factor ends in `+ iu·carry·T`, so `φ_SVJ(−i) = F/S₀` and it
is the **jump factor alone** that must be 1. `φ_jump(−i) = 1` exactly is what pins the compensator,
and both forms are asserted so the correction is visible rather than silently substituted.

### Changed — the Gil-Pelaez inversion extracted at its second consumer

`engine/fourier_inversion.py` (new) holds the inversion, `FourierGrid` and the `_LOWER` cutoff;
`heston_fourier.py` keeps what is Heston-specific. SVJ differs from Heston in *exactly one thing* —
its characteristic function — so §6's rule-of-two fires on a genuinely shared shape, and the
alternative (a sixth argument on `heston_probabilities`) would have broken §7's limit for something
that is not vocabulary. **The move is proved bit-identical on 120 Heston reference values** — 360
doubles, same interpreter, before versus after, every bit equal.

**And the first committed form of that proof was wrong in a way CI caught.** The captured bit
patterns were asserted for *equality*, which turns an in-process measurement into a cross-platform
claim — and §9 states the rule it breaks: transcendental results differ 1-2 ULP across runtimes.
`cmath.exp`/`log`/`sqrt` duly disagreed with the capturing machine by **1 to 4 ULP** on both
runners, 12 tests failing. A second attempt asserted a **ULP distance** — which sounded stronger and
was the wrong instrument, and is the more interesting half of the lesson: a Fourier price is a sum of
4001 transcendental terms, so its cross-platform difference accumulates as a **relative** quantity,
while ULP counts representable steps whose absolute size follows the exponent. A price of `0.00158`
came out **25,475 ULP** away while being **3.5e-12** relative. The committed form is a relative
tolerance of `1e-9` against a measured worst of 3.5e-12 — six orders tighter than the tightest
behavioural oracle in the tree. The in-process bit-identity remains the purity argument; the portable
regression is what a test running elsewhere is entitled to assert, and reaching for the
strongest-sounding metric rather than the one the quantity calls for cost two CI rounds.

**And the reference set is 54 of 60, with the six exclusions named.** A regression needs a number the
engine can resolve, and six deep out-of-the-money short-dated points sit below `1e-3` on a spot of
100 — quadrature residue rather than prices. One of them is a finding in its own right, raised for
ruling rather than regression-locked: at `σ_v = 0.9`, 30 days, `K = 130`, tranche 33's committed grid
returns **−2.5e-04** where an `upper=1200, count=24001` reference gives **+1.7e-05** — a *negative
call price*, wrong by 2.7e-04 absolute. Same regime and same cause as this tranche's wider SVJ grid,
in code that merged sixteen tranches ago. Pinning it would have preserved a wrong number.

The extraction also made the inversion testable **with no model at all**, which a Heston-only test
could never do: handed a Gaussian characteristic function, both probabilities have exact closed
forms, and `P1`'s — the **Esscher tilt**, which shifts the mean by exactly `σ²` — pins the
share-measure line against a formula derived in the test. (A point mass, `φ ≡ 1`, is the obvious
degenerate check and a trap: its CF never decays, so the truncated quadrature measures the grid
rather than the inversion.)

And one guard the move created: `tests/L4/test_heston_fourier.py` had been patching
`heston_fourier._LOWER`, which after the move would have been a **silent no-op** — the test passing
while measuring nothing, the §9 can't-fail shape inside a test. The constant's home is now asserted.

### Fixed — SVJ needs a wider truncation than Heston, and it hides where no one looks

Jump tails make `φ_SVJ` decay more slowly in `u` than `φ_Heston`, and the damage is worst at
**short** maturity, where the diffusion has contributed least. Measured against an
`upper=1200, count=24001` reference at 60 days, tranche 33's `upper=200` is wrong by **2.9e-05**
relative while `upper=400` is at 2.7e-11; at 730 days both sit at 1e-14. So a Heston-shaped default
prices long-dated options perfectly and short-dated ones wrongly. `_DEFAULT_GRID` is now
`upper=400`, and the constant's *value* is oracled rather than chosen.

### Verification

2027 → **2089** tests (the extraction's own file plus this tranche's 66, less the six excluded references). Mutation:
**applied 22/22, killed 22, survived 0** — after one real hole that took two attempts to close, and
the second attempt is the lesson. Narrowing `_DEFAULT_GRID` back to Heston's 200 survived the whole
file: the grid constant's *value* was oracled, but the **registered pricer's use of it** was not,
because the pricer test ran at 730 days where the narrow grid is already accurate to 1e-14 (the T33b
input-population shape one level up — the constant was tested, the code path reading it was not).
The first fix added a short-dated pricer test **at the money** and the mutant *still* survived:
truncation is an **absolute** error, so it is negligible against a large price and ruinous against a
small one — 1.7e-08 at the money versus 2.9e-05 at `K=130` and 1.4e-04 at `K=150`. The fixture had to
be deep out of the money *and* short dated, which is exactly the region an SVJ model is bought for.
Both premises are now asserted, including that the at-the-money fixture is invisible.

Also oracled: the compensator identity `φ_jump(−i) = 1` with its non-vacuity shown; the
factorisation as a structural identity; put-call parity across strikes and maturities; an independent
MC of the jump diffusion banded at `4·SE` with the inversion's own error measured separately and the
band's resolution stated (the jump effect is 13.8% of the price against a `4·SE` band of ~2%, a ~7×
margin); and the **cross-arc tie-back** — an SVI fit (T46) to an SVJ smile has a materially steeper
short-dated at-the-money skew than one fitted to a Heston smile, asserted as an **ordering** only,
since ch 5's results are plots.

## [0.54.0] - 2026-11-16

### Added — tranche 48, volatility derivatives (the volatility swap and its convexity)

`products/volatility_swap.py` (L2) and `engine/volatility_swap.py` (L4): the fair volatility-swap
strike `K_vol = E[√V_T]` under Heston, and the **convexity adjustment** `√(K_var) − K_vol ≥ 0`. The
second build of the vol-surface arc (T47 was dropped by the §1 pull gate). Reuses T31's variance
convention and T33's Heston parameters whole; no new L0 primitive.

The engine is Gatheral eq 11.6 — `E[√W_T] = (1/2√π)∫₀^∞(1 − E[e^{−ψW_T}])ψ^{−3/2}dψ` on the
**total** variance `W_T = ∫₀^T v_t dt` — closed by the CIR Laplace transform `A·e^{−ψv₀B}`. The swap
pays the *annualised* `V_T = W_T/T`, so `K_vol = E[√W_T]/√T`; at `T = 1` the two normalisations
coincide, so every oracle runs at two maturities and neither is a year. A mutant dropping that
`1/√T` dies there.

**Both renderings lose the overbar on `v̄`, so §4's dual-rendering rule is powerless — exactly as
T46's lost the radical.** `-raw` and `-layout` both print `A`'s exponent as `2λv/η²` and the mean as
`(1−e^{−λT})/λ·(v − v) + vT`: the long-run and initial variances are indistinguishable in the
extraction. What resolves it is a **requirement oracle against code already held** — the transform's
slope at zero must be `E[W_T]`, and T33's `HestonParams.integrated_variance` computes
`θT + (v₀−θ)(1−e^{−κT})/κ`, which is precisely the mean the text prints. That pins `A`'s exponent to
the long-run variance and the exponential to the initial one, and it reproduces it to **1.5e-8**.

**And the source hands over the degenerate fixture.** The oracle above discriminates only when
`v₀ ≠ θ`, and Gatheral's own illustrative parameters — Figures 11.2 (Heston-Nandi) and 11.3
(Bakshi-Cao-Chen) — both set `v = v̄ = 0.04`, as does the `var[W_T]` derivation that assumes it.
Building on them would have hidden the overbar completely: swapping `v₀` and `θ` moves the mean by
8.5% on our fixture and by **exactly nothing** on the source's. Asserted both ways.

**The printed transform overflows, so it is evaluated rearranged.** `e^{φT}` grows with `√ψ` and the
quadrature needs `ψ` to 40,000; at `η = 1, T = 5` the printed form raises before the integrand
decays. Dividing numerator and denominator by `e^{φT}` is the **same quantity in exact arithmetic**
and underflows harmlessly instead; agreement with the printed form measured at **2.1e-15** (a few
ULP) wherever the printed form computes at all — §4's pure-refactor standard, an exact-arithmetic
identity plus a ULP band, since the reordering moves the last digits and bit-identity is
unavailable.

### Fixed — a source-level error, recorded rather than reproduced

**Ch 11's order-of-magnitude claim for the convexity adjustment is wrong by a power.** p.146 derives
`var[W_T] = v̄T·η²/λ² + O(T⁰)` as `T → ∞`, concludes that the standard deviation of annualised
*volatility* scales as `η/λ`, and predicts the Bakshi-Cao-Chen convexity to be "roughly 3.39 times"
the Heston-Nandi one. But Taylor-expanding the square root gives
`√E[V] − E[√V] ≈ var[V]/(8·E[V]^{3/2})`, which is **quadratic** in `η/λ`: the predicted ratio is
`3.3913² = 11.50`, and the measured ratio is **11.49**. Ch 11's results are plots, so §4's "no number
is fitted to a figure" forbids settling this against Figures 11.2/11.3 — the derivation settles it,
and the oracle asserts the square while excluding the first-order reading. The spec also called this
a *small*-`T` statement; it is the `T → ∞` limit. Both halves corrected.

### Verification

107 new tests (1852 → **1959**). Mutation: **applied 20/20, killed 20, survived 0** — after one real
hole. The first run left `M16 pricer: drop the discount factor` alive, because the fixture's curve
was flat at **zero**: `df(T) = 1` exactly, so the pricer's discounting was a multiplication by one.
That is the degenerate-fixture catalogue's *first* shape ("a zero makes a factor invisible") in its
purest form, in a factor no amount of reading would have questioned. Fixed with a 3% rate and the
premise asserted (`df < 0.95`, and the undiscounted number shown to differ).

Oracles: the Laplace requirements (`F(0) = 1`, decreasing, in `(0,1]`); the slope-at-zero overbar pin
above; **`ρ`-independence**, since `W_T` is a functional of the variance path and the variance SDE
carries no correlation; the Jensen bound across a 60-point parameter sweep; the exact `η → 0`
reduction with a companion asserting the convexity is **materially** non-zero (T39 — the reduction is
exact *because* it zeroes the parameter the convexity lives in); an independent Monte-Carlo of
`√V_T` banded at `4·SE` with the quadrature's share measured separately at **below 1e-12** rather
than folded into one flat percentage; the band's **resolution** stated (band ~0.9% against a
convexity of ~9% of `K_vol`, a 10× margin — on the source's Heston-Nandi parameters the convexity is
~0.6%, *inside* the band, so an MC check there could not see a dropped convexity at all); the
discretisation **null averaged over six seeds**, because an oracle whose expected answer is "no
difference" is the one a single seed most easily fakes (T21); and a **log-strip cross-product** for
`K_var` — Heston option prices through T33's Fourier engine against the integrated-variance
transform, different code each side — asserted by its one-directional truncation signature
(`narrow < wide < exact`) rather than a tolerance.

The MC instrument uses numpy's generator rather than the tree's `UniformSource`, deliberately: an
oracle sharing an RNG with the production engines shares a component with them, and
`standard_normals(source.uniforms(n))` round-trips through Python lists at ~19× the vectorised draw —
enough alone to push the file past the `slow` ceiling (21.8s → 2.0s against a 10s budget).

## [0.53.0] - 2026-11-16

### Added — tranche 46, the SVI surface parameterization (project six opens)

`models/svi.py` (L3): Gatheral eq 3.20 as a closed form in five parameters, with the no-arbitrage
conditions that decide whether a fitted slice is admissible. First build of the volatility-surface
arc; `GATHERAL-2006` → `worked-from`.

**The quantity is TOTAL variance `w(k) = σ²_BS(k)·T`, and the spec was wrong about it before the
build.** The spec wrote Lee's wing bound as `b(1+|ρ|) ≤ 2` against *annualised* variance; ch 7 bounds
`σ²_BS(k,T)·T/|k|`, so the annualised form carries a `T` and the two agree **only at `T = 1`**. That
is the `τ=1` degeneracy living in a **formula** rather than a fixture — a one-year fixture would have
passed it silently. Caught by a dimension check before RED, ruled by Cowork, recorded as **C14**
(guarded: the Heston↔SVI path reprices end-to-end at a measured **0.0030%** against an 0.05% band,
while the same path with the `×T` omitted reads **81.1%**, asserted in a companion test). The
archived predecessor had the same defect shipped — it declares `w = σ²_BS × T` and then tests
`b(1+|ρ|) ≤ 4/T`.

**Neither `pdftotext` rendering shows eq 3.20's square root, and §4's dual-rendering rule cannot
see it** — both are faithful to a lossy extraction. What recovers it is two things the renderings
*do* carry plus one the source carries three chapters later:

* `√` survives extraction **123 times** elsewhere in the book, so 3.20 is not a generic glyph loss;
  what sits in its place is a large-math **vinculum**, which also appears over displayed fractions.
* The vinculum's **measured span** pins the *scope*: it begins at column 47 and covers the radicand
  at columns 51–66, while `ρ (k − m)` sits at columns 33–45, outside it. The v0.4.0 mis-nesting
  hazard is therefore covered by the extraction.
* The radical's **identity** is forced by ch 7: with it the wings are linear in `k`, without it `w`
  is quadratic, which violates Lee's model-independent `β*, α* ∈ [0,2]` unboundedly. §4's
  derivation-step independence, **across chapters** rather than across two steps of one derivation.

**The butterfly condition is derived, not transcribed.** The 2006 text carries none for SVI, and the
rigorous `g(k) ≥ 0` is Gatheral–Jacquier 2014, which is **not held** (acquire-source note in
`docs/to_go_deeper.md`). So the density comes from Breeden–Litzenberger on the fitted smile, and the
trap is in the first step: for a `(e^X − e^k)⁺` payoff `∂c/∂k = −e^k(1 − P(k))`, so
`e^k p = c'' − **c'**` and the reflex `p = e^{−k}c''` is wrong by `O(1)` (measured: 24.5% at the
money). Carried through with Black's `n(d₊) = e^k n(d₋)` it collapses to `p = g·n(d₋)/√w` with
`g = (1 − kw'/2w)² − (w'²/4)(1/w + 1/4) + w''/2`, confirmed three independent ways — against a
numerical second difference of Black prices (~1e-7), by integrating to **1.0 ± 3e-11**, and by
collapsing **exactly** onto the Black–Scholes normal density at `b = 0`, where `g ≡ 1`.

**A derived result replaced a constant the sweep showed was dead config.** The butterfly scan
originally widened the quoted span by 50%; coarsening it back killed nothing, and working out why no
fixture could exercise it gave the better answer: as `|k| → ∞`, `w' → b(1±ρ)`, `w'' → 0` and the skew
term → ½, so

    g(±∞) = ¼ − (b(1±ρ))²/16 ≥ 0   ⟺   b(1±ρ) ≤ 2

— **the asymptotic no-butterfly condition is exactly Lee's bound** (verified to 8 decimals on both
wings of three slices). So the tails are certified analytically and the interior numerically, the
margin is deleted, and what the pair does *not* cover — the region between the far quote and the
asymptotic regime — is named in the code rather than papered over, which is what makes the
Gatheral–Jacquier acquire-source note an earned one.

### Changed

`foundation/distributions.py` gains **`norm_pdf`**, promoted at its **third** consumer. Two L3
modules each carried a private copy, and the first one's docstring said *"§6's rule-of-two has not
fired, so it has not been promoted"* — true when written, false from the moment the second copy
appeared and nobody noticed. Both were the same expression in the same operation order, so the
consolidation is **bit-identical** (§4: claimable exactly when the refactor preserves op order), not
merely equal to a tolerance.

`models/svi.py`'s start heuristic is the second version. The first estimated each wing's slope from
the vertex to its endpoint, which has a degenerate case no obvious fixture shows: a **one-sided**
smile puts the smallest quote *at* an endpoint, so that side has no data, its slope is exactly zero,
`ρ₀` pins to its clip, and the solve walks out of evaluations at a cost of 2e-10 — a good fit
reported as non-convergence. Replaced by a one-sided estimate with `ρ₀ = 0`, deliberately
start-neutral; a sweep of 432 slices finds it beats a constant start on exactly **one**, which is now
its test.

### Verification

68 new tests (1784 → **1852**). Mutation: **applied 27/27, killed 26, survived 1**, and the applied
count is load-bearing — the first run was reported `26/27 killed` on **stale bytecode**, which
silently un-applies a mutation (Python compares mtime and size, and a patch written inside one mtime
tick is ignored), changing three verdicts. That is §9's "a mutant that did not apply is not a mutant
that died" one level down: not a mutant that failed to apply, but one that applied to a file the
interpreter then declined to recompile, with every row still printing a confident verdict.

The sweep found **five** unreached guards, all closed test-only: the fit's Lee refusal (the predicate
was tested, the fit *consulting* it was not — T33b's input-population degeneracy), `annualised_vol`
with no test at all, the minimum-quote count, the start heuristic, and the butterfly margin (deleted
instead). The surviving mutant coarsens the butterfly scan from 401 points to 3 and is **not a
hole**: SVI's density breaches are structurally wide — measured at **0.58 in log-strike**, ~200× the
scan step, with none narrower than 0.05 across a 324-slice sweep — so the failing case is unreachable
given `g`'s own smoothness. Asserted as a resolution oracle rather than argued.

## [0.52.1] - 2026-11-16

### Fixed — tranche 45, re-audit #4 remediation (test-only, no `src/` change)

Re-audit #4's **proactive invariant-6 sweep found five blind oracles on its first run** — the guard
present and correct in `src/`, and no fixture making it bite. All five closed the way the degeneracy
always is: a fixture where the two branches produce **different numbers**, so dropping the guard
fails a test.

| # | site | why it was invisible | the fixture now |
|---|---|---|---|
| F-4.1 | `engine/equity_vanilla.py` | the expired-option test struck the call at 105 against a spot of 100, so the fall-through **intrinsic was `0`** — the same number PV-0 gives | expired and **in the money** (85), so intrinsic 15 ≠ 0 |
| F-4.2 | `engine/heston_fourier.py` | the registered pricer's guard had **no test at all** — every test called `heston_call_price` directly | an engine-level expired **ITM** option, plus a live-option premise |
| F-4.3 | `engine/forward_cds.py` | no expired-swaption fixture existed | an expired option on a **live** underlying, struck deep in the money |
| F-4.4 | `engine/yoy.py` (fixed leg) | the matured test values the swap *after its last payment*, so every period is settled and the comprehension is empty either way | a valuation date **inside** the schedule |
| F-4.5 | `engine/yoy.py` (par annuity) | same cause | the same straddling fixture |

Severity, for the record: F-4.1/4.2/4.3 would have priced an expired in-the-money option at its
intrinsic instead of PV 0 — a live number for a settled trade. F-4.4/4.5 would discount a settled
period at **negative `t`**, a factor above 1, in an engine shipped since tranche 4; the new fixture
asserts that factor exceeds 1, so the defect is named rather than merely excluded.

**The matured YoY test is kept beside the new one**, because it asserts a *different* line — the
whole-swap invariant-6 guard — which the straddling fixture cannot reach. Not a degenerate fixture
left lying around: a narrower one, now labelled.

**Verified as the spec demands, with the applied count** (§9's new clause): all six sites —
the five plus `yoy.py:101` as a control — **applied 6/6, killed 6, survived 0**.

### The shape question, ruled

Not a new eleventh shape. The five split across two **existing** shapes and surfaced together only
because they share one guard: F-4.3/4.4/4.5 are the **tenth** (the boundary is never reached), while
F-4.1/4.2 are the catalogue's **first** — "a zero makes a factor invisible" — wearing the
invariant-6 guard's clothes. The fixture straddled correctly and the branches still coincided
*through a zero*. The catalogue is of mechanisms, not of the guards they are found behind.

### Charter — two edits, both corrections rather than additions

1. **§4: the tenth-shape premise sharpened.** The standing sweep asserted "the fixture straddles";
   its own first run proved that **necessary but not sufficient**. The general premise is that the
   two branches must *differ* — which for an option bypass additionally requires the expired option
   **in the money**.
2. **§9: the sweep's own instrument.** A whole-sweep "zero survivors" is meaningless without the
   applied count beside it. This run's first execution printed *"survivors: 0"* from 21 mutants that
   had all **failed on the command line** (a missing `pytest-timeout`), caught only because every
   row read "error" rather than "killed". The per-mutant form of that rule was old; the aggregate
   form is the dangerous one, because the sweep runs unattended.


## [0.52.0] - 2026-11-15

### Added — tranche 44b, base correlation (the credit arc's last build)

The market's quoting convention for index tranches: the single `ρ_B(K)` that, applied to the
**equity `[0,K]`** tranche, reprices the quote (`OKANE-2008` ch 20.2–20.3). The curve is
bootstrapped shortest-strike-first, and any `[K₁,K₂]` tranche is the **difference of two
base-correlation equity tranches**.

- `engine/base_correlation.py` — `HomogeneousPool`, `BaseCorrelationCurve`,
  `bootstrap_base_correlation`, `base_tranche_expected_loss`, `base_tranche_par_spread`. **L4
  because it inverts the engine**: each strike's correlation is found by repricing `[0,K]` through
  tranche 43's LHP pricer, which §2 puts beside the Jarrow-Yildirim calibrator.
- `products/synthetic_cdo_tranche.py` — `TrancheQuote`, at **L2 and not L1** though it is market
  data: a quote is *about* a `TrancheBounds`, and `market/` importing a product would be the upward
  edge `acyclic` forbids.

The legs are tranche 39's and 42's unchanged — the base-tranche survival satisfies
`credit_legs.SurvivalFunction`, so a base-correlation tranche reaches the same arithmetic as every
other credit product in the library.

### The spec's headline oracle was vacuous, and the restatement is the tranche's substance

The spec asked for a refusal of a non-monotone positive quote set. There is no such thing:

    EL([K₁,K₂]) > 0  ⟺  Q < 1 and decreasing  ⟺  protection > 0  ⟺  par > 0

is exact, so **every positive quote bootstraps to a monotone curve by construction** — measured
down to a **1bp** mezzanine quote. Asserting it would have the suite reject something that cannot
exist: an oracle that cannot fail, on the requirement base correlation exists to satisfy. *(Raised
at build and blessed by Cowork before the oracle was written, rather than at the PR.)*

So the monotonicity is asserted as **the theorem it is** — over extreme quotes, with the
implication chain written out link by link, so the suite documents *why* it holds and why compound
correlation cannot promise it. And the non-vacuous guard is the other side of the same iff:

**a quote with no base correlation at all.** Given `ρ_B(3%) = 0.25`, the `[3%,7%]` par spread
reachable over `ρ ∈ (0.02, 0.98)` is `[−729, +1467] bp`, so a 1600bp quote has no root. It is
refused in the bootstrap's own vocabulary, with the admissible range *reported* — because
`bracketed_root`'s docstring says a non-straddling bracket is the caller's bug and that the caller
should have refused an out-of-range target first. A second oracle asserts the message names the
tranche and the reachable interval and that the root-finder's vocabulary does not leak: dropping
the guard still *raises*, so "it raises" would have passed.

### Oracles

The **known-curve round trip** is the anchor: a non-flat curve (ρ_B = 0.25/0.45/0.62) generates a
realistic skew — **4254 / 661 / 131 bp** — and bootstrapping those quotes back recovers the curve to
**1e-12**, with the quotes repriced through it to 1e-10. Plus the **single-strike reduction** onto
tranche 43 (one `[0,K]` quote at `ρ` bootstraps to exactly `ρ`), and the implication chain above.

### Interpolation is deliberately absent

Every tranche priced here is bounded by *quoted* strikes, so `correlation_at` **refuses** an
unquoted one. Interpolating `ρ_B` is a modelling choice the quotes do not pin, nothing in scope
consumes it (§6), and doing it silently would be a plausible wrong number (§3.4). Like tranche 43's
index-handle question, the ruling dissolved rather than being made — it waits for a bespoke tranche
to pull it.

### A guard deleted for being unreachable

A first draft checked the quotes were in increasing strike order. Contiguity forces
`attachment == previous_detachment` and `TrancheBounds` forces `detachment > attachment`, so the
detachments increase **already** — `detachment <= strikes[-1]` cannot hold. Two guards implying a
third makes the third dead code (the T15 rule); it was deleted and the suite now pins the
implication, so an out-of-order quote set is still refused, just not by the guard a reader might
expect.

### The `slow` gate again, and the third local-vs-CI miss

Reported green on a 3.4s local reading; **Windows failed at 11.2s against 10.0s** while Linux
passed. Third time this arc, and the first *since* the §9 line saying CI is the measurement went in
— which I had written and then not applied to my own report.

The fix is the number rather than a trim: a base-correlation tranche costs **two** `Φ₂` evaluations
per step (one model per leg of the difference) where a single-correlation tranche costs one, so the
LHP's 800 steps was simply the wrong count here. Against a 1600-step reference the par spread
converges at second order — 2.8e-06, 7.0e-07, 1.7e-07, 3.3e-08 at 100/200/400/800 — so **200** is
committed at **7.0e-07**, four significant figures tighter than a tranche is quoted, for 3 ms
against 10 ms. The file runs 1.1s and projects to ~4s on Windows.

### A false claim of mine, and the third bite of the tenth shape

I told Cowork this tranche's fixture straddled the valuation date, as tranches 43 and 44a's do.
**It did not.** The schedule began *at* `VAL`, there was no straddle assertion, and dropping the
invariant-6 filter in `base_tranche_par_spread` **survived the whole file**. Found only because
Cowork declined to take the report's word — a reported premise it had not re-derived is content and
not an oracle (§5, one level up) — and I re-derived it before marking the PR ready.

That makes §4's tenth degenerate shape **bite a third time** (tranches 40, 42, and here), which is
the escalation the charter flag arms: the ruling that the counter stood at two bites rested on my
claim, and the claim was wrong. Fixture fixed, the straddle asserted as a premise, and the filter
pinned — the mutation now dies four ways.

### Mutation sweep

11 mutants, **no survivors**. Four single-killers, all refusal guards with one condition each —
inherent — except the reachability refusal, which is the tranche's non-vacuous arbitrage oracle and
was hardened to two.


## [0.51.0] - 2026-11-15

### Added — tranche 44a, the exact loss distribution by recursion

The LHP (tranche 43) is the infinite-pool *approximation*; this is the exact distribution for a
finite homogeneous pool (`OKANE-2008` ch 18). Conditional on the market factor the credits are
independent, so adding one credit at a time convolves its loss into the buckets.

- `models/loss_recursion.py` — `RecursionModel(market, entity, beta, recovery, pool_size)` with the
  conditional and unconditional distributions, the capped expected loss and the tranche
  composition.
- `engine/loss_recursion.py` — **a second pricer for the same product.** A `SyntheticCDOTranche`
  against `LHPModel` gets the approximation, against `RecursionModel` the exact answer: the
  Black-vs-Heston shape the `(product, model)` registry key exists for, with the registry refusing
  a duplicate pair so neither can silently replace the other.

**The arithmetic is not duplicated.** A tranche is a CDS on its own survival whichever model
produced it, so the legs and the pricer body are tranche 43's, reached through a new `TrancheModel`
protocol — two present consumers sharing a shape, not a word.

### The anchor closes the arc's loop

Tranche 43's closed form is this recursion's `N → ∞` limit, and the convergence is **exactly
`O(1/N)`** — error ratios of 1.81, 1.97, 2.00, 1.99 per doubling of the pool, measured before a
line was implemented. So the approximation built first becomes the reference the exact model is
checked against, by a *rate* rather than a tolerance. The conditional distribution is additionally
pinned **pointwise against the binomial in closed form**, before any integration over `Z` — which
is where a dropped or swapped branch is visible, since afterwards it would still sum to one.

### Config moved onto the models, because cost is a model property

The protection-leg step count now lives on the model rather than the engine (§3.5). A closed-form
survival is cheap to sample, so the LHP commits 800; the recursion costs a 96-node sweep over an
`N`-credit convolution *per step*, so it commits **100**, where the measured error is 8.1e-07 —
four significant figures tighter than a tranche is quoted. The engine cannot know this, and one
hard-coded number would make the LHP needlessly coarse or the recursion unusably slow.

Two cost fixes beyond that, both exact: the two caps in a tranche's expected loss now share **one**
market-factor sweep (the subtraction moves inside a linear expectation), and the whole test file
fell from **44s to 5.3s**.

### Vectorised, because CI is not my machine

The `slow` gate failed on **both** runners — 18.3s on Linux, 10.1s on Windows, against 10.0s —
while the file measured **5.3s locally**. That is §9's "a timing reading is only valid where it is
taken", and the second time this arc my local headroom was not headroom (tranche 42 was the first).

The charter's remedy is to vectorise, so the recursion now advances **all quadrature nodes as one
array**: its only `Z`-dependence is the scalar `q = p(T|Z)`, so the convolution is identical at
every node. `nodes × N` interpreter steps become `N` array operations, and the file runs **1.1s**.
The outer sum stays `math.fsum` over weight-times-value, exactly as `standard_normal_expectation`
does it, so only the inner convolution was reordered — agreement with the scalar form is exact in ℝ
and **≤1 ULP**, which is the claim available and the one made.

Two things that fell out of doing it properly:

- **`scipy.special.ndtr` would have vectorised `Φ` in one call, and was rejected.** It would mean
  this model's `Φ` is not the `Φ` the copula and the LHP use — a convention divergence of exactly
  the kind the register exists to prevent — and it bought **nothing**: with the library's own
  `norm_cdf` called per node the file still runs 1.1s, because the convolution was the cost, not
  the 96 scalar evaluations.
- **There are now two implementations of the recursion**, and the binomial anchor pinned only the
  scalar one while pricing ran the vectorised one. An anchor on the wrong implementation is
  worthless, so the two are asserted equal and the binomial anchor reaches the fast path
  transitively.

### The duplication the sweep exposed, collapsed — `models/tranche_loss.py`

`tranche_expected_loss` and `tranche_survival` were written out in **both** models, identical code
over a differing `expected_loss_capped`. That is why the width-division hole was found twice: at
tranche 43 (fixed there, against the LHP) and again at tranche 44a in the second copy. The lesson is
not "apply the shared oracle to both implementations" but **stop having two implementations**.

Extracted as **free functions over a `TrancheLossModel` protocol**, not a base class or mixin: a
mixin is the template-method polymorphism §6 keeps these frozen value objects away from, and
functions-over-a-protocol is the library's own shape (`credit_legs`, the registry, tranche 42's
`SurvivalFunction`). *(Ruled by Cowork.)* The width oracle moved onto the shared function and is
parametrised over both models, so **one mutation of the division now fails four tests across both**
— the hole cannot reappear in a copy that does not exist.

Proved a **pure refactor**: the shipped LHP path is **bit-identical over 24 values** (PV, par
spread, annuity, protection leg, survival and expected loss across four tranches), which is the §4
bar for touching tranche 43.

It cost the recursion's fused two-cap sweep, measured at **1.74×** the shared form — affordable
(projected 6.4s on CI against a 10s gate) and worth it for one composition.

### Mutation sweep

12 mutants. Two survivors, and both are lessons about where an oracle *lives* rather than about the
recursion:

- **the width division in `tranche_survival` — the same hole tranche 43 had.** It was fixed there,
  but the oracle lives in the LHP's file against the LHP model, so the recursion's own copy was
  never covered. A second implementation of a shared composition needs the shared oracle applied to
  **it**, not merely present somewhere in the tree. (The composition is genuinely duplicated
  between the two models rather than shared — worth a look.)
- **the committed step count — because the protection-leg test TELESCOPES.** That test reads
  `model.protection_steps`, so it asserts the engine against the grid the engine actually uses:
  honest about *which* grid and, for exactly that reason, blind to whether the grid is adequate. A
  mutation setting the count to 2 moves both sides together and survived. Two different questions,
  conflated — now separated, with the resolution asserted against an independent finer reference.

### A third concrete annotation denying a correct reuse

`TrancheSurvival` was annotated `LHPModel` while the recursion model is a legitimate consumer —
pyright caught it. That is the **third** time in this arc a correct reuse was contradicted by a
concrete type (tranche 42's `risky_pv01`, tranche 42's `BasketSurvival`, this): widened to a
`TrancheLossModel` protocol, so the signature states the shape the code actually needs.


## [0.50.0] - 2026-11-15

### Added — tranche 43, the LHP synthetic-CDO tranche

Vasicek's infinite-homogeneous-pool limit of the one-factor Gaussian copula — the model that
defined the correlation market (`OKANE-2008` ch 16.2, 16.7.1). The law of large numbers collapses
the conditional loss to a point mass, so the whole loss distribution is closed form.

- `models/lhp.py` — `LHPModel(market, entity, beta, recovery)` with `A(K)`, `E[min(L,K)]`, the
  tranche expected loss and survival; plus `TrancheSurvival`, which satisfies
  `credit_legs.SurvivalFunction`.
- `products/synthetic_cdo_tranche.py` — `SyntheticCDOTranche(bounds, schedule, spread, notional,
  reference_entity)` and `TrancheBounds(attachment, detachment)`.
- `engine/synthetic_cdo_tranche.py` — `@register(SyntheticCDOTranche, LHPModel)`. **A tranche is a
  CDS on its own survival**, so both legs are reused unchanged: the premium leg paying its spread
  on the amortising outstanding notional *is* `risky_pv01(Q)`, and the protection leg is tranche
  42's `−dQ` form.

### Eq 16.1 is our `Φ₂`, and the correlation argument is `−β` — verified, not assumed

The check Cowork asked for, and it was not cosmetic. Against an independently computed market-factor
integral, **`−β` agrees to 1e-14 while `+β` is 62% to 240% wrong**. A companion test asserts the
wrong sign *is* materially wrong, so the check is proven to have power rather than merely passing.

### The closed form is undefined at exactly the two points its headline identity needs

`A(K)` takes `Φ⁻¹(K/(1−R))`, so it needs `0 < K < 1−R` **strictly**, and `inverse_norm_cdf` refuses
the endpoints rather than returning `±∞`. But the capital-structure partition
`Σ EL([aₖ₋₁,aₖ]) = (1−R)p` runs from `K = 0` to `K = 1`. So the two branches — `K ≤ 0 → 0` and
`K ≥ 1−R → (1−R)p`, the cap being unable to bind because **the maximum portfolio loss is `1−R`, not
1** — are load-bearing rather than defensive. With them the partition is exact to 2e-16; without
them it raises.

### A truncation defect in my own reference, caught by a plateau

The two-method cross-check needed a **kink-split** integral: `min((1−R)p(T|Z), K)` has a kink where
the cap binds, and Gauss-Hermite is stuck near 1e-3 there at any practical node count — six orders
worse than the split form, so using it would have forced a band wide enough to hide a real defect.

The split reference then disagreed with the closed form by 2.41e-10 at one strike, and the gap
**did not move** at 160, 400 or 900 nodes. That plateau is the tell (§4's rate detector, pointed at
the reference itself): not quadrature error but **truncation** — the tail window was `[z*, z*+12]`,
and where the kink sits deep in the left tail that stops short of the upper tail. With an absolute
upper limit the worst gap over the whole grid is **1.7e-14**, so the band stayed tight instead of
being loosened around a reference bug.

### Both of the spec's `Ask:`s

- **The index-handle question dissolves rather than needing a choice.** The LHP is the infinite
  homogeneous limit, so there is no per-name structure to reference: the pool's identity *is* its
  one default curve, which `CurveSet.survivals` already keys by `ReferenceEntity` (tranche 41b). An
  index is a reference entity for keying purposes — no new type, and a new one would have no second
  consumer (§6).
- **The coupon convention** is the running-spread-on-outstanding-notional form the spec describes.
  The standardised-coupon/upfront detail is not in scope and is raised rather than assumed.

### Mutation sweep

21 mutants. Three survivors, and they shared one cause worth naming: **every oracle was relative.**
The par reprice, the linearity in the spread and the equity/mezzanine/senior ordering all survive a
uniform rescaling of either leg, so between them they pinned the *shape* of the tranche's economics
and nothing about its *level* — §4's magnitude trap in a new place, a complete set of assertions
about the wrong aspect.

- the tranche survival's **width division** (`Q = 1 − EL/(K₂−K₁)`) was unpinned: dropping it leaves
  `Q` in `[0,1]` and decreasing, and the premium-leg test fed `risky_pv01` the *same* mutated curve,
  so it telescoped;
- the protection leg applying **`(1−R)` a second time** — the recovery is already inside the
  expected tranche loss — rescales it by 0.6 and the par reprice rescales with it. Closed by an
  independent discounted-loss sum, with a companion asserting the doubled version is 40% off;
- the **pool-entity mismatch** check survived because the only fixture used an entity *absent* from
  the snapshot, which already fails on the curve lookup. What the check guards is an entity that
  **is** present but is not the model's pool — closed with a two-pool snapshot.

An earlier draft also asserted the naive integral's error sequence was *non-monotone*. It is at some
fixtures and not at the one the test uses, so the claim pinned an accident — the same mistake made
at tranche 41 and recorded there. Twice is enough to state it: assert the **size** of an error, not
the shape of its approach, unless the shape itself is the theorem.


## [0.49.1] - 2026-11-15

### Changed — tranche 42b, the gates-gate covers the standalone CI gates too (tooling only)

Tranche 38's `verify.py gates` closed "a gate stops running and nothing notices" — but only for
`verify.py` subcommands. `ci.yml` also runs **`ruff`** and **`pyright`**, and `pyright` caught a
tranche-42 type defect a local run missed; either could have been deleted with nothing noticing,
because neither the charter nor the gate knew they should be there. The same defect, one category
wider, and statically decidable.

§9's gate list now names `ruff` and `pyright`, and the gate requires `ci.yml` to invoke each
**named** gate whether it is a subcommand or a standalone step. The reverse direction stays scoped
to subcommands: classifying an arbitrary `ci.yml` step as gate-or-scaffolding needs judgement, and
a check needing an exemption list cannot fail (the τ=1 over-reach, declined). So a standalone gate
is enforced by being *named*, and naming it is the deliberate act.

### An invocation is not an installation, and that is the design

Both `ruff` and `pyright` appear in the workflow beyond their gate steps, `ruff` in the `pip
install` line. A check asking only "is the name in the file?" would pass with the `ruff check` step
deleted — **a gate satisfied by its own dependency list**. Install steps are skipped, and the
seeded-drift test deletes the *invocation* while leaving the *install* in place, which is the case
that separates a real check from a self-satisfying one. (`pyright` cannot demonstrate it: it
arrives via `npx`, so it has no install line — a first draft of that test assumed both names were
on it and failed on `pyright` for a premise that was mine, not the gate's.)

### `pytest` is deliberately not named, which is a finding

The spec listed it among the three to name. It has **no direct `ci.yml` step at all**: it runs
*through* `verify.py tests` and `verify.py slow`, both already named and invoked. Naming it would
either fail outright or, under a looser check, pass on the `pip install` line. Raised for Cowork
rather than guessed, and `tests` already covers it.

### The charter's gate list must stay a list

The first draft of the §9 edit put the explanation *inside* the list sentence, and a backticked
word in that prose parses as a gate name — so the gate immediately flagged a gate that does not
exist. **The gate caught the charter edit that was adding the gate.** The explanation now sits past
the parse boundary, and says so.

### Mutation sweep

Four mutants of the extension, each killed: ignoring non-`verify.py` names (7), counting install
lines as invocations (2), returning no standalone invocations (7), dropping the
named-but-not-invoked direction (4).


## [0.49.0] - 2026-11-15

### Added — tranche 42, the first-to-default basket (the first product through the copula)

A first-to-default basket **is a CDS on the basket survival curve**, which is what makes the method
exact and the reuse large (`OKANE-2008` ch 15.1–15.2):

    Q_B(t|Z) = Π_i Q_i(t|Z)        — conditional independence (tranche 41's copula)
    Q_B(t)   = ∫ Q_B(t|Z) φ(Z) dZ  — Gauss-Hermite over the market factor

- `models/basket.py` — `BasketSurvival(copula, nodes)` and `BasketModel(copula, recovery)`.
- `products/first_to_default_basket.py` — frozen; at least two names, no duplicates.
- `engine/first_to_default_basket.py` — `@register(FirstToDefaultBasket, BasketModel)`, with
  `par_spread`, `basket_annuity` and `basket_protection` as named surface because the oracles are
  about *them*. Deterministic, so `standard_error` stays `None`; the Monte-Carlo is the
  cross-check, not the engine (the tranche-41 pattern).
- `market/credit_legs.py` — `protection_leg_from_survival`, the `−dQ` form, on a new
  `SurvivalFunction` protocol.

### Two of the spec's instructions did not survive measurement

1. **`credit_legs.protection_leg` cannot be reused for the basket.** The spec said to reuse the
   legs whole with "no leg re-transcription". The premium leg does reuse — `risky_pv01` reads only
   `survival(t)`. The protection leg cannot: its correctness rests on the hazard being constant
   between pillars, true of a bootstrapped `SurvivalCurve` and false of `Q_B`, whose implied hazard
   varies continuously. Fed `Q_B` it **plateaus 1.56e-03 from the truth** — identical at 50 and 200
   points per segment, §4's signature of a model error rather than a discretisation one. So the
   basket uses the `−dQ_B` form the spec's own displayed formula writes, which needs no hazard and
   converges cleanly (7.3e-06 / 4.5e-07 / 2.7e-08 at 50/200/800 steps).

   The two are **not** a duplication to collapse: they are the right quadrature for two different
   classes of survival function. Which applies where is `docs/conventions_register.md` C13's
   business, and extending that row is Cowork's — raised, not decided.
2. **Recovery is on the model, not the product.** The spec listed `R` as a product field; tranche
   38 ruled the opposite and that ruling is the older and more considered — `R` is a market
   assumption, and a recovery on the product would let two baskets on the same names carry
   different ones. `BasketModel` composes the copula exactly as `CreditSwaptionModel` composes
   `CreditModel`.

The spec's layer question (`Q_B` at L1 or L3) is settled by the import direction: it is built from
a `GaussianCopula` (L3), so an L1 module holding it would be the upward edge `acyclic` forbids.

### The ρ = 1 limit had to be taken analytically

At `|β| = 1` each conditional survival is an **indicator** in `Z`, so their product is a step, and
Gauss-Hermite cannot integrate a step — tranche 41's unit-β finding, one level up. Measured: `Q_B`
came out **6.1% below** `min_i Q_i` at `t = 5` and, at `t = 3`, **above** it, which is an impossible
probability — the Fréchet upper bound breached. The closed form is elementary (the product is
`1{Z > max_i C_i}`, so `Q_B = 1 − Φ(max_i C_i) = min_i Q_i`, exact for `β = ±1` alike). A basket
**mixing** unit and sub-unit loadings is refused: the integrand is then a step times a smooth part,
the quadrature is silently wrong there, and nothing needs it.

**The Fréchet sandwich is what caught it.** `Π Q_i ≤ Q_B ≤ min Q_i` is a requirement oracle that
survives transcription — and it does something a value-against-value check cannot: it recognised
`Q_B > min Q_i` as *impossible* rather than merely unexpected.

### Oracles

Single-name reduction `Q_B = Q₁` (exact in ℝ, **≤4 ULP** — see below), the `ρ = 0` product and
`ρ = ±1` widest-name limits,
the Fréchet sandwich at four horizons, monotonicity, the rate-asserted market-factor integral, the
par reprice, linearity in the spread, and the spread bounded below by the widest single name and
above by their sum (210.96bp ≤ 491.28bp ≤ 554.27bp). Plus a Monte-Carlo cross-check at 40k paths
whose **realised correlation pins `β_iβ_j` pair by pair** (tranche 17 — a factorisation is pinned
where it is observable, not through a price) and whose realised variances pin the `√(1−β²)` weight.

### Two of my own over-claims, caught by CI rather than by reading

- **`risky_pv01` was annotated with the concrete `SurvivalCurve`** while the basket passes a
  `BasketSurvival`. The "the premium leg reuses cleanly" claim was true at runtime and the type
  said otherwise — pyright caught it, and the fix is the honest one: the premium leg takes the
  `SurvivalFunction` protocol (it reads only `survival(t)`), while the pillar protection leg stays
  concrete because it genuinely needs `.times` and `.hazard`. The signatures now *encode* the
  finding above rather than contradicting it.
- **Two reductions were asserted bit-exact and are not.** `Q_B = Q₁` and
  `Q_B = min_i Q_i` are exact identities in ℝ, but each compares *different computations* — a
  64-node Gauss-Hermite sum over `Φ` against `exp(−∫λ)`, and a `Φ`/`Φ⁻¹` round trip. They agreed
  to the last bit on one platform, so the first draft used `==`; **CI failed on both Linux and
  Windows at 1 ULP**, which is what §9's "tolerance-based oracles are a hard rule" exists to
  prevent. Now asserted at ≤4 ULP, with the one place bit-equality *is* available (the engine
  delegating to `risky_pv01` with identical arguments and ordering) kept as `==` and the contrast
  stated — bit-equality is a claim about the computation, not about the mathematics.

### Performance — the `slow` gate caught redundant work in the engine, not just the tests

`CreditName.conditional_default` recomputed its `Φ⁻¹` threshold and its `√(1−β²)` weight on every
call, and neither depends on the market factor — so a 64-node quadrature did them 64 times over,
at each of the protection leg's 800 steps. The gate failed on Windows at 10.6s against a 10.0s
budget; locally the file ran in 3s, which is the §9 point that a timing reading is only valid where
it is taken.

Hoisted into `conditional_default_at(horizon)`, a closure over the `Z`-independent parts, with
`conditional_default` now defined as that closure evaluated at a point — so eq 13.4 is still
**transcribed once** and the basket composes it rather than re-deriving it (§4). **Bit-identical
over 150 (name, t, z) points**, and 4.6× faster: the two protection-leg convergence tests went
from 0.69s to 0.15s each.

### Mutation sweep

22 mutants, full suite each, no `-x`. Four survivors, **every one a fixture gap and not a bug** —
and the last is the one worth recording:

- **the notional scaling** was unpinned: the par reprice is ~0 at any scale, and the linearity check
  compares differences, so dropping the factor entirely survived;
- **the contract's own entity list** was unpinned, because the fixture gave the basket *exactly* the
  copula's three names, making "the contract's subset" and "all of them" the same operation. A
  fourth name in the copula separates them;
- **the invariant-6 filter** was unpinned, because the premium schedule began at the valuation date
  — so no period was historical and the filter had nothing to filter. That is *precisely* the blind
  spot that hid tranche 40's 73% annuity overstatement, found in tranche 41b. Having fixed that one
  tranche earlier, this tranche's fixture reproduced its shape: the convention "schedules start
  today" is systemic, not per-tranche, and the sweep is what catches it;
- the `−dQ` leg's non-positive-horizon guard is a **proved equivalent** mutant — every survival
  function is flat at 1 below zero, so each difference is exactly zero and the sum is `0.0` with or
  without the guard (bit-identical over 45 combinations). Kept for symmetry with `protection_leg`,
  where the same line is load-bearing, and the test says plainly that it pins the outcome and
  cannot kill the mutation.


## [0.48.2] - 2026-11-14

### Added — tranche 41c, `verify.py conventions` (tooling only)

The conventions register had **no mechanical integrity check**, which is how tranche 41b found two
`C11` rows — the tranche-39 protection-leg row took an id the `model.market` row already held, and
**two review passes read past it, including the Cowork pass that ratified "C11" by name**. The
register is cited from `src/` docstrings, `OPEN.md`, the CHANGELOG and CLAUDE.md, so a duplicate id
silently redirects every one of those citations. Statically decidable, no exemption list possible,
so gateable under §9's boundary where the τ=1 fixture guard was not.

`verify.py conventions` asserts: unique ids within each table and no overlap between them, a
**gap-free ascending** live run, and that every `C<n>` the register mentions resolves to a row.
Listed in §9 and invoked in `ci.yml`, so the tranche-38 gates-gate covers it.

**The live floor is read out of the table, not pinned.** The spec said the live sequence starts at
"the stated floor… C10"; it actually starts at **C7** — the register's "new live rows continue at
C10" note is about where *new* rows go, and C7–C9 are live too. Hard-coding 10 would have made the
gate wrong about the first three rows it exists to protect.

**Out of scope deliberately:** whether a row's *claim* is true. That a `guarded` row really is
guarded needs judgement, judgement needs an exemption list, and a gate with an exemption list
cannot fail — the τ=1 guard's lesson.

### Fixed — the register's own layout

The live table was split in two by a stray blank line and by the C10 addendum sitting inside the
table body, so the rows after the break rendered without a header. C13 also sat between C10 and
C11, because tranche 41b's renumber edited the id in place. Reordered to C7–C13, one table, with
the addendum below it — and the ascending check now holds that.

### Mutation sweep

Seven mutants of the gate, each killed. Two survivors on the first pass, both of them findings
about the gate itself:

- **a live row reusing an *archived* id** was unchecked by any test. The archived ids are kept
  stable precisely so a live row may not reuse one, and renumbering a live row to `C1` also trips
  the ascending and gap checks — so the test asserts the *specific* message, or it would pass for
  the wrong reason.
- **the row-header stripping in `register_citations` was dead code.** It existed so a row's own id
  would not count as a citation, but `known` is built from those same headers, so a row id
  resolves to itself by construction and could never dangle. Deleted (§6, the T15 rule), and the
  function renamed to `referenced_convention_ids` to say what it now means — marginally stronger
  too, since a malformed header like `| C99X |` does not parse as a row but is still mentioned, so
  it surfaces instead of vanishing.


## [0.48.1] - 2026-11-14

### Changed — tranche 41b, survival curves keyed by reference entity

A §3-conformance refactor the multi-name products force. `CurveSet.survivals` was keyed by
**`Currency`**, so N names in one currency were **not representable in the snapshot** — which is why
tranche 41's copula held its per-name curves on the model, and why tranche 42's basket engine would
have read market data from outside `model.market`. The currency key conflated two things: a CDS has
a **currency** (whose discount curve it uses) and a **reference entity** (whose survival curve it
uses).

- `market/survival_curve.py` — `ReferenceEntity(name)`, mirroring `EquityAsset(ticker)`. Just a
  name: no sector, seniority, or currency-of-issue, which would be a hierarchy nothing asks for.
- `CurveSet.survivals` re-keyed, `survival(entity)` replacing `survival(currency)`.
- `CreditDefaultSwap`, `ForwardCDS`, `CreditDefaultSwaption` and `CreditName` carry an explicit
  entity; the T38–T40 engines look the curve up by it. The **discount** curve stays currency-keyed.
- `products/forward_cds.py` — **`ForwardPeriod(schedule, start)`**, the pair both forward products
  carry. The bundle is *compelled*: the new entity field would otherwise put both at six fields,
  over §7's limit, which is un-bundled vocabulary rather than a reason to raise it. It also removes
  a real duplication — the "split must be a schedule date" and "must precede the maturity"
  validations were written out twice, once per product, which is the duplicated-guard shape §4
  warns about. One definition now, so one fixture suffices.

### Oracles

- **Bit-identical single-name path.** The pre-change values were captured as **hex literals and
  committed in the RED test** before the re-key, which is what makes it a pre-registration rather
  than a re-run: a refactor compared against a freshly recomputed number proves only that two runs
  of the same code agree. All of the CDS PV, risky PV01, protection leg, par spread, the forward CDS
  and the swaption reproduce exactly.
- **The capability that was impossible before**: two distinct entities in one currency, each CDS
  pricing against its own name's curve, with the two curves asserted materially different (2.41×)
  so "each sees its own" has content.
- The two keys are now **independent**, asserted directly: the same name in another currency finds
  the same survival curve, and an unknown **name** fails however familiar the currency. The old
  test ("a survival curve for another *currency* is a PricingFailure") was exactly the conflation
  this removes, so it was replaced rather than patched.

### Fixed — a real defect in tranche 40's forward engine, found by this tranche's sweep

`_legs_to` summed `schedule.periods()` **whole**, so a seasoned contract's already-paid premiums
entered the annuity and were discounted at a negative `t` — giving factors **above 1**, which
compounds a settled payment forward. On a schedule straddling the valuation date by two years the
annuity read **4.81 against a correct 2.78, a 73% overstatement**. This violates invariant 6 ("a
flow on or before `valuation_date` is excluded and never discounted with non-positive `t`") and
bypassed `is_historical`, the single atom every pay-boundary site must use
(`docs/conventions_register.md` C10).

It was invisible because **every tranche-40 fixture begins at the valuation date**, so no schedule
had a historical period: the filter had nothing to filter. The oracle now is a cross-product with
exact economics behind it — once the forward start has passed, a forward CDS **is** a spot CDS over
the remaining window, and the two agree to 2.2e-16 with different code on each side.

### Mutation sweep

8 mutants, full suite each, no `-x`. Reverting either engine's lookup to the currency key dies 36
and 19 times; returning *any* curve instead of refusing an unknown entity dies 6 times. One
survivor, which is what uncovered the defect above: `ForwardPeriod.maturity` returning the **split**
instead of the end survived, because it only matters when the forward start has passed and the
maturity has not — and no fixture was seasoned. Chasing it found the invariant-6 violation; the
seasoned cross-product now kills it 4 ways.


## [0.48.0] - 2026-11-14

### Added — tranche 41, the one-factor Gaussian copula (the multi-name half opens)

`models/gaussian_copula.py` — `CreditName(beta, survival)` and `GaussianCopula(market, names)`,
with the threshold `C_i = Φ⁻¹(1−Q_i)`, the asset correlation `ρ_ij = β_iβ_j`, the conditional
default `p_i(T|Z)`, the pairwise joint `p_ij = Φ₂(C_i,C_j;ρ_ij)`, and the same joint by the
conditional-independence integral. No product and no `price()` — tranche 42's basket pulls it.

`CreditName` bundles the pair **not** to meet the §7 field limit (`(market, betas, survivals)`
would be three fields) but because parallel tuples make a mis-pairing expressible: name 0's beta
against name 1's curve is a wrong number with no symptom. Pairing them makes it unrepresentable
rather than merely tested.

### Oracles

- **The anchor is marginal recovery through the market-factor integral**: `E_Z[p_i(T|Z)] = 1−Q_i`
  to 1e-10, a genuinely different computation from the threshold that produced it. Its companion
  half, `Φ(C_i) = 1−Q_i`, telescopes by construction and is labelled as a guard.
- The three exact limits (`ρ = 0` product, `ρ = 1` minimum, `ρ = −1` Fréchet lower bound), each to
  1e-10, with a generic-`ρ` companion asserted strictly between the first two.
- **Φ₂ against the conditional-independence integral** — two genuinely different computations of
  the same joint, which is also the statement of §13.4.
- The threshold against `scipy.special.ndtri`, the conditional probability against an independent
  transcription of 13.4, monotonicity in the market factor, and the joint bounded by each marginal.

### Fixed — L0: a Gauss-Hermite rule that cannot be built is now refused

`foundation/quadrature.py` (tranche 12b) returned a **silently wrong answer** for large node
counts. `numpy.polynomial.hermite.hermgauss` loses its weights to underflow: measured on numpy
2.4.3, correct to 370 nodes, and at **371 every weight is `0.0`, so `E[1]` comes back as `0.0`** —
finite, plausible, and 100% wrong, which is strictly worse than the `NaN` that appears from ~380.
Either way a caller asking for a finer rule than the library can build received a number instead of
a refusal, against §3.4.

The guard asserts this module's **own stated invariant** (`Σwᵢ = E[1] = 1`, which its docstring
already names as an oracle) rather than a hard-coded 370, so the usable range can move with numpy
without the guard moving. Found because tranche 41's market-factor integral wanted 256 nodes at
high factor loadings — inside the range, one refinement from the cliff.

### Fixed — the unit-β marginal is taken analytically

At `|β| = 1` the conditional default probability is an **indicator** in `Z`, and Gauss-Hermite
cannot integrate a step: measured 13%, 18%, 19% and 5% wrong at 16, 64, 256 and 370 nodes —
non-monotone, and never right at any count. The expectation is elementary instead (`β = +1` gives
`Z < C`, `β = −1` gives `Z > −C`, both `Φ(C) = 1−Q_i`), so it is computed directly. This is the
configuration the `ρ = ±1` limits make natural, so the naive route would have been a silent 18%
error in a case the model invites. A companion test asserts the quadrature *would* be materially
wrong and does not converge, which is what makes the special case necessary rather than assumed.

### Two of the spec's premises did not survive measurement

1. **Φ₂ is the accurate side at high correlation, not the integral.** The spec expected Φ₂ to
   degrade as `|ρ| → 1` and the market-factor integral to be the reference there. Against
   `scipy.stats.multivariate_normal`, at `ρ = 0.9801` **Φ₂ is accurate to 8.9e-16** while the
   integral at the committed 64 nodes is **1.8e-02** out — the integrand approaches a step as
   `β → 1`, so the quadrature is what degrades. Φ₂ is exact at `ρ = ±1` by construction, being
   derived from Plackett's identity rather than a quadrature in `ρ`; tranche 36b's `|ρ| ≤ 0.99`
   regime concerns deep-tail *values* (`Φ₂ ≳ 1e-8`), and these probabilities are ≈0.05.
2. **The `ρ = −1` reduction is vacuous on a low-default fixture.** `max(p_i + p_j − 1, 0)` is `0`
   when `p_i + p_j < 1`, which an implementation returning zero also satisfies. The fixture carries
   deliberately high hazards for that case, with `p_i + p_j > 1` asserted as the premise.

An earlier draft of this suite also asserted that the marginal-recovery convergence was
**non-monotone** in the node count. It is at one fixture (`C = −2.5`) and is not at the one the
test uses (`C = −2.33`), so the claim would have pinned an accident; replaced by the robust
statement — the gap between a coarse rule and the committed one, 1e07×.

### Mutation sweep

20 mutants, full suite each, no `-x`. One survivor and three load-bearing single-killers:

- swapping `C_i` and `C_j` in the joint **survives**: `Φ₂` is symmetric in its arguments, so the
  mutation is **exact in ℝ** and differs by **≤2 ULP** (measured over 216 argument combinations) —
  the §4 reordering category, not a hole. Worth noting that the suite's own symmetry assertion is
  what makes it unobservable: asserting an invariance creates an equivalent-mutant class under it.
- the unit-β indicator branch (direction, and its removal) had one killer each → 2, via the new
  unit-β marginal recovery.
- integrating **name 0 for every index** had one killer because the recovery test looped over the
  names inside a single test. Parametrised per name, it now fails 6 cases — a looping test is one
  deletion away from reopening the hole.

The remaining single-killers are refusal guards with one condition each; recorded in
`docs/audit_log.md` for re-audit #4.


## [0.47.0] - 2026-11-14

### Added — tranche 40, the forward CDS and the knockout default swaption

Closes the single-name CDS vertical (`OKANE-2008` ch 9.2/9.3).

- `market/credit_legs.py` (extended) — `CdsLegs(annuity, protection)`, `forward_annuity`,
  `forward_spread`. The forward spread is written as the **protection difference** over the annuity
  difference, which is what O'Kane's 9.2.1 derivation produces.
- `products/forward_cds.py`, `products/credit_default_swaption.py` — frozen data, each carrying one
  schedule spanning `[t, T]` plus a date **on** it that splits it. That makes period alignment a
  property of the representation: a separate near-schedule field could be handed two unaligned
  grids, which is a wrong number nothing would catch.
- `models/credit_swaption.py` — `CreditSwaptionModel(credit, volatility)`, **composing** the
  tranche-38 `CreditModel` rather than restating `(market, recovery)`, so a forward CDS and the
  swaption written on it cannot disagree about the curves or the recovery.
- `engine/forward_cds.py` — both pricers. `SwaptionDirection` already satisfies Black's
  `OptionDirection` protocol (payer is a call on the spread), so it passes straight through.
- The **knockout needs no indicator**: `1{τ>tE}` is already inside `RPV01(0,tE,T)`, since both
  annuities weight by survival from today. A separate factor would double-count it.

### Oracles

- **The anchor is a cross-product, not a formula restatement**: the forward CDS equals
  `price(CDS to T, S₀) − price(CDS to tF, S₀)` through tranche 38's engine, for every contractual
  spread **and both directions** — 1e-12, different code each side.
- **Put-call parity** `VPay − VRec = (S(0,tE,T) − K)·RPV01(0,tE,T)` tied back to the forward CDS,
  at every vol — survives any slip in Black's `d±`.
- **`tF → t` reduces to tranche 38's `par_spread`** (cross-product, different code).
- **Black's ATM asymptotic** `price/(annuity·N·F) → σ√tE/√(2π)`, asserted as a *rate*: the error is
  3.3e-05 / 1.3e-04 / 5.3e-04 at σ = 0.02/0.04/0.08, exactly quadratic in σ.
- A pointwise transcription of the 9.3 composition against `norm_cdf` directly, plus the zero-vol
  intrinsic and monotonicity in σ.

### Three oracles the spec proposed that turned out to be weaker than they looked

Reported rather than quietly adjusted, since two of them are the kind that pass while asserting
little:

1. **Eq 9.2 and the 9.2.1 derivation are the same expression.** The par spread *is*
   `Prot/RPV01`, so substituting gives one from the other identically — measured relative gap
   exactly `0.0`. That is an algebraic tautology, not the transcription independence it resembles;
   it is kept and **labelled as an identity**, with the real independence coming from the
   cross-product.
2. **"At `S₀ = S(0,tF,T)` the forward CDS is worth zero" telescopes** — the forward spread is
   *defined* as the ratio that zeroes it. Kept as a guard; the non-telescoping version is the same
   statement asserted through the engine difference, where nothing was constructed to cancel.
3. **A flat *hazard* is not a flat *spread curve*.** The spec's flat-curve reduction was asserted
   against a flat hazard, and it fails: with a non-flat discount curve the par spread still drifts
   with maturity (1.35e-03 between 2y and 5y here), so the forward spread is legitimately 1.06e-03
   from the 5y spot. Hiding that would have meant loosening the band to 2e-03. Replaced by the
   condition that actually makes 9.2.1 exact — **equal par spreads**, asserted on constructed legs
   to within a few ULP — plus the curve-level version requiring **both** curves flat (3e-08), and
   the misleading case asserted as a *non*-reduction so it cannot be re-added as a wrong test.

### Validity regime

`forward_spread` divides by a difference of annuities, so a forward start close to the maturity is
a cancellation: the reduction identity holds to ~69 ULP with annuities 2.42 and 2.43 against ~1 ULP
for well-separated ones. An empty or inverted window is refused; a *nearly* empty one is not, and
returns a spread whose last digits are noise. Measured, asserted, and named in the docstring.

### Mutation sweep

23 mutants, full suite each, no `-x`. One survivor and three load-bearing single-killers, all
closed in-tranche:

- dropping `CdsLegs.par_spread`'s annuity guard **survived** — the `tF → t` reduction constructs
  exactly the degenerate leg, so the guard is reachable and simply had no test;
- the forward CDS's **direction factor** was killed by one test; the cross-product anchor is now
  parametrised over both directions (1 → 4 killers);
- the swaption's **total variance** was killed only by the pointwise transcription, so `σ²` alone
  and `σ·tE` both had a single killer each. Black's ATM asymptotic separates them independently —
  they read `0.707×` and `4.98×` of it — taking both to 2 killers.

The remaining single-killers are refusal guards with one condition each, which is inherent;
recorded in `docs/audit_log.md` for re-audit #4.


## [0.46.0] - 2026-11-14

### Added — tranche 39, the CDS survival-curve bootstrap

- `market/cds_bootstrap.py` — `bootstrap_survival_curve(quotes, discount, recovery)`: a par CDS
  strip to a piecewise-flat forward hazard curve, solved maturity by maturity, shortest first
  (`OKANE-2008` ch 7.4/7.5). L1 because of what it **produces** — market data from quotes, using
  only L0+L1, never the L4 engine. A quote is `(schedule, spread)`, not maturity → spread, because
  the schedule carries the premium frequency as well as the maturity; a mapping would silently
  assume one frequency across the curve.
- A **root-find**, unlike tranche 8's closed-form bootstrap: `λₘ` enters the protection integral
  inside `exp(−∫λ)` *and* the premium leg's survivals, so `V = 0` is genuinely implicit. `V` is
  strictly increasing in `λₘ`, which is what makes a bracketed solve right and gives the arbitrage
  test its exact form.
- **Arbitrage is refused by a sign test, not a tolerance** (ch 7.7): since `V` increases in `λₘ`, a
  root with `λₘ ≥ 0` exists iff `V(λₘ=0) ≤ 0`, so a quote too low to be consistent with the shorter
  maturities is detected exactly and refused in domain vocabulary rather than surfacing as a
  solver bracket failure.
- `market/credit_legs.py` — the CDS leg arithmetic at L1, shared by the bootstrap and the engine.

### Oracles

- **The known-curve round trip** (tranche 8's anchor): an independently constructed
  term-structured hazard curve → par spreads computed in the test → bootstrapped back → hazards
  recovered to **4.8e-07**, converging second order (1.9e-06 / 4.8e-07 / 1.2e-07 at 25/50/100
  points per segment). Independent in its **quadrature**, not only its code: the test integrates
  `½(Z(a)+Z(b))·(Q(a)−Q(b))`, exact in the survival differences, where the implementation
  integrates `Z·λ·Q`.
- **The credit triangle, inverted** — and restructured from the spec. The gap to `λ = S/(1−R)` is
  **not** the integration grid (identical at 50 and 400 points per segment): it is the premium
  discretisation, and it is first order in the premium period — 1.015e-02, 5.065e-03, 1.687e-03,
  3.89e-04, 5.54e-05 at 2, 4, 12, 52, 365 periods a year. So it is asserted as a **convergence in
  the premium frequency** with that measured rate, which is sharper than a band at any one
  frequency.
- **The extraction is proved pure by bit-identity**: `risky_pv01` lifted out of the engine leaves
  its values unchanged to the last bit across **324 fixtures × 5 values**. Nothing was reordered,
  so the strong claim was available and is the one made.
- Requirement oracles: `Q` monotone non-increasing, every hazard non-negative, each quote
  repricing to par across **two different quadratures** (max 1.4e-08).

### Changed

- `engine/credit_default_swap.py` delegates `risky_pv01` to the shared L1 function. Its protection
  leg does **not** delegate — see below.

### Ruled and fixed in this tranche — `OPEN.md` OI-4, closed

`λ(t)Q(t)` is **discontinuous at every hazard pillar**. The engine integrated `default_density(t)`
pointwise on a uniform grid, so a trapezoid straddled those jumps and **lost an order** — and at a
segment's end `default_density` returns the hazard of the segment *starting* there, the wrong one
for the piece ending there. The shared L1 functional reads the hazard once per pillar segment,
exact between pillars, and recovers second order.

Code raised this rather than ruling it, since correcting the engine moves a **shipped** number.
**Cowork ruled the engine adopts the per-segment integration** (thread 0214): the +1.846e-05 move
is provably *toward* an independent 20k/segment reference, so it is a ruled correctness change and
not the silent reordering §4's pure-refactor rule forbids. Measured after the switch, at the same
committed count:

- the engine's term-structured protection leg: **1.8e-05 → 6.3e-11** against a 200k/segment
  reference, and the test band tightened by five orders with it (a band left at 3e-05 would have
  gone on passing while asserting nothing);
- the **flat**-hazard closed-form anchor: **unchanged at +5.672e-10**, bit for bit — a flat hazard
  has no interior pillars, so one segment of 2000 points *is* the quadrature v0.45.0 shipped, which
  is why that version's published figure stays exactly reproducible;
- `riemann_bracket` breaks at the same pillars, since on a uniform grid it would bracket a
  *different* integral from the trapezoid it asserts about;
- the duplication **collapsed** — `credit_legs.protection_leg` is the one formula the engine and
  the bootstrap call — so OI-4 closed rather than persisting, and the agreement became a guarded
  convention, `docs/conventions_register.md` **C11** (**renumbered to C13 at
  v0.48.1** — the id was already held by the `model.market` row; left as published
  with the correction beside it, per the v0.45.0 precedent);
- a new oracle became available that could not exist while the two disagreed: a bootstrapped curve
  reprices its own calibrating CDS **through the engine** to **−5.8e-18 of notional** on the same
  grid, with the residual on a coarser grid falling at second order (the rate distinguishes a
  discretisation from a defect where no single value can).

**The finding that outlived the defect.** Tranche 38's rung-1 closed-form anchor uses a **flat**
hazard — which has no interior pillars, so it is exact *because* it degenerates the very parameter
this error lives in, and was structurally blind to a first-vs-second-order integration defect.
That earned a charter §4 clause on the reduction-anchor blind spot: the companion must be
non-degenerate in the specific parameter the reduction zeroes, checked by convergence **rate**
where the error is an order loss rather than a value.

### Mutation sweep

19 mutants, full suite each, no `-x`. Three first-pass survivors, two of them real and both in the
**tests**:

- A refusal oracle **passed with its guard deleted**: `match="recovery"` was satisfied by an
  unrelated bracket-expansion failure whose message also contained the word. A refusal oracle must
  name the condition it refuses, or it only asserts that *something* went wrong. Both paths now
  pinned separately.
- The arbitrage test's `>` versus `>=` boundary is reached only at `V(λ=0) = 0` **exactly**, which
  happens only for a **0bp quote** — where zero hazard is the right answer, not an arbitrage. No
  fixture reached it until one was added.
- Reading the hazard at a segment's start rather than its midpoint is a **proved equivalent**
  mutant: `breaks` is built from the pillars and `hazard` is right-continuous at one, so both reads
  land in the same constant piece — bit-identical over 32 curve/horizon combinations.

Re-swept after the OI-4 fix, with five mutants added for the newly shared engine path. One new
single-killer surfaced and was hardened rather than recorded: a bracket computed on a uniform grid
was caught by exactly one test. Investigating why exposed a weak oracle — *"the trapezoid sits
inside the bracket"* is close to arithmetically necessary, since the trapezoid **is** the mean of
the left and right sums, differing only in summation order. The claim a bracket exists to make is
that it contains the **true value**, and that discriminates sharply: pillar-aware it contains a
200k/segment reference at every refinement, while on a uniform grid it **excludes** the truth by
~6% at every refinement, bracketing nothing. Asserted now, which both strengthens the oracle and
gives the mutation a second independent killer.


## [0.45.0] - 2026-11-14

### Added — tranche 38, the credit default swap (project five opens: O'Kane credit)

- `products/credit_default_swap.py` — `CreditDefaultSwap(schedule, spread, notional, direction)`.
  Recovery is deliberately *not* a product field: `R` is a market/model assumption, not a term of
  the contract, so it lives on `CreditModel` where two counterparties pricing the same contract
  may disagree about it.
- `models/credit.py` — `CreditModel(market, recovery)`; refuses a recovery outside `[0,1]`, a
  snapshot with no survival curve, and a survival curve on a different clock than the snapshot.
- `engine/credit_default_swap.py` — O'Kane eq 6.2/6.4: `risky_pv01` (premium plus
  accrual-on-default, per unit notional), `protection_leg`, `riemann_bracket`, `par_spread`, and
  the registered `price(CreditDefaultSwap, CreditModel)`.
- `market/curve_set.py` — survival curves are keyed **in the curve set**, with their own typed
  accessor and their own backing store.

### Oracles

- **Rung 1, the credit triangle**: `par_spread → λ(1−R)` in the continuous flat-hazard limit,
  with a companion proving a term-structured hazard *breaks* the triangle (so the reduction is
  not vacuous).
- **Rung 1, the protection leg**: against the elementary closed form
  `(1−R)λ(1−e^{−(r+λ)T})/(r+λ)` under flat hazard and flat rates — 5.7e-10 relative at the
  committed 2000 steps, where a right-Riemann rule reads 4.1e-05, plus the second-order
  convergence *rate*.
- **Pointwise, eq 6.4**: the risky annuity term by term against a transcription whose survivals
  are summed from the hazard pillars, on an aligned *and* a straddling schedule (1e-13).
- **Cross-arc**: the protection leg reconciles against tranche 25's `unilateral_cva` on a flat
  unit exposure, both sides converging to the same integral — different code each side.
- **Measured, not toleranced**: the trapezoid sits inside O'Kane's explicit Riemann bracket, the
  bracket tightens at `O(1/M)`, and the committed step count resolves its own error (~2e-05).
  **Superseded at v0.46.0** — that ~2e-05 was the *uniform-grid* quadrature's resolution. Tranche
  39 found it loses an order at every hazard pillar and ruled the engine onto the per-segment
  integration (`OPEN.md` OI-4, closed); the same committed count now resolves **6.3e-11**. The
  figure above is left as published, with the correction beside it, rather than quietly edited.

### Changed

- `MarketSnapshot` keeps **five** fields. Survival curves went to `CurveSet` rather than becoming
  a sixth: tranche 1 predicted survival would arrive as a `CurveRole` key, but `SurvivalCurve`
  answers `survival`/`default_density` and not `df_at`, so keying it as a role would have made a
  hazard curve answer a discounting question (§6, a shared word and not a shared shape). It gets
  its own store in the same frozen value object — §7's remedy, not an exemption, and the tree
  still has no `fields-exempt`.
- `engine/credit_default_swap.py` — the `PricingFailure` guard wraps the whole computation rather
  than probing the discount curve alone. The probe front-loaded one missing-curve failure and let
  the survival lookup escape as a raw `KeyError`, against invariant 4. Found by the new
  wrong-currency oracle; pre-existing in this tranche, never shipped.
- `market/survival_curve.py` is now the single owner of the negative-time clamp; the engine's
  duplicate `max(·, 0)` is gone (§4/T30 — consolidate a duplicated guard rather than multiply
  fixtures). Proved redundant, bit-identical over 399 negative times.
- `.github/workflows/ci.yml` runs `verify.py references`. The gate was built at tranche 26b and
  listed in charter §9 but **never wired into CI** — so it has been firing only when run locally,
  which is how it caught `OKANE-2008` left at `in-hand` today. A gate that is not run is not a
  gate.

### Added — `verify.py gates`, closing the clause this tranche ratified

Charter §9 now says a gate named there "is not live until `ci.yml` invokes it… the charter's gate
list and `ci.yml` must agree, **and the agreement is itself checked**." Nothing checked it, so the
clause asserted a practice that did not exist — the defect §9 itself names about the merge-method
clause. `verify.py gates` parses §9's gate list and the workflow's `verify.py` invocations and
rejects three divergences: a gate named but never invoked (T38's actual defect), one invoked but
never named, and — the shape a set comparison is blind to — a name the two sides **agree** on that
`verify.py` does not implement. It is listed in §9 and invoked in `ci.yml`, so it covers itself.
Statically decidable, which is what makes it gateable where T37's τ=1 fixture guard was not.

Seeded-drift tests in `tests/tooling/test_gates_gate.py` (the T26b references-gate pattern):
removing the `references` line from a copy of the real workflow reproduces the T38 defect and the
gate rejects it. Five mutations of the gate, all killed.

### Mutation sweep

18 mutants, full suite each, no `-x`. Two first-pass survivors, both test gaps and no `src/`
defect: the **accrual-on-default term** (`(Q(tₙ₋₁)+Q(tₙ))` → `2Q(tₙ)`, −0.59% on the annuity,
invisible because every existing oracle asserted presence and sign but never magnitude), and the
negative-time clamp, which turned out to be a **proved equivalent** mutant — the loop's own
`break` already covers `t ≤ 0`, bit-identical over 8 probes including zero. Four single-killer
oracles were hardened or recorded: the quadrature rule 1→3 killers, the direction factor 1→2; the
recovery-range and shared-clock refusals stay at one killer each, which is inherent to a guard
with one behaviour to assert. Recorded for re-audit #4.


## [0.44.1] - 2026-11-13

Tranche 37 — **accrual-oracle hardening.** Test-only; **no `src/` change**. Remediates re-audit #3,
whose mutation sweep found two real oracle holes — code correct, oracles blind — in pre-Jäckel
tranches.

### Two holes closed

- **F-3.1, the nominal bootstrap (T8).** `annuity = sum(ψ · df)` → `sum(df)` **survived all 1430
  tests**: the fixture's dates are `ANCHOR + 365·k` on ACT/365F, so **every ψ is exactly 1.0**. On a
  semi-annual schedule each annuity term is overstated **~2.01×**. Closed with a semi-annual fixture,
  a `ψ ≠ 1` premise, a known-curve round trip and a par-reprice companion; the mutant now dies.
- **F-3.2, the implied-vol surface (T9).** Dropping `ψ` from `discounted_unit` **survived all 1433
  tests**. The vol is backed out of `floorlet / discounted_unit`, so on a semi-annual period a
  **9.00% vol reads as 5.80%** — a 36% understatement of every node the later tranches calibrate
  against. Closed the same way; the mutant now dies.

**`τ = 1` has now bitten three times** — the linker (F-1.2, T3), the bootstrap, the vol surface —
all pre-Jäckel, all from one fixture convention, and all found only by a re-audit's sweep.

### The exhaustive sweep (re-audit #3, target 1)

Every accrual multiplication in `src/` was mutated, not a sample: `swap_annuity` (killed by 5), the
bootstrap's last accrual (2), Market Model I's `τ/(1+τ)` (1), MM-II's (4), the LMM drift's `fτ` (15),
the Bermudan annuity (14), the cap/floor caplet (2), the linker's `ψ` (**2**, up from 1 — direct
evidence that re-audit #2's second-killer remedy works), and the YoY first period's `ψ` (1). Two
survivors, both above, both fixed.

### A second independent killer for Sobol's Gray code

`G(n) = n ⊕ ⌊n/2⌋` is load-bearing for the whole low-discrepancy construction and was killed by
**exactly one test** — which held both the printed-table oracle and the one-bit property, so one
deletion would have reopened it silently. Added: the same point set built **directly** from `G(n)`'s
set bits with no incremental state — a second *method*, not a second assertion — and the one-bit
property split into its own test. The mutation now dies to 2.

### The mechanical guard: assessed and NOT built, with the evidence

Re-audit #3 asked whether a CI gate could catch a future degenerate fixture. Two mechanisms were
built and tested:

- **Textual** (scan tests for annual date arithmetic) — **rejected**: it flags
  `test_inflation_capfloor.py`, whose accrual mutation is in fact killed by two tests. It cannot
  tell whether an accrual is load-bearing in the path a fixture exercises.
- **Behavioural** (instrument `CouponSchedule.periods()` and record the accruals each test module
  actually exercises) — works, is cheap, and has total discovery by construction. But as a *gate*
  it flagged **7 modules, of which zero are real holes**: four (`index_ratio`, `jy_calibration`,
  `jy_full_calibration`, `seasoned`) have **no accrual multiplication in their subject code at all**,
  and the other three are killed by tests elsewhere. 7/7 false positives.

**Verdict: no clean, non-brittle mechanism exists.** The missing half is *whether the subject code
uses the accrual*, which only mutation testing answers — and that is already the re-audit's job. The
guard goes to `to_go_deeper`; the exhaustive sweep and §4's premise rule stand. Reported rather than
shipped, because a gate with a 7-entry exemption list would be a gate that cannot fail.

## [0.44.0] - 2026-11-13

Tranche 36b — **wrong-way risk: equity CVA with correlated counterparty default.** The capstone's
second seam, and the **last build of the equity & hybrids arc**. Brockhaus 14.5.1, on 13.1.2's
truncated-Gaussian hazard apparatus.

### Added

- **`foundation/bivariate_normal.py` (L0)** — `Φ₂`, a genuinely new primitive: the tree had none,
  and `src/` deliberately does not use scipy. **Derived** from Plackett's correlation-derivative
  identity rather than transcribed.
- **`risk/wrong_way_risk.py` (L5)** — eq 13.4's `a_T` solve (and its forward formula as named
  surface), `GaussianHazard`, eq 14.5.1's four-term `risky_call_value`, and `wrong_way_cva`.
- **`risk/wrong_way_mc.py` (L5)** — the correlated-default simulation with the **exact** indicator
  `1{U < e^{−H_T}}`, the independent anchor.

### `Φ₂` was chosen for a property, not for elegance

`Φ₂ = Φ(h)Φ(k) + ∫₀^ρ φ₂` makes **independence exact at `ρ = 0`** (the integral is empty). That is
what lets the capstone's reduction be bit-identical: a series-in-`ρ` implementation would have made
the arc's headline anchor a tolerance question.

**Its validity regime is measured and stated, in BOTH value and correlation** — relative accuracy to
~1e-7 while `Φ₂ ≳ 1e-8`, and `|ρ| ≲ 0.99` (4.3e-08 at 0.999, 9.9e-05 at 0.9999; `|ρ| = 1` is exact
by closed form). The WWR consumer **asserts every one of its four `Φ₂` calls lands inside both
bounds**, and refuses a correlation outside them, rather than trusting the primitive silently.

### The `ρ → 0` reduction is exact, and exact for a reason

At zero correlation the first two terms sum to `F·Φ(d₊)·[Φ(−a/σ_H) + e^{−a+σ_H²/2}Φ(a/σ_H−σ_H)]` —
and that bracket **is eq 13.4**, i.e. `p(T)`. So `V = p(T)·V̂` and `CVA = (1−p(T))·V̂`, which is
tranche 36a exactly. Two independent things make it bit-identical: `Φ₂`'s exactness at `ρ = 0`, and
the `a_T` solve being the inverse of the very identity the reduction collapses through — so a solver
error and a formula error cannot cancel.

### The wrong-way direction is DERIVED, never intuited

`H_T` is increasing in `Y`, and `Y` correlates with the equity's `X` at `ρ` — so **`ρ > 0` makes
default likelier exactly when a long call's exposure is largest**, which is the wrong-way case and
must raise CVA. For a long put the same `ρ` would be right-way. The convention is stated in the test
so no later reader re-imports "wrong-way means bigger" as a fact about `ρ`'s sign, and the
monotonicity is confirmed independently by the simulation. (The T35 lesson, applied before the fact
rather than after.)

### Two findings in the source, both resolved by the finance

- **A dropped relation symbol.** `pdftotext` renders the book's limit as "If σ_H ␣␣ a_T" in **both**
  renderings — the operator is gone. Read as `≫` the stated approximation gives **p = 7.03**, not a
  probability; read as `≪` it matches the exact `p` to **six digits**. The requirement `p ≤ 1`
  decides it. Same hazard as eq 11.1's and 13.4's missing `Φ`, one level down: this time the
  casualty was a *relation operator*.
- **scipy is the weaker side in the deep joint tail.** My pre-registered `Φ₂` test asserted relative
  agreement at `h=k=−6`; measured, **scipy returns exactly 0.0** there (ours gives 3.5e-17 at
  `h=k=−4, ρ=−0.5`). The high-precision reference fails before we do — T20a's "a reference is code
  too" in its sharpest form yet.

### Oracles

- The eq-13.4 solve **inverting its own forward formula**, and matching the book's own small-`σ_H`
  limit (a derivation-step check, §4's stronger rung).
- The exact `ρ → 0` reduction to T36a, with a material companion at `ρ ≠ 0`.
- **The independent correlated MC** agreeing across the whole `ρ` grid (0.00–0.80 SE), with the
  equity drift, the **13.4 solve** (realised survival frequency) and **`ρ`** (realised correlation)
  each pinned *before* any price — and the band shown to resolve a flipped `ρ` (>20 SE) and the
  independent approximation (>10 SE).
- **Eight mutants killed; one PROVED equivalent** — the MC's `max(H,0)` is redundant because
  `H < 0 ⇒ e^{−H} > 1 > U`, bit-identical indicators on 200k paths. Proved rather than assumed, with
  the open-interval premise it rests on pinned.
- The `Φ₂` primitive carries seven mutants of its own, including the `max(·,0)` floor at `ρ = −1`
  that the sweep caught unexercised (without it, a **negative probability**) — §4's ninth shape
  firing twice more in this tranche.

## [0.43.0] - 2026-11-13

Tranche 36a — **CVA of an equity option under independent default.** The arc capstone's first seam:
**project three (XVA, T25–T29) meets project four (equity, T30–T35)**. Split from T36 at build time
on measured evidence; wrong-way risk is T36b.

### Added

- **`risk/equity_cva.py` (L5)** — `equity_option_cva`: `CVA = (1−R)·(1−p₀(T))·V₀`. New arithmetic in
  neither arc — a *consumer* of both.

### The integral collapses, and the derivation is the content

Brockhaus 14.2.1 states the general form `CVA = (1−R)N₀·E[(1/N_τ)1{0<τ≤T}·V_τ]` — a weighted sum
over default **dates**, not a single number. It collapses because, under independence of `τ` and
`(N,S)`, the **discounted option value is a martingale**: `E[V_t/N_t] = V₀/N₀` for every fixed `t`,
so the expectation does not depend on *when* default happens, and integrating against the default
density leaves `(V₀/N₀)(1−p₀(T))`.

**The book's forward contract in the same section does *not* collapse**, and the contrast is the
point: a forward's exposure `(F_t−K)⁺` is not its value — a forward can be worth less than nothing —
so the martingale argument cannot reach it. An option's exposure *is* its value. Derived here rather
than transcribed (§4 rates a derivation above a transcription), and verified independently at review.

### The headline oracle is a cross-ARC reduction

Tranche 25's `unilateral_cva` — a different arc, a different source, a different composition,
integrating `DF(t)·EPE(t)·dPD(t)` over a time grid — reproduces the closed form, and the gap
**converges** under grid refinement (10× over 50→800 points). Convergence rather than a point match
is what makes them the same quantity instead of two close numbers (the T31 lesson).

### Model-agnostic by construction

The collapse needs only that the *contract's* value is a martingale — nothing about the volatility
model — so the option value is taken through `dispatch` rather than from a named pricer. Layer 1
therefore prices CVA for **Black (T30), Heston (T33), the Hull-White hybrid (T34) and the
defaultable-equity model (T35)** without knowing which, which is tranche 33b's `(product, model)`
registry paying off in a consumer rather than a test.

### Oracles

- The cross-arc reduction and its convergence rate.
- Two zeros (`h → 0`, `R → 1`); the full-loss bound at `p → 0`; monotone in the hazard and the
  recovery; a **non-flat** survival curve read beyond its last pillar (T25's degeneracy, inherited).
- **Five mutants, all killed.**

## [0.42.0] - 2026-11-13

Tranche 35 — **defaultable equity under independent default.** The second hybrid, and the tranche
where the equity vertical meets the **credit** vertical: it reuses the `SurvivalCurve` built in the
XVA arc rather than declaring a second notion of a hazard rate.

### Added

- **`models/defaultable_equity.py` (L3)** — `DefaultableEquityModel` (snapshot, the *surviving*
  process's volatility, and the reused `SurvivalCurve`), refusing a survival curve anchored to a
  different date than the snapshot (C12's family).
- **`engine/defaultable_equity.py` (L4)** — the modified Black formulas, registered under
  `(EuropeanEquityOption, DefaultableEquityModel)`: the **fourth** pricer on the equity vanilla,
  after Black (T30), Heston (T33) and the equity-rate hybrid (T34).
- **`engine/defaultable_equity_mc.py` (L4)** — the independent simulation of the definition
  `S_t = S̃_t·1{τ>t}`, with the default indicator drawn from a **disjoint slice** of the stream so
  it is independent of the diffusion, which is the model's premise made operational.

### The source prints both forms, which is a free transcription check

`p·BSCall(F/p, K) = BSCall(F, p·K)` is Black's degree-1 homogeneity — and the book supplies *both*
sides, so asserting their agreement checks the scaled-strike transcription against the scaled-forward
one without inventing an identity. The engine uses the **scaled-strike** form: it never divides by
`p` (which vanishes as the hazard grows), and at `p = 1` it is *literally* the default-free call, so
the reduction is bit-identical rather than close.

### Two of my own assumptions were wrong, and the algebra said so

The pre-registered tests asserted that default makes the **call cheaper** and that a certain default
makes it **worthless**. Both are wrong under this model, and the formula is right:

- `E[S_t] = F(t)` is held **fixed** (eq 13.3 forces it), so raising the hazard does not lower the
  stock's expectation — it *redistributes* it. Mass moves to zero while the surviving process is
  worth `F/p`, which rises. That is a **mean-preserving spread**, and it makes every convex payoff
  dearer: call *and* put both rise, monotonically.
- As `p → 0` the call tends to the **forward**, not to zero: the strike scales away while `F` is
  held, and the vanishing surviving branch is worth `F/p → ∞`; the two balance exactly.

Confirmed by a direct independent simulation before the tests were rewritten (31.19/17.86 at
`h = 0`, 58.55/45.22 at `h = 0.15`, 89.16/75.83 at `h = 0.4` — closed form and MC agreeing at each).
The *observed* effect of worsening credit — a falling stock — is a change in **spot**, which this
comparison deliberately holds fixed; correlated default is a different model and is not built here.

### Oracles

- The **`h → 0` reduction is bit-identical** to T30's Black, call and put.
- **Put-call parity with the default terms cancelling exactly**: `BSCall(F,pK) − (1−p)K −
  BSPut(F,pK) = F − K`, which holds for *every* `p` — blind to a slip in both formulas at once and
  sharp against one in either.
- The source's two forms agreed to 1e-14; the surviving-forward identity `p·F̃ = F`.
- **The independent MC**: `E[S_t] = F` and the realised default frequency pinned *before* any price,
  then call and put within 4·SE across the money — with the band shown to resolve an inverted
  scaling, an unscaled strike, and a missing default lump.
- A **non-flat** survival curve read **beyond its last pillar**, so a mutant reading `hazard(0)` and
  exponentiating is separated (the tranche-25 degeneracy, inherited rather than rediscovered).
- **Seven mutants, all killed.**

## [0.41.0] - 2026-11-12

Tranche 34 — **the equity-rate hybrid under Hull-White.** The arc's first hybrid, and the first
tranche to price equity against the rates machinery the tree already had.

### Added

- **`models/equity_hull_white.py` (L3)** — `EquityHullWhiteModel`: the snapshot, the **reused**
  `HullWhiteParams`, the equity vol and the equity-rate correlation. One Hull-White convention in
  the library, not a second spelling of `kappa, sigma`.
- **`engine/equity_rate_hybrid.py` (L4)** — eq (11.1): Black on the forward with the **total**
  log-spot volatility and both `d±` shifted. Registered under
  `(EuropeanEquityOption, EquityHullWhiteModel)` — the **third** pricer for the equity vanilla,
  coexisting with Black (T30) and Heston (T33) only because T33b keyed the registry on the model.
- **`engine/hybrid_mc.py` (L4)** — the correlated two-factor simulation, written in **deflated
  form**: `E[(B⁻¹S − K·B⁻¹)⁺]`, so the spot is never reconstructed by dividing one noisy quantity
  by another.

### Two findings in the source, both of which changed the code

1. **The printed §11.1.2 bond integrand cannot be read naively.** It gives
   `(σ^B_t)²t = ∫₀ᵗ(v_s(T) − v_s(t))²ds`, and for a European option the payoff date *is* `t` — so a
   naive `T = t` makes it **identically zero**, collapsing the formula to Black while still passing
   its own reduction oracle. §11.1.1 settles it: `B` is the **cash bond** (`E[B_t⁻¹] = P(t)`), so
   `σ_B` is the volatility of `log B_t`, which for Hull-White is `σ_r²∫₀ᵗβ(s,t)²ds`.
2. **Take the product (11.1) needs straight from the derivation step, not via (11.3).** The printed
   (11.3) **is** consistent with the book's own derivation — *provided* every symbol is the
   integral-defined **effective** quantity (`ρ_{B,S/B}` is time-averaged, not the instantaneous `ρ`;
   `σ_B` is the effective bond vol, not `σ_r`). The hazard is numerical: reconstructing
   `ρ_{B,S}·σ_B` through `ρ_{B,S}` routes via a quantity that **divides by `σ_B`**, which vanishes at
   the `σ_r → 0` limit the anchor lives at. Taken from the derivation step the product is
   `(σ_B² + cross)/σ_S` — no division by `σ_B`, exactly zero at zero rate vol.

Both were found by **deriving before implementing**, and both corrected my own pre-registered tests.
*(The first write-up of finding 2 claimed the book's numerator was wrong; it is not — the error was
the reconstruction route. Corrected at review, and recorded because §5 cuts both ways: a false claim
*about* a source is as bad as a number copied from one.)*

### Oracles

- The **`σ_r → 0` reduction is bit-identical** to T30's Black — which is also what identifies the
  cumulative-normal glyph `pdftotext` drops in *both* renderings of (11.1).
- **(11.2) and (11.3) as derivation-step identities** over many parameter sets, plus `|ρ_{B,S}| ≤ 1`.
- The integrated variances against the test's own quadrature.
- **The independent anchor**: the hybrid MC agrees at **0.00 / 0.05 / 0.11 SE** across the money,
  with the martingale identity pinning the drift first and `ρ` pinned by the **realised covariance
  of the increments** (0.10%) rather than through a price — T17's instrument, because `ρ` is in the
  diffusion.
- **Eight mutants, all killed.**

### What the anchor can and cannot resolve, measured

- A wrong **total vol** moves the main fixture by only 2.7 SE (3.9 SE even at 200k paths), because
  the two vols differ by 2% there. Resolved by a **power fixture with a larger rate vol**, where it
  lands at **18.9 SE**. More paths were the wrong lever; a fixture where the term is *large* was the
  right one.
- The **`d±` shift cannot be resolved by any price at any path count**, and the reason is exact:
  `d(price)/d(shift) = df·[F·n(d₊) − K·n(d₋)]`, which Black's identity makes **exactly zero**
  (verified to machine precision at every strike). It is pinned instead by its defining identity
  `shift = ρ_{B,S}σ_B√t`, and the suite says so rather than letting the MC appear to cover it.

### Two sweep survivors, both closed by relocating the check

- **Dropping the shift** survived — no price sees it. Closed by the identity above.
- **A left-endpoint cash-bond integrator** survived, because the deflator is normalised to `P(t)`
  from the sample and that absorbs a first-order bias in its *mean*. The **variance** is where it
  shows: `Var[∫r du]` is 0.1% from the closed form with a trapezoid and **4.6%** with a
  left-endpoint rule. That oracle doubles as an independent cross-method check on
  `integrated_bond_variance` itself.

### Also

A single-seed convergence assertion failed for T21's reason — one seed made 2,500 paths look better
than 40,000 — and is now averaged over seeds with a 50× path ratio. The registry test that counted
equity pricers was generalised to assert *shape* rather than a hard number, since the count grows
with every hybrid.

## [0.40.0] - 2026-11-12

Tranche 33b — **the `(product, model)` engine registry.** Infrastructure, pulled by tranche 33: one
product now legitimately has several models, and the registry was the one place that did not know it.

### The footgun this removes

`_PRICERS` was keyed on the **product type alone**, so a second `@register(SomeProduct)` *replaced*
the first silently. Tranche 33 proved it empirically: registering the Heston pricer for
`EuropeanEquityOption` would have removed tranche 30's Black pricer from the table and repriced every
equity vanilla in the tree **without a single test failing**, because the two signatures are
identical.

### Added

- **`_PRICERS` keyed on `(product_type, model_key)`**, where the model key is a **concrete type OR a
  `@runtime_checkable` protocol**.
- **`DuplicateRegistration`** — registering the same pair twice now *raises*. A pricer silently
  replaced is debt incurred without a ledger entry (§8).
- **`AmbiguousRegistration`** — a model satisfying two registered protocols for one product is a
  loud error, not an arbitrary pick.
- **`registered_pricers()`** — a *copy* of the table, so a test can see the registry without being
  able to mutate it through the back door.
- **`HestonModel`** (params + snapshot, §3) and **`price_european_equity_option_heston`**, registered
  under `(EuropeanEquityOption, HestonModel)` — **the registration that was impossible before.** The
  put uses put-call parity rather than a second inversion, so it inherits the call's accuracy.

### The spec's premise was wrong, and execution said so

The spec read *"each current engine has one model type — a mechanical pass."* It is false for **two
of the eight**: `IndexRatioModel` is a `@runtime_checkable` **Protocol**, and the inflation cap/floor
and year-on-year pricers are polymorphic over it, serving `JarrowYildirim`, `MarketModelI`,
`MarketModelII` and a stub defined in `tests/`. Keying them concretely broke **63 tests** and would
have encoded a falsehood about what those pricers are for. Reported rather than coded around (§10);
Cowork corrected the spec to the concrete-OR-protocol key.

**Dispatch resolves exact first, then the unique protocol match.** A pricer naming a concrete model
makes a more specific claim than one naming a capability. A model satisfying a registered protocol
needs no registration of its own — which is why a stub defined in `tests/` dispatches without `src/`
knowing it exists.

### Oracles

- **Pure refactor, proved bit-identical**: the inflation cap/floor resolved through the *protocol*
  path with a `JarrowYildirim` model returns `13957.888028738482`, identical to calling the
  registered pricer directly. The lookup key changed; no arithmetic did.
- Coexistence, loud duplicate failure, loud ambiguity, exact-beats-protocol, protocol-polymorphism
  across all four models, and every production registration surviving the widening.
- **Six mutants, all killed.**

### A hole the sweep found

Dropping the `isinstance` check in the protocol match **survived all 1140 tests** — every model the
suite hands to a protocol-registered pricer happens to satisfy that protocol, so the check was
unexercised. Without it, a model that does *not* satisfy the protocol would be handed to the pricer
anyway and fail deep inside a valuation instead of returning a clean `PricingFailure` at the
boundary (§3.4). The degenerate fixture was the **population of models the suite uses**, not any one
value. Closed with a protocol carrying a real member (an empty `Protocol` is satisfied by
everything, so it could not have discriminated).

### Also

`models/__init__.py`'s docstring sat below five imports, so the module had none; it is now first.
Four `# noqa` suppressions introduced while writing the tests were removed by fixing their causes
(§7) — the directory's `conftest.py` owns both the path setup and the registry-populating import.

## [0.39.0] - 2026-11-12

Tranche 33 — **Heston by Fourier inversion.** Fourth unit of the equity & hybrids arc, and the first
model in the tree whose skew comes from *dynamics* rather than from a fitted surface.

### Added

- **`models/heston.py` (L3)** — `HestonParams` (the five parameters as one bundle), the
  characteristic function on its **stable branch**, and `naive_characteristic_function` — kept
  deliberately as the oracle's wrong-variant companion, not as a fallback.
- **`engine/heston_fourier.py` (L4)** — `FourierGrid` (truncation and node count as explicit
  numerical config, §3.5) plus `heston_probabilities` and `heston_call_price` by Gil-Pelaez
  inversion. `P1` is the *same* characteristic function tilted into the share measure, not a second
  transcription — so an error moves both probabilities together and surfaces in put-call parity.
- **`engine/heston_mc.py` (L4)** — the independent correlated log-Euler simulation with a
  **full-truncation** CIR variance.

### RED-first, restored

All three oracle files were **committed before their implementations existed** — the fix agreed
after tranche 32's lapse (thread 0169): the oracles that need no implementation to state get written
first and sit red while the derivation happens. `φ(0)=1`, `φ(−i)=F(T)/S₀`, Hermitian symmetry,
`|φ|≤1`, put-call parity, `P1,P2 ∈ [0,1]` and the no-arbitrage bounds are all properties of *any*
valid characteristic function or price, so none could be reverse-engineered from the code they test.

### The Little Heston Trap, oracled by grid refinement

The naïve complex branch returns a *plausible* number and is simply wrong at long maturity, so a
value oracle at one maturity sees nothing. The discriminator is **structural**: a genuine
discontinuity keeps its height under refinement while smooth variation shrinks with the step.
Measured over `T ∈ [0.5, 20]` at `u = 1`:

| nodes | stable | naïve |
|---|---|---|
| 80 | 6.97e-03 | 1.497 |
| 320 | 1.73e-03 | 1.494 |
| 1280 | 4.31e-04 | **1.493** |
| 5120 | 1.08e-04 | **1.493** |

The stable branch falls 4× per 4× refinement; the naïve branch does not move. A first attempt
asserted an absolute bound on the largest step and was wrong — on a coarse grid a smooth function
also moves, so the bound measured the step size rather than the branch.

### A bias found by the reduction, and pinned as a relationship

`1/(iu)` is undefined at `u = 0`, so the quadrature starts at `_LOWER` and omits the sliver below.
That omission is **one-directional and linear in the cutoff**, and **grid-independent** — which is
how it was identified: the `σ_v → 0` reduction's error sat at 1.16e-08 across three different grids
and moved only when the cutoff did. Measured at `K/F = 1.25`: 1.16e-06 at `1e-6`, 1.16e-08 at
`1e-8`, 1.16e-10 at `1e-10`, 1.16e-12 at `1e-12`, reaching the floating-point floor by `1e-14`.
Committed at `1e-12`, and the test asserts the **linearity** rather than a bound — so if the ratio
ever stops tracking the cutoff, the error has a second source.

### Oracles

- The `σ_v → 0` reduction to verified Black on the integrated variance — with a premise asserting
  the stochastic price is materially different, so the reduction is **proved vacuous alone**.
- **The non-degenerate anchor**: Fourier against the independent MC within 4·SE, on a fixture whose
  **Feller condition is deliberately violated** (`2κθ = 0.12 < 0.36 = σ_v²`) so the variance
  genuinely reaches zero and full truncation is exercised rather than assumed away.
- The martingale identity `E[S_T] = F(T)` pinning the drift **first**, so a drift error cannot be
  read as a volatility error.
- The band's power asserted: a flipped `ρ` lands outside it.
- **Twelve mutants, all killed** — including reflection-instead-of-truncation on the variance.

### Raised, not decided

The spec asked for a `@register`ed Fourier pricer. **Dispatch is keyed on product type alone**
(`_PRICERS[product_type]`), so registering Heston for `EuropeanEquityOption` **silently replaces**
tranche 30's Black pricer — verified empirically. The Heston pricer is therefore a named function,
every oracle reaches it directly, and the two-models-one-product question is an `Ask:` for Cowork.

## [0.38.0] - 2026-11-11

Tranche 32 — **the implied-vol surface and Dupire local volatility.** Third unit of the equity &
hybrids arc, and the modelling foundation the rest of it stands on.

### Added

- **`ImpliedVolSkew` widened to `(K, T)`** — `σ(K,T) = atm + term_slope·T + slope·log(K/F(T))`, with
  **exact derivatives** (`∂σ/∂K`, `∂²σ/∂K²`, `∂σ/∂T`). The term dimension arrived with its second
  consumer, since Dupire differentiates in `T` and a flat-in-maturity surface has nothing to
  differentiate. T31's numbers are unchanged (`term_slope` defaults to 0 and adding `0.0` is exact).
- **`models/local_volatility.py` (L3)** — `ForwardState`, `black_partials`, `surface_partials` and
  `dupire_local_variance`.
- **`engine/local_vol_mc.py` (L4)** — the log-Euler local-volatility diffusion, re-evaluating
  `σ_loc(S,t)` each step.

### Analytic derivatives, ruled at spec time

Dupire's `∂C/∂T`, `∂C/∂K`, `∂²C/∂K²` are taken **exactly by the chain rule**, not by finite
differences. Measured on this very surface, a finite-difference `∂²σ/∂K²` is **1% wrong at
`h = 1e-5`**, best near `h = 1e-2` (3.5e-08), and degrading again beyond — a noise-dominated
reference with a step-size sweet spot that would need arguing in every regime it is used (§4's "a
reference is code too", the T20a shape). With the derivatives exact, the inversion band is **pure
Monte-Carlo standard error** and a Dupire defect has nowhere to hide in it.

**The chain rule is the content**: `σ` depends on `K` and `T`, so `∂C/∂K` picks up `C_v·v_K` and
`∂²C/∂K²` picks up three cross terms. All four are pinned — dropping any one of them is a mutant
that dies.

### Oracles

- Every composed partial against the test's **own** finite differences, on its own grid, of its
  **own** price function — with a premise asserting the chain-rule terms are not negligible, so the
  comparison is not vacuous.
- **The flat reduction is exact**: a flat surface gives `σ_loc = σ` to machine precision at every
  strike, because every chain term vanishes and Black's partials compose back to `σ²`.
- **The mandatory companion** — Dupire's *rule of two*: on the skewed surface the local skew is
  measured at **2.17×** the implied skew, which the no-op `σ_loc = σ_impl` cannot produce.
- **The inversion anchor**: `σ_loc` simulated by log-Euler reprices the parametrized surface within
  4·SE (measured deviations 0.01–0.18 SE), with the **martingale identity** `E[S_T] = F(T)` pinning
  the drift *first* so a drift error cannot be mistaken for a vol error.
- **Ten mutants, all killed**, including all three `∂²C/∂K²` cross terms.

### A degeneracy caught before the oracle was written

The `(r−q)·K·∂C/∂K` drift term is invisible at a near-zero carry. Measured: dropping it moves the
worst repricing by **3.2 SE at `carry = 0.005`** — *inside* a 4-SE band, so it would have survived —
and **8.9 SE at `carry = 0.04`**. The fixture now carries `r − q = 0.05` with that materiality
asserted as a premise. Found by running the mutation against a draft fixture rather than after
committing one.

### Also fixed, while writing its own docstring

`∂σ/∂T` at fixed strike has **two** terms, not one: the explicit `term_slope`, **plus** the skew read
against a forward that itself moves with `T` (`−slope·carry`). A surface skewed in log-moneyness is
implicitly time-dependent even at `term_slope = 0`, so dropping that term is a silent error on any
skewed surface with a non-zero carry. The first draft returned only `term_slope` while its docstring
described both; the mutation that drops it now dies.

## [0.37.0] - 2026-11-11

Tranche 31 — **the variance swap by static replication.** Second unit of the equity & hybrids arc,
and the first **model-independent** price in the tree: no volatility dynamics enter it at all.

### Added

- **`market/equity_skew.py` (L1)** — `ImpliedVolSkew`, a two-parameter log-moneyness skew. Market
  data, not a model: it is the *price source* a model-free replication reads across strikes. The
  full surface and Dupire local vol are T32.
- **`products/variance_swap.py` (L2)** — `VarianceSwap`, struck in **variance** rather than
  volatility, because the payoff is linear in variance and converting at the boundary invites the
  squaring to happen twice or not at all.
- **`models/variance_replication_model.py` (L3)** — `StrikeGrid` and `VarianceReplicationModel`.
  There is no pricing model: the object carries a snapshot (§3) and the **strike grid**, which is
  numerical config and so is explicit rather than defaulted (§3.5). `StrikeGrid` lives at L3 because
  an L3 model may not import L4.
- **`engine/variance_replication.py` (L4)** — `fair_variance` (the `1/K²`-weighted strip of OTM
  vanillas) and the registered `price_variance_swap`.
- **`MarketSnapshot.equity_skews`**.

### The forward sits ON the strike grid — a real quadrature fix, not a loosened assertion

The strip switches from puts to calls at `K = F`. The integrand is *continuous* there (put-call
parity makes the two prices equal) but its **derivative is not**, so an interval straddling `F`
contains a kink and the trapezoid rule loses its second-order rate on it. Measured on a uniform grid
that ignored `F`, successive refinements gave error ratios of **0.85, 22.1, 0.61, 3.25** — noise
around no rate at all, because the error was dominated by how close `F` happened to fall to a node.

`StrikeGrid.strikes` now puts the forward on the grid and integrates each side separately. The
ratios become **4.00, 4.02, 4.00, 4.00** — clean second order. The convergence-rate oracle is what
exposed this; a tolerance alone would have passed.

### Oracles

- **The anchor is a cross-product reduction**: under flat vol the strip integrates to exactly `σ²`,
  with the strip integrator and `black_price` being *different code*. It fails under any error in the
  `1/K²` weight, the `2/T` factor, the OTM put/call split, or the forward.
- **The log-contract identity by a third route** — `−(2/T)E[log(S_T/F)]` computed by numerical
  integration over the lognormal density, independent of both the strip and the `−½σ²T` algebra.
- **The mandatory skew companion.** A flat surface satisfies the reduction *trivially*, so the
  reduction alone is vacuous; a material skew must lift `K_var` off the ATM variance by the
  convexity premium. Also pinned: **either sign** of skew raises it, since the premium comes from
  curvature in the weighted strip — a test asserting only "negative skew raises it" would pass an
  implementation that responded to `|slope|` in the wrong place.
- **Eight mutants, all killed.**

### Two degeneracies found by the sweep

- **`T = 1` hid the `2/T` factor.** The fixture expired in exactly one year, so `2/T · area` and
  `2 · area` are the same number and dropping the division survived all 1038 tests. §4's *first*
  recorded trap (`τ = 1`), reappearing in the time axis rather than an accrual. Closed with a
  2.5-year fixture and its non-unit premise asserted.
- **The grid's accuracy is two-sided**, which a single band hides: at **low** `σ√T` the integrand is
  peaked near the money and widening the grid at fixed node count makes it **worse** (discretisation);
  at **high** `σ√T` truncation dominates and widening makes it **better**. Measured 1.1e-04 at
  `σ√T = 0.12`, 1.6e-06 at 0.44, 4.0e-04 at 0.63 — so the committed grid's sweet spot is roughly
  `0.15 ≤ σ√T ≤ 0.5`, and both directions are now asserted rather than left as one tolerance. §4's
  "a reference needs its accuracy argument in the regime where it is used."

## [0.36.0] - 2026-11-10

Tranche 30 — **the equity underlying and the European equity option.** First unit of project four,
the **equity & hybrids arc** (`BROCKHAUS-2016`) — and the first equity vertical in the tree.

### Added

- **`market/equity.py` (L1)** — `EquityAsset`, `DividendSchedule` (all three quoted conventions at
  once: discrete cash, proportional, continuous yield), `EquityUnderlying` (spot, repo/borrow,
  dividends) and `dividend_forward`. Quote-side equity data, the same status as a discount curve.
- **`products/equity_option.py` (L2)** — `EuropeanEquityOption` and `EquityOptionKind`.
- **`models/equity_black.py` (L3)** — `EquityBlackModel`: one flat vol, carrying its snapshot.
- **`engine/equity_vanilla.py` (L4)** — the registered pricer: compute the forward, then **reuse**
  `black_price`. The equity content of a vanilla is the *forward*, not the option formula.
- **`MarketSnapshot.equities`** — the equity quote side per asset, reached only through
  `model.market` (§3).
- **`black_price`'s `kind` parameter now depends on the `omega` *shape*** (`OptionDirection`), not on
  one product's enum. Rule-of-two met by a shared shape rather than a shared word: `CapFloorKind`
  (T5) and `EquityOptionKind` (T30) are both present consumers. The body already read only
  `kind.omega`, so no operation changed.

### Oracles

- **The anchor is the exact reduction to the verified core**: with no dividends and no repo the price
  is **bit-identically** `black_price` on `S₀/df(T)`. Black has been verified since T5 and this
  tranche must not perturb it. Companion: a material dividend stream moves the call by >15%.
- The forward against an **independently composed** reference; the continuous-yield form against
  `S₀·e^{(r−q)T}`; discrete cash proved **PV-discounted rather than raw-summed** (the commonest slip,
  with the premise that discounting matters on the fixture asserted).
- **A cross-form oracle**: a fine strip of small proportional dividends converges to the continuous
  yield — two different formulas for one object, so a shared transcription error surfaces.
- Put-call parity (model-free, so it survives a transcription error in the forward); the forward
  monotone decreasing in every dividend and in the repo; the call monotone in vol and inside the
  no-arbitrage bounds.

### The degeneracy here is a DATE boundary, and it was found half-guarded

A dividend after expiry cannot enter `F(T)`, and **every value-based oracle is blind to dropping the
`tᵢ ≤ T` restriction if the fixture puts all dividends before expiry** — the same family as
invariant 6's boundary, where the analogous slip was a 41% mispricing.

The fixture carried a post-expiry **cash** dividend from the start. The mutation sweep then showed
that dropping the restriction on the **proportional** branch survived all 1010 tests: the boundary is
a *separate line of code per convention*, and only one had a post-expiry entry to exclude. The trap
this module's own docstring names, half-applied by its author and caught by the sweep. Closed with a
post-expiry proportional dividend plus its materiality companion; **all seven mutants now die**.

### Also

`BROCKHAUS-2016` flipped `in-hand` → `worked-from` — **caught mid-tranche by the `references` gate
built at T26b**, which fired on its first real opportunity, in the tranche that created the drift
rather than five tranches later. And the re-audit #2 ruling: a **second independent killer** for
F-1.2 (the coupon leg is frequency-invariant at a zero real rate — a requirement, not another
transcription), proven to kill the dropped-`ψ` mutant *alone* with the original oracle deselected.

## [0.35.0] - 2026-11-10

Tranche 29 — **collateral / CSA: the collateralized exposure over the margin period of risk.** The
last build of the XVA spine arc; `CP_xva_spine.md` and re-audit #2 follow.

### Added

- **`risk/collateral.py` (L5)** — `CollateralAgreement` (a zero-threshold, zero-MTA variation-margin
  CSA: its margin period `δ` and the deterministic initial mark) and `collateralized_exposure`,
  which returns `(V(t) − V(t−δ))⁺` path by path.

**Collateral enters only through the exposure.** The gap is assembled into the same
`SimulatedExposure` the uncollateralized engine produces, so T25/T27's `credit_adjustment` prices it
**unchanged** — no collateralized CVA function, no second integral, no branch in the credit code. A
CSA changes *what is exposed*, not how an adjustment is computed. The positive part lives here
rather than downstream, because a negative gap means we are over-collateralized and an
`ExposureProfile` is a non-negative object by construction.

### Oracles

- **Two exact reductions, each with its non-degenerate companion.** `δ = 0` gives exactly zero
  (`V(t) − V(t) = 0`, by construction, not a limit); `agreement=None` returns the input **object**,
  so the uncollateralized case is a structural special case and T26/T27's numbers provably cannot
  move. Both reductions are *trivially* exact, which is stated plainly — the companion is what makes
  the pair non-vacuous: a material `δ` on a volatile mark where the collateralized CVA is **less
  than half** the uncollateralized value and strictly positive.
- **A rate oracle, not just monotonicity.** The gap's standard deviation is `σ√δ`, so the adjustment
  is proportional to `√δ`: halving `δ` must multiply the CVA by `1/√2 = 0.7071`. Measured 0.772,
  0.710, 0.706 over successive halvings. A wrong power of `δ` passes a monotonicity check and fails
  this one.
- **A structural signature** separating the gap from the mark: the gap's law does not depend on `t`,
  so its EPE is **flat** (2.4% across the horizon, sampling noise on a `t`-independent law) while
  the mark's EPE grows (17.4%). Asserted as the *separation*, not as two fixture-tuned bounds.
- The gap EPE against T25's `expected_positive_exposure` on the gap's own moments, at the MC rate.
  **Derived, not transcribed** — there is no source equation for the gap construction, and §4 rates
  a derivation above a transcription.
- The process is **not** re-anchored: the gap is a functional of the very paths T26's Leg B validated
  against the numéraire-relative martingale identity, so it inherits that validation. Inventing a
  fresh anchor over the same paths would add ceremony, not evidence.
- Requirement oracles: collateral can never *increase* the adjustment, and the adjustment is monotone
  increasing in `δ`.

### A real hole found by mutation, and the fixture that closes it

Reversing the gap to `(V(t−δ) − V(t))⁺` — the **wrong** exposure — initially **survived all 986
tests**. It is not an equivalent mutant. It survived because the fixture's mark was **driftless**,
making `V(t) − V(t−δ) ~ N(0, σ²δ)` symmetric about zero: the reversed gap then has the *identical
distribution*, and every mean-based oracle is structurally blind to the direction. Measured, the two
EPEs agree to the last digit with no drift and differ by **87%** at a drift of 24/year (572% at 60).

Closed with a **drifting-mark fixture** whose closed form carries the drift (`N(μδ, σ²δ)`), a premise
asserting the drift is material against the gap's own standard deviation, and a companion asserting
the band is tight enough to *reject* the reversed value — rather than assuming the new fixture would
catch it. All six mutants now die.

This is §4's degenerate-fixture rule in a form not yet recorded there: the degeneracy is a
**distributional symmetry**, not a specific parameter value (`τ = 1`) nor a uniformity across a
slice. The guard is therefore not "the values differ" but "the fixture breaks the symmetry the
mutation exploits".

### Scope fence

Zero threshold and zero MTA, variation margin only. Thresholds/MTA, **initial margin** and its MVA,
ColVA (the funding cost of collateral), and gap/wrong-way risk are each deferred, not approximated.

## [0.34.0] - 2026-11-09

Tranche 28 — **FVA: the funding adjustment, a fixed point with a closed form.** Fourth and last
**core** unit of the XVA spine arc: CVA · DVA · FVA complete. Collateral (T29) is optional after it,
then the checkpoint closes the arc.

### Added

- **`risk/funding_valuation_adjustment.py` (L5)** — `funding_valuation_adjustment` (Crépey eq 17's
  closed form `γ/(1+γ)·(P−CVA)⁺`), `fva_fixed_point_residual` (eq 19's defining equation as named
  surface, because it is the tranche's correctness anchor and §4 says extract what an oracle must
  reach), and `iterate_fva` (the Picard iteration, returning the whole sequence so a test can read
  the *rate* and not only the limit).

### Oracles

- **The anchor is that the closed form satisfies its own defining equation** (§4: a formula and its
  stated inverse must actually invert), asserted to machine precision across the spread range. It is
  derived from the finance, so it survives a transcription error — and it is sharp against exactly
  the mistake the source's typography invites.
- **With the companion that proves the anchor has power:** the natural reflow misreading
  `γ·(P−CVA)⁺` is asserted **not** to satisfy the fixed point. Without that, "the closed form
  satisfies its fixed point" could be a property of a whole family of formulas rather than of this
  one (§4, the Box-Muller pattern).
- The `γ/(1+γ)` coefficient is printed as a **stacked fraction**, so a `-layout` reflow splits
  numerator from denominator across lines — the same hazard that produced the eq-25 KVA error. Read
  in `-raw` and `-layout` and reconciled, **and independently derived** from eq 19 (with the
  positive part open, `FVA(1+γ) = γ(P−CVA)`), which §4 rates above a transcription.
- An **independent root-find** on the defining equation (different code from its algebraic solution)
  agrees across five spreads.
- **Both regimes of the `⁺`**: a net payer with `P−CVA` materially positive (asserted, else the
  formula is invisible) and a **net receiver**, whose FVA is *exactly* zero — the `⁺` binding is a
  statement about the economics, not a rounding outcome.
- **Seven mutants, all killed.** The reflow slip (dropping `1/(1+γ)`) dies to **18** tests.

### Corrected — a spec claim, measured

The tranche spec justified the Picard iteration as "a contraction, factor `γ/(1+γ) < 1`". Measured,
the error ratio is **`γ`**: `γ/(1+γ)` is the *solution's* coefficient, not the map's Lipschitz
constant, since `e_{n+1} = −γ·e_n` exactly while the positive part stays open. The consequence is not
cosmetic — the iteration converges only for `γ < 1`, **oscillates at `γ = 1` and diverges above it**,
while the closed form keeps satisfying the fixed point for every `γ ≥ 0`. Both facts are now pinned
by tests rather than left in prose. This strengthens the ruling that the closed form is production: a
"solve the fixed point numerically" instinct would have produced code that silently fails outside
`γ < 1`.

### Stated assumption

`CA = CVA` holds because this spine is **uncollateralized with no initial margin**; in general
`CA = CVA + FVA` (eq 11). MVA and the collateral terms that make `CA ≠ CVA` are deferred (T29 /
`to_go_deeper`), so the assumption is named in the module rather than assumed silently.

## [0.33.0] - 2026-11-09

Tranche 27 — **bilateral CVA + DVA: both parties can default.** Third unit of the XVA spine arc. The
step from "only the counterparty defaults" to the mirror case, and with it the first-to-default
conditioning the unilateral spine dropped.

### Added

- **`credit_adjustment(exposure, defaulting, surviving, discount, lgd)` (L5)** — the one integral
  both credit legs are, extracted at the second real consumer (§6's rule-of-two: T25 was the first,
  DVA is the second). CVA weights the EPE against the counterparty's default; DVA the ENE against
  our own; the unilateral case is `surviving=None`.
- **`CreditParties`** (both curves and both LGDs, §7's compelled bundling — it removes the commonest
  bilateral error, passing the two curves or the two LGDs in the wrong order) and
  **`BilateralAdjustment`** (`cva`, `dva`, with `bcva` a **property** so the net cannot drift from
  its parts; both legs are always reported, since the same small net can be two large offsetting
  legs or two negligible ones).
- **`bilateral_credit_adjustment`** and the pathwise **`mc_bilateral_adjustment`** (a
  `PricingResult` with its standard error, invariant 7), plus
  **`ExposureEstimate.negative_profile()`** for the ENE side.

### Oracles

- **The reduction doubles as the purity proof.** `λ_own → 0` gives `DVA = 0` and the CVA leg exactly
  T25's unilateral CVA. And T25's `unilateral_cva` routed through the extracted integral is
  **bit-identical** — verified against the pre-refactor code from git on four grids (5 to 2001
  points), every digit. `surviving=None` *skips* the multiplication rather than multiplying by 1.0,
  which is what keeps the operation order literally T25's and makes §4's stronger proof available
  instead of an identical-to-a-few-ULP argument.
- **The party-swap symmetry is the independent, non-telescoping anchor** — our DVA equals the
  counterparty's CVA against us, computed from a different composition of the pieces, and on
  simulated paths `ENE(V) = EPE(−V)` exercises the **positive**-part code path on a negated
  exposure. An ENE computed as `−mean(V)` rather than `mean(V⁻)`, or a swapped curve assignment,
  dies here and in no within-DVA check.
- **The first-to-default term is asserted by magnitude, not only sign** (§4's documented blind
  spot): our own default risk *strictly reduces* the CVA, and the term is measured at **2.7%** of
  the unilateral CVA on the fixture — percent, not epsilon.
- `BCVA = CVA − DVA` exactly; both legs non-negative; BCVA either sign; monotonicity in each party's
  hazard and LGD. The pathwise SE is asserted **not** to equal the quadrature sum of the two legs —
  CVA and DVA share paths, so the difference's error is not the sum, which is why the pathwise
  estimator is required rather than convenient.
- **Eight mutants, all killed**: wrong surviving curve, swapped defaulting/surviving, dropped
  surviving factor, flipped net sign, DVA fed the EPE, swapped LGDs, pathwise CVA dropping own
  survival, `negative_profile` returning the EPE.
- Non-degeneracy premises asserted: a genuinely **two-sided** exposure (EPE and ENE within 3× of
  each other, so DVA is not ~0), two **differing** hazards, and two **differing** LGDs.

### Open question raised, not ruled

Crépey conditions through the **bank survival measure** (`EY = (1−γ)⁻¹E*(Y°)`, eq 7 p.37) — his
first-to-default conditioning carries a `(1−γ)⁻¹` **normalisation**. This tranche builds the
market-standard **explicit survival factor**, the natural extension of T25/T26, neither of which
carries a normalisation. Both forms have the term present and both satisfy `BCVA = CVA − DVA`; they
differ by that factor, measured at **4.7%** of the CVA on the fixture. Which is the identity and
which the refinement is a design call reserved for Cowork by the tranche spec, and is raised as an
`Ask:` rather than settled here.

## [0.32.0] - 2026-11-09

Tranche 26b — **the `verify.py references` gate: the register, self-policed.** A standalone
governance tranche, run before T27. §7 floated this gate as *"proposed, not built, until the first
drift makes it earn its place"* — **five** drifts earned it.

### The finding that pulled it

Committing T26, one word of a Cowork edit (`CREPEY-2026 (worked-from)`) did not match
`docs/references.md`. Pulling that thread found five stale statuses, none caught by careful reading:
`CREPEY-2026` and `CAPINSKI-ZASTAWNIAK` still `in-hand` though T25 had built against them;
`JAECKEL-2002` **self-contradictory** (header `in-hand`, body `worked-from`, 18 citers);
`MCINERNEY-ZASTAWNIAK` `in-hand` with **10** citers and a body promising `worked-from` *"until T10
lands"* — sixteen tranches earlier; and `ACKLAM-2000` `in-hand` with its coefficients transcribed in
`distributions.py`. This is §9's re-audit target (3), *register accuracy*, failing by accident
between re-audits eleven tranches apart.

### Added

- **`verify.py references`** — three checks, keyed on the register's header status field and the
  `source:` citations `provenance` already walks: (1) every cited key **resolves** to an
  `in-hand`/`worked-from` status (never `referenced-by` — §7's no-reference-by-inference, never
  absent); (2) every key cited by ≥1 module is **`worked-from`**, the register's own vocabulary for
  *held and built/oracled against*; (3) every `worked-from` key has ≥1 **citer**.
- `references` added to the §9 gate list; charter §7 and the register's own note record the gate as
  built and name the five drifts.

### Oracles — seeded drift, mutation applied to a gate

- The pure core `check_reference_statuses(statuses, citations)` is **injectable**, and each real
  drift shape is reconstructed and asserted to be **rejected**: a cited key left `in-hand`; a cited
  key absent from the register; a cited key only `referenced-by`; an orphan `worked-from`.
  Extracting the core was necessary rather than stylistic — a gate whose only entry point reads the
  real repo can be asserted to pass today and never *shown to fail*.
- Two **legitimate** states asserted to pass: a held-but-unbuilt source at `in-hand` with no citer
  (exactly `CREPEY-2026`'s state between registration and T25), and a cited key at `worked-from`. A
  gate that blocked honest work is the other way to be useless.
- **The gate's first run on the then-current tree failed**, catching `MCINERNEY-ZASTAWNIAK` and
  `ACKLAM-2000` — two drifts that had survived a hand-correction pass three commits earlier. That is
  the argument for gating a discipline rather than restating it, and it is why this is not a gate
  asserting what was already true.

### Fixed

- `MCINERNEY-ZASTAWNIAK` and `ACKLAM-2000` statuses corrected to `worked-from`, each with the
  tranche and the built modules named (following the same correction to `JAECKEL-2002`,
  `CREPEY-2026` and `CAPINSKI-ZASTAWNIAK` at v0.31.0).

### Also fixed — the gate that floors the gates

- **`verify.py`'s `tests` gate discovered only `tests/L*`**, so this tranche's own `tests/tooling/`
  tests ran under plain `pytest` and passed while the gate counted 902 against a floor of 911 —
  present, green, and **never floored**. That is the "filter with no floor" defect `run_tests`'s own
  docstring warns about, occurring in its *discovery* rather than its `--layer` flag. Discovery is
  now total (layer dirs first, so the lowest layer still fails fast), extracted into
  `discover_test_dirs` so a test can reach it (§4's reachability rule), and pinned by three tests.
  Found because the floor failed on a tranche that *added* tests — the one circumstance that makes
  an unfloored directory visible.

### Scope fence

Status and citation **consistency** only. The gate does not check that a held PDF exists on disk
(they are gitignored and absent in CI, so the check could not fail there — this project's named
defect class), does not parse or fact-check the prose body, and does not verify a formula against
its source. That last is §4's per-tranche job and no gate can do it.

## [0.31.0] - 2026-11-08

Tranche 26 — **the Monte-Carlo exposure engine: CVA of a simulated netting set.** Second unit of the
XVA spine arc. T25 integrated an exposure given in closed form; this one produces an exposure by
simulation, which changes what the oracle can be — T25's closed form is now *our own code* on the
other side, so it validates the estimator and nothing more.

### Added

- **`risk/mc_exposure.py` (L5)** — `SimulatedExposure` (the value matrix, kept as paths),
  `ExposureEstimate` (per-date EPE/ENE as `MonteCarloEstimate`s), `mc_exposure_cva` (the pathwise
  CVA, returning a `PricingResult` with its `standard_error` — invariant 7),
  `swap_values_on_paths` (a forward-starting payer swap on simulated LMM forwards, in both the
  time-`t` and numéraire-relative forms) and `live_exposure_times`.
- **`foundation/time_measure.is_historical(pay, valuation_date)` (L0)** — invariant 6's boundary as
  one atom. All **eight** pay-boundary sites across the five engines (`cash`, `inflation`,
  `nominal_cap`, `capfloor`, `yoy`×4) now route through it, plus the exposure grid. **Register C10
  upgrades `per-module` → `guarded — structurally`**: a divergence between engines is no longer
  *expressible*. Extracted because a real consumer pulled it (§6), not speculatively.

### Oracles — two legs, because self-consistency telescopes

- **Leg A (the estimator)** — the simulated EPE against T25's closed form on a Gaussian mark, within
  `4·SE` at every date, with the error falling at the MC rate (RMS over the grid, since one date's
  error is a single noisy draw). Plus the MC CVA against T25's **analytic** CVA: a cross-product
  oracle, different exposure code each side.
- **Leg B (the process)** — the numéraire-relative swap value reproducing `V(0)/B(0,T_end)` at every
  observation date: **exact by construction, consulting T25 nowhere**. Deviations of 0.11/0.35/1.04
  SE against real path dispersion. This is what a wrong drift or measure cannot survive and what
  Leg A is structurally blind to.
- **The pathwise/profile identity** — the pathwise CVA mean equals T25's integral of the same EPE
  profile to 1e-12, on *both* the Gaussian and the simulated-swap exposures. The module claimed it;
  the suite now asserts it, which is what makes the reported standard error trustworthy.
- **Nine mutants, all killed.** Informatively, dropping the numéraire division died to **exactly one
  test — Leg B** — direct evidence that leg is load-bearing rather than redundant; dropping `DF(t)`
  died only to the T25 cross-check. Flipping the `is_historical` atom's `<=` fails six tests.
- Non-degeneracy premises asserted: non-zero exposure variance, more than one future date, a
  non-flat numéraire across the observed dates, and the swap exposure measurably **skewed** (so
  Leg A's Gaussian closed form provably cannot reach it).

### Design notes

- **The CVA is estimated pathwise, not by combining per-date standard errors.** Each path
  contributes its own `∫DF·V(t)⁺·dPD`. Quadrature-summing per-date errors would assume the dates
  independent when one path visits every one of them, understating the error — and the pathwise form
  satisfies invariant 7's correctness point directly, since the error comes from the same paths as
  the price.
- **The swap is observed only before its first reset, deliberately.** The martingale identity holds
  for a *fixed* claim; once a coupon has paid, the remaining swap is a different claim and
  `E[V/B_N] = V(0)/B_N(0)` is false. Observing a forward-starting swap keeps Leg B **exact** rather
  than approximate, and the engine **refuses** a later observation index rather than silently
  asserting something untrue. The exposure is still lognormal, which is the point of the leg.

## [0.30.0] - 2026-11-08

Tranche 25 — **unilateral CVA and the survival curve: the XVA spine opens.** First unit of the XVA
arc (`docs/xva_spine_arc.md`), and the first adjustment this tree has ever priced — it replaces the
archived predecessor's faked `risk/xva.py`, whose register rows named a CVA module that never
existed (pruned by tranche 24).

### Added

- **`market/survival_curve.py` (L1)** — `SurvivalCurve`, survival probability from a
  piecewise-constant hazard: `S(t) = exp(-cumulative_hazard(t))`, with `cumulative_hazard` and
  `default_density` (`lambda(t)S(t)`) as named surface rather than steps buried in the exponential.
  Market data, not a model: a hazard rate is quote-side credit, the same status as a zero rate.
  Source `CAPINSKI-ZASTAWNIAK` ch 2.
- **`risk/credit_valuation_adjustment.py` (L5)** — `expected_positive_exposure` (the Bachelier
  positive part `m*Phi(m/s) + s*phi(m/s)`), `ExposureProfile` (+ `.gaussian`), `unilateral_cva` and
  the extracted one-period step `unilateral_cva_one_period`. Crepey Prop 1 eq (17)'s
  `CVA = E[(1-J)P+]` under **exposure/default independence**, which is the simplification this
  tranche makes and the thing wrong-way risk would recover. L5's second consumer after Greeks.
- **A shared-clock guard (register `C12`)** — `unilateral_cva` refuses a survival and a discount
  curve on different `TimeMeasure`s. The profile's `t` is read on both, so two anchors would make
  one `t` denote two dates and the integral would mix them. The C7/C8 family, guarded rather than
  recorded; its test is mutation-proved (removing the guard fails it).
- **`DiscountCurve.df_at(t)`** — the discount factor at a year fraction, with `df(date)`
  delegating. A time-grid integral holds `t` and no date; the delegation preserves operation order,
  so the extraction is proved **bit-identical** rather than ULP-bounded (§4).

### Oracles

- The EPE against the test's **own numerical integral** (independent route) inside the regime where
  that route is accurate, and against scipy's `ndtr` composition at the **extreme ratios the
  profile actually visits** (`m/s` reaches 66), with relative tolerance — an absolute one is blind
  where the EPE is ~1e-24 (T20a's rule).
- The CVA against scipy quadrature of an **independently built integrand**, banded at 2e-04 from a
  measured 9.4e-05, **plus the second-order convergence rate** — the rate is what proves it is the
  same integral rather than merely a close number.
- Crepey's requirement oracles: **two separate zeros** (`LGD = 0` and `lambda = 0`), sign,
  monotonicity in both the hazard level and `LGD`, linearity in `LGD`, and the sandwich.
- The one-period step pinned **bit-identically** to a hand computation.
- **Six mutants, all killed** — dropped `LGD`, density replaced by the survival level, EPE replaced
  by the forward, flat hazard cumulation, dropped `DF(t)`, hazard segment off-by-one. The
  non-degeneracy premises (non-flat hazard, non-flat discount curve, `m/s = O(1)`) are each
  asserted, and each is what kills one of these.

### Fixed

- A **defect in this tranche's own test reference**, caught by the convergence rate rather than by a
  tolerance: the EPE quadrature integrated from `-m/s` over a fixed-width window, which truncates
  above the distribution's mass once the ratio is large — it returned ~0 where the true EPE was
  99.975. It made the CVA error *plateau* at 3e-04 instead of converging. The references now carry
  explicit validity regimes and are asserted to agree where both hold. §4's "a reference is code
  too", fourth occurrence.

## [0.29.0] - 2026-11-08

Tranche 24 — **the conventions register rebuilt to this tree** (re-audit #1 findings F-3.1, F-3.2,
F-2.1). Second and last remediation tranche; **re-audit entry #1 is complete**. Specced as docs-only
— and the honest re-derivation earned §2's optional guard and **found a second hole the audit had
not**.

### A second hole: the inflation cap/floor's valuation-date boundary, worth 41%
Re-deriving the invariant-6 register row meant establishing what the five engines actually agree on,
and the honest way to establish that is to flip the boundary and see who notices. `>` → `>=` **survives
in three of five**. The first instinct — "three engines have an untested boundary" — was **wrong for
two of them**: `nominal_cap` and `yoy` are **proved equivalent mutants**, prices bit-identical across
four schedules including the boundary, because a Hull-White caplet whose expiry has passed is worth
exactly zero so including or dropping such a period changes nothing. I nearly wrote a guard on that
false premise; measuring stopped me.

`capfloor` is the one that mattered. A period that has already paid has **both index levels known** —
the payoff is *determined*, not an expired option — so it carries real value, and including it moved
the price from **5,929.24 to 8,363.52, about +41%**. A genuine mispricing a wrong comparison would
have caused, with nothing to catch it. Closed by
`test_INVARIANT_6_excludes_a_period_PAYING_EXACTLY_on_the_valuation_date`.

### The row is `per-module`, not `guarded`, and that is the honest entry
A cross-module test comparing all five engines would catch **nothing** the per-engine boundary tests
do, while duplicating three market fixtures and claiming a guarantee that does not exist. What would
make C10 structurally guarded is what makes C9 guarded: **an atom**. Five engines inline
`pay > valuation_date`, so a divergence is expressible; extracting one `is_historical(...)` is recorded
as the row's upgrade trigger rather than done, since this was ruled a docs tranche.

### The register, and one of my own rows was overstated
C1–C6 described `risk/exposure.py`, `risk/xva.py`, `engine/linear.py`, `engine/equity.py`,
`models/sabr_model.py` and seven more modules that **do not exist here**. They are now a labelled
*archived* section (lessons only, not inventory), IDs **kept stable** because the handoffs, the audit
docs and CLAUDE.md cite them. The live register is C7–C11, and a check confirms every module it names
exists in `src/`.

**My new C11 row was wrong twice before commit**, the same failure mode as the rows it replaced:
"every L3 model carries a `MarketSnapshot`" (six do; six others are closed forms and parameter objects
that never read market data — that is the convention's *scope*, not exceptions), and "there is no
second place to pass a snapshot from" (false inside an engine — `inflation_leg_value`,
`resolved_index_level` and the calibration objectives all take one legitimately; the guarantee is at
the **entry point**, where every registered pricer is `(product, model)`). A register row is only worth
the grep behind it.

### Also
- `to_go_deeper`: "Six `crosscheck:` markers" → **23** real targets (of 45 headers; the other 22 are
  explicit `none — …` exemptions). "L5 is empty" → corrected: tranche 22 filled it; the *inflation*
  risk layer the route describes is still genuinely unbuilt.
- F-2.1 recorded: the driftless-forward terminal is inlined in `black_mc.py` and `sensitivities.py` —
  rule-of-two has fired, extract at the next touch of either.
- 821 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.28.0] - 2026-11-07

Tranche 23 — **the linker coupon accrual oracle** (re-audit #1 finding F-1.2). The first
remediation tranche from the first whole-tree re-audit. **Test-only: `src/` is unchanged** — the
production code was correct and only its coverage was wrong.

### The hole, reproduced before it was closed
Dropping the `ψ` accrual factor from `engine/inflation.py`'s linker coupon sum **survived all 818
tests**, which I confirmed independently from a clean bytecode state rather than taking the finding on
report. The term is load-bearing in production and was invisible to every oracle.

### The existing oracle computed a `ψ` and was still blind to it
`test_the_linker_matches_the_closed_form_with_accrual_fractions` evaluates
`psi = (pay - start).days / 365.0` over a fixture whose every period is exactly 365 days — so every
`ψ` it computes is `1.0` and `C·ψ = C`. Its name promises accrual fractions the fixture never
delivers. That is a sharper form of the charter's `τ = 1` trap than the existing examples: the test
does not merely *omit* the accrual, it visibly *handles* it, which is why twenty tranches of readers
and two review passes saw a covered term. **A fixture can defeat an oracle that looks correct line by
line.**

### The fixture, and why the accruals are unequal
Semi-annual periods of **182 and 183 days** on ACT/365F, so `ψ ≈ 0.4986` and `0.5014`. Two properties,
the second not required by the spec: **not 1** (which makes the accrual visible at all) and **not
equal to each other** (which kills a mutant replacing `ψ` by any single constant — a fixture of equal
non-unit accruals would have closed F-1.2 while leaving `ψ → 0.5` alive).

| mutant | before | after |
|---|---|---|
| `ψ` dropped (the F-1.2 hole) | **SURVIVED** | **KILLED** |
| `ψ → 1.0` | survived | **KILLED** |
| `ψ → 0.5` (a constant) | — | **KILLED** (needs the unequal accruals) |
| `ψ` applied to the redemption | — | **KILLED** |

### The non-degeneracy premise is itself a test
`test_the_SEMI_ANNUAL_fixtures_accruals_are_NOT_degenerate` asserts every accrual is more than 0.4
from 1 and that they are not all equal. This is the guard the tranche exists to install: a later
refactor **cannot** quietly flatten the schedule back to annual periods, which is precisely how the
hole survived. Without it the new oracle would keep passing while silently losing its power over `ψ`.

### Notes
- **No `Ask:` raised and no `src` change**: the non-degenerate oracle passed against unmodified
  production on the first run, so there is no real `ψ` bug — the cleanest outcome for a
  latent-coverage finding, stated plainly rather than left as an absence.
- F-1.2 was found only because the re-audit's sweep was extended past the first pass's domain — §4's
  own domain rule applied to the audit itself. Worth making the default for re-audit #2.
- 820 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.27.0] - 2026-11-07

Tranche 22 — **Greeks by simulation** (`JAECKEL-2002` ch 11), **the last Jäckel unit**. Three methods
— finite differencing with path recycling (§11.3), pathwise differentiation (§11.5), the likelihood
ratio (§11.6/§11.9.1) — all anchored on Black's analytic Greeks. `models/black.py` gains those;
`risk/sensitivities.py` is **the first L5 module**. **Nineteen mutations, all nineteen killed on the
first pass.** Ch 11 closes, and with it the nine-chapter spine.

### The anchor did not exist — the spec assumed a capability we never built
The spec names "Black's analytic Greeks (T9)" as the anchor twice. **There were none.** T9 built the
implied-vol *surface*; a grep of `src/` for any delta/gamma/vega definition found nothing. So the
tranche built its own anchor, and the consequence that matters is that the anchor needed **its own
oracle** rather than being assumed. The chain is now four independently checkable links: `black_price`
(verified since tranche 5) → its central difference → the analytic Greek → the Monte Carlo Greek.

Worth recording as a pattern: a spec can assert a *capability* as confidently as it asserts a number,
and this is the first time the missing thing was a whole function rather than a value.

### The analytic Greeks are proved as a LIMIT, not a point comparison
A first version compared each Greek to one central difference at `h = 1e-5` and **failed** deep
out-of-the-money by ~4e-6. The finite difference was the weaker side: varying the step, its gap falls
5.63e-2 → 5.60e-4 → 5.60e-6 → 5.60e-8 → 3.28e-10 — textbook `O(h^2)`, converging **onto** the analytic
value. So the formula was right and the reference was truncating by a hundred times the bound. The
oracle now asserts the **rate**, which proves the closed form is the *limit* of the numerical
derivative — a stronger claim, and one a wrong `d+` could not fake. Measured ratios 87 to 102 per
tenfold step. Gamma's floor is looser (a second difference divides by `h^2`), parameterised and stated
rather than hidden under one bound for both.

### The companion that earns the second method
Pathwise matching on a call is vacuous alone — finite differencing would do. What justifies a second
method is a payoff where pathwise is *wrong*: a digital's payoff is flat almost everywhere, so its
pathwise derivative is **exactly 0.0 on every path** while the true delta is strictly positive. The
oracle asserts the zero as a number against an analytic digital delta built for the purpose, with the
likelihood ratio holding there. **Pathwise is also refused for gamma**, because §11.5 refuses it — it
gives a first derivative per path, not a second, so the method raises rather than approximating.

### The variance ranking inverts the obvious guess
200,000 paths, at-the-money, SE as a percentage of the analytic value:

| Greek | finite difference | pathwise | likelihood ratio |
|---|---|---|---|
| delta | **0.25%** | **0.25%** | 0.56% |
| gamma | 4.35% | *refused* | **1.44%** |
| vega | **0.42%** | **0.42%** | 1.44% |

The likelihood ratio is 2-3x noisier for delta and vega — its weights carry `1/(sigma sqrt(dt))`, as
Jäckel's own caveat implies — but **3x sharper for gamma**, because a second finite difference divides
by `bump^2` and amplifies what it differences. And FD's gamma is visibly **off-centre** (55.47 against
49.62, 2.7 SE — inside the oracle's band but a truncation bias, not noise). So for gamma the likelihood
ratio is both sharper and better centred; the ranking is not a matter of taste.

### L5 is open, deliberately without a protocol
The charter's layer box has carried "L5 RISK & CAPITAL — on the engine + a Priceable protocol" since
tranche 1, and L5 has been empty for twenty-one tranches. **No `Priceable` protocol**: this module
reads Black's closed form and the MC machinery directly — one shape of consumer — so a protocol would
be an abstraction with a single implementation, which §6 forbids. It arrives at the second kind of
priceable that needs differentiating; the charter clause is half-satisfied and says so.

`BlackState` is the bundle §7's limit forced (`black_mc_vega` was six arguments). `years` is read only
by vega — not a carried-but-unread field, since `total_variance = sigma^2 * years` makes the horizon
what separates a one-year 20% option from a four-year 10% one.

### Notes
- The likelihood-ratio weights are adapted from spot to **forward**: Jäckel writes
  `S_T = S_0 exp((r-d-sigma^2/2)T + sigma sqrt(T) z)`; a forward is driftless under its own measure, so
  the weights carry over with `S_0 -> F_0` and `r-d -> 0`. Stated, because an unstated convention
  change is how a right formula becomes a wrong number.
- Eq 11.32 (the LR Vega) **is** printed in full for the single-horizon case, so the Acklam pattern did
  not bite here.
- 818 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.26.0] - 2026-11-07

Tranche 21 — **the Brownian bridge** (`JAECKEL-2002` §10.8.3, appendix §10.9.2), and Sobol wired into
the LMM path. `foundation/brownian_bridge.py` (L0) plus the engine's `Sampling` bundle. **Thirteen
mutations, all thirteen killed.** This **completes chapter 10**; only ch 11 (Greeks) remains before the
nine-chapter spine closes and the first whole-tree re-audit fires.

### The anchor is exact — no Monte Carlo in it at all
The spec asked for the covariance identity "to tolerance". It needs none. The bridge is a **linear** map
of its variates, so its matrix `A` is recovered exactly from unit basis vectors and "same Gaussian,
reordered" becomes `A A^T = min(t_i, t_j)` — the Wiener covariance — checked to machine precision at 2,
3, 4, 5, 8, **14** and 16 steps. There is no estimator, so there was no tolerance to choose. The same
holds one level down: the increment map's Gram matrix is exactly the identity.

14 steps is in the sweep because it is the source's own worked example (figure 10.4) and is **not** a
power of two — the case §10.8.3 singles out for its general "halve into two parts containing
approximately the same number of points" rule.

### A source misprint, faithful in both renderings
Appendix §10.9.2's eq **10.76** prints `W_tk = W_tj + sqrt(t_k - t_i) z`, contradicting eq 10.77 three
lines later (`z := (W_tk - W_ti)/sqrt(t_k - t_i)` forces `W_ti`); with `W_tj` the first increment is
counted twice. **Both renderings print it identically**, so transcription independence is powerless and
only the internal inconsistency catches it. It does not reach the build — eqs 10.33/10.34 are what is
implemented, and they were re-derived independently rather than read from the typo'd sketch. Worth
recording because §4 treats a derivation as an independent check on its result, and here the check
itself had to be repaired before it could be used.

### The claim holds, but only one half of it is assertable
Measured through the engine on an LMM caplet at 32 Sobol dimensions, the fitted exponents are
**pseudo-random -0.255, Sobol+incremental -0.616, Sobol+bridge -0.940** — the bridge nearly reaching
the theoretical -1, and at 16,384 paths **20x better than pseudo-random, 3x better than Sobol alone**.

But the spec asked for the pseudo-random comparison as a fitted exponent and **that is not assertable**:
one seed is one noisy realisation, and its slope swings from -0.637 to -1.056 depending only on where
the range ends — on the shorter range it *beats the bridge*, by luck. So bridge-vs-incremental is
compared on the **exponent** (both Sobol, hence deterministic) and the pseudo-random comparison on the
**RMS error over the range**: bridge/incremental 0.35, bridge/pseudo 0.17, identical to two decimals
across both ranges. Third tranche running in which a headline claim had to be restated in the form
measurable above its own noise.

### The null result needed averaging too, which is the subtler half
Under pseudo-random draws the bridge must change nothing — it is a reordering, so it gives a different
*sample*, not a better one. A first version of that oracle used one seed and measured the bridge **6x
better**: that looked like a result and was sampling luck. Averaged over twelve seeds at two path counts
the RMS ratio is **1.31** (and 0.92 / 1.26 / 1.29 under three other choices) — neutral, as the algebra
requires. **A null measured on one sample is not a measurement.**

### The ordering is the purpose, and no output property could see it
Twelve mutations died at once; the survivor was **depth-first instead of breadth-first subdivision**.
A real hole, not an equivalent mutant: it still places every point once, still brackets correctly, still
yields a valid Wiener path — so the exact covariance identity **cannot see it** — while breaking the
entire purpose, since the second variate would shape a small early region rather than the global
midpoint. Tranche 18's lesson in a new place. The distinguishing property is the bracket **width**:
non-increasing up to a slack of one (the source's own "approximately the same number of points"; a
first attempt demanded strict monotonicity and failed at 14 steps, where unequal halving legitimately
puts a width-3 bracket before a width-4 one).

### Notes
- `Sampling(source, construction, mirrored)` is the bundle §7's five-argument limit forced, and the
  right answer: three sampling choices now have real consumers (Sobol, the bridge, antithetic). It
  defaults to exactly tranche 16's behaviour, which is why all 784 existing tests passed unchanged.
- Jäckel's "all of the weighting coefficients can be precalculated" became
  `BrownianBridge.increment_matrix()`, so the engine reorders every path and factor in one `tensordot`
  rather than a Python loop per path — the cost pattern I walked into in tranche 19.
- **The Bermudan's own path still cannot use Sobol**: six forwards over eight steps is 48 dimensions,
  past the 32 Table 8.3 covers. So T20b's deferred acquisition is now load-bearing for a real consumer.
- 787 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.25.0] - 2026-11-07

Tranche 20c — **`norm_cdf` moves to `0.5·erfc(-x/sqrt2)`**, fixing the lower-tail cancellation T20a
found. A correctness refactor of our own L0, no external source. **OI-3 closed.** Six mutations, all
six killed — including a revert to the superseded form, which is the one that proves the new oracles
guard the fix.

### The defect was worse than the ledger recorded: 100%, not 6.1e-06
OI-3 cited 6.1e-06 relative error at `x = -7.03`. Pre-registering the oracle showed the worst case over
`[-9, 9]` is **1.0** — below about `x = -8.3` the old form returns **exactly 0.0**. At `x = -9` the true
probability is ~1.1e-19 and the function returned zero: not an inaccurate probability but a lost one.
The honest characterisation is not "degrades in the tail" but **"returns zero past a threshold"**, and
T20a's single measurement happened to land just above the collapse.

### Purity as a measured partition, because a flat ULP claim would have been false
The exact-arithmetic identity holds (`erfc(-y) == 1 + erf(y)`), but the bounded-agreement half cannot be
one tolerance: `x >= 0` agrees to **1 ULP** (bit-identical at 88% of a fine grid; both forms accurate to
~2e-16 against the reference), tens of ULP down to `x = -4`, then 3.9e-11 at `-5` and 1.3e-10 at `-6`
— and **that divergence is the OLD form's error**. It is *attributed* rather than noted: the oracle
requires the new form to track `scipy.special.ndtr` (`< 1e-14`) **and the old one not to** (`> 1e-11`),
because an unexplained divergence in a pure refactor is a drift.

The partition bounds the blast radius for the nineteen tranches that read `norm_cdf`: anything
evaluating `Phi` above about `x = -4` moves by at most tens of ULP, which no tolerance in the suite can
resolve. The suite stayed green unchanged at 760 tests — the corroboration this predicts, not the proof.

**My first claim was too strong and my own oracle caught it:** I asserted *bit*-identity on `x >= 0`
and it failed at `x = 1.2`. For positive `x` both routes produce a value near 1 and round differently
in the last bit — no cancellation, so no accuracy loss, but not identical. One ULP is the true claim.

### The workaround was removed, not left alongside
T20a added a private `erfc` sibling because `norm_cdf` could not be trusted in the tail. `norm_cdf`
**is** that form now, so the sibling is deleted and there is one CDF again (§6). Checked first that no
oracle depended on the two forms *differing* — none did; what needed updating was a comment explaining
why the reference avoided `norm_cdf`, a reason that no longer exists.

### A side finding: we resolve further than the reference
At `x = -38` scipy's `ndtr` underflows to **exactly 0.0** while `erfc(-x/sqrt2)/2` still returns a
subnormal (~2.9e-316). So the deep-tail bound loosens with depth (1e-13 to `x = -20`, 1e-12 to `-37`)
and that is **the reference's** accuracy, not ours — a first version asserted 1e-13 everywhere and
failed at `x = -30`. That is §4's "a reference is code too" — the note this arc added one tranche
earlier — biting in the tranche whose oracle *is* a reference comparison.

### Notes
- No candidate charter rule from this tranche, which is worth saying: every discipline it needed was
  already in §4. The charter earned its keep rather than growing.
- 761 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.24.0] - 2026-11-07

Tranche 20b — **Sobol' low-discrepancy numbers** (`JAECKEL-2002` §8.3), consuming tranche 20a's
inversion. `foundation/sobol.py` (L0, finance-free). **Fifteen mutations, all fifteen killed on the
first pass.** The fitted convergence exponent on a Black call is **-0.906** against pseudo-random's
**-0.481** (theory -1 and -0.5); at N = 32,768 the error is 8.4e-07 versus 2.7e-05 — 32x better.

### A ~350-number table, extracted twice by machine rather than read twice by eye
Table 8.3 gives 32 dimensions x (degree, primitive polynomial, ten initialisation numbers). §4's
dual-rendering rule exists for exactly this, but *reading* it twice by eye is the **weaker** method at
that scale — eye fatigue is correlated between passes in a way `pdftotext` is not. Both renderings
were parsed and reconciled mechanically (`-raw` reflows the table to one value per line and needed a
different parser, which is itself evidence the two are independent): **zero disagreements** across 32
rows, every exponent confirmed as `32 - l`. The shipped table was *generated* from the reconciled
extraction, never typed.

### What makes the table verifiable is a property the source states about it
T20a showed how weak an oracle over a transcribed table can be — there the refinement repaired small
coefficient errors. Here the caption hands over something sharper: *"Initialisation numbers providing
property A up to dimension 32."* Property A is testable — in any binary segment of length `2^d`
aligned to a multiple of `2^d`, each of the `2^d` sub-hypercubes holds **exactly one** point — and one
mistyped direction integer breaks a count. Verified over four aligned segments per dimension for
`d = 1..8`; beyond that a segment is more draws than is practical (`d = 32` is 4 billion), so the
higher dimensions rest on the source's two bit constraints over all 32x32 direction integers, which is
how the recurrence is verified where the book prints no values at all.

### The first segment is short by one, and telling that apart from a defect was the work
Property A failed first run at "15 of 16". Not a defect: the property is stated over the sequence
*including* `n = 0`, the origin, which we skip because `0.0` maps to `-inf` under `Phi^-1` and T20a's
guard refuses it. §8.3.4 says exactly what follows — the first segment "will have no point in the
sub-hypercube `[0, 1/2)^d`". The oracle now asserts the **exact** expected shape (first segment hits
`2^d - 1` cells, the missing one being the all-lower cell) rather than loosening a count, because "one
cell short" is also what a transcription error looks like.

### Dimension 1 is the construction at `g_k = 0`, not a special case
Its polynomial is degree 0, so there are zero freely-chosen initialisation numbers and eq 8.19 reads
`v_1l = (v_1l >> 0)` — vacuous, which the first implementation reported honestly as an `IndexError`.
What remains is `v_1l = 2^(b-l)`, the unit initialisation of eq 8.24, which is exactly what Table 8.3
prints for `k = 1`. So dimension 1 is the van der Corput sequence in base 2. Note the asymmetry the
source flags: unit initialisation is valid everywhere and "leads to surprisingly bad results ... and
should be avoided" — it is right for dimension 1 only because there the construction forces it.

### Where the advantage washes out (the spec's TO MEASURE item)
Discrepancy proxy at 1,024 points: **10-20x** better through `d = 5`, **1.8x** by `d = 8`, **1.0x —
gone — by `d = 24`**, and at `d = 32` Sobol is *worse* than pseudo-random. With the caveat that
matters: this is a statement about `N = 1,024`, not about Sobol — property A at `d = 32` needs `2^32`
draws, so the sequence has had no chance to equidistribute. The honest reading is that *at feasible
path counts the advantage is a low-dimension phenomenon*, which is precisely what tranche 21's
Brownian bridge exists to exploit. In 1-D the proxy is **exactly zero** (with `N = 2^k` Sobol hits
every dyadic box), which crashed the first measurement on a division — and is why the spec forbids
oracling discrepancy in 1-D alone.

### Notes
- **Sobol is not wired into the LMM path engine.** An LMM path needs one dimension per forward per
  step (6 forwards x 8 steps = 48), past where the advantage has washed out at feasible path counts;
  wiring it before the Brownian bridge would add surface that does not pay. Left for T21.
- Dimension 33 is **refused**: the source's polynomials beyond Table 8.3 are "on the accompanying CD"
  we do not hold — the third time this book has sent a reader to something it does not contain.
- 756 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.23.0] - 2026-11-07

Tranche 20a — **the inverse normal CDF**, closing the inversion tranche 15 deliberately deferred.
`ACKLAM-2000` registered `in-hand` (acquired by Bernardo) — the algorithm `JAECKEL-2002` §9.3
recommends and prints nowhere, which is what stopped T15. Acklam's rational approximation plus one
Halley step against an exact CDF: worst miss **7.66e-16** against an independent implementation,
sub-ULP. **Sobol is NOT built — it is T20b** (see `docs/tranches/20_inversion_sobol.md` §0).

### Our `norm_cdf` is relatively inaccurate in the lower tail, by nine orders of magnitude
The refinement needs `Phi(x)`, and using our own `norm_cdf` broke the deep tail: it computes
`(1 + erf(x/sqrt2))/2`, and as `erf -> -1` the sum **cancels catastrophically**. Measured against an
independent high-precision CDF at `x = -7.03`: ours **6.1e-06** relative, `erfc(-x/sqrt2)/2`
**7.4e-15**. Acklam prints both forms side by side ("use this line ... OR this; not both") and says
he used Cody's `erfc` — this is why. The refinement therefore uses `erfc`, and the upper half is
mapped onto the lower by the exact oddness of `Phi^-1`, so the CDF is only evaluated where it is
accurate.

`norm_cdf` itself is **raised, not patched**: it has shipped since tranche 5, nineteen tranches of
oracles read it, and the change is not bit-identical. Worth noting why its own oracle missed this —
it asserts "tails -> 0/1", an **absolute** claim that passes however bad the relative error gets.
§4's domain rule, found in a function green for fifteen tranches.

### Three times the reference was the weaker side
A root-find built on `norm_cdf` disagreed with the implementation by 3.9e-07 in the deep tail — and
**ours was right**, the reference carried the cancellation. At `p = 1 - 1e-12` the reference could not
resolve a root at all (`Phi` is flat at 1 in double precision), so it became unresolvable before the
code became wrong. At `p = 0.999` it missed an independent implementation by 3.74e-15 while ours was
**bit-exact**. The primary cross-check became `scipy.special.ndtri` — a different algorithm and a
different provider, exactly the arrangement tranche 11 set up by moving `distributions.py` off scipy
"so the oracle and the implementation no longer share a provider". The lesson: *a reference is not the
authority merely because a test calls it one.*

### Halley's third-order term is not measurable from Acklam's start
From a start good to 1.15e-9, Newton squares the error to ~1e-18 — below double precision — so Halley
and Newton **land on the same answer**: worst gap 4.44e-16, machine epsilon. A first version of the
oracle asserted Halley was measurably better and failed with `4.44e-16 < 4.44e-16`. The denominator is
still transcribed (the source specifies it) and its necessity is shown from a deliberately degraded
start, where cubic and quadratic convergence separate — otherwise it would be unfalsifiable surface.

### The oddness oracle could only be exact where the inputs allow it
`Phi^-1(1-p) = -Phi^-1(p)` is exact in arithmetic, but testing it needs `1 - (1 - p)` to return `p`,
which in binary holds only for **dyadic** `p`. At `p = 1e-10` only about four digits survive, so a
first version measured asymmetry in the **inputs** rather than the function. Split: exact on dyadics,
elsewhere bounded by each input's own measured representation loss.

### Mutation: sixteen run, and the four survivors are the resolution rather than holes
The Halley refinement **repairs** small coefficient errors, so the refined output is nearly
indifferent to the transcription. Two survivors were proved **bit-identical** (a 1e-14 coefficient
change; a 1e-4 break-point change — which is why Acklam put the break-point where both branches
agree). The only oracle with grip is the published bound on the *unrefined* approximation, measured
at **1.1288e-09 against a published 1.15e-09 — 98% of it** — with a resolution of ~20 ULP per
coefficient; a 4.5e-9 relative error moves it to 4.40e-08, 38x the bound. An oracle proving the bound
**can fail** was added, since otherwise the one check guarding 21 magic numbers would be unverified.

A wrong diagnosis on the way, recorded because it was stated confidently: the four survivors were
exactly the length-preserving edits, so I concluded Python's bytecode cache had run stale code and
reported false survivors. It had not — `-B` with a cleared `__pycache__` gave the same four, and a
direct probe confirmed the mutations execute. The correlation with edit length was coincidence.

### Notes
- `inverse_norm_cdf` refuses the closed interval: `Phi^-1(0)` is `-inf`, not a large number. This is
  where tranche 15's removed redraw loop relocated to — that branch was unreachable from any test,
  and the domain check belongs where a test can call it with `0.0` directly.
- 698 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.22.0] - 2026-11-05

Tranche 19 — **variance reduction I**: antithetic sampling (`JAECKEL-2002` §10.1) and control
variates (§10.3). First unit of the Jäckel completion arc. Both reducers at L0
(`foundation/mc_estimator.py`, finance-free), both wired at L4 — antithetic as a mirrored-draw mode on
the path engine, the control variate into the Bermudan pricer. **Sixteen mutations, all sixteen
killed on the first pass.**

### The spec's headline was refuted: the Bermudan sandwich is structural, not sampling error
The spec expected the control variate to narrow tranche 18's 15.75% bracket. It cannot. Measured over
a **64x path increase**, the width holds at 15.19% / 15.75% / 15.08% / 15.39% while the standard error
falls 151.25 -> 18.75 — a factor of 8 across 64x paths, the `-1/2` rate exactly. Measured in standard
errors the gap *grows* from 5.9 to 48.8. The width is the perfect-foresight bias plus the policy's
suboptimality, both structural, and tranche 18's own numbers already implied it (the gap was 12 SE
wide, not one). So the deliverable is **sharper error bars**, not a narrower bracket — and the oracle
now asserts that the reduction shrinks the SE and leaves the bracket *invariant*, which is itself the
check, since a reduction that moved the bracket would be introducing bias.

### The spec's chosen control was biased; an exactly-centred one exists and is better
Eq 10.8 is unbiased *only* because `g*` is the **exact** `E[g]`. The spec named the Rebonato swaption,
whose mean is an approximation (§5.11: the LMM swap rate is not lognormal) — centring on it injects
the ~0.5% freezing error as a permanent bias, which is what the spec's own oracle 1 forbids.

The alternative needs no model result at all: under the `P_N`-forward measure every
asset-over-numeraire ratio is a **martingale**, so the numeraire-relative annuity `A/B_N` has
expectation `A(0)/B_N(0)` exactly, by construction. Measured correlation with the Bermudan payoff
**+0.7368**, a **54% variance reduction** (SE x 0.676), zero bias, with the martingale property
verified (simulated mean within 0.66 SE of the analytic value). The intrinsic at the first exercise
date correlates marginally better (0.7444) but has no closed-form mean, so it is not a control.

### The constant-control fallback is the finance, and it saved tranche 18's sharpest oracle
For a single-period residual swap `A/B_N` collapses to exactly `tau` on every path — the identity
behind tranche 17's one-period reduction — so the control has zero variance and `beta* = Cov/V[g]` is
undefined. A constant control is the *absence* of one, and the engine falls back to the plain
estimator. This is also how the issue surfaced: four tranche-18 oracles went red on the first wiring,
because T18's oracle 1 demands a one-exercise-date Bermudan match the T17 European **bit-identically**
and reducing one side broke it. Widening that back into an SE band would have undone the power T18
gained — it was precisely an 8-11%-resolving band that hid a 1.2% defect there.

### Jäckel does not print the variance factor
Eqs 10.7/10.8/10.9 are printed and both renderings agree exactly. The factor `V[v](1 - rho^2)` is
**not in the source** — the second time this book has sent us to a formula it does not print (after
T15's inverse CDF). It is *derived* from 10.9 in two lines and asserted as a **theorem**, verified to
2e-17 against a synthetic pair whose correlation is fixed by construction. §10.3 also offers a third
route for `beta` the spec's framing omitted — a pilot simulation, which is both optimal and provably
unbiased; `control_variate_estimate` accepts an explicit `beta` so a caller can supply one.

### Notes
- Two of my own test errors, caught by execution: "antithetic has exactly zero SE on a linear payoff"
  is zero in R and **1.5e-18** in floating point (the summands round, not the estimator), and I had
  the centring-bias **sign** backwards — eq 10.8 adds `beta(g* - g)`, so raising `g*` raises the mean.
- A process error worth recording: I backgrounded `verify.py slow` and then ran the full suite and
  pyright on top of it. The gate reported a file at **690.6s** against a true 1.75s — a 400x artefact
  of CPU contention I created. A timing gate's reading is only valid when nothing else competes.
- Three `# noqa: E731` suppressions were removed rather than ledgered (lambdas became named
  functions), and a 6-argument test helper I introduced in T17 was bundled to satisfy §7.
- 674 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.21.0] - 2026-11-05

Tranche 18 — the **Bermudan swaption under the LMM**, Jäckel's headline instrument, priced inside a
sandwich because it has no closed form. Third and final unit of the BGM/J arc (T16 → T17 → T18), each
anchored on the one before, with tranche 17's caplet→Black exact anchor underneath all of it.
`models/exercise_boundary.py` (L3), `products/bermudan_swaption.py` (L2),
`engine/bermudan_swaption_mc.py` (L4), and the path widening in `engine/mc_path_engine.py`.
**Eighteen mutations, all eighteen killed.**

### The source is internally inconsistent — eq 12.38 contradicts eq 12.39
Both renderings of eq 12.38 print `+ p_i3`. But eq 12.39 pins the final slice to `(0, 0, K)`, and
§12.7 states what that must mean: *"at the very last exercise opportunity we have exact knowledge if
exercise is optimal, namely when the residual swap is in the money."* At the last date the residual
swap is one period, so its swap rate IS `f_(n-1)`; in-the-money for a payer is `f > K`. The printed
sign gives `f + K > 0` — true for every positive rate, so it exercises a deep out-of-the-money payer,
and it misclassifies every OTM payer across seven forwards and both directions. Ruled by the finance
per §4: 12.39 carries exact financial content, 12.38's sign convention is arbitrary.

**And the arbitrariness explains how the error survives publication.** At every slice except the last,
`p_i3` is a free fitted parameter, so the optimiser simply converges to `-p_i3` and the two
conventions describe the same family of boundaries. The inconsistency is invisible wherever the
parameters are trained and decisive only at the one slice where they are pinned. Both facts are
oracled, and the printed variant is asserted to **fail**, so the ruling is proved necessary rather
than preferred. Recorded in `docs/references.md` beside the five found in `BERHAN-2012` and the one in
`MCINERNEY-ZASTAWNIAK`. No ordinal claimed: `BERHAN-2012`'s 4.1.3 and 4.2.2 are internal
inconsistencies as well, so "the second" would have been wrong.

### Jäckel's prescribed optimiser cannot train Jäckel's objective
§12.8 says to use BFGS. **BFGS makes no progress at all** on eq 12.41, which averages an
indicator-selected quantity and is therefore piecewise constant in `p`. Measured at the first training
slice: start (the European policy) `U = 0.00290435`; BFGS `0.00290435` at **iteration 0**; Powell
`0.00363794`; Nelder-Mead `0.00374813`; attainable ceiling `0.00420242`. So BFGS captures 0% of the
available gain and Nelder-Mead 65%. The finite-difference gradient is exactly zero at `h = 1e-8` and
`1e-6`, non-zero only at `1e-4` — what a gradient method sees is a step-size artefact, not a
derivative. `train_slice` uses Nelder-Mead and the deviation is **oracled**: a test asserts BFGS
leaves the objective untouched while the derivative-free method improves it.

### The sandwich, and what it resolves
A 6-non-call-2 struck at the forward swap rate, 4,000 training / 8,000 pricing paths: exercisable only
at its last date 1,484.64; **trained policy (lower bound) 5,860.27**; **perfect foresight (upper
bound) 6,783.50**. The early-exercise premium is resolved at 14x the combined standard error. The
**sandwich is 15.75% wide** and that IS the tranche's resolution, reported rather than assumed tight.
In-sample bias measured at +1.12%, which is Jäckel's "slight upward bias" and the reason the
train/price split is a correctness requirement.

**One honest negative:** the lower bound does **not** improve monotonically with training paths —
500 → 5,890.09, 1,000 → 5,862.98, 2,000 → 5,849.68, 4,000 → 5,860.27, 8,000 → 5,880.16, a ~0.7%
wobble with no trend. The three-parameter boundary is already at its capacity for this trade, so extra
training moves the fitted boundary within a plateau rather than improving it. The place to spend
effort is a richer boundary, not more paths.

### Five mutations survived the first pass, and one was this tranche repeating tranche 17's lesson
A dropped numéraire division in the annuity hid inside oracle 1's `4 x combined SE` band, which at
these path counts resolves only **8-11%** of the price — exactly the trap tranche 17 ratified into §4,
walked into one tranche later. The fix was not more paths but noticing the comparison can be **exact**:
after this tranche's widening a one-date Bermudan's schedule *is* `PathGrid(T, steps)`, so both engines
consume the same stream; oracle 1 now asserts bit-identical PV and standard error.

The other four were all pieces of §12.8's *procedure* — the induction direction, the continuation
roll, the `intrinsic > 0` guard, and the choice of second coordinate (`s_{i+1}` vs `s_i`, §12.6's
entire finding) — and **no price-level bound could see any of them**, because a valid-but-wrong
procedure still satisfies the output bounds. Each is now pinned by a pointwise check on a named step,
which is why `roll_continuations`, `train_exercise_policy` and `ExerciseSlices` were extracted from
private helpers into named, oracled surface.

### The engine widening
`simulate_forward_paths` is now the single definition of eq 13.2, with `simulate_terminal_forwards` a
last-slice read of it — the rule-of-two consolidation tranche 17 deferred to this tranche's real second
consumer. Proved **bit-identically**, the right bar because the refactor reorders nothing: a
one-segment schedule reproduces the old terminal result, and a two-segment schedule reaches the same
endpoint, which also proves segmenting does not disturb the random stream.

### Notes
- The non-recombining tree (ch 13) is **not** built, per the spec: Jäckel uses it to *discover* the
  coordinate pair, and he states the pair he settles on. Recorded as a `to_go_deeper` route.
- 657 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.20.1] - 2026-11-05

**§3 invariant 7 implemented** — *an estimate carries its error*. Ruled by Cowork as tranche 17's
finding 5 and accepted by Bernardo the same day; `OPEN.md` OI-2 ledgered the gap between the charter
and the code for exactly as long as it existed, and is now closed.

`PricingResult` gains `standard_error: float | None = None`. `price_european_swaption` sets it,
scaled by the same deterministic `discount x notional` that scales the PV, and the banded oracles
read `out.standard_error` instead of re-simulating to recover a number the engine already had.

`None` is load-bearing and is not `0.0`: zero would claim a perfectly precise simulation, while
`None` says *this is not an estimate*. Exact pricers inherit it from the default, so no other engine
was touched — the field rides along only where it is meaningful, which is the half of the invariant
that keeps it from becoming surface on sixteen engines that will never set it.

### One part of Cowork's build note did not apply
It said "the caplet and swaption pricers set it". `mc_caplet_price` returns a `MonteCarloEstimate`,
not a `PricingResult`, and that type has carried `mean`, `standard_error` and `n_samples` since
tranche 15 — its docstring already states the invariant's principle ("an MC engine that returned only
a price would make every downstream oracle invent a tolerance"). The caplet pricer already honoured
invariant 7 by a different route and needed no change; converting it to return a `PricingResult` would
be a design decision nobody ruled, so it was not done.

### Notes
- `_estimate_from_paths` survives in the oracle with a narrower and better purpose: an independent
  recomputation of the payoff on the same paths, pinning the engine's mean *and* its error bar,
  rather than supplying a number the engine could have been wrong about.
- Seventeen mutations: the thirteen tranche-17 mutants behave exactly as before (twelve killed, the
  transposed-rho equivalent mutant still proved bit-identical), and four new mutants on the error
  bar's scaling — unscaled, zeroed, left `None`, and double-scaled — all die.
- 606 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.20.0] - 2026-11-05

Tranche 17 — the **European swaption under the LMM**, anchored by an exact reduction and a bounded
approximation. Second of the three-unit BGM/J arc. `models/swap_rate_vol.py` (L3, Rebonato's
approximation), `products/swaption.py` (L2), `engine/swaption_mc.py` (L4), plus the correlation
widened from a scalar to a full matrix — the widening tranche 16 said would arrive "with the unit that
needs one", and the swaption is that unit.

### The anchor: a one-period swaption IS a caplet, exactly
A one-period swap's rate is exactly its forward and its weight exactly 1, so Rebonato's vol collapses
to `v = sigma_i` to 1e-15 at every index. This engine and tranche 16's caplet engine then agree to
**1e-12** on one seed after a deterministic scaling — a cross-product oracle, different code each
side, no Monte Carlo error in the comparison. It fails under any error in the annuity, the swap rate,
the numeraire bond or the discount index, and in mutation it did.

### The cross-source independence the spec counted on does not exist
The spec specified the swap-rate-vol approximation as transcribed from both books and reconciled —
"the strongest independence we have". Read in both, they are **not one formula**. Both write the
variance as `Z C Z^T`; McInerney eq 5.40 uses the **frozen** weight `w_k F_k / S`, Jaeckel eqs
12.18/12.21 the **exact** derivative `(dSR/df_k)(f_k/SR)`. Jaeckel's is McInerney's first term plus
the two terms arising because the weights themselves depend on the forwards — and McInerney's own
Remark 5.13 says 5.40 can be improved by not freezing them. Two approximations of different order,
differing by **0.45%** in vol, not two renderings of one. Claiming the independence would have been
false; Jaeckel's form is kept as a companion whose gap is asserted small *and non-zero*. The signature
that they are different objects: `sum Z_frozen = 1` exactly, while `sum Z_exact` is 0.9955.

### A non-uniform fixture can still be degenerate
Two of thirteen mutations survived all 603 tests, both because of fixtures rather than code, and
neither is the flat-`tau` trap the charter already names. A correlation `exp(-beta|i-j|)` is
non-uniform yet **translation-invariant**, so `rho[first+m][first+l]` equals `rho[m][l]` and a mutant
dropping the span offset is invisible; all-`0.5` accruals make `accruals[:count]` and
`accruals[first:first+count]` the same slice, hiding the offset a second time. Jaeckel eq 12.12
decays in the actual times `t_i`, not index distance — so the faithful reading is also the
non-degenerate one. Sixth appearance of the trap, first where the degeneracy is *uniformity across a
slice* rather than a specific value. A third survivor — transposing `rho` — is a proved equivalent
mutant: symmetry is enforced at construction, and the vol is bit-identical across seven spans.

### An oracle claimed a power it did not have, and its own power assertion caught it
The bounded cross-check first asserted agreement inside one flat 3% band, called that "the freezing
slack", and claimed a 10% vol error would fall outside. It **failed**, and both halves were wrong.
The freezing error measured at 200,000 paths is 0.15%-0.96%, settling near 0.5%; the ~2.3% seen at the
committed path count is mostly Monte Carlo error, so one flat percentage was standing in for two
different errors and would have let a real 2% defect pass as noise. And the band had no power at all:
at 6,000 paths it resolves only a ~20% vol error. Measured resolution 20% at 6,000 paths, 10% at
12,000, 5% at 25,000. The oracle is now banded as `4 SE + the measured freezing slack` with the SE
measured from the paths the engine consumes, split into a breadth check whose coarse resolution is
stated and a dedicated power test at the count where a 10% error provably falls outside.

### Oracle 5's specced instrument does not generalise — replaced by something stronger
Tranche 16's two-numeraire identity worked because only the *drift* differed, so the diffusion
cancelled to the last bit. `rho` enters the **factorisation**, so two models differing in `rho` give
genuinely different paths and nothing cancels; the price-level sensitivity also cannot be tightly
bounded, since differencing amplifies relative error past 100% of the move. Replaced by a direct pin:
the realised covariance of the simulated log-increments must reproduce `sigma_i sigma_j rho_ij dt` for
**every pair** — all 36 entries individually, where a price pins one weighted scalar. Worst pairwise
error 3.2% at 20,000 paths, 1.66% at 80,000, halving as `sqrt(N)` quadruples.

### Notes
- The engine's payoff is vectorised over paths. The first version walked each path in Python while
  consuming an already-vectorised simulation — the failure the `slow` gate's own docstring names.
- 604 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.19.0] - 2026-11-04

Tranche 16 — the **LIBOR market model evolved by Monte Carlo**, anchored on our own Black caplet.
First of the three-unit BGM/J arc (`JAECKEL-2002` ch 12), and the only one with a closed-form anchor.
Three pieces: the pseudo-square-root at L0, the LMM at L3, the path engine at L4 — the last extracted
now because the LMM is the **second** consumer of the MC machinery (rule-of-two).

### The drift's magnitude was unconstrained, and thirteen green oracles did not see it
Three mutations of eq 12.8 survived all 550 tests: dropping the drift's `dt` factor (an 8x drift),
feeding the drift the raw forwards instead of the `(f tau)/(1 + f tau)·sigma` ratios (5x), and dropping
that denominator (1.04x). The cause was structural. Every drift oracle asserted the drift was
**present** (the paired-numeraire test, at 297-372 standard errors) or correctly **signed** — neither
constrains its size. And the closed-form anchor runs under the measure where the target forward's drift
is **exactly zero**: the exactness that makes the anchor sharp is the same reason it is blind to the
drift. A complete set of assertions about the wrong quantity.

The fix removes Monte Carlo error instead of fighting it. Two models differing only in numeraire index,
one step, one seed consume the same normals from the same state, so the diffusion and `-c_ii/2` terms
cancel to the last bit and `log f_i^A - log f_i^B = (mu_i^A - mu_i^B)·dt` **exactly**, every path.
Agreement with a fresh transcription of eq 12.8 is 1e-13 relative, cross-path spread 7e-16 — precision
no path count buys, and what lets it see the 3.8% mutant. The step is a **half** year deliberately: at
`dt = 1.0` the `dt` scaling is invisible, which is how that mutant survived. All eight now die.

### The new oracle found an error in its own reference before it found one in the code
Its transcription used the off-diagonal `rho = 0.80` for every pair, including `rho_ii = 1`. Forward 0
agreed to 1e-13 and forwards 1-3 disagreed in the third digit — and the **asymmetry located the bug**:
forward 0 is the only index whose drift sums never reach the diagonal. The model was right. Recorded
because it is the dual of the usual worry: an independent transcription can introduce a fresh error
rather than repeat the implementation's, and the disagreement *pattern* is what says which side is wrong.

### Added
- `foundation/linalg.py` — the pseudo-square-root (12.4) with spectral ordering, and the m-factor
  rescaling (12.6/12.7) making `SUM_j a_ij^2 = c_ii` hold **exactly** at any factor count.
- `models/libor_market_model.py` — `LmmTenor`, `LiborMarketModel`, eq 12.8 split into `drift_matrix`
  (its constant part) and `state_ratios` (its state-dependent part) so the scalar `drift_at` and the
  vectorised engine share **one** definition rather than the engine re-deriving the formula.
- `engine/mc_path_engine.py` — eq 13.2's log-Euler step, vectorised over paths.

### Notes
- `black_mc` is **not** folded onto the new engine. The spec allowed it only if bit-identical; it is
  not (single step to expiry vs a stepped grid, so different draws in a different order), so it stays.
- The engine oracle file went 8.4s to 1.2s — vectorisation then caching the deterministic
  simulations — against the `slow` gate's 10s ceiling, which CI meets roughly 4x slower.
- 554 tests, all eight gates green.

## [0.18.0] - 2026-11-04

Tranche 15 — the Monte Carlo spine. **Project two's first unit**, from `JAECKEL-2002`, and the root
of the MC dependency graph: a seeded source, normal variates, an estimator with its standard error,
and a simulated price checked against our own closed form.

### A hard stop at the source, and the ruling that resolved it
The spec said to transcribe `Φ⁻¹` "from Jäckel ch 9". **Jäckel does not print it.** §9.3 *recommends*
Acklam's and Moro's interpolation formulas, §2.3 says the implementation is *"included on the
accompanying CD"* we do not have, and a grep of all 235 pages for either algorithm's characteristic
coefficients returns nothing. Both works are `referenced-by` only, so §7 forbids citing them.

Ruled (Cowork): **trigonometric Box-Muller, eq 9.6** — printed in full in the held source, **exact**
rather than an approximation, and transcribable today. `scipy.stats.norm.ppf` was rejected on the
grounds that trusted-and-pinned is the fallback for a component *no held source specifies* (PCG64's
bits), not a first choice when a held source gives an exact method. Acklam is deferred to the Sobol
unit, where a monotone transform genuinely earns its keep.

### Added
- `foundation/rng.py` — `UniformSource`, numpy `Generator(PCG64(seed))` **wrapped so no global or
  default RNG is reachable**, with the first five draws pinned as the cross-platform contract and the
  global numpy state proved untouched after 10,000 draws. PCG64's bits are *trusted, not transcribed*
  — it postdates Jäckel — which is the §4 boundary drawn explicitly.
- `foundation/variates.py` — Box-Muller eq 9.6. **The anchor oracle is the exact identity
  `x² + y² = −2 ln u`**, which carries no sampling noise; the N(0,1) moments and pair-independence are
  its companions, with the bands *measured* from the sample size. A wrong angle (`πv` for `2πv`) is
  shown to satisfy the identity but fail the pointwise transcription — §4's two rules meeting.
- `foundation/mc_estimator.py` — mean and `σ̂/√N`, with the `N−1` divisor and the `√N` each pinned by
  the two wrong variants being asserted to differ.
- `models/black_mc.py` — the forward simulated as the driftless lognormal martingale it is, averaged
  into a price **with its standard error**. 12 oracles: within four *measured* SEs of our
  `black_price` (with a companion forbidding a vacuously wide band), the **−½ convergence rate**
  fitted over four path counts, unbiasedness across forty seeds, byte-identical at a fixed seed,
  **exact** at zero variance, and the martingale `E[F_T] = F₀` as a payoff-independent check on the
  `−V²/2` term.

### Three of my own errors, one found only by mutation
My SE-scaling oracle asserted `SE(4N) = SE(N)/2`; with the `N−1` divisor the exact law is
`1/√(N−1)` and the inverse-root-N form is only asymptotic — replaced with the closed form, which
pins the divisor and the `√N` together. My asymptotic check asserted `≈2.0` at `rel=1e-3` where the
true ratio is `√(1023/255) = 2.00294`; now exact.

And **mutation found an unreachable branch**: the source originally redrew endpoints to deliver the
open interval `Φ⁻¹` needs, but an exact `0.0` has probability ~2⁻⁵³ — so the branch was unoracled, a
mutant breaking it passed all 468 tests, and had it ever been reached it would have *hung*. Deleted;
the domain guard moved to `standard_normals`, where a test can call it with `0.0`. Charter §4 gains
this as a **third survivor kind** — unreachable code, distinct from both a hole and an equivalent
mutant.

### Changed
- `TEST_FLOOR` 453 → 490.

## [0.17.0] - 2026-11-04

Tranche 14 — Market Model II's calibration (BERHAN-2012 §5.4). **This closes the book**: ch 1–5 are
built, leaving only §1.3's risk material outside. Checkpoint: `docs/checkpoints/CP_inflation_book_complete.md`.

### §5.4.1 is an identity; §5.4.2's bootstrap is structural
`σn,i` **is** the quoted Black ATM cap vol — an exact round-trip, not a residual (contrast §5.3.1,
where JY needs a Hull-White cap engine and a real two-parameter fit). And the 1-Year-Floor bootstrap
works for a statable reason: the first period's lead is zero, so `Di` vanishes and `V²` collapses to
`σ_I,1²·(T₁−T₀)` — one unknown, one price, a plain Black inversion with no solver. Both the structure
and the bootstrap's blindness to every later parameter are oracled.

### The thesis's expiry claim is almost right, and the exception is the finding
*"Since each expiry has it's own set of parameters, there is no need to restrict the number of
contracts in the expiry dimension."* Measured: **every index vol recovers to ~1e-16 except the
terminal one**, which from a low start lands at **0.002141 against a truth of 0.020000 — 89% wrong
at a residual of 8.6e-17.**

`σ_I,i` enters its period's variance *quadratically* and its drift *linearly and negatively*, so the
caplet price is **non-monotone** in it (vertex at `ρ_II·σ_I,i−1·lead/(lead+span)`, predicted 0.0115,
located by sweep at 0.0115). Two vols give one price. Every vol but the terminal one appears in
**two** periods and the second appearance picks the root; the terminal date appears once.

**And whether a fit lands on the wrong root is start-dependent** — a high start recovers the truth, a
low one falls into the conjugate, and **both report ~1e-17**. So a single-start MM-II calibration is
unsafe at the terminal point and no residual will say so. Multi-start is load-bearing, not
decoration.

### The cross-model comparison the arc was for
Tranche 13 put the three models' *pricing* gap at 7.7e-09 to 3.7e-05. The difference that matters:
**JY leaves three global parameters unidentifiable** (one at the wrong sign), **MM-II leaves one
locally ambiguous point.** So the thesis's *"this approach seems the most promising"* holds — and the
reason is identifiability, not price, which the thesis does not say.

### Added
- `engine/mmii_calibration.py` — `nominal_vol_from_black_atm` (the §5.4.1 identity, named and
  oracled because it is a claim about the model), `index_vol_from_first_floor` (the bootstrap by
  Black inversion), `calibrate_mmii_to_atm_floors`, `MMIICalibration`.
- 23 oracles, including the source's **signed** skew direction (fitted at ATM, MM-II underprices OTM
  and overprices ITM) with the companion that its ATM fit is essentially exact — two distinct axes.

### Changed — MM-II's index vols became a term structure, bit-identically
`MMIIParams.index_vols` is now keyed by the forward index's own maturity, so a period reads two
adjacent points — which is what §5.4.2's four-parameter-per-expiry set requires. A scalar would have
collapsed two of the four and made any identifiability finding an artefact of our simplification.
`MMIIParams.flat()` is the tranche 13 shape, verified **bit-identical** across the six values that
tranche measured.

### Fixed — two of my own errors
I assumed tranche 13's recursion depth transferred: it measured the `index_vol_step` refusal at two
steps on *annual* periods, and this semi-annual fixture survives **three**. And my "quote one period
further" recommendation was wrong — truncating the strip makes the ambiguity *disappear*, not move,
because reachability of the conjugate root depends on the period's own lead and its neighbour's vol.
Replaced with the start-dependence finding, which is true and more useful.

### Changed
- `TEST_FLOOR` 430 → 453.

## [0.16.0] - 2026-11-04

Tranche 13 — Market Model II (BERHAN-2012 ch 4), the **third** model, through the same
`IndexRatioModel` seam with no change to it. The thesis's model shelf is now complete.

### MM-II's identity: no real-rate volatility, anywhere
The source's words: *"contrary to Market Model I, the correction term does not depend on the
volatility of real rates."* `MMIIParams` has **no real-vol field to set** — asserted structurally on
`__dataclass_fields__`, so a later edit cannot violate it silently. The real *curve* still enters,
through the forward levels in `1 + τFr`; it is the real *volatility* that cannot reach the model.

### Two hazards in the source, each settled and pinned
**Eq 4.1.3's denominator contradicts `Di`'s, in both renderings** — 4.1.3 prints `Fn/(1 + Fn)`,
`Di(t)` prints `τᵢFn/(1 + τᵢFn)`. An internal inconsistency, so §4's derivation-step rule fired and
found a *discrepancy* rather than a confirmation. `1 + τᵢFn` is taken as correct (the standard LMM
coefficient, what MM-I's freezing uses, dimensionally sound, and the form used downstream), pinned
by an oracle.

**`Vi(t)` is defined twice for two different quantities** — 4.2.2's lead-only form, and an augmented
`sqrt(Vᵢ² + σ_I²(Tᵢ−Tᵢ₋₁))` before 4.2.4. The caplet needs the augmented one: 4.2.1 is the
*forward*-index ratio at `Tᵢ₋₁` while the payoff is the *spot* ratio at `Tᵢ`. Dropping it reuses
MM-I's `σ_I²·span` shape and loses the lead term — guarded both ways, including an exact reduction
to `σ_I²·span` at `ρ_{I,I} = 1`.

### Added
- `models/market_model_ii.py` — `MarketModelII`, `MMIIParams`, `MMIICorrelations`. 22 oracles: the
  **four-path** zero-vol reduction (MM-II, MM-I, JY, the curves); a **second exact anchor** at
  `ρ_{I,n} = 0`, where the source says the formula is exact and the freezing is unused, oracled from
  both sides; pointwise transcriptions of 4.1.6 and 4.2.2; three models required to **disagree**
  away from the limits; and `cap − floor = the YoY leg` under a third model.

### Fixed — the unit-accrual trap, fifth appearance, on a source-discrepancy oracle
The test pinning the 4.1.3-vs-`Di` denominator was powerless: on annual ACT/365F periods `τ = 1.0`
exactly, so `τFn` and `Fn` coincide and the two readings were **indistinguishable** — leaving `τ`
unoracled throughout the file. Fixed with a semi-annual fixture (`τ ≈ 0.4932`). First time this trap
hit an oracle about a *source discrepancy* rather than a formula.

### MM-I's real-vol dependence is real but weak, and the size is the finding
A contrast oracle failed at `rel=1e-6` — not because the dependence is absent but because it is
small. Sweeping `σr` 5% → 50%: **7.7e-09** at a 0.49y lead and `ρ_rI = −0.20`, rising to **3.7e-05**
at 1.48y and `ρ_rI = −0.90`. So MM-II's "no real-rate vol" is a genuine *structural* simplification
whose *pricing* impact is modest here — stated rather than hidden behind a loosened tolerance, with
the oracle now asserting the trend in `|ρ_rI|` and the lead.

### Changed
- `TEST_FLOOR` 408 → 430.

## [0.15.0] - 2026-11-04

Tranche 12(b) — Market Model I (BERHAN-2012 ch 3), the library's **second model**, flowing through
tranche 12(a)'s capability seam with no change to either pricer.

### The thesis's "approximately constant" does not hold — the freezing drives `σ_I` negative
Mercurio's freezing relation `σ_I,i−1 ≈ σ_I,i + σr·τFr/(1+τFr) − σn·τFn/(1+τFn)`, walked backwards
from `σ_I = 0.012` with `σn = 25%`, `σr = 20%` on this library's sloped test curves: **+0.012000 →
+0.005605 → −0.000542.** Two steps to a negative volatility. The adjustment is negative whenever
`σn·τFn/(1+τFn) > σr·τFr/(1+τFr)` — the ordinary case, since nominal vol and nominal forwards both
exceed real — so the source's *"we can still assume that the volatilities σ_I,i are all constant"*
does not survive a realistic parameter set. `index_vol_step` refuses a non-positive result rather
than letting it reach a `√` and surface as a `nan` layers away.

**The third honest negative of the inflation arc**, after JY's flat smile and §5.3.2's structural
underdetermination.

### Where the approximation is — not where the spec expected
Given its per-period inputs, eq 3.1.11 is an **exact** expectation; the freezing governs the
index-vol *term structure* between periods, not any single period's price. Its error bar, measured:
a 200bp forward move shifts `σ_I,i−1` by **71% of `σ_I`**, near-linear in the shock. A
**sensitivity** bound, not a distributional one — the latter needs an LMM simulation this tranche
does not build and does not claim.

### Added
- `models/market_model_i.py` — `MarketModelI`, `MMIParams`, `MMICorrelations`, `index_vol_step`.
  `MMICorrelations` carries **two** correlations, not JY's three: neither eq 3.1.11 nor §3.2 reads a
  nominal-index correlation, and sharing JY's type would force the carried-unread field §6 forbids.
- `foundation/quadrature.py` — `standard_normal_expectation`, a fixed Gauss-Hermite rule. Chosen
  over `scipy.integrate.quad` because an adaptive integrator picks its own nodes per integrand and
  can shift between scipy versions, neither of which belongs under a pinned reproducibility
  contract. **No tolerance knob**; accuracy is measured against moments known in closed form.
- 33 oracles. **The anchor is a three-path reduction**: at zero vol MM-I's integral, JY's `e^C → 1`
  and the curves directly all agree to 1e-13, sharing nothing but the curves — with a companion
  requiring the vols to move MM-I's ratio, and a further oracle requiring MM-I and JY to *disagree*
  away from the limit.

### The transcription is corroborated by a second path inside the source
`-layout` mangles eq 3.1.11 into broken columns, so §4's dual-rendering rule could not settle it.
Page 25 gives the same integral in its **pre-substitution** form, and the two agree to **2.2e-16** —
a misprint would have to appear identically in both forms to survive. Both are oracled. Stronger
than two renderings of one line, and worth reaching for whenever a source derives in steps.

### Fixed — two of my own test defects, both found by execution not review
**Mutation found a hole properties alone left open.** Dropping `/(1+τFr)` from the freezing
coefficient survived all 404 tests — not an equivalent mutant, since at `τF ≈ 0.03` the denominator
is worth ~3%. My freezing oracles asserted only properties and never transcribed the formula
pointwise. Fixed, with a companion requiring the un-divided form to differ measurably.

**And a threshold that read as reassurance.** My error-bar oracle asserted `worst < 0.01`, implying
the freezing error is small, while the measurement says it is 71% of `σ_I`. It now asserts the
magnitude and its linearity instead.

### Changed
- `TEST_FLOOR` 375 → 408.

## [0.14.0] - 2026-11-03

Tranche 12(a) — the model capability seam, so a second model can price the same products. The
inflation vertical's first structural change since the spine.

### The design question was settled by building: ONE protocol, and the source supplies its shape
A second model (Market Model I) prices the *same* products, and the registry dispatches on product
type — so a second model must not become an `isinstance`-on-model branch. The open question was
whether one protocol method could carry both models' formula or whether a two-key
`(product, model)` registry was the honest seam.

**One protocol with two methods fits both, and no formula was contorted.** Factoring BERHAN's two
year-on-year leg formulas side by side, eq 2.5.12's later-period term and eq 3.1.11's are both
`psi_i · Pn(t,Ti) · [E − 1]`, with `E` the `Ti`-forward expectation of `I(Ti)/I(Ti−1)` — JY's
`(Pn/Pn)(Pr/Pr)e^C`, MM-I's bivariate-lognormal integral. The cap/floors are both Black on
`(E, V²)`. The seam is the *source's* decomposition, and both quantities are ones BERHAN names (`m`
eq 2.6.2, `V²` eq 2.6.4). Had the protocol exposed JY's `convexity` instead, MM-I would have had to
manufacture a fake `C` from the log of its integral — the contortion that was to be avoided.

### Added
- `models/index_ratio.py` — the `IndexRatioModel` protocol: `expected_index_ratio(start, pay)` and
  `index_ratio_variance(start, pay)`, in **dates** (tranche 7's lesson). Introduced at its second
  consumer, not in anticipation — the same timing as Black in tranche 9 and the Hull-White core in
  tranche 10.
- `JarrowYildirim` implements it, delegating to its existing year-fraction methods so eq 2.5.11 keeps
  exactly one implementation.
- 9 oracles, including **the one a type annotation cannot be**: a stub model implementing only the
  two methods prices correctly through both engines, and the cap/floor's output is *exactly* Black on
  the stub's two numbers — proving the pricers read only `m` and `V²`. Six mutations on the seam, all
  caught.

### Changed — the refactor is NOT bit-identical, and that is reported not claimed
`engine/yoy.py` and `engine/capfloor.py` now depend on the protocol. The factoring that makes one
seam fit two models reorders a division, so JY's last digits move: **36119.025432954419 →
36119.025432954477, a relative 1.6e-15 (5.8e-13 bp of notional)**. Every tranche 4 and 5 oracle
passes unchanged. Keeping bit-identity would have required exposing `convexity`, which is what makes
a second model impossible — so the substitute is agreement to 1.6e-15 with the cause named.

`caplet_value` now takes the `(start, pay, accrual)` schedule period rather than a pair of year
fractions, removing a date → year-fraction → date round trip.

### Fixed — a wrong requirement of mine, caught by its own test
I asserted `m > 1` on an inflationary curve pair. False at 3y→4y (`m = 0.9914`): a negative
convexity overcomes the 2.1% carry. A requirement that ignores the convexity is not a requirement of
this model. Replaced with the decomposition — at `σr = 0` the convexity vanishes exactly and `m > 1`
*is* then forced — which also pins the sign of `C`'s contribution.

### Changed
- `TEST_FLOOR` 366 → 375.

## [0.13.0] - 2026-11-03

Tranche 11 — the full six-parameter JY calibration (BERHAN-2012 §5.3.2), and reproducing **why it
fails**. The closing rung of the JY arc, and deliberately a negative result.

### This tranche does not claim a successful fit — the thesis says there isn't one
§5.3.2 frees all six of `{κr, σr, σI, ρnr, ρrI, ρnI}` and reports failure: *"we cannot fit to the
whole surface"*, *"the presence of a strike skew makes calibration unfeasible for non-ATM
contracts"*. So "the fit matches the market" cannot be the oracle — it would assert something the
source denies. What is delivered instead: correct machinery, the documented failure reproduced, and
the failure *explained*.

### Two distinct failures, and the thesis names only one
**Underdetermination, structural.** The index correlations enter `yoy_variance` only as products
with `σI`, so `V²` is a quadratic in `σI` whose linear coefficient is a single *pooled* combination
of `(ρnI, ρrI)`; and `ρnI` is absent from `yoy_convexity` entirely. Both are asserted against the
code. On **perfect ATM synthetic data**, four starts: `(κr, σr, ρnr)` recover exactly, while `σI`
spreads 27.5%, `ρrI` 21.5%, and `ρnI` **469%** — recovered at **−0.597 against a truth of +0.15, the
wrong sign, at an rms of 8.1e-15.** A confident wrong answer on flawless data, with no skew present.

**Misspecification — the flat smile.** The implied vol is identical at every strike to 12 decimals.
Kept as a guard with its near-tautology stated, since it is the reason a market skew is unfittable —
the failure the thesis does name.

**BERHAN's diagnosis is therefore true but incomplete:** the fit is already underdetermined at ATM,
before any skew. Both failures are asserted separately so the reproduction is honest rather than a
restatement.

### Added
- `engine/jy_full_calibration.py` — `calibrate_jy_to_floor_surface`, `JYFreeParams` (the model's own
  three bundles, not six loose floats), `JYSurfaceCalibration`. Its docstring says plainly that
  `free` is *what the fit produced, not what it determined*, and which three components are one
  point on a valley. `(κn, σn)` arrive fixed from §5.3.1 and are never re-fitted.
- 18 oracles. O1 recovers the identifiable subset from every start; **O2 is the discriminator — the
  multi-start spread plus distance from truth**, behavioural and platform-independent; O3 the flat
  smile as a guard; O4 the two failures asserted as distinct, the underdetermination holding on
  ATM-only data; O5 failure is a value.

### No condition number is asserted anywhere, and that was measured not assumed
The spec's first draft proposed a Jacobian condition-number threshold, and claimed it *blows up* as
non-ATM strikes are added. Both withdrawn before building: a Gram matrix cannot lose rank by gaining
rows (measured, it **improves** — 5.2e10 → 2.2e10), and the smallest singular value sits below the
finite-difference noise floor, swinging three orders of magnitude with the step (2.5e7 → 2.1e10) and
differing again under central differences. A threshold would assert noise and flake across the two
CI operating systems. `s₅/s₁ ≈ 3.4e-6` is the only stable number and is documented, never gated.

### The `slow` gate failed on CI after passing locally, and the first fix damaged the finding
Caching the deterministic multi-start fits took the file 16.5s → 4.4s, green locally — then CI's
Ubuntu runner (~3× slower) still exceeded the 10s ceiling, which is the gate working as intended: a
time ceiling must hold on the slowest runner. Trimming the maturity strip from six points to four
then cost `κr` its identifiability (0.04036 against a truth of 0.04), silently weakening the
partition oracles 1 and 2 assert. The speed came from the number of starts instead, leaving the
strip intact: **2.44s** with 4× headroom and the finding unchanged. A speed optimisation can
degenerate a fixture exactly as a careless one can.

### Changed
- `TEST_FLOOR` 348 → 366.

## [0.12.0] - 2026-11-03

Tranche 10(b) — the `(κn, σn)` fit to ATM nominal cap volatilities (BERHAN-2012 §5.3.1), completing
tranche 10.

### Two of the thesis's six calibration parameters now have a source
`σn` and `κn` have been hand-supplied since tranche 6, because BERHAN sources them in §5.3.1 and
declines to publish the forward model. The fit's output feeds `JYFixedParams` directly, so **T6's
"two of six" becomes four of six**. Still unfitted: `κr` and the three correlations, part of a
six-parameter fit the thesis itself reports does not work.

### Added
- `engine/hw_cap_calibration.py` — `calibrate_hull_white_to_atm_caps`, least-squaring the **price**
  residual `Cap_HullWhite − Cap_Black(σ_ATM)` exactly as §5.3.1 states it. L4, because it inverts
  the cap engine. Plus `black_cap_price` and `black_cap_implied_vol` (a flat vol across the strip,
  which is what a market cap quote is), kept local since they have one consumer.
- `AtmCapVolQuote` (schedule, ATM strike, Black vol) and `HullWhiteCalibration` (params, rms,
  n_quotes, iterations) — the fit's output is a complete `HullWhiteParams`, so it is directly a
  model's input with no unpacking step that could pair a fitted `σn` with the wrong `κn`.
- 16 oracles. B1 recovers a known `(κn, σn)`; **B2 is its identifiability companion** — each
  parameter moves the strip, four distinct starts land on one point, the sensitivity Jacobian is
  well conditioned, and a **one-maturity strip is asserted to be unidentifiable** so the multi-start
  check is known to have power. B3 the residual vanishes at the truth; B4 failure is a value.

### The Black caplet's variance horizon is the EXPIRY — the mirror of tranche 9
A rate caplet fixes at `Tᵢ₋₁`, so its variance is `σ²·Tᵢ₋₁`; an inflation floorlet accumulates over
the period, `σ_I²·(Tᵢ − Tᵢ₋₁)`. The two instruments genuinely differ, which is why `black_price`'s
parameter is `variance_years` and not `expiry`. Pinned by an oracle requiring the two horizons to
give prices more than 10% apart.

### Fixed — the solver stopped 0.09% short until the residuals were scaled to basis points
At unit notional the price residuals are O(1e-3), so scipy's relative tolerances were met while
`κn` was still 0.08007 against a truth of 0.08. Scaling the solver's residuals to basis points — the
market's own quoting unit — recovers it exactly (`rms` 7.0e-7 → 9.9e-17). `rms_residual` is still
reported in price units. The pre-registered `rel=1e-6` is what exposed this; a looser tolerance
would have passed.

### Changed
- `TEST_FLOOR` 332 → 348.

## [0.11.0] - 2026-11-02

Tranche 10(a) — the Hull-White nominal cap engine. **The first non-inflation instrument**, pulled in
because BERHAN-2012 §5.3.1 sources `σn` by fitting `(κn, σn)` to ATM nominal cap vols, and `σn` has
been a hand-supplied input since tranche 6.

### New source: `MCINERNEY-ZASTAWNIAK`, `in-hand`
BERHAN performs §5.3.1 with a Hull-White cap formula it **declines to publish** ("beyond the scope
of this thesis"), and charter §7 forbids citing Brigo–Mercurio, which is only `referenced-by`. The
build stopped there rather than inventing an anchor; Bernardo acquired McInerney & Zastawniak,
*Stochastic Interest Rates*, which supplies §2.7's caplet-as-a-put-on-a-zero-coupon-bond, §3.5's
bond option (3.16–3.18), `ν(0,S)` (3.19/3.20), the caplet (3.21) — and (3.26), the calibration
scheme itself, which unblocks (b) too.

### Added
- `models/hull_white.py` gains `bond_option_variance` (eq 3.20), `zero_bond_option` (3.16/3.17),
  `hw_caplet` (3.21) and `HullWhiteModel`. The Hull-White core (`HullWhiteParams`, `hw_beta`) was
  relocated here from `models/jarrow_yildirim.py` at its second consumer — lifted **verbatim** and
  proven **bit-identical across 15 captured values**, with tranche 4's oracles green.
- `products/nominal_cap.py`, `engine/nominal_cap.py` — a first-class L2 product priced through
  `price()`. Deliberately **not** sharing a type with `InflationCapFloor` despite identical fields:
  one strikes an index ratio, the other a nominal rate — the shared *word*, not a shared shape.
- 34 oracles. The caplet is checked against an **independent longhand transcription** of 3.16–3.21;
  `σ → 0` gives the intrinsic with the companion that `σ` moves the price; and the headline
  cross-product identity **cap − floor = the forward-rate swap**, whose right-hand side is built from
  discount factors alone with no option formula in it.

### A HW bond option *is* Black, so no second transcription was written
`zero_bond_option` composes `black_price` (tranche 9's extraction), because a bond option is Black on
the forward bond price `B(0,T)/B(0,S)` with total variance `ν`, discounted by `B(0,S)`. The only
genuinely new formula is `ν`. `ν` itself is written through `hw_beta`, since eq 3.19's `D(S,T)` **is**
`β` — verified by reducing 3.19 to the printed 3.20.

### The source misprints `d₋` in both renderings, and parity cannot catch it
Both `-raw` and `-layout` print `d₋ = d₊ − ν(0,S)` while `ν` is a variance. **Put-call parity is
satisfied identically for any `d₊`/`d₋`** (`Φ(d) + Φ(−d) = 1` collapses both terms), and the `ν → 0`
limit converges either way — so both obvious discriminators are checks that cannot fail here. What
settles it is that the literal reading prices a **call at −0.0049**. That has its own oracle.

### Fixed — the unit-accrual trap left a real hole, found by mutation
Every fixture used annual ACT/365F periods where `τ = 1.0` exactly, so `1 + τK` and `1 + K` are the
same number: a mutation dropping the accrual from eq 3.21 **passed all 312 tests**. The L3 caplet
test now sweeps `τ ∈ {0.25, 0.5, 1.0, 2.0}` and the L4 fixture is semi-annual, with the non-unit
accrual asserted as a premise; the mutation now fails 23 tests. Fourth appearance of this trap.

### Changed
- `TEST_FLOOR` 224 → 332.

## [0.10.0] - 2026-11-01

Tranche 9(b) — the floor→floorlet strip and the implied-vol surface (BERHAN-2012 §5.2), completing
tranche 9. Decided by Cowork, split per its §4.

### Fixed — `black_implied_vol`'s `expiry` parameter was the wrong NAME, worth up to 50%
Shipped in v0.9.0 as `expiry`. Its second consumer showed the divisor is the horizon the *variance*
accumulates over, which for a year-on-year floorlet is the **period length**: the floorlet on
`[2y, 3y]` expires at 3y and accumulates one year of variance, since `V² = σ_I²·(Tᵢ − Tᵢ₋₁)` under
simplified JY. Passing the expiry under-states `σ_I` by **29.3% / 42.3% / 50.0%** at the 2y / 3y / 4y
nodes — and **0.0% at the 1y node**, whose period starts at the anchor, so a single-period fixture
could never have caught it. Renamed `variance_years`; the trap is pinned by its own oracle across
four periods, with the error required to grow with maturity.

*If a name needs a comment to explain it, the name is wrong.* This one needed one.

### Added
- `market/vol_surface.py` — `FloorQuote` (a quoted floor: schedule, strike, price, notional) and
  `ImpliedVolSurface`, nodes keyed by `(payment date, strike)`. **Dates, not year fractions** —
  tranche 7 lost a pointwise oracle to exactly that mismatch. **No interpolation**, and an off-node
  request raises, so the absence is a decision with a test rather than an oversight.
- `models/implied_vol_surface.py` — `implied_vol_surface(market, quotes)`: difference the strip,
  invert Black per node. Under the thesis's simplified JY (§5.2 Prop 2.3.1, JY at `σn = σr = 0`) the
  convexity vanishes, so the forward is a pure curve relationship. Quotes are grouped **by strike
  and stripped independently** — differencing across strike would mix two option series while still
  producing a plausible node count.
- 21 oracles. The anchor recovers a known **sloped** generating vol at every node, with the test
  pricing the floorlets itself. Cowork's sharpening is built in: the count is the population floor,
  the per-node values are the discriminator, and three premises are asserted — pairwise-distinct
  floorlets, a **reversed** strip giving a different middle node, and `floor(T₀) = 0`.

### Changed
- Out-of-order floor quotes are **sorted, not refused** (a feed's order is not an error); a
  **duplicate** maturity at one strike is refused, since it differences to a zero-price floorlet.
- `TEST_FLOOR` 204 → 224.

## [0.9.0] - 2026-11-01

Tranche 9(a) — Black-76 extracted to L3 with its inverse, the first half of the implied-vol surface
inversion (BERHAN-2012 §5.2). Decided by Cowork (`docs/tranches/09_implied_vol_surface.md`), split
per its §4.

### Added
- `models/black.py` — `black_price` and `black_implied_vol`. **L3, not L0**: strikes and call/put
  are finance and L0 is finance-free, with the migration tree's own `foundation/black.py` as the
  named precedent for that drift. **Parameterised by total variance, not `(σ, T)`** — JY's `V²`
  (eq 2.6.4) is an eight-term total variance and not `σ²·T`, so a `(σ, T)` signature would make
  every caller invent a `σ` the model does not have.
- `foundation/solvers.py` gains `bracketed_root` — a separate entry point from `least_squares_fit`
  because a monotone 1-D inverse carries a stronger guarantee: given a sign change Brent converges,
  so there is no non-convergence to report. The seam's second consumer, so the rule of two is now
  satisfied for the seam itself.
- `black_implied_vol` **refuses rather than guessing**: a price below intrinsic or at/above the
  no-arbitrage limit has no Black vol, and raises — `PricingFailure` is the *engine* contract (§3),
  while a constructor validates at its trust boundary (the tranche 8 ruling). Its bracket is
  structural: `V² = 0` below, doubling until the price is exceeded above, so no constant decides the
  range.
- 58 oracles. Put-call parity to 1e-15 as a **requirement** (derived from the payoff, so it survives
  a transcription error the code and the reference could share), the no-arbitrage bounds,
  monotonicity of the inverse, and the refusals. The round trip is kept but **labelled as
  telescoping**: it earns its keep only on tranche 5's independent forward oracle, which the test
  states.

### Changed
- `engine/capfloor.py` composes `black_price` instead of computing Black inline — the rule-of-two
  extraction at its second consumer, so a Black convention cannot diverge between pricing and
  implying. **Verified bit-identical**: eight strip prices captured as `repr()` before and after are
  unchanged to the last bit, with tranche 5's oracles green.
- `TEST_FLOOR` 146 → 204.

## [0.8.0] - 2026-11-01

Tranche 8 — the nominal curve bootstrap (BERHAN-2012 §5.1), the cross-check presence gate, and the
per-module dispositions. Decided by Cowork (`docs/tranches/08_nominal_bootstrap.md`).

### The thesis's §5.1 is misprinted, twice — and both renderings agree
Transcribed in `-raw` and `-layout` per charter §4; the renderings match each other and the two
equations still contradict each other. **5.1.1 carries a spurious leading minus** (every par rate
would be negative) and **5.1.2 sums `j = 0..i−1`** where inverting 5.1.1 requires `j = 1..i−1` —
with `P(0,T0) := 1` defined on the same line, so the printed form adds a spurious unit to the
annuity. Against a known flat 3% curve the implemented form recovers `P(0,T₁) = 0.97044553` exactly;
the printed form gives 0.94089107.

**A new sub-case of §4:** transcription independence protects against a *rendering* mangling a
formula. It is powerless when the **source** is wrong, because both renderings reproduce the typo
faithfully. What catches that is the **requirement that a formula and its stated inverse actually
invert** — which is why the anchor starts from a known curve. Both misprints now have their own
tests.

### Added
- `market/nominal_bootstrap.py` — `nominal_curve_from_par_swaps`, quotes → `DiscountCurve`, plus the
  shared `swap_annuity` atom (`Σ τ·df`, reading the schedule's own day count) so a future swap
  pricer composes the same annuity. **Closed form per pillar, no root-find** — corrected 5.1.2 is an
  exact rearrangement, and solving it numerically would be "solving a closed form"
  (`market/inflation_curve.py:54`). That also makes the negative-rate guarantee unconditional: no
  bracket, so nothing for a negative rate to fall outside.
- 17 oracles. The anchor is a **known sloped curve round-tripping through its own par rates,
  pointwise to 1e-12** — non-telescoping, since the test owns the forward map. The par-reprice is
  kept only as a consistency guard. No identifiability discriminator, and that is a reason not an
  omission: a sequential bootstrap is exactly determined (triangular), so there is no valley.
- `verify.py provenance` gains the **cross-check presence gate**: every module under `market/`,
  `models/` and `engine/` carries a real `crosscheck:` target or `crosscheck: none — <reason>`, the
  reason required. **Mutation-tested**: deleting tranche 7's `fixing_series.py` marker — the exact
  omission that slipped through — fails the gate, and so does a reasonless `none`.

### Changed
- Charter §8: the marker is the provenance-header `crosscheck:` form (no `#`), **gated on presence**
  rather than form. Form-only cannot catch an omission, and tranche 7 produced one in the same hour
  the form-only rule was made.
- Charter §2: a **curve bootstrap is L1**, not L3. The clause said L3 while naming tranche 2's
  break-even inversion as its example — which lives in `market/inflation_curve.py` at L1, so it
  misdescribed its own example. Model-free *parameter* calibration stays L3; engine-inverting stays
  L4.
- `TEST_FLOOR` 129 → 146.

## [0.7.0] - 2026-11-01

Tranche 7 — the index fixing series and seasoned pricing (`t > 0`), BERHAN-2012 eq 2.5.12. First
rung of finishing the book. Decided by Cowork (`docs/tranches/07_seasoned_index_series.md`).

### Fixed — the YoY swap returned a WRONG NUMBER at every seasoned valuation date
Eq 2.5.13 is 2.5.12 specialised to `t = 0`, where `I(0)/I(T₀) = 1`. Tranche 4 applied that unit
ratio **unconditionally**, so valuing a seasoned trade the engine returned the answer for "the index
has not moved since the last reset" — never reading the published reset `I(T_{ι(t)−1})`, and never
refusing either. On a 3y annual swap 18 months in: **99.0 bp of notional and 34.9% of the inflation
leg at 2%/yr realised inflation** (197.0 bp at 4%).

Every YoY oracle valued at inception, where the two forms coincide exactly — so the oracle's domain
excluded the failure (§4's governing clause, fifth instance in this vertical).

### Added
- `market/fixing_series.py` — `FixingSeries` (published levels keyed by reference month) and
  `reference_level`, which applies the index's publication lag and interpolation. **One atom, three
  engines**, so the ZCIIS, the linker and the YoY swap cannot disagree about which print `I(t)` is.
  `LINEAR` refuses when the following month is missing rather than falling back to flat — a flat
  fallback returns a plausible number under the wrong convention.
- `InflationIndex` gains `publication_lag` and `interpolation` (`FLAT` for UK gilts, `LINEAR` for
  TIPS), **required, not defaulted**: "3 months, linear" is wrong for UK gilts, so a default would
  silently price the wrong convention.
- 26 oracles. The anchor is the exact reduction — seasoned equals inception at `ι(t)=1`, the same
  division giving `1.0` — with the mandatory companion that a real elapsed period moves the price
  by the amount the ratio implies. Resolution is oracled on a **sloped series read mid-month**, and
  the two ways that fixture could lose its power (a flat series; a date on the 1st, where `LINEAR`
  degenerates to `FLAT`) are asserted rather than assumed.

### Changed — BREAKING: `schema_version` 1 → 2
A product's `"index"` went from a bare name to the nested `InflationIndex.to_dict()`, because a name
cannot say which monthly print `I(t)` is. `InflationIndex.from_dict` **refuses a bare string** rather
than defaulting the conventions. `check_schema_version` cannot catch this alone — it only rejects
payloads *newer* than the build — so the refusal comes from the reader and has its own oracle.

`MarketSnapshot.index_fixings: Mapping[index, float]` → `fixings: Mapping[index, FixingSeries]`.
The scalar is the degenerate one-point case, which is what makes the replacement provably faithful
rather than a rewrite.

A YoY swap **at inception still needs no fixing at all** — eq 2.5.13 contains no index level, as
tranche 4's oracle asserts. The seasoning guard is on the dates (`valuation_date ≤ reset_date`),
never on whether a series is present, since "no series ⇒ ratio 1" was the original defect.

### Docs
- Convention register **C9** — index resolution is one atom; the register's first **structurally**
  guarded row (one function, so a divergence is not expressible without duplicating it first).
- `OPEN.md` **OI-1 reopened**: the finding that there were zero `crosscheck:` markers was Code's own
  error — there are **six**, written in the provenance-header form while the charter specified the
  `#`-comment form. Charter §4's defect class committed by the person invoking it. The live question
  is only which spelling §8 should name.

## [0.6.0] - 2026-10-31

Tranche 6 — calibrating Jarrow–Yildirim's volatilities to a year-on-year floor strip
(BERHAN-2012 ch 5). Two stages, and only the vol stage is fitted: the curve stage stays pinned by
tranche 2, because refitting curves against vol quotes launders a curve error into a vol.

### Rejected — the stock's calibration fits volatilities to the wrong instrument
`_stock/.../jarrow_yildirim.py::jy_calibrate` fits `(σn, σr, σI)` to **zero-coupon** inflation swap
quotes, which under correct JY carry **no volatility information at all**: the fair rate is
`Pr(0,T)/Pn(0,T) − 1` (eqs 1.2.13/1.2.16) — it is *because* the ZCIIS is model-free that tranche 2
built the real curve from it. Its objective is non-flat only through two defects: model-implied
Vasicek `θ = r₀` curves where the market curves belong, and a convexity adjustment applied to the
ZCIIS when JY puts convexity on the **year-on-year** swap. Measured at a 5y quote, doubling `σI`
moves its fair rate **+10.3 bp** and doubling `σr` **+15.1 bp** — enough to drive a least-squares
fit against basis-point quotes, and entirely artefact. It converges on its own bugs and reports
success. Not purified; mined for content only (the fixed-vs-free split).

Third instance of **stock-is-not-authority** in this vertical, and the first in a *calibration*
rather than a formula.

### Added
- `engine/jy_calibration.py` — `calibrate_jy_to_yoy_floors`, fitting `(σr, σI)` to a strip of ATM
  floors through tranche 5's pricer. `JYFixedParams` bundles what the thesis sources elsewhere
  (both mean reversions, the correlations, and `σn` from *nominal* ATM caps — an engine this
  library has no reason to own, so `σn` is a named input with no default and the fit is
  conditional on it). Returns `PricingFailure`, never a NaN and never a raise.
- `foundation/solvers.py` — a thin bounded-least-squares adapter on the seam `distributions.py`
  established: scipy stays behind our own API, so models and calibrators are scipy-free and a port
  has one swap point. Its signature is exactly its one caller's need; non-convergence is a
  *reported value*, which is what lets the calibrator return a failure instead of a confident
  number.
- 16 oracles. A1 recovers the generating parameters from synthetic quotes; **A2a** proves the
  objective is not *flat* and **A2b** proves it is not *degenerate*; A3 the residual vanishes at
  the truth and descends from a wrong start; A4 the cross-stage break-even round trip; A5 failure
  is a value.

### The oracle lesson: "not flat" and "not degenerate" are different failures
A parameter-recovery round-trip is vacuous on a flat objective — the stock's failure — so it needs
a discriminator. Cowork's review pass (thread 0035) found the discriminator I wrote was itself
insufficient: per-parameter sensitivity proves the objective is not flat, but for a *two*-parameter
fit it cannot prove the optimum is **unique**, since `σr` and `σI` can each move the strip a lot
while moving it in nearly collinear directions.

Demonstrated rather than argued. On a one-quote strip (two parameters, one equation), A2a passes
comfortably — `σr` +10% moves the quote 83× its gate, `σI` +10% moves it 12× — and the fit reaches
`rms = 8.7e-11`, a *perfect* fit, at `σI = 0.0039` where the truth is `0.0120`. **Wrong by a factor
of three, reporting success**, with three starts giving three different answers. A2b adds
multi-start uniqueness and a bound on the sensitivity Jacobian's condition number: `inf` on that
degenerate strip, `6.7e2` on the real six-quote one against a `1e4` gate — so it bites where it
must and does not bind where it must not.

### Fixed — A4's own premise check could not fail, and it bit during the build
A4 asserts the rebuilt real curve equals the direct one before comparing fits, or it would compare
two calibrations of two different markets. The first version checked that **only at the quote
maturities** — exactly where the two agree by construction — and passed while the curves differed
below 2y, where `real_curve_from_breakeven` can only flat-extrapolate under its shortest quote. The
fit then missed `σI` by 0.14%, which is how the hole surfaced. The premise is now checked on a
30-day grid across the whole horizon the floors touch. The tranche's own subject, found in its own
test.

### Changed
- `TEST_FLOOR` 87 → 103.

## [0.5.0] - 2026-10-31

### Fixed — a WRONG FORMULA shipped in v0.4.0, found by this tranche's cross-product oracle
**Eq 2.5.11's bracket nesting was wrong.** The `ρ_nr` cross term belongs **inside** the
`βr(t,Tᵢ₋₁)` factor; v0.4.0 placed it beside it. Error size: **1.8% of the YoY convexity** at a
2y–3y period, 1.5% at 4y–5y, and **unbounded** at `t = Tᵢ₋₁`, where the correct value is exactly
zero and v0.4.0's was 2.6e-4.

**Why nothing in tranche 4 caught it, which is the part worth keeping:**
- **The sweep oracle compared two copies of the same mistake.** T4's "independent transcription"
  came from the same `pdftotext -layout` output as the implementation — and that rendering *merges
  2.5.11's parentheses*. **Two transcriptions of one corrupted source are not independent**, so a
  108-point sweep passed with both sides wrong identically. The telescoping failure mode in a new
  disguise: we guarded against reading a formula from the *stock*, then read it twice from one
  mangled rendering of the *source*.
- **The `σr → 0` anchor could not see it** — `σr` is an overall factor under either nesting.
- **What found it:** this tranche's `cap − floor = T4's YoY leg` oracle, disagreeing by 0.65%. The
  cross-product check doing exactly its job: different code on each side, so a shared transcription
  error has nowhere to hide.
- **What identified the cause:** a **structural requirement**, not a number. `C` corrects
  `E^{Tᵢ₋₁}[Pr(Tᵢ₋₁,Tᵢ)|Ft]`, so at `t = Tᵢ₋₁` the quantity is already known and `C` must be
  **exactly zero** — a property of the object, true for every parameter set. The wrong nesting
  violates it; a test for it now exists and should have existed in T4.

**Two practices follow:** re-read a load-bearing formula in the **`-raw`** extraction as well as
`-layout` (layout reflows fractions and merges parentheses; raw preserves token order); and **ask
what the object must satisfy regardless of parameters** — `C(t=Tᵢ₋₁)=0`, `V² ≥ 0`, parity — because
those outlive a transcription error.

### Added — tranche 5: the inflation cap/floor under Jarrow-Yildirim
Eq 2.6.5 per period, summed. Source `BERHAN-2012` §1.2.4 + ch 2.6, **read first-hand**. Proposal,
review and results: `docs/tranches/05_inflation_capfloor.md`.

- **L3** `yoy_variance` — eq 2.6.4's **eight terms**; `JYCorrelations.nominal_index` (`ρ_nI`) lands
  as **its first consumer**, since it enters `V²`'s eighth term and nothing else in the model.
  Required rather than defaulted: a default would be a hidden knob on a real model parameter.
- **L0** `CapFloorKind` — an enum, not a bool: `InflationCapFloor(..., True)` is unreadable and
  silently reversible. · `foundation/distributions.py`, a thin scipy adapter whose oracles test
  **the adapter** (`Φ(x)+Φ(−x)=1`, which the parity oracle leans on), never scipy's numerics.
- **L2** `InflationCapFloor(index, schedule, strike, notional, kind)` — **a strike, not a leg.**
  Ruled: a leg accrues and discounts, a strike enters `d₁/d₂`; same two fields, different
  *operation*, so reusing `YoYLeg` would have put a strike in `fixed_rate` and made its "accruing"
  docstring false. The real shared shape is **`CouponSchedule`**, which gets its honest second
  consumer.
- **L4** `price_inflation_capfloor` + `caplet_value`.

### Oracles — 87 green
- **Primary:** eq 2.6.5 against an independent transcription, swept over four strikes × both kinds;
  and eq 2.6.4 swept over 4 × 3 × 3 × 4 parameter combinations. Primary *because* the reduction
  below kills six of `V²`'s eight terms.
- **Reduction + companion:** `σn = σr = 0` ⇒ `C = 0`, `V² = σI²Δ` ⇒ plain Black on the
  deterministic forward ratio, exactly — paired with a test that the rate vols **move** the price,
  without which the reduction passes an implementation ignoring seven terms.
- **Parity:** `cap − floor = N·Σψ·Pn(Tᵢ)·(m − K)`, and at `K = 1` it **is** tranche 4's YoY leg —
  the oracle that found the bug above.
- **Structural:** `C(t=Tᵢ₋₁) = 0` exactly; `V² ≥ 0` across adverse correlations; `C` invariant to
  `ρ_nI`; deep-ITM → forward annuity, deep-OTM → 0; monotone in strike; no-vol → intrinsic.
- `TEST_FLOOR` 64 → 87.

## [0.4.0] - 2026-10-31

### Added — tranche 4: the year-on-year inflation swap and Jarrow-Yildirim
**The first model-dependent instrument.** Eq 2.5.13 with the JY convexity 2.5.11, read in the held
PDF. Source `BERHAN-2012` ch 2. Proposal, correction and results:
`docs/tranches/04_yoy_jarrow_yildirim.md`.

- **L3** `hw_beta` (eqs 2.4.3/2.4.16, shared by both factors so nominal and real cannot drift) ·
  `HullWhiteParams` · `JYCorrelations` · `JYParams` · `yoy_convexity` (2.5.11) ·
  `JarrowYildirim(market, params)`.
- **L0** `YoYLeg` — **separate from `RealLeg`**: rule-of-two fired at the second leg type and the
  answer was *no*. A real leg pays a rate on an uplifted notional, a YoY leg an index *ratio*; one
  name over both would invite a kind-of-leg branch, which the no-`isinstance` law forbids.
  Rule-of-two triggers an abstraction when consumers share a *shape*, not a *word*.
- **L2** `YearOnYearInflationSwap` — three fields and **no base index**: a YoY floating side is a
  chain of period-on-period ratios, so it has no single `I0`.
- **L4** `price_yoy_inflation_swap`, `yoy_inflation_leg_value`, `yoy_fixed_leg_value`,
  `yoy_par_rate`. **Nothing new at L1.**

### Fixed — the handoff's anchor oracle was wrong, found by reading the source
The oracle plan specified `ρ → 0` as T4's anchor, **taken from the stock's
`yoy_convexity_adjustment` rather than from the thesis.** That function computes
`σI·σn·ρ·tenor·year` — a first-order approximation in which `ρ` *is* a factor — and it **takes a
`real_vol` argument it never uses** (the carried-but-unread smell, fourth instance, every one of
them in stock code we are declining to purify).

In JY, **`σr` is eq 2.5.11's overall factor and `ρ` is not**; the thesis says so in the line after
it. Measured: all-parameters `C = −1.4738e-02`; **at ρ=0, −1.4486e-02** (barely moves); **at σr=0,
exactly 0**. So the anchor is **`σr → 0`**, which is also *better* — a structural factor gives an
exact reduction rather than a numerical limit.

**Had the ρ→0 anchor been built it would have failed against the correct formula** — and the danger
is not the red but that a red anchor invites bending the formula until it passes. Cowork owned the
error and recorded the lesson: **oracle anchors are read from the held source's formulas, never
inferred from the stock's code — the stock is content and regression, the source is authority.**

### Oracles — 64 green
- **The anchor:** `σr → 0` gives the deterministic-real-rate limit exactly, **paired with a test
  that the convexity MOVES the leg** — without which the anchor would pass on an implementation
  that ignored `C` entirely. A reduction proves the general case *collapses* correctly and says
  nothing about whether it does anything at all.
- **`ρ = 0` does NOT vanish**, asserted against the surviving `−½βr²` term, so nobody simplifies
  2.5.11 back toward the approximation.
- 2.5.11 against an independent transcription over a **108-point** parameter sweep · `β`'s closed
  form **and its `κ → 0` limit** (the limit that catches a misplaced `κ`) · eq 2.5.13 transcribed
  independently · **the first period equals tranche 2's ZCIIS leg** (the thesis's own structure, and
  different code on each side) · par-rate reprice (a telescoping consistency guard, kept as such) ·
  matured → PV 0 · **prices with no index fixing**, since 2.5.13 contains no index level.

### Changed — three gate catches, all fixed at the cause
- **`PLR0913`**: `yoy_convexity` took 7 arguments → bundled into `JYParams`, the model's own
  structure rather than a split to satisfy a count.
- **`layers`**: the word *"payoff"* in a docstring tripped the L0 finance-free gate, which greps
  prose as well as code. **Reworded rather than loosening the gate** — a gate that over-reports on
  comments is still doing its job, and relaxing it so my own docstring passes is the wrong
  direction.
- **`debt`**: four `# noqa: E731` on test lambdas → real local functions. There was no
  justification to give, which is the gate's point.
- `TEST_FLOOR` 50 → 64.

## [0.3.0] - 2026-10-31

### Added — tranche 3: the inflation-linked bond (linker), model-free
`ILB(t,TM) = I(t)/I0 · N[ Σ C·ψᵢ·Pr(t,Tᵢ) + Pr(t,TM) ]` — BERHAN 1.2.5 generalised with the accrual
fraction a real linker pays. **No nominal curve is touched**: a linker is a real bond scaled by the
index ratio, the foreign-currency analogy at its clearest. Source `BERHAN-2012` §1.2.1. Proposal,
review and results: `docs/tranches/03_inflation_linked_bond.md`.

- **L0** `CouponSchedule(dates, day_count)` — period **boundaries**, so an accrual has both ends
  rather than inferring the first start · `RealLeg(schedule, coupon_rate)`. Both are §3b-compelled
  bundling (seven loose fields, two over the limit) and both stay **concrete frozen bundles** — no
  base class, no polymorphism; whether `RealLeg` generalises is decided at the *second* leg type.
- **L2** `InflationLinkedBond(index, base_index, leg, notional, indexation_lag)`.
- **L4** `price_inflation_linked_bond`.
- **Nothing new at L1 or L3** — a second inflation product cost no market-data change at all, which
  is what keys-not-fields buys.

### Oracles — 50 green, first run
- **The closed form** with every `ψ` and `Pr` computed in the test, swept over nominal rates that
  must **not** move the price (1.2.5 has no `Pn`; a leak shows immediately).
- **The exact reduction:** every period is 365 days so each `ψᵢ = 1`, and the general form *is* the
  paper's annual 1.2.5. The premise itself is asserted alongside it.
- **Cross-product reconciliation (the first of its kind here):** the linker equals
  `Σ C·ψᵢ·ZCIB(Tᵢ) + ZCIB(TM)` with each `ZCIB` rebuilt from **tranche 2's** `inflation_leg_value`
  — different code on each side, so a divergent index-ratio application in one product but not the
  other breaks it while each product's own oracle still passes.
- Zero-coupon limit → eq 1.2.4 scaled · `Pr = Pn`, `I = I0` prices as a **nominal** bond (catches
  the ratio on the wrong leg, or `Pr`/`Pn` transposed) · doubling the index doubles the **whole**
  price · past coupons excluded, checked by equality against the same bond with those dates removed
  · matured → PV 0 · missing curve/fixing → failure · the round-trip **exercises the lag**.

### Changed — the `day_count` question, answered a third way
Cowork spotted that pricing BERHAN's literal annual 1.2.5 would leave `CouponSchedule.day_count`
**carried but unread**, and offered two resolutions (read it and register the convention, or drop to
a dates-only schedule). **Took a third: price the general `C·ψᵢ` form and make 1.2.5 its exact
reduction.** Better on three counts — the field is read so nothing is smuggled, the instrument is
the real one rather than the paper's simplification, and it buys a reduction oracle neither option
offered.
- **Register gains C8** — a schedule's accrual day count and its curve's `TimeMeasure` are one
  convention. A **C7 sibling**: OWED, currently unrepresentable (one day count exists), trigger
  shared. Register now reads five of eight not cross-module guarded.
- **The indexation lag is carried and round-tripped, on Cowork's sharper test.** I had argued
  "contract data vs market metadata" and asked to be ruled rather than decide in my own favour. Its
  test is better: **constitutive of the instrument's identity, not "does the pricer read it."** Two
  linkers on one index with different lags are *different contracts*, so the round-trip **is** its
  present consumer — exactly what T2's publication-lag field lacked. Oracle asserts both that the
  lag survives serialisation and that `lag=3 ≠ lag=8` as contracts.
- `TEST_FLOOR` 37 → 50.

## [0.2.0] - 2026-10-31

### Added — tranche 2: the inflation spine (ZCIIS, model-free)
The zero-coupon inflation-indexed swap, priced with **no dynamics** — pure no-arbitrage from the
foreign-currency analogy (real ↔ foreign, nominal ↔ domestic, index ↔ FX spot). Source:
**`BERHAN-2012`** §1.2.2. Proposal, review and results: `docs/tranches/02_inflation_spine.md`.

- **L0** `InflationIndex` — the asset class's *entire* L0 footprint, a bare identity ·
  **`Term(start, maturity, day_count)`**, which exists because the contract needs six loose fields
  and the rule's answer is to bundle the cohesive ones, never to raise the limit. It also puts the
  compounding exponent in one place: `TM` is measured from **inception**, since the fixed payout
  was fixed at trade time and does not shrink as the trade seasons.
- **L1** `CurveRole.REAL` — **a new curve KEY, not a new snapshot field**: a second discount curve
  costs one enum member while `MarketSnapshot`, `CurveSet` and every existing pricer stay as they
  were · `MarketSnapshot.index_fixings` · **`real_curve_from_breakeven`**, eq 1.2.16.
- **L2** `ZeroCouponInflationSwap`, serialised and version-enforced.
- **L4** `price_zc_inflation_swap` + `inflation_leg_value`, composing `df` from **both** curves.

### The design call: there is exactly ONE `Pr`
`Pr` is a curve in the snapshot; eq 1.2.16 is **how it is built** from break-even quotes, not a
second way to price. The pricer never recomputes it. Two oracles guard that, with different jobs:
the **par identity** proves construction and pricing share one `Pr`, and a **pointwise** 1.2.16
check proves the constructor is right on its own.

### Oracles — 37 green
- **1.2.14** off curves the test specifies, to 1e-9 · **the exact reduction**: 1.2.13 with
  `I(t) = I0` *is* 1.2.14, so there is one formula and not two · the index enters as a **ratio**
  (double it and the real-bond term doubles while `−Pn` does not — a difference would shift by a
  constant) · the net swap equals leg **minus** fixed leg, so the net PV cannot become its own
  formula · zero-inflation degenerate case · sign (the receiver gains on higher break-even, loses
  on a higher strike) · matured → PV 0 · missing real curve / missing fixing → failure · round-trip
  with the version enforced.

### Fixed — a must-fix found a second bug, in the API rather than the test
- **`real_curve_from_breakeven` now takes quote DATES, not year fractions.** Cowork's must-fix
  (the par oracle telescopes, so the constructor needs its own pointwise oracle) made that oracle
  **fail at the 2.5-year pillar by 6.6e-6**: a caller saying "2.5 years" converts to 912 whole
  days — 2.4986 years on the curve's own measure — so the caller's `T` and the curve's `T`
  disagreed by a day. Fixed structurally: one `date → t` map, the curve's, and a caller cannot
  hold a second. **The par oracle could never have seen this**, because a one-day disagreement
  shared by construction and pricing telescopes away — which is precisely the must-fix's argument,
  demonstrated.
- **A matured swap returns PV 0, not `PricingFailure`** (must-fix 2). Conflating *historical* with
  *unpriceable* would make a settled trade look broken; checked before any market lookup, so a
  matured trade needs no curves.

### Changed
- **No lag field on `InflationIndex`, and the stock is why.** Its `inflation_indices.py` carries one
  `publication_lag_months`: 3 for most indices, **8 for UK index-linked gilts**. Those are two
  quantities conflated — UK RPI publishes ~1 month after its reference month, and the 8 months is
  the **indexation lag the gilt's contract specifies**, which two linkers on the same index can set
  differently. So the indexation lag is L2 contract data (arriving with the linker in T3), and even
  the publication lag is omitted here because **nothing reads it** — the snapshot carries the
  already-resolved `I(t)`. Ratified by Cowork, which confirmed the 8-month figure against the stock.
- **Conventions register gains C7** (Cowork's review condition): the break-even construction, the
  nominal curve's `TimeMeasure` and the contract's `Term.day_count` share **one** day count, and the
  par identity holds only under it. Recorded **OWED**, because *no test could fail on it* — the par
  oracle telescopes when the conventions match, which they always do today. Sharpened while
  recording it: with one day count in the library the convention is not merely unguarded but
  **unviolable**, so a guard now would be an imagined problem. Trigger: the first second day count.
- **`# crosscheck: quantlib …` markers** at both sites wanting external validation (thread 0014),
  so the later optional QuantLib tier has a floored worklist instead of a from-memory sweep.
- `TEST_FLOOR` 25 → 37.

## [0.1.1] - 2026-10-31

### Fixed — a defect in v0.1.0, found by Cowork's independent review
- **The schema version was written and never read.** Both `to_dict`s stamped `schema_version` and
  **nothing dereferenced it**, so a payload from a future build would have loaded silently under
  v0.1.0 code — defeating the exact guarantee the field's own docstring claims ("a format with no
  version cannot change without silently breaking a reader"). It was therefore not merely an unused
  field but a **false claim in a comment**, which is the worse of the two. The project's named
  defect shape, **resolve-don't-just-exist**, and v0.1.0's test asserted only *presence*, which
  could never have caught it.
  Fixed with **one** shared `check_schema_version(payload)` that both `from_dict`s call — one
  definition of what is readable, so the two products cannot drift — returning the validated
  version so a future migration branches on it rather than re-reading the raw field.
- **Six oracles that assert REFUSAL, not presence:** a future version, a missing one, six
  malformed ones (`"1"`, `0`, `-3`, `1.5`, `True`, `None`), the nested payload, and **the
  ordering** — the version is checked *before* `kind`, because in a format we cannot read the
  `kind` field's meaning is itself unknown, so reporting "not a fixed_cashflow" would be a guess
  about a payload we have already failed to understand.

### Changed
- **`numerics` threading dropped from `dispatch`.** Purified from the stock as-is, it inspected
  each pricer's signature to decide whether to pass an optional numerical config — and **ran in no
  test**, since the only pricer is closed-form. Three reasons agree: the config's own rule is that
  it *lands with the first engine that reads it*; there is no consumer; and an unexercised dispatch
  branch at the baseline is the same dead-code shape the debt gate caught in `zero_rate`. It
  returns with the first pricer that integrates or samples — with a consumer and a test.
- `TEST_FLOOR` 18 → 25.

### Process note
v0.1.0 merged before this review round landed, so the fix ships as a patch rather than as part of
the tranche. The sequence Cowork set is **propose → review → land**; here it ran propose → build →
land → review → patch. No harm done (the defect was caught and is fixed in the next commit on
`main`), but the review's value is highest *before* a merge, which is where the beat belongs.

## [0.1.0] - 2026-10-31

### Added — tranche 1: the minimal spine
A single fixed cashflow priced end to end through the stateless
`price(product, model) → PricingResult`. The backbone every product needs and none can avoid, so
it prejudges no product while proving the engine contract. Proposal and results:
`docs/tranches/01_minimal_spine.md`.

- **L0** `Money`/`Currency` (currency mixing is an error, never a silent addition) ·
  `DayCountConvention` + `year_fraction` (ACT/365F only) · `TimeMeasure` (the one `date → t` map) ·
  `Cashflow` · `PricingResult`/`PricingFailure`.
- **L1** `DiscountCurve` + the `CurveHandle` protocol the engine depends on · `CurveSet`/`CurveKey`
  (curves keyed, not positional) · `MarketSnapshot`, which **carries the valuation date** — there
  is no clock read anywhere in the library.
- **L2** `FixedCashflow` — pure data, with `to_dict`/`from_dict` from the first product.
- **L3** `DiscountingModel`, carrying the snapshot it was built on.
- **L4** the engine registry (structural dispatch, no `isinstance` ladder) + `price_fixed_cashflow`.

### Oracles — 18 green
- **`PV = N·e^{−r·t}` exact to 1e-12** across 48 points. The reference is arithmetic written in the
  test, independent of every path under it — which is the point: the predecessor shipped a 5.14%
  error whose oracle restated the builder's own composition and so validated it against itself.
- **Analytic vs finite-difference DV01** to 1e-6 — two different computations, so agreement is not
  tautological.
- The **valuation-date boundary** in both directions: a flow *on* the date is excluded, one *day
  after* is included and positive. Without the second, "excluded" could be implemented as "always
  zero" and the first would still pass.
- Statelessness (byte-identical reprice, snapshot untouched); failure-is-a-value on three distinct
  failures; currency mixing refused; the representation round-trips and carries a schema version.
- **Interpolation tested on a SLOPED curve.** A flat curve returns one rate everywhere and cannot
  tell linear interpolation from any other scheme — the fixture trap the predecessor hit twice.

### Stock
11 of 13 modules **purified** from `_stock/`, 2 built fresh, nothing over 100 lines. Purification
was mostly *trimming*: `money.py` 289 → 90 lines (the open-currency *insight* kept, the registry
machinery left behind until something needs named constants); `day_count.py` 332 → 45 (one
convention); `snapshot.py` 154 → 30 (curves only — its other shapes grew one asset class at a
time and return when a product pulls them).

### Changed
- **`CalibratedModel` not carried.** A protocol declares a capability, and a discounting model has
  none to declare beyond carrying its market, so carrying it would be an empty abstraction. The
  registry's model parameter is `object` until a pricer needs a real capability.
- **`DiscountCurve.zero_rate` uses `bisect` and carries no suppression.** The first draft ended in
  an unreachable `raise` needing `# pragma: no cover`, which the debt gate caught. The answer to
  dead code is to remove the deadness, not to ledger the suppression — and `bisect` is O(log n),
  which will matter at a bootstrapped curve.
- `TEST_FLOOR` 1 → 18.

## [0.0.0] - 2026-10-30

### Added
- **The clean project scaffold.** An empty, CI-green library at `v0.0.0` — no pricing code.
  Package `pricebook`, MIT, public. `pyproject.toml`, one trivial test, the ubuntu+windows CI
  matrix on Python 3.12, and `verify.py` carried from the migration tree and **trimmed to the
  gates that still apply**.
- **The stock.** The whole previous tree — the built core, the un-migrated `quarry/`, its `docs/`
  and all 77 oracle test files — shelved in `_stock/` and **gitignored**: reference material,
  mined by hand for *content*, never for *structure*, never imported. Its pushed backup is the
  archived `pricebook-migration` repo at tag `migration-archive`.
- **`docs/handoffs/phase_R_record.md`** — what Phase R did, the three parameters Bernardo ruled
  before execution, and the four `verify.py` calls taken under delegated mechanics.
- **`docs/conventions_register.md`** carried as a tracked living doc, because it describes the
  built core rather than the migration — including its one **OWED** item, the vol clock, which
  the rebuild re-lays *with* its cross-module oracle.

### Changed
- **The migration frame is retired** (`docs/handoffs/rulings_reset_to_fresh_repo.md`). The unit of
  work is now a product you want, pulling in the core it needs; `done` is no longer "the quarry is
  empty". The architecture, the oracle discipline and the conventions are kept; the obligation to
  cross every old module is dropped.
- **`verify.py` trimmed:** the `quarry` import-floor gate and the drawdown reconciliation are
  **deleted** — both measured migration progress, and there is no migration to measure. Kept:
  `acyclic`, `layers`, `fields`, `debt`, `provenance`, `version`, `slow`, `tests`.
- **Provenance header: `quarry:` → `stock:`**, same role against the new reference tree. Because
  `_stock/` is gitignored and therefore absent in CI, the resolve-check **reports** that it could
  not dereference paths rather than skipping silently — a gate that quietly passes is this
  project's named defect class. `source:` and `oracle:` are verified in both environments.
- **License corrected to MIT** in `pyproject.toml`, matching the `LICENSE` file (the old tree's
  metadata said Proprietary).
