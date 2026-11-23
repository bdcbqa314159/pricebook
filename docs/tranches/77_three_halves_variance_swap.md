# T77 — the 3/2-model variance swap (a new non-affine stochastic-vol model)

**Cowork spec, 2026-11-23.** Fourth tranche of the vol-derivatives arc (project ten, `KWOK-ZHENG`;
`docs/vol_derivatives_arc.md`). The arc's **second stochastic-vol model** after Heston — the **3/2 model**,
non-affine, under which the *variance swap* has a closed form. Deliberately scoped to the **variance swap
only** (the first moment / fair strike): the full 3/2 transform (options on variance) is the confluent-
hypergeometric object and is deferred — this tranche is the model plus its one closed form.

## Target
Price the **fair variance under the 3/2 model** — `price(VarianceSwap, ThreeHalvesModel) → PricingResult`.
**No new product:** `VarianceSwap` (T31) already exists; this registers it against a *new model*, exactly as
the same product can price model-free (`VarianceReplicationModel`) or model-based. The deliverable is the
new L3 model and the L4 engine that reads its closed form.

## The route — TWO independent routes (RULED 0409, superseding 0408 after the source paper was acquired)
**Step-1 (0407) found the book's transform eq (3.30) fails `L(0)=1`; the source paper now resolves it.**
Eq (3.30) attributes to Zhao et al. (2018), which Bernardo acquired (`ZHAO-CUI-FLORESCU-2018`, held). Its
**Proposition 3 / eq (19)** — the canonical Carr–Sun (2007) form — writes the confluent hypergeometric as
`M(α, γ, −2/(σ²y_T))` (**argument last**) and **satisfies `L(0)=1` to 1e-10** (verified); the book put the
argument first, which diverges. So eq (3.30) is **repaired, not quarantined**, and T77 has **two independent
routes to the same first moment** — the strongest possible anchor (the T75 oracle-1 cross-METHOD shape):

- **Production (route c) — the reciprocal-CIR Poisson-ladder, scipy-free.** Price `∫₀^T E[1/Y_t]dt` on the
  held reciprocal-CIR map (below), reusing `foundation/noncentral_chisquare.py`'s existing ladder — **no new
  L0 primitive, no scipy on the production path.**
- **Oracle — transcribe eq (19) from the held paper**, differentiated at `λ=0` for the first moment
  (`−∂L/∂λ|₀`), using **scipy's `hyp1f1` in the oracle only** (the T20a `distributions.py`-off-scipy
  arrangement — scipy is allowed in tests, never in `src/`). The two routes must agree; disagreement is a
  finding. `L(0)=1` on the transcribed form is a free requirement oracle the book's rendering failed.

**The production route derives the fair variance from the held, printed reciprocal-CIR map (eq 4.35):**

The 3/2 variance process (the source's **eq (2.75d)** — vol-of-vol named **`epsilon`** in code; `sigma`/`σ`
collides with Heston's and the CEV exponent, the T71 β-collision lesson):

    dv = κ·v·(θ − v) dt + epsilon·v^{3/2} dW

Under `Y = 1/V`, Itô gives a **time-homogeneous CIR** process `dY = [(κ+ε²) − κθ·Y]dt − ε√Y dW`
(`κ_Y = κθ`, `θ_Y = (κ+ε²)/(κθ)` — **verify the algebra independently** before relying on it; it is the
standard Lewis/Carr–Sun result and eq (4.35) is its printed, held form). Hence `E[v_t] = E[1/Y_t]` is the
**inverse moment of a non-central χ²**, and the fair variance is `E[∫₀^T v_t dt] = ∫₀^T E[1/Y_t] dt`,
**asserted as a theorem (T19)** — it is derived, not transcribed, so a source misprint cannot reach it, and
`L(0)=1` holds **by construction** (the requirement eq (3.30) failed).

**The special-function / layer question — resolved by the two-route split.** The production inverse moment
is the **Poisson-ladder mixture** `E[1/Y] = Σ_j Poisson(j; λ/2)·1/(c·(d/2+j−1))` — `₁F₁` is only the
closed-form *name* for that series — so it **reuses `foundation/noncentral_chisquare.py`'s existing
Poisson-ladder**: no new L0 primitive, no scipy in `src/`. The `₁F₁` lives only in the **oracle** (eq (19)
transcription via scipy's `hyp1f1`, T20a). Confirm the ladder closes at build; only if it genuinely does not
does the new-L0-primitive sub-`Ask:` return.

## Pieces, layer, reuse
- **The model** (new, L3) — a `ThreeHalvesModel` frozen dataclass carrying the 3/2 parameters
  (`v0, kappa, theta, epsilon`, and the `MarketSnapshot` it is calibrated to per §3) — ≤5 fields (§7);
  bundle into a value object if over. Sibling of `HestonParams`/`CevModel`, **not** a subclass (§6 — a
  different SDE, a different accrual).
- **The engine** (new, L4) — `@register(VarianceSwap, ThreeHalvesModel)`, computing `∫₀^T E[1/Y_t] dt` via
  the reciprocal-CIR inverse moment (route c), projecting it into a `PricingResult` (fair strike × notional ×
  discount, the §3 engine-result shape). Reuse: the **registry/`PricingResult` plumbing**, the product, and
  — if the ladder form closes — `foundation/noncentral_chisquare.py`'s Poisson-ladder. Reuses nothing of
  `variance_replication` (that is the model-free strip; this is a model quantity).
- **The MC companion** (new or reuse) — the 3/2 SDE needs its **own** Euler/Milstein simulation of
  `v_t` (`heston_mc` simulates the *spot* under Heston's affine variance, a different process), so either a
  small `three_halves_mc` or an extension of `mc_path_engine`. The realized `∫v dt` from simulated variance
  paths against the closed form, banded per §4 (`k·SE + measured discretisation error`, resolution stated) —
  and watch the **3/2 process's fatter tails / explosion risk** (`v^{3/2}` diffusion): a naive Euler scheme
  can blow up, so the companion's own accuracy must be established before it judges anything (the "a
  reference is code too" rule), e.g. a fine-step / full-truncation or log-Euler scheme, measured.

## Oracles (RED-first, §4) — reconstructed-formula anchoring (route c; MC is NOT the sole anchor → in-method)
1. **The cross-METHOD anchor — the two routes must agree.** The production reciprocal-CIR Poisson-ladder
   value `== −∂L/∂λ|₀` of the **transcribed eq (19)** (oracle, scipy `hyp1f1`), two genuinely different
   derivations of one number (reciprocal-CIR inverse moment vs the Carr–Sun transform differentiated),
   different code each side — the T75 oracle-1 shape. Transcribe eq (19) pointwise in `-raw`+`-layout` (the
   argument is **last**: `M(α,γ,−2/(σ²y_T))`, the book's defect was argument-first); assert **`L(0)=1`** on
   the transcribed transform (the free requirement the book's rendering failed, now a green not a red).
   Verify the `Y=1/V` Itô algebra and the CIR coefficients `(κ_Y, θ_Y)` independently (eq 4.35 is the held
   printed cross-check for the production side).
2. **The exact reduction: `epsilon → 0` ⇒ the deterministic logistic ODE** `dv/dt = κv(θ−v)`, whose solution
   is closed-form `v(t) = θ·v₀ / (v₀ + (θ−v₀)e^{−κθ t})` and whose integral `∫₀^T v dt =
   (1/κ)·ln[(v₀ + (θ−v₀)e^{−κθT}) / θ] + θT` (**the test derives and asserts this independently** — a
   genuine closed form, not a numerical limit). **Companion required (§4/T39):** a **genuine `epsilon > 0`**
   fixture checked against oracle 1 — the `epsilon → 0` reduction is *structurally blind* to the vol-of-vol
   term (it zeroes exactly the parameter the stochastic part lives in, the T39 trap), so the companion must
   exercise a material `epsilon`, checked by convergence **rate** in `epsilon` if the closed form degrades
   near zero.
3. **The `α→0`-type numerics watch (T66/T67, rate-in-a-regime measured PER model).** The logistic integral
   carries a `(1/κ)·ln[…]` and the fair variance likely a `(1−e^{−κθT})/(κθ)`-type factor — a **removable
   singularity** at `κθ → 0` (`0/0`). If so, the closed form will diverge from its own limit *before*
   reaching it (catastrophic cancellation), so assert an **O(κθ) convergence rate over the regime where the
   form is numerically sound**, with a companion showing it **degrades** outside — and **measure the sound
   boundary for THIS model** (T67: the boundary depends on how many times the form divides by the vanishing
   parameter; do not inherit T66's or Vasicek's). The degenerate case (`κθ=0`) is its own closed form. If no
   removable singularity appears, say so and skip — do not manufacture the watch.
4. **The central-χ² `ε→0` limit of `Y` as an inverse-moment cross-check (IF cheap).** As `ε→0` the CIR `Y`
   loses its noncentrality structure toward the deterministic reciprocal, so the inverse-moment ladder should
   agree with the logistic reduction's own value by a second route — a cheap internal consistency check on
   the ladder. Not load-bearing (1–3 already pin it); skip if it does not fall out.
5. **Requirement oracles (structural, survive a transcription error — F-6.1):** the fair variance `> 0`;
   **monotone increasing in `v₀`** (more initial variance ⇒ more expected variance); and the long-horizon
   limit `(1/T)·E[∫v] → the 3/2 model's STATIONARY mean` as `T → ∞`. **Corrected (0411): that stationary
   mean is `θ/(1 + ε²/(2κ))`, NOT `θ` — the `v^{3/2}` vol-of-vol pulls the long-run variance *below* the
   reversion level** (θ is only the `ε→0` case; verified to machine precision, a 26% gap at ε=1.2). My
   original "→ θ" was wrong: the obligation-not-form lesson applied to a *requirement oracle's value* — the
   stationary mean of a non-affine process is not its reversion parameter, and a requirement oracle's target
   must be derived, not read off the naive parameter. Assert the corrected limit (Code's `stationary_variance`
   = `θ/(1+ε²/(2κ))`), with `θ` asserted as the `ε→0` reduction of it.

## Degeneracy premises (§4 / §9)
- The `epsilon → 0` companion is **non-degenerate in `epsilon`** (a material vol-of-vol, T39) — state it;
  `epsilon = 0` is the reduction in disguise and blind to the whole stochastic term.
- The monotonicity/limit fixtures use **`v₀ ≠ θ`** (mean-reverting *away* from the long-run level) — at
  `v₀ = θ` the process starts at its mean and several terms collapse, hiding a mean-reversion error (the
  τ=1 shape: `v₀ = θ` is the degenerate value).
- The MC companion's **own accuracy is established first** (the 3/2's `v^{3/2}` explosion risk — a crude
  Euler scheme is the weaker side; measure the scheme against the closed form at a fine step before using it
  to judge the closed form).

## Provenance / gates
- Header `source: KWOK-ZHENG §2 eq (2.75d) [SDE] + §4 eq (4.35) [reciprocal-CIR, production] + ZHAO-CUI-
  FLORESCU-2018 Prop 3 eq (19) [transform, oracle]`. The book's §3.2.2 eq (3.30) is **not** cited (its
  argument-order rendering is defective; the paper carries the correct form). `oracle:`; `crosscheck:`
  (QuantLib has no 3/2 engine — likely `crosscheck: none — a non-affine model QuantLib does not implement`).
- `references.md`: **`ZHAO-CUI-FLORESCU-2018` flips `in-hand → worked-from` here** (it gains a `src/` citer —
  the oracle transcribes eq (19); run `verify.py references`).
- `references.md`: `KWOK-ZHENG` already `worked-from`; no flip.
- Version → **v0.79.0**; one floor bump; one `CHANGELOG` entry. **Re-time the 3/2 MC on BOTH CI runners
  (§9)** — the eighth heavy-numerics file in the arc; measure CI, a lossless pass before any OI-8-style
  exemption. If a `hyp1f1` production-path decision lands (the Ask above), its own §9/provenance follows.

## Out of scope (carried / deferred → T78+)
- **Options on variance under the 3/2 model** — the full confluent-hypergeometric transform; `to_go_deeper`
  (this tranche is the variance swap / first moment only).
- **The time-changed-Lévy framework + the discrete-monitoring adjustment** — T78 (which 0401 made a pricing
  tranche, not only a convergence study).
- **Timer options** — T79 (the capstone, off the Heston transform).
