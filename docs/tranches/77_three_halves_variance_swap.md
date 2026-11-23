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

## The model and its closed form (re-transcribe from the source — NOT this spec, §5)
Read **Kwok–Zheng §4.2.2** (the 3/2 model variance swap) in `-raw` and `-layout` and reconcile (§4). The
3/2 variance process (name the vol-of-vol **`epsilon`** in code — `sigma`/`σ` collides with Heston's and
with the CEV exponent, the T71 β-collision lesson):

    dv = κ·v·(θ − v) dt + epsilon·v^{3/2} dW

— note the mean-reversion **speed scales with `v`** (`κv(θ−v)`, *not* Heston's `κ(θ−v)`), which is what makes
the deterministic limit a **logistic** ODE (below). Transcribe the §4.2.2 closed form for
`E[∫₀^T v_t dt]` pointwise (the F-6.1 rule); the spec names the obligation, not the form (§4).

**The special-function / layer question — raise it before building, do not resolve it silently.** If the
§4.2.2 fair-variance closed form needs a **confluent hypergeometric / Kummer `M`** (or any function beyond
`math.exp/log/lgamma` and the L0 `noncentral_chisquare` machinery already in the tree), that is a
**production-path** dependency, and scipy is **oracle-only** here (the T20a `distributions.py`-off-scipy
arrangement). So: if the variance swap (first moment alone) reduces to **elementary + gamma/incomplete-gamma**
functions — which it may, since the *first moment* is far simpler than the full transform — build it on the
existing machinery. If it genuinely needs `hyp1f1` on the production path, **`Ask:`** — that is a new L0
primitive decision (its own oracle, the T20a reference arrangement), not a silent `scipy` import in `src/`.
Report which it is at step 1.

## Pieces, layer, reuse
- **The model** (new, L3) — a `ThreeHalvesModel` frozen dataclass carrying the 3/2 parameters
  (`v0, kappa, theta, epsilon`, and the `MarketSnapshot` it is calibrated to per §3) — ≤5 fields (§7);
  bundle into a value object if over. Sibling of `HestonParams`/`CevModel`, **not** a subclass (§6 — a
  different SDE, a different accrual).
- **The engine** (new, L4) — `@register(VarianceSwap, ThreeHalvesModel)`, reading the §4.2.2 closed form for
  the fair variance and projecting it into a `PricingResult` (fair strike × notional × discount, the §3
  engine-result shape). Reuses nothing of `variance_replication` (that is the model-free strip; this is a
  model closed form) — the reuse is the **registry/`PricingResult` plumbing** and the product.
- **The MC companion** (new or reuse) — the 3/2 SDE needs its **own** Euler/Milstein simulation of
  `v_t` (`heston_mc` simulates the *spot* under Heston's affine variance, a different process), so either a
  small `three_halves_mc` or an extension of `mc_path_engine`. The realized `∫v dt` from simulated variance
  paths against the closed form, banded per §4 (`k·SE + measured discretisation error`, resolution stated) —
  and watch the **3/2 process's fatter tails / explosion risk** (`v^{3/2}` diffusion): a naive Euler scheme
  can blow up, so the companion's own accuracy must be established before it judges anything (the "a
  reference is code too" rule), e.g. a fine-step / full-truncation or log-Euler scheme, measured.

## Oracles (RED-first, §4)
1. **Pointwise transcription of the §4.2.2 fair-variance closed form** — the load-bearing new formula; the
   test forms its own reference independently (and in two renderings, §4). If a derivation step precedes the
   final form in §4.2.2, use the earlier step as the independent check (the T12(b) derivation-structure
   anchor).
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
4. **The CIR-reciprocal cross-check (a second independent route, IF it transcribes cheaply).** `u = 1/v`
   under the 3/2 SDE follows a CIR/Heston-type affine process, so the 3/2 fair variance has an alternative
   expression through the reciprocal map — a genuinely different derivation (different code) of the same
   number, the T75-oracle-1 cross-METHOD shape. **Pull only if §4.2.2 or §4.2.1 gives the reciprocal form
   cheaply**; else `to_go_deeper` (an independent anchor, available, not load-bearing — the reductions +
   transcription + MC already pin it).
5. **Requirement oracles (structural, survive a transcription error — F-6.1):** the fair variance `> 0`;
   **monotone increasing in `v₀`** (more initial variance ⇒ more expected variance); and the long-horizon
   limit `(1/T)·E[∫v] → θ` as `T → ∞` (the fair *rate* tends to the long-run variance — a requirement the
   mean-reversion forces), swept where non-degenerate.

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
- Header `source: KWOK-ZHENG §4.2.2`; `oracle:`; `crosscheck:` (QuantLib has no 3/2 engine — likely
  `crosscheck: none — a non-affine model QuantLib does not implement`, but check before asserting).
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
