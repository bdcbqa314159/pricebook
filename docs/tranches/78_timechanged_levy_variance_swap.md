# T78 — variance swaps under time-changed Lévy processes, and the discrete-monitoring adjustment (a pricing tranche)

**Cowork spec, 2026-11-23.** Fifth tranche of the vol-derivatives arc (project ten, `KWOK-ZHENG`). The arc's
**jump** pillar: a time-changed Lévy variance process, under which a variance swap has a closed form, **and**
— the deliverable 0401 made this a *pricing* tranche, not only a convergence study — the **discrete-monitoring
adjustment**: the discretely-sampled realized-variance swap priced as *the continuous counterpart plus an
adjustment*, with the discrete→continuous convergence **rate** as how the adjustment is oracled.

## Sourcing (pinned from the TOC — the arc plan's "4.3.3" was wrong)
Read in `-raw` + `-layout` and reconcile (§4):
- **§2.4 — time-changed Lévy processes** (the framework: §2.4.1 subordinators / activity rates; §2.4.4 the
  Barndorff–Nielsen–Shephard model). The **model**.
- **§4.4 — variance swaps under time-changed Lévy** (the pricing): **§4.4.1** continuous fair strike,
  **§4.4.2 swaps on discrete realized variance** (the adjustment — the pricing deliverable), §4.4.3
  generalized (out of scope here — T74 already did gamma/corridor under replication), **§4.4.4 convergence
  of fair strikes** and **§4.4.5 conditions on convergence in expectation** (the rate, how the adjustment is
  oracled).
- **§1.5.3 — replication of swaps on discrete realized variance** (the *model-free* discrete result, a
  cross-check on §4.4.2's model-based one).

## Target
`price(VarianceSwap, LevyModel) → PricingResult` for the **continuous** fair strike, and
`price(DiscreteVarianceSwap, LevyModel) → PricingResult` for the **discretely-monitored** one, the second
built as *continuous + the §4.4.2 adjustment*.

## The model, and the one thing to settle at step 1
**Transcribe the specific time-changed Lévy model §4.4 works its closed form for — do NOT assume one**
(the arc's repeated spec-miss is guessing the source's choice; §5). The framework is general (a Lévy process
under a stochastic time change); §4.4 will work a concrete instance. **Likely tractable** (a hint, not a
commitment): a Variance-Gamma or similarly elementary-cumulant model, whose variance-swap strike and
discrete adjustment are functions of the Lévy **cumulants** (elementary, no special functions). **Step-1
Ask trigger:** if the model §4.4 uses needs a **special-function primitive** the tree lacks (a Bessel for
NIG, etc.) on the production path, raise the T20a layer `Ask:` (scipy oracle-only, build-our-own-in-`src/`)
before building — the T77 pattern.

## Pieces, layer, reuse
- **The model** (new, L3) — a `LevyModel` (name per the concrete model chosen) frozen dataclass carrying the
  Lévy/time-change parameters + the `MarketSnapshot` (§3). ≤5 fields (§7). Sibling of
  `HestonParams`/`ThreeHalvesModel`, not a subclass (§6).
- **The discrete product** (new, L2) — a `DiscreteVarianceSwap` carrying the **monitoring schedule / n**
  (the continuous `VarianceSwap`, T31, is reused for the continuous arm). ≤5 fields.
- **The engines** (new, L4) — `@register(VarianceSwap, LevyModel)` (continuous, §4.4.1) and
  `@register(DiscreteVarianceSwap, LevyModel)` (discrete = continuous + §4.4.2 adjustment). Reuse: the
  registry/`PricingResult` plumbing; the continuous strike is the reduction target for the discrete one.
- **The MC companion** — a new time-changed-Lévy path simulation (the existing `heston_mc` is a diffusion
  sim, different process); realized variance (continuous-proxy at fine steps, and discretely-monitored at
  `n`) against the closed forms. Establish the simulator's own accuracy first (jump processes are
  heavy-tailed — "a reference is code too").

## Oracles (RED-first, §4)
1. **The continuous Lévy variance-swap strike transcribed pointwise** (§4.4.1), with the **exact reduction
   to the pure-diffusion case** ⇒ T31's continuous variance swap when the jumps vanish (the Lévy measure → 0),
   and a **companion with genuine jumps** (T39 — the reduction zeroes exactly the jump part the whole tranche
   is about).
2. **The discrete-monitoring adjustment (§4.4.2), transcribed — anchored by its exact pure-diffusion value.**
   **CORRECTED (0419): my original "the adjustment vanishes with jumps" was wrong** — it is a discrete-sampling
   effect that exists with **no jumps at all**. Prop 4.3's correction is `(1/T)Σ(E[Δln S])² + (1/T)Σ
   var(E[Δln S|F^Z])`, whose first term survives in the pure-diffusion case as **exactly `(r−q−σ²/2)²·T/N`**
   (the classic discrete-variance drift bias; measured `adjustment × N` constant to 8 digits over `N=4…2520`).
   So the anchor is **"pure-diffusion adjustment = `(r−q−σ²/2)²T/N` exactly"** — a nonzero closed form,
   strictly stronger than a zero (which would also be the vacuous kind §4 warns of, satisfied by a no-op), and
   it **is simultaneously oracle 3's O(1/N) rate constant**. The adjustment vanishes as **`N→∞`, not as
   jumps→0.** Companion with **genuine jumps** (T39), where the extra Lévy term appears. Cross-check §4.4.2's
   model-based adjustment against **§1.5.3's model-free** discrete replication where both apply.
3. **The convergence RATE (§4.4.4/4.4.5) — how the adjustment is oracled, measured not plotted (the T25
   rate-detector).** The discrete strike → the continuous strike as `n → ∞`; assert the **rate** (O(1/n)
   expected — transcribe the order §4.4.4 states) by varying `n` and fitting, not a single flat band. A
   plateau rather than convergence is the detector's red (the T25 "a reference is code too" caught-by-rate
   case). The adjustment × n → the stated constant is the sharp form.
4. **The MC companion** measures the discrete and continuous strikes from simulated Lévy paths, banded
   `k·SE + measured error`, resolution stated (§4), swept where the adjustment is resolvable above the SE.
5. **Requirement oracles:** fair variance `> 0`; the adjustment's **sign** (§4.4.2 forces it — discrete
   realized variance sits above/below continuous by the jump structure; transcribe and assert it); the
   discrete strike monotone toward continuous as `n` grows (if §4.4.5's conditions give monotonicity).

## Degeneracy premises (§4 / §9)
- The pure-diffusion reduction's companion is **non-degenerate in the jumps** (a genuine Lévy measure, T39) —
  state it; zero jumps is the reduction in disguise and blind to the adjustment, which is the whole product.
- The convergence fixture reaches an `n` where the adjustment is **resolvable above the MC SE** (else "discrete
  ≈ continuous" asserts only that both are noisy — the band-resolution rule).
- **Any κ→0 / `(1−e^{−x})/x`-style numerics watch is scoped to REFERENCES, not production** (Code's T77 input,
  0412 — T77's removable singularity lived in a test reference, not the stochastic path; expect the same here,
  where a rate constant or an activity-rate integral may carry one). Measure the sound regime for this model
  (T67), do not inherit.

## Provenance / gates
- Header `source: KWOK-ZHENG §2.4 [model] + §4.4.1/§4.4.2/§4.4.4–5 [pricing + convergence] + §1.5.3 [model-free
  discrete cross-check]`; `oracle:`; `crosscheck:` (check QuantLib for a time-changed-Lévy variance engine
  before asserting `none`).
- `references.md`: `KWOK-ZHENG` already `worked-from`; no flip.
- Version → **v0.80.0**; one floor bump; one `CHANGELOG` entry. **Re-time the Lévy MC on BOTH CI runners
  (§9)** — the arc's heavy-numerics family; a lossless pass before any OI-8-style exemption.

## Out of scope (carried / deferred → T79)
- **Generalized variance swaps under Lévy (§4.4.3)** — gamma/corridor already built under replication (T74);
  their Lévy-model versions are a `to_go_deeper` note, not pulled.
- **The BN-S model specifically**, if §4.4 uses a different concrete instance — `to_go_deeper` as a second
  time-changed model unless §4.4 itself works it.
- **Timer options** — T79, the capstone.
