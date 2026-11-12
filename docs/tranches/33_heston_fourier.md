# Tranche 33 — stochastic volatility (Heston) by the Fourier / affine method

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.39.0`.
**Source:** `BROCKHAUS-2016` ch 6.2 (stochastic volatility) + ch 16.2 (affine processes — European
options and the Fourier transform). **Pre-registered RED.** Fourth unit of the equity & hybrids arc;
follows T32. The last piece of the **equity core** before the hybrids (T34–T36).

## 0. The product worth having
**The European option under the Heston model, priced by Fourier inversion of its characteristic
function** — the second vol model, and the one whose vanilla is *semi-closed-form* rather than a PDE/MC
grind. Heston:

```
dS = (r−q) S dt + √v · S dW_S
dv = κ(θ − v) dt + σ_v √v dW_v ,   d⟨W_S, W_v⟩ = ρ dt
```

The log-spot characteristic function `φ_T(u) = E[e^{iu log S_T}]` is **exponential-affine** in `v₀`
(ch 16.2), so the call is a one-dimensional Fourier integral of `φ` (Carr–Madan damping, or the
Lewis/Heston form). `σ_v` and `ρ` generate the smile the local-vol surface (T32) only *fits*; Heston
*produces* it from dynamics.

## 1. The pieces and their layers
- **L3 `models/` — the Heston model:** `HestonParams(kappa, theta, sigma_v, rho, v0)` (a frozen bundle,
  §7 vocabulary) carrying the `MarketSnapshot`. The characteristic function `φ_T(u)` (ch 16.2) lives
  here or in L0 as a complex-valued helper.
- **L4 `engine/` — the Fourier vanilla pricer:** the 1-D inversion integral of `φ` (a real quadrature
  with a stated damping/truncation — numerical config arriving with it, §3.5, the T31 `(market, grid)`
  shape). `@register`ed.
- **L4 `engine/` — the Heston Monte-Carlo** (the independent cross-check the non-degenerate anchor
  needs): the two correlated factors + a **CIR variance** simulation (full-truncation Euler — `v`
  floored at 0) on the Jäckel path engine (RNG, correlation, variance reduction reused). Carries its SE.

## 2. The oracle (RULED)
The `σ_v → 0` reduction is exact but degenerate; the non-degenerate case is pinned by an **independent
method** (MC) plus **characteristic-function requirement identities** that survive a transcription error.

- **The `σ_v → 0` exact reduction — the preferred anchor (§4), with the mandatory companion.** At
  vol-of-vol zero, `v(t) = θ + (v₀−θ)e^{−κt}` is **deterministic**, so the Heston vanilla is **exactly**
  Black on the time-integrated variance `∫₀ᵀ v(t) dt` (our verified Black, T5/T30/T32). Exact by
  construction; but a no-op (ignoring stoch vol) also satisfies it — so it is vacuous alone. **Companion:
  `σ_v > 0, ρ ≠ 0`** — where Heston produces a genuine skew, asserted material.
- **Characteristic-function requirement oracles (§4, survive a transcription error) — the sharp ones.**
  `φ_T(0) = 1` (total mass); `φ_T(−i) = E[S_T/S₀]·… = F(T)/S₀` (the **martingale/forward identity** —
  the CF at `u = −i` returns the forward, pinning the drift independently of the vol dynamics); and
  `φ` is a valid CF (|φ| ≤ 1 on the real axis). These pin the normalisation and drift of the CF whatever
  the `κ,θ,σ_v,ρ` transcription.
- **The independent-method cross-check (the non-degenerate correctness anchor).** The Fourier price vs
  the **Heston MC** (2-factor + CIR, independent code and method) within SE, on `σ_v>0, ρ≠0`. A
  Fourier/CF transcription error that survives the σ_v=0 reduction and the special points is caught here
  — different method entirely. Band = `k·SE` (MC) with the Fourier quadrature error measured separately
  and negligible (the T17 two-source split). Pin the MC's own martingale (`E[S_T]=F`) so its drift is
  checked independently of `v`.
- **THE LITTLE HESTON TRAP — flagged, because it is the known implementation hazard here.** The CF
  contains a complex `√` and `log`; the *naïve* principal branch is **discontinuous** in `u` and blows
  up at long maturity, giving wrong prices — Albrecher et al's "little Heston trap". Transcribe the CF
  in the **stable (Gatheral/Albrecher) form** (the branch that stays continuous), and **oracle the trap
  directly**: the Fourier price must match the MC at a **long maturity** (e.g. `T = 5–10y`) where the
  naïve branch visibly fails. A requirement, not a value — the price must not diverge from MC as `T`
  grows. This is `§4`'s "a source/standard formula has a known trap; pin the property the trap breaks."
- **Fourier-inversion validity regime (§4, T25/T32).** The damping `α`, the integration truncation and
  the node count each have a regime; state the smallest mispricing the committed grid resolves, and that
  the truncation/discretization error sits below the MC band — never a flat tolerance hiding both.
- **Requirement + mutation.** Put-call parity on the Fourier prices; arbitrage-free (monotone, convex in
  `K`). **Mutation:** flip the `ρ` sign (→ skew flips, caught by MC and by the skew companion); drop the
  `κθ` drift in the CF; wrong sign in the complex `√`; drop the `q`; use the naïve branch (→ the
  long-maturity trap oracle fails). Each must die.

## 3. Process
Branch `tranche/33-heston-fourier`. `test` (RED: the σ_v=0 reduction + the CF special-point identities +
the MC cross-check + the long-maturity trap oracle + the mutation table) → `feat` → `docs` →
`chore(release): v0.39.0`. **Open the draft PR early; pre-register the CF requirement oracles first**
(they need no implementation — the T32 RED-first fix). **Re-time `slow`** — the Heston MC is new cost
(T19). `source:` `BROCKHAUS-2016`; transcribe the CF (ch 16.2) in **both renderings** and prefer a
**derivation step** where the book gives one (§4's stronger rung — the affine ODEs for the CF are
derived, so an earlier step checks the final `A(u,T), B(u,T)`). `crosscheck: quantlib
AnalyticHestonEngine` (a real target). **Raise an `Ask:`** if the tranche is too large (the Fourier
pricer and the Heston MC may split T33a/T33b, the T20 precedent) or if the Feller condition
`2κθ ≥ σ_v²` regime needs a ruling for the MC's zero-boundary behaviour.

## 4. What this tranche is NOT (scope fence)
No **local-stochastic volatility** (the LSV mix of T32 and T33 — ch 6.3, `to_go_deeper`). No **jump
diffusion** (ch 6.4) and no Bates (Heston+jumps). No **calibration** of Heston to a market surface
(fitting `κ,θ,σ_v,ρ,v₀` — a real route, `to_go_deeper`; T33 prices *given* params, oracled by the
reduction + MC, not by a market fit). No **FFT** performance work (a single-strike quadrature is enough;
Carr–Madan FFT across strikes is `to_go_deeper`). The equity core closes here; T34 begins the hybrids.

---

## 5. Cowork's ruling on T33 — PR #45 (§10 step 4, 2026-11-12)

Accepted — v0.39.0, 1130 green, nine gates, twelve mutants, held as a draft for review (the premature-
ready not repeated). The equity core's last unit; stochastic vol is in.

**RED-first restored — the T32 fix working as intended.** All three oracle files committed before their
implementations; the CF requirement identities (`φ(0)=1`, `φ(−i)=F`, Hermitian symmetry, `|φ|≤1`,
`P1,P2∈[0,1]`, the bounds) state without any implementation, so pre-registering cost nothing and none
could be reverse-engineered. The clean version of the discipline; the T32 lapse is closed.

**The Little Heston Trap oracle — ratified, and your correction is the sharp part.** Your first version
(an absolute bound on the step-to-step change in `T`) measured the *step size*, not the branch — a
smooth function also moves on a coarse grid. The discriminating property is **refinement-invariance**:
a genuine discontinuity keeps its height as the grid refines (naïve branch 1.493 unchanged) while
smooth variation shrinks (stable 6.97e-03 → 1.08e-04). That is §4's convergence-rate instrument used to
*identify* a defect rather than measure one — the same tool as T31's quadrature rate, which is why **no
new clause** is earned: the existing instrument applied to a branch cut. Keeping
`naive_characteristic_function` as the proven wrong-variant companion (§4 "assert the wrong variant
fails") is exactly right. Flagging the trap in the spec turned a famously silent wrong price into an
oracle — the method at its best.

**The σ_v→0 reduction finding a cutoff bias — ratified; asserting the *linearity* is the discipline.**
The error constant across three grids (1.16e-08) is the signature of a bias the grid cannot fix — the
quadrature's lower-cutoff omission, one-directional and **linear in the cutoff** (1.16e-06 → 1.16e-08 →
…). Asserting the scaling (so a second error source shows as the ratio drifting) rather than a bound is
the §8/§4 "a single data point inherits its own domain" applied to a bias. Good, and unlooked-for —
the reduction earned its keep.

**The registry two-models-one-product question — RULED: dispatch on `(product, model)`, as a small
tranche T33b before T34.** You verified empirically that `_PRICERS[product_type]` lets a second
`@register(EuropeanEquityOption)` **silently replace** T30's Black — a real correctness footgun, and
the first time one product has two model-pricers (Black T30, Heston T33, local-vol-MC T32). The fix is
the §3 contract made literal: `price(product, model)` already takes the model, so **dispatch keys on
`(product_type, model_type)`**. This is rule-of-two, not speculation — the equity option is the second
consumer that decides it (§6) — and it removes the silent clobber before the hybrids (T34–T36) add more
product×model pricers. **Rejected:** a model-specific product type (violates §2/§6 — a European option
is one product whatever prices it). `heston_call_price` stays a named function until **T33b**
`@register`s it under `(EuropeanEquityOption, HestonModel)`. Spec: `docs/tranches/33b_engine_registry.md`.

**The two smaller notes:** `models/__init__.py`'s missing docstring (imports above it) — cosmetic, fix
it in T33b while the registry is open. The deliberately Feller-violating MC fixture (`2κθ < σ_v²`)
exercises the zero boundary — good, the Feller `Ask:` is moot.

**No charter edit** — the trap and the bias findings are existing §4 instruments (the rate/refinement
detector, the domain sweep) applied well, not new rules. The charter earned its keep.

**Next:** T33b (the registry) then T34 (the equity-rate hybrid) — both mine to spec; T33b is on disk,
T34 before its build.
