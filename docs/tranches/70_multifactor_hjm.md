# T70 — multi-factor HJM

**Cowork spec, 2026-11-21.** Fifth tranche of the single-curve short-rate arc (project nine,
`MCINERNEY-ZASTAWNIAK`; ch 4.4). The one-factor Gaussian HJM (T69) with **N driving factors** — the
drift condition becomes a **sum over factors**, and the bond variance gains **cross terms** between factors,
so the T17/T43 correlation discipline returns (as it did at G2++). T69 is the exact `N=1` reduction, so this
tranche's correctness anchor is built already.

## Target
Price a zero-coupon bond and a caplet under **N-factor** Gaussian HJM —
`price(product, MultiFactorHjmModel) → PricingResult` — with
`df(t,T) = Σₖ αₖ(t,T)dt + Σₖ σₖ(t,T)dWₖ`, the `Wₖ` (possibly correlated) Brownian factors.

## Pieces, layer, reuse
- **The model** (new, L3) — `MultiFactorHjmModel` carrying the `MarketSnapshot` (§3), the **N forward-rate
  volatility functions** `σₖ(t,T)`, and the **factor correlation** (or independent factors — the source's
  convention; read it). Reuse T69's `GaussianHjmModel` structure where the per-factor machinery is identical.
  Over the field limit → bundle (§7). **Distinct from `GaussianHjmModel`** only if N-factor is a different
  idea; if it is `GaussianHjmModel` with a list of vols, it may be the *same* type generalised — Code rules at
  the second consumer (§6), raising the vocabulary call at review with the coupling, as at the Asian family.
- **The drift** — `αₖ(t,T) = σₖ(t,T)·∫_t^T (Σⱼ ρₖⱼ σⱼ(t,u)) du` (the multi-factor HJM no-arbitrage condition —
  a **sum over factors** inside the integral). Still a **requirement** oracle (no-arbitrage forces it), the
  T69 anchor generalised.
- **The bond / caplet** — the bond variance is `Σₖ Σⱼ ρₖⱼ ∫Bₖ∫Bⱼ`-type (cross terms); **reuse `zero_bond_option`**
  on the total variance. The MC simulates N correlated factors — reuse **`pseudo_square_root`** (`linalg`,
  the G2++/T63 reuse).

## The mathematics (re-transcribe from the source — NOT this spec, §5)
Read **McInerney–Zastawniak ch 4.4 (multi-factor HJM)** in `-raw` and `-layout` and reconcile (§4). The
load-bearing new structure is the **factor sum** in the drift condition and the **cross terms** in the bond
variance. The §5 hazard is live again (the source may print the SDE, not the integrated variance) — if so,
**pin by the identity** (the discounted bond is driftless; the bond reprices), the T68/T69 move, rather than
transcribing a form the book does not print. The spec names the obligation, not the form (§4).

## Oracles (RED-first, §4)
1. **The multi-factor drift condition as a REQUIREMENT oracle** — the discounted bond is driftless under the
   risk-neutral measure for **all** `{σₖ}` and correlations (no-arbitrage), to numerical zero. Transcribe the
   factor-sum form pointwise once (T12(b) — the requirement never replaces the transcription), with a mutation
   dropping a factor from the sum (a dropped `j` term must break the driftlessness).
2. **Reduction: `N=1` ⇒ the T69 one-factor Gaussian HJM, exactly.** The arc's self-verifying anchor again —
   at one factor the model IS T69's, so assert agreement against the **already-verified** `gaussian_hjm`
   (consistency guard, not independence, §4 same-object caveat). **Companion required (§4/T39):** a genuine
   **≥2-factor** case with non-trivial inter-factor correlation — the `N=1` reduction is blind to the cross
   terms and the correlation, which is everything this tranche adds.
3. **The correlation is pinned TWO ways (T17/T43, the G2++/T63 discipline).** The inter-factor `ρₖⱼ` enters
   the bond variance as a **product** with the vols, so (a) assert the **value** of the cross terms against an
   independent transcription at non-trivial `ρ` (T43 — presence/sign is not magnitude), and (b) pin `ρ`
   **directly observable** via the **realised covariance** of the simulated factor increments (T17), proved
   unbiased before the price band is credited (T19).
4. **The MC companion** simulates the N correlated factors (Cholesky), banded per §4; the closed-form bond
   inside the band; the simulated drift pinned against the drift condition first.
5. **Watch for `α→0`-type limits** in the cross-term `∫` integrals — rate-in-a-regime, **measured for this
   model** (the finding-2 clause; the cross terms may divide by a *sum* of speeds, so the regime is its own,
   per T67).
6. **Requirement oracles:** `P(T,T)=1`; `f(0,T)` reproduces the initial `DiscountCurve`; and a **two-factor
   HJM with the right vols reduces to G2++** (T67) if the structures coincide — a cross-model consistency
   check against already-verified code if the reduction is exact (rule at build whether it is).

## Degeneracy premises (§4 / §9)
- The `N=1` companion is **non-degenerate in the factor count and correlation** (≥2 factors, `ρₖⱼ ≠ 0`,
  distinct `σₖ` — T39).
- **`ρₖⱼ ≠ 0`, distinct vols** so the cross terms are live (at `ρ=0`/identical vols a dropped cross term
  survives); prefer a **mutation to a reading** (drop a factor from the drift sum, drop a cross term, flip a
  `ρₖⱼ` sign, `ρ→ρ²`).
- The realised-covariance pin needs enough paths that a factorisation bias is resolvable above the SE (T17).

## Provenance / gates
- Header `source: MCINERNEY-ZASTAWNIAK §4.4`; `oracle:`; `crosscheck:` (QuantLib has a multi-factor HJM /
  Gaussian engine — a real target).
- Version → **v0.72.0**; one floor bump; one `CHANGELOG` entry. Re-time the N-factor MC on **both** CI runners
  (§9 — N correlated factors have real cost).

## Out of scope (carried / deferred → T71)
- **The ch-8 volatility-smile short-rate models** (Normal/Bachelier, CEV, displaced-diffusion) — T71, the last
  spine tranche.
- **Non-Gaussian / stochastic factor vols** — `to_go_deeper` (the Gaussian case is the spine).
- **Full N-factor calibration to a surface** — the model carries `{σₖ}` and the correlation; the joint fit is
  `to_go_deeper` (the T11 identifiability discipline applies — N factors are famously under-identified by a
  single surface). The initial-curve reproduction (oracle 6) is the calibration this tranche carries.
