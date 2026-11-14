# Tranche 41 — the one-factor Gaussian latent-variable model (the copula)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.48.0`.
**Source:** `OKANE-2008` ch 13 (the Gaussian latent variable model) — §13.2 (the single-name latent
variable + threshold), §13.3 (the multi-name factor form), §13.4 (conditional independence).
**Pre-registered RED.** Fourth unit of the credit arc and the **first multi-name model** — the
"Black-Scholes of the correlation market." It opens phase 2; the basket (T42) and the LHP tranche (T43)
price through it.

## 0. The product worth having
**The one-factor Gaussian copula: a portfolio's default dependence from a single market factor.** Each
name `i` carries a latent Gaussian `A_i` and defaults by `T` iff `A_i < C_i(T)`; the names are coupled
through one common factor `Z`:

```
A_i = β_i·Z + √(1−β_i²)·ε_i ,   Z, ε_i ~ N(0,1) independent                      (13.3)
Threshold:        C_i(T) = Φ⁻¹(1 − Q_i(T))                   — calibrated to the CDS-implied marginal
Asset correlation: ρ_ij = β_i·β_j                                                 (13.2)
Conditional default (given Z):  p_i(T|Z) = Φ( (C_i(T) − β_i·Z) / √(1−β_i²) )       (13.4)
Pairwise joint default:         p_ij(T)  = Φ₂( C_i(T), C_j(T); ρ_ij )              — reuses Φ₂ (T36b)
```

Conditional on `Z` the defaults are **independent** — the property every correlation product's pricing
exploits. This tranche builds the model and its probability functions, oracled directly; it has **no
`price()` engine yet** — T42's first-to-default basket is the first product that prices through it
(product-pull: the copula is not speculative, two queued tranches pull it immediately).

## 1. The pieces and their layers
- **L3 `models/` — `GaussianCopula`.** Carries the `MarketSnapshot`, the per-name **factor loadings
  `β_i`** (a frozen tuple; `ρ_ij = β_iβ_j`), and the per-name **`SurvivalCurve`s** (the marginals
  `Q_i`, reused from T25). Named, oracled surface (functions, not an engine): the **threshold**
  `C_i(T) = Φ⁻¹(1−Q_i(T))`, the **conditional default probability** `p_i(T|Z)`, and the **pairwise
  joint** `p_ij(T) = Φ₂(C_i,C_j;ρ_ij)`. Fields ≤ 5 (bundle the per-name `(β, survival)` pairs as
  vocabulary if needed, §7). Reuses `Φ`/`Φ⁻¹` (L0 `distributions`) and **`Φ₂`** (T36b, L0).
- **L1/L0 — the market-factor quadrature.** The integral over `Z` (`∫ f(Z)φ(Z)dZ`) is **Gauss-Hermite**
  (Z is Gaussian) — numerical config arriving with this first consumer (§3.5). Reuse an existing
  Gaussian quadrature if one is in `foundation/`; else build the Gauss-Hermite nodes/weights here, L0,
  with its own oracle (it integrates polynomials exactly to its order; `∫φ = 1`, `∫Z²φ = 1`).
- **No engine, no product this tranche.** The copula is a model with closed-form + MC oracles; T42
  (`FirstToDefaultBasket`) brings the first `@register`ed `price()`. (Like `SurvivalCurve` at T25:
  computable, oracled market/model surface before a product fully exercises it.)

## 2. The oracle (RULED)
The **marginal recovery** is the correctness anchor (the model must reproduce its CDS-implied inputs);
the **three exact limit reductions** O'Kane gives are the closed-form anchors; the **joint via Φ₂ vs
the market-factor integral** is a two-method cross-check; an MC pins `ρ` where it is observable.

- **Marginal recovery — the "inverse must invert" requirement (§4), the correctness anchor.** Two
  halves: (a) `Φ(C_i(T)) = 1 − Q_i(T)` exactly, by construction of the threshold (`C_i = Φ⁻¹(1−Q_i)`);
  (b) the conditional probability **integrated over the market factor recovers the marginal**:
  `∫ p_i(T|Z) φ(Z) dZ = 1 − Q_i(T)` — the factor model must not distort the single-name defaults it was
  calibrated to. This is non-telescoping (the integral is a different computation from the threshold)
  and is the analogue of T8's known-curve round trip. Assert the integral's **convergence rate** in the
  Gauss-Hermite order (the T25/T39 rate-detector), not a flat band.
- **The three exact reductions (§4), all closed-form from O'Kane §13.3.** (a) **Independence** `ρ=0`
  (`β_i=β_j=0`): `p_ij = (1−Q_i)(1−Q_j)`. (b) **Max dependence** `ρ=1` (`β_i=β_j=1`):
  `p_ij = min(1−Q_i, 1−Q_j)`. (c) **Min dependence** `ρ=−1` (`β_i=−β_j=1`):
  `p_ij = max(1−Q_i−Q_j, 0)`. Each exact by construction. **Companion (non-degenerate):** a generic
  `0<ρ<1` with **distinct** `β_i` (so `ρ_ij=β_iβ_j` is a real product, not uniform) and **distinct**
  marginals `Q_i` (so the names are not interchangeable — the degenerate-fixture discipline: a fixture
  with equal `β` and equal `Q` hides an index/pairing bug), asserted material.
- **Joint default: Φ₂ closed form vs the market-factor integral — the two-method cross-check (§4,
  different code).** `p_ij = Φ₂(C_i,C_j;ρ_ij)` (the reused T36b bivariate normal) must equal
  `∫ p_i(T|Z)·p_j(T|Z)·φ(Z)dZ` (the conditional-independence integral) — two genuinely different
  computations of the same joint. **Φ₂'s T36b regime applies:** assert the agreement inside `|ρ|≤0.99`
  and `p_ij≳1e-8` (where Φ₂ is accurate), and use the market-factor integral as the reference in the
  `|ρ|→1` corner Φ₂ degrades in — the "a reference is code too" boundary, already mapped at T36b.
- **MC cross-check — the non-degenerate anchor, `ρ` pinned where observable (§4, T17).** Simulate
  `A_i = β_iZ + √(1−β_i²)ε_i`, count defaults (`A_i < C_i`), recover `p_i(T|Z)` and `p_ij` within `k·SE`.
  Because `ρ_ij = β_iβ_j` enters the **factorisation** of the `A_i`, a joint-probability scalar is a weak
  test of it — pin it by the **realised correlation of the simulated `A_i`** (reproducing `β_iβ_j`), the
  T17 realised-covariance pin, halving as `√N`. Also assert the faster threshold/uniform default-time
  method (§13.2: `τ_i = Q_i⁻¹(Φ(A_i))`) agrees with the direct `A_i<C_i` indicator.
- **Mutation (prove power).** Each must die: drop the `√(1−β_i²)` idiosyncratic weight (→ `A_i` loses
  unit variance; the marginal recovery fails); use `β_i` not `β_iβ_j` for `ρ_ij` (→ dies on distinct
  `β`, not on equal `β` — the degenerate premise); swap `C_i`/`C_j` or a sign in the conditional
  `(C_i−β_iZ)/√(1−β_i²)` (→ marginal recovery + MC die); break `Φ₂` back to `Φ·Φ` (→ the `ρ≠0` joint
  cross-check dies). **Premises:** distinct `β_i`, distinct `Q_i`, `0<|ρ|<1`, Gauss-Hermite order stated.

## 3. Process
Branch `tranche/41-gaussian-copula`. `test` (RED: marginal recovery (both halves) + the three exact
reductions + the Φ₂-vs-integral cross-check + the MC with the realised-ρ pin + the mutation table) →
`feat` → `docs` → `chore(release): v0.48.0`. **Open the draft PR early; pre-register the marginal
recovery and the three reductions first** (closed-form, no implementation needed). **Transcribe 13.1–13.4
in both renderings**; O'Kane derives the conditional probability from `A_i<C_i` conditional on `Z`
(§13.4) — use that derivation step as the independent check on `p_i(T|Z)` (§4 stronger rung). `source:`
`OKANE-2008`; `crosscheck: quantlib` — QuantLib's `OneFactorGaussianCopula` / `GaussianLHPLossModel`
machinery is a real target for the conditional probability. **Raise an `Ask:`** if a Gauss-Hermite
quadrature is **not** already in `foundation/` (then it is a small L0 add this tranche, with its own
exact-polynomial oracle — flag it, don't fold silently), or if the per-name `(β, survival)` bundling
presses the §7 field limit (a value object, §6 vocabulary).

## 4. What this tranche is NOT (scope fence)
No **product / `price()`** — the copula is the model; T42's first-to-default basket brings the first
engine. No **basket or tranche** pricing (T42, T43). No **loss distribution** (T44). No **multi-factor**
copula (§13 is one-factor; multi-factor is `to_go_deeper`). No **t-copula / skew copulas** (ch 21 —
`to_go_deeper`). No **base/implied correlation** (T44/ch 19–20). No **random recovery**. T41 is the one
object: the one-factor Gaussian latent-variable model, its conditional and joint default probabilities,
oracled by marginal recovery + the exact limits + the Φ₂/integral cross-check; the products pull it next.

---

## 5. Cowork's ruling on T41 — (reserved for §10 step 4, after the build)
