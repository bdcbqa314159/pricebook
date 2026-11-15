# Tranche 43 — the LHP synthetic-CDO tranche (large homogeneous pool)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.50.0`.
**Source:** `OKANE-2008` ch 16.2 (the LHP model) + ch 16.7.1 (the technical appendix: the expected
tranche loss and the bivariate-normal formula eq 16.1). **Pre-registered RED.** Sixth unit of the credit
arc — the **single-tranche synthetic CDO** priced in the large-homogeneous-pool closed form, the model
that *defined* the correlation market. Reuses the copula (T41) and **`Φ₂`** (T36b).

## 0. The product worth having
**A synthetic CDO tranche — protection on the portfolio losses between an attachment `K₁` and a
detachment `K₂`** — priced by the **LHP model** (Vasicek's infinite-homogeneous-pool limit of the
one-factor Gaussian copula). In the limit of infinitely many identical credits (same `p`, `R`, `β`), the
law of large numbers collapses the conditional loss to a point mass at its mean, so the whole loss
distribution is a **closed form**:

```
Conditional loss (point mass):   L(T|Z) = (1−R)·p(T|Z),   p(T|Z) = Φ((C − βZ)/√(1−β²)),  C = Φ⁻¹(p(T))
Loss-exceedance threshold:       A(K) = (C(T) − √(1−β²)·Φ⁻¹(K/(1−R))) / β ,   Pr(L ≥ K) = Φ(A(K))
Expected loss capped at K (ch 16.7.1, eq 16.x):
    E[min(L(T), K)] = (1−R)·Φ₂(C(T), −A(K); −β) + K·Φ(A(K))                 ← reuses Φ₂ (T36b), eq 16.1 IS our Φ₂
Tranche [K₁,K₂] expected loss:   EL(T) = E[min(L,K₂)] − E[min(L,K₁)]
Tranche survival:                Q(T,K₁,K₂) = 1 − EL(T)/(K₂−K₁)
```

Then the tranche prices like a CDS on the **tranche notional that amortises as losses eat it**: the
protection leg pays the expected-tranche-loss increments, the premium leg pays the spread on the
outstanding `(K₂−K₁)·Q(T,K₁,K₂)`.

## 1. The pieces and their layers
- **L3 `models/` — the LHP model (homogeneous one-factor Gaussian).** Built on the `GaussianCopula`
  (T41) specialised to **homogeneity** (one `β`/`ρ=β²`, one `R`, one default curve `p(T)`) and the
  infinite-pool limit. The named, oracled surface: `A(K)`, `E[min(L,K)]` (the eq-16.x closed form), the
  tranche expected loss `EL(T)` and survival `Q(T,K₁,K₂)`. Reuses `Φ`/`Φ⁻¹` (L0) and **`Φ₂`** (T36b, L0).
- **L2 `products/` — `SyntheticCDOTranche`.** Frozen: attachment `K₁`, detachment `K₂` (`0 ≤ K₁ < K₂ ≤ 1`),
  the homogeneous pool spec (the single `ReferenceEntity`-set or an index handle, the shared `β`, `R`),
  the premium schedule, spread, notional, maturity. Prices nothing.
- **L4 `engine/` — the tranche pricer**, `@register(SyntheticCDOTranche, LHPModel)`. Builds `EL(T)` on
  the premium schedule, then premium leg (spread on the amortising outstanding notional) + protection
  leg (the `EL` increments discounted) — the CDS-on-a-tranche-survival shape, reusing the discounting
  arithmetic. **Deterministic** (the LHP is closed form); the **finite-pool Monte-Carlo** is the
  cross-check oracle (the LHP's own approximation error), not the engine.

## 2. The oracle (RULED)
The **capital-structure identities** are the transcription-proof anchors; the **Φ₂ closed form vs the
market-factor integral** is a two-method cross-check; the **ρ→0 limit** is exact; the **finite-pool MC**
is the non-degenerate anchor and measures the LHP's approximation error.

- **Capital-structure requirement identities (survive a transcription error, §4).** (a) **The full
  structure [0,1] is the pool's expected loss:** `E[min(L,1)] = E[L] = (1−R)·p(T)` exactly (`L ≤ 1`
  always) — a closed-form the Φ₂ expression must reproduce. (b) **Tranches partition the loss:** over a
  partition `0 = a₀ < a₁ < … < aₙ = 1`, `Σ EL([aₖ₋₁,aₖ]) = E[L] = (1−R)p(T)` — the equity + mezz + senior
  losses sum to the pool loss, for all parameters. (c) `EL` **decreasing in `K₁`**, **increasing in
  `K₂`**; `0 ≤ Q(T,K₁,K₂) ≤ 1`; `EL ≥ 0`. These are derived from the finance and pin the Φ₂ composition
  no transcription can fake.
- **The Φ₂ closed form vs the market-factor integral — the two-method cross-check (§4, different code).**
  `E[min(L,K)] = (1−R)Φ₂(C,−A(K);−β) + KΦ(A(K))` (the reused T36b bivariate normal, eq 16.1 confirmed
  to **be** our Φ₂) must equal `∫ E[min(L,K)|Z] φ(Z) dZ = ∫ [(1−R)p(T|Z)·𝟙(...) …] φ(Z) dZ` by
  Gauss-Hermite — two genuinely different computations of the capped expected loss. **Φ₂'s T36b regime
  carries** (`|ρ|≤0.99`): the Φ₂ argument is `(C, −A(K); −β)`, so `|β|<1` keeps it accurate; the `β→1`
  limit is handled analytically (the T41 unit-β step lesson — at full correlation the pool is comonotone,
  the loss is the single-name loss, closed form).
- **The ρ→0 (β→0) exact reduction (§4).** At zero correlation the law of large numbers removes the `Z`
  dispersion: the loss is a **point mass at `(1−R)p(T)`**, so `E[min(L,K)] = min((1−R)p(T), K)` exactly
  and the tranche loss is deterministic. **Companion (non-degenerate):** `0<β<1` with `0<K₁<K₂<1` where
  the Φ₂ term is live and the loss is genuinely dispersed — asserted material, the degenerate-fixture
  discipline (a `K₁=0` or `K₂=1` fixture degenerates one Φ₂ argument; use an interior tranche).
- **The finite-pool Monte-Carlo — the non-degenerate anchor AND the LHP's approximation error (§4,
  banded).** Simulate `N` homogeneous credits (the copula `A_i = βZ + √(1−β²)ε_i`), compute the realised
  portfolio loss and tranche loss, → the LHP as `N→∞`. O'Kane: the LHP-vs-exact gap is ~5% at finite `N`,
  the conditional-loss variance falling `O(1/N)`. **Band the check `k·SE + the measured LHP-vs-finite
  gap`** (the §4 two-error-source rule — never one flat percentage), assert the **convergence** of the
  finite-pool price to the LHP as `N→∞` (the LHP is the limit, not a wrong number), and pin `ρ=β²` by
  the realised correlation of the `A_i` (T17). The LHP is an *approximation* (§1) and this is its
  oracle: exact reduction at β→0, closed-form capital-structure identities, and a convergent finite-pool
  cross-check — all three, so it stays in method.
- **Mutation (prove power).** Each must die: drop the `K·Φ(A(K))` term in `E[min(L,K)]` (→ the [0,1]=E[L]
  identity dies); swap the Φ₂ arguments or the `−β` sign (→ the partition sum + the MC die); use `β` not
  `√(1−β²)` in `A(K)` (→ the reduction + MC die); drop the `(1−R)` (→ every identity dies). **Premises:**
  `0<β<1`, `0<K₁<K₂<1` (interior tranche), `p` term-structured, semi-annual schedule (the tenth
  degenerate-fixture shape — the schedule straddles the valuation date where the historical filter bites).

## 3. Process
Branch `tranche/43-lhp-tranche`. `test` (RED: the capital-structure identities + the Φ₂-vs-integral
cross-check + the β→0 reduction + the finite-pool MC convergence with the realised-ρ pin + the mutation
table) → `feat` → `docs` → `chore(release): v0.50.0`. **Open the draft PR early; pre-register the
capital-structure identities first** (closed-form, no implementation). **Transcribe `A(K)`, `E[min(L,K)]`
(eq 16.x) and the bivariate-normal eq 16.1 in both renderings** and **confirm eq 16.1 is our `Φ₂`**
(the T17 same-formula check — it is the standard bivariate normal, but confirm the `ρ` sign convention
matches before reusing). **Re-time `slow`** — the finite-pool MC is new cost (T19). `source:`
`OKANE-2008`; `crosscheck: quantlib` — QuantLib's `GaussianLHPLossModel` is a real target. **Raise an
`Ask:`** if the tranche notional amortisation (the outstanding `(K₂−K₁)Q` the premium leg pays on)
needs a convention our CDS leg arithmetic does not carry (the running-vs-upfront tranche coupon, the
standardised-coupon detail — rule in-scope vs defer), or if the homogeneous pool wants an **index
handle** (a named pool) rather than an explicit entity set (a §2/§6 representation question).

## 4. What this tranche is NOT (scope fence)
No **inhomogeneous pool** (per-name `p`/`R`/`β` — the exact recursion, ch 18 / T44, not the LHP). No
**base correlation** (ch 20 / T44 — the LHP prices *given* `β`; the base-correlation *calibration* is
T44). No **LH+ / idiosyncratic** refinement (ch 17.3). No **tail dependence / Student-t** LHP (ch 16.5,
`to_go_deeper`). No **implied/compound correlation** (ch 19). No **tranche risk/hedging** (ch 17). T43 is
the one closed form: the LHP expected tranche loss and the single-tranche CDO priced on it, oracled by
the capital-structure identities + the Φ₂/integral cross-check + the convergent finite-pool MC. T44
brings the exact loss distribution and base correlation and closes the arc.

---

## 5. Cowork's ruling on T43 — (reserved for §10 step 4, after the build)
