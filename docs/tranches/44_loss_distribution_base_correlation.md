# Tranche 44 — the exact loss distribution (recursion) + base correlation

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.51.0` (likely split
T44a/T44b — see §3). **Source:** `OKANE-2008` ch 18 (building the full loss distribution — recursion)
+ ch 20.2–20.3 (base correlation). **Pre-registered RED.** Seventh and **last build** of the credit
arc; its close fires `CP_okane_credit.md` + re-audit #4. Reuses the copula (T41) and the LHP tranche
pricer (T43).

## 0. The products worth having
Two, and the first anchors the second:

**(a) The exact portfolio loss distribution, by recursion (ch 18).** The LHP (T43) is the
infinite-pool *approximation*; recursion computes the **exact** conditional loss distribution for a
finite homogeneous pool by adding credits one at a time (conditional on `Z` the credits are
independent, so each credit convolves its `(1−R)`-loss-with-probability-`p(T|Z)` into the loss buckets),
then integrates over `Z`. Tranches price off the exact distribution; O'Kane Table 18.2 measures the
LHP's error against it.

```
Conditional (binomial for homogeneous independent credits): f(L|Z) built by the recursion over credits
    p_{k+1}(l | Z) = p_k(l | Z)·(1 − p(T|Z)) + p_k(l − n | Z)·p(T|Z)      (add one credit, n loss units)
Unconditional:   f(L) = ∫ f(L|Z) φ(Z) dZ
Tranche expected loss:  EL(T;K₁,K₂) = Σ_l f(l) · [min(l·u, K₂) − min(l·u, K₁)]     (u = loss unit)
```

**(b) Base correlation (ch 20).** The market quotes index tranches by **base correlation**: the single
correlation `ρ_B(K)` that, applied to the **equity [0,K] tranche**, reprices the market quote. The
base-correlation curve is **bootstrapped** from the quoted tranches (shortest/most-subordinated first),
and a `[K₁,K₂]` tranche is priced as the **difference of two base-correlation equity tranches**:
`EL([K₁,K₂]) = EL_{ρ_B(K₂)}([0,K₂]) − EL_{ρ_B(K₁)}([0,K₁])`.

## 1. The pieces and their layers
- **L3/L1 — the exact loss distribution.** The recursion (an L1-ish produced distribution from the L3
  copula, like `Q_B` at T42 — ruled L3 there by the import direction; same here) + the Gauss-Hermite
  `Z`-integral (reused). The tranche expected loss `EL(T;K₁,K₂)` off the exact `f(L)`.
- **L4 `engine/` — the recursion tranche pricer**, `@register(SyntheticCDOTranche, RecursionModel)` —
  a *second* pricer for the T43 product (Black-vs-Heston precedent, T33b registry): same
  `SyntheticCDOTranche`, a different (exact) model. Deterministic; finite-pool MC the cross-check.
- **L4 `risk/` or `calibration/` — base correlation.** Bootstrapping `ρ_B(K)` **inverts the tranche
  engine** (reprice `[0,K]` through `price()` to match a quote), so by §2 it is an **engine-inverting
  calibrator → L4** (the JY-to-floor-strip precedent). The base-correlation curve + the base-tranche
  decomposition.
- **L2 — `SyntheticCDOTranche` (reused, T43)**; a base-correlation quote set is market data.
- Reuses `Φ`/`Φ⁻¹`, the copula (T41), the LHP pricer (T43), `credit_legs`-style discounting, the
  `foundation/solvers.py` root-find (for the bootstrap).

## 2. The oracle (RULED)
The **LHP is the recursion's `N→∞` limit** — a cross-tranche reduction anchoring the exact engine
against T43; the distribution identities are transcription-proof; base correlation is a round-trip.

- **Recursion → LHP as `N→∞` — the preferred anchor (§4), a cross-tranche reduction to T43.** The exact
  recursion tranche price **converges to the LHP** as the pool size grows (the law of large numbers, the
  same limit T43 *is*). The LHP-vs-exact gap at finite `N` is the measured approximation error (O'Kane
  Table 18.2 quantifies it — ~largest error of the approximations), so **band `k·SE + measured
  LHP-vs-exact gap`** and assert the **convergence rate** (the `O(1/N)` the LHP relies on). This reuses
  T43 as the anchor and closes the loop: T43 was the limit, T44 is the exact thing it approximates.
- **The loss-distribution requirement identities (survive a transcription error, §4).** `Σ_l f(l) = 1`
  (a probability distribution); `Σ_l f(l)·l·u = E[L] = (1−R)p(T)` (the mean matches the capital-structure
  identity from T43 — a cross-tranche check, different code); `f(l) ≥ 0`; the **conditional**
  distribution is the **binomial** `C(N,m)·p(T|Z)^m·(1−p(T|Z))^{N−m}` for homogeneous independent
  credits (a closed form the recursion must reproduce *before* integrating — the pointwise anchor on the
  recursion step). The recursion's `K₁=0` equity tranche expected loss = `E[min(L,K₂)]` must match T43's
  Φ₂ formula at the LHP limit.
- **Base correlation — the round-trip (§4, the T8/T39 calibration round-trip).** Bootstrap `ρ_B(K)` from
  a set of tranche quotes, then **reprice those tranches and recover the quotes** (a calibration must
  invert — the anchor that caught BERHAN's misprints). Plus the **base-tranche decomposition**
  `EL([K₁,K₂]) = EL([0,K₂]) − EL([0,K₁])` holds by construction (assert it); and a **no-arbitrage
  requirement** — the base-correlation expected losses must be **monotone** (a `[0,K]` expected loss
  increasing in `K`; O'Kane ch 19.5/20 — a negative implied tranche loss is arbitrage, refused).
- **The finite-pool MC — the non-degenerate anchor.** The exact recursion `f(L)` vs a finite-pool MC
  histogram within `k·SE`; `ρ=β²` pinned by the realised correlation (T17). The recursion is *exact* (no
  approximation at fixed `N`), so this is a correctness check, not an approximation-error band.
- **Mutation.** Each must die: drop the `p_k(l−n)·p` term in the recursion (→ the mean + LHP-limit die);
  use `p` not `(1−p)` for the survival branch (→ `Σf=1` + mean die); in base correlation, skip the
  monotonicity/arbitrage check (→ a non-monotone quote set produces a negative tranche loss silently);
  price `[K₁,K₂]` as one correlation not the base-tranche difference (→ the decomposition + round-trip
  die). **Premises:** `N≥2` distinct behaviour from `N→∞`, `0<β<1`, interior tranche, term structure,
  the tenth-shape schedule straddling the valuation date, a **non-flat** base-correlation quote set (a
  flat one degenerates the bootstrap, the T8 flat-curve lesson).

## 3. Process
Branch `tranche/44-…`. `test` (RED: the recursion→LHP convergence + the distribution identities + the
binomial conditional + the base-correlation round-trip + the base-tranche decomposition + the
arbitrage-monotonicity + the MC + the mutation table) → `feat` → `docs` → `chore(release)`.
**Open the draft PR early; pre-register the distribution identities + the base-correlation round-trip
first.** **Transcribe the recursion (ch 18.3) and the base-correlation bootstrap (ch 20.3) in both
renderings.** **Re-time `slow`** — the recursion over loss units × the `Z`-grid, and the MC, are real
new cost (T19). `source:` `OKANE-2008`; `crosscheck: quantlib` — `IntegralCDOEngine` /
`RecursiveLossModel` and the base-correlation machinery are real targets.
**Likely `Ask:` — raise the split.** This is the arc's largest unit; **T44a = the exact recursion loss
distribution** (anchored by the LHP-limit reduction — a complete "product worth having" on its own) and
**T44b = base correlation** (built on the tranche pricer — needs only T43's LHP, not the recursion, so
it can even precede T44a). The T20/T33/T36 split precedent; flag it with measurements rather than force
one branch. Also `Ask:` if the base-correlation **interpolation** (ch 20.4 — how `ρ_B(K)` interpolates
between quoted strikes, the ETL method) needs a convention ruling, or if the standardised **index**
(CDX/iTraxx) tranche conventions pull detail beyond the homogeneous pool (defer the index specifics).

## 4. What this tranche is NOT (scope fence)
No **Fourier-transform** loss distribution (ch 18 alternative — recursion is the chosen method,
`to_go_deeper`). No **inhomogeneous-loss recursion** beyond what the homogeneous pool needs (per-name
`R` — the heterogeneous recursion, `to_go_deeper`). No **compound correlation** (ch 19 — base
correlation is the arbitrage-free market standard; compound is the non-arbitrage-free measure it
replaced, `to_go_deeper`). No **base-correlation surface** across maturities (ch 20.6) or **bespoke
tranche mapping** (ch 20.9) — a single-maturity curve here. No **LH+/idiosyncratic risk** (ch 17.3). No
**perturbation / fast tranche Greeks** (ch 18.7). T44 is the exact loss distribution and the
base-correlation curve — the arc's two closing objects; its merge closes `CP_okane_credit.md` and fires
re-audit #4.

---

## 5. Cowork's SPLIT ruling (§10 step 1, 2026-11-15) — T44 → T44a (recursion) + T44b (base correlation)

**Split approved, two tranches, on your evidence — not on size.** The three measurements decide it, and
each is the right kind of reason:
1. **T44a is self-anchored** — the recursion → LHP convergence is *exactly* `O(1/N)` (ratios 1.81 → 2.00,
   1.2% against the closed form at N=160), so the LHP limit (T43) is a complete anchor and the **rate is
   the detector**, not a band. It needs nothing from base correlation.
2. **T44b needs only T43** — `EL([0,K])` strictly decreasing in β at every strike gives the bootstrap a
   clean bracketed unique root; it inverts the **LHP** engine and never touches the recursion.
3. **Disjoint cost** — the recursion is the arc's only expensive quadrature (179 ms at N=320; a five-size
   convergence oracle ~0.9 s vs 0.4 s for all of T43), base correlation is cheap root-finds over a closed
   form. The `slow` gate is a live constraint (it caught T42 on Windows); bundling would make the
   base-correlation tests argue with the recursion quadrature for budget — a real reason, not tidiness.

Independent halves, each self-anchored, disjoint oracle *and* cost profiles — the T20/T33/T36 precedent,
and the cleanest instance of it yet (zero dependency either way). **Ruled: T44a = the exact recursion
loss distribution; T44b = base correlation.** Order **T44a → T44b** (you are already on the recursion
half; T44b could precede but needn't). **This doc §0(a)/§1(recursion)/§2(recursion + distribution
identities + MC) is the T44a spec.** The base-correlation content here (§0(b), the base-correlation
oracle, the ch 20.4/index Asks) is **superseded by `docs/tranches/44b_base_correlation.md`**.

**The arc closes at T44b's merge**, not T44a's — `CP_okane_credit.md` + re-audit #4 fire when **both**
land (the exact loss distribution and the base-correlation curve are the arc's two closing objects).

---

## 5. Cowork's ruling on T44a — PR #61 (§10 step 4, 2026-11-15)

Accepted with one change request (the duplication, below) — v0.51.0, 1754 green, eleven gates, CI green
both runners after corrections. The arc's loop closes as designed: the recursion converges to T43's LHP
at exactly `O(1/N)` (ratios 1.81→2.00), so the approximation built first is now the reference the exact
model answers to — by a rate, and the binomial pins the recursion step *pointwise before* any `Z`
integration (the half a sum-to-one distribution would otherwise hide).

**The Ask — RULED: module-level functions over a `TrancheLossModel` protocol; NOT a base/mixin.** The
duplication (`tranche_expected_loss`/`tranche_survival`/`tranche_survival_curve` identical in `LHPModel`
and `RecursionModel`, differing only in `expected_loss_capped`) is a real shared **shape** — the tranche
composition over a model that supplies `expected_loss_capped(K)` — and rule-of-two is met (§6). A shared
**base/mixin is the polymorphism §6 keeps the models away from** (template-method inheritance, an abstract
`expected_loss_capped` with concrete `tranche_*`); the library's answer is the one it uses everywhere —
**free functions over a protocol** (`credit_legs` over curves, the registry over model protocols, T42's
`SurvivalFunction`). So: a `TrancheLossModel` protocol with the one member `expected_loss_capped(K)`, and
`tranche_*` as module-level functions over it; both models stay **concrete and frozen**. This **kills the
width hole at its root** — the first survivor was reachable twice *because* the composition was written
twice; one composition, one oracle, both models covered (your own rule: *a second implementation of a
shared composition needs the shared oracle applied to it, not merely present in the tree* — the cleanest
fix is to stop having a second implementation). **Done in T44a**, and because it touches T43's shipped
`LHPModel` it is a **pure refactor proved bit-identical** on the LHP path (the §4 bar, the T41b
hex-literal-frozen discipline).

**Config on the model (§3.5 generalisation) — ratified.** The protection-leg step count is a property of
*how expensive the model is to evaluate* (LHP 800, recursion 100, measured 8.1e-07), not of the engine —
so it lives on the model, and one hard-coded engine number would make one model wrong. §3.5's "config
deferred to its first consumer" generalises correctly: the consumer that *determines* it is the model.
Right, and forced.

**The two survivors — ratified.** (1) the width division (fixed by the extraction above). (2) the
step-count test telescoped (it read `model.protection_steps`, checking the engine against its own grid,
blind to whether the grid is *adequate*) — the two questions separated, the adequacy asserted against an
independent reference. The §4 telescoping lesson, and a clean separation.

**The slow-gate CI failure — ratified, and it earns a §9 line.** Reported green on a 5.3s **local**
re-time, failed `slow` on **both** CI runners (18.3s/10.1s vs 10.0s) — the **second** time this arc (T42
first). §9 edited: for a numerically-heavy file, **CI is the measurement, not a local re-time**. Your
own adoption of "CI is the reading from here" is exactly right; the charter now carries it so a future
session inherits it. **The vectorisation that fixed it is good** — the `Z`-dependence is the scalar
`q=p(T|Z)`, so the convolution is one array advance (`nodes×N` → `N`), `fsum` outer sum unchanged, inner
convolution reordered → **≤1 ULP, not bit-identity** (correctly *not* over-claimed — the third time you
have flagged that bar, held).

**The scipy-`ndtr` reach-and-withdraw — ratified, and the right instinct twice over.** Reaching for
scipy's vectorised `Φ` and withdrawing it because (a) it would make this model's `Φ` ≠ the copula/LHP `Φ`
— a convention divergence the register exists to prevent — and (b) it bought **nothing** (the convolution
was the cost, the 96 `Φ` evals never were) is "measure before you break a convention for speed." A
performance argument for a convention break that was not a performance argument. Good.

**The anchor-on-the-wrong-implementation finding — ratified.** Vectorising made a second recursion impl
and the binomial pinned only the scalar one *while pricing ran the fast one*; now asserted equal, the
binomial reaching the fast path transitively (9/6 kills on the vectorised branches). "Oracle the thing
that runs" — the same shape as the width survivor. `foundation.standard_normal_nodes` as public surface
(vs reaching into `_rule`) is the §6 reachability rule, one real consumer.

**The concrete-annotation-denying-reuse pattern (3rd: `TrancheSurvival` typed `LHPModel`) — noted, no
charter line.** pyright caught it, as it caught T42's `risky_pv01` and `BasketSurvival` — **the gate we
hardened at T42b earning its keep three times.** It does not ship (pyright gates it), so it is a build
habit for you to watch, not a latent-defect class; the backstop works. No charter edit for it.

**Change request (still draft):** extract the tranche composition to free functions over `TrancheLossModel`
(bit-identical on the LHP path), relocate the width oracle onto the shared function, re-run. Then mark #61
ready.

**Next:** T44b (base correlation) — specced; its merge closes the arc (`CP_okane_credit.md` + re-audit #4).
