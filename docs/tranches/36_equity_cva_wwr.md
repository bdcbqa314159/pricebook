# Tranche 36 — equity CVA with wrong-way risk (the capstone: the XVA arc meets the equity arc)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.43.0`.
**Source:** `BROCKHAUS-2016` ch 14.2–14.2.1 (CVA on standard products), ch 14.3 + ch 14.5 (wrong-way
risk, reduced-form), built on T35's ch 13.1.2 apparatus. **Pre-registered RED.** Seventh and final
spine unit of the equity & hybrids arc — **the capstone**, where project three (the XVA spine, T25–T29)
and project four (equity, T30–T35) meet. Its close fires `CP_equity_hybrids.md` and **re-audit #3** (§9).

## 0. The product worth having
**The CVA on an equity-option position held against a defaultable counterparty, with wrong-way risk** —
the counterparty-risk charge on an equity derivative, first *without* and then *with* correlation
between the option's exposure and the counterparty's default. This is the seam the whole arc was aimed
at: the XVA machinery (T25's `unilateral_cva`, T26's `mc_exposure_cva`) priced exposure against a
`SurvivalCurve`; the equity arc built the option and (T35) the defaultable underlying; T36 charges CVA
on the *equity* exposure and then lets the equity and the default **correlate**, which is the one thing
the elementary XVA spine held independent.

Two layers, the second generalising the first:

- **Independent default (ch 14.2.1), the clean closed form and the cross-arc reduction.** For a
  European equity option whose exposure settles at `T`, independent default gives
  `CVA = (1−R)·(1−p₀(T))·P₀(T)·BSCall(F₀(T), K, σ̂, T)` (eq collapsing 9802→9810) — the option's
  discounted price times the default probability times the loss rate. It reuses T30's Black and T25's
  survival `p₀`, and it **must equal our own `unilateral_cva`/`mc_exposure_cva` run on the
  equity-option exposure** — different code on each side, the two arcs reconciled.
- **Wrong-way risk (ch 14.3 + 14.5, reduced-form), the correlation.** The equity and the counterparty's
  Gaussian hazard correlate at `ρ` (T35's deferred ch 13.1.2 apparatus: the `a_T` solve of eq 13.4, the
  truncated-Gaussian `H_T ≈ (a_T + σ_H Y)⁺`, the bivariate-normal terms). Positive exposure-default
  correlation raises the charge (wrong-way); negative lowers it (right-way).

## 1. The pieces and their layers
- **L5 `risk/` — the equity-CVA caller (independent default).** A thin function composing T25's CVA
  weights with the equity-option exposure, reusing `unilateral_cva`/`mc_exposure_cva` and `black_price`
  — not a new engine, a new *consumer* of both arcs at L5 (CVA is risk, above the engine, §2). No new
  product type; the exposure is the `EuropeanEquityOption` priced through the registry.
- **L3 `models/` — the correlated equity-credit model (WWR).** A frozen bundle extending T35's
  `DefaultableEquityModel` idea with the equity/hazard correlation `ρ` and the Gaussian-hazard
  volatility `σ_H` (the 13.1.2 parameters), carrying the `MarketSnapshot` and the `SurvivalCurve`.
  Fields grouped to the §7 limit (if the parameter count presses the limit, bundle the Gaussian-hazard
  pair as vocabulary, §6 — it *is* one object, the hazard's law).
- **L5 `risk/` — the WWR CVA (closed form + MC).** The closed-form `CVA = V̂ − V` via the bivariate
  normal (ch 14.3/14.5), and an independent **correlated-default Monte-Carlo** (equity + correlated
  Gaussian hazard, the *exact* indicator with no truncated-Gaussian approximation) carrying its SE — the
  non-degenerate anchor and the measure of the approximation's own error.

## 2. The oracle (RULED)
The `ρ → 0` reduction is exact and doubles as the cross-arc reconciliation; the non-degenerate WWR case
is pinned by an independent correlated MC with `ρ` pinned where it is directly observable, plus
requirement identities and the book's step-wise derivation.

- **The `ρ → 0` exact reduction — the preferred anchor, and it is a DOUBLE anchor (§4).** At zero
  correlation the WWR closed form must collapse **exactly** to the independent-default closed form
  `(1−R)(1−p₀(T))P₀(T)·BSCall` (ch 14.2.1); and that closed form must in turn equal our
  `unilateral_cva`/`mc_exposure_cva` (T25/T26) evaluated on the equity-option exposure — **different
  code each side, the XVA arc and the equity arc reconciled** (the §4 cross-product oracle, now
  cross-*arc*). A no-op (ignoring `ρ`) satisfies the first half, so it is vacuous alone. **Companion —
  `ρ ≠ 0`:** the WWR CVA must differ **materially** from the independent CVA, asserted non-degenerate.
- **Wrong-way sign and monotonicity — DERIVED from the model, never from intuition (the T35 lesson,
  applied before the fact).** CVA must be **monotone in the exposure-default correlation**, and the
  *direction* of that monotonicity is asserted only after deriving it from the joint law — not from the
  phrase "wrong-way risk makes CVA bigger," which is a real-world heuristic whose sign depends on the
  exposure's own direction and the correlation's reference. T35 shipped a test with the economics
  backwards because an intuited direction contradicted the model's constraint; here the guard is built
  in: compute the comparative static (or read it off the MC across a `ρ` grid) and assert the sign the
  **model** produces, with the independent case `ρ=0` as the pivot. State the convention explicitly in
  the test (which `ρ` sign is wrong-way for *this* exposure) so a later reader cannot re-import the
  intuition.
- **The independent correlated MC — the non-degenerate correctness anchor, with `ρ` pinned where it is
  observable (§4, T17).** The closed-form WWR CVA vs an independent MC that simulates the equity and a
  **correlated Gaussian hazard** with the *exact* default indicator (no truncated-Gaussian
  approximation), computing `(1−R)·E[1{τ≤T}·V_τ⁺]` within `k·SE`. Because `ρ` enters the **joint law**
  (the factorisation of the correlated equity/hazard draws), a single CVA scalar is a weak test of it —
  so pin `ρ` **directly** by the realised correlation of the simulated equity and hazard increments
  (the T17 realised-covariance pin), not only by the CVA it produces. Band `k·SE + the truncated-
  Gaussian approximation's own MEASURED error` (measured at a huge path count where sampling noise is
  negligible), never one flat percentage for both (§4 banded-resolution); **assert the band's
  resolution** — the smallest WWR effect a green rules out at the committed path count.
- **Requirement oracles (§4, survive a transcription error).** `CVA ≥ 0` for a positive exposure;
  `CVA → 0` as `h → 0` (no default, no charge) and as `R → 1` (no loss); CVA **bounded above** by the
  full-loss value `(1−R)·P₀(T)·BSCall` (the `p₀→0` limit); monotone in `(1−R)` and in the hazard level.
  These are derived from the finance and hold whatever the bivariate-normal transcription. **Transcribe
  the `a_T` solve (eq 13.4) and the `V̂−V` terms (eq 14.5 ff) in both renderings** and use the book's
  **derivation steps** (13.4 solved for `a_T`; `V̂−V` assembled from the four terms) as the independent
  check on the final line (§4's stronger rung — the approximation is derived in steps, so an earlier
  step checks the last).
- **Mutation (prove the oracle has power).** Each must die: set `ρ = 0` in the WWR path but keep the
  independent path (→ companion dies; the two must differ at `ρ≠0`); flip the `ρ` sign (→ the
  monotonicity-direction test dies, the realised-correlation pin dies); drop the `(1−p₀)` factor in the
  independent closed form (→ the cross-arc reduction to `unilateral_cva` dies); drop the `(1−R)` loss
  factor (→ the `R→1` requirement dies). **Assert the non-degeneracy premises:** `ρ ≠ 0`, `h > 0`,
  `R < 1`, and a survival curve read beyond its first segment (the T25/T35 non-flat premise), so no
  mutant hides behind a fixture where its term is invisible.

## 3. Process
Branch `tranche/36-equity-cva-wwr`. `test` (RED: the `ρ→0` double reduction — to the independent closed
form AND to `unilateral_cva`; the derived WWR sign/monotonicity; the correlated MC with the realised-ρ
pin; the requirement identities; the mutation table) → `feat` → `docs` → `chore(release): v0.43.0`.
**Open the draft PR early; pre-register the requirement identities and the cross-arc reduction first**
(they need no WWR implementation — RED-first, clean since T33). **Re-time `slow`** — the correlated MC
is new cost and the arc's `slow` margin is already the tightest at `test_local_volatility` (~7.7s/23%);
re-time, do not assume (T19). `source:` `BROCKHAUS-2016` (ch 14.2.1, 14.3, 14.5, 13.1.2);
`crosscheck: quantlib` — a real target only if a QuantLib reduced-form equity-CVA/WWR engine is
confirmed to match this convention; otherwise `crosscheck: none — <reason>` per §8 (do not name a
target that may not match).

**Raise an `Ask:` — and this one is likely.** This is the largest spine unit of the arc. If the
bivariate-normal WWR closed form plus the correlated MC prove too large alongside the independent-default
seam, **split: T36a = independent-default equity CVA** (the cross-arc reduction, a complete "product
worth having" on its own — the seam is the headline, and it closes the arc's *reduction* even if WWR
lands a tranche later), **T36b = WWR** (the correlated closed form + MC). The T20/T33 split precedent.
Flag it rather than forcing one branch. Also `Ask:` if, at build, the truncated-Gaussian WWR
approximation turns out **not** to be oracle-able as the method stands (its error pinnable only by a
twin-MC with no closed-form, cross-check, or requirement anchor) — then WWR is `out_of_method` (§1) and
T36 is the independent-default seam, with WWR recorded as a candidate. *(The reading says it IS
oracle-able — ρ→0 and h→0 reduce exactly and the approximation error is MC-measurable — so this is a
guard, not an expectation.)*

## 4. What this tranche is NOT (scope fence)
No **structural / asset-barrier WWR** (ch 14.4, the Merton/CreditGrades first-hitting-time family,
14.4.1–14.4.3) — a whole new model family we do not hold and the arc did not plan; reduced-form is the
route our `SurvivalCurve` and T35 already live on. `to_go_deeper` if interest ever pulls it. No
**portfolio / netting-set** CVA (the `V = ΣV_i` of 14.4, multiple exposures and netting) — a single
equity-option exposure here. No **equity FVA/DVA/KVA on top** (T28's FVA is held; stacking it on the
equity exposure is a later pull, not the capstone). No **collateralised equity CVA** (T29's MPoR/CSA
applied to the equity exposure — a clean later tranche, `to_go_deeper`). No **CDS/defaultable-bond
calibration** of the counterparty curve (an L1 bootstrap, pulled when needed). T36 is the one capstone:
equity CVA, independent then wrong-way, reduced-form. Its close ends the arc's builds and triggers the
checkpoint + re-audit #3.

---

## 5. Cowork's ruling on T36 — (reserved for §10 step 4, after the build)
