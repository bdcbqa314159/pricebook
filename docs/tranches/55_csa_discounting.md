# Tranche 55 — CSA discounting (the collateralised trade)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.58.0`. **Source:**
`LICHTERS-2015` ch 5 ("CSA Discounting") + ch 3's OIS/projection split. **Pre-registered RED.** The second
build of the collateralised-discounting arc; its arc plan (and the **single-curve design invariant**) is
`docs/modern_discounting_arc.md`. Builds on T54's bootstrap.

## 0. The product worth having
**A collateralised trade priced under CSA discounting** — a swap whose floating leg projects off its
**tenor projection curve** and whose cashflows discount at the **OIS (collateral) rate**, because a
perfectly-collateralised trade (cash CSA in the trade currency) earns the collateral rate on posted
collateral. This is the first pricer that reads **both** curves T54 built, and the first that can be run in
**either era** — single-curve (the pre-crisis / `MCINERNEY-ZASTAWNIAK` world) or multi-curve/CSA — which is
exactly the parallel Bernardo wants to show. Cross-currency / foreign-collateral discounting is **T56**;
collateral-choice, thresholds and negative-rate floors (ch 5.3–5.6) are deferred (§4 fence).

## 1. The pieces and their layers
- **L0 — a `Tenor` value object** (new vocabulary, §6, ruled at T54 Ask (b)). A tenor (3M, 6M, …) is the
  dimension a projection curve is keyed by, and there is none in the tree today (it is implicit in a
  `CouponSchedule`). Finance-free (a period/frequency), so **L0**. Introduced here, where first consumed —
  not speculatively at T54.
- **L1 — the `projections` store on the curve-set** (ruled at T54 Ask (b)): a separate store keyed by
  `(currency, Tenor)`, **not** a widened `CurveKey` (which would hang a `None` tenor on every DISCOUNT/REAL
  key — §6 carried-but-unread), the `survivals`-own-store precedent. A typed accessor returns a projection
  curve for `(currency, tenor)`.
- **L4 — the multi-curve / CSA swap pricer.** Given a model whose snapshot carries a `CurveSet` (OIS
  discount + projections), it **projects** each floating coupon off the coupon's tenor curve and
  **discounts** every cashflow off the OIS curve. **The existing single-curve swap pricer is untouched**
  (the invariant): the pricer discounts off a *source* that is either a single discount curve (single-curve
  mode) or the `CurveSet`'s OIS curve (multi-curve mode), chosen by what the snapshot holds (§3).
  **Read-before-asserting (§10/§5):** this spec assumes an existing swap/leg product and the single-curve
  swap pricer — confirm what exists and reuse it (the T54 lesson: an anchor may *exist and not be what the
  spec assumes*; grep before building).

## 2. The oracle (RULED)
The **pre-crisis reduction** ties it to what we hold; the **single-curve invariant** is Bernardo's
parallel-era guarantee and its own first real test; the **CSA-discounting transcription** is the correctness
anchor.

- **The CSA-discounting formula — transcribed pointwise, two renderings (§4).** The collateralised swap PV
  is the tenor-projected floating leg minus the fixed leg, each cashflow discounted at OIS (ch 5). Carry a
  **requirement oracle**: a **par** collateralised swap (fixed rate = the multi-curve par rate) is worth
  **0** under CSA discounting — derived from the finance, so it survives a transcription error (and it is
  non-telescoping *only* if the par rate is computed by independent code; a par rate read back from the
  pricer telescopes and is kept as the consistency guard, the correctness anchor being the reduction + the
  cross-check).
- **The pre-crisis reduction — the exact anchor to what we hold (§4).** With **OIS = projection** (the
  degenerate single-curve case), the CSA price must equal the **existing single-curve swap price** to the
  ULP (a reordering, so ~few-ULP relative, stated — the T54/T49 bit-identity-vs-tolerance discipline, not
  claimed bit-identical). Different construction, same number.
- **The single-curve invariant — the first REAL test (the arc invariant + T33b guard).** Price the *same*
  swap in **genuine single-curve mode** (the existing path, a single discount curve) and in
  **multi-curve/CSA mode**, and assert: **they AGREE where the eras coincide** (OIS = projection, the
  reduction) and **DIFFER by the basis where a real OIS/projection basis exists** (a measured, non-zero,
  maturity-growing gap — the T54-finding-2 shape: 0 at one period, growing with maturity). This proves both
  modes are *reachable and exercised*, not that one collapses to the other — so a parallel single-curve
  project is supported and the single-curve path cannot silently atrophy (T33b: a mode whose branch leaves
  the test population). **Premise:** a **real basis** in the fixture (OIS ≠ projection) and **≥ 2 periods**
  (one period is the single-curve degeneracy — the df cancels).
- **Cross-check vs QuantLib (§4, second source).** `crosscheck: quantlib` — QuantLib prices a multi-curve
  collateralised swap; agree to a measured tolerance (the dev-only tier, never a merge gate — §8).
- **Mutation (premises asserted up front).** Each must die: discount the floating leg at the **projection**
  curve instead of OIS (→ the single-curve error; killed by the **real-basis companion**, assert it);
  project a coupon off the **OIS** curve instead of its tenor (→ same shape); make the pricer ignore the
  snapshot's `CurveSet` and fall back to a single curve (→ the single-curve-invariant *differ* assertion
  fails — the mode-reachability guard biting). **Premises:** a real basis, ≥ 2 periods, a multi-tenor fixture
  if more than one tenor is in play.

## 3. The convention register (RULED at T54, opened here)
Open the `conventions_register.md` row **guarded** (not `owed`), the CSA pricer its first consumer:
**"A collateralised trade discounts at OIS; a floating leg projects off its tenor curve."** Guarded by the
single-curve-invariant oracle (which asserts the discounting *is* OIS and the projection *is* the tenor —
a mutant swapping them dies). Cross-module: the pricer (L4) ↔ the `CurveSet`/`projections` (L1) ↔ the
`Tenor` (L0).

## 4. Process
Branch `tranche/55-csa-discounting`. `test` (RED: the CSA formula transcription + the par requirement + the
pre-crisis reduction + **the single-curve-invariant agree/differ** + the mutation table) → `feat` → `docs`
(the register row; the arc plan's invariant now has its first guarding oracle) → `chore(release): v0.58.0`.
**Transcribe the CSA-discounting relation (ch 5) in both renderings (§4).** `source:` `LICHTERS-2015`.
**Raise an `Ask:`** only if the existing swap/leg product is not reusable as assumed (the T54 read-before-
asserting shape), or if the single-curve "mode" is better carried as a snapshot variant than a source the
pricer selects — a §3/§6 vocabulary call, surfaced by the build.

## 5. What this is NOT (scope fence)
No **cross-currency / foreign-collateral** discounting (T56 — the discount curve changing with collateral
currency, ch 4). No **collateral-choice option / cheapest-to-deliver**, **thresholds / asymmetric CSA**,
**negative-rate floors**, or **initial margin** (ch 5.2–5.6 — `to_go_deeper`). No **re-pricing of every
product onto multi-curve** — multi-curve pricing is pulled by the swap here; other products inherit it when
they pull it (§6). The single-curve pricer is **preserved, not replaced** (the invariant). T55 is one
collateralised swap priced under CSA/OIS discounting, and the proof the two eras are both reachable.

---

## 6. Cowork's ruling on T55 — (reserved for §10 step 4, after the build)
