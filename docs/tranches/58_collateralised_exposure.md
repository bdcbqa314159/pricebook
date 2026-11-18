# Tranche 58 — collateralised exposure under a CSA: the arc capstone

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.60.0`. **Source:**
`LICHTERS-2015` Part II/III (collateralised exposure) + `CREPEY-2026` (the gap frame, already held). **The
arc's LAST build**; its merge closes the collateralised-discounting arc (`CP_collateralised_discounting.md`
+ re-audit #6). **Reuse anchors GREPPED at step 1 (§10) — and the grep reshaped the scope.**

## 0. What the grep found, and the corrected scope
The spec I almost wrote — "build collateralised exposure under a CSA" — describes something that **already
exists**: `risk/collateral.py` (T29) implements the **zero-threshold, zero-MTA MPoR gap**
`collateralised_exposure(t) = (V(t) − V(t−δ))⁺`, feeds it into the **same `SimulatedExposure` the T25/T27
`credit_adjustment` prices unchanged** (a CSA changes *what* is exposed, not *how* the adjustment is
computed), and **explicitly defers** thresholds, MTA, initial margin, ColVA and gap-WWR. So T58 is **not**
that build. Its genuine, pulled content is two things:
1. **The arc capstone — collateralised CVA of the arc's collateralised interest-rate swap, LGM-driven, in
   BOTH discounting eras.** Wire the arc together: the **LGM (T57)** simulates the rate paths → the
   **collateralised IRS (T55)** marks `V(t)` → **`collateral.py` (T29)** takes the MPoR gap → **`credit_
   adjustment` (T25)** prices the CVA. This is the product worth having: a real collateralised trade's
   counterparty-risk charge, computed in the post-Libor multi-curve world and the pre-crisis single-curve
   world. The era invariant's **last and most-exposed test** (see §2).
2. **A non-zero THRESHOLD extension to `collateral.py`** — the one CSA mechanic T29 deferred that most
   changes the economics (collateral posted only once the mark passes a trigger). MTA, initial margin/MVA,
   ColVA, gap-WWR stay deferred (T29's fence holds; `to_go_deeper`).

## 1. The pieces and their layers (all reuse anchors grepped present)
- **L5 — the collateralised-CVA capstone**, composing `risk/mc_exposure.py` (the MC exposure engine, T26),
  `risk/collateral.py` (`CollateralAgreement`, `collateralized_exposure`, T29), and
  `risk/credit_valuation_adjustment.py` (`credit_adjustment`, T25). No new CVA function — the T29 design
  (collateral enters only through the exposure) is reused whole.
- **The RFE is the LGM (T57, `models/lgm.py`)** driving the IRS mark; the trade is the **collateralised IRS
  (T55, `engine/interest_rate_swap.py` + `market/multi_curve.py`)**. All confirmed present by the step-1
  grep.
- **The threshold** extends `collateral.py`'s gap: with threshold `H`, posted collateral is
  `max(V(t−δ) − H, 0)`, so the exposure is `(V(t) − max(V(t−δ) − H, 0))⁺`. A field on `CollateralAgreement`,
  not a new type.
- **The era invariant is carried as a RULED design constraint (Code's 0315 flag, Cowork's separability
  ruling):** collateralisation and the discounting era are **separable**. The CSA mechanics (the MPoR gap,
  the threshold) are what make exposure collateralised; the **discount source (single curve or OIS) stays a
  parameter** read off the snapshot. The engine **must not require a `CurveSet`** — single-curve
  collateralised exposure is meaningful (the pre-crisis analog, part of Bernardo's parallel) and is
  **pre-registered**, not added after. The failure mode here is not a dropped branch but a *reasonable-
  looking requirement* ("collateralised exposure means OIS"), so the invariant oracle is load-bearing.

## 2. The oracle (RULED)
- **Reduction to the uncollateralised CVA — the exact anchor (§4).** No CSA (`agreement=None`) or
  `threshold → ∞` ⇒ the exposure is `V(t)` ⇒ the **T25 CVA exactly** (bit-identical where `collateral.py`
  already guarantees it). Reuse of the existing reduction.
- **Reduction to the T29 zero-threshold gap (§4).** `threshold = 0` ⇒ `collateral.py`'s existing MPoR gap,
  to the ULP — the new threshold mechanic must collapse to the built one. **Companion (T39):** a non-zero
  threshold with a real `δ`, so the reduction is not the only check (a threshold that did nothing would pass
  `H=0` and `H→∞` and fail here); assert the exposure is **monotone in `H`** (more threshold → more exposure,
  between the two reductions).
- **The era invariant — the last test (the arc invariant, T33b guard, Code's 0315 flag).** Compute the
  collateralised CVA of the *same* swap in **genuine single-curve mode** and in **multi-curve/OIS mode** and
  assert both halves: **agree** where the eras coincide (OIS=projection) **and differ by the basis** where a
  real basis exists. A capstone that quietly required OIS would pass "agree" (trivially, one mode) and fail
  "differ" — the guard biting on the *reasonable requirement* failure mode. **Premise:** a real basis, the
  single-curve path pre-registered.
- **Cross-product: the LGM-simulated swap exposure vs the HW analytic (§4).** The LGM (T57) reduces to HW,
  so the simulated IRS exposure profile's EPE matches an HW-anchored reference within SE — tying the new RFE
  wiring to the verified model (different code each side).
- **Collateral never increases the adjustment (§4 requirement, reused from T29)** — collateralised CVA ≤
  uncollateralised, for all parameters.
- **Mutation (premises asserted up front).** Each must die: drop the `max(·−H,0)` threshold flooring (→ the
  `H=0` vs non-`H` companion); make the engine require a `CurveSet` / ignore a single-curve snapshot (→ the
  era-invariant *differ*/*reachable* assertion — the invariant guard); feed the gap at the wrong `δ` (→ the
  T29 reductions). **Premises:** a real basis, `κ≠0` (structural via `HullWhiteParams`), `δ>0` and `H>0` in
  the non-degenerate companion, the single-curve path exercised.

## 3. Process
Branch `tranche/58-collateralised-exposure`. `test` (RED: the two reductions + their companions + the era
invariant + the LGM-vs-HW exposure cross-product + the monotone-in-`H` and collateral-never-increases
requirements + the mutation table) → `feat` → `docs` (close the arc; the checkpoint is Cowork's on merge) →
`chore(release): v0.60.0`. **An MC capstone — re-time on BOTH CI runners and cache loop-invariant
simulations proactively** (the three-ceiling watch, §9/T53/T55). `source:` `LICHTERS-2015` + `CREPEY-2026`;
`crosscheck: quantlib`. **Raise an `Ask:`** if the threshold genuinely does not reduce cleanly to T29's gap
(then it is a different object, defer it and keep the capstone integration alone), or if wiring LGM→IRS→gap
needs a surface none of the four modules exposes (the T57 "extract a buried step" shape).

## 4. What this is NOT (scope fence)
No rebuild of the MPoR gap, the CVA, or the exposure engine (all reused — T25/T26/T29). No **MTA, initial
margin / MVA, ColVA, or gap / wrong-way risk** (T29's fence holds; `to_go_deeper`) — only the threshold is
pulled. No **cross-currency collateralised exposure** (needs the FX layer T56 is deferred for). No
**multi-factor / multi-currency LGM**. The single-curve era stays first-class (the invariant). T58 is the
arc's capstone: a collateralised swap's CVA, LGM-driven, in both eras, with the one CSA mechanic that most
changes the economics — and the proof the whole arc composes.

---

## 5. Cowork's ruling on T58 — (reserved for §10 step 4, after the build)
