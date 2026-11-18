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

## 6. Cowork's ruling on T55 (§10 step 4, 2026-11-18) — RATIFIED

Built as ruled, 2273 tests, eleven gates green both runners, 14/14 killed, `is_clean`, C15 opened guarded.
The collateralised swap prices under CSA/OIS discounting. Ratified.

**The single-curve invariant came out STRONGER than I specced it, and the realisation is yours.** I specced
"a source the pricer selects", which is a flag; you found the era is a property of the **market** (a
snapshot *with* projections is multi-curve, *without* is single-curve), so there is **one pricer and no
second path to atrophy** — nothing is reachable by accident. That is the §6 "remove the thing that could
diverge" shape (T53's delegation, T49's extraction) applied to Bernardo's parallel-era invariant, and it is
a better guarantee than two pricers. The T33b guard still bites (forcing the single-curve branch dies on the
*differ* half), so the invariant is both structural and tested. This is the best realisation of the
parallel-era requirement; it goes in the arc checkpoint.

**The read-before-asserting miss is MINE, and now twice — a §10 note.** The spec named "the existing swap
pricer"; the tree has never had a vanilla IRS. That is the second consecutive spec asserting a reuse anchor
I did not grep (T54: exists-but-misunderstood; T55: does-not-exist). §10 puts that grep at **step 1, with
Cowork** — I leaned on your build-catch backstop instead. Charter §10 now records both, and I grep reuse
anchors at spec-time going forward. The cost here (T55 brought the **IRS product, L2** — a larger unit than
specced) is owned: the product was genuinely needed, and the design survived because it never depended on
the absent pricer.

**Finding 1 — the *differ* premise was my error, repeated from T54 one layer up.** I wrote "one period is
degenerate (the df cancels)", carrying T54's **curve-level** 1y-pillar identity into a **price-level**
claim — but a price also differs in its *discounting*, so a one-period swap already differs (−59.93 on 10M).
Owned; your fix-to-the-measurement with the premise named is right, and worth keeping *because* it is the
second time the curve-vs-price layer confusion has bitten.

**Finding 2 — the reduction's relative band was vacuous at a zero PV → a §4 edit (the mirror clause).** A
par-rate reduction's PV is **zero by construction**, so a relative tolerance has nothing to be relative to
and fails on noise (`rel=1.25` on 2.1e-09). §4 gains the dual of its near-zero clause: a value that *is*
zero gets an **absolute** band against the scale (notional), which is your rebanding. Good catch — it is the
trap you hit *by reaching for* the existing near-zero rule.

**The `SwapDirection` Ask — keep it per-family, do NOT consolidate. Ruled.** The four ±1 enums
(`CapFloorKind`, `EquityOptionKind`, `SwaptionKind`, `SwapDirection`) share a **word and an `omega`, not a
concept** — a swap is not a swaption, exactly the §6 real-leg-vs-YoY-leg case, and a swap field typed
`SwaptionKind` is the shared-word coupling §6 forbids. **The shared SHAPE is already abstracted the right
way:** `black.py`'s `OptionDirection` **protocol** unifies the `omega` arithmetic *where it is shared*, while
the concrete types stay distinct — the library's "free functions over a protocol, not a base class" pattern.
So the thing that should be shared (the math) *is*, and the things that should be distinct (the instruments)
*are*; four distinct direction enums behind one protocol is not a smell, it is that pattern working. Giving
the swap its own was correct. (The rule-of-two to *consolidate the types* would trigger only if a consumer
ever took "any direction" and read `omega` with no instrument distinction — which is what the protocol
already is.)

**And T52's instrument fired on you from the committed side** — its no-op guard refused a placeholder
mutation mid-build. First time one of those clauses caught Code in real use rather than in the tool's own
tests; the instrument earning its keep exactly as the T52 decision bet.

Two charter edits (§4 mirror clause, §10 reuse-anchor-grep note), both earned — the §10 one by my own miss,
twice. T56 (cross-currency) is next; I grep its reuse anchors before handing it.
