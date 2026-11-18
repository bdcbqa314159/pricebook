# Tranche 54 — the OIS / multi-curve bootstrap

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.57.0`. **Source:**
`LICHTERS-2015` ch 3 ("Clearing House Pricing" — building the OIS curve and the forward-projection curves).
**Pre-registered RED.** The first build of the collateralised-discounting arc (project seven); its arc plan
is `docs/modern_discounting_arc.md`. Registers `LICHTERS-2015` → `worked-from`.

## 0. The product worth having
**The multi-curve bootstrap — one OIS discount curve and the separate forward-projection curves** the
post-crisis world prices on. A collateralised trade discounts at the **OIS (collateral) rate**, while each
floating leg projects its forwards off a **tenor-specific projection curve** — so the single discount curve
the tree has today is replaced by a *set*: an OIS discount curve calibrated to OIS swaps, and one projection
curve per tenor calibrated to that tenor's par/basis swaps **using the OIS curve to discount**. The value
worth having is the curve-set a desk prices every linear instrument off.

## 1. The pieces and their layers
- **L1 — the multi-curve bootstrap, in `market/`.** Market quotes (OIS swap rates; per-tenor par-swap and
  tenor-basis quotes) → curves, using only L0+L1 — a **curve bootstrap**, so **L1** (§2, the T2/T8 split;
  it *produces* market data). It extends, and is checked against, the existing **`market/nominal_bootstrap.py`
  (T8)** and composes into **`market/curve_set.py`**. **Read-before-asserting (§10/§5):** this spec names
  `nominal_bootstrap.py` and `curve_set.py` as the reuse anchors from their names; confirm at build that
  they are what they are assumed to be (the T22 guard — a one-line read before the build, not after).
- **The sequential dependency is the content:** OIS discount is bootstrapped **first** (OIS swaps discount
  themselves), then each projection curve is bootstrapped **against the OIS discount curve** (the tenor
  swaps discount at OIS, project at tenor). The order is load-bearing and is a premise the oracle asserts.
- **Reuse:** T8's par-swap bootstrap machinery and the discount/forward value types. No new L0 primitive
  expected.

## 2. The oracle (RULED)
The **known-curve round trip** is the correctness anchor (a bootstrap must invert); the **single-curve
reduction** ties it to what we already hold.

- **The round trip — the correctness anchor (§4, the T8 pattern verbatim).** Bootstrap the OIS and
  projection curves from a quote set, then **reprice every calibrating instrument** (the OIS swaps off the
  OIS curve; each tenor's swaps off its projection curve discounted at OIS) and recover par (0) to
  tolerance. Non-telescoping where the repricing composes different code than the bootstrap; this is the
  anchor that caught BERHAN's misprints, on a two-curve construction.
- **The single-curve reduction — the exact anchor to what we hold (§4).** When the OIS curve and the
  projection curve are **the same** (OIS quotes = tenor quotes, the pre-crisis degenerate case), the
  multi-curve bootstrap must reproduce **`nominal_bootstrap.py`'s single-curve output to the ULP** — a
  cross-product regression against T8, different construction, same number. **Companion (required, T39):**
  a **genuinely multi-curve** fixture where OIS ≠ projection (a real basis), so the reduction is not the
  only check — a multi-curve bootstrap that silently collapsed to single-curve would pass the reduction and
  fail here.
- **The sequential-dependency requirement (§4 requirement oracle).** The projection curve is bootstrapped
  *against* the OIS discount curve, so **perturbing the OIS curve moves the projection curve's implied
  forwards** — assert that a shift in an OIS quote changes a projection-curve forward (it must; a projection
  curve that ignored the OIS discount would be the old single-curve error). This pins the order the box
  cannot show.
- **Mutation (premises asserted up front).** Each must die: discount the tenor swaps at the **projection**
  curve instead of OIS (→ the single-curve error; killed only by the **OIS ≠ projection companion**, assert
  it); bootstrap the curves in the wrong **order** (→ the sequential-dependency requirement); drop the basis
  between OIS and projection (→ the companion). **Premises:** a **non-flat** OIS curve and a **real basis**
  (OIS ≠ projection, so the multi-curve-ness is visible — the single-curve case is the degenerate one, the
  §4 "flat curve / degenerate reduction" trap stated); ≥ 2 tenors so a per-tenor bug is visible; the
  projection bootstrap reads the OIS curve, not a copy.

## 3. Process
Branch `tranche/54-ois-multicurve-bootstrap`. `test` (RED: the round trip + the single-curve reduction & its
OIS≠projection companion + the sequential-dependency requirement + the mutation table) → `feat` → `docs` →
`chore(release): v0.57.0`. **Transcribe the OIS/projection bootstrap relations (ch 3) in both renderings
(§4).** `source:` `LICHTERS-2015`; `crosscheck: quantlib` — QuantLib has multi-curve bootstrapping, a real
target. **Open the draft PR early; pre-register the round trip + the single-curve reduction first.**
**Raise an `Ask:`** on (a) the **curve conventions** — which instruments calibrate the OIS vs each
projection curve, the interpolation (log-linear on discount factors vs the Hagan–West the tree already
uses), and whether these want a **`conventions_register.md`** row (the OIS-vs-projection discounting split
is exactly a cross-module convention, §7); (b) whether the curve-set is one **`CurveSet`** value object or
the OIS and projection curves are separate snapshot members — a §6 vocabulary question decided by what the
pricer reads.

## 4. What this is NOT (scope fence)
No **CSA discounting** yet — that is T55 (this builds the curves; T55 prices a collateralised trade off
them). No **cross-currency / foreign collateral** (T56). No **re-pricing of every existing product onto
multi-curve** — the existing single-curve products keep working, and **single-curve stays a first-class,
deliberately-selectable mode** (the arc's design invariant, `modern_discounting_arc.md`): multi-curve is a
sibling, not a replacement. The single-curve path is **kept exercised in genuine single-curve mode**, not
only as the `OIS=projection` degenerate reduction — the reduction proves the eras *coincide*, the retained
single-curve oracle proves single-curve is *reachable* (so a parallel single-curve project is supported and
the path cannot atrophy, the T33b input-population guard). Multi-curve pricing itself arrives with the first
product that pulls it, not speculatively (§6). No
**OIS short-rate model** (that is the LGM, T57, under exposure). T54 is the curve-set and its bootstrap, and
the proof it reduces to what we hold.

---

## 5. Cowork's ruling on T54 (§10 step 4, 2026-11-18) — RATIFIED

Built as ruled, 2212 tests, eleven gates green both runners, 14/14 killed, `is_clean`. The multi-curve
bootstrap is in, the single-curve invariant honoured by construction (the OIS curve *is*
`nominal_bootstrap.py`, untouched and still the single-curve path; only the projection bootstrap is new).
Ratified. And **read-before-asserting earned itself on day one** — my spec said "extends
`nominal_bootstrap.py`"; you found the OIS half *is* it verbatim (an OIS par swap discounts its own
cashflows, the float leg telescopes, BERHAN 5.1.2), so T54 is one new projection function, not a module. The
spec over-credited the new work; the grep before the build caught it, which is the T22 guard paying off.

**Ask (a) — the register row opens at T55, not T54. Ratified.** "A collateralised trade discounts at OIS, a
floating leg projects at its tenor" is a cross-module convention only once a **second** module (a pricer)
reads both — at T54 the bootstrap alone touches it, so a row now would be `owed` on arrival, which §7's
folklore line is not about. Open it at T55 **`guarded`**, with the CSA pricer as its first consumer and
oracle. No new convention from the interpolation (you kept the tree's).

**Ask (b) — projection keying: a separate store, not a widened `CurveKey`. Ratified as the direction,
decided concretely at T55.** The projection curves are keyed by a **new dimension (tenor)** that
DISCOUNT/REAL keys do not have, so widening `CurveKey` to `(role, currency, tenor)` would hang a `None`
tenor on every existing key — a carried-but-unread field (§6). The `survivals`-as-its-own-store precedent
(keyed by its own dimension) is the right shape: `projections` keyed by `(currency, tenor)` with a typed
accessor. **And T55 owns the missing piece you surfaced — there is no tenor / rate-index type in the tree
(the tenor is implicit in a `CouponSchedule`).** A `Tenor` value object is new vocabulary introduced where
it is first consumed (§6 compelled bundling), at T55, not speculatively at T54. T54 correctly adds no
snapshot member.

**Finding 3 — T8's skipped-pillar `KeyError` → `OPEN.md` OI-6, not a T54 fix. Ruled.** A quote set that
skips a pillar passes `_validate`'s prefix check and then crashes with a `KeyError` on the unsolved pillar —
a precondition surfacing as a **raise-and-hope instead of a `PricingFailure`**, which is a real §3.4
roughness, not merely cosmetic (so "no reshaping where already correct" does **not** shield it — it is not
correct). But it is **T8's** module, not T54's subject, so it does not ride T54's PR (one unit per tranche);
**add OI-6** (low severity — it crashes on malformed input, it does not ship a wrong number) with the
re-open trigger **"the next tranche to touch `nominal_bootstrap.py`, or re-audit #6."** Your T54 module
refusing in domain vocabulary is the right contrast and is what surfaced it — the new thing built right
showing the old thing's roughness, the OI-4-at-T49 shape.

**Findings 1 & 2 affirmed, no charter edit.** (1) The reduction is **1 ULP, not bit-identical** (reordered
ops, §4's condition fails) — committed as a relative tolerance with the ULP recorded, the T12(a)/T49
discipline; my "to the ULP" was loose and you stated which it is, correctly. (2) The 1y-pillar
inter-era difference being **0** is the **single-period degeneracy** (one period, the discount factor
cancels whichever curve carries it — the catalogue's "single node, a horizon invisible") — fixed to the
measurement (0 → 7.8e-05 at 5y) with the premise asserted, so the companion is only non-degenerate at ≥ 2
periods. Good catch; it is the degenerate-fixture discipline on your own guess.

Nothing new in the charter — the arc opens lean. T55 (CSA discounting) carries the register row, the
projection keying + `Tenor` type, and the first real test of the single-curve invariant (a pricer that reads
both modes).
