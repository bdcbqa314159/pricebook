# Tranche 41b — survival curves keyed by a credit reference entity

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.48.1` (a structural
refactor; the single-name re-key is bit-identical, so the user-visible numbers do not move). **Source:**
none external — a §3-conformance refactor the multi-name products force, pulled by T41's Ask. **Runs
before T42** (the first multi-name engine). The T33b precedent: infrastructure before the products
multiply the consumers.

## 0. Why this is a tranche (the §3 tension T41 surfaced)
`CurveSet.survivals` is keyed by **`Currency`**, so a multi-name portfolio — N reference names, often one
currency — is **not representable in the snapshot**. T41's copula therefore holds the per-name survival
curves on the *model* (`CreditName.survival`), which is fine with no engine (§3 binds engines) but a real
violation once **T42's basket engine** reaches market data that lives outside `model.market`. The
currency-keying was a single-name simplification: a CDS has a **currency** (its discount curve) *and* a
**reference entity** (its survival curve) — two different keys conflated into one. The tree already holds
the fix's shape: `equities` is keyed by `EquityAsset(ticker)`, not by currency.

## 1. The change
- **L0/L1 — `ReferenceEntity`.** A frozen identifier mirroring `EquityAsset(ticker)`: `ReferenceEntity(
  name: str)` (or `ticker`), the key a snapshot's credit data is held under. Finance-free value type.
- **L1 — re-key `CurveSet.survivals`** from `Mapping[Currency, SurvivalCurve]` to
  `Mapping[ReferenceEntity, SurvivalCurve]`, and `survival(currency)` → `survival(entity)` (raising on a
  missing entity, as it does now — the §3.4 failure-is-a-value shape stays).
- **L2 — `CreditDefaultSwap` gains an explicit `reference_entity: ReferenceEntity`** (the name it
  protects), so the engine looks the curve up by entity, not by the notional's currency. `CreditName`
  (T41) carries a `ReferenceEntity` too, so a copula portfolio is N entities × N curves in the snapshot.
- **L4 — T38–T40 engines** read `model.market.curves.survival(product.reference_entity)` instead of
  `survival(currency)`. The discount curve stays keyed by currency (unchanged — it *is* a currency's).

## 2. The oracle (RULED)
A **pure refactor** on the single-name path, plus the new capability (a multi-name snapshot) demonstrated.

- **Bit-identical single-name (§4 pure refactor).** Every existing T38–T40 oracle prices
  **bit-identically** through the entity-keyed lookup — the key changed, not an operation (the same
  `SurvivalCurve` is returned, found by a different key). Prove bit-identity (this is a key change, a no-op
  on the arithmetic), not ULP — the equivalent-mutant/extraction bar (T33b's `risky_pv01`, the Black/HW
  extractions). The whole suite staying green is corroboration; the bit-identity on a representative CDS
  is the proof.
- **The new capability — a multi-name snapshot (the RED that was impossible before).** A snapshot holding
  **two distinct reference entities in one currency** with **distinct** survival curves, each CDS pricing
  against **its own** entity's curve — the thing the currency key could not express. Pre-register it red
  (it cannot be built on the currency-keyed snapshot). Assert that a CDS on entity A does **not** see
  entity B's curve (the §6 "another name's data cannot be reached" guard, the equity-snapshot precedent).
- **Missing-entity failure (§3.4).** `survival(unknown_entity)` raises a clear error the engine turns into
  a `PricingFailure`, never defaults to another entity's curve (a silent wrong name is a plausible wrong
  number). Assert it.
- **Mutation.** Revert the lookup key to the notional's currency (→ the two-entities-one-currency test
  fails — two CDS get the same curve); drop the missing-entity guard (→ the failure test fails). Each dies.

## 3. Process
Branch `tranche/41b-reference-entity-keying`. `test` (RED: the bit-identical single-name reprice + the
two-entities-one-currency capability + the missing-entity failure + the mutation table) → `feat`
(introduce `ReferenceEntity`, re-key, thread the entity through the products and engines) → `docs` →
`chore(release): v0.48.1`. `source:` none (a refactor); `crosscheck: none — a keying change, no numerics`.
**Raise an `Ask:`** only if a product's reference entity cannot be derived cleanly (e.g. a multi-name
product's per-name entities vs a single `reference_entity` field — but T41's `CreditName` already carries
one, so the shape exists).

## 4. What this is NOT (scope fence)
No new pricing, no new model, no new product beyond the `reference_entity` field — only the key and the
lookups. Not a credit-entity *hierarchy* (sector, seniority, currency-of-issue — speculative, §6; a name
is one `ReferenceEntity`). The discount curve stays currency-keyed. T41b is the one structural change the
snapshot needs before T42 prices a portfolio.

---

## 5. Cowork's ruling on T41b — PR #56 (§10 step 4, 2026-11-14)

Accepted — v0.48.1, 1620 green, ten gates. The snapshot can now hold a multi-name portfolio, and the
single-name path is a proved pure refactor. Two things rode the tranche that the spec did not foresee,
both reported not asked (§7 and §8 left no deferral) — correct, and one is a real shipped defect.

**The bit-identical proof done RIGHT — ratified, and the discipline is the point.** Code captured the
six pre-change values as **hex literals committed in the RED test *before* touching the code**. That is
the correct pure-refactor proof: a frozen reference that predates the change, not a number recomputed
after it (which only proves two runs of the same code agree — the §4 "independence" rule applied to a
refactor's own anchor). All six reproduce exactly. The two-entities-one-currency capability is the RED
that was impossible before, and replacing the old "another *currency* is a PricingFailure" test (which
*was* the conflation) rather than patching it is right.

**`ForwardPeriod` — a compelled bundle, ratified (§7), with a §6 dedup bonus.** The entity put
`ForwardCDS`/`CreditDefaultSwaption` at six fields; §7 says bundle, never raise — so `ForwardPeriod` is
*compelled* vocabulary, not chosen. That both products carried `(schedule, split)` with the **same two
validations written twice** (the §4/T30 duplicated-guard shape) and now have one definition is the
dedup §6 wants — the bundle paid for itself.

**The T40 invariant-6 defect — ratified fix, and it is the degenerate-fixture family again.** `_legs_to`
summed `schedule.periods()` **whole**, so a seasoned forward's settled premiums entered the annuity
discounted at **negative `t`** (factors > 1, compounding paid flows forward) — a **73% annuity
overstatement** on a schedule straddling valuation by two years. A real §3-invariant-6 violation that
**shipped at T40 (v0.47.0)**, latent because **every T40 fixture begins at the valuation date** so the
historical-flow filter had nothing to filter — the "a guard whose branch no fixture exercises" shape (the
T33b/T30 family), compounded by `_legs_to` **bypassing the `is_historical` atom entirely** (the C10
non-caller, below). No published number was wrong (no shipped fixture had a seasoned forward), so no
CHANGELOG correction is owed — but the fix is a genuine correctness repair. The cross-product oracle (a
seasoned forward **is** a spot CDS over the remaining window, 2.2e-16, different code each side) is the
right anchor.

**The C10 Guard-column addendum — RULED: ratified, verdict unchanged, no charter edit.** Code correctly
deferred the ratified-row change to me. The verdict "guarded — structurally" is **accurate** — a
divergence *between callers* of `is_historical` is not expressible. What it never claimed, and what
`_legs_to` exposed, is that a site which **never calls the atom** is outside the structural guarantee
entirely; caller-completeness is the mutation sweep's / §9 register-pass's job, which is exactly how this
was found. The addendum states this precisely; I ratified it in the register and noted the atom now has a
tenth caller. **No charter line** — this is the existing §4 "an oracle must contain the failure in its
domain" (a guarantee covers only its domain; a thing outside it is invisible) on a convention guard, the
same shape as the τ=1 non-caller fixtures and the CI-gate-not-run. The register (content) carries the
specific; the principle is already law.

**No Ask was owed and none made** — `ForwardPeriod` forced by §7, the invariant-6 fix forced by §8
(a defect in code this tranche touched cannot be deferred). Both reported, correctly.

**Next:** T42 (the first-to-default basket) — the first product priced through the copula — mine to
spec. **Mark #56 ready (step 6) → Bernardo merges.**
