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

## 5. Cowork's ruling on T41b — (reserved for §10 step 4, after the build)
