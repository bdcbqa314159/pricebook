# T63 — spread options: Margrabe (exact) + Kirk (approximation)

**Cowork spec, 2026-11-19.** Fourth tranche of the commodity arc (project eight, `CLARK-2014`). The
**spread option** — an option on `S₁ − S₂` (crack/spark/calendar spreads are the commodity desk's bread and
butter). Two pieces: **Margrabe** (the zero-strike exchange option, **exact**) and **Kirk** (the
finite-strike spread, an **approximation**). Both collapse to a **1-D Black-76 on a combined volatility**, so
both reuse `black_price` — **no bivariate normal** (though `foundation/bivariate_normal.py` exists if a later
spread route needs it). The subtle work of this tranche is entirely in the **correlation oracle**: ρ enters
as a *product* `σ₁σ₂ρ`, which is where T17/T34/T43 concentrate.

## Target
`price(CommoditySpreadOption, model) → PricingResult` on two commodity underlyings, by Margrabe when the
strike is zero and Kirk when it is not.

## Pieces, layer, reuse
- **The product** (new, L2) — `CommoditySpreadOption`: two underlyings (`asset_long`, `asset_short`, each a
  `CommodityAsset` from T60), strike, expiry, direction, notional. That is six fields → **bundle** (§7): the
  two legs into an asset *pair* value object, or Code's best vocabulary — raise if the grouping is not
  obvious. The two legs are an ordered pair (long − short), not a set.
- **The model** (new, L3) — a two-asset commodity Black model carrying the two volatilities and the
  **correlation** ρ (the one genuinely-new parameter). `CommoditySpreadModel(market, vol₁, vol₂, rho)` or a
  pair of `CommodityBlackModel`s plus ρ — Code rules the shape; ρ is a model/market parameter, **not** a
  product field. It is distinct from `CommodityBlackModel` (two assets + a correlation is a different idea),
  so no merge — the T60 two-Black-models precedent.
- **The engine** (new, L4) — assembles the **combined volatility** and calls `black_price`. Reuses T60's
  forwards for each leg and `black_price` whole; the only new closed-form code is the combined-vol assembly
  and Kirk's shifted forward.

## The mathematics (re-transcribe from the source — NOT this spec, §5)
Read **Clark ch 2.9.1 (Margrabe) and 2.9.2 (Kirk)** in `-raw` and `-layout` and reconcile (§4).
- **Margrabe** is Black with zero rate, forward `F₁`, strike `F₂`, and combined vol
  `σ_M = √(σ₁² + σ₂² − 2ρσ₁σ₂)`, discounted appropriately — **exact**.
- **Kirk** approximates the finite-strike spread by treating `S₂ + K` as a single lognormal, giving a
  combined vol `σ_K` with ρ entering through a weight that depends on `F₂/(F₂+K)` — an **approximation**.
The load-bearing term both share is the **`−2ρσ₁σ₂` cross term**; transcribe `σ_M` pointwise, and Kirk's
weighted form pointwise. The spec states no formula as authority.

## Oracles (RED-first, §4)
1. **Pointwise transcription of Margrabe's `σ_M`** — the test forms its own `√(σ₁²+σ₂²−2ρσ₁σ₂)`
   independently. The cross term is the mutation target.
2. **The correlation oracle is the heart — pin the PRODUCT `σ₁σ₂ρ`, not the factors (T43/T17).** ρ enters
   `σ_M` only through `σ₁σ₂ρ`; a set of oracles that confirm ρ is *present and correctly signed* is **silent
   about its magnitude** (the T43 trap — "complete about presence and sign, silent about magnitude"). So
   assert the **value** of the combined vol against an independent transcription at a non-trivial ρ (e.g.
   ρ = 0.4, −0.3), not merely that larger ρ lowers the spread vol. And pin ρ **where it is directly
   observable** in the MC companion: the **realised covariance** of the two simulated log-increments must
   reproduce `σ₁σ₂ρ·t` for the pair (the T17 pin — a price pins only the ρ-weighted scalar; the realised
   covariance pins ρ itself, and a factorisation bias shows as failure-to-improve as √N).
3. **Reduction: Kirk → Margrabe as strike K → 0** (exact — at K=0 Kirk's weight is 1 and `σ_K = σ_M`).
   **Companion required (§4/T39):** a **finite-strike** Kirk case (K ≠ 0) checked against the MC companion,
   since the K→0 reduction is blind to exactly the shifted-forward approximation Kirk introduces.
4. **The MC companion MEASURES Kirk's finite-strike error (§4), and proves ρ unbiased first.** Simulate two
   **correlated** GBM forwards (reuse `foundation/linalg.py` Cholesky + `mc_estimator`), price the spread by
   discounted payoff. **Before** crediting the companion, verify the simulated correlation is unbiased (the
   realised-covariance pin, #2) — a variance reduction or a factorisation must be proved unbiased before it
   is trusted (the T19 rule). Band Kirk-vs-MC as `k·SE + measured Kirk error`, the Kirk error measured at a
   huge path count, resolution stated — Kirk degrades with strike-away-from-zero and with |ρ|→1, so sweep
   both and record where it exceeds ~0.5% (the honest negative, as T62).
5. **Requirement oracles — the ρ = ±1 degenerate limits.** At **ρ = +1** the two logs are perfectly
   correlated, so `σ_M = |σ₁ − σ₂|` and the spread is (near-)deterministic for equal vols; at **ρ = −1**,
   `σ_M = σ₁ + σ₂` (maximal). Assert both closed forms — they survive a transcription error the source-value
   oracle might share, and they pin the cross term's sign. Also **Margrabe put–call / exchange symmetry**:
   an option to exchange A for B relates to the reverse by a parity the finance forces.
6. **Reduction: σ₂ → 0 ⇒ an ordinary option on S₁** (a spread against a deterministic leg is a vanilla) —
   reduces to the T60 commodity option, an exact cross-product check against a different engine.

## Degeneracy premises (§4 / §9)
- **ρ must be non-zero and the two vols unequal** in the main fixtures — at ρ = 0 the cross term vanishes
  (a dropped `−2ρσ₁σ₂` survives), and at σ₁ = σ₂ with ρ = 1 the combined vol is exactly 0 (hiding structure).
  Assert the premise; prefer a **mutation** (drop the cross term, flip its sign, drop the factor 2, swap
  ρ→ρ²) to a reading.
- The realised-covariance pin needs **enough paths that a factorisation bias is resolvable above the MC SE**
  (T17: ~1.66% at 80k paths halving as √N) — state the count and that a bias shows as failure-to-improve.
- The Kirk-error sweep must reach a (strike, ρ) region where the error is **resolvable above the SE** — else
  "Kirk agrees with MC" asserts only that both are noisy (the band-resolution rule).

## Provenance / gates
- `source: CLARK-2014 §2.9.1, §2.9.2`; `crosscheck:` (QuantLib has Kirk/spread-option analytic engines — a
  real target); `oracle:`.
- Version → **v0.65.0**; one floor bump; one `CHANGELOG` entry (lead with the measured Kirk-error surface).

## Out of scope (carried / deferred)
- **Calendar spread options and Asian spread options** (ch 2.9.3–2.9.4) — `to_go_deeper` (the Asian spread
  composes T61/T62 with this; a real route, unpulled).
- **Three-plus-asset / basket spreads** — would pull the bivariate→multivariate normal; `to_go_deeper`.
- Stochastic / term-structure correlation — a deterministic ρ is the spine (the frozen-basis precedent, T58).
