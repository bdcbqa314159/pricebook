# Tranche 44b — base correlation

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.52.0`. **Source:**
`OKANE-2008` ch 20.2–20.3 (base correlation; building the base-correlation curve). **Pre-registered RED.**
The second half of the split T44 (T44a = the exact recursion loss distribution). **Its merge closes the
credit arc** — `CP_okane_credit.md` + re-audit #4 fire when it and T44a have both landed. Needs only the
LHP tranche pricer (T43), **not** the recursion (T44a) — the two are independent.

## 0. The product worth having
**Base correlation — the market's quoting convention for index tranches.** A single correlation
`ρ_B(K)` that, applied to the **equity `[0,K]` tranche**, reprices the market quote; the
base-correlation *curve* is bootstrapped from the quoted tranches, and any `[K₁,K₂]` tranche is priced
as the **difference of two base-correlation equity tranches**:

```
Base correlation ρ_B(K):   the ρ such that EL_ρ([0,K]) reprices the market [0,K] (equity) quote
Bootstrap:                 solve ρ_B(K₁), then ρ_B(K₂), … shortest/most-subordinated strike first
Pricing a [K₁,K₂] tranche: EL([K₁,K₂]) = EL_{ρ_B(K₂)}([0,K₂]) − EL_{ρ_B(K₁)}([0,K₁])
```

Base correlation replaced **compound correlation** (ch 19) because it is **arbitrage-free** where
compound is not — that is its whole reason to exist, and the oracle must hold it to that.

## 1. The pieces and their layers
- **L4 `calibration/` (or `risk/`) — the base-correlation bootstrap.** It **inverts the tranche engine**
  — reprices `[0,K]` through `price()` (the T43 LHP pricer) to match a quote — so by §2 it is an
  **engine-inverting calibrator → L4** (the JY-to-floor-strip precedent, confirmed by the import
  direction). A bracketed root-find (`foundation/solvers.py`) per strike, shortest first; `EL([0,K])` is
  strictly decreasing in β (measured, T44a's handoff), so the root is unique and well-bracketed.
- **L1/L2 — a base-correlation quote set is market data** (strikes + tranche quotes); the
  `SyntheticCDOTranche` (T43) is the product priced through the curve.
- **L4 `engine/` — the `[K₁,K₂]` pricer via the base-tranche difference**, reusing the T43 LHP pricer
  at the two bootstrapped correlations. Reuses `foundation/solvers.py` and T43 whole.

## 2. The oracle (RULED)
The **round-trip** is the correctness anchor (a calibration must invert); the **base-tranche
decomposition** and the **no-arbitrage monotonicity** are transcription-proof requirements.

- **The round-trip — the correctness anchor (§4, the T8/T39 calibration-inverts anchor).** Bootstrap
  `ρ_B(K)` from a set of tranche quotes, then **reprice those tranches through the curve and recover the
  quotes** to tolerance. Non-telescoping (an independent quote set in, the repriced quotes out) — this is
  the anchor that caught BERHAN's misprints, applied to the base-correlation bootstrap. **Companion:** a
  **non-flat** quote set (a flat base-correlation curve degenerates the bootstrap — the T8 flat-curve
  lesson; a flat curve makes every strike's `ρ_B` equal and hides a per-strike bug).
- **The base-tranche decomposition — a requirement (§4).** `EL([K₁,K₂]) = EL([0,K₂]) − EL([0,K₁])` must
  hold for the bootstrapped curve by construction; assert it against a direct `[K₁,K₂]` LHP price at a
  **single** correlation **only where `ρ_B(K₁)=ρ_B(K₂)`** (a flat local curve) — elsewhere the two differ
  precisely because base correlation is a *curve*, which is the content, not a bug (the T43 scale/shape
  discipline: assert the identity the model asserts, not one it does not).
- **No-arbitrage monotonicity — the requirement that is base correlation's reason to exist (§4).** The
  base-correlation expected losses must be **monotone**: `EL([0,K])` increasing in `K` (more capital
  below absorbs more loss), so an implied `[K₁,K₂]` tranche loss is **non-negative**. A quote set that
  implies a negative tranche loss is **arbitrageable** and must be **refused** (the §3-invariant-4
  shape — a `PricingFailure`/raise, never a silent negative loss). O'Kane ch 19.5/20: this is exactly
  where compound correlation fails and base correlation does not — assert a known-arbitrage quote set is
  rejected, and a known-arbitrage-free one accepted.
- **Reduction — a single quoted strike recovers the LHP (§4).** A base-correlation curve from one `[0,K]`
  quote at correlation `ρ` reprices that tranche at exactly `ρ` — the bootstrap of a single strike is the
  LHP inverse, reducing to T43. Exact.
- **Mutation.** Each must die: price `[K₁,K₂]` as **one** correlation rather than the base-tranche
  difference (→ the decomposition + round-trip die on a non-flat curve); skip the monotonicity/arbitrage
  check (→ a non-monotone quote set yields a silent negative tranche loss); bootstrap the strikes in the
  wrong order / reuse a prior strike's `ρ_B` (→ the round-trip fails to recover the quotes). **Premises:**
  a **non-flat** quote set, `0<K₁<K₂<1`, ≥ 2 quoted strikes (a one-strike curve is the reduction), the
  bootstrap read beyond its first strike.

## 3. Process
Branch `tranche/44b-base-correlation`. `test` (RED: the round-trip + the base-tranche decomposition + the
arbitrage-monotonicity refusal + the single-strike reduction + the mutation table) → `feat` → `docs` →
`chore(release): v0.52.0`. **Open the draft PR early; pre-register the round-trip and the arbitrage
requirement first.** **Transcribe the bootstrap (ch 20.3) in both renderings.** `source:` `OKANE-2008`;
`crosscheck: quantlib` — the base-correlation / `BaseCorrelationLossModel` machinery is a real target.
**Raise an `Ask:`** on (a) the **interpolation** of `ρ_B(K)` between quoted strikes (ch 20.4 — linear in
`K`, or the ETL method; rule the convention, since the interpolation *is* a modelling choice the quotes
do not pin), and (b) whether the standardised **index** (CDX/iTraxx) tranche conventions — the fixed
attach/detach points, the standardised coupon — pull detail beyond a generic quote set (defer the index
specifics; a quote set is strikes + spreads).

## 4. What this is NOT (scope fence)
No **compound correlation** (ch 19 — the non-arbitrage-free measure base correlation replaced;
`to_go_deeper` only if a reason to show the smile arises). No **base-correlation surface** across
maturities (ch 20.6) or **bespoke-tranche mapping** (ch 20.9 — a single-maturity index curve here). No
**interpolation zoo** beyond the one convention ruled at build. No **tranche risk/hedging** (ch 20.7).
T44b is the base-correlation curve and the base-tranche pricing; with T44a's exact loss distribution it
closes the credit arc.

---

## 5. Cowork's ruling on T44b — (reserved for §10 step 4, after the build)
