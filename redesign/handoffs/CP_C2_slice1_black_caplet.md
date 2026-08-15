# Checkpoint — C2 opening (slice 1: Black-76 caplet)

**Version:** v0.93.0 · **Slice:** `slice/07-black-caplet` · **Baseline:** v0.92.2, C1 closed + audit-hardened.

This slice OPENS the models & calibration cluster (C2): the first dynamics model, the first capability
beyond `Discounting`, and the first `surfaces` snapshot shape. New capability protocol + new vocabulary →
checkpoint required (CLAUDE.md §6). The four review inputs + doc-22 verification + next checkpoint follow.

## 1. The load-bearing rebase — `CalibratedModel` promoted, capability model under load
Doc 22 designed `CalibratedModel` + capability protocols but deferred the code to the second model (rule of
two). C2 is that second model, so the protocols landed and the **engine + registry now depend on the
CAPABILITY (`model.market`), not concrete `DiscountingModel`** — no `isinstance`. Refactor-under-green: the
whole slice-1..6c linear reprice-to-par suite (L3 186, L4 207) stayed green through the rebase, no new red.

**Doc-22 "extends additively" verified (§0).** Promoting `CalibratedModel` + adding `BlackVol` changed the
MEANING of **no** ratified type: `DiscountingModel` is unchanged (still `.market`, structurally a
`CalibratedModel`); the engine rebased onto a protocol `DiscountingModel` already satisfied; `MarketSnapshot`
gained `surfaces`, but that is **filling a pre-ratified closed shape** (doc 19 §2), not ad-hoc field growth.
The second capability *extended* the model additively — it did not break it.

## 2. Oracle-quality audit
| Oracle | Class | Result |
|---|---|---|
| Caplet reprices to Black-76 for a known (F,K,vol,df,t) | **closed-form**, independent code path (inline `erf` vs the scipy `norm_cdf` adapter) | <1e-12 |
| Put-call parity: caplet − floorlet = df·N·τ·(F−K) | closed-form identity | <1e-12 |
| vol→0 degenerate = discounted intrinsic df·N·τ·(F−K)+ | closed-form limit | <1e-12 |
| Linear reprice-to-par (slices 1–6c) through the rebase | refactor guard | green, no new red |

Top-tier: the caplet oracle is a genuine closed-form cross-check (the reference Black is an independent
inline implementation), stronger than self-consistency.

## 3. Challenge-me — the C2-opening design
- **`BlackVol` semantic contract (Q3′a).** Written as: lognormal implied vol of the index forward over its
  accrual, optionlet expiring at `expiry` struck at `strike`, under the **T-forward measure** (T = pay date),
  units annualized lognormal. *Challenge:* normal/Bachelier vol is a **distinct** capability (deferred) — the
  contract is what stops a Bachelier surface being silently read as lognormal.
- **Surface is flat-minimal.** One `flat_vol` at every (expiry, strike); `at()` ignores its args by design.
  *Challenge:* is a flat surface too thin to prove the shape? It proves the *snapshot shape* + the *capability
  wiring*; the smile/term-structure grid earns its complexity with a calibrated-surface consumer (rule of two).
- **`black()` keeps df OUT.** Undiscounted E[(F−K)+]; the engine multiplies df·N·τ. *Challenge:* keeps the
  analytic block ≤5 args and composable, at the cost of the engine knowing the df·N·τ wrapper — acceptable
  (the engine already owns discounting).
- **Capability validated by `isinstance(model, BlackVol)`.** This is a `runtime_checkable` **protocol** check
  (capability satisfaction, Q3′b), NOT an `isinstance`-on-concrete-model ladder — the §1 law is intact.
- **`t` (vol time) is ACT/365F from valuation to expiry.** *Challenge:* hardcoded convention; fine for the
  single-index oracle, revisit if a product needs a different expiry-time basis.

## 4. Smell + debt scan
- `verify.py` acyclic/fields/layers/provenance/debt all green. **`layers` confirms** Black-76 sits at **L3**,
  not L0 (the `foundation/black.py`-at-L0 precedent honoured); `vol_surface` is L1 market data.
- Field/arg discipline: `black()` 5 args; `Caplet` 4 fields; `Surface` 1; `SurfaceKey` 1; `MarketSnapshot`
  4 (≤5). No new suppressions (`debt` balanced).
- Exception-count (§3d gauge): 0 concrete-type switches in the engine/registry; the only `isinstance` is the
  capability-protocol check.

## Drawdown (§4) — 19/793, tick 0 (PARTIAL crosses)
Retire-read of the quarry option sources — both **partial**, tick 0 (ratified expectation):
- `python/pricebook/models/black76.py` — **Black-76 price CROSSES** (ng `black()`). Deferred, forward-linked:
  `black76_delta/gamma/vega/theta` → **C3 risk**; `bachelier_*` (normal vol) → **2nd vol consumer** (Bachelier
  capability). Partial: greeks + Bachelier remain resident.
- `python/pricebook/options/capfloor.py` — **single caplet PRICE path crosses** (ng `Caplet` + `price_caplet`).
  Deferred: floorlet product → its consumer; `CapFloor` strip → **cap/floor slice (B5)**; `strip_caplet_vols`/
  `StrippedCapletVol` → **vol-calibration slice (C2)**; `calibrate_capfloor_sabr` → **SABR (B4)**. Partial.

## Deferred (named triggers)
Bachelier/normal vol (2nd vol consumer) · vol calibration / surface stripping (its slice) · swaption + annuity
numeraire (next product, B5) · 2D smile interpolation (gridded-surface consumer) · engine numerics-config
(first engine with a real method choice — Bachelier/MC — invariant 5; **name it distinctly from calibration's
`SolveConfig`**) · Bermudan / numerically-priced targets ((A)-fork re-open trigger) · SABR/HW (later B4).
**Doc fix (not code, deferred):** rename doc 22's `SolverConfig` → `SolveConfig` (shipped calibration name).

## Named next checkpoint
**C2 slice 2 — Black swaption** (annuity numeraire, §3d — the C1 `rpv01` becomes the swaption's numeraire),
or **vol calibration** (first `SolveConfig` consumer beyond curves). Checkpoint at the first of ≤6 slices or
the next capability boundary.
