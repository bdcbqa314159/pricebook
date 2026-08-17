# Checkpoint — C2 slice 2 (Black European swaption)

**Version:** v0.94.0 · **Slice:** `slice/08-black-swaption` · **Baseline:** v0.93.0 (Black caplet).

Second capability on one model → the capability-model rule-of-two now has its **multi-capability** case
(one `BlackModel`, opt-in `BlackVol` + `SwaptionVol`). Amends the C2-opening checkpoint.

## 1. The multi-capability moment — verified additive
`BlackModel` now satisfies `CalibratedModel` + `BlackVol` + `SwaptionVol`. **`BlackVol` and `BlackModel`'s
existing meaning are UNCHANGED** — `swaption_vol` is a new method reading a new key type; `black_vol` and
`.market` are byte-identical to slice 1. This is doc 22 Q1's "one or more capability protocols" exercised
for real: a second capability **extends** the model, it does not break it. The engine still dispatches
structurally (registry, no `isinstance`-on-concrete-type); the only `isinstance` is the runtime-checkable
`SwaptionVol` capability check (Q3′b).

## 2. Oracle-quality audit
| Oracle | Class | Result |
|---|---|---|
| Swaption reprices to `N·rpv01·black(S,K,vol,t)` for known inputs | **closed-form**, independent inline `erf` reference | <1e-12 |
| Payer − receiver = `N·rpv01·(S−K)` | closed-form identity (annuity parity) | <1e-12 |
| vol→0 = `N·rpv01·(S−K)+` | closed-form limit | <1e-12 |
| **§3d identity:** swaption(payer−receiver) == the swap engine's PV of the same swap | cross-engine consistency | <1e-9 |

The §3d oracle is the load-bearing one: it proves the swaption engine's `S`/annuity are the SAME composition
the swap engine/calibrator use (`rpv01`, `float_leg_pv`) — not a second annuity loop. If it failed, engine and
calibrator would have diverging swap-rate compositions (the "calibrates-to-par-but-prices-wrong" trap §3d
warns of). It passes → the atoms are genuinely shared.

*Note (§7c):* oracles run at **unit notional** so the `<1e-12` absolute tolerance is a tight O(1) check, not
machine-epsilon-relative at 1e6 scale. PV is exactly linear in notional — identical computation.

## 3. Challenge-me
- **`SwaptionVol` is a distinct capability, not a generalized `BlackVol`.** A swaption vol is the vol of the
  swap rate under the annuity measure; a caplet vol is an index-forward vol under the T-forward measure —
  different underlying, measure, numeraire. Q3′(a) forbids one signature covering both (that's shared-signature/
  per-model-semantics = isinstance-with-extra-steps). *Challenge:* is the duplication (two `black_vol`-shaped
  methods) a smell? No — the *contracts* differ, and `black()` (the math) is shared; only the vol *lookup* is
  per-capability. Correct per Q3′.
- **`_swap_tenor` derives the tenor from the schedule** (`round((end−start)/365)` in years). *Challenge:* a
  rounding heuristic for the surface key. Fine for whole-year tenors (the only ones a flat surface distinguishes);
  when a gridded surface with fractional/odd tenors lands, the tenor should travel on the product, not be
  re-derived. Forward-linked to the gridded-surface slice.
- **Engine composes `float_leg_pv`/`rpv01`, not `_par_rate`.** Ratified (D4) — the calibrator's `_par_rate` is
  private; the engine reaches the *atoms* it's built from. The §3d oracle proves they agree.
- **Surface is still flat-minimal.** `SwaptionSurfaceKey(index, swap_tenor)` distinguishes tenors, but the
  `Surface` returns one vol at every (expiry, strike). The 2D smile + swaption cube earn their complexity with a
  calibrated-cube consumer (rule of two).

## 4. Smell + debt scan
- `verify.py` acyclic/fields/layers/provenance/debt all green. Field/arg discipline: `Swaption` 3 fields,
  `SwaptionSurfaceKey` 2, `swaption_vol` 4 args, `price_swaption` composes atoms (no god-function). No new
  suppressions (`debt` balanced).
- Exception-count (§3d): 0 concrete-type switches; one capability-protocol `isinstance`. `MarketSnapshot.surfaces`
  key type widened to `SurfaceKey | SwaptionSurfaceKey` — a union of key types over one shape, not a new field
  (doc 19 closed-shape × open-keys honoured).

## Drawdown (§4) — 19/793, tick 0 (PARTIAL cross)
`options/swaption.py`: the **European Black×annuity swaption price CROSSES** (ng `Swaption`+`price_swaption`).
Resident/deferred (forward-linked): `price_swaption_sabr_hw` (SABR/HW blend) → SABR (B4); greeks → C3;
cash-settled/IRR → first cash-settled consumer. Separate quarry files travel with their slices:
`swaption_vol_cube.py` → cube/vol-calibration; `bermudan_swaption*.py` → (A)-fork Bermudan (numerically priced);
`midcurve_swaption.py` → midcurve slice. Partial → tick 0.

## Deferred (named triggers)
Cash-settled/IRR swaptions · swaption vol cube + 2D smile · vol calibration/surface fitting · Bermudan swaption
((A)-fork re-open trigger) · SABR/HW swaption vols (B4) · midcurve swaptions · swaption greeks (C3).

## Named next checkpoint
**C2 slice 3 — vol calibration / surface stripping** (the first `SolveConfig` consumer beyond curves — a model
built by *solving*, not reading), or another Black product. Checkpoint at the first of ≤6 slices or the next
capability/calibration-family boundary.
