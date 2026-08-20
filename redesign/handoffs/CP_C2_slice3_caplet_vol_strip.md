# Checkpoint — C2 slice 3 (caplet-vol stripping — the first solved surface)

**Version:** v0.95.0 · **Slice:** `slice/09-caplet-vol-strip` · **Baseline:** v0.94.0 (Black caplet + swaption).

First calibrated **volatility** object, and **`SolveConfig`'s rule-of-two second consumer** (the first
beyond curves). The (A) fork is exercised on a non-curve target. New calibration family → checkpoint.

## 1. `SolveConfig` reused without strain — the rule-of-two case for the config
The strip reuses `SolveConfig` (SEQUENTIAL/Brent) verbatim — it is now consumed by two families (curve
bootstrap + vol strip), which is exactly the rule-of-two that justified the config being a shared type
rather than a curve-only one. **No strain:** the strip needs only `tolerance` (convergence grading) and
the SEQUENTIAL/Brent method; the vol solve is 1-D per pillar, structurally the curve bootstrap in a new
family. A **separate per-family entry** (`strip_caplet_vols`, not folded into `calibrate()`) — a `Surface`
is a different target than a `CurveSet`, and vols are downstream of a finished curve. Config shared,
**entry point not** (folding one `calibrate()` over both is speculative until a joint curve+vol consumer).

## 2. (A) fork on a non-curve target — confirmed
The cap/caplet target is closed-form-priceable (`black()`), so the strip calibrates by repricing the
targets through the model's OWN closed-form capability — **fork (A), no `StateProcess`/`Payoff`**. This is
the first time the (A) fork is exercised outside curves; it holds (the target is analytic).

## 3. Oracle-quality audit
| Oracle | Class | Result |
|---|---|---|
| Flat round-trip: strip a flat cap → every caplet vol == the flat vol | **closed-form anchor** | <1e-12 (exact) |
| §3d backstop: caplet priced through the L4 engine off the stripped surface == independent Black | closed-form cross-engine | <1e-12 |
| Reprice-to-quote: stripped surface reprices each cap to its quote | self-consistency | <1e-9 |
| Invariant 4: an infeasible marginal (target outside the vol-attainable range) → `CalibrationFailure` | value-not-raise | asserted |

The flat round-trip is the load-bearing closed-form anchor: with all quotes at one flat vol, the marginal
telescopes so each stripped caplet vol must equal the flat vol *exactly* — a stronger check than
reprice-to-quote (which telescopes for any self-consistent strip). The §3d backstop proves the strip and
the L4 engine share `black`/`forward`/`df` — the strip does NOT import the L4 engine (L3 ⊥ L4) nor a
private caplet formula; it composes the same atoms.

## 4. Challenge-me
- **"variance-linear" implemented as linear-in-σ² over expiry** (not linear-in-total-variance σ²·t). The
  ratified spec said "variance-linear"; `Surface` has 3 fields and **no valuation origin**, so total-variance
  (which needs `t` from valuation) isn't representable self-containedly. Linear-in-σ² over expiry ordinals is
  origin-free, **exact at pillars and for flat** (the only cases the oracles exercise), and monotone. *This is
  a faithful reading, surfaced for correction:* if Cowork meant total-variance σ²·t, the fix is to carry the
  valuation date on `Surface` (→4 fields) and interpolate σ²·t — a 1-line change, deferrable to the first
  off-pillar caplet consumer (between-pillar interpolation is not oracle-tested this slice).
- **Each `CapQuote` maps to one caplet pillar** (maturity = fixing date, accrual = one index period). The
  quarry handled multi-caplet-per-maturity caps awkwardly; the ng model is one-caplet-per-quote, clean and
  sequential. A denser cap grid (quarterly caplets under annual quotes) is a schedule the strip would build
  from `frequency` — deferred to that consumer (the spec has room for a `frequency` field ≤5).
- **Surface grow was a refactor under green** — flat = 1-pillar degenerate (`Surface.flat`); slices 1–2 flat
  consumers migrated and stayed green (no new red).

## 5. Smell + debt scan
- `verify.py` acyclic/fields/layers/provenance/debt all green. **acyclic** confirms the L3 strip composes
  L1 (`forward`, `df`, `CapQuote`) + L3 (`black`) atoms with **no L4 import** (L3 ⊥ L4). Field/arg discipline:
  `VolCalibrationSpec` 4 fields, `CapQuote` 3, `Surface` 3, `_Leg` 4 — all ≤5. No new suppressions.
- Exception-count (§3d): 0 type-switches; the strip is one composition path.

## Drawdown (§4) — 19/793, tick 0 (PARTIAL crosses)
`options/capfloor.py`: the caplet-vol *term-structure strip* crosses (ng `strip_caplet_vols`);
`calibrate_capfloor_sabr` + smile/cube generation resident → SABR/smile slice; `CapFloor` container + greeks
already deferred (slice 1). `options/vol_calibration.py`: a multi-asset surface calibrator — only the IR
caplet-term-structure portion of `calibrate_ir_surface` is superseded; `calibrate_fx/equity/commodity_surface`,
`CalibratedSABRNode`, `CalibratedVolSurface` travel with their asset topics + the SABR slice. Both partial → tick 0.

## Deferred (named triggers)
Strike-smile / 2nd surface axis (its consumer) · swaption-cube fitting (its slice) · SABR/parametric vols (B4) ·
SIMULTANEOUS vol solve (2nd consumer) · caplet-stripping convention variants (spot-vs-forward vol) · vol-surface
serialisation (persistence) · total-variance (σ²·t) term interpolation (first off-pillar caplet consumer) ·
denser cap-caplet schedule via `frequency` (its consumer).

## Named next checkpoint
**C2 slice 4** — swaption-vol surface / cube fitting (the swaption analogue, 2D), or SABR parametric vols (B4),
or close C2 and open C3 (risk). Checkpoint at the first of ≤6 slices or the next capability/family boundary.
