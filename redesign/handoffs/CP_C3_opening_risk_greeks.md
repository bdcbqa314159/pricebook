# Checkpoint — C2 close-out + C3 opening (risk layer, L5)

**Version:** v0.96.0 · **Slice:** `slice/10-risk-greeks` · **Baseline:** v0.95.0 (C2: caplet, swaption, vol strip).

This slice **opens the risk layer (L5)** — a new layer + a new protocol (`Priceable`) — and with it
**closes the C2 cluster**. Two review inputs below: a C2 cluster close-out, then the C3-opening review.

---

## Part 1 — C2 cluster close-out (models & calibration)

C2 delivered: Black caplet (slice 1), Black swaption (slice 2), caplet-vol stripping (slice 3).
- **Capability model proven under load** — one `BlackModel` opt-in satisfies `CalibratedModel` + `BlackVol`
  + `SwaptionVol`; each capability ships with its Q3′(a) semantic contract (T-forward vs annuity measure).
  Additive throughout — no ratified type's meaning changed across the three slices.
- **`SolveConfig` proven beyond curves** — the caplet strip is its rule-of-two second consumer (a per-family
  entry reusing the config, not folded into `calibrate()`).
- **(A) fork exercised on a non-curve target** — cap/caplet stripping is closed-form-priceable → no
  `StateProcess`/`Payoff`. Bermudan remains the (A)-fork re-open trigger.
- **Oracle quality:** every C2 slice has a closed-form anchor (Black closed form + put-call/payer-receiver
  parity + vol→0 intrinsic; flat round-trip for the strip) plus a §3d cross-engine identity. All at 1e-12.
- **Carried out of C2 (deferred, named triggers):** Bachelier/normal vol · swaption cube + 2D smile ·
  SABR/HW (B4) · Bermudan · midcurve · analytic option greeks · total-variance (σ²·t) term interpolation
  (the slice-3 "variance-linear = linear-in-σ²" flag — 1-line change at its first off-pillar consumer).

## Part 2 — C3 opening review (risk, L5)

### 1. The design under load — key-blind core, no ratified-type meaning change
Risk lands at L5, depends DOWN on the engine + the `Priceable` protocol, and **never inspects an
instrument's type** (the engine registry dispatches; `Priceable` is `snapshot → PV`). The FD core
`central_diff` is **key-type-blind** — `bump.apply` is the only shape-aware call; the curve-vs-surface
split lives in `CurveBump`/`SurfaceBump` strategies, **not an isinstance/if-elif in the core** (§1 law;
§3d no type-switch). Opening L5 changed **no** ratified L1/L4 type's meaning: `bumped()` on curve/surface
and `with_curve` are **additive immutable transforms** (invariant 3 — base never mutated); the engine and
snapshot are untouched. `acyclic` confirms nothing at L≤4 imports L5.

### 2. Oracle-quality audit
| Oracle | Class | Result |
|---|---|---|
| DV01: `ir_delta` (curve bump, central FD) vs analytic `−Σ N·τ·(F−K)·t·DF` | **closed-form** cross-check | <1e-6 |
| Vega: `vol_vega` (surface bump, central FD) vs `df·N·τ·black_vega` (closed form) | **closed-form** cross-check | <1e-6 |
| Generic-over-keys: `ir_delta` and `vol_vega` both == `central_diff` with their `Bump` | structural (one core) | exact |
| isinstance-free: risk/ has no isinstance except failure-as-value on `PricingFailure` | source-structural | asserted |
| Immutability + failure-as-value: base dfs unchanged after a bump; a failed reprice → `PricingFailure` | invariant 3 / 4 | asserted |

Both greeks are FD-vs-analytic closed-form checks — the strongest kind. The generic-over-keys + no-key-switch
oracles are the design guards: they prove the "generic engine over open keys" (CLAUDE.md §1 / doc 19 §2) is
real, not a per-asset ladder in disguise.

### 3. Challenge-me
- **`ir_delta`/`vol_vega` return the RAW derivative** (`∂PV/∂r`, `∂PV/∂σ`), not pre-scaled DV01/vega-per-point.
  The ratified D-line parenthesised "(DV01)"/"(1 vol point)", but the ratified **oracle** compares to the raw
  analytic `−Σcf·t·DF` / `df·N·τ·black_vega`, so the raw derivative is what the oracle demands. DV01 = `ir_delta
  · 1bp`, vega-per-point = `vol_vega · 0.01` — documented, a caller-side scaling. *Flagged:* if a caller wants
  the pre-scaled value, it's a 1-line wrapper; surfaced in case Cowork wants the scaled convention as default.
- **Sticky-model market delta** (bump the built curves/surfaces, re-wrap the model) — NOT recalibration. The
  par/recalibration delta (bump quotes, re-solve) is a distinct greek, deferred to its slice (doc 18 C3).
- **`bumped` on the `CurveHandle` capability** (not just concrete `DiscountCurve`) — risk depends on the
  capability, not the concrete type (§1). One method added to the protocol; every impl (only `DiscountCurve`)
  satisfies it.
- **`Bump` is the extension point for new key kinds** — the third key (FX-spot `ScalarBump`) is a new strategy
  at its first consumer, **not** a core edit. This is the payoff of the key-blind core.

### 4. Smell + debt scan
- `verify.py` acyclic/fields/layers/provenance/debt all green. **acyclic** is the load-bearing one here — it
  proves L5 is a leaf (no L≤4 import). Field discipline: `CurveBump`/`SurfaceBump` 1 field each; `Priceable`
  closure captures 2. No new suppressions.
- Exception-count (§3d): the core has **zero** type-switches; the only `isinstance` is failure-as-value on
  `PricingFailure` (asserted by a source-structural test).

### Drawdown (§4) — 19/793, tick 0 (PARTIAL crosses)
The generic bump-and-reprice greek *mechanism* now exists (discharging the accrued "greeks exist at L5"
→C3 forward-links — any priceable gets DV01/vega free). Every quarry risk file retains breadth beyond ng's
parallel first-order pair: `risk/greeks.py` (gamma/theta), `curves/key_rate_risk.py` (key-rate/bucketed),
`curves/curve_risk.py` (Jacobian/rolldown), `risk/pathwise_greeks.py` (AAD), `curves/curve_bumper.py` (pillar
bumps), `core/greeks.py` (`Greeks` aggregate), `models/black76.py` (analytic delta/gamma/theta). All partial → tick 0.

### Deferred (named triggers)
Recalibration / par delta (bump quotes + re-solve) · key-rate / bucketed DV01 (its consumer) · analytic/AAD
greeks (AAD topic) · gamma / 2nd-order / cross-greeks (their consumer) · scalar (FX-spot) delta — 3rd key kind,
`ScalarBump` (first FX-delta consumer) · greek-result aggregate (`Greeks`, portfolio consumer) · XVA/RWA ·
L6 Trade/Book/portfolio aggregation (the rest of C3).

## Named next checkpoint
**C3 continues** — L6 Trade/Book/portfolio + risk aggregation (the imperative shell), or key-rate/bucketed
DV01, or recalibration greeks. Checkpoint at the first of ≤6 slices or the next layer/family boundary.
