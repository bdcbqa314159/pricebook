# L1 — `pricebook.curves` Layer Audit

**Started:** 2026-06-11 (immediately after L0 audit closure)
**Scope:** `pricebook.curves.*` — 33 modules.
**Method:** Fresh read per module; cross-reference existing `MODULE_HEALTH.md` findings; document status, real bugs, doc/test gaps, slicing proposals.
**Prerequisites:** L0 audit complete (`AUDIT_L0_CORE.md`); legacy-debt ledger active (LD.1–11).

## Pass plan

| Pass | Scope | Modules |
|---|---|---|
| **A — calibration / bootstrap core** | The pricing-critical curve construction logic | `bootstrap`, `global_solver`, `multicurve_solver`, `ncurve_solver`, `rfr_bootstrap`, `bond_curve` |
| B — parametric curve forms | Closed-form curve families | `nelson_siegel`, `smith_wilson`, `seasonal_curve`, `inflation_curve`, `ndf_implied` |
| C — risk / bumping / scenarios | Sensitivity / scenario tooling | `curve_bumper`, `curve_risk`, `curve_scenarios`, `key_rate_risk` |
| D — builders / engines | Higher-level curve orchestration | `curve_builder`, `em_curve_builder`, `curve_blending`, `curve_advanced`, `curve_engine`, `synthetic_market_data`, `curve_storage`, `curve_diffusion` |
| E — internal numerics | Curve-side numerical primitives | `linalg`, `quadrature`, `sparse`, `sparse_grids` |
| F — AAD subsystem | Forward-mode autodiff for curves | `aad`, `aad_calibration`, `aad_curves`, `aad_interp`, `aad_pricing` |

---

## Pass A — calibration / bootstrap core

| # | Module | Status | Confirmed bugs |
|---|---|---|---|
| A.1 | `bootstrap.py` | ⚠️ | **2 (HW convexity wrong by 10–22×; float-leg conventions are no-ops)** |
| A.2 | `global_solver.py` | ❓ | |
| A.3 | `multicurve_solver.py` | ❓ | |
| A.4 | `ncurve_solver.py` | ❓ | |
| A.5 | `rfr_bootstrap.py` | ❓ | |
| A.6 | `bond_curve.py` | ❓ | |

---

## A.1 — `curves/bootstrap.py`

**Purpose:** sequential single-curve bootstrap from deposits, FRAs, futures, and swap par rates. Now also produces a `CalibrationResult` (G1 P1 Slice 5) and accepts an optional `market_snapshot` for audit linkage (G1 P2 Slice 2).

**Internal deps:** `core.day_count`, `core.discount_curve`, `core.interpolation`, `core.schedule`, `core.solvers` (brentq), `core.calendar`, `calibration`. Plus indirect: `pricebook.market_data` (TYPE_CHECKING).

**Size:** ~560 lines (was 252 pre-G1).

**Tests:** `test_bootstrap.py`, `test_curve_bootstrap_snapshot.py`, others.

### Status: ⚠️ Two HIGH-severity bugs verified live

### Confirmed bugs

#### A.1 B1 — Inlined Hull-White convexity adjustment is mathematically wrong  *[HIGH]*

**Location:** `bootstrap.py:113-122` (inside the `futures` loop within `bootstrap(...)`).

```python
def _B(s, t):
    return (1 - _math.exp(-hw_convexity_a * (t - s))) / hw_convexity_a
conv_adj = 0.5 * hw_convexity_sigma**2 * _B(t_start, t_end) * (
    _B(0, t_end) - _B(0, t_start)
)
```

That formula is `CA = ½σ² · B(T1,T2) · [B(0,T2) − B(0,T1)]`, which does not match the standard Hull-White convexity result. The library *already has* the canonical implementation in `pricebook.fixed_income.ir_futures.hw_convexity_adjustment`:

```python
# Canonical (correct):
CA = ½σ² · B(T1,T2) · [B(T1,T2) · G(t,T1) + B(t,T1) · (T2 − T1)]
where  B(s,t) = (1 − e^{−a(t−s)})/a,
       G(t,T) = (1 − e^{−2a(T−t)})/(2a)
```

**Live numeric comparison** (a=0.03, σ=0.01, t=0):

| Future | Canonical CA | Bootstrap-inlined CA | Ratio |
|---|---|---|---|
| 5y → 5.25y (3M ED) | **0.2785 bp** | 0.0267 bp | 10.4× under-stated |
| 10y → 10.25y (3M ED) | **0.5022 bp** | 0.0230 bp | 21.9× under-stated |

The error gets *worse* with maturity — exactly the regime where convexity matters most for futures-stub curve building. The inlined formula essentially treats convexity as a tiny perturbation when it should be the dominant correction for long-dated futures.

**Downstream impact:** any USD/EUR curve built using IR futures (the standard short-end stub between deposits and swaps) under-corrects for convexity. Implied DFs at futures pillars are wrong by 0.2–0.5 bp at typical vol levels, propagating into swap/swaption pricing.

**Fix shape:** delete the inlined formula; import and call `from pricebook.fixed_income.ir_futures import hw_convexity_adjustment`. (Single-line fix once the import is in place; mostly a refactoring of the bootstrap signature so HW params travel via a single object rather than two scalars.) Risk: the canonical function uses `t` (current time) explicitly while the inlined one assumed `t=0`. Migration needs to pass `t=0` explicitly or rebase to a different anchor.

#### A.1 B2 — `float_day_count` and `float_frequency` are silent no-ops in single-curve mode  *[MEDIUM]*

**Location:** `bootstrap.py:174-181` (inside the swap-solver `objective` function).

```python
pv_float = 0.0
for i in range(1, len(_flsched)):
    d1, d2 = _flsched[i - 1], _flsched[i]
    df1 = trial_curve.df(d1)
    df2 = trial_curve.df(d2)
    yf = year_fraction(d1, d2, float_day_count)
    fwd = (df1 - df2) / (yf * df2)
    pv_float += fwd * yf * df2
```

Algebraically:
```
fwd * yf * df2 = ((df1 − df2) / (yf · df2)) * yf * df2 = df1 − df2
```

So the loop sum telescopes to `df(start) − df(end)`. The intermediate schedule dates cancel and `float_day_count` cancels out of every term. The user-supplied `float_day_count` and `float_frequency` (which determines the schedule grid) have **no effect on the computed PV** — but the API advertises them as if they did. Tests built around "swap with 6M EURIBOR float leg vs 3M EURIBOR float leg" would silently produce identical PVs even when the underlying conventions actually differ.

**Why it doesn't catch the standard single-curve case:** for a single-curve world (legacy pre-OIS-discounting), the float leg in fact reduces to `1 − df(T)` and the convention parameters genuinely don't matter at this level. But the API exposes the parameters as if they did, which is misleading. The dual-curve variant (`global_solver` / `multicurve_solver`) doesn't telescope because the projection curve and the OIS curve differ, so it correctly uses the conventions.

**Fix shape:** two options.
- *Documentation:* explicitly document in the `bootstrap()` docstring that in pure single-curve mode the float-leg conventions are nominal (telescoping makes them no-ops). This is the cheap fix.
- *Refactor:* deprecate the single-curve `bootstrap()` entry point's `float_*` params; route callers to `global_bootstrap()` or `multicurve_newton()` for proper conventions. Larger surface change.

### Other concerns (not bugs)

- **No ordering check on `fras` / `futures` input lists.** Mismatched ordering can produce inconsistent bootstrap progress. Validation is one-liner.
- **`brentq` bracket `[1e-6, 3.0]`** is wide enough for negative rates down to ~−3% but the swap-solver might fail for steep deeply-negative-rate curves (e.g. JPY pre-2025).
- **TOY (turn-of-year) spread** is applied only when `start_date.year != end_date.year`. That catches the typical year-end stub but misses cases where the year-end falls inside the period without crossing it (rare in practice, ~impossible for futures).

### Test coverage

Existing tests cover deposits + swaps + the snapshot linkage (from G1 P2 Slice 2). Missing:
- HW-convexity test comparing bootstrap output to `ir_futures.hw_convexity_adjustment` directly (would catch A.1 B1).
- Float-leg-convention-changes-don't-affect-PV xfail test (would document A.1 B2).
- FRA ordering validation tests.

### Slicing items

1. **A.1 B1 fix** — replace inlined convexity with `from pricebook.fixed_income.ir_futures import hw_convexity_adjustment`. Characterise via xfail test for 5y and 10y futures pre-fix; assert exact match post-fix. **Estimated 1 slice + characterisation slice.**
2. **A.1 B2 fix** — for the MED interpretation: clarify docstring. For the HIGH interpretation: deprecate the params with `DeprecationWarning`. Defer the larger refactor to Gate 2.
3. **FRA/futures ordering validation** — single defensive slice.

---

*(audit continues — next module: A.2 `global_solver.py`)*
