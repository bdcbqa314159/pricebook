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
| A.1 | `bootstrap.py` | ⚠️ | **2 (HW convexity wrong by 10–22× — but LATENT no current caller; float-leg conventions are no-ops in single-curve mode)** |
| A.2 | `global_solver.py` | ⚠️ | **1 HIGH (residual collision when two instruments share a maturity)** |
| A.3 | `multicurve_solver.py` | ⚠️ | **1 HIGH (PV_float skips the first accrual period; annuity includes it → systematic bias)** |
| A.4 | `ncurve_solver.py` | ⚠️ | 2 simplifications labelled as such (BasisSwap annuity ≈ τ·df(T); OISSwap schedule is a crude calendar-year stepper) |
| A.5 | `rfr_bootstrap.py` | ✅ | 0 (correctly pre-applies canonical `rfr_futures_convexity`; doesn't trigger A.1 B1) |
| A.6 | `bond_curve.py` | ✅ | 0 (correctly passes ICMA refs in `_price_bond`) |

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

**A.1 follow-up correction:** the inlined HW convexity *bug exists* but is currently **latent** — `rfr_bootstrap.py` pre-applies the canonical `rfr_futures_convexity` and then passes the resulting forward rates as FRAs (not futures) to `bootstrap()`. A grep for direct `bootstrap(futures=...)` callers in the library returns zero. So the bug only fires for **future code paths** that call `bootstrap()` directly with `futures=[...]` and non-zero HW params. Still HIGH-severity-in-shape but **MED in present impact**.

---

## A.2 — `curves/global_solver.py`

**Purpose:** `global_bootstrap` (single curve via global Newton) and `coupled_bootstrap` (OIS + projection jointly).

### Status: ⚠️ One HIGH bug

#### A.2 B1 — Residual collision silently drops constraints when instruments share a maturity  *[HIGH]*

**Location:** `global_solver.py:80-98` (single-curve `_residuals`), and the analogue inside `coupled_bootstrap`.

The residual vector is indexed by `pillar_idx[mat]`. When two instruments share a maturity — e.g. a 1Y deposit AND a 1Y swap, two swaps at the same node, or duplicate deposits — they write to the **same residual row**. Loop iterates deposits first, then swaps; the swap residual silently overwrites the deposit. The Newton solve now has `n` equations but only `n − k` independent constraints (`k` = collisions). The Jacobian's affected row drops the deposit's direct `df_i` term, so Newton converges to a curve that does **not reprice all input instruments**.

**Live repro** — deliberately inconsistent inputs to expose the silent overwrite:

```
deposits = [(2025-01-01, 0.05)]   # 1Y depo at 5%
swaps    = [(2025-01-01, 0.04)]   # 1Y swap at 4% — SAME maturity, DIFFERENT rate
global_bootstrap(2024-01-01, deposits, swaps)
  → zero_rate(2025-01-01) = 3.9607%
```

Only the swap survived. The deposit is silently dropped. A real-world scenario where this matters: USD curve with a 1Y depo *and* a 1Y OIS swap (both standard market quotes at the 1Y pillar). One of them is silently ignored.

**Fix shape:**
- Option A: detect duplicates up-front and `raise ValueError("Duplicate maturity at ...")`.
- Option B: build a residual vector indexed by *instrument number* (not pillar); add a pillar→row mapping; require the system to be square only after explicit dedup.

Option A is the conservative one-slice fix. Option B is the right design but more invasive.

---

## A.3 — `curves/multicurve_solver.py`

**Purpose:** `multicurve_newton` — solve OIS + projection jointly via damped Newton.

### Status: ⚠️ One HIGH bug

#### A.3 B1 — Projection swap PV_float skips the first accrual period; annuity does not  *[HIGH]*

**Location:** `multicurve_solver.py:159-176`.

```python
dates_up_to = [d for d in projection_pillar_dates if d <= inst['maturity']]
pv_float = 0.0
for j in range(1, len(dates_up_to)):
    d_start = dates_up_to[j - 1]
    d_end   = dates_up_to[j]
    ...
    pv_float += fwd_j * tau_j * ois.df(d_end)
annuity = _compute_annuity(ois, dates_up_to, day_count)
model_rate = pv_float / max(annuity, 1e-10) if annuity > 0 else 0
```

The loop starts at `j=1`, so the FIRST segment — from `reference_date` to `dates_up_to[0]` (the first pillar) — is silently skipped. But `_compute_annuity` initialises `prev = reference_date` and **does** include that first period. Result: `pv_float` has `N-1` segments while `annuity` has `N` segments. For any projection swap with `len(dates_up_to) ≥ 2`, the model rate is systematically biased: model_rate ≈ `(N-1)/N` × true_rate.

For a 2-pillar projection swap (single intermediate plus maturity), `pv_float` has 1 segment and `annuity` has 2 segments — error is ≈50% of the rate. For a 5-pillar projection (e.g. 1y+2y+3y+5y → 5y swap), error is ≈20%.

**Even worse for single-pillar cases:** `len(dates_up_to) == 1`, the forward loop does nothing (`range(1, 1)`), `pv_float == 0`, `model_rate == 0`. The error becomes `0 − inst['rate']` = `−inst['rate']`, which the solver tries to drive to zero by collapsing the DFs to nonsense values.

This is the bug that drove the existing test `test_paper_01_multicurve::TestMulticurveNewton::test_multicurve_solver_runs` to emit `RuntimeWarning: did not converge` (residual 2.86e-03) — the warning has been there the whole time, but it's the bug talking.

**Fix shape:** start the loop at `j=0` using `reference_date` as the initial `d_start`, OR prepend `reference_date` to `dates_up_to`. Either way, ensure `pv_float` and `annuity` cover the same period set.

---

## A.4 — `curves/ncurve_solver.py`

**Purpose:** Generalises dual-curve bootstrap to N curves solved simultaneously. Protocol-based pricers (`DepositPricer`, `OISSwapPricer`, `BasisSwapPricer`).

### Status: ⚠️ Two real approximations marked as "simplified" but used in production

#### A.4 B1 — `BasisSwapPricer` annuity is single-period approximation  *[MED, documented as "simplified"]*

**Location:** `ncurve_solver.py:141-152`.

```python
# At par: float_pay + spread×annuity = float_recv
# Error: (1-df_pay(T)) + spread×annuity - (1-df_recv(T))
df_pay  = pay.df(self.maturity)
df_recv = recv.df(self.maturity)
t = year_fraction(self.reference_date, self.maturity, self.dc)
annuity = t * disc.df(self.maturity)  # simplified
return (1 - df_pay) + self.basis_spread * annuity - (1 - df_recv)
```

For a 5y quarterly basis swap, the correct annuity is `Σ τ_i × df(t_i)` for ~20 cashflows. The simplified version `t × df(T) ≈ 5 × 0.82 = 4.1`. Correct ≈ `0.25 × 4.6 = 1.15`. **The simplified version overstates annuity by ~3-4×** → spread contribution → curve fit by the same factor.

The "simplified" comment is honest, but the function is exported and used. Any caller solving for basis spreads against this pricer gets systematically biased results.

#### A.4 B2 — `OISSwapPricer` schedule uses crude calendar-year stepper  *[MED]*

**Location:** `ncurve_solver.py:106-120`.

```python
for i in range(1, n + 1):
    t = i / self.frequency
    y = self.reference_date.year + int(t)
    m = self.reference_date.month
    d_i = date(y, m, min(self.reference_date.day, 28))
    if d_i > self.maturity:
        d_i = self.maturity
```

This builds coupon dates as `(ref.year + int(t), ref.month, min(ref.day, 28))` — a calendar-year stride. Misses business-day adjustment, EOM convention, day-count drift, and the actual market schedule generation (`generate_schedule` exists for this). The clamp to day-28 prevents Feb-31-style invalid dates but adds ad-hoc drift. Adequate for 1y rough OIS, wrong for non-trivial maturities and any locale that doesn't follow "same day-of-month" payment.

**Fix shape:** route through `pricebook.core.schedule.generate_schedule(ref, maturity, frequency, ...)` like every other swap in the library.

---

## A.5 — `curves/rfr_bootstrap.py`

### Status: ✅ Clean

`bootstrap_rfr` correctly pre-applies the canonical `rfr_futures_convexity` via `rfr_futures_to_forwards`, then passes FRAs (not futures) to the underlying `bootstrap` / `global_bootstrap`. This **prevents** A.1 B1 from firing in the production RFR-curve path. If A.1 B1 is fixed later, no change needed here.

---

## A.6 — `curves/bond_curve.py`

### Status: ✅ Clean — ICMA refs passed correctly

`_price_bond` at line 173-176 correctly passes `ref_start=t_start, ref_end=t_end, frequency=quote.frequency` to `year_fraction(...)` for ACT/ACT ICMA, so bond curves built from sovereign quotes get exact ICMA semantics. (This is the right pattern; matches the fix landed in `FixedLeg` for A.1 B1 Slice 3.)

A handful of internal methods (`_bootstrap_sequential`, `_bootstrap_global`, `_bootstrap_parametric`) — well-structured; couldn't find correctness issues on quick reads.

---

## Pass A — summary

6 modules audited; **3 confirmed bugs** (1 HIGH active, 1 HIGH active, 1 HIGH latent) + **2 production-quality approximations**.

| # | Module | Bug | Severity (impact) | Fix shape |
|---|---|---|---|---|
| A.1 B1 | `bootstrap` | HW convexity wrong | HIGH-shape, MED-impact (no current caller) | Replace inlined formula with `ir_futures.hw_convexity_adjustment` |
| A.1 B2 | `bootstrap` | Float-leg conventions no-ops | MED | Document / deprecate single-curve `float_*` params |
| A.2 B1 | `global_solver` | Residual collision silently drops constraint | **HIGH (active)** | Detect duplicates and raise |
| A.3 B1 | `multicurve_solver` | PV_float skips first period | **HIGH (active)** | Start loop at j=0 (anchor on reference_date) |
| A.4 B1 | `ncurve_solver` | BasisSwap annuity ≈ τ·df(T) | MED | Sum proper annuity |
| A.4 B2 | `ncurve_solver` | Crude OIS schedule stepper | MED | Route through `generate_schedule` |
| A.5 | `rfr_bootstrap` | — | — | clean |
| A.6 | `bond_curve` | — | — | clean |

Two HIGH-impact bugs are active today (`A.2 B1` and `A.3 B1`); the `multicurve_newton::did not converge` warning that the existing test suite emits is *the bug talking*. Fixes for those are next.

---

## Pass B — parametric curve forms

| # | Module | LoC | Status | Confirmed bugs |
|---|---|---:|---|---|
| B.1 | `nelson_siegel.py` | 150 | 📝 | 0 bugs; day-count drift ~0.3 bp at 10y when materialising as `DiscountCurve`; non-convex objective with discontinuous barrier (Nelder-Mead unfriendly) |
| B.2 | `smith_wilson.py` | 135 | ✅ | 0; clean EIOPA implementation. Same `date_from_year_fraction`/ACT-365F drift as B.1 when materialised. |
| B.3 | `seasonal_curve.py` | 188 | 📝 | 0; deterministic overlay; `vars(self)` to_dict footgun on `SeasonalPattern` (not yet swept — outside L0) |
| B.4 | `inflation_curve.py` | 90 | 📝 | 0; `relativedelta(years=int(t), months=...)` introduces a sub-day drift between input `(t, r)` and curve-stored `(d, df)` (acceptable approximation) |
| B.5 | `ndf_implied.py` | 267 | 📝 | 0; CIP formula correct; **silent skip on `df_em > 2.0`** at line 154 (data-quality concern — should log) |

### Pass B detail

#### B.1 — `nelson_siegel.py`

- **`nelson_siegel_yield` / `svensson_yield`** match the cited 1987/1994 references. Limit at `t=0` (returns `β₀ + β₁`) and sign conventions are correct.
- **`_ns_factor2(t, tau)`** at line 31-36 — short-circuits at `x < 1e-10` but for `1e-10 < x < ~1e-6` there's a potential catastrophic-cancellation band (the formula computes `(1-exp(-x))/x - exp(-x)` which is `O(x)` for small `x`). Not a bug today (no caller probes that region) but worth noting.
- **Calibration objective** uses `if tau <= 0.01: return 1e10` — a step discontinuity that Nelder-Mead can't gracefully navigate (the simplex straddles the barrier and oscillates). Robustness concern, not a correctness bug.
- **Underdetermination guard absent**: calibrating 4-param NS against ≤ 3 pillars or 6-param Svensson against ≤ 5 pillars silently succeeds with whatever the optimiser converges to. Should at minimum warn.
- **Day-count drift**: `ns_discount_curve(...)` builds dates via `date_from_year_fraction(ref, t)` (365.25 days/yr) but the resulting `DiscountCurve` uses ACT/365F by default. Querying `curve.zero_rate(d)` at the constructed date returns a value drifting by ~0.26 bp at t=10y for a typical NS curve (live measured). Smaller than the 4 bp claimed in MODULE_HEALTH for that specific case but still real.

#### B.2 — `smith_wilson.py`

- **`_wilson_function(t, u, alpha, ufr)`** matches the standard EIOPA technical specification (Solvency II QIS5 onwards).
- **`smith_wilson_calibrate`** solves the linear `W @ ζ = market_df - exp(-ufr·t)` — well-posed when maturities are distinct.
- **`smith_wilson_forward`** uses a 1-day finite difference for instantaneous forward — adequate; could be analytical from `dW/dt` if needed.
- Same `date_from_year_fraction` drift as NS when materialised; same docstring fix applies.

#### B.3 — `seasonal_curve.py`

- Deterministic seasonal overlay (year-end / quarter-end / month-end premia). Reasonable approach. Currency-specific pre-built patterns (`USD_SEASONAL`, `EUR_SEASONAL`, `GBP_SEASONAL`) have the right ordering (USD year-end > GBP > EUR).
- `SeasonalPattern.to_dict` returns `vars(self)` — footgun pattern from L0 audit, not yet swept in `curves/*` (outside L0 scope).

#### B.4 — `inflation_curve.py`

- Joint real+nominal builder. Uses `relativedelta(years=int(t), months=int((t%1)*12))` to construct dates from year-fraction tenors — fine for whole/half years; can introduce sub-day drift between the input `t` and the curve-stored interpretation. The DF is computed from `exp(-r * t)` using the input `t` (not the constructed date's recomputed year fraction), so there's the same flavour of drift as B.1/B.2 but smaller.
- BEI calculation `nominal_rate - real_rate` at line 84 is standard.

#### B.5 — `ndf_implied.py`

- **CIP relationship correct**: `df_em = df_base × Spot / NDF` with Spot quoted base/em (USD/CNY style). Matches Della Corte-Sarno-Tsiakas.
- **Silent skip on `df_em > 2.0`** at line 154: implies negative EM rate ≤ −69% — clearly bad data. Currently `continue` silently drops the pillar. Should at least emit a `RuntimeWarning` so the caller knows data was filtered.
- **`df_em` clamp at 1e-10**: silently. Same concern.

### Pass B — summary

5 modules audited. **0 confirmed bugs.** Recurring theme: `date_from_year_fraction` (365.25) ↔ DiscountCurve's ACT/365F day-count mismatch introduces ~0.2-0.3 bp drift at 10y when materialising parametric curves. Below the 1-bp resolution that practical curve work usually demands, but worth a coordinated fix when the breaking schema bump (LD bundle) lands — the parametric builders should accept a `day_count` parameter and use it consistently.

The five LOW-priority items (silent skips in ndf_implied, vars(self) in seasonal, day-count drift across all 4 parametric builders) are all stylistically related. A single "parametric-curve hygiene" slice could close them all.

---

*(audit continues — Pass C risk / bumping / scenarios)*
