# pricebook_ng — audit findings (third-party review)

Independent audit of `pricebook_ng` (v0.92.0) by an external reviewer. **No changes were made to this repo.** This is a handoff document: pass to Cowork, which relays actions to the agent working on the original repo. Each item lists location, a concrete reproduction, severity, and a suggested fix — the implementing agent decides the final approach.

Method: full source read (L0 foundation → L1 market → L2 products → L3 calibration/models → L4 engine) plus two adversarial passes (numerical + correctness). Code was **not executed** (no `pricebook_ng` install in the review env); findings are source-level, with hand-calculation where noted. The numerical core is sound — exposure is concentrated at input/range boundaries.

Fix order: **#1 → #3 first** (they crash/hang on realistic input); the rest mispriced or degrade quietly.

---

## Confirmed bugs

### 1 — [HIGH] Negative rates crash the default calibration path
- **Where:** `calibration/calibrate.py` — `_bootstrap` and `_bootstrap_xccy`, both `brent(residual, 1e-6, 1.0)`.
- **Two defects in one bracket:**
  1. A negative-rate pillar has DF > 1 (`deposit_df(-0.005, τ=1) = 1.00502`), outside `[1e-6, 1.0]` — the bracket bakes in a positive-rates assumption. EUR/CHF/JPY OIS (2015–2022; JPY still) cannot be built sequentially.
  2. `brentq` then raises `ValueError: f(a) and f(b) must have different signs`, uncaught by `_calibrate_sequential`/`calibrate` — so it escapes as an exception, violating the module's stated "failure is a value, never raised (invariant 4)".
- **Repro:** single-currency spec, one `DepositQuote(rate=-0.005)`, default `SEQUENTIAL`. `residual(1e-6) ≈ -1.005` and `residual(1.0) ≈ -0.005` — same sign.
- **Scope note:** the `SIMULTANEOUS` path (via `root_nd`, which catches) is unaffected; the **default** `SEQUENTIAL` path is the one that breaks.
- **Suggested fix:** widen the upper bound (`hi=5.0`+, or expand-until-sign-change) and wrap the per-pillar solve so a non-bracketing `ValueError` returns `CalibrationFailure` (mirror `root_nd`'s catch set: `ValueError, FloatingPointError, ZeroDivisionError`).

### 2 — [MED] Hagan–West DF silently extrapolates past the last pillar
- **Where:** `market/curve.py` `df()` HAGAN_WEST branch + `foundation/hagan_west.py` `MonotoneConvex.integral`.
- **Defect:** for `t > times[-1]`, `integral` sums every interval's full contribution and drops the tail, so `df(t) = df[-1]` (flat, zero forward). This contradicts the docstring's promise that a date past the last pillar RAISEs (log-linear does), and disagrees with `value()`, which flat-extrapolates the *forward* (nonzero).
- **Repro:** query a HW-interpolated curve at any `t > times[-1]` → wrong DF, no raise.
- **Suggested fix:** add the past-last-pillar guard to the HW branch (raise), or make `integral` include the `averages[-1]·(t − knots[-1])` tail so `value`/`integral` agree — then pick raise-vs-extrapolate consistently with the log-linear branch.

### 3 — [MED] `_act_act_icma` infinite loop / silently-wrong on bad frequency
- **Where:** `foundation/day_count.py` `_act_act_icma` (guard only rejects `frequency <= 0`).
- **Defect:** `frequency > 12` ⇒ `period_months = 12 // freq = 0`, `_icma_add_months(d, 0)` is the identity, both normalization `while` loops spin forever (hard hang). `freq ∈ {5,7,8,9,10,11}` (doesn't divide 12) ⇒ misaligned notional grid ⇒ terminating but silently wrong DCF.
- **Repro:** `year_fraction(..., ACT_ACT_ICMA, coupon_period=CouponPeriod(..., frequency=13))` hangs.
- **Suggested fix:** tighten the guard to `1 <= freq <= 12 and 12 % freq == 0`, raising otherwise.

### 4 — [MED] `RegularPeriod` with coincident/reversed anchors builds a zero-length period
- **Where:** `foundation/schedule.py` `_unadjusted` RegularPeriod branch.
- **Defect:** `first_regular_date == last_regular_date` yields `grid = [X, X]`; a reversed pair yields a backwards span. The resulting `SchedulePeriod` has `accrual_start == accrual_end`, which raises downstream in `Accrual.__post_init__` ("accrual must be ordered").
- **Repro:** `ScheduleTerms(..., stub=RegularPeriod(X, X))` with `start < X < end`.
- **Suggested fix:** validate `first_reg < last_reg` up front, or drop a trailing duplicate knot.

---

## Robustness / lower severity

### 5 — [LOW] `RegularPeriod()` default anchors don't flag a short stub
- **Where:** `foundation/schedule.py` RegularPeriod branch.
- **Defect:** with a non-dividing tenor, the short final period gets `is_stub=False`, so a leg builder reading `is_stub` picks the wrong ICMA reference period. Downstream mispricing suspected but not confirmed (leg-builder consumer not located).
- **Confirm with:** a schedule test using `RegularPeriod()` + a non-dividing tenor, asserting `periods[-1].is_stub`.

### 6 — [LOW] `WeekendSchedule.on` assumes sorted transitions
- **Where:** `foundation/calendars.py` `WeekendSchedule.on` (constructor doesn't enforce ordering).
- **Defect:** takes the last transition in list order with `year >= since`; an out-of-order tuple silently returns the wrong weekend. `((2013,FRI_SAT),(2000,SAT_SUN))`, `on(2015)` → SAT_SUN (should be FRI_SAT).
- **Suggested fix:** sort transitions in the constructor, or validate ascending order.

### 7 — [LOW] Convergence gate decoupled from the configured tolerance
- **Where:** `calibration/calibrate.py` — `_calibrate_sequential` and the xccy path grade convergence with a literal `1e-10`, not `spec.solve.tolerance`.
- **Suggested fix:** derive the gate from `spec.solve.tolerance`.

### 8 — [LOW, already-documented deferral] HW reconstruction rebuilt per `df()` call
- **Where:** `market/curve.py` `_forward_reconstruction`.
- **Note:** O(n) reconstruction inside every lookup ⇒ O(n²) full-curve revaluation; hits the non-local HW curve hardest. Flagged in-code as a Topic-1 caching deferral — listed for completeness, not urgent.

---

## Verified clean (checked, no action)

- **Hagan–West** — all four regions hand-differentiated/integrated: `g(0)=g0`, `g(1)=g1`, `∫₀¹g=0`, `_region_integral` is the exact antiderivative, the `(g0,g1)`-plane partition has no gap/overlap, region (iv) never divides by zero.
- **Day counts** — 30U/360 Feb ordering, 30E/360-ISDA termination rule, ICMA long-coupon grid (UST semi coupon = exactly 2.0000), ACT/365L annual-vs-frequent, ACT/ACT-AFB leap-to-leap.
- **`accrued_rate`** — COMPOUNDED normalization, EXPONENTIAL BUS/252 annualization, observation-shift-vs-lookback day-count consistency, lockout freezing; empty-window / zip-length / divide-by-zero guards.
- **Calendars** — FRI_SAT substitution, `christmas_boxing` collision, furikae, year-boundary spill, `add_business_days(d, 0)` guard.
- **Serialization** — Money/Quantity/Cashflow/Leg/Accrual round-trips; JointCalendar "A+B" identity split.

---

*Reviewer stance: third-party expert. This document is advisory; implementation and final design choices belong to the owning repo's agent.*
