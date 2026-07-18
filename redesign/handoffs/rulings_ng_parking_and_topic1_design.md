# Cowork → Build ruling — ng parking + Topic-1 design corrections from the quarry read

Two outcomes of the Topic-1 scoping read (`parked/topic-01-yield-curve/MANIFEST.md`).

---

## 1. ng parking — ng becomes a second quarry

**Ruled.** The out-of-topic ng modules move to **`ng_parked/`**, with their tests preserved as the
**re-base oracles** for when their topic opens.

**Why (this reverses my earlier degenerate-config advice):** keeping ~30 out-of-topic modules live
*couples the Topic-1 rebuild to re-basing all of them in every slice*. Each time `CurveSet` replaces
the snapshot's curve fields, forty modules must move with it. That is precisely the "extra noise."
Parked, they are proven, oracle-gated material belonging to blocks that are not open yet — a
**second source pool** alongside the quarry.

**Stays active (~20, Topic 1):**
`foundation/`: time · calendar · schedule · money · cashflow · numerical_config · results · solvers · distributions
`market/`: discount_curve · snapshot · keys · `models/discounting_model`
`products/`: deposit · fra · ois · swap · fixed_rate_bond · leg · fixed_cashflow
`engine/`: discounting · swap · fra · `risk/`: priceable · greeks · `calibration/discount_curve`

**Parks to `ng_parked/` (~30):**
`foundation/black` · `models/{hull_white, credit_model}` · `market/survival_curve` ·
`products/{cds, fx_forward, fx_option, equity_option, commodity_option, inflation, swaption}` ·
`engine/{cds, fx_forward, fx_option, spot_option, swaption, swaption_mc}` ·
`risk/{credit_greeks, exposure, saccr}` · `shell/{booking, xva_report}` ·
`calibration/{hull_white, survival_curve}`

**Rules:** `git mv` (history preserved) · tests move with their modules and are **excluded from the
active suite but retained** · a `ng_parked/MANIFEST.md` records each module, its topic, and its
re-base oracle · un-parking is a `git mv` back plus a re-base slice.
This also disposes of `black.py` cleanly — it parks with the options material; no move needed now.

---

## 2. Design corrections forced by the quarry read

Three are flaws we would otherwise have inherited. **These amend artifact #14 (Topic-1 object model).**

### C1 — Calendars keyed by identity, not currency
The quarry registers 37 calendars **by ISO currency code**; its own comment admits this cannot
express a USD+GBP joint calendar or NY-vs-SIFMA. **ng: key calendars by calendar identity**
(`TARGET`, `LONDON`, `NEW_YORK_SIFMA`, …), with **currency → calendar as a lookup**, never an
identity. `JointCalendar` composes identities.

### C2 — Schedules must return adjusted AND unadjusted dates
`generate_schedule` returns one flat list, adjusted in place, so **unadjusted accrual dates are
unrecoverable** — but accrual uses unadjusted dates while payment uses adjusted ones. **ng: the
schedule result carries both** (per period: unadjusted start/end, adjusted start/end, payment date).

### C3 — Pillar placement is load-bearing; make it explicit (biggest one)
Our reprice-to-par oracle **will fail** on naive pillar placement. Evidence from `bootstrap.py`:
- The swap pillar sits at `max(fixed_sched[-1], float_sched[-1])` — the **rolled schedule end**, not
  the quoted maturity — else business-day rolls leave `df(end)` extrapolated at solve time and
  interpolated on the final curve (their documented "W8" ~1e-6 residual).
- **FRAs pin `df(start)` as an extra pillar**, else later swap pillars reshape the interpolation over
  that segment and the FRA stops repricing (their documented "W3 structural gap").
- `bootstrap_forward_curve` places pillars at the **quoted maturity** — internally inconsistent with
  the above.
**ng: `CurveBuild` owns pillar placement as an explicit, stated rule** (default: rolled schedule end;
FRA start pinned), uniform across single- and multi-curve. Oracle: every pillar reprices to par
*after* the full curve is built, not just at solve time.

### C4 — Extrapolation decided deliberately
Quarry: left extrapolation is always flat and not overridable; on the right **only log-linear does
flat-forward** — linear/cubic/monotone/Akima silently go flat-in-*level*. **ng: `Interpolation`
states its extrapolation policy explicitly on both ends**; flat-forward is the default right policy
for curve use. No silent per-scheme divergence.

### C5 — Solver curve addressing
`ncurve_solver`'s `InstrumentPricer.reprice(curves: dict[str, Curve])` addresses curves **by name**.
Our `CurveSet` is keyed by `(currency, collateral)` / `RateIndex` — richer. **Resolve when designing
`CurveBuild`:** the solver needs generic addressing over the state vector; a `CurveRef` that resolves
into the `CurveSet` is the likely answer. Not blocking; flag at the slice.

**Kept as-is (validated by the read):** our `RateIndex` carries a `RollRule` (hence a calendar),
fixing a real quarry gap — theirs has no calendar field and infers it from currency.

---

## 3. Not migration — new work (do not mistake for ports)
- **Par deltas**: `curve_risk.input_jacobian` (d zero/d quote) and `curve_bumper` (∂PV/∂z) both exist
  but are **never composed**. The quarry never computed par sensitivities.
- **Key-rate buckets that sum to parallel DV01**: the quarry's do **not** — the docstring claims
  partition of unity, the code performs **no normalisation**, and two incompatible bucket definitions
  coexist. Our "Σ buckets == DV01" oracle is a real gate the quarry never had.
- **Parametric methods are post-fits**, not construction: NS/Svensson/Smith-Wilson fit an
  already-bootstrapped curve. Treat as presentation/smoothing, not solvers.
