# Checkpoint — Topic 0 gate, post-Slice-5 (cadence ≤6)

**When:** after Slice 5 of `HANDOFF_topic0_gate.md` (the handoff's `►► CHECKPOINT`).
**Version:** `0.72.0`. **Branches (stacked on `main`):** `slice/l0-tenor-frequency-roll` (v0.71.0) →
`slice/l0-numerics` (v0.72.0). **Stop-and-wait:** do not begin Slices 6–8 or Topic 1 until ruled.

**Claim:** every `[BLOCK]` item in Slices 1–5 is landed and oracle-gated; L0's numeric base is now
scipy-backed (S17). **Ask:** ratify Slices 1–5 → clear Slices 6–8 (open domains · results/invariants ·
serialisation) → then the Topic 0 gate + parking.

---

## Slice status against the canonical handoff

| slice | items | state |
|---|---|---|
| 1 `l0-tenor-frequency-roll` | **S7** `date+Tenor`, **S3** Frequency tenor-step, **S8** roll anchor | **built this session** — S7 `Tenor.__radd__` + S8 `RollConvention.IMM`/`CDS` new; S3 (28-day/daily/BULLET) was already on main |
| 2 `l0-rate-basis` | **S12** `Compounding`+`convert_rate`, rename→`AccrualMethod` | **already on main** (verified: invariant recorded, no `CompoundingMethod` remains) |
| 3 `l0-index` | **F2** index-carries-RollRule, **F3** decompose, **F4+S10** `Underlying`+siblings | **already on main** (verified: `accrued_rate` reads `index.accrual.roll.calendar`; no `calendar_for_currency`; 4 sibling types w/ `delivery_location`/`grade`) |
| 4 `l0-flows` | **S13** signed, **S2** `Leg` of `Cashflow\|Delivery`, **S14** degenerate raise | **already on main** (verified) |
| 5 `l0-numerics` | scipy interpolation/solvers/distributions, **F1**, **S15** | **built this session** — scipy adapters replace hand-rolled; F1/S15 were already on main |

All Slice 1–5 `[BLOCK]` and `[solid]` items are landed. Slices 2–4 arrived via the earlier gate-rework
merges; I verified each against this handoff's exact wording rather than re-building.

---

## Review input 1 — Oracle-quality audit

| item | oracle | tier |
|---|---|---|
| S7 `date+Tenor` | 31 Jan +1M clamps to 28/29 Feb; day/week exact; +3M/+1Y | closed-form date arithmetic |
| S8 IMM/CDS anchor | IMM schedule lands on 3rd Wednesdays of Mar/Jun/Sep/Dec **regardless of effective date**; CDS on the 20th | published roll tables |
| S5 distributions | `norm_cdf(0)=0.5`, published `Φ(1)`, `ppf` inverts `cdf`, `p∉(0,1)` raises | closed-form / scipy reference |
| S5 solvers | brent/newton/secant find `√2` to 1e-10–1e-12; LM recovers a linear fit's `(a,b)` | closed-form |
| S5 interpolation | cubic/PCHIP/Akima reproduce nodes; **PCHIP no-overshoot** on a flat-then-rise series; per-end FLAT/CONTINUE_SLOPE/RAISE | closed-form / shape property |
| regression (gate) | **SOFR-on-SIFMA ≠ generic-USD** over Columbus Day; **BRL constructible end-to-end** (Currency·BUS/252·São Paulo) | self-consistency / regression |

No oracle rests on quarry cross-check. scipy's numerics are the reference for the adapters (their own
battle-tested test suites); our oracles assert the *adapter contract* + published values.

## Review input 2 — Quarry-drawdown reconciliation

**No parking this checkpoint** — parking is the final gate step (after Slices 6–8), per the handoff's
gate checklist. Drawdown unchanged at **0 physically parked** (the prior parking branch was not merged;
`quarry_reconciliation.md` remains the pre-replan record + topic roll-up). The re-classifications the
spotcheck ruled (`data_registry`→dead, `notional`→L2-absorbed, `fixings`→market-data) are recorded and
will land with the parking manifest. My earlier L0/Topic-1 `fixings` split was ratified as correct.

## Review input 3 — "Challenge me" (design choices to contest)

1. **`CONTINUE_SLOPE` uses a one-sided numerical derivative** at the boundary node (a 1e-6 step
   inside the range), method-agnostic. Simple and uniform across linear/cubic/PCHIP/Akima, but for a
   spline it is the spline's boundary slope, not the last-segment secant. Acceptable, or do you want
   the analytic end-derivative from scipy's `.derivative()`?
2. **scipy interpolators are rebuilt per `interpolate()` call** (no caching of the spline object). L0
   is the *correctness* environment; a hot curve caches the interpolator itself in Topic 1. Flagged so
   it is a deliberate ponytail, not an oversight.
3. **`RollConvention.IMM`/`CDS` on `roll_day` (a union `int | RollConvention | None`)** rather than a
   separate field. Keeps `ScheduleTerms` at 4 fields; the anchor is "how the interior rolls" either way.
   Contest if you'd rather a distinct `roll_rule` field.
4. **`least_squares` uses method='lm'** (no bounds). Fine for L0/unconstrained calibration; bounded
   (`trf`) arrives with its first consumer. Deferred capability, not a gap.
5. **`numpy`/`scipy` pinned `==2.4.3`/`==1.17.1` in CI.** Exact pins maximise convergence
   reproducibility (the ruling's concern) at the cost of upgrade friction. Prefer a compatible range?

## Review input 4 — Smell + debt scan

- **Debt:** `verify.py debt` green — **0 suppressions**; no new `OPEN.md` entry. The one PLR0913 that
  appeared (`_extrapolate`, 6 args) was fixed by deriving `at_left` from `x`, **not** suppressed.
- **Fields:** `verify.py fields` green on merit; new `ExtrapolationEnds` is 2 fields; sole
  `fields-exempt` remains `PricingResult` (output record, §3b-legal).
- **Layers:** `verify.py layers` green — scipy/numpy are numerics, no finance vocabulary enters L0.
- **No duplicates:** hand-rolled `bisect_root`/`nelder_mead` deleted, not shadowed (S17 "no duplicates").
- **Spine conformance (5th input):** foundation modules stay L0 finance-free; scipy is wrapped behind
  our API with provenance headers — the single documented C++-port swap point; engines/models will call
  `foundation.*`, never scipy.

---

## Named next checkpoint

**Topic 0 gate CLOSE** — after Slices 6–8 and parking. Note: Slices **6** (S1 open `Currency` ✓ v0.70 ·
S4 open `Unit` ✓ · S11 daycount completeness ✓ `ONE_ONE`/`ACT_ACT_AFB`) and **7** (S5 `day_type` ✓ v0.69
· S6 `PricingResult` ✓ · S9/S16 invariants) are **largely pre-satisfied on main** from the earlier
rework; the genuine remaining build is **Slice 8** (serialisation on a hard nested case) plus the gate's
**parking** step. I will confirm 6–7 line-by-line against the handoff and build 8 once this checkpoint is
ruled.

**Do not begin Slices 6–8 or Topic 1 until ruled** (handoff `►► CHECKPOINT`).
