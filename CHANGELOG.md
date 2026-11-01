# Changelog

All notable changes are recorded here, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. `pricebook.__version__` is the
single source of truth and CI asserts it matches the top entry.

## [Unreleased]

## [0.7.0] - 2026-11-01

Tranche 7 — the index fixing series and seasoned pricing (`t > 0`), BERHAN-2012 eq 2.5.12. First
rung of finishing the book. Decided by Cowork (`docs/tranches/07_seasoned_index_series.md`).

### Fixed — the YoY swap returned a WRONG NUMBER at every seasoned valuation date
Eq 2.5.13 is 2.5.12 specialised to `t = 0`, where `I(0)/I(T₀) = 1`. Tranche 4 applied that unit
ratio **unconditionally**, so valuing a seasoned trade the engine returned the answer for "the index
has not moved since the last reset" — never reading the published reset `I(T_{ι(t)−1})`, and never
refusing either. On a 3y annual swap 18 months in: **99.0 bp of notional and 34.9% of the inflation
leg at 2%/yr realised inflation** (197.0 bp at 4%).

Every YoY oracle valued at inception, where the two forms coincide exactly — so the oracle's domain
excluded the failure (§4's governing clause, fifth instance in this vertical).

### Added
- `market/fixing_series.py` — `FixingSeries` (published levels keyed by reference month) and
  `reference_level`, which applies the index's publication lag and interpolation. **One atom, three
  engines**, so the ZCIIS, the linker and the YoY swap cannot disagree about which print `I(t)` is.
  `LINEAR` refuses when the following month is missing rather than falling back to flat — a flat
  fallback returns a plausible number under the wrong convention.
- `InflationIndex` gains `publication_lag` and `interpolation` (`FLAT` for UK gilts, `LINEAR` for
  TIPS), **required, not defaulted**: "3 months, linear" is wrong for UK gilts, so a default would
  silently price the wrong convention.
- 26 oracles. The anchor is the exact reduction — seasoned equals inception at `ι(t)=1`, the same
  division giving `1.0` — with the mandatory companion that a real elapsed period moves the price
  by the amount the ratio implies. Resolution is oracled on a **sloped series read mid-month**, and
  the two ways that fixture could lose its power (a flat series; a date on the 1st, where `LINEAR`
  degenerates to `FLAT`) are asserted rather than assumed.

### Changed — BREAKING: `schema_version` 1 → 2
A product's `"index"` went from a bare name to the nested `InflationIndex.to_dict()`, because a name
cannot say which monthly print `I(t)` is. `InflationIndex.from_dict` **refuses a bare string** rather
than defaulting the conventions. `check_schema_version` cannot catch this alone — it only rejects
payloads *newer* than the build — so the refusal comes from the reader and has its own oracle.

`MarketSnapshot.index_fixings: Mapping[index, float]` → `fixings: Mapping[index, FixingSeries]`.
The scalar is the degenerate one-point case, which is what makes the replacement provably faithful
rather than a rewrite.

A YoY swap **at inception still needs no fixing at all** — eq 2.5.13 contains no index level, as
tranche 4's oracle asserts. The seasoning guard is on the dates (`valuation_date ≤ reset_date`),
never on whether a series is present, since "no series ⇒ ratio 1" was the original defect.

### Docs
- Convention register **C9** — index resolution is one atom; the register's first **structurally**
  guarded row (one function, so a divergence is not expressible without duplicating it first).
- `OPEN.md` **OI-1 reopened**: the finding that there were zero `crosscheck:` markers was Code's own
  error — there are **six**, written in the provenance-header form while the charter specified the
  `#`-comment form. Charter §4's defect class committed by the person invoking it. The live question
  is only which spelling §8 should name.

## [0.6.0] - 2026-10-31

Tranche 6 — calibrating Jarrow–Yildirim's volatilities to a year-on-year floor strip
(BERHAN-2012 ch 5). Two stages, and only the vol stage is fitted: the curve stage stays pinned by
tranche 2, because refitting curves against vol quotes launders a curve error into a vol.

### Rejected — the stock's calibration fits volatilities to the wrong instrument
`_stock/.../jarrow_yildirim.py::jy_calibrate` fits `(σn, σr, σI)` to **zero-coupon** inflation swap
quotes, which under correct JY carry **no volatility information at all**: the fair rate is
`Pr(0,T)/Pn(0,T) − 1` (eqs 1.2.13/1.2.16) — it is *because* the ZCIIS is model-free that tranche 2
built the real curve from it. Its objective is non-flat only through two defects: model-implied
Vasicek `θ = r₀` curves where the market curves belong, and a convexity adjustment applied to the
ZCIIS when JY puts convexity on the **year-on-year** swap. Measured at a 5y quote, doubling `σI`
moves its fair rate **+10.3 bp** and doubling `σr` **+15.1 bp** — enough to drive a least-squares
fit against basis-point quotes, and entirely artefact. It converges on its own bugs and reports
success. Not purified; mined for content only (the fixed-vs-free split).

Third instance of **stock-is-not-authority** in this vertical, and the first in a *calibration*
rather than a formula.

### Added
- `engine/jy_calibration.py` — `calibrate_jy_to_yoy_floors`, fitting `(σr, σI)` to a strip of ATM
  floors through tranche 5's pricer. `JYFixedParams` bundles what the thesis sources elsewhere
  (both mean reversions, the correlations, and `σn` from *nominal* ATM caps — an engine this
  library has no reason to own, so `σn` is a named input with no default and the fit is
  conditional on it). Returns `PricingFailure`, never a NaN and never a raise.
- `foundation/solvers.py` — a thin bounded-least-squares adapter on the seam `distributions.py`
  established: scipy stays behind our own API, so models and calibrators are scipy-free and a port
  has one swap point. Its signature is exactly its one caller's need; non-convergence is a
  *reported value*, which is what lets the calibrator return a failure instead of a confident
  number.
- 16 oracles. A1 recovers the generating parameters from synthetic quotes; **A2a** proves the
  objective is not *flat* and **A2b** proves it is not *degenerate*; A3 the residual vanishes at
  the truth and descends from a wrong start; A4 the cross-stage break-even round trip; A5 failure
  is a value.

### The oracle lesson: "not flat" and "not degenerate" are different failures
A parameter-recovery round-trip is vacuous on a flat objective — the stock's failure — so it needs
a discriminator. Cowork's review pass (thread 0035) found the discriminator I wrote was itself
insufficient: per-parameter sensitivity proves the objective is not flat, but for a *two*-parameter
fit it cannot prove the optimum is **unique**, since `σr` and `σI` can each move the strip a lot
while moving it in nearly collinear directions.

Demonstrated rather than argued. On a one-quote strip (two parameters, one equation), A2a passes
comfortably — `σr` +10% moves the quote 83× its gate, `σI` +10% moves it 12× — and the fit reaches
`rms = 8.7e-11`, a *perfect* fit, at `σI = 0.0039` where the truth is `0.0120`. **Wrong by a factor
of three, reporting success**, with three starts giving three different answers. A2b adds
multi-start uniqueness and a bound on the sensitivity Jacobian's condition number: `inf` on that
degenerate strip, `6.7e2` on the real six-quote one against a `1e4` gate — so it bites where it
must and does not bind where it must not.

### Fixed — A4's own premise check could not fail, and it bit during the build
A4 asserts the rebuilt real curve equals the direct one before comparing fits, or it would compare
two calibrations of two different markets. The first version checked that **only at the quote
maturities** — exactly where the two agree by construction — and passed while the curves differed
below 2y, where `real_curve_from_breakeven` can only flat-extrapolate under its shortest quote. The
fit then missed `σI` by 0.14%, which is how the hole surfaced. The premise is now checked on a
30-day grid across the whole horizon the floors touch. The tranche's own subject, found in its own
test.

### Changed
- `TEST_FLOOR` 87 → 103.

## [0.5.0] - 2026-10-31

### Fixed — a WRONG FORMULA shipped in v0.4.0, found by this tranche's cross-product oracle
**Eq 2.5.11's bracket nesting was wrong.** The `ρ_nr` cross term belongs **inside** the
`βr(t,Tᵢ₋₁)` factor; v0.4.0 placed it beside it. Error size: **1.8% of the YoY convexity** at a
2y–3y period, 1.5% at 4y–5y, and **unbounded** at `t = Tᵢ₋₁`, where the correct value is exactly
zero and v0.4.0's was 2.6e-4.

**Why nothing in tranche 4 caught it, which is the part worth keeping:**
- **The sweep oracle compared two copies of the same mistake.** T4's "independent transcription"
  came from the same `pdftotext -layout` output as the implementation — and that rendering *merges
  2.5.11's parentheses*. **Two transcriptions of one corrupted source are not independent**, so a
  108-point sweep passed with both sides wrong identically. The telescoping failure mode in a new
  disguise: we guarded against reading a formula from the *stock*, then read it twice from one
  mangled rendering of the *source*.
- **The `σr → 0` anchor could not see it** — `σr` is an overall factor under either nesting.
- **What found it:** this tranche's `cap − floor = T4's YoY leg` oracle, disagreeing by 0.65%. The
  cross-product check doing exactly its job: different code on each side, so a shared transcription
  error has nowhere to hide.
- **What identified the cause:** a **structural requirement**, not a number. `C` corrects
  `E^{Tᵢ₋₁}[Pr(Tᵢ₋₁,Tᵢ)|Ft]`, so at `t = Tᵢ₋₁` the quantity is already known and `C` must be
  **exactly zero** — a property of the object, true for every parameter set. The wrong nesting
  violates it; a test for it now exists and should have existed in T4.

**Two practices follow:** re-read a load-bearing formula in the **`-raw`** extraction as well as
`-layout` (layout reflows fractions and merges parentheses; raw preserves token order); and **ask
what the object must satisfy regardless of parameters** — `C(t=Tᵢ₋₁)=0`, `V² ≥ 0`, parity — because
those outlive a transcription error.

### Added — tranche 5: the inflation cap/floor under Jarrow-Yildirim
Eq 2.6.5 per period, summed. Source `BERHAN-2012` §1.2.4 + ch 2.6, **read first-hand**. Proposal,
review and results: `docs/tranches/05_inflation_capfloor.md`.

- **L3** `yoy_variance` — eq 2.6.4's **eight terms**; `JYCorrelations.nominal_index` (`ρ_nI`) lands
  as **its first consumer**, since it enters `V²`'s eighth term and nothing else in the model.
  Required rather than defaulted: a default would be a hidden knob on a real model parameter.
- **L0** `CapFloorKind` — an enum, not a bool: `InflationCapFloor(..., True)` is unreadable and
  silently reversible. · `foundation/distributions.py`, a thin scipy adapter whose oracles test
  **the adapter** (`Φ(x)+Φ(−x)=1`, which the parity oracle leans on), never scipy's numerics.
- **L2** `InflationCapFloor(index, schedule, strike, notional, kind)` — **a strike, not a leg.**
  Ruled: a leg accrues and discounts, a strike enters `d₁/d₂`; same two fields, different
  *operation*, so reusing `YoYLeg` would have put a strike in `fixed_rate` and made its "accruing"
  docstring false. The real shared shape is **`CouponSchedule`**, which gets its honest second
  consumer.
- **L4** `price_inflation_capfloor` + `caplet_value`.

### Oracles — 87 green
- **Primary:** eq 2.6.5 against an independent transcription, swept over four strikes × both kinds;
  and eq 2.6.4 swept over 4 × 3 × 3 × 4 parameter combinations. Primary *because* the reduction
  below kills six of `V²`'s eight terms.
- **Reduction + companion:** `σn = σr = 0` ⇒ `C = 0`, `V² = σI²Δ` ⇒ plain Black on the
  deterministic forward ratio, exactly — paired with a test that the rate vols **move** the price,
  without which the reduction passes an implementation ignoring seven terms.
- **Parity:** `cap − floor = N·Σψ·Pn(Tᵢ)·(m − K)`, and at `K = 1` it **is** tranche 4's YoY leg —
  the oracle that found the bug above.
- **Structural:** `C(t=Tᵢ₋₁) = 0` exactly; `V² ≥ 0` across adverse correlations; `C` invariant to
  `ρ_nI`; deep-ITM → forward annuity, deep-OTM → 0; monotone in strike; no-vol → intrinsic.
- `TEST_FLOOR` 64 → 87.

## [0.4.0] - 2026-10-31

### Added — tranche 4: the year-on-year inflation swap and Jarrow-Yildirim
**The first model-dependent instrument.** Eq 2.5.13 with the JY convexity 2.5.11, read in the held
PDF. Source `BERHAN-2012` ch 2. Proposal, correction and results:
`docs/tranches/04_yoy_jarrow_yildirim.md`.

- **L3** `hw_beta` (eqs 2.4.3/2.4.16, shared by both factors so nominal and real cannot drift) ·
  `HullWhiteParams` · `JYCorrelations` · `JYParams` · `yoy_convexity` (2.5.11) ·
  `JarrowYildirim(market, params)`.
- **L0** `YoYLeg` — **separate from `RealLeg`**: rule-of-two fired at the second leg type and the
  answer was *no*. A real leg pays a rate on an uplifted notional, a YoY leg an index *ratio*; one
  name over both would invite a kind-of-leg branch, which the no-`isinstance` law forbids.
  Rule-of-two triggers an abstraction when consumers share a *shape*, not a *word*.
- **L2** `YearOnYearInflationSwap` — three fields and **no base index**: a YoY floating side is a
  chain of period-on-period ratios, so it has no single `I0`.
- **L4** `price_yoy_inflation_swap`, `yoy_inflation_leg_value`, `yoy_fixed_leg_value`,
  `yoy_par_rate`. **Nothing new at L1.**

### Fixed — the handoff's anchor oracle was wrong, found by reading the source
The oracle plan specified `ρ → 0` as T4's anchor, **taken from the stock's
`yoy_convexity_adjustment` rather than from the thesis.** That function computes
`σI·σn·ρ·tenor·year` — a first-order approximation in which `ρ` *is* a factor — and it **takes a
`real_vol` argument it never uses** (the carried-but-unread smell, fourth instance, every one of
them in stock code we are declining to purify).

In JY, **`σr` is eq 2.5.11's overall factor and `ρ` is not**; the thesis says so in the line after
it. Measured: all-parameters `C = −1.4738e-02`; **at ρ=0, −1.4486e-02** (barely moves); **at σr=0,
exactly 0**. So the anchor is **`σr → 0`**, which is also *better* — a structural factor gives an
exact reduction rather than a numerical limit.

**Had the ρ→0 anchor been built it would have failed against the correct formula** — and the danger
is not the red but that a red anchor invites bending the formula until it passes. Cowork owned the
error and recorded the lesson: **oracle anchors are read from the held source's formulas, never
inferred from the stock's code — the stock is content and regression, the source is authority.**

### Oracles — 64 green
- **The anchor:** `σr → 0` gives the deterministic-real-rate limit exactly, **paired with a test
  that the convexity MOVES the leg** — without which the anchor would pass on an implementation
  that ignored `C` entirely. A reduction proves the general case *collapses* correctly and says
  nothing about whether it does anything at all.
- **`ρ = 0` does NOT vanish**, asserted against the surviving `−½βr²` term, so nobody simplifies
  2.5.11 back toward the approximation.
- 2.5.11 against an independent transcription over a **108-point** parameter sweep · `β`'s closed
  form **and its `κ → 0` limit** (the limit that catches a misplaced `κ`) · eq 2.5.13 transcribed
  independently · **the first period equals tranche 2's ZCIIS leg** (the thesis's own structure, and
  different code on each side) · par-rate reprice (a telescoping consistency guard, kept as such) ·
  matured → PV 0 · **prices with no index fixing**, since 2.5.13 contains no index level.

### Changed — three gate catches, all fixed at the cause
- **`PLR0913`**: `yoy_convexity` took 7 arguments → bundled into `JYParams`, the model's own
  structure rather than a split to satisfy a count.
- **`layers`**: the word *"payoff"* in a docstring tripped the L0 finance-free gate, which greps
  prose as well as code. **Reworded rather than loosening the gate** — a gate that over-reports on
  comments is still doing its job, and relaxing it so my own docstring passes is the wrong
  direction.
- **`debt`**: four `# noqa: E731` on test lambdas → real local functions. There was no
  justification to give, which is the gate's point.
- `TEST_FLOOR` 50 → 64.

## [0.3.0] - 2026-10-31

### Added — tranche 3: the inflation-linked bond (linker), model-free
`ILB(t,TM) = I(t)/I0 · N[ Σ C·ψᵢ·Pr(t,Tᵢ) + Pr(t,TM) ]` — BERHAN 1.2.5 generalised with the accrual
fraction a real linker pays. **No nominal curve is touched**: a linker is a real bond scaled by the
index ratio, the foreign-currency analogy at its clearest. Source `BERHAN-2012` §1.2.1. Proposal,
review and results: `docs/tranches/03_inflation_linked_bond.md`.

- **L0** `CouponSchedule(dates, day_count)` — period **boundaries**, so an accrual has both ends
  rather than inferring the first start · `RealLeg(schedule, coupon_rate)`. Both are §3b-compelled
  bundling (seven loose fields, two over the limit) and both stay **concrete frozen bundles** — no
  base class, no polymorphism; whether `RealLeg` generalises is decided at the *second* leg type.
- **L2** `InflationLinkedBond(index, base_index, leg, notional, indexation_lag)`.
- **L4** `price_inflation_linked_bond`.
- **Nothing new at L1 or L3** — a second inflation product cost no market-data change at all, which
  is what keys-not-fields buys.

### Oracles — 50 green, first run
- **The closed form** with every `ψ` and `Pr` computed in the test, swept over nominal rates that
  must **not** move the price (1.2.5 has no `Pn`; a leak shows immediately).
- **The exact reduction:** every period is 365 days so each `ψᵢ = 1`, and the general form *is* the
  paper's annual 1.2.5. The premise itself is asserted alongside it.
- **Cross-product reconciliation (the first of its kind here):** the linker equals
  `Σ C·ψᵢ·ZCIB(Tᵢ) + ZCIB(TM)` with each `ZCIB` rebuilt from **tranche 2's** `inflation_leg_value`
  — different code on each side, so a divergent index-ratio application in one product but not the
  other breaks it while each product's own oracle still passes.
- Zero-coupon limit → eq 1.2.4 scaled · `Pr = Pn`, `I = I0` prices as a **nominal** bond (catches
  the ratio on the wrong leg, or `Pr`/`Pn` transposed) · doubling the index doubles the **whole**
  price · past coupons excluded, checked by equality against the same bond with those dates removed
  · matured → PV 0 · missing curve/fixing → failure · the round-trip **exercises the lag**.

### Changed — the `day_count` question, answered a third way
Cowork spotted that pricing BERHAN's literal annual 1.2.5 would leave `CouponSchedule.day_count`
**carried but unread**, and offered two resolutions (read it and register the convention, or drop to
a dates-only schedule). **Took a third: price the general `C·ψᵢ` form and make 1.2.5 its exact
reduction.** Better on three counts — the field is read so nothing is smuggled, the instrument is
the real one rather than the paper's simplification, and it buys a reduction oracle neither option
offered.
- **Register gains C8** — a schedule's accrual day count and its curve's `TimeMeasure` are one
  convention. A **C7 sibling**: OWED, currently unrepresentable (one day count exists), trigger
  shared. Register now reads five of eight not cross-module guarded.
- **The indexation lag is carried and round-tripped, on Cowork's sharper test.** I had argued
  "contract data vs market metadata" and asked to be ruled rather than decide in my own favour. Its
  test is better: **constitutive of the instrument's identity, not "does the pricer read it."** Two
  linkers on one index with different lags are *different contracts*, so the round-trip **is** its
  present consumer — exactly what T2's publication-lag field lacked. Oracle asserts both that the
  lag survives serialisation and that `lag=3 ≠ lag=8` as contracts.
- `TEST_FLOOR` 37 → 50.

## [0.2.0] - 2026-10-31

### Added — tranche 2: the inflation spine (ZCIIS, model-free)
The zero-coupon inflation-indexed swap, priced with **no dynamics** — pure no-arbitrage from the
foreign-currency analogy (real ↔ foreign, nominal ↔ domestic, index ↔ FX spot). Source:
**`BERHAN-2012`** §1.2.2. Proposal, review and results: `docs/tranches/02_inflation_spine.md`.

- **L0** `InflationIndex` — the asset class's *entire* L0 footprint, a bare identity ·
  **`Term(start, maturity, day_count)`**, which exists because the contract needs six loose fields
  and the rule's answer is to bundle the cohesive ones, never to raise the limit. It also puts the
  compounding exponent in one place: `TM` is measured from **inception**, since the fixed payout
  was fixed at trade time and does not shrink as the trade seasons.
- **L1** `CurveRole.REAL` — **a new curve KEY, not a new snapshot field**: a second discount curve
  costs one enum member while `MarketSnapshot`, `CurveSet` and every existing pricer stay as they
  were · `MarketSnapshot.index_fixings` · **`real_curve_from_breakeven`**, eq 1.2.16.
- **L2** `ZeroCouponInflationSwap`, serialised and version-enforced.
- **L4** `price_zc_inflation_swap` + `inflation_leg_value`, composing `df` from **both** curves.

### The design call: there is exactly ONE `Pr`
`Pr` is a curve in the snapshot; eq 1.2.16 is **how it is built** from break-even quotes, not a
second way to price. The pricer never recomputes it. Two oracles guard that, with different jobs:
the **par identity** proves construction and pricing share one `Pr`, and a **pointwise** 1.2.16
check proves the constructor is right on its own.

### Oracles — 37 green
- **1.2.14** off curves the test specifies, to 1e-9 · **the exact reduction**: 1.2.13 with
  `I(t) = I0` *is* 1.2.14, so there is one formula and not two · the index enters as a **ratio**
  (double it and the real-bond term doubles while `−Pn` does not — a difference would shift by a
  constant) · the net swap equals leg **minus** fixed leg, so the net PV cannot become its own
  formula · zero-inflation degenerate case · sign (the receiver gains on higher break-even, loses
  on a higher strike) · matured → PV 0 · missing real curve / missing fixing → failure · round-trip
  with the version enforced.

### Fixed — a must-fix found a second bug, in the API rather than the test
- **`real_curve_from_breakeven` now takes quote DATES, not year fractions.** Cowork's must-fix
  (the par oracle telescopes, so the constructor needs its own pointwise oracle) made that oracle
  **fail at the 2.5-year pillar by 6.6e-6**: a caller saying "2.5 years" converts to 912 whole
  days — 2.4986 years on the curve's own measure — so the caller's `T` and the curve's `T`
  disagreed by a day. Fixed structurally: one `date → t` map, the curve's, and a caller cannot
  hold a second. **The par oracle could never have seen this**, because a one-day disagreement
  shared by construction and pricing telescopes away — which is precisely the must-fix's argument,
  demonstrated.
- **A matured swap returns PV 0, not `PricingFailure`** (must-fix 2). Conflating *historical* with
  *unpriceable* would make a settled trade look broken; checked before any market lookup, so a
  matured trade needs no curves.

### Changed
- **No lag field on `InflationIndex`, and the stock is why.** Its `inflation_indices.py` carries one
  `publication_lag_months`: 3 for most indices, **8 for UK index-linked gilts**. Those are two
  quantities conflated — UK RPI publishes ~1 month after its reference month, and the 8 months is
  the **indexation lag the gilt's contract specifies**, which two linkers on the same index can set
  differently. So the indexation lag is L2 contract data (arriving with the linker in T3), and even
  the publication lag is omitted here because **nothing reads it** — the snapshot carries the
  already-resolved `I(t)`. Ratified by Cowork, which confirmed the 8-month figure against the stock.
- **Conventions register gains C7** (Cowork's review condition): the break-even construction, the
  nominal curve's `TimeMeasure` and the contract's `Term.day_count` share **one** day count, and the
  par identity holds only under it. Recorded **OWED**, because *no test could fail on it* — the par
  oracle telescopes when the conventions match, which they always do today. Sharpened while
  recording it: with one day count in the library the convention is not merely unguarded but
  **unviolable**, so a guard now would be an imagined problem. Trigger: the first second day count.
- **`# crosscheck: quantlib …` markers** at both sites wanting external validation (thread 0014),
  so the later optional QuantLib tier has a floored worklist instead of a from-memory sweep.
- `TEST_FLOOR` 25 → 37.

## [0.1.1] - 2026-10-31

### Fixed — a defect in v0.1.0, found by Cowork's independent review
- **The schema version was written and never read.** Both `to_dict`s stamped `schema_version` and
  **nothing dereferenced it**, so a payload from a future build would have loaded silently under
  v0.1.0 code — defeating the exact guarantee the field's own docstring claims ("a format with no
  version cannot change without silently breaking a reader"). It was therefore not merely an unused
  field but a **false claim in a comment**, which is the worse of the two. The project's named
  defect shape, **resolve-don't-just-exist**, and v0.1.0's test asserted only *presence*, which
  could never have caught it.
  Fixed with **one** shared `check_schema_version(payload)` that both `from_dict`s call — one
  definition of what is readable, so the two products cannot drift — returning the validated
  version so a future migration branches on it rather than re-reading the raw field.
- **Six oracles that assert REFUSAL, not presence:** a future version, a missing one, six
  malformed ones (`"1"`, `0`, `-3`, `1.5`, `True`, `None`), the nested payload, and **the
  ordering** — the version is checked *before* `kind`, because in a format we cannot read the
  `kind` field's meaning is itself unknown, so reporting "not a fixed_cashflow" would be a guess
  about a payload we have already failed to understand.

### Changed
- **`numerics` threading dropped from `dispatch`.** Purified from the stock as-is, it inspected
  each pricer's signature to decide whether to pass an optional numerical config — and **ran in no
  test**, since the only pricer is closed-form. Three reasons agree: the config's own rule is that
  it *lands with the first engine that reads it*; there is no consumer; and an unexercised dispatch
  branch at the baseline is the same dead-code shape the debt gate caught in `zero_rate`. It
  returns with the first pricer that integrates or samples — with a consumer and a test.
- `TEST_FLOOR` 18 → 25.

### Process note
v0.1.0 merged before this review round landed, so the fix ships as a patch rather than as part of
the tranche. The sequence Cowork set is **propose → review → land**; here it ran propose → build →
land → review → patch. No harm done (the defect was caught and is fixed in the next commit on
`main`), but the review's value is highest *before* a merge, which is where the beat belongs.

## [0.1.0] - 2026-10-31

### Added — tranche 1: the minimal spine
A single fixed cashflow priced end to end through the stateless
`price(product, model) → PricingResult`. The backbone every product needs and none can avoid, so
it prejudges no product while proving the engine contract. Proposal and results:
`docs/tranches/01_minimal_spine.md`.

- **L0** `Money`/`Currency` (currency mixing is an error, never a silent addition) ·
  `DayCountConvention` + `year_fraction` (ACT/365F only) · `TimeMeasure` (the one `date → t` map) ·
  `Cashflow` · `PricingResult`/`PricingFailure`.
- **L1** `DiscountCurve` + the `CurveHandle` protocol the engine depends on · `CurveSet`/`CurveKey`
  (curves keyed, not positional) · `MarketSnapshot`, which **carries the valuation date** — there
  is no clock read anywhere in the library.
- **L2** `FixedCashflow` — pure data, with `to_dict`/`from_dict` from the first product.
- **L3** `DiscountingModel`, carrying the snapshot it was built on.
- **L4** the engine registry (structural dispatch, no `isinstance` ladder) + `price_fixed_cashflow`.

### Oracles — 18 green
- **`PV = N·e^{−r·t}` exact to 1e-12** across 48 points. The reference is arithmetic written in the
  test, independent of every path under it — which is the point: the predecessor shipped a 5.14%
  error whose oracle restated the builder's own composition and so validated it against itself.
- **Analytic vs finite-difference DV01** to 1e-6 — two different computations, so agreement is not
  tautological.
- The **valuation-date boundary** in both directions: a flow *on* the date is excluded, one *day
  after* is included and positive. Without the second, "excluded" could be implemented as "always
  zero" and the first would still pass.
- Statelessness (byte-identical reprice, snapshot untouched); failure-is-a-value on three distinct
  failures; currency mixing refused; the representation round-trips and carries a schema version.
- **Interpolation tested on a SLOPED curve.** A flat curve returns one rate everywhere and cannot
  tell linear interpolation from any other scheme — the fixture trap the predecessor hit twice.

### Stock
11 of 13 modules **purified** from `_stock/`, 2 built fresh, nothing over 100 lines. Purification
was mostly *trimming*: `money.py` 289 → 90 lines (the open-currency *insight* kept, the registry
machinery left behind until something needs named constants); `day_count.py` 332 → 45 (one
convention); `snapshot.py` 154 → 30 (curves only — its other shapes grew one asset class at a
time and return when a product pulls them).

### Changed
- **`CalibratedModel` not carried.** A protocol declares a capability, and a discounting model has
  none to declare beyond carrying its market, so carrying it would be an empty abstraction. The
  registry's model parameter is `object` until a pricer needs a real capability.
- **`DiscountCurve.zero_rate` uses `bisect` and carries no suppression.** The first draft ended in
  an unreachable `raise` needing `# pragma: no cover`, which the debt gate caught. The answer to
  dead code is to remove the deadness, not to ledger the suppression — and `bisect` is O(log n),
  which will matter at a bootstrapped curve.
- `TEST_FLOOR` 1 → 18.

## [0.0.0] - 2026-10-30

### Added
- **The clean project scaffold.** An empty, CI-green library at `v0.0.0` — no pricing code.
  Package `pricebook`, MIT, public. `pyproject.toml`, one trivial test, the ubuntu+windows CI
  matrix on Python 3.12, and `verify.py` carried from the migration tree and **trimmed to the
  gates that still apply**.
- **The stock.** The whole previous tree — the built core, the un-migrated `quarry/`, its `docs/`
  and all 77 oracle test files — shelved in `_stock/` and **gitignored**: reference material,
  mined by hand for *content*, never for *structure*, never imported. Its pushed backup is the
  archived `pricebook-migration` repo at tag `migration-archive`.
- **`docs/handoffs/phase_R_record.md`** — what Phase R did, the three parameters Bernardo ruled
  before execution, and the four `verify.py` calls taken under delegated mechanics.
- **`docs/conventions_register.md`** carried as a tracked living doc, because it describes the
  built core rather than the migration — including its one **OWED** item, the vol clock, which
  the rebuild re-lays *with* its cross-module oracle.

### Changed
- **The migration frame is retired** (`docs/handoffs/rulings_reset_to_fresh_repo.md`). The unit of
  work is now a product you want, pulling in the core it needs; `done` is no longer "the quarry is
  empty". The architecture, the oracle discipline and the conventions are kept; the obligation to
  cross every old module is dropped.
- **`verify.py` trimmed:** the `quarry` import-floor gate and the drawdown reconciliation are
  **deleted** — both measured migration progress, and there is no migration to measure. Kept:
  `acyclic`, `layers`, `fields`, `debt`, `provenance`, `version`, `slow`, `tests`.
- **Provenance header: `quarry:` → `stock:`**, same role against the new reference tree. Because
  `_stock/` is gitignored and therefore absent in CI, the resolve-check **reports** that it could
  not dereference paths rather than skipping silently — a gate that quietly passes is this
  project's named defect class. `source:` and `oracle:` are verified in both environments.
- **License corrected to MIT** in `pyproject.toml`, matching the `LICENSE` file (the old tree's
  metadata said Proprietary).
