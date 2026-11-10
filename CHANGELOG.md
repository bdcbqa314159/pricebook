# Changelog

All notable changes are recorded here, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. `pricebook.__version__` is the
single source of truth and CI asserts it matches the top entry.

## [Unreleased]

## [0.35.0] - 2026-11-10

Tranche 29 — **collateral / CSA: the collateralized exposure over the margin period of risk.** The
last build of the XVA spine arc; `CP_xva_spine.md` and re-audit #2 follow.

### Added

- **`risk/collateral.py` (L5)** — `CollateralAgreement` (a zero-threshold, zero-MTA variation-margin
  CSA: its margin period `δ` and the deterministic initial mark) and `collateralized_exposure`,
  which returns `(V(t) − V(t−δ))⁺` path by path.

**Collateral enters only through the exposure.** The gap is assembled into the same
`SimulatedExposure` the uncollateralized engine produces, so T25/T27's `credit_adjustment` prices it
**unchanged** — no collateralized CVA function, no second integral, no branch in the credit code. A
CSA changes *what is exposed*, not how an adjustment is computed. The positive part lives here
rather than downstream, because a negative gap means we are over-collateralized and an
`ExposureProfile` is a non-negative object by construction.

### Oracles

- **Two exact reductions, each with its non-degenerate companion.** `δ = 0` gives exactly zero
  (`V(t) − V(t) = 0`, by construction, not a limit); `agreement=None` returns the input **object**,
  so the uncollateralized case is a structural special case and T26/T27's numbers provably cannot
  move. Both reductions are *trivially* exact, which is stated plainly — the companion is what makes
  the pair non-vacuous: a material `δ` on a volatile mark where the collateralized CVA is **less
  than half** the uncollateralized value and strictly positive.
- **A rate oracle, not just monotonicity.** The gap's standard deviation is `σ√δ`, so the adjustment
  is proportional to `√δ`: halving `δ` must multiply the CVA by `1/√2 = 0.7071`. Measured 0.772,
  0.710, 0.706 over successive halvings. A wrong power of `δ` passes a monotonicity check and fails
  this one.
- **A structural signature** separating the gap from the mark: the gap's law does not depend on `t`,
  so its EPE is **flat** (2.4% across the horizon, sampling noise on a `t`-independent law) while
  the mark's EPE grows (17.4%). Asserted as the *separation*, not as two fixture-tuned bounds.
- The gap EPE against T25's `expected_positive_exposure` on the gap's own moments, at the MC rate.
  **Derived, not transcribed** — there is no source equation for the gap construction, and §4 rates
  a derivation above a transcription.
- The process is **not** re-anchored: the gap is a functional of the very paths T26's Leg B validated
  against the numéraire-relative martingale identity, so it inherits that validation. Inventing a
  fresh anchor over the same paths would add ceremony, not evidence.
- Requirement oracles: collateral can never *increase* the adjustment, and the adjustment is monotone
  increasing in `δ`.

### A real hole found by mutation, and the fixture that closes it

Reversing the gap to `(V(t−δ) − V(t))⁺` — the **wrong** exposure — initially **survived all 986
tests**. It is not an equivalent mutant. It survived because the fixture's mark was **driftless**,
making `V(t) − V(t−δ) ~ N(0, σ²δ)` symmetric about zero: the reversed gap then has the *identical
distribution*, and every mean-based oracle is structurally blind to the direction. Measured, the two
EPEs agree to the last digit with no drift and differ by **87%** at a drift of 24/year (572% at 60).

Closed with a **drifting-mark fixture** whose closed form carries the drift (`N(μδ, σ²δ)`), a premise
asserting the drift is material against the gap's own standard deviation, and a companion asserting
the band is tight enough to *reject* the reversed value — rather than assuming the new fixture would
catch it. All six mutants now die.

This is §4's degenerate-fixture rule in a form not yet recorded there: the degeneracy is a
**distributional symmetry**, not a specific parameter value (`τ = 1`) nor a uniformity across a
slice. The guard is therefore not "the values differ" but "the fixture breaks the symmetry the
mutation exploits".

### Scope fence

Zero threshold and zero MTA, variation margin only. Thresholds/MTA, **initial margin** and its MVA,
ColVA (the funding cost of collateral), and gap/wrong-way risk are each deferred, not approximated.

## [0.34.0] - 2026-11-09

Tranche 28 — **FVA: the funding adjustment, a fixed point with a closed form.** Fourth and last
**core** unit of the XVA spine arc: CVA · DVA · FVA complete. Collateral (T29) is optional after it,
then the checkpoint closes the arc.

### Added

- **`risk/funding_valuation_adjustment.py` (L5)** — `funding_valuation_adjustment` (Crépey eq 17's
  closed form `γ/(1+γ)·(P−CVA)⁺`), `fva_fixed_point_residual` (eq 19's defining equation as named
  surface, because it is the tranche's correctness anchor and §4 says extract what an oracle must
  reach), and `iterate_fva` (the Picard iteration, returning the whole sequence so a test can read
  the *rate* and not only the limit).

### Oracles

- **The anchor is that the closed form satisfies its own defining equation** (§4: a formula and its
  stated inverse must actually invert), asserted to machine precision across the spread range. It is
  derived from the finance, so it survives a transcription error — and it is sharp against exactly
  the mistake the source's typography invites.
- **With the companion that proves the anchor has power:** the natural reflow misreading
  `γ·(P−CVA)⁺` is asserted **not** to satisfy the fixed point. Without that, "the closed form
  satisfies its fixed point" could be a property of a whole family of formulas rather than of this
  one (§4, the Box-Muller pattern).
- The `γ/(1+γ)` coefficient is printed as a **stacked fraction**, so a `-layout` reflow splits
  numerator from denominator across lines — the same hazard that produced the eq-25 KVA error. Read
  in `-raw` and `-layout` and reconciled, **and independently derived** from eq 19 (with the
  positive part open, `FVA(1+γ) = γ(P−CVA)`), which §4 rates above a transcription.
- An **independent root-find** on the defining equation (different code from its algebraic solution)
  agrees across five spreads.
- **Both regimes of the `⁺`**: a net payer with `P−CVA` materially positive (asserted, else the
  formula is invisible) and a **net receiver**, whose FVA is *exactly* zero — the `⁺` binding is a
  statement about the economics, not a rounding outcome.
- **Seven mutants, all killed.** The reflow slip (dropping `1/(1+γ)`) dies to **18** tests.

### Corrected — a spec claim, measured

The tranche spec justified the Picard iteration as "a contraction, factor `γ/(1+γ) < 1`". Measured,
the error ratio is **`γ`**: `γ/(1+γ)` is the *solution's* coefficient, not the map's Lipschitz
constant, since `e_{n+1} = −γ·e_n` exactly while the positive part stays open. The consequence is not
cosmetic — the iteration converges only for `γ < 1`, **oscillates at `γ = 1` and diverges above it**,
while the closed form keeps satisfying the fixed point for every `γ ≥ 0`. Both facts are now pinned
by tests rather than left in prose. This strengthens the ruling that the closed form is production: a
"solve the fixed point numerically" instinct would have produced code that silently fails outside
`γ < 1`.

### Stated assumption

`CA = CVA` holds because this spine is **uncollateralized with no initial margin**; in general
`CA = CVA + FVA` (eq 11). MVA and the collateral terms that make `CA ≠ CVA` are deferred (T29 /
`to_go_deeper`), so the assumption is named in the module rather than assumed silently.

## [0.33.0] - 2026-11-09

Tranche 27 — **bilateral CVA + DVA: both parties can default.** Third unit of the XVA spine arc. The
step from "only the counterparty defaults" to the mirror case, and with it the first-to-default
conditioning the unilateral spine dropped.

### Added

- **`credit_adjustment(exposure, defaulting, surviving, discount, lgd)` (L5)** — the one integral
  both credit legs are, extracted at the second real consumer (§6's rule-of-two: T25 was the first,
  DVA is the second). CVA weights the EPE against the counterparty's default; DVA the ENE against
  our own; the unilateral case is `surviving=None`.
- **`CreditParties`** (both curves and both LGDs, §7's compelled bundling — it removes the commonest
  bilateral error, passing the two curves or the two LGDs in the wrong order) and
  **`BilateralAdjustment`** (`cva`, `dva`, with `bcva` a **property** so the net cannot drift from
  its parts; both legs are always reported, since the same small net can be two large offsetting
  legs or two negligible ones).
- **`bilateral_credit_adjustment`** and the pathwise **`mc_bilateral_adjustment`** (a
  `PricingResult` with its standard error, invariant 7), plus
  **`ExposureEstimate.negative_profile()`** for the ENE side.

### Oracles

- **The reduction doubles as the purity proof.** `λ_own → 0` gives `DVA = 0` and the CVA leg exactly
  T25's unilateral CVA. And T25's `unilateral_cva` routed through the extracted integral is
  **bit-identical** — verified against the pre-refactor code from git on four grids (5 to 2001
  points), every digit. `surviving=None` *skips* the multiplication rather than multiplying by 1.0,
  which is what keeps the operation order literally T25's and makes §4's stronger proof available
  instead of an identical-to-a-few-ULP argument.
- **The party-swap symmetry is the independent, non-telescoping anchor** — our DVA equals the
  counterparty's CVA against us, computed from a different composition of the pieces, and on
  simulated paths `ENE(V) = EPE(−V)` exercises the **positive**-part code path on a negated
  exposure. An ENE computed as `−mean(V)` rather than `mean(V⁻)`, or a swapped curve assignment,
  dies here and in no within-DVA check.
- **The first-to-default term is asserted by magnitude, not only sign** (§4's documented blind
  spot): our own default risk *strictly reduces* the CVA, and the term is measured at **2.7%** of
  the unilateral CVA on the fixture — percent, not epsilon.
- `BCVA = CVA − DVA` exactly; both legs non-negative; BCVA either sign; monotonicity in each party's
  hazard and LGD. The pathwise SE is asserted **not** to equal the quadrature sum of the two legs —
  CVA and DVA share paths, so the difference's error is not the sum, which is why the pathwise
  estimator is required rather than convenient.
- **Eight mutants, all killed**: wrong surviving curve, swapped defaulting/surviving, dropped
  surviving factor, flipped net sign, DVA fed the EPE, swapped LGDs, pathwise CVA dropping own
  survival, `negative_profile` returning the EPE.
- Non-degeneracy premises asserted: a genuinely **two-sided** exposure (EPE and ENE within 3× of
  each other, so DVA is not ~0), two **differing** hazards, and two **differing** LGDs.

### Open question raised, not ruled

Crépey conditions through the **bank survival measure** (`EY = (1−γ)⁻¹E*(Y°)`, eq 7 p.37) — his
first-to-default conditioning carries a `(1−γ)⁻¹` **normalisation**. This tranche builds the
market-standard **explicit survival factor**, the natural extension of T25/T26, neither of which
carries a normalisation. Both forms have the term present and both satisfy `BCVA = CVA − DVA`; they
differ by that factor, measured at **4.7%** of the CVA on the fixture. Which is the identity and
which the refinement is a design call reserved for Cowork by the tranche spec, and is raised as an
`Ask:` rather than settled here.

## [0.32.0] - 2026-11-09

Tranche 26b — **the `verify.py references` gate: the register, self-policed.** A standalone
governance tranche, run before T27. §7 floated this gate as *"proposed, not built, until the first
drift makes it earn its place"* — **five** drifts earned it.

### The finding that pulled it

Committing T26, one word of a Cowork edit (`CREPEY-2026 (worked-from)`) did not match
`docs/references.md`. Pulling that thread found five stale statuses, none caught by careful reading:
`CREPEY-2026` and `CAPINSKI-ZASTAWNIAK` still `in-hand` though T25 had built against them;
`JAECKEL-2002` **self-contradictory** (header `in-hand`, body `worked-from`, 18 citers);
`MCINERNEY-ZASTAWNIAK` `in-hand` with **10** citers and a body promising `worked-from` *"until T10
lands"* — sixteen tranches earlier; and `ACKLAM-2000` `in-hand` with its coefficients transcribed in
`distributions.py`. This is §9's re-audit target (3), *register accuracy*, failing by accident
between re-audits eleven tranches apart.

### Added

- **`verify.py references`** — three checks, keyed on the register's header status field and the
  `source:` citations `provenance` already walks: (1) every cited key **resolves** to an
  `in-hand`/`worked-from` status (never `referenced-by` — §7's no-reference-by-inference, never
  absent); (2) every key cited by ≥1 module is **`worked-from`**, the register's own vocabulary for
  *held and built/oracled against*; (3) every `worked-from` key has ≥1 **citer**.
- `references` added to the §9 gate list; charter §7 and the register's own note record the gate as
  built and name the five drifts.

### Oracles — seeded drift, mutation applied to a gate

- The pure core `check_reference_statuses(statuses, citations)` is **injectable**, and each real
  drift shape is reconstructed and asserted to be **rejected**: a cited key left `in-hand`; a cited
  key absent from the register; a cited key only `referenced-by`; an orphan `worked-from`.
  Extracting the core was necessary rather than stylistic — a gate whose only entry point reads the
  real repo can be asserted to pass today and never *shown to fail*.
- Two **legitimate** states asserted to pass: a held-but-unbuilt source at `in-hand` with no citer
  (exactly `CREPEY-2026`'s state between registration and T25), and a cited key at `worked-from`. A
  gate that blocked honest work is the other way to be useless.
- **The gate's first run on the then-current tree failed**, catching `MCINERNEY-ZASTAWNIAK` and
  `ACKLAM-2000` — two drifts that had survived a hand-correction pass three commits earlier. That is
  the argument for gating a discipline rather than restating it, and it is why this is not a gate
  asserting what was already true.

### Fixed

- `MCINERNEY-ZASTAWNIAK` and `ACKLAM-2000` statuses corrected to `worked-from`, each with the
  tranche and the built modules named (following the same correction to `JAECKEL-2002`,
  `CREPEY-2026` and `CAPINSKI-ZASTAWNIAK` at v0.31.0).

### Also fixed — the gate that floors the gates

- **`verify.py`'s `tests` gate discovered only `tests/L*`**, so this tranche's own `tests/tooling/`
  tests ran under plain `pytest` and passed while the gate counted 902 against a floor of 911 —
  present, green, and **never floored**. That is the "filter with no floor" defect `run_tests`'s own
  docstring warns about, occurring in its *discovery* rather than its `--layer` flag. Discovery is
  now total (layer dirs first, so the lowest layer still fails fast), extracted into
  `discover_test_dirs` so a test can reach it (§4's reachability rule), and pinned by three tests.
  Found because the floor failed on a tranche that *added* tests — the one circumstance that makes
  an unfloored directory visible.

### Scope fence

Status and citation **consistency** only. The gate does not check that a held PDF exists on disk
(they are gitignored and absent in CI, so the check could not fail there — this project's named
defect class), does not parse or fact-check the prose body, and does not verify a formula against
its source. That last is §4's per-tranche job and no gate can do it.

## [0.31.0] - 2026-11-08

Tranche 26 — **the Monte-Carlo exposure engine: CVA of a simulated netting set.** Second unit of the
XVA spine arc. T25 integrated an exposure given in closed form; this one produces an exposure by
simulation, which changes what the oracle can be — T25's closed form is now *our own code* on the
other side, so it validates the estimator and nothing more.

### Added

- **`risk/mc_exposure.py` (L5)** — `SimulatedExposure` (the value matrix, kept as paths),
  `ExposureEstimate` (per-date EPE/ENE as `MonteCarloEstimate`s), `mc_exposure_cva` (the pathwise
  CVA, returning a `PricingResult` with its `standard_error` — invariant 7),
  `swap_values_on_paths` (a forward-starting payer swap on simulated LMM forwards, in both the
  time-`t` and numéraire-relative forms) and `live_exposure_times`.
- **`foundation/time_measure.is_historical(pay, valuation_date)` (L0)** — invariant 6's boundary as
  one atom. All **eight** pay-boundary sites across the five engines (`cash`, `inflation`,
  `nominal_cap`, `capfloor`, `yoy`×4) now route through it, plus the exposure grid. **Register C10
  upgrades `per-module` → `guarded — structurally`**: a divergence between engines is no longer
  *expressible*. Extracted because a real consumer pulled it (§6), not speculatively.

### Oracles — two legs, because self-consistency telescopes

- **Leg A (the estimator)** — the simulated EPE against T25's closed form on a Gaussian mark, within
  `4·SE` at every date, with the error falling at the MC rate (RMS over the grid, since one date's
  error is a single noisy draw). Plus the MC CVA against T25's **analytic** CVA: a cross-product
  oracle, different exposure code each side.
- **Leg B (the process)** — the numéraire-relative swap value reproducing `V(0)/B(0,T_end)` at every
  observation date: **exact by construction, consulting T25 nowhere**. Deviations of 0.11/0.35/1.04
  SE against real path dispersion. This is what a wrong drift or measure cannot survive and what
  Leg A is structurally blind to.
- **The pathwise/profile identity** — the pathwise CVA mean equals T25's integral of the same EPE
  profile to 1e-12, on *both* the Gaussian and the simulated-swap exposures. The module claimed it;
  the suite now asserts it, which is what makes the reported standard error trustworthy.
- **Nine mutants, all killed.** Informatively, dropping the numéraire division died to **exactly one
  test — Leg B** — direct evidence that leg is load-bearing rather than redundant; dropping `DF(t)`
  died only to the T25 cross-check. Flipping the `is_historical` atom's `<=` fails six tests.
- Non-degeneracy premises asserted: non-zero exposure variance, more than one future date, a
  non-flat numéraire across the observed dates, and the swap exposure measurably **skewed** (so
  Leg A's Gaussian closed form provably cannot reach it).

### Design notes

- **The CVA is estimated pathwise, not by combining per-date standard errors.** Each path
  contributes its own `∫DF·V(t)⁺·dPD`. Quadrature-summing per-date errors would assume the dates
  independent when one path visits every one of them, understating the error — and the pathwise form
  satisfies invariant 7's correctness point directly, since the error comes from the same paths as
  the price.
- **The swap is observed only before its first reset, deliberately.** The martingale identity holds
  for a *fixed* claim; once a coupon has paid, the remaining swap is a different claim and
  `E[V/B_N] = V(0)/B_N(0)` is false. Observing a forward-starting swap keeps Leg B **exact** rather
  than approximate, and the engine **refuses** a later observation index rather than silently
  asserting something untrue. The exposure is still lognormal, which is the point of the leg.

## [0.30.0] - 2026-11-08

Tranche 25 — **unilateral CVA and the survival curve: the XVA spine opens.** First unit of the XVA
arc (`docs/xva_spine_arc.md`), and the first adjustment this tree has ever priced — it replaces the
archived predecessor's faked `risk/xva.py`, whose register rows named a CVA module that never
existed (pruned by tranche 24).

### Added

- **`market/survival_curve.py` (L1)** — `SurvivalCurve`, survival probability from a
  piecewise-constant hazard: `S(t) = exp(-cumulative_hazard(t))`, with `cumulative_hazard` and
  `default_density` (`lambda(t)S(t)`) as named surface rather than steps buried in the exponential.
  Market data, not a model: a hazard rate is quote-side credit, the same status as a zero rate.
  Source `CAPINSKI-ZASTAWNIAK` ch 2.
- **`risk/credit_valuation_adjustment.py` (L5)** — `expected_positive_exposure` (the Bachelier
  positive part `m*Phi(m/s) + s*phi(m/s)`), `ExposureProfile` (+ `.gaussian`), `unilateral_cva` and
  the extracted one-period step `unilateral_cva_one_period`. Crepey Prop 1 eq (17)'s
  `CVA = E[(1-J)P+]` under **exposure/default independence**, which is the simplification this
  tranche makes and the thing wrong-way risk would recover. L5's second consumer after Greeks.
- **A shared-clock guard (register `C12`)** — `unilateral_cva` refuses a survival and a discount
  curve on different `TimeMeasure`s. The profile's `t` is read on both, so two anchors would make
  one `t` denote two dates and the integral would mix them. The C7/C8 family, guarded rather than
  recorded; its test is mutation-proved (removing the guard fails it).
- **`DiscountCurve.df_at(t)`** — the discount factor at a year fraction, with `df(date)`
  delegating. A time-grid integral holds `t` and no date; the delegation preserves operation order,
  so the extraction is proved **bit-identical** rather than ULP-bounded (§4).

### Oracles

- The EPE against the test's **own numerical integral** (independent route) inside the regime where
  that route is accurate, and against scipy's `ndtr` composition at the **extreme ratios the
  profile actually visits** (`m/s` reaches 66), with relative tolerance — an absolute one is blind
  where the EPE is ~1e-24 (T20a's rule).
- The CVA against scipy quadrature of an **independently built integrand**, banded at 2e-04 from a
  measured 9.4e-05, **plus the second-order convergence rate** — the rate is what proves it is the
  same integral rather than merely a close number.
- Crepey's requirement oracles: **two separate zeros** (`LGD = 0` and `lambda = 0`), sign,
  monotonicity in both the hazard level and `LGD`, linearity in `LGD`, and the sandwich.
- The one-period step pinned **bit-identically** to a hand computation.
- **Six mutants, all killed** — dropped `LGD`, density replaced by the survival level, EPE replaced
  by the forward, flat hazard cumulation, dropped `DF(t)`, hazard segment off-by-one. The
  non-degeneracy premises (non-flat hazard, non-flat discount curve, `m/s = O(1)`) are each
  asserted, and each is what kills one of these.

### Fixed

- A **defect in this tranche's own test reference**, caught by the convergence rate rather than by a
  tolerance: the EPE quadrature integrated from `-m/s` over a fixed-width window, which truncates
  above the distribution's mass once the ratio is large — it returned ~0 where the true EPE was
  99.975. It made the CVA error *plateau* at 3e-04 instead of converging. The references now carry
  explicit validity regimes and are asserted to agree where both hold. §4's "a reference is code
  too", fourth occurrence.

## [0.29.0] - 2026-11-08

Tranche 24 — **the conventions register rebuilt to this tree** (re-audit #1 findings F-3.1, F-3.2,
F-2.1). Second and last remediation tranche; **re-audit entry #1 is complete**. Specced as docs-only
— and the honest re-derivation earned §2's optional guard and **found a second hole the audit had
not**.

### A second hole: the inflation cap/floor's valuation-date boundary, worth 41%
Re-deriving the invariant-6 register row meant establishing what the five engines actually agree on,
and the honest way to establish that is to flip the boundary and see who notices. `>` → `>=` **survives
in three of five**. The first instinct — "three engines have an untested boundary" — was **wrong for
two of them**: `nominal_cap` and `yoy` are **proved equivalent mutants**, prices bit-identical across
four schedules including the boundary, because a Hull-White caplet whose expiry has passed is worth
exactly zero so including or dropping such a period changes nothing. I nearly wrote a guard on that
false premise; measuring stopped me.

`capfloor` is the one that mattered. A period that has already paid has **both index levels known** —
the payoff is *determined*, not an expired option — so it carries real value, and including it moved
the price from **5,929.24 to 8,363.52, about +41%**. A genuine mispricing a wrong comparison would
have caused, with nothing to catch it. Closed by
`test_INVARIANT_6_excludes_a_period_PAYING_EXACTLY_on_the_valuation_date`.

### The row is `per-module`, not `guarded`, and that is the honest entry
A cross-module test comparing all five engines would catch **nothing** the per-engine boundary tests
do, while duplicating three market fixtures and claiming a guarantee that does not exist. What would
make C10 structurally guarded is what makes C9 guarded: **an atom**. Five engines inline
`pay > valuation_date`, so a divergence is expressible; extracting one `is_historical(...)` is recorded
as the row's upgrade trigger rather than done, since this was ruled a docs tranche.

### The register, and one of my own rows was overstated
C1–C6 described `risk/exposure.py`, `risk/xva.py`, `engine/linear.py`, `engine/equity.py`,
`models/sabr_model.py` and seven more modules that **do not exist here**. They are now a labelled
*archived* section (lessons only, not inventory), IDs **kept stable** because the handoffs, the audit
docs and CLAUDE.md cite them. The live register is C7–C11, and a check confirms every module it names
exists in `src/`.

**My new C11 row was wrong twice before commit**, the same failure mode as the rows it replaced:
"every L3 model carries a `MarketSnapshot`" (six do; six others are closed forms and parameter objects
that never read market data — that is the convention's *scope*, not exceptions), and "there is no
second place to pass a snapshot from" (false inside an engine — `inflation_leg_value`,
`resolved_index_level` and the calibration objectives all take one legitimately; the guarantee is at
the **entry point**, where every registered pricer is `(product, model)`). A register row is only worth
the grep behind it.

### Also
- `to_go_deeper`: "Six `crosscheck:` markers" → **23** real targets (of 45 headers; the other 22 are
  explicit `none — …` exemptions). "L5 is empty" → corrected: tranche 22 filled it; the *inflation*
  risk layer the route describes is still genuinely unbuilt.
- F-2.1 recorded: the driftless-forward terminal is inlined in `black_mc.py` and `sensitivities.py` —
  rule-of-two has fired, extract at the next touch of either.
- 821 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.28.0] - 2026-11-07

Tranche 23 — **the linker coupon accrual oracle** (re-audit #1 finding F-1.2). The first
remediation tranche from the first whole-tree re-audit. **Test-only: `src/` is unchanged** — the
production code was correct and only its coverage was wrong.

### The hole, reproduced before it was closed
Dropping the `ψ` accrual factor from `engine/inflation.py`'s linker coupon sum **survived all 818
tests**, which I confirmed independently from a clean bytecode state rather than taking the finding on
report. The term is load-bearing in production and was invisible to every oracle.

### The existing oracle computed a `ψ` and was still blind to it
`test_the_linker_matches_the_closed_form_with_accrual_fractions` evaluates
`psi = (pay - start).days / 365.0` over a fixture whose every period is exactly 365 days — so every
`ψ` it computes is `1.0` and `C·ψ = C`. Its name promises accrual fractions the fixture never
delivers. That is a sharper form of the charter's `τ = 1` trap than the existing examples: the test
does not merely *omit* the accrual, it visibly *handles* it, which is why twenty tranches of readers
and two review passes saw a covered term. **A fixture can defeat an oracle that looks correct line by
line.**

### The fixture, and why the accruals are unequal
Semi-annual periods of **182 and 183 days** on ACT/365F, so `ψ ≈ 0.4986` and `0.5014`. Two properties,
the second not required by the spec: **not 1** (which makes the accrual visible at all) and **not
equal to each other** (which kills a mutant replacing `ψ` by any single constant — a fixture of equal
non-unit accruals would have closed F-1.2 while leaving `ψ → 0.5` alive).

| mutant | before | after |
|---|---|---|
| `ψ` dropped (the F-1.2 hole) | **SURVIVED** | **KILLED** |
| `ψ → 1.0` | survived | **KILLED** |
| `ψ → 0.5` (a constant) | — | **KILLED** (needs the unequal accruals) |
| `ψ` applied to the redemption | — | **KILLED** |

### The non-degeneracy premise is itself a test
`test_the_SEMI_ANNUAL_fixtures_accruals_are_NOT_degenerate` asserts every accrual is more than 0.4
from 1 and that they are not all equal. This is the guard the tranche exists to install: a later
refactor **cannot** quietly flatten the schedule back to annual periods, which is precisely how the
hole survived. Without it the new oracle would keep passing while silently losing its power over `ψ`.

### Notes
- **No `Ask:` raised and no `src` change**: the non-degenerate oracle passed against unmodified
  production on the first run, so there is no real `ψ` bug — the cleanest outcome for a
  latent-coverage finding, stated plainly rather than left as an absence.
- F-1.2 was found only because the re-audit's sweep was extended past the first pass's domain — §4's
  own domain rule applied to the audit itself. Worth making the default for re-audit #2.
- 820 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.27.0] - 2026-11-07

Tranche 22 — **Greeks by simulation** (`JAECKEL-2002` ch 11), **the last Jäckel unit**. Three methods
— finite differencing with path recycling (§11.3), pathwise differentiation (§11.5), the likelihood
ratio (§11.6/§11.9.1) — all anchored on Black's analytic Greeks. `models/black.py` gains those;
`risk/sensitivities.py` is **the first L5 module**. **Nineteen mutations, all nineteen killed on the
first pass.** Ch 11 closes, and with it the nine-chapter spine.

### The anchor did not exist — the spec assumed a capability we never built
The spec names "Black's analytic Greeks (T9)" as the anchor twice. **There were none.** T9 built the
implied-vol *surface*; a grep of `src/` for any delta/gamma/vega definition found nothing. So the
tranche built its own anchor, and the consequence that matters is that the anchor needed **its own
oracle** rather than being assumed. The chain is now four independently checkable links: `black_price`
(verified since tranche 5) → its central difference → the analytic Greek → the Monte Carlo Greek.

Worth recording as a pattern: a spec can assert a *capability* as confidently as it asserts a number,
and this is the first time the missing thing was a whole function rather than a value.

### The analytic Greeks are proved as a LIMIT, not a point comparison
A first version compared each Greek to one central difference at `h = 1e-5` and **failed** deep
out-of-the-money by ~4e-6. The finite difference was the weaker side: varying the step, its gap falls
5.63e-2 → 5.60e-4 → 5.60e-6 → 5.60e-8 → 3.28e-10 — textbook `O(h^2)`, converging **onto** the analytic
value. So the formula was right and the reference was truncating by a hundred times the bound. The
oracle now asserts the **rate**, which proves the closed form is the *limit* of the numerical
derivative — a stronger claim, and one a wrong `d+` could not fake. Measured ratios 87 to 102 per
tenfold step. Gamma's floor is looser (a second difference divides by `h^2`), parameterised and stated
rather than hidden under one bound for both.

### The companion that earns the second method
Pathwise matching on a call is vacuous alone — finite differencing would do. What justifies a second
method is a payoff where pathwise is *wrong*: a digital's payoff is flat almost everywhere, so its
pathwise derivative is **exactly 0.0 on every path** while the true delta is strictly positive. The
oracle asserts the zero as a number against an analytic digital delta built for the purpose, with the
likelihood ratio holding there. **Pathwise is also refused for gamma**, because §11.5 refuses it — it
gives a first derivative per path, not a second, so the method raises rather than approximating.

### The variance ranking inverts the obvious guess
200,000 paths, at-the-money, SE as a percentage of the analytic value:

| Greek | finite difference | pathwise | likelihood ratio |
|---|---|---|---|
| delta | **0.25%** | **0.25%** | 0.56% |
| gamma | 4.35% | *refused* | **1.44%** |
| vega | **0.42%** | **0.42%** | 1.44% |

The likelihood ratio is 2-3x noisier for delta and vega — its weights carry `1/(sigma sqrt(dt))`, as
Jäckel's own caveat implies — but **3x sharper for gamma**, because a second finite difference divides
by `bump^2` and amplifies what it differences. And FD's gamma is visibly **off-centre** (55.47 against
49.62, 2.7 SE — inside the oracle's band but a truncation bias, not noise). So for gamma the likelihood
ratio is both sharper and better centred; the ranking is not a matter of taste.

### L5 is open, deliberately without a protocol
The charter's layer box has carried "L5 RISK & CAPITAL — on the engine + a Priceable protocol" since
tranche 1, and L5 has been empty for twenty-one tranches. **No `Priceable` protocol**: this module
reads Black's closed form and the MC machinery directly — one shape of consumer — so a protocol would
be an abstraction with a single implementation, which §6 forbids. It arrives at the second kind of
priceable that needs differentiating; the charter clause is half-satisfied and says so.

`BlackState` is the bundle §7's limit forced (`black_mc_vega` was six arguments). `years` is read only
by vega — not a carried-but-unread field, since `total_variance = sigma^2 * years` makes the horizon
what separates a one-year 20% option from a four-year 10% one.

### Notes
- The likelihood-ratio weights are adapted from spot to **forward**: Jäckel writes
  `S_T = S_0 exp((r-d-sigma^2/2)T + sigma sqrt(T) z)`; a forward is driftless under its own measure, so
  the weights carry over with `S_0 -> F_0` and `r-d -> 0`. Stated, because an unstated convention
  change is how a right formula becomes a wrong number.
- Eq 11.32 (the LR Vega) **is** printed in full for the single-horizon case, so the Acklam pattern did
  not bite here.
- 818 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.26.0] - 2026-11-07

Tranche 21 — **the Brownian bridge** (`JAECKEL-2002` §10.8.3, appendix §10.9.2), and Sobol wired into
the LMM path. `foundation/brownian_bridge.py` (L0) plus the engine's `Sampling` bundle. **Thirteen
mutations, all thirteen killed.** This **completes chapter 10**; only ch 11 (Greeks) remains before the
nine-chapter spine closes and the first whole-tree re-audit fires.

### The anchor is exact — no Monte Carlo in it at all
The spec asked for the covariance identity "to tolerance". It needs none. The bridge is a **linear** map
of its variates, so its matrix `A` is recovered exactly from unit basis vectors and "same Gaussian,
reordered" becomes `A A^T = min(t_i, t_j)` — the Wiener covariance — checked to machine precision at 2,
3, 4, 5, 8, **14** and 16 steps. There is no estimator, so there was no tolerance to choose. The same
holds one level down: the increment map's Gram matrix is exactly the identity.

14 steps is in the sweep because it is the source's own worked example (figure 10.4) and is **not** a
power of two — the case §10.8.3 singles out for its general "halve into two parts containing
approximately the same number of points" rule.

### A source misprint, faithful in both renderings
Appendix §10.9.2's eq **10.76** prints `W_tk = W_tj + sqrt(t_k - t_i) z`, contradicting eq 10.77 three
lines later (`z := (W_tk - W_ti)/sqrt(t_k - t_i)` forces `W_ti`); with `W_tj` the first increment is
counted twice. **Both renderings print it identically**, so transcription independence is powerless and
only the internal inconsistency catches it. It does not reach the build — eqs 10.33/10.34 are what is
implemented, and they were re-derived independently rather than read from the typo'd sketch. Worth
recording because §4 treats a derivation as an independent check on its result, and here the check
itself had to be repaired before it could be used.

### The claim holds, but only one half of it is assertable
Measured through the engine on an LMM caplet at 32 Sobol dimensions, the fitted exponents are
**pseudo-random -0.255, Sobol+incremental -0.616, Sobol+bridge -0.940** — the bridge nearly reaching
the theoretical -1, and at 16,384 paths **20x better than pseudo-random, 3x better than Sobol alone**.

But the spec asked for the pseudo-random comparison as a fitted exponent and **that is not assertable**:
one seed is one noisy realisation, and its slope swings from -0.637 to -1.056 depending only on where
the range ends — on the shorter range it *beats the bridge*, by luck. So bridge-vs-incremental is
compared on the **exponent** (both Sobol, hence deterministic) and the pseudo-random comparison on the
**RMS error over the range**: bridge/incremental 0.35, bridge/pseudo 0.17, identical to two decimals
across both ranges. Third tranche running in which a headline claim had to be restated in the form
measurable above its own noise.

### The null result needed averaging too, which is the subtler half
Under pseudo-random draws the bridge must change nothing — it is a reordering, so it gives a different
*sample*, not a better one. A first version of that oracle used one seed and measured the bridge **6x
better**: that looked like a result and was sampling luck. Averaged over twelve seeds at two path counts
the RMS ratio is **1.31** (and 0.92 / 1.26 / 1.29 under three other choices) — neutral, as the algebra
requires. **A null measured on one sample is not a measurement.**

### The ordering is the purpose, and no output property could see it
Twelve mutations died at once; the survivor was **depth-first instead of breadth-first subdivision**.
A real hole, not an equivalent mutant: it still places every point once, still brackets correctly, still
yields a valid Wiener path — so the exact covariance identity **cannot see it** — while breaking the
entire purpose, since the second variate would shape a small early region rather than the global
midpoint. Tranche 18's lesson in a new place. The distinguishing property is the bracket **width**:
non-increasing up to a slack of one (the source's own "approximately the same number of points"; a
first attempt demanded strict monotonicity and failed at 14 steps, where unequal halving legitimately
puts a width-3 bracket before a width-4 one).

### Notes
- `Sampling(source, construction, mirrored)` is the bundle §7's five-argument limit forced, and the
  right answer: three sampling choices now have real consumers (Sobol, the bridge, antithetic). It
  defaults to exactly tranche 16's behaviour, which is why all 784 existing tests passed unchanged.
- Jäckel's "all of the weighting coefficients can be precalculated" became
  `BrownianBridge.increment_matrix()`, so the engine reorders every path and factor in one `tensordot`
  rather than a Python loop per path — the cost pattern I walked into in tranche 19.
- **The Bermudan's own path still cannot use Sobol**: six forwards over eight steps is 48 dimensions,
  past the 32 Table 8.3 covers. So T20b's deferred acquisition is now load-bearing for a real consumer.
- 787 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.25.0] - 2026-11-07

Tranche 20c — **`norm_cdf` moves to `0.5·erfc(-x/sqrt2)`**, fixing the lower-tail cancellation T20a
found. A correctness refactor of our own L0, no external source. **OI-3 closed.** Six mutations, all
six killed — including a revert to the superseded form, which is the one that proves the new oracles
guard the fix.

### The defect was worse than the ledger recorded: 100%, not 6.1e-06
OI-3 cited 6.1e-06 relative error at `x = -7.03`. Pre-registering the oracle showed the worst case over
`[-9, 9]` is **1.0** — below about `x = -8.3` the old form returns **exactly 0.0**. At `x = -9` the true
probability is ~1.1e-19 and the function returned zero: not an inaccurate probability but a lost one.
The honest characterisation is not "degrades in the tail" but **"returns zero past a threshold"**, and
T20a's single measurement happened to land just above the collapse.

### Purity as a measured partition, because a flat ULP claim would have been false
The exact-arithmetic identity holds (`erfc(-y) == 1 + erf(y)`), but the bounded-agreement half cannot be
one tolerance: `x >= 0` agrees to **1 ULP** (bit-identical at 88% of a fine grid; both forms accurate to
~2e-16 against the reference), tens of ULP down to `x = -4`, then 3.9e-11 at `-5` and 1.3e-10 at `-6`
— and **that divergence is the OLD form's error**. It is *attributed* rather than noted: the oracle
requires the new form to track `scipy.special.ndtr` (`< 1e-14`) **and the old one not to** (`> 1e-11`),
because an unexplained divergence in a pure refactor is a drift.

The partition bounds the blast radius for the nineteen tranches that read `norm_cdf`: anything
evaluating `Phi` above about `x = -4` moves by at most tens of ULP, which no tolerance in the suite can
resolve. The suite stayed green unchanged at 760 tests — the corroboration this predicts, not the proof.

**My first claim was too strong and my own oracle caught it:** I asserted *bit*-identity on `x >= 0`
and it failed at `x = 1.2`. For positive `x` both routes produce a value near 1 and round differently
in the last bit — no cancellation, so no accuracy loss, but not identical. One ULP is the true claim.

### The workaround was removed, not left alongside
T20a added a private `erfc` sibling because `norm_cdf` could not be trusted in the tail. `norm_cdf`
**is** that form now, so the sibling is deleted and there is one CDF again (§6). Checked first that no
oracle depended on the two forms *differing* — none did; what needed updating was a comment explaining
why the reference avoided `norm_cdf`, a reason that no longer exists.

### A side finding: we resolve further than the reference
At `x = -38` scipy's `ndtr` underflows to **exactly 0.0** while `erfc(-x/sqrt2)/2` still returns a
subnormal (~2.9e-316). So the deep-tail bound loosens with depth (1e-13 to `x = -20`, 1e-12 to `-37`)
and that is **the reference's** accuracy, not ours — a first version asserted 1e-13 everywhere and
failed at `x = -30`. That is §4's "a reference is code too" — the note this arc added one tranche
earlier — biting in the tranche whose oracle *is* a reference comparison.

### Notes
- No candidate charter rule from this tranche, which is worth saying: every discipline it needed was
  already in §4. The charter earned its keep rather than growing.
- 761 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.24.0] - 2026-11-07

Tranche 20b — **Sobol' low-discrepancy numbers** (`JAECKEL-2002` §8.3), consuming tranche 20a's
inversion. `foundation/sobol.py` (L0, finance-free). **Fifteen mutations, all fifteen killed on the
first pass.** The fitted convergence exponent on a Black call is **-0.906** against pseudo-random's
**-0.481** (theory -1 and -0.5); at N = 32,768 the error is 8.4e-07 versus 2.7e-05 — 32x better.

### A ~350-number table, extracted twice by machine rather than read twice by eye
Table 8.3 gives 32 dimensions x (degree, primitive polynomial, ten initialisation numbers). §4's
dual-rendering rule exists for exactly this, but *reading* it twice by eye is the **weaker** method at
that scale — eye fatigue is correlated between passes in a way `pdftotext` is not. Both renderings
were parsed and reconciled mechanically (`-raw` reflows the table to one value per line and needed a
different parser, which is itself evidence the two are independent): **zero disagreements** across 32
rows, every exponent confirmed as `32 - l`. The shipped table was *generated* from the reconciled
extraction, never typed.

### What makes the table verifiable is a property the source states about it
T20a showed how weak an oracle over a transcribed table can be — there the refinement repaired small
coefficient errors. Here the caption hands over something sharper: *"Initialisation numbers providing
property A up to dimension 32."* Property A is testable — in any binary segment of length `2^d`
aligned to a multiple of `2^d`, each of the `2^d` sub-hypercubes holds **exactly one** point — and one
mistyped direction integer breaks a count. Verified over four aligned segments per dimension for
`d = 1..8`; beyond that a segment is more draws than is practical (`d = 32` is 4 billion), so the
higher dimensions rest on the source's two bit constraints over all 32x32 direction integers, which is
how the recurrence is verified where the book prints no values at all.

### The first segment is short by one, and telling that apart from a defect was the work
Property A failed first run at "15 of 16". Not a defect: the property is stated over the sequence
*including* `n = 0`, the origin, which we skip because `0.0` maps to `-inf` under `Phi^-1` and T20a's
guard refuses it. §8.3.4 says exactly what follows — the first segment "will have no point in the
sub-hypercube `[0, 1/2)^d`". The oracle now asserts the **exact** expected shape (first segment hits
`2^d - 1` cells, the missing one being the all-lower cell) rather than loosening a count, because "one
cell short" is also what a transcription error looks like.

### Dimension 1 is the construction at `g_k = 0`, not a special case
Its polynomial is degree 0, so there are zero freely-chosen initialisation numbers and eq 8.19 reads
`v_1l = (v_1l >> 0)` — vacuous, which the first implementation reported honestly as an `IndexError`.
What remains is `v_1l = 2^(b-l)`, the unit initialisation of eq 8.24, which is exactly what Table 8.3
prints for `k = 1`. So dimension 1 is the van der Corput sequence in base 2. Note the asymmetry the
source flags: unit initialisation is valid everywhere and "leads to surprisingly bad results ... and
should be avoided" — it is right for dimension 1 only because there the construction forces it.

### Where the advantage washes out (the spec's TO MEASURE item)
Discrepancy proxy at 1,024 points: **10-20x** better through `d = 5`, **1.8x** by `d = 8`, **1.0x —
gone — by `d = 24`**, and at `d = 32` Sobol is *worse* than pseudo-random. With the caveat that
matters: this is a statement about `N = 1,024`, not about Sobol — property A at `d = 32` needs `2^32`
draws, so the sequence has had no chance to equidistribute. The honest reading is that *at feasible
path counts the advantage is a low-dimension phenomenon*, which is precisely what tranche 21's
Brownian bridge exists to exploit. In 1-D the proxy is **exactly zero** (with `N = 2^k` Sobol hits
every dyadic box), which crashed the first measurement on a division — and is why the spec forbids
oracling discrepancy in 1-D alone.

### Notes
- **Sobol is not wired into the LMM path engine.** An LMM path needs one dimension per forward per
  step (6 forwards x 8 steps = 48), past where the advantage has washed out at feasible path counts;
  wiring it before the Brownian bridge would add surface that does not pay. Left for T21.
- Dimension 33 is **refused**: the source's polynomials beyond Table 8.3 are "on the accompanying CD"
  we do not hold — the third time this book has sent a reader to something it does not contain.
- 756 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.23.0] - 2026-11-07

Tranche 20a — **the inverse normal CDF**, closing the inversion tranche 15 deliberately deferred.
`ACKLAM-2000` registered `in-hand` (acquired by Bernardo) — the algorithm `JAECKEL-2002` §9.3
recommends and prints nowhere, which is what stopped T15. Acklam's rational approximation plus one
Halley step against an exact CDF: worst miss **7.66e-16** against an independent implementation,
sub-ULP. **Sobol is NOT built — it is T20b** (see `docs/tranches/20_inversion_sobol.md` §0).

### Our `norm_cdf` is relatively inaccurate in the lower tail, by nine orders of magnitude
The refinement needs `Phi(x)`, and using our own `norm_cdf` broke the deep tail: it computes
`(1 + erf(x/sqrt2))/2`, and as `erf -> -1` the sum **cancels catastrophically**. Measured against an
independent high-precision CDF at `x = -7.03`: ours **6.1e-06** relative, `erfc(-x/sqrt2)/2`
**7.4e-15**. Acklam prints both forms side by side ("use this line ... OR this; not both") and says
he used Cody's `erfc` — this is why. The refinement therefore uses `erfc`, and the upper half is
mapped onto the lower by the exact oddness of `Phi^-1`, so the CDF is only evaluated where it is
accurate.

`norm_cdf` itself is **raised, not patched**: it has shipped since tranche 5, nineteen tranches of
oracles read it, and the change is not bit-identical. Worth noting why its own oracle missed this —
it asserts "tails -> 0/1", an **absolute** claim that passes however bad the relative error gets.
§4's domain rule, found in a function green for fifteen tranches.

### Three times the reference was the weaker side
A root-find built on `norm_cdf` disagreed with the implementation by 3.9e-07 in the deep tail — and
**ours was right**, the reference carried the cancellation. At `p = 1 - 1e-12` the reference could not
resolve a root at all (`Phi` is flat at 1 in double precision), so it became unresolvable before the
code became wrong. At `p = 0.999` it missed an independent implementation by 3.74e-15 while ours was
**bit-exact**. The primary cross-check became `scipy.special.ndtri` — a different algorithm and a
different provider, exactly the arrangement tranche 11 set up by moving `distributions.py` off scipy
"so the oracle and the implementation no longer share a provider". The lesson: *a reference is not the
authority merely because a test calls it one.*

### Halley's third-order term is not measurable from Acklam's start
From a start good to 1.15e-9, Newton squares the error to ~1e-18 — below double precision — so Halley
and Newton **land on the same answer**: worst gap 4.44e-16, machine epsilon. A first version of the
oracle asserted Halley was measurably better and failed with `4.44e-16 < 4.44e-16`. The denominator is
still transcribed (the source specifies it) and its necessity is shown from a deliberately degraded
start, where cubic and quadratic convergence separate — otherwise it would be unfalsifiable surface.

### The oddness oracle could only be exact where the inputs allow it
`Phi^-1(1-p) = -Phi^-1(p)` is exact in arithmetic, but testing it needs `1 - (1 - p)` to return `p`,
which in binary holds only for **dyadic** `p`. At `p = 1e-10` only about four digits survive, so a
first version measured asymmetry in the **inputs** rather than the function. Split: exact on dyadics,
elsewhere bounded by each input's own measured representation loss.

### Mutation: sixteen run, and the four survivors are the resolution rather than holes
The Halley refinement **repairs** small coefficient errors, so the refined output is nearly
indifferent to the transcription. Two survivors were proved **bit-identical** (a 1e-14 coefficient
change; a 1e-4 break-point change — which is why Acklam put the break-point where both branches
agree). The only oracle with grip is the published bound on the *unrefined* approximation, measured
at **1.1288e-09 against a published 1.15e-09 — 98% of it** — with a resolution of ~20 ULP per
coefficient; a 4.5e-9 relative error moves it to 4.40e-08, 38x the bound. An oracle proving the bound
**can fail** was added, since otherwise the one check guarding 21 magic numbers would be unverified.

A wrong diagnosis on the way, recorded because it was stated confidently: the four survivors were
exactly the length-preserving edits, so I concluded Python's bytecode cache had run stale code and
reported false survivors. It had not — `-B` with a cleared `__pycache__` gave the same four, and a
direct probe confirmed the mutations execute. The correlation with edit length was coincidence.

### Notes
- `inverse_norm_cdf` refuses the closed interval: `Phi^-1(0)` is `-inf`, not a large number. This is
  where tranche 15's removed redraw loop relocated to — that branch was unreachable from any test,
  and the domain check belongs where a test can call it with `0.0` directly.
- 698 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.22.0] - 2026-11-05

Tranche 19 — **variance reduction I**: antithetic sampling (`JAECKEL-2002` §10.1) and control
variates (§10.3). First unit of the Jäckel completion arc. Both reducers at L0
(`foundation/mc_estimator.py`, finance-free), both wired at L4 — antithetic as a mirrored-draw mode on
the path engine, the control variate into the Bermudan pricer. **Sixteen mutations, all sixteen
killed on the first pass.**

### The spec's headline was refuted: the Bermudan sandwich is structural, not sampling error
The spec expected the control variate to narrow tranche 18's 15.75% bracket. It cannot. Measured over
a **64x path increase**, the width holds at 15.19% / 15.75% / 15.08% / 15.39% while the standard error
falls 151.25 -> 18.75 — a factor of 8 across 64x paths, the `-1/2` rate exactly. Measured in standard
errors the gap *grows* from 5.9 to 48.8. The width is the perfect-foresight bias plus the policy's
suboptimality, both structural, and tranche 18's own numbers already implied it (the gap was 12 SE
wide, not one). So the deliverable is **sharper error bars**, not a narrower bracket — and the oracle
now asserts that the reduction shrinks the SE and leaves the bracket *invariant*, which is itself the
check, since a reduction that moved the bracket would be introducing bias.

### The spec's chosen control was biased; an exactly-centred one exists and is better
Eq 10.8 is unbiased *only* because `g*` is the **exact** `E[g]`. The spec named the Rebonato swaption,
whose mean is an approximation (§5.11: the LMM swap rate is not lognormal) — centring on it injects
the ~0.5% freezing error as a permanent bias, which is what the spec's own oracle 1 forbids.

The alternative needs no model result at all: under the `P_N`-forward measure every
asset-over-numeraire ratio is a **martingale**, so the numeraire-relative annuity `A/B_N` has
expectation `A(0)/B_N(0)` exactly, by construction. Measured correlation with the Bermudan payoff
**+0.7368**, a **54% variance reduction** (SE x 0.676), zero bias, with the martingale property
verified (simulated mean within 0.66 SE of the analytic value). The intrinsic at the first exercise
date correlates marginally better (0.7444) but has no closed-form mean, so it is not a control.

### The constant-control fallback is the finance, and it saved tranche 18's sharpest oracle
For a single-period residual swap `A/B_N` collapses to exactly `tau` on every path — the identity
behind tranche 17's one-period reduction — so the control has zero variance and `beta* = Cov/V[g]` is
undefined. A constant control is the *absence* of one, and the engine falls back to the plain
estimator. This is also how the issue surfaced: four tranche-18 oracles went red on the first wiring,
because T18's oracle 1 demands a one-exercise-date Bermudan match the T17 European **bit-identically**
and reducing one side broke it. Widening that back into an SE band would have undone the power T18
gained — it was precisely an 8-11%-resolving band that hid a 1.2% defect there.

### Jäckel does not print the variance factor
Eqs 10.7/10.8/10.9 are printed and both renderings agree exactly. The factor `V[v](1 - rho^2)` is
**not in the source** — the second time this book has sent us to a formula it does not print (after
T15's inverse CDF). It is *derived* from 10.9 in two lines and asserted as a **theorem**, verified to
2e-17 against a synthetic pair whose correlation is fixed by construction. §10.3 also offers a third
route for `beta` the spec's framing omitted — a pilot simulation, which is both optimal and provably
unbiased; `control_variate_estimate` accepts an explicit `beta` so a caller can supply one.

### Notes
- Two of my own test errors, caught by execution: "antithetic has exactly zero SE on a linear payoff"
  is zero in R and **1.5e-18** in floating point (the summands round, not the estimator), and I had
  the centring-bias **sign** backwards — eq 10.8 adds `beta(g* - g)`, so raising `g*` raises the mean.
- A process error worth recording: I backgrounded `verify.py slow` and then ran the full suite and
  pyright on top of it. The gate reported a file at **690.6s** against a true 1.75s — a 400x artefact
  of CPU contention I created. A timing gate's reading is only valid when nothing else competes.
- Three `# noqa: E731` suppressions were removed rather than ledgered (lambdas became named
  functions), and a 6-argument test helper I introduced in T17 was bundled to satisfy §7.
- 674 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.21.0] - 2026-11-05

Tranche 18 — the **Bermudan swaption under the LMM**, Jäckel's headline instrument, priced inside a
sandwich because it has no closed form. Third and final unit of the BGM/J arc (T16 → T17 → T18), each
anchored on the one before, with tranche 17's caplet→Black exact anchor underneath all of it.
`models/exercise_boundary.py` (L3), `products/bermudan_swaption.py` (L2),
`engine/bermudan_swaption_mc.py` (L4), and the path widening in `engine/mc_path_engine.py`.
**Eighteen mutations, all eighteen killed.**

### The source is internally inconsistent — eq 12.38 contradicts eq 12.39
Both renderings of eq 12.38 print `+ p_i3`. But eq 12.39 pins the final slice to `(0, 0, K)`, and
§12.7 states what that must mean: *"at the very last exercise opportunity we have exact knowledge if
exercise is optimal, namely when the residual swap is in the money."* At the last date the residual
swap is one period, so its swap rate IS `f_(n-1)`; in-the-money for a payer is `f > K`. The printed
sign gives `f + K > 0` — true for every positive rate, so it exercises a deep out-of-the-money payer,
and it misclassifies every OTM payer across seven forwards and both directions. Ruled by the finance
per §4: 12.39 carries exact financial content, 12.38's sign convention is arbitrary.

**And the arbitrariness explains how the error survives publication.** At every slice except the last,
`p_i3` is a free fitted parameter, so the optimiser simply converges to `-p_i3` and the two
conventions describe the same family of boundaries. The inconsistency is invisible wherever the
parameters are trained and decisive only at the one slice where they are pinned. Both facts are
oracled, and the printed variant is asserted to **fail**, so the ruling is proved necessary rather
than preferred. Recorded in `docs/references.md` beside the five found in `BERHAN-2012` and the one in
`MCINERNEY-ZASTAWNIAK`. No ordinal claimed: `BERHAN-2012`'s 4.1.3 and 4.2.2 are internal
inconsistencies as well, so "the second" would have been wrong.

### Jäckel's prescribed optimiser cannot train Jäckel's objective
§12.8 says to use BFGS. **BFGS makes no progress at all** on eq 12.41, which averages an
indicator-selected quantity and is therefore piecewise constant in `p`. Measured at the first training
slice: start (the European policy) `U = 0.00290435`; BFGS `0.00290435` at **iteration 0**; Powell
`0.00363794`; Nelder-Mead `0.00374813`; attainable ceiling `0.00420242`. So BFGS captures 0% of the
available gain and Nelder-Mead 65%. The finite-difference gradient is exactly zero at `h = 1e-8` and
`1e-6`, non-zero only at `1e-4` — what a gradient method sees is a step-size artefact, not a
derivative. `train_slice` uses Nelder-Mead and the deviation is **oracled**: a test asserts BFGS
leaves the objective untouched while the derivative-free method improves it.

### The sandwich, and what it resolves
A 6-non-call-2 struck at the forward swap rate, 4,000 training / 8,000 pricing paths: exercisable only
at its last date 1,484.64; **trained policy (lower bound) 5,860.27**; **perfect foresight (upper
bound) 6,783.50**. The early-exercise premium is resolved at 14x the combined standard error. The
**sandwich is 15.75% wide** and that IS the tranche's resolution, reported rather than assumed tight.
In-sample bias measured at +1.12%, which is Jäckel's "slight upward bias" and the reason the
train/price split is a correctness requirement.

**One honest negative:** the lower bound does **not** improve monotonically with training paths —
500 → 5,890.09, 1,000 → 5,862.98, 2,000 → 5,849.68, 4,000 → 5,860.27, 8,000 → 5,880.16, a ~0.7%
wobble with no trend. The three-parameter boundary is already at its capacity for this trade, so extra
training moves the fitted boundary within a plateau rather than improving it. The place to spend
effort is a richer boundary, not more paths.

### Five mutations survived the first pass, and one was this tranche repeating tranche 17's lesson
A dropped numéraire division in the annuity hid inside oracle 1's `4 x combined SE` band, which at
these path counts resolves only **8-11%** of the price — exactly the trap tranche 17 ratified into §4,
walked into one tranche later. The fix was not more paths but noticing the comparison can be **exact**:
after this tranche's widening a one-date Bermudan's schedule *is* `PathGrid(T, steps)`, so both engines
consume the same stream; oracle 1 now asserts bit-identical PV and standard error.

The other four were all pieces of §12.8's *procedure* — the induction direction, the continuation
roll, the `intrinsic > 0` guard, and the choice of second coordinate (`s_{i+1}` vs `s_i`, §12.6's
entire finding) — and **no price-level bound could see any of them**, because a valid-but-wrong
procedure still satisfies the output bounds. Each is now pinned by a pointwise check on a named step,
which is why `roll_continuations`, `train_exercise_policy` and `ExerciseSlices` were extracted from
private helpers into named, oracled surface.

### The engine widening
`simulate_forward_paths` is now the single definition of eq 13.2, with `simulate_terminal_forwards` a
last-slice read of it — the rule-of-two consolidation tranche 17 deferred to this tranche's real second
consumer. Proved **bit-identically**, the right bar because the refactor reorders nothing: a
one-segment schedule reproduces the old terminal result, and a two-segment schedule reaches the same
endpoint, which also proves segmenting does not disturb the random stream.

### Notes
- The non-recombining tree (ch 13) is **not** built, per the spec: Jäckel uses it to *discover* the
  coordinate pair, and he states the pair he settles on. Recorded as a `to_go_deeper` route.
- 657 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.20.1] - 2026-11-05

**§3 invariant 7 implemented** — *an estimate carries its error*. Ruled by Cowork as tranche 17's
finding 5 and accepted by Bernardo the same day; `OPEN.md` OI-2 ledgered the gap between the charter
and the code for exactly as long as it existed, and is now closed.

`PricingResult` gains `standard_error: float | None = None`. `price_european_swaption` sets it,
scaled by the same deterministic `discount x notional` that scales the PV, and the banded oracles
read `out.standard_error` instead of re-simulating to recover a number the engine already had.

`None` is load-bearing and is not `0.0`: zero would claim a perfectly precise simulation, while
`None` says *this is not an estimate*. Exact pricers inherit it from the default, so no other engine
was touched — the field rides along only where it is meaningful, which is the half of the invariant
that keeps it from becoming surface on sixteen engines that will never set it.

### One part of Cowork's build note did not apply
It said "the caplet and swaption pricers set it". `mc_caplet_price` returns a `MonteCarloEstimate`,
not a `PricingResult`, and that type has carried `mean`, `standard_error` and `n_samples` since
tranche 15 — its docstring already states the invariant's principle ("an MC engine that returned only
a price would make every downstream oracle invent a tolerance"). The caplet pricer already honoured
invariant 7 by a different route and needed no change; converting it to return a `PricingResult` would
be a design decision nobody ruled, so it was not done.

### Notes
- `_estimate_from_paths` survives in the oracle with a narrower and better purpose: an independent
  recomputation of the payoff on the same paths, pinning the engine's mean *and* its error bar,
  rather than supplying a number the engine could have been wrong about.
- Seventeen mutations: the thirteen tranche-17 mutants behave exactly as before (twelve killed, the
  transposed-rho equivalent mutant still proved bit-identical), and four new mutants on the error
  bar's scaling — unscaled, zeroed, left `None`, and double-scaled — all die.
- 606 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.20.0] - 2026-11-05

Tranche 17 — the **European swaption under the LMM**, anchored by an exact reduction and a bounded
approximation. Second of the three-unit BGM/J arc. `models/swap_rate_vol.py` (L3, Rebonato's
approximation), `products/swaption.py` (L2), `engine/swaption_mc.py` (L4), plus the correlation
widened from a scalar to a full matrix — the widening tranche 16 said would arrive "with the unit that
needs one", and the swaption is that unit.

### The anchor: a one-period swaption IS a caplet, exactly
A one-period swap's rate is exactly its forward and its weight exactly 1, so Rebonato's vol collapses
to `v = sigma_i` to 1e-15 at every index. This engine and tranche 16's caplet engine then agree to
**1e-12** on one seed after a deterministic scaling — a cross-product oracle, different code each
side, no Monte Carlo error in the comparison. It fails under any error in the annuity, the swap rate,
the numeraire bond or the discount index, and in mutation it did.

### The cross-source independence the spec counted on does not exist
The spec specified the swap-rate-vol approximation as transcribed from both books and reconciled —
"the strongest independence we have". Read in both, they are **not one formula**. Both write the
variance as `Z C Z^T`; McInerney eq 5.40 uses the **frozen** weight `w_k F_k / S`, Jaeckel eqs
12.18/12.21 the **exact** derivative `(dSR/df_k)(f_k/SR)`. Jaeckel's is McInerney's first term plus
the two terms arising because the weights themselves depend on the forwards — and McInerney's own
Remark 5.13 says 5.40 can be improved by not freezing them. Two approximations of different order,
differing by **0.45%** in vol, not two renderings of one. Claiming the independence would have been
false; Jaeckel's form is kept as a companion whose gap is asserted small *and non-zero*. The signature
that they are different objects: `sum Z_frozen = 1` exactly, while `sum Z_exact` is 0.9955.

### A non-uniform fixture can still be degenerate
Two of thirteen mutations survived all 603 tests, both because of fixtures rather than code, and
neither is the flat-`tau` trap the charter already names. A correlation `exp(-beta|i-j|)` is
non-uniform yet **translation-invariant**, so `rho[first+m][first+l]` equals `rho[m][l]` and a mutant
dropping the span offset is invisible; all-`0.5` accruals make `accruals[:count]` and
`accruals[first:first+count]` the same slice, hiding the offset a second time. Jaeckel eq 12.12
decays in the actual times `t_i`, not index distance — so the faithful reading is also the
non-degenerate one. Sixth appearance of the trap, first where the degeneracy is *uniformity across a
slice* rather than a specific value. A third survivor — transposing `rho` — is a proved equivalent
mutant: symmetry is enforced at construction, and the vol is bit-identical across seven spans.

### An oracle claimed a power it did not have, and its own power assertion caught it
The bounded cross-check first asserted agreement inside one flat 3% band, called that "the freezing
slack", and claimed a 10% vol error would fall outside. It **failed**, and both halves were wrong.
The freezing error measured at 200,000 paths is 0.15%-0.96%, settling near 0.5%; the ~2.3% seen at the
committed path count is mostly Monte Carlo error, so one flat percentage was standing in for two
different errors and would have let a real 2% defect pass as noise. And the band had no power at all:
at 6,000 paths it resolves only a ~20% vol error. Measured resolution 20% at 6,000 paths, 10% at
12,000, 5% at 25,000. The oracle is now banded as `4 SE + the measured freezing slack` with the SE
measured from the paths the engine consumes, split into a breadth check whose coarse resolution is
stated and a dedicated power test at the count where a 10% error provably falls outside.

### Oracle 5's specced instrument does not generalise — replaced by something stronger
Tranche 16's two-numeraire identity worked because only the *drift* differed, so the diffusion
cancelled to the last bit. `rho` enters the **factorisation**, so two models differing in `rho` give
genuinely different paths and nothing cancels; the price-level sensitivity also cannot be tightly
bounded, since differencing amplifies relative error past 100% of the move. Replaced by a direct pin:
the realised covariance of the simulated log-increments must reproduce `sigma_i sigma_j rho_ij dt` for
**every pair** — all 36 entries individually, where a price pins one weighted scalar. Worst pairwise
error 3.2% at 20,000 paths, 1.66% at 80,000, halving as `sqrt(N)` quadruples.

### Notes
- The engine's payoff is vectorised over paths. The first version walked each path in Python while
  consuming an already-vectorised simulation — the failure the `slow` gate's own docstring names.
- 604 tests, all eight gates green, ruff and pyright clean, zero suppressions.

## [0.19.0] - 2026-11-04

Tranche 16 — the **LIBOR market model evolved by Monte Carlo**, anchored on our own Black caplet.
First of the three-unit BGM/J arc (`JAECKEL-2002` ch 12), and the only one with a closed-form anchor.
Three pieces: the pseudo-square-root at L0, the LMM at L3, the path engine at L4 — the last extracted
now because the LMM is the **second** consumer of the MC machinery (rule-of-two).

### The drift's magnitude was unconstrained, and thirteen green oracles did not see it
Three mutations of eq 12.8 survived all 550 tests: dropping the drift's `dt` factor (an 8x drift),
feeding the drift the raw forwards instead of the `(f tau)/(1 + f tau)·sigma` ratios (5x), and dropping
that denominator (1.04x). The cause was structural. Every drift oracle asserted the drift was
**present** (the paired-numeraire test, at 297-372 standard errors) or correctly **signed** — neither
constrains its size. And the closed-form anchor runs under the measure where the target forward's drift
is **exactly zero**: the exactness that makes the anchor sharp is the same reason it is blind to the
drift. A complete set of assertions about the wrong quantity.

The fix removes Monte Carlo error instead of fighting it. Two models differing only in numeraire index,
one step, one seed consume the same normals from the same state, so the diffusion and `-c_ii/2` terms
cancel to the last bit and `log f_i^A - log f_i^B = (mu_i^A - mu_i^B)·dt` **exactly**, every path.
Agreement with a fresh transcription of eq 12.8 is 1e-13 relative, cross-path spread 7e-16 — precision
no path count buys, and what lets it see the 3.8% mutant. The step is a **half** year deliberately: at
`dt = 1.0` the `dt` scaling is invisible, which is how that mutant survived. All eight now die.

### The new oracle found an error in its own reference before it found one in the code
Its transcription used the off-diagonal `rho = 0.80` for every pair, including `rho_ii = 1`. Forward 0
agreed to 1e-13 and forwards 1-3 disagreed in the third digit — and the **asymmetry located the bug**:
forward 0 is the only index whose drift sums never reach the diagonal. The model was right. Recorded
because it is the dual of the usual worry: an independent transcription can introduce a fresh error
rather than repeat the implementation's, and the disagreement *pattern* is what says which side is wrong.

### Added
- `foundation/linalg.py` — the pseudo-square-root (12.4) with spectral ordering, and the m-factor
  rescaling (12.6/12.7) making `SUM_j a_ij^2 = c_ii` hold **exactly** at any factor count.
- `models/libor_market_model.py` — `LmmTenor`, `LiborMarketModel`, eq 12.8 split into `drift_matrix`
  (its constant part) and `state_ratios` (its state-dependent part) so the scalar `drift_at` and the
  vectorised engine share **one** definition rather than the engine re-deriving the formula.
- `engine/mc_path_engine.py` — eq 13.2's log-Euler step, vectorised over paths.

### Notes
- `black_mc` is **not** folded onto the new engine. The spec allowed it only if bit-identical; it is
  not (single step to expiry vs a stepped grid, so different draws in a different order), so it stays.
- The engine oracle file went 8.4s to 1.2s — vectorisation then caching the deterministic
  simulations — against the `slow` gate's 10s ceiling, which CI meets roughly 4x slower.
- 554 tests, all eight gates green.

## [0.18.0] - 2026-11-04

Tranche 15 — the Monte Carlo spine. **Project two's first unit**, from `JAECKEL-2002`, and the root
of the MC dependency graph: a seeded source, normal variates, an estimator with its standard error,
and a simulated price checked against our own closed form.

### A hard stop at the source, and the ruling that resolved it
The spec said to transcribe `Φ⁻¹` "from Jäckel ch 9". **Jäckel does not print it.** §9.3 *recommends*
Acklam's and Moro's interpolation formulas, §2.3 says the implementation is *"included on the
accompanying CD"* we do not have, and a grep of all 235 pages for either algorithm's characteristic
coefficients returns nothing. Both works are `referenced-by` only, so §7 forbids citing them.

Ruled (Cowork): **trigonometric Box-Muller, eq 9.6** — printed in full in the held source, **exact**
rather than an approximation, and transcribable today. `scipy.stats.norm.ppf` was rejected on the
grounds that trusted-and-pinned is the fallback for a component *no held source specifies* (PCG64's
bits), not a first choice when a held source gives an exact method. Acklam is deferred to the Sobol
unit, where a monotone transform genuinely earns its keep.

### Added
- `foundation/rng.py` — `UniformSource`, numpy `Generator(PCG64(seed))` **wrapped so no global or
  default RNG is reachable**, with the first five draws pinned as the cross-platform contract and the
  global numpy state proved untouched after 10,000 draws. PCG64's bits are *trusted, not transcribed*
  — it postdates Jäckel — which is the §4 boundary drawn explicitly.
- `foundation/variates.py` — Box-Muller eq 9.6. **The anchor oracle is the exact identity
  `x² + y² = −2 ln u`**, which carries no sampling noise; the N(0,1) moments and pair-independence are
  its companions, with the bands *measured* from the sample size. A wrong angle (`πv` for `2πv`) is
  shown to satisfy the identity but fail the pointwise transcription — §4's two rules meeting.
- `foundation/mc_estimator.py` — mean and `σ̂/√N`, with the `N−1` divisor and the `√N` each pinned by
  the two wrong variants being asserted to differ.
- `models/black_mc.py` — the forward simulated as the driftless lognormal martingale it is, averaged
  into a price **with its standard error**. 12 oracles: within four *measured* SEs of our
  `black_price` (with a companion forbidding a vacuously wide band), the **−½ convergence rate**
  fitted over four path counts, unbiasedness across forty seeds, byte-identical at a fixed seed,
  **exact** at zero variance, and the martingale `E[F_T] = F₀` as a payoff-independent check on the
  `−V²/2` term.

### Three of my own errors, one found only by mutation
My SE-scaling oracle asserted `SE(4N) = SE(N)/2`; with the `N−1` divisor the exact law is
`1/√(N−1)` and the inverse-root-N form is only asymptotic — replaced with the closed form, which
pins the divisor and the `√N` together. My asymptotic check asserted `≈2.0` at `rel=1e-3` where the
true ratio is `√(1023/255) = 2.00294`; now exact.

And **mutation found an unreachable branch**: the source originally redrew endpoints to deliver the
open interval `Φ⁻¹` needs, but an exact `0.0` has probability ~2⁻⁵³ — so the branch was unoracled, a
mutant breaking it passed all 468 tests, and had it ever been reached it would have *hung*. Deleted;
the domain guard moved to `standard_normals`, where a test can call it with `0.0`. Charter §4 gains
this as a **third survivor kind** — unreachable code, distinct from both a hole and an equivalent
mutant.

### Changed
- `TEST_FLOOR` 453 → 490.

## [0.17.0] - 2026-11-04

Tranche 14 — Market Model II's calibration (BERHAN-2012 §5.4). **This closes the book**: ch 1–5 are
built, leaving only §1.3's risk material outside. Checkpoint: `docs/checkpoints/CP_inflation_book_complete.md`.

### §5.4.1 is an identity; §5.4.2's bootstrap is structural
`σn,i` **is** the quoted Black ATM cap vol — an exact round-trip, not a residual (contrast §5.3.1,
where JY needs a Hull-White cap engine and a real two-parameter fit). And the 1-Year-Floor bootstrap
works for a statable reason: the first period's lead is zero, so `Di` vanishes and `V²` collapses to
`σ_I,1²·(T₁−T₀)` — one unknown, one price, a plain Black inversion with no solver. Both the structure
and the bootstrap's blindness to every later parameter are oracled.

### The thesis's expiry claim is almost right, and the exception is the finding
*"Since each expiry has it's own set of parameters, there is no need to restrict the number of
contracts in the expiry dimension."* Measured: **every index vol recovers to ~1e-16 except the
terminal one**, which from a low start lands at **0.002141 against a truth of 0.020000 — 89% wrong
at a residual of 8.6e-17.**

`σ_I,i` enters its period's variance *quadratically* and its drift *linearly and negatively*, so the
caplet price is **non-monotone** in it (vertex at `ρ_II·σ_I,i−1·lead/(lead+span)`, predicted 0.0115,
located by sweep at 0.0115). Two vols give one price. Every vol but the terminal one appears in
**two** periods and the second appearance picks the root; the terminal date appears once.

**And whether a fit lands on the wrong root is start-dependent** — a high start recovers the truth, a
low one falls into the conjugate, and **both report ~1e-17**. So a single-start MM-II calibration is
unsafe at the terminal point and no residual will say so. Multi-start is load-bearing, not
decoration.

### The cross-model comparison the arc was for
Tranche 13 put the three models' *pricing* gap at 7.7e-09 to 3.7e-05. The difference that matters:
**JY leaves three global parameters unidentifiable** (one at the wrong sign), **MM-II leaves one
locally ambiguous point.** So the thesis's *"this approach seems the most promising"* holds — and the
reason is identifiability, not price, which the thesis does not say.

### Added
- `engine/mmii_calibration.py` — `nominal_vol_from_black_atm` (the §5.4.1 identity, named and
  oracled because it is a claim about the model), `index_vol_from_first_floor` (the bootstrap by
  Black inversion), `calibrate_mmii_to_atm_floors`, `MMIICalibration`.
- 23 oracles, including the source's **signed** skew direction (fitted at ATM, MM-II underprices OTM
  and overprices ITM) with the companion that its ATM fit is essentially exact — two distinct axes.

### Changed — MM-II's index vols became a term structure, bit-identically
`MMIIParams.index_vols` is now keyed by the forward index's own maturity, so a period reads two
adjacent points — which is what §5.4.2's four-parameter-per-expiry set requires. A scalar would have
collapsed two of the four and made any identifiability finding an artefact of our simplification.
`MMIIParams.flat()` is the tranche 13 shape, verified **bit-identical** across the six values that
tranche measured.

### Fixed — two of my own errors
I assumed tranche 13's recursion depth transferred: it measured the `index_vol_step` refusal at two
steps on *annual* periods, and this semi-annual fixture survives **three**. And my "quote one period
further" recommendation was wrong — truncating the strip makes the ambiguity *disappear*, not move,
because reachability of the conjugate root depends on the period's own lead and its neighbour's vol.
Replaced with the start-dependence finding, which is true and more useful.

### Changed
- `TEST_FLOOR` 430 → 453.

## [0.16.0] - 2026-11-04

Tranche 13 — Market Model II (BERHAN-2012 ch 4), the **third** model, through the same
`IndexRatioModel` seam with no change to it. The thesis's model shelf is now complete.

### MM-II's identity: no real-rate volatility, anywhere
The source's words: *"contrary to Market Model I, the correction term does not depend on the
volatility of real rates."* `MMIIParams` has **no real-vol field to set** — asserted structurally on
`__dataclass_fields__`, so a later edit cannot violate it silently. The real *curve* still enters,
through the forward levels in `1 + τFr`; it is the real *volatility* that cannot reach the model.

### Two hazards in the source, each settled and pinned
**Eq 4.1.3's denominator contradicts `Di`'s, in both renderings** — 4.1.3 prints `Fn/(1 + Fn)`,
`Di(t)` prints `τᵢFn/(1 + τᵢFn)`. An internal inconsistency, so §4's derivation-step rule fired and
found a *discrepancy* rather than a confirmation. `1 + τᵢFn` is taken as correct (the standard LMM
coefficient, what MM-I's freezing uses, dimensionally sound, and the form used downstream), pinned
by an oracle.

**`Vi(t)` is defined twice for two different quantities** — 4.2.2's lead-only form, and an augmented
`sqrt(Vᵢ² + σ_I²(Tᵢ−Tᵢ₋₁))` before 4.2.4. The caplet needs the augmented one: 4.2.1 is the
*forward*-index ratio at `Tᵢ₋₁` while the payoff is the *spot* ratio at `Tᵢ`. Dropping it reuses
MM-I's `σ_I²·span` shape and loses the lead term — guarded both ways, including an exact reduction
to `σ_I²·span` at `ρ_{I,I} = 1`.

### Added
- `models/market_model_ii.py` — `MarketModelII`, `MMIIParams`, `MMIICorrelations`. 22 oracles: the
  **four-path** zero-vol reduction (MM-II, MM-I, JY, the curves); a **second exact anchor** at
  `ρ_{I,n} = 0`, where the source says the formula is exact and the freezing is unused, oracled from
  both sides; pointwise transcriptions of 4.1.6 and 4.2.2; three models required to **disagree**
  away from the limits; and `cap − floor = the YoY leg` under a third model.

### Fixed — the unit-accrual trap, fifth appearance, on a source-discrepancy oracle
The test pinning the 4.1.3-vs-`Di` denominator was powerless: on annual ACT/365F periods `τ = 1.0`
exactly, so `τFn` and `Fn` coincide and the two readings were **indistinguishable** — leaving `τ`
unoracled throughout the file. Fixed with a semi-annual fixture (`τ ≈ 0.4932`). First time this trap
hit an oracle about a *source discrepancy* rather than a formula.

### MM-I's real-vol dependence is real but weak, and the size is the finding
A contrast oracle failed at `rel=1e-6` — not because the dependence is absent but because it is
small. Sweeping `σr` 5% → 50%: **7.7e-09** at a 0.49y lead and `ρ_rI = −0.20`, rising to **3.7e-05**
at 1.48y and `ρ_rI = −0.90`. So MM-II's "no real-rate vol" is a genuine *structural* simplification
whose *pricing* impact is modest here — stated rather than hidden behind a loosened tolerance, with
the oracle now asserting the trend in `|ρ_rI|` and the lead.

### Changed
- `TEST_FLOOR` 408 → 430.

## [0.15.0] - 2026-11-04

Tranche 12(b) — Market Model I (BERHAN-2012 ch 3), the library's **second model**, flowing through
tranche 12(a)'s capability seam with no change to either pricer.

### The thesis's "approximately constant" does not hold — the freezing drives `σ_I` negative
Mercurio's freezing relation `σ_I,i−1 ≈ σ_I,i + σr·τFr/(1+τFr) − σn·τFn/(1+τFn)`, walked backwards
from `σ_I = 0.012` with `σn = 25%`, `σr = 20%` on this library's sloped test curves: **+0.012000 →
+0.005605 → −0.000542.** Two steps to a negative volatility. The adjustment is negative whenever
`σn·τFn/(1+τFn) > σr·τFr/(1+τFr)` — the ordinary case, since nominal vol and nominal forwards both
exceed real — so the source's *"we can still assume that the volatilities σ_I,i are all constant"*
does not survive a realistic parameter set. `index_vol_step` refuses a non-positive result rather
than letting it reach a `√` and surface as a `nan` layers away.

**The third honest negative of the inflation arc**, after JY's flat smile and §5.3.2's structural
underdetermination.

### Where the approximation is — not where the spec expected
Given its per-period inputs, eq 3.1.11 is an **exact** expectation; the freezing governs the
index-vol *term structure* between periods, not any single period's price. Its error bar, measured:
a 200bp forward move shifts `σ_I,i−1` by **71% of `σ_I`**, near-linear in the shock. A
**sensitivity** bound, not a distributional one — the latter needs an LMM simulation this tranche
does not build and does not claim.

### Added
- `models/market_model_i.py` — `MarketModelI`, `MMIParams`, `MMICorrelations`, `index_vol_step`.
  `MMICorrelations` carries **two** correlations, not JY's three: neither eq 3.1.11 nor §3.2 reads a
  nominal-index correlation, and sharing JY's type would force the carried-unread field §6 forbids.
- `foundation/quadrature.py` — `standard_normal_expectation`, a fixed Gauss-Hermite rule. Chosen
  over `scipy.integrate.quad` because an adaptive integrator picks its own nodes per integrand and
  can shift between scipy versions, neither of which belongs under a pinned reproducibility
  contract. **No tolerance knob**; accuracy is measured against moments known in closed form.
- 33 oracles. **The anchor is a three-path reduction**: at zero vol MM-I's integral, JY's `e^C → 1`
  and the curves directly all agree to 1e-13, sharing nothing but the curves — with a companion
  requiring the vols to move MM-I's ratio, and a further oracle requiring MM-I and JY to *disagree*
  away from the limit.

### The transcription is corroborated by a second path inside the source
`-layout` mangles eq 3.1.11 into broken columns, so §4's dual-rendering rule could not settle it.
Page 25 gives the same integral in its **pre-substitution** form, and the two agree to **2.2e-16** —
a misprint would have to appear identically in both forms to survive. Both are oracled. Stronger
than two renderings of one line, and worth reaching for whenever a source derives in steps.

### Fixed — two of my own test defects, both found by execution not review
**Mutation found a hole properties alone left open.** Dropping `/(1+τFr)` from the freezing
coefficient survived all 404 tests — not an equivalent mutant, since at `τF ≈ 0.03` the denominator
is worth ~3%. My freezing oracles asserted only properties and never transcribed the formula
pointwise. Fixed, with a companion requiring the un-divided form to differ measurably.

**And a threshold that read as reassurance.** My error-bar oracle asserted `worst < 0.01`, implying
the freezing error is small, while the measurement says it is 71% of `σ_I`. It now asserts the
magnitude and its linearity instead.

### Changed
- `TEST_FLOOR` 375 → 408.

## [0.14.0] - 2026-11-03

Tranche 12(a) — the model capability seam, so a second model can price the same products. The
inflation vertical's first structural change since the spine.

### The design question was settled by building: ONE protocol, and the source supplies its shape
A second model (Market Model I) prices the *same* products, and the registry dispatches on product
type — so a second model must not become an `isinstance`-on-model branch. The open question was
whether one protocol method could carry both models' formula or whether a two-key
`(product, model)` registry was the honest seam.

**One protocol with two methods fits both, and no formula was contorted.** Factoring BERHAN's two
year-on-year leg formulas side by side, eq 2.5.12's later-period term and eq 3.1.11's are both
`psi_i · Pn(t,Ti) · [E − 1]`, with `E` the `Ti`-forward expectation of `I(Ti)/I(Ti−1)` — JY's
`(Pn/Pn)(Pr/Pr)e^C`, MM-I's bivariate-lognormal integral. The cap/floors are both Black on
`(E, V²)`. The seam is the *source's* decomposition, and both quantities are ones BERHAN names (`m`
eq 2.6.2, `V²` eq 2.6.4). Had the protocol exposed JY's `convexity` instead, MM-I would have had to
manufacture a fake `C` from the log of its integral — the contortion that was to be avoided.

### Added
- `models/index_ratio.py` — the `IndexRatioModel` protocol: `expected_index_ratio(start, pay)` and
  `index_ratio_variance(start, pay)`, in **dates** (tranche 7's lesson). Introduced at its second
  consumer, not in anticipation — the same timing as Black in tranche 9 and the Hull-White core in
  tranche 10.
- `JarrowYildirim` implements it, delegating to its existing year-fraction methods so eq 2.5.11 keeps
  exactly one implementation.
- 9 oracles, including **the one a type annotation cannot be**: a stub model implementing only the
  two methods prices correctly through both engines, and the cap/floor's output is *exactly* Black on
  the stub's two numbers — proving the pricers read only `m` and `V²`. Six mutations on the seam, all
  caught.

### Changed — the refactor is NOT bit-identical, and that is reported not claimed
`engine/yoy.py` and `engine/capfloor.py` now depend on the protocol. The factoring that makes one
seam fit two models reorders a division, so JY's last digits move: **36119.025432954419 →
36119.025432954477, a relative 1.6e-15 (5.8e-13 bp of notional)**. Every tranche 4 and 5 oracle
passes unchanged. Keeping bit-identity would have required exposing `convexity`, which is what makes
a second model impossible — so the substitute is agreement to 1.6e-15 with the cause named.

`caplet_value` now takes the `(start, pay, accrual)` schedule period rather than a pair of year
fractions, removing a date → year-fraction → date round trip.

### Fixed — a wrong requirement of mine, caught by its own test
I asserted `m > 1` on an inflationary curve pair. False at 3y→4y (`m = 0.9914`): a negative
convexity overcomes the 2.1% carry. A requirement that ignores the convexity is not a requirement of
this model. Replaced with the decomposition — at `σr = 0` the convexity vanishes exactly and `m > 1`
*is* then forced — which also pins the sign of `C`'s contribution.

### Changed
- `TEST_FLOOR` 366 → 375.

## [0.13.0] - 2026-11-03

Tranche 11 — the full six-parameter JY calibration (BERHAN-2012 §5.3.2), and reproducing **why it
fails**. The closing rung of the JY arc, and deliberately a negative result.

### This tranche does not claim a successful fit — the thesis says there isn't one
§5.3.2 frees all six of `{κr, σr, σI, ρnr, ρrI, ρnI}` and reports failure: *"we cannot fit to the
whole surface"*, *"the presence of a strike skew makes calibration unfeasible for non-ATM
contracts"*. So "the fit matches the market" cannot be the oracle — it would assert something the
source denies. What is delivered instead: correct machinery, the documented failure reproduced, and
the failure *explained*.

### Two distinct failures, and the thesis names only one
**Underdetermination, structural.** The index correlations enter `yoy_variance` only as products
with `σI`, so `V²` is a quadratic in `σI` whose linear coefficient is a single *pooled* combination
of `(ρnI, ρrI)`; and `ρnI` is absent from `yoy_convexity` entirely. Both are asserted against the
code. On **perfect ATM synthetic data**, four starts: `(κr, σr, ρnr)` recover exactly, while `σI`
spreads 27.5%, `ρrI` 21.5%, and `ρnI` **469%** — recovered at **−0.597 against a truth of +0.15, the
wrong sign, at an rms of 8.1e-15.** A confident wrong answer on flawless data, with no skew present.

**Misspecification — the flat smile.** The implied vol is identical at every strike to 12 decimals.
Kept as a guard with its near-tautology stated, since it is the reason a market skew is unfittable —
the failure the thesis does name.

**BERHAN's diagnosis is therefore true but incomplete:** the fit is already underdetermined at ATM,
before any skew. Both failures are asserted separately so the reproduction is honest rather than a
restatement.

### Added
- `engine/jy_full_calibration.py` — `calibrate_jy_to_floor_surface`, `JYFreeParams` (the model's own
  three bundles, not six loose floats), `JYSurfaceCalibration`. Its docstring says plainly that
  `free` is *what the fit produced, not what it determined*, and which three components are one
  point on a valley. `(κn, σn)` arrive fixed from §5.3.1 and are never re-fitted.
- 18 oracles. O1 recovers the identifiable subset from every start; **O2 is the discriminator — the
  multi-start spread plus distance from truth**, behavioural and platform-independent; O3 the flat
  smile as a guard; O4 the two failures asserted as distinct, the underdetermination holding on
  ATM-only data; O5 failure is a value.

### No condition number is asserted anywhere, and that was measured not assumed
The spec's first draft proposed a Jacobian condition-number threshold, and claimed it *blows up* as
non-ATM strikes are added. Both withdrawn before building: a Gram matrix cannot lose rank by gaining
rows (measured, it **improves** — 5.2e10 → 2.2e10), and the smallest singular value sits below the
finite-difference noise floor, swinging three orders of magnitude with the step (2.5e7 → 2.1e10) and
differing again under central differences. A threshold would assert noise and flake across the two
CI operating systems. `s₅/s₁ ≈ 3.4e-6` is the only stable number and is documented, never gated.

### The `slow` gate failed on CI after passing locally, and the first fix damaged the finding
Caching the deterministic multi-start fits took the file 16.5s → 4.4s, green locally — then CI's
Ubuntu runner (~3× slower) still exceeded the 10s ceiling, which is the gate working as intended: a
time ceiling must hold on the slowest runner. Trimming the maturity strip from six points to four
then cost `κr` its identifiability (0.04036 against a truth of 0.04), silently weakening the
partition oracles 1 and 2 assert. The speed came from the number of starts instead, leaving the
strip intact: **2.44s** with 4× headroom and the finding unchanged. A speed optimisation can
degenerate a fixture exactly as a careless one can.

### Changed
- `TEST_FLOOR` 348 → 366.

## [0.12.0] - 2026-11-03

Tranche 10(b) — the `(κn, σn)` fit to ATM nominal cap volatilities (BERHAN-2012 §5.3.1), completing
tranche 10.

### Two of the thesis's six calibration parameters now have a source
`σn` and `κn` have been hand-supplied since tranche 6, because BERHAN sources them in §5.3.1 and
declines to publish the forward model. The fit's output feeds `JYFixedParams` directly, so **T6's
"two of six" becomes four of six**. Still unfitted: `κr` and the three correlations, part of a
six-parameter fit the thesis itself reports does not work.

### Added
- `engine/hw_cap_calibration.py` — `calibrate_hull_white_to_atm_caps`, least-squaring the **price**
  residual `Cap_HullWhite − Cap_Black(σ_ATM)` exactly as §5.3.1 states it. L4, because it inverts
  the cap engine. Plus `black_cap_price` and `black_cap_implied_vol` (a flat vol across the strip,
  which is what a market cap quote is), kept local since they have one consumer.
- `AtmCapVolQuote` (schedule, ATM strike, Black vol) and `HullWhiteCalibration` (params, rms,
  n_quotes, iterations) — the fit's output is a complete `HullWhiteParams`, so it is directly a
  model's input with no unpacking step that could pair a fitted `σn` with the wrong `κn`.
- 16 oracles. B1 recovers a known `(κn, σn)`; **B2 is its identifiability companion** — each
  parameter moves the strip, four distinct starts land on one point, the sensitivity Jacobian is
  well conditioned, and a **one-maturity strip is asserted to be unidentifiable** so the multi-start
  check is known to have power. B3 the residual vanishes at the truth; B4 failure is a value.

### The Black caplet's variance horizon is the EXPIRY — the mirror of tranche 9
A rate caplet fixes at `Tᵢ₋₁`, so its variance is `σ²·Tᵢ₋₁`; an inflation floorlet accumulates over
the period, `σ_I²·(Tᵢ − Tᵢ₋₁)`. The two instruments genuinely differ, which is why `black_price`'s
parameter is `variance_years` and not `expiry`. Pinned by an oracle requiring the two horizons to
give prices more than 10% apart.

### Fixed — the solver stopped 0.09% short until the residuals were scaled to basis points
At unit notional the price residuals are O(1e-3), so scipy's relative tolerances were met while
`κn` was still 0.08007 against a truth of 0.08. Scaling the solver's residuals to basis points — the
market's own quoting unit — recovers it exactly (`rms` 7.0e-7 → 9.9e-17). `rms_residual` is still
reported in price units. The pre-registered `rel=1e-6` is what exposed this; a looser tolerance
would have passed.

### Changed
- `TEST_FLOOR` 332 → 348.

## [0.11.0] - 2026-11-02

Tranche 10(a) — the Hull-White nominal cap engine. **The first non-inflation instrument**, pulled in
because BERHAN-2012 §5.3.1 sources `σn` by fitting `(κn, σn)` to ATM nominal cap vols, and `σn` has
been a hand-supplied input since tranche 6.

### New source: `MCINERNEY-ZASTAWNIAK`, `in-hand`
BERHAN performs §5.3.1 with a Hull-White cap formula it **declines to publish** ("beyond the scope
of this thesis"), and charter §7 forbids citing Brigo–Mercurio, which is only `referenced-by`. The
build stopped there rather than inventing an anchor; Bernardo acquired McInerney & Zastawniak,
*Stochastic Interest Rates*, which supplies §2.7's caplet-as-a-put-on-a-zero-coupon-bond, §3.5's
bond option (3.16–3.18), `ν(0,S)` (3.19/3.20), the caplet (3.21) — and (3.26), the calibration
scheme itself, which unblocks (b) too.

### Added
- `models/hull_white.py` gains `bond_option_variance` (eq 3.20), `zero_bond_option` (3.16/3.17),
  `hw_caplet` (3.21) and `HullWhiteModel`. The Hull-White core (`HullWhiteParams`, `hw_beta`) was
  relocated here from `models/jarrow_yildirim.py` at its second consumer — lifted **verbatim** and
  proven **bit-identical across 15 captured values**, with tranche 4's oracles green.
- `products/nominal_cap.py`, `engine/nominal_cap.py` — a first-class L2 product priced through
  `price()`. Deliberately **not** sharing a type with `InflationCapFloor` despite identical fields:
  one strikes an index ratio, the other a nominal rate — the shared *word*, not a shared shape.
- 34 oracles. The caplet is checked against an **independent longhand transcription** of 3.16–3.21;
  `σ → 0` gives the intrinsic with the companion that `σ` moves the price; and the headline
  cross-product identity **cap − floor = the forward-rate swap**, whose right-hand side is built from
  discount factors alone with no option formula in it.

### A HW bond option *is* Black, so no second transcription was written
`zero_bond_option` composes `black_price` (tranche 9's extraction), because a bond option is Black on
the forward bond price `B(0,T)/B(0,S)` with total variance `ν`, discounted by `B(0,S)`. The only
genuinely new formula is `ν`. `ν` itself is written through `hw_beta`, since eq 3.19's `D(S,T)` **is**
`β` — verified by reducing 3.19 to the printed 3.20.

### The source misprints `d₋` in both renderings, and parity cannot catch it
Both `-raw` and `-layout` print `d₋ = d₊ − ν(0,S)` while `ν` is a variance. **Put-call parity is
satisfied identically for any `d₊`/`d₋`** (`Φ(d) + Φ(−d) = 1` collapses both terms), and the `ν → 0`
limit converges either way — so both obvious discriminators are checks that cannot fail here. What
settles it is that the literal reading prices a **call at −0.0049**. That has its own oracle.

### Fixed — the unit-accrual trap left a real hole, found by mutation
Every fixture used annual ACT/365F periods where `τ = 1.0` exactly, so `1 + τK` and `1 + K` are the
same number: a mutation dropping the accrual from eq 3.21 **passed all 312 tests**. The L3 caplet
test now sweeps `τ ∈ {0.25, 0.5, 1.0, 2.0}` and the L4 fixture is semi-annual, with the non-unit
accrual asserted as a premise; the mutation now fails 23 tests. Fourth appearance of this trap.

### Changed
- `TEST_FLOOR` 224 → 332.

## [0.10.0] - 2026-11-01

Tranche 9(b) — the floor→floorlet strip and the implied-vol surface (BERHAN-2012 §5.2), completing
tranche 9. Decided by Cowork, split per its §4.

### Fixed — `black_implied_vol`'s `expiry` parameter was the wrong NAME, worth up to 50%
Shipped in v0.9.0 as `expiry`. Its second consumer showed the divisor is the horizon the *variance*
accumulates over, which for a year-on-year floorlet is the **period length**: the floorlet on
`[2y, 3y]` expires at 3y and accumulates one year of variance, since `V² = σ_I²·(Tᵢ − Tᵢ₋₁)` under
simplified JY. Passing the expiry under-states `σ_I` by **29.3% / 42.3% / 50.0%** at the 2y / 3y / 4y
nodes — and **0.0% at the 1y node**, whose period starts at the anchor, so a single-period fixture
could never have caught it. Renamed `variance_years`; the trap is pinned by its own oracle across
four periods, with the error required to grow with maturity.

*If a name needs a comment to explain it, the name is wrong.* This one needed one.

### Added
- `market/vol_surface.py` — `FloorQuote` (a quoted floor: schedule, strike, price, notional) and
  `ImpliedVolSurface`, nodes keyed by `(payment date, strike)`. **Dates, not year fractions** —
  tranche 7 lost a pointwise oracle to exactly that mismatch. **No interpolation**, and an off-node
  request raises, so the absence is a decision with a test rather than an oversight.
- `models/implied_vol_surface.py` — `implied_vol_surface(market, quotes)`: difference the strip,
  invert Black per node. Under the thesis's simplified JY (§5.2 Prop 2.3.1, JY at `σn = σr = 0`) the
  convexity vanishes, so the forward is a pure curve relationship. Quotes are grouped **by strike
  and stripped independently** — differencing across strike would mix two option series while still
  producing a plausible node count.
- 21 oracles. The anchor recovers a known **sloped** generating vol at every node, with the test
  pricing the floorlets itself. Cowork's sharpening is built in: the count is the population floor,
  the per-node values are the discriminator, and three premises are asserted — pairwise-distinct
  floorlets, a **reversed** strip giving a different middle node, and `floor(T₀) = 0`.

### Changed
- Out-of-order floor quotes are **sorted, not refused** (a feed's order is not an error); a
  **duplicate** maturity at one strike is refused, since it differences to a zero-price floorlet.
- `TEST_FLOOR` 204 → 224.

## [0.9.0] - 2026-11-01

Tranche 9(a) — Black-76 extracted to L3 with its inverse, the first half of the implied-vol surface
inversion (BERHAN-2012 §5.2). Decided by Cowork (`docs/tranches/09_implied_vol_surface.md`), split
per its §4.

### Added
- `models/black.py` — `black_price` and `black_implied_vol`. **L3, not L0**: strikes and call/put
  are finance and L0 is finance-free, with the migration tree's own `foundation/black.py` as the
  named precedent for that drift. **Parameterised by total variance, not `(σ, T)`** — JY's `V²`
  (eq 2.6.4) is an eight-term total variance and not `σ²·T`, so a `(σ, T)` signature would make
  every caller invent a `σ` the model does not have.
- `foundation/solvers.py` gains `bracketed_root` — a separate entry point from `least_squares_fit`
  because a monotone 1-D inverse carries a stronger guarantee: given a sign change Brent converges,
  so there is no non-convergence to report. The seam's second consumer, so the rule of two is now
  satisfied for the seam itself.
- `black_implied_vol` **refuses rather than guessing**: a price below intrinsic or at/above the
  no-arbitrage limit has no Black vol, and raises — `PricingFailure` is the *engine* contract (§3),
  while a constructor validates at its trust boundary (the tranche 8 ruling). Its bracket is
  structural: `V² = 0` below, doubling until the price is exceeded above, so no constant decides the
  range.
- 58 oracles. Put-call parity to 1e-15 as a **requirement** (derived from the payoff, so it survives
  a transcription error the code and the reference could share), the no-arbitrage bounds,
  monotonicity of the inverse, and the refusals. The round trip is kept but **labelled as
  telescoping**: it earns its keep only on tranche 5's independent forward oracle, which the test
  states.

### Changed
- `engine/capfloor.py` composes `black_price` instead of computing Black inline — the rule-of-two
  extraction at its second consumer, so a Black convention cannot diverge between pricing and
  implying. **Verified bit-identical**: eight strip prices captured as `repr()` before and after are
  unchanged to the last bit, with tranche 5's oracles green.
- `TEST_FLOOR` 146 → 204.

## [0.8.0] - 2026-11-01

Tranche 8 — the nominal curve bootstrap (BERHAN-2012 §5.1), the cross-check presence gate, and the
per-module dispositions. Decided by Cowork (`docs/tranches/08_nominal_bootstrap.md`).

### The thesis's §5.1 is misprinted, twice — and both renderings agree
Transcribed in `-raw` and `-layout` per charter §4; the renderings match each other and the two
equations still contradict each other. **5.1.1 carries a spurious leading minus** (every par rate
would be negative) and **5.1.2 sums `j = 0..i−1`** where inverting 5.1.1 requires `j = 1..i−1` —
with `P(0,T0) := 1` defined on the same line, so the printed form adds a spurious unit to the
annuity. Against a known flat 3% curve the implemented form recovers `P(0,T₁) = 0.97044553` exactly;
the printed form gives 0.94089107.

**A new sub-case of §4:** transcription independence protects against a *rendering* mangling a
formula. It is powerless when the **source** is wrong, because both renderings reproduce the typo
faithfully. What catches that is the **requirement that a formula and its stated inverse actually
invert** — which is why the anchor starts from a known curve. Both misprints now have their own
tests.

### Added
- `market/nominal_bootstrap.py` — `nominal_curve_from_par_swaps`, quotes → `DiscountCurve`, plus the
  shared `swap_annuity` atom (`Σ τ·df`, reading the schedule's own day count) so a future swap
  pricer composes the same annuity. **Closed form per pillar, no root-find** — corrected 5.1.2 is an
  exact rearrangement, and solving it numerically would be "solving a closed form"
  (`market/inflation_curve.py:54`). That also makes the negative-rate guarantee unconditional: no
  bracket, so nothing for a negative rate to fall outside.
- 17 oracles. The anchor is a **known sloped curve round-tripping through its own par rates,
  pointwise to 1e-12** — non-telescoping, since the test owns the forward map. The par-reprice is
  kept only as a consistency guard. No identifiability discriminator, and that is a reason not an
  omission: a sequential bootstrap is exactly determined (triangular), so there is no valley.
- `verify.py provenance` gains the **cross-check presence gate**: every module under `market/`,
  `models/` and `engine/` carries a real `crosscheck:` target or `crosscheck: none — <reason>`, the
  reason required. **Mutation-tested**: deleting tranche 7's `fixing_series.py` marker — the exact
  omission that slipped through — fails the gate, and so does a reasonless `none`.

### Changed
- Charter §8: the marker is the provenance-header `crosscheck:` form (no `#`), **gated on presence**
  rather than form. Form-only cannot catch an omission, and tranche 7 produced one in the same hour
  the form-only rule was made.
- Charter §2: a **curve bootstrap is L1**, not L3. The clause said L3 while naming tranche 2's
  break-even inversion as its example — which lives in `market/inflation_curve.py` at L1, so it
  misdescribed its own example. Model-free *parameter* calibration stays L3; engine-inverting stays
  L4.
- `TEST_FLOOR` 129 → 146.

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
