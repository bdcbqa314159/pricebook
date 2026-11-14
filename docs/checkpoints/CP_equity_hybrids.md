# Checkpoint — the equity & hybrids arc (project four, `BROCKHAUS-2016`)

**Closed 2026-11-13 at `v0.44.0`, 1430 tests, nine gates green.** The fourth vertical and the
capstone — equity on its own closed forms, then equity coupled to the rates, credit, and
counterparty machinery the three earlier arcs built. The spine was committed in full (T30→T36b) by
Bernardo on 2026-11-10 and built straight through.

## What was built — the equity core and the three hybrids
- **T30 — the equity underlying + Black-Scholes vanilla.** Spot, repo, three dividend conventions
  (continuous / proportional / discrete-cash), the dividend forward, the European option on
  `models/black`. `v0.36.0`.
- **T31 — the variance swap.** Model-independent static replication (the log-contract, a strike
  continuum of vanillas) — an exact, model-free result. `v0.37.0`.
- **T32 — Dupire local volatility.** The implied-vol surface and the local-vol function, anchored by
  the round-trip (local vol reprices the vanillas it was calibrated to, a non-telescoping check).
  `v0.38.0`.
- **T33 — Heston stochastic volatility by Fourier inversion.** The characteristic-function vanilla,
  with the Little Heston Trap (the branch-cut discontinuity) turned into a refinement-invariance
  oracle. `v0.39.0`.
- **T33b — the `(product, model)` registry.** Infrastructure the hybrids forced: one product (the
  equity option) now carries several model-pricers, so dispatch keys on the pair, with a protocol
  path for the polymorphic pricers. The silent clobber became a loud failure. `v0.40.0`.
- **T34 — the equity-rate hybrid.** Equity under a stochastic Hull-White short rate, closed form —
  the **first true hybrid**, reusing `models/hull_white` (and its `hw_beta`, verified back at T4/T10a)
  whole. `v0.41.0`.
- **T35 — defaultable equity.** Reduced-form, independent default (Brockhaus 13.1.1) — a modified-Black
  vanilla reusing the `SurvivalCurve` (T25) whole. The equity-credit seam. `v0.42.0`.
- **T36a — equity CVA (independent default).** CVA on an equity-option exposure, the **cross-arc
  seam**: our own `unilateral_cva` (project three) reproduces the closed form and the gap *converges*
  under grid refinement. Prices CVA for all four equity models through one `dispatch` — the T33b
  payoff in a consumer. `v0.43.0`.
- **T36b — wrong-way risk.** The equity/default correlation (reduced-form, Brockhaus 14.3/14.5 on the
  13.1.2 apparatus), via a new L0 bivariate-normal CDF `Φ₂`; `ρ→0` reduces *bit-identically* to T36a.
  The capstone. `v0.44.0`.

## What the arc reused — the map that makes it the capstone
The last third spent most of its new effort on *seams*, not rebuilding: `models/hull_white` and the
LMM machinery (the Jäckel arc, T15–T22) fed T34; the `SurvivalCurve` (T25) fed T35 and T36; the XVA
engines `unilateral_cva`/`mc_exposure_cva` (T25/T26) fed T36a; `models/black` (T5/T30) anchored every
equity closed form. Project four is where the three prior verticals meet.

## Method advances this arc produced (charter accretions)
- **§4 — the zero-order-invisibility detector (T34).** A parameter whose first-order effect on the
  chosen observable is *identically zero* by a structural identity cannot be pinned by that observable
  at any precision; the detector is the analytic derivative, and the fix is to pin by the defining
  identity and make the suite *state* the price oracle does not cover it. (The hybrid's `d±`
  correlation shift: `F·n(d₊)=K·n(d₋)` exactly.)
- **§4 — the ninth degenerate-fixture shape (T33b), fired three more times (T36b).** Input-population
  uniformity: a validation/type-guard's reject branch goes untested because every supplied input is
  well-formed; assert non-degeneracy by supplying the input the guard rejects. Sub-trap: an empty
  `runtime_checkable Protocol` is satisfied by everything. The clause then caught `Φ₂`'s `max(·,0)`
  floor (unfloored → negative probability).
- **The lessons that did NOT grow the charter — the healthy outcome.** T35 (a pre-registered test with
  backwards economics, corrected from the model's `E[S]=F` constraint) and T34/T36b (source
  mis-readings and a dropped relation operator, settled by requirement oracles) were ruled *instances*
  of the existing "derive before asserting" and "a reference is code too" / dual-rendering clauses —
  the charter earning its keep by covering cases it was not written for rather than accreting
  near-duplicates. The arc closed with the charter leaner-relative-to-coverage than it opened.

## The notable oracles — the arc's best method moments
- **The cross-arc reduction (T36a)** — two independently built towers (the equity closed form and the
  XVA engine) reconciled, and *converging* under refinement rather than merely agreeing.
- **The martingale collapse (T36a)** — the CVA integral collapses to `V₀(1−p₀)` because the discounted
  option value is a martingale; a *forward* does not collapse, and that contrast is the content.
  Derived before building, verified against simulation.
- **Φ₂-chosen-for-its-property (T36b)** — the Plackett form picked so its `ρ=0` integral is empty,
  making the capstone's reduction bit-identical, and collapsing *through* the eq-13.4 identity so a
  solver error and a four-term error cannot cancel.
- **The Little Heston Trap (T33)** — a famously silent wrong price turned into a refinement-invariance
  oracle.

## Honest completion ratio — what the spine does NOT do (the deferred ledger)
Built as the spine: the equity underlying + vanilla, the variance swap, local vol, Heston, and the
three hybrids. Everything below is `to_go_deeper` (oracle-able with our discipline, unpulled) —
**nothing in this arc is `out_of_method`; equity is closed-form country:**
- The **exotic equity zoo** (barriers, cliquets, asians, lookbacks, autocallables, ch 3) — MC
  consumers, pulled when interest wants one.
- **Implied-vol dynamics** (ch 7); **local-stochastic volatility** (ch 6.3) and **jump-diffusion /
  Bates** (ch 6.4).
- **Correlation & copulas / multi-asset** (ch 8–9) — a whole sub-arc.
- **FX & quanto** (ch 15).
- **Heston calibration** to a market surface (T33 prices given params); **Carr-Madan FFT** across
  strikes.
- **Portfolio / netting-set CVA** and **collateralised** (MPoR) equity CVA; **structural / asset-barrier
  WWR** (ch 14.4, Merton/CreditGrades — a new model family).
- **Correlated-default defaultable equity for the STOCK itself** (Brockhaus 13.1.2): T36b built the
  13.1.2 apparatus for the *counterparty's* default in WWR, but the correlated defaultable-*equity*
  vanilla (stock skew from jump-to-default correlated with the diffusion) was not pulled — it now has
  a cheaper route since `Φ₂` exists.

## Governance this arc
- **T33b's registry** removed the silent-clobber footgun before the hybrids multiplied product×model
  pricers — rule-of-two deciding an abstraction at its second consumer, not before.
- **The split at the capstone (T36 → T36a/T36b)** — ruled on Code's evidence (a new L0 primitive +
  the 13.1.2 apparatus + the correlated MC, each a tranche's worth), the T20/T33 precedent. T36a
  landed a complete, reconciled seam regardless of how WWR's oracle-gate resolved.
- **The draft-PR mechanism held** — every tranche opened draft, marked ready only at step 6; no
  premature merge this arc.

## Close
The equity & hybrids arc is complete on its builds. The close triggers **re-audit #3** (§9, arc-close
trigger — the eleven-tranche counter sits at nine since re-audit #2 at T29, so arc-close is the firing
trigger). Its findings are recorded in `docs/audit/reaudit_03_findings.md` and ruled by Cowork. After
it, the library is at a fresh **arc boundary**: the next direction is Bernardo's interest plus a Cowork
spec, written to the queue — not raised in the build channel.
