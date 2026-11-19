# Checkpoint — the collateralised-discounting arc (project seven, `LICHTERS-2015`)

**Closed 2026-11-19 at `v0.60.0`, 2322 tests, eleven gates green.** The seventh vertical: the
**CSA / collateralised-discounting layer** the tree had never had — the post-crisis reality that a
collateralised trade discounts at the *collateral (OIS) rate*, with separate forward-projection curves — and
its extension into collateralised credit exposure. Committed by Bernardo on 2026-11-18 as "the biggest spine
the reuse makes affordable"; it built in **four** tranches, not five — T56 deferred for want of an FX layer.

## What was built — and what the pull gate deferred
- **T54 — the OIS / multi-curve bootstrap (ch 3).** The OIS discount curve and the per-tenor forward-
  projection curves. L1. Anchored by **T8's known-curve round trip** on a two-curve construction and the
  **single-curve reduction** (OIS=projection ⇒ `nominal_bootstrap.py`'s output). The OIS half **is** T8's
  bootstrap (an OIS swap discounts its own cashflows), so only the projection bootstrap was new. `v0.57.0`.
- **T55 — CSA discounting (ch 5).** A collateralised swap projects off its tenor curve and discounts at OIS;
  perfect cash-CSA = OIS discounting. Brought the **vanilla interest-rate swap** (the tree had none), the
  `projections` store keyed by `(currency, Tenor)`, and a new `Tenor` L0 type. Convention **C15** guarded.
  `v0.58.0`.
- **~~T56 — cross-currency collateralisation (ch 4)~~ — DEFERRED (§1 pull gate).** The step-1 grep found the
  tree has **no FX layer at all**; foreign-collateral discounting is inherently multi-currency, so it needs
  FX built first — not pulled by the single-currency spine. `to_go_deeper`, needing the FX prerequisite.
- **T57 — the Linear Gauss-Markov (LGM) short-rate model (ch 11).** The exactly-simulable rate model for
  exposure RFE — a driftless Gaussian state with deterministic variance. L3. Anchored by the **exact
  reduction to Hull-White** (LGM is HW reparameterised) **with a mandatory MC companion** (the reduction is
  blind to the simulation, T39). `v0.59.0`.
- **T58 — collateralised exposure under a CSA (the capstone).** The arc composed end to end: LGM → the
  collateralised IRS → the **MPoR gap** (`risk/collateral.py`, reused from T29) → the **CVA** (T25, reused),
  in both eras, plus a non-zero **threshold**. L5. `v0.60.0`.

## What the arc reused — why "big" was affordable
Three major pieces were reused, not rebuilt, each with a cross-check: **CVA/DVA and the MPoR-gap collateral**
(the XVA arc T25–T29 — T58 added no CVA function, collateral enters only through the exposure), **Hull-White**
(T57's LGM is HW reparameterised — an exact reduction), and **T8's par-swap bootstrap** (T54's OIS half is it
verbatim). The arc's new effort went to the genuinely-new objects: the OIS/projection split, CSA discounting,
the LGM parameterisation, and the capstone wiring.

## The single-curve design invariant — Bernardo's parallel, realised BY CONSTRUCTION
Bernardo's requirement was that the library always price in **either era** — pre-crisis single-curve (the
`MCINERNEY-ZASTAWNIAK`/Hull-White world) and post-Libor multi-curve — so parallel projects can show the two
side by side. This was ruled a **design invariant** (not just a reduction oracle): single-curve and
multi-curve are co-equal, deliberately-selectable modes, guarded against silent atrophy (T33b). It held at
every discounting-touching tranche, and ended up **structural, not branch-based**, in two places: **T55's
pricer** (the era is the snapshot shape — a snapshot with projections is multi-curve, without is
single-curve, one pricer, no second path), and **T58's basis factor** (identically 1 in single-curve mode,
so one code path serves both eras). The invariant is a guarantee the design makes, not a promise a test
keeps.

## Method advances this arc produced (charter accretions — two, both earned)
- **§10 — the step-1 reuse-anchor grep (bit twice: T54/T55).** A spec's REUSE anchors (not only a named
  prior capability) are grepped by **Cowork at step 1**, not caught by the build backstop. Earned by two
  consecutive specs asserting reuse Cowork had not verified (`nominal_bootstrap.py` misunderstood; "the
  existing swap pricer" absent). It then **reshaped the scope four times** this arc (T54/T55/T56/T58) — the
  highest-leverage edit of the arc.
- **§4 — the relative-tolerance mirror (T55).** A *relative* tolerance is vacuous where the true value is
  **zero by construction** (a par reprice, a reduction-to-zero) — the dual of the near-zero clause, and the
  trap you hit by reaching for it. Match the tolerance to where the value lives; a zero value gets an
  absolute band against the scale.

## The lessons that did NOT grow the charter — the healthy outcome
- **The method caught Cowork's OWN transcription errors.** T57's eq (11.4) and T58's threshold formula were
  both wrong in my specs and caught by §5's unconditional re-transcription + the *named reductions* — the
  spec being wrong was survivable because the oracle it specified was right. Process change (mark spec
  formulas "re-transcribe from eq X"), not a clause.
- **T29's oracle narrowed, not a new clause.** "Collateral never increases the adjustment" is a
  **plausibility check** (it can be violated pathwise), not a finance-forced requirement — the §4 floor rule
  on an oracle's own description. Narrowed in re-audit #6's register pass; "implausible-input detector vs
  theorem check" is a lens the floor rule covers.
- **The frozen-tenor-basis choice** (named, not buried; the stochastic basis a separate model) and the
  **loop-invariant-cache** lens (the third ceiling-watch file, cleared losslessly) were both the right calls
  without accretion.

## Honest completion ratio — the deferred ledger
Built as the spine: the OIS/multi-curve bootstrap, CSA discounting, the LGM, and the collateralised-exposure
capstone with a threshold. Deferred to `to_go_deeper` (oracle-able, unpulled): **cross-currency /
foreign-collateral collateralisation** (needs the FX layer — the single largest deferral), **collateral-
choice / MTA / initial margin / MVA / ColVA / gap-WWR** (T29/T58's fence), **stochastic tenor basis**,
**fair-value hedge accounting** (ch 6, §0-adjacent), the **FX / equity RFE zoo** and **multi-currency LGM**
(ch 12–14), **Dodgson–Kainth inflation** (ch 13.3). **`out_of_method` candidate:** **exposure-model
backtesting** (the book's backtesting part + the held `kerkhof.pdf`) — statistical/historical validation,
no closed-form oracle. **Open debt:** `OPEN.md` **OI-6** alone (T8's skipped-pillar `KeyError`, a §3.4
roughness T54 surfaced; re-open at the next `nominal_bootstrap.py` touch or re-audit #6).

## Governance this arc
- **The step-1 grep is the arc's signature** — four scope reshapes caught *before* a wrong spec was built,
  the §10 edit paying for itself more than any other.
- **A pull-gate deferral mid-arc (T56)**, cleanly, on the grep — the T47 discipline, now in a second arc.
- **The draft-PR mechanism held**; every tranche opened draft, marked ready only at step 6.

## Close
The collateralised-discounting arc is complete on its builds. The close triggers **re-audit #6** (§9 —
arc-close trigger, and ~nine tranches since re-audit #5 at the Gatheral close). Its scope and findings are
`docs/audit/reaudit_06_findings.md`, with the **proactive invariant-6 sweep** as its standing first item,
the **single-curve invariant's reachability** swept across the arc's discounting-touching modules, **MM-I's
owed second killer**, and the **T29 docstring narrowing** in the register pass. After it, the library is at
a fresh **arc boundary**: the next direction is Bernardo's interest plus a Cowork spec.
