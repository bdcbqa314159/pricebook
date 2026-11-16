# Re-audit #4 — the credit-derivatives arc close (v0.52.0, 2026-11-16)

Fourth whole-tree re-audit (§9), fired by the **arc-close trigger** — the eleven-tranche cadence
counter sits at ten since re-audit #3 (T37: T38, 39, 40, 41, 41b, 41c, 42, 42b, 43, 44a, 44b is
eleven builds but T37→now is the span; arc-close fires first regardless, as it has every time). Run
with independence: Cowork sets scope and rules; the mechanical mutation sweep is Code's execution on
that scope; the register-accuracy and charter-staleness passes (no execution) are done here by Cowork.

Four §9 targets: **(1) oracle power** (full-tree mutation sweep — Code, PENDING), **(2) cross-tranche
consistency & duplication**, **(3) register accuracy**, **(4) charter staleness**. Per §9, (1) and (3)
inform each other and are run together.

**This arc's own lesson governs this re-audit's conduct:** a surface's report that it did something is
content, not the artifact (§4, the tenth-shape third-bite rule). So where a finding below rests on a
claim — including Code's sweep claims and the "the fixture straddles" premises — the claim is content
until the artifact is shown. Code runs #4 **oldest-first** and re-derives each premise it reports.

---

## (1) Oracle-power mutation sweep — PENDING (Code), scope set

Full-tree, **oldest-first** (§9's binding emphasis — re-audit #1's only hole was in the *oldest*
instrument, not the arc just built; re-audit #3's confirmed it again, the only hole pre-Jäckel). The
scope, in order:

1. **FIRST NAMED ITEM — the proactive invariant-6 / historical-flow sweep (the tenth-shape
   escalation's first run).** The tenth degenerate-fixture shape (premium schedules that begin at the
   valuation date, so the invariant-6 filter has nothing to filter) **bit a third time at T44b** and so
   earned the proactive sweep, the `τ=1` precedent (§4/§9). This run sweeps **every module with a
   historical-flow filter** — drop or bypass the `is_historical` / date-boundary filter and confirm a
   *straddling* fixture kills it — not only the credit arc's. The known at-risk surfaces: the five rates
   engines (C10's nine pay-boundary sites), the CDS premium leg and bootstrap, the forward CDS, the
   tranche engines, `base_tranche_par_spread` (the T44b site, now fixed — confirm the fix holds under
   the sweep). **Verify, do not trust, the T43/T44a/T44b straddle claims** — T44a and T43 were reported
   to straddle and T44b was reported to and did not; re-derive all three.
2. **The single-point-of-failure re-run table** — every load-bearing mutant recorded as killed by
   exactly one test (the "Accumulating for re-audit #4" table in `docs/audit_log.md`). Re-run each;
   a killer silently deleted since it was recorded reopens the hole with the suite green. Highest value:
   MM-I's `τ/(1+τ)` denominator (second killer still owed), the YoY first `ψ` (an already-shipped hole,
   F-1.2), tranche 26's Leg B numéraire divisor.
3. **The new L0 and credit closed forms, swept for the first time** — `Φ₂` (L0, built T36b, now load-
   bearing for the whole copula/LHP tower), the credit triangle, RPV01, the protection-leg quadrature,
   the copula conditional probability, the LHP formula, the exact loss recursion, the base-correlation
   bootstrap. The scale-invariance blind spot (T43) and the reduction-anchor blind spot (T39) are this
   arc's two new failure modes — probe the tranche oracles for a surviving *uniform rescaling*, and the
   convergence-anchored oracles for a surviving *order loss* their exact anchor cannot see.
4. **The rest of the tree (T1–T37)** as the standing full-tree sweep.

Survivors ruled by Cowork, appended here.

## (2) Cross-tranche consistency & duplication — PARTIAL (Cowork read; Code sweep informs)

- **The T44a tranche-composition duplication — flagged, likely resolved, VERIFY.** T44a's handoff
  flagged that the tranche composition was duplicated across the LHP and recursion models (the reason a
  width-division mutant was reachable). T44a's change-request built one `TrancheLossModel` protocol +
  free functions (`tranche_expected_loss`/`tranche_survival`) in `models/tranche_loss.py`, both models
  satisfying it — which *should* have single-sourced the composition. **Cowork believes this is
  resolved; per this re-audit's governing lesson it is content until Code shows the artifact** — confirm
  there is one composition both models call, not two. If a copy remains, it is a rule-of-two consolidation
  overdue at its second consumer (§6).
- **The `dividends_before(T)` dividend-filter consolidation — still live, correctly deferred.** Re-flagged
  at re-audit #3 "for the next equity touch"; the credit arc touched no equity, so it legitimately
  remains open in `to_go_deeper` route 1. Not a defect (the §4 coverage-gap hole is closed with
  per-branch fixtures); the consolidation that makes a single fixture suffice is still owed.
- **C2 (vol lives on one clock) — still OWED and the widest-spread, least-guarded convention** (10
  modules). Untouched this arc (no vol-surface work in credit). Carried.
- **C7/C8 (day-count conventions) — OWED but unrepresentable** (one day count in the tree); trigger is
  the first second day count. Unchanged.
- **The credit arc's own new conventions C12/C13 — guarded.** C13 (the protection-leg quadrature, one
  functional, bit-identical engine/bootstrap) is structurally and comparison-guarded; C12 (CVA curves
  share one clock) guarded. No new OWED convention opened this arc.

## (3) Register-accuracy pass — DONE (Cowork)

- **`to_go_deeper.md` — two stale arc headers fixed in place.** (a) The `OKANE-2008` header read
  "T38–, in progress"; the arc is closed, now "T38–T44b, **CLOSED** v0.52.0", pointing at the
  checkpoint's completion ratio. (b) The **`CREPEY-2026` header read "T25–, in progress"** — stale since
  the XVA arc closed at re-audit #2 (T29, v0.35.0), and **missed by re-audits #2 and #3** (both ran the
  register pass and read past it). Now "T25–T29, CLOSED v0.35.0". A register-accuracy miss worth naming:
  a *closed-arc header* is exactly the kind of row the register pass exists to catch, and it survived two
  passes — the register pass should check every arc header's status against `docs/checkpoints/`, not only
  the arc just closed. (Same class as re-audit #3's BROCKHAUS header fix; the recurrence says the check
  should be mechanical — a candidate `verify.py` extension is noted under (4).)
- **`references.md` — clean.** `OKANE-2008` is `worked-from` with `src/` citers; the `verify.py
  references` gate (now CI-wired since T38) passes by construction. All seven source keys consistent.
- **`OPEN.md` — clean.** Zero suppressions (`verify.py debt`: suppressions − entries = 0). No open item
  was carried into or out of this arc; the T44 split and the T44b restatement were rulings, not debt.
- **`conventions_register.md` — clean and gated.** C13 (added T39 as a duplicate `C11`, renumbered
  T41c) is correct; `verify.py conventions` (T41c) now guards id-uniqueness and citation resolution, so
  the duplicate-id class that slipped past two reviews cannot recur.
- **Arc plan (`okane_credit_arc.md`) — current.** The queue shows the T44 split (T44a/T44b) and the
  three infrastructure inserts (T41b/T41c/T42b); the completion ratio matches the checkpoint.

## (4) Charter-staleness pass — DONE (Cowork)

- **No clause found stale or in tension.** The arc's §4 additions (the T39 reduction-anchor blind spot,
  the T43 scale-invariance blind spot, the tenth degenerate-fixture shape and its third-bite escalation)
  describe reality and are each anchored to the tranche that earned them. The §9 additions (the CI-liveness
  gate and `verify.py gates`, `verify.py conventions`, the gates-gate widening, the local≠CI timing line)
  describe live gates. The §3.5 "config lives on the model" generalisation describes the T44a/T44b step
  counts. The deliberate *non*-additions (three dissolved questions, the vacuous-oracle restatement ruled
  an instance of the empty-oracle clause) left the charter leaner-relative-to-coverage than it opened —
  the intended outcome, as at the equity capstone.
- **One mechanical candidate, not a charter change — a register-header gate.** Two re-audits in a row
  fixed a stale "in progress" arc header that a closed checkpoint contradicts (BROCKHAUS at #3, OKANE +
  CREPEY at #4). This is *statically decidable* (an arc header's status vs the existence of its
  `CP_*.md`), so by §9's own gateability boundary it is a candidate for `verify.py` rather than a
  recurring manual catch — unlike the τ=1 degeneracy, which is dynamic and stayed the sweep's job. **Not
  built here; flagged for a ruling** — it is a small, real, thrice-adjacent drift (the `references`-gate
  precedent: gate a discipline once it has drifted enough to earn it). Cowork to decide at the next
  boundary whether it earns its place yet.

---

---

## (1) Oracle-power sweep — DONE (Code, 2026-11-16). **5 survivors, all real, all one family.**

Ran oldest-first. **Methodology note first, because it nearly invalidated the run:** the sweep's
first execution reported *"survivors: 0"* from 21 runs that had all **failed on the command line**
(`--timeout=600` with no `pytest-timeout` installed). Every row said "error" rather than "killed",
which is the only reason it was caught. **A sweep that reports zero survivors must also report how
many mutants actually applied** — "a mutant that does not apply is not a mutant that died", at the
scale of a whole sweep rather than one line. Re-run correctly, 21/21 applied.

### F-4.1 … F-4.5 — the proactive invariant-6 sweep (the first named item): 21 sites, 5 survivors

Every historical-flow filter in the tree was bypassed individually (`is_historical` flipped so
nothing is historical), full suite each. 16 killed, **5 survived**, and all five are the tenth
degenerate shape — the guard is present and correct, and no fixture exercises it:

| # | site | why the guard is invisible |
|---|---|---|
| **F-4.1** | `engine/equity_vanilla.py:45` | there *is* an expired-option test, but the option is **out of the money** (strike 105, spot 100), so the intrinsic the bypass returns is `0` — the same answer. **A moneyness degeneracy, not a date one**: a new sub-shape of the tenth family. |
| **F-4.2** | `engine/heston_fourier.py:166` | same shape, same fixture family. |
| **F-4.3** | `engine/forward_cds.py:118` | the swaption's expiry guard — no expired-swaption fixture at all. |
| **F-4.4** | `engine/yoy.py:136` | the fixed-leg comprehension guard: the matured-swap test has **every** period settled, so the comprehension is empty with or without the filter. Only a **straddling** schedule discriminates. |
| **F-4.5** | `engine/yoy.py:152` | the par-rate annuity guard, same cause. |

**Severity.** F-4.1/4.2/4.3: an expired *in-the-money* option would price at its intrinsic instead
of PV 0 — a live number for a settled trade. F-4.4/4.5: a settled period's `nominal.df(pay)` is a
discount factor at **negative `t`**, i.e. above 1 — the same defect class as T40's 73% annuity
overstatement, in an engine that has shipped since T4.

**The escalation earned its keep on its first run**, and the T44b fix holds under it (its site is
killed). **The three straddle claims were re-derived, not taken:** T43, T44a and T44b all straddle
(2 settled / 10 live each) with the premise asserted in each file.

### Items 2–4 — clean

- **Single-killer re-run table:** all four recorded load-bearing single-killers still die (MM-I's
  two `τ/(1+τ)` denominators, the YoY first `ψ`, tranche 26's Leg B numéraire divisor). **No killer
  has been deleted**; each is still killed by exactly one test, so the recorded status stands.
- **New L0/credit closed forms, this arc's two blind spots probed:** a *uniform rescaling* of every
  tranche surface (LHP capped loss, the recursion, the shared composition, the base-correlation
  difference) and an *order loss* in every credit quadrature (both `credit_legs` legs, the RPV01
  trapezoid) — **no survivors**. T43's scale-invariance lesson holds across the whole tower. `Φ₂`,
  now load-bearing for the copula/LHP/recursion tower, dies **80** ways on a 1% rescale.
  *(One probe reported PATTERN NOT FOUND and was re-run against the real line rather than counted.)*
- **Oldest-first full-tree sweep (T1–T37):** 12 mutants applied across the discount curve, the cash
  engine, the linker accrual, Black's `d±`, the T8 annuity, MM-I's freezing weight, Box-Muller's
  angle, `inverse_norm_cdf`'s tail branch and the CVA integrand — **no survivors**. *(Five reported
  PATTERN NOT FOUND on the first pass and were re-run against their real lines.)*

## (2) Duplication — VERIFIED by artifact, not by report

The T44a tranche-composition extraction single-sources it: **zero** composition definitions remain
in `LHPModel` or `RecursionModel`, three in `models/tranche_loss.py`, and dropping the shared width
division **once** fails 4 tests across *both* models' suites. One composition, one oracle, both
covered — as claimed, now shown.

## Status

- (3) and (4): **DONE** (Cowork), above. Two stale headers fixed in place; no fix-tranche.
- (1) and (2): **scope set, PENDING Code's execution** oldest-first, the proactive invariant-6 sweep the
  first named item. Survivors and the duplication verdict appended here and ruled by Cowork.
- Remediation (if any survivor is a real oracle hole): a test-only fix-tranche, the T23/T37 pattern.

Re-audit #4 **logged open**; the arc boundary (the next direction) is settled once the sweep lands and
any remediation is ruled.

---

## Cowork's rulings — F-4.1 … F-4.5, the shape question, and the sweep's own instrument (2026-11-16)

**The escalation is vindicated.** Five real holes on the proactive sweep's *first* run — the third-bite
decision (T44b) paid for itself immediately. Three of the five (F-4.1/4.2/4.3) would price an expired
**in-the-money** option at its intrinsic instead of PV 0; two (F-4.4/4.5) discount a settled period at
negative `t` (a factor above 1), T40's defect class, in an engine shipped since **T4**. All five are blind
oracles — `src/` correct, no fixture exercises the guard — the F-1.2/F-3.1/F-3.2 pattern.

**Ruling: all five are real oracle-power holes → one test-only fix-tranche, T45 (the T23/T37 pattern).**
Code's remedy reading is ratified: an expired fixture that makes the two branches **differ** for each
site —
- F-4.1/4.2 (`equity_vanilla`, `heston_fourier`): an expired **in-the-money** option, so the fall-through
  intrinsic ≠ PV 0.
- F-4.3 (`forward_cds`): an expired-swaption fixture (there is none today).
- F-4.4/4.5 (`yoy` ×2): a **straddling** schedule, so the comprehensions are non-empty and the filter bites.

**The shape question — not a new eleventh shape.** The five split across two *existing* shapes, surfaced
together only because they share the one guard:
- **F-4.3/4.4/4.5 are the tenth** (the boundary is not reached — no expired fixture, or an all-settled
  schedule with empty comprehensions either way).
- **F-4.1/4.2 are the primordial "a zero makes a factor invisible" shape** (the catalog's first example),
  not the tenth: the fixture *does* straddle, but the OTM moneyness makes intrinsic `0`, so the two
  branches coincide through a zero. Calling it a new sub-shape over-claims; it is the oldest shape wearing
  the invariant-6 guard's clothes.
- **What it *does* earn is a sharpening of the tenth-shape sweep's premise**, now charter: the straddle is
  **necessary, not sufficient** — the general premise (the two branches must differ) specialises to "the
  excluded flow is worth a different amount excluded than fallen-through", which for an option bypass needs
  the expired option ITM. The standing proactive sweep asserted "straddle"; its own first run proved that
  incomplete, so the clause is corrected, not grown.

**The sweep's own instrument — a §9 hardening, charter.** The run first printed "survivors: 0" from 21
mutants that had *every one errored* (no `pytest-timeout`), caught only by the per-row "error" label. A
proactive sweep runs unattended and a false `0` ends the re-audit declaring the tree sound when nothing
ran — the most dangerous re-audit failure. Ruled: **the sweep reports applied/killed/survived, and a `0`
survivors is read only against a full applied count** (§9, the §4 domain rule turned on the audit's own
instrument).

**No charter growth beyond these two corrections.** F-4.1/4.2 are an instance, not a shape; the remedy is
the standing T23/T37 test-only pattern. T45 spec: `docs/tranches/45_reaudit4_remediation.md`.
