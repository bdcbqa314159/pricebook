# Re-audit #3 — the equity & hybrids arc close (v0.44.0, 2026-11-13)

Third whole-tree re-audit (§9), fired by the **arc-close trigger** — the eleven-tranche cadence counter
sits at nine since re-audit #2 (T29), so arc-close is the firing trigger, as at #2. Run with
independence: Cowork sets scope and rules; the mechanical mutation sweep is Code's execution on that
scope; the register-accuracy and charter-staleness passes (no execution) are done here by Cowork.

Four §9 targets: **(1) oracle power** (full-tree mutation sweep — Code, pending), **(2) cross-tranche
consistency & duplication**, **(3) register accuracy**, **(4) charter staleness**. Per §9, (1) and (3)
inform each other and are run together.

---

## (3) Register-accuracy pass — DONE (Cowork)

- **`to_go_deeper.md` — one stale header, fixed.** The `BROCKHAUS-2016` section read "the equity arc:
  T30–, in progress … lands at the checkpoint"; the arc is closed, so the header now reads
  "T30–T36b, CLOSED v0.44.0" and points at `CP_equity_hybrids.md`. **Added route 0** — the
  correlated-default defaultable-equity vanilla for the *stock* (Brockhaus 13.1.2), which T36b built the
  apparatus for (counterparty side) but never pulled for the equity itself; now cheap since `Φ₂` exists.
- **`to_go_deeper.md` route 1 (dividend-filter consolidation) — still live, and a mild flag.** T30's
  "do it at the next touch of the dividend/forward code" did **not** happen across the whole arc
  (T32/T34/T35 touched equity pricing; `market/equity.py` still has the `tᵢ ≤ T` filter written per
  convention, no `dividends_before(T)`). It is a code-hygiene route, not a capability, and oracle-able
  trivially; **re-flagged for the next equity touch**, no longer "likely T32". Not a defect — the §4
  coverage-gap hole it came from is closed with per-branch fixtures — but the consolidation that would
  make a single fixture suffice is overdue (rule-of-two well past, §6).
- **`references.md` — clean.** `BROCKHAUS-2016` is `worked-from` with 22 `src/` citers; the
  `verify.py references` gate (T26b) passes by construction. `CREPEY-2026`, `CAPINSKI-ZASTAWNIAK`,
  `CAPINSKI-ZASTAWNIAK`-family keys consistent.
- **`OPEN.md` — clean.** Zero suppressions (`verify.py debt`: suppressions − entries = 0), OI-2 and
  OI-3 closed. No new open item was opened this arc; the capstone split (T36→T36a/T36b) was a ruling,
  not debt.
- **Arc plan (`equity_hybrids_arc.md`) — current.** Queue updated at the split to show T36a/T36b; the
  completion-ratio section matches the checkpoint.

## (4) Charter-staleness pass — DONE (Cowork)

- **No clause found stale or in tension.** The arc's §4 additions (the T34 zero-order-invisibility
  detector; the T33b ninth degenerate-fixture shape) describe reality and fired again this arc (the
  ninth shape caught `Φ₂`'s `max(·,0)` hole at T36b). The deliberate *non*-additions (T35's
  backwards-economics test, T34/T36b's source mis-readings and the dropped relation operator) were
  ruled instances of existing clauses, so the charter covered cases it was not written for rather than
  growing — the intended outcome, and evidence the clauses are the right abstraction, not staleness.
- **One cosmetic note, not a change:** `docs/audit_log.md` phrased the next-due as "≈ T40", which could
  be misread as "not due until T40"; the arc-close trigger fires first (as it did at #2). Clarified in
  the log below rather than in the charter (§9's cadence text is correct as written).

## (2) Cross-tranche consistency & duplication — PARTIAL (Cowork read; Code sweep pending)

- **The `(product, model)` registry (T33b) is working as the consolidation it was meant to be** — four
  equity-option pricers (Black, Heston, HW-hybrid, defaultable) coexist, and T36a consumed all four
  through one `dispatch`. No duplicated dispatch logic re-emerged.
- **`hull_white` / `black` / `SurvivalCurve` reuse is real, not copy-paste** — T34 reused
  `HullWhiteParams`+`hw_beta`, T35/T36 the `SurvivalCurve`, every equity closed form `black_price`.
  No value-object duplication spotted on read.
- **The dividend-filter duplication (above) is the one standing rule-of-two-overdue item.**
- The mechanical duplication/oracle-power judgment is Code's sweep (below).

## (1) Oracle power — full-tree mutation sweep — DONE (Code executed; Cowork ruled)

**31 mutations full-tree, oldest-first; 30 killed, ONE survived — and it is in T8, one of the oldest
tranches.** The scoping instruction was vindicated precisely: every one of the eight mutations aimed at
the arc just built (T30–T36b) died, and the only hole in the tree is pre-Jäckel. A sweep scoped to the
arc just built would have returned a clean bill and been wrong — the signature defect, avoided.

### F-3.1 (the survivor) — the nominal bootstrap drops the accrual, nothing notices. RULED: real hole → T37.
`market/nominal_bootstrap.py` `annuity = Σ psi·discount[pay]` → `Σ discount[pay]` **survives all 1430
tests.** Not an equivalent mutant: the fixture `tests/L1/test_nominal_bootstrap.py` builds annual dates
on ACT/365F, so **every ψ = 1.0 exactly** and `psi·df ≡ df` (confirmed: annual ψ = [1.0, 1.0]). On a
semi-annual schedule ψ ≈ 0.496/0.504 and each annuity term is overstated ~2×, so the solved discount
factors — and every zero rate off them — are materially wrong. **The `src/` code is correct; the ORACLE
is blind** (a weak fixture, not a live mispricing). This is **F-1.2's exact shape — the `τ=1` degeneracy,
the charter's first-recorded §4 fixture trap — in a second instrument**, and the second time it is
pre-Jäckel and found only by re-audit (F-1.2 was the linker at re-audit #1). **Ruled: a real oracle-power
hole → the fix is a test-only tranche T37** (non-degenerate semi-annual fixture + a `ψ ≠ 1` premise,
the exact remedy T23 applied to F-1.2). No `src/` change, no OPEN.md debt (no suppression, code correct);
recorded here and queued as T37.

**The systemic reading — `τ=1` has now bitten TWICE, so T37 sweeps PROACTIVELY, not just the one found.**
Two accrual-bearing instruments (linker F-1.2, bootstrap F-3.1) carried the identical latent oracle hole,
both pre-Jäckel, both invisible until a re-audit. "Contain the failure in its domain" (§4) says: do not
wait for re-audit #4 to find a third. **T37 audits every accrual-bearing fixture in the tree for the
`τ=1`/`ψ=1` degeneracy** and asserts the non-degeneracy premise wherever an accrual is load-bearing.

### Two single-point-of-failure oracles (flagged as requested). RULED.
| mutation | killers | ruling |
|---|---|---|
| Sobol **Gray code** `G(n)=n⊕⌊n/2⌋` (T20b) | **1** | load-bearing for the whole low-discrepancy construction — the thinnest load-bearing SPOF in the tree. **T37 adds a second independent killer.** |
| T26 **Leg B** numéraire divisor | **1** | already ruled single-by-design at re-audit #2; **recorded for re-audit #4**, unchanged (no cheap second killer that isn't Leg B). |

Five oracles sit at **2 killers** (JY convexity T4; the linker ψ F-1.2, now 2 after T23's killer; the
capfloor boundary C10; the collateral-gap sign T29; the dividend PV T30) — recorded, not hardened; above
the single-point line, so the floor is met.

### Code's instrument defect, found and fixed mid-sweep — ratified as good discipline.
The first pass used `pytest -x` (stop at first failure), so **every** kill reported one killer and the
single-point-of-failure question was unanswerable — the instrument would have given a confident, useless
answer to the exact question asked. Discarded and re-run without `-x` (real counts 1–392). Five
`PATTERN-MISS` mutations (source not hit) were corrected and re-run (Box-Muller 93, Gauss-Hermite 13,
bridge 11, SE n−1 4, Gray code 1). The §9 "a timing/measurement reading is only valid when measured
correctly" spirit, applied to the sweep's own instrument. No charter line — it is the existing
measure-the-instrument discipline, and Code caught it itself.

---

### F-3.2 (the proactive sweep's find) — the vol surface drops the accrual too. RULED: same class → T37 part 1b.
Part 2's proactive sweep immediately found the **third** instance: `models/implied_vol_surface.py` drops
`ψ` from `discounted_unit = notional·ψ·df` and **survives all 1433 tests** (annual-date fixture, every
`ψ=1`). Measured: a 9.00% vol reads as **5.80%** semi-annually — a ~36% understatement of every surface
node, and the surface is T10+'s calibration target. The `src/` is correct; the oracle was blind. Folded
into **T37 part 1b** (same remedy, confirmed by mutation) rather than a separate tranche — τ=1 is now a
known class with a known fix, so same-class survivors are fixed in-tranche, not raised one-by-one (the
scope ruling). A *different*-class survivor would still be raised.

### The pattern — τ=1 has bitten THREE pre-Jäckel instruments from ONE fixture convention. RULED: gate it (T26b precedent).
F-1.2 (linker, T3, re-audit #1), F-3.1 (bootstrap, T8), F-3.2 (vol surface, T9) — all pre-Jäckel, all
from annual dates on ACT/365F making every accrual exactly 1.0, all invisible to manual review and all
caught only by a mutation sweep. Three drifts of one discipline is the **T26b situation** — the project
gates a repeatedly-drifting discipline rather than restating it (the references gate earned its place at
five drifts; this has three and a measured 36% latent mispricing behind it). **Ruled: T37 part 3 assesses
and, if a clean mechanism exists, builds a mechanical guard** flagging an accrual-bearing fixture whose
accruals are all unit; if no clean mechanism, the exhaustive sweep + §4's existing premise rule stand and
the guard is recorded. **No charter prose line now** — the gate *is* the artifact (documented like the
references gate when it lands at T37's review); restraint holds, the method is already in §4.

### Single-point-of-failure oracles — updated count: FOUR. RULED.
The sweep (run correctly, without `-x`) found two more beyond §(1)'s first pair: **MM-I's freezing
denominator `τ/(1+τ)` (T13, 1 killer)** and the **YoY first period's `ψ` (T5, 1 killer)**. With the Sobol
Gray code (1) and T26's Leg B (1, single-by-design), that is **four** single-killer oracles. **Ruled:**
T37 adds a second killer to the Gray code (load-bearing, the thinnest) and to MM-I / YoY if cheap; else
recorded for re-audit #4 (floor = record, ceiling = harden, §9). The linker `ψ` (F-1.2) is now at **2
killers** (T30 added the second per re-audit #2's ruling) — direct evidence the second-killer remedy works.

## Disposition
Register and charter passes clean. The mutation sweep found **two real oracle-power holes of one class**
(F-3.1 bootstrap, F-3.2 vol surface — the `τ=1` degeneracy, third and second-of-three with F-1.2) and
**four single-point-of-failure oracles** (Gray code, MM-I freezing denominator, YoY first ψ, T26 Leg B).
All → **T37**, a test-only oracle-hardening tranche: fix F-3.1 (done) + F-3.2, an **exhaustive** proactive
τ=1 sweep fixing every same-class survivor in-tranche, a **mechanical guard** for the τ=1 fixture
convention if a clean mechanism exists (the T26b "gate a thrice-drifted discipline" precedent), and second
killers for the thin oracles. No `src/` defect (the code is correct throughout; the oracles were blind),
no new debt, no charter staleness. **Re-audit #3's reading is complete; its remediation (T37) is specced
and queued** — the arc boundary opens once T37 lands, since a re-audit does not leave the holes it found
open across the boundary (its own backstop logic). The oldest-first scope is the finding of record: every
T30–T36b mutation died; all holes were pre-Jäckel.
