# Audit log — the periodic whole-tree re-audit

Charter §9: a full re-audit **every 11 tranches, and at each arc close**, run with independence (the
`analysis/` read-only copy + `AUDIT_FINDINGS.md` pointed at current `main`). It targets what the
per-push gates cannot — oracle power (a full-tree mutation sweep), cross-tranche consistency and
duplication, register accuracy (`to_go_deeper`/`OPEN`/`references`/arc plans), and charter staleness.
Findings are ruled by Cowork and become tranches or ledger entries.

This file is the **counter**, so the trigger is mechanical, not remembered.

| re-audit | at version | tranche | findings | notes |
|---|---|---|---|---|
| — | — | — | — | mechanism established 2026-11-07 (cadence 11, Bernardo). |
| #1 (**complete**) | v0.27.0 | T22 | **2 HOLES + 1 MED + 3 LOW** | first re-audit, Jäckel arc close, whole tree T1–T22; run independently of the builder from two fresh contexts (Code built all 22), findings → `docs/audit/reaudit_01_findings.md`, ruled by Cowork. See below. |
| #2 (**complete**) | v0.35.0 | T29 | **CLEAN — 0 holes + 1 LOW (fixed)** | **XVA arc close** (arc-boundary trigger, not the cadence counter — eight tranches since #1). Whole tree **T1–T29**, independent. No oracle holes, no silent weakening from the XVA refactors, charter clean; one LOW (stale conventions-register summary) fixed in place. Findings + rulings → `docs/audit/reaudit_02_findings.md`. |

## Re-audit #2 — result (2026-11-10, v0.35.0, T1–T29)

A **clean bill**, cleaner than #1. Fired by the XVA arc close. Findings/rulings in
`docs/audit/reaudit_02_findings.md`.

- **Oracle power: CLEAN.** ~48 mutations, 0 holes; one proven-equivalent survivor. The signature-defect
  checks — did generalizing T25 into `credit_adjustment` (T27), extending T26's engine with collateral
  (T29), or the `is_historical` atom across five engines weaken an older oracle? — all held. F-1.2
  confirmed closed; the new `references` gate itself powered.
- **Duplication: CLEAN** (F-2.1 still one consumer short, correctly open).
- **Register accuracy: 1 LOW.** F-3.1 — the conventions-register "Immediate reading" summary was T24
  prose not updated under T26 (called C10 unguarded, counted five conventions). Rows were correct;
  summary **fixed in place**, no fix-tranche.
- **Charter staleness: CLEAN.**

## Re-audit #1 — result (2026-11-07, v0.27.0, T1–T22)

Two independent mutation passes (fresh contexts; mutants regenerated, not reused — Code's T15–T22
sweeps did not survive its session) plus the three tree-wide judgment passes. Findings and rulings in
`docs/audit/reaudit_01_findings.md`.

- **Oracle power — Jäckel arc (T15–T22): CLEAN.** 26 mutations, 0 holes; 2 proven-equivalent survivors.
- **Oracle power — inflation arc (T2–T14): 1 HOLE.** F-1.2 [MED] — the linker coupon `ψ` accrual
  survives a drop under all 818 tests (every fixture `ψ=1`, the §4 `τ=1` trap); production code
  correct, oracle degenerate; ~8.4% latent mispricing on a semi-annual linker. → **fix-tranche T23**
  (oracle hardening). *Found only because the sweep was extended past the first pass's domain — the
  §4 rule applied to the audit itself.*
- **Duplication & consistency:** 2 LOW (F-2.1 driftless-forward terminal inlined twice → rule-of-two,
  `to_go_deeper`; F-2.2 discount-from-forwards ×3 but different shapes → no action).
- **Register accuracy:** 1 MED + 1 LOW. F-3.1 — `conventions_register.md` C1–C6 describe the archived
  predecessor tree → **fix-tranche T24** (rebuild). F-3.2 — `to_go_deeper` stale lines → folded into T24.
- **Charter staleness: CLEAN.**

**Remediation: COMPLETE.** **T23** (v0.28.0) closed F-1.2 — the linker accrual oracle, test-only, the
drop-`ψ` mutant reproduced then killed. **T24** (v0.29.0) rebuilt the conventions register, corrected
`to_go_deeper`'s two stale lines and recorded F-2.1.

**T24 found a SECOND hole the audit had not, which is why the count above reads 2.** Re-deriving the
invariant-6 row meant mutating the valuation-date boundary in all five engines rather than reading
them: `>` → `>=` **survives** in three. Two of those three (`nominal_cap`, `yoy`) are **proved
equivalent mutants** — bit-identical prices across four schedules including the boundary, because a
Hull-White caplet whose expiry has passed is worth exactly zero. The third (`capfloor`) was a **real
41% mispricing** (5,929 → 8,364 on a schedule paying exactly on the valuation date), closed by
`test_INVARIANT_6_excludes_a_period_PAYING_EXACTLY_on_the_valuation_date`.

Worth recording for re-audit #2: the finding came from **re-deriving a register row by mutation
instead of by reading the code**. The register rebuild was scoped as a docs tranche; the hole was a
by-product of doing it honestly. Target 3 (register accuracy) and Target 1 (oracle power) are less
separable than the §9 list suggests.

**Next due (superseded — re-audit #2 ran at the T29 arc close; see its result above):** re-arms at
whichever of {next arc close, 11 tranches since **re-audit #2** at T29 ≈ **T40**} comes first.

---

## Re-audit #3 — equity & hybrids arc close (v0.44.0, T36b, 2026-11-13)

Fired by the **arc-close trigger** (the eleven-tranche cadence counter is at nine since re-audit #2 at
T29: T30, 31, 32, 33, 33b, 34, 35, 36a, 36b — so arc-close fires first, as at #2). Findings:
`docs/audit/reaudit_03_findings.md`.

- **Register-accuracy + charter-staleness passes (Cowork): clean.** One stale `to_go_deeper` header
  fixed (BROCKHAUS arc marked closed; the 13.1.2-for-the-stock route added); one overdue code-hygiene
  route re-flagged (the dividend-filter consolidation, `dividends_before(T)`, never done across the
  arc). `references.md`, `OPEN.md` (zero suppressions), the arc plan all current. No charter clause
  stale or in tension — the arc's lessons were ruled instances of existing clauses, so the charter
  covered new cases rather than growing.
- **Oracle-power mutation sweep (Code): PENDING**, scope set — full-tree, **oldest-first** (§9's
  binding emphasis; re-audit #1's only hole was in the oldest instrument), single-point-of-failure
  oracles flagged, the new L0 `Φ₂` and the equity closed forms swept for the first time. Survivors
  ruled by Cowork, appended to the findings doc.

**Next due:** re-arms at whichever of {next arc close, 11 tranches since **re-audit #3**} comes first.
(The "≈ T40" in re-audit #2's note was the cadence-counter estimate; the arc-close trigger fired first,
as it has every time — the counter has never been the firing trigger.)
