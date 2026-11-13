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

## (1) Oracle power — full-tree mutation sweep — PENDING (Code executes on this scope)

**Scope Cowork sets (handed to Code in thread 0199):**
- **Full-tree, oldest-first.** §9's binding emphasis: *"full-tree means the oldest tranches, not the
  arc just built."* Re-audit #1's only mutation-pass hole (F-1.2) was in the **oldest** instrument.
  So the sweep weights the pre-equity tree (L0 foundation, the inflation spine T2–T14, the Jäckel arc
  T15–T22, the XVA spine T25–T29) at least as heavily as T30–T36b — an arc-close re-audit scoped only
  to the arc just built would repeat the signature defect committed against the audit itself.
- **Flag single-point-of-failure oracles.** Any load-bearing mutation killed by exactly one test is
  recorded (a later test deletion silently reopens it); those guarding a previously-shipped hole want a
  second independent killer, per re-audit #2's F-1.2 / numéraire-divisor precedent.
- **The new L0 `Φ₂` and the equity closed forms** get their first full sweep here (they were
  per-tranche-mutated at build, never re-swept).
- Survivors are **ruled by Cowork** (hole vs equivalent-mutant-proved-bit-identical vs unreachable),
  and become tranches or `to_go_deeper`/`OPEN` entries.

**Status:** awaiting Code's execution and survivor report. Findings and rulings appended here.

---

## Disposition
Register and charter passes clean (one stale header fixed, one overdue hygiene route re-flagged). No
new debt, no charter staleness, no capability defect found on the documentation/consistency side. The
clean bill is provisional on the mutation sweep (1), which Code executes on the scope above; this doc is
updated with its survivors and Cowork's rulings before the re-audit is logged complete.
