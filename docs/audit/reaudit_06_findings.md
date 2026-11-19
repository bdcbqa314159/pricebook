# Re-audit #6 — the collateralised-discounting arc close (v0.60.0, 2026-11-19)

Sixth whole-tree re-audit (§9), fired by the **arc-close trigger** (~nine tranches since re-audit #5 at the
Gatheral close — T50–T53 the Gatheral tail, T54/T55/T57/T58 this arc; arc-close fires first, as every time).
Run with independence: Cowork sets scope and rules; the mutation sweep is Code's execution on the **committed
instrument** (`tests/tooling/mutation_sweep.py`, T52); the register-accuracy and charter-staleness passes are
done here by Cowork.

Four §9 targets: **(1) oracle power** (full-tree sweep — Code, PENDING), **(2) duplication**, **(3) register
accuracy**, **(4) charter staleness**. (1) and (3) run together. Conduct rule (the house discipline): a
surface's report is content, not the artifact — Code re-derives each premise; the sweep reports
applied/killed/survived with the bytecode and ERRORED guards (the instrument encodes them).

---

## (1) Oracle-power mutation sweep — PENDING (Code), scope set

Full-tree, **oldest-first** (§9), in order:

1. **FIRST — the proactive invariant-6 / historical-flow sweep** (the tenth-shape escalation's standing run).
   Drop the filter on every module that has one; the arc added the IRS, the CSA pricer and the collateralised
   exposure. The option-bypass sub-shape (F-4.1 moneyness) and the straddle premise both apply.
2. **The single-curve design invariant — a NEW named sweep this arc, because its failure is silent (T33b).**
   The arc's invariant is that single-curve and multi-curve are co-equal selectable modes. Sweep the
   discounting-touching modules (T55's pricer, T58's basis factor) for a mutation that makes a `CurveSet`
   **mandatory** (ignore a single-curve snapshot / force projections): the *differ* half of the invariant
   oracle must kill it. The invariant is structural in two places (the snapshot shape, the basis factor = 1)
   — confirm the structure actually holds, do not trust the report.
3. **MM-I's owed second killer** (`market_model_i.py:154/:155`) — two load-bearing holes behind the *same one*
   test, recorded since re-audit #5. **This re-audit's highest-value single-killer target**: add the second
   independent killer, or confirm it was added. Plus the standing single-killer re-run table
   (`docs/audit_log.md`).
4. **The new closed forms, first sweep** — the OIS/projection bootstrap (T54), the CSA pricer (T55), the LGM
   and its MC companion (T57), the threshold + the capstone wiring (T58). Probe the T39 reduction blind spot
   the arc paid twice (my eq-11.4 and threshold errors) and the T43 scale-invariance; confirm the LGM's HW
   reduction + MC companion both bite.
5. **The rest of the tree (T1–T53)** as the standing full-tree sweep.

Survivors ruled by Cowork, appended here. Any real hole → a test-only fix-tranche (the T23/T37/T45/T51
pattern).

## (2) Duplication — PARTIAL (Cowork read; Code sweep informs)

- **C2 (vol lives on one clock) — still OWED, the widest-unguarded.** Untouched this arc. Carried.
- **C14 (total-variance dimension, T46) and C15 (OIS-vs-projection, T55) — both guarded.** No new OWED
  convention opened this arc.
- **`dividends_before(T)` consolidation — still owed** (no equity touch this arc). Carried.
- No new duplicated value object this arc; the IRS/`Tenor`/`projections` are single-sourced (Code verifies).

## (3) Register-accuracy pass — DONE (Cowork)

- **`risk/collateral.py` (T29) — the oracle wording NARROWED (the T58 ruling, done here).** "Collateral
  never increasing the adjustment" overclaimed a theorem; it is a **plausibility check** (aggregate, a
  well-behaved mark; a gross breach signals implausible inputs — which is how it caught T58's reciprocal
  basis). The `oracle:` provenance line now says so — the §4 floor rule applied to an oracle's own
  description. No code/test change; the check stays, its description is honest.
- **`to_go_deeper.md` — the `LICHTERS-2015` header fixed** ("T54–, in progress" → "T54–T58 CLOSED v0.60.0");
  the T56 cross-currency deferral and the T29-fence routes are current.
- **`references.md` — clean.** `LICHTERS-2015` is `worked-from` with `src/` citers.
- **`OPEN.md` — OI-6 open and correct** (the T8 skipped-pillar `KeyError`, §3.4, low severity, re-open at
  the next `nominal_bootstrap.py` touch or this re-audit). It is the only open item. **Candidate to fix in
  the remediation tranche** if this re-audit produces one — a one-line `PricingFailure` guard with a test,
  a natural `nominal_bootstrap.py` touch.
- **`conventions_register.md` — clean and gated** (C15 added guarded, T55; `verify.py conventions` holds).
- **Arc plan (`modern_discounting_arc.md`) — current.** T56 struck, queue closed, single-curve invariant
  recorded, completion ratio matches the checkpoint.

## (4) Charter-staleness pass — DONE (Cowork): clean

- **No clause stale or in tension.** The two arc additions (§10 the step-1 reuse grep; §4 the
  relative-tolerance mirror) describe reality and each cite the earning tranche. The deliberate non-additions
  (the Cowork transcription errors caught by §5 + named reductions; the T29 narrowing; the frozen-basis
  choice; the loop-invariant lens) left the charter leaner-relative-to-coverage, as at the last three
  capstones.
- **Method-watch items carried (not charter):** the three-ceiling heavy-MC-file watch (a fourth earns the
  `slow`-budget conversation, §9, Bernardo's); the glyph-recovery sub-kinds (2 instances, from the Gatheral
  arc) and the first-shape-twice counter (Gatheral) — both still at two, carried.

---

## Status

- (3) and (4): **DONE** (Cowork). The T29 narrowing done; one stale header fixed; charter clean.
- (1) and (2): **scope set, PENDING Code's execution** oldest-first, the proactive invariant-6 sweep first,
  **the single-curve-invariant reachability a named new sweep**, MM-I's second killer the highest-value
  single-killer target.
- Remediation (if any survivor is a real hole, or MM-I's killer is still owed): a test-only fix-tranche,
  which can also close OI-6 (a natural `nominal_bootstrap.py` touch).

Re-audit #6 **logged open**; the arc boundary is settled once the sweep lands and any remediation is ruled.
