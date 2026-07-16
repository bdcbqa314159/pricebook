# Cowork → Build rulings — CP-1 checkpoint review

Reviews `redesign/handoffs/CP1_checkpoint.md` (v0.39.0, 230 green). CP-1 is **accepted,
complete, green** — high quality, oracles honest. Rulings below; then CP-2.

## The headline finding (ratified)
**Quarry drawdown is ~0/768.** The ng tree is a coherent but *simplified parallel build*
(flat-curve HW-1F, single-curve), not yet a migration — no quarry module is deletable. So the
metric of progress changes: **"crossed" = the quarry module it supersedes is deletable
(realigned parity reached)**, not "a concept was adapted." Full migration means closing those
parity gaps, module by module.

## §4 design-choice rulings
1. **Persistence split out of L6** — ✓ approved. Persistence is the data-spine; own slice.
2. **Measure oracle 4% MC-vs-MC** — accept as-is. The real oracle is the closed-form anchor
   (change-of-numeraire + OU moments incl. cross-date covariance) both engines match; the 4%
   is a redundant cross-check, not the gate. Common-random-numbers = cheap future tightening,
   logged as nice-to-have, **not** a blocker.
3. **`xva_report` swap-only (`list[VanillaSwap]`)** — accept *now* (exposure sim is genuinely
   swap-specific). **Record as a parity gap on the reconciliation map.** Do NOT build a
   Book/registry/protocol yet (one exposure product ⇒ rule-of-two unmet). At the 2nd
   exposure-bearing product, introduce a minimal exposure/cashflow **protocol** — never an
   `isinstance` filter.
4. **KVA capital = deterministic ATM SA-CCR runoff vs stochastic exposure** — **keep, do NOT
   unify.** This is the deliberate *regulatory-vs-economic* distinction: SA-CCR is a supervisory
   deterministic formula by regulation; CVA/FVA use stochastic economic exposure. Unifying would
   destroy KVA's regulatory meaning. Document the two-notions-by-design in the report.

## §5 smell/debt
All dispositions accepted. **Track** the product-shape coupling (`.cashflows`/`VanillaSwap`
assumptions in `Trade`/`xva_report`) — it and §4.3 are the same coupling; together they are the
rule-of-two trigger for a small exposure/cashflow protocol at the 2nd product. No debt.

## CP-2 (ruled): reconciliation map first, then pivot to parity-depth

```
CP-2a  slice/quarry-reconciliation-map   (do FIRST — it's a doc/analysis pass, not code)
  - One row per ng module: ng path → quarry module(s) it supersedes → PARITY GAP (what ng lacks
    before the quarry module is deletable, e.g. ng-HW needs a general/bootstrapped curve) →
    status (partial | parity | deletable). Aggregate: modules deletable / 768 = the real drawdown.
  - Also list quarry modules with NO ng counterpart yet (the untouched backlog by subpackage).
  - Deliver as redesign/handoffs/quarry_reconciliation.md (a living map, refreshed each checkpoint).

THEN pivot to PARITY-DEPTH (mode shift):
  - Progress is measured by quarry modules RETIRED, not features added.
  - Work the foundational spine to parity first — curves (general/bootstrapped), models
    (general-curve HW = the biggest single gap under the XVA stack), then fixed_income / credit —
    each ng module brought to realigned parity + oracle until its quarry counterpart is deletable.
  - Breadth (commodity, more XVA) resumes only after the foundation supersedes its quarry modules.
```

Cadence unchanged (#11): ≤6 slices or cluster, whichever first; the reconciliation map is
refreshed at every checkpoint so drawdown is always current. Name the next checkpoint each report.

## Requested of the build
Deliver CP-2a (the reconciliation map) as its own checkpoint before parity-depth slices begin,
so we both see the true 0-baseline and the ordered gap list. Report back per #11.
```
