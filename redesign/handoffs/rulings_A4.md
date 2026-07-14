# Cowork → Build rulings (Amendment A4) — answers to the v0.2–v0.8 report §5s

Ratified in `redesign/02_spine.md` Amendment A4 and `CLAUDE.md`. Read those; summary +
the one action slice below.

## Rulings

- **A4.1 — demand-driven vertical migration: confirmed.** Keep pulling quarry entries as
  slices need them; the report ledger-delta tables are the crossing record. No per-layer
  ledger gate.
- **A4.2 — promote `SurvivalCurve` into `MarketSnapshot`** (the one that needs code). The
  hazard curve is market data; move it out of `CreditModel` into the snapshot alongside
  `discount_curve` + `FixingHistory`. `CreditModel` reads it from the snapshot. Then rebuild
  `credit01` to go through the **same `Priceable` / bump-the-snapshot** path as `dv01` —
  delete the bespoke `credit01(cds, model)` bypass.
- **A4.3 — building-block math boundary: ratified.** Curve/model may expose closed-form
  blocks (`RPV01`, `cds_par_spread`, HW `zero_bond_option`); the L4 engine composes them;
  the product stays pure data. No relocation needed — the current placements are fine.
- **A4.4 — confirmed defaults:** engine/model registry lands with the first mixed trade
  (not before); `PricingResult` fields grow on demand; demand-migrate the minimum (stdlib
  `random` is fine); `accrued` nominal + `clean = dirty − accrued` ratified. No action.
- **Idle `FixingHistory`** — resolved: the seasoned-swap `float-leg-fixings` slice now
  consumes it. No action.
- **Slice-plan renumber** — done on the Cowork side (`04_slice_plan.md` now records the real
  descriptive history). No action.

## The one action slice

```
BRANCH slice/survival-in-snapshot   (Amendment A4.2)
  - Move SurvivalCurve into MarketSnapshot (first-class, like discount_curve + fixings).
    CreditModel is built from the snapshot's survival curve, not given one directly.
  - Rebuild credit greeks: credit01 bumps the SNAPSHOT's survival curve and reprices through
    the Priceable protocol — same machinery as dv01. Remove the bespoke credit01(cds, model).
  - Guard/oracle: behaviour-preserving — CDS PV and credit01 values byte-identical to pre-move
    (reuse the par-CDS→0 and independent-hazard-FD oracles). Add a test that credit01 now goes
    through Priceable (structural, like the A1 binding test).
  - verify.py all + ruff green both OSes; docs/provenance; CHANGELOG (Changed); version MINOR;
    rebase-and-merge.
```

After this, credit and rates risk share one `Priceable` path (restores A1), and forward work
resumes: next asset-class or curve refinement, each oracle-gated, on the corrected contract.
Report drift back as usual.
```
