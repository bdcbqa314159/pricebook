# Claude Code kickoff — Slice 0 (walking skeleton)

Paste the block below into a fresh Claude Code session opened at the repo root
(`~/work/pricebook`). It launches the build under the ratified guardrails.

---

```
Read CLAUDE.md at the repo root — it is law, not suggestion. Then read redesign/02_spine.md,
redesign/03_vocabulary.md, redesign/04_slice_plan.md, redesign/05_migration_and_debt_policy.md,
redesign/07_branching_and_commit_policy.md, and redesign/09_verification_and_audit.md. Also read
redesign/L0_ledger.xlsx for the foundation entries' dispositions (the YOUR CALL column).

We are building Pricebook by MIGRATION, bottom-up, one slice at a time. The old tree
(python/pricebook/) is a READ-ONLY QUARRY — never edit or delete it; it is your reference oracle.
The new tree grows at src/pricebook_ng/ (imports as `pricebook_ng`; it takes the `pricebook` name
only at v1.0 when the quarry is empty).

Your task is SLICE 0 only — the walking skeleton from redesign/04_slice_plan.md: a single fixed
cashflow discounted on a flat curve, priced end-to-end L0→L6 through the stateless engine.

Work on branch `slice/00-walking-skeleton` off main, with meaningful commits in this order
(red before green is a HARD rule):
  1. chore(bootstrap): scaffold src/pricebook_ng/ layer dirs (foundation, market, instruments,
     models, engine, risk, shell) + packaging with __version__ = "0.0.0"; create CHANGELOG.md
     (Keep-a-Changelog); create verify.py with subcommands acyclic / tests --layer N / debt /
     provenance / version / all. Leave RELEASE_NOTES.md and python/pricebook/ untouched.
  2. test:  Slice 0 oracle RED — a failing reference-value test:
              PV == notional * exp(-r * t)   (t via ACT/365F)  to < 1e-12
              analytic DV01 == -notional*t*exp(-r*t)*1e-4 vs finite-difference to < 1e-6
              statelessness: repricing is byte-identical
  3. feat:  make it GREEN — L0 Money + Cashflow (Cashflow promoted from fixed_income to L0);
            L1 flat MarketSnapshot + DiscountCurve behind a CurveHandle (df=exp(-r t));
            L2 FixedCashflowTrade (frozen dataclass, NO pv method); L4 DiscountingEngine.price(...)
            returning PricingResult; L5 DV01 by bumping the snapshot; L6 book(trade).value(date).
  4. docs:  provenance header (quarry path / source / oracle / slice) on each new module;
            CHANGELOG.md entry.
  5. chore(release): bump __version__ to 0.0.1.

Honour the invariants: stateless engine (referentially transparent, no ambient state, frozen
inputs, PricingFailure not exceptions, explicit NumericalConfig); instruments are pure data;
dependencies point down only (verify.py acyclic must pass); no speculative abstraction (CLAUDE.md
§6b — solve exactly Slice 0, nothing more); every suppression logged in OPEN.md.

Land with rebase-and-merge onto main only when the Slice-0 oracle is green AND
`verify.py tests --layer 0`, `acyclic`, `debt`, and `version` all pass.

Do NOT proceed past Slice 0. When the L0 layer's slices are complete, STOP and write the Layer
Completion Report to redesign/handoffs/L0_report.md per redesign/08_handoff_protocol.md, then emit
the one-line return message so we resume in the design workspace.
```

---

## After Slice 0
Slice 0 proves the spine end-to-end. The next slices (S1 day-count/schedule, S2 discount-curve
bootstrap) continue the L0→L1 migration per redesign/04_slice_plan.md, each ruled in the ledger
first. When all L0 entries are landed, the L0 report brings you back here to rule L1.
