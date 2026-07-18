# redesign/ — reading order

Docs accumulated as the design evolved. **Several early ones were written under assumptions later
corrected** — status is marked so nothing stale is read as current.

---

## ► START HERE (3 docs — everything needed to begin)

| # | doc | what it is |
|---|---|---|
| 1 | **`handoffs/handoff_topic0_foundation.md`** | **the active hand-off** — 7 slices, what to mine, oracles, checkpoint |
| 2 | **`16_topic0_foundation.md`** | **the Topic 0 spec** — what the foundation *is*, cross-asset complete |
| 3 | **`../CLAUDE.md`** | the guardrails — law, not suggestion |

Current plan in one line: **park all ng → build Topic 0 (foundation, complete) → park the ~12 quarry
foundational modules → then Topic 1 (yield curves).**

## ► THE METHOD (why it's shaped this way)

| # | doc | status |
|---|---|---|
| 4 | `13_topic_migration_and_parking.md` | **current** — topics, tick mechanism (§3), target/use/**apply the policy** (§5) |
| 5 | `12_domain_build_order.md` | **current** — block order replaced demand-driven; drawdown reports, never steers |
| 6 | `11_checkpoint_and_review_cadence.md` | **current** — ≤6 slices or cluster; the five review inputs |

## ► REFERENCE WHILE BUILDING

| # | doc | status |
|---|---|---|
| 7 | `15_foundation_comparison.md` | **current** — where each quarry `core/` module actually belongs |
| 8 | `02_spine.md` | **current** — layers + **Amendments A1–A6** (engine contract, temporality, keyed snapshot, measure) |
| 9 | `09_verification_and_audit.md` · `10_ci_and_cross_platform.md` | **current** — `verify.py`, CI matrix, tolerance oracles |
| 10 | `06_versioning_and_release_policy.md` · `07_branching_and_commit_policy.md` | **current** — 0.x→1.0, branch per slice, red-before-green |

## ► LATER (Topic 1 — do not read yet)

| doc | when |
|---|---|
| `14_topic1_object_model.md` | when Topic 0's gate is green — the yield-curve objects (`CurveSet`, `RateIndex` capstone, curve risk) |
| `parked/topic-01-yield-curve/MANIFEST.md` | Topic 1 file set + the critical findings from the quarry read |

## ► HISTORICAL / PARTLY SUPERSEDED

| doc | status |
|---|---|
| `01_scope_contract.md` | scope decision (full cross-asset, one discipline bar) **still stands**; its drawdown framing is superseded by #12 |
| `03_vocabulary.md` | absorbed into #14/#16 — read those instead |
| `04_slice_plan.md` | **superseded** — the topic method (#13) replaced it; kept for the Slice-0 walking-skeleton record |
| `05_migration_and_debt_policy.md` | debt rules **still valid**; migration ordering superseded by #12/#13 |
| `08_handoff_protocol.md` | extended by #11 — read #11 |
| `L0_ledger.xlsx` | retired as a tracker (per-topic manifests replaced it) |
| `kickoff_slice0.md` | historical — the original build kickoff |

## ► handoffs/ — the build↔design record

Rulings and checkpoint reports, newest last. **`handoff_topic0_foundation.md` is the only active
hand-off**; `handoff_topic1_conventions.md` is explicitly marked SUPERSEDED. The rest
(`CP1`–`CP4` checkpoints, `rulings_*`) are the historical record of decisions — useful for *why*,
not *what to do next*.

Notable ones worth knowing exist:
- `rulings_deletable_definition.md` — what "deletable" means (supersede, not clone)
- `rulings_spotcheck_retire_1.md` — the evidence protocol for a `dead` claim
- `rulings_CP3_correction.md` — the phantom-residual rule
- `rulings_spine_conformance.md` — semantic layer conformance (the `black.py` precedent)
