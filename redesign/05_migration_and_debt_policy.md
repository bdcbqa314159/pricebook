# Artifact #5 — Migration & Debt Policy (DRAFT)

**Status:** Draft. The rules for crossing an entry from the quarry into the new tree, and
for tracking any debt incurred. This is the artifact that becomes enforcement: a trimmed
version of it + the spine lands as the repo-root `CLAUDE.md` for the Claude Code build.

---

## Part A — Migration: crossing an entry from quarry to spine

### The quarry rule
The old tree (`python/pricebook/…` today) is a **read-only quarry**. It is never edited
in place and never deleted. The new tree grows beside it. "Done" is defined as **the
quarry empty** — every entry either migrated or explicitly archived.

### Copy-ADAPT, never copy-paste
An entry does not move; a *new* aligned version is written in the new tree, informed by
the quarry original. Every crossing must do at least one of: conform to the spine layer,
speak the ratified vocabulary, drop self-pricing behaviour into the engine, or shed debt.
A byte-for-byte copy is a failed migration — if nothing changed, the entry was already
aligned and belongs to a different slice, or it is being copy-pasted (forbidden).

### The green-oracle gate (nothing crosses grey)
No entry is "landed" until a **red/green oracle** proves it correct against a known value:
- closed form where one exists (the strongest oracle),
- QuantLib or an ISDA/reference model cross-check,
- self-consistency (inputs reprice to par / to zero NPV),
- a trusted external mark.
"It runs and looks right" is **not** an oracle. Eyeballed correctness does not count —
pricing bugs are numerical, not syntactic, and only comparison surfaces them.

### Provenance (the educational constraint, enforced)
Each migrated entry records, at minimum: the quarry path it descended from, the
paper/book/model it implements (a `documents/` reference or citation), and the oracle
that gates it. This is what keeps the library legible and learnable.

### The per-entry checklist
```
[ ] AUDIT   quarry original read; correctness, deps, and debt understood
[ ] ALIGN   rewritten to spine layer + vocabulary; instrument behaviour → engine
[ ] ORACLE  named oracle chosen; red first, then green
[ ] DEPS    imports point strictly downward (per-commit acyclic check passes)
[ ] PROVEN  green to stated tolerance; statelessness holds (repeatable, no mutation)
[ ] PROV.   provenance + oracle recorded in the ledger
[ ] MARK    ledger Status → landed
```
A slice may batch several entries, but every entry passes the whole checklist.

### Layer discipline during migration
Migrate **bottom-up**: an entry may only land if everything it depends on has already
landed. This is why the ledger is ordered by layer and Slice 0 is the foundation
walking skeleton. An entry that "needs" something not yet migrated signals either a
mis-layering (fix the design) or that its slice is out of order.

---

## Part B — Debt: logged, never silenced

### The one rule
**No debt is silenced silently.** Every suppressed warning, `# type: ignore`, empty
`except`, skipped test, shim, or load-bearing TODO is written into the debt ledger with a
rationale and a re-open trigger — or it does not go in at all.

### The ledger
`OPEN.md` is the existing, working debt ledger and stays the single source of truth. Its
structure is already good and is adopted wholesale:
- **Hot debt** — active, being paid down (e.g. the serialisation-typing mypy sweep).
- **Held by design** — deliberate, with a rationale *and* a named re-open trigger
  (e.g. "registry duplicate-registration silent ignore — re-open if it masks a real
  double-register").
- **Deferred bundles** — coordinated multi-slice work with a trigger condition
  (e.g. the schema-v2 shim removal, triggered at a release boundary or when shim count
  crosses ~25).
- **Open & pickable** — small one-slice items ready to take.

### Rules for incurring debt during a migration
1. **Never** to pass the oracle gate. Debt may never buy a green light; a slice that
   can't go green honestly is not done.
2. Allowed only for *deferred scope* (a feature not in this slice), never for *hidden
   wrongness* (a suppressed error).
3. Every new debt entry names: what, why, the trigger to re-open, and the owning slice.
4. A suppression with no ledger entry is a **defect**, caught in review, not merged.

### The debt invariant (checkable)
The gap between "warnings the toolchain would emit" and "warnings suppressed" must equal
the debt ledger — no more, no less. Suppressions minus ledger entries = 0. This is the
machine form of "no silent debt," and can be a CI check.

---

## Part C — What becomes `CLAUDE.md` (design → build handoff)

When the design phase is done (all five artifacts + Slice 0 specified), a **trimmed**
composite becomes the repo-root `CLAUDE.md` that governs the Claude Code build agent —
guardrails, not suggestions:

1. **The spine** — the six layers and the one-way dependency law (from #2).
2. **The vocabulary** — the stable value types and the ratified decisions (from #3).
3. **The stateless-engine contract** — the five invariants (from #2).
4. **The migration rules** — quarry read-only, copy-adapt, green-oracle gate, bottom-up
   (Part A).
5. **The debt rule** — log never silence, ledger discipline, the suppression = ledger
   invariant (Part B).
6. **The slice discipline** — one vertical cut, ships with an oracle, small enough to
   check in one pass (from #4).

Everything else in these artifacts is design rationale and stays in `redesign/`; only the
enforceable guardrails go into `CLAUDE.md`.

---

## Definition of done for the design phase
All five artifacts exist and Slice 0 is fully specified with its oracle. ✅ (pending
ratification of this artifact). At that point work moves to Claude Code under the
generated `CLAUDE.md`, and the first build task is Slice 0.
