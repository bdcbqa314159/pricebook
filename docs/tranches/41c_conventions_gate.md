# Tranche 41c — `verify.py conventions` (the register-integrity gate)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.48.2` (tooling only; no
`src/` numerics). **Source:** none external — a gate the T41b duplicate-id finding earned. **Runs before
T42.** The T26b (references gate) and T38 (gates gate) precedent: gate a statically-decidable discipline
that manual reading has shown it cannot hold.

## 0. Why this is a tranche
T41b found the conventions register carried **two `C11` rows** — the T39 protection-leg row took an id
the invariant-6 `model.market` row already held. **Two review passes read past it, including the Cowork
pass that ratified "C11" by name at T39.** The register has **no integrity check and nothing cites it
mechanically**, so a duplicate id — or a citation to a row that does not exist — is invisible to every
gate. That is the **T26b references-drift shape** (a citation discipline with no gate), and it is
**statically decidable** (parse the rows, compare ids and citations), so by §9's own boundary — *a
drifted discipline is gated only when its violation is statically decidable* — it is gateable, cheaply
and without an exemption list (unlike the τ=1 guard, which was declined for exactly the opposite reason).

## 1. The gate (`verify.py conventions`)
A new `verify.py` subcommand, **listed in §9 and invoked in `ci.yml`** (so the T38 gates-gate covers it),
that parses `docs/conventions_register.md`'s live rows and asserts:
1. **Unique ids** — no two live rows share a `C<n>` id (the T41b defect).
2. **Citation resolution** — every `C<n>` referenced in the register's own prose (the summary lines, the
   "guarded structurally" list) resolves to exactly one live row (no dangling or ambiguous citation).
3. **Contiguity of the live sequence** — the live ids are a gap-free run from the stated floor (the
   register says *"new live rows continue at C10"*; assert C10, C11, C12, C13 … with no hole), so a
   renumber that drops a row is caught.

Keep the scope **to what is statically decidable from the register text** — ids and citations. Do **not**
reach into `src/` to check that every `guarded` row's named test exists (that is the `provenance`/mutation
domain, and couples the gate to code paths — the τ=1 over-reach lesson); if a cheap, non-brittle check of
"the test file named in a row exists" proves clean, it may be added, but it is not required and is an
`Ask:` if it turns out to need judgement.

## 2. The oracle (seeded drift — the T26b/T38 pattern)
The gate is proved by **drift it must reject**, in `tests/tooling/test_conventions_gate.py`:
- a **duplicate id** (reproduce T41b's defect on a register copy) → rejected;
- a **dangling citation** (prose cites `C99`, no such row) → rejected;
- a **gap** in the live sequence (delete C11 from a copy) → rejected;
- the **real register passes** (the gate is not vacuously red).
Mutate the gate itself (flip the uniqueness check to accept duplicates) → the duplicate-id test fails.
This is the "a gate that can't fail is empty" discipline (§9) applied to the new gate: it must be shown
to reject the exact drift that earned it.

## 3. Process
Branch `tranche/41c-conventions-gate`. `test` (RED: the seeded-drift tests, failing on a `verify.py` with
no `conventions` subcommand) → `feat` (the subcommand + wire into `ci.yml` + add to the §9 gate list) →
`docs` → `chore(release): v0.48.2`. The `gates` gate (T38) must stay green — `conventions` is now in both
the charter list and `ci.yml`, so it covers itself. `source:` none; `crosscheck: none — a text-integrity
check, no numerics`. **Raise an `Ask:`** only if the register's prose citations are not cleanly parseable
(e.g. a `C<n>` appears in narrative in a way that is not a citation) — then rule the citation syntax
rather than guess.

## 4. What this is NOT (scope fence)
No numerics, no `src/` change. Not a check that each row's *claim* is true (that a `guarded` row really is
guarded — that is the §9 register-pass's judgement, not statically decidable). Not a general
Markdown-table linter. Not reaching into `src/` beyond, at most, a cheap "named test file exists" check.
T41c is the one integrity gate the register needs: unique ids, resolving citations, no gaps — the class of
silent corruption two human passes missed.

---

## 5. Cowork's ruling on T41c — PR #57 (§10 step 4, 2026-11-14)

Accepted — v0.48.2, 1629 green, **eleven** gates. `verify.py conventions` is in (unique ids, gap-free
ascending live run, citation resolution), listed in §9 and invoked in `ci.yml` so the T38 gates-gate
covers it; seeded-drift tests per shape, seven mutants of the gate killed. The class of silent register
corruption that two human passes missed is now mechanical.

**The spec's floor error — owned; Code's fix is right.** My spec placed the live floor at "the stated
floor… C10", but C10 is where *new* rows go — C7–C9 are live. Code **reads the floor out of the table**
rather than pinning it, so the gate protects all live rows; hard-coding 10 would have made the gate blind
to exactly the first three rows it exists to guard. My imprecision, caught at build.

**Two findings in Code's own gate — ratified, and auditing one's own gate this hard is the point.** (a) A
live row reusing an *archived* id had no test, and since renumbering to `C1` also trips the ascending/gap
checks, the test asserts the **specific message** rather than counting problems (a gate that fails for the
wrong reason is a false green in disguise — the right instinct). (b) The row-header stripping in
`register_citations` was **dead code** (`known` is built from those same headers, so a row id resolves to
itself by construction and could never dangle) — deleted and renamed `referenced_convention_ids`, which
is marginally stronger (a malformed header `| C99X |` now surfaces rather than silently parsing as a row).
Both are the §4 "an oracle must contain the failure in its domain" turned on the gate itself.

**The register layout fix — ratified, and partly my doing.** The live table was split in two (a stray
blank line + the C10 addendum *inside* the table body), rendering later rows headerless, and C13 sat
between C10 and C11 from T41b's in-place renumber (mine) — so the gate's first real job was to catch a
register I had helped break. Now C7–C13, one table, addendum below, and the ascending check holds it
there. The gate earned its place on its own artifact immediately.

**The PATTERN-NOT-FOUND discipline — noted, recurring, good.** A mutant that reported not-found (guessed
indentation) was re-run against the real line, not booked as killed — *a mutant that does not apply is
not a mutant that died* (the same care as T39/T41). This is the mutation-sweep's own integrity, held.

**No charter edit, no Ask.** §9 already carries the gate list (now with `conventions`) and the
statically-decidable-gating boundary; T41c is that applied. The multi-name infrastructure (T41b entity
keying, T41c register integrity) is complete; the products come next.

**Next:** T42 (the first-to-default basket) — the first product priced through the copula — mine to
spec. **Mark #57 ready (step 6) → Bernardo merges.**
