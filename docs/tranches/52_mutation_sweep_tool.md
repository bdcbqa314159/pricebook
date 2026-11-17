# Tranche 52 — the mutation-sweep as a committed instrument (tooling)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.56.0`. **Source:** Code's 0285
scope question, ruled 2026-11-17. The mutation sweep is a **standing §9 obligation** (every arc close + every
11 tranches), yet it has been an ad-hoc scratchpad rewritten from scratch each re-audit — **the one part of
the method with no provenance**, and the one that has re-introduced the same tooling bugs twice.

## 0. What this is — and the boundary that matters
A committed mutation-sweep **engine** in `tests/tooling/`, versioned and reviewed, that encodes the
instrument lessons the charter has paid for as **code** rather than re-deriving them each time. **It is the
engine, not the audit.** The ruling that matters: **which mutations to apply stays Cowork's per-re-audit
judgement** — where a transcription error would land, the blind-spot probes (T43 rescale, T39 reduction),
the oldest-first scoping. The tool mechanizes *applying and scoring* a mutation, never *choosing* it. A tool
that chose its own mutations would be the can't-fail gate §9 forbids (it would sweep where it is easy, not
where the failure lives). So this is an instrument the re-audit's judgement drives, and the spec fences that
explicitly.

This is **not speculative** (§6): five re-audits are well past rule-of-two, and the consumer is a standing
obligation, not a hypothetical future.

## 1. What it must encode (the instrument clauses, as code)
Each of these is a charter lesson the sweep must not be able to get wrong again:
- **Bytecode-cache invalidation per mutant** (§9, T46) — the mutated source *is* the code that runs; a
  false *"killed"* from a stale `.pyc` is structurally impossible in the committed tool.
- **applied / killed / survived reporting** (§9, re-audit #4) — a `0` survivors is meaningless without the
  applied count; the tool reports all three and refuses to headline a survivor count alone.
- **Positional mutation on textually-identical lines** (the decision this tranche carries, bitten **twice**
  in re-audit #5 — `yoy.py:136/:152`, `credit_legs.py:140/:250`): a string-replacement sweep refuses
  identical load-bearing lines as non-unique and silently leaves two real sites uncounted. The tool mutates
  **by position**, so identical lines in different functions are each a real, counted site. (The
  applied-count clause catches the miss today; the tool removes the failure mode.)
- **PATTERN-NOT-FOUND is surfaced, never counted as a kill** (§9) — a mutation that did not apply is
  reported as *not applied*, loud.

## 2. The oracle (RULED) — the tool is tested like any instrument
A sweep engine that mis-scores is worse than none (it launders false confidence), so it carries its own
tests in `tests/tooling/`:
- **A known-hole fixture: the tool must report a planted survivor as survived**, and a planted killable
  mutation as killed — against a tiny fixture module with a deliberately weak test, so the tool's verdict is
  checked against a known-answer, not trusted.
- **The bytecode guard is proven**: a mutation written within one mtime tick is *still* detected as applied
  (the T46 defect, now a regression test on the tool itself — the instrument carries the proof of the bug
  it was built to prevent).
- **Identical-line positional mutation is proven**: two textually-identical lines in different functions are
  reported as **two** applied sites, both scored (the re-audit #5 defect, as a regression test).
- **The floor gate already discovers `tests/tooling/`** (T26b), so the tool's own tests are floored.

## 3. Process
Branch `tranche/52-mutation-sweep-tool`. `test` (RED: the tool's own known-hole / bytecode / identical-line
tests) → `feat` (the committed engine in `tests/tooling/`) → `docs` → `chore(release): v0.56.0`. **The §9
re-audit bullet gains one line, landing WITH this tranche** (not before — a charter must not state a
practice that does not yet exist, §9's own rule): *the mutation sweep is now a committed instrument in
`tests/tooling/`, run by the re-audit's judgement, with the instrument clauses (bytecode, applied-count,
positional, PATTERN-NOT-FOUND) encoded as code rather than re-derived.* Draft PR; ready only at step 6.

## 4. What this is NOT
**Not a gate** — it does not run on every push, and it does not choose its own mutations; it is run by
Cowork at a re-audit with a chosen mutation set (§9's "a static gate for a dynamic property is a can't-fail
gate" — the sweep's *value* is the judgement of what to mutate, which is not mechanizable). Not a mutation
*library* enumerating all of `src/` (re-audit #5's item-5 honesty: a sample where transcription errors land,
not an exhaustive enumeration — the tool applies a given set, it does not define the universe). Not a change
to any existing oracle. On merge, the sweep has provenance and the twice-bitten tooling bugs cannot recur.

---

## 5. Cowork's ruling on T52 — (reserved for §10 step 4, after the build)
