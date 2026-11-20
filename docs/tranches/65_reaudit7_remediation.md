# T65 — re-audit #7 remediation (one test/tooling-only tranche → v0.67.0)

**Cowork spec, 2026-11-20.** Re-audit #7 (`docs/audit/reaudit_07_findings.md`) swept **77 applied / 76
killed / 1 real hole**. One blind oracle (F-7.1), one duplication (F-7.2), one tooling gap in the sweep
instrument itself (F-7.3), and one honest non-probe note (F-7.4, no action). Per the reaudit/05–06 precedent
this is **one test/tooling-only tranche** — no `src/` behaviour changes, every fix under one `CHANGELOG`
entry, one `__version__` bump, one floor bump. Two charter edits ride with it (drafted below); they land in
**this** merge so neither describes a practice that does not yet exist (§9's own anti-pattern).

## What the tranche carries

### 1. F-7.1 — the blind oracle: a reference sharing a coincidental literal the fixture makes correct
`reference_level` in `market/fixing_series.py` hard-codes `days_in_month = 30.0` and **survives all 2625
tests**. The source is right, so it is a blind oracle. The doubled cause is the lesson: the only pointwise
linear value test reads **2026-04-15 — April has exactly 30 days** — *and that test's own reference writes
`/ 30` too*. So §4's independence rule is satisfied **formally** (the test computes its own reference) while
the reference is independent of the code but **not of the coincidence**. Magnitude: −6.67% on the February
weight, ~0.03% on the index level.
- **Fix (one test):** a pointwise linear value test in a **non-30-day month** (February = 28, or a 31-day
  month) whose reference **derives** the month length (`calendar.monthrange`, or the days-between-month-ends)
  **rather than writing a literal `30`**. This breaks both coincidences at once — the fixture exercises a
  real `days_in_month ≠ 30`, and the reference no longer shares the magic constant. Confirm the hard-`30`
  mutant dies (0 → ≥1 killer); prefer the mutation to a reading (§4).

### 2. F-7.2 — four byte-identical ascending-time-grid guards (duplication, no hole)
Four **byte-identical** strictly-ascending time-grid guards (two sharing three consecutive lines), predating
the commodity arc — six prior pass-(2) runs did not name it. §6 rule-of-two has long fired on a shared
**shape** (not merely the word "ascending").
- **Fix:** extract the one byte-identical guard to a single home — **guard-only, not a `TimeGrid` value
  object** (the T62 precedent: nothing compels a bundle, no class is near the field limit, and the shared
  thing is one piece of *knowledge* — "a time grid is strictly ascending"). Code rules the home (foundation
  is the natural layer — it is finance-free). **Exclude the separate group of four *domain-specific*
  ascending guards** Code identified: merging those would be the shared-word-not-shared-shape error (their
  surrounding conditions differ). Pure refactor — prove it by the existing oracles staying green (and
  bit-identity in-process if op order is preserved, the T49 discipline).

### 3. F-7.3 — the single-killer count belongs to the instrument, not a hand-rolled loop
§9 makes the single-killer re-run standing, but `sweep()` returns only `applied/killed/survived`, so every
re-audit since #2 has hand-rolled a counting loop **beside** the committed instrument — inheriting none of
its verified-baseline or `try/finally` guards. This run paid for it: a command timeout killed the counter
**between mutating `schwartz.py` and restoring it**, and the next run measured against a corrupted baseline
(caught, restored, 2625 re-confirmed — nothing shipped, but it is exactly the T52 failure arriving through
the one obligation the instrument does not cover).
- **Fix:** extend `tests/tooling/mutation_sweep.py` so `SweepResult` carries the **per-mutant failing-test
  set**, and the single-killer count is read from it — so the count inherits the instrument's verified
  baseline and `try/finally` restoration. Known-answer test for the new field (the instrument's own
  test-suite pattern, T52). This is the fourth hand-roll of the baseline/restore; committing it is the T52
  argument applied to the obligation T52 left out.

### 4. F-7.4 — no action (recorded)
One item-4 probe mutated **prose inside an f-string** (`"strictly ascending"` → `"ascending"`) and survived
correctly — the condition, the raise and the matched substring are all unchanged. Not a hole: the corrected
probes (condition loosened, guard bypassed) both die, so the guard is pinned. Recorded so the applied-count
is honest (a mis-aimed probe is not a kill and not a survivor-of-interest). No code.

## Charter edits (Cowork's; land with this tranche)

**§4 — the degenerate-fixture catalogue gains its eleventh shape** (append to that bullet, past tense,
tied to F-7.1): *a reference computed independently can still be blind if it shares a **coincidental literal
constant** with the code that the chosen fixture happens to make correct — §4 independence is then satisfied
formally while the reference is not independent of the coincidence. `reference_level`'s hard-coded
`days_in_month = 30` survived all 2625 tests because the one pointwise fixture read a 30-day month (April)
and the test's own reference wrote `/ 30` too. The premise: a reference **derives** a magic constant the
code should compute (the month length), it does not write the same literal; and the fixture exercises a
value where that constant is not its coincidental default. Eleventh `τ=1`-family instance, first where the
degeneracy is a **shared literal between fixture and reference** rather than a property of the fixture
alone. (F-7.1, re-audit #7.)*

**§9 — the committed-instrument clause notes the single-killer count is the instrument's** (append):
*the sweep also returns the per-mutant failing-test set, so the standing single-killer re-run reads its
counts from the instrument — inheriting the verified baseline and `try/finally` restoration — rather than
from a loop hand-rolled beside it, which for five re-audits inherited none of the guards and once measured
against a baseline a timeout had corrupted mid-mutation. (F-7.3, T65.)*

## Gates / provenance
- RED-first for the F-7.1 test and the F-7.3 known-answer test; the F-7.2 extraction proved by green + purity
  (§4/T49).
- Fill the `docs/audit_log.md` single-killer table's T60–T64 row **from the sweep's measured counts** (Code
  already did at #7; confirm it carries over).
- Version → **v0.67.0**; one floor bump; one `CHANGELOG` entry naming re-audit #7.

## Out of scope (carried)
- **C2 (vol lives on one clock)** — still the widest OWED; no vol touch.
- **`dividends_before(T)` consolidation** — still owed; no equity touch.
- **OI-7** (`MarketSnapshot` fields-exempt) — stays open, its trigger unchanged (the next keying dimension).
- **Method-watch (not this tranche):** the duplication pass missed four byte-identical guards for six runs;
  a mechanical byte-identical-block scan could aid it. Noted for the re-audit, not gated (identical lines
  have legitimate instances; it is an aid, not a can't-fail gate).
