# Closure Verification — response to the three foundation audits

**Date:** 2026-07-19 · **Verifier:** Cowork (design authority) · **Subject:** `closed_AUDIT.md` ·
`closed_PONYTAIL_AUDIT.md` · `closed_PONYTAIL-DEBT.md`
**Tree state:** branch `fix/foundation-audit-closure`, `__version__ = 0.82.0`, 33 commits ahead of
`main`, spanning v0.75.0–v0.82.0. 147 test functions. Not yet pushed.

## What this document is

A **control document**, not a summary. Its purpose is to make the closure *independently
re-checkable* — by a reviewer who does not trust the closure report, the builder, or this file. Every
claim below carries the handle needed to falsify it.

It exists because a closed audit is a claim about the past that gets weaker with time. Six months
from now nobody will remember which of the 48 findings were fixed, which were deferred on purpose,
and which were quietly dropped. That ambiguity is the failure mode this guards against.

**Scope limit, stated plainly.** This pass verified *structural* claims: file existence, tracking
status, ledger completeness, test-name presence, disposition coverage, and internal consistency
between the reports, `OPEN.md`, and `.gitignore`. It did **not** re-run the suite, re-derive the
numerical counterexamples, or independently re-audit the source. Those are separate acts and are
specified in Part 5.

---

## Verdict

**The closure is sound and the disposition discipline held.** All 48 findings across the three
reports are accounted for. No finding is closed by silence. The one item that mattered most —
ICMA Rule 251.2, where a wrong implementation returns a plausible number forever — was pinned to a
published ISDA example rather than to reprice-to-par, which was available and would have looked fine.

**Three verification findings follow (V1–V3), one of them material.** None reopens the audit. V1 is
an active instruction to bypass a control and should be fixed before this branch is pushed.

---

# Part 1 — Disposition census

Verified against `closed_AUDIT.md` §"Closure disposition", `OPEN.md` §"Foundation audit closure",
and the repo.

| disposition | count | meaning |
|---|---|---|
| **Fixed with a test** | 24 | Tier 1 (5) · Tier 2 (4) · Tier 3 (7, incl. 3.7 sub-items) · PONYTAIL 1–5, 8 |
| **Fixed, named sub-part deferred** | 3 | AC-2.2b · AC-2.4b · AC-3.6b |
| **Fixed on reclassification** | 2 | AC-T4.2 (invalid input accepted) · EURIBOR_6M (v0.82.0) |
| **Ledgered with a re-open trigger** | 19 | 17 remaining Tier-4 + 3 deferred sub-parts + AC-PD.1 |
| **Rejected with recorded reason** | 2 | PONYTAIL 6 (`exponential_growth`) · 7 (`_rule_set`) |

**Coverage check: complete.** Every finding in all three reports resolves to exactly one of these
five states. No finding appears in two states; none appears in none.

**The reclassification held under pressure.** When instructed to eliminate deferrals, the correct
answer was *not* to build them — zero-consumer machinery is precisely what Phase 4 had just deleted
(`Underlying` siblings, `NumericalConfig`, three `PricingResult` fields). The ledger was re-tested
against a single question — *does this produce a wrong answer or accept invalid input today?* — and
exactly one item (AC-T4.2, non-positive `Tenor`) answered yes. It was promoted to a fix. That is the
right outcome and the right reasoning: **no latent wrongness in the ledger, deferred capability
freely.**

---

# Part 2 — Verification handles

How to falsify each class of claim without trusting this document.

**Fixed-with-a-test (24).** Every Tier 1–3 fix maps to a named test in `tests_ng/L0/
test_audit_closure.py`, numbered to the finding. Confirmed present:
`test_1_1_afb_leap_to_leap_is_exactly_whole_years` · `test_1_2_log_linear_continue_slope_stays_positive` ·
`test_1_3_business_days_between_is_start_inclusive_end_exclusive` ·
`test_1_4b_backward_eom_keys_on_the_maturity_anchor` ·
`test_1_5_furikae_golden_week_substitute_deterministic` ·
`test_2_2_sifma_good_friday_is_a_holiday` · `test_2_2_sifma_saturday_new_year_does_not_shift_to_friday` ·
`test_3_5_act_act_icma_long_first_coupon` · `test_3_6_fx_spot_date` ·
`test_t4_2_non_positive_tenor_is_rejected`.
*To falsify:* delete a fix, re-run — the correspondingly numbered test must fail. A fix with no
failing counterpart is not fixed.

**The single highest-risk fix.** ICMA Rule 251.2 (finding 3.5) is anchored to ISDA 2006 §4.16, the
long-first-coupon example 15 Aug 2002 → 15 Jul 2003 semi-annual, documented value
153/(2·184) + 181/(2·181) = **0.9157608695652174**, matched to 1e-12. Arithmetic independently
confirmed here: 153/368 = 0.4157608696, 181/362 = 0.5 exactly.
*Why it matters:* a wrong quasi-period split is self-consistent — it reprices to par and stays wrong
forever. This is the one finding where the available cheap oracle would have certified a bug.

**Ledgered (19).** All carry ids (`AC-T4.*`, `AC-2.2b`, `AC-2.4b`, `AC-3.6b`, `AC-PD.1`), a module,
and a re-open trigger, in `OPEN.md`.
*To falsify:* pick five at random and check the trigger names something that will actually occur.

**Rejected (2).** Both carry reasons in the disposition block. A rejection is a ruling; a ruling
without recorded reasoning is indistinguishable from an oversight.

**Test-quality context.** The suite is ~92% oracle / ~8% self-consistency by the builder's own
skeptical re-count, after an automated sweep returning 97.4% was **rejected** — the classifier keyed
on a test's apparent subject rather than on whether its expected value was derivable without running
the code, which is the identical failure mode as the three bug-ratifying tests that triggered the
sweep. Five self-consistency tests were then deleted rather than reclassified, including one that
duplicated the `verify.py fields` gate. *This matters more upward than here:* L0 is where external
oracles are cheapest, so 92% is a floor, not an achievement. At L3/L4 the closed forms run out.

---

# Part 3 — Findings from this verification pass

### V1 — `OPEN.md` instructs readers to bypass the control it is part of · **MATERIAL**

`OPEN.md` line ~10 states: *"This file is gitignored (`/*.md` rule). Edit freely; no commits
needed."*

**This is false.** `.gitignore:58` carries an explicit `!/OPEN.md` exception and `git ls-files`
returns it — the file is tracked, as `CLAUDE.md §7c` requires so CI has `verify.py`'s inputs.

**Why it is material:** the entire 21-item deferral ledger lives in this file. A reader following the
instruction edits the ledger locally and never commits, so deferrals silently diverge between working
copies and CI sees a stale ledger. The instruction is a leftover from the quarry era, where the file
genuinely was untracked — the exception was added later and the self-description was never updated.
It is the one thing found here that actively undermines the closure.

**Fix:** delete the sentence; replace with a statement that the file is tracked and every edit lands
in a commit.

### V2 — the ng ledger lives inside a stale quarry document

`OPEN.md`'s header reads *"Last reviewed: 2026-06-19 (post-v1.119 audit + warnings closure)"* and
*"8268/8268 in L≤3, 12,797/12,797 full suite"*. Those are **quarry** figures at v1.119; the new tree
is at v0.82.0 with 147 tests. The ng audit-closure ledger was appended as a section beneath a header
describing a different codebase, so a reader cannot tell which library the file is about.

**Fix:** split the ng ledger to its own tracked file, or re-head the document to state that it spans
both trees and mark each section's subject.

### V3 — the 21 ledgered items are covered by no automated check

They are deliberately excluded from the `verify.py debt` balance, and that call is **correct**: §5's
CI invariant is `suppressions − ledger entries = 0`, and a deferred capability suppresses nothing, so
including them would corrupt the balance.

The consequence must be stated anyway: **nothing automated will ever notice these items again.**
`verify.py`'s seven gates (`acyclic`, `tests`, `fields`, `debt`, `provenance`, `version`, `layers`)
are all silent on them. They survive on human review alone.

Compounding this, six triggers are **condition-driven, not topic-driven** — AC-T4.7/8/9 (fire only if
a future calendar or `NEAREST` consumer introduces the pattern), AC-T4.11 (error-message quality),
AC-T4.17 (a real-world weekend-rule change), AC-T4.18 (a fourth consumer, rule of three). The builder
flagged these honestly rather than dressing them as roadmap items, which is the right call. But it
means **no amount of roadmap progress will fire them.** They need a calendar-based sweep, not a
trigger.

**Fix:** add the ledger to the checkpoint's standing review inputs — at every checkpoint, re-read the
21 items and ask of each whether its trigger has fired. That converts a passive list into a recurring
obligation, which is what `CLAUDE.md §5`'s "or it does not exist" actually demands.

---

# Part 4 — Standing safeguards

What must remain true for this closure to keep meaning something.

1. **`closed_` is earned, and reversible.** The prefix means *every* finding is fixed-with-a-test,
   ledgered-with-a-trigger, or rejected-with-reason. If a disposition is later found to be false, the
   file is renamed **back**. The prefix must be able to be wrong, or it is decoration.
2. **A deferral without a live trigger is a forgotten item with paperwork.** Reviewed at each
   checkpoint (V3).
3. **Nothing re-enters the ledger that produces a wrong answer today.** The Phase-5 reclassification
   question is the standing test: *wrong answer or invalid input accepted?* Yes ⇒ it is a bug, and
   bugs are not deferrable. No ⇒ absent capability, defer freely.
4. **Deleted speculative types stay deleted until a real consumer arrives.** The `Underlying`
   siblings, `NumericalConfig` and the three `PricingResult` fields return **with** the layer that
   uses them, shaped by it. Re-adding them ahead of that consumer re-runs the exact error Phase 4
   corrected.
5. **Oracle standard, forward-committed:** the expected value must be writable from an external
   source *before the code exists*, at every layer. Reprice-to-par does not count — it is
   self-consistent by construction (doc 19 §6).
6. **The three deferred sub-parts are the likeliest to be mistaken for complete.** AC-2.2b, AC-2.4b
   and AC-3.6b sit under findings marked FIXED. A future reader scanning dispositions sees "2.2
   FIXED" and may not read to the deferral. Their ids exist for exactly this reason — cite the id,
   never the parent finding.

---

# Part 5 — Re-verification runbook

For an auditor re-checking this closure cold.

```
1. Structural (minutes)
   git ls-files redesign/independent_audits/     -> three closed_*.md
   git ls-files OPEN.md CHANGELOG.md             -> both tracked
   grep -c "AC-" OPEN.md                         -> 21 ledgered ids
   Each closed_*.md has a disposition block; every finding appears exactly once.

2. Executable (one run)
   verify.py all        -> seven gates green
   pytest tests_ng/     -> 147 green
   Then: revert any one fix and confirm its numbered test in
   test_audit_closure.py fails. A fix with no failing counterpart is unproven.

3. Numerical spot-check (the ones that hide)
   ICMA 251.2   vs ISDA 2006 §4.16     -> 0.9157608695652174
   AFB          2004-02-29→2008-02-29  -> exactly 4.0
   LOG_LINEAR   CONTINUE_SLOPE t=30    -> +0.209, never negative
   BUS/252      one primitive, [start, end), agrees with CDI accrual

4. Ledger integrity (the part that decays)
   Five ledgered ids at random: does the trigger name something that
   will actually occur? Six are condition-driven by design (V3) — confirm
   they are still flagged as such and not quietly reclassified as scheduled.
```

---

## Bottom line

Forty-eight findings, all accounted for: 26 fixed with tests, 19 deferred with triggers, 2 rejected
with reasons, 3 partially deferred under named ids. The discipline that matters most held in the two
places it was tested — a published oracle was used where a self-consistent one would have passed, and
a flattering automated measurement was rejected by the person it flattered.

**One thing to fix before pushing:** V1. A ledger carrying an instruction not to commit it is a
control with its own bypass written into the first paragraph.
