# Tranche 23 — the linker coupon accrual oracle (re-audit #1 F-1.2): a degenerate fixture closed

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.28.0`.
**Source:** `BERHAN-2012` ch 1 (the inflation-linked bond), `worked-from` — the same source the
instrument was built from (T3); no new reference. **Pre-registered RED (by mutation), as always.**
First of two re-audit-remediation tranches (T23 → T24); neither opens a new source arc.

## 0. The defect, and why it is a tranche not a patch
Re-audit #1 (Target 1 cont., `docs/audit/reaudit_01_findings.md` F-1.2) found that dropping the `ψ`
accrual factor from the inflation-linked-bond coupon sum at `engine/inflation.py:135` **survives all
818 tests**. The production code is **correct** — `ψ` is present and computed — but every linker
coupon oracle runs on a fixture of exactly-365-day ACT/365F periods, so every `ψᵢ = 1.0` and `C·ψ = C`
identically. The test even named `test_the_linker_matches_the_closed_form_with_accrual_fractions`
computes `ψ = days/365 = 1.0` on those dates: the name promises non-unit accruals the fixture never
delivers. This is the charter's own degenerate-fixture trap (§4, `τ=1` hiding the accrual), on the
oldest instrument in the tree, caught only because the re-audit's mutation sweep was extended to the
inflation arc (T2–T14). Measured blast radius: a dropped `ψ` overstates a semi-annual 3% linker's PV
by **~887 bp of notional (~8.4%)**. The *same* accrual-drop is caught everywhere else `ψ` lives (YoY
legs, cap/floor, bootstrap annuity, both calibration accruals) — the linker is the lone instrument
with a degenerate accrual fixture.

## 1. The change
**Test-only.** No `src` change is expected — the code is right. The tranche adds the non-degenerate
oracle the instrument never had. If the non-degenerate fixture surfaces a real `ψ` bug (the audit says
it will not), that is a **finding to raise as an `Ask:`, not a silent fix** (§10).

## 2. The oracle (RULED)
- **Pointwise, on a non-unit-`ψ` fixture.** A linker whose coupon periods are **not** exactly 365 days
  — a semi-annual schedule on ACT/365F (each period ≈ 0.5), or an ACT/ACT schedule — so `ψᵢ ≠ 1`.
  Assert the engine's PV against an **independently transcribed** closed form (the test computes its
  own `ψᵢ = day_count(start_i, end_i)` and the indexed cashflow sum — never reads a value back from
  the engine), matched to the usual tolerance. This pins the code's `ψ` handling *pointwise*, which the
  existing degenerate test cannot.
- **Non-degeneracy as a premise (§4).** Assert in the test that the fixture's accruals are not all 1 —
  e.g. `assert max(abs(ψᵢ − 1)) > <bound>` — so a later refactor that quietly flattens the fixture back
  to annual periods fails loudly rather than silently re-hiding the hole. This is the "assert the
  non-degeneracy as a premise" rule, the exact guard this tranche exists to install.
- **Mutation is the proof of power (§4, missed-pre-registration recovery).** Because the pre-register
  was missed at T3, the empirical proof RED-first would have given is supplied by mutation: the new
  oracle **must fail** a `ψ`-dropped mutant (`C·ψ → C`) and **pass** against production. Report both
  outcomes; a new oracle that does not kill the drop-`ψ` mutant has not closed the hole.
- Keep the existing annual-period test as a consistency guard; the new non-degenerate oracle is the
  correctness anchor (§4 — self-consistency telescopes, the non-degenerate structure does not).

## 3. Process
Branch `tranche/23-linker-accrual-oracle`. `test` (RED proof = the drop-`ψ` mutant dying under the new
oracle, pre-registered via mutation) → `feat`/`docs` as needed → `chore(release): v0.28.0`. `source:`
`BERHAN-2012`. No `OPEN.md` item opened or closed (the hole was latent in coverage, not shipped debt).
**Open as a DRAFT PR** (§10 step 3). Raise any real-`ψ` discrepancy as an `Ask:`, do not rule it.


---

# BUILT — results (appended at build; §0–3 above are Cowork's spec, unedited)

`v0.28.0`. 820 tests, all eight gates green, ruff and pyright clean. **Test-only: `src/` is
unchanged**, exactly as the audit predicted. The hole is closed.

## 4. The hole, reproduced before it was closed

I did not take F-1.2 on report. Dropping `ψ` from `engine/inflation.py`'s coupon sum, from a clean
bytecode state:

```
drop-psi mutant: SURVIVED  818 passed
```

So the finding is independently confirmed: the term is load-bearing in production and invisible to
every oracle.

## 5. What the existing oracle was doing, which is the part worth keeping

`test_the_linker_matches_the_closed_form_with_accrual_fractions` **computes a `ψ` and is still blind
to it.** It evaluates `psi = (pay − start).days / 365.0` over `DATES`, where every period is exactly
365 days — so every `ψ` it computes is `1.0`, and `C·ψ = C`. The name promises accrual fractions the
fixture never delivers.

That is a sharper version of the `τ = 1` trap than the charter's existing examples: the test is not
merely *silent* about the accrual, it visibly *handles* it, which is why twenty tranches of readers
(and two review passes) saw a covered term. A fixture can defeat an oracle that looks correct line by
line.

## 6. The fixture, and why the accruals are unequal

Semi-annual periods of **182 and 183 days** on ACT/365F: `ψ ≈ 0.4986` and `0.5014`. Two properties
were chosen deliberately, and the second is not in the spec:

* **not 1** — which is what makes the accrual visible at all;
* **not equal to each other** — which makes a mutant that replaces `ψ` by any single *constant* fail
  too. A fixture of equal non-unit accruals would have closed F-1.2 while leaving `ψ → 0.5` alive.

Mutation confirms both halves:

| mutant | before T23 | after T23 |
|---|---|---|
| `ψ` dropped (the F-1.2 hole) | **SURVIVED** (818 passed) | **KILLED** |
| `ψ → 1.0` | survived | **KILLED** |
| `ψ → 0.5` (a constant) | — | **KILLED** (needs the unequal accruals) |
| `ψ` applied to the redemption | — | **KILLED** |
| production | 818 pass | **820 pass** |

## 7. The non-degeneracy premise is itself a test

`test_the_SEMI_ANNUAL_fixtures_accruals_are_NOT_degenerate` asserts the fixture's own property: every
accrual is more than 0.4 away from 1, and they are not all equal. This is the guard the tranche exists
to install — **a later refactor cannot quietly flatten the schedule back to annual periods**, which is
precisely how the hole survived. Without it, the new oracle would keep passing while silently losing
all power over `ψ`, and the next re-audit would find the same finding again.

## 8. No `src` change, and no `Ask:`

The spec required that a real `ψ` discrepancy be raised rather than fixed silently. There is none: the
non-degenerate oracle passes against unmodified production on the first run. The production code was
right and only its coverage was wrong, which is the cleanest possible outcome for a latent-coverage
finding and worth stating plainly rather than leaving as an absence.

## 9. Findings for Cowork (§10 — raised, not decided)

1. **A candidate §4 sharpening, and I am ambivalent about it.** The existing degenerate-fixture bullet
   describes a fixture where the parameter *"sits where it cannot be seen"*. F-1.2 is worse in a
   specific way: the oracle **visibly computes** the quantity (`psi = days/365`) and is still blind,
   because the fixture makes the computation a no-op. So *"an oracle that handles a parameter is not
   thereby testing it — check the fixture makes the handling observable."* Against adding it: §4
   already has five instances of this trap and is long, and the operational rule (mutation, plus the
   non-degeneracy premise) is already there and is what caught this. My inclination is **no new
   clause** — the detection machinery worked; it was only ever applied to too small a domain. Yours to
   rule.
2. **The re-audit's own domain is the real lesson, and it is already recorded.** F-1.2 was found only
   because the sweep was extended past the first pass's scope — §4's own rule applied to the audit.
   Worth making the *default* for re-audit #2: sweep the whole tree, not the recent arc.

---

## 10. Cowork's ruling on T23 — PR #32 (§10 step 4, 2026-11-08)

Accepted — v0.28.0, test-only, `src` unchanged exactly as the audit predicted, 820 green, the hole
reproduced before it was closed (drop-`ψ` SURVIVED 818 from clean bytecode) and killed by the new
non-degenerate oracle. F-1.2 closed.

**The unequal-accrual refinement is the right instinct, and better than my spec.** I asked for
non-unit `ψ`; you also made them **unequal** (182/183 days → 0.4986/0.5014), which kills the `ψ → 0.5`
constant-mutant a merely-non-unit fixture would have left alive. That is the charter's existing rule
applied correctly — "the premise to assert is not 'the values differ' but 'the block the code reads
differs from the block a mutation would read'" (§4, the T17 uniformity-across-a-slice appearance): a
constant-`ψ` mutant reads a *uniform* block, so the fixture must be non-uniform, not merely non-unit.
No charter change — it is an instance of a rule already there, and the `test_..._NOT_degenerate`
premise test is exactly the guard §4 prescribes.

**Your two §9 findings:**

1. **No new §4 clause — agreed, and the restraint is right (your second arc running).** The
   "visibly computes `ψ` and is still blind" sharpness is real, but it is a point about *review
   psychology* — why humans and the two review passes saw a covered term — not a new *detection* rule.
   The detection machinery (mutation + assert the non-degeneracy as a premise) already existed and is
   what caught this; and the fixture is, structurally, an instance of §4's existing "a varying fixture
   can still be degenerate if it is invariant under the transformation the code performs" — `C·ψ = C`
   when every `ψ = 1`, degenerate under multiply-by-`ψ`. Sixth instance of the trap, not a sixth kind.
   §4 is long and earned its keep by catching this; it does not grow. (If the "visibly handles it"
   line ever recurs as a *distinct* failure the premise test does not guard, re-open it — but once is
   an anecdote.)
2. **Whole-tree as the default sweep scope — ruled, and it lands in §9, not §4.** This is the real
   lesson, and it is the signature defect committed against the audit itself: the text already said
   "full-tree mutation sweep", the first run scoped to the recent arc anyway, and the one hole lived
   in the oldest tranche. §9 gains a precedent line (F-1.2 as the cite) that full-tree means the
   oldest tranches, not the arc just built — a precedent, not a new rule, since the rule was already
   right and only under-applied. Edited.

**Next: T24** (the register rebuild) per `docs/tranches/24_reaudit_register_rebuild.md`, on the normal
rhythm. Entry #1 completes when T24 lands.
