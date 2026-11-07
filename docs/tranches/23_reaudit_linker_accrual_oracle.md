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
