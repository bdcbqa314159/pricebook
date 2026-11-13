# Tranche 37 — accrual-oracle hardening (re-audit #3's F-3.1 + the τ=1 proactive sweep)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.44.1` (a test-only patch
bump — no `src/` change). **Source:** none external — a re-audit finding, like T23 (which fixed
F-1.2). **Pre-registered RED by construction** (the fix *is* a failing-then-passing oracle). Driven by
`docs/audit/reaudit_03_findings.md` §(1).

## 0. Why this is a tranche
Re-audit #3's full-tree mutation sweep found **F-3.1**: dropping the accrual `ψ` in the nominal
bootstrap (`market/nominal_bootstrap.py`, `annuity = Σ ψ·discount[pay]`) **survives all 1430 tests**,
because `tests/L1/test_nominal_bootstrap.py` uses annual ACT/365F dates so every `ψ = 1.0` exactly and
`ψ·df ≡ df`. The `src/` code is **correct**; the oracle is blind. This is **F-1.2's exact shape** — the
`τ=1` degeneracy, the charter's first-recorded §4 fixture trap — in a second instrument, and the second
time it is pre-Jäckel and surfaced only by a re-audit. T23 fixed F-1.2 the same way; T37 does the same
for F-3.1, **and** acts on the systemic signal: `τ=1` has now bitten twice, so it is swept proactively.

## 1. What T37 does (test-only; no `src/` change)
1. **Fix F-3.1.** Add a **semi-annual** schedule fixture to `tests/L1/test_nominal_bootstrap.py` (ACT/365F,
   so `ψ ≈ 0.496/0.504 ≠ 1`), and assert the **non-degeneracy premise** `ψ ≠ 1` on it, so the
   dropped-accrual mutant dies. The reference discount factors/zero rates are computed independently
   (the test's own summation, per §4 independence), not read back from the bootstrap.
2. **Proactive τ=1 sweep (the systemic half — "contain the failure in its domain", §4).** Audit **every
   accrual-bearing fixture in the tree** — anywhere a year-fraction/accrual `τ` or `ψ` multiplies a term
   (the inflation legs, the caps/floors, the swap annuities, the YoY leg, the XVA exposure weights) —
   and wherever the accrual is load-bearing **and** the fixture is annual-or-unit, add a non-unit-accrual
   case and assert the `τ ≠ 1` premise. The deliverable is: a list of every accrual-bearing oracle, each
   marked already-non-degenerate or newly-hardened. Two instruments carried this hole undetected; the
   point is to find a third now rather than at re-audit #4.
3. **A second independent killer for the Sobol Gray code (T20b).** `G(n) = n ⊕ ⌊n/2⌋` is killed by
   exactly one test and is load-bearing for the whole low-discrepancy construction — the thinnest
   single-point-of-failure oracle in the tree (re-audit #3 §(1)). Add a second, independent killer
   (a different property the Gray code must satisfy — e.g. successive-value single-bit-change, or the
   reflected-binary count — distinct code from the existing killer), so a later deletion of one test
   does not silently reopen it. *(T26's Leg B numéraire divisor stays single-by-design per re-audit #2;
   not touched here.)*

## 2. The oracle
Each addition is itself the oracle: the new semi-annual bootstrap case **fails** on the ψ-dropped
mutant and passes on the real code (re-run the F-3.1 mutation to confirm it now dies — the mutation is
the proof of power, §4); each newly-hardened accrual fixture likewise; the second Gray-code killer must
die on a Gray-code mutation and pass on the real one. **Confirm by mutation**, not by reading — the
whole finding is that reading missed it twice. No `src/` touched, so the existing 1430 stay green and
the new cases add to the floor (`verify.py tests`).

## 3. Process
Branch `tranche/37-accrual-oracle-hardening`. `test` (the semi-annual bootstrap case + the proactive
accrual-fixture hardening + the second Gray-code killer, each pre-registered against its mutation) →
*(no `feat` — test-only)* → `docs` → `chore(release): v0.44.1`. `source:` none; this is a re-audit
remediation. **Re-run the F-3.1 and Gray-code mutations after the adds to confirm they now die.** CI
floor rises by the number of cases added. **Raise an `Ask:`** only if the proactive sweep turns up a
*second* real survivor (an accrual-bearing oracle blind the same way) — that is a finding to rule, not
a mechanical add.

## 4. What this is NOT
No `src/` change (the code is correct; this hardens oracles). No new capability. Not a re-opening of
T26's Leg B single-killer (ruled single-by-design). Not a general mutation re-sweep (that is re-audit
#4's job on its own cadence) — T37 acts only on re-audit #3's recorded findings.

---

## 5. Cowork's ruling on T37 — (reserved for §10 step 4, after the build)
