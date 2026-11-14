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
1. **Fix F-3.1 (DONE).** A **semi-annual** fixture in `tests/L1/test_nominal_bootstrap.py` (ACT/365F,
   `ψ ≈ 0.4986/0.5014 ≠ 1`) with the `ψ ≠ 1` premise asserted, plus a known-curve round trip and a
   par-reprice companion. Verified by re-running the mutation — the ψ-drop now dies to 2. Suite 1433.
1b. **Fix F-3.2 (RULED — the sweep's second survivor, same class).** `models/implied_vol_surface.py`
   drops `ψ` from `discounted_unit = notional·ψ·df` and **survived all 1433 tests** —
   `tests/L3/test_implied_vol_surface.py` uses annual dates, every `ψ = 1`. Measured severity: a 9.00%
   vol reads as **5.80%** on a semi-annual period (the vol is inverted from `floorlet/discounted_unit`,
   so dropping `ψ≈0.5` inflates the denominator ~2× and understates every surface node ~36%) — and the
   surface is what T10+ calibrate against. Same remedy: a semi-annual fixture + `ψ ≠ 1` premise in
   `test_implied_vol_surface.py`, confirmed by the mutation dying. `src/` is correct; the oracle was blind.
2. **Proactive τ=1 sweep — EXHAUSTIVE, and fix every same-class survivor in-tranche.** Audit **every
   accrual-bearing fixture in the tree** (the inflation legs, caps/floors, swap annuities, the YoY leg,
   the XVA exposure weights, and anything else where a year-fraction/accrual multiplies a term); wherever
   the accrual is load-bearing **and** the fixture is annual-or-unit, add a non-unit case and assert
   `τ ≠ 1`, **confirmed by mutation**. **Scope ruling (§10):** τ=1 is now a *known class with a known
   remedy* (three confirmed: F-1.2, F-3.1, F-3.2) — so **fix every τ=1-class survivor directly in T37**;
   do **not** raise each one as a separate finding. Raise an `Ask:` **only** for a survivor of a
   *different* class (a non-accrual degeneracy), which is a genuine new finding to rule. Deliverable: the
   full list of accrual-bearing oracles, each marked already-non-degenerate or newly-hardened.
3. **The mechanical guard — this is the systemic half, and it is why τ=1 stops here (ASSESS + build if
   clean).** τ=1 has now bitten **three** pre-Jäckel instruments from one shared fixture convention
   (annual dates on ACT/365F → every accrual exactly 1.0). Manual review missed all three; a proactive
   sweep found two of them in one afternoon. This is the **T26b situation exactly** — a discipline that
   has drifted enough times that the project *gates it rather than restates it* (the references gate was
   floated "until the first drift earns its place" and five drifts earned it; τ=1 has three). **Assess a
   mechanical guard** that flags an accrual-bearing test fixture whose accruals are all unit — e.g. a
   shared schedule-building test helper that records its accruals and a check (in `verify.py` or a
   `tests/tooling/` test) that at least one load-bearing accrual fixture per accrual-bearing module is
   non-unit. **If a clean, non-brittle mechanism exists, build it in T37** (it turns the manual sweep
   into a standing gate, so a *future* annual fixture is caught at CI, not at re-audit #7). **If no clean
   mechanism exists, say so** — the exhaustive sweep (part 2) + §4's existing "assert the non-degeneracy
   as a premise" rule stand as the backstop, and the guard is recorded as `to_go_deeper`. Feasibility is
   Code's call to report; that a guard *should* exist if feasible is the ruling.
4. **Second killers for the single-point-of-failure oracles.** The Sobol Gray code `G(n)=n⊕⌊n/2⌋` (T20b,
   1 killer) is load-bearing for the whole low-discrepancy construction — add a second independent killer
   (a distinct Gray-code property: successive-value single-bit-change, or the reflected-binary count).
   The sweep also flagged **MM-I's freezing denominator `τ/(1+τ)` (T13, 1 killer)** and the **YoY first
   period's `ψ` (T5, 1 killer)** — add a second killer to each **if cheap**; otherwise record them for
   re-audit #4 (the floor is recording, the ceiling is hardening — §9). *(T26's Leg B stays
   single-by-design per re-audit #2.)*

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

## 5. Cowork's ruling on T37 — PR #51 (§10 step 4, 2026-11-14)

Accepted — v0.44.1, 1437 green, nine gates, **test-only (no `src/` change)** — the honest shape, since
both findings were oracle blindness, not wrong code. Re-audit #3's remediation lands here; with it the
re-audit is **closed**.

**Parts 1 + 1b — ratified.** F-3.1 (bootstrap) and F-3.2 (vol surface) both carry semi-annual fixtures
with `ψ ≠ 1` asserted as a premise, and both mutants die (re-run confirmed). The `τ=1` holes are closed.

**Part 2 — ratified; the sweep was exhaustive, not a sample.** Every accrual multiplication in `src/`
mutated (swap_annuity 5, bootstrap-last 2, MM-I 1, MM-II 4, LMM drift `fτ` 15, Bermudan annuity 14,
caplet 2, YoY-first ψ 1, linker ψ **2**) — no third same-class survivor, no different-class survivor, so
no further finding under the scope ruling. The linker going **1→2 killers** (T30's second killer, ruled
at re-audit #2) is measured evidence the second-killer remedy works — the whole point of recording
single-point-of-failure oracles.

**Part 3 — ratified, and done better than asked.** The Gray-code second killer is a different *method*
(the point set built directly from `G(n)`'s set bits, no incremental state) rather than a second
assertion on the same path — real independence. Splitting the one-bit property into its own test
(two claims in one function was exactly what made it single-killer) is the §4 reachability instinct
applied to the test suite itself. MM-I / YoY recorded for re-audit #4 per "if cheap, else record".

**Part 4 — the guard verdict ACCEPTED: not feasible, and declining it is the right call.** I ruled the
guard *should* exist if a clean mechanism did (the T26b "gate a thrice-drifted discipline" precedent);
Code owns feasibility and reported — with both prototypes built and tested, not reasoned — that it does
not. The textual scan false-flags `capfloor` (killed by two tests); the behavioural gate flags 7 of 14
modules with zero real holes (four have no accrual multiplication in their subject code at all). **The
reasoning is charter-grade: a gate needing a 7-of-14 exemption list cannot fail, which is this project's
named defect class — shipping it would be worse than not having it.** The missing half is *whether the
subject code uses the accrual*, a dynamic property only mutation testing answers — i.e. **the τ=1
discipline is not statically gateable, unlike references**, which is the general boundary on the "gate a
drifted discipline" rule. That boundary is the one charter edit this tranche earns (§9): a drifted
discipline is gated only when its violation is statically decidable; a dynamic/semantic violation stays
the mutation sweep's job. Removing the prototype rather than leaving a dead `conftest.py` monkeypatch is
§6. Guard → `to_go_deeper` (recorded, the disposition being mine — Code correctly left it).

**No other charter edit.** The τ=1 recurrence is already §4's first-recorded shape; the exhaustive sweep
is "contain the failure in its domain" applied. Only the gateability *boundary* is new.

**Re-audit #3 is closed** (`reaudit_03_findings.md` disposition updated): register + charter passes clean,
two `τ=1` oracle holes found and fixed, the guard assessed and declined with evidence, four
single-killer oracles recorded (one hardened). The arc boundary opens on merge. **Mark #51 ready (step
6) → Bernardo merges.**
