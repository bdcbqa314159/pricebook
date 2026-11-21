# T73 — re-audit #8 remediation (one tranche → v0.75.0)

**Cowork spec, 2026-11-21.** Re-audit #8 swept **68/63/5** — one real blind oracle (F-8.6), five
previously-shipped holes on single killers (F-8.3), a dead field (F-8.4), an instrument gap (F-8.2), a
register miss + a bin ruling (F-8.1), and the C16 strip-walk extraction (pass 2). **One tranche**, the T59
precedent (test/tooling + small src changes together). The one charter edit (§4, F-8.6's `pytest.approx`
floor) rides this merge. Order the RED commit L0-first.

## What the tranche carries

### 1. F-8.6 — the `pytest.approx` relative-floor blind oracle (test-only + the clause)
`pytest.approx(x, rel=r)` floors at `1e-12` absolute, so a `rel=` band is secretly absolute in a tail — the
T20a/OI-3 `norm_cdf` survival defect (`1−cdf`) survived all 3069 tests. **Fix:** `abs=0.0` on the native
survival site (measured `0 → 7` killers), **and audit the blind subset** — the `approx(rel=)` sites where the
expected can fall `≪1e-12` (tails, differences-of-equals; ~357 sites tree-wide, the blind ones a small
subset) — pinning `abs=0.0` there. The charter clause is in §4's T20a bullet (read before staging, §10).

### 2. F-8.3 — second independent killers for the five previously-shipped holes (test-only, HIGHEST VALUE)
F-4.1…F-4.5 (the expired-option bypasses, shipped and remediated at T45) are each guarded by a **single**
killer, and §9 says a previously-shipped hole wants a second independent one (the F-1.2/T37 rule, never
applied here). The killers are **fixture-dependent** (an **in-the-money** expired fixture — the fall-through
intrinsic must differ from PV 0), so a later edit can neutralise them silently. Add a **second independent
killer** for each (a different ITM-expired construction — e.g. the opposite-side floorlet pattern T72 used),
and **record all five in the standing single-killer table** (they were never there).

### 3. F-8.4 — delete `VasicekParams.long_run_mean` (src, one field)
Dead surface: zero consumers in `src/` or `tests/`, §4's third survivor kind, in the arc's oldest tranche.
Delete it (nothing speculative, §6); confirm the model computes the long-run level from `(κ, θ)` where needed.

### 4. F-8.2 — make the sweep instrument's restore SIGTERM-safe (tooling)
`tests/tooling/mutation_sweep.py`'s `finally`-based restore is bypassed by SIGTERM, which stranded a mutation
**twice** this sitting (once corrupting `sobol.py`'s Gray code in the working tree). Add a **signal-handled
restore** (trap SIGTERM/SIGINT → restore → re-raise) **and a post-sweep working-tree-clean assertion** (the
sweep fails loudly if any source is left mutated). A known-answer test for the signal path (the instrument's
own-test discipline, T52/T65). The "never `git add -A`" rule is a backstop, not the guard.

### 5. F-8.1 — the register miss + the ch-8.5 bin (docs + §1)
`to_go_deeper.md` held **none** of the arc's T71/T72 routes though pass (3) and the checkpoint said it did —
transcribe the rulings (the CEV-short-rate / drifted-CEV / surface-calibration routes, the T68 routes already
there). And **ch-8.5 stochastic-vol smile moves to `out_of_method`** as a **candidate** (no closed form →
fails the §4 oracle gate → not come-back-same-method debt; re-examined only if a source with a closed-form
approximation — Hagan SABR — is acquired, §1). Fix `references.md`'s ch-8.5 row to point at the right bin.

### 6. C16 — extract the strip walk across the THREE `NominalCap`-family engines (src refactor)
The four caplet engines share a strip skeleton, but only **three** share the **shape**: `smile_caplet`'s two +
`cev_caplet` (build `CapletPeriod`, compute the **fixing-horizon** variance, value per-period, accumulate).
Hull-White shares the *walk* and not the *variance* (two dates vs one — stays separate); `capfloor`/`yoy`
take different payoffs (§6 leg-vs-YoY-leg — stay separate). **Extract a per-period-valuation callable (or a
small protocol)** the three share, with the per-period valuation (Bachelier / displaced-diffusion / CEV) the
varying part. **Preserve each engine's exception set explicitly** — `cev_caplet`'s `ArithmeticError` is the
one a careless merge drops silently. The extraction *creates* the shared fixing-horizon-variance value C16
has lacked, so **C16 flips `per-module → guarded`** (a divergence becomes inexpressible, the C9 structural
form). Purity: the existing oracles green + the per-engine prices bit-identical in-process (or an ULP bound if
op order moves, T12(a)). If the protocol design proves large, raise an `Ask:` and I split it to T74 — default
is one tranche.

## Gates / provenance
- RED-first for F-8.6's `abs=0.0` tests, F-8.3's five second killers, F-8.2's signal known-answer test.
- Re-time `test_cev_caplet` and any touched heavy file on **both** CI runners (§9 — F-8.9's margin).
- Version → **v0.75.0**; one floor bump; one `CHANGELOG` entry naming re-audit #8.

## Out of scope (carried)
- **C2** (vol clock) — the widest OWED, now touched by the smile models' forward clock; the C16 extraction may
  give it a handle but does not close it.
- **`dividends_before(T)`** — still owed. **OI-7** — stays open.
- The slow-budget-margin question (F-8.9) — **Bernardo's**, not this tranche's.
