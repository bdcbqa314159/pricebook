# Tranche 45 — re-audit #4 remediation (test-only)

**Status:** DECIDED by Cowork (§10 step 1). **Target:** `v0.52.1` (a patch — test-only, no `src/`
change). **Source of the holes:** `docs/audit/reaudit_04_findings.md`, F-4.1 … F-4.5. **The T23/T37
pattern** — a re-audit found blind oracles (production code correct, the fixture degenerate); this
tranche hardens the oracles and nothing else.

## 0. What this is
Re-audit #4's proactive invariant-6 / historical-flow sweep found **five blind oracles** — the guard is
present and correct in `src/`, and no fixture makes it bite. All five are closed the same way the
degeneracy is always closed: a fixture that makes the **two branches produce different numbers** (§4), so
dropping the guard fails a test. **No `src/` line changes**; if any does, the hole was misdiagnosed and
that is a finding to raise, not fix silently.

## 1. The five holes and the fixture each needs
| # | site | the guard | the fixture that makes it bite |
|---|---|---|---|
| F-4.1 | `engine/equity_vanilla.py:45` | expired option → PV 0 | an expired option **in the money** (so fall-through intrinsic ≠ 0) |
| F-4.2 | `engine/heston_fourier.py:166` | same | an expired **ITM** option |
| F-4.3 | `engine/forward_cds.py:118` | expired swaption → PV 0 | an **expired-swaption** fixture (there is none today); with a payoff that is non-zero when excluded vs fallen-through |
| F-4.4 | `engine/yoy.py:136` | fixed-leg comprehension excludes settled periods | a **straddling** schedule (some settled, some live — not all-settled) |
| F-4.5 | `engine/yoy.py:152` | par-rate annuity, same | the same straddling schedule |

**The premise each new fixture asserts (the sharpened tenth-shape premise, now charter):** the straddle /
expiry is **necessary, not sufficient** — the excluded flow must be worth a *different* amount excluded
than it is when the dropped guard lets it fall through. For F-4.4/4.5 that is the straddle; for
F-4.1/4.2/4.3 the expired instrument must additionally be **in the money / non-zero**, because an OTM
expired option's intrinsic is `0`, the same number PV-0 gives (F-4.1/4.2 were exactly this — straddled but
OTM).

## 2. The oracle (RULED)
For each site: **a RED mutation test** that drops/bypasses the historical-flow guard and asserts the
fixture above now **fails** (the mutant dies), plus the GREEN assertion that the correct code returns PV 0
for the settled/expired flow. Pre-register each RED before the fixture lands (§4). Where an existing
expired/matured test is the degenerate one (F-4.1/4.2's OTM option, F-4.4/4.5's all-settled schedule),
**strengthen it in place** (make it ITM / straddling) rather than adding a second — a degenerate fixture
left beside a good one is a future trap.

**Mutation is the proof of power** (§4): each new fixture must kill the exact mutant the sweep flagged —
show it, do not assume it. Re-run the sweep's five sites after the fixtures land and confirm 0 survivors
*with the applied count* (§9 — the sweep reports applied/killed/survived; a `0` is read only against a
full applied count, the clause this re-audit added).

## 3. The two charter edits ride with this tranche (§10 — Cowork wrote them, Code stages by path)
Both are in `CLAUDE.md`, ruled in the findings doc, staged under this tranche's `docs:` commit (the
re-audit-#3 → T37 precedent: a re-audit's charter accretions land with its remediation):
1. **§4 tenth-shape premise sharpened** — the straddle is necessary, not sufficient; the two branches must
   differ (for an option bypass, expired **ITM**). Earned by F-4.1/4.2.
2. **§9 sweep-instrument hardening** — a whole-sweep "0 survivors" is meaningless without the applied
   count; the sweep reports applied/killed/survived. Earned by the 21-errored-mutants false zero.

## 4. What this is NOT
No `src/` change (it is test-only; a needed `src/` change means a misdiagnosis → raise it). No new shape in
the catalog (F-4.1/4.2 are the primordial "zero makes a factor invisible" shape, F-4.3/4.4/4.5 the tenth —
ruled). No new product. This closes re-audit #4's oracle-power remediation; on its merge the re-audit is
complete and the library is at a fresh arc boundary.

## 5. Process
Branch `tranche/45-reaudit4-remediation`. `test` (RED: the five mutation tests + strengthened fixtures) →
`feat` is **empty or absent** (test-only; if a GREEN `src/` change appears, stop and raise it) → `docs`
(the two charter edits, the findings-doc rulings, this spec) → `chore(release): v0.52.1`. Draft PR;
mark ready only at §10 step 6.

## 6. Cowork's ruling (reserved for §10 step 4, after the build)
