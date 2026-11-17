# Tranche 51 — re-audit #5 oracle hardening (test-only)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.55.2` (test-only, no `src/`
change). **Source of the holes:** `docs/audit/reaudit_05_findings.md`, F-5.1/F-5.2/F-5.3 (the three sweep
survivors). The **T45/T23/T37 pattern** — a re-audit found blind oracles (production code presumed correct,
the oracle degenerate); this tranche hardens the oracles and nothing else.

## 0. What this is
Three blind oracles from re-audit #5's discount-factor sweep, closed the usual way: a fixture/oracle that
makes the dropped-`df` mutation **die**. **No `src/` line changes**; if one does, the hole was misdiagnosed
and that is a finding to raise, not a silent fix.

## 1. The three holes and the killer each needs
| # | site | why it survived | the killer |
|---|---|---|---|
| F-5.1 | `heston_fourier.py:136` | the registered pricer's **put arm** is never taken — no put priced through the engine | price a **put through the registered Heston engine** (enter the put arm), so dropping its `df` fails |
| F-5.2 | `bates_fourier.py:117` | same, inherited; the parity test computes the put **test-side** from the call | price a **put through the registered Bates engine** — not a test-side parity reconstruction |
| F-5.3 | `local_vol_mc.py:151` | the MC repricing band **absorbs** the dropped `df` (6.18% inside an 11.8–33.1% band) | a **discounting-specific resolution oracle** — a differential/higher-path check that resolves a dropped `df` where the generic band cannot |

## 2. The oracle (RULED)
- **F-5.1 / F-5.2 — reach the engine's own put arm (§4 reachability, T18).** The lesson F-5.2 carries:
  **asserting put-call parity with the put computed test-side never enters the engine's parity code** — a
  relation satisfied by test-side arithmetic leaves the engine's own side untested. So the killer prices a
  **put through `price(product, model)`** (the registered dispatch), takes the engine's put arm, and a
  dropped `df` there fails. Pre-register each as a RED mutation test. If an existing parity test computes
  the put itself, **strengthen it in place** to route through the engine (the T45 strengthen-in-place rule),
  unless that test is the sole killer of a *distinct* guard (then keep both — the T45 "count what a fixture
  is the sole killer of" rule).
- **F-5.3 — a per-defect resolution oracle (§4, the line this re-audit added).** The band's resolution is
  per-defect: the existing Dupire-drift resolution test is correct and silent about discounting. Add a
  **dedicated** oracle that resolves a dropped `df` — a differenced/seed-shared run or a higher-path power
  test measured so the dropped `df` falls **outside** its band, with the resolution **stated** (the smallest
  discounting defect a green rules out). Assert the Dupire-drift resolution test still holds (it guards a
  different defect — do not weaken it).
- **Mutation is the proof (§4).** Each killer must kill the exact sweep survivor — show it, do not assume
  it. Re-run the three sites after the tests land: applied/killed/survived with the applied count (§9), and
  confirm the mutated code ran (the T46 bytecode clause).

## 3. Process
Branch `tranche/51-reaudit5-hardening`. `test` (RED: the two put-through-engine tests + the discounting
resolution oracle) → `feat` **empty or absent** (test-only; a GREEN `src/` change means a misdiagnosis —
stop and raise it) → `docs` → `chore(release): v0.55.2`. Draft PR; ready only at step 6.

## 4. What this is NOT
No `src/` change. No new product. No re-sweep of the whole tree (that was re-audit #5). The MM-I
single-killer (`market_model_i.py:154/:155`, item 3) is **not** in scope here — its second killer rides
whatever next touches MM-I (recorded for re-audit #6), not forced. On merge, re-audit #5's three blind
oracles are closed.

---

## 5. Cowork's ruling on T51 — (reserved for §10 step 4, after the build)
