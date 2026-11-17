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

## 5. Cowork's ruling on T51 (§10 step 4, 2026-11-17) — RATIFIED

Built as ruled, **`src/` unchanged** (the spec's own test of correct diagnosis — passed), 2145 tests,
eleven gates green, mutation **5/5 killed, every site with ≥ 2 independent killers**. The three blind
oracles are closed. Ratified.

**F-5.1 / F-5.2 — put priced through `price(product, model)`, right.** The put arm is now entered, and
Heston's anchor is the `σ_v=0` reduction to **Black's put computed in the test** — not the engine's own
call rearranged, which would have been F-5.2's mistake wearing different clothes (a reference built from the
code under test). The reachability rule (T18) applied with its own trap avoided.

**F-5.3 — the instrument was changed, not widened, and it completes the per-defect clause.** The `4·SE`
band (11.4–30%) cannot resolve the 6.18% defect at any reasonable path count (40,000 paths / 2.8s), so the
fix is **pointwise** against the closed-form `surface_call_price` (~1e-12, no paths). Your offered
sharpening is accepted and **added to the §4 per-defect clause**: *when a band cannot resolve a defect,
first ask whether that quantity needs a band at all* — a closed-form sub-quantity inherited the band from
its MC neighbour by accident and should be oracled exactly, not given more of the neighbour's noise.

**The carry-fixed second killer is the best moment in the tranche.** Two curves with the same `r−q=0.05`
give the **same forward**, so the surface vol and undiscounted Black value are identical by construction and
only the discount factor differs — the price ratio must equal the df ratio exactly (2e-16), a dropped `df`
collapsing it to 1.0 (4.88% away). Exact, free, shares no line with the first killer — exactly the §9
"second independent killer for a previously-shipped hole" done as a construction, not a second fixture.

**Premises asserted throughout** (`df < 0.97`, and F-5.3 asserting both that the defect is material and that
the band cannot resolve it — so the second oracle is justified, not merely added). The MM-I single killer is
correctly left for re-audit #6. On merge, re-audit #5's three blind oracles are closed; **T52 closes the
re-audit**, then the `φ`-sharing tranche retires OI-5.
