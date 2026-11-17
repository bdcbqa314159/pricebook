# Tranche 50 — OI-4: the Fourier-inversion truncation defect (src fix)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.55.1` (a patch — a correctness
fix to shipped code, no new capability). **Source of the defect:** `OPEN.md` OI-4, mapped at re-audit #5
(`docs/audit/reaudit_05_findings.md`). **Priority** over T51/T52 — a shipped wrong number at a quoted strike.

## 0. What this is
The Heston/SVJ Fourier-inversion pricer returns a **negative call price** (an arbitrage violation) in the
short-dated high-vol out-of-the-money wing, because the truncation grid (`upper`, `n`) is too coarse where
`φ` decays slowly. Mapped at re-audit #5: **12 of 64 points wrong** over `σ_v×T×K/F`; boundary **`T ≤ 90d
∧ K/F ≥ 1.15`, monotone in `σ_v`, nothing fails at 365d**; and it reaches a **quoted** point —
`σ_v ∈ {0.9,1.2}, 30d, K/F=1.15` reads **4.0%/4.4%** wrong. This is the **"a reference is code too"** hole
(§4): T33's own oracle grid was the weak side there, so the oracle could not see production was wrong.

## 1. The fix — RULED: a regime-aware / adaptive truncation, NOT a wider constant
A flat wider `upper` would tax **every** pricing call to repair a corner (§6 — do not pay everywhere for a
regional defect; Code's 0281 flag, agreed). The fix keys the truncation to the regime that needs it:
- **An adaptive upper bound** driven by `φ`'s decay (the integrand's tail is set by `σ_v`, `T`, and
  moneyness — the slowest decay is short-`T`, high-`σ_v`, OTM, exactly the mapped boundary), or a
  **refinement that triggers** on that boundary. The mechanism is Code's to choose by measurement; the
  **ruling is only that it must not be a flat constant** and must cover the mapped region.
- It lives where the inversion lives — `engine/fourier_inversion.py` (the shared Gil-Pelaez, T49) and/or
  the Heston/SVJ grid config. Config-on-the-model (T44a) if the bound is model-specific.

## 2. The oracle (RULED)
- **The no-arbitrage requirement, over the mapped region (§4 requirement oracle).** The call price is
  **non-negative** and the vertical/butterfly spreads are monotone across **every** `(σ_v, T, K/F)` in and
  around the failing boundary — a mutant restoring the old coarse grid must fail here. This is the
  correctness anchor: an arbitrage violation is a property the finance forbids, so it survives any grid.
- **Convergence against a high-precision reference IN the failing region (§4 "a reference is code too").**
  The reference is an independent high-precision inversion (`upper=1600, n=32001`, or `mpmath`/`scipy`
  quadrature kept off the production path, the T20a arrangement) evaluated **in the bad corner**, not at a
  benign grid where both agree. The production price converges to it there; state the residual.
- **The reduction still holds (§4).** The fix must not move the prices where the old grid was already
  right (365d, ATM) beyond tolerance — assert the benign region is unchanged (a pure-regime fix), so the
  correction is *local* to the defect, not a global reshaping.
- **Mutation.** Restoring the coarse `upper`/`n` must fail the no-arbitrage region oracle; the premise is a
  fixture **in the mapped region** (`T ≤ 90d, K/F ≥ 1.15, σ_v ≥ 0.6`) — a benign fixture is the degenerate
  one (the defect is invisible at 365d/ATM, the "contain the failure in its domain" floor).

## 3. Process
Branch `tranche/50-oi4-fourier-truncation`. `test` (RED: the no-arbitrage region oracle + the
high-precision-in-the-corner convergence, both failing on the committed grid) → `feat` (the adaptive
truncation) → `docs` (close OI-4 in `OPEN.md` with the resolution; the §8 severity-by-domain note) →
`chore(release): v0.55.1`. `slow` is a live risk if the reference is evaluated at high `n` — keep the
high-precision reference in a **dedicated** test at a measured count, not the production path (§9 CI-timing).
Draft PR; mark ready only at step 6.

## 4. What this is NOT
No reshaping of the inversion where it was already correct. No new product or model. No **SVJ-specific**
grid beyond what the shared inversion needs (if the SVJ's slower decay needs a wider regime than Heston,
that is config-on-the-model, ruled by measurement). On merge, OI-4 is CLOSED and the re-audit's one shipped
defect is repaired.

---

## 5. Cowork's ruling on T50 (§10 step 4, 2026-11-17) — RATIFIED; OI-4 CLOSED, OI-5 accepted

Built as ruled — adaptive truncation reading the **actual** `φ` decay (so one rule serves Heston and SVJ,
no second constant), fixed node spacing so only truncation error moves. **OI-4 closed**: 113 materially
negative prices on the shipped grid → **zero** on the fixed one over 600 points, the mapped corner
repricing to a high-precision reference at 1e-6; mutation 17/17. Ratified.

**Item 4 — the `slow-test-exempt` (OI-5) is accepted as ledgered; T50 does NOT wait.** OI-4 is a shipped
wrong number at a quoted strike — its correctness beats a performance refactor, so the exemption is the
right call (your lean, agreed). And it is the exemption done *right*: lossless reductions exhausted first
(memoised pricer, merged redundant region sweeps where the highest-strike non-negativity implies the rest,
halved the reference grid, split the arithmetic-floor oracles along a real seam), then the two gaming moves
**declined** — trimming the region (the oracle's domain, which caught a 16-tranche defect) and over-splitting
the file. That discipline is now a **§9 line** (an honest exemption beats a gamed green; the timing gate must
not shrink an oracle's domain). **OI-5's removal is its own tranche** — sharing one `φ(u)` across a maturity's
strikes in `gil_pelaez_probabilities` (a 4–10× smile speedup in production, the real finding), queued after
T51/T52, raised as its own unit per your "no reshaping where already correct" fence, not smuggled into T50.

**Items 1–3 — self-caught, affirmed, no charter edit beyond item 4's.** (1) The first threshold tested the
integrand *at* the cutoff, not the tail integral, and doubled benign call cost — the "don't tax every call"
ruling drifted because it was a *rationale, not a test*; making the cost property an oracle is the guarded-
vs-folklore fix (§7), and "a rationale that is not a test erodes" is a sharp statement of it. (2) The
arithmetic floor and the **three-branch `checked_call_price`** (price / honest-0 / **refusal** below the
~2e-13 cancellation floor) is the right §3.4 call — a bare clamp would have turned OI-4's own −2.5e-04 into a
plausible `0.0`, the §4 trap; the non-cancelling inversion (Carr–Madan/COS) is correctly `to_go_deeper` route
4, a new object. (3) The cost-not-tested gap (item 1 again) and the SVJ floor-guard bypass (T33b input-
population: every SVJ fixture priced above the floor, so a shared-but-unrouted guard went untested) are both
known shapes, handled.

On merge OI-4 is CLOSED and OI-5 is the standing exemption with its removal tranche queued. T51, T52 follow.
