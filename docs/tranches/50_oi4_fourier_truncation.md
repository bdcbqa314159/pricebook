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

## 5. Cowork's ruling on T50 — (reserved for §10 step 4, after the build)
