# Handoff Report — the L5 risk & capital run (v0.19.0 → v0.37.0)

Date: 2026-07-16   ·   Version: `0.37.0`   ·   Tests: 222 green (`verify.py all`)

Fourth build → design report. Covers the long L5 run after "resume forward with risk to L5":
generic greeks, the unified calibration front (L3), and the full **XVA / regulatory-capital
stack**. This is a **checkpoint for a Cowork brainstorm**, not a layer-completion hard stop —
L5 is deep but not exhausted, and §4 carries the open design questions. The headline is a
**measure-consistency decision** the MPOR slice forced into the open.

---

## 1. Slices landed

| Slice | Version | What |
|---|---|---|
| curve greeks | 0.19 | generic keyed `curve01`/`bump_curve`; `credit01` an alias |
| bootstrapped dv01 | 0.20 | `DiscountCurve.bumped` (parallel zero shift) |
| key-rate buckets | 0.21 | `key_rate_dv01` — per-pillar KR01, Σ = dv01 |
| calibration front | 0.22 | `calibration/` (L3); `calibrate_hull_white` to a ZCB-option (caplet) |
| bootstraps under front | 0.23 | rate/credit bootstraps migrated L1 → L3 |
| HW cap strip | 0.24 | joint `(a,σ)` least-squares; stdlib `nelder_mead` (L0) |
| CVA | 0.25 | `cva` = protection-leg on the exposure profile |
| MC exposure engine | 0.26 | `exposure_profiles` (EPE) — **per-date T-forward measure** |
| BCVA | 0.27 | DVA + bilateral; EPE **and** ENE from one pass |
| FVA | 0.28 | funding annuity on net exposure |
| KVA | 0.29 | cost-of-capital annuity; extracted `_annuity_adjustment` |
| SA-CCR | 0.30 | `saccr_ead` = α(RC+PFE); 5.5%-of-notional anchor |
| forward EAD → KVA | 0.31 | EAD runoff → capital profile → KVA |
| stochastic EAD | 0.32 | RC = EPE; unifies exposure + supervisory PFE |
| netting-set SA-CCR | 0.33 | signed notionals, maturity buckets, correlations |
| MVA | 0.34 | funding annuity on the IM profile (3rd annuity consumer) |
| PFE / dynamic-IM | 0.35 | `pfe_profile` — exposure quantiles (monotone-transform oracle) |
| margined exposure | 0.36 | threshold CSA; `E = min(max(±V,0), H)` |
| **MPOR paths** | 0.37 | risk-neutral OU **joint paths**; close-out gap `max(V(t)−V(t−MPOR),0)` |

Every slice is red→green→release with a named oracle; `verify all` green on each.

## 2. The shape that emerged

Two integral families, both fed by the exposure/SA-CCR machinery:

```
                       ┌── protection-leg weight  (1−R)·ΔQ ──►  CVA · DVA · BCVA
 exposure engine ──────┤
   (EPE/ENE, PFE)       └── funding-annuity weight  S·τ    ──►  FVA
                                                              ┌►  KVA  ← SA-CCR EAD runoff → capital
 SA-CCR (single+netting) ─────────────────────────────────────┤
   (RC + PFE, α=1.4)                                          └►  (RWA)
 PFE quantile → dynamic IM ────────── funding-annuity ────────►  MVA
```

- **One annuity** (`_annuity_adjustment`, `rate·Σ profile·DF·S·τ`) backs FVA, KVA, MVA.
- **One protection leg** (the CDS `cds_pv` math) backs CVA/DVA/BCVA.
- **One simulator core** (`_simulate_swap_values`) backs EPE/ENE, PFE quantiles, and threshold
  collateral. SA-CCR EAD is shared single-trade ↔ netting-set ↔ forward runoff ↔ stochastic.
- Everything is keyed to survival curves in the snapshot (A5), so `bump_curve`/`credit01`
  already differentiates the whole stack.

## 3. Oracles (how each is pinned, non-circular)

- CVA == CDS protection leg at zero spread; FVA/KVA/MVA on unit profile == `rate·RPV01`.
- MC exposure: `σ=0` deterministic (exact) **and** `P(0,t)·EPE(t) ==` the analytic co-terminal
  swaption (Jamshidian) — discounted EE *is* a swaption strip.
- PFE quantile == `V` at the `r`-quantile (monotone transform of a Gaussian).
- SA-CCR: 10y ATM $100mm IRS EAD ≈ 5.5% notional (published anchor); netting mirror-hedge → 0.
- MPOR path simulator: reproduces the analytic HW/OU moments (marginal `α(t)`, variance, and
  **cross-date covariance** `e^{−a(t−s)}·var(s)`).

---

## 4. Open design questions — for the brainstorm

### 4.1 — HEADLINE: measure consistency across the exposure stack
The EE/PFE/collateral profiles simulate **each date independently under its own T-forward
measure** (`forward_short_rate`). The MPOR slice needed cross-date correlation, so it added a
**risk-neutral joint-path simulator** (`_simulate_rate_paths`). We now have **two measures**
in one engine.

- For the local MPOR *difference* `V(t)−V(t−MPOR)` the measure is second-order (flagged
  provisional in the code + CHANGELOG).
- But it must be settled before the measures mix further. Options:
  1. **Migrate everything to risk-neutral paths.** One simulator; EE = `mean max(V,0)` on paths;
     the "discounted-EE == swaption strip" oracle changes (risk-neutral EE ≠ forward-measure EE;
     stochastic discounting re-enters). Most physically standard for CVA/exposure systems.
  2. **Keep both, scoped:** forward-measure marginals where a clean analytic oracle exists
     (EPE/PFE), risk-neutral paths only for genuinely path-dependent quantities (MPOR, callables).
     Cheaper now, but two engines to keep consistent.
  3. **Terminal-measure path simulation** with the forward-measure marginal preserved per date
     (drift-adjust so marginals match), reconciling the two.
- **My lean:** (2) short-term (it's what we have and each engine is internally oracle'd), with
  (1) as the eventual target once a path-based EE oracle is in place. Wants your ruling.

### 4.2 — A unified `xva_report`
CVA/DVA/BCVA/FVA/KVA/MVA are separate calls, each re-running the exposure engine. Propose a
single `xva_report(trade, counterparty, self, funding, ...)` that simulates **once** and returns
all adjustments (+ EE/PFE/EAD profiles). Rule-of-two is met. Question: the right input bundle /
does it belong at L5 or the L6 shell (it's portfolio-level)?

### 4.3 — The HW-1F / flat-curve ceiling
All exposure is Hull-White one-factor on a **flat** curve. Real desks need a bootstrapped-curve
HW and ideally multi-factor / a second asset. This is the biggest *modelling* gap under the whole
stack. Is it worth a "general-curve HW" slice now, or after breadth?

### 4.4 — L6 shell is untouched
We've gone deep in L5 while **L6 (trade / book / benefit table / realized-vs-mark)** is empty and
the quarry empties slowly. Per CLAUDE.md the realized-P&L / lifecycle lives there. Time to cut a
vertical up into L6 (book a trade, run its life) to exercise the A3 decomposition end-to-end?

### 4.5 — Breadth vs depth: how far down XVA?
Still open below the current line: first-to-default BCVA (survival cross-weighting), wrong-way
risk, SIMM (sensitivity-based IM vs the current PFE-proxy), margined SA-CCR (MPOR maturity
factor), capital floors, collateral haircuts, ColVA. All real, none urgent. Where's the line?

---

## 5. Recommendation

Checkpoint here. The L5 stack is coherent and fully oracle-gated, but **§4.1 (measure)** is a
foundational fork that should be ruled before more path-dependent work, and **§4.4 (L6)** is the
migration-health question — the quarry empties by *breadth up the layers*, not by exhausting L5.
Suggested next after the brainstorm: rule §4.1, then either the `xva_report` consolidation (§4.2)
or the first L6 vertical (§4.4).
