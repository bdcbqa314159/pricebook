# Tranche 26 — the Monte-Carlo exposure engine: CVA of a simulated netting set

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.31.0`.
**Sources:** `CREPEY-2026` (the exposure/CVA frame) + the library's own MC machinery (Jäckel arc,
T15–T21) and T25's closed-form EE as the reduction anchor. **Pre-registered RED, as always.** Second
unit of the XVA spine arc (`docs/xva_spine_arc.md`); follows T25.

## 0. The product worth having
**CVA of an uncollateralized netting set whose exposure is produced by simulation** — the step from
T25's *analytic* exposure to exposure through the engine. T25 priced CVA from a Gaussian exposure given
in closed form; T26 builds the thing that produces an exposure profile for a **real instrument** by
simulating its mark-to-market on a date grid and taking `EPE(t) = E[V(t)⁺]`, then feeds T25's
`unilateral_cva` functional unchanged.

This is where "L5 on the engine + a `Priceable` protocol" (§2) finally materialises, and where T25's
closed form stops being the whole story and becomes a **reduction check** on the estimator.

## 1. The pieces and their layers
- **L5 `risk/` — the MC exposure engine.** A generic estimator: given a process that simulates a
  tradeable's value `V(t)` on a time grid (reusing `engine/mc_path_engine` and the Jäckel sampling
  machinery — antithetic, the bridge, Sobol where it pays), produce `EPE(t) = mean(V(t)⁺)` and
  `ENE(t) = mean(V(t)⁻)` **with their per-date standard error**, assembled into the existing
  `ExposureProfile` (T25). The engine is **instrument-agnostic** — it reprices whatever product the
  caller hands it at each date; it does not know "swap".
- **The first real consumer — a single uncollateralized interest-rate swap under the LMM.** The swap's
  remaining fixed-vs-float value is repriced from the simulated forwards at each exposure date (the LMM
  MC is already in the tree, T16–T21). Its exposure is **not Gaussian** (forwards are lognormal) — that
  is the point: it is the non-degenerate case T25's closed form cannot reach.
- **`PricingResult` with SE (invariant 7).** The CVA is now an **estimate**: `EPE(t)` carries sampling
  error, so the CVA it feeds does too. The MC-exposure CVA returns a `PricingResult` carrying its
  `standard_error`, projected from the exposure estimator's per-date SE through the deterministic
  `∫ DF·dPD` weights — the same projection pattern as the swaption engine (T17/v0.20.1). T25's
  closed-form `unilateral_cva` is unchanged and keeps returning an exact number (`SE = None`).
- **The `is_historical` atom (C10's trigger), pulled here.** The exposure grid runs over future dates
  only — the engine reads the valuation-date boundary — so this is the consumer that **pulls** the
  `is_historical(pay, valuation_date)` extraction the re-audit flagged (T24 §9.2), upgrading C10 from
  `per-module` to `guarded — structurally`. Extract it now because a real consumer needs it (not
  speculatively, §6).

## 2. The oracle (RULED) — two legs, because self-consistency alone telescopes
T25's closed form is now **load-bearing for the whole arc**. A T26 checked *only* against it is
self-consistency against our own code (§4 rung 3), which passes even on a shared wrong convention. So
T26 carries an **independent** anchor beside the reduction — this is the ruling from PR #34's review
(thread 0127), and the reason the arc plan's T26 line already says so.

- **Leg A — the exact reduction to T25 (validates the *estimator*).** Feed the exposure engine a
  **Gaussian** process — an arithmetic Brownian mark `V(t) = V₀ + σ·W(t)`, whose `V(t) ~ N(V₀, σ²t)` is
  exactly T25's Bachelier case — and assert the simulated `EPE(t)` converges to T25's closed-form
  `expected_positive_exposure` within the sampling band, at the `−1/2` MC rate. This pins the
  estimator's positive-part/averaging against a known closed form. Necessary but **telescoping** (both
  sides share the exposure convention), so it is not the whole oracle.
- **Leg B — the independent anchor (validates the *process*, consulting T25 nowhere).** Under the
  pricing numéraire `B_N`, every traded value over the numéraire is a **martingale**, so
  `E[V(t)/B_N(t)] = V(0)/B_N(0)` **exactly, by construction — no model result, no T25**. Assert the
  simulated numéraire-relative swap value reproduces `V(0)/B_N(0)` at every exposure date, within its
  SE and improving as `√N`. This is the T19 free-control identity reused as an exposure oracle: it
  catches a mis-specified drift or measure that Leg A (which only checks the `⁺` transform against a
  closed form sharing the same paths) cannot see. *(If the swap value's martingale check is awkward to
  pin directly, pin the simulated forwards' own martingale property under their terminal measure — the
  T16/T19 identity — which the swap value inherits; state which was used.)*
- **Banded resolution (§4).** The Leg-A reduction band is `k·SE + 0` (the reduction is to an *exact*
  closed form, so there is no approximation error to add — unlike T17's MC-vs-approximation case);
  state the committed path count's resolution and add a higher-path power test at the count where a
  defect of the size you care about falls outside. The CVA estimate's SE must be **measured from the
  same paths** the engine consumes (invariant 7's correctness point), never re-simulated.
- **Requirement oracles carry over from T25** and must still hold on the simulated exposure: CVA ≥ 0,
  the two zeros (`LGD = 0`, `λ ≡ 0`), monotonicity in the hazard and `LGD`. And a **cross-check to
  T25 at the Gaussian case**: the MC-exposure CVA on the Brownian mark must match T25's closed-form
  CVA within the band — a cross-product oracle (different code each side: simulated vs analytic
  exposure, same CVA integral).
- **Degenerate-fixture guards (post-F-1.2, mandatory).** Assert the non-degeneracy premises: the
  exposure has **non-zero variance** (else EPE = forward and the `⁺` is invisible); the grid has **more
  than one future date** (else a single-date reduction hides the profile's time structure); the
  martingale check runs on a **non-flat** numéraire (else `B_N(t)/B_N(0) = 1` and a dropped
  discounting survives). **Mutation is the proof of power** — report the table: drop the `⁺` (use the
  raw mean), drop a date's discounting in the numéraire-relative check, mis-align the exposure grid to
  the survival grid, feed the estimator the forward instead of the simulated value. Each must die.

## 3. Process
Branch `tranche/26-mc-exposure-engine`. `test` (RED: the Leg-A reduction + the Leg-B martingale
identity + the mutation table, pre-registered) → `feat` → `docs` → `chore(release): v0.31.0`.
`source:` names `CREPEY-2026`; the engine module `wants` a `crosscheck:` target — QuantLib has CVA /
exposure tooling (`crosscheck: quantlib <exposure/CVA> — deferred to the QuantLib tier`, or
`none — <reason>` if judged outside the tier, §8). The CVA-returning engine carries `standard_error`
(invariant 7). **Open as a DRAFT PR** (§10 step 3). Raise any source discrepancy, or any difficulty
pinning Leg B directly, as an `Ask:` — do not rule it or silently weaken the anchor to a
self-consistency check.

## 4. What this tranche is NOT (scope fence)
No bilateral CVA/DVA (T27). No FVA (T28). No collateral/CSA (T29 or deferred). No wrong-way risk — the
exposure/default independence of T25 still holds (WWR is the thing that independence drops, deferred).
No new credit model — the T25 survival curve is reused. KVA, BSDEs, neural-net regression, and the
risk-measure chapters remain in `out_of_method.md`.

**On the model for the real consumer:** the LMM MC already in the tree (T16–T21) is the intended
process for the swap's exposure. If repricing the swap's remaining legs from the simulated forwards at
each date turns out to need a capability the tree lacks, that is an `Ask:` (read-before-asserting on
our own tree, §10/§5) — not a reason to substitute a different instrument without ruling.
