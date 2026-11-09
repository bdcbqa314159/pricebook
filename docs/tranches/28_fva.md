# Tranche 28 — FVA: the funding valuation adjustment, a fixed point with a closed form

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.34.0`.
**Source:** `CREPEY-2026` Prop 1 — eq (17) the closed form, eq (19) the fixed point it solves.
**Pre-registered RED.** Fourth and last **core** unit of the XVA spine arc (`docs/xva_spine_arc.md`);
follows T27. Collateral (T29) is optional after it, then the checkpoint closes the arc.

## 0. The product worth having
**The funding valuation adjustment of the uncollateralized netting set** — the cost of funding the
position at the bank's own spread, which completes the elementary spine (CVA · DVA · FVA). Crépey
defines it as a **semilinear fixed point** (eq 19):

> FVA = γ (P − CVA − FVA)⁺

and proves (Prop 1, eq 17) it has the unique **closed form**

> FVA = γ (P − CA)⁺ = (γ / (1+γ)) · (P − CVA)⁺,  with CA = CVA in the uncollateralized, no-IM setting.

`P` is the clean (risk-free) mark of the netting set, `CVA` is T27's counterparty adjustment, `γ` the
funding spread (an input, a probability-like quantity over the horizon). The `⁺` says a **net receiver**
(`P − CVA ≤ 0`) has no funding cost: FVA = 0.

## 1. The pieces and their layers
- **L5 `risk/` — the FVA functional.** Thin: `fva = γ/(1+γ) · max(P − CVA, 0)`, reusing T27's CVA and
  the clean mark `P` (the time-0 netting-set value already computed by the exposure/swap machinery). No
  new L1, no new model. `CA = CVA` here because the spine is uncollateralized with no initial margin —
  **state that assumption in the module**; MVA and the collateral terms that would make `CA ≠ CVA` are
  deferred (T29 / `to_go_deeper`).
- If `P` and `CVA` are MC estimates, FVA inherits their SE (invariant 7); if closed-form, exact. The
  pathwise estimator carries over where the inputs are simulated.

## 2. The oracle (RULED) — settle the fixed-point question
Code's spec-time flag, ruled: **the closed form is the production value and the primary pointwise
transcription; the fixed-point equation is the strong requirement oracle; the iteration is a secondary
guard.** In order:

- **Pointwise transcription (§4, closed form — the top rung).** `FVA = γ/(1+γ) · (P−CVA)⁺` transcribed
  from Crépey eq 17 in **both `-raw` and `-layout`**. **The `γ/(1+γ)` fraction is a reflow hazard — the
  exact shape that bit eq (25)** (a `-layout` split the numerator from the denominator across lines);
  read it in both renderings and reconcile before trusting it. This is the number production returns.
- **The defining-equation requirement oracle — the anchor (§4, "a formula and its stated inverse must
  actually invert").** The closed form must **satisfy its own fixed point**: substitute it back and
  assert `FVA == γ·(P − CVA − FVA)⁺` to machine precision. This is derived from the finance, so it
  **survives a transcription error** — and it is sharp: a wrong closed form (e.g. `γ·(P−CVA)⁺` without
  the `1/(1+γ)`, the obvious slip) does **not** satisfy the fixed point, so this oracle kills exactly
  the mistake the reflow hazard invites. Pre-register it as the RED.
- **Convergence of the iteration — a consistency guard, not the anchor (§4, self-consistency
  telescopes).** `FVA_{n+1} = γ(P − CVA − FVA_n)⁺` from `FVA_0 = 0` converges to the closed form (the
  map is a contraction, factor `γ/(1+γ) < 1`); assert it reaches the closed form and report the rate.
  Keep it as a guard — it shares `γ` and `P−CVA` with the closed form, so it telescopes; the
  fixed-point-satisfaction check above is the non-telescoping correctness anchor.
- **Both regimes of the `⁺` (degenerate-fixture guard, mandatory).** Two fixtures: `P − CVA > 0`
  **materially** (FVA > 0, the `⁺` open — assert the non-degeneracy, else FVA ≡ 0 and the whole formula
  is invisible, the §4 trap), and `P − CVA < 0` (FVA = 0, the `⁺` binds — a **net receiver** has no
  funding cost; assert the exact zero). The fixed point is degenerate (FVA=0) in the second regime by
  construction — assert that the iteration and the closed form agree there too.
- **Reduction:** `γ = 0` ⇒ FVA = 0 (no funding spread) — exact. Monotonicity: FVA ↑ in `γ` and in
  `(P − CVA)⁺`.
- **Mutation table (report it):** drop the `1/(1+γ)` (→ fails fixed-point satisfaction); drop the `⁺`
  (→ a net receiver wrongly gets a negative FVA — the net-receiver fixture kills it); use `P + CVA` or
  `P` alone instead of `P − CVA`; flip a sign. Each must die.

## 3. Process
Branch `tranche/28-fva`. `test` (RED: the fixed-point-satisfaction oracle + the two `⁺`-regime fixtures
+ the mutation table) → `feat` → `docs` → `chore(release): v0.34.0`. **Open the draft PR early.**
`source:` `CREPEY-2026`. `references` gate stays green (no new key). `crosscheck:` QuantLib FVA or
`none — <reason>` (§8). **Raise as an `Ask:`** if the source's `CA = CVA` identity needs a term the
spine doesn't have (it should not, uncollateralized) — do not silently substitute.

## 4. What this tranche is NOT (scope fence)
No MVA (margin valuation adjustment) and no collateral — those are what make `CA ≠ CVA`, deferred to
T29 / `to_go_deeper`. No KVA — the cost-of-capital adjustment and the `(1−γ)⁻¹` fininsurance
normalisation are `to_go_deeper`/`out_of_method` (the deferred frontier; T27 route 2). No wrong-way
risk. This closes the elementary spine's core; the arc checkpoint `CP_xva_spine.md` follows (after the
optional collateral tranche, if interest pulls it).
