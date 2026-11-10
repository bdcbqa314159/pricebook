# Tranche 29 — collateral / CSA: the collateralized exposure over the margin period of risk

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.35.0`.
**Source:** `CREPEY-2026` ch III §3.B (collateral and default cash flows) for the frame; the
margin-period-of-risk (MPoR) gap exposure's closed form is **derived** (standard, Bachelier on the
increment), not transcribed. **Pre-registered RED.** The optional unit of the XVA spine arc
(`docs/xva_spine_arc.md`), chosen before the checkpoint; **after it lands, `CP_xva_spine.md` closes
the arc and re-audit #2 fires (§9).**

## 0. The product worth having
**The collateralized exposure under a CSA, and the CVA/DVA it feeds.** A CSA means the parties post
**variation margin (VM)** against the mark, so the exposure is no longer `V(t)⁺` but the **gap** the
collateral fails to cover. The realistic gap is the **margin period of risk**: at a default the last
VM call is stale by δ (≈ 2 weeks), so the exposure is the move in `V` over δ, not the level of `V`.

The headline: **collateral reduces CVA**, and with continuous margining (δ → 0) kills it. T29 makes
that quantitative and reuses everything above it — the collateralized exposure is just a different
`ExposureProfile`, and T25/T27's `credit_adjustment` functional prices CVA/DVA on it **unchanged**.

## 1. The pieces and their layers
- **A `CollateralTerms` value object (frozen vocabulary, §6).** The elementary CSA: **MPoR `δ`**,
  **threshold = 0**, **MTA = 0**, bilateral VM, cash collateral. Thresholds/MTA (which make collateral
  a *nonlinear* function of `V`), initial margin, segregation and rehypothecation are **deferred**
  (`to_go_deeper`). Keep it concrete and frozen; whether it generalises is decided at a second CSA
  variant.
- **L5 `risk/` — the collateralized exposure.** Extends T26's exposure engine with the collateral
  rule: at default date `t` the posted collateral is `C(t) = V(t − δ)` (zero threshold ⇒ collateral =
  the last marked value), so the collateralized exposure is `E[(V(t) − V(t−δ))⁺]` — the **gap over the
  MPoR**. ENE mirrors it for DVA. The resulting `ExposureProfile` flows into T25/T27's `credit_adjustment`
  with **no change to the CVA/DVA code** — collateral enters only through the exposure.
- No new L1, no new model. Reuses the survival curve (both parties), the discount curve, the LMM/analytic
  mark, and the CVA/DVA functionals.

## 2. The oracle (RULED)
- **Two exact reductions — the preferred anchors (§4), one to zero and one to the prior tranche.**
  - **δ → 0 (continuous margining) ⇒ collateral tracks `V` exactly ⇒ collateralized exposure → 0 ⇒
    collateralized CVA → 0.** Exact by construction (`V(t) − V(t) = 0`). Perfect collateralization kills
    the adjustment.
  - **Collateral off (threshold → ∞, or `C ≡ 0`) ⇒ collateralized exposure = the uncollateralized
    exposure ⇒ collateralized CVA → T26/T27's uncollateralized CVA**, bit-identically. The second
    reduction is also the **rule-of-two purity proof**: uncollateralized is the `C ≡ 0` special case of
    the collateralized engine, so the T26/T27 path must stay bit-identical (or exact-arithmetic identity
    if ops reorder, §4 — report which).
  - **Each reduction needs the non-degenerate companion** (a reduction a no-op also satisfies is
    vacuous): a material `δ > 0` with a volatile mark, where the collateralized CVA is strictly between
    0 and the uncollateralized value.
- **Closed-form anchor on a Gaussian mark (pointwise, derived).** For a Bachelier mark the gap
  `V(t) − V(t−δ) ~ N(μ_gap, σ²δ)` is Gaussian, so the collateralized EPE is T25's
  `expected_positive_exposure` on the **gap** distribution — a closed form the MC estimator must reach
  within its band at the `−1/2` rate (T26's Leg-A pattern). **Derive it** (standard), two renderings not
  needed since there is no source equation to transcribe — but assert it against T25's own function
  applied to the gap moments, computed independently.
- **The independent process anchor carries over (T26 Leg B).** The simulated `V` still satisfies the
  numéraire-relative martingale identity; the collateralized exposure is a functional of the same
  validated paths, so no new independence anchor is needed — but **state that the gap `V(t) − V(t−δ)`
  reuses the Leg-B-validated process**, not a fresh unproven one.
- **Requirement oracles (the finance).** `collateralized CVA ≤ uncollateralized CVA` for every
  parameter (collateral cannot increase exposure); **monotone increasing in `δ`** (staler collateral ⇒
  more exposure); `= 0` at `δ = 0`. Each derived from the finance, surviving a transcription error.
- **Degenerate-fixture guards (mandatory, §4).** `δ > 0` **materially** and the mark **volatile**, else
  the gap ≈ 0 and collateral is invisible (assert the non-degeneracy). The uncollateralized and
  collateralized values must be **materially different** in the main fixture, else the reductions are
  the only thing tested and the middle is a hole. **Mutation table:** use `V(t)` instead of the gap
  `V(t) − V(t−δ)` (→ collapses to uncollateralized, the δ-monotonicity and the δ→0 reduction kill it);
  post collateral at `t` instead of `t−δ` (→ exposure 0 always, the non-degenerate companion kills it);
  drop the `⁺`. Each must die.

## 3. Process
Branch `tranche/29-collateral-csa`. `test` (RED: the two reductions + the Gaussian-gap closed form + the
δ-monotonicity + the mutation table) → `feat` → `docs` → `chore(release): v0.35.0`. **Open the draft PR
early.** `source:` `CREPEY-2026`. The collateralized-exposure module `wants` a `crosscheck:` (QuantLib
has collateralized CVA, or `none — <reason>`, §8). `references` gate green (no new key). **Raise an
`Ask:`** if the zero-threshold/zero-MTA simplification hides a term the frame needs — do not silently
add thresholds.

## 4. What this tranche is NOT (scope fence)
No **thresholds / MTA** (collateral as a nonlinear function of `V`), no **initial margin** and hence no
**MVA**, no segregation/rehypothecation — all `to_go_deeper`. No **FVA-with-collateral** (the funding
benefit of posted VM, ColVA) — the T28 FVA stays on the uncollateralized mark; the collateral/funding
interaction is deferred. No **wrong-way / gap-risk correlation** between exposure and default. No KVA
(`out_of_method`). **This is the last build of the arc**: `CP_xva_spine.md` closes it next, writing the
honest completion ratio and the deferred ledger, and re-audit #2 (§9) fires on the arc close.

---

## 5. Cowork's ruling on T29 — PR #40 (§10 step 4, 2026-11-09)

Accepted — v0.35.0, 990 green, nine gates, six mutants (all dead after the fix), draft PR. The design
held: collateral enters only through the exposure, the gap feeds T25/T27's `credit_adjustment`
unchanged, and the `agreement=None` reduction is structural (the input object returns), so T26/T27's
numbers provably cannot move. **The XVA spine arc's builds are complete.**

**The directional hole → §4 gains a clause.** Reversing the gap to the *wrong sign* survived all 986
tests because the fixture's mark was driftless, so `V(t)−V(t−δ)` is symmetric about zero and the
reversed gap has the identical distribution — a mean-based oracle cannot see direction. This is a
**new form of the degenerate-fixture trap**: not a special value (`τ=1`, a zero) and not uniformity
across a slice, but a **distributional symmetry the mutation respects**. The guard is "break the
symmetry the mutation exploits" — a material drift (`E[gap⁺] ≠ E[(−gap)⁺]`: 0% driftless, 572% at
60/yr). Seventh appearance; §4 edited with the T29 cite. Your close (drift asserted material against
the gap's own sd + a companion asserting the band *rejects* the reversed value) is the right fix, not
just a new fixture.

**The `√δ` oracle — ratified, and the correction made it stronger.** Your `0.35×` was wrong (true
0.387), and replacing it with the scaling law — halving δ multiplies the CVA by `1/√2` (measured
0.772/0.710/0.706) — catches a **wrong power of δ** that monotonicity alone cannot. §10's
report-what-execution-shows working on your own assertion, and the result is a better oracle.

**The floor gate caught its author — exactly the point of T26b.** Staging the stamp with
`TEST_FLOOR = 990` against a 986-test tree failed the floor gate on both runners, correctly. The gate
built to floor the tests floored its own builder; fixed in its own `test(...)` commit. No action — the
mechanism did its job, and it is a journal beat.

**Next — the arc closes.** After #40 merges I write `CP_xva_spine.md` (the honest completion ratio and
the deferred ledger — `to_go_deeper`/`out_of_method`) and **re-audit #2 fires** (§9, arc close *and*
~eleven tranches since re-audit #1 at T22). Settled with Bernardo, not opened here.
