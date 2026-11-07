# Tranche 22 — Greeks by simulation: the last Jäckel unit, and the first L5 risk module

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.27.0`.
**Source:** `JAECKEL-2002` ch 11 — FD with path recycling (§11.3), pathwise differentiation (§11.5),
the likelihood-ratio method (§11.6), and the LR Vega/Rho appendix (§11.9). **Last of the Jäckel arc.**
Closes ch 11 (the ninth chapter), **opens the empty L5 risk layer**, and closes BERHAN's §1.3 route.
On landing, the §9 whole-tree re-audit fires at the Jäckel checkpoint.

## 0. The point — make the price *move*, anchored on Greeks we already own in closed form
Every model so far prices; none reports a sensitivity. Greeks by simulation give delta/gamma/vega from
the same paths, and — the anchor — **we have Black's analytic Greeks** (T9), so an MC delta/gamma/vega
must converge to them. This is the price-convergence discipline applied to the derivative of the price.

## 1. The methods (from Jäckel) and the anchor (ours)
- **FD with path recycling (§11.3):** bump `S₀`, reprice on the *same* paths (common random numbers),
  difference. The seed-sharing is what makes the difference low-variance — a direct use of the §3
  determinism invariant.
- **Pathwise (§11.5):** differentiate the payoff along the path, `E[∂payoff/∂S₀]` — valid only for a
  Lipschitz payoff.
- **Likelihood ratio (§11.6):** differentiate the density, not the payoff — a score-function weight,
  valid for a *discontinuous* payoff where pathwise is not.
- **Anchor:** all three converge to **Black's closed-form delta/gamma/vega** (T9), within the Greek
  estimate's own standard error (invariant 7 extends to a Greek — it is an estimate).

## 2. Pieces and layers — this OPENS L5
```
L5  risk/sensitivities.py — the first L5 module: a sensitivity computed ON the L4 engine. Per §2, L5
                            sits on the engine + a Priceable protocol. Whether L5 needs a NEW Priceable
                            protocol or bumps an input and calls the existing registry `price()` is a
                            build question — settle by what it reads, raise if unobvious (rule-of-two:
                            the FD method prices many products via `price()`, so the "thing that can be
                            bumped and repriced" may already be the registry interface).
L4/L0  the pathwise and LR estimators attach to the MC engine (they differentiate its payoff/density);
                            the FD method is pure bump-and-reprice over the existing engine.
```
This is a milestone: the layer box has had L5 empty since T1. It fills here, and the charter's "L5 on
the engine + a Priceable protocol" stops being an aspiration.

## 3. The oracle (RULED unless TO MEASURE)
- **RULED — exact anchor: MC delta/gamma/vega converge to Black's analytic Greeks** (T9) within the
  Greek's measured SE. The closed form we own is the judge, as for the price.
- **RULED — cross-method agreement: pathwise, LR and FD-path-recycling agree** on a smooth payoff
  (three independent estimators, different code each side — a cross-method oracle). A wrong one
  diverges from the other two.
- **RULED — the reduction+companion that is the whole point of having two methods: on a Lipschitz
  payoff (a call) pathwise is valid and matches; on a DISCONTINUOUS payoff (a digital) pathwise is
  invalid and LR is the one that holds.** Assert both — pathwise agreeing on the call is vacuous
  without the digital showing why LR exists.
- **RULED — the −½ rate on the Greek estimate**, and the null/averaging discipline (§4) where a
  sensitivity's expected value is zero (e.g. a Greek that must vanish by symmetry — average it).
- **RULED — determinism; failure-is-a-value** (a Greek that cannot be formed → a value, not a NaN),
  and the Greek carries its SE (invariant 7).
- **TO MEASURE — variance ranking:** report which method is sharpest for delta vs gamma vs vega (the
  §11.7 comparison), measured, not asserted from the book's figures (no number off a chart, §4).

## 4. Stock & process
Mine `_stock/.../models/mc_greeks*.py` and `risk/pathwise_greeks.py` for shape and the migration
edge cases; the pathwise/LR formulas are read from **Jäckel §11.5/§11.6/§11.9**, not the stock. The LR
Vega/Rho formulas (§11.9) are load-bearing — transcribe pointwise, and check whether Jäckel prints
them in full or leaves them to the reader (the Acklam/`(1−ρ²)` pattern — derive-and-assert if unprinted).
Branch `tranche/22-greeks`. `test` (RED pre-registered: MC delta vs Black delta) → `feat` → `docs` →
`chore(release): v0.27.0`. Mutation on the pathwise derivative and the LR score; the digital-payoff
companion must fail a pathwise estimate. `crosscheck: quantlib (MC Greeks)` on the L5 module.
`source:` cites `JAECKEL-2002` ch 11; names Black's analytic Greeks as the anchor. **Open as a DRAFT
PR** (§10 step 3).

## 5. What this closes
The nine-chapter Jäckel spine (2,3,4,7,8,9,10,11,12) is complete. Write `CP_jaeckel_arc.md`, finalise
the `to_go_deeper` Jäckel section, and **run the first §9 whole-tree re-audit** (T1–T22, independent,
via the `analysis/` pattern) — recording it in `docs/audit_log.md`. Then the next arc is Bernardo's
interest-pick (he has a source in mind).
