# Checkpoint — the Jäckel arc: Monte Carlo, nine chapters of the book's spine (T15–T22)

**Date:** 2026-11-07 · **By:** Cowork (step 4) · **Baseline:** v0.27.0, PR #31 · **Source:**
`JAECKEL-2002` (`worked-from`). Third source of the reset repo (after `BERHAN-2012` and the BGM/J
application within Jäckel ch 12). Closes the completion arc of `docs/jaeckel_completion_arc.md`.

## 0. What is built — the book's working spine, nine of fourteen chapters

| ch | capability | tranche |
|---|---|---|
| 2 | the estimator, standard error, CLT / −½ rate | T15 |
| 3 | Brownian motion, lognormal processes | T15/T16 |
| 4 | the predictor-corrector evolution scheme | T16 |
| 7 | the pseudo-random generator (PCG64, ruled) | T15 |
| 9 | Box-Muller + the inverse-CDF (Acklam + Halley on our `erfc`) | T15, T20a |
| 8 | Sobol / low-discrepancy | T20b |
| 10 | variance reduction: antithetic, control variates, the Brownian bridge | T19, T21 |
| 11 | Greeks: FD-path-recycling, pathwise, likelihood-ratio | T22 |
| 12 | BGM/J: the LMM, the swaptions, the Bermudan | T16–T18 (BGM/J arc) |

Plus **T20c**, a correctness fix (OI-3, `norm_cdf`→`erfc`) pulled by T20a.
**v0.27.0 · 818 tests · eight gates green on both OSes · ruff + pyright clean · zero suppressions.**
**L5 (risk) opened at T22** — the layer box's last empty tier now has its first module.

## 1. The spine of the method
Every MC number was anchored on a closed form the library already owned and had itself verified —
Black, then the caplet, the swaption, the Greeks as the *limit* of a finite difference. Jäckel never
supplied a reference number (his results are convergence diagrams, §14.4); the library's own earlier
work was the judge throughout. "Simulate well" — the book's actual thesis — landed as: generate
(7,8,9), drive (3,4), reduce variance (10), differentiate (11), apply to the market model (12).

## 2. Oracle shapes this arc added to §4 (the discipline grew, mostly from honest misses)
- statistical-but-deterministic (fixed-seed MC is a pure function; the −½ rate as the requirement);
- the one-seed differencing identity, and its boundary (drift only — a diffusion quantity doesn't cancel);
- bands separate SE from approximation error and assert their resolution (T17);
- two sources are independent only if they render the SAME formula — else different-order approximations (T17);
- a large table's independence is two parsers, not two eye-reads, with a property-A oracle (T20b);
- a variance reduction must be proved UNBIASED; an approximation is not a control; martingale ratios are free controls (T19);
- a reference is code too — it can be the weaker side; absolute tolerance is blind to relative tail error (T20a, four times over);
- a null needs averaging as much as an effect does (T21);
- a ledger entry's severity is swept over the defect's domain, not one sample (§8, T20c);
- a spec's cited anchor is verified to exist before handoff (§10, T22).

## 3. Source defects — Jäckel, THREE; plus a separate pattern of three unprinted formulas

**Corrected at commit.** The draft read *"Jäckel, four"* and reached it by merging two categories that
`references.md` deliberately keeps apart — counting two *unprinted* formulas as defects, bundling the
two ch-12 defects as one, and omitting the third omission. The authoritative list is in
`references.md`, and it separates them for a reason: a misprint is a defect in what the book *says*,
while an omission is a gap in what it *contains*, and the two need different responses.

**Source defects — three:**
1. **§12.8 eqs 12.38 vs 12.39** — an internal sign inconsistency; the printed condition exercises a
   deep out-of-the-money payer. Ruled by the finance, pinned by an oracle the print fails (T18).
2. **§12.8 eq 12.41** — the prescribed BFGS cannot train the stated objective (piecewise constant;
   BFGS terminates at iteration 0). A limitation of the *method*, not its printing (T18).
3. **§10.9 eq 10.76** — an appendix inconsistency, **non-load-bearing**: it reaches no implemented
   formula, and both renderings print it identically so only the internal contradiction catches it
   (T21).

**The "sends you to a formula it does not print" pattern — three instances**, its own note in the
references entry rather than a defect list: `Φ⁻¹` (§9.3, on the CD — closed by acquiring
`ACKLAM-2000`); the `(1−ρ²)` control-variate variance factor (§10.3 — derived and asserted as a
theorem); the Sobol primitive polynomials past 32 dimensions (§8.3, on the CD — recorded as a
conditional acquisition).

*(Second time an arc checkpoint has merged defect counts with omission counts; the first was
`CP_bgmj_arc.md`, corrected the same way. The tally is checkable against `references.md` and the rank
is not — state the former.)*

## 4. The two-surface split, measured over the arc

**Corrected at commit:** the draft said *"five of eight tranches (T17 ×4, T19 ×2, T20a, T22)"*, whose
sentence and parenthetical disagree (four tranches named, eight corrections counted) and whose
denominator is wrong — the arc ran **ten** tranches (T15, T16, T17, T18, T19, T20a, T20b, T20c, T21,
T22). Enumerated from the tranche docs instead:

A Cowork spec premise was corrected in **five** of the ten, **seven** corrections in total:

| tranche | premise corrected |
|---|---|
| T17 | the cross-source independence did not exist; oracle 5's one-seed instrument did not generalise |
| T18 | oracle 3's training monotonicity is not observable at this parametrisation |
| T19 | the sandwich is structural, not sampling error; the Rebonato control would inject bias |
| T21 | "beats pseudo-random" is not assertable as a fitted exponent |
| T22 | the anchor capability did not exist |

Every one made the oracle stronger and is recorded in its tranche doc. Cowork's review set each oracle
and caught weak discriminators that the build would not have. **The split works in both directions**,
which is the whole argument for it (§10) — and the direction that is easy to forget is that Cowork's
*oracle choices* held every time, including the ones only measurement made legible.

Code also began resisting charter growth (T20c, T22), which is the healthier equilibrium: §4 earning
its keep rather than accumulating a clause per tranche.

## 5. Honest scope — what the arc did NOT build (`to_go_deeper`, Jäckel)
Salvaging a non-PSD correlation matrix (ch 6); the non-recombining tree + tight dual Bermudan bound
(ch 13); copulæ (ch 5/§9.4); engineering (ch 14); Sobol past 32 dimensions (CD data, conditional on a
sampling-bound high-dim consumer); the exotic processes (ch 3), Milstein (ch 4), and the
variance-reduction techniques beyond the core (ch 10). L5 has one module (sensitivities); no
`Priceable` protocol until a second consumer.

## 6. Next
The first **§9 whole-tree re-audit (T1–T22)** runs now, independent of the builder — scoped by Cowork,
recorded in `docs/audit_log.md`. Then the next arc is Bernardo's interest-pick (a source already in
mind), decided at the boundary per §10.
