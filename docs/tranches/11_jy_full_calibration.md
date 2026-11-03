# Tranche 11 — the full six-parameter JY calibration, and the two reasons it fails (§5.3.2)

**Status:** RE-SPEC by Cowork after Code verified the first spec's empirical claims and three were
false (thread 0064). Awaiting build. **Target:** `v0.13.0`. **Source:** `BERHAN-2012` §5.3.2.
**Chosen by Bernardo** (2026-11-03) as the honest capstone of the JY arc.

## 0. A negative-result tranche — and the corrected finding is STRONGER than the thesis

BERHAN §5.3.2 fits all six `{κr, σr, σI, ρnr, ρrI, ρnI}` to the YoY-floor **surface** and reports it
fails, blaming the **strike skew** ("unfeasible for non-ATM"). Code verified against the code and
the thesis is **incomplete**: the six-parameter fit is **already underdetermined at ATM, before any
skew enters**. There are **two distinct failures**, and T11 asserts both:

1. **Underdetermination** — three parameters are not identifiable from YoY floors *at any strike*,
   provable from the formula (§1). The thesis does not name this.
2. **Misspecification (the flat smile)** — JY is Gaussian, so it produces no skew (§1). This is what
   the thesis names.

`(κn, σn)` arrive **fixed from §5.3.1** (T10); T6 fitted `(σr, σI)`; T11 frees all six and finds the
limit. The capstone is the honest, explained reproduction — reproduce the structure *including*
where the source's own diagnosis stops short.

## 1. The two structural facts, both read off the code (not asserted blind this time)

**(a) Underdetermination.** In `yoy_variance`, the index correlations enter **only as products with
`σI`** — `ρnI` via `2·ρnI·(σn·σI/κn)·(span−βn)`, `ρrI` via `2·ρrI·(σr·σI/κr)·(span−βr)` — and `ρnI`
appears **nowhere in the convexity `C`**. So a YoY floor price sees `(ρnI, ρrI)` only through a
single pooled, `σI`-scaled combination whose per-maturity weights are near-proportional; a maturity
strip barely separates them. Multi-start on **perfect ATM synthetic data** (thread 0064): `(κr, σr,
ρnr)` recover exactly; `(σI, ρrI, ρnI)` do not — `ρnI` wanders +0.15 → −0.55, every fit at a perfect
residual. Three of six identifiable, three not, by construction.

**(b) Flat smile.** JY is Gaussian; `V²` (eq 2.6.4) contains **no strike**, so the implied YoY-cap
vol is constant across strikes (verified: `σI = 0.0195048871` at every κ ∈ {0,1,2,3,4}%). The model
has no skew to offer, so fitting a skewed market leaves irreducible OTM error.

## 2. Pieces and layer

```
L4  engine/jy_calibration.py — EXTEND T6's fit to free all six params; report the identifiable
                               subset and the degeneracy. (κn, σn) fixed inputs. No new type.
```

Engine-inverting → **L4**. The free set is `(HullWhiteParams_real, index_vol, JYCorrelations)` — the
model's own shape (T6's bundle widened), ≤5 fields.

## 3. The oracle (CORRECTED — no condition-number threshold anywhere)

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **recover the IDENTIFIABLE subset `(κr, σr, ρnr)` exactly** from synthetic ATM data, multi-start — all starts agree on these three to tolerance | 2 — round trip | the solver works on what *is* determined; leans on T4/T5's forward (state it) |
| 2 | **the discriminator — the multi-start SPREAD, behavioural and step-/platform-independent:** on perfect data `(σI, ρrI, ρnI)` do **not** converge (spread above a floor) while `(κr, σr, ρnr)` do (spread ≈ 0), and the reason is asserted structurally — `(ρnI, ρrI)` enter only pooled with `σI`, `ρnI` absent from `C` | **1 — structural** | this IS the underdetermination; a fit that reported all six converged would be hiding the valley. **No condition number** — the smallest singular value is below the finite-difference noise floor (swings three orders with the step; flaky across OSes). A singular-value *ratio* `s₅/s₁ ≈ 3.4e-6` is the only stable number and is **documented, not gated** |
| 3 | **flat smile (structural guard):** JY's implied YoY-cap vol is constant across strikes to tolerance | 1 — structural | near-tautological for our code (`V²` has no strike) — kept as a documentation + regression guard, its tautology noted, NOT a discriminator |
| 4 | **the two failures are distinct and both present:** underdetermination (oracle 2) holds on **ATM-only** data, *before* any skew; the smile failure (oracle 3) is a separate axis — assert each in isolation, and that the thesis names only the second | structural | conflating them (e.g. "it fails because of the skew") would reproduce the thesis's incomplete diagnosis |
| 5 | failure-is-a-value: non-convergence → `PricingFailure`; the identifiable sub-fit still returns a usable result | structural | the solver seam's `converged` flag |

## 4. Design decisions (ruled)

- **No condition-number gate — ruled out, not deferred.** A finite-difference Jacobian condition
  number here is noise (Code measured it swinging 2.5e7 → 2.1e10 across step sizes, differing again
  under central differences). An oracle must be measurable above its own numerical noise; this one
  is not. The **behavioural multi-start spread** is the identifiability oracle, and it is
  step-independent and the same on both OSes.
- **The finding beyond the thesis is the deliverable** — stated plainly in the module, the
  `CHANGELOG`, and the coverage map: "the six-parameter JY fit fails for two reasons — three
  parameters are underdetermined at ATM (provable from `V²`'s `σI·ρ` pooling and `ρnI`'s absence
  from `C`), and the Gaussian flat smile cannot fit a skew; §5.3.2 names only the second."
- **`(κn, σn)` fixed from §5.3.1, not re-fitted** (the don't-re-fit-the-prior-stage rule).
- **Cross-check disposition unchanged** — the calibrator keeps T6's `crosscheck:` (QuantLib has no
  JY-to-floor calibrator; the forward map is tagged at `capfloor.py`).

## 5. Stock
Mine `jy_calibrate` for the surface-fit objective's shape only; the underdetermination and the flat
smile are read from **the formula and the source**, not from the stock's code.

## 6. Process
Branch `tranche/11-jy-full-calibration` (the equivalent-mutant charter commit is already there).
`test` (RED, **pre-registered** — the one thing missed in T10a) → `feat` → `docs` → `chore(release):
v0.13.0`. Mutation on the six-param objective; oracle 3 uses a genuinely **skewed** surface fixture
(an ATM-only one hides the smile failure — the degenerate-fixture lesson). `source:` cites
`BERHAN-2012` §5.3.2.
