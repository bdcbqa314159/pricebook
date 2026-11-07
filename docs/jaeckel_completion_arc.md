# The Jäckel completion arc — closing the book's working spine

**Decided by Cowork with Bernardo, 2026-11-05.** How we close `JAECKEL-2002` as a satisfying arc —
the same shape as inflation: complete the book's **working spine**, ledger the rest as `to_go_deeper`
debt, write a checkpoint, then move on. "Closing a chapter" means its spine is built and oracled, NOT
every sub-section — exactly as BERHAN "closed" at ch 1–5 with risk/HJM/smile deferred.

## The book's thesis, and why these chapters
Jäckel's argument is not *simulate* but **simulate well**: generate good randomness (7,8,9) → drive
processes (2,3,4) → **reduce the variance** (10) → get Greeks (11) → apply to the market model (12) →
cross-check with trees (13). T15–T18 built the generation core, the process core, and the whole
application (12). The arc makes the MC *sharp* — smaller standard error at the same path count, so
every banded oracle resolves a finer defect. **Note the correction T19 forced:** the Bermudan's 15.75%
sandwich is **structural** (perfect-foresight bias + policy suboptimality — its width is invariant as
the SE falls), *not* sampling error, so variance reduction sharpens each price but does not narrow that
bracket; tightening it needs a better exercise policy or the dual bound (`to_go_deeper`).

## Chapter map (at arc start, v0.21.0)

| ch | status after T15–T18 | arc target |
|----|----------------------|-----------|
| 1 Introduction | n/a (prose) | — |
| 2 Mathematics behind MC | **core built** (estimator, SE, CLT) | — |
| 3 Stochastic dynamics | **core built** (BM, lognormal) | — |
| 4 Process-driven sampling | **core built** (predictor-corrector) | — |
| 5 Correlation & co-movement (copulæ) | partial | → `to_go_deeper` |
| 6 Salvaging a correlation matrix | not built | → `to_go_deeper` |
| 7 Pseudo-random numbers | **settled** (PCG64 ruled) | — |
| 8 Low-discrepancy (Sobol) | not built | **T20** |
| 9 Non-uniform variates | partial (Box-Muller; inversion deferred) | **T20** (inversion) |
| 10 Variance reduction | not built | **T19 + T21** |
| 11 Greeks | not built | **T22** |
| 12 BGM/J | **fully built** | — |
| 13 Non-recombining trees | not built | → `to_go_deeper` |
| 14 Miscellanea | n/a (engineering) | — |

## The arc — four tranches
- **T19 — Variance reduction I: antithetic (10.1) + control variates (10.3).** *(Built, v0.22.0.)*
  The control must have an **exact** mean — the specced Rebonato control was an approximation and would
  have injected bias; replaced by the numéraire-relative annuity (exact by the martingale property),
  ρ=0.74, 54% variance reduction, zero bias. Oracle: unbiasedness (exact, against our closed forms) +
  measured reduction.
- **T20 — Inverse-CDF + Sobol (8).** The deferred inversion finally built (Sobol needs a monotone
  transform, not Box-Muller). **Not a correctness dependency on Acklam:** `Φ⁻¹` is built primarily by
  inverting our own exact `norm_cdf` (a guarded root-find — self-contained, exact, monotone), with
  **Acklam's rational approximation as the fast path** — recommended in Jäckel **§9.3** ("Normal
  variates") and ch 2, never printed there; the note itself is now `in-hand` in
  `_stock/NewDocuments/` alongside the other held sources. *(Corrected at commit: the draft cited
  §9.1, which is "Inversion of the cumulative probability function" — the general inversion method,
  relevant but a different thing from where Acklam is named. `references.md` already said §9.3.)*
  the two **cross-checked** against each other (different code, Acklam's coefficients untrusted until
  they match the root-find) and both against the round-trip. Register `ACKLAM-2000` `in-hand` when
  specced. Oracle: beats the −½ rate. **TODO (queued, convenience only):** produce a gitignored `.md`
  reading-extract of the Acklam note under `_stock/NewDocuments/` — clearly marked *derived, not the
  authority*; Code still transcribes from the `.htm` in two renderings (the shared-rendering rule).
  - **T20a — the inverse-CDF (Acklam + Halley on our `erfc`).** *(Built, v0.23.0.)* One production
    function; the root-find is the oracle's independent cross-check, not a shipped second path (§6).
  - **T20b — Sobol / low-discrepancy (§8).** *(Next.)* Oracle: beats the −½ rate; the inversion feeds it.
  - **T20c — fix `norm_cdf`'s lower-tail cancellation (OI-3).** Move it to `0.5·erfc(−x/√2)` — §4
    pure-refactor (identity where the forms agree, to ULP) + a relative-accuracy oracle vs
    `scipy.special.ndtri`; retire the T20a duplicate. Pulled by T20a's finding; before T21.
- **T21 — Path construction / the Brownian bridge (10.8).** Path-dependent payoffs; the classic
  partner to Sobol (front-loads variance into the low-dimension coordinates).
- **T22 — Greeks (11): pathwise + likelihood ratio.** Opens L5 risk; closes BERHAN's §1.3 route too.

## The completion ratio — honest
At arc end: chapters **2, 3, 4, 7, 8, 9, 10, 11, 12 — nine of fourteen — built and oracled as the
working spine.** Deferred wholesale: 1 (intro), 5 (copulæ), 6 (salvaging correlation), 13 (trees), 14
(engineering). Plus per-chapter debt *inside* the nine (Feynman-Kac/Moore-Penrose in 2; Bessel/CEV/
displaced in 3; Milstein/convergence theory in 4; copula draws in 9; stratified/importance/
moment-matching/Latin-hypercube in 10). All of it in `docs/to_go_deeper.md` under `JAECKEL-2002`.

A checkpoint (`CP_jaeckel_arc.md`) closes the arc when T22 lands, as `CP_bgmj_arc` closed T16–18.
