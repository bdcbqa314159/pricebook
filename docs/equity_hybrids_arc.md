# The equity & hybrids arc — project four (`BROCKHAUS-2016`)

**Proposed by Cowork for Bernardo, 2026-11-10 (arc direction his pick, "dig in and take off a
consequent main spine").** The first *equity* vertical, and the one that **ties the whole library
together**: equity on its own closed forms first, then equity coupled to the rates, credit, and
counterparty machinery we already built. Same discipline as the three closed arcs — build the
oracle-able spine, ledger the rest, close with a checkpoint.

## The thesis, and why this book is the capstone
Brockhaus's own order *is* the spine: equity vanilla → the implied-vol surface → vol models → fixed
income → **hybrids** → credit → defaultable equity → **equity CVA**. The last third is the point: the
book's ch 10 (HW/CIR/HJM), ch 12 (credit), ch 14 (CVA/FVA/WWR) and ch 17 (MC) are things **we already
hold**, so the equity arc spends most of its new effort on the equity core and then *connects* it — the
hybrids are where three verticals meet. Method-fit is strong: ch 18 ("Gauss") is a closed-form appendix
(Black-Scholes, barrier, useful integrals) that hands us anchors directly.

## What the equity spine reuses (the map onto the tree)
| Brockhaus | we already have | reused by |
|---|---|---|
| ch 17 Monte Carlo (RNG, Brownian bridge, variance reduction, Greeks) | the Jäckel arc (T15–T22) | every MC equity tranche |
| ch 10 Fixed income (Hull-White, CIR) | `models/hull_white`, the LMM | the equity-rate hybrid |
| ch 12 Credit, ch 13.1 reduced-form (Gaussian hazard) | the survival curve (T25) | defaultable equity |
| ch 14 CVA / FVA / WWR | the XVA spine (T25–T29) | equity CVA |
| ch 18.2 Black-Scholes | `models/black` | the equity vanilla, the local-vol anchor |

So the arc is light on re-building and heavy on the genuinely new equity objects + the seams.

## The main spine — the proposed tranche sequence
Each unit names its product, its oracle anchor, and what it pulls. Cowork writes the per-tranche spec
before each build (§10); this is the queue, not the specs.

- **T30 — the equity underlying: forward, dividends, and the Black-Scholes vanilla.** The equity base —
  spot, repo, discrete/proportional/deterministic dividends (ch 2, 5.1–5.3), and the European option
  (ch 18.2), reusing `models/black` on the equity forward. **Oracle:** closed-form Black-Scholes
  (ch 18.2, exact); put-call parity; the dividend forward `F = (S − PV(divs))·e^{(r−q)T}` reductions.
  The foundation everything equity pulls.
- **T31 — the variance swap: model-independent static replication.** The log-contract replication
  (ch 3.7, 5.8) — a *beautiful exact* result (the fair variance is a strike-continuum of vanillas),
  model-free. **Oracle:** the replication identity (exact, closed form) + reduction to the
  flat-vol Black-Scholes case; a self-contained "product worth having" that barely needs a model.
- **T32 — the implied-vol surface and Dupire local volatility.** The skew parametrization (ch 4) and
  Dupire's local vol from the surface (ch 6.1). **Oracle:** the **consistency anchor** — local vol
  *reprices the vanillas it was calibrated to* (a round-trip, non-telescoping when the local-vol PDE/MC
  is a different engine than the surface); the Dupire formula transcribed pointwise. The equity
  modelling foundation.
- **T33 — stochastic volatility (Heston) by the Fourier / affine method.** The characteristic-function
  vanilla (ch 6.2 + ch 16.2). **Oracle:** the affine closed-form call via Fourier inversion, anchored
  by the `σ_vol → 0` reduction to Black-Scholes and by an independent MC (the Jäckel engine) within SE.
- **T34 — the equity-interest-rate hybrid.** Equity under a stochastic short rate (ch 11), reusing
  `hull_white`. **Oracle:** the constant-equity-vol closed form (ch 11.1, a Gaussian-rates/lognormal-
  equity hybrid has a closed-form vanilla) + the `rate-vol → 0` reduction to the deterministic-rate
  equity vanilla (T30). The **first true hybrid** — equity meets the rates vertical.
- **T35 — defaultable equity (equity-credit).** Reduced-form defaultable equity with a Gaussian/CIR
  hazard (ch 13.1), reusing the survival curve. **Oracle:** the reduction to the default-free equity
  vanilla as hazard → 0; the defaultable-equity closed form (ch 13.1, Fig 13.1's structure) pinned by
  a requirement identity. Equity meets the credit vertical.
- **T36 — equity CVA (the capstone).** CVA on an equity position (ch 14.2, 14.5), reusing the T25–T29
  XVA machinery, with wrong-way risk (ch 14.3) as the equity/default correlation the elementary XVA
  spine deferred. **Oracle:** reduction to the T25/T27 uncollateralized CVA at zero correlation;
  the WWR sign/monotonicity. Project three and project four meet here.
- **Checkpoint** `CP_equity_hybrids.md` closes the arc; its close fires re-audit #3 (§9).

## The completion ratio — honest (proposed)
Built as the spine: the equity underlying + vanilla, the variance swap, local vol, stochastic vol
(Heston/affine), and the three hybrids (equity-rate, defaultable equity, equity CVA). Deferred to
`to_go_deeper` (oracle-able, unpulled): the **exotic equity zoo** (barriers/cliquets/asians/lookbacks/
autocallables, ch 3 — pulled as MC consumers when interest wants one), implied-vol *dynamics* (ch 7),
**correlation & copulas / multi-asset** (ch 8–9, a whole sub-arc), FX & quanto (ch 15), local-
stochastic vol and jump-diffusion (ch 6.3–6.4), portfolio credit (ch 12.4), the KVA/wrong-way-risk
frontier beyond the elementary. Nothing here is `out_of_method` — equity is closed-form country.

## Decision — the full spine is committed (Bernardo, 2026-11-10)
Bernardo chose the **full seven-tranche spine** (T30 → T36 + checkpoint) over a trimmed version: the
complete capstone, equity core *and* all three hybrids. The queue is T30 → T31 → T32 → T33 → **T33b (the (product,model) registry, ruled at T33's
review — infrastructure before the hybrids multiply the product×model pricers)** → T34 → T35 → **T36a (independent-default equity CVA — the cross-arc seam, built v0.43.0)** →
**T36b (wrong-way risk — the equity/default correlation; split from T36 at build, ruled 2026-11-13)** →
`CP_equity_hybrids.md`; Cowork writes each per-tranche spec before its build (§10). T30 is
specced (`docs/tranches/30_equity_underlying.md`); the rest follow in turn.
