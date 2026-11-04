# Tranche 15 — the minimal Monte Carlo spine, judged by a closed form we already own

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.18.0`.
**Source:** `JAECKEL-2002` (`in-hand` → `worked-from` on landing) ch 2, 9, 14 for the method; the
**oracle is this library's own `black_price`** (T9), not Jäckel. **Chosen by Bernardo:** project two,
after BERHAN closed — the capability BERHAN's §2.1–2.4 route (`to_go_deeper`) was waiting on.

## 0. The point — the T1 of Monte Carlo: the smallest engine whose output we can already judge

This is the root of the whole MC dependency graph. It builds nothing clever — a seeded generator, a
normal variate, a naive path, an estimator with a standard error — and it earns its place by
**repricing a product we have in closed form and converging to that price**. Every later technique
(variance reduction, Sobol, the bridge, Greeks, BGM/J) is defined as an *improvement on this baseline*
and cannot be oracled until the baseline exists and is measured. So this lands first, regardless of
chapter order.

**What it is NOT:** not a general process library, not variance reduction, not QMC, not Greeks. Those
arrive when a product pulls them (§1, §6 — nothing speculative). The stock has all of them
(`mc_base_inventory.md`); none is purified here.

## 1. The method (transcribed from Jäckel) vs the answer (our closed form)

The division that governs every MC tranche:
- **Algorithm from Jäckel, pointwise (§4):** the normal-variate construction (ch 9 — inverse-CDF
  `z = Φ⁻¹(u)`), and the estimator/standard-error definitions (ch 2, 14). Read `-raw` + `-layout`.
- **Correctness from us:** price a European call under GBM by simulating `S_T = S₀·exp((r−½σ²)T +
  σ√T·z)` and discounting the payoff mean; it must converge to `black_price(forward, strike,
  σ²T, kind)` — the T9 closed form — within the sampling error.

## 2. Pieces and layers

```
L0  foundation/rng.py        — a DETERMINISTIC seeded uniform source: seed in → reproducible uniforms.
                               RULED (§3): numpy `Generator(PCG64(seed))`, wrapped. Purify the stock's
                               statistics/rng.py SHAPE only.
L0  foundation/variates.py   — standard-normal variates by inverse-CDF Φ⁻¹ (reuse foundation's Φ if
                               present; else transcribe from Jäckel ch 9). Finance-free. THIS is the
                               §4-transcribed, pointwise-oracled piece — not the uniform bits.
L0  foundation/mc_estimator.py — the sample mean + standard error (σ̂/√N) of a vector of payoffs.
                               Finance-free statistics.
L4  engine/mc_european.py    — the simulating pricer: GBM terminal draw, payoff, discount, estimate;
                               returns a price AND its standard error. Reads model + product, like
                               every L4 engine.
```

The RNG/variate/estimator stack is finance-free → L0 (beside `quadrature.py`). The simulating pricer
reads a product and a model → L4. This split is **ours**, re-derived from the charter; the stock's
`models/mc_*` placement carries no authority (§5).

## 3. Rulings this unit forces (the methods-material firsts) — ALL DECIDED, nothing left open

- **The RNG is RULED: numpy `Generator(PCG64(seed))`, wrapped.** Not left to the build. Reasoning:
  the generator must be deterministic from an explicit seed, read no global/ambient state, and —
  the binding constraint — produce **byte-identical streams on ubuntu and windows** so the §9
  same-process statelessness check is meaningful. A seeded `PCG64` is a pure function of its seed
  (§3.1/§3.5), numpy guarantees its stream is reproducible across platforms, and numpy is already a
  dependency. We do **not** hand-roll a generator — that is surface §6 forbids; reproducible normals
  are the real problem, a novel RNG is not. The wrapper's whole job is to ensure **no global/default
  RNG is ever touched** — the seed enters as explicit config (§3.5), threaded to the engine.
- **What §4-transcription covers, and what it does not.** PCG64 postdates Jäckel (2002) and is not in
  it, so its uniform bits are **not** transcribed — they are trusted (a documented, version-pinned
  library stream) and **pinned by a cross-platform reproducibility regression** (same seed ⇒
  byte-identical uniforms on both OSes). The transcribed, pointwise-oracled piece is the **normal
  variate** — the inverse-CDF `Φ⁻¹` from Jäckel ch 9 applied to those uniforms — and, later, Sobol
  direction numbers. (Do not use numpy's ziggurat `standard_normal`: the oracle must rest on *our*
  transcribed inversion, not a library's internal algorithm.)
- **Determinism / §3 preserved.** Same seed ⇒ same uniforms ⇒ same normals ⇒ same price, in-process,
  so referential transparency (§3.1) and the §9 byte-identical statelessness check both hold.
- **The oracle is statistical — a new shape, ruled honest up front.** An MC price never equals the
  closed form; the oracle is **|MC − Black| < k·SE** with the SE *measured* (σ̂/√N), not a magic
  tolerance (the "tolerance is a finding, not a knob" rule; the T11 noise-floor philosophy). `k` is
  stated (e.g. 3–4) and justified as a confidence level, and the **seed is fixed** so the test is
  deterministic — a fixed-seed MC is a pure function, so this is a hard assertion, not a flaky one.
- **No number fitted to a figure** — Jäckel's convergence diagrams are read for *shape*, never pixels.

## 4. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **convergence to Black within standard error:** at a fixed seed and large N, `|MC − black_price| < k·SE` | **closed form (ours) — the anchor** | a wrong drift/vol in the terminal draw, a discounting error, a payoff error |
| 2 | **the rate — a requirement oracle, stronger than any single N:** halving the error needs ~4× the paths; fit `log SE` vs `log N`, slope ≈ **−½** to tolerance | **structural** | a biased estimator passes oracle 1 at one N but has the wrong rate; this catches it |
| 3 | **unbiasedness across seeds:** the mean of the MC estimate over many seeds is within SE/√(n_seeds) of Black — the estimator has no systematic offset | structural | a construction bias (e.g. `exp` of a mis-centred normal) invisible at one seed |
| 4 | **determinism / statelessness:** same seed ⇒ byte-identical paths and price, in-process (§9); a second pricing never touches global RNG state; the uniform stream is byte-identical ubuntu vs windows | structural | an ambient-entropy leak (the §3 violation), or a non-portable generator |
| 5 | **the variate is standard normal:** `Φ⁻¹(u)` draws match N(0,1) in mean/variance to SE, AND a known-quantile input maps to the known z (a pointwise transcription check, not only a moment check) | 1 — pointwise + requirement | a wrong inverse-CDF (the properties-don't-replace-transcription rule, §4) |
| 6 | **failure is a value:** `n_paths ≤ 0` / a non-finite draw → `PricingFailure`, never a `NaN` price (invariant 4; a stock migration-hardening case — `test_mcengine_n_paths_validation`) | structural | the estimator swallowing a bad input |

Oracle 1 is the anchor (our own closed form); oracle 2 is the one that makes "it's Monte Carlo" a
*testable* claim rather than an excuse for a loose number. Fold the stock's migration-hardening cases
(RNG seeding reproducibility, `n_paths` validation) into oracles 4 and 6 as pre-registered REDs.

## 5. Stock
Purify the **shape** of `statistics/rng.py` (seed→draws) and `numerical/_mc.py` (estimator/SE),
re-layered to L0; mine `tests/test_mc_rng_seeding.py` and `test_l2_t4_mcengine_n_paths_validation.py`
for **edge cases** to pre-register (see `docs/stock/mc_base_inventory.md`). Read the method from
**Jäckel ch 2/9/14**; the GBM draw and the oracle are ours. Nothing imported from `_stock/`.

## 6. Process
Branch `tranche/15-mc-spine`. `test` (RED **pre-registered**) → `feat` → `docs` →
`chore(release): v0.18.0`. Mutation on the drift/vol/discount of the terminal draw and on the SE
formula; the degenerate-fixture guard applies (do **not** oracle at `σ=0` or `T=0`, where the draw is
deterministic and the whole method is invisible — the fifth-time lesson). `crosscheck: quantlib
(GBM European MC) — deferred` on `engine/mc_european.py`; the L0 numerics carry `crosscheck: none —
finance-free numerics`. `source:` cites `JAECKEL-2002` for the method and names `black_price` as the
oracle. **Open as a DRAFT PR** (§10 step 3).

## 7. What this unlocks (not built here)
Once the spine converges to Black: variance reduction (ch 10, with Black as a **control variate**),
Sobol (ch 8, oracle: beats the −½ rate), the Brownian bridge (ch 10.8, path-dependent payoffs),
Greeks (ch 11 — and BERHAN's §1.3 risk route), BGM/J (ch 12, late). Each is its own tranche, pulled
by a product, judged against a closed form or this baseline.
