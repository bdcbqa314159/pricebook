# Stock inventory — the Monte Carlo base (for the Jäckel arc)

Everything MC the stock holds, catalogued so each Jäckel tranche can **cite what exists to purify**
rather than rediscover it. Per §5 this is **content, not structure**: we mine formulas, edge cases,
reference values and the migration's hardening lessons; the stock's own layering
(`models/mc_*`, one 477-line engine) carries **no authority** — we re-derive the layer split from the
charter (finance-free numerics → L0, the simulating pricer → L4), and **every purified piece earns
its own oracle** (the stock's tests are regression hints, never the oracle).

There are three trees in the stock, and for MC the split is **very lopsided** — worth stating plainly
so no one assumes a hardened migrated MC exists to lean on:

- **LEGACY** `_stock/quarry/pricebook/` — the pre-migration quarry, **~40 MC files**: essentially all
  the content (engine, processes, variance reduction, QMC, RNG, several Greek engines incl. AD, exact
  simulation, rough paths, Heston/SABR, a credit MC).
- **MIGRATED** `_stock/src/pricebook/` — the reset-era tree, **only two MC files** (`risk/greeks.py`,
  `risk/monte_carlo.py`). The abandoned migration **never reached the MC core**; it got as far as a
  risk/exposure MC and some Greeks.
- **PARKED-NG** `_stock/quarry/parked/ng/pricebook_ng/` — one file worth reading, `engine/swaption_mc.py`,
  for its clean `engine/` layout (closest to our L4 target).

So we are effectively **rebuilding MC fresh from the legacy content + Jäckel** — there is no hardened
migrated MC to inherit, which is cleaner (no half-migrated state to reconcile). Where the trees
disagree, none wins by location — the source (Jäckel) and our own closed forms decide (§5).

## Core numerics — L0 candidates (finance-free)

| stock file | lines | content to mine | likely home |
|---|---|---|---|
| `quarry/pricebook/statistics/rng.py` | 73 | pseudo-random (numpy Generator) + quasi-random (Sobol) normal generation, **reproducible seed management** — the shape of "seed in, deterministic draws out" | L0 `foundation/` (beside `quadrature.py`) |
| `quarry/pricebook/numerical/_qmc.py` | 233 | Sobol, Halton, Latin-hypercube, sparse grids; claims O(1/N) vs O(1/√N) | L0 — but the Sobol **direction numbers** get transcribed from Jäckel ch 8, not trusted from here |
| `quarry/pricebook/numerical/_mc.py` | 242 | core MC estimator / standard-error machinery | L0 estimator + L4 engine split |
| `quarry/pricebook/models/brownian.py` | 155 | Brownian motion / increment construction | L0 (path primitives) |

## Engine & techniques — L4 candidates (read a product + model, produce a price)

| stock file | lines | content to mine | Jäckel ch |
|---|---|---|---|
| `models/mc_engine.py` | 477 | `MCEngine` / `ProcessSpec` / `TimeGrid` / `MCResult` — the simulate-a-process-and-a-payoff frame. Mine the **shape**, not the god-object size | 2, 5 |
| `models/mc_pricer.py` | 105 | the product→price glue | — |
| `models/mc_processes.py` | 809 | SDE process library (GBM and beyond) — large; take only what a product pulls | — |
| `models/mc_variance_reduction.py` | 207 | antithetic, control variates, etc. | 10 |
| `models/mc_payoffs.py` | — | payoff library (european_call, path-dependent) | — |
| `models/mc_greeks*.py` | 467+ | pathwise, likelihood-ratio, and AD Greeks | 11 |
| `models/mc_config.py` | 245 | MC config (n_paths, seed, grid) — informs the **explicit-seed-as-config** ruling (§3.5) | 14 |
| `models/mc_diagnostics.py` | 287 | convergence diagnostics, standard-error reporting — the **statistical-oracle** machinery | 14.4 |
| `options/heston_mc.py`, `options/sabr_mc.py` | — | SV model simulation — relevant if the smile route (BERHAN `to_go_deeper` #3) is pulled | — |
| `parked/ng/pricebook_ng/engine/swaption_mc.py` | — | a swaption MC in the parked NG layering — closest to our target structure | 12 |

## The migration hardening — mine this, do not copy it

MC was **already migrated once** in the quarry, and the scars are recorded:

- `models/mc_migrate.py` (a migration shim) and `tests/test_mc_migration{,_full,_wave2,_wave3,_wave4}.py`
  — four waves of migration tests. These encode the **edge cases and bugs the previous effort hit**
  (RNG seeding reproducibility, path-discount sweeps, Greek exception-safety, `n_paths` validation,
  path-dependent discounting — see `tests/test_mc_rng_seeding.py`, `test_l2_t4_*`,
  `test_mc_greek_param_name.py`). That is exactly the **content** §5 says to mine: fold each hardening
  case into our own pre-registered oracles, so we inherit the lessons without inheriting the structure.
- `tests/test_l2_t1_1_mlmc_giles.py` — a multilevel-MC (Giles) test; a route, not a near-term unit.

## What this means for the dig

- **The minimal spine (T15)** can purify `rng.py`'s seed→draws shape and `_mc.py`'s estimator,
  re-layered to L0, oracled against our **Black** closed form (convergence within SE, 1/√N rate). The
  stock gives the shape; the oracle is ours.
- **The RNG is ruled (T15 §3):** numpy `Generator(PCG64(seed))`, wrapped so only an explicit seed
  enters and no global state is read — deterministic, cross-platform byte-identical, not hand-rolled
  (§6). The §4 transcription is for the **normal variate** (inverse-CDF, Jäckel ch 9) and later Sobol,
  not the uniform bits. So `rng.py` is mined for its seed-management *shape*, not its generator.
- Everything here is a **candidate**; a file enters a tranche only when a product pulls it, and is
  re-oracled on arrival. Nothing is imported from `_stock/`, ever.
