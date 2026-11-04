"""Black by simulation (L3) — the Monte Carlo spine's first priced number. T15.

Simulates the forward as the driftless lognormal martingale it is under its own forward measure,

    F_T = F₀·exp(−V²/2 + V·z),   z ~ N(0,1)

and averages `max(ω(F_T − K), 0)`. That is precisely the expectation `black_price` evaluates in
closed form — so **our own closed form is the oracle**, which is the whole reason the MC spine starts
here rather than at an instrument with no analytic counterpart.

**Why this sits beside `black.py` at L3 rather than being an `engine/mc_european.py` on a European
product, which is what T15's §2 specified.** The library has no vanilla-option product, and nothing
pulls one: hosting a simulation would mean adding an L2 product *and* an L4 engine for an instrument
with no consumer, which is the speculative surface §6 forbids. The tranche's purpose is the Monte
Carlo machinery, and the simulated counterpart of `black_price` has a real consumer — the oracles
that check it against the closed form. *(Raised as an `Ask:` to Cowork rather than taken silently;
the engine placement returns the moment a product genuinely needs simulating, which is the first
model here without a closed form.)*

**No discounting, deliberately.** `black_price` returns the undiscounted expectation per unit of
notional, so this returns the same quantity and the two are directly comparable. Discounting is the
caller's — and every existing pricer already multiplies by its own `df`.

**The `−V²/2` is the whole correctness question in one term.** Without it `E[F_T] ≠ F₀` and every
option price is biased upward while still looking plausible — so the martingale oracle
(`E[F_T] = F₀`, checked at a zero strike where the payoff *is* the forward) is independent of the
option formula and catches exactly that.

**The oracle shape, ruled honest up front.** An MC price never equals the closed form, so the anchor
is `|MC − Black| < k·SE` with the standard error **measured**, never a tolerance chosen to pass; the
seed is explicit config so a fixed-seed run is a *pure function* (§3) and the assertion is hard
rather than flaky; and the `−½` convergence rate is asserted separately, because a biased estimator
can sit inside a 4-SE band at one `N` by luck but cannot fake the slope.

Provenance:
  stock:  _stock/quarry/pricebook/... `_mc.py` / `models/mc_*` — mined for the SHAPE and for the
          migration-hardening cases (path-count validation, seeding reproducibility) only; its
          placement under `models/` carries no authority and its code is not imported (§5)
  source: JAECKEL-2002 (docs/references.md) ch 2 and ch 14 for the standard error as the measure of
          MC convergence and for the −½ rate; the lognormal martingale form is standard Black-76,
          the same expectation `models/black.py` evaluates analytically
  oracle: tests/L3/test_black_mc.py — within four MEASURED standard errors of our `black_price`,
          with a companion forbidding the band from being vacuously wide; the `−½` rate fitted over
          four path counts; unbiasedness across forty seeds against the standard error of their
          mean; byte-identical at a fixed seed; EXACT agreement at zero variance; and the
          martingale `E[F_T] = F₀` as a payoff-independent check on the `−V²/2` term
  crosscheck: none — the independent side of this oracle is already our own closed form, which is
          itself oracled against the source in tranches 5 and 9, so an external engine would add a
          third party to a check that already has two
  slice:  tranche 15 — the Monte Carlo spine
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.mc_estimator import MonteCarloEstimate, estimate
from pricebook.foundation.rng import UniformSource
from pricebook.foundation.variates import standard_normals


@dataclass(frozen=True)
class MonteCarloConfig:
    """How many paths, and from which seed — the numerical config a simulation needs.

    **The seed is explicit config and not a default, which is §3.5 and the reason any of this is
    testable.** A simulation that seeded itself from entropy would read ambient state, so it could
    not be a pure function of its arguments and neither §3.1's referential transparency nor §9's
    byte-identical check would hold. Bundled with the path count because the two are the whole of
    what the numerics need, and the charter's rule is to group the cohesive ones."""

    paths: int
    seed: int

    def __post_init__(self) -> None:
        if self.paths < 2:
            raise ValueError(
                f"a Monte Carlo estimate needs at least two paths to have a standard error, "
                f"got {self.paths}"
            )
        if self.seed < 0:
            raise ValueError(f"a seed must be non-negative, got {self.seed}")


def black_mc_price(
    forward: float,
    strike: float,
    total_variance: float,
    kind: CapFloorKind,
    config: MonteCarloConfig,
) -> MonteCarloEstimate:
    """`E[max(ω(F_T − K), 0)]` by simulation — the same quantity `black_price` returns in closed form.

    Returns the estimate **with its standard error**, because a Monte Carlo price without its error
    bar cannot be compared to anything and would force every caller to invent a tolerance."""
    omega = kind.omega
    if total_variance <= 0.0:
        # Exact, not a limit: with no variance every path lands on the forward, so the payoff is
        # deterministic. Simulating it would be arithmetic noise around a known answer, and the
        # standard error is exactly zero rather than merely small.
        intrinsic = max(omega * (forward - strike), 0.0)
        return MonteCarloEstimate(
            mean=intrinsic, standard_error=0.0, n_samples=config.paths
        )

    # Box-Muller consumes uniforms in pairs, so an odd path count draws one extra normal and
    # discards it — the request is honoured exactly rather than silently rounded, and the discard
    # is deterministic because the stream is.
    pairs_needed = config.paths + (config.paths % 2)
    uniforms = UniformSource(seed=config.seed).uniforms(pairs_needed)
    normals = standard_normals(uniforms)[: config.paths]

    volatility = math.sqrt(total_variance)
    drift = -0.5 * total_variance
    payoffs = [
        max(omega * (forward * math.exp(drift + volatility * z) - strike), 0.0)
        for z in normals
    ]
    return estimate(payoffs)
