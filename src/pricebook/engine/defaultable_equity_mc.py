"""The defaultable-equity Monte-Carlo (L4) - the independent method the modified Black is checked against.

Draws the two sources of randomness **separately**, which is the model's premise made operational:
a lognormal `S~` with forward `F/p`, and a default indicator `1{U < p}` from a uniform drawn
independently of the equity variate. If the two shared a stream the simulation would encode a
dependence the closed form assumes away, and agreement would stop being evidence about the formula.

    payoff = (S~ * 1{survive} - K)^+

No Black formula, no scaled strike, no survival algebra - just the definition `S_t = S~_t 1{tau>t}`.
So a strike-scaling error, an inverted scaling, or a missing default lump shows up as a price gap.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 13.1.1 - the definition (13.2) simulated directly
  oracle: tests/L4/test_defaultable_equity_mc.py - `E[S_t] = F(t)` pinning the default weighting
          before any price is trusted, and agreement with the closed form within `k*SE` on a
          fixture whose survival probability is materially below 1
  crosscheck: none - a direct simulation of the model's own definition
  slice:  tranche 35 - defaultable equity, the equity-credit seam
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.foundation.mc_estimator import MonteCarloEstimate, estimate
from pricebook.foundation.rng import UniformSource
from pricebook.foundation.variates import standard_normals


def simulate_defaultable_spots(
    forward: float, survival: float, volatility: float, years: float, config
) -> np.ndarray:
    """`S~ * 1{survive}` per path, with the default indicator drawn INDEPENDENTLY of the diffusion."""
    if not 0.0 < survival <= 1.0:
        raise ValueError(f"the survival probability must lie in (0, 1], got {survival}")
    source = UniformSource(config.seed)
    width = volatility * math.sqrt(years)
    draws = np.array(standard_normals(source.uniforms(config.paths)))
    # A SECOND, disjoint slice of the same stream - independent of the normals above, which is the
    # model's premise. Reusing the equity uniforms would correlate default with the diffusion.
    uniforms = np.array(source.uniforms(config.paths))
    surviving = (forward / survival) * np.exp(-0.5 * width * width + width * draws)
    return surviving * (uniforms < survival)


def defaultable_call_estimate(spots: np.ndarray, strike: float) -> MonteCarloEstimate:
    """`E[(S_t - K)^+]`, undiscounted."""
    return estimate(np.maximum(spots - strike, 0.0).tolist())


def defaultable_put_estimate(spots: np.ndarray, strike: float) -> MonteCarloEstimate:
    """`E[(K - S_t)^+]`, undiscounted - the one that must find the default lump."""
    return estimate(np.maximum(strike - spots, 0.0).tolist())


def defaultable_forward_estimate(spots: np.ndarray) -> MonteCarloEstimate:
    """`E[S_t]`, which must be `F(t)` - the default weighting, pinned before any price."""
    return estimate(spots.tolist())
