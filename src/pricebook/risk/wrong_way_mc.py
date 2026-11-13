"""The correlated-default Monte-Carlo (L5) - the independent anchor for wrong-way risk.

Simulates the equity and the hazard **jointly correlated**, and applies the default indicator
**exactly** - `1{U < e^{-H_T}}` with `U` uniform - rather than through the truncated-Gaussian
approximation the closed form uses. That is the point: the two differ in both method *and* model
approximation, so this measures the approximation's own error rather than merely re-deriving it.

Two draws per path:

* `X`, the equity's Gaussian, and `Y`, the hazard's, correlated at `rho` by the usual factorisation;
* `U`, an independent uniform, deciding survival against `e^{-H_T}` with `H_T = (a_T + s_H Y)^+`.

**What this does and does not isolate.** Running it with the *same* truncated-Gaussian `H_T` as the
closed form isolates the **four-term bivariate-normal transcription** - the only difference left is
analytic-versus-simulated. That is the comparison the oracle makes, because it is the one with a
known answer. The truncated Gaussian's own error against a genuinely non-Gaussian hazard is a
*modelling* question this tranche does not pretend to settle, and saying so is cheaper than implying
the MC settles it.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 13.1.2 / 14.5.1 - the definitions simulated
          directly: `{tau > T} = {U < e^{-H_T}}` with `H_T ~ (a_T + s_H Y)^+`
  oracle: tests/L5/test_wrong_way_mc.py - the realised correlation of the simulated `X` and `Y`
          pinning `rho` where it is observable (T17), the realised survival frequency pinning
          `p(T)`, and the closed form within `k*SE` across a `rho` grid with the band's resolution
          asserted
  crosscheck: none - a direct simulation of the model's own definitions
  slice:  tranche 36b - wrong-way risk, the independent anchor
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.foundation.mc_estimator import MonteCarloEstimate, estimate
from pricebook.foundation.rng import UniformSource
from pricebook.foundation.variates import standard_normals
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.risk.wrong_way_risk import GaussianHazard


@dataclass(frozen=True)
class CorrelatedDefaultPaths:
    """Terminal equity values, survival indicators, and the raw Gaussians.

    The Gaussians are kept because `rho` is pinned on *them* - it lives in the joint law, so a
    single CVA scalar is a weak test of it (T17). Discarding them would force a second simulation
    to recover what this one already drew.
    """

    spots: np.ndarray
    survived: np.ndarray
    equity_draws: np.ndarray
    hazard_draws: np.ndarray


def simulate_correlated_default(
    forward: float,
    equity_vol: float,
    hazard: GaussianHazard,
    config: MonteCarloConfig,
) -> CorrelatedDefaultPaths:
    """Correlated equity and hazard, with the EXACT default indicator."""
    source = UniformSource(config.seed)
    equity = np.array(standard_normals(source.uniforms(config.paths)))
    other = np.array(standard_normals(source.uniforms(config.paths)))
    rho = hazard.correlation
    hazard_draws = rho * equity + math.sqrt(1.0 - rho * rho) * other
    uniforms = np.array(source.uniforms(config.paths))

    integrated = np.maximum(hazard.truncation_level + hazard.hazard_vol * hazard_draws, 0.0)
    spots = forward * np.exp(equity_vol * equity - 0.5 * equity_vol * equity_vol)
    return CorrelatedDefaultPaths(
        spots=spots,
        survived=uniforms < np.exp(-integrated),
        equity_draws=equity,
        hazard_draws=hazard_draws,
    )


def risky_call_estimate(paths: CorrelatedDefaultPaths, strike: float) -> MonteCarloEstimate:
    """`E[(S_T - K)^+ 1{tau>T}]` - the quantity eq 14.5.1 evaluates in closed form."""
    payoff = np.maximum(paths.spots - strike, 0.0) * paths.survived
    return estimate(payoff.tolist())


def forward_estimate(paths: CorrelatedDefaultPaths) -> MonteCarloEstimate:
    """`E[S_T]`, which must be the forward - the equity drift, pinned before any price."""
    return estimate(paths.spots.tolist())


def realised_survival(paths: CorrelatedDefaultPaths) -> float:
    """The realised survival frequency, which must reproduce `p(T)` - the eq-13.4 solve, observed."""
    return float(paths.survived.mean())


def realised_correlation(paths: CorrelatedDefaultPaths) -> float:
    """The realised correlation of the two Gaussians - where `rho` is DIRECTLY observable (T17)."""
    return float(np.corrcoef(paths.equity_draws, paths.hazard_draws)[0, 1])
