"""The Heston Monte-Carlo (L4) - the independent method the Fourier pricer is checked against.

    dv     = kappa (theta - v) dt + sigma_v sqrt(v) dW2
    dlog S = (carry - v/2) dt + sqrt(v) dW1,            corr(dW1, dW2) = rho

No characteristic function appears anywhere here, which is the entire point: the Fourier pricer
inverts a complex integral and this integrates a path, so the two share no expression and a
transcription error in one cannot be reproduced by the other. The `sigma_v -> 0` reduction only
shows Fourier agrees with Black where the variance is *deterministic*; this is what makes the
stochastic, correlated case an assertion rather than a hope.

**FULL TRUNCATION on the variance, and the choice matters.** The exact CIR process stays positive
only when Feller's `2*kappa*theta >= sigma_v^2` holds, which the equity regime routinely violates -
a steep skew needs a large `sigma_v`, and this tranche's own fixture has `0.12 < 0.36`. A discretised
`v` therefore *will* go negative, and what happens next is a modelling decision rather than a detail:

* **absorption** (`v := max(v, 0)`) kills the variance permanently once it touches zero;
* **reflection** (`v := |v|`) injects variance the process never had;
* **full truncation** - carry `v` forward signed, but use `max(v, 0)` everywhere it is *read* - does
  neither. The drift still pulls the state back up, so a path that dips below zero recovers, and the
  bias is the smallest of the three (Lord, Koekkoek & van Dijk).

The spot uses **log-Euler**, so `S` stays positive by construction whatever `v` does.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 6.2 - the Heston dynamics; the full-truncation
          treatment of the discretised CIR variance is the standard scheme for the sub-Feller regime
  oracle: tests/L4/test_heston_mc.py (PRE-REGISTERED before this module existed) - the martingale
          identity `E[S_T] = F(T)` pinning the drift independently of the variance; agreement with
          the Fourier price within simulation error on a genuinely stochastic, correlated fixture
          whose Feller condition is deliberately violated; and the band's power, asserted by showing
          a flipped `rho` lands outside it
  crosscheck: quantlib HestonProcess / MCEuropeanHestonEngine - deferred to the optional QuantLib tier
  slice:  tranche 33 - Heston by Fourier inversion
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from pricebook.foundation.mc_estimator import MonteCarloEstimate, estimate
from pricebook.foundation.rng import UniformSource
from pricebook.foundation.variates import standard_normals
from pricebook.market.curve import CurveHandle
from pricebook.market.equity import EquityUnderlying, dividend_forward
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.heston import HestonParams


def simulate_heston_spots(
    params: HestonParams,
    underlying: EquityUnderlying,
    expiry: date,
    discount: CurveHandle,
    sampling: tuple[MonteCarloConfig, int],
) -> np.ndarray:
    """Terminal spots under Heston, by correlated log-Euler with a full-truncation variance.

    `sampling` bundles the path count with the step count: they are one decision about cost against
    discretisation bias, and separating them would push past §7's limit while hiding that they trade
    off against each other.
    """
    config, steps = sampling
    if steps < 1:
        raise ValueError(f"the simulation needs at least one step, got {steps}")
    years = discount.time_measure.year_fraction(expiry)
    if years <= 0.0:
        raise ValueError(f"the simulation needs a positive horizon, got {years}")

    forward = dividend_forward(underlying, expiry, discount)
    carry = math.log(forward / underlying.spot) / years
    dt = years / steps
    root_dt = math.sqrt(dt)
    orthogonal = math.sqrt(1.0 - params.rho * params.rho)

    source = UniformSource(config.seed)
    log_spots = np.full(config.paths, math.log(underlying.spot), dtype=float)
    variances = np.full(config.paths, params.v0, dtype=float)

    for _ in range(steps):
        first = np.array(standard_normals(source.uniforms(config.paths)))
        second = np.array(standard_normals(source.uniforms(config.paths)))
        correlated = params.rho * first + orthogonal * second

        # Full truncation: `v` is carried forward SIGNED, but every *read* of it is floored. The
        # drift term therefore still pulls a negative state back toward `theta`, which absorption
        # at zero would prevent and reflection would overcorrect.
        used = np.maximum(variances, 0.0)
        root_v = np.sqrt(used)
        log_spots = log_spots + (carry - 0.5 * used) * dt + root_v * root_dt * first
        variances = (
            variances
            + params.kappa * (params.theta - used) * dt
            + params.sigma_v * root_v * root_dt * correlated
        )
    return np.exp(log_spots)


def heston_forward_estimate(spots: np.ndarray) -> MonteCarloEstimate:
    """`E[S_T]` - what the martingale identity pins, independently of the variance process."""
    return estimate(spots.tolist())


def heston_call_estimate(
    spots: np.ndarray, strike: float, discount_factor: float
) -> MonteCarloEstimate:
    """The simulated call price and its standard error - the band the Fourier price is held to."""
    return estimate((np.maximum(spots - strike, 0.0) * discount_factor).tolist())
