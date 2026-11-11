"""The local-volatility Monte Carlo (L4) - the independent half of Dupire's inversion check.

Simulates the spot under the local-volatility diffusion

    dS = (r - q) S dt + sigma_loc(S, t) S dW

by log-Euler, with `sigma_loc` re-evaluated at each step from the Dupire surface. This exists to
**reprice the very surface Dupire was applied to**: "local volatility reproduces the vanillas it was
calibrated to" is the defining property, and with analytic Dupire it is a genuine inversion check
rather than a telescoping one - a *wrong* Dupire yields a `sigma_loc` whose simulated prices miss
the surface (§4's "a formula and its stated inverse must actually invert").

**Different code on each side is what makes it an oracle.** The analytic Dupire formula and this
diffusion share no expression: one differentiates a closed-form price, the other integrates a path.
A transcription error in either shows up as a price gap, and the band is **pure Monte-Carlo standard
error** because the analytic-derivative ruling removed the only other error source.

**The drift is pinned separately from the vol**, by the martingale identity `E[S_T] = F(T)` that
tranches 19 and 26 established: it is exact by construction and involves no model result, so a
mis-specified drift is caught *before* any statement about `sigma_loc` is made. Without it a drift
error and a vol error would be indistinguishable in a repricing gap.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 6.1 - the local-volatility diffusion; the log-Euler
          scheme is tranche 16's, reused
  oracle: tests/L4/test_local_volatility.py - the martingale identity pins the drift; the repricing
          of the parametrized surface within Monte-Carlo standard error is the inversion anchor,
          asserted on a SKEWED surface where a dropped Dupire term bites
  crosscheck: quantlib LocalVolMonteCarlo / GeneralizedBlackScholesProcess - deferred to the
          optional QuantLib tier
  slice:  tranche 32 - the implied-vol surface and Dupire local volatility
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from pricebook.foundation.mc_estimator import MonteCarloEstimate, estimate
from pricebook.foundation.rng import UniformSource
from pricebook.foundation.variates import standard_normals
from pricebook.market.curve import CurveHandle
from pricebook.market.equity import EquityUnderlying, dividend_forward
from pricebook.market.equity_skew import ImpliedVolSkew
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.local_volatility import dupire_local_variance

#: Below this the diffusion is effectively deterministic over a step; clamping keeps a surface's
#: floored wing (or a momentarily negative Dupire variance in a far tail) from producing a `nan`
#: that would silently poison every path's average.
_MIN_LOCAL_VARIANCE = 1e-12


def simulate_local_vol_spots(
    surface: ImpliedVolSkew,
    underlying: EquityUnderlying,
    expiry: date,
    discount: CurveHandle,
    sampling: tuple[MonteCarloConfig, int],
) -> np.ndarray:
    """Terminal spots under `sigma_loc(S,t)`, by log-Euler on `steps` uniform time steps.

    `sampling` bundles the Monte-Carlo config with the step count because they are one decision
    about the simulation's cost and accuracy, and splitting them would push this past §7's argument
    limit while hiding that they trade off against each other.
    """
    config, steps = sampling
    if steps < 1:
        raise ValueError(f"the simulation needs at least one step, got {steps}")
    years = discount.time_measure.year_fraction(expiry)
    if years <= 0.0:
        raise ValueError(f"the simulation needs a positive horizon, got {years}")

    rate = -math.log(discount.df(expiry)) / years
    carry = rate - underlying.dividends.continuous_yield - underlying.repo_rate
    dt = years / steps
    anchor = discount.time_measure.anchor

    source = UniformSource(config.seed)
    spots = np.full(config.paths, underlying.spot, dtype=float)
    for step in range(steps):
        elapsed = step * dt
        remaining = years - elapsed
        # `sigma_loc` is a function of (level, calendar time). Dupire is stated in terms of a strike
        # and a maturity, and the diffusion reads it at (S_t, t) - the same surface, read on the
        # diagonal rather than across strikes.
        when = anchor + timedelta(days=max(1, round(remaining * 365)))
        variances = np.array(
            [
                max(_MIN_LOCAL_VARIANCE, _safe_variance(surface, underlying, float(s), when, discount))
                for s in spots
            ]
        )
        draws = np.array(standard_normals(source.uniforms(config.paths)))
        spots = spots * np.exp((carry - 0.5 * variances) * dt + np.sqrt(variances * dt) * draws)
    return spots


def _safe_variance(
    surface: ImpliedVolSkew,
    underlying: EquityUnderlying,
    level: float,
    when: date,
    discount: CurveHandle,
) -> float:
    """`sigma_loc^2` at a level, or the floor where the surface is locally un-invertible.

    A Dupire variance can go non-positive in a far wing of a parametrized surface (the
    parametrization is not guaranteed arbitrage-free everywhere). Clamping *here*, at the point of
    use, keeps the diffusion well-posed while leaving `dupire_local_variance` free to report the
    negative value honestly to anyone who asks it directly - which is what the arbitrage oracle
    needs it to do.
    """
    try:
        variance = dupire_local_variance(surface, underlying, level, when, discount)
    except (ValueError, ZeroDivisionError):
        return _MIN_LOCAL_VARIANCE
    return variance if math.isfinite(variance) else _MIN_LOCAL_VARIANCE


def local_vol_call_estimate(
    spots: np.ndarray, strike: float, discount_factor: float
) -> MonteCarloEstimate:
    """The simulated call price and its standard error - the estimate the band is measured against."""
    payoffs = np.maximum(spots - strike, 0.0) * discount_factor
    return estimate(payoffs.tolist())


def forward_estimate(spots: np.ndarray) -> MonteCarloEstimate:
    """`E[S_T]` - what the martingale identity pins against `F(T)`, independently of the vol."""
    return estimate(spots.tolist())


def surface_call_price(
    surface: ImpliedVolSkew,
    underlying: EquityUnderlying,
    strike: float,
    expiry: date,
    discount: CurveHandle,
) -> float:
    """The parametrized surface's own call price - the target the simulation must reproduce."""
    from pricebook.models.black import black_price
    from pricebook.products.equity_option import EquityOptionKind

    years = discount.time_measure.year_fraction(expiry)
    forward = dividend_forward(underlying, expiry, discount)
    volatility = surface.volatility(strike, forward, years)
    return discount.df(expiry) * black_price(
        forward, strike, volatility * volatility * years, EquityOptionKind.CALL
    )
