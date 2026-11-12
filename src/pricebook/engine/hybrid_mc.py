"""The equity-rate hybrid Monte-Carlo (L4) - the independent method eq 11.1 is checked against.

Simulates the Hull-White short rate and the equity **jointly and correlated**, accumulating the
cash-bond factor `B_t = exp(integral r du)` along each path, and prices the option as
`E[B_t^-1 (S_t - K)^+]` - the expectation eq (11.1) evaluates in closed form.

**It shares no expression with the closed form.** One assembles integrated variances and shifts
Black's `d+/-`; the other steps two correlated SDEs and averages a payoff. So a variance-assembly
error, a wrong cross term, or a missing `d+/-` shift shows up as a price gap - and crucially, the
shift is the term that **put-call parity cannot see**: parity holds for *any* shift, since
`N(x) + N(-x) = 1`, so without this simulation the shift would rest on the transcription alone.

**`rho` is pinned where it is directly observable, not through a price.** The correlation enters the
**factorisation** of the two increments, so a price pins only one correlation-weighted scalar and is
a weak test of it - tranche 17 measured a price giving a 13.9% band where the direct pin gave 1.66%.
So the oracle checks the **realised covariance** of the simulated rate and equity increments against
`rho*sigma_r*sigma*dt` directly. (Tranche 16's one-seed drift-differencing identity does *not* apply:
it works because the quantity under test enters only the drift and everything else cancels, and
`rho` is in the diffusion.)

The short rate is simulated around a **zero** mean-reversion level, with the deterministic drift that
would reprice today's curve omitted deliberately: the closed form's `F(t)` and `P(t)` already carry
the curve, and the quantity being checked is the *stochastic* correction. The simulation's own drift
is pinned separately by the martingale identity.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 11.1.2 - the Hull-White specialisation; the
          correlated Euler scheme is tranche 16's, reused
  oracle: tests/L4/test_hybrid_mc.py - the martingale identity `E[B_t^-1 S_t] = P(t)F(t)` pinning the
          drift independently of the vol; the realised covariance of the increments pinning `rho`
          directly; and the closed form within `k*SE` on a non-degenerate fixture, with the band's
          resolution asserted
  crosscheck: quantlib AnalyticBSMHullWhiteEngine / MC - deferred to the optional QuantLib tier
  slice:  tranche 34 - the equity-rate hybrid, the arc's first
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from pricebook.foundation.mc_estimator import MonteCarloEstimate, estimate
from pricebook.foundation.rng import UniformSource
from pricebook.foundation.variates import standard_normals
from pricebook.market.curve import CurveHandle
from pricebook.market.equity import EquityUnderlying, dividend_forward
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.equity_hull_white import EquityHullWhiteModel


@dataclass(frozen=True)
class HybridPaths:
    """What one simulation produces: terminal spots, cash-bond factors, and the raw increments.

    The increments are kept because `rho` is pinned on *them* rather than on a price - discarding
    them would force a second simulation to recover what this one already drew, and the T19 lesson
    is that an oracle re-deriving its own inputs is checking a different sample.
    """

    deflated_spots: np.ndarray
    deflators: np.ndarray
    #: `integral_0^t r du` per path - kept because it is the ONE place the cash-bond integrator is
    #: directly observable. The deflator is normalised to `P(t)`, which absorbs a first-order
    #: discretisation bias in its *mean*, so the price cannot see the integrator at all; the
    #: *variance* of this can, and the oracle checks it against the closed form (§4 - pin a quantity
    #: where it is observable rather than through a price that barely depends on it).
    rate_integrals: np.ndarray
    rate_increments: np.ndarray
    equity_increments: np.ndarray


def simulate_hybrid(
    model: EquityHullWhiteModel,
    underlying: EquityUnderlying,
    expiry: date,
    discount: CurveHandle,
    sampling: tuple[MonteCarloConfig, int],
) -> HybridPaths:
    """Correlated Euler on the short rate and log-equity, accumulating the cash bond."""
    config, steps = sampling
    if steps < 1:
        raise ValueError(f"the simulation needs at least one step, got {steps}")
    years = discount.time_measure.year_fraction(expiry)
    if years <= 0.0:
        raise ValueError(f"the simulation needs a positive horizon, got {years}")

    kappa, sigma_r = model.rate_params.kappa, model.rate_params.sigma
    sigma_s, rho = model.equity_vol, model.correlation
    dt = years / steps
    root_dt = math.sqrt(dt)
    orthogonal = math.sqrt(1.0 - rho * rho)

    source = UniformSource(config.seed)
    paths = config.paths
    rate = np.zeros(paths)            # the STOCHASTIC part of the short rate, around zero
    accumulated = np.zeros(paths)     # integral r du, the log cash-bond factor
    log_relative = np.zeros(paths)    # log(S_t/B_t) relative to its forward
    rate_steps, equity_steps = [], []

    for _ in range(steps):
        first = np.array(standard_normals(source.uniforms(paths)))
        second = np.array(standard_normals(source.uniforms(paths)))
        correlated = rho * first + orthogonal * second

        rate_increment = -kappa * rate * dt + sigma_r * root_dt * first
        # Trapezoid on `integral r du`: the rate moves within a step, and a left-endpoint rule
        # biases the cash-bond factor first-order in `dt` - which would show up as a martingale
        # failure and be mistaken for a correlation error.
        accumulated = accumulated + (rate + 0.5 * rate_increment) * dt
        rate = rate + rate_increment

        equity_step = -0.5 * sigma_s * sigma_s * dt + sigma_s * root_dt * correlated
        log_relative = log_relative + equity_step
        rate_steps.append(sigma_r * root_dt * first)
        equity_steps.append(sigma_s * root_dt * correlated)

    forward = dividend_forward(underlying, expiry, discount)

    # **The deflator is built so its sample mean is `P(t)` by construction**, using only simulated
    # data. `E[B_t^-1] = P(t)` requires a convexity correction `exp(-Var[X]/2)`, and taking `Var[X]`
    # from the closed form would make this simulation a restatement of the thing it is supposed to
    # check. Normalising by the sample's own `mean(exp(-X))` is the same correction estimated from
    # the paths, so the two methods stay independent - and the martingale oracle then tests the
    # *equity* drift rather than an identity the construction forced.
    deflator = discount.df(expiry) * np.exp(-accumulated)
    deflator = deflator / np.mean(np.exp(-accumulated))

    # `B_t^-1 S_t = S_0 exp(-qt - sigma^2 t/2 + sigma W^S)` EXACTLY: the `integral r du` in the
    # equity's drift cancels against the one in the cash bond, which is why no rate path appears
    # here. The hybrid effect survives entirely in the *correlation* between this and the deflator.
    deflated_spot = forward * discount.df(expiry) * np.exp(log_relative)
    return HybridPaths(
        rate_integrals=accumulated,
        deflated_spots=deflated_spot,
        deflators=deflator,
        rate_increments=np.column_stack(rate_steps),
        equity_increments=np.column_stack(equity_steps),
    )


def hybrid_call_estimate(paths: HybridPaths, strike: float) -> MonteCarloEstimate:
    """`E[B_t^-1 (S_t - K)^+]` - exactly the expectation eq (11.1) evaluates in closed form.

    Written as `E[(B^-1 S - K B^-1)^+]`, so the spot itself is never reconstructed: the deflated
    spot and the deflator are each exact, while `S_t = (B^-1 S)/B^-1` would divide one noisy
    quantity by another for no gain.
    """
    payoff = np.maximum(paths.deflated_spots - strike * paths.deflators, 0.0)
    return estimate(payoff.tolist())


def hybrid_forward_estimate(paths: HybridPaths) -> MonteCarloEstimate:
    """`E[B_t^-1 S_t]`, which must be `P(t)F(t)` - the drift, pinned independently of the vol."""
    return estimate(paths.deflated_spots.tolist())


def realised_increment_covariance(paths: HybridPaths) -> float:
    """The realised covariance of the rate and equity increments, per step.

    This is where `rho` is directly observable: it must reproduce `rho*sigma_r*sigma_S*dt`. A
    factorisation bias shows as a persistent gap that does not shrink as `sqrt(N)`.
    """
    rate = paths.rate_increments
    equity = paths.equity_increments
    return float(np.mean((rate - rate.mean(axis=0)) * (equity - equity.mean(axis=0))))


def realised_rate_integral_variance(paths: HybridPaths) -> float:
    """`Var[integral r du]` from the simulated paths - the cash-bond integrator, made observable.

    Must reproduce `integrated_bond_variance`. This is the only oracle in the tranche that can see
    the integration rule: measured, a trapezoid rule reproduces the closed form to **0.1%** while a
    left-endpoint rule is **4.6%** low, and neither moves the price by as much as a tenth of its
    standard error once the deflator is normalised.
    """
    return float(np.var(paths.rate_integrals, ddof=1))
