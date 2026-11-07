"""Greeks by simulation (L5). JAECKEL-2002 ch 11 — §11.3, §11.5, §11.6 and appendix §11.9.1.

Three estimators of the same quantity, each with a different idea about where the derivative lives:

* **finite differencing with path recycling (§11.3)** — bump the input, reprice on the *same* draws,
  difference. Sharing the draws is what makes the difference low-variance rather than two
  independent noisy prices subtracted. It is common random numbers, and tranche 19 established that
  CRN is variance-*reduced*, not noise-free.
* **pathwise differentiation (§11.5)** — differentiate the payoff along the path. *"For
  Lipschitz-continuous payoff functions ... the order of differentiation and expectation can be
  interchanged."* Not otherwise, which is the whole reason the third method exists.
* **the likelihood ratio method (§11.6)** — differentiate the *density* instead. The payoff is left
  alone and carries a weight `ω`; *"it does not require any kind of continuity of the payoff
  function."*

Jäckel's printed weights, for the single-horizon case this module covers:

    ω_Δ = z / (S₀ σ √Δt₁)                            (11.30)
    ω_Γ = (z² − z σ √Δt₁ − 1) / (S₀² σ² Δt₁)         (11.31)
    ω_V = (z² − 1)/σ − z √T                          (11.32, appendix §11.9.1)

**Adapted from spot to forward, which is a convention change rather than a transcription.** Jäckel
writes `S_T = S₀·exp((r − d − ½σ²)T + σ√T z)`; our Black is a *forward* Black, undiscounted, and a
forward is driftless under its own measure: `F_T = F₀·exp(−½σ²T + σ√T z)`. The weights carry over
with `S₀ → F₀`, `r − d → 0` and a single horizon `Δt₁ = T`. Stated because an unstated convention
change is how a right formula becomes a wrong number.

**Jäckel's own caveat, worth keeping visible:** the weights are inversely proportional to `σ√Δt₁` and
`σ²Δt₁`, so *"the variance incurred by this calculation of Delta and Gamma may increase dramatically
for low volatilities or very short time intervals"*. The likelihood ratio is the method that always
*applies*, not the method that is always sharpest.

**Pathwise is not offered for gamma, because the source does not offer it:** *"for the calculation of
Gamma, we still have to implement a finite differencing scheme for two individually calculated
Deltas."* It gives a derivative per path, not a second derivative, and a method list that quietly
included it would be inventing a capability.

Provenance:
  stock:  _stock/quarry/pricebook/... mc Greeks and `risk/pathwise_greeks.py` — read for SHAPE and
          migration edge cases only; the weights are read from the source (§5)
  source: JAECKEL-2002 (docs/references.md) ch 11 — §11.3 (finite differencing with path recycling),
          §11.5 (pathwise differentiation and its Lipschitz condition), §11.6 eqs 11.26-11.31 (the
          likelihood ratio and its Delta/Gamma weights), appendix §11.9.1 eq 11.32 (the Vega weight,
          printed in full for the single-horizon case). All read in `-raw` and `-layout`, which agree
  oracle: tests/L5/test_sensitivities.py — every method against Black's analytic Greeks within the
          GREEK's own standard error; the three-method cross-check (different routes, one answer);
          and the reduction-with-companion that earns the second method — pathwise valid on a call
          and returning EXACTLY zero on a digital, where the likelihood ratio holds; the -1/2 rate on
          a Greek; determinism; the refusals. The analytic Greeks themselves are oracled at
          tests/L3/test_black.py as the LIMIT of a central difference of the price verified since
          tranche 5 — they did not exist before this tranche, despite the spec assuming they did
  crosscheck: quantlib (MC Greeks) — deferred to the optional QuantLib tier
  slice:  tranche 22 — Greeks by simulation; the first L5 module
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.mc_estimator import MonteCarloEstimate, estimate
from pricebook.foundation.rng import UniformSource
from pricebook.foundation.variates import standard_normals
from pricebook.models.black_mc import MonteCarloConfig

#: The relative bump for finite differencing. Large enough that the differenced prices do not cancel
#: away their significant digits, small enough that the `O(h^2)` truncation stays below the Monte
#: Carlo standard error -- which is the binding error here, unlike the analytic oracle where the
#: truncation was the only error and the step had to be driven to zero.
_BUMP = 1e-3


@dataclass(frozen=True)
class BlackState:
    """The option a Greek is taken of: `forward`, `strike`, `total_variance` and the `years` to it.

    **A bundle because §7's five-argument limit forced one**, and the limit was right to force it:
    `black_mc_vega(forward, strike, total_variance, method, config, years)` was six arguments, and
    §7's prescribed answer is to group into a frozen value object rather than raise the limit or
    suppress the check. Four present consumers share this triple-plus-horizon, so it is compelled
    vocabulary, not speculation (§6).

    **`years` is read only by vega, and that is not a carried-but-unread field.** `total_variance` is
    `σ²·years`, so the horizon is what separates a one-year 20% option from a four-year 10% one —
    constitutive of the option's identity, which is §6's test. Vega needs it because eq 11.32's
    weight contains a bare `√T`; delta and gamma do not, because theirs do not."""

    forward: float
    strike: float
    total_variance: float
    years: float = 1.0

    def __post_init__(self) -> None:
        if self.forward <= 0.0:
            raise ValueError(f"a forward must be positive to be lognormal, got {self.forward}")
        if self.total_variance <= 0.0:
            raise ValueError(
                f"a positive variance is needed to differentiate a density, got "
                f"{self.total_variance}; at zero variance there is no distribution, only the "
                f"intrinsic"
            )
        if self.years <= 0.0:
            raise ValueError(f"the horizon must be positive, got {self.years}")

    @property
    def sigma(self) -> float:
        """`σ = √(V²/T)` — the annualised volatility eq 11.32's weight is written in."""
        return math.sqrt(self.total_variance / self.years)


class GreekMethod(Enum):
    """Which of chapter 11's three estimators to use."""

    FINITE_DIFFERENCE = "finite_difference"
    PATHWISE = "pathwise"
    LIKELIHOOD_RATIO = "likelihood_ratio"


def _variates(config: MonteCarloConfig) -> list[float]:
    """The shared draws. Validation lives on `BlackState`, where it cannot be bypassed."""
    uniforms = UniformSource(seed=config.seed).uniforms(config.paths + config.paths % 2)
    return list(standard_normals(uniforms)[: config.paths])


def _terminal(forward: float, total_variance: float, z: float) -> float:
    """`F_T = F_0 * exp(-V^2/2 + V z)` — the driftless forward under its own measure."""
    root = math.sqrt(total_variance)
    return forward * math.exp(-0.5 * total_variance + root * z)


def _call(value: float, strike: float) -> float:
    return max(value - strike, 0.0)


def black_mc_delta(
    state: BlackState, method: GreekMethod, config: MonteCarloConfig
) -> MonteCarloEstimate:
    """`∂/∂F₀ E[(F_T − K)⁺]` by simulation — a call's delta, undiscounted.

    Returns the estimate **with its standard error**: a Greek is an estimate, so §3's invariant 7
    applies to it exactly as to a price, and a banded oracle reads the error rather than inventing a
    tolerance."""
    forward, strike, total_variance = state.forward, state.strike, state.total_variance
    draws = _variates(config)

    if method is GreekMethod.FINITE_DIFFERENCE:
        bump = forward * _BUMP
        samples = [
            (
                _call(_terminal(forward + bump, total_variance, z), strike)
                - _call(_terminal(forward - bump, total_variance, z), strike)
            )
            / (2.0 * bump)
            for z in draws
        ]
    elif method is GreekMethod.PATHWISE:
        # d/dF0 of (F_T - K)^+ is (F_T/F_0) when in the money and 0 otherwise -- the payoff is
        # Lipschitz, so §11.5 permits the interchange. *"It is perfectly consistent to assign a Delta
        # of zero to all paths that terminate out of the money, and a Delta equal to dS_T/dS_0."*
        samples = []
        for z in draws:
            terminal = _terminal(forward, total_variance, z)
            samples.append(terminal / forward if terminal > strike else 0.0)
    else:
        root = math.sqrt(total_variance)
        samples = [
            _call(_terminal(forward, total_variance, z), strike) * (z / (forward * root))
            for z in draws
        ]
    return estimate(samples)


def black_mc_gamma(
    state: BlackState, method: GreekMethod, config: MonteCarloConfig
) -> MonteCarloEstimate:
    """`∂²/∂F₀² E[(F_T − K)⁺]` by simulation.

    **Pathwise is refused, not approximated.** §11.5 gives a derivative per path, not a second one;
    the source's own remedy is to finite-difference two pathwise deltas, which is a different method
    from the one named."""
    if method is GreekMethod.PATHWISE:
        raise ValueError(
            "pathwise differentiation gives a first derivative per path, not a second; §11.5's own "
            "remedy for gamma is to finite-difference two pathwise deltas, which is a different "
            "method — use FINITE_DIFFERENCE or LIKELIHOOD_RATIO"
        )
    forward, strike, total_variance = state.forward, state.strike, state.total_variance
    draws = _variates(config)

    if method is GreekMethod.FINITE_DIFFERENCE:
        bump = forward * _BUMP
        samples = [
            (
                _call(_terminal(forward + bump, total_variance, z), strike)
                - 2.0 * _call(_terminal(forward, total_variance, z), strike)
                + _call(_terminal(forward - bump, total_variance, z), strike)
            )
            / bump**2
            for z in draws
        ]
    else:
        root = math.sqrt(total_variance)
        samples = [
            _call(_terminal(forward, total_variance, z), strike)
            * ((z * z - z * root - 1.0) / (forward * forward * total_variance))
            for z in draws
        ]
    return estimate(samples)


def black_mc_vega(
    state: BlackState, method: GreekMethod, config: MonteCarloConfig
) -> MonteCarloEstimate:
    """`∂/∂σ E[(F_T − K)⁺]` by simulation, with `σ = √(V²/years)`.

    The horizon comes from the state because eq 11.32's weight contains `√T` on its own, not only
    inside the variance — so unlike delta and gamma this Greek genuinely needs it, and deriving a
    `∂/∂σ` from something never told `T` would be inventing one."""
    forward, strike, total_variance = state.forward, state.strike, state.total_variance
    draws = _variates(config)
    years, sigma, root_t = state.years, state.sigma, math.sqrt(state.years)

    if method is GreekMethod.FINITE_DIFFERENCE:
        bump = sigma * _BUMP
        samples = []
        for z in draws:
            up = ((sigma + bump) ** 2) * years
            down = ((sigma - bump) ** 2) * years
            samples.append(
                (
                    _call(_terminal(forward, up, z), strike)
                    - _call(_terminal(forward, down, z), strike)
                )
                / (2.0 * bump)
            )
    elif method is GreekMethod.PATHWISE:
        # d/dsigma of F_T = F_T * (sqrt(T) z - sigma T), so the pathwise vega is that times the
        # in-the-money indicator -- the same Lipschitz argument as for delta.
        samples = []
        for z in draws:
            terminal = _terminal(forward, total_variance, z)
            derivative = terminal * (root_t * z - sigma * years)
            samples.append(derivative if terminal > strike else 0.0)
    else:
        samples = [
            _call(_terminal(forward, total_variance, z), strike)
            * ((z * z - 1.0) / sigma - z * root_t)
            for z in draws
        ]
    return estimate(samples)


def digital_mc_delta(
    state: BlackState, method: GreekMethod, config: MonteCarloConfig
) -> MonteCarloEstimate:
    """`∂/∂F₀ P(F_T > K)` by simulation — the case that justifies the likelihood ratio.

    A digital's payoff is flat almost everywhere, so **pathwise returns exactly zero on every path**
    while the true delta is strictly positive. That is not a loss of accuracy but a wrong answer, and
    it is why §11.6's method exists: it differentiates the density and never touches the payoff, so
    the payoff's discontinuity is irrelevant to it.

    Offered for all three methods precisely so the failure is *demonstrable* rather than described."""
    forward, strike, total_variance = state.forward, state.strike, state.total_variance
    draws = _variates(config)

    def indicator(value: float) -> float:
        return 1.0 if value > strike else 0.0

    if method is GreekMethod.FINITE_DIFFERENCE:
        bump = forward * _BUMP
        samples = [
            (
                indicator(_terminal(forward + bump, total_variance, z))
                - indicator(_terminal(forward - bump, total_variance, z))
            )
            / (2.0 * bump)
            for z in draws
        ]
    elif method is GreekMethod.PATHWISE:
        # Zero on every path: the indicator is locally constant, so its derivative vanishes wherever
        # it is defined. This is the method being WRONG, reproduced faithfully rather than patched.
        samples = [0.0 for _ in draws]
    else:
        root = math.sqrt(total_variance)
        samples = [
            indicator(_terminal(forward, total_variance, z)) * (z / (forward * root))
            for z in draws
        ]
    return estimate(samples)
