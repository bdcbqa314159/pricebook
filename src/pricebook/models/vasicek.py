"""The Vasicek short-rate model, and Merton as the model it reduces to (L3).
`MCINERNEY-ZASTAWNIAK` §3.3, §3.4.

The canonical mean-reverting Gaussian short rate, and the first tranche of the single-curve arc.
Its significance for this tree is structural as well as financial: a short-rate model **makes** the
discount curve out of `r`, so the single-curve mode of the design invariant is exercised **by
construction** here rather than selected by the shape of a snapshot.

**The source's notation, which is not the only one in circulation.** Written out because the two
differ in a way that matters:

    Vasicek  (§3.4):  dr(t) = (theta - alpha r(t)) dt + sigma dW(t)
    Merton   (§3.3):  dr(t) = alpha dt + sigma dW(t)

So **`alpha` is the mean-reversion speed in Vasicek and a plain drift in Merton** - one letter for
two things, three pages apart - and Vasicek's long-run mean is **`theta/alpha`**, not `theta`. The
tranche spec used `dr = kappa(theta - r)dt`, the other common parameterisation, in which `theta` *is*
the level; this module follows the source (§5), so `theta` is the drift constant. The distinction is
not cosmetic: the two readings of `theta` differ by a factor of `alpha`, which is 5.6x on the
oracle's own fixture.

    D(t,T)    = integral_t^T e^{-alpha(s-t)} ds = (1 - e^{-alpha(T-t)})/alpha          (3.5)
    E[r(s)]   = r(0)e^{-alpha s} + theta(1 - e^{-alpha s})/alpha
    Var[r(s)] = sigma^2 (1 - e^{-2 alpha s})/(2 alpha)   -> sigma^2/(2 alpha)
    B(t,T)    = exp( -r(t)D(t,T) - theta I1 + 1/2 sigma^2 I2 )                         (3.6)
                with I1 = integral_t^T D(u,T) du,  I2 = integral_t^T D(u,T)^2 du
    Merton:   B(t,T) = exp( -r(t)(T-t) - 1/2 alpha (T-t)^2 + (1/6) sigma^2 (T-t)^3 )   (3.4)

**`D(t,T)` IS `hw_beta`, and that reuse is the arc's premise.** `(1-e^{-alpha tau})/alpha` is the
affine function tranche 10 defined for Hull-White, and every model in the book's chapter 3 shares
it, so it is called rather than rewritten. Their agreement is a **consistency guard, not
cross-source independence** - one object read twice (§4's same-formula caveat).

**The source prints (3.6) as integrals and leaves the closed forms as Exercise 3.2.** Worked here:

    I1 = (tau - D)/alpha
    I2 = (tau - 2D + (1 - e^{-2 alpha tau})/(2 alpha)) / alpha^2

and oracled by **quadrature of the printed integrands** - a different computation path, not a second
transcription of a closed form the source never prints.

**`alpha > 0` is structural, and the limit is analytic rather than numerical.** Both closed forms
divide by `alpha` (`I2` twice), so zero is undefined rather than slow. More importantly the
`alpha -> 0` limit **cannot be reached by evaluating at a small alpha**: `I1 = (tau - D)/alpha` is a
`0/0` and `I2` is a `0/0` over `alpha^2`, so catastrophic cancellation sets in well before the
limit - measured against quadrature, machine-exact at `alpha = 0.1`, 5e-11 at `1e-3`, and **1.4%
wrong at `1e-6`**. The formula *diverges* from its own limit rather than approaching it to a few
ULP, so this is not the T61 "exact in R, banded in float" case; it is a **stated-regime** case, and
the oracle asserts the reduction as a convergence **rate** over the range where the closed form is
sound (the T25 rate-detector). Merton is therefore its own small closed form (3.4) rather than a
limit to be evaluated - which is also how the source presents it.

**Its own parameter type rather than `HullWhiteParams` plus a theta.** `alpha` and `sigma` play the
same algebraic role and `D` *is* `hw_beta` - but Vasicek's `theta` is a **constant** where
Hull-White's `theta(t)` is fitted to the initial curve, which is the difference between a model that
*produces* a curve and one that *reproduces* it. Same SDE family, different idea (§6). Tranche 57's
LGM reuse is the contrast that draws the line: there the LGM's numbers literally *are* Hull-White's,
reparameterised.

Provenance:
  stock:  none - the archived predecessor had no Vasicek or Merton model
  source: MCINERNEY-ZASTAWNIAK (docs/references.md) §3.3 (Merton, eq 3.4) and §3.4 (Vasicek, the
          SDE, the short-rate moments, eq 3.5's `D` and eq 3.6's bond, whose closed form is the
          book's Exercise 3.2); transcribed in both renderings and reconciled
  oracle: tests/L3/test_vasicek.py - `I1`, `I2` and the bond each oracled against QUADRATURE of the
          source's printed integrands (a different computation path, since the source never prints
          the closed form); `D` cross-checked against `hw_beta` and labelled a consistency guard;
          `B(T,T) = 1` exactly as a requirement companion; the short-rate moments pinned pointwise
          with the stationary variance MEASURED to saturate rather than pre-asserted (§4's
          obligation-not-form clause) and the long-run mean asserted to be `theta/alpha` rather
          than `theta`; Merton's (3.4) transcribed pointwise; and the reduction asserted as an
          O(alpha) convergence RATE inside the regime where the closed form is accurate, with a
          companion asserting that it DEGRADES outside it so the regime is stated and not assumed
  crosscheck: quantlib Vasicek / AnalyticDiscountFactor - deferred to the optional QuantLib tier
  slice:  tranche 66 - the Vasicek short-rate model; the single-curve short-rate arc opens
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.hull_white import hw_beta


def _checked_finite(**values: float) -> None:
    """Every parameter finite, named in the message so a reader knows which one failed."""
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")


@dataclass(frozen=True)
class VasicekParams:
    """`dr = (theta - alpha r)dt + sigma dW` - the source's parameterisation (§3.4).

    `theta` is the **drift constant**, so the long-run mean is `theta/alpha`. See the module
    docstring: the other common parameterisation makes `theta` the level, and the two differ by a
    factor of `alpha`.
    """

    alpha: float
    theta: float
    sigma: float

    def __post_init__(self) -> None:
        _checked_finite(alpha=self.alpha, theta=self.theta, sigma=self.sigma)
        if self.alpha <= 0.0:
            raise ValueError(f"mean reversion must be positive, got {self.alpha}")
        if self.sigma < 0.0:
            raise ValueError(f"volatility cannot be negative, got {self.sigma}")

    @property
    def long_run_mean(self) -> float:
        """`theta/alpha` - the level the short rate reverts to, named so a caller cannot mistake
        `theta` for it."""
        return self.theta / self.alpha


@dataclass(frozen=True)
class MertonParams:
    """`dr = drift dt + sigma dW` - §3.3, where the source calls the drift `alpha`.

    Named `drift` here rather than `alpha` **on purpose**: the source reuses the letter for
    Vasicek's mean reversion three pages later, and a shared name across two different meanings is
    exactly the confusion §4's independence rules exist to prevent. The vocabulary is the fix.
    """

    drift: float
    sigma: float

    def __post_init__(self) -> None:
        _checked_finite(drift=self.drift, sigma=self.sigma)
        if self.sigma < 0.0:
            raise ValueError(f"volatility cannot be negative, got {self.sigma}")


@dataclass(frozen=True)
class VasicekModel:
    """The Vasicek parameters and the current short rate, carrying the snapshot (§3).

    `short_rate` is the model's **initial state**, not market data reached through the snapshot:
    the snapshot carries curves, and recovering `r(0)` from one would mean differentiating an
    interpolated curve to get an instantaneous forward - a real piece nothing pulls yet (§6). A
    short-rate model's curve is its *output*, which is the whole point of this arc.
    """

    market: MarketSnapshot
    params: VasicekParams
    short_rate: float

    def __post_init__(self) -> None:
        _checked_finite(short_rate=self.short_rate)


def vasicek_drift_integral(params: VasicekParams, t: float, maturity: float) -> float:
    """`I1 = integral_t^T D(u,T) du = (tau - D)/alpha` - the source's Exercise 3.2, worked.

    Named surface rather than a local inside the bond, so a test can transcribe it against
    quadrature of the printed integrand (the F-6.1 rule).

    Numerically this is a difference of near-equal quantities divided by `alpha`, so it loses
    precision as `alpha -> 0`; the module docstring states the regime and the oracle measures it.
    """
    if maturity < t:
        raise ValueError(f"a bond cannot mature ({maturity}) before it is observed ({t})")
    return (maturity - t - hw_beta(params.alpha, t, maturity)) / params.alpha


def vasicek_variance_integral(params: VasicekParams, t: float, maturity: float) -> float:
    """`I2 = integral_t^T D(u,T)^2 du` - Exercise 3.2's second integral, worked:

        I2 = (tau - 2D + (1 - e^{-2 alpha tau})/(2 alpha)) / alpha^2

    The `alpha^2` divisor is why this is the less stable of the two as `alpha -> 0`.
    """
    if maturity < t:
        raise ValueError(f"a bond cannot mature ({maturity}) before it is observed ({t})")
    tau = maturity - t
    decay = hw_beta(params.alpha, t, maturity)
    tail = -math.expm1(-2.0 * params.alpha * tau) / (2.0 * params.alpha)
    return (tau - 2.0 * decay + tail) / (params.alpha * params.alpha)


def vasicek_bond(
    params: VasicekParams, short_rate: float, t: float, maturity: float
) -> float:
    """`B(t,T)` - eq (3.6), with both integrals in closed form.

    Exactly `1.0` at `t = T`: every term carries a factor of `tau`, so the exponent is zero there
    rather than nearly so - which is why the oracle can assert equality and not a tolerance.
    """
    if maturity < t:
        raise ValueError(f"a bond cannot mature ({maturity}) before it is observed ({t})")
    if maturity == t:
        return 1.0
    exponent = (
        -short_rate * hw_beta(params.alpha, t, maturity)
        - params.theta * vasicek_drift_integral(params, t, maturity)
        + 0.5 * params.sigma * params.sigma * vasicek_variance_integral(params, t, maturity)
    )
    return math.exp(exponent)


def vasicek_short_rate_moments(
    params: VasicekParams, short_rate: float, horizon: float
) -> tuple[float, float]:
    """`(E[r(s)], Var[r(s)])` under Q - the printed moments of §3.4.

    Returned as a pair because they come from one walk of the same solved SDE and every consumer
    (the simulator, the oracle, a future calibration) needs both. The variance is bounded by
    `sigma^2/(2 alpha)`, which is the structural difference from a driftless short rate whose
    variance grows linearly without bound.
    """
    if horizon < 0.0:
        raise ValueError(f"a horizon cannot be negative, got {horizon}")
    decay = math.exp(-params.alpha * horizon)
    mean = short_rate * decay + params.theta * (1.0 - decay) / params.alpha
    variance = (
        params.sigma
        * params.sigma
        * -math.expm1(-2.0 * params.alpha * horizon)
        / (2.0 * params.alpha)
    )
    return mean, variance


def merton_bond(
    params: MertonParams, short_rate: float, t: float, maturity: float
) -> float:
    """`B(t,T) = exp(-r(t)tau - 1/2 drift tau^2 + (1/6) sigma^2 tau^3)` - eq (3.4).

    Its own closed form rather than Vasicek evaluated at `alpha = 0`, and for two reasons: the
    source presents Merton as its own model (§3.3), and Vasicek's closed form cannot be evaluated
    there at all (the module docstring's regime note).
    """
    if maturity < t:
        raise ValueError(f"a bond cannot mature ({maturity}) before it is observed ({t})")
    tau = maturity - t
    exponent = (
        -short_rate * tau
        - 0.5 * params.drift * tau * tau
        + params.sigma * params.sigma * tau * tau * tau / 6.0
    )
    return math.exp(exponent)
