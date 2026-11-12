"""The Heston stochastic-volatility model and its characteristic function (L3).

    dS/S = (r - q) dt + sqrt(v) dW1
    dv   = kappa (theta - v) dt + sigma_v sqrt(v) dW2,   d<W1,W2> = rho dt

Heston's value is that it has a **closed-form characteristic function** while its variance is
genuinely stochastic and correlated with the spot. The correlation `rho` is what produces a skew
from dynamics rather than by assumption - tranche 32's local volatility reproduces today's surface
exactly but says nothing about how it moves, and this is the first model in the tree that does.

**The characteristic function is of the log return**, `phi_T(u) = E[exp(iu*log(S_T/S_0))`], which
makes two identities immediate and transcription-independent: `phi(0) = 1`, and `phi(-i) =
E[S_T]/S_0 = exp(carry*T)` - the martingale identity, which pins the drift no matter how the
variance part is written.

**THE LITTLE HESTON TRAP.** The function contains a complex square root and a complex logarithm, and
the *naive* branch - the one that falls out of the derivation first, with `g1 = (beta+d)/(beta-d)`
and `e^{+dT}` - crosses a branch cut as maturity grows. It does not fail loudly: it returns a
plausible complex number that is simply **wrong**, and a price built on it is wrong with it. The
stable branch (`g2 = (beta-d)/(beta+d)` and `e^{-dT}`, Albrecher-Mayer-Schoutens-Tistaert) is the
same function analytically continued, and agrees with the naive one at short maturity where neither
has crossed anything.

`naive_characteristic_function` is kept **deliberately**, and it is not dead code: it is the
companion that proves the trap oracle has power. A test asserting "the stable branch is smooth"
means nothing unless the fixture is one where the naive branch demonstrably is not (§4 - the
Box-Muller pattern, where the wrong variant must be shown to fail).

Provenance:
  stock:  none - the archived predecessor had no stochastic-volatility model in this tree
  source: BROCKHAUS-2016 (docs/references.md) ch 6.2 / 16.2 - the Heston dynamics and its
          characteristic function; the stable branch follows Albrecher et al.'s correction, flagged
          in the tranche spec before the build
  oracle: tests/L3/test_heston_cf.py (PRE-REGISTERED before this module existed) - `phi(0)=1`,
          `phi(-i)=exp(carry*T)`, Hermitian symmetry, `|phi| <= 1`, the deterministic-variance
          reduction at `sigma_v = 0` integrated independently by the test, and the trap: the stable
          branch stays smooth in maturity where the naive one jumps, with the naive jump asserted
          so the fixture is proved to trip it
  crosscheck: quantlib AnalyticHestonEngine - deferred to the optional QuantLib tier
  slice:  tranche 33 - Heston by Fourier inversion
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HestonParams:
    """The five Heston parameters - one bundle, because no four of them mean anything alone.

    `kappa` mean-reversion speed, `theta` long-run variance, `sigma_v` vol-of-vol, `rho` spot/vol
    correlation, `v0` initial variance. All five are *variance*-dimensioned where they are
    variances, never volatilities: `theta = 0.04` is a 20% long-run vol, and converting at the
    boundary is how the squaring happens twice or not at all.
    """

    kappa: float
    theta: float
    sigma_v: float
    rho: float
    v0: float

    def __post_init__(self) -> None:
        if self.kappa <= 0.0:
            raise ValueError(f"the mean-reversion speed must be positive, got {self.kappa}")
        if self.theta < 0.0 or self.v0 < 0.0:
            raise ValueError(f"variances may not be negative, got theta={self.theta}, v0={self.v0}")
        if self.sigma_v < 0.0:
            raise ValueError(f"the vol of vol may not be negative, got {self.sigma_v}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"the correlation must lie in [-1, 1], got {self.rho}")

    @property
    def satisfies_feller(self) -> bool:
        """`2*kappa*theta >= sigma_v^2` - whether the variance can reach zero.

        Reported rather than enforced: the equity regime routinely violates it (a steep skew needs a
        large `sigma_v`), so refusing it would refuse the realistic case. What it governs is whether
        a variance *simulation* must handle the zero boundary, which is a question for the engine.
        """
        return 2.0 * self.kappa * self.theta >= self.sigma_v * self.sigma_v

    def integrated_variance(self, years: float) -> float:
        """`E[integral v_t dt]` - exact, and the deterministic variance when `sigma_v = 0`."""
        decay = (1.0 - math.exp(-self.kappa * years)) / self.kappa
        return self.theta * years + (self.v0 - self.theta) * decay


def _deterministic_variance_cf(
    params: HestonParams, u: complex, years: float, carry: float
) -> complex:
    """The `sigma_v = 0` limit, where the variance is deterministic and Black's CF applies.

    Handled as its own branch rather than as a limit of the general formula, because that formula
    divides by `sigma_v^2`. A limit taken numerically would be the degenerate case computed least
    accurately, which is the reverse of what a reduction oracle needs.
    """
    integrated = params.integrated_variance(years)
    return cmath.exp(1j * u * (carry * years - 0.5 * integrated) - 0.5 * u * u * integrated)


def heston_characteristic_function(
    params: HestonParams, u: complex, years: float, carry: float
) -> complex:
    """`phi_T(u) = E[exp(iu*log(S_T/S_0))]` - the STABLE branch.

    `g = (beta - d)/(beta + d)` with `e^{-dT}`: the formulation that stays on one branch of the
    complex logarithm for every maturity. See the module docstring for why the other one does not.
    """
    if years <= 0.0:
        raise ValueError(f"the characteristic function needs a positive horizon, got {years}")
    if params.sigma_v == 0.0:
        return _deterministic_variance_cf(params, u, years, carry)

    sigma2 = params.sigma_v * params.sigma_v
    beta = params.kappa - params.rho * params.sigma_v * 1j * u
    d = cmath.sqrt(beta * beta + sigma2 * (1j * u + u * u))
    # The stable pairing: the MINUS root with `exp(-dT)`, so the exponential decays and the
    # logarithm's argument stays near 1 instead of wandering around the cut.
    numerator = beta - d
    g = numerator / (beta + d)
    decay = cmath.exp(-d * years)

    d_term = numerator / sigma2 * (1.0 - decay) / (1.0 - g * decay)
    c_term = (
        params.kappa
        * params.theta
        / sigma2
        * (numerator * years - 2.0 * cmath.log((1.0 - g * decay) / (1.0 - g)))
    )
    return cmath.exp(c_term + d_term * params.v0 + 1j * u * carry * years)


def naive_characteristic_function(
    params: HestonParams, u: complex, years: float, carry: float
) -> complex:
    """The TRAP branch - `g = (beta + d)/(beta - d)` with `e^{+dT}`. **Never price with this.**

    Kept because the trap oracle needs it: asserting that the stable branch is smooth proves nothing
    unless the same fixture is shown to make this one jump. It is the wrong-variant companion §4
    requires, not a fallback and not dead code.
    """
    if params.sigma_v == 0.0:
        return _deterministic_variance_cf(params, u, years, carry)

    sigma2 = params.sigma_v * params.sigma_v
    beta = params.kappa - params.rho * params.sigma_v * 1j * u
    d = cmath.sqrt(beta * beta + sigma2 * (1j * u + u * u))
    numerator = beta + d
    g = numerator / (beta - d)
    grow = cmath.exp(d * years)

    d_term = numerator / sigma2 * (1.0 - grow) / (1.0 - g * grow)
    c_term = (
        params.kappa
        * params.theta
        / sigma2
        * (numerator * years - 2.0 * cmath.log((1.0 - g * grow) / (1.0 - g)))
    )
    return cmath.exp(c_term + d_term * params.v0 + 1j * u * carry * years)
