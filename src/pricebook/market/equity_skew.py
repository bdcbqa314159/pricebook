"""A minimal implied-volatility skew (L1) - the price source a replication integrates over.

The variance swap needs option prices **across strikes**, so it needs a volatility per strike. That
is all this is: a two-parameter log-moneyness skew, the smallest thing that makes the strip
non-trivial.

    sigma(K) = atm_volatility + slope * log(K / F)

**Deliberately not a surface.** No term structure, no calibration, no arbitrage-free interpolation -
those arrive with tranche 32, pulled by a product that needs them. A skew is what tranche 31's strip
consumes, and building the surface first would be the speculation §6 forbids. It is also why this is
L1 market data rather than a model: it is a *quote* convention (how the market's vols vary by
strike), and the replication that reads it is model-free.

**The floor is load-bearing, not defensive.** A linear skew extrapolated far enough produces a
negative volatility, and the strip integrates to a strike of `0.05 F` - deep in that territory. A
negative `sigma` would reach `black_price` as a negative `total_variance`, which its documented
contract treats as the deterministic case and returns an intrinsic for: a silently wrong price
rather than an error. So the skew floors at a small positive volatility, where the option is
worthless anyway and the strip's `1/K^2` weight is largest.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 4.1 - the implied-volatility skew parametrization
          in log-moneyness
  oracle: tests/L4/test_variance_swap.py - the skew is exercised as the variance swap's price
          source: a flat skew (slope 0) must reproduce `sigma^2` exactly through the strip, and a
          material skew must lift the fair variance off the ATM variance by the convexity premium,
          which is the behaviour only a correctly weighted strip produces
  crosscheck: none - a quote-side parametrization, not a numerical convention a library implements
  slice:  tranche 31 - the variance swap: the strip's price source
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Below this the option is worthless at any strike the strip visits, so flooring here changes no
#: price while keeping `black_price` out of its deterministic branch (see the module docstring).
_VOLATILITY_FLOOR = 1e-6


@dataclass(frozen=True)
class ImpliedVolSkew:
    """`sigma(K,T) = atm + term_slope*T + slope*log(K/F(T))`, floored positive.

    `slope < 0` is the typical equity skew; `term_slope` is the term structure (ch 4.3), zero by
    default because tranche 31's replication reads a single expiry and the term dimension arrived
    with its second consumer - tranche 32's Dupire, which differentiates in `T` and would have
    nothing to differentiate on a flat-in-maturity surface.

    **It stays a closed form, and that is now load-bearing rather than merely tidy.** Dupire needs
    `dC/dT`, `dC/dK` and `d2C/dK2`, and tranche 32 takes them **analytically** by the chain rule -
    so this object must supply its own exact derivatives. A discrete node map could not (that is
    exactly why tranche 9's `ImpliedVolSurface` does not generalise here: it refuses to interpolate,
    so it has no derivatives to take).
    """

    atm_volatility: float
    slope: float
    term_slope: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.atm_volatility) or self.atm_volatility < 0.0:
            raise ValueError(
                f"the atm volatility must be finite and non-negative, got {self.atm_volatility}"
            )
        for name, value in (("skew slope", self.slope), ("term slope", self.term_slope)):
            if not math.isfinite(value):
                raise ValueError(f"the {name} must be finite, got {value}")

    def _raw(self, strike: float, forward: float, years: float) -> float:
        return self.atm_volatility + self.term_slope * years + self.slope * math.log(strike / forward)

    def volatility(self, strike: float, forward: float, years: float = 0.0) -> float:
        """The volatility at `(strike, years)`, read in log-moneyness against `forward`."""
        if strike <= 0.0 or forward <= 0.0:
            raise ValueError(f"strike and forward must be positive, got {strike}, {forward}")
        return max(self._raw(strike, forward, years), _VOLATILITY_FLOOR)

    def is_floored(self, strike: float, forward: float, years: float) -> bool:
        """Whether the floor is binding here - where the exact derivatives below stop applying.

        The floor makes `sigma` piecewise, so its derivatives are zero in the floored region and the
        parametrization's elsewhere. A caller taking analytic derivatives has to know which side it
        is on rather than differentiate through a `max`, which is the kind of silent kink tranche 31
        measured the cost of.
        """
        return self._raw(strike, forward, years) < _VOLATILITY_FLOOR

    def d_volatility_d_strike(self, strike: float, forward: float, years: float) -> float:
        """`dsigma/dK = slope / K` - exact."""
        return 0.0 if self.is_floored(strike, forward, years) else self.slope / strike

    def d2_volatility_d_strike2(self, strike: float, forward: float, years: float) -> float:
        """`d2sigma/dK2 = -slope / K^2` - exact."""
        return 0.0 if self.is_floored(strike, forward, years) else -self.slope / (strike * strike)

    def d_volatility_d_time(
        self, strike: float, forward: float, years: float, carry: float
    ) -> float:
        """`dsigma/dT` at **fixed strike** - two terms, not one.

        The explicit `term_slope`, **plus** the skew read against a forward that itself moves with
        `T`: `log(K/F(T))` with `dlog F/dT = carry = r - q` contributes `-slope * carry`. So a
        surface skewed in log-moneyness is *implicitly* time-dependent even when `term_slope = 0`,
        and dropping that second term is a silent error on any skewed surface with a non-zero carry.

        The caller passes `carry` rather than this object assuming zero, because the forward's drift
        belongs to the underlying (dividends, repo, rates) and not to the vol parametrization - the
        surface cannot know it, and guessing it would be the kind of hidden assumption §4 keeps
        finding on the wrong side of an oracle.
        """
        if self.is_floored(strike, forward, years):
            return 0.0
        return self.term_slope - self.slope * carry
