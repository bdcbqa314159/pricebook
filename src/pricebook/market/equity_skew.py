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
    """`sigma(K) = atm + slope*log(K/F)`, floored positive. `slope < 0` is the typical equity skew."""

    atm_volatility: float
    slope: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.atm_volatility) or self.atm_volatility < 0.0:
            raise ValueError(
                f"the atm volatility must be finite and non-negative, got {self.atm_volatility}"
            )
        if not math.isfinite(self.slope):
            raise ValueError(f"the skew slope must be finite, got {self.slope}")

    def volatility(self, strike: float, forward: float) -> float:
        """The volatility at `strike`, read in log-moneyness against `forward`."""
        if strike <= 0.0 or forward <= 0.0:
            raise ValueError(f"strike and forward must be positive, got {strike}, {forward}")
        return max(self.atm_volatility + self.slope * math.log(strike / forward), _VOLATILITY_FLOOR)
