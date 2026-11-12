"""The Heston vanilla by Fourier inversion (L4).

    C = df(T) * [ F*P1 - K*P2 ]

with `P2 = Prob(S_T > K)` under the T-forward measure and `P1` the same event under the **share**
measure, each recovered from the characteristic function by the Gil-Pelaez inversion

    P = 1/2 + (1/pi) * integral_0^inf Re[ exp(-iu*k) * phi(u) / (iu) ] du,   k = log(K/S_0)

`P1` uses `phi(u - i)/phi(-i)`, which is the same characteristic function tilted into the share
measure - *not* a second transcription. That matters: a single `phi` means a transcription error
moves both probabilities together and shows up in put-call parity, rather than hiding in one of two
independently written formulas.

**The integrand is regular at `u = 0` even though `1/(iu)` is not.** `phi(u) -> 1` there, so the
numerator vanishes at the same rate and the real part tends to a finite limit; the quadrature starts
just above zero rather than pretending otherwise, and the oracle asserts convergence under
refinement instead of trusting the endpoint.

**Truncation and discretisation are numerical config, not constants** (§3.5): `FourierGrid` carries
them explicitly, in the same shape tranche 31 used for its strike strip. A defaulted upper limit
would silently decide how much of the tail the price contains.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 16.2 - the Fourier inversion of the Heston
          characteristic function; the Gil-Pelaez form of the inversion integral
  oracle: tests/L4/test_heston_fourier.py (PRE-REGISTERED before this module existed) - `P1, P2` in
          `[0,1]` and falling in the strike; the `sigma_v -> 0` reduction to verified Black on the
          integrated variance; put-call parity; the no-arbitrage bounds; the skew's direction under
          `rho < 0`; and the quadrature's own regime (refinement converges, early truncation is
          visible). The non-degenerate anchor is the Heston MC in tests/L4/test_heston_mc.py
  crosscheck: quantlib AnalyticHestonEngine - deferred to the optional QuantLib tier
  slice:  tranche 33 - Heston by Fourier inversion
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from datetime import date

from pricebook.market.curve import CurveHandle
from pricebook.market.equity import EquityUnderlying, dividend_forward
from pricebook.models.heston import HestonParams, heston_characteristic_function

#: Where the quadrature starts, since `1/(iu)` is undefined at 0 while the integrand's limit there
#: is finite. **The omitted sliver `[0, _LOWER]` is a bias, and it is LINEAR in this constant** -
#: measured against the `sigma_v = 0` Black reduction, the relative price error is 1.16e-06 at
#: `1e-6`, 1.16e-08 at `1e-8`, 1.16e-10 at `1e-10` and 1.16e-12 at `1e-12`, reaching the
#: double-precision floor (1.4e-14) by `1e-14`. One-directional and grid-independent, so refining or
#: extending the grid never removes it - which is exactly how it was identified: the reduction's
#: error sat unchanged across three grids and only moved when this did.
#:
#: Set to `1e-12`, where the bias is five orders below the tightest oracle in this tranche. Not
#: `1e-14`: the integrand is evaluated at `u` down there and a smaller start buys nothing once the
#: bias is under the floating-point noise of everything else.
_LOWER = 1e-12


@dataclass(frozen=True)
class FourierGrid:
    """Where the inversion integral is truncated, and how finely it is sampled.

    Both are error sources and they fail differently: too low an `upper` **omits** tail
    contribution (a one-directional bias), too small a `count` adds trapezoid error of either sign.
    The oracle asserts each separately rather than hiding them in one tolerance.
    """

    upper: float
    count: int

    def __post_init__(self) -> None:
        if self.upper <= 0.0:
            raise ValueError(f"the integration upper limit must be positive, got {self.upper}")
        if self.count < 2:
            raise ValueError(f"the quadrature needs at least two nodes, got {self.count}")

    def nodes(self) -> tuple[float, ...]:
        step = (self.upper - _LOWER) / (self.count - 1)
        return tuple(_LOWER + step * k for k in range(self.count))


def heston_probabilities(
    params: HestonParams,
    underlying: EquityUnderlying,
    strike: float,
    expiry: date,
    quadrature: tuple[CurveHandle, FourierGrid],
) -> tuple[float, float]:
    """`(P1, P2)` by Gil-Pelaez inversion - named surface, because the oracles reach for them.

    Returning the probabilities rather than only the price is what lets the suite assert they are
    *probabilities* (in `[0,1]`, falling in the strike) and lets put-call parity be checked from the
    same two numbers the call is built from (§4 - extract the step a test must see).
    """
    discount, grid = quadrature
    years = discount.time_measure.year_fraction(expiry)
    if years <= 0.0:
        raise ValueError(f"the inversion needs a positive horizon, got {years}")

    forward = dividend_forward(underlying, expiry, discount)
    carry = math.log(forward / underlying.spot) / years
    log_strike = math.log(strike / underlying.spot)
    shift = heston_characteristic_function(params, -1j, years, carry)

    def integrands(u: float) -> tuple[float, float]:
        phase = cmath.exp(-1j * u * log_strike)
        plain = heston_characteristic_function(params, complex(u), years, carry)
        tilted = heston_characteristic_function(params, complex(u) - 1j, years, carry) / shift
        return (tilted * phase / (1j * u)).real, (plain * phase / (1j * u)).real

    nodes = grid.nodes()
    values = [integrands(u) for u in nodes]
    first = _trapezoid([v[0] for v in values], nodes)
    second = _trapezoid([v[1] for v in values], nodes)
    return 0.5 + first / math.pi, 0.5 + second / math.pi


def _trapezoid(values: list[float], nodes: tuple[float, ...]) -> float:
    area = 0.0
    for i in range(len(nodes) - 1):
        area += 0.5 * (values[i] + values[i + 1]) * (nodes[i + 1] - nodes[i])
    return area


def heston_call_price(
    params: HestonParams,
    underlying: EquityUnderlying,
    strike: float,
    expiry: date,
    quadrature: tuple[CurveHandle, FourierGrid],
) -> float:
    """`df(T) * (F*P1 - K*P2)` - the European call under Heston."""
    discount, _ = quadrature
    first, second = heston_probabilities(params, underlying, strike, expiry, quadrature)
    forward = dividend_forward(underlying, expiry, discount)
    return discount.df(expiry) * (forward * first - strike * second)
