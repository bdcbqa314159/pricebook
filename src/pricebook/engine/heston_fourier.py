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

**The inversion itself moved to `engine/fourier_inversion.py` at tranche 49**, when SVJ became its
second consumer; `FourierGrid` and `_LOWER` went with it, and this module keeps what is specific to
Heston — binding the characteristic function, and the `(product, model)` registration. The move was
verbatim and is proved bit-identical on 120 reference values, so every number here is the number it
was at tranche 33.

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

import math
from datetime import date

from pricebook.engine.fourier_inversion import FourierGrid, gil_pelaez_probabilities
from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.market.curve import CurveHandle
from pricebook.market.equity import EquityUnderlying, dividend_forward
from pricebook.models.heston import HestonModel, HestonParams, heston_characteristic_function
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption

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
    def characteristic(u: complex) -> complex:
        return heston_characteristic_function(params, u, years, carry)

    return gil_pelaez_probabilities(characteristic, log_strike, grid)


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


#: The inversion grid the registered pricer uses. Numerical config with a *default* here rather than
#: on the function, because `dispatch` passes only `(product, model)`: the grid is part of how this
#: engine computes, and tranche 31's `(market, grid)` model shape is the alternative if a caller
#: ever needs to vary it. Wide and fine enough that truncation sits far below the oracles' bands.
_DEFAULT_GRID = FourierGrid(upper=200.0, count=4001)


@register(EuropeanEquityOption, HestonModel)
def price_european_equity_option_heston(
    product: EuropeanEquityOption, model: HestonModel
) -> PricingResult | PricingFailure:
    """The equity vanilla under Heston - **the registration that was impossible before T33b.**

    On the product-only registry this decorator would have *replaced* tranche 30's Black pricer,
    silently repricing every equity vanilla in the tree. The `(product, model)` key is what lets the
    two coexist, each reached by the model it was written for.
    """
    market = model.market
    currency = product.notional.currency
    if is_historical(product.expiry, market.valuation_date):
        return PricingResult(pv=Money(0.0, currency))

    underlying = market.equities.get(product.asset)
    if underlying is None:
        return PricingFailure(f"no equity data for {product.asset.ticker} in the snapshot")
    try:
        discount = market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the equity option under Heston: {exc}")

    call = heston_call_price(
        model.params, underlying, product.strike, product.expiry, (discount, _DEFAULT_GRID)
    )
    if product.kind is EquityOptionKind.PUT:
        # Put-call parity rather than a second inversion: the identity is exact and model-free, so
        # the put inherits the call's accuracy instead of acquiring its own quadrature error.
        forward = dividend_forward(underlying, product.expiry, discount)
        call -= discount.df(product.expiry) * (forward - product.strike)
    return PricingResult(pv=Money(product.notional.amount * call, currency))
