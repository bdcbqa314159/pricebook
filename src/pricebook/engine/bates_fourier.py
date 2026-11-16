"""The SVJ vanilla by Fourier inversion (L4). `GATHERAL-2006` ch 5.

**Tranche 33's engine, reused whole.** The only thing SVJ changes is the characteristic function it
inverts, so this module binds `phi_SVJ` and hands it to `gil_pelaez_probabilities`; the quadrature,
the share-measure tilt and the `df*(F*P1 - K*P2)` assembly are the ones tranche 33 wrote and are
proved unchanged by it (the inversion was extracted at this tranche and is bit-identical on 120
Heston references — `tests/L4/test_fourier_inversion.py`).

The one-line assembly below is duplicated from `heston_fourier.py` rather than abstracted: §6 waits
for the third occurrence, and two callers of `df * (forward * first - strike * second)` is
coincidence, not a shape.

Provenance:
  stock:  none - the archived predecessor had no SVJ model
  source: GATHERAL-2006 (docs/references.md) ch 5 p.65 - the SVJ characteristic function; the
          inversion itself is BROCKHAUS-2016 ch 16.2 by way of tranche 33
  oracle: tests/L4/test_bates_svj.py - the double reduction (Heston at zero intensity, the Merton
          series at zero vol-of-vol), put-call parity, an independent Monte-Carlo of the
          jump-diffusion banded at 4*SE with the inversion's own error measured separately, and the
          cross-arc tie-back that an SVI fit to an SVJ smile has a steeper short-dated skew than
          one fitted to a Heston smile
  crosscheck: quantlib BatesEngine - deferred to the optional QuantLib tier
  slice:  tranche 49 - the SVJ/Bates model
"""

from __future__ import annotations

import math
from datetime import date

from pricebook.engine.fourier_inversion import (
    FourierGrid,
    gil_pelaez_probabilities,
)
from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.market.curve import CurveHandle
from pricebook.market.equity import EquityUnderlying, dividend_forward
from pricebook.models.bates import BatesModel, BatesParams, bates_characteristic_function
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption

#: The inversion grid the registered pricer uses. **Wider than tranche 33's `upper=200`, and the
#: difference is structural rather than cautious.** Jumps fatten the tails of the return
#: distribution, so `phi_SVJ` decays more slowly in `u` than `phi_Heston` does — and the effect is
#: worst at *short* maturity, where the diffusion has contributed least. Measured against an
#: `upper=1200, count=24001` reference at 60 days: `upper=200` is wrong by **2.9e-05** relative,
#: `upper=400` by **2.7e-11**. At 730 days both sit at 1e-14, which is why a Heston-shaped default
#: looks fine until a short-dated option asks for a price. Asserted in
#: `tests/L4/test_bates_svj.py::TestTheInversionIsConverged`.
_DEFAULT_GRID = FourierGrid(upper=400.0, count=4001)


def bates_probabilities(
    params: BatesParams,
    underlying: EquityUnderlying,
    strike: float,
    expiry: date,
    quadrature: tuple[CurveHandle, FourierGrid],
) -> tuple[float, float]:
    """`(P1, P2)` under SVJ — the Heston path with `phi_SVJ` bound in place of `phi_Heston`."""
    discount, grid = quadrature
    years = discount.time_measure.year_fraction(expiry)
    if years <= 0.0:
        raise ValueError(f"the inversion needs a positive horizon, got {years}")

    forward = dividend_forward(underlying, expiry, discount)
    carry = math.log(forward / underlying.spot) / years
    log_strike = math.log(strike / underlying.spot)

    def characteristic(u: complex) -> complex:
        return bates_characteristic_function(params, u, years, carry)

    return gil_pelaez_probabilities(characteristic, log_strike, grid)


def bates_call_price(
    params: BatesParams,
    underlying: EquityUnderlying,
    strike: float,
    expiry: date,
    quadrature: tuple[CurveHandle, FourierGrid],
) -> float:
    """`df(T) * (F*P1 - K*P2)` - the European call under SVJ."""
    discount, _ = quadrature
    first, second = bates_probabilities(params, underlying, strike, expiry, quadrature)
    forward = dividend_forward(underlying, expiry, discount)
    return discount.df(expiry) * (forward * first - strike * second)


@register(EuropeanEquityOption, BatesModel)
def price_european_equity_option_bates(
    product: EuropeanEquityOption, model: BatesModel
) -> PricingResult | PricingFailure:
    """The equity vanilla under SVJ, reached by the `(product, model)` key tranche 33b built."""
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
        return PricingFailure(f"cannot price the equity option under SVJ: {exc}")

    call = bates_call_price(
        model.params, underlying, product.strike, product.expiry, (discount, _DEFAULT_GRID)
    )
    forward = dividend_forward(underlying, product.expiry, discount)
    if product.kind is EquityOptionKind.CALL:
        unit = call
    else:
        unit = call - discount.df(product.expiry) * (forward - product.strike)
    return PricingResult(pv=Money(product.notional.amount * unit, currency))
