"""The variance swap's static replication (L4) - a `1/K^2`-weighted strip of vanillas.

    K_var = (2/T) * [ integral_0^F P(K)/K^2 dK + integral_F^inf C(K)/K^2 dK ] / df(T)

**Model-free, which is the whole reason this instrument matters.** No volatility dynamics enter the
price: the fair variance is a portfolio of traded options, so a variance swap can be hedged and
valued without committing to a model. The volatility skew appears only as the *source* of the option
prices, never as an assumption about how they evolve.

**Puts below the forward, calls above it.** Both are OTM there, which is where the options actually
trade and where the quoted skew is meaningful; using ITM options instead would be arithmetically
equivalent in a parity-consistent world and badly conditioned in a real one. At `K = F` the two
coincide exactly (put-call parity gives `C - P = df*(F - K) = 0`), so the split point carries no
discontinuity.

**The discount factor cancels, and the code says so rather than performing it.** The textbook form
multiplies by `e^{rT}` and integrates *present-value* option prices. Our `black_price` is already
undiscounted, so `(2/(T*df)) * integral df*black/K^2 dK` is `(2/T) * integral black/K^2 dK` exactly -
the `df` round-trip is written out of the formula instead of multiplied and divided back, which keeps
one fewer operation between the source and the number.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 3.7 / 5.8 - the variance swap's static replication
          by a `1/K^2`-weighted strip and its log-contract form
  oracle: tests/L4/test_variance_swap.py - **the cross-product reduction**: under a flat vol the
          strip integrates to exactly `sigma^2`, with the strip integrator and `black_price` being
          different code; the **log-contract** identity `-(2/T)E[log(S_T/F)]` by a third route
          (numerical integration over the lognormal density, pinning the `-2` and the `2/T`); the
          grid's **validity regime** (a truncated grid underprices, widening converges upward,
          refining improves at the trapezoid rate); and the mandatory **skew companion** - a flat
          surface satisfies the reduction trivially, so a material skew must lift `K_var` off the
          ATM variance by the convexity premium
  crosscheck: quantlib VarianceSwap / replicating-portfolio engine - deferred to the optional
          QuantLib tier
  slice:  tranche 31 - the variance swap by static replication
"""

from __future__ import annotations

from datetime import date

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.market.curve import CurveHandle
from pricebook.market.equity import EquityUnderlying, dividend_forward
from pricebook.market.equity_skew import ImpliedVolSkew
from pricebook.models.black import black_price
from pricebook.models.variance_replication_model import StrikeGrid, VarianceReplicationModel
from pricebook.products.equity_option import EquityOptionKind
from pricebook.products.variance_swap import VarianceSwap


def fair_variance(
    underlying: EquityUnderlying,
    expiry: date,
    discount: CurveHandle,
    skew: ImpliedVolSkew,
    grid: StrikeGrid,
) -> float:
    """`K_var` - the fair variance, as the `1/K^2`-weighted strip of OTM vanillas.

    Named surface rather than a step inside the registered pricer, because every oracle in this
    tranche reaches for *this* number: the reduction to `sigma^2`, the log-contract identity and the
    grid's convergence are all statements about the strip, not about a swap's PV (§4 - extract what
    the oracle must reach).
    """
    forward = dividend_forward(underlying, expiry, discount)
    years = discount.time_measure.year_fraction(expiry)
    if years <= 0.0:
        raise ValueError(f"the fair variance needs a positive horizon, got {years}")

    strikes = grid.strikes(forward)
    weighted = []
    for strike in strikes:
        volatility = skew.volatility(strike, forward, years)
        kind = EquityOptionKind.PUT if strike < forward else EquityOptionKind.CALL
        undiscounted = black_price(forward, strike, volatility * volatility * years, kind)
        weighted.append(undiscounted / (strike * strike))

    area = 0.0
    for i in range(len(strikes) - 1):
        area += 0.5 * (weighted[i] + weighted[i + 1]) * (strikes[i + 1] - strikes[i])
    return 2.0 / years * area


@register(VarianceSwap, VarianceReplicationModel)
def price_variance_swap(
    product: VarianceSwap, model: VarianceReplicationModel
) -> PricingResult | PricingFailure:
    """`N * df(T) * (K_var - K)` - the discounted variance difference."""
    market = model.market
    currency = product.variance_notional.currency

    # Invariant 6: a matured swap is the shell's business, PV 0 before any market lookup.
    if is_historical(product.expiry, market.valuation_date):
        return PricingResult(pv=Money(0.0, currency))

    underlying = market.equities.get(product.asset)
    skew = market.equity_skews.get(product.asset)
    if underlying is None or skew is None:
        return PricingFailure(
            f"no equity underlying and skew for {product.asset.ticker} in the snapshot"
        )
    try:
        discount = market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the variance swap: {exc}")

    variance = fair_variance(underlying, product.expiry, discount, skew, model.grid)
    payoff = product.variance_notional.amount * (variance - product.variance_strike)
    return PricingResult(pv=Money(discount.df(product.expiry) * payoff, currency))
