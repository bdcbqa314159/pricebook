"""The defaultable-equity vanilla (L4) - a modified Black formula, Brockhaus 13.1.1.

With `S_t = S~_t 1{tau>t}` and `tau` independent of the diffusion, the option prices collapse to
Black on a **scaled strike**:

    E[(S_t - K)^+] = p BSCall(F/p, K, s, t) = BSCall(F, p K, s, t)
    E[(K - S_t)^+] = (1-p)K + p BSPut(F/p, K, s, t) = (1-p)K + BSPut(F, p K, s, t)

**The book prints both forms, and this module uses the scaled-STRIKE one.** They are equal by
Black's degree-1 homogeneity in `(forward, strike)` - which is itself a source-level check, since
the text supplies both sides rather than leaving the identity to be invented. The scaled-strike form
is preferred for two reasons that are not stylistic: it never divides by `p`, which goes to zero as
the hazard grows, and at `p = 1` it is **literally** the default-free call, so the reduction is
bit-identical rather than merely close.

**The put's `(1-p)K` is the default lump**, the strike collected when the stock is worthless, and it
is what makes put-call parity survive: `BSCall(F,pK) - (1-p)K - BSPut(F,pK) = (F - pK) - (1-p)K =
F - K`. The survival scaling and the lump cancel exactly, so parity holds for **every** `p` - which
makes it blind to a slip in both formulas at once and sharp against a slip in either one.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 13.1.1 - eq (13.3) and the two modified
          Black-Scholes formulas, both forms, read in `-raw` and `-layout` and reconciled
  oracle: tests/L4/test_defaultable_equity.py (PRE-REGISTERED) - the source's own two forms agreed
          to 1e-14 (the homogeneity identity); the `h -> 0` reduction **bit-identical** to tranche
          30, with the mandatory companion moving call and put in opposite directions; put-call
          parity with the default terms cancelling; the `p -> 0` limit (worthless call, strike-worth
          put); and a NON-FLAT survival curve read beyond its first segment, so a mis-integrated
          hazard is separated (the tranche-25 degeneracy)
  crosscheck: quantlib (defaultable equity) - deferred to the optional QuantLib tier
  slice:  tranche 35 - defaultable equity, the equity-credit seam
"""

from __future__ import annotations

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.market.equity import dividend_forward
from pricebook.models.black import black_price
from pricebook.models.defaultable_equity import DefaultableEquityModel
from pricebook.products.equity_option import EquityOptionKind, EuropeanEquityOption


def surviving_forward(forward: float, survival: float) -> float:
    """`F~(t) = F(t)/p(t)` - eq (13.3), the forward of the process conditioned on survival.

    Exposed because it is the identity that pins the scaling independently of the volatility:
    `E[S_t] = p F~ = F`, the default-risky forward is the plain forward. The engine itself does not
    use it - the scaled-strike form avoids the division - but the oracle reaches for it.
    """
    if survival <= 0.0:
        raise ValueError(f"the surviving forward needs a positive survival probability, got {survival}")
    return forward / survival


def defaultable_call(
    forward: float, strike: float, survival: float, volatility: float, years: float
) -> float:
    """`BSCall(F, p K, s, t)` - undiscounted."""
    return black_price(forward, survival * strike, volatility * volatility * years, EquityOptionKind.CALL)


def defaultable_put(
    forward: float, strike: float, survival: float, volatility: float, years: float
) -> float:
    """`(1-p)K + BSPut(F, p K, s, t)` - undiscounted, the default lump plus the scaled-strike put."""
    return (1.0 - survival) * strike + black_price(
        forward, survival * strike, volatility * volatility * years, EquityOptionKind.PUT
    )


@register(EuropeanEquityOption, DefaultableEquityModel)
def price_european_equity_option_defaultable(
    product: EuropeanEquityOption, model: DefaultableEquityModel
) -> PricingResult | PricingFailure:
    """The modified Black formula, discounted - the fourth pricer on the equity vanilla."""
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
        return PricingFailure(f"cannot price the defaultable equity option: {exc}")

    years = discount.time_measure.year_fraction(product.expiry)
    forward = dividend_forward(underlying, product.expiry, discount)
    survival = model.survival.survival(years)
    undiscounted = (
        defaultable_call(forward, product.strike, survival, model.volatility, years)
        if product.kind is EquityOptionKind.CALL
        else defaultable_put(forward, product.strike, survival, model.volatility, years)
    )
    return PricingResult(
        pv=Money(discount.df(product.expiry) * product.notional.amount * undiscounted, currency)
    )
