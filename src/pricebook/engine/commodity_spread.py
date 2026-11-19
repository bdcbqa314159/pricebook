"""The commodity spread option engine (L4) - one combined volatility, then verified Black.

`CLARK-2014` ch 2.9.1 (Margrabe, **exact**) and 2.9.2 (Kirk, an **approximation**). Both collapse to
1-D Black-76 on a combined volatility, so both reuse `black_price` and no bivariate normal is
needed - `foundation/bivariate_normal.py` exists and is read by five other modules, but a spread
priced this way never reaches for it.

**The source, transcribed from both renderings and reconciled (§4).** pp.74-76:

    Margrabe (K = 0), eq (2.138):
        [sigma^{(1;2)}]^2 = [sigma^(1)]^2 + [sigma^(2)]^2 - 2 rho sigma^(1) sigma^(2)
        V = S0^(1) e^{-r^{f;1} T} N(d1) - S0^(2) e^{-r^{f;2} T} N(d2)

    Kirk (K != 0), eq (2.139) and p.76:
        Y_t = S_t^(2) + K e^{-r^d (T-t)},    sigma~^(2) = sigma^(2) S_t^(2) / Y_t
        sigma_Z = sqrt( [sigma^(1)]^2 + [sigma~^(2)]^2 - 2 rho sigma^(1) sigma~^(2) )
        V0 = e^{-r^d T} [ F^(1) N(d1^Z) - Y0 N(d2^Z) ],   Z0 = F1 / (F2 + K)

**`S0 e^{-r^f T}` is `F df(T)`**, since tranche 60's forward is `F = S e^{(r^d - r^f) T}`. So
Margrabe is exactly `df(T) * black_price(F1, F2, sigma_M^2 T, call)` - Black-76 with the **second
forward as the strike** - and Kirk is the same with the strike `F2 + K`. Clark's own driftless
restatement on p.76 states it in that form, where killing the drift collapses `S` and `F` so his
`Z0 = S0^(1)/(S0^(2)+K)` and the forward ratio are one thing rather than two.

**ONE function serves both, and the reduction is structural rather than a second code path.** Kirk's
weight `w = F2/(F2+K)` is exactly `1` at `K = 0`, where `sigma_Z` becomes `sigma_M` and the strike
becomes `F2`. So the same code is **exact** at zero strike and an **approximation** away from it,
which is a better arrangement than two near-duplicate functions and makes the Margrabe reduction a
property of the arithmetic instead of a thing to keep in step.

**And it is exactly why the reduction needs its companion (§4, T39):** `K = 0` is exact *because* it
kills the weight, which is the one quantity Kirk introduces, so the anchor is structurally blind to
the whole approximation. The finite-strike companion is mandatory, and the oracle measures how far
the weight has moved.

**The weight multiplies BOTH the squared term and the cross term.** `sigma~^(2)` appears in
`[sigma~^(2)]^2` *and* in `-2 rho sigma^(1) sigma~^(2)`; weighting only one is the mutation this most
plausibly attracts, and the oracle measures that the two halves differ.

**Where the correlation is pinned, and why it takes two instruments.** `rho` enters here only
through the product `sigma_1 sigma_2 rho`, so oracles establishing that it is present and correctly
signed would be complete about presence and sign and **silent about magnitude** (T43) - hence the
combined variance is transcribed **by value**. And a price pins only one rho-weighted scalar, so it
sees rho weakly (T17): the companion pins rho where it is directly observable, as the realised
covariance of the simulated log-increments reproducing `sigma_1 sigma_2 rho t`.

Provenance:
  stock:  none - the archived predecessor had no spread option engine
  source: CLARK-2014 (docs/references.md) ch 2.9.1 eq (2.138) and 2.9.2 eqs (2.139)/(2.140) plus
          p.76's sigma_Z and driftless Black-76 restatement; transcribed in both renderings and
          reconciled
  oracle: tests/L4/test_commodity_spread.py - the combined variance transcribed POINTWISE BY VALUE
          over a (rho, strike) grid (T43: presence-and-sign is silent about magnitude); the weight
          asserted to multiply BOTH the squared and the cross term; the rho = +-1 limits
          (`|sigma_1 - sigma_2|` and `sigma_1 + sigma_2`) as requirement oracles pinning the cross
          term's SIGN; monotonicity in rho and non-negativity across the whole range; MARGRABE
          exact against the simulation at 4*SE with NO approximation term; Kirk's error MEASURED
          and growing with the strike, banded as `4*SE + the measured error`; the simulated
          correlation proved unbiased FIRST (T19) by the realised-covariance pin, with the
          error halving as sqrt(N) over six seeds (T21: a null needs averaging too); the zero-short-
          vol reduction to tranche 60's vanilla, a CROSS-PRODUCT check against a different engine;
          put-call parity; invariant 6 with an in-the-money expired fixture
  crosscheck: quantlib KirkSpreadOptionEngine - deferred to the optional QuantLib tier
  slice:  tranche 63 - spread options: Margrabe and Kirk
"""

from __future__ import annotations

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.market.commodity import commodity_forward
from pricebook.models.black import black_price
from pricebook.models.commodity_spread import CommoditySpreadModel
from pricebook.products.commodity_spread import CommoditySpreadOption


def kirk_weight(forward_short: float, strike: float) -> float:
    """`w = F2 / (F2 + K)` - the share of the shifted short leg that is actually stochastic.

    Kirk's device is to treat `S^(2) + K` as a single lognormal; only the `S^(2)` part of it
    diffuses, so its effective volatility is scaled by this weight (`sigma~^(2) = w sigma^(2)`,
    p.76). At `K = 0` the weight is exactly `1` and the method is Margrabe; as the strike grows the
    short leg contributes less volatility, which is also where the approximation decays - Kirk's
    stated assumption is `K << S^(2)`.

    Named surface rather than a local, so a test can transcribe it and the reduction can be asserted
    on the weight itself (the F-6.1 rule).

    The caller is responsible for `F2 + K > 0`; the engine refuses a shifted forward that is not
    positive, since there is then no lognormal to match and the method is undefined.
    """
    return forward_short / (forward_short + strike)


def spread_total_variance(
    vol_long: float, vol_short: float, correlation: float, weight: float, years: float
) -> float:
    """`sigma_Z^2 T = ([s1]^2 + [w s2]^2 - 2 rho s1 (w s2)) T` - p.76, Margrabe's at `weight = 1`.

    The weight is applied to the short-leg volatility **once**, and that weighted value is then used
    in both the squared term and the cross term - which is what the print says and what a half-right
    mutation gets wrong. Written with `weighted` named for exactly that reason: the alternative
    spelling, with `weight` appearing twice inside one expression, invites weighting one term and not
    the other.

    Non-negative for `|rho| <= 1` because the expression is `(s1 - w s2)^2 + 2 s1 (w s2)(1 - rho)`;
    the model type refuses a correlation outside that range, so the square root downstream is always
    defined rather than defensively guarded here.
    """
    weighted = vol_short * weight
    return (
        vol_long * vol_long + weighted * weighted - 2.0 * correlation * vol_long * weighted
    ) * years


@register(CommoditySpreadOption, CommoditySpreadModel)
def price_commodity_spread_option(
    product: CommoditySpreadOption, model: CommoditySpreadModel
) -> PricingResult | PricingFailure:
    """`df(T) * N * Black(F1, F2 + K, sigma_Z^2 T, omega)` - Margrabe at `K = 0`, Kirk otherwise."""
    market = model.market
    currency = product.notional.currency

    # Invariant 6: an expired option is the shell's business, PV 0, checked before any market lookup
    # so a settled trade needs neither leg's data.
    if is_historical(product.expiry, market.valuation_date):
        return PricingResult(pv=Money(0.0, currency))

    legs = product.legs
    for asset in (legs.long, legs.short):
        if asset not in market.commodities:
            return PricingFailure(
                f"no commodity data for {asset.symbol} in the snapshot"
            )
    try:
        discount = market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the spread option: {exc}")

    forward_long = commodity_forward(
        market.commodities[legs.long], product.expiry, discount
    )
    forward_short = commodity_forward(
        market.commodities[legs.short], product.expiry, discount
    )
    shifted = forward_short + product.strike
    if shifted <= 0.0:
        return PricingFailure(
            f"the shifted short forward {forward_short} + {product.strike} = {shifted} is "
            "non-positive; Kirk matches a lognormal to it, so there is nothing to match"
        )

    years = discount.time_measure.year_fraction(product.expiry)
    total_variance = spread_total_variance(
        model.vol_long,
        model.vol_short,
        model.correlation,
        kirk_weight(forward_short, product.strike),
        years,
    )
    undiscounted = black_price(forward_long, shifted, total_variance, product.kind)
    return PricingResult(
        pv=Money(
            discount.df(product.expiry) * product.notional.amount * undiscounted, currency
        )
    )
