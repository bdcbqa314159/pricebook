"""The geometric-average commodity option engine (L4) - two adjustments, then verified Black.

**Exact, not approximate, and that is the whole reason this is a closed form.** The geometric
average of lognormals is itself lognormal, so an average-rate option on it is Black-76 on an
**adjusted forward** and an **adjusted variance** - nothing is approximated. (The arithmetic average
is not lognormal, which is why it will need a moment match.) The only new code here is the two
adjustments; `black_price` is reused whole, exactly as tranche 60's European engine does.

**The source, transcribed from both renderings and reconciled (§4).** `CLARK-2014` pp.63-65, with
`r^d` the risk-free rate and `r^f` the convenience yield (his FX analogy, carried from tranche 60):

    ln A_G = ln S_0 + (1/n) sum_i (r^d - r^f - 1/2 sigma^2) t_i
                    + (1/n) sum_i sigma W_{t_i}                            (2.111)

    Var[ln A_G] = (sigma^2 / n^2) sum_{i,j=1..n} min(t_i, t_j)
                = (sigma^2 / n^2) sum_{i=1..n} t_i [2(n-i)+1]              p.64

**The source prints the load-bearing quantity three ways, which is stronger than two renderings of
one line.** The variance appears as the `min(t_i,t_j)` double sum, as its sorted single-sum
evaluation, and - for an equally-spaced grid - as the closed form `sigma^2 T [1 - (1/6)(4n+1)(n-1)/n^2]`
(2.114b), whose `n -> infinity` limit is `sigma^2 T / 3` (2.115b). Three different computation paths
for one number: a misprint would have to corrupt all three identically to survive. They agree to
1e-13 across `n = 1..60`.

**Built on the GENERAL forms, which is what makes the specialisations independent checks.** Real
fixing schedules are not equally spaced, and Clark says the uniform grid is "purely to simplify the
algebra below (the computations can easily be performed for arbitrary fixing schedules)". So the
double sum is what is implemented, and (2.114a/b) are oracles rather than restatements of the code.

**The adjusted forward.** `A_G` is lognormal, so `F_G = E[A_G] = exp(E[ln A_G] + 1/2 Var[ln A_G])`.
Written through tranche 60's `commodity_forward` per fixing, so the carry is defined in exactly one
place: the log-mean is the average of the fixing forwards' logs, less `1/2 sigma^2 tbar`, and the
convexity `+1/2 sigma_G^2` is added back. At `n = 1` the two corrections cancel exactly and `F_G`
collapses to `F(t_1)` - the reduction anchor, exact by construction.

**Why it sits BELOW the geometric mean of the forwards**, which is a requirement oracle rather than
a transcription: `sigma_G^2 <= sigma^2 tbar` because averaging destroys variance, so the net
convexity `1/2 (sigma_G^2 - sigma^2 tbar)` is non-positive, with equality only when the fixings
coincide.

**No historical fixings - and the source scopes itself the same way.** Clark restricts to "the case
where no historical fixings have taken place yet (i.e. with `j = 0` in his notation)", and so does
this engine. A partly-fixed Asian needs the realised prints, which the snapshot does not carry for a
commodity; pricing it as though unfixed would be a plausible wrong number, so it is refused in
domain vocabulary. A **fully** settled one is invariant 6's business and is PV 0.

Provenance:
  stock:  none - the archived predecessor had no Asian engine
  source: CLARK-2014 (docs/references.md) ch 2.7.3 (Kemna and Vorst, 1990) - eq (2.111), the
          variance double sum and its sorted evaluation (p.64), and the equally-spaced
          specialisations (2.114a/b) with their continuous limit (2.115b); transcribed in both
          renderings and reconciled
  oracle: tests/L4/test_commodity_geometric_asian.py - the variance's THREE independent printed
          forms required to agree; `F_G` and `sigma_G^2` transcribed POINTWISE against references
          the test computes itself (the F-6.1 lesson - a formula living only inside a composite is
          never pinned, so both are named surface); the n=1 reduction to the tranche-60 European,
          EXACT, with a MANDATORY companion non-degenerate in fixing count and spacing (T39 - the
          reduction collapses the averaging and is blind to the whole double sum); an MC companion
          on the real geometric average, banded at 4*SE with no approximation term because the
          closed form is exact, and its resolution measured; averaging reduces variance with
          equality iff the fixings coincide; put-call parity swept off the money
  crosscheck: quantlib AnalyticDiscreteGeometricAveragePriceAsianEngine - deferred to the optional
          QuantLib tier
  slice:  tranche 61 - geometric Asian options, exact closed form
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.market.commodity import CommodityUnderlying, commodity_forward
from pricebook.market.curve import CurveHandle
from pricebook.models.black import black_price
from pricebook.models.commodity_black import CommodityBlackModel
from pricebook.products.commodity_asian import CommodityGeometricAsian


def geometric_average_variance(fixing_years: Sequence[float], volatility: float) -> float:
    """`Var[ln A_G] = (sigma^2/n^2) sum_{i,j} min(t_i,t_j)` - CLARK-2014 p.64.

    The **double sum** is implemented rather than its sorted evaluation, because the double sum is
    the general statement and the single sum assumes an ascending schedule. It is `O(n^2)` in the
    fixing count, which for a real averaging schedule (tens of dates, not thousands) is nothing, and
    the clarity is worth more than the loop: this is the covariance of the log-prices, written as
    the covariance of the log-prices.

    Named surface rather than a local inside the pricer, so a test can transcribe it pointwise -
    the F-6.1 lesson, where a formula that only ever appeared inside a composite was never pinned.
    """
    n = len(fixing_years)
    if n == 0:
        raise ValueError("the geometric average needs at least one fixing time")
    return (
        volatility
        * volatility
        / (n * n)
        * math.fsum(min(a, b) for a in fixing_years for b in fixing_years)
    )


def geometric_average_forward(
    underlying: CommodityUnderlying,
    fixings: Sequence[date],
    discount: CurveHandle,
    volatility: float,
) -> float:
    """`F_G = E[A_G] = exp(E[ln A_G] + 1/2 Var[ln A_G])` - the lognormal mean of the average.

    Composed from tranche 60's `commodity_forward` per fixing, so the cost of carry and the
    convenience yield are defined in exactly one place and this function adds only the averaging.
    Writing `ln F(t_i) = ln S_0 + (r - delta) t_i`, eq (2.111)'s drift is the mean of those logs
    less `1/2 sigma^2 tbar`, and the lognormal mean adds `1/2 sigma_G^2` back.

    At `n = 1` the two corrections are `-1/2 sigma^2 t_1` and `+1/2 sigma^2 t_1` and cancel
    **exactly in R**, so `F_G = F(t_1)` identically. In floating point they do not quite: this
    composition evaluates `exp(log F - a + a)` where tranche 60 evaluates `F` directly, so a
    log/exp round trip stands between them and the two differ by **1 ULP (2.22e-16 relative)**.
    That is the §4 distinction between an exact-arithmetic identity and bit-identity, and claiming
    the latter here would be false - the ops are legitimately reordered. The tell that the
    cancellation really is algebraic is that the gap does **not depend on the volatility**: a
    defect in either half of the pair would scale with `sigma^2`, and the oracle asserts the
    invariance rather than just the smallness.
    """
    years = [discount.time_measure.year_fraction(f) for f in fixings]
    n = len(years)
    if n == 0:
        raise ValueError("the geometric average needs at least one fixing date")
    log_mean = (
        math.fsum(math.log(commodity_forward(underlying, f, discount)) for f in fixings) / n
    )
    mean_time = math.fsum(years) / n
    variance = geometric_average_variance(years, volatility)
    return math.exp(log_mean - 0.5 * volatility * volatility * mean_time + 0.5 * variance)


@register(CommodityGeometricAsian, CommodityBlackModel)
def price_commodity_geometric_asian(
    product: CommodityGeometricAsian, model: CommodityBlackModel
) -> PricingResult | PricingFailure:
    """`df(t_n) * N * Black(F_G, K, sigma_G^2, omega)`."""
    market = model.market
    currency = product.notional.currency
    settlement = product.fixings[-1]

    # Invariant 6: every fixing settled means the trade is the shell's business, PV 0 - checked
    # before any market lookup, so a settled trade needs no commodity data at all.
    if is_historical(settlement, market.valuation_date):
        return PricingResult(pv=Money(0.0, currency))

    # A PARTLY fixed schedule is a different instrument: its price depends on prints the snapshot
    # does not carry. Treating it as unfixed is a plausible wrong number, so it is refused. The
    # source draws the same line - Clark prices "the case where no historical fixings have taken
    # place yet".
    fixed = [f for f in product.fixings if is_historical(f, market.valuation_date)]
    if fixed:
        return PricingFailure(
            f"{len(fixed)} of {len(product.fixings)} fixings are historical (through "
            f"{fixed[-1]}); a partly-fixed average needs the realised prints, which this "
            "snapshot does not carry"
        )

    underlying = market.commodities.get(product.asset)
    if underlying is None:
        return PricingFailure(
            f"no commodity data for {product.asset.symbol} in the snapshot"
        )
    try:
        discount = market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the average-rate option: {exc}")

    years = [discount.time_measure.year_fraction(f) for f in product.fixings]
    forward = geometric_average_forward(underlying, product.fixings, discount, model.volatility)
    variance = geometric_average_variance(years, model.volatility)
    undiscounted = black_price(forward, product.strike, variance, product.kind)
    return PricingResult(
        pv=Money(discount.df(settlement) * product.notional.amount * undiscounted, currency)
    )
