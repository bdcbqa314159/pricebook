"""The Schwartz one-factor commodity option engine (L4). `CLARK-2014` ch 2.10.1.1.

Two closed forms and then verified Black, like every engine in this arc - but the two closed forms
here are the model itself rather than a carry adjustment.

**The source, transcribed from both renderings and reconciled (§4).** Clark pp.81-82:

    Var[X_T] = (sigma^2 / (2 kappa)) (1 - e^{-2 kappa T})                       (2.150b)
    F(0,T)   = S0^{e^{-kappa T}} exp( alpha_hat (1 - e^{-kappa T})
                                      + (sigma^2 / (4 kappa)) (1 - e^{-2 kappa T}) )   (2.156)
    V        = omega e^{-r T} [ F(0,T) N(omega d1) - K N(omega d2) ]               (2.151a)
    d_{1,2}  = [ ln(F/K) +- (1/2) sigma_{0,T}^2 T ] / (sigma_{0,T} sqrt(T))        (2.151b)

**`X_T` is Gaussian, so the option is Black-76 on the model's own forward and variance** - which is
why `black_price` is reused whole and (2.150b) is fed to it as the **total** variance, not as an
annualised one. Clark's `sigma_{0,T}^2 T` *is* `Var[X_T]`; the text says "where, from (2.150b)".

**The `sigma^2/(4 kappa)` in the futures price is half the variance, and that is not a coincidence.**
`E[S_T] = exp(E[X_T] + (1/2) Var[X_T])` is the lognormal mean (Clark quotes it as (2.152)), so the
futures price carries exactly half of (2.150b). A reader checking the two formulae against each
other should see the `4 kappa` as `2 * 2 kappa` rather than as an independent constant - and the
oracle asserts the relation rather than only the two values.

**The kappa -> 0 limit does NOT give the cost-of-carry forward, and the tranche-64 spec's claim that
it did is corrected here.** There is no `r - delta` anywhere in (2.147): the Schwartz drift is pure
mean reversion. So as `kappa -> 0` the log-spot becomes driftless arithmetic Brownian motion and

    Var[X_T] -> sigma^2 T          and          F(0,T) -> S0 exp(sigma^2 T / 2)

Measured on the oracle's fixture the limit forward is 88.5136 against tranche 60's carry forward of
83.9785 - different models, not a discrepancy. Clark is explicit on p.82 that matching market
forwards means **calibrating** `alpha_hat` ("to choose `alpha_X` such that `E^d[S_T] = F_{0,T}`"),
which this tranche does not build (nothing pulls it yet, §6).

The cross-engine anchor the spec wanted does survive, with the true condition attached: the two
agree exactly when tranche 60's carry reproduces the `kappa -> 0` Schwartz drift, i.e. at
`delta = r - sigma^2/2`. The oracle asserts that, and asserts the two **disagree** at a genuine
`kappa` so the anchor is not vacuous.

**The simulator is tranche 57's, and the reuse is exact.** The LGM state is `z_T ~ N(0, zeta(T))`
with `zeta(T) = sigma^2 (e^{2 kappa T} - 1)/(2 kappa)`, so

    Var(e^{-kappa T} z_T) = e^{-2 kappa T} zeta(T) = sigma^2 (1 - e^{-2 kappa T})/(2 kappa)

is identically (2.150b) - the same `X_hat_t = e^{kappa t} X_t` integrating factor Clark uses to
integrate the SDE. The Schwartz state is the LGM state scaled by `e^{-kappa T}` plus a deterministic
mean. That is the reuse the whole arc was built toward, and it is a **consistency guard rather than
independence** (one simulator, two readings - the T17 same-object caveat).

Provenance:
  stock:  none - the archived predecessor had no mean-reverting commodity engine
  source: CLARK-2014 (docs/references.md) ch 2.10.1.1 - eqs (2.150b), (2.151a/b), (2.152), (2.156)
          and the stationary limit (2.155); transcribed in both renderings and reconciled
  oracle: tests/L4/test_schwartz.py - (2.150b) and (2.156) transcribed POINTWISE over a
          (kappa, horizon) grid; `F(0,0) = S0` and zero variance at zero horizon as requirement
          companions; the STATIONARY variance `sigma^2/(2 kappa)` as the structural fact that
          separates this from GBM, with the term structure MEASURED against the closed form and the
          Samuelson effect reported rather than pre-asserted (§4's obligation-not-form clause); the
          kappa -> 0 variance and futures limits, with the CORRECTED driftless-log target and the
          cross-engine agreement at `delta = r - sigma^2/2` plus its non-vacuity companion; the
          simulated OU mean AND variance pinned against (2.150a/b) BEFORE the price band is
          credited (T19), with the error improving as sqrt(N) over six seeds (T21); the LGM scaling
          identity checked on the closed forms rather than by measurement; an MC price band at
          4*SE with no approximation term and its resolution named against the mean-reversion
          factor; put-call parity; invariant 6 with an in-the-money expired fixture
  crosscheck: quantlib ExtendedOrnsteinUhlenbeckProcess / a one-factor mean-reverting commodity
          engine - deferred to the optional QuantLib tier
  slice:  tranche 64 - the Schwartz one-factor model, the commodity arc capstone
"""

from __future__ import annotations

import math

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.models.black import black_price
from pricebook.models.schwartz import SchwartzModel, SchwartzParams
from pricebook.products.commodity_option import CommodityOption


def schwartz_total_variance(params: SchwartzParams, years: float) -> float:
    """`Var[X_T] = (sigma^2/(2 kappa))(1 - e^{-2 kappa T})` - eq (2.150b).

    The **total** variance, which is what Black-76 takes; Clark's `sigma_{0,T}^2 T` is this
    quantity, not an annualised rate. Bounded above by `sigma^2/(2 kappa)` as `T -> infinity`, which
    is the whole qualitative difference from GBM's unbounded `sigma^2 T` - and the reason a
    mean-reverting commodity has a volatility term structure at all.

    Named surface rather than a local inside the pricer, so a test can transcribe it pointwise and
    sweep its term structure (the F-6.1 rule).

    `kappa > 0` is guaranteed by `SchwartzParams`, so there is no division guard here; the
    `kappa -> 0` limit `sigma^2 T` is approached analytically and never evaluated.
    """
    if years < 0.0:
        raise ValueError(f"the variance needs a non-negative horizon, got {years}")
    return (
        params.sigma
        * params.sigma
        / (2.0 * params.kappa)
        * -math.expm1(-2.0 * params.kappa * years)
    )


def schwartz_futures(params: SchwartzParams, spot: float, years: float) -> float:
    """`F(0,T) = S0^{e^{-kappa T}} exp(alpha_hat (1 - e^{-kappa T}) + half the variance)` - (2.156).

    Written with the convexity term as **half of `schwartz_total_variance`** rather than as its own
    `sigma^2/(4 kappa)` expression, because that is what it is: `E[S_T] = exp(E[X_T] + Var[X_T]/2)`
    is the lognormal mean (Clark's (2.152)), so the `4 kappa` in the print is `2 * 2 kappa`. Sharing
    the one function makes the relation structural instead of two constants that must be kept in
    step - and the oracle checks the relation, not only the two values.

    At `T = 0` the decay is 1, both exponential terms vanish and this is `S0` exactly.
    """
    if years < 0.0:
        raise ValueError(f"a futures price needs a non-negative horizon, got {years}")
    if spot <= 0.0:
        raise ValueError(f"the spot must be positive, got {spot}")
    decay = math.exp(-params.kappa * years)
    exponent = params.level * (1.0 - decay) + 0.5 * schwartz_total_variance(params, years)
    return spot**decay * math.exp(exponent)


@register(CommodityOption, SchwartzModel)
def price_commodity_option_schwartz(
    product: CommodityOption, model: SchwartzModel
) -> PricingResult | PricingFailure:
    """`df(T) * N * Black(F(0,T), K, Var[X_T], omega)` - eqs (2.151a/b).

    The same `CommodityOption` product tranche 60 priced under a flat Black vol, now priced under
    mean reversion - which is the point of a stateless engine keyed on `(product, model)`: the trade
    is unchanged and the model is the thing that was swapped.
    """
    market = model.market
    currency = product.notional.currency

    # Invariant 6: an expired option is the shell's business, PV 0, checked before any market lookup
    # so a settled trade needs no commodity data at all.
    if is_historical(product.expiry, market.valuation_date):
        return PricingResult(pv=Money(0.0, currency))

    underlying = market.commodities.get(product.asset)
    if underlying is None:
        return PricingFailure(
            f"no commodity data for {product.asset.symbol} in the snapshot"
        )
    try:
        discount = market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the commodity option: {exc}")

    years = discount.time_measure.year_fraction(product.expiry)
    forward = schwartz_futures(model.params, underlying.spot, years)
    total_variance = schwartz_total_variance(model.params, years)
    undiscounted = black_price(forward, product.strike, total_variance, product.kind)
    return PricingResult(
        pv=Money(
            discount.df(product.expiry) * product.notional.amount * undiscounted, currency
        )
    )
