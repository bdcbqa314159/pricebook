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
  source: BROCKHAUS-2016 (docs/references.md) ch 3.7 / 5.8 - the variance swap's static replication;
          KWOK-ZHENG §1.4 - Proposition 1.2 (the gamma swap, eq 1.24: the `1/K` weight with its
          `1/F0` normalisation) and eq (1.26) (the corridor swap, the `1/K^2` strip restricted to
          the contract's corridor), both transcribed in `-raw` and `-layout` and reconciled
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
from pricebook.market.equity_skew import SmileLike
from pricebook.models.black import black_price
from pricebook.models.variance_replication_model import StrikeGrid, VarianceStrip, VarianceReplicationModel
from pricebook.products.equity_option import EquityOptionKind
from pricebook.products.generalized_variance_swap import CorridorVarianceSwap, GammaSwap
from pricebook.products.variance_swap import VarianceSwap


def generalized_fair_variance(
    underlying: EquityUnderlying,
    expiry: date,
    discount: CurveHandle,
    skew: SmileLike,
    strip: VarianceStrip,
) -> float:
    """The fair variance of a generalized variance swap — one strip walk, three products.

    `2/T · ∫ term(OTM(K), K, F) dK` over the strip's range, where `term` carries the weight. The
    vanilla variance swap (`1/K²`, full range), the gamma swap (`1/(K·F₀)`, full range — eq 1.24)
    and the corridor variance swap (`1/K²`, restricted to the corridor — eq 1.26) differ **only** in
    that term and that range, which is why they share this walk rather than three copies of it.

    **The rule of two fired at the third consumer, and the extraction is what makes the weighting a
    comparable value** — the same move convention C16 earned at tranche 73: with three engines each
    writing their own `1/K^n` inline, nothing could assert they agreed about the prefactor or the
    put/call split; with one walk a divergence is not expressible.

    Named surface rather than a step inside a pricer, because every oracle reaches for *this* number
    — the reduction to `σ²`, the corridor ordering, the gamma/vanilla relation (§4).
    """
    forward = dividend_forward(underlying, expiry, discount)
    years = discount.time_measure.year_fraction(expiry)
    if years <= 0.0:
        raise ValueError(f"the fair variance needs a positive horizon, got {years}")

    strikes = strip.restricted(forward)
    weighted = []
    for strike in strikes:
        volatility = skew.volatility(strike, forward, years)
        kind = EquityOptionKind.PUT if strike < forward else EquityOptionKind.CALL
        undiscounted = black_price(forward, strike, volatility * volatility * years, kind)
        weighted.append(strip.term(undiscounted, strike, forward))

    area = 0.0
    for i in range(len(strikes) - 1):
        area += 0.5 * (weighted[i] + weighted[i + 1]) * (strikes[i + 1] - strikes[i])
    return 2.0 / years * area


def fair_variance(
    underlying: EquityUnderlying,
    expiry: date,
    discount: CurveHandle,
    skew: SmileLike,
    grid: StrikeGrid,
) -> float:
    """`K_var` - the fair variance, as the `1/K^2`-weighted strip of OTM vanillas.

    Tranche 31's entry point, kept because every caller and every oracle names it, and because the
    vanilla `1/K²` strip is the thing the other two are generalizations *of*. The walk moved to
    `generalized_fair_variance` at tranche 74 when the gamma and corridor swaps became its second
    and third consumers; this is the same number, **bit-identically** — the weighted term is one
    division either way, so no operation was reordered (T12(a)).
    """
    return generalized_fair_variance(
        underlying, expiry, discount, skew, VarianceStrip.vanilla(grid)
    )


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


def _price_generalized(
    product: GammaSwap | CorridorVarianceSwap,
    model: VarianceReplicationModel,
    strip: VarianceStrip,
) -> PricingResult | PricingFailure:
    """`N · df(T) · (K_generalized − K)` — the same discounted variance difference T31 pays.

    One body for both products: they differ in the **strip** the caller builds, which is the whole
    content of a generalized variance swap, and in nothing else about the pricing. The strip is
    built by each registered pricer rather than here, because it is where the contract's own terms
    (the corridor) enter.
    """
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
    # `ValueError` as well as `KeyError`, and the first draft of this caught only `KeyError`:
    # `VarianceStrip.restricted` refuses a corridor that does not contain the forward, and that
    # refusal was propagating as an exception **out of a registered pricer** — a §3.4 violation
    # (failure is a value, never raise-and-hope). Found by asking what a second killer for the
    # guard would look like, not by review.
    try:
        discount = market.curves.discount(currency)
        variance = generalized_fair_variance(underlying, product.expiry, discount, skew, strip)
    except (KeyError, ValueError) as exc:
        return PricingFailure(f"cannot price the generalized variance swap: {exc}")
    payoff = product.variance_notional.amount * (variance - product.variance_strike)
    return PricingResult(pv=Money(discount.df(product.expiry) * payoff, currency))


@register(GammaSwap, VarianceReplicationModel)
def price_gamma_swap(
    product: GammaSwap, model: VarianceReplicationModel
) -> PricingResult | PricingFailure:
    """The gamma swap — the `1/(K·F₀)` strip of eq (1.24)."""
    return _price_generalized(product, model, VarianceStrip.gamma(model.grid))


@register(CorridorVarianceSwap, VarianceReplicationModel)
def price_corridor_variance_swap(
    product: CorridorVarianceSwap, model: VarianceReplicationModel
) -> PricingResult | PricingFailure:
    """The corridor variance swap — the `1/K²` strip of eq (1.26), restricted to the contract's
    corridor rather than to the grid's truncation bounds."""
    lower, upper = product.corridor
    return _price_generalized(
        product, model, VarianceStrip.corridor(model.grid, lower, upper)
    )
