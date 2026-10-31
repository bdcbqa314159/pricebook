"""The year-on-year inflation swap under Jarrow-Yildirim (L4). BERHAN-2012 eq 2.5.13.

    YYIIS(0,T,Psi,N) = N*psi_1*[Pr(0,T1) - Pn(0,T1)]
                     + N*sum_{i=2..M} psi_i*[ Pn(0,Ti-1)*Pr(0,Ti)/Pr(0,Ti-1)*e^C - Pn(0,Ti) ]

**The first cash flow is priced by the ZERO-COUPON inflation leg formula** (1.2.13), because its
previous fixing is already known -- the thesis is explicit about this. So the first term composes
tranche 2's verified `inflation_leg_value` rather than restating it, and the cross-product
reconciliation oracle is a statement of the formula's own structure.

**Every later period carries `e^C`, the convexity** (2.5.11), which the model supplies. `C` is
where the model enters at all: with `sigma_r = 0` it is exactly zero and the whole swap collapses
to the deterministic-real-rate limit -- the tranche's anchor oracle, exact by construction rather
than numerically.

**At inception this prices off curves alone** -- eq 2.5.13 contains no index level, the first term
being `Pr - Pn` and the rest forward ratios. Requiring a fixing would be a plausible-looking error,
so a test asserts a YoY swap prices without one.

**t = 0 only.** The seasoned form (eq 2.5.12, `iota(t) > 1`) needs the *historical* fixing
`I(T_{iota-1})`, i.e. an index **series** in the snapshot -- a new market-data shape, deferred with
the indexation-lag resolution it shares a consumer with.

  crosscheck: quantlib YoYInflationSwap -- deferred to the optional QuantLib tier (0013/0014).

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/jarrow_yildirim.py (the ZC/YoY structure, mined for
          shape). Its `inflation_advanced.yoy_convexity_adjustment` is deliberately NOT used -- a
          first-order approximation with rho as a factor and an unused real_vol argument.
  source: BERHAN-2012 (docs/references.md) eqs 1.2.17-1.2.18, 2.5.9-2.5.13
  oracle: eq 2.5.13 against an independent transcription; sigma_r -> 0 gives the deterministic
          limit exactly (the anchor), with a companion test that the convexity MOVES the leg so
          the anchor is not vacuous; the first period equals tranche 2's ZCIIS leg; par-rate
          reprice; matured -> PV 0; prices with no index fixing
  slice:  tranche 4 -- the year-on-year swap and Jarrow-Yildirim
"""

from __future__ import annotations

import math

from pricebook.engine.registry import register
from pricebook.foundation import Money, PricingFailure, PricingResult
from pricebook.models.jarrow_yildirim import JarrowYildirim
from pricebook.products.inflation import YearOnYearInflationSwap


def yoy_inflation_leg_value(swap: YearOnYearInflationSwap, model: JarrowYildirim) -> float:
    """The inflation leg of eq 2.5.13. Raises on missing market data; the pricer turns that into a
    `PricingFailure`."""
    market = model.market
    currency = swap.notional.currency
    nominal = market.curves.discount(currency)
    real = market.curves.real(currency)
    measure = nominal.time_measure
    periods = [p for p in swap.leg.schedule.periods() if p[1] > market.valuation_date]
    if not periods:
        return 0.0

    # The FIRST remaining period is the zero-coupon form (1.2.13) with its previous fixing known:
    # `Pr(0,T1) - Pn(0,T1)`, which is exactly tranche 2's leg at a unit index ratio.
    _, first_pay, first_psi = periods[0]
    total = first_psi * (real.df(first_pay) - nominal.df(first_pay))

    # Later periods carry the forward ratio and the convexity.
    for start, pay, psi in periods[1:]:
        c = model.convexity(0.0, measure.year_fraction(start), measure.year_fraction(pay))
        forward_real = real.df(pay) / real.df(start)
        total += psi * (nominal.df(start) * forward_real * math.exp(c) - nominal.df(pay))

    return swap.notional.amount * total


def yoy_fixed_leg_value(swap: YearOnYearInflationSwap, model: JarrowYildirim) -> float:
    """`N·Σ ψᵢ·C_fix·Pn(0,Tᵢ)` — eq 1.2.18 discounted. Historical periods excluded (invariant 6)."""
    market = model.market
    nominal = market.curves.discount(swap.notional.currency)
    return swap.notional.amount * sum(
        swap.leg.fixed_rate * psi * nominal.df(pay)
        for _, pay, psi in swap.leg.schedule.periods()
        if pay > market.valuation_date
    )


def yoy_par_rate(swap: YearOnYearInflationSwap, model: JarrowYildirim) -> float:
    """The fixed rate that makes the swap worth zero: the leg over the annuity.

    Solved in closed form rather than by iteration — the fixed leg is linear in the rate — and it
    composes the **same** `yoy_inflation_leg_value` and annuity the pricer composes, so a par
    reprice cannot disagree with the price. (That also means the par oracle telescopes, which is
    why it is kept as a consistency guard only and the anchor is the `σr → 0` reduction.)"""
    market = model.market
    nominal = market.curves.discount(swap.notional.currency)
    annuity = swap.notional.amount * sum(
        psi * nominal.df(pay)
        for _, pay, psi in swap.leg.schedule.periods()
        if pay > market.valuation_date
    )
    if annuity == 0.0:
        raise ValueError("a matured swap has no par rate")
    return yoy_inflation_leg_value(swap, model) / annuity


@register(YearOnYearInflationSwap)
def price_yoy_inflation_swap(
    swap: YearOnYearInflationSwap, model: JarrowYildirim
) -> PricingResult | PricingFailure:
    """Mark a YoY inflation swap from the inflation **receiver's** side: leg minus fixed leg."""
    market = model.market
    currency = swap.notional.currency

    # Invariant 6: all periods historical ⇒ the trade is settled. PV 0 before any market lookup,
    # per tranche 2's precedent, so a matured swap needs no curves.
    if all(pay <= market.valuation_date for _, pay, _ in swap.leg.schedule.periods()):
        return PricingResult(pv=Money(0.0, currency))

    try:
        pv = yoy_inflation_leg_value(swap, model) - yoy_fixed_leg_value(swap, model)
    except KeyError as exc:
        return PricingFailure(f"cannot price the year-on-year swap: {exc}")
    return PricingResult(pv=Money(pv, currency))
