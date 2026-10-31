"""Zero-coupon inflation swap pricer (L4) — model-free. BERHAN-2012 §1.2.2.

The whole instrument is no-arbitrage: no dynamics, no vol, no correlation. Its value comes from
the foreign-currency analogy the thesis derives it from — **real ↔ foreign, nominal ↔ domestic,
the index ↔ the FX spot** (eqs 1.2.8–1.2.12) — which is why the inflation leg's value is the index
level times a *real* bond, exactly as a foreign-currency claim is the spot times a foreign bond.

    inflation leg   N[ I(t)/I0 · Pr(t,TM) − Pn(t,TM) ]      (1.2.13)
    at t=0, I=I0    N[ Pr(0,TM) − Pn(0,TM) ]                (1.2.14)
    fixed leg       N[ (1+b)^TM − 1 ] · Pn(t,TM)            (1.2.7, discounted)

1.2.14 is **not a second formula** — it is 1.2.13 at `I(t)=I0`, and a test asserts that reduction
rather than implementing both. The compounding exponent `TM` is the term's **inception-to-maturity**
fraction, because the fixed payout was fixed at trade time and does not shrink as the trade seasons.

  crosscheck: quantlib ZeroCouponInflationSwap — deferred to the optional QuantLib tier
  (thread 0013/0014).

Provenance:
  stock:  _stock/quarry/parked/ng/pricebook_ng/engine/inflation.py (mined for CONTENT and
          extended: that version priced only the inception form, having no base index)
  source: BERHAN-2012 (docs/references.md) §1.2.2, eqs 1.2.6–1.2.7, 1.2.13–1.2.14
  oracle: 1.2.14 off independently-specified curves to 1e-9; the 1.2.13→1.2.14 reduction; the
          index entering as a RATIO; the par identity at the break-even; matured → PV 0
  slice:  tranche 2 — the inflation spine
"""

from __future__ import annotations

from pricebook.engine.registry import register
from pricebook.foundation import Money, PricingFailure, PricingResult
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.discounting_model import DiscountingModel
from pricebook.products.inflation import ZeroCouponInflationSwap


def inflation_leg_value(swap: ZeroCouponInflationSwap, market: MarketSnapshot) -> float:
    """`N[ I(t)/I0 · Pr(t,TM) − Pn(t,TM) ]` (1.2.13) — the value of the payout `N[I(TM)/I0 − 1]`.

    Exposed because it is the quantity the thesis calls `ZCIIS(t,TM)`, so the oracle can check the
    paper's own formula directly rather than only the net swap it appears inside. Raises on missing
    market data; the pricer turns that into a `PricingFailure`."""
    currency = swap.notional.currency
    nominal = market.curves.discount(currency)
    real = market.curves.real(currency)
    try:
        index_level = market.index_fixings[swap.index]
    except KeyError:
        raise KeyError(
            f"no index fixing for {swap.index}; refusing to assume the base level"
        ) from None
    maturity = swap.term.maturity
    ratio = index_level / swap.base_index
    return swap.notional.amount * (ratio * real.df(maturity) - nominal.df(maturity))


@register(ZeroCouponInflationSwap)
def price_zc_inflation_swap(
    swap: ZeroCouponInflationSwap, model: DiscountingModel
) -> PricingResult | PricingFailure:
    """Mark a ZC inflation swap from the inflation **receiver's** side: leg minus fixed leg."""
    market = model.market
    currency = swap.notional.currency

    # Invariant 6, and tranche 1's precedent: the single exchange is on or before the valuation
    # date, so it is HISTORICAL — the shell's business, excluded from the mark. PV 0, not a
    # failure: `PricingFailure` is for inputs that cannot be priced, and a settled trade is not
    # unpriceable. Checked before any market lookup, so a matured trade needs no curves.
    if swap.term.maturity <= market.valuation_date:
        return PricingResult(pv=Money(0.0, currency))

    try:
        inflation = inflation_leg_value(swap, market)
        fixed = (
            swap.notional.amount
            * ((1.0 + swap.fixed_rate) ** swap.term.years() - 1.0)
            * market.curves.discount(currency).df(swap.term.maturity)
        )
    except KeyError as exc:  # a missing curve or fixing → a value, not a raise
        return PricingFailure(f"cannot price the inflation swap: {exc}")

    return PricingResult(pv=Money(inflation - fixed, currency))
