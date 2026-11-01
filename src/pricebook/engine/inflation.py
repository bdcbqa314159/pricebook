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
from pricebook.foundation.inflation import InflationIndex
from pricebook.market.fixing_series import reference_level
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.discounting_model import DiscountingModel
from pricebook.products.inflation import InflationLinkedBond, ZeroCouponInflationSwap


def resolved_index_level(index: InflationIndex, market: MarketSnapshot) -> float:
    """`I(t)` for the valuation date, through the index's own lag and interpolation.

    **One atom, three engines.** The ZCIIS, the linker and the year-on-year swap all need `I(t)`;
    resolving it in one place is what stops them disagreeing about which print that is while every
    within-product oracle stays green — the shape a cross-module convention bug takes. Raises on a
    missing series or a missing print; the pricers turn that into a `PricingFailure`, and nothing
    here guesses a level."""
    try:
        series = market.fixings[index]
    except KeyError:
        raise KeyError(
            f"no fixing series for {index}; refusing to assume the reference level"
        ) from None
    return reference_level(series, market.valuation_date, index)


def inflation_leg_value(swap: ZeroCouponInflationSwap, market: MarketSnapshot) -> float:
    """`N[ I(t)/I0 · Pr(t,TM) − Pn(t,TM) ]` (1.2.13) — the value of the payout `N[I(TM)/I0 − 1]`.

    Exposed because it is the quantity the thesis calls `ZCIIS(t,TM)`, so the oracle can check the
    paper's own formula directly rather than only the net swap it appears inside. Raises on missing
    market data; the pricer turns that into a `PricingFailure`."""
    currency = swap.notional.currency
    nominal = market.curves.discount(currency)
    real = market.curves.real(currency)
    index_level = resolved_index_level(swap.index, market)
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


@register(InflationLinkedBond)
def price_inflation_linked_bond(
    bond: InflationLinkedBond, model: DiscountingModel
) -> PricingResult | PricingFailure:
    """`I(t)/I0 · N[ Σ C·ψᵢ·Pr(t,Tᵢ) + Pr(t,TM) ]` — BERHAN 1.2.5, generalised with the accrual.

    **The paper's 1.2.5 is the annual, unit-accrual case.** A real linker pays `C·ψᵢ`, so this
    prices the general form and 1.2.5 is its *reduction* at `ψᵢ = 1` — asserted as an oracle rather
    than implemented twice. It is also why `CouponSchedule.day_count` is read here: a carried field
    nothing consumes is the thing this project keeps catching, so the accrual is computed from the
    schedule's own convention and from nowhere else.

    **No nominal curve is touched.** A linker is a real bond scaled by the index ratio; reaching
    for `Pn` here would be the kind of error oracle `test_the_nominal_curve_does_NOT_affect_the_price`
    exists to catch.

      crosscheck: quantlib CPIBond — deferred to the optional QuantLib tier (thread 0013/0014).
    """
    market = model.market
    currency = bond.notional.currency
    vd = market.valuation_date
    schedule = bond.leg.schedule

    # Invariant 6: every remaining flow is in the future. A matured bond has none, so it is
    # historical in full — PV 0, and checked before any market lookup, per tranche 2's precedent.
    future = [(pay, psi) for _, pay, psi in schedule.periods() if pay > vd]
    principal_due = schedule.dates[-1] > vd
    if not future and not principal_due:
        return PricingResult(pv=Money(0.0, currency))

    try:
        real = market.curves.real(currency)
        index_level = resolved_index_level(bond.index, market)
    except KeyError as exc:
        return PricingFailure(f"cannot price the inflation-linked bond: {exc}")

    coupons = sum(bond.leg.coupon_rate * psi * real.df(pay) for pay, psi in future)
    redemption = real.df(schedule.dates[-1]) if principal_due else 0.0
    ratio = index_level / bond.base_index
    return PricingResult(pv=Money(bond.notional.amount * ratio * (coupons + redemption), currency))
