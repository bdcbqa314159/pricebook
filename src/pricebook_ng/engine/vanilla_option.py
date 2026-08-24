"""Vanilla optionlet pricer — Black-76 caplet, registered behind the L4 registry.

`price_caplet` composes the `forward` atom (F over the accrual, §3d) and `df` (discounting) with
the L3 `black` closed form: `PV = df(pay) · N · τ · black(F, K, vol, t, CALL)`, undiscounted Black
times the discount factor. The model must carry the `BlackVol` capability — validated STRUCTURALLY
against the runtime-checkable protocol (Q3′b: the registry validates capability satisfaction, it does
not `isinstance`-on-concrete-type nor auto-select a method). Market is reached only through
`model.market` (A1); failure is a value (invariant 4).

Provenance:
  quarry: python/pricebook/options/capfloor.py (caplet pricing)
  source: CLAUDE.md §1 (capability, not concrete type) · §2 (A1) · §3d (shared atoms); Black (1976)
  oracle: caplet reprices to Black-76 to <1e-12; put-call parity; vol→0 discounted intrinsic
  slice:  black-caplet (C2 slice 1)
"""

from __future__ import annotations

from pricebook_ng.engine.registry import register
from pricebook_ng.engine.seasoned import current_period_failure
from pricebook_ng.foundation import (
    Money,
    PricingFailure,
    PricingResult,
    Tenor,
    TenorUnit,
    future_periods,
)
from pricebook_ng.market.building_blocks import float_leg_pv, forward, rpv01
from pricebook_ng.models.black import black, vol_time_measure
from pricebook_ng.models.protocols import BlackVol, CalibratedModel, SwaptionVol
from pricebook_ng.products.option import Caplet, OptionType, Swaption
from pricebook_ng.products.swap import VanillaSwap


@register(Caplet)
def price_caplet(caplet: Caplet, model: CalibratedModel) -> PricingResult | PricingFailure:
    """Mark a Black-76 caplet. Requires the `BlackVol` capability (validated structurally)."""
    if not isinstance(model, BlackVol):  # capability check — a protocol, not a concrete-type ladder
        return PricingFailure("caplet requires a model with the BlackVol capability")
    accrual, index = caplet.accrual, caplet.index
    currency = index.id.currency
    vd = model.market.valuation_date
    if accrual.end <= vd:  # invariant 6: a paid optionlet has no future PV
        return PricingResult(pv=Money(0.0, currency))
    seasoned = current_period_failure(accrual.start, vd)  # #3a: fixing past, not yet paid
    if seasoned is not None:
        return seasoned
    curves = model.market.curves
    try:
        expiry = accrual.start  # T = valuation → fixing, ACT/365F for the vol
        t = vol_time_measure(vd).year_fraction(expiry)
        pay_df = curves.discount(currency).df(accrual.end)
        fwd = forward(curves.projection(index), accrual)
        tau = accrual.year_fraction()
        if (fwd <= 0.0 or caplet.strike <= 0.0) and t > 0.0:  # #15: lognormal Black undefined
            return PricingFailure("forward ≤ 0: lognormal Black undefined — normal/shifted deferred")
        vol = model.black_vol(index, expiry, caplet.strike)
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        return PricingFailure(str(exc))
    pv = pay_df * caplet.notional * tau * black(fwd, caplet.strike, vol, t, OptionType.CALL)
    return PricingResult(pv=Money(pv, currency))


def underlying_tenor(swap: VanillaSwap) -> Tenor:
    """The underlying swap's EXACT tenor — period count × the schedule's coupon step (the #7 enrichment),
    NOT `round(days/365)`. A 6M underlying keys `6M`, 18M keys `18M`, 5Y normalizes to `5Y` (whole years
    collapse to YEAR so existing surfaces stay keyed as before). This is the swaption vol's key dimension
    and must be exact — `SwaptionSurfaceKey` exists to distinguish 2Y5Y from 2Y10Y (#5)."""
    sched = swap.fixed_leg.schedule
    months = len(sched.periods) * (12 // sched.terms.frequency.per_year())
    if months % 12 == 0:
        return Tenor(months // 12, TenorUnit.YEAR)
    return Tenor(months, TenorUnit.MONTH)


@register(Swaption)
def price_swaption(swaption: Swaption, model: CalibratedModel) -> PricingResult | PricingFailure:
    """Mark a Black European swaption on the annuity numeraire. Requires the `SwaptionVol`
    capability (validated structurally). Composes the SHARED atoms `rpv01` (annuity = numeraire)
    and `float_leg_pv` (→ forward swap rate `S = float_leg_pv/annuity`) — never the calibrator's
    private `_par_rate` — so engine and calibrator cannot disagree (§3d). PV = N·annuity·black(S,K,·)."""
    if not isinstance(model, SwaptionVol):  # capability check — a protocol, not a concrete-type ladder
        return PricingFailure("swaption requires a model with the SwaptionVol capability")
    swap = swaption.swap
    index = swap.float_leg.index
    currency = swap.currency
    strike = swap.fixed_leg.rate
    vd = model.market.valuation_date
    if swaption.expiry <= vd:  # #20: an expired swaption has no future optional value (exercise = L6)
        return PricingResult(pv=Money(0.0, currency))
    curves = model.market.curves
    try:
        discount = curves.discount(currency, swap.collateral)  # same discount the swap engine uses
        projection = curves.projection(index)
        # invariant 6: the underlying's mark is FUTURE PV only (a spot-forward swaption loses none)
        fixed_sched = future_periods(swap.fixed_leg.schedule, vd)
        float_sched = future_periods(swap.float_leg.schedule, vd)
        seasoned = current_period_failure(fixed_sched.periods[0].accrual_start, vd) if fixed_sched.periods else None
        if seasoned is not None:  # #3a: underlying's current period is in progress
            return seasoned
        annuity = rpv01(fixed_sched, swap.fixed_leg.day_count, discount)
        s = float_leg_pv(float_sched, swap.float_leg.day_count, discount, projection) / annuity
        if (s <= 0.0 or strike <= 0.0):  # #15: lognormal Black undefined (t > 0 guaranteed above)
            return PricingFailure("forward swap rate ≤ 0: lognormal Black undefined — normal/shifted deferred")
        swap_tenor = underlying_tenor(swap)
        vol = model.swaption_vol(index, swaption.expiry, swap_tenor, strike)
        t = vol_time_measure(vd).year_fraction(swaption.expiry)
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        return PricingFailure(str(exc))
    pv = swap.notional * annuity * black(s, strike, vol, t, swaption.option_type)
    return PricingResult(pv=Money(pv, currency))
