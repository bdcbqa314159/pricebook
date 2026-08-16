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
from pricebook_ng.foundation import (
    DayCountConvention,
    Money,
    PricingFailure,
    PricingResult,
    Schedule,
    Tenor,
    TenorUnit,
    TimeMeasure,
)
from pricebook_ng.market.building_blocks import float_leg_pv, forward, rpv01
from pricebook_ng.models.black import black
from pricebook_ng.models.protocols import BlackVol, CalibratedModel, SwaptionVol
from pricebook_ng.products.option import Caplet, OptionType, Swaption


@register(Caplet)
def price_caplet(caplet: Caplet, model: CalibratedModel) -> PricingResult | PricingFailure:
    """Mark a Black-76 caplet. Requires the `BlackVol` capability (validated structurally)."""
    if not isinstance(model, BlackVol):  # capability check — a protocol, not a concrete-type ladder
        return PricingFailure("caplet requires a model with the BlackVol capability")
    accrual, index = caplet.accrual, caplet.index
    currency = index.id.currency
    curves = model.market.curves
    try:
        pay_df = curves.discount(currency).df(accrual.end)
        fwd = forward(curves.projection(index), accrual)
        tau = accrual.year_fraction()
        # T = time from the valuation date to the optionlet expiry (fixing), ACT/365F for the vol
        expiry = accrual.start
        t = TimeMeasure(model.market.valuation_date, DayCountConvention.ACT_365_FIXED).year_fraction(expiry)
        vol = model.black_vol(index, expiry, caplet.strike)
    except (ValueError, KeyError) as exc:
        return PricingFailure(str(exc))
    pv = pay_df * caplet.notional * tau * black(fwd, caplet.strike, vol, t, OptionType.CALL)
    return PricingResult(pv=Money(pv, currency))


def _swap_tenor(schedule: Schedule) -> Tenor:
    """The underlying swap's tenor (whole years, start→end) — the swaption vol's key dimension."""
    start = schedule.periods[0].accrual_start
    end = schedule.periods[-1].accrual_end
    return Tenor(round((end - start).days / 365.0), TenorUnit.YEAR)


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
    curves = model.market.curves
    try:
        discount = curves.discount(currency, swap.collateral)  # same discount the swap engine uses
        projection = curves.projection(index)
        annuity = rpv01(swap.fixed_leg.schedule, swap.fixed_leg.day_count, discount)
        s = float_leg_pv(swap.float_leg.schedule, swap.float_leg.day_count, discount, projection) / annuity
        swap_tenor = _swap_tenor(swap.fixed_leg.schedule)
        vol = model.swaption_vol(index, swaption.expiry, swap_tenor, strike)
        t = TimeMeasure(model.market.valuation_date, DayCountConvention.ACT_365_FIXED).year_fraction(swaption.expiry)
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        return PricingFailure(str(exc))
    pv = swap.notional * annuity * black(s, strike, vol, t, swaption.option_type)
    return PricingResult(pv=Money(pv, currency))
