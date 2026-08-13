"""Cash-instrument pricers — Deposit, FRA, Future, registered behind the L4 registry.

Each composes the SAME L1 atoms the calibration instruments compose (§3d), so the L3
residual and the L4 price cannot drift: the deposit pricer and `DepositInstrument` share
`deposit_df`; the FRA/future pricers and `FRAInstrument`/`FutureInstrument` share `forward`.
Market is reached only through `model.market` (A1); failure is a value (invariant 4). A
future's forward rate is `1 − price` (convexity deferred to the models topic).

Provenance:
  quarry: python/pricebook/pricing/ (deposit / FRA / future engines)
  source: CLAUDE.md §1 (registry) · §2 (A1) · §3d (shared atoms); doc 18 §2 (futures = forwards)
  oracle: each reprices to zero at par through the engine; future reproduces 1 − price at IMM
  slice:  cash-instruments (T1 slice 3, Step B)
"""

from __future__ import annotations

from pricebook_ng.engine.registry import register
from pricebook_ng.foundation import Money, PricingFailure, PricingResult
from pricebook_ng.market.building_blocks import deposit_df, forward
from pricebook_ng.models.protocols import CalibratedModel
from pricebook_ng.products.cash import FRA, Deposit, Future

__all__ = ["price_deposit", "price_fra", "price_future"]


@register(Deposit)
def price_deposit(deposit: Deposit, model: CalibratedModel) -> PricingResult | PricingFailure:
    """Lender PV = N·(df(end)/deposit_df − df(start)); zero when df(end) = df(start)·deposit_df."""
    try:
        discount = model.market.curves.discount(deposit.currency)
        a = deposit.accrual
        pv = deposit.notional * (
            discount.df(a.end) / deposit_df(deposit.rate, a) - discount.df(a.start)
        )
    except (ValueError, KeyError) as exc:
        return PricingFailure(str(exc))
    return PricingResult(pv=Money(pv, deposit.currency))


@register(FRA)
def price_fra(fra: FRA, model: CalibratedModel) -> PricingResult | PricingFailure:
    """PV = N·τ·df_disc(end)·(forward(proj) − rate); zero when the projected forward = rate."""
    try:
        curves = model.market.curves
        discount = curves.discount(fra.currency)
        projection = curves.projection(fra.index)
        pv = (
            fra.notional
            * fra.accrual.year_fraction()
            * discount.df(fra.accrual.end)
            * (forward(projection, fra.accrual) - fra.rate)
        )
    except (ValueError, KeyError) as exc:
        return PricingFailure(str(exc))
    return PricingResult(pv=Money(pv, fra.currency))


@register(Future)
def price_future(future: Future, model: CalibratedModel) -> PricingResult | PricingFailure:
    """PV = N·τ·df_disc(end)·(forward(proj) − (1 − price)); the forward approximation to a
    future (no convexity). Zero when the projected forward = 1 − price."""
    try:
        curves = model.market.curves
        discount = curves.discount(future.currency)
        projection = curves.projection(future.index)
        pv = (
            future.notional
            * future.accrual.year_fraction()
            * discount.df(future.accrual.end)
            * (forward(projection, future.accrual) - (1.0 - future.price))
        )
    except (ValueError, KeyError) as exc:
        return PricingFailure(str(exc))
    return PricingResult(pv=Money(pv, future.currency))
