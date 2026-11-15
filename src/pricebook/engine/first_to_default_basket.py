"""The first-to-default basket pricer (L4). `OKANE-2008` ch 15.2.

    Premium PV    = S₁ · RPV01(Q_B) · N
    Protection PV = (1−R) ∫ Z(t)(−dQ_B(t)) · N
    V             = Protection − Premium          (long protection)
    S₁(par)       = Protection / (RPV01(Q_B) · N)

**A first-to-default basket is a CDS on the basket survival curve**, which is what makes this
engine small: the only new quantity is `Q_B` (L3, `models/basket.py`), and the legs are
`credit_legs`'.

**The reuse is clean on one leg and not the other, which was measured rather than assumed.**
`risky_pv01` reads only `survival(t)`, so the basket curve feeds it unchanged. The protection leg
could not be reused: `credit_legs.protection_leg` reads one hazard per pillar segment, exact for a
piecewise-constant-hazard curve and wrong for `Q_B`, where it **plateaus 1.56e-03 from the truth**
at every refinement. So this uses `protection_leg_from_survival`, which differences `Q` and needs
no hazard — the form `OKANE-2008` ch 15.2 itself writes.

**Deterministic, so no standard error.** The market-factor integral prices; the Monte-Carlo is the
cross-check oracle, not the engine (the tranche-41 pattern). §3.7's SE field stays `None`, which is
what it is for.

Provenance:
  stock:  none — the archived predecessor had no basket pricer
  source: OKANE-2008 ch 15.2; the legs are market/credit_legs.py (tranche 39), the basket survival
          is models/basket.py (tranche 42) on tranche 41's copula
  oracle: tests/L4/test_first_to_default_basket.py — the par reprice, linearity in the spread, and
          the spread bounded below by the widest single name and above by their sum
  crosscheck: quantlib NthToDefault — deferred, dev-only
  slice:  tranche 42 — the first-to-default basket
"""

from __future__ import annotations

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.market.credit_legs import protection_leg_from_survival, risky_pv01
from pricebook.models.basket import BasketModel, BasketSurvival
from pricebook.products.first_to_default_basket import FirstToDefaultBasket

#: Steps in the protection-leg quadrature. Measured against a 4000-step reference on the
#: tranche-42 fixture: 7.3e-06 at 50, 4.5e-07 at 200, 2.7e-08 at 800. 800 is committed — it
#: resolves better than 1e-07 and the whole price costs a few milliseconds.
_DEFAULT_STEPS = 800


def _live_periods(product: FirstToDefaultBasket, valuation_date):
    """The premium periods still to pay — invariant 6, through the one shared atom (C10)."""
    return [
        period
        for period in product.schedule.periods()
        if not is_historical(period[1], valuation_date)
    ]


def _basket_for(product: FirstToDefaultBasket, model: BasketModel) -> BasketSurvival:
    """The basket curve for exactly the names this contract protects.

    Checked against the copula rather than assumed: a basket naming an entity the copula does not
    carry must fail as a value, never quietly price the copula's own names instead — which would be
    a plausible number for the wrong contract.
    """
    for entity in product.entities:
        model.curve_for(entity)
    named = {entity for entity in product.entities}
    selected = tuple(name for name in model.copula.names if name.entity in named)
    from dataclasses import replace

    return BasketSurvival(replace(model.copula, names=selected))


def basket_annuity(product: FirstToDefaultBasket, model: BasketModel) -> float:
    """`RPV01(Q_B)` per unit notional — named surface because an oracle is about it."""
    market = model.market
    discount = market.curves.discount(product.notional.currency)
    return risky_pv01(
        _live_periods(product, market.valuation_date), discount, _basket_for(product, model))


def basket_protection(
    product: FirstToDefaultBasket, model: BasketModel, steps: int = _DEFAULT_STEPS
) -> float:
    """`(1−R)∫Z(−dQ_B)` per unit notional."""
    market = model.market
    discount = market.curves.discount(product.notional.currency)
    periods = _live_periods(product, market.valuation_date)
    if not periods:
        return 0.0
    horizon = discount.time_measure.year_fraction(periods[-1][1])
    return protection_leg_from_survival(
        discount, _basket_for(product, model), model.recovery, horizon, steps)


def par_spread(
    product: FirstToDefaultBasket, model: BasketModel, steps: int = _DEFAULT_STEPS
) -> float:
    """The spread making the basket worth zero — the quoted first-to-default spread."""
    annuity = basket_annuity(product, model)
    if annuity <= 0.0:
        raise ValueError(
            "a par spread needs a positive risky annuity; this basket has no premium periods left")
    return basket_protection(product, model, steps) / annuity


@register(FirstToDefaultBasket, BasketModel)
def price_first_to_default_basket(
    product: FirstToDefaultBasket, model: BasketModel
) -> PricingResult | PricingFailure:
    """`Protection − S₁·RPV01`, long protection."""
    currency = product.notional.currency
    if not _live_periods(product, model.market.valuation_date):
        # Invariant 6: a matured basket is the shell's business, PV 0 before any market lookup.
        return PricingResult(pv=Money(0.0, currency))
    try:
        protection = basket_protection(product, model)
        premium = product.spread * basket_annuity(product, model)
    except KeyError as exc:
        return PricingFailure(f"cannot price the first-to-default basket: {exc}")
    return PricingResult(
        pv=Money((protection - premium) * product.notional.amount, currency))
