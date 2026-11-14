"""The forward CDS and the knockout default swaption (L4). `OKANE-2008` ch 9.2/9.3.

    Forward annuity   RPV01(t,tF,T) = RPV01(t,T) − RPV01(t,tF)
    Forward spread    S(t,tF,T)     = [Prot(t,T) − Prot(t,tF)] / RPV01(t,tF,T)          (9.2)
    Forward CDS       V(t)          = ω·(S(t,tF,T) − S₀)·RPV01(t,tF,T)·N
    Default swaption  V(0)          = RPV01(0,tE,T)·Black(S(0,tE,T), K, σ²tE, ω)·N      (9.3)

**Why Black applies, in one sentence.** Under the measure whose numéraire is the risky annuity, the
forward spread is a martingale — `E^P[S(tE,T)] = S(0,tE,T)`, eq 9.6 — so the option is a vanilla on
it, and the annuity reappears as the discounting factor. Same structure as the LMM swaption
(tranches 17/19), with `RPV01` in place of the swap annuity.

**The knockout needs no indicator.** `1{τ>tE}` is already inside `RPV01(0,tE,T)`: both annuities
weight by survival from today, so their difference carries "and the name has not defaulted by
`tE`". A separate factor would double-count it.

**Both legs come from the shared L1 functional, so a forward CDS and a spot CDS cannot disagree
about what a protection leg is** (`docs/conventions_register.md` C11). The composition here is the
only new arithmetic, which is why the suite transcribes *it* pointwise and anchors the rest on a
cross-product: the forward CDS must equal the difference of two spot CDS prices through tranche
38's engine, for every contractual spread — different code on each side.

Provenance:
  stock:  none — the archived predecessor had neither product
  source: OKANE-2008 ch 9.2 (eq 9.2, the forward spread), ch 9.3 (the knockout default swaption,
          eq 9.6 the annuity-measure martingale); Black's bracket is models/black.py (tranche 5)
  oracle: tests/L4/test_forward_cds_default_swaption.py — the cross-product to two spot CDS,
          put-call parity against the forward CDS, the tF→t reduction to tranche 38's par spread,
          the zero-vol intrinsic, and a pointwise transcription of the Black composition
  crosscheck: quantlib CdsOption / BlackCdsOptionEngine — deferred, dev-only
  slice:  tranche 40 — the forward CDS and the default swaption
"""

from __future__ import annotations

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.schedule import CouponSchedule
from pricebook.foundation.time_measure import is_historical
from pricebook.market import credit_legs as legs
from pricebook.models.credit import CreditModel
from pricebook.models.credit_swaption import CreditSwaptionModel
from pricebook.models.black import black_price
from pricebook.products.credit_default_swaption import CreditDefaultSwaption
from pricebook.products.forward_cds import ForwardCDS

#: Points per pillar segment in the protection legs — the engine-wide committed count, imported
#: rather than restated so a forward CDS and a spot CDS cannot be priced on different grids.
from pricebook.engine.credit_default_swap import _DEFAULT_PER_SEGMENT


def _legs_to(dates: tuple, day_count, model, currency) -> legs.CdsLegs:
    """The spot legs to `dates[-1]`, per unit notional, from the shared L1 functional."""
    discount = model.market.curves.discount(currency)
    survival = model.survival(currency)
    schedule = CouponSchedule(dates, day_count)
    horizon = discount.time_measure.year_fraction(dates[-1])
    return legs.CdsLegs(
        annuity=legs.risky_pv01(schedule.periods(), discount, survival),
        protection=legs.protection_leg(
            discount, survival, model.recovery, horizon, _DEFAULT_PER_SEGMENT))


def forward_quote(product, model) -> tuple[float, float]:
    """`(S(t,tF,T), RPV01(t,tF,T))` — named surface because the oracles are about *these*.

    Works for both products: each carries a schedule spanning `[t, T]` and a date on it (the
    forward start, or the option expiry) that splits it.
    """
    currency = product.notional.currency
    split = product.forward_start if isinstance(product, ForwardCDS) else product.expiry
    index = product.schedule.dates.index(split)
    near = _legs_to(product.schedule.dates[: index + 1], product.schedule.day_count,
                    model, currency)
    far = _legs_to(product.schedule.dates, product.schedule.day_count, model, currency)
    return legs.forward_spread(near, far), legs.forward_annuity(near, far)


@register(ForwardCDS, CreditModel)
def price_forward_cds(
    product: ForwardCDS, model: CreditModel
) -> PricingResult | PricingFailure:
    """`ω·(S(t,tF,T) − S₀)·RPV01(t,tF,T)·N`."""
    currency = product.notional.currency
    if is_historical(product.schedule.dates[-1], model.market.valuation_date):
        # Invariant 6: a matured contract is the shell's business, PV 0 before any market lookup.
        return PricingResult(pv=Money(0.0, currency))
    try:
        spread, annuity = forward_quote(product, model)
    except KeyError as exc:
        return PricingFailure(f"cannot price the forward CDS: {exc}")
    return PricingResult(
        pv=Money(
            product.direction.omega * (spread - product.spread) * annuity
            * product.notional.amount,
            currency))


@register(CreditDefaultSwaption, CreditSwaptionModel)
def price_credit_default_swaption(
    product: CreditDefaultSwaption, model: CreditSwaptionModel
) -> PricingResult | PricingFailure:
    """`RPV01(0,tE,T)·Black(S(0,tE,T), K, σ²tE, ω)·N` — eq 9.3."""
    currency = product.notional.currency
    if is_historical(product.expiry, model.market.valuation_date):
        # An expired option is the shell's business: exercise or lapse already happened.
        return PricingResult(pv=Money(0.0, currency))
    try:
        forward, annuity = forward_quote(product, model)
        discount = model.market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the default swaption: {exc}")

    expiry_years = discount.time_measure.year_fraction(product.expiry)
    total_variance = model.volatility * model.volatility * expiry_years
    # `SwaptionDirection` already satisfies Black's `OptionDirection` protocol — payer is a call
    # on the spread (ω=+1), receiver a put (ω=−1) — so it is passed straight through rather than
    # translated into another product's enum. The protocol exists for exactly this (tranche 33b).
    #
    # Black's bracket is undiscounted and per unit (tranche 5); the **risky annuity** is the
    # numéraire, so it — not a discount factor — scales it back to today. A `df_at(tE)` here would
    # discount twice and drop the survival weighting the knockout relies on.
    bracket = black_price(forward, product.strike, total_variance, product.direction)
    return PricingResult(pv=Money(bracket * annuity * product.notional.amount, currency))
