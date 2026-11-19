"""The collateralised swap under CSA discounting (L4). `LICHTERS-2015` ch 5.

A perfectly-collateralised trade (cash CSA in the trade currency) earns the collateral rate on
posted collateral, so **every cashflow discounts at OIS** while **each floating coupon projects off
its tenor curve**:

    PV = omega * N * [ sum_i f^tenor(t_{i-1}, t_i) * tau_i * df^OIS_i
                       - c * sum_i tau_i * df^OIS_i ]

with `f^tenor` from eq (1.1) on the projection curve. Two curves, one discounting - which is the
whole content of the post-crisis change, and the reason the floating leg no longer telescopes
(eq (1.2) holds only when one curve both projects and discounts; see `market/multi_curve.py`).

**One pricer, two eras, chosen by what the snapshot holds.** The arc's design invariant is that
single-curve and multi-curve are *co-equal, deliberately-selectable* modes rather than a world and
its degenerate case. This engine honours it in the strongest available form: there is **no second
code path to atrophy**. A snapshot carrying a projection curve for the swap's tenor is the
post-crisis era; a snapshot carrying only a discount curve is the pre-crisis one, and the pricer
then projects off that discount curve - which is precisely what the single-curve world does. The era
is a property of the market, not a flag on the call, so neither mode can be reached by accident and
neither can be forgotten.

That matters because the way a mode rots is silent (the tranche 33b input-population degeneracy): if
every new fixture carried projections, the single-curve branch would leave the test population and
could break with the suite green. So the oracle asserts the pair - the eras **agree** where they
coincide and **differ by the basis** where a real basis exists - and the second half is the one that
fails if this engine ever quietly falls back to one curve.

**Invariant 6 is applied per period, not per trade.** A swap straddling the valuation date has
settled coupons and live ones; the settled periods are the shell's business and are excluded, which
is also the only way the remaining legs are discounted at positive `t`.

Provenance:
  stock:  none - the archived predecessor had no vanilla IRS pricer
  source: LICHTERS-2015 (docs/references.md) ch 5 (CSA discounting: a collateralised trade
          discounts at the collateral rate) with ch 3.5's OIS/projection split and eq (1.1) for the
          forward; transcribed in both renderings and reconciled
  oracle: tests/L4/test_csa_discounting.py - the legs transcribed pointwise against the test's own
          arithmetic; a swap struck at an INDEPENDENTLY computed multi-curve par rate is worth
          zero (the pricer's own par rate kept as a labelled consistency guard, since it
          telescopes); the pre-crisis reduction to the single-curve price at OIS = projection; and
          the single-curve invariant as a pair - agreeing where the eras coincide, differing by the
          basis where one exists
  crosscheck: quantlib VanillaSwap with a separate discounting curve (multi-curve collateralised
          pricing) - deferred to the optional QuantLib tier
  slice:  tranche 55 - CSA discounting
"""

from __future__ import annotations

from datetime import date

from pricebook.engine.registry import register
from pricebook.foundation.money import Currency, Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.schedule import CouponSchedule
from pricebook.foundation.tenor import Tenor
from pricebook.foundation.time_measure import is_historical
from pricebook.market.curve import CurveHandle
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.discounting_model import DiscountingModel
from pricebook.products.interest_rate_swap import InterestRateSwap


def _live(schedule: CouponSchedule, valuation_date: date) -> list[tuple[date, date, float]]:
    """The periods still to pay. Invariant 6 per period: a coupon on or before the valuation date
    has settled, so it is excluded from the mark rather than discounted at a non-positive `t`."""
    return [
        (start, pay, accrual)
        for start, pay, accrual in schedule.periods()
        if not is_historical(pay, valuation_date)
    ]


def floating_leg_value(
    schedule: CouponSchedule, projection: CurveHandle, discount: CurveHandle, valuation_date: date
) -> float:
    """`sum_i f^tenor(t_{i-1},t_i) * tau_i * df^OIS_i`, per unit notional.

    **Projection and discounting are separate arguments, and that is the point of the tranche.**
    Passing one curve twice is the single-curve era and is a legitimate call; passing the projection
    curve as the discount is the pre-crisis *error*, and a named surface is what lets the oracle say
    so. The accruals cancel in `f * tau` by eq (1.1), but they are kept written out: the schedule's
    own day count is what defines both, and collapsing them would hide which convention is in use.
    """
    total = 0.0
    for start, pay, accrual in _live(schedule, valuation_date):
        forward = (projection.df(start) / projection.df(pay) - 1.0) / accrual
        total += forward * accrual * discount.df(pay)
    return total


def fixed_leg_value(
    schedule: CouponSchedule, fixed_rate: float, discount: CurveHandle, valuation_date: date
) -> float:
    """`c * sum_i tau_i * df^OIS_i`, per unit notional - the rate times the OIS annuity."""
    return fixed_rate * sum(
        accrual * discount.df(pay) for _, pay, accrual in _live(schedule, valuation_date)
    )


def _curves(market: MarketSnapshot, currency: Currency, tenor: Tenor) -> tuple[CurveHandle, CurveHandle]:
    """`(projection, discount)` for the era this snapshot describes — `CurveSet.curves_for_era`.

    This used to carry the lookup itself and claim to be "the one place the mode is selected",
    which the collateralised capstone's inline copy made false. Re-audit #6 (F-6.6) moved the
    choice to L1, where `projections` lives and below both consumers; the claim is true now.
    """
    return market.curves.curves_for_era(currency, tenor)


def par_swap_rate(
    schedule: CouponSchedule, market: MarketSnapshot, currency: Currency, tenor: Tenor
) -> float:
    """The fixed rate making the collateralised swap worth zero: the floating leg over the annuity.

    Composes the **same** legs the pricer composes, so a par reprice telescopes - it is kept as a
    consistency guard and labelled as one in the oracle, the correctness anchors being the
    independently computed par rate and the pre-crisis reduction (§4: a par-reprice-to-zero is
    satisfied by construction when construction and pricing share a convention, even a wrong one).
    """
    projection, discount = _curves(market, currency, tenor)
    annuity = fixed_leg_value(schedule, 1.0, discount, market.valuation_date)
    if annuity == 0.0:
        raise ValueError("a swap with no live periods has no par rate")
    return floating_leg_value(schedule, projection, discount, market.valuation_date) / annuity


@register(InterestRateSwap, DiscountingModel)
def price_interest_rate_swap(
    product: InterestRateSwap, model: DiscountingModel
) -> PricingResult | PricingFailure:
    """`omega * N * (floating - fixed)`, both legs discounted at OIS."""
    market = model.market
    currency = product.notional.currency
    try:
        projection, discount = _curves(market, currency, product.tenor)
    except KeyError as exc:
        return PricingFailure(f"cannot price the swap: {exc}")

    floating = floating_leg_value(product.schedule, projection, discount, market.valuation_date)
    fixed = fixed_leg_value(product.schedule, product.fixed_rate, discount, market.valuation_date)
    unit = product.direction.omega * (floating - fixed)
    return PricingResult(pv=Money(product.notional.amount * unit, currency))
