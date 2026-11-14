"""The credit default swap pricer (L4) - O'Kane ch 6.

    Premium leg  = S0 * RPV01,   RPV01 = 1/2 * sum_n D_n * Z(t_n) * (Q(t_{n-1}) + Q(t_n))   (6.2/6.4)
    Protection   = (1-R) * integral_0^T Z(t) * lambda(t) * Q(t) dt
    V(buyer)     = Protection - Premium,   S_par = Protection / RPV01

**The protection leg IS tranche 25's CVA integral**, and that is the capstone seam rather than a
coincidence. `(1-R) integral Z*lambda*Q dt` is `LGD * integral DF*EPE*lambda*S dt` with the exposure
flat at 1: a CDS is a CVA on a unit notional. So this tranche's correctness anchor is **cross-arc** -
project five's leg reconciled against project three's integral, different code from different sources
- which no within-CDS check could replace.

**RPV01 uses O'Kane's 6.4, and that form is an APPROXIMATION, named as one.** The exact risky annuity
(6.3) integrates the coupon accrued at default over each period; 6.4 replaces that integral with
`1/2 * D_n * Z(t_n) * (Q(t_{n-1}) - Q(t_n))`, on the observation that a default lands on average
halfway through a period. O'Kane puts its error at `O(10^-5)` on the PV01 and notes it discounts to
the period end rather than the midpoint "to have one less discount factor to calculate". Combining
the two terms gives the averaged-survival form above. The oracle asserts the approximation's error
against the exact integral rather than assuming the book's figure.

**The `1/2 (Q_{n-1} + Q_n)` is where this goes wrong silently.** Replace it by `Q_n` and the
accrual-on-default half disappears; on a flat hazard with annual periods the difference is small
enough to hide, which is why the fixture carries a term-structured hazard and non-unit unequal
accruals from the first line. Three tranches (3, 8, 9) shipped oracles blind to an accrual because
their fixtures were annual on ACT/365F; this one is built to see it.

**`par_spread` and `risky_pv01` are named surface before their consumers exist**, which is normally
the speculation §6 forbids - except that two consumers are already in the committed queue (the CDS
bootstrap at T39 inverts `par_spread`; the forward CDS and default swaption at T40 need `RPV01`), so
rule-of-two is met by named, scheduled consumers rather than by hope. Extracting them also makes the
oracle's job possible at all: the credit triangle is a statement about `par_spread`, not about a PV.

Provenance:
  stock:  none
  source: OKANE-2008 (docs/references.md) ch 6 - eqs 6.2/6.3/6.4 (the premium leg and risky PV01)
          and ch 3 (the protection leg and the credit triangle `S = lambda(1-R)`), read in `-raw`
          and `-layout` and reconciled
  oracle: tests/L4/test_credit_default_swap.py (PRE-REGISTERED) - the credit triangle as a
          convergent reduction under flat hazard/rate; **the protection leg against tranche 25's
          `unilateral_cva` on a flat unit exposure** (cross-arc, with the gap converging under joint
          refinement so the two are one integral rather than two close numbers); the trapezoid
          inside O'Kane's Riemann bracket with the bracket's `O(1/M)` tightening asserted; RPV01
          positive and decreasing in the hazard; the protection leg increasing in it; `V(S_par) = 0`
          as a telescoping guard; `dV/dS0 = -RPV01` exactly; buyer = -seller
  crosscheck: quantlib IsdaCdsEngine - deferred to the optional QuantLib tier
  slice:  tranche 38 - the credit default swap: project five opens
"""

from __future__ import annotations

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.market import credit_legs as legs
from pricebook.foundation.time_measure import is_historical
from pricebook.models.credit import CreditModel
from pricebook.products.credit_default_swap import CreditDefaultSwap

#: Points **per pillar segment** in the protection-leg quadrature — not points in total, since
#: tranche 39 made the integration break at every hazard pillar (`market/credit_legs.py`). Numerical
#: config arriving with its first consumer (§3.5), and the count is **measured, not chosen**.
#:
#: Against a 200,000-per-segment reference on the term-structured fixture: 1.0e-07 at 50,
#: 6.3e-09 at 200, 1.0e-09 at 500, 6.3e-11 at 2000 — clean second order, 4x per 2x refinement.
#: Cost at 2000 is 2.5 ms for a three-segment curve.
#:
#: 2000 is committed, and the reason is not accuracy alone: on a **flat** hazard there are no
#: interior pillars, so one segment of 2000 points is bit-for-bit the quadrature v0.45.0 shipped —
#: which keeps that version's published closed-form figure (+5.672e-10) exactly reproducible while
#: the term-structured case improves from 1.8e-05 to 6.3e-11. 500 would cost a quarter as much for
#: 1.0e-09, which is also ample; the point of stating the table is that the choice is defensible
#: rather than inherited.
_DEFAULT_PER_SEGMENT = 2000


def _live_periods(product: CreditDefaultSwap, valuation_date):
    """The premium periods still to pay - invariant 6, through the one shared atom."""
    return [
        period
        for period in product.schedule.periods()
        if not is_historical(period[1], valuation_date)
    ]


def risky_pv01(product: CreditDefaultSwap, model: CreditModel) -> float:
    """`RPV01` per unit notional - O'Kane 6.4, premium payments plus coupon accrued at default.

    `1/2 * sum D_n * Z(t_n) * (Q(t_{n-1}) + Q(t_n))`. The averaged survival is the combined form:
    the surviving premium `Q(t_n)` plus the accrual-on-default `1/2 (Q(t_{n-1}) - Q(t_n))`.
    """
    market = model.market
    currency = product.notional.currency
    return legs.risky_pv01(
        _live_periods(product, market.valuation_date),
        market.curves.discount(currency),
        model.survival(currency))


def protection_leg(
    product: CreditDefaultSwap, model: CreditModel, per_segment: int = _DEFAULT_PER_SEGMENT
) -> float:
    """`(1−R) ∫₀ᵀ Z(t)λ(t)Q(t)dt` times the notional — the ONE shared L1 functional.

    The engine integrated this itself until tranche 39, pointwise in `default_density` on a uniform
    grid. That loses an order at every hazard pillar, where `λQ` jumps, and sat 1.846e-05 below the
    truth on a term-structured curve; the shared functional reads the hazard once per pillar
    segment, which is exact between pillars and second order. Adopting it moved a shipped number
    **toward** an independent reference — a ruled correctness fix (`OPEN.md` OI-4, closed), not a
    silent reordering — and it makes this the single formula the engine and the bootstrap share,
    so a bootstrapped curve now reprices its own calibrating CDS to zero through the engine
    (`docs/conventions_register.md` C11, guarded).
    """
    market = model.market
    currency = product.notional.currency
    discount = market.curves.discount(currency)

    periods = _live_periods(product, market.valuation_date)
    if not periods:
        return 0.0
    horizon = discount.time_measure.year_fraction(periods[-1][1])
    return legs.protection_leg(
        discount, model.survival(currency), model.recovery, horizon, per_segment
    ) * product.notional.amount


def riemann_bracket(
    product: CreditDefaultSwap, model: CreditModel, per_segment: int = _DEFAULT_PER_SEGMENT
) -> tuple[float, float]:
    """O'Kane's explicit lower and upper Riemann sums for the protection leg.

    Named surface because the oracle's claim is that the trapezoid lies **inside** the bracket - a
    requirement for all parameters, where a tolerance would be a guess - and because the bracket's
    width is what tightens at `O(1/M)`, which is the rate the suite asserts rather than a band.
    Returned as `(lower, upper)` whichever way the integrand happens to be monotone, so the caller
    need not know.
    """
    market = model.market
    currency = product.notional.currency
    discount = market.curves.discount(currency)
    survival = model.survival(currency)

    periods = _live_periods(product, market.valuation_date)
    if not periods:
        return 0.0, 0.0
    horizon = discount.time_measure.year_fraction(periods[-1][1])
    if horizon <= 0.0:
        return 0.0, 0.0
    scale = (1.0 - model.recovery) * product.notional.amount

    # The SAME breaks as the trapezoid it brackets. On a uniform grid these sums would bracket a
    # different integral, because `λQ` jumps at a pillar and a cell straddling one is bounded by
    # neither of its endpoints.
    left = right = 0.0
    for lo, hi in zip(segment_breaks := legs.segment_breaks(survival, horizon),
                      segment_breaks[1:]):
        hazard = survival.hazard(0.5 * (lo + hi))
        step = (hi - lo) / per_segment
        for index in range(per_segment):
            a_t, b_t = lo + step * index, lo + step * (index + 1)
            left += discount.df_at(a_t) * survival.survival(a_t) * step * hazard
            right += discount.df_at(b_t) * survival.survival(b_t) * step * hazard
    first, second = scale * left, scale * right
    return min(first, second), max(first, second)


def par_spread(
    product: CreditDefaultSwap, model: CreditModel, per_segment: int = _DEFAULT_PER_SEGMENT
) -> float:
    """`S_par = Protection / (RPV01 * notional)` - the spread making the contract worth zero.

    The credit triangle is a statement about *this*, which is why it is extracted rather than left
    inside a PV: under a flat hazard, flat rates and the continuous limit it is exactly
    `lambda(1-R)`, whatever the curve, because the `integral ZQ` common to both legs cancels.
    """
    annuity = risky_pv01(product, model)
    if annuity <= 0.0:
        raise ValueError("a par spread needs a positive risky annuity; the contract has no premium legs")
    return protection_leg(product, model, per_segment) / (annuity * product.notional.amount)


@register(CreditDefaultSwap, CreditModel)
def price_credit_default_swap(
    product: CreditDefaultSwap, model: CreditModel
) -> PricingResult | PricingFailure:
    """`omega * (Protection - S0 * RPV01 * notional)`."""
    market = model.market
    currency = product.notional.currency
    if not _live_periods(product, market.valuation_date):
        # Invariant 6: a matured contract is the shell's business, PV 0 before any market lookup.
        return PricingResult(pv=Money(0.0, currency))
    # The guard wraps the whole computation, not a probe of the discount curve alone: the
    # survival lookup happens inside the legs, so a probe front-loads one missing-curve failure
    # and lets the other escape as a raw KeyError, against invariant 4 (failure is a value).
    try:
        protection = protection_leg(product, model)
        premium = product.spread * risky_pv01(product, model) * product.notional.amount
    except KeyError as exc:
        return PricingFailure(f"cannot price the credit default swap: {exc}")
    return PricingResult(
        pv=Money(product.direction.omega * (protection - premium), currency)
    )
