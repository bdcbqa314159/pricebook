"""SwaptionEngine — Hull-White European swaption via Jamshidian (L4).

Stateless. A European swaption on a fixed-vs-float swap equals an option on the
coupon bond built from the fixed leg (coupons + notional redemption), struck at
par. Under HW the bond price at expiry is monotonic in the single state r(T0), so
(Jamshidian) the coupon-bond option decomposes into a portfolio of options on the
constituent zero-coupon bonds, each struck at its value at the critical rate r*:

    receiver swaption = sum_i c_i * ZBC(T0, t_i, K_i)   (call on the coupon bond)
    payer swaption    = sum_i c_i * ZBP(T0, t_i, K_i)   (put on the coupon bond)

where sum_i c_i * P(T0, t_i; r*) = notional and K_i = P(T0, t_i; r*). The ZBC/ZBP
are the S07 HW analytic ZCB options; r* is found by bisection (the coupon-bond
value is monotonically decreasing in r).

Provenance:
  quarry: python/pricebook/pricing/ (swaption)
  source: Jamshidian (1989); Brigo & Mercurio s.3.3
  oracle: put-call parity + ATM symmetry + sigma->0 intrinsic (S08)
  slice:  S08
"""

from __future__ import annotations

from datetime import date

from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.foundation.solvers import bisect_root
from pricebook_ng.products.swap import VanillaSwap
from pricebook_ng.products.swaption import Swaption
from pricebook_ng.models.hull_white import HullWhite


def coupon_bond_cashflows(swap: VanillaSwap) -> tuple[list[date], list[float], float]:
    """The coupon bond a swap's fixed leg decomposes into: fixed coupons plus the
    notional redeemed at the last date. Shared by the analytic + MC swaption engines
    and the MC exposure engine."""
    notional = swap.float_leg.face.amount
    dates = [cf.date for cf in swap.fixed_leg.cashflows]
    amounts = [cf.amount.amount for cf in swap.fixed_leg.cashflows]
    amounts[-1] += notional
    return dates, amounts, notional


class SwaptionEngine:
    """Prices a European swaption under Hull-White by Jamshidian decomposition."""

    def price(
        self,
        swaption: Swaption,
        model: HullWhite,
        numerics: NumericalConfig,
    ) -> PricingResult | PricingFailure:
        expiry = swaption.expiry
        if expiry < model.market.valuation_date:
            return PricingFailure(f"swaption expiry {expiry} precedes valuation")

        currency = swaption.swap.float_leg.face.currency
        dates, amounts, notional = coupon_bond_cashflows(swaption.swap)

        def coupon_bond(short_rate: float) -> float:
            return sum(a * model.zero_bond(expiry, d, short_rate) for a, d in zip(amounts, dates))

        # r*: the short rate at expiry that prices the coupon bond at par (notional).
        r_star = bisect_root(lambda r: coupon_bond(r) - notional, -1.0, 1.0)

        is_call = not swaption.swap.pay_fixed  # payer swaption = put on the coupon bond
        value = sum(
            a * model.zero_bond_option(expiry, d, model.zero_bond(expiry, d, r_star), is_call)
            for a, d in zip(amounts, dates)
        )
        return PricingResult(pv=Money(value, currency))
