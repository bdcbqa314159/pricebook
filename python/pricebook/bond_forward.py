"""Bond forward: forward pricing, forward DV01, forward spread.

A bond forward is an agreement to buy/sell a bond at a future date.
Forward price is determined by cash-and-carry arbitrage.

    from pricebook.bond_forward import BondForward

    fwd = BondForward(bond, settlement, delivery, repo_rate)
    result = fwd.price(curve)

References:
    Tuckman & Serrat, *Fixed Income Securities*, Wiley, 2012, Ch. 15.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.bond import FixedRateBond
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


@dataclass
class BondForwardResult:
    """Bond forward pricing result."""
    forward_dirty: float
    forward_clean: float
    spot_dirty: float
    carry: float
    repo_cost: float
    coupon_income: float
    forward_dv01: float


class BondForward:
    """Forward contract on a fixed-rate bond.

    Args:
        bond: the underlying bond.
        settlement: spot settlement date (today or T+1/T+2).
        delivery: forward delivery date.
        repo_rate: financing rate for the carry period.
    """

    def __init__(
        self,
        bond: FixedRateBond,
        settlement: date,
        delivery: date,
        repo_rate: float,
    ):
        if delivery <= settlement:
            raise ValueError(f"delivery ({delivery}) must be after settlement ({settlement})")
        self.bond = bond
        self.settlement = settlement
        self.delivery = delivery
        self.repo_rate = repo_rate

    def _coupon_income(self) -> float:
        """Sum of coupons received between settlement and delivery."""
        total = 0.0
        for cf in self.bond.coupon_leg.cashflows:
            if self.settlement < cf.payment_date <= self.delivery:
                total += cf.amount
        return total / self.bond.face_value * 100.0

    def price(self, curve: DiscountCurve) -> BondForwardResult:
        """Compute forward price from spot price + carry.

        Forward dirty = spot dirty × (1 + repo × T) - coupon_income.
        """
        spot_dirty = self.bond.dirty_price(curve)
        days = (self.delivery - self.settlement).days
        dt = days / 365.0
        repo_cost = spot_dirty * self.repo_rate * dt
        coupon_income = self._coupon_income()
        fwd_dirty = spot_dirty + repo_cost - coupon_income
        accrued_at_delivery = self.bond.accrued_interest(self.delivery)
        fwd_clean = fwd_dirty - accrued_at_delivery
        carry = coupon_income - repo_cost

        # Forward DV01: bump yield 1bp and recompute forward
        ytm = self.bond.yield_to_maturity(spot_dirty, self.settlement)
        price_up = self.bond._price_from_ytm(ytm + 0.0001, self.settlement)
        fwd_up = price_up + price_up * self.repo_rate * dt - coupon_income
        fwd_dv01 = fwd_dirty - fwd_up

        return BondForwardResult(
            forward_dirty=fwd_dirty,
            forward_clean=fwd_clean,
            spot_dirty=spot_dirty,
            carry=carry,
            repo_cost=repo_cost,
            coupon_income=coupon_income,
            forward_dv01=fwd_dv01,
        )
