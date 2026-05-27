"""Zero-coupon (discount) bonds.

T-Bills, LTN (Brazil), CETES (Mexico), and other discount securities
that pay no coupons — just face value at maturity.

    from pricebook.fixed_income.zero_coupon_bond import ZeroCouponBond

    bill = ZeroCouponBond(issue, maturity, day_count=DayCountConvention.ACT_360)
    price = bill.price(discount_curve)
    ytm = bill.yield_from_price(price)

Pricing:
    Price = Face × df(T)         [curve-based]
    Price = Face / (1 + r × τ)   [simple yield, money market]
    Price = Face × exp(-r × τ)   [continuous yield]

References:
    Fabozzi (2012). Bond Markets, Analysis and Strategies, Ch 6.
"""

from __future__ import annotations

import math
from datetime import date

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import Calendar, BusinessDayConvention


class ZeroCouponBond:
    """A zero-coupon (discount) bond.

    Pays face_value at maturity, no intermediate coupons.

    Args:
        issue_date: issue / settlement date.
        maturity: maturity date.
        face_value: principal (default 100).
        day_count: for yield/price calculations.
        calendar: business day calendar.
        settlement_days: T+N for settlement.
    """

    def __init__(
        self,
        issue_date: date,
        maturity: date,
        face_value: float = 100.0,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        calendar: Calendar | None = None,
        settlement_days: int = 0,
    ):
        if issue_date >= maturity:
            raise ValueError(f"issue_date ({issue_date}) must be before maturity ({maturity})")
        self.issue_date = issue_date
        self.maturity = maturity
        self.face_value = face_value
        self.day_count = day_count
        self.calendar = calendar
        self.settlement_days = settlement_days
        self.coupon_rate = 0.0
        self.frequency = None

    def price(self, curve: DiscountCurve) -> float:
        """Price from a discount curve: Face × df(T)."""
        return self.face_value * curve.df(self.maturity)

    # Alias for compatibility with FixedRateBond
    dirty_price = price

    def clean_price(self, curve: DiscountCurve, settlement: date | None = None) -> float:
        """Clean price = dirty price (no accrued for zeros)."""
        return self.price(curve)

    def pv(self, curve: DiscountCurve) -> float:
        """Present value (alias for price)."""
        return self.price(curve)

    def price_from_yield_simple(
        self, yield_rate: float, settlement: date | None = None,
    ) -> float:
        """Price from simple (money-market) yield.

        P = Face / (1 + r × τ)

        This is the T-Bill convention (US, Mexico CETES).
        """
        settle = settlement or self.issue_date
        tau = year_fraction(settle, self.maturity, self.day_count)
        if tau <= 0:
            return self.face_value
        return self.face_value / (1.0 + yield_rate * tau)

    def price_from_discount_rate(
        self, discount_rate: float, settlement: date | None = None,
    ) -> float:
        """Price from discount rate (bank discount basis).

        P = Face × (1 - d × τ)

        Used for US T-Bill quoting convention.
        """
        settle = settlement or self.issue_date
        tau = year_fraction(settle, self.maturity, self.day_count)
        return self.face_value * (1.0 - discount_rate * tau)

    def price_from_yield_continuous(
        self, yield_rate: float, settlement: date | None = None,
    ) -> float:
        """Price from continuously compounded yield.

        P = Face × exp(-r × τ)
        """
        settle = settlement or self.issue_date
        tau = year_fraction(settle, self.maturity, self.day_count)
        return self.face_value * math.exp(-yield_rate * tau)

    def yield_simple(
        self, market_price: float, settlement: date | None = None,
    ) -> float:
        """Simple (money-market) yield from price.

        r = (Face / P - 1) / τ
        """
        settle = settlement or self.issue_date
        tau = year_fraction(settle, self.maturity, self.day_count)
        if tau <= 0 or market_price <= 0:
            return 0.0
        return (self.face_value / market_price - 1.0) / tau

    def discount_rate(
        self, market_price: float, settlement: date | None = None,
    ) -> float:
        """Bank discount rate from price.

        d = (1 - P / Face) / τ
        """
        settle = settlement or self.issue_date
        tau = year_fraction(settle, self.maturity, self.day_count)
        if tau <= 0:
            return 0.0
        return (1.0 - market_price / self.face_value) / tau

    def yield_continuous(
        self, market_price: float, settlement: date | None = None,
    ) -> float:
        """Continuously compounded yield from price.

        r = -ln(P / Face) / τ
        """
        settle = settlement or self.issue_date
        tau = year_fraction(settle, self.maturity, self.day_count)
        if tau <= 0 or market_price <= 0:
            return 0.0
        return -math.log(market_price / self.face_value) / tau

    def dv01(self, curve: DiscountCurve) -> float:
        """DV01: price change for 1bp parallel shift in curve."""
        p_base = self.price(curve)
        p_up = self.price(curve.bumped(0.0001))
        return abs(p_up - p_base)

    def modified_duration(
        self, market_price: float, settlement: date | None = None,
    ) -> float:
        """Modified duration of a zero = τ / (1 + r × τ) ≈ τ for short bills."""
        settle = settlement or self.issue_date
        tau = year_fraction(settle, self.maturity, self.day_count)
        r = self.yield_simple(market_price, settlement)
        denom = 1.0 + r * tau
        return tau / denom if denom > 0 else tau

    def pv_ctx(self, ctx) -> float:
        """PV using PricingContext."""
        curve = ctx.discount_curve
        if curve is None:
            raise ValueError("No discount curve in context")
        return self.pv(curve)

    @classmethod
    def from_convention(cls, conv, issue_date, maturity, face_value=100.0):
        """Create ZeroCouponBond from a convention object."""
        from pricebook.core.calendar import get_calendar
        cal = get_calendar(conv.calendar_currency) if hasattr(conv, 'calendar_currency') else None
        return cls(
            issue_date=issue_date, maturity=maturity, face_value=face_value,
            day_count=conv.day_count, calendar=cal,
            settlement_days=getattr(conv, 'settlement_days', 0),
        )

    def to_dict(self) -> dict:
        return {
            "type": "zero_coupon_bond",
            "issue_date": self.issue_date.isoformat(),
            "maturity": self.maturity.isoformat(),
            "face_value": self.face_value,
            "day_count": self.day_count.value,
            "settlement_days": self.settlement_days,
        }
