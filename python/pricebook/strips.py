"""STRIPS: Separate Trading of Registered Interest and Principal of Securities.

Strip a coupon bond into individual zero-coupon securities:
- C-STRIPS: each coupon payment becomes a separate zero.
- P-STRIPS: the principal payment at maturity.

    from pricebook.strips import strip_bond, price_strip

    zeros = strip_bond(bond)
    pv = price_strip(face=100, maturity=date(2030,1,15), curve=curve)

References:
    Tuckman & Serrat, *Fixed Income Securities*, Wiley, 2012, Ch. 1.
    US Treasury STRIPS program, 31 CFR Part 356.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.bond import FixedRateBond
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


@dataclass
class StripSecurity:
    """A single zero-coupon strip."""
    strip_type: str          # "C-STRIP" or "P-STRIP"
    maturity: date
    face_value: float        # cashflow amount
    source_bond_coupon: float | None   # original bond coupon rate (for C-STRIP)


@dataclass
class StripResult:
    """Result of stripping a bond."""
    c_strips: list[StripSecurity]
    p_strip: StripSecurity
    total_face: float


def strip_bond(bond: FixedRateBond) -> StripResult:
    """Strip a coupon bond into C-STRIPS and P-STRIP.

    Each coupon cashflow becomes a C-STRIP with face = coupon amount.
    The principal becomes a P-STRIP with face = bond face value.
    """
    c_strips = []
    for cf in bond.coupon_leg.cashflows:
        c_strips.append(StripSecurity(
            strip_type="C-STRIP",
            maturity=cf.payment_date,
            face_value=cf.amount,
            source_bond_coupon=bond.coupon_rate,
        ))

    p_strip = StripSecurity(
        strip_type="P-STRIP",
        maturity=bond.maturity,
        face_value=bond.face_value,
        source_bond_coupon=None,
    )

    total = sum(s.face_value for s in c_strips) + p_strip.face_value
    return StripResult(c_strips=c_strips, p_strip=p_strip, total_face=total)


def price_strip(
    face_value: float,
    maturity: date,
    curve: DiscountCurve,
) -> float:
    """Price a zero-coupon strip: PV = face × df(maturity)."""
    return face_value * curve.df(maturity)


def strip_yield(
    market_price: float,
    face_value: float,
    settlement: date,
    maturity: date,
    day_count: DayCountConvention = DayCountConvention.ACT_ACT_ISDA,
) -> float:
    """Yield of a zero-coupon strip (continuously compounded).

    y = -ln(price/face) / T.
    """
    import math
    t = year_fraction(settlement, maturity, day_count)
    if t <= 0 or market_price <= 0:
        return 0.0
    return -math.log(market_price / face_value) / t


def reconstruct_bond_price(
    strips: StripResult,
    curve: DiscountCurve,
) -> float:
    """Reconstruct bond price from its strips.

    Sum of all strip PVs should equal the bond dirty price.
    """
    total = 0.0
    for s in strips.c_strips:
        total += price_strip(s.face_value, s.maturity, curve)
    total += price_strip(strips.p_strip.face_value, strips.p_strip.maturity, curve)
    return total
