"""
European swaption — the right to enter an interest rate swap.

A payer swaption gives the right to enter a payer swap (pay fixed).
A receiver swaption gives the right to enter a receiver swap (receive fixed).

Priced with Black-76 on the forward swap rate:

    price = annuity * Black76(forward_swap_rate, strike, vol, T_expiry)

where:
    forward_swap_rate = par rate of the forward-starting underlying swap
    annuity = sum of year_frac * df for the fixed leg of the underlying
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pricebook.black76 import OptionType, black76_price
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency, StubType
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.calendar import Calendar, BusinessDayConvention


class SwaptionType(Enum):
    PAYER = "payer"        # right to pay fixed (call on swap rate)
    RECEIVER = "receiver"  # right to receive fixed (put on swap rate)


class Swaption:
    """
    European swaption on a vanilla interest rate swap.

    Args:
        expiry: option expiry date (= start of the underlying swap).
        swap_end: maturity of the underlying swap.
        strike: the fixed rate of the underlying swap.
        swaption_type: PAYER or RECEIVER.
        notional: notional of the underlying swap.
        fixed_frequency: coupon frequency of the fixed leg.
        float_frequency: coupon frequency of the floating leg.
        fixed_day_count: day count for the fixed leg.
        float_day_count: day count for the floating leg.
        calendar: business day calendar.
        convention: business day convention.
        stub: stub type for schedule generation.
        eom: end-of-month rule.
    """

    def __init__(
        self,
        expiry: date,
        swap_end: date,
        strike: float,
        swaption_type: SwaptionType = SwaptionType.PAYER,
        notional: float = 1_000_000.0,
        fixed_frequency: Frequency = Frequency.SEMI_ANNUAL,
        float_frequency: Frequency = Frequency.QUARTERLY,
        fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360,
        float_day_count: DayCountConvention = DayCountConvention.ACT_360,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        stub: StubType = StubType.SHORT_FRONT,
        eom: bool = True,
    ):
        if expiry >= swap_end:
            raise ValueError("expiry must be before swap_end")

        self.expiry = expiry
        self.swap_end = swap_end
        self.strike = strike
        self.swaption_type = swaption_type
        self.notional = notional

        # Build the underlying forward-starting swap (payer by convention)
        self.underlying = InterestRateSwap(
            start=expiry,
            end=swap_end,
            fixed_rate=strike,
            direction=SwapDirection.PAYER,
            notional=notional,
            fixed_frequency=fixed_frequency,
            float_frequency=float_frequency,
            fixed_day_count=fixed_day_count,
            float_day_count=float_day_count,
            calendar=calendar,
            convention=convention,
            stub=stub,
            eom=eom,
        )

    def forward_swap_rate(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """Par rate of the underlying forward-starting swap."""
        return self.underlying.par_rate(curve, projection_curve)

    def annuity(self, curve: DiscountCurve) -> float:
        """Annuity of the underlying swap's fixed leg."""
        return self.underlying.fixed_leg.annuity(curve)

    def pv(
        self,
        curve: DiscountCurve,
        vol_surface,
        projection_curve: DiscountCurve | None = None,
        valuation_date: date | None = None,
    ) -> float:
        """
        Swaption price using Black-76.

        price = notional * annuity * Black76(F, K, vol, T)

        Args:
            curve: discount curve.
            vol_surface: object with vol(expiry, strike) method.
            projection_curve: forward projection curve (None = single-curve).
            valuation_date: date for computing time to expiry.
                Defaults to curve's reference date.
        """
        if valuation_date is None:
            valuation_date = curve.reference_date

        fwd = self.forward_swap_rate(curve, projection_curve)
        ann = self.annuity(curve)

        time_to_expiry = year_fraction(
            valuation_date, self.expiry, DayCountConvention.ACT_365_FIXED,
        )

        vol = vol_surface.vol(self.expiry, self.strike)

        option_type = (
            OptionType.CALL
            if self.swaption_type == SwaptionType.PAYER
            else OptionType.PUT
        )

        # Black-76 with df=1 because the annuity already contains discounting
        unit_price = black76_price(fwd, self.strike, vol, time_to_expiry, df=1.0,
                                   option_type=option_type)

        return self.notional * ann * unit_price
