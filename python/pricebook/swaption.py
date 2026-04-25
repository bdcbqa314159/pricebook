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

from pricebook.black76 import OptionType, black76_price, black76_delta, black76_gamma, black76_vega
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.greeks import Greeks
from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_context import PricingContext
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
        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency
        self.fixed_day_count = fixed_day_count
        self.float_day_count = float_day_count

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

        if self.expiry <= valuation_date:
            # Expired: return intrinsic value
            if self.swaption_type == SwaptionType.PAYER:
                return self.notional * ann * max(fwd - self.strike, 0.0)
            else:
                return self.notional * ann * max(self.strike - fwd, 0.0)

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

    def pv_ctx(
        self,
        ctx: PricingContext,
        vol_surface_name: str = "ir",
        projection_curve_name: str | None = None,
    ) -> float:
        """
        Price the swaption from a PricingContext.

        Args:
            ctx: pricing context with curves and vol surfaces.
            vol_surface_name: key for the vol surface in the context.
            projection_curve_name: key for the projection curve.
                If None, single-curve pricing (discount curve used for both).
        """
        if ctx.discount_curve is None:
            raise ValueError("PricingContext must have a discount_curve")

        vol_surface = ctx.get_vol_surface(vol_surface_name)
        projection_curve = (
            ctx.get_projection_curve(projection_curve_name)
            if projection_curve_name is not None
            else None
        )

        return self.pv(
            ctx.discount_curve,
            vol_surface,
            projection_curve,
            valuation_date=ctx.valuation_date,
        )

    def greeks(
        self,
        curve: DiscountCurve,
        vol_surface,
        projection_curve: DiscountCurve | None = None,
        valuation_date: date | None = None,
    ) -> Greeks:
        """Swaption Greeks via Black-76 analytical formulas."""
        if valuation_date is None:
            valuation_date = curve.reference_date

        fwd = self.forward_swap_rate(curve, projection_curve)
        ann = self.annuity(curve)
        T = year_fraction(valuation_date, self.expiry, DayCountConvention.ACT_365_FIXED)
        vol = vol_surface.vol(self.expiry, self.strike)

        option_type = (
            OptionType.CALL if self.swaption_type == SwaptionType.PAYER
            else OptionType.PUT
        )

        price = self.notional * ann * black76_price(fwd, self.strike, vol, T, 1.0, option_type)
        delta = self.notional * ann * black76_delta(fwd, self.strike, vol, T, 1.0, option_type)
        gamma = self.notional * ann * black76_gamma(fwd, self.strike, vol, T, 1.0)
        vega = self.notional * ann * black76_vega(fwd, self.strike, vol, T, 1.0)

        return Greeks(
            price=price,
            delta=delta,
            gamma=gamma,
            vega=vega,
        )
