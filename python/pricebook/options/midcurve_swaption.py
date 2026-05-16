"""Mid-curve swaption: option expiry before the underlying swap starts.

A mid-curve swaption (e.g., 1Y into 5Y starting in 2Y) gives the right
to enter a forward-starting swap. The option expires at T_option, but
the underlying swap starts at T_swap_start > T_option.

    from pricebook.options.midcurve_swaption import midcurve_swaption, MidCurveResult

    result = midcurve_swaption(
        spot_date, option_expiry, swap_start, swap_end,
        strike=0.03, vol=0.50, curve=ois,
    )

References:
    Brigo & Mercurio (2006). Interest Rate Models, Ch. 6.
    Rebonato (2002). Modern Pricing of Interest-Rate Derivatives, Ch. 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.models.black76 import black76_price, black76_delta, black76_vega, OptionType
from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.schedule import Frequency, generate_schedule
from pricebook.fixed_income.swap import InterestRateSwap, SwapDirection


@dataclass
class MidCurveResult:
    """Mid-curve swaption pricing result."""
    price: float
    forward_swap_rate: float
    annuity: float
    time_to_expiry: float
    gap: float                 # time between option expiry and swap start
    delta: float
    vega: float

    def to_dict(self) -> dict:
        return {
            "price": self.price, "forward_swap_rate": self.forward_swap_rate,
            "annuity": self.annuity, "T_expiry": self.time_to_expiry,
            "gap": self.gap, "delta": self.delta, "vega": self.vega,
        }


def midcurve_swaption(
    reference_date: date,
    option_expiry: date,
    swap_start: date,
    swap_end: date,
    strike: float,
    vol: float,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
    is_payer: bool = True,
    notional: float = 1_000_000,
    fixed_frequency: Frequency = Frequency.SEMI_ANNUAL,
    fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360,
) -> MidCurveResult:
    """Price a mid-curve swaption via Black-76 on the forward swap rate.

    Key difference from standard swaption: option expires at T_option,
    but the underlying swap starts at T_swap_start > T_option.

    The forward swap rate is the par rate of the swap starting at
    T_swap_start and ending at T_swap_end, as seen from T_option.

    The convexity adjustment from the gap (T_swap_start - T_option)
    is typically small and is ignored here (first-order pricing).

    Args:
        option_expiry: date the option expires.
        swap_start: date the underlying swap starts (>= option_expiry).
        swap_end: date the underlying swap ends.
        strike: fixed rate of the underlying swap.
        vol: swaption vol (lognormal, Black-76).
    """
    if option_expiry < reference_date:
        raise ValueError(f"option_expiry ({option_expiry}) must be >= reference_date")
    if swap_start < option_expiry:
        raise ValueError(f"swap_start ({swap_start}) must be >= option_expiry")
    if swap_end <= swap_start:
        raise ValueError(f"swap_end ({swap_end}) must be after swap_start")

    proj = projection_curve or curve
    T_expiry = year_fraction(reference_date, option_expiry, DayCountConvention.ACT_365_FIXED)
    gap = year_fraction(option_expiry, swap_start, DayCountConvention.ACT_365_FIXED)

    # Build forward-starting swap to get par rate and annuity
    fwd_swap = InterestRateSwap(
        start=swap_start, end=swap_end, fixed_rate=strike,
        direction=SwapDirection.PAYER, notional=notional,
        fixed_frequency=fixed_frequency, fixed_day_count=fixed_day_count,
    )
    forward_rate = fwd_swap.par_rate(curve, proj)

    # Annuity: PV01 of fixed leg
    schedule = generate_schedule(swap_start, swap_end, fixed_frequency)
    annuity = 0.0
    for i in range(1, len(schedule)):
        yf = year_fraction(schedule[i - 1], schedule[i], fixed_day_count)
        annuity += yf * curve.df(schedule[i])

    # Black-76 pricing
    opt = OptionType.CALL if is_payer else OptionType.PUT
    price = black76_price(forward_rate, strike, vol, T_expiry, annuity, opt) * notional

    # Greeks
    delta = black76_delta(forward_rate, strike, vol, T_expiry, annuity, opt) * notional
    vega = black76_vega(forward_rate, strike, vol, T_expiry, annuity) * notional

    return MidCurveResult(
        price=float(price),
        forward_swap_rate=float(forward_rate),
        annuity=float(annuity),
        time_to_expiry=float(T_expiry),
        gap=float(gap),
        delta=float(delta),
        vega=float(vega),
    )
