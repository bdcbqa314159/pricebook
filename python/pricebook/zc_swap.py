"""Zero-coupon swaps and IR digital options.

A zero-coupon swap exchanges a single compounded fixed amount for a
single compounded floating amount at maturity. Digital caps/floors
pay a fixed amount if the rate breaches the strike.

    from pricebook.zc_swap import ZeroCouponSwap, digital_capfloor
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from dateutil.relativedelta import relativedelta

from pricebook.black76 import black76_price, OptionType
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency, generate_schedule
from pricebook.swap import InterestRateSwap, SwapDirection


# ---- Zero-coupon swap ----

class ZeroCouponSwap:
    """Zero-coupon swap: single compounded payment at maturity.

    Fixed leg pays: notional × ((1 + K)^T - 1) at maturity.
    Floating leg pays: notional × (∏(1 + L_i × τ_i) - 1) at maturity,
    which equals notional × (1/df(T) - 1) under single-curve pricing.

    PV(payer) = notional × (1/df(T) - 1) × df(T) - notional × ((1+K)^T - 1) × df(T)
              = notional × (1 - df(T)) - notional × ((1+K)^T - 1) × df(T)

    Args:
        start: swap start date.
        end: swap maturity date.
        fixed_rate: annual compounded fixed rate.
        direction: PAYER (pay fixed, receive floating) or RECEIVER.
        notional: swap notional.
    """

    def __init__(
        self,
        start: date,
        end: date,
        fixed_rate: float,
        direction: SwapDirection = SwapDirection.PAYER,
        notional: float = 1_000_000.0,
    ):
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")
        self.start = start
        self.end = end
        self.fixed_rate = fixed_rate
        self.direction = direction
        self.notional = notional

    def _tenor_years(self, day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED) -> float:
        return year_fraction(self.start, self.end, day_count)

    def fixed_amount(self) -> float:
        """Compounded fixed amount paid at maturity."""
        T = self._tenor_years()
        return self.notional * ((1 + self.fixed_rate) ** T - 1)

    def floating_pv(self, curve: DiscountCurve) -> float:
        """PV of the floating leg: notional × (1 - df(T))."""
        df_T = curve.df(self.end)
        return self.notional * (1 - df_T)

    def fixed_pv(self, curve: DiscountCurve) -> float:
        """PV of the fixed leg: fixed_amount × df(T)."""
        df_T = curve.df(self.end)
        return self.fixed_amount() * df_T

    def pv(self, curve: DiscountCurve) -> float:
        """PV for the specified direction.

        Payer: receive floating - pay fixed.
        Receiver: receive fixed - pay floating.
        """
        float_pv = self.floating_pv(curve)
        fix_pv = self.fixed_pv(curve)
        if self.direction == SwapDirection.PAYER:
            return float_pv - fix_pv
        return fix_pv - float_pv

    def par_rate(self, curve: DiscountCurve) -> float:
        """Fixed rate that makes PV = 0.

        (1 + K)^T = 1/df(T)  →  K = df(T)^(-1/T) - 1
        """
        df_T = curve.df(self.end)
        T = self._tenor_years()
        if df_T <= 0 or T <= 0:
            return 0.0
        return df_T ** (-1.0 / T) - 1


# ---- IR Digital cap/floor ----

def digital_capfloor(
    start: date,
    end: date,
    strike: float,
    payout: float,
    curve: DiscountCurve,
    vol: float,
    option_type: OptionType = OptionType.CALL,
    notional: float = 1_000_000.0,
    frequency: Frequency = Frequency.QUARTERLY,
) -> float:
    """Price an IR digital cap (or floor) as a strip of digital caplets.

    Each digital caplet pays `payout` if rate > strike (call) or
    rate < strike (put). Priced via tight call-spread approximation.

    Args:
        payout: fixed amount paid per digital observation.
        vol: Black-76 vol for the forward rate.
    """
    ref = curve.reference_date
    schedule = generate_schedule(start, end, frequency)
    total = 0.0
    spread_width = 0.0001  # 1bp call spread

    for i in range(1, len(schedule)):
        d1 = schedule[i - 1]
        d2 = schedule[i]
        if d2 <= ref:
            continue

        yf = year_fraction(d1, d2, DayCountConvention.ACT_360)
        fwd = curve.forward_rate(d1, d2)
        t_fix = max(year_fraction(ref, d1, DayCountConvention.ACT_365_FIXED), 1e-6)
        df = curve.df(d2)

        # Digital via call spread: (C(K) - C(K + ε)) / ε
        if option_type == OptionType.CALL:
            c_low = black76_price(fwd, strike, vol, t_fix, df, OptionType.CALL)
            c_high = black76_price(fwd, strike + spread_width, vol, t_fix, df, OptionType.CALL)
            digital = (c_low - c_high) / spread_width
        else:
            p_low = black76_price(fwd, strike - spread_width, vol, t_fix, df, OptionType.PUT)
            p_high = black76_price(fwd, strike, vol, t_fix, df, OptionType.PUT)
            digital = (p_high - p_low) / spread_width

        total += notional * payout * digital

    return total


# ---- Digital CMS cap ----

def digital_cms_cap(
    start: date,
    end: date,
    strike: float,
    payout: float,
    cms_tenor: int,
    curve: DiscountCurve,
    vol: float,
    notional: float = 1_000_000.0,
    frequency: Frequency = Frequency.QUARTERLY,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Digital option on a CMS rate.

    Pays `payout` if CMS rate > strike (call) or < strike (put).
    CMS forward includes convexity adjustment.
    """
    from pricebook.cms import cms_convexity_adjustment

    ref = curve.reference_date
    schedule = generate_schedule(start, end, frequency)
    total = 0.0
    spread_width = 0.0001

    for i in range(1, len(schedule)):
        d1 = schedule[i - 1]
        d2 = schedule[i]
        if d2 <= ref:
            continue

        fixing = d1 if d1 > ref else ref
        swap_end = fixing + relativedelta(years=cms_tenor)

        fwd_swap = InterestRateSwap(
            fixing, swap_end, 0.05, SwapDirection.PAYER, 1_000_000.0,
        )
        fwd_rate = fwd_swap.par_rate(curve)
        ann = fwd_swap.fixed_leg.annuity(curve)

        t_fix = max(year_fraction(ref, d1, DayCountConvention.ACT_365_FIXED), 1e-6)
        adj = cms_convexity_adjustment(fwd_rate, ann, cms_tenor, vol, t_fix)
        cms_fwd = fwd_rate + adj

        df = curve.df(d2)

        if option_type == OptionType.CALL:
            c_low = black76_price(cms_fwd, strike, vol, t_fix, df, OptionType.CALL)
            c_high = black76_price(cms_fwd, strike + spread_width, vol, t_fix, df, OptionType.CALL)
            digital = (c_low - c_high) / spread_width
        else:
            p_low = black76_price(cms_fwd, strike - spread_width, vol, t_fix, df, OptionType.PUT)
            p_high = black76_price(cms_fwd, strike, vol, t_fix, df, OptionType.PUT)
            digital = (p_high - p_low) / spread_width

        total += notional * payout * digital

    return total
