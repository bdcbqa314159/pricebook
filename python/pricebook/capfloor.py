"""Interest rate caps and floors."""

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency, StubType, generate_schedule
from pricebook.calendar import Calendar, BusinessDayConvention
from pricebook.black76 import black76_price, OptionType


class CapFloor:
    """
    An interest rate cap or floor: a strip of caplets/floorlets.

    A caplet is a call option on a forward rate (pays max(L - K, 0) * yf * notional).
    A floorlet is a put option on a forward rate (pays max(K - L, 0) * yf * notional).

    Each caplet/floorlet is priced with Black-76:
        caplet = notional * yf * df(T_pay) * Black76_call(F, K, vol, T_fix)
        floorlet = notional * yf * df(T_pay) * Black76_put(F, K, vol, T_fix)

    The vol for each caplet comes from a vol surface at the fixing expiry.
    """

    def __init__(
        self,
        start: date,
        end: date,
        strike: float,
        option_type: OptionType = OptionType.CALL,
        notional: float = 1_000_000.0,
        frequency: Frequency = Frequency.QUARTERLY,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    ):
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")

        self.start = start
        self.end = end
        self.strike = strike
        self.option_type = option_type
        self.notional = notional
        self.frequency = frequency
        self.day_count = day_count

        schedule = generate_schedule(
            start, end, frequency, calendar, convention,
            StubType.SHORT_FRONT, True,
        )

        # Each caplet/floorlet covers one accrual period
        self.periods: list[tuple[date, date]] = []
        for i in range(1, len(schedule)):
            self.periods.append((schedule[i - 1], schedule[i]))

    def pv(self, curve: DiscountCurve, vol_surface) -> float:
        """
        PV of the cap/floor.

        Args:
            curve: discount curve (also used for forward rate projection).
            vol_surface: object with vol(expiry, strike) method.
        """
        total = 0.0
        for accrual_start, accrual_end in self.periods:
            yf = year_fraction(accrual_start, accrual_end, self.day_count)
            # Forward rate for this period
            df1 = curve.df(accrual_start)
            df2 = curve.df(accrual_end)
            fwd = (df1 / df2 - 1.0) / yf

            # Time to fixing (= accrual start)
            t_fix = year_fraction(curve.reference_date, accrual_start, self.day_count)
            if t_fix <= 0:
                # Already fixed — use intrinsic value
                if self.option_type == OptionType.CALL:
                    payoff = max(fwd - self.strike, 0.0)
                else:
                    payoff = max(self.strike - fwd, 0.0)
                total += self.notional * yf * curve.df(accrual_end) * payoff
                continue

            vol = vol_surface.vol(accrual_start, self.strike)
            # Black-76 price per unit notional
            optlet = black76_price(fwd, self.strike, vol, t_fix, 1.0, self.option_type)
            total += self.notional * yf * curve.df(accrual_end) * optlet

        return total
