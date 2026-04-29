"""Interest rate caps and floors."""

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency, StubType, generate_schedule
from pricebook.calendar import Calendar, BusinessDayConvention
from pricebook.black76 import black76_price, black76_vega, OptionType
from pricebook.solvers import brentq


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
        self.calendar = calendar
        self.convention = convention

        schedule = generate_schedule(
            start, end, frequency, calendar, convention,
            StubType.SHORT_FRONT, True,
        )

        # Each caplet/floorlet covers one accrual period
        self.periods: list[tuple[date, date]] = []
        for i in range(1, len(schedule)):
            self.periods.append((schedule[i - 1], schedule[i]))

    def pv(
        self,
        curve: DiscountCurve,
        vol_surface,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """
        PV of the cap/floor.

        Args:
            curve: discount curve.
            vol_surface: object with vol(expiry, strike) method.
            projection_curve: forward projection curve. If None, single-curve pricing.
        """
        proj = projection_curve if projection_curve is not None else curve
        total = 0.0
        for accrual_start, accrual_end in self.periods:
            yf = year_fraction(accrual_start, accrual_end, self.day_count)
            # Forward rate for this period (from projection curve)
            df1 = proj.df(accrual_start)
            df2 = proj.df(accrual_end)
            fwd = (df1 - df2) / (yf * df2)

            # Time to fixing — always ACT/365F for Black-76
            t_fix = year_fraction(curve.reference_date, accrual_start, DayCountConvention.ACT_365_FIXED)
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

    def caplet_pvs(
        self,
        curve: DiscountCurve,
        vol_surface,
        projection_curve: DiscountCurve | None = None,
    ) -> list[dict]:
        """Individual caplet/floorlet PVs with forward rates and vols."""
        proj = projection_curve if projection_curve is not None else curve
        results = []
        for accrual_start, accrual_end in self.periods:
            yf = year_fraction(accrual_start, accrual_end, self.day_count)
            df1 = proj.df(accrual_start)
            df2 = proj.df(accrual_end)
            fwd = (df1 - df2) / (yf * df2)
            t_fix = year_fraction(curve.reference_date, accrual_start, DayCountConvention.ACT_365_FIXED)
            vol = vol_surface.vol(accrual_start, self.strike) if t_fix > 0 else 0.0
            if t_fix > 0:
                optlet = black76_price(fwd, self.strike, vol, t_fix, 1.0, self.option_type)
            else:
                if self.option_type == OptionType.CALL:
                    optlet = max(fwd - self.strike, 0.0)
                else:
                    optlet = max(self.strike - fwd, 0.0)
            pv = self.notional * yf * curve.df(accrual_end) * optlet
            results.append({
                "accrual_start": accrual_start,
                "accrual_end": accrual_end,
                "forward": fwd,
                "vol": vol,
                "pv": pv,
            })
        return results


def strip_caplet_vols(
    cap_flat_vols: list[tuple[date, float]],
    strike: float,
    curve: DiscountCurve,
    start: date,
    frequency: Frequency = Frequency.QUARTERLY,
    day_count: DayCountConvention = DayCountConvention.ACT_360,
) -> list[tuple[date, float]]:
    """Strip individual caplet vols from flat cap vols.

    Given flat (quoted) cap vols at increasing maturities, bootstrap
    the individual forward vol for each caplet period.

    The flat vol prices the entire cap. The caplet vol is the vol that
    prices just the marginal caplet added at each maturity.

    Returns list of (fixing_date, caplet_vol).
    """
    from pricebook.vol_surface import FlatVol

    sorted_quotes = sorted(cap_flat_vols, key=lambda x: x[0])
    caplet_vols: list[tuple[date, float]] = []
    prev_pv = 0.0
    prev_periods: list[tuple[date, date]] = []

    for mat, flat_vol in sorted_quotes:
        cap = CapFloor(start, mat, strike, OptionType.CALL, 1.0, frequency, day_count)
        total_pv = cap.pv(curve, FlatVol(flat_vol))
        marginal_pv = total_pv - prev_pv

        if cap.periods and len(cap.periods) > len(prev_periods):
            # The new caplet is the last period
            new_period = cap.periods[-1]
            accrual_start, accrual_end = new_period
            yf = year_fraction(accrual_start, accrual_end, day_count)
            df1 = curve.df(accrual_start)
            df2 = curve.df(accrual_end)
            fwd = (df1 - df2) / (yf * df2)
            t_fix = year_fraction(curve.reference_date, accrual_start, DayCountConvention.ACT_365_FIXED)

            if t_fix > 0 and marginal_pv > 0:
                target = marginal_pv / (yf * df2)

                def obj(v):
                    return black76_price(fwd, strike, v, t_fix, 1.0, OptionType.CALL) - target

                try:
                    caplet_vol = brentq(obj, 0.001, 2.0)
                except Exception:
                    caplet_vol = flat_vol
                caplet_vols.append((accrual_start, caplet_vol))

        prev_pv = total_pv
        prev_periods = list(cap.periods)

    return caplet_vols

from pricebook.serialisable import serialisable as _serialisable
_serialisable("capfloor", ["start", "end", "strike", "option_type", "notional", "frequency", "day_count"])(CapFloor)
