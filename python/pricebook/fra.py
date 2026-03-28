"""Forward rate agreement."""

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


class FRA:
    """
    A forward rate agreement: a single-period contract on a forward rate.

    The buyer locks in a borrowing rate (the strike). At settlement:
        PV = notional * (forward_rate - strike) * year_frac * df(payment_date)

    Settlement is at the start of the forward period (standard FRA convention),
    but the accrual runs from start to end. The payment is discounted from
    the end date back to the valuation date.
    """

    def __init__(
        self,
        start: date,
        end: date,
        strike: float,
        notional: float = 1_000_000.0,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
    ):
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")

        self.start = start
        self.end = end
        self.strike = strike
        self.notional = notional
        self.day_count = day_count
        self.year_frac = year_fraction(start, end, day_count)

    def forward_rate(self, curve: DiscountCurve) -> float:
        """Implied forward rate from the curve for this FRA period."""
        df1 = curve.df(self.start)
        df2 = curve.df(self.end)
        return (df1 / df2 - 1.0) / self.year_frac

    def pv(self, curve: DiscountCurve) -> float:
        """Present value of the FRA."""
        fwd = self.forward_rate(curve)
        return self.notional * (fwd - self.strike) * self.year_frac * curve.df(self.end)

    def par_rate(self, curve: DiscountCurve) -> float:
        """The strike rate that makes PV = 0 (equals the forward rate)."""
        return self.forward_rate(curve)
