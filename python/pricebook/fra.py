"""Forward rate agreement."""

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


class FRA:
    """
    A forward rate agreement: a single-period contract on a forward rate.

    The buyer locks in a borrowing rate (the strike). At settlement (start date):
        settled = notional * (fwd - strike) * year_frac / (1 + fwd * year_frac)
        PV = settled * df(start)

    Supports dual-curve pricing:
        - projection_curve: used to compute the forward rate
        - discount_curve: used to discount the payment
    Single-curve is the special case where both are the same.
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

    def forward_rate(self, projection_curve: DiscountCurve) -> float:
        """Implied forward rate from the projection curve for this FRA period."""
        df1 = projection_curve.df(self.start)
        df2 = projection_curve.df(self.end)
        return (df1 / df2 - 1.0) / self.year_frac

    def pv(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """
        Present value of the FRA.

        Args:
            curve: discount curve.
            projection_curve: forward projection curve. If None, single-curve pricing.
        """
        proj = projection_curve if projection_curve is not None else curve
        fwd = self.forward_rate(proj)
        # FRA settles at start date: discounted amount at start, then PV to today
        # Settled amount = N × (fwd - K) × τ / (1 + fwd × τ)
        settled = self.notional * (fwd - self.strike) * self.year_frac / (1.0 + fwd * self.year_frac)
        return settled * curve.df(self.start)

    def par_rate(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """The strike rate that makes PV = 0."""
        proj = projection_curve if projection_curve is not None else curve
        return self.forward_rate(proj)
