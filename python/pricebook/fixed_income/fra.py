"""Forward rate agreement."""

from datetime import date

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve


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
        return (df1 - df2) / (self.year_frac * df2)

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

    def pv_ctx(self, ctx) -> float:
        """PV using PricingContext.

        Looks up ``ctx.projection_curves`` keyed by this FRA's day-count
        name (e.g. ``"ACT/360"``).  Falls back to the discount curve if
        the context has no projection curves at all.

        Fix T4-FRA1: pre-fix this routine silently fell back to
        ``next(iter(ctx.projection_curves.values()))`` when the keyed
        lookup missed.  In a multi-curve setup (e.g. ACT/360 USD-LIBOR
        vs ACT/365 GBP), an ACT/360 FRA could silently get priced
        against an ACT/365 projection curve — wrong-curve forward rates
        with no diagnostic.  Now raises ``KeyError`` with the offending
        day-count and the available keys.
        """
        curve = ctx.discount_curve
        if curve is None:
            raise ValueError("No discount curve in context")
        proj = None
        if hasattr(ctx, 'projection_curves') and ctx.projection_curves:
            # Try keyed lookup by day-count convention name.
            dc_key = self.day_count.value if hasattr(self, 'day_count') else None
            if dc_key and dc_key in ctx.projection_curves:
                proj = ctx.projection_curves[dc_key]
            else:
                raise KeyError(
                    f"FRA.pv_ctx: no projection curve for day_count "
                    f"{dc_key!r}; ctx has keys "
                    f"{list(ctx.projection_curves.keys())!r}.  Pre-fix this "
                    "silently picked the first available curve, producing a "
                    "wrong-curve forward rate."
                )
        return self.pv(curve, proj)

@classmethod
def _fra_from_convention(cls, conv, start, end, strike, notional=1_000_000.0):
    """Create FRA from a convention (uses float day_count)."""
    dc = getattr(conv, 'float_day_count', getattr(conv, 'day_count', None))
    if dc is None:
        from pricebook.core.day_count import DayCountConvention
        dc = DayCountConvention.ACT_360
    return cls(start, end, strike, notional, dc)

FRA.from_convention = _fra_from_convention

from pricebook.core.serialisable import serialisable as _serialisable
_serialisable("fra", ["start", "end", "strike", "notional"])(FRA)
