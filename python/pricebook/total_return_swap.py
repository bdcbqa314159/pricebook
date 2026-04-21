"""Total Return Swap (TRS) on bonds.

Pay leg: total return of a reference bond (price change + coupons).
Receive leg: floating rate + spread.

Used for: leverage, synthetic bond exposure, balance sheet optimisation.

    from pricebook.total_return_swap import TotalReturnSwap

    trs = TotalReturnSwap(bond, notional, trs_spread)
    mtm = trs.mark_to_market(initial_dirty, current_curve, projection_curve)

References:
    Choudhry, *The Bond and Money Markets*, Butterworth-Heinemann, 2001, Ch. 35.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.bond import FixedRateBond
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


@dataclass
class TRSResult:
    """Total return swap mark-to-market result."""
    total_return_leg: float    # PV of total return (price + coupon)
    funding_leg: float         # PV of floating + spread payments
    mtm: float                 # net MTM (positive = total return receiver gains)
    price_return: float        # dirty price change component
    coupon_return: float       # coupon income component


class TotalReturnSwap:
    """Total return swap on a fixed-rate bond.

    Total return receiver gets: (dirty_end - dirty_start) + coupons.
    Total return payer gets: floating rate + TRS spread.

    Args:
        bond: reference bond.
        notional: face amount of the bond position.
        trs_spread: spread over floating on the funding leg (annualised).
        start: TRS effective date.
        end: TRS maturity (reset date).
    """

    def __init__(
        self,
        bond: FixedRateBond,
        notional: float,
        trs_spread: float,
        start: date,
        end: date,
    ):
        self.bond = bond
        self.notional = notional
        self.trs_spread = trs_spread
        self.start = start
        self.end = end

    def mark_to_market(
        self,
        initial_dirty: float,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> TRSResult:
        """Mark-to-market the TRS.

        Args:
            initial_dirty: dirty price at TRS inception (per 100 face).
            curve: current discount curve.
            projection_curve: for floating rate projection. If None, uses curve.
        """
        proj = projection_curve if projection_curve is not None else curve
        ref = curve.reference_date

        # Total return leg: price change + coupons
        current_dirty = self.bond.dirty_price(curve)
        price_return = (current_dirty - initial_dirty) / 100.0 * self.notional

        # Coupons received between start and now
        coupon_return = 0.0
        for cf in self.bond.coupon_leg.cashflows:
            if self.start < cf.payment_date <= ref:
                coupon_return += cf.amount / self.bond.face_value * self.notional

        total_return = price_return + coupon_return

        # Funding leg: floating + spread for accrued period
        yf_elapsed = year_fraction(self.start, ref, DayCountConvention.ACT_360)
        # Forward rate for the period
        if ref < self.end:
            fwd = proj.forward_rate(self.start, self.end)
        else:
            fwd = 0.0
        funding = self.notional * (fwd + self.trs_spread) * yf_elapsed

        return TRSResult(
            total_return_leg=total_return,
            funding_leg=funding,
            mtm=total_return - funding,
            price_return=price_return,
            coupon_return=coupon_return,
        )

    def breakeven_spread(
        self,
        initial_dirty: float,
        expected_dirty_at_end: float,
        expected_coupon_income: float,
        funding_rate: float,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
    ) -> float:
        """Breakeven TRS spread: spread at which total return = funding cost.

        Returns the TRS spread that makes the swap fair at inception.
        """
        yf = year_fraction(self.start, self.end, day_count)
        if yf <= 0:
            return 0.0

        total_return_pct = (expected_dirty_at_end - initial_dirty) / 100.0
        total_return_pct += expected_coupon_income / self.notional

        return total_return_pct / yf - funding_rate
