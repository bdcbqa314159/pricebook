"""Equity Total Return Swap.

Pay leg: total return of a stock or index (price change + dividends).
Receive leg: floating rate + TRS spread.

Used for: leveraged equity exposure, balance sheet management, tax efficiency.

    from pricebook.equity_trs import EquityTRS

    trs = EquityTRS(ticker, notional, trs_spread, start, end)
    result = trs.mark_to_market(initial_price, current_price, divs_received, curve)

References:
    Choudhry, *The Bond and Money Markets*, Butterworth-Heinemann, 2001, Ch. 36.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


@dataclass
class EquityTRSResult:
    """Equity TRS mark-to-market result."""
    total_return_leg: float    # price return + dividends
    funding_leg: float         # floating + spread
    mtm: float                 # net (positive = TR receiver gains)
    price_return: float
    dividend_return: float


class EquityTRS:
    """Total return swap on equity (stock or index).

    Total return receiver gets: (S_end - S_start)/S_start × notional + dividends.
    Total return payer gets: (floating rate + spread) × notional × accrual.

    Args:
        ticker: underlying identifier.
        notional: swap notional (face amount of equity exposure).
        trs_spread: spread over floating on the funding leg (annualised).
        start: TRS effective date.
        end: TRS maturity / reset date.
    """

    def __init__(
        self,
        ticker: str,
        notional: float,
        trs_spread: float,
        start: date,
        end: date,
    ):
        self.ticker = ticker
        self.notional = notional
        self.trs_spread = trs_spread
        self.start = start
        self.end = end

    def mark_to_market(
        self,
        initial_price: float,
        current_price: float,
        dividends_received: float,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> EquityTRSResult:
        """Mark-to-market the equity TRS.

        Args:
            initial_price: stock price at TRS inception.
            current_price: current stock price.
            dividends_received: total dividends per share received so far.
            curve: discount curve.
            projection_curve: for floating rate. If None, uses curve.
        """
        proj = projection_curve if projection_curve is not None else curve
        ref = curve.reference_date

        # Total return leg
        price_return = (current_price - initial_price) / initial_price * self.notional
        div_return = dividends_received / initial_price * self.notional
        total_return = price_return + div_return

        # Funding leg
        if ref <= self.start:
            yf = 0.0
        else:
            yf = year_fraction(self.start, ref, DayCountConvention.ACT_360)
        if ref < self.end:
            fwd = proj.forward_rate(self.start, self.end)
        else:
            fwd = 0.0
        funding = self.notional * (fwd + self.trs_spread) * yf

        return EquityTRSResult(
            total_return_leg=total_return,
            funding_leg=funding,
            mtm=total_return - funding,
            price_return=price_return,
            dividend_return=div_return,
        )

    def breakeven_return(
        self,
        funding_rate: float,
    ) -> float:
        """Annualised equity return that equals the funding cost.

        breakeven = funding_rate + trs_spread
        """
        return funding_rate + self.trs_spread
