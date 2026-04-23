"""Interest rate futures: SOFR futures (1M/3M) with convexity adjustment."""

from __future__ import annotations

import math
from datetime import date, timedelta
from enum import Enum

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


class FuturesType(Enum):
    SOFR_1M = "sofr_1m"
    SOFR_3M = "sofr_3m"


class IRFuture:
    """SOFR interest rate future.

    Quoted as 100 - rate. Settlement based on compounded daily SOFR
    over the reference period.

    Args:
        accrual_start: first day of the reference period.
        accrual_end: last day of the reference period.
        futures_type: 1M or 3M SOFR.
        tick_value: dollar value of one basis point (default $41.67 for 3M).
        notional: contract notional (default $1,000,000).
        day_count: convention for accrual (ACT/360 for SOFR).
    """

    def __init__(
        self,
        accrual_start: date,
        accrual_end: date,
        futures_type: FuturesType = FuturesType.SOFR_3M,
        tick_value: float | None = None,
        notional: float = 1_000_000.0,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
    ):
        self.accrual_start = accrual_start
        self.accrual_end = accrual_end
        self.futures_type = futures_type
        self.notional = notional
        self.day_count = day_count

        tau = year_fraction(accrual_start, accrual_end, day_count)
        if tick_value is not None:
            self.tick_value = tick_value
        else:
            # Standard: notional * tau / 100 per 1bp
            self.tick_value = notional * tau / 10_000

    @property
    def accrual_fraction(self) -> float:
        return year_fraction(self.accrual_start, self.accrual_end, self.day_count)

    def implied_forward(self, curve: DiscountCurve) -> float:
        """Simply compounded forward rate for the reference period.

        Uses the futures day count (ACT/360), not the curve's internal day count.
        """
        df1 = curve.df(self.accrual_start)
        df2 = curve.df(self.accrual_end)
        tau = self.accrual_fraction
        return (df1 - df2) / (tau * df2)

    def futures_rate(
        self,
        curve: DiscountCurve,
        convexity: float = 0.0,
    ) -> float:
        """Futures rate = forward rate + convexity adjustment.

        The convexity adjustment accounts for daily margining of futures
        vs single settlement of FRAs.
        """
        fwd = self.implied_forward(curve)
        return fwd + convexity

    def price(
        self,
        curve: DiscountCurve,
        convexity: float = 0.0,
    ) -> float:
        """Futures price = 100 - futures_rate * 100."""
        return 100.0 - self.futures_rate(curve, convexity) * 100.0

    def pv(
        self,
        curve: DiscountCurve,
        trade_price: float,
        convexity: float = 0.0,
    ) -> float:
        """Mark-to-market P&L vs trade price.

        PV = (current_price - trade_price) * tick_value * 100
        """
        current = self.price(curve, convexity)
        return (current - trade_price) * self.tick_value * 100

    def dv01(self, curve: DiscountCurve, convexity: float = 0.0) -> float:
        """Dollar value of 1bp rate move."""
        return self.tick_value


# ---------------------------------------------------------------------------
# Convexity adjustment
# ---------------------------------------------------------------------------


def hw_convexity_adjustment(
    a: float,
    sigma: float,
    t: float,
    T1: float,
    T2: float,
) -> float:
    """Hull-White analytical convexity adjustment for rate futures.

    The futures rate exceeds the forward rate by:
        CA = 0.5 * sigma^2 * B(T1, T2) * [B(T1, T2) * G(t, T1) + (T2 - T1)]
    where:
        B(s, t) = (1 - exp(-a*(t-s))) / a
        G(t, T) = (1 - exp(-2*a*(T-t))) / (2*a)

    Simplified form for constant a, sigma:
        CA ≈ 0.5 * sigma^2 * B(t, T1) * B(T1, T2) * (T1 - t) / a
           (approximate for small a*(T-t))

    Args:
        a: mean reversion speed.
        sigma: short-rate volatility.
        t: current time (year fraction).
        T1: accrual start (year fraction).
        T2: accrual end (year fraction).

    Returns:
        Convexity adjustment (positive: futures rate > forward rate).
    """
    if a < 1e-10:
        # No mean reversion: simple formula
        return 0.5 * sigma**2 * (T2 - T1) * (T1 - t)

    B_T1_T2 = (1.0 - math.exp(-a * (T2 - T1))) / a
    B_t_T1 = (1.0 - math.exp(-a * (T1 - t))) / a
    G_t_T1 = (1.0 - math.exp(-2.0 * a * (T1 - t))) / (2.0 * a)

    return 0.5 * sigma**2 * B_T1_T2 * (B_T1_T2 * G_t_T1 + B_t_T1 * (T2 - T1))


def futures_strip_rates(
    futures: list[IRFuture],
    curve: DiscountCurve,
    a: float = 0.0,
    sigma: float = 0.0,
) -> list[dict]:
    """Compute rates for a strip of futures with HW convexity.

    Returns list of dicts with tenor, forward, convexity, futures_rate, price.
    """
    ref = curve.reference_date
    results = []
    for fut in futures:
        t = year_fraction(ref, fut.accrual_start, fut.day_count)
        T1 = t
        T2 = year_fraction(ref, fut.accrual_end, fut.day_count)

        ca = hw_convexity_adjustment(a, sigma, 0.0, T1, T2) if sigma > 0 else 0.0
        fwd = fut.implied_forward(curve)
        fut_rate = fwd + ca

        results.append({
            "start": fut.accrual_start,
            "end": fut.accrual_end,
            "forward": fwd,
            "convexity": ca,
            "futures_rate": fut_rate,
            "price": 100.0 - fut_rate * 100.0,
        })

    return results
