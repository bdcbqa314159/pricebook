"""Interest rate futures: SOFR/Euribor futures, convexity, packs, bundles, butterflies.

* :class:`IRFuture` — single SOFR/Euribor future with HW convexity.
* :func:`hw_convexity_adjustment` — analytical convexity for rate futures.
* :func:`futures_strip_rates` — compute rates for a strip with convexity.
* :func:`futures_pack` — 4-quarter pack (average price of 4 consecutive futures).
* :func:`futures_bundle` — multi-year bundle (average price of N consecutive futures).
* :func:`futures_butterfly` — weighted butterfly on 3 consecutive futures.
* :func:`fed_funds_implied_probability` — implied probability of a rate move from FF futures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve


class FuturesType(Enum):
    SOFR_1M = "sofr_1m"
    SOFR_3M = "sofr_3m"
    EURIBOR_3M = "euribor_3m"


class IRFuture:
    """Interest rate future (SOFR, Euribor).

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



    def pv_ctx(self, ctx) -> float:
        """PV using PricingContext."""
        curve = ctx.discount_curve
        if curve is None:
            raise ValueError("No discount curve in context")
        return self.pv(curve)

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


# ---------------------------------------------------------------------------
# Pack / Bundle / Butterfly
# ---------------------------------------------------------------------------


@dataclass
class PackResult:
    """Pack pricing result."""
    price: float              # average price of the 4 futures
    dv01: float               # total DV01 of the pack
    implied_rate: float       # average implied rate
    n_contracts: int

    def to_dict(self) -> dict:
        return vars(self)


def futures_pack(
    futures: list[IRFuture],
    curve: DiscountCurve,
    a: float = 0.0,
    sigma: float = 0.0,
) -> PackResult:
    """Price a pack of 4 consecutive quarterly futures.

    A pack is the average price of 4 consecutive quarterly futures,
    traded as a single unit. Standard packs: white (Y1), red (Y2),
    green (Y3), blue (Y4), gold (Y5).

    Args:
        futures: exactly 4 consecutive quarterly futures.
    """
    if len(futures) != 4:
        raise ValueError(f"pack requires exactly 4 futures, got {len(futures)}")

    strips = futures_strip_rates(futures, curve, a, sigma)
    avg_price = sum(s["price"] for s in strips) / 4
    avg_rate = sum(s["futures_rate"] for s in strips) / 4
    total_dv01 = sum(f.tick_value for f in futures)

    return PackResult(
        price=avg_price,
        dv01=total_dv01,
        implied_rate=avg_rate,
        n_contracts=4,
    )


@dataclass
class BundleResult:
    """Bundle pricing result."""
    price: float
    dv01: float
    implied_rate: float
    n_contracts: int
    years: int

    def to_dict(self) -> dict:
        return vars(self)


def futures_bundle(
    futures: list[IRFuture],
    curve: DiscountCurve,
    a: float = 0.0,
    sigma: float = 0.0,
) -> BundleResult:
    """Price a bundle of consecutive quarterly futures.

    A bundle is the average price of N consecutive quarterly futures
    spanning multiple years. Standard bundles: 2Y (8 contracts),
    3Y (12), 5Y (20).

    Args:
        futures: list of consecutive quarterly futures (4, 8, 12, 16, or 20).
    """
    n = len(futures)
    if n < 4 or n % 4 != 0:
        raise ValueError(f"bundle requires a multiple of 4 futures, got {n}")

    strips = futures_strip_rates(futures, curve, a, sigma)
    avg_price = sum(s["price"] for s in strips) / n
    avg_rate = sum(s["futures_rate"] for s in strips) / n
    total_dv01 = sum(f.tick_value for f in futures)

    return BundleResult(
        price=avg_price,
        dv01=total_dv01,
        implied_rate=avg_rate,
        n_contracts=n,
        years=n // 4,
    )


@dataclass
class FuturesButterfly:
    """Butterfly on 3 consecutive futures."""
    spread: float       # 2 × mid - front - back
    front_price: float
    mid_price: float
    back_price: float

    def to_dict(self) -> dict:
        return vars(self)


def futures_butterfly(
    front: IRFuture,
    mid: IRFuture,
    back: IRFuture,
    curve: DiscountCurve,
    a: float = 0.0,
    sigma: float = 0.0,
) -> FuturesButterfly:
    """Price a butterfly on 3 consecutive quarterly futures.

    Butterfly = 2 × mid - front - back.
    Positive when the curve is convex (mid trades rich).

    Args:
        front, mid, back: three consecutive quarterly futures.
    """
    strips = futures_strip_rates([front, mid, back], curve, a, sigma)
    fp, mp, bp = strips[0]["price"], strips[1]["price"], strips[2]["price"]

    return FuturesButterfly(
        spread=2 * mp - fp - bp,
        front_price=fp,
        mid_price=mp,
        back_price=bp,
    )


# ---------------------------------------------------------------------------
# Fed Funds implied probability
# ---------------------------------------------------------------------------

def fed_funds_implied_probability(
    futures_rate: float,
    current_rate: float,
    move_size: float = 0.0025,
) -> float:
    """Implied probability of a rate move from Fed Funds futures.

    P(hike) = (futures_rate - current_rate) / move_size

    Args:
        futures_rate: implied rate from the Fed Funds future (1 - price/100).
        current_rate: current effective Fed Funds rate.
        move_size: expected move size (default 25bp = 0.0025).

    Returns:
        Probability of a move (0 to 1). Values outside [0, 1] indicate
        the market prices more than one move or a different-sized move.
    """
    if abs(move_size) < 1e-10:
        raise ValueError("move_size cannot be zero")
    return (futures_rate - current_rate) / move_size
