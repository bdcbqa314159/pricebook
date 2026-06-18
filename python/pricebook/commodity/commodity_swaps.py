"""Commodity swaps and swaptions.

Fixed-for-floating commodity swaps with Asian-style settlement,
and swaptions priced via Black-76 on the swap rate.

* :class:`CommoditySwapResult` — swap pricing result.
* :func:`commodity_swap_price` — fixed-for-floating commodity swap.
* :func:`commodity_swaption_price` — option to enter commodity swap.
* :func:`asian_commodity_swap` — swap with averaging settlement.

References:
    Geman, *Commodities and Commodity Derivatives*, Ch. 4, 2005.
    Hull, *Options, Futures, and Other Derivatives*, 11th ed., Ch. 35.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.models.black76 import (
    OptionType, black76_price, black76_delta, black76_vega,
)


@dataclass
class CommoditySwapResult:
    """Commodity swap pricing result."""
    pv: float               # PV to fixed-rate payer
    fair_fixed: float        # fair fixed price
    notional_per_period: float
    n_periods: int
    total_notional: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def commodity_swap_price(
    forward_prices: list[float],
    payment_times: list[float],
    fixed_price: float,
    rate: float = 0.04,
    quantity_per_period: float = 1000.0,
) -> CommoditySwapResult:
    """Price a fixed-for-floating commodity swap.

    Floating leg pays the spot/average price each period.
    Fixed leg pays the agreed fixed price.

    PV = Σ (F_i − K) × Q × df_i

    where F_i = forward price for period i, K = fixed price.

    Args:
        forward_prices: forward prices for each settlement period.
        payment_times: settlement times (years) per period.
        fixed_price: agreed fixed price.
        rate: risk-free rate.
        quantity_per_period: physical quantity per period.
    """
    pv = 0.0
    for F, t in zip(forward_prices, payment_times):
        df = math.exp(-rate * t)
        pv += (F - fixed_price) * quantity_per_period * df

    # Fair fixed: PV-weighted average of forwards
    annuity = sum(math.exp(-rate * t) for t in payment_times)
    fair_fixed = sum(
        F * math.exp(-rate * t)
        for F, t in zip(forward_prices, payment_times)
    ) / annuity if annuity > 0 else 0

    return CommoditySwapResult(
        pv=pv,
        fair_fixed=fair_fixed,
        notional_per_period=quantity_per_period,
        n_periods=len(forward_prices),
        total_notional=quantity_per_period * len(forward_prices),
    )


@dataclass
class CommoditySwaptionResult:
    """Commodity swaption pricing result."""
    premium: float
    delta: float
    vega: float             # per 1% vol
    forward_swap_rate: float
    strike: float
    vol: float
    option_type: str

    def to_dict(self) -> dict:
        return dict(vars(self))


def commodity_swaption_price(
    forward_prices: list[float],
    payment_times: list[float],
    strike_price: float,
    vol: float,
    expiry_years: float,
    rate: float = 0.04,
    quantity_per_period: float = 1000.0,
    option_type: str = "call",
) -> CommoditySwaptionResult:
    """Price a commodity swaption via Black-76.

    Option to enter a commodity swap at the strike fixed price.
    Call = right to receive floating (benefit from high prices).
    Put = right to pay floating (benefit from low prices).

    Uses the forward swap rate as the underlying.

    Args:
        forward_prices: forward prices for swap settlement periods.
        payment_times: settlement times (years).
        strike_price: swap fixed price strike.
        vol: lognormal vol of the forward swap rate.
        expiry_years: swaption expiry.
        option_type: "call" (payer) or "put" (receiver).
    """
    df_expiry = math.exp(-rate * expiry_years)
    otype = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT

    # Forward swap rate
    annuity = sum(math.exp(-rate * t) for t in payment_times)
    fwd_rate = sum(
        F * math.exp(-rate * t)
        for F, t in zip(forward_prices, payment_times)
    ) / annuity if annuity > 0 else 0

    # Total notional for the swap
    total_q = quantity_per_period * len(forward_prices)

    # Price via Black-76
    premium = black76_price(fwd_rate, strike_price, vol, expiry_years, df_expiry, otype)
    premium *= annuity * quantity_per_period

    delta = black76_delta(fwd_rate, strike_price, vol, expiry_years, df_expiry, otype)
    vega = black76_vega(fwd_rate, strike_price, vol, expiry_years, df_expiry) * 0.01
    vega *= annuity * quantity_per_period

    return CommoditySwaptionResult(
        premium=premium,
        delta=delta,
        vega=vega,
        forward_swap_rate=fwd_rate,
        strike=strike_price,
        vol=vol,
        option_type=option_type,
    )


def asian_commodity_swap(
    daily_forwards: list[float],
    averaging_days: int,
    fixed_price: float,
    rate: float = 0.04,
    T: float = 1.0,
    quantity: float = 1000.0,
) -> CommoditySwapResult:
    """Commodity swap with arithmetic averaging settlement.

    The floating payment is based on the arithmetic average
    of daily prices over the averaging period.

    PV = (avg(F_i) − K) × Q × df

    Args:
        daily_forwards: forward prices for each averaging day.
        averaging_days: number of days in averaging period.
        fixed_price: agreed fixed price.
        rate: risk-free rate.
        T: time to settlement (years).
        quantity: total quantity for the period.
    """
    # Use first averaging_days forwards
    avg_prices = daily_forwards[:averaging_days]
    if not avg_prices:
        return CommoditySwapResult(0, fixed_price, quantity, 0, 0)

    avg_forward = sum(avg_prices) / len(avg_prices)
    df = math.exp(-rate * T)
    pv = (avg_forward - fixed_price) * quantity * df

    return CommoditySwapResult(
        pv=pv,
        fair_fixed=avg_forward,
        notional_per_period=quantity,
        n_periods=1,
        total_notional=quantity,
    )
