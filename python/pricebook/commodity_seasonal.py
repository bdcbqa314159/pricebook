"""Commodity seasonality and storage cost models.

Seasonal forward curves for natural gas, power, etc. Monthly seasonal
factors overlay the base forward curve. Storage cost model computes
convenience yield term structure.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.black76 import OptionType, black76_price


class SeasonalFactors:
    """Monthly seasonal adjustment factors.

    Factor > 1.0 means premium (e.g. winter gas), < 1.0 means discount.
    Factors should average to ~1.0 over a year.
    """

    def __init__(self, monthly_factors: list[float]):
        if len(monthly_factors) != 12:
            raise ValueError("Need exactly 12 monthly factors")
        self.factors = monthly_factors

    def factor(self, d: date) -> float:
        return self.factors[d.month - 1]

    @classmethod
    def natural_gas(cls) -> "SeasonalFactors":
        """Typical US natural gas seasonality: winter premium, summer discount."""
        return cls([1.15, 1.10, 1.02, 0.92, 0.88, 0.90,
                    0.92, 0.93, 0.95, 1.00, 1.08, 1.15])

    @classmethod
    def power(cls) -> "SeasonalFactors":
        """Typical power seasonality: summer + winter peaks."""
        return cls([1.10, 1.05, 0.95, 0.90, 0.95, 1.10,
                    1.15, 1.10, 0.95, 0.90, 0.95, 1.05])

    @classmethod
    def flat(cls) -> "SeasonalFactors":
        return cls([1.0] * 12)


class SeasonalForwardCurve:
    """Commodity forward curve with seasonal adjustment.

    Args:
        base_price: annual average or flat base forward price.
        seasonal: monthly seasonal factors.
        reference_date: valuation date.
    """

    def __init__(
        self,
        base_price: float,
        seasonal: SeasonalFactors,
        reference_date: date,
    ):
        self.base_price = base_price
        self.seasonal = seasonal
        self.reference_date = reference_date

    def forward(self, delivery_date: date) -> float:
        """Seasonally adjusted forward price."""
        return self.base_price * self.seasonal.factor(delivery_date)

    def forwards(self, dates: list[date]) -> list[float]:
        return [self.forward(d) for d in dates]


class StorageCostModel:
    """Convenience yield / storage cost model.

    The cost of carry for commodities:
        F(T) = S * exp((r - y + c) * T)
    where r = risk-free rate, y = convenience yield, c = storage cost.

    Args:
        storage_cost_rate: annual storage cost as fraction of spot.
        convenience_yield: annual convenience yield.
    """

    def __init__(
        self,
        storage_cost_rate: float = 0.02,
        convenience_yield: float = 0.0,
    ):
        self.storage_cost = storage_cost_rate
        self.convenience_yield = convenience_yield

    @property
    def net_cost_of_carry(self) -> float:
        """Net cost of carry = storage - convenience yield."""
        return self.storage_cost - self.convenience_yield

    def implied_forward(self, spot: float, rate: float, T: float) -> float:
        """Forward price from cost-of-carry model."""
        return spot * math.exp((rate + self.net_cost_of_carry) * T)

    def implied_convenience_yield(
        self, spot: float, forward: float, rate: float, T: float,
    ) -> float:
        """Implied convenience yield from market forward."""
        if T <= 0 or spot <= 0 or forward <= 0:
            return 0.0
        return rate + self.storage_cost - math.log(forward / spot) / T


def calendar_spread_option(
    forward_near: float,
    forward_far: float,
    vol_near: float,
    vol_far: float,
    correlation: float,
    T: float,
    df: float,
    strike: float = 0.0,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """Calendar spread option: payoff = max(F_near - F_far - K, 0).

    Approximation via Margrabe/Kirk's formula for spread options.
    """
    spread_vol = math.sqrt(
        vol_near**2 + vol_far**2 - 2 * correlation * vol_near * vol_far
    )
    spread_forward = forward_near - forward_far
    if spread_forward + strike <= 0:
        return 0.0

    return black76_price(
        max(spread_forward, 1e-10), max(strike, 1e-10),
        spread_vol, T, df, option_type,
    )
