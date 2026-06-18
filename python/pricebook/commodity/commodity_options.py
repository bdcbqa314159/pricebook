"""Commodity futures options: crude, gold, grain, natural gas.

Options on commodity futures with seasonal vol term structure,
option-vs-futures expiry mismatch, and American exercise.

* :class:`CommodityOptionResult` — pricing result with Greeks.
* :func:`commodity_option_price` — price commodity futures option.
* :func:`seasonal_vol` — seasonal volatility adjustment.
* :func:`commodity_option_strip` — price strip across delivery months.

References:
    Black, *The Pricing of Commodity Contracts*, JFE, 1976.
    Geman, *Commodities and Commodity Derivatives*, Ch. 6, 2005.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum

import numpy as np

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.models.black76 import (
    OptionType, black76_price, black76_delta, black76_gamma,
    black76_vega, black76_theta,
)


class CommodityType(Enum):
    """Commodity sector."""
    ENERGY = "energy"
    METALS = "metals"
    AGRICULTURE = "agriculture"
    SOFTS = "softs"


@dataclass
class CommodityOptionResult:
    """Commodity futures option pricing result."""
    price: float
    price_per_contract: float
    delta: float
    gamma: float
    vega: float          # per 1% vol
    theta: float         # per day
    futures_price: float
    strike: float
    vol: float
    seasonal_adj: float  # seasonal vol multiplier applied
    expiry_years: float
    option_type: str
    commodity: str

    def to_dict(self) -> dict:
        return dict(vars(self))


# ---- Seasonal volatility ----

# Empirical seasonal patterns (multiplier vs annual average)
_SEASONAL_PATTERNS: dict[str, list[float]] = {
    # Natural gas: winter peak (heating demand)
    "NG": [1.3, 1.2, 1.0, 0.85, 0.80, 0.75, 0.80, 0.85, 0.90, 1.0, 1.15, 1.3],
    # Crude oil: moderate seasonality (driving season)
    "CL": [1.0, 1.0, 1.05, 1.10, 1.10, 1.05, 1.0, 0.95, 0.95, 0.95, 1.0, 1.0],
    # Corn: summer weather premium
    "ZC": [0.85, 0.90, 0.95, 1.0, 1.10, 1.20, 1.25, 1.15, 1.0, 0.90, 0.85, 0.85],
    # Wheat: similar to corn
    "ZW": [0.85, 0.90, 0.95, 1.05, 1.15, 1.20, 1.20, 1.10, 0.95, 0.90, 0.85, 0.85],
    # Soybeans: summer growing season
    "ZS": [0.85, 0.90, 0.95, 1.0, 1.10, 1.20, 1.25, 1.15, 1.0, 0.90, 0.85, 0.85],
    # Gold: low seasonality
    "GC": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    # Silver: moderate
    "SI": [1.0, 1.0, 1.05, 1.05, 1.0, 0.95, 0.95, 1.0, 1.05, 1.05, 1.0, 0.95],
}


def seasonal_vol(
    base_vol: float,
    delivery_month: int,
    ticker: str = "CL",
) -> float:
    """Apply seasonal volatility adjustment.

    Args:
        base_vol: annual average implied vol.
        delivery_month: futures delivery month (1-12).
        ticker: commodity ticker for seasonal pattern.

    Returns:
        Seasonally adjusted vol.
    """
    pattern = _SEASONAL_PATTERNS.get(ticker.upper(), [1.0] * 12)
    idx = max(0, min(delivery_month - 1, 11))
    return base_vol * pattern[idx]


def vol_term_structure(
    base_vol: float,
    expiry_years: float,
    samuelson_alpha: float = 0.5,
) -> float:
    """Samuelson effect: front-month vol > back-month vol.

    σ(T) = base_vol × exp(−α × T)

    where α > 0 means vol decays as maturity increases.

    Args:
        base_vol: spot vol.
        expiry_years: time to expiry.
        samuelson_alpha: decay rate (higher = steeper term structure).
    """
    return base_vol * math.exp(-samuelson_alpha * expiry_years)


# ---- Pricing ----

def commodity_option_price(
    futures_price: float,
    strike: float,
    vol: float,
    expiry_date: date,
    valuation_date: date,
    option_type: str = "call",
    rate: float = 0.04,
    ticker: str = "CL",
    multiplier: float = 1000.0,
    delivery_month: int | None = None,
    apply_seasonal: bool = True,
    apply_samuelson: bool = True,
    samuelson_alpha: float = 0.3,
) -> CommodityOptionResult:
    """Price a commodity futures option.

    Applies seasonal vol adjustment and Samuelson effect.

    Args:
        futures_price: current futures price.
        strike: option strike.
        vol: base implied vol.
        expiry_date: option expiry.
        valuation_date: pricing date.
        option_type: "call" or "put".
        rate: risk-free rate.
        ticker: commodity ticker (for seasonal pattern).
        multiplier: contract multiplier.
        delivery_month: futures delivery month (1-12).
        apply_seasonal: apply seasonal vol adjustment.
        apply_samuelson: apply Samuelson term structure effect.
        samuelson_alpha: Samuelson decay parameter.
    """
    T = year_fraction(valuation_date, expiry_date, DayCountConvention.ACT_365_FIXED)
    df = math.exp(-rate * max(T, 0))
    otype = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT

    # Vol adjustments
    adj_vol = vol
    seasonal_mult = 1.0

    if apply_seasonal and delivery_month is not None:
        seasonal_mult = seasonal_vol(1.0, delivery_month, ticker)
        adj_vol *= seasonal_mult

    if apply_samuelson and T > 0:
        adj_vol = vol_term_structure(adj_vol, T, samuelson_alpha)

    # Price
    premium = black76_price(futures_price, strike, adj_vol, T, df, otype)

    # Greeks
    delta = black76_delta(futures_price, strike, adj_vol, T, df, otype)
    gamma = black76_gamma(futures_price, strike, adj_vol, T, df)
    vega = black76_vega(futures_price, strike, adj_vol, T, df) * 0.01
    theta = black76_theta(futures_price, strike, adj_vol, T, df, otype) / 365.0

    return CommodityOptionResult(
        price=premium,
        price_per_contract=premium * multiplier,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        futures_price=futures_price,
        strike=strike,
        vol=adj_vol,
        seasonal_adj=seasonal_mult,
        expiry_years=T,
        option_type=otype.value,
        commodity=ticker.upper(),
    )


def commodity_option_strip(
    futures_prices: list[float],
    strike: float,
    vol: float,
    expiry_dates: list[date],
    delivery_months: list[int],
    valuation_date: date,
    option_type: str = "call",
    rate: float = 0.04,
    ticker: str = "CL",
    multiplier: float = 1000.0,
) -> list[CommodityOptionResult]:
    """Price a strip of commodity options across delivery months.

    Returns one result per delivery month.
    """
    results = []
    for fp, exp, dm in zip(futures_prices, expiry_dates, delivery_months):
        r = commodity_option_price(
            fp, strike, vol, exp, valuation_date,
            option_type, rate, ticker, multiplier, dm,
        )
        results.append(r)
    return results


# ---- Implied vol from premium ----

def commodity_implied_vol(
    premium: float,
    futures_price: float,
    strike: float,
    expiry_years: float,
    rate: float = 0.04,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Extract implied vol from observed option premium via Newton-Raphson."""
    df = math.exp(-rate * expiry_years)
    otype = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT

    sigma = 0.30  # initial guess
    for _ in range(max_iter):
        price = black76_price(futures_price, strike, sigma, expiry_years, df, otype)
        vega = black76_vega(futures_price, strike, sigma, expiry_years, df)
        if abs(vega) < 1e-15:
            break
        sigma -= (price - premium) / vega
        sigma = max(sigma, 0.001)
        if abs(price - premium) < tol:
            break

    return sigma
