"""Freight derivatives: Forward Freight Agreements (FFAs) and freight options.

Covers dry bulk and tanker freight rate derivatives, time-charter equivalents,
seasonal forward curve construction, and bunker fuel sensitivity.

* :class:`FFAResult` — FFA pricing result.
* :class:`FreightOptionResult` — Black-76 freight option with Greeks.
* :class:`ContangoBackwardation` — curve shape classification.
* :func:`ffa_price` — Forward Freight Agreement fair value.
* :func:`freight_option_price` — Black-76 option on FFA rate.
* :func:`time_charter_equivalent` — TCE calculation.
* :func:`freight_forward_curve` — seasonal forward curve from spot.
* :func:`bunker_spread` — P&L sensitivity to bunker fuel price.
* :func:`freight_curve_shape` — classify curve as contango or backwardation.

References:
    Alizadeh, A. H., & Nomikos, N. K. (2009). *Shipping Derivatives and Risk
        Management*. Palgrave Macmillan.
    Kavussanos, M. G., & Visvikis, I. D. (2006). *Derivatives and Risk
        Management in Shipping*. Witherbys Publishing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from pricebook.models.black76 import (
    OptionType,
    black76_price,
    black76_delta,
    black76_gamma,
    black76_vega,
    black76_theta,
)


# ---------------------------------------------------------------------------
# Dataclasses and enums
# ---------------------------------------------------------------------------

@dataclass
class FFAResult:
    """Forward Freight Agreement pricing result."""
    fair_value: float       # FFA rate ($/day or $/ton)
    basis: float            # FFA rate minus current spot rate
    settlement_type: str    # "avg" or "point"
    route: str              # Baltic route identifier (e.g. "C5", "TD3C")

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class FreightOptionResult:
    """Black-76 option pricing result on an FFA rate."""
    price: float    # option premium ($/day)
    delta: float    # dPrice/dFFA
    gamma: float    # d²Price/dFFA²
    vega: float     # dPrice/dVol (per 1% vol)
    theta: float    # dPrice/dTime (per calendar day)

    def to_dict(self) -> dict:
        return dict(vars(self))


class ContangoBackwardation(Enum):
    """Freight forward curve shape classification."""
    CONTANGO = "contango"           # forward rates > spot (upward sloping)
    BACKWARDATION = "backwardation" # forward rates < spot (downward sloping)
    FLAT = "flat"                   # rates approximately equal


# ---------------------------------------------------------------------------
# FFA pricing
# ---------------------------------------------------------------------------

def ffa_price(
    spot_rate: float,
    forward_curve_rates: list[float],
    T: float,
    settlement: str = "avg",
    route: str = "C5",
) -> FFAResult:
    """Fair value of a Forward Freight Agreement (FFA).

    An FFA is settled against either the arithmetic average of the relevant
    Baltic Exchange rate over the delivery period (``settlement="avg"``) or a
    single rate at expiry (``settlement="point"``).

    For average settlement the fair value is the simple mean of the expected
    daily rates over the period, since freight rates are not storable and
    risk-neutral expectations drive the term structure.

    Args:
        spot_rate: current Baltic Exchange spot rate ($/day or $/ton).
        forward_curve_rates: expected daily rates for each period in the
            settlement window.  Length determines the averaging window.
        T: time in years to the start of the settlement period.
        settlement: ``"avg"`` (arithmetic average over window) or
            ``"point"`` (single expiry fixing).
        route: Baltic route identifier string, stored in result for reference.

    Returns:
        :class:`FFAResult` with fair_value, basis, settlement_type, route.
    """
    if not forward_curve_rates:
        raise ValueError("forward_curve_rates must contain at least one rate.")

    if settlement == "avg":
        fair_value = sum(forward_curve_rates) / len(forward_curve_rates)
    else:
        # Point settlement: use the last rate in the curve
        fair_value = forward_curve_rates[-1]

    basis = fair_value - spot_rate

    return FFAResult(
        fair_value=fair_value,
        basis=basis,
        settlement_type=settlement,
        route=route,
    )


# ---------------------------------------------------------------------------
# Freight option pricing
# ---------------------------------------------------------------------------

def freight_option_price(
    ffa_rate: float,
    strike: float,
    vol: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> FreightOptionResult:
    """Black-76 option price on an FFA rate.

    FFA options are cash-settled against the FFA rate at expiry.  Black-76 is
    the standard model, treating the FFA rate as the forward price and using
    the risk-free discount factor for the option premium.

    Args:
        ffa_rate: current FFA rate ($/day), used as the forward price.
        strike: option strike rate ($/day).
        vol: implied volatility of the FFA rate (annual).
        T: time to expiry in years.
        r: risk-free rate (annual, continuous).
        option_type: ``"call"`` or ``"put"``.

    Returns:
        :class:`FreightOptionResult` with price and Greeks.
    """
    if ffa_rate <= 0 or strike <= 0:
        raise ValueError("ffa_rate and strike must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if vol <= 0:
        raise ValueError("vol must be positive")

    ot = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT
    df = math.exp(-r * T)

    price = black76_price(ffa_rate, strike, vol, T, df, ot)
    delta = black76_delta(ffa_rate, strike, vol, T, df, ot)
    gamma = black76_gamma(ffa_rate, strike, vol, T, df)
    vega = black76_vega(ffa_rate, strike, vol, T, df) * 0.01   # per 1% vol
    theta = black76_theta(ffa_rate, strike, vol, T, df, ot) / 365.0  # per day

    return FreightOptionResult(price=price, delta=delta, gamma=gamma,
                               vega=vega, theta=theta)


# ---------------------------------------------------------------------------
# Time-Charter Equivalent
# ---------------------------------------------------------------------------

def time_charter_equivalent(
    revenue_per_day: float,
    voyage_costs: float,
    port_costs: float,
    days_at_sea: float,
    days_in_port: float,
) -> dict:
    """Time-Charter Equivalent (TCE) rate for a voyage.

    TCE normalises voyage profitability to a per-day measure, stripping out
    variable voyage expenses.  It allows comparison across different vessel
    sizes and route lengths.

        TCE = (Gross Revenue − Voyage Costs − Port Costs) / Total Days

    Args:
        revenue_per_day: freight revenue per sea-day ($/day).
        voyage_costs: bunker and canal costs for the entire voyage ($).
        port_costs: port dues, pilotage, and stevedoring ($).
        days_at_sea: laden and ballast sea days.
        days_in_port: loading and discharging days.

    Returns:
        dict with ``gross_revenue``, ``total_costs``, ``net_revenue``,
        ``total_days``, and ``tce`` ($/day).
    """
    total_days = days_at_sea + days_in_port
    if total_days <= 0:
        raise ValueError("Total days (sea + port) must be positive.")

    gross_revenue = revenue_per_day * days_at_sea
    total_costs = voyage_costs + port_costs
    net_revenue = gross_revenue - total_costs
    tce = net_revenue / total_days

    return {
        "gross_revenue": gross_revenue,
        "total_costs": total_costs,
        "net_revenue": net_revenue,
        "total_days": total_days,
        "tce": tce,
    }


# ---------------------------------------------------------------------------
# Freight forward curve
# ---------------------------------------------------------------------------

def freight_forward_curve(
    spot_rate: float,
    seasonality_factors: list[float],
    n_months: int = 24,
) -> list[float]:
    """Build a monthly freight forward curve from spot rate and seasonal pattern.

    Each monthly forward rate is the spot rate scaled by the corresponding
    seasonal factor (cyclically applied if the pattern is shorter than
    *n_months*).  This reflects the well-documented seasonality in dry bulk
    and tanker markets (e.g. winter demand peaks for tankers, grain-season
    peaks for bulk).

    Args:
        spot_rate: current spot rate ($/day).
        seasonality_factors: list of multiplicative seasonal adjustments,
            typically 12 monthly factors (1.0 = no adjustment).
        n_months: number of monthly forward rates to generate.

    Returns:
        List of *n_months* forward rates ($/day), month 1 first.
    """
    if not seasonality_factors:
        raise ValueError("seasonality_factors must be non-empty.")

    n_factors = len(seasonality_factors)
    return [
        spot_rate * seasonality_factors[i % n_factors]
        for i in range(n_months)
    ]


# ---------------------------------------------------------------------------
# Bunker spread
# ---------------------------------------------------------------------------

def bunker_spread(
    freight_rate: float,
    bunker_price: float,
    consumption_per_day: float,
    days: float,
) -> dict:
    """P&L sensitivity of a voyage to bunker fuel cost.

    Bunker (marine fuel) is typically the largest variable cost in shipping.
    The bunker spread captures how much of the freight rate is consumed by
    fuel expense.

    Args:
        freight_rate: all-in freight revenue for the voyage ($).
        bunker_price: bunker fuel price ($/MT, typically VLSFO or HSFO).
        consumption_per_day: vessel fuel consumption at sea (MT/day).
        days: voyage duration in days.

    Returns:
        dict with ``total_bunker_cost``, ``bunker_pct_of_revenue``,
        ``net_after_bunker``, and ``breakeven_bunker_price``.
    """
    total_bunker_cost = bunker_price * consumption_per_day * days
    net_after_bunker = freight_rate - total_bunker_cost
    bunker_pct = total_bunker_cost / freight_rate * 100.0 if freight_rate != 0 else float("nan")

    total_consumption = consumption_per_day * days
    breakeven = freight_rate / total_consumption if total_consumption > 0 else float("inf")

    return {
        "total_bunker_cost": total_bunker_cost,
        "bunker_pct_of_revenue": bunker_pct,
        "net_after_bunker": net_after_bunker,
        "breakeven_bunker_price": breakeven,
    }


# ---------------------------------------------------------------------------
# Curve shape classification
# ---------------------------------------------------------------------------

_FLAT_THRESHOLD = 0.02   # 2% band around spot to classify as flat


def freight_curve_shape(forward_rates: list[float]) -> ContangoBackwardation:
    """Classify the freight forward curve as contango, backwardation, or flat.

    Compares the average of the back end of the curve (second half) against
    the front end (first half).  A 2% tolerance band defines a flat market.

    Args:
        forward_rates: ordered list of forward rates (nearest first).

    Returns:
        :class:`ContangoBackwardation` enum value.

    Raises:
        ValueError: if fewer than two rates are provided.
    """
    if len(forward_rates) < 2:
        raise ValueError("At least two forward rates are required.")

    mid = len(forward_rates) // 2
    front_avg = sum(forward_rates[:mid]) / mid
    back_avg = sum(forward_rates[mid:]) / (len(forward_rates) - mid)

    if front_avg == 0:
        return ContangoBackwardation.FLAT

    relative_diff = (back_avg - front_avg) / front_avg

    if relative_diff > _FLAT_THRESHOLD:
        return ContangoBackwardation.CONTANGO
    elif relative_diff < -_FLAT_THRESHOLD:
        return ContangoBackwardation.BACKWARDATION
    else:
        return ContangoBackwardation.FLAT
