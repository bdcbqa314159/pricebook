"""Advanced inflation models: YoY convexity, LPI pricing, inflation swaptions.

Extends :mod:`pricebook.inflation` and :mod:`pricebook.inflation_vol` with:

* :func:`yoy_convexity_adjustment` — ZC-to-YoY convexity correction.
* :func:`lpi_swap_price` — limited price index (capped/floored YoY).
* :func:`inflation_swaption_price` — option on breakeven swap rate.
* :func:`real_rate_swaption_price` — option on real rate.

References:
    Mercurio, *Pricing Inflation-Indexed Derivatives*, QF, 2005.
    Kerkhof, *Inflation Derivatives Explained*, Lehman Brothers, 2005.
    Brigo & Mercurio, *Interest Rate Models*, Ch. 15.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.black76 import black76_price, OptionType


# ---- YoY convexity adjustment ----

@dataclass
class YoYConvexityResult:
    """Year-on-year convexity adjustment result."""
    zc_rate: float
    yoy_rate: float
    convexity_adjustment: float
    tenor: float


def yoy_convexity_adjustment(
    zc_inflation_rate: float,
    nominal_vol: float,
    real_vol: float,
    inflation_vol: float,
    nominal_real_corr: float,
    tenor: float,
    year: int,
) -> YoYConvexityResult:
    """Convexity adjustment from ZC inflation rate to YoY forward rate.

    The YoY forward rate differs from the ZC rate because of the
    covariance between the inflation index ratio I(T_i)/I(T_{i-1})
    and the nominal discount factor P(t, T_i).

    YoY_i ≈ ZC_i + σ_I × σ_n × ρ_{I,n} × (T_i − T_{i-1})

    where σ_I = inflation vol, σ_n = nominal vol, ρ = correlation.

    For multi-year: the adjustment grows with the year index.

    Args:
        zc_inflation_rate: zero-coupon (breakeven) inflation rate.
        nominal_vol: nominal rate volatility.
        real_vol: real rate volatility.
        inflation_vol: inflation index volatility.
        nominal_real_corr: correlation between nominal rate and inflation.
        tenor: payment period length (typically 1.0 for annual).
        year: year index (1, 2, ..., n).
    """
    # Convexity correction grows with year index
    adj = inflation_vol * nominal_vol * nominal_real_corr * tenor * year

    yoy_rate = zc_inflation_rate + adj

    return YoYConvexityResult(zc_inflation_rate, yoy_rate, adj, tenor)


# ---- Limited Price Index (LPI) ----

@dataclass
class LPIResult:
    """Limited Price Index swap pricing result."""
    price: float
    effective_rate: float
    cap_rate: float
    floor_rate: float
    n_periods: int
    method: str


def lpi_swap_price(
    notional: float,
    zc_inflation_rate: float,
    cap: float,
    floor: float,
    maturity: float,
    inflation_vol: float,
    discount_rate: float = 0.04,
    periods_per_year: int = 1,
) -> LPIResult:
    """Price a Limited Price Index (LPI) swap.

    LPI pays capped/floored annual inflation:
        LPI_i = min(cap, max(floor, YoY_i))

    Decomposition:
        LPI = YoY + floor_put − cap_call
    where floor_put protects the floor, cap_call caps the upside.

    Each year is a separate caplet/floorlet on the inflation rate.

    Args:
        notional: swap notional.
        zc_inflation_rate: expected annual inflation rate.
        cap: annual cap on inflation indexation.
        floor: annual floor on inflation indexation.
        maturity: swap maturity in years.
        inflation_vol: annual inflation vol for option pricing.
        discount_rate: nominal discount rate.
        periods_per_year: typically 1 for annual LPI.
    """
    n_periods = int(maturity * periods_per_year)
    dt = 1.0 / periods_per_year

    total_pv = 0.0

    for i in range(1, n_periods + 1):
        t = i * dt
        df = math.exp(-discount_rate * t)

        # Forward inflation rate for this period (simplified: flat)
        fwd_infl = zc_inflation_rate

        # Base YoY payment
        base = notional * fwd_infl * dt * df

        # Cap cost: short a call on inflation rate at cap level
        if fwd_infl > 0 and inflation_vol > 0:
            cap_cost = notional * dt * df * black76_price(
                fwd_infl, cap, t, inflation_vol, 1.0, OptionType.CALL,
            )
            # Floor benefit: long a put on inflation rate at floor level
            floor_benefit = notional * dt * df * black76_price(
                fwd_infl, max(floor, 1e-6), t, inflation_vol, 1.0, OptionType.PUT,
            )
        else:
            cap_cost = 0.0
            floor_benefit = 0.0

        total_pv += base - cap_cost + floor_benefit

    effective_rate = zc_inflation_rate  # approximate
    if n_periods > 0:
        avg_df = math.exp(-discount_rate * maturity / 2)
        if notional * maturity * avg_df > 0:
            effective_rate = total_pv / (notional * maturity * avg_df * dt)

    return LPIResult(total_pv, effective_rate, cap, floor, n_periods, "capfloor_decomposition")


# ---- Inflation swaption ----

@dataclass
class InflationSwaptionResult:
    """Inflation swaption pricing result."""
    price: float
    forward_breakeven: float
    strike: float
    vol: float
    is_payer: bool


def inflation_swaption_price(
    notional: float,
    forward_breakeven: float,
    strike: float,
    expiry: float,
    swap_tenor: float,
    vol: float,
    discount_rate: float = 0.04,
    is_payer: bool = True,
) -> InflationSwaptionResult:
    """Price a swaption on a breakeven inflation swap.

    A payer inflation swaption gives the right to enter a swap
    paying fixed (strike) and receiving inflation. Exercise if
    breakeven > strike.

    Priced via Black-76 on the forward breakeven rate.

    Args:
        notional: swap notional.
        forward_breakeven: forward breakeven inflation rate.
        strike: swaption strike (fixed rate).
        expiry: option expiry (years).
        swap_tenor: underlying swap tenor (years).
        vol: breakeven rate volatility.
        discount_rate: for discounting.
        is_payer: True = right to pay fixed (bullish on inflation).
    """
    df = math.exp(-discount_rate * expiry)
    annuity = sum(math.exp(-discount_rate * (expiry + i))
                  for i in range(1, int(swap_tenor) + 1))

    opt_type = OptionType.CALL if is_payer else OptionType.PUT
    unit_price = black76_price(forward_breakeven, strike, expiry, vol, 1.0, opt_type)

    price = notional * annuity * unit_price

    return InflationSwaptionResult(price, forward_breakeven, strike, vol, is_payer)


# ---- Real rate swaption ----

@dataclass
class RealRateSwaptionResult:
    """Real rate swaption pricing result."""
    price: float
    forward_real_rate: float
    strike: float
    vol: float


def real_rate_swaption_price(
    notional: float,
    forward_real_rate: float,
    strike: float,
    expiry: float,
    swap_tenor: float,
    real_rate_vol: float,
    discount_rate: float = 0.04,
    is_payer: bool = True,
) -> RealRateSwaptionResult:
    """Price a swaption on a real rate swap.

    The real rate is the difference between nominal and breakeven:
        real_rate = nominal_rate − breakeven_rate

    A payer real rate swaption exercises when real rates rise
    (nominal up or inflation down).

    Args:
        notional: swap notional.
        forward_real_rate: forward real rate.
        strike: real rate strike.
        expiry: option expiry.
        swap_tenor: underlying swap tenor.
        real_rate_vol: volatility of the real rate.
        discount_rate: for discounting.
        is_payer: True = right to pay fixed real rate.
    """
    df = math.exp(-discount_rate * expiry)
    annuity = sum(math.exp(-discount_rate * (expiry + i))
                  for i in range(1, int(swap_tenor) + 1))

    opt_type = OptionType.CALL if is_payer else OptionType.PUT

    # Real rate can be negative, use shifted Black if needed
    shift = max(0.0, 0.02 - forward_real_rate, 0.02 - strike)
    f_shifted = forward_real_rate + shift
    k_shifted = strike + shift

    unit_price = black76_price(f_shifted, k_shifted, expiry, real_rate_vol, 1.0, opt_type)

    price = notional * annuity * unit_price

    return RealRateSwaptionResult(price, forward_real_rate, strike, real_rate_vol)
