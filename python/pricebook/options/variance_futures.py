"""VIX futures and variance swaps on futures.

Fair value of VIX futures from variance strip, term structure
analysis, and variance swap pricing.

* :class:`VIXFuturesResult` — VIX futures pricing result.
* :func:`vix_futures_fair_value` — fair value from variance strip.
* :func:`variance_swap_price` — variance swap on futures.
* :func:`vix_term_structure` — VIX futures term structure analytics.
* :func:`vol_of_vol` — vol-of-vol from VIX option prices.

References:
    CBOE, *VIX White Paper*, 2019.
    Carr & Wu, *A Tale of Two Indices*, JFM, 2006.
    Demeterfi et al., *More Than You Ever Wanted to Know About
    Volatility Swaps*, Goldman Sachs QS, 1999.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.black76 import (
    OptionType, black76_price, black76_vega, _norm_cdf, _norm_pdf,
)


@dataclass
class VIXFuturesResult:
    """VIX futures pricing result."""
    fair_value: float           # VIX futures fair value
    spot_vix: float
    basis: float                # futures - spot
    basis_pct: float            # basis as % of spot
    days_to_expiry: int
    roll_yield_annual: float    # annualised roll yield

    def to_dict(self) -> dict:
        return vars(self)


def vix_futures_fair_value(
    spot_vix: float,
    days_to_expiry: int,
    term_premium: float = 0.02,
    mean_reversion_speed: float = 3.0,
    long_run_vol: float = 20.0,
) -> VIXFuturesResult:
    """VIX futures fair value via mean-reversion model.

    F(T) = VIX_∞ + (VIX_0 − VIX_∞) × exp(−κT) + term_premium × √T

    VIX futures trade in contango (F > spot) on average because
    the vol risk premium is negative (sellers of vol insurance are
    compensated).

    Args:
        spot_vix: current VIX level.
        days_to_expiry: days until futures settlement.
        term_premium: annualised term premium.
        mean_reversion_speed: κ — mean reversion speed.
        long_run_vol: long-run VIX level (VIX_∞).
    """
    T = days_to_expiry / 365.0

    # Mean-reversion component
    mr_component = long_run_vol + (spot_vix - long_run_vol) * math.exp(-mean_reversion_speed * T)

    # Term premium (convexity adjustment)
    premium = term_premium * math.sqrt(max(T, 0))

    fair_value = mr_component + premium
    basis = fair_value - spot_vix
    basis_pct = basis / spot_vix * 100 if spot_vix > 0 else 0

    # Roll yield: annualised return from rolling short futures
    roll_yield = -basis / spot_vix / max(T, 1/365) if spot_vix > 0 and T > 0 else 0

    return VIXFuturesResult(
        fair_value=fair_value,
        spot_vix=spot_vix,
        basis=basis,
        basis_pct=basis_pct,
        days_to_expiry=days_to_expiry,
        roll_yield_annual=roll_yield,
    )


@dataclass
class VarianceSwapResult:
    """Variance swap pricing result."""
    fair_variance: float        # fair strike variance (annualised)
    fair_vol: float             # fair strike vol (√variance)
    pv: float                   # mark-to-market PV
    vega_notional: float
    realised_variance: float    # if provided

    def to_dict(self) -> dict:
        return vars(self)


def variance_swap_price(
    spot: float,
    strikes: list[float],
    call_prices: list[float],
    put_prices: list[float],
    T: float,
    r: float = 0.04,
    realised_variance: float | None = None,
    vega_notional: float = 100_000.0,
) -> VarianceSwapResult:
    """Price a variance swap from option strip.

    Fair variance = (2/T) × Σ (ΔK_i / K_i²) × O_i × exp(rT)

    where O_i is the OTM option price at strike K_i.
    This is the model-free replication of variance.

    Args:
        spot: current spot/futures price.
        strikes: option strikes (sorted ascending).
        call_prices: call prices at each strike.
        put_prices: put prices at each strike.
        T: time to expiry (years).
        r: risk-free rate.
        realised_variance: if given, compute MTM PV.
        vega_notional: variance swap notional.
    """
    if T <= 0 or len(strikes) < 2:
        return VarianceSwapResult(0, 0, 0, vega_notional, realised_variance or 0)

    # Use OTM options: puts below spot, calls above
    integral = 0.0
    for i, K in enumerate(strikes):
        if i == 0:
            dK = strikes[1] - strikes[0]
        elif i == len(strikes) - 1:
            dK = strikes[-1] - strikes[-2]
        else:
            dK = (strikes[i + 1] - strikes[i - 1]) / 2.0

        # Use OTM option
        if K < spot:
            price = put_prices[i]
        elif K > spot:
            price = call_prices[i]
        else:
            price = (call_prices[i] + put_prices[i]) / 2.0

        integral += dK / (K * K) * price

    fair_variance = 2.0 / T * math.exp(r * T) * integral
    fair_vol = math.sqrt(max(fair_variance, 0))

    # MTM PV
    if realised_variance is not None:
        pv = vega_notional / (2 * fair_vol) * (realised_variance - fair_variance) if fair_vol > 0 else 0
    else:
        pv = 0.0

    return VarianceSwapResult(
        fair_variance=fair_variance,
        fair_vol=fair_vol * 100,  # in vol points
        pv=pv,
        vega_notional=vega_notional,
        realised_variance=realised_variance or 0,
    )


@dataclass
class VIXTermStructurePoint:
    """Single point on VIX term structure."""
    days: int
    futures_price: float
    basis: float
    contango: bool

    def to_dict(self) -> dict:
        return vars(self)


def vix_term_structure(
    spot_vix: float,
    futures_prices: list[tuple[int, float]],
) -> list[VIXTermStructurePoint]:
    """Analyse VIX futures term structure.

    Args:
        spot_vix: current VIX level.
        futures_prices: list of (days_to_expiry, futures_price).

    Returns:
        Term structure analysis per contract.
    """
    results = []
    for days, price in sorted(futures_prices):
        basis = price - spot_vix
        results.append(VIXTermStructurePoint(
            days=days,
            futures_price=price,
            basis=basis,
            contango=price > spot_vix,
        ))
    return results


def vol_of_vol(
    vix_spot: float,
    vix_option_prices: list[tuple[float, float, str]],
    T: float,
    r: float = 0.04,
) -> float:
    """Implied vol-of-vol from VIX options.

    Uses ATM VIX option to extract vol of the VIX itself.

    Args:
        vix_spot: current VIX level.
        vix_option_prices: list of (strike, premium, "call"/"put").
        T: time to expiry.
        r: risk-free rate.

    Returns:
        Implied vol-of-vol (annualised).
    """
    df = math.exp(-r * T)

    # Find nearest ATM option
    best_diff = float('inf')
    best_price = 0.0
    best_type = OptionType.CALL

    for strike, premium, otype in vix_option_prices:
        diff = abs(strike - vix_spot)
        if diff < best_diff:
            best_diff = diff
            best_price = premium
            best_type = OptionType.CALL if otype == "call" else OptionType.PUT

    # Newton-Raphson for implied vol
    sigma = 0.80  # initial guess (VIX vol is typically high)
    for _ in range(100):
        price = black76_price(vix_spot, vix_spot, sigma, T, df, best_type)
        vega = black76_vega(vix_spot, vix_spot, sigma, T, df)
        if abs(vega) < 1e-15:
            break
        sigma -= (price - best_price) / vega
        sigma = max(sigma, 0.01)
        if abs(price - best_price) < 1e-6:
            break

    return sigma
