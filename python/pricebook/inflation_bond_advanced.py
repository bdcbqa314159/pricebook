"""Inflation bonds deepening: real yield curve, breakevens, deflation floor.

* :func:`real_yield_curve_bootstrap` — bootstrap real yields from linker prices.
* :func:`breakeven_trade` — nominal − linker decomposition.
* :func:`seasonality_adjusted_breakeven` — seasonal correction.
* :func:`linker_asw` — real asset swap spread.
* :func:`deflation_floor_value` — TIPS deflation floor option.

References:
    Deacon, Derry & Mirfendereski, *Inflation-Indexed Securities*, Wiley, 2004.
    Barclays, *US TIPS: A Guide for Investors*, 2012.
    Kerkhof, *Inflation Derivatives Explained*, Lehman Brothers, 2005.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RealYieldCurveResult:
    """Real yield curve bootstrap result."""
    maturities: np.ndarray
    real_yields: np.ndarray
    nominal_yields: np.ndarray | None
    breakevens: np.ndarray | None


def real_yield_curve_bootstrap(
    linker_prices: list[float],
    notionals: list[float],
    coupon_rates: list[float],
    maturities: list[float],
    cpi_base: float,
    cpi_current: float,
    n_coupons_per_year: int = 2,
) -> RealYieldCurveResult:
    """Bootstrap real yield curve from TIPS/linker prices.

    For each linker, solve for real yield y such that:
        Price = Σ c × (CPI/CPI_base) × e^{-yT} + N × (CPI/CPI_base) × e^{-yT_final}

    Simplified: treat as flat real yield per bond.

    Args:
        linker_prices: dirty prices (adjusted for inflation accrual).
        cpi_base: CPI at bond issue.
        cpi_current: current CPI.
    """
    index_ratio = cpi_current / cpi_base
    mats = np.array(maturities)
    real_yields = np.zeros(len(mats))

    for i, (price, notional, coupon, T) in enumerate(
            zip(linker_prices, notionals, coupon_rates, maturities)):
        # Total cash flows (adjusted for inflation)
        adjusted_notional = notional * index_ratio
        coupon_per_period = notional * coupon / n_coupons_per_year * index_ratio

        # Solve: price = Σ cpn × e^{-y × t_k} + adjusted_N × e^{-y × T}
        # via bisection
        lo, hi = -0.05, 0.20
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            pv = 0.0
            for k in range(1, int(T * n_coupons_per_year) + 1):
                t_k = k / n_coupons_per_year
                pv += coupon_per_period * math.exp(-mid * t_k)
            pv += adjusted_notional * math.exp(-mid * T)
            if pv > price:
                lo = mid
            else:
                hi = mid
        real_yields[i] = 0.5 * (lo + hi)

    return RealYieldCurveResult(
        maturities=mats,
        real_yields=real_yields,
        nominal_yields=None,
        breakevens=None,
    )


@dataclass
class BreakevenTradeResult:
    """Breakeven trade decomposition."""
    nominal_yield_pct: float
    real_yield_pct: float
    breakeven_pct: float            # nominal − real
    risk_premium_est: float         # estimated inflation risk premium


def breakeven_trade(
    nominal_yield_pct: float,
    real_yield_pct: float,
    expected_inflation_pct: float | None = None,
) -> BreakevenTradeResult:
    """Breakeven = nominal yield − real yield.

    Breakeven ≈ expected inflation + inflation risk premium.
    If expected inflation given, can estimate risk premium.

    Args:
        nominal_yield_pct: matched-maturity nominal bond yield.
        real_yield_pct: TIPS/linker real yield.
        expected_inflation_pct: survey/model expected inflation (optional).
    """
    be = nominal_yield_pct - real_yield_pct
    if expected_inflation_pct is not None:
        risk_prem = be - expected_inflation_pct
    else:
        risk_prem = 0.0

    return BreakevenTradeResult(
        nominal_yield_pct=nominal_yield_pct,
        real_yield_pct=real_yield_pct,
        breakeven_pct=float(be),
        risk_premium_est=float(risk_prem),
    )


@dataclass
class SeasonalBreakevenResult:
    """Seasonality-adjusted breakeven."""
    raw_breakeven_pct: float
    seasonal_adjustment_pct: float
    adjusted_breakeven_pct: float


def seasonality_adjusted_breakeven(
    raw_breakeven_pct: float,
    seasonal_factor: float,
) -> SeasonalBreakevenResult:
    """Adjust breakeven for CPI seasonality.

    CPI has strong seasonal pattern (e.g., higher in summer for energy).
    The raw breakeven includes seasonal distortion in the near months.

    adjusted = raw − seasonal_factor

    Args:
        raw_breakeven_pct: unadjusted breakeven.
        seasonal_factor: CPI seasonal distortion (positive = CPI seasonally high).
    """
    adjusted = raw_breakeven_pct - seasonal_factor
    return SeasonalBreakevenResult(
        raw_breakeven_pct=raw_breakeven_pct,
        seasonal_adjustment_pct=float(seasonal_factor),
        adjusted_breakeven_pct=float(adjusted),
    )


@dataclass
class LinkerASWResult:
    """Linker (real) asset swap spread."""
    linker_real_yield_pct: float
    real_swap_rate_pct: float
    real_asw_bps: float


def linker_asw(
    linker_real_yield_pct: float,
    real_swap_rate_pct: float,
) -> LinkerASWResult:
    """Real asset swap spread = linker real yield − real swap rate.

    Analogous to nominal ASW but in real space. Positive = linker cheap
    relative to swaps (supply/demand imbalance).
    """
    spread = (linker_real_yield_pct - real_swap_rate_pct) * 100
    return LinkerASWResult(
        linker_real_yield_pct=linker_real_yield_pct,
        real_swap_rate_pct=real_swap_rate_pct,
        real_asw_bps=float(spread),
    )


@dataclass
class DeflationFloorResult:
    """Deflation floor option value."""
    floor_value: float
    breakeven_pct: float
    vol: float
    T: float
    probability_deflation: float


def deflation_floor_value(
    breakeven_pct: float,
    inflation_vol_pct: float,
    T: float,
    discount_factor: float = 1.0,
) -> DeflationFloorResult:
    """TIPS deflation floor: guaranteed return of par at maturity.

    US TIPS have a deflation floor: principal paid at maturity is
    max(original_par, inflation_adjusted_par). This is a put on the CPI.

    Floor value ≈ DF × N(−d₂) × par (simplified Black model).

    In practice: breakeven near 0 → floor is valuable.

    Args:
        breakeven_pct: current breakeven inflation rate.
        inflation_vol_pct: annual inflation vol.
        T: years to maturity.
    """
    from scipy.stats import norm

    if inflation_vol_pct <= 0 or T <= 0:
        # Deterministic: floor = max(0, -(breakeven × T)) discounted
        deflation = max(-breakeven_pct / 100 * T, 0)  # convert pct to decimal
        prob = 1.0 if breakeven_pct < 0 else 0.0
        return DeflationFloorResult(
            floor_value=float(discount_factor * deflation),
            breakeven_pct=breakeven_pct,
            vol=inflation_vol_pct,
            T=T,
            probability_deflation=prob,
        )

    # Forward CPI ratio ≈ exp(breakeven × T)
    # Floor strike = 1 (return of original par)
    # Put = DF × [N(-d2) − F × N(-d1)]  where F = exp(be × T)
    F = math.exp(breakeven_pct / 100 * T) if abs(breakeven_pct) < 50 else 1.0
    K = 1.0
    sigma = inflation_vol_pct / 100
    sigma_sqrt_T = sigma * math.sqrt(T)

    if sigma_sqrt_T > 1e-10:
        d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        put = discount_factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        prob_defl = norm.cdf(-d2)
    else:
        put = discount_factor * max(K - F, 0)
        prob_defl = 1.0 if F < K else 0.0

    return DeflationFloorResult(
        floor_value=float(max(put, 0.0)),
        breakeven_pct=breakeven_pct,
        vol=inflation_vol_pct,
        T=T,
        probability_deflation=float(prob_defl),
    )
