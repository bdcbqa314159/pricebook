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

from pricebook.calibration import curve_calibration_record
from pricebook.core.solvers import brentq


@dataclass
class RealYieldCurveResult:
    """Real yield curve bootstrap result."""
    maturities: np.ndarray
    real_yields: np.ndarray
    nominal_yields: np.ndarray | None
    breakevens: np.ndarray | None
    calibration_result: object = None  # canonical record, set by the bootstrap



    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items() if k != "calibration_result"}

    def to_calibration_result(self):
        """`ProvenanceCarrier`: the calibration record this result carries, or None."""
        return self.calibration_result


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
    residuals: list[float] = []

    for i, (price, notional, coupon, T) in enumerate(
            zip(linker_prices, notionals, coupon_rates, maturities)):
        # Total cash flows (adjusted for inflation)
        adjusted_notional = notional * index_ratio
        coupon_per_period = notional * coupon / n_coupons_per_year * index_ratio

        # Solve: price = Σ cpn × e^{-y × t_k} + adjusted_N × e^{-y × T}
        def _pv_obj(y: float) -> float:
            pv = 0.0
            for k in range(1, int(T * n_coupons_per_year) + 1):
                t_k = k / n_coupons_per_year
                pv += coupon_per_period * math.exp(-y * t_k)
            pv += adjusted_notional * math.exp(-y * T)
            return pv - price

        real_yields[i] = brentq(_pv_obj, -0.05, 0.30)
        # Residual = linker repricing error at the solved real yield (~0, brentq).
        residuals.append(_pv_obj(real_yields[i]))

    result = RealYieldCurveResult(
        maturities=mats,
        real_yields=real_yields,
        nominal_yields=None,
        breakevens=None,
    )
    result.calibration_result = curve_calibration_record(
        model_class="real_yield_curve_bootstrap",
        parameters={f"real_yield_{float(T):g}y": float(y)
                    for T, y in zip(maturities, real_yields)},
        residuals=residuals,
        quotes_fitted=[f"linker_{float(T):g}y" for T in maturities],
        algorithm="bootstrap",  # per-linker brentq on the price equation
        iterations=len(mats),
        converged=True,
        diagnostics_extra={"index_ratio": float(index_ratio), "n_linkers": len(mats)},
    )
    return result


@dataclass
class BreakevenTradeResult:
    """Breakeven trade decomposition."""
    nominal_yield_pct: float
    real_yield_pct: float
    breakeven_pct: float            # nominal − real
    risk_premium_est: float         # estimated inflation risk premium



    def to_dict(self) -> dict:
        return dict(vars(self))
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



    def to_dict(self) -> dict:
        return dict(vars(self))
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



    def to_dict(self) -> dict:
        return dict(vars(self))
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



    def to_dict(self) -> dict:
        return dict(vars(self))
def deflation_floor_value(
    breakeven: float,
    inflation_vol: float,
    T: float,
    discount_factor: float = 1.0,
) -> DeflationFloorResult:
    """TIPS deflation floor: guaranteed return of par at maturity.

    US TIPS have a deflation floor: principal paid at maturity is
    max(original_par, inflation_adjusted_par). This is a put on the CPI.

    Floor value via Black model on the CPI index ratio.

    Args:
        breakeven: current breakeven inflation rate (decimal, e.g. 0.025 = 2.5%).
        inflation_vol: annual inflation vol (decimal, e.g. 0.01 = 1%).
        T: years to maturity.
    """
    from scipy.stats import norm

    if inflation_vol <= 0 or T <= 0:
        # Fix T4-INF1: pre-fix used the linearised approximation
        # ``-breakeven·T`` (first-order Taylor of ``1 - exp(breakeven·T)``).
        # For breakeven=-5%, T=30y this gave 1.5 — exceeding the maximum
        # possible deflation (1.0 = 100% loss).  Now uses the exact
        # deterministic-limit ``max(1 - exp(breakeven·T), 0) = max(K - F, 0)``
        # at K=1 and F=exp(breakeven·T), matching the interior Black put limit.
        F = math.exp(breakeven * T)
        K = 1.0
        deflation = max(K - F, 0.0)
        prob = 1.0 if breakeven < 0 else 0.0
        return DeflationFloorResult(
            floor_value=float(discount_factor * deflation),
            breakeven_pct=breakeven * 100,
            vol=inflation_vol * 100,
            T=T,
            probability_deflation=prob,
        )

    F = math.exp(breakeven * T)
    K = 1.0
    sigma_sqrt_T = inflation_vol * math.sqrt(T)

    if sigma_sqrt_T > 1e-10:
        d1 = (math.log(F / K) + 0.5 * inflation_vol**2 * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        put = discount_factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        prob_defl = norm.cdf(-d2)
    else:
        put = discount_factor * max(K - F, 0)
        prob_defl = 1.0 if F < K else 0.0

    return DeflationFloorResult(
        floor_value=float(max(put, 0.0)),
        breakeven_pct=breakeven * 100,
        vol=inflation_vol * 100,
        T=T,
        probability_deflation=float(prob_defl),
    )
