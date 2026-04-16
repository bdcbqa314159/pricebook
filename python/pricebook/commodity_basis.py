"""Commodity basis: crude grade, locational, quality spreads.

* :class:`BasisCurve` — term structure of basis differentials.
* :func:`wti_brent_basis` — WTI/Brent basis curve analytics.
* :func:`power_locational_basis` — PJM hub vs zone differentials.
* :func:`gas_basis_curve` — HH vs regional gas hubs.
* :func:`quality_basis` — sulphur, API, heat content adjustments.

References:
    Clewlow & Strickland, *Energy Derivatives*, Wiley, 2000, Ch. 8.
    Eydeland & Wolyniec, *Energy and Power Risk Management*, Wiley, 2003.
    Lyle & Elliott, *A Model for the Dynamics of Locational Prices*, RE, 2009.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---- Basis curve ----

@dataclass
class BasisCurve:
    """Term structure of basis differentials.

    Basis = price_derivative − price_benchmark.
    Stored as (tenor, basis) pairs with interpolation.
    """
    tenors: np.ndarray
    basis_values: np.ndarray    # can be negative (discount) or positive (premium)
    benchmark: str
    derivative: str

    def basis(self, T: float) -> float:
        """Interpolated basis at tenor T."""
        if T <= self.tenors[0]:
            return float(self.basis_values[0])
        if T >= self.tenors[-1]:
            return float(self.basis_values[-1])
        return float(np.interp(T, self.tenors, self.basis_values))

    def forward_price(self, T: float, benchmark_forward: float) -> float:
        """Derivative forward price at tenor T."""
        return benchmark_forward + self.basis(T)


def basis_curve_from_futures(
    tenors: list[float],
    benchmark_forwards: list[float],
    derivative_forwards: list[float],
    benchmark: str = "benchmark",
    derivative: str = "derivative",
) -> BasisCurve:
    """Construct basis curve from aligned forward prices."""
    T = np.array(tenors)
    B = np.array(benchmark_forwards)
    D = np.array(derivative_forwards)
    basis = D - B
    return BasisCurve(T, basis, benchmark, derivative)


# ---- WTI / Brent ----

@dataclass
class WTIBrentResult:
    """WTI/Brent basis analysis."""
    basis_curve: BasisCurve
    near_month_basis: float
    long_end_basis: float
    backwardation_ratio: float      # signed measure of shape


def wti_brent_basis(
    tenors: list[float],
    wti_forwards: list[float],
    brent_forwards: list[float],
) -> WTIBrentResult:
    """Analyse WTI/Brent basis (typically Brent > WTI since ~2011).

    Positive basis (Brent − WTI) typical for geographic / quality premium.
    """
    curve = basis_curve_from_futures(tenors, wti_forwards, brent_forwards,
                                       "WTI", "Brent")

    return WTIBrentResult(
        basis_curve=curve,
        near_month_basis=float(curve.basis_values[0]),
        long_end_basis=float(curve.basis_values[-1]),
        backwardation_ratio=float(curve.basis_values[-1] - curve.basis_values[0]),
    )


# ---- Power locational basis ----

@dataclass
class PowerLocationalBasis:
    """Power locational basis result (e.g. PJM hub vs zone)."""
    hub_forwards: np.ndarray
    node_forwards: np.ndarray
    basis_values: np.ndarray
    tenors: np.ndarray
    mean_basis: float
    max_congestion: float           # max absolute basis (congestion rent proxy)


def power_locational_basis(
    tenors: list[float],
    hub_forwards: list[float],
    node_forwards: list[float],
) -> PowerLocationalBasis:
    """Locational basis for power (hub vs zonal or nodal pricing).

    Basis = node_price − hub_price. Captures congestion and losses.
    """
    T = np.array(tenors)
    H = np.array(hub_forwards)
    N = np.array(node_forwards)
    basis = N - H

    return PowerLocationalBasis(
        hub_forwards=H,
        node_forwards=N,
        basis_values=basis,
        tenors=T,
        mean_basis=float(basis.mean()),
        max_congestion=float(np.abs(basis).max()),
    )


# ---- Gas basis ----

@dataclass
class GasBasisResult:
    """Gas basis result (HH vs regional hub)."""
    basis_curve: BasisCurve
    seasonal_peak_month: int         # 1-12, month of maximum basis
    annual_mean_basis: float


def gas_basis_curve(
    tenors: list[float],
    hh_forwards: list[float],
    regional_forwards: list[float],
    hub_name: str = "CityGate",
) -> GasBasisResult:
    """Gas basis between Henry Hub and a regional hub.

    Regional hubs typically have winter premium (CityGate, Algonquin)
    or summer discount (AECO).

    Args:
        tenors: tenors (years; assumed monthly).
        hh_forwards: Henry Hub forwards.
        regional_forwards: regional hub forwards.
    """
    curve = basis_curve_from_futures(tenors, hh_forwards, regional_forwards,
                                       "HH", hub_name)

    # Identify peak month from monthly pattern
    basis = curve.basis_values
    peak_idx = int(np.argmax(basis))
    # Convert index to month (assuming starts in current month)
    month = (peak_idx % 12) + 1

    return GasBasisResult(
        basis_curve=curve,
        seasonal_peak_month=month,
        annual_mean_basis=float(basis.mean()),
    )


# ---- Quality basis ----

@dataclass
class QualityBasisResult:
    """Quality basis adjustment result."""
    reference_price: float
    adjusted_price: float
    total_adjustment: float
    adjustments_breakdown: dict[str, float]


def quality_basis(
    reference_price: float,
    api_gravity_delta: float = 0.0,
    sulphur_pct_delta: float = 0.0,
    heat_content_delta: float = 0.0,
    api_coefficient: float = 0.10,
    sulphur_coefficient: float = -5.0,
    heat_coefficient: float = 1.0,
) -> QualityBasisResult:
    """Quality-adjusted commodity price.

    Typical adjustments for crude:
    - Higher API gravity (lighter crude) → price premium.
    - Higher sulphur → price discount (sour crude).
    - Higher heat content → price premium.

    Default coefficients are indicative (calibrated empirically).

    Args:
        reference_price: reference crude price (e.g. WTI spot).
        api_gravity_delta: API gravity difference vs reference (e.g. +5°).
        sulphur_pct_delta: sulphur content difference (e.g. +0.5%).
        heat_content_delta: heat content (MMBtu/bbl) difference.
        api_coefficient, sulphur_coefficient, heat_coefficient: USD per unit.
    """
    adjustments = {
        "api": api_gravity_delta * api_coefficient,
        "sulphur": sulphur_pct_delta * sulphur_coefficient,
        "heat": heat_content_delta * heat_coefficient,
    }
    total = sum(adjustments.values())
    adjusted = reference_price + total

    return QualityBasisResult(
        reference_price=reference_price,
        adjusted_price=float(adjusted),
        total_adjustment=float(total),
        adjustments_breakdown=adjustments,
    )
