"""Inflation basis: ZC vs YoY, cross-market, basis trading.

* :func:`zc_yoy_basis` — ZC vs YoY inflation swap basis.
* :func:`cross_market_inflation_basis` — HICP vs CPI vs RPI.
* :func:`inflation_basis_trade` — basis trade construction.

References:
    Kerkhof, *Inflation Derivatives Explained*, Lehman Brothers, 2005.
    Deacon et al., *Inflation-Indexed Securities*, Wiley, 2004.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class ZCYoYBasisResult:
    zc_rate: float
    yoy_rate: float
    basis_bps: float
    tenor: float
    convexity_component: float

def zc_yoy_basis(
    zc_rate: float, yoy_rate: float, tenor: float,
    convexity_adj: float = 0.0,
) -> ZCYoYBasisResult:
    """ZC vs YoY inflation swap basis.
    Basis = ZC − YoY (should be ≈ convexity adjustment).
    """
    basis = (zc_rate - yoy_rate) * 100 * 100  # in bps
    return ZCYoYBasisResult(zc_rate, yoy_rate, float(basis), tenor, convexity_adj * 10000)


@dataclass
class CrossMarketBasisResult:
    market1: str
    market2: str
    rate1: float
    rate2: float
    basis_bps: float
    z_score: float

def cross_market_inflation_basis(
    market1: str, rate1: float, market2: str, rate2: float,
    historical_basis_bps: list[float] | None = None,
) -> CrossMarketBasisResult:
    """Cross-market basis: HICP vs CPI vs RPI."""
    basis = (rate1 - rate2) * 10000
    if historical_basis_bps and len(historical_basis_bps) > 1:
        arr = np.array(historical_basis_bps)
        z = (basis - arr.mean()) / max(arr.std(), 1e-10)
    else:
        z = 0.0
    return CrossMarketBasisResult(market1, market2, rate1, rate2, float(basis), float(z))


@dataclass
class InflationBasisTradeResult:
    leg1_rate: float
    leg2_rate: float
    basis_bps: float
    dv01_per_bp: float
    expected_pnl: float

def inflation_basis_trade(
    leg1_rate: float, leg2_rate: float,
    notional: float, dv01_per_bp: float,
    target_basis_bps: float = 0.0,
) -> InflationBasisTradeResult:
    """Construct basis trade: long one leg, short another.
    Expected P&L = (current_basis − target) × DV01.
    """
    basis = (leg1_rate - leg2_rate) * 10000
    expected = (basis - target_basis_bps) * dv01_per_bp
    return InflationBasisTradeResult(leg1_rate, leg2_rate, float(basis),
                                      dv01_per_bp, float(expected))
