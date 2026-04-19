"""Inflation carry: real yield roll-down, linker carry, seasonal carry.

* :func:`real_yield_rolldown` — roll-down P&L on real yield curve.
* :func:`linker_carry_decomposition` — real yield + breakeven carry.
* :func:`inflation_carry_vs_vol` — Sharpe-like signal.

References:
    Barclays, *US TIPS: A Guide for Investors*, 2012.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class RealYieldRolldownResult:
    real_yield_now: float
    real_yield_rolled: float
    rolldown_bps: float
    horizon_years: float

def real_yield_rolldown(
    tenors: list[float], real_yields: list[float],
    position_tenor: float, horizon: float = 1.0,
) -> RealYieldRolldownResult:
    """Roll-down on real yield curve."""
    T = np.array(tenors); Y = np.array(real_yields)
    y_now = float(np.interp(position_tenor, T, Y))
    y_rolled = float(np.interp(position_tenor - horizon, T, Y))
    rolldown = (y_now - y_rolled) * 100
    return RealYieldRolldownResult(y_now, y_rolled, float(rolldown), horizon)


@dataclass
class LinkerCarryResult:
    total_carry_bps: float
    real_yield_carry: float
    breakeven_carry: float
    financing_cost: float

def linker_carry_decomposition(
    real_yield: float, breakeven: float, repo_rate: float,
    duration: float, horizon: float = 1.0,
) -> LinkerCarryResult:
    """Linker carry = real yield carry + breakeven carry − financing.
    Real yield carry ≈ real_yield × horizon × 100.
    Breakeven carry ≈ breakeven × horizon × 100 (CPI accrual).
    Financing ≈ repo_rate × horizon × 100.
    """
    ry_carry = real_yield * horizon * 100
    be_carry = breakeven * horizon * 100
    fin = repo_rate * horizon * 100
    total = ry_carry + be_carry - fin
    return LinkerCarryResult(float(total), float(ry_carry), float(be_carry), float(fin))


@dataclass
class InflationCarryVolResult:
    carry_bps: float
    vol_bps: float
    sharpe: float
    signal: str

def inflation_carry_vs_vol(
    carry_bps: float, breakeven_vol_bps: float,
) -> InflationCarryVolResult:
    """Carry / vol ratio for inflation trades."""
    sharpe = carry_bps / max(breakeven_vol_bps, 1e-10)
    signal = "strong_buy" if sharpe > 1 else "buy" if sharpe > 0.5 else "neutral" if sharpe > -0.5 else "sell"
    return InflationCarryVolResult(carry_bps, breakeven_vol_bps, float(sharpe), signal)
