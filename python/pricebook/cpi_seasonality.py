"""CPI seasonality: Canty-Heider model, monthly factors, deseasonalised breakevens.

* :func:`estimate_seasonal_factors` — extract from historical CPI.
* :func:`deseasonalise_breakeven` — remove seasonal distortion.
* :func:`seasonal_carry_signal` — carry signal from CPI seasonality.

References:
    Canty & Heider, *Seasonality in CPI*, BIS, 2012.
    Barclays, *Inflation Seasonality and TIPS Valuation*, 2011.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class SeasonalFactors:
    monthly_factors: np.ndarray   # (12,) multiplicative factors
    mean_annual_inflation: float
    residual_std: float
    method: str

def estimate_seasonal_factors(
    monthly_cpi_changes_pct: list[float],
) -> SeasonalFactors:
    """Estimate monthly CPI seasonal factors from historical data.
    Factor for month m = mean(CPI_change_m) / mean(all months).
    """
    data = np.array(monthly_cpi_changes_pct)
    n_years = len(data) // 12
    if n_years < 1:
        return SeasonalFactors(np.ones(12), float(data.mean()), 0.0, "insufficient_data")
    monthly = data[:n_years*12].reshape(n_years, 12)
    monthly_means = monthly.mean(axis=0)
    overall_mean = monthly_means.mean()
    if abs(overall_mean) > 1e-10:
        factors = monthly_means / overall_mean
    else:
        factors = np.ones(12)
    residuals = monthly - monthly_means
    return SeasonalFactors(factors, float(overall_mean * 12), float(residuals.std()), "canty_heider")


@dataclass
class DeseasonalisedBreakevenResult:
    raw_breakeven: float
    seasonal_adjustment: float
    deseasonalised: float
    current_month: int

def deseasonalise_breakeven(
    raw_breakeven: float, seasonal_factors: SeasonalFactors,
    current_month: int, months_to_maturity: int,
) -> DeseasonalisedBreakevenResult:
    """Remove CPI seasonality from observed breakeven.
    Adjustment: difference between remaining seasonal profile and annual average.
    """
    factors = seasonal_factors.monthly_factors
    # Average seasonal factor over remaining months
    remaining_months = [(current_month + i) % 12 for i in range(months_to_maturity)]
    avg_factor = float(factors[remaining_months].mean()) if remaining_months else 1.0
    adjustment = (avg_factor - 1.0) * seasonal_factors.mean_annual_inflation / 12
    deseasonalised = raw_breakeven - adjustment
    return DeseasonalisedBreakevenResult(raw_breakeven, float(adjustment),
                                          float(deseasonalised), current_month)


@dataclass
class SeasonalCarrySignal:
    signal: float           # positive = buy inflation before high-CPI months
    current_month: int
    next_3m_seasonal: float
    historical_mean: float

def seasonal_carry_signal(
    seasonal_factors: SeasonalFactors, current_month: int,
) -> SeasonalCarrySignal:
    """Carry signal from CPI seasonality.
    If next 3 months have above-average CPI → buy inflation (positive signal).
    """
    factors = seasonal_factors.monthly_factors
    next_3 = [factors[(current_month + i) % 12] for i in range(3)]
    avg_next_3 = float(np.mean(next_3))
    signal = avg_next_3 - 1.0  # positive if above average
    return SeasonalCarrySignal(signal, current_month, avg_next_3, 1.0)
