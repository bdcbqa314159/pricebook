"""Inflation-commodity link: oil-breakeven regression, hybrid products.

* :func:`oil_breakeven_regression` — oil price → breakeven regression.
* :func:`commodity_inflation_hybrid` — commodity-linked inflation swap.

References:
    Ciccarelli & Mojon, *Global Inflation*, ECB WP, 2010.
    Hobijn, *Commodity Prices and Inflation*, FRBSF, 2008.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class OilBreakevenRegressionResult:
    beta: float             # bps breakeven per $1 oil move
    alpha: float            # intercept
    r_squared: float
    residual_std: float
    n_observations: int

def oil_breakeven_regression(
    oil_changes: list[float], breakeven_changes_bps: list[float],
) -> OilBreakevenRegressionResult:
    """Regress breakeven changes on oil price changes."""
    x = np.array(oil_changes); y = np.array(breakeven_changes_bps)
    if len(x) < 3:
        return OilBreakevenRegressionResult(0, 0, 0, 0, len(x))
    x_mean = x.mean(); y_mean = y.mean()
    cov = ((x - x_mean) * (y - y_mean)).sum()
    var_x = ((x - x_mean)**2).sum()
    beta = cov / max(var_x, 1e-10)
    alpha = y_mean - beta * x_mean
    fitted = alpha + beta * x
    ss_res = ((y - fitted)**2).sum()
    ss_tot = ((y - y_mean)**2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    return OilBreakevenRegressionResult(float(beta), float(alpha), float(r2),
                                          float(math.sqrt(ss_res / max(len(x)-2, 1))), len(x))


@dataclass
class CommodityInflationHybridResult:
    price: float
    commodity_delta: float
    inflation_delta: float

def commodity_inflation_hybrid(
    commodity_paths: np.ndarray, inflation_paths: np.ndarray,
    commodity_weight: float, inflation_weight: float,
    strike: float, notional: float, discount_factor: float,
) -> CommodityInflationHybridResult:
    """Commodity-linked inflation swap: payoff = w_c × commodity_return + w_i × inflation − K.
    """
    c_ret = commodity_paths[:, -1] / commodity_paths[:, 0] - 1
    i_ret = inflation_paths[:, -1] / inflation_paths[:, 0] - 1
    combined = commodity_weight * c_ret + inflation_weight * i_ret
    payoff = notional * np.maximum(combined - strike, 0.0)
    price = float(discount_factor * payoff.mean())
    c_delta = float(np.corrcoef(c_ret, payoff)[0, 1]) if payoff.std() > 0 else 0
    i_delta = float(np.corrcoef(i_ret, payoff)[0, 1]) if payoff.std() > 0 else 0
    return CommodityInflationHybridResult(price, c_delta, i_delta)
