"""Commodity-rates link: inflation-commodity regression, cross-asset PCA.

* :func:`inflation_commodity_factor_model` — PCA on rates + commodity + inflation.
* :func:`commodity_inflation_swap` — commodity-linked inflation swap.

References:
    Ciccarelli & Mojon, *Global Inflation*, ECB, 2010.
    Stock & Watson, *Forecasting Inflation*, JME, 1999.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class CrossAssetPCAResult:
    n_factors: int
    explained_variance: np.ndarray
    cumulative_explained: np.ndarray
    factor_loadings: np.ndarray     # (n_factors, n_series)
    series_names: list[str]
    dominant_factor: str

def inflation_commodity_factor_model(
    series_data: dict[str, list[float]],
    n_factors: int = 3,
) -> CrossAssetPCAResult:
    """PCA factor model across rates, commodity, and inflation series."""
    names = list(series_data.keys())
    data = np.column_stack([np.array(series_data[n]) for n in names])
    centered = data - data.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
    total = eigvals.sum()
    explained = eigvals / max(total, 1e-10)
    cumulative = np.cumsum(explained)

    loadings = eigvecs[:, :n_factors].T

    # Dominant: series with highest loading on PC1
    dominant_idx = int(np.argmax(np.abs(loadings[0])))

    return CrossAssetPCAResult(n_factors, explained[:n_factors],
                                 cumulative[:n_factors], loadings,
                                 names, names[dominant_idx])


@dataclass
class CommodityInflationSwapResult:
    price: float
    commodity_return_component: float
    inflation_component: float
    notional: float

def commodity_inflation_swap(
    commodity_paths: np.ndarray,
    cpi_paths: np.ndarray,
    commodity_weight: float,
    cpi_weight: float,
    fixed_rate: float,
    notional: float,
    discount_factor: float,
) -> CommodityInflationSwapResult:
    """Commodity-linked inflation swap: floating = w_c × commodity_return + w_i × CPI_return.
    Fixed leg pays fixed_rate. PV = notional × DF × (floating − fixed).
    """
    c_ret = commodity_paths[:, -1] / commodity_paths[:, 0] - 1
    i_ret = cpi_paths[:, -1] / cpi_paths[:, 0] - 1
    floating = commodity_weight * c_ret + cpi_weight * i_ret
    pv = notional * discount_factor * (floating - fixed_rate)
    price = float(pv.mean())
    c_component = float(commodity_weight * c_ret.mean())
    i_component = float(cpi_weight * i_ret.mean())
    return CommodityInflationSwapResult(price, c_component, i_component, notional)
