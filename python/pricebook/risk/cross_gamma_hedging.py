"""Cross-gamma hedging: optimal multi-asset rehedge, vega netting, min-variance.

* :func:`optimal_multi_asset_hedge` — minimise portfolio variance.
* :func:`cross_asset_vega_netting` — net vega across asset classes.
* :func:`correlation_aware_sizing` — position size with ρ constraints.
* :func:`minimum_variance_exotic_hedge` — optimal vanilla basket for exotic.

References:
    Alexander, *Market Risk Analysis*, Vol. IV, Wiley, 2008.
    Hull, *Options, Futures, and Other Derivatives*, Ch. 19.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class OptimalHedgeResult:
    """Optimal multi-asset hedge result."""
    hedge_weights: np.ndarray
    portfolio_variance: float
    unhedged_variance: float
    variance_reduction_pct: float
    n_instruments: int

def optimal_multi_asset_hedge(
    target_greeks: np.ndarray,       # (n_greeks,) portfolio Greeks to neutralise
    instrument_greeks: np.ndarray,   # (n_instruments, n_greeks) per instrument
    covariance: np.ndarray | None = None,  # (n_greeks, n_greeks) optional for variance calc
) -> OptimalHedgeResult:
    """Find hedge weights to neutralise target Greeks.
    min ||target + A'w||² where A = instrument_greeks.
    """
    A = np.array(instrument_greeks)
    b = np.array(target_greeks)
    # Solve: A'w ≈ -b → min ||Aw + b||²
    w, _, _, _ = np.linalg.lstsq(A.T, -b, rcond=None)

    residual = b + A.T @ w
    if covariance is not None:
        hedged_var = float(residual @ covariance @ residual)
        unhedged_var = float(b @ covariance @ b)
    else:
        hedged_var = float(np.sum(residual**2))
        unhedged_var = float(np.sum(b**2))

    reduction = (1 - hedged_var / max(unhedged_var, 1e-10)) * 100

    return OptimalHedgeResult(w, hedged_var, unhedged_var, float(reduction), len(w))


@dataclass
class VegaNettingResult:
    """Cross-asset vega netting result."""
    gross_vega: float
    net_vega: float
    netting_benefit_pct: float
    per_asset_vega: dict[str, float]

def cross_asset_vega_netting(
    vega_by_asset: dict[str, float],
    correlations: dict[tuple[str, str], float] | None = None,
) -> VegaNettingResult:
    """Net vega across asset classes.
    Gross = Σ |vega_i|; Net = √(Σ vega² + 2 Σ ρ vega_i vega_j).
    """
    assets = list(vega_by_asset.keys())
    vegas = np.array([vega_by_asset[a] for a in assets])
    gross = float(np.sum(np.abs(vegas)))

    # Compute net vega with correlations
    n = len(assets)
    var = float(np.sum(vegas**2))
    if correlations:
        for i in range(n):
            for j in range(i+1, n):
                key = (assets[i], assets[j])
                rho = correlations.get(key, correlations.get((assets[j], assets[i]), 0.0))
                var += 2 * rho * vegas[i] * vegas[j]

    net = math.sqrt(max(var, 0))
    benefit = (1 - net / max(gross, 1e-10)) * 100

    return VegaNettingResult(gross, float(net), float(benefit), dict(vega_by_asset))


@dataclass
class CorrelationAwareSizingResult:
    """Position sizing with correlation constraints."""
    optimal_size: float
    max_loss_at_conf: float
    correlation_impact: float

def correlation_aware_sizing(
    expected_pnl: float,
    pnl_vol: float,
    correlation_to_portfolio: float,
    portfolio_vol: float,
    max_portfolio_vol_increase_pct: float = 5.0,
) -> CorrelationAwareSizingResult:
    """Size a new position accounting for correlation to existing book.
    Marginal portfolio vol = ∂σ_p / ∂w ≈ ρ × σ_new × σ_p / σ_p.
    Max size: such that marginal vol increase ≤ threshold.
    """
    if pnl_vol <= 0:
        return CorrelationAwareSizingResult(0, 0, 0)

    # Marginal contribution: ρ × σ_new / σ_portfolio
    marginal = abs(correlation_to_portfolio) * pnl_vol
    max_increase = portfolio_vol * max_portfolio_vol_increase_pct / 100

    if marginal > 1e-10:
        max_size = max_increase / marginal
    else:
        max_size = float("inf")

    # Optimal: size for best risk-adjusted return
    sharpe = expected_pnl / max(pnl_vol, 1e-10)
    optimal = min(max_size, abs(expected_pnl) / max(pnl_vol**2, 1e-10))

    max_loss = 2.33 * pnl_vol * optimal  # 99% VaR
    corr_impact = correlation_to_portfolio * pnl_vol * optimal

    return CorrelationAwareSizingResult(float(optimal), float(max_loss), float(corr_impact))


@dataclass
class MinVarianceExoticHedgeResult:
    """Minimum-variance hedge for multi-asset exotic."""
    hedge_weights: np.ndarray
    hedged_std: float
    unhedged_std: float
    variance_reduction_pct: float

def minimum_variance_exotic_hedge(
    exotic_pnl: np.ndarray,         # (n_scenarios,) P&L of exotic
    hedge_pnls: np.ndarray,         # (n_scenarios, n_hedges) P&L of hedge instruments
) -> MinVarianceExoticHedgeResult:
    """Find optimal basket of vanilla hedges for a multi-asset exotic.
    min_w Var(exotic_pnl + Σ w_i × hedge_pnl_i).
    Solved via OLS: regress exotic on hedge instruments.
    """
    # w that minimise Var(exotic + H w) = w that solve H'H w = -H'exotic
    w, _, _, _ = np.linalg.lstsq(hedge_pnls, -exotic_pnl, rcond=None)

    hedged = exotic_pnl + hedge_pnls @ w
    hedged_std = float(hedged.std())
    unhedged_std = float(exotic_pnl.std())
    reduction = (1 - hedged_std**2 / max(unhedged_std**2, 1e-10)) * 100

    return MinVarianceExoticHedgeResult(w, hedged_std, unhedged_std, float(reduction))
