"""Efficient frontier computation, tangency portfolio, capital market line.

* :func:`efficient_frontier` — full mean-variance frontier.
* :func:`tangency_portfolio` — maximum Sharpe ratio portfolio.
* :func:`minimum_variance_portfolio` — global minimum variance.
* :func:`capital_market_line` — CML from risk-free to tangency.
* :func:`frontier_with_constraints` — constrained frontier.

References:
    Markowitz, *Portfolio Selection*, JF, 1952.
    Merton, *An Analytic Derivation of the Efficient Portfolio Frontier*,
    JFQA, 1972.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class FrontierPoint:
    """Single point on the efficient frontier."""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: np.ndarray

    def to_dict(self) -> dict:
        return {
            "return": self.expected_return,
            "vol": self.volatility,
            "sharpe": self.sharpe_ratio,
        }


@dataclass
class EfficientFrontierResult:
    """Complete efficient frontier."""
    points: list[FrontierPoint]
    tangency: FrontierPoint | None
    min_variance: FrontierPoint
    n_assets: int
    risk_free_rate: float

    def to_dict(self) -> dict:
        return {
            "n_points": len(self.points),
            "n_assets": self.n_assets,
            "tangency_sharpe": self.tangency.sharpe_ratio if self.tangency else None,
            "min_var_vol": self.min_variance.volatility,
        }


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float = 0.0,
    n_points: int = 50,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> EfficientFrontierResult:
    """Compute the full mean-variance efficient frontier.

    Sweeps target returns from minimum-variance to maximum-return,
    solving a QP at each point.

    Args:
        mu: expected returns (N,).
        cov: covariance matrix (N, N).
        risk_free_rate: for Sharpe ratio and CML.
        n_points: frontier resolution.
        long_only: constrain weights ≥ 0.
        max_weight: maximum weight per asset.
    """
    N = len(mu)

    # Global minimum variance
    mv = minimum_variance_portfolio(cov, long_only, max_weight)
    mv_ret = float(mu @ mv.weights)
    mv.expected_return = mv_ret
    mv.sharpe_ratio = (mv_ret - risk_free_rate) / mv.volatility if mv.volatility > 0 else 0

    # Maximum return (100% in highest-return asset, or solve if constrained)
    max_ret = float(np.max(mu))

    # Sweep
    targets = np.linspace(mv_ret, max_ret, n_points)
    points = []
    best_sharpe = -1e10
    tangency = None

    for target in targets:
        w = _solve_mv_target(mu, cov, float(target), long_only, max_weight)
        if w is None:
            continue

        vol = float(np.sqrt(w @ cov @ w))
        ret = float(mu @ w)
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0

        pt = FrontierPoint(ret, vol, sharpe, w)
        points.append(pt)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            tangency = pt

    return EfficientFrontierResult(
        points=points,
        tangency=tangency,
        min_variance=mv,
        n_assets=N,
        risk_free_rate=risk_free_rate,
    )


def tangency_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float = 0.0,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> FrontierPoint:
    """Maximum Sharpe ratio (tangency) portfolio.

    max (μ'w − rf) / √(w'Σw)
    s.t. Σw = 1, w ≥ 0 (if long_only)
    """
    N = len(mu)
    excess = mu - risk_free_rate

    def neg_sharpe(w):
        ret = excess @ w
        vol = math.sqrt(max(w @ cov @ w, 1e-12))
        return -ret / vol

    w0 = np.ones(N) / N
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if long_only:
        bounds = [(0, max_weight)] * N
    else:
        bounds = [(-max_weight, max_weight)] * N

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints)

    w = result.x if result.success else w0
    vol = float(np.sqrt(w @ cov @ w))
    ret = float(mu @ w)
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0

    return FrontierPoint(ret, vol, sharpe, w)


def minimum_variance_portfolio(
    cov: np.ndarray,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> FrontierPoint:
    """Global minimum variance portfolio.

    min w'Σw  s.t. Σw = 1
    """
    N = cov.shape[0]

    if not long_only:
        # Analytical: w* = Σ⁻¹ 1 / (1' Σ⁻¹ 1)
        try:
            inv_cov = np.linalg.inv(cov)
            ones = np.ones(N)
            w = inv_cov @ ones / (ones @ inv_cov @ ones)
            vol = float(np.sqrt(w @ cov @ w))
            return FrontierPoint(0, vol, 0, w)
        except np.linalg.LinAlgError:
            pass

    # Numerical
    def portfolio_var(w):
        return float(w @ cov @ w)

    w0 = np.ones(N) / N
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * N if long_only else [(-max_weight, max_weight)] * N

    result = minimize(portfolio_var, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints)

    w = result.x if result.success else w0
    vol = float(np.sqrt(w @ cov @ w))
    return FrontierPoint(0, vol, 0, w)


def capital_market_line(
    tangency_point: FrontierPoint,
    risk_free_rate: float,
    n_points: int = 20,
    max_leverage: float = 2.0,
) -> list[dict]:
    """Capital Market Line from risk-free to tangency (and beyond with leverage).

    CML: E[r] = rf + (Sharpe × σ)

    Args:
        tangency_point: tangency portfolio.
        max_leverage: maximum portfolio weight on risky (>1 = leveraged).
    """
    sharpe = tangency_point.sharpe_ratio
    max_vol = tangency_point.volatility * max_leverage

    vols = np.linspace(0, max_vol, n_points)
    cml = []
    for vol in vols:
        ret = risk_free_rate + sharpe * vol
        leverage = vol / tangency_point.volatility if tangency_point.volatility > 0 else 0
        cml.append({
            "vol": float(vol),
            "return": float(ret),
            "leverage": float(leverage),
        })
    return cml


def _solve_mv_target(mu, cov, target, long_only, max_weight):
    """Solve min w'Σw s.t. μ'w = target, Σw = 1."""
    N = len(mu)

    def obj(w):
        return float(w @ cov @ w)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: mu @ w - target},
    ]
    bounds = [(0, max_weight)] * N if long_only else [(-max_weight, max_weight)] * N

    w0 = np.ones(N) / N
    result = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)

    return result.x if result.success else None
