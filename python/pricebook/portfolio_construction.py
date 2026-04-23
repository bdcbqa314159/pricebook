"""Portfolio construction: mean-variance, Black-Litterman, risk parity, rebalancing.

    from pricebook.portfolio_construction import (
        mean_variance, black_litterman, risk_parity, rebalance,
    )

References:
    Meucci, *Risk and Asset Allocation*, Springer, 2005.
    Black & Litterman, *Global Portfolio Optimization*, FAJ, 1992.
    Roncalli, *Introduction to Risk Parity and Budgeting*, CRC, 2013.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ============================================================================
# Mean-variance
# ============================================================================

@dataclass
class PortfolioOptResult:
    """Portfolio optimisation result."""
    weights: np.ndarray
    names: list[str]
    expected_return: float
    expected_vol: float
    sharpe: float
    method: str


def mean_variance(
    expected_returns: np.ndarray | list[float],
    covariance: np.ndarray,
    names: list[str] | None = None,
    risk_free: float = 0.0,
    long_only: bool = True,
    target_return: float | None = None,
    max_weight: float = 1.0,
) -> PortfolioOptResult:
    """Mean-variance optimisation (max Sharpe or target return).

        result = mean_variance(mu, cov, names=["SPX", "UST", "GOLD"])
    """
    mu = np.asarray(expected_returns, dtype=float)
    n = len(mu)
    names = names or [f"asset_{i}" for i in range(n)]

    # Analytical max-Sharpe: w* = Σ^{-1} (μ - rf) / 1'Σ^{-1}(μ - rf)
    try:
        inv_cov = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(covariance)

    excess = mu - risk_free
    raw = inv_cov @ excess
    w = raw / raw.sum() if abs(raw.sum()) > 1e-10 else np.ones(n) / n

    if long_only:
        w = np.maximum(w, 0.0)
        w_sum = w.sum()
        if w_sum > 1e-10:
            w = w / w_sum

    w = np.clip(w, -max_weight, max_weight)
    w_sum = w.sum()
    if abs(w_sum) > 1e-10:
        w = w / w_sum

    ret = float(w @ mu)
    vol = float(math.sqrt(w @ covariance @ w))
    sr = (ret - risk_free) / vol if vol > 1e-10 else 0.0

    return PortfolioOptResult(w, names, ret, vol, sr, "mean_variance")


# ============================================================================
# Black-Litterman
# ============================================================================

@dataclass
class BlackLittermanResult:
    """Black-Litterman posterior."""
    posterior_returns: np.ndarray
    posterior_covariance: np.ndarray
    optimal_weights: np.ndarray
    names: list[str]
    method: str


def black_litterman(
    market_weights: np.ndarray,
    covariance: np.ndarray,
    views: np.ndarray,
    view_confidences: np.ndarray,
    P: np.ndarray,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    names: list[str] | None = None,
) -> BlackLittermanResult:
    """Black-Litterman model: combine market equilibrium with investor views.

    Args:
        market_weights: equilibrium (market-cap) weights.
        covariance: asset covariance matrix.
        views: Q vector — expected returns for each view.
        view_confidences: diagonal of Ω (view uncertainty).
        P: pick matrix — maps views to assets.
        risk_aversion: λ (market risk aversion, typically 2-3).
        tau: scaling factor for prior uncertainty (typically 0.01-0.05).

        result = black_litterman(mkt_w, cov, Q, omega_diag, P)
    """
    n = len(market_weights)
    names = names or [f"asset_{i}" for i in range(n)]

    # Equilibrium excess returns: π = λ Σ w_mkt
    pi = risk_aversion * covariance @ market_weights

    # View uncertainty matrix
    Omega = np.diag(view_confidences)

    # Posterior: μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} × [(τΣ)^{-1}π + P'Ω^{-1}Q]
    tau_sigma = tau * covariance
    try:
        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(Omega)
    except np.linalg.LinAlgError:
        inv_tau_sigma = np.linalg.pinv(tau_sigma)
        inv_omega = np.linalg.pinv(Omega)

    M = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
    posterior_mu = M @ (inv_tau_sigma @ pi + P.T @ inv_omega @ views)
    posterior_cov = covariance + M

    # Optimal weights from posterior
    opt = mean_variance(posterior_mu, posterior_cov, names)

    return BlackLittermanResult(
        posterior_mu, posterior_cov, opt.weights, names, "black_litterman",
    )


# ============================================================================
# Risk parity
# ============================================================================

def risk_parity(
    covariance: np.ndarray,
    risk_budgets: np.ndarray | None = None,
    names: list[str] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> PortfolioOptResult:
    """Risk parity: equalise risk contribution from each asset.

    Each asset contributes equally to portfolio risk:
        RC_i = w_i × (Σw)_i / σ_p = budget_i

        result = risk_parity(cov, names=["stocks", "bonds", "commodities"])
    """
    n = covariance.shape[0]
    names = names or [f"asset_{i}" for i in range(n)]
    budgets = risk_budgets if risk_budgets is not None else np.ones(n) / n

    # Newton iteration on w_i ∝ budget_i / (Σw)_i
    w = np.ones(n) / n

    for _ in range(max_iter):
        sigma_w = covariance @ w
        port_vol = math.sqrt(float(w @ sigma_w))
        if port_vol < 1e-15:
            break

        rc = w * sigma_w / port_vol
        target_rc = budgets * port_vol

        # Update weights proportional to budget / marginal risk
        marginal = sigma_w / port_vol
        w_new = budgets / marginal
        w_new = w_new / w_new.sum()

        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    port_vol = math.sqrt(float(w @ covariance @ w))

    return PortfolioOptResult(w, names, 0.0, port_vol, 0.0, "risk_parity")


# ============================================================================
# Rebalancing
# ============================================================================

@dataclass
class RebalanceResult:
    """Rebalancing decision."""
    should_rebalance: bool
    current_weights: np.ndarray
    target_weights: np.ndarray
    trades: np.ndarray            # target - current (positive = buy)
    turnover: float               # sum of absolute trades
    reason: str


def rebalance(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    threshold: float = 0.05,
    min_trade: float = 0.01,
    names: list[str] | None = None,
) -> RebalanceResult:
    """Decide whether to rebalance and compute trades.

    Args:
        threshold: max allowed drift before rebalancing.
        min_trade: minimum trade size (ignore smaller drifts).

        result = rebalance(current, target, threshold=0.05)
        if result.should_rebalance:
            execute(result.trades)
    """
    current = np.asarray(current_weights, dtype=float)
    target = np.asarray(target_weights, dtype=float)
    drift = np.abs(current - target)
    max_drift = float(drift.max())

    should = max_drift > threshold
    trades = target - current
    # Zero out tiny trades
    trades[np.abs(trades) < min_trade] = 0.0
    turnover = float(np.abs(trades).sum())

    reason = f"max drift {max_drift:.1%} > threshold {threshold:.1%}" if should else "within threshold"

    return RebalanceResult(should, current, target, trades, turnover, reason)
