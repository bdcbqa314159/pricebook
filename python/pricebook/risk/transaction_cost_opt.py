"""Transaction cost-aware portfolio optimisation.

* :func:`tc_aware_rebalance` — rebalance with turnover penalty.
* :func:`optimal_rebalance_frequency` — cost-benefit of rebalancing.
* :func:`no_trade_region` — Leland-Davis no-trade bands.

References:
    Davis & Norman, *Portfolio Selection with Transaction Costs*, MOR, 1990.
    Leland, *Optimal Portfolio Implementation*, JF, 2000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class TCRebalanceResult:
    """Transaction cost-aware rebalance result."""
    target_weights: np.ndarray
    actual_weights: np.ndarray
    turnover: float
    transaction_cost: float
    net_benefit: float          # expected improvement − cost

    def to_dict(self) -> dict:
        return {
            "turnover": self.turnover,
            "transaction_cost": self.transaction_cost,
            "net_benefit": self.net_benefit,
        }


def tc_aware_rebalance(
    mu: np.ndarray,
    cov: np.ndarray,
    current_weights: np.ndarray,
    tc_bps: float = 10.0,
    risk_aversion: float = 1.0,
    long_only: bool = True,
) -> TCRebalanceResult:
    """Rebalance with turnover penalty in objective.

    max μ'w − (λ/2)w'Σw − tc × ||w − w_current||₁

    The transaction cost penalty creates a no-trade region
    around the current portfolio.

    Args:
        mu: expected returns.
        cov: covariance matrix.
        current_weights: current portfolio weights.
        tc_bps: transaction cost in basis points (per unit turnover).
        risk_aversion: risk aversion parameter.
    """
    N = len(mu)
    tc = tc_bps / 10_000

    def neg_obj(w):
        ret = float(mu @ w)
        risk = 0.5 * risk_aversion * float(w @ cov @ w)
        cost = tc * float(np.sum(np.abs(w - current_weights)))
        return -(ret - risk - cost)

    w0 = current_weights.copy()
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * N if long_only else [(-1, 1)] * N

    result = minimize(neg_obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = result.x if result.success else current_weights

    turnover = float(np.sum(np.abs(w - current_weights)))
    cost = tc * turnover
    benefit_no_tc = float(mu @ w) - 0.5 * risk_aversion * float(w @ cov @ w)
    benefit_current = float(mu @ current_weights) - 0.5 * risk_aversion * float(current_weights @ cov @ current_weights)

    return TCRebalanceResult(
        target_weights=w,
        actual_weights=current_weights,
        turnover=turnover,
        transaction_cost=cost,
        net_benefit=benefit_no_tc - benefit_current - cost,
    )


def no_trade_region(
    mu: np.ndarray,
    cov: np.ndarray,
    optimal_weights: np.ndarray,
    tc_bps: float = 10.0,
    risk_aversion: float = 1.0,
) -> list[dict]:
    """Leland-Davis no-trade bands around optimal weights.

    No-trade band width ≈ (3/2 × tc / (λ × σ²))^{1/3}

    Within the band, transaction costs exceed the benefit of rebalancing.

    Args:
        optimal_weights: target (unconstrained) optimal weights.
        tc_bps: transaction cost in basis points.
    """
    N = len(mu)
    tc = tc_bps / 10_000
    diag_var = np.diag(cov)

    bands = []
    for i in range(N):
        sigma_i = math.sqrt(max(diag_var[i], 1e-10))
        # Approximate band width
        if risk_aversion * sigma_i**2 > 0:
            half_width = (1.5 * tc / (risk_aversion * sigma_i**2)) ** (1.0 / 3.0)
        else:
            half_width = 0.0
        half_width = min(half_width, 0.5)

        bands.append({
            "asset": i,
            "optimal": float(optimal_weights[i]),
            "lower": max(float(optimal_weights[i] - half_width), 0),
            "upper": min(float(optimal_weights[i] + half_width), 1),
            "half_width": half_width,
        })

    return bands


def optimal_rebalance_frequency(
    mu: np.ndarray,
    cov: np.ndarray,
    tc_bps: float = 10.0,
    risk_aversion: float = 1.0,
    vol_of_drift: float = 0.01,
) -> dict:
    """Optimal rebalancing frequency (cost vs benefit).

    Rebalancing benefit ∝ σ²_drift × Δt (variance of weight drift).
    Rebalancing cost = tc × expected_turnover.
    Optimal: Δt* ∝ (tc / σ²_drift)^{2/3}.

    Args:
        vol_of_drift: how fast weights drift per day.
    """
    tc = tc_bps / 10_000
    avg_var = float(np.mean(np.diag(cov)))

    # Optimal interval (days)
    if vol_of_drift > 0:
        optimal_days = (tc / (risk_aversion * vol_of_drift**2)) ** (2.0 / 3.0)
    else:
        optimal_days = 252  # annual if no drift

    optimal_days = max(1, min(optimal_days, 252))

    return {
        "optimal_days": optimal_days,
        "rebalances_per_year": 252 / optimal_days,
        "annual_tc_cost": tc * 252 / optimal_days * float(np.sum(np.sqrt(np.diag(cov)))),
        "tc_bps": tc_bps,
    }
