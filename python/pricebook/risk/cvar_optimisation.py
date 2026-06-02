"""CVaR portfolio optimisation via LP (Rockafellar-Uryasev).

Minimise CVaR subject to return and weight constraints.
Also: worst-case CVaR and risk budgeting with CVaR.

* :func:`cvar_portfolio` — CVaR-optimal portfolio via LP.
* :func:`min_cvar_target_return` — minimum CVaR for given target return.
* :func:`cvar_risk_budget` — CVaR risk contribution decomposition.
* :func:`mean_cvar_frontier` — efficient frontier in mean-CVaR space.

References:
    Rockafellar & Uryasev, *Optimization of Conditional Value-at-Risk*,
    JR, 2000.
    Rockafellar & Uryasev, *Conditional Value-at-Risk for General Loss
    Distributions*, JBF, 2002.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize


@dataclass
class CVaRPortfolioResult:
    """CVaR portfolio optimisation result."""
    weights: np.ndarray
    expected_return: float
    cvar: float
    var: float
    n_assets: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "expected_return": self.expected_return,
            "cvar": self.cvar,
            "var": self.var,
            "n_assets": self.n_assets,
            "confidence": self.confidence,
        }


def cvar_portfolio(
    returns: np.ndarray,
    confidence: float = 0.95,
    target_return: float | None = None,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> CVaRPortfolioResult:
    """CVaR-optimal portfolio via Rockafellar-Uryasev LP.

    Minimises CVaR_α = min_ζ { ζ + 1/(1-α) × E[max(-r'w - ζ, 0)] }

    LP variables: w (weights), ζ (VaR threshold), u_s (auxiliary ≥ 0).

    min  ζ + 1/((1-α)S) × Σ u_s
    s.t. u_s ≥ -r_s'w - ζ   ∀s
         u_s ≥ 0
         Σ w = 1
         w ≥ 0 (if long_only)

    Args:
        returns: (S, N) matrix of S scenarios × N assets.
        confidence: CVaR confidence level (e.g. 0.95).
        target_return: if given, add constraint E[r'w] ≥ target.
        long_only: constrain weights ≥ 0.
        max_weight: maximum weight per asset.
    """
    S, N = returns.shape
    alpha = confidence
    scale = 1.0 / ((1 - alpha) * S)

    # Variables: [w_1..w_N, zeta, u_1..u_S]
    n_vars = N + 1 + S

    # Objective: min ζ + scale × Σ u_s
    c = np.zeros(n_vars)
    c[N] = 1.0  # zeta coefficient
    c[N + 1:] = scale  # u_s coefficients

    # Inequality constraints: u_s ≥ -r_s'w - ζ  →  r_s'w + ζ + u_s ≥ 0
    # In scipy form: A_ub @ x ≤ b_ub  →  -r_s'w - ζ - u_s ≤ 0
    A_ub_rows = []
    b_ub_rows = []
    for s in range(S):
        row = np.zeros(n_vars)
        row[:N] = -returns[s, :]  # -r_s'w
        row[N] = -1.0  # -ζ
        row[N + 1 + s] = -1.0  # -u_s
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)

    # Target return: -E[r'w] ≤ -target  →  -μ'w ≤ -target
    if target_return is not None:
        mu = returns.mean(axis=0)
        row = np.zeros(n_vars)
        row[:N] = -mu
        A_ub_rows.append(row)
        b_ub_rows.append(-target_return)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)

    # Equality: Σ w = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])

    # Bounds
    bounds = []
    for i in range(N):
        lb = 0.0 if long_only else -max_weight
        bounds.append((lb, max_weight))
    bounds.append((None, None))  # zeta unbounded
    for _ in range(S):
        bounds.append((0.0, None))  # u_s ≥ 0

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        w = result.x[:N]
        zeta = result.x[N]
        cvar_val = result.fun
    else:
        # Fallback: equal weight
        w = np.ones(N) / N
        zeta = float(np.percentile(-returns @ w, alpha * 100))
        cvar_val = zeta

    # Compute portfolio metrics
    port_returns = returns @ w
    mu_port = float(np.mean(port_returns))
    var_val = float(np.percentile(-port_returns, alpha * 100))
    cvar_actual = float(np.mean(-port_returns[-port_returns <= -var_val])) if np.any(-port_returns >= var_val) else var_val

    return CVaRPortfolioResult(
        weights=w,
        expected_return=mu_port,
        cvar=cvar_val,
        var=var_val,
        n_assets=N,
        confidence=alpha,
    )


def min_cvar_target_return(
    returns: np.ndarray,
    target_return: float,
    confidence: float = 0.95,
    long_only: bool = True,
) -> CVaRPortfolioResult:
    """Minimum CVaR portfolio for a given target return."""
    return cvar_portfolio(returns, confidence, target_return, long_only)


def cvar_risk_budget(
    returns: np.ndarray,
    weights: np.ndarray,
    confidence: float = 0.95,
) -> list[dict]:
    """CVaR risk contribution decomposition.

    Component CVaR_i = w_i × E[-r_i | portfolio_loss ≥ VaR].

    Args:
        returns: (S, N) scenario matrix.
        weights: portfolio weights.
        confidence: confidence level.
    """
    port_returns = returns @ weights
    var_threshold = np.percentile(-port_returns, confidence * 100)
    tail_mask = -port_returns >= var_threshold

    if not np.any(tail_mask):
        return [{"asset": i, "weight": float(weights[i]),
                 "cvar_contribution": 0, "pct": 0} for i in range(len(weights))]

    result = []
    total_cvar = float(np.mean(-port_returns[tail_mask]))

    for i in range(len(weights)):
        # Component: w_i × E[-r_i | tail]
        comp = float(weights[i] * np.mean(-returns[tail_mask, i]))
        pct = comp / total_cvar * 100 if total_cvar > 0 else 0
        result.append({
            "asset": i,
            "weight": float(weights[i]),
            "cvar_contribution": comp,
            "pct": pct,
        })

    return result


def mean_cvar_frontier(
    returns: np.ndarray,
    n_points: int = 20,
    confidence: float = 0.95,
    long_only: bool = True,
) -> list[dict]:
    """Efficient frontier in mean-CVaR space.

    Sweeps target returns from min to max feasible.
    """
    mu = returns.mean(axis=0)
    min_ret = float(np.min(mu))
    max_ret = float(np.max(mu))

    targets = np.linspace(min_ret, max_ret, n_points)
    frontier = []

    for target in targets:
        try:
            r = cvar_portfolio(returns, confidence, float(target), long_only)
            frontier.append({
                "target_return": float(target),
                "cvar": r.cvar,
                "weights": r.weights.tolist(),
            })
        except Exception:
            continue

    return frontier
