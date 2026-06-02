"""Robust portfolio optimisation: worst-case, uncertainty sets.

* :func:`robust_mean_variance` — worst-case mean-variance.
* :func:`ellipsoidal_uncertainty` — ellipsoidal return uncertainty.
* :func:`box_uncertainty` — box (interval) return uncertainty.

References:
    Goldfarb & Iyengar, *Robust Portfolio Selection Problems*, MOR, 2003.
    Ben-Tal & Nemirovski, *Robust Optimization*, Princeton, 2009.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class RobustPortfolioResult:
    """Robust portfolio result."""
    weights: np.ndarray
    worst_case_return: float
    nominal_return: float
    volatility: float
    uncertainty_radius: float
    method: str

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "worst_case_return": self.worst_case_return,
            "nominal_return": self.nominal_return,
            "volatility": self.volatility,
            "method": self.method,
        }


def robust_mean_variance(
    mu: np.ndarray,
    cov: np.ndarray,
    epsilon: float = 0.1,
    risk_aversion: float = 1.0,
    long_only: bool = True,
    uncertainty_type: str = "ellipsoidal",
) -> RobustPortfolioResult:
    """Worst-case mean-variance portfolio.

    max_w min_μ̃  μ̃'w − (λ/2) w'Σw
    s.t. ||μ̃ − μ|| ≤ ε  (uncertainty set)

    The inner min has closed form:
    worst-case μ̃'w = μ'w − ε × ||Σ^{1/2} w|| (ellipsoidal)
    worst-case μ̃'w = μ'w − ε × ||w||₁ (box)

    Args:
        mu: nominal expected returns.
        cov: covariance matrix.
        epsilon: uncertainty radius.
        risk_aversion: λ in mean-variance objective.
        uncertainty_type: "ellipsoidal" or "box".
    """
    N = len(mu)

    if uncertainty_type == "ellipsoidal":
        return ellipsoidal_uncertainty(mu, cov, epsilon, risk_aversion, long_only)
    else:
        return box_uncertainty(mu, cov, epsilon, risk_aversion, long_only)


def ellipsoidal_uncertainty(
    mu: np.ndarray,
    cov: np.ndarray,
    epsilon: float = 0.1,
    risk_aversion: float = 1.0,
    long_only: bool = True,
) -> RobustPortfolioResult:
    """Ellipsoidal uncertainty set: ||μ̃ − μ||_{Σ⁻¹} ≤ ε.

    Worst case: μ̃'w = μ'w − ε × √(w'Σw).
    Objective: max μ'w − ε√(w'Σw) − (λ/2)w'Σw.
    """
    N = len(mu)

    def neg_obj(w):
        port_var = float(w @ cov @ w)
        port_vol = math.sqrt(max(port_var, 1e-12))
        worst_ret = float(mu @ w) - epsilon * port_vol
        return -(worst_ret - 0.5 * risk_aversion * port_var)

    w0 = np.ones(N) / N
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * N if long_only else [(-1, 1)] * N

    result = minimize(neg_obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = result.x if result.success else w0

    vol = float(np.sqrt(w @ cov @ w))
    nom_ret = float(mu @ w)
    worst_ret = nom_ret - epsilon * vol

    return RobustPortfolioResult(w, worst_ret, nom_ret, vol, epsilon, "ellipsoidal")


def box_uncertainty(
    mu: np.ndarray,
    cov: np.ndarray,
    epsilon: float = 0.1,
    risk_aversion: float = 1.0,
    long_only: bool = True,
) -> RobustPortfolioResult:
    """Box uncertainty: |μ̃_i − μ_i| ≤ ε for each i.

    Worst case: μ̃'w = μ'w − ε × Σ|w_i| = μ'w − ε × ||w||₁.
    For long-only: ||w||₁ = 1, so worst = μ'w − ε.
    """
    N = len(mu)

    def neg_obj(w):
        port_var = float(w @ cov @ w)
        worst_ret = float(mu @ w) - epsilon * float(np.sum(np.abs(w)))
        return -(worst_ret - 0.5 * risk_aversion * port_var)

    w0 = np.ones(N) / N
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * N if long_only else [(-1, 1)] * N

    result = minimize(neg_obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = result.x if result.success else w0

    vol = float(np.sqrt(w @ cov @ w))
    nom_ret = float(mu @ w)
    worst_ret = nom_ret - epsilon * float(np.sum(np.abs(w)))

    return RobustPortfolioResult(w, worst_ret, nom_ret, vol, epsilon, "box")
