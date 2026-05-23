"""Maximum entropy option pricing — model-free risk-neutral density.

Recovers the risk-neutral density that maximises Shannon entropy subject
to reproducing observed option prices. No parametric model assumed.

    from pricebook.options.entropy_pricing import (
        max_entropy_density, entropy_implied_vol, MaxEntropyResult,
    )

References:
    Buchen & Kelly (1996). The Maximum Entropy Distribution of an Asset
    Inferred from Option Prices. JFQA.
    Stutzer (1996). A Simple Nonparametric Approach to Derivative Security
    Valuation. Journal of Finance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class MaxEntropyResult:
    """Result of maximum entropy density recovery."""
    grid: np.ndarray             # (M,) price grid
    density: np.ndarray          # (M,) risk-neutral density q(S)
    entropy: float               # Shannon entropy of the density
    forward: float               # risk-neutral forward price
    repricing_errors: list[float]  # model - market for each constraint
    n_constraints: int
    converged: bool

    def expected_value(self, payoff_fn: callable) -> float:
        """Price any payoff under the max-entropy density.

        E[payoff(S)] = Σ payoff(S_i) × q(S_i) × ΔS
        """
        ds = self.grid[1] - self.grid[0] if len(self.grid) > 1 else 1.0
        payoffs = np.array([payoff_fn(s) for s in self.grid])
        return float(np.sum(payoffs * self.density * ds))

    def call_price(self, strike: float, df: float = 1.0) -> float:
        """European call price under max-entropy density."""
        return df * self.expected_value(lambda s: max(s - strike, 0))

    def put_price(self, strike: float, df: float = 1.0) -> float:
        """European put price under max-entropy density."""
        return df * self.expected_value(lambda s: max(strike - s, 0))

    def implied_vol_at(self, strike: float, forward: float, T: float, df: float = 1.0) -> float:
        """Implied vol at a given strike from the max-entropy density."""
        from pricebook.core.solvers import brentq
        price = self.call_price(strike, df)
        intrinsic = max(df * (forward - strike), 0)
        if price <= intrinsic:
            return 0.0

        def objective(sigma):
            from scipy.stats import norm
            d1 = (math.log(forward / strike) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            bs = df * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
            return bs - price

        try:
            return brentq(objective, 0.001, 5.0)
        except ValueError:
            return 0.0

    def to_dict(self) -> dict:
        return {
            "entropy": self.entropy,
            "forward": self.forward,
            "n_constraints": self.n_constraints,
            "converged": self.converged,
            "repricing_errors": self.repricing_errors,
        }


def max_entropy_density(
    strikes: list[float],
    call_prices: list[float],
    forward: float,
    df: float = 1.0,
    n_grid: int = 200,
    grid_range: tuple[float, float] | None = None,
) -> MaxEntropyResult:
    """Recover the maximum entropy risk-neutral density from option prices.

    Solves: max H(q) = -Σ q(S) log q(S) ΔS
    subject to:
        Σ q(S) ΔS = 1                          (normalisation)
        Σ S × q(S) ΔS = forward                (forward constraint)
        Σ max(S-K_i, 0) × q(S) ΔS = C_i / df   (option prices)

    Uses Lagrange multipliers → exponential family density:
        q(S) ∝ exp(-λ₀ - λ₁S - Σ λᵢ max(S-Kᵢ, 0))

    Args:
        strikes: list of option strike prices.
        call_prices: list of observed call prices (undiscounted: C/df).
        forward: risk-neutral forward price.
        df: discount factor to maturity.
        n_grid: number of grid points for the density.
        grid_range: (S_min, S_max) for the grid.

    Returns:
        MaxEntropyResult with the recovered density.
    """
    if len(strikes) != len(call_prices):
        raise ValueError("strikes and call_prices must have same length")

    # Undiscounted call prices
    undiscounted = [c / df for c in call_prices]

    # Grid
    if grid_range is None:
        s_min = forward * 0.3
        s_max = forward * 2.5
    else:
        s_min, s_max = grid_range
    grid = np.linspace(s_min, s_max, n_grid)
    ds = grid[1] - grid[0]

    n_constraints = len(strikes)

    # Payoff matrix: (n_grid, n_constraints)
    payoffs = np.zeros((n_grid, n_constraints))
    for i, k in enumerate(strikes):
        payoffs[:, i] = np.maximum(grid - k, 0)

    # Dual formulation (Buchen-Kelly):
    # q*(S) = exp(λ₀ + λ₁S + Σ λᵢ payoff_i(S)) / Z(λ)
    # Minimise: L(λ) = log Z(λ) - λ₀ - λ₁F - Σ λᵢCᵢ
    # where Z = ∫ exp(λ₀ + λ₁S + Σ λᵢ payoff_i(S)) dS
    n_lambdas = 1 + n_constraints

    def neg_dual(lam):
        exponent = lam[0] * grid
        for i in range(n_constraints):
            exponent += lam[1 + i] * payoffs[:, i]
        max_e = exponent.max()
        log_Z = max_e + math.log(max(np.sum(np.exp(exponent - max_e)) * ds, 1e-300))
        return log_Z - lam[0] * forward - sum(
            lam[1 + i] * undiscounted[i] for i in range(n_constraints))

    def neg_dual_grad(lam):
        exponent = lam[0] * grid
        for i in range(n_constraints):
            exponent += lam[1 + i] * payoffs[:, i]
        max_e = exponent.max()
        q = np.exp(exponent - max_e)
        Z = max(np.sum(q) * ds, 1e-300)
        q_n = q / Z
        grad = np.zeros(n_lambdas)
        grad[0] = np.sum(grid * q_n) * ds - forward
        for i in range(n_constraints):
            grad[1 + i] = np.sum(payoffs[:, i] * q_n) * ds - undiscounted[i]
        return grad

    x0 = np.zeros(n_lambdas)
    result = minimize(neg_dual, x0, jac=neg_dual_grad, method="L-BFGS-B",
                      options={"maxiter": 1000, "ftol": 1e-14})

    lam = result.x
    exponent = lam[0] * grid
    for i in range(n_constraints):
        exponent += lam[1 + i] * payoffs[:, i]
    max_e = exponent.max()
    q = np.exp(exponent - max_e)
    Z = max(np.sum(q) * ds, 1e-300)
    q_norm = q / Z

    # Entropy
    mask = q_norm > 1e-300
    entropy = float(-np.sum(q_norm[mask] * np.log(q_norm[mask])) * ds)

    # Repricing errors
    errors = []
    for i in range(n_constraints):
        model_price = np.sum(payoffs[:, i] * q_norm) * ds
        errors.append(float(model_price - undiscounted[i]))

    # Model forward
    model_forward = float(np.sum(grid * q_norm) * ds)

    return MaxEntropyResult(
        grid=grid,
        density=q_norm,
        entropy=entropy,
        forward=model_forward,
        repricing_errors=errors,
        n_constraints=n_constraints,
        converged=result.success,
    )


def entropy_implied_vol(
    strikes: list[float],
    call_prices: list[float],
    forward: float,
    T: float,
    df: float = 1.0,
    query_strikes: list[float] | None = None,
) -> list[dict]:
    """Extract implied vol smile from max-entropy density.

    Args:
        strikes: observed option strikes.
        call_prices: observed call prices.
        forward: risk-neutral forward.
        T: time to expiry.
        df: discount factor.
        query_strikes: strikes at which to compute implied vol.

    Returns list of {strike, implied_vol, call_price}.
    """
    result = max_entropy_density(strikes, call_prices, forward, df)

    if query_strikes is None:
        query_strikes = np.linspace(forward * 0.7, forward * 1.3, 20).tolist()

    smile = []
    for k in query_strikes:
        price = result.call_price(k, df)
        iv = result.implied_vol_at(k, forward, T, df)
        smile.append({"strike": k, "implied_vol": iv, "call_price": price})

    return smile
