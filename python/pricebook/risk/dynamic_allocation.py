"""Multi-period dynamic allocation: CPPI, target-date, lifecycle.

* :func:`cppi_allocation` — constant proportion portfolio insurance.
* :func:`target_date_glide` — target-date fund glide path.
* :func:`multi_period_mv` — multi-period mean-variance.

References:
    Black & Jones, *Simplifying Portfolio Insurance*, JPM, 1987.
    Merton, *Optimum Consumption and Portfolio Rules*, JET, 1971.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CPPIResult:
    """CPPI simulation result."""
    portfolio_values: np.ndarray
    floor_values: np.ndarray
    equity_allocations: np.ndarray
    final_value: float
    max_drawdown: float
    floor_breached: bool

    def to_dict(self) -> dict:
        return {
            "final_value": self.final_value,
            "max_drawdown": self.max_drawdown,
            "floor_breached": self.floor_breached,
            "n_periods": len(self.portfolio_values),
        }


def cppi_allocation(
    initial_value: float,
    floor_pct: float = 0.80,
    multiplier: float = 5.0,
    risky_return: float = 0.08,
    risky_vol: float = 0.15,
    safe_rate: float = 0.03,
    n_periods: int = 252,
    seed: int = 42,
) -> CPPIResult:
    """CPPI: Constant Proportion Portfolio Insurance.

    Equity allocation = m × (V − F) / V,
    capped at 100%, floored at 0%.

    Where V = portfolio value, F = floor, m = multiplier.

    Args:
        floor_pct: floor as fraction of initial value.
        multiplier: CPPI multiplier (higher = more aggressive).
        risky_return: expected return of risky asset.
        risky_vol: volatility of risky asset.
        safe_rate: risk-free rate.
        n_periods: simulation periods.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / n_periods

    portfolio = np.zeros(n_periods + 1)
    floors = np.zeros(n_periods + 1)
    eq_alloc = np.zeros(n_periods + 1)

    portfolio[0] = initial_value
    floor = initial_value * floor_pct
    floors[0] = floor

    max_val = initial_value
    max_dd = 0.0
    breached = False

    for t in range(n_periods):
        V = portfolio[t]
        cushion = max(V - floor, 0)
        eq_weight = min(multiplier * cushion / V, 1.0) if V > 0 else 0
        eq_alloc[t] = eq_weight

        # Returns
        r_risky = (risky_return - 0.5 * risky_vol**2) * dt + risky_vol * math.sqrt(dt) * rng.standard_normal()
        r_safe = safe_rate * dt

        V_new = V * (eq_weight * math.exp(r_risky) + (1 - eq_weight) * math.exp(r_safe))
        portfolio[t + 1] = V_new

        # Floor grows at safe rate
        floor *= math.exp(r_safe)
        floors[t + 1] = floor

        # Drawdown
        max_val = max(max_val, V_new)
        dd = (max_val - V_new) / max_val
        max_dd = max(max_dd, dd)

        if V_new < floor:
            breached = True

    eq_alloc[-1] = eq_alloc[-2]

    return CPPIResult(
        portfolio_values=portfolio,
        floor_values=floors,
        equity_allocations=eq_alloc,
        final_value=float(portfolio[-1]),
        max_drawdown=max_dd,
        floor_breached=breached,
    )


@dataclass
class GlidePathResult:
    """Target-date glide path result."""
    equity_weights: list[float]
    bond_weights: list[float]
    years_to_retirement: list[float]

    def to_dict(self) -> dict:
        return {
            "n_points": len(self.equity_weights),
            "initial_equity": self.equity_weights[0],
            "final_equity": self.equity_weights[-1],
        }


def target_date_glide(
    years_to_retirement: float = 30.0,
    initial_equity: float = 0.90,
    final_equity: float = 0.30,
    glide_type: str = "linear",
) -> GlidePathResult:
    """Target-date fund glide path.

    Equity allocation decreases as retirement approaches.

    Args:
        years_to_retirement: total horizon.
        initial_equity: equity weight at start.
        final_equity: equity weight at retirement.
        glide_type: "linear", "convex" (slow then fast), "concave" (fast then slow).
    """
    years = np.arange(0, years_to_retirement + 1)
    n = len(years)

    if glide_type == "linear":
        eq = np.linspace(initial_equity, final_equity, n)
    elif glide_type == "convex":
        # Slow decrease initially, accelerates
        t = np.linspace(0, 1, n)
        eq = initial_equity + (final_equity - initial_equity) * t**2
    elif glide_type == "concave":
        # Fast decrease initially, decelerates
        t = np.linspace(0, 1, n)
        eq = initial_equity + (final_equity - initial_equity) * np.sqrt(t)
    else:
        eq = np.linspace(initial_equity, final_equity, n)

    bonds = 1.0 - eq

    return GlidePathResult(
        equity_weights=eq.tolist(),
        bond_weights=bonds.tolist(),
        years_to_retirement=(years_to_retirement - years).tolist(),
    )


def multi_period_mv(
    mu: np.ndarray,
    cov: np.ndarray,
    n_periods: int = 12,
    risk_aversion: float = 1.0,
    rebalance_cost_bps: float = 5.0,
) -> list[dict]:
    """Multi-period mean-variance with myopic rebalancing.

    At each period, solve single-period MV. Track cumulative
    return and risk accounting for rebalancing costs.

    Args:
        mu: expected returns (annual).
        cov: covariance matrix (annual).
        n_periods: number of rebalancing periods.
        rebalance_cost_bps: cost per rebalance in bps.
    """
    from pricebook.risk.portfolio_construction import mean_variance

    dt = 1.0 / n_periods
    mu_period = mu * dt
    cov_period = cov * dt
    tc = rebalance_cost_bps / 10_000

    periods = []
    prev_weights = np.ones(len(mu)) / len(mu)

    for t in range(n_periods):
        r = mean_variance(mu_period, cov_period, risk_aversion=risk_aversion)
        turnover = float(np.sum(np.abs(r.weights - prev_weights)))
        cost = tc * turnover

        periods.append({
            "period": t,
            "weights": r.weights.tolist(),
            "expected_return": float(mu_period @ r.weights),
            "volatility": float(np.sqrt(r.weights @ cov_period @ r.weights)),
            "turnover": turnover,
            "cost": cost,
        })
        prev_weights = r.weights

    return periods
