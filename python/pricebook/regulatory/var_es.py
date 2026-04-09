"""VaR and Expected Shortfall engine with backtesting.

Parametric (normal + t-distribution), historical simulation, and
Monte Carlo VaR. Portfolio VaR with component/marginal decomposition.
Basel traffic light backtesting with Kupiec test.

    from pricebook.regulatory.var_es import (
        parametric_var, parametric_es, historical_var, historical_es,
        monte_carlo_var, portfolio_var, backtest_var, quick_var,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import norm, t as t_dist, chi2


# ---- Configuration ----

class VaRMethod(Enum):
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"


def scale_var(var_1day: float, horizon_days: int, method: str = "sqrt") -> float:
    """Scale 1-day VaR to N-day: sqrt-of-time or linear."""
    if method == "sqrt":
        return var_1day * math.sqrt(horizon_days)
    return var_1day * horizon_days


def get_z_score(confidence: float, distribution: str = "normal", df: int = 5) -> float:
    """Critical value for confidence level."""
    if distribution == "normal":
        return float(norm.ppf(confidence))
    return float(t_dist.ppf(confidence, df))


# ---- Parametric VaR ----

def parametric_var(
    returns: list | np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    distribution: str = "normal",
    df: int = 5,
    position_value: float | None = None,
) -> dict:
    """Parametric VaR: VaR = -μ + σ × z_α."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        raise ValueError("Need at least 2 returns")

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    z = get_z_score(confidence, distribution, df)

    var_1day = -mu + z * sigma
    var_pct = scale_var(var_1day, horizon_days)
    var_abs = var_pct * position_value if position_value else None

    return {
        "method": "parametric", "confidence": confidence,
        "horizon_days": horizon_days, "distribution": distribution,
        "var_pct": var_pct, "var_1day_pct": var_1day, "var_abs": var_abs,
        "mean_return": mu, "volatility": sigma,
        "annualized_volatility": sigma * math.sqrt(252),
        "z_score": z, "num_observations": len(r),
    }


def parametric_es(
    returns: list | np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    distribution: str = "normal",
    df: int = 5,
    position_value: float | None = None,
) -> dict:
    """Parametric Expected Shortfall (CVaR)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    alpha = 1 - confidence

    if distribution == "normal":
        z = float(norm.ppf(confidence))
        es_1day = -mu + sigma * float(norm.pdf(z)) / alpha
    elif distribution == "t":
        z = float(t_dist.ppf(confidence, df))
        es_1day = -mu + sigma * float(t_dist.pdf(z, df)) * (df + z ** 2) / ((df - 1) * alpha)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    es_pct = scale_var(es_1day, horizon_days)
    es_abs = es_pct * position_value if position_value else None

    return {
        "method": "parametric", "confidence": confidence,
        "horizon_days": horizon_days, "distribution": distribution,
        "es_pct": es_pct, "es_1day_pct": es_1day, "es_abs": es_abs,
        "mean_return": mu, "volatility": sigma, "num_observations": len(r),
    }


# ---- Historical VaR ----

def historical_var(
    returns: list | np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    position_value: float | None = None,
) -> dict:
    """Historical simulation VaR."""
    r = np.sort(np.asarray(returns, dtype=float))
    r = r[~np.isnan(r)]
    if len(r) < 10:
        raise ValueError("Need at least 10 returns")

    idx = int((1 - confidence) * len(r))
    var_1day = float(-r[idx])
    var_pct = scale_var(var_1day, horizon_days)
    var_abs = var_pct * position_value if position_value else None

    return {
        "method": "historical", "confidence": confidence,
        "horizon_days": horizon_days,
        "var_pct": var_pct, "var_1day_pct": var_1day, "var_abs": var_abs,
        "mean_return": float(np.mean(r)), "volatility": float(np.std(r, ddof=1)),
        "min_return": float(r[0]), "max_return": float(r[-1]),
        "num_observations": len(r),
    }


def historical_es(
    returns: list | np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    position_value: float | None = None,
) -> dict:
    """Historical Expected Shortfall."""
    r = np.sort(np.asarray(returns, dtype=float))
    r = r[~np.isnan(r)]
    idx = int((1 - confidence) * len(r))
    tail = r[:idx + 1]

    es_1day = float(-np.mean(tail))
    es_pct = scale_var(es_1day, horizon_days)
    es_abs = es_pct * position_value if position_value else None

    return {
        "method": "historical", "confidence": confidence,
        "horizon_days": horizon_days,
        "es_pct": es_pct, "es_1day_pct": es_1day, "es_abs": es_abs,
        "var_pct": float(-r[idx]),
        "tail_observations": len(tail), "num_observations": len(r),
    }


# ---- Monte Carlo VaR ----

def monte_carlo_var(
    mean_return: float,
    volatility: float,
    confidence: float = 0.99,
    horizon_days: int = 1,
    num_simulations: int = 10_000,
    distribution: str = "normal",
    df: int = 5,
    position_value: float | None = None,
    seed: int = 42,
) -> dict:
    """Monte Carlo VaR."""
    rng = np.random.default_rng(seed)
    h = horizon_days

    if distribution == "normal":
        sims = rng.normal(mean_return * h, volatility * math.sqrt(h), num_simulations)
    elif distribution == "t":
        sims = mean_return * h + volatility * math.sqrt(h) * rng.standard_t(df, num_simulations) * math.sqrt((df - 2) / df)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    sorted_sims = np.sort(sims)
    idx = int((1 - confidence) * num_simulations)
    var_pct = float(-sorted_sims[idx])
    es_pct = float(-np.mean(sorted_sims[:idx + 1]))

    var_abs = var_pct * position_value if position_value else None
    es_abs = es_pct * position_value if position_value else None

    return {
        "method": "monte_carlo", "confidence": confidence,
        "horizon_days": horizon_days, "distribution": distribution,
        "var_pct": var_pct, "var_abs": var_abs,
        "es_pct": es_pct, "es_abs": es_abs,
        "mean_return": mean_return, "volatility": volatility,
        "num_simulations": num_simulations,
    }


# ---- Portfolio VaR ----

def portfolio_var(
    weights: list | np.ndarray,
    returns_matrix: np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    method: str = "parametric",
    position_value: float | None = None,
) -> dict:
    """Portfolio VaR with component and marginal decomposition."""
    w = np.asarray(weights, dtype=float)
    R = np.asarray(returns_matrix, dtype=float)
    n_assets = len(w)
    port_ret = R @ w

    if method == "parametric":
        var_r = parametric_var(port_ret, confidence, horizon_days, position_value=position_value)
        es_r = parametric_es(port_ret, confidence, horizon_days, position_value=position_value)
    else:
        var_r = historical_var(port_ret, confidence, horizon_days, position_value=position_value)
        es_r = historical_es(port_ret, confidence, horizon_days, position_value=position_value)

    cov = np.atleast_2d(np.cov(R, rowvar=False))
    port_std = math.sqrt(float(w @ cov @ w))
    z = get_z_score(confidence)
    marginal = (cov @ w) * z / port_std
    component = w * marginal
    total_comp = float(np.sum(component))
    pct_contrib = (component / total_comp * 100).tolist() if total_comp > 0 else [0.0] * n_assets

    standalone = []
    for i in range(n_assets):
        if method == "parametric":
            sv = parametric_var(R[:, i], confidence, horizon_days)
        else:
            sv = historical_var(R[:, i], confidence, horizon_days)
        standalone.append(sv["var_pct"])

    div_benefit = sum(wi * sv for wi, sv in zip(w, standalone)) - var_r["var_pct"]

    return {
        "method": method, "confidence": confidence, "horizon_days": horizon_days,
        "portfolio_var_pct": var_r["var_pct"], "portfolio_var_abs": var_r["var_abs"],
        "portfolio_es_pct": es_r["es_pct"], "portfolio_es_abs": es_r["es_abs"],
        "portfolio_volatility": port_std,
        "marginal_var": marginal.tolist(), "component_var": component.tolist(),
        "pct_contribution": pct_contrib, "standalone_var": standalone,
        "diversification_benefit": div_benefit,
        "n_assets": n_assets, "n_observations": len(port_ret),
    }


# ---- Backtesting ----

class BacktestZone(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


BACKTESTING_PLUS_FACTORS = {
    0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00,
    5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85,
}


def backtest_var(
    returns: list | np.ndarray,
    var_estimates: list | np.ndarray,
    confidence: float = 0.99,
) -> dict:
    """Backtest VaR: traffic light zones, plus factor, Kupiec test."""
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var_estimates, dtype=float)
    if len(r) != len(v):
        raise ValueError("Returns and VaR estimates must have same length")

    n = len(r)
    exceptions = r < -v
    n_exc = int(np.sum(exceptions))
    exc_rate = n_exc / n
    expected = (1 - confidence) * n

    if n_exc <= 4:
        zone = BacktestZone.GREEN
    elif n_exc <= 9:
        zone = BacktestZone.YELLOW
    else:
        zone = BacktestZone.RED

    plus_factor = BACKTESTING_PLUS_FACTORS.get(n_exc, 1.0)

    # Kupiec POF test
    lr_pof = None
    p_value = None
    if 0 < n_exc < n:
        alpha = 1 - confidence
        lr_pof = 2 * (
            n_exc * math.log(exc_rate / alpha)
            + (n - n_exc) * math.log((1 - exc_rate) / (1 - alpha))
        )
        p_value = float(1 - chi2.cdf(lr_pof, 1))

    return {
        "n_observations": n, "n_exceptions": n_exc,
        "exception_rate_pct": exc_rate * 100, "expected_exceptions": expected,
        "zone": zone.value, "plus_factor": plus_factor,
        "kupiec_lr_statistic": lr_pof, "kupiec_p_value": p_value,
        "exception_indices": np.where(exceptions)[0].tolist(),
        "confidence": confidence,
    }


# ---- Convenience ----

def quick_var(
    returns: list | np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    method: str = "parametric",
    position_value: float | None = None,
) -> dict:
    """Quick VaR and ES in one call."""
    r = np.asarray(returns, dtype=float)
    if method == "parametric":
        var_r = parametric_var(r, confidence, horizon_days, position_value=position_value)
        es_r = parametric_es(r, confidence, horizon_days, position_value=position_value)
    elif method == "historical":
        var_r = historical_var(r, confidence, horizon_days, position_value=position_value)
        es_r = historical_es(r, confidence, horizon_days, position_value=position_value)
    elif method == "monte_carlo":
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))
        return monte_carlo_var(mu, sigma, confidence, horizon_days, position_value=position_value)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "method": method, "confidence": confidence, "horizon_days": horizon_days,
        "var_pct": var_r["var_pct"], "var_abs": var_r.get("var_abs"),
        "es_pct": es_r["es_pct"], "es_abs": es_r.get("es_abs"),
        "volatility": var_r["volatility"],
        "annualized_volatility": var_r["volatility"] * math.sqrt(252),
        "mean_return": var_r["mean_return"],
        "num_observations": var_r["num_observations"],
    }


def compare_var_methods(
    returns: list | np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    position_value: float | None = None,
) -> dict:
    """Compare VaR across all three methods."""
    r = np.asarray(returns, dtype=float)
    results = {m: quick_var(r, confidence, horizon_days, m, position_value)
               for m in ["parametric", "historical", "monte_carlo"]}
    var_vals = {m: r["var_pct"] for m, r in results.items()}
    es_vals = {m: r["es_pct"] for m, r in results.items()}
    return {
        "confidence": confidence, "horizon_days": horizon_days,
        "methods": results,
        "var_comparison": var_vals, "es_comparison": es_vals,
        "var_range": (min(var_vals.values()), max(var_vals.values())),
        "es_range": (min(es_vals.values()), max(es_vals.values())),
    }
