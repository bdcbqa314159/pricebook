"""Statistical toolkit: cointegration, regime detection, bootstrap, rolling analytics.

    from pricebook.statistics import (
        cointegration_test, regime_detect, bootstrap_ci, rolling_stats,
    )

References:
    Hamilton, *Time Series Analysis*, Princeton, 1994.
    Engle & Granger, *Co-Integration and Error Correction*, Econometrica, 1987.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ============================================================================
# Cointegration
# ============================================================================

@dataclass
class CointegrationResult:
    """Cointegration test result."""
    is_cointegrated: bool
    hedge_ratio: float       # β in y = α + β×x + ε
    spread_mean: float
    spread_std: float
    adf_statistic: float
    critical_value: float
    half_life: float         # mean reversion half-life in periods


def cointegration_test(
    y: np.ndarray,
    x: np.ndarray,
    significance: float = 0.05,
) -> CointegrationResult:
    """Engle-Granger two-step cointegration test.

    1. Regress y on x to get hedge ratio β
    2. Test residuals for stationarity (ADF)

        result = cointegration_test(spread_y, spread_x)
        if result.is_cointegrated:
            trade_the_spread(result.hedge_ratio)
    """
    n = len(y)
    # Step 1: OLS regression y = α + β×x
    X = np.column_stack([np.ones(n), x[:n]])
    coeffs = np.linalg.lstsq(X, y[:n], rcond=None)[0]
    alpha, beta = coeffs[0], coeffs[1]

    # Residuals (spread)
    spread = y[:n] - alpha - beta * x[:n]
    spread_mean = float(spread.mean())
    spread_std = float(spread.std())

    # Step 2: ADF test on residuals (simplified)
    # Δspread_t = γ × spread_{t-1} + ε_t
    # t-stat on γ; reject unit root if t < critical value
    ds = np.diff(spread)
    s_lag = spread[:-1]
    if len(s_lag) > 1 and s_lag.std() > 1e-10:
        gamma = float(np.sum(ds * s_lag) / np.sum(s_lag ** 2))
        se_gamma = float(np.sqrt(np.sum((ds - gamma * s_lag) ** 2) / (len(ds) * np.sum(s_lag ** 2))))
        adf = gamma / se_gamma if se_gamma > 1e-10 else 0.0
    else:
        gamma = 0.0
        adf = 0.0

    # Critical values (approximate, from MacKinnon)
    crit = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}
    cv = crit.get(significance, -2.86)

    is_coint = adf < cv

    # Half-life: HL = -ln(2) / ln(1 + γ)
    if gamma < 0:
        hl = -math.log(2) / math.log(1 + gamma) if abs(1 + gamma) > 1e-10 else float('inf')
    else:
        hl = float('inf')

    return CointegrationResult(is_coint, beta, spread_mean, spread_std, adf, cv, hl)


# ============================================================================
# Regime detection
# ============================================================================

@dataclass
class RegimeResult:
    """Regime detection result."""
    n_regimes: int
    current_regime: int
    regime_labels: np.ndarray     # regime at each time step
    regime_means: list[float]
    regime_vols: list[float]
    transition_matrix: np.ndarray | None


def regime_detect(
    returns: np.ndarray,
    n_regimes: int = 2,
    method: str = "threshold",
    vol_window: int = 20,
) -> RegimeResult:
    """Simple regime detection.

    Methods:
        - "threshold": low vol = regime 0, high vol = regime 1
        - "momentum": positive trend = regime 0, negative = regime 1

        regimes = regime_detect(returns, n_regimes=2)
    """
    n = len(returns)
    labels = np.zeros(n, dtype=int)

    if method == "threshold":
        # Rolling vol based
        for i in range(vol_window, n):
            vol = returns[i - vol_window:i].std()
            median_vol = np.median([returns[max(0, j - vol_window):j].std()
                                     for j in range(vol_window, n)])
            labels[i] = 0 if vol < median_vol else 1
    elif method == "momentum":
        for i in range(vol_window, n):
            cum = returns[i - vol_window:i].sum()
            labels[i] = 0 if cum > 0 else 1
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute regime statistics
    means = []
    vols = []
    for r in range(n_regimes):
        mask = labels == r
        if mask.sum() > 1:
            means.append(float(returns[mask].mean()))
            vols.append(float(returns[mask].std()))
        else:
            means.append(0.0)
            vols.append(0.0)

    return RegimeResult(n_regimes, int(labels[-1]), labels, means, vols, None)


# ============================================================================
# Bootstrap
# ============================================================================

@dataclass
class BootstrapCI:
    """Bootstrap confidence interval."""
    estimate: float
    lower: float
    upper: float
    confidence: float
    n_bootstrap: int


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap confidence interval for a statistic.

        ci = bootstrap_ci(returns, statistic="sharpe", confidence=0.95)
    """
    rng = np.random.default_rng(seed)
    n = len(data)

    def _stat(x):
        if statistic == "mean":
            return float(x.mean())
        elif statistic == "median":
            return float(np.median(x))
        elif statistic == "std":
            return float(x.std())
        elif statistic == "sharpe":
            mu = x.mean()
            sigma = x.std()
            return float(mu / sigma * math.sqrt(252)) if sigma > 1e-10 else 0.0
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

    point = _stat(data)
    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = _stat(sample)

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(boot_stats, alpha * 100))
    upper = float(np.percentile(boot_stats, (1 - alpha) * 100))

    return BootstrapCI(point, lower, upper, confidence, n_bootstrap)


# ============================================================================
# Rolling analytics
# ============================================================================

@dataclass
class RollingStatsResult:
    """Rolling statistics."""
    rolling_mean: np.ndarray
    rolling_vol: np.ndarray
    rolling_sharpe: np.ndarray
    rolling_skew: np.ndarray
    rolling_kurt: np.ndarray
    window: int


def rolling_stats(
    returns: np.ndarray,
    window: int = 60,
    annualise: int = 252,
) -> RollingStatsResult:
    """Compute rolling statistics over a window.

        stats = rolling_stats(daily_returns, window=60)
    """
    n = len(returns)
    r_mean = np.full(n, np.nan)
    r_vol = np.full(n, np.nan)
    r_sharpe = np.full(n, np.nan)
    r_skew = np.full(n, np.nan)
    r_kurt = np.full(n, np.nan)

    for i in range(window, n):
        w = returns[i - window:i]
        mu = w.mean()
        sigma = w.std()
        r_mean[i] = mu * annualise
        r_vol[i] = sigma * math.sqrt(annualise)
        r_sharpe[i] = mu / sigma * math.sqrt(annualise) if sigma > 1e-10 else 0.0

        # Skewness and kurtosis
        if sigma > 1e-10:
            z = (w - mu) / sigma
            r_skew[i] = float(np.mean(z ** 3))
            r_kurt[i] = float(np.mean(z ** 4))
        else:
            r_skew[i] = 0.0
            r_kurt[i] = 3.0

    return RollingStatsResult(r_mean, r_vol, r_sharpe, r_skew, r_kurt, window)
