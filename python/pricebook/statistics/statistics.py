"""Statistical toolkit: cointegration, regime detection, bootstrap, rolling analytics.

    from pricebook.statistics.statistics import (
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



    def to_dict(self) -> dict:
        return dict(vars(self))
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



    def to_dict(self) -> dict:
        return dict(vars(self))
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



    def to_dict(self) -> dict:
        return dict(vars(self))
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



    def to_dict(self) -> dict:
        return dict(vars(self))
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


# ============================================================================
# Time Series Diagnostics
# ============================================================================

def acf(x: np.ndarray, max_lag: int = 40) -> np.ndarray:
    """Autocorrelation function.

    Returns array of length max_lag+1 where acf[0] = 1.0.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    xm = x - x.mean()
    c0 = np.dot(xm, xm) / n
    if c0 < 1e-15:
        return np.zeros(min(max_lag + 1, n))
    lags = min(max_lag, n - 1)
    result = np.zeros(lags + 1)
    result[0] = 1.0
    for k in range(1, lags + 1):
        result[k] = np.dot(xm[:n - k], xm[k:]) / (n * c0)
    return result


def pacf(x: np.ndarray, max_lag: int = 40) -> np.ndarray:
    """Partial autocorrelation function via Levinson-Durbin recursion.

    Returns array of length max_lag+1 where pacf[0] = 1.0.
    """
    r = acf(x, max_lag)
    lags = len(r) - 1
    result = np.zeros(lags + 1)
    result[0] = 1.0
    if lags == 0:
        return result

    # Levinson-Durbin
    phi = np.zeros((lags + 1, lags + 1))
    phi[1, 1] = r[1]
    result[1] = r[1]

    for k in range(2, lags + 1):
        num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
        if abs(den) < 1e-15:
            break
        phi[k, k] = num / den
        result[k] = phi[k, k]
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

    return result


@dataclass
class LjungBoxResult:
    """Ljung-Box Q test result."""
    statistic: float
    p_value: float
    lags: int
    reject: bool           # True = significant serial correlation

    def to_dict(self) -> dict:
        return dict(vars(self))


def ljung_box(x: np.ndarray, lags: int = 20, significance: float = 0.05) -> LjungBoxResult:
    """Ljung-Box Q test for serial correlation.

    H0: no autocorrelation up to lag k.
    Q = n(n+2) Σ_{k=1}^{K} rho_k^2 / (n-k)

    Uses chi-squared critical value with K degrees of freedom.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    r = acf(x, lags)
    Q = n * (n + 2) * sum(r[k] ** 2 / (n - k) for k in range(1, min(lags + 1, len(r))))

    # Chi-squared CDF approximation (Wilson-Hilferty)
    k = lags
    z = ((Q / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    # Normal CDF approximation for p-value
    p_value = 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

    return LjungBoxResult(
        statistic=float(Q),
        p_value=max(0.0, min(1.0, p_value)),
        lags=lags,
        reject=p_value < significance,
    )


@dataclass
class ADFResult:
    """Augmented Dickey-Fuller test result."""
    statistic: float
    p_value: float         # approximate
    lags_used: int
    reject: bool           # True = reject unit root → series is stationary
    critical_values: dict  # 1%, 5%, 10%

    def to_dict(self) -> dict:
        return dict(vars(self))


def adf_test(x: np.ndarray, max_lag: int | None = None,
             significance: float = 0.05) -> ADFResult:
    """Augmented Dickey-Fuller unit root test.

    H0: unit root (non-stationary).
    Rejects when test statistic < critical value (more negative).

    Uses OLS regression: Δy_t = α + γ y_{t-1} + Σ β_i Δy_{t-i} + ε_t
    Test statistic = γ / se(γ).

    Critical values from MacKinnon (1994) for constant, no trend.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if max_lag is None:
        max_lag = int(np.floor((n - 1) ** (1 / 3)))

    dx = np.diff(x)
    # Build regression matrix: [1, y_{t-1}, Δy_{t-1}, ..., Δy_{t-p}]
    T = len(dx) - max_lag
    if T < 5:
        return ADFResult(0.0, 1.0, max_lag, False, {})

    Y = dx[max_lag:]
    X = np.ones((T, 2 + max_lag))
    X[:, 1] = x[max_lag:max_lag + T]  # y_{t-1}
    for j in range(max_lag):
        X[:, 2 + j] = dx[max_lag - 1 - j:max_lag - 1 - j + T]

    # OLS
    try:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return ADFResult(0.0, 1.0, max_lag, False, {})

    resid = Y - X @ beta
    sigma2 = float(np.dot(resid, resid) / max(T - X.shape[1], 1))
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return ADFResult(0.0, 1.0, max_lag, False, {})

    gamma = beta[1]
    se_gamma = math.sqrt(max(cov[1, 1], 1e-20))
    t_stat = gamma / se_gamma

    # MacKinnon critical values (constant, no trend, T → ∞)
    criticals = {"1%": -3.43, "5%": -2.86, "10%": -2.57}

    # Approximate p-value (linear interpolation of MacKinnon table)
    if t_stat < -3.43:
        p = 0.005
    elif t_stat < -2.86:
        p = 0.01 + (t_stat + 3.43) / (-2.86 + 3.43) * 0.04
    elif t_stat < -2.57:
        p = 0.05 + (t_stat + 2.86) / (-2.57 + 2.86) * 0.05
    elif t_stat < -1.94:
        p = 0.10 + (t_stat + 2.57) / (-1.94 + 2.57) * 0.15
    else:
        p = 0.25 + min(0.75, max(0, (t_stat + 1.94) / 3.0))

    return ADFResult(
        statistic=float(t_stat),
        p_value=float(p),
        lags_used=max_lag,
        reject=t_stat < criticals.get(f"{int(significance * 100)}%", -2.86),
        critical_values=criticals,
    )


def durbin_watson(residuals: np.ndarray) -> float:
    """Durbin-Watson statistic for first-order serial correlation in residuals.

    DW ≈ 2: no autocorrelation.
    DW < 2: positive autocorrelation.
    DW > 2: negative autocorrelation.
    """
    r = np.asarray(residuals, dtype=float)
    if len(r) < 2:
        return 2.0
    diff = np.diff(r)
    return float(np.dot(diff, diff) / np.dot(r, r))
