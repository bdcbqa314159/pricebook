"""Distribution fitting: MLE, goodness-of-fit tests, Q-Q plot data.

    from pricebook.statistics.distribution_fit import (
        fit_normal, fit_student_t, fit_gev,
        ks_test, anderson_darling, qq_data,
    )

References:
    Coles (2001). An Introduction to Statistical Modeling of Extreme Values.
    D'Agostino & Stephens (1986). Goodness-of-Fit Techniques.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Fit results
# ═══════════════════════════════════════════════════════════════

@dataclass
class FitResult:
    """Distribution fit result."""
    distribution: str
    params: dict[str, float]
    log_likelihood: float
    aic: float              # Akaike Information Criterion
    n_obs: int

    def to_dict(self) -> dict:
        return dict(vars(self))


# ═══════════════════════════════════════════════════════════════
# MLE fitting
# ═══════════════════════════════════════════════════════════════

def fit_normal(data: np.ndarray) -> FitResult:
    """MLE fit of normal distribution: mu, sigma."""
    x = np.asarray(data, dtype=float)
    n = len(x)
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=0))  # MLE uses ddof=0
    ll = -0.5 * n * (math.log(2 * math.pi) + 2 * math.log(max(sigma, 1e-15))) - \
         0.5 * np.sum((x - mu) ** 2) / max(sigma ** 2, 1e-30)
    return FitResult("normal", {"mu": mu, "sigma": sigma}, float(ll),
                     float(-2 * ll + 2 * 2), n)


def fit_student_t(data: np.ndarray) -> FitResult:
    """MLE fit of Student-t distribution: mu, sigma, nu.

    Uses grid search on nu (degrees of freedom) then analytical mu/sigma.
    """
    from scipy.stats import t as t_dist
    from scipy.optimize import minimize_scalar

    x = np.asarray(data, dtype=float)
    n = len(x)

    def neg_ll(nu):
        if nu <= 2:
            return 1e10
        mu = np.mean(x)
        sigma = np.std(x) * math.sqrt((nu - 2) / nu)
        sigma = max(sigma, 1e-10)
        return -float(np.sum(t_dist.logpdf(x, df=nu, loc=mu, scale=sigma)))

    result = minimize_scalar(neg_ll, bounds=(2.1, 100), method="bounded")
    nu = result.x
    mu = float(np.mean(x))
    sigma = float(np.std(x) * math.sqrt(max(nu - 2, 0.1) / nu))
    ll = -result.fun
    return FitResult("student_t", {"mu": mu, "sigma": sigma, "nu": float(nu)},
                     float(ll), float(-2 * ll + 2 * 3), n)


def fit_gev(data: np.ndarray) -> FitResult:
    """MLE fit of Generalised Extreme Value distribution: mu, sigma, xi.

    GEV: F(x) = exp(-(1 + xi*(x-mu)/sigma)^{-1/xi})
    xi > 0: Fréchet (heavy tail), xi < 0: Weibull (bounded), xi = 0: Gumbel.
    """
    from scipy.stats import genextreme
    xi, mu, sigma = genextreme.fit(data)
    ll = float(np.sum(genextreme.logpdf(data, xi, loc=mu, scale=sigma)))
    return FitResult("gev", {"mu": float(mu), "sigma": float(sigma), "xi": float(-xi)},
                     float(ll), float(-2 * ll + 2 * 3), len(data))


# ═══════════════════════════════════════════════════════════════
# Goodness-of-fit tests
# ═══════════════════════════════════════════════════════════════

@dataclass
class KSResult:
    """Kolmogorov-Smirnov test result."""
    statistic: float
    p_value: float
    reject: bool
    n_obs: int

    def to_dict(self) -> dict:
        return dict(vars(self))


def ks_test(
    data: np.ndarray,
    distribution: str = "normal",
    params: dict | None = None,
    significance: float = 0.05,
) -> KSResult:
    """Kolmogorov-Smirnov goodness-of-fit test.

    H0: data comes from the specified distribution.

    Args:
        distribution: 'normal', 'student_t', or 'uniform'.
        params: distribution parameters (if None, fit from data).
    """
    from scipy import stats

    x = np.asarray(data, dtype=float)
    n = len(x)

    if distribution == "normal":
        if params is None:
            mu, sigma = np.mean(x), np.std(x)
        else:
            mu, sigma = params.get("mu", 0), params.get("sigma", 1)
        stat, p = stats.kstest(x, "norm", args=(mu, sigma))
    elif distribution == "student_t":
        if params is None:
            r = fit_student_t(x)
            nu, mu, sigma = r.params["nu"], r.params["mu"], r.params["sigma"]
        else:
            nu, mu, sigma = params["nu"], params.get("mu", 0), params.get("sigma", 1)
        stat, p = stats.kstest(x, "t", args=(nu, mu, sigma))
    elif distribution == "uniform":
        stat, p = stats.kstest(x, "uniform", args=(x.min(), x.max() - x.min()))
    else:
        raise ValueError(f"unsupported distribution: {distribution!r}")

    return KSResult(float(stat), float(p), p < significance, n)


@dataclass
class ADResult:
    """Anderson-Darling test result."""
    statistic: float
    critical_values: dict[str, float]
    reject_at_5pct: bool
    n_obs: int

    def to_dict(self) -> dict:
        return {"statistic": self.statistic, "reject_at_5pct": self.reject_at_5pct,
                "n_obs": self.n_obs}


def anderson_darling(data: np.ndarray, distribution: str = "normal") -> ADResult:
    """Anderson-Darling test (more powerful in tails than KS).

    H0: data comes from the specified distribution.
    """
    from scipy.stats import anderson

    x = np.asarray(data, dtype=float)
    dist_map = {"normal": "norm", "exponential": "expon", "logistic": "logistic"}
    dist = dist_map.get(distribution, distribution)

    result = anderson(x, dist=dist)
    criticals = {f"{sl}%": float(cv) for sl, cv in zip(result.significance_level, result.critical_values)}
    reject = result.statistic > result.critical_values[2]  # 5% level

    return ADResult(float(result.statistic), criticals, reject, len(x))


# ═══════════════════════════════════════════════════════════════
# Q-Q plot data
# ═══════════════════════════════════════════════════════════════

@dataclass
class QQData:
    """Q-Q plot data."""
    theoretical: np.ndarray
    empirical: np.ndarray
    distribution: str

    def to_dict(self) -> dict:
        return {"distribution": self.distribution,
                "theoretical": self.theoretical.tolist(),
                "empirical": self.empirical.tolist()}


def qq_data(data: np.ndarray, distribution: str = "normal") -> QQData:
    """Generate Q-Q plot data (theoretical vs empirical quantiles).

    Args:
        distribution: 'normal' or 'student_t'.
    """
    from scipy import stats

    x = np.sort(np.asarray(data, dtype=float))
    n = len(x)
    probs = (np.arange(1, n + 1) - 0.5) / n

    if distribution == "normal":
        theoretical = stats.norm.ppf(probs)
    elif distribution == "student_t":
        r = fit_student_t(data)
        theoretical = stats.t.ppf(probs, df=r.params["nu"])
    else:
        theoretical = stats.norm.ppf(probs)  # fallback

    return QQData(theoretical, x, distribution)
