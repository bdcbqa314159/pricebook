"""Probability distributions: clean interface over scipy.stats.

Replaces direct scipy.stats.norm calls throughout the codebase.

    from pricebook.numerical import Normal, StudentT, LogNormal

    Normal.cdf(1.96)       # 0.975
    Normal.pdf(0.0)        # 0.3989
    Normal.ppf(0.975)      # 1.96
    Normal.rvs(1000, rng)  # 1000 samples
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class Normal:
    """Standard normal distribution N(mu, sigma²).

    Default (mu=0, sigma=1) matches scipy.stats.norm.
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma

    def cdf(self, x) -> float | np.ndarray:
        """Cumulative distribution function."""
        z = (np.asarray(x) - self.mu) / self.sigma
        return _norm_cdf(z)

    def pdf(self, x) -> float | np.ndarray:
        """Probability density function."""
        z = (np.asarray(x) - self.mu) / self.sigma
        return _norm_pdf(z) / self.sigma

    def ppf(self, p) -> float | np.ndarray:
        """Percent point function (inverse CDF / quantile)."""
        from scipy.stats import norm
        return float(norm.ppf(p, loc=self.mu, scale=self.sigma)) if np.isscalar(p) \
            else norm.ppf(np.asarray(p), loc=self.mu, scale=self.sigma)

    def logpdf(self, x) -> float | np.ndarray:
        """Log of the probability density function."""
        z = (np.asarray(x) - self.mu) / self.sigma
        return -0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) - math.log(self.sigma)

    def rvs(self, size: int = 1, rng=None) -> np.ndarray:
        """Random variates."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(self.mu, self.sigma, size)

    def moment(self, n: int) -> float:
        """n-th central moment: E[(X - mu)^n]. n=1: 0, n=2: sigma^2, n=3: 0, n=4: 3*sigma^4."""
        if n == 1:
            return 0.0  # first central moment is always 0
        if n == 2:
            return self.sigma ** 2
        if n == 3:
            return 0.0
        if n == 4:
            return 3.0 * self.sigma ** 4
        raise ValueError(f"moment {n} not supported")

    def to_dict(self) -> dict:
        return {"type": "Normal", "mu": self.mu, "sigma": self.sigma}


# Module-level standard normal for convenience (most common use case)
_std_normal = Normal()


class StudentT:
    """Student's t-distribution with df degrees of freedom.

    Heavier tails than normal for small df. Converges to normal as df → ∞.
    """

    def __init__(self, df: float, mu: float = 0.0, sigma: float = 1.0):
        if df <= 0:
            raise ValueError(f"df must be positive, got {df}")
        self.df = df
        self.mu = mu
        self.sigma = sigma

    def cdf(self, x) -> float | np.ndarray:
        from scipy.stats import t
        return t.cdf(x, self.df, loc=self.mu, scale=self.sigma)

    def pdf(self, x) -> float | np.ndarray:
        from scipy.stats import t
        return t.pdf(x, self.df, loc=self.mu, scale=self.sigma)

    def ppf(self, p) -> float | np.ndarray:
        from scipy.stats import t
        return t.ppf(p, self.df, loc=self.mu, scale=self.sigma)

    def rvs(self, size: int = 1, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return self.mu + self.sigma * rng.standard_t(self.df, size)

    def tail_dependence(self) -> float:
        """Lower tail dependence coefficient (for copula context)."""
        from scipy.stats import t
        return 2 * t.cdf(-math.sqrt((self.df + 1) / (self.df - 1 + 1e-10)),
                         self.df + 1)

    def to_dict(self) -> dict:
        return {"type": "StudentT", "df": self.df, "mu": self.mu, "sigma": self.sigma}


class LogNormal:
    """Log-normal distribution: log(X) ~ N(mu, sigma²).

    If X ~ LogNormal(mu, sigma), then E[X] = exp(mu + sigma²/2).
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma

    def cdf(self, x) -> float | np.ndarray:
        x = np.asarray(x, dtype=float)
        z = np.where(x > 0, (np.log(x) - self.mu) / self.sigma, -np.inf)
        return _norm_cdf(z)

    def pdf(self, x) -> float | np.ndarray:
        x = np.asarray(x, dtype=float)
        result = np.where(
            x > 0,
            _norm_pdf((np.log(x) - self.mu) / self.sigma) / (x * self.sigma),
            0.0,
        )
        return result

    def ppf(self, p) -> float | np.ndarray:
        from scipy.stats import norm
        return np.exp(self.mu + self.sigma * norm.ppf(p))

    def rvs(self, size: int = 1, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return np.exp(rng.normal(self.mu, self.sigma, size))

    def mean(self) -> float:
        return math.exp(self.mu + 0.5 * self.sigma ** 2)

    def variance(self) -> float:
        return (math.exp(self.sigma ** 2) - 1) * math.exp(2 * self.mu + self.sigma ** 2)

    def to_dict(self) -> dict:
        return {"type": "LogNormal", "mu": self.mu, "sigma": self.sigma}


class Uniform:
    """Uniform distribution on [a, b]."""

    def __init__(self, a: float = 0.0, b: float = 1.0):
        if a >= b:
            raise ValueError(f"a ({a}) must be < b ({b})")
        self.a = a
        self.b = b

    def cdf(self, x) -> float | np.ndarray:
        return np.clip((np.asarray(x) - self.a) / (self.b - self.a), 0, 1)

    def pdf(self, x) -> float | np.ndarray:
        x = np.asarray(x)
        return np.where((x >= self.a) & (x <= self.b), 1.0 / (self.b - self.a), 0.0)

    def ppf(self, p) -> float | np.ndarray:
        return self.a + np.asarray(p) * (self.b - self.a)

    def rvs(self, size: int = 1, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.a, self.b, size)

    def to_dict(self) -> dict:
        return {"type": "Uniform", "a": self.a, "b": self.b}


class Exponential:
    """Exponential distribution with rate lambda."""

    def __init__(self, rate: float = 1.0):
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate

    def cdf(self, x) -> float | np.ndarray:
        x = np.asarray(x)
        return np.where(x >= 0, 1 - np.exp(-self.rate * x), 0.0)

    def pdf(self, x) -> float | np.ndarray:
        x = np.asarray(x)
        return np.where(x >= 0, self.rate * np.exp(-self.rate * x), 0.0)

    def ppf(self, p) -> float | np.ndarray:
        return -np.log(1 - np.asarray(p)) / self.rate

    def rvs(self, size: int = 1, rng=None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.exponential(1.0 / self.rate, size)

    def mean(self) -> float:
        return 1.0 / self.rate

    def to_dict(self) -> dict:
        return {"type": "Exponential", "rate": self.rate}


# ═══════════════════════════════════════════════════════════════
# Internal helpers (avoid scipy for the hot path)
# ═══════════════════════════════════════════════════════════════

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_INV_SQRT_2PI = 1.0 / _SQRT_2PI


def _norm_cdf(z):
    """Standard normal CDF via math.erf (no scipy needed)."""
    if isinstance(z, np.ndarray):
        from scipy.special import erf
        return 0.5 * (1.0 + erf(z / _SQRT_2))
    return 0.5 * (1.0 + math.erf(z / _SQRT_2))


def _norm_pdf(z):
    """Standard normal PDF."""
    return np.exp(-0.5 * np.asarray(z) ** 2) * _INV_SQRT_2PI
