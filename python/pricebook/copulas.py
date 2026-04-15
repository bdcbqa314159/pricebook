"""Copula models: Student-t, Clayton, Frank, Gumbel, nested copulas.

Extends the Gaussian one-factor copula in :mod:`pricebook.basket_cds`
with richer dependence structures for portfolio credit risk.

* :class:`GaussianCopula` — standard Gaussian (baseline).
* :class:`StudentTCopula` — tail dependence via degrees of freedom ν.
* :class:`ClaytonCopula` — lower tail dependence (defaults cluster in stress).
* :class:`FrankCopula` — symmetric, no tail dependence.
* :class:`GumbelCopula` — upper tail dependence.
* :func:`copula_default_simulation` — simulate correlated defaults.
* :func:`tranche_pricing_copula` — CDO tranche pricing under any copula.

References:
    Li, *On Default Correlation: A Copula Function Approach*, J. Fixed Income, 2000.
    McNeil, Frey & Embrechts, *Quantitative Risk Management*, Princeton, 2005.
    Cherubini, Luciano & Vecchiato, *Copula Methods in Finance*, Wiley, 2004.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, t as t_dist


# ---- Abstract copula ----

class Copula(ABC):
    """Abstract base class for bivariate/multivariate copulas."""

    @abstractmethod
    def sample(self, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n draws of d uniform [0,1] variates with copula dependence.

        Returns: (n, d) array of uniform marginals.
        """

    def default_indicators(
        self,
        marginal_pds: list[float],
        n_sims: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Simulate correlated default indicators.

        Args:
            marginal_pds: per-name default probability.
            n_sims: number of simulations.

        Returns:
            (n_sims, n_names) boolean array.
        """
        rng = np.random.default_rng(seed)
        d = len(marginal_pds)
        U = self.sample(n_sims, d, rng)
        pds = np.array(marginal_pds)
        return U < pds[np.newaxis, :]


# ---- Gaussian copula ----

class GaussianCopula(Copula):
    """Gaussian copula with equi-correlation ρ (one-factor model).

    U_i = Φ(√ρ M + √(1−ρ) ε_i) where M, ε ~ N(0,1).
    """

    def __init__(self, rho: float):
        self.rho = rho

    def sample(self, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
        M = rng.standard_normal(n)
        eps = rng.standard_normal((n, d))
        Z = math.sqrt(self.rho) * M[:, np.newaxis] + math.sqrt(1 - self.rho) * eps
        return norm.cdf(Z)


# ---- Student-t copula ----

class StudentTCopula(Copula):
    """Student-t copula with equi-correlation ρ and ν degrees of freedom.

    Has tail dependence: λ = 2 t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))) > 0.
    As ν → ∞, converges to Gaussian copula.
    """

    def __init__(self, rho: float, nu: float = 5.0):
        self.rho = rho
        self.nu = nu

    def sample(self, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
        # Sample from multivariate t via: T = Z / √(W/ν) where W ~ χ²(ν)
        M = rng.standard_normal(n)
        eps = rng.standard_normal((n, d))
        Z = math.sqrt(self.rho) * M[:, np.newaxis] + math.sqrt(1 - self.rho) * eps
        W = rng.chisquare(self.nu, n)
        T = Z / np.sqrt(W[:, np.newaxis] / self.nu)
        return t_dist.cdf(T, self.nu)

    @property
    def tail_dependence(self) -> float:
        """Lower tail dependence coefficient."""
        arg = -math.sqrt((self.nu + 1) * (1 - self.rho) / (1 + self.rho))
        return 2 * t_dist.cdf(arg, self.nu + 1)


# ---- Clayton copula ----

class ClaytonCopula(Copula):
    """Clayton copula: C(u,v) = (u^{−θ} + v^{−θ} − 1)^{−1/θ}.

    Lower tail dependence: λ_L = 2^{−1/θ} > 0.
    θ > 0 for positive dependence. θ → 0 gives independence.
    """

    def __init__(self, theta: float):
        if theta <= 0:
            raise ValueError("Clayton θ must be > 0")
        self.theta = theta

    def sample(self, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
        """Marshall-Olkin algorithm for Archimedean copulas.

        For Clayton: frailty V ~ Gamma(1/θ, 1).
        """
        V = rng.gamma(1.0 / self.theta, 1.0, n)
        E = rng.exponential(1.0, (n, d))
        # X_i = (E_i / V)^{-1/θ} ... but for Clayton the transform is:
        # U_i = (1 + E_i / V)^{-1/θ}
        U = (1 + E / V[:, np.newaxis]) ** (-1.0 / self.theta)
        return U

    @property
    def lower_tail_dependence(self) -> float:
        return 2 ** (-1.0 / self.theta)


# ---- Frank copula ----

class FrankCopula(Copula):
    """Frank copula: no tail dependence (symmetric).

    C(u,v) = −(1/θ) ln(1 + (e^{−θu}−1)(e^{−θv}−1)/(e^{−θ}−1)).
    """

    def __init__(self, theta: float):
        if theta == 0:
            raise ValueError("Frank θ must be ≠ 0")
        self.theta = theta

    def sample(self, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
        """Approximate sampling via conditional method for d=2,
        extended to d>2 via sequential conditioning."""
        U = np.zeros((n, d))
        U[:, 0] = rng.uniform(0, 1, n)

        for j in range(1, d):
            v = rng.uniform(0, 1, n)
            u1 = U[:, 0]
            # Conditional CDS: C_{2|1}(v|u) via Frank formula
            a = np.exp(-self.theta * u1)
            b = np.exp(-self.theta)
            # Inverse of conditional: u2 = -ln(1 + v(b-1)/(v(a-1) - (a-1))) / θ
            num = v * (a - 1)
            den = v * (a - 1) - (b - 1)
            # Guard against division issues
            safe_den = np.where(np.abs(den) < 1e-15, 1e-15, den)
            arg = 1 + (b - 1) * (a - 1) / (safe_den * (np.exp(-self.theta) - 1 + 1e-15))
            arg = np.clip(arg, 1e-15, None)
            U[:, j] = -np.log(arg) / self.theta
            U[:, j] = np.clip(U[:, j], 0.0, 1.0)

        return U


# ---- Gumbel copula ----

class GumbelCopula(Copula):
    """Gumbel copula: upper tail dependence.

    C(u,v) = exp(−((−ln u)^θ + (−ln v)^θ)^{1/θ}).
    θ ≥ 1. θ = 1 gives independence. Upper tail dep = 2 − 2^{1/θ}.
    """

    def __init__(self, theta: float):
        if theta < 1:
            raise ValueError("Gumbel θ must be ≥ 1")
        self.theta = theta

    def sample(self, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
        """Marshall-Olkin: frailty from stable distribution."""
        # Stable(1/θ) sampling via Chambers-Mallows-Stuck
        alpha = 1.0 / self.theta
        if abs(alpha - 1.0) < 1e-10:
            # θ = 1 → independence
            return rng.uniform(0, 1, (n, d))

        # Sample stable(alpha, 1, 0, 0) using CMS method
        phi = (rng.uniform(0, 1, n) - 0.5) * math.pi
        W = rng.exponential(1.0, n)
        V = np.sin(alpha * phi) / np.cos(phi) ** (1.0 / alpha) * \
            (np.cos(phi * (1 - alpha)) / W) ** ((1 - alpha) / alpha)
        V = np.maximum(V, 1e-10)

        E = rng.exponential(1.0, (n, d))
        U = np.exp(-(E / V[:, np.newaxis]) ** (1.0 / self.theta))
        return np.clip(U, 0, 1)

    @property
    def upper_tail_dependence(self) -> float:
        return 2 - 2 ** (1.0 / self.theta)


# ---- Default simulation + tranche pricing ----

@dataclass
class CopulaDefaultResult:
    """Result of copula-based default simulation."""
    default_rate: float
    n_defaults_mean: float
    loss_distribution: np.ndarray  # (n_sims,) losses
    correlation_estimate: float


def copula_default_simulation(
    copula: Copula,
    marginal_pds: list[float],
    lgd: float = 0.6,
    n_sims: int = 50_000,
    seed: int | None = None,
) -> CopulaDefaultResult:
    """Simulate portfolio defaults under a copula model."""
    defaults = copula.default_indicators(marginal_pds, n_sims, seed)
    n_names = len(marginal_pds)

    n_defaults = defaults.sum(axis=1)
    losses = n_defaults * lgd / n_names  # loss as fraction of portfolio

    # Estimate pairwise default correlation
    if n_names >= 2:
        p1 = defaults[:, 0].mean()
        p2 = defaults[:, 1].mean()
        p12 = (defaults[:, 0] & defaults[:, 1]).mean()
        denom = math.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
        corr = (p12 - p1 * p2) / denom if denom > 0 else 0.0
    else:
        corr = 0.0

    return CopulaDefaultResult(
        float(defaults.any(axis=1).mean()),
        float(n_defaults.mean()),
        losses,
        corr,
    )


@dataclass
class TranchePricingResult:
    """CDO tranche pricing under a copula."""
    expected_loss: float
    tranche_spread: float
    copula_name: str


def tranche_pricing_copula(
    copula: Copula,
    marginal_pds: list[float],
    attach: float,
    detach: float,
    lgd: float = 0.6,
    T: float = 5.0,
    rate: float = 0.05,
    n_sims: int = 100_000,
    seed: int | None = None,
) -> TranchePricingResult:
    """Price a CDO tranche under any copula model.

    Args:
        attach: attachment point (e.g. 0.03 for 3%).
        detach: detachment point (e.g. 0.07 for 7%).
    """
    result = copula_default_simulation(copula, marginal_pds, lgd, n_sims, seed)

    # Tranche loss: max(min(portfolio_loss, detach) - attach, 0) / (detach - attach)
    thickness = detach - attach
    tranche_loss = np.maximum(
        np.minimum(result.loss_distribution, detach) - attach, 0.0
    ) / thickness

    el = float(tranche_loss.mean())

    # Spread ≈ EL / risky_annuity (simplified)
    annuity = sum(math.exp(-rate * t) for t in range(1, int(T) + 1))
    spread = el / annuity if annuity > 0 else 0.0

    name = type(copula).__name__

    return TranchePricingResult(el, spread, name)
