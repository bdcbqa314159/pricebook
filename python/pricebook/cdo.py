"""
CDO tranche pricing: loss distribution, tranching, base correlation.

Vasicek large homogeneous pool: conditional on systematic factor M,
defaults are independent. Integrate over M for unconditional loss.

    from pricebook.cdo import portfolio_loss_distribution, tranche_expected_loss

    loss_dist = portfolio_loss_distribution(pd=0.02, rho=0.3, lgd=0.6, n_names=100)
    eq_loss = tranche_expected_loss(loss_dist, 0.0, 0.03)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from pricebook.quadrature import gauss_hermite
from pricebook.solvers import brentq


def vasicek_conditional_pd(pd: float, rho: float, M: float) -> float:
    """Conditional PD given systematic factor M (Vasicek one-factor).

    P(default | M) = Phi((Phi^{-1}(PD) - sqrt(rho)*M) / sqrt(1-rho))
    """
    if rho <= 0:
        return pd
    threshold = norm.ppf(pd)
    return float(norm.cdf((threshold - math.sqrt(rho) * M) / math.sqrt(1 - rho)))


def portfolio_loss_distribution(
    pd: float,
    rho: float,
    lgd: float,
    n_names: int = 100,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Portfolio loss distribution via Vasicek large pool.

    Returns: (loss_fractions, probabilities).
    loss_fractions: array of loss levels as fraction of portfolio notional.
    probabilities: corresponding probability density (discrete).
    """
    # Loss fraction grid: 0 to lgd (max loss = all default)
    loss_grid = np.linspace(0, lgd, n_points)
    density = np.zeros(n_points)

    # Integrate over M using Gauss-Hermite (M ~ N(0,1), weight = exp(-M^2))
    # Transform: integral over M with N(0,1) density
    n_quad = 32

    for i in range(n_points):
        L = loss_grid[i]
        if L <= 0:
            continue

        # For loss fraction L: need P(N_defaults >= L/(lgd/n)) given M
        # In the large pool limit: loss fraction = conditional_PD * lgd
        # So P(loss = L) = density of conditional_PD at L/lgd

        def integrand(M):
            cond_pd = vasicek_conditional_pd(pd, rho, M)
            expected_loss = cond_pd * lgd
            # Gaussian kernel around this loss level
            sigma_kernel = lgd / (2 * n_points)
            if sigma_kernel <= 0:
                return 0.0
            return math.exp(-0.5 * ((L - expected_loss) / sigma_kernel)**2) / \
                (sigma_kernel * math.sqrt(2 * math.pi))

        result = gauss_hermite(
            lambda m: integrand(m * math.sqrt(2)) * math.sqrt(2),
            n=n_quad,
        )
        density[i] = max(result.value / math.sqrt(math.pi), 0)

    # Normalise
    total = density.sum()
    if total > 0:
        density /= total

    return loss_grid, density


def tranche_expected_loss(
    loss_grid: np.ndarray,
    density: np.ndarray,
    attach: float,
    detach: float,
) -> float:
    """Expected tranche loss for tranche [attach, detach].

    Tranche loss = min(max(portfolio_loss - attach, 0), detach - attach)
    """
    thickness = detach - attach
    if thickness <= 0:
        return 0.0

    tranche_losses = np.minimum(np.maximum(loss_grid - attach, 0), thickness)
    return float((tranche_losses * density).sum())


def tranche_spread(
    loss_grid: np.ndarray,
    density: np.ndarray,
    attach: float,
    detach: float,
    T: float = 5.0,
    risk_free_rate: float = 0.05,
) -> float:
    """Implied spread for a CDO tranche (simplified).

    spread ≈ expected_tranche_loss / (thickness * risky_annuity)
    """
    el = tranche_expected_loss(loss_grid, density, attach, detach)
    thickness = detach - attach
    if thickness <= 0:
        return 0.0

    # Simplified annuity: sum of df * survival ≈ sum of df
    annuity = sum(math.exp(-risk_free_rate * t) for t in range(1, int(T) + 1))

    return el / (thickness * annuity) if annuity > 0 else 0.0


def base_correlation(
    target_spread: float,
    detach: float,
    pd: float,
    lgd: float,
    T: float = 5.0,
    risk_free_rate: float = 0.05,
) -> float:
    """Base correlation: flat correlation that reprices the [0, detach] tranche.

    Solve for rho such that tranche_spread([0, detach], rho) = target_spread.
    """
    def objective(rho: float) -> float:
        if rho <= 0.001 or rho >= 0.999:
            return 1e10
        loss_grid, density = portfolio_loss_distribution(pd, rho, lgd)
        spread = tranche_spread(loss_grid, density, 0.0, detach, T, risk_free_rate)
        return spread - target_spread

    return brentq(objective, 0.01, 0.95)
