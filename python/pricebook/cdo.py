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
    """Portfolio loss distribution via Vasicek large-pool analytic formula.

    In the large-pool limit, portfolio loss fraction L = PD(M) × LGD where
    PD(M) is the conditional default probability given systematic factor M.
    The loss density is obtained by change of variable from M to L:

        f(L) = sqrt((1-rho)/rho) × phi(z) / phi(Phi^{-1}(L/LGD)) × (1/LGD)

    where z = (Phi^{-1}(L/LGD) × sqrt(1-rho) - Phi^{-1}(PD)) / sqrt(rho).

    Returns: (loss_fractions, probabilities).
    """
    loss_grid = np.linspace(0, lgd, n_points)
    density = np.zeros(n_points)

    if rho <= 0 or pd <= 0 or pd >= 1 or lgd <= 0:
        # Degenerate: all loss at one point
        idx = min(int(pd * lgd / lgd * n_points), n_points - 1)
        density[idx] = 1.0
        return loss_grid, density

    threshold = norm.ppf(pd)
    sqrt_rho = math.sqrt(rho)
    sqrt_1_rho = math.sqrt(1 - rho)

    for i in range(1, n_points):  # skip L=0
        L = loss_grid[i]
        p = L / lgd  # implied default fraction
        if p <= 0 or p >= 1:
            continue

        # Inverse of conditional PD formula
        z_p = norm.ppf(p)
        # M value that gives this conditional PD
        z = (z_p * sqrt_1_rho - threshold) / sqrt_rho

        # Density via change of variable (Jacobian)
        density[i] = (sqrt_1_rho / sqrt_rho) * norm.pdf(z) / max(norm.pdf(z_p), 1e-300) / lgd

    # Normalise: density is a discrete probability mass function (should sum to 1)
    dl = loss_grid[1] - loss_grid[0] if n_points > 1 else 1.0
    density *= dl  # convert density to probability mass
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

    # Search right branch (rho > 0.05) where spread is typically monotone in rho
    # for equity tranches. Fall back to full range if needed.
    for lo, hi in [(0.05, 0.999), (0.001, 0.999)]:
        try:
            return brentq(objective, lo, hi)
        except ValueError:
            continue
    # No root found — return boundary with smallest error
    err_low = abs(objective(0.001))
    err_high = abs(objective(0.999))
    return 0.001 if err_low < err_high else 0.999
