"""Advanced bond relative value: issuer curve fitting, invoice spread, PCA.

* :func:`issuer_curve_fit` — Nelson-Siegel on issuer bonds.
* :func:`invoice_spread` — bond ASW vs futures-implied ASW.
* :func:`ois_asw_decomposition` — IBOR-OIS-ASW decomposition.
* :func:`bond_yield_pca` — PCA on yield changes.

References:
    Tuckman & Serrat, *Fixed Income Securities*, Wiley, 2012.
    Litterman & Scheinkman, *Common Factors Affecting Bond Returns*, JFI, 1991.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class IssuerCurveFitResult:
    """Issuer curve fit result."""
    beta0: float
    beta1: float
    beta2: float
    tau: float
    residuals: np.ndarray       # per-bond residual (bps)
    rms_residual: float
    rich_cheap: dict[str, float]  # bond_id → residual


def _ns_yield(t, b0, b1, b2, tau):
    if t < 1e-10:
        return b0 + b1
    x = t / tau
    ex = math.exp(-x)
    return b0 + b1 * (1 - ex) / x + b2 * ((1 - ex) / x - ex)


def issuer_curve_fit(
    bond_ids: list[str],
    maturities: list[float],
    yields_pct: list[float],
) -> IssuerCurveFitResult:
    """Fit Nelson-Siegel to a single issuer's bonds. Residuals = rich/cheap."""
    T = np.array(maturities)
    Y = np.array(yields_pct)

    def objective(params):
        b0, b1, b2, tau = params
        if tau < 0.01:
            return 1e10
        model = np.array([_ns_yield(t, b0, b1, b2, tau) for t in T])
        return float(np.sum((model - Y) ** 2))

    b0_init = float(Y[-1]) if len(Y) > 0 else 0.05
    b1_init = float(Y[0] - Y[-1]) if len(Y) > 1 else 0.0
    result = minimize(objective, [b0_init, b1_init, 0.0, 2.0], method='Nelder-Mead',
                      options={'maxiter': 5000})

    b0, b1, b2, tau = result.x
    tau = max(tau, 0.01)
    fitted = np.array([_ns_yield(t, b0, b1, b2, tau) for t in T])
    residuals_bps = (Y - fitted) * 100
    rms = float(np.sqrt(np.mean(residuals_bps**2)))

    rich_cheap = {bid: float(r) for bid, r in zip(bond_ids, residuals_bps)}

    return IssuerCurveFitResult(b0, b1, b2, tau, residuals_bps, rms, rich_cheap)


@dataclass
class InvoiceSpreadResult:
    """Invoice spread result."""
    bond_asw_bps: float
    futures_implied_asw_bps: float
    invoice_spread_bps: float


def invoice_spread(
    bond_asw_bps: float,
    futures_implied_asw_bps: float,
) -> InvoiceSpreadResult:
    """Invoice spread: bond ASW − futures-implied ASW.

    Reflects repo specials, delivery option, and liquidity.
    """
    inv_spread = bond_asw_bps - futures_implied_asw_bps
    return InvoiceSpreadResult(bond_asw_bps, futures_implied_asw_bps, float(inv_spread))


@dataclass
class OISASWResult:
    """OIS-asset swap decomposition."""
    bond_yield_bps: float
    swap_rate_bps: float
    ois_rate_bps: float
    ibor_asw_bps: float         # bond - swap
    ois_asw_bps: float          # bond - OIS
    ibor_ois_basis_bps: float   # swap - OIS


def ois_asw_decomposition(
    bond_yield_pct: float,
    swap_rate_pct: float,
    ois_rate_pct: float,
) -> OISASWResult:
    """IBOR-OIS asset swap decomposition.

    bond_yield = OIS + IBOR-OIS basis + credit/liquidity spread
    IBOR ASW = bond - swap
    OIS ASW = bond - OIS = IBOR ASW + IBOR-OIS basis
    """
    b = bond_yield_pct * 100
    s = swap_rate_pct * 100
    o = ois_rate_pct * 100
    return OISASWResult(b, s, o,
                         float(b - s), float(b - o), float(s - o))


@dataclass
class PCAResult:
    """PCA on bond yield changes."""
    explained_variance_ratio: np.ndarray
    components: np.ndarray          # (n_components, n_tenors)
    n_components: int
    cumulative_explained: np.ndarray


def bond_yield_pca(
    yield_changes: np.ndarray,      # (n_obs, n_tenors)
    n_components: int = 3,
) -> PCAResult:
    """PCA on bond yield changes (Litterman-Scheinkman).

    Typically 3 factors explain >95% of yield curve moves:
    1. Level (parallel shift)
    2. Slope (steepening/flattening)
    3. Curvature (butterfly)

    Args:
        yield_changes: (n_observations, n_tenors) matrix of daily yield changes.
        n_components: number of components to extract.
    """
    # Centre
    centered = yield_changes - yield_changes.mean(axis=0)
    # Covariance
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = eigenvalues.sum()
    explained = eigenvalues / max(total_var, 1e-10)
    cumulative = np.cumsum(explained)

    components = eigenvectors[:, :n_components].T

    return PCAResult(
        explained_variance_ratio=explained[:n_components],
        components=components,
        n_components=n_components,
        cumulative_explained=cumulative[:n_components],
    )
