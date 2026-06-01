"""Index replication and tracking error.

Replicate a CDS index using a subset of constituents, with
L1-regularised optimisation for sparse portfolios.

* :class:`ReplicationResult` — replication weights and tracking error.
* :func:`replicate_index` — find optimal weights for index replication.
* :func:`tracking_error` — annualised tracking error vs full index.

References:
    O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*,
    Ch. 8, 2008.
    Joshi & Stacey, *Intensity Gamma*, Risk, 2006.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ReplicationResult:
    """Index replication result."""
    weights: np.ndarray
    tracking_error: float
    n_active: int        # number of non-zero weights
    r_squared: float     # explained variance
    residual_spread_bp: float

    def to_dict(self) -> dict:
        return {
            "n_active": self.n_active,
            "tracking_error_bp": self.tracking_error * 10_000,
            "r_squared": self.r_squared,
            "residual_spread_bp": self.residual_spread_bp,
        }


def replicate_index(
    index_spreads: np.ndarray,
    constituent_spreads: np.ndarray,
    n_select: int | None = None,
    l1_penalty: float = 0.0,
    long_only: bool = True,
) -> ReplicationResult:
    """Find optimal weights to replicate an index from constituents.

    Minimises: ||index − constituent_spreads @ w||² + λ × ||w||₁

    When n_select is given, selects the n_select names with highest
    correlation to the index (greedy selection).

    Args:
        index_spreads: (T,) array of index spread history.
        constituent_spreads: (T, N) array of constituent spread histories.
        n_select: if given, select top-N names before optimisation.
        l1_penalty: L1 regularisation for sparsity.
        long_only: if True, constrain weights ≥ 0.
    """
    T, N = constituent_spreads.shape
    idx = index_spreads.copy()

    # Name selection: pick top-n by correlation
    if n_select is not None and n_select < N:
        corrs = np.array([
            np.corrcoef(idx, constituent_spreads[:, j])[0, 1]
            for j in range(N)
        ])
        selected = np.argsort(-np.abs(corrs))[:n_select]
        X = constituent_spreads[:, selected]
    else:
        selected = np.arange(N)
        X = constituent_spreads.copy()

    n_sel = X.shape[1]

    # Solve via iteratively reweighted least squares with L1
    if l1_penalty > 0:
        w = _lasso_cd(X, idx, l1_penalty, long_only)
    else:
        if long_only:
            w = _nnls(X, idx)
        else:
            w = np.linalg.lstsq(X, idx, rcond=None)[0]

    # Map back to full constituent space
    full_weights = np.zeros(N)
    full_weights[selected] = w

    # Tracking error
    residual = idx - X @ w
    te = np.std(residual) * math.sqrt(252)  # annualised

    # R-squared
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((idx - np.mean(idx)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-15)

    residual_bp = np.mean(np.abs(residual)) * 10_000

    return ReplicationResult(
        weights=full_weights,
        tracking_error=te,
        n_active=int(np.sum(np.abs(full_weights) > 1e-8)),
        r_squared=max(r2, 0.0),
        residual_spread_bp=residual_bp,
    )


def tracking_error(
    index_returns: np.ndarray,
    replica_returns: np.ndarray,
    annualise: bool = True,
) -> float:
    """Annualised tracking error between index and replica.

    TE = std(index_returns − replica_returns) × √252.

    Args:
        index_returns: daily index spread changes.
        replica_returns: daily replica spread changes.
        annualise: if True, multiply by √252.
    """
    diff = index_returns - replica_returns
    te = np.std(diff)
    if annualise:
        te *= math.sqrt(252)
    return float(te)


# ---- Internal solvers ----

def _nnls(X: np.ndarray, y: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """Non-negative least squares via active set method."""
    n = X.shape[1]
    w = np.zeros(n)
    XtX = X.T @ X
    Xty = X.T @ y

    for _ in range(max_iter):
        gradient = XtX @ w - Xty
        # Find most violating constraint
        violations = gradient.copy()
        violations[w > 0] = 0  # active constraints only
        if np.all(violations >= -1e-10):
            break
        j = np.argmin(violations)
        # Coordinate step
        step = -gradient[j] / max(XtX[j, j], 1e-10)
        w[j] = max(w[j] + step, 0)

    return w


def _lasso_cd(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    long_only: bool,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Coordinate descent for L1-regularised least squares (LASSO).

    Minimises: (1/2T) ||y − Xw||² + λ ||w||₁
    """
    T, n = X.shape
    w = np.zeros(n)
    Xty = X.T @ y / T
    XtX_diag = np.sum(X ** 2, axis=0) / T

    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(n):
            r_j = Xty[j] - (X.T @ (X @ w))[j] / T + XtX_diag[j] * w[j]
            # Soft thresholding
            if r_j > lam:
                w[j] = (r_j - lam) / max(XtX_diag[j], 1e-10)
            elif r_j < -lam and not long_only:
                w[j] = (r_j + lam) / max(XtX_diag[j], 1e-10)
            else:
                w[j] = 0.0
            if long_only:
                w[j] = max(w[j], 0.0)

        if np.max(np.abs(w - w_old)) < tol:
            break

    return w
