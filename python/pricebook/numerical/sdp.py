"""Semidefinite programming (SDP) interface.

Nearest correlation matrix as SDP, factor model covariance bounds,
and general SDP via alternating projections.

* :func:`nearest_psd` — project matrix onto PSD cone.
* :func:`nearest_correlation_sdp` — nearest correlation via SDP.
* :func:`factor_covariance_bounds` — bounds on covariance from factor model.
* :func:`sdp_solve` — general small-scale SDP via projected gradient.

References:
    Higham, *Computing the Nearest Correlation Matrix*, IMA JNA, 2002.
    Boyd & Vandenberghe, *Convex Optimization*, Ch. 4.6 (SDP).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SDPResult:
    """SDP solution result."""
    X: np.ndarray               # optimal matrix
    objective: float
    feasible: bool
    iterations: int
    psd_gap: float              # min eigenvalue (should be ≥ 0)

    def to_dict(self) -> dict:
        return {
            "objective": self.objective,
            "feasible": self.feasible,
            "iterations": self.iterations,
            "psd_gap": self.psd_gap,
            "size": self.X.shape[0],
        }


def nearest_psd(M: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone.

    Clips negative eigenvalues to zero.
    """
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals_clipped = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def nearest_correlation_sdp(
    A: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> SDPResult:
    """Nearest correlation matrix via alternating projections (Higham 2002).

    Minimise ||X − A||_F subject to:
    - X is positive semidefinite
    - diag(X) = 1 (unit diagonal)

    This is equivalent to an SDP:
    min ||X − A||_F  s.t.  X ≽ 0, X_ii = 1

    Solved via Dykstra's alternating projections between the PSD cone
    and the unit-diagonal affine set.

    Args:
        A: input matrix (should be symmetric).
        max_iter: maximum Dykstra iterations.
        tol: convergence tolerance on Frobenius distance change.
    """
    n = A.shape[0]
    S = np.zeros_like(A)  # Dykstra correction
    Y = A.copy()

    for iteration in range(max_iter):
        # Project onto PSD cone (with Dykstra correction)
        R = Y - S
        X = nearest_psd(R)
        S = X - R

        # Project onto unit diagonal
        Y_new = X.copy()
        np.fill_diagonal(Y_new, 1.0)

        # Convergence check
        diff = float(np.linalg.norm(Y_new - Y, 'fro'))
        Y = Y_new

        if diff < tol:
            break

    # Final PSD projection
    X_final = nearest_psd(Y)
    np.fill_diagonal(X_final, 1.0)

    min_eig = float(np.min(np.linalg.eigvalsh(X_final)))
    frob_dist = float(np.linalg.norm(X_final - A, 'fro'))

    return SDPResult(
        X=X_final,
        objective=frob_dist,
        feasible=min_eig >= -1e-8,
        iterations=iteration + 1,
        psd_gap=min_eig,
    )


def factor_covariance_bounds(
    factor_loadings: np.ndarray,
    factor_cov: np.ndarray,
    idiosyncratic_var: np.ndarray,
) -> dict:
    """Compute covariance matrix bounds from factor model.

    Σ = B Σ_F B' + D where B = loadings, Σ_F = factor covariance, D = diag(idio).

    Returns the full covariance, factor contribution, and
    minimum-variance portfolio weights.

    Args:
        factor_loadings: (N, K) factor loading matrix.
        factor_cov: (K, K) factor covariance matrix.
        idiosyncratic_var: (N,) idiosyncratic variance per asset.
    """
    N, K = factor_loadings.shape
    B = factor_loadings
    D = np.diag(idiosyncratic_var)

    # Full covariance
    Sigma = B @ factor_cov @ B.T + D

    # Factor contribution to variance
    factor_var = B @ factor_cov @ B.T
    idio_var = D

    # Eigenvalue decomposition for condition number
    eigvals = np.linalg.eigvalsh(Sigma)
    condition = float(eigvals[-1] / max(eigvals[0], 1e-15))

    # Minimum variance portfolio
    try:
        inv_Sigma = np.linalg.inv(Sigma)
        ones = np.ones(N)
        w_mv = inv_Sigma @ ones / (ones @ inv_Sigma @ ones)
        mv_vol = float(np.sqrt(w_mv @ Sigma @ w_mv))
    except np.linalg.LinAlgError:
        w_mv = np.ones(N) / N
        mv_vol = float(np.sqrt(w_mv @ Sigma @ w_mv))

    return {
        "covariance": Sigma,
        "factor_contribution": factor_var,
        "idiosyncratic_contribution": idio_var,
        "condition_number": condition,
        "min_eigenvalue": float(eigvals[0]),
        "min_variance_weights": w_mv,
        "min_variance_vol": mv_vol,
        "n_assets": N,
        "n_factors": K,
    }


def sdp_solve(
    C: np.ndarray,
    constraints: list[dict] | None = None,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> SDPResult:
    """General small-scale SDP via projected gradient descent.

    min tr(C X)  s.t. X ≽ 0, tr(A_i X) = b_i

    Each constraint is {"A": matrix, "b": scalar}.
    Projection onto PSD cone after each gradient step.

    Warning: Only practical for small n (≤ 50). For large-scale,
    use a dedicated SDP solver (e.g. CVXPY + SCS/MOSEK).

    Args:
        C: objective matrix (n × n, symmetric).
        constraints: list of equality constraints.
        max_iter: maximum iterations.
        tol: convergence tolerance.
    """
    n = C.shape[0]
    X = np.eye(n)  # initial feasible point

    step_size = 1.0 / (np.linalg.norm(C, 'fro') + 1)

    for iteration in range(max_iter):
        # Gradient of objective: ∇ tr(CX) = C
        grad = C.copy()

        # Add constraint penalty gradient
        if constraints:
            for con in constraints:
                A_i = con["A"]
                b_i = con["b"]
                violation = float(np.trace(A_i @ X)) - b_i
                grad += violation * A_i  # penalty gradient

        # Gradient step
        X_new = X - step_size * grad

        # Symmetrise
        X_new = 0.5 * (X_new + X_new.T)

        # Project onto PSD cone
        X_new = nearest_psd(X_new)

        # Convergence
        diff = float(np.linalg.norm(X_new - X, 'fro'))
        X = X_new

        if diff < tol:
            break

    obj = float(np.trace(C @ X))
    min_eig = float(np.min(np.linalg.eigvalsh(X)))

    return SDPResult(
        X=X, objective=obj, feasible=min_eig >= -1e-8,
        iterations=iteration + 1, psd_gap=min_eig,
    )
