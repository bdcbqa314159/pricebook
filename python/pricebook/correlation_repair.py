"""Correlation matrix repair: nearest PD matrix, eigenvalue floor, interpolation.

Market-observed correlation matrices are often not positive-definite due
to asynchronous data, missing observations, or inconsistent estimation
windows. This module provides tools to repair them.

* :func:`nearest_correlation_matrix` — Higham (2002) alternating projections.
* :func:`eigenvalue_floor` — simple fix: clip negative eigenvalues to ε.
* :func:`is_positive_definite` — check via Cholesky.
* :func:`correlation_interpolation` — interpolate between two PD matrices.

References:
    Higham, *Computing the Nearest Correlation Matrix — A Problem from
    Finance*, IMA J. Numer. Anal., 2002.
    Rebonato & Jäckel, *The Most General Methodology to Create a Valid
    Correlation Matrix for Risk Management*, 2000.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---- Positive-definiteness check ----

def is_positive_definite(matrix: np.ndarray | list[list[float]]) -> bool:
    """Check if a matrix is positive-definite via Cholesky."""
    A = np.asarray(matrix, dtype=float)
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def is_valid_correlation(matrix: np.ndarray | list[list[float]]) -> bool:
    """Check if a matrix is a valid correlation matrix.

    Requirements: symmetric, unit diagonal, PD, all entries in [-1, 1].
    """
    A = np.asarray(matrix, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n):
        return False
    if not np.allclose(A, A.T, atol=1e-10):
        return False
    if not np.allclose(np.diag(A), 1.0, atol=1e-10):
        return False
    if np.any(A < -1.0 - 1e-10) or np.any(A > 1.0 + 1e-10):
        return False
    return is_positive_definite(A)


# ---- Eigenvalue floor ----

def eigenvalue_floor(
    matrix: np.ndarray | list[list[float]],
    floor: float = 1e-6,
) -> np.ndarray:
    """Repair a correlation matrix by flooring negative eigenvalues.

    Decomposes A = V Λ V', clips eigenvalues at *floor*, rebuilds,
    then rescales to unit diagonal.

    Simple but does NOT minimise the distance to the original matrix.
    For the optimal nearest correlation matrix, use :func:`nearest_correlation_matrix`.
    """
    A = np.asarray(matrix, dtype=float)
    eigenvalues, V = np.linalg.eigh(A)
    eigenvalues = np.maximum(eigenvalues, floor)
    B = V @ np.diag(eigenvalues) @ V.T
    # Rescale to unit diagonal
    d = np.sqrt(np.diag(B))
    d[d == 0] = 1.0
    D_inv = np.diag(1.0 / d)
    return D_inv @ B @ D_inv


# ---- Higham nearest correlation matrix ----

@dataclass
class NearestCorrResult:
    """Result of nearest correlation matrix computation."""
    matrix: np.ndarray
    iterations: int
    frobenius_distance: float
    is_pd: bool


def nearest_correlation_matrix(
    matrix: np.ndarray | list[list[float]],
    max_iter: int = 100,
    tol: float = 1e-10,
) -> NearestCorrResult:
    """Compute the nearest correlation matrix in Frobenius norm.

    Implements the alternating projections algorithm of Higham (2002):
    alternates between projecting onto the set of PSD matrices (eigenvalue
    clipping) and the set of matrices with unit diagonal.

    The result is the closest matrix (in Frobenius norm) that is both
    positive semi-definite and has unit diagonal.

    Args:
        matrix: input (possibly invalid) correlation matrix.
        max_iter: maximum number of alternating projection iterations.
        tol: convergence tolerance on the Frobenius norm change.

    Returns:
        :class:`NearestCorrResult`.
    """
    A = np.asarray(matrix, dtype=float)
    n = A.shape[0]
    Y = A.copy()
    delta_S = np.zeros_like(A)

    for iteration in range(max_iter):
        # Dykstra correction
        R = Y - delta_S

        # Project onto PSD cone
        eigenvalues, V = np.linalg.eigh(R)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        X = V @ np.diag(eigenvalues) @ V.T

        delta_S = X - R

        # Project onto unit-diagonal matrices
        Y_new = X.copy()
        np.fill_diagonal(Y_new, 1.0)

        # Convergence check
        diff = np.linalg.norm(Y_new - Y, 'fro')
        Y = Y_new

        if diff < tol:
            break

    # Ensure strict PD by nudging any near-zero eigenvalues
    evals, V = np.linalg.eigh(Y)
    if np.any(evals < 1e-14):
        evals = np.maximum(evals, 1e-14)
        Y = V @ np.diag(evals) @ V.T
        # Re-enforce unit diagonal after nudge
        d = np.sqrt(np.diag(Y))
        D_inv = np.diag(1.0 / d)
        Y = D_inv @ Y @ D_inv

    frob = float(np.linalg.norm(Y - A, 'fro'))
    pd = is_positive_definite(Y)

    return NearestCorrResult(
        matrix=Y,
        iterations=iteration + 1,
        frobenius_distance=frob,
        is_pd=pd,
    )


# ---- Correlation interpolation ----

def correlation_interpolation(
    corr_a: np.ndarray | list[list[float]],
    corr_b: np.ndarray | list[list[float]],
    weight: float,
) -> np.ndarray:
    """Interpolate between two correlation matrices.

    Uses linear interpolation in matrix space followed by nearest
    correlation matrix projection to ensure the result is PD.

        C(w) = nearest_corr((1 - w) × A + w × B)

    A more sophisticated approach would use geodesic interpolation on
    SPD(n), but this is adequate for most practical purposes.

    Args:
        corr_a: first correlation matrix.
        corr_b: second correlation matrix.
        weight: interpolation weight (0 = corr_a, 1 = corr_b).

    Returns:
        Interpolated correlation matrix (guaranteed PD with unit diagonal).
    """
    A = np.asarray(corr_a, dtype=float)
    B = np.asarray(corr_b, dtype=float)
    C = (1 - weight) * A + weight * B
    result = nearest_correlation_matrix(C)
    return result.matrix
