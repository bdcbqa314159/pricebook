"""Linear algebra tools: PCA, eigendecomposition, SVD, condition numbers.

Foundation for yield curve factor models, risk decomposition, and
dimensionality reduction. Builds on numpy but exposes structured results
with financial interpretation.

* :func:`pca` — principal component analysis with explained variance.
* :func:`eigendecomposition` — sorted eigenvalues + eigenvectors.
* :func:`svd_decomposition` — thin SVD with singular values.
* :func:`condition_number` — matrix conditioning diagnostic.
* :func:`explained_variance` — cumulative explained variance ratios.

References:
    Litterman & Scheinkman, *Common Factors Affecting Bond Returns*, 1991.
    (Yield curve PCA: level, slope, curvature explain >95% of variance.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---- PCA ----

@dataclass
class PCAResult:
    """Principal component analysis result.

    Attributes:
        components: (n_components, n_features) — principal component directions.
        eigenvalues: (n_components,) — variance along each PC.
        explained_ratio: (n_components,) — fraction of total variance per PC.
        cumulative_ratio: (n_components,) — cumulative explained variance.
        mean: (n_features,) — sample mean (subtracted before PCA).
        n_components: number of components retained.
    """
    components: np.ndarray
    eigenvalues: np.ndarray
    explained_ratio: np.ndarray
    cumulative_ratio: np.ndarray
    mean: np.ndarray
    n_components: int

    def project(self, data: np.ndarray) -> np.ndarray:
        """Project data onto the first n_components PCs.

        Args:
            data: (n_samples, n_features) or (n_features,).

        Returns:
            (n_samples, n_components) or (n_components,) scores.
        """
        centered = data - self.mean
        return centered @ self.components.T

    def reconstruct(self, scores: np.ndarray) -> np.ndarray:
        """Reconstruct data from PC scores.

        Args:
            scores: (n_samples, n_components) or (n_components,).

        Returns:
            (n_samples, n_features) or (n_features,) reconstructed data.
        """
        return scores @ self.components + self.mean


def pca(
    data: np.ndarray | list[list[float]],
    n_components: int | None = None,
) -> PCAResult:
    """Principal component analysis via eigendecomposition of the covariance.

    Args:
        data: (n_samples, n_features) matrix of observations.
            Rows are observations, columns are features (e.g. tenor points).
        n_components: number of components to retain. If None, retains all.

    Returns:
        :class:`PCAResult` with sorted components, eigenvalues, and
        explained variance ratios.
    """
    X = np.asarray(data, dtype=float)
    n_samples, n_features = X.shape
    mean = X.mean(axis=0)
    centered = X - mean

    cov = centered.T @ centered / max(n_samples - 1, 1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Ensure non-negative (numerical noise can give tiny negatives)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    total_var = eigenvalues.sum()
    explained = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)
    cumulative = np.cumsum(explained)

    if n_components is None:
        n_components = n_features
    n_components = min(n_components, n_features)

    return PCAResult(
        components=eigenvectors[:, :n_components].T,  # (n_comp, n_feat)
        eigenvalues=eigenvalues[:n_components],
        explained_ratio=explained[:n_components],
        cumulative_ratio=cumulative[:n_components],
        mean=mean,
        n_components=n_components,
    )


# ---- Eigendecomposition ----

@dataclass
class EigenResult:
    """Eigendecomposition result, sorted by eigenvalue magnitude."""
    eigenvalues: np.ndarray   # (n,) sorted descending by |λ|
    eigenvectors: np.ndarray  # (n, n) columns are eigenvectors
    is_positive_definite: bool


def eigendecomposition(
    matrix: np.ndarray | list[list[float]],
) -> EigenResult:
    """Eigendecomposition of a symmetric matrix, sorted descending.

    Uses ``np.linalg.eigh`` (assumes symmetry for stability).
    """
    A = np.asarray(matrix, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    is_pd = bool(np.all(eigenvalues > -1e-10))

    return EigenResult(eigenvalues, eigenvectors, is_pd)


# ---- SVD ----

@dataclass
class SVDResult:
    """Singular value decomposition result."""
    U: np.ndarray          # (m, k) left singular vectors
    singular_values: np.ndarray  # (k,)
    Vt: np.ndarray         # (k, n) right singular vectors
    rank: int              # numerical rank (singular values > tol)


def svd_decomposition(
    matrix: np.ndarray | list[list[float]],
    tol: float = 1e-10,
) -> SVDResult:
    """Thin SVD with numerical rank computation.

    Args:
        matrix: (m, n) matrix.
        tol: singular values below this are considered zero.
    """
    A = np.asarray(matrix, dtype=float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    rank = int(np.sum(s > tol))
    return SVDResult(U, s, Vt, rank)


# ---- Condition number ----

@dataclass
class ConditionReport:
    """Matrix conditioning diagnostic."""
    condition_number: float
    is_well_conditioned: bool  # cond < 1e8
    is_ill_conditioned: bool   # cond > 1e12
    max_singular: float
    min_singular: float
    recommendation: str


def condition_number(
    matrix: np.ndarray | list[list[float]],
) -> ConditionReport:
    """Compute the condition number and diagnose conditioning.

    Uses the 2-norm condition number: κ = σ_max / σ_min.
    """
    A = np.asarray(matrix, dtype=float)
    s = np.linalg.svd(A, compute_uv=False)

    max_s = float(s[0]) if len(s) > 0 else 0.0
    min_s = float(s[-1]) if len(s) > 0 else 0.0
    cond = max_s / min_s if min_s > 0 else float("inf")

    if cond < 1e8:
        rec = "well-conditioned"
        well = True
        ill = False
    elif cond < 1e12:
        rec = "moderate conditioning — results may lose a few digits"
        well = False
        ill = False
    else:
        rec = "ill-conditioned — results unreliable, consider regularisation"
        well = False
        ill = True

    return ConditionReport(cond, well, ill, max_s, min_s, rec)


# ---- Explained variance utility ----

def explained_variance(
    eigenvalues: np.ndarray | list[float],
    threshold: float = 0.95,
) -> int:
    """Number of components needed to explain *threshold* fraction of variance.

    Args:
        eigenvalues: sorted descending.
        threshold: cumulative variance target (default 0.95 = 95%).

    Returns:
        Minimum number of components n such that
        sum(eigenvalues[:n]) / sum(eigenvalues) >= threshold.
    """
    evals = np.asarray(eigenvalues, dtype=float)
    total = evals.sum()
    if total <= 0:
        return 0
    cumsum = np.cumsum(evals) / total
    indices = np.where(cumsum >= threshold)[0]
    return int(indices[0] + 1) if len(indices) > 0 else len(evals)
