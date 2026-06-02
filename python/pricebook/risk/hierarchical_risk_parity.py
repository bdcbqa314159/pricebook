"""Hierarchical Risk Parity (HRP) — López de Prado (2016).

Tree-based portfolio allocation that doesn't require covariance
inversion, making it robust to estimation error.

* :func:`hrp_portfolio` — full HRP allocation.
* :func:`cluster_assets` — hierarchical clustering of assets.
* :func:`quasi_diagonalise` — reorder covariance by cluster.

References:
    López de Prado, *Building Diversified Portfolios that Outperform
    Out of Sample*, JFM, 2016.
    López de Prado, *Advances in Financial Machine Learning*, Ch. 16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


@dataclass
class HRPResult:
    """HRP portfolio result."""
    weights: np.ndarray
    cluster_order: list[int]
    n_assets: int
    n_clusters: int

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "n_assets": self.n_assets,
            "n_clusters": self.n_clusters,
        }


def hrp_portfolio(
    returns: np.ndarray,
    method: str = "single",
) -> HRPResult:
    """Hierarchical Risk Parity allocation.

    Three steps:
    1. Tree clustering: hierarchical clustering on correlation distance.
    2. Quasi-diagonalisation: reorder covariance by cluster structure.
    3. Recursive bisection: allocate inversely proportional to cluster variance.

    Args:
        returns: (T, N) return matrix.
        method: linkage method ("single", "complete", "average", "ward").
    """
    cov = np.cov(returns, rowvar=False)
    corr = np.corrcoef(returns, rowvar=False)

    # Step 1: Hierarchical clustering
    dist = _correlation_distance(corr)
    link = linkage(squareform(dist), method=method)
    order = list(leaves_list(link))

    # Step 2: Quasi-diagonalise
    cov_ordered = cov[np.ix_(order, order)]

    # Step 3: Recursive bisection
    weights_ordered = _recursive_bisection(cov_ordered, list(range(len(order))))

    # Map back to original order
    weights = np.zeros(len(order))
    for i, orig_idx in enumerate(order):
        weights[orig_idx] = weights_ordered[i]

    # Count clusters (using a cut at median distance)
    n_clusters = min(len(order), max(2, len(order) // 3))

    return HRPResult(
        weights=weights,
        cluster_order=order,
        n_assets=len(order),
        n_clusters=n_clusters,
    )


def cluster_assets(
    returns: np.ndarray,
    method: str = "single",
) -> tuple[list[int], np.ndarray]:
    """Hierarchical clustering of assets by correlation distance.

    Returns:
        (leaf_order, linkage_matrix).
    """
    corr = np.corrcoef(returns, rowvar=False)
    dist = _correlation_distance(corr)
    link = linkage(squareform(dist), method=method)
    order = list(leaves_list(link))
    return order, link


def quasi_diagonalise(
    cov: np.ndarray,
    order: list[int],
) -> np.ndarray:
    """Reorder covariance matrix by cluster structure."""
    return cov[np.ix_(order, order)]


# ---- Internal helpers ----

def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Distance matrix from correlation: d = √(0.5 × (1 − ρ))."""
    N = corr.shape[0]
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = math.sqrt(max(0.5 * (1 - corr[i, j]), 0))
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _recursive_bisection(cov: np.ndarray, indices: list[int]) -> np.ndarray:
    """Recursive bisection: split indices, allocate by inverse variance."""
    n = len(indices)
    weights = np.ones(n)

    if n <= 1:
        return weights

    # Split in half
    mid = n // 2
    left = indices[:mid]
    right = indices[mid:]

    # Cluster variance = 1'Σ⁻¹1 (inverse-variance of the cluster)
    var_left = _cluster_variance(cov, left)
    var_right = _cluster_variance(cov, right)

    # Allocate inversely proportional to variance
    total_inv = 1.0 / max(var_left, 1e-10) + 1.0 / max(var_right, 1e-10)
    alpha_left = (1.0 / max(var_left, 1e-10)) / total_inv
    alpha_right = 1.0 - alpha_left

    # Recurse
    w_left = _recursive_bisection(cov, left)
    w_right = _recursive_bisection(cov, right)

    # Scale
    result = np.zeros(n)
    result[:mid] = alpha_left * w_left
    result[mid:] = alpha_right * w_right

    return result


def _cluster_variance(cov: np.ndarray, indices: list[int]) -> float:
    """Variance of an inverse-variance weighted cluster."""
    sub_cov = cov[np.ix_(indices, indices)]
    # Inverse-variance weights within cluster
    diag = np.diag(sub_cov)
    inv_var = 1.0 / np.maximum(diag, 1e-10)
    w = inv_var / inv_var.sum()
    return float(w @ sub_cov @ w)
