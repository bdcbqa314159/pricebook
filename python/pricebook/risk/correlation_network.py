"""Correlation network — MST portfolio, HRP, community detection.

    from pricebook.risk.correlation_network import (
        correlation_to_distance, mst_portfolio,
        hierarchical_risk_parity, community_detection,
    )

References:
    Mantegna (1999). Hierarchical Structure in Financial Markets.
    López de Prado (2016). Building Diversified Portfolios that Outperform OOS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pricebook.numerical._graph import minimum_spanning_tree


def correlation_to_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix.

    d(i,j) = √(2(1 - ρ(i,j)))   (Mantegna 1999)

    ρ = 1 → d = 0 (identical)
    ρ = 0 → d = √2 (independent)
    ρ = -1 → d = 2 (opposite)
    """
    corr = np.asarray(corr)
    return np.sqrt(2 * (1 - np.clip(corr, -1, 1)))


@dataclass
class MSTResult:
    """Minimum spanning tree of asset correlations."""
    edges: list[tuple[int, int, float]]
    asset_names: list[str]
    total_distance: float
    avg_correlation: float

    def to_dict(self) -> dict:
        return {
            "n_assets": len(self.asset_names),
            "n_edges": len(self.edges),
            "total_distance": self.total_distance,
            "avg_correlation": self.avg_correlation,
        }


def mst_portfolio(
    returns: np.ndarray,
    asset_names: list[str] | None = None,
) -> MSTResult:
    """Build MST from return correlations.

    Args:
        returns: (T, N) return matrix.
        asset_names: optional names for assets.
    """
    corr = np.corrcoef(returns, rowvar=False)
    dist = correlation_to_distance(corr)
    n = dist.shape[0]

    if asset_names is None:
        asset_names = [f"asset_{i}" for i in range(n)]

    edges = minimum_spanning_tree(dist)
    total_dist = sum(w for _, _, w in edges)
    avg_corr = float(np.mean(corr[np.triu_indices(n, k=1)]))

    return MSTResult(
        edges=edges,
        asset_names=asset_names,
        total_distance=total_dist,
        avg_correlation=avg_corr,
    )


@dataclass
class HRPResult:
    """Hierarchical Risk Parity allocation result."""
    weights: dict[str, float]
    asset_names: list[str]
    cluster_order: list[int]

    def to_dict(self) -> dict:
        return {"weights": self.weights}


def hierarchical_risk_parity(
    returns: np.ndarray,
    asset_names: list[str] | None = None,
) -> HRPResult:
    """López de Prado (2016) Hierarchical Risk Parity.

    Steps:
    1. Compute correlation distance matrix
    2. Build hierarchical clustering (single-linkage)
    3. Quasi-diagonalise covariance
    4. Recursive bisection: allocate inversely proportional to cluster variance
    """
    cov = np.cov(returns, rowvar=False)
    corr = np.corrcoef(returns, rowvar=False)
    n = cov.shape[0]

    if asset_names is None:
        asset_names = [f"asset_{i}" for i in range(n)]

    # Step 1-2: hierarchical clustering via distance
    dist = correlation_to_distance(corr)
    order = _quasi_diag(dist)

    # Step 3-4: recursive bisection
    weights = np.ones(n)
    _recursive_bisection(cov, weights, order)

    # Normalise
    weights /= weights.sum()

    return HRPResult(
        weights={asset_names[i]: float(weights[i]) for i in range(n)},
        asset_names=asset_names,
        cluster_order=order,
    )


def community_detection(
    adj_matrix: np.ndarray,
    n_communities: int = 3,
) -> list[set[int]]:
    """Spectral clustering on graph Laplacian.

    Args:
        adj_matrix: (N, N) adjacency/similarity matrix.
        n_communities: number of communities.

    Returns:
        List of sets of node indices.
    """
    n = adj_matrix.shape[0]
    A = np.abs(adj_matrix)
    D = np.diag(A.sum(axis=1))
    L = D - A  # unnormalised Laplacian

    # Eigenvectors of L (smallest eigenvalues)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Take first n_communities eigenvectors (skip the trivial one)
    X = eigenvectors[:, :n_communities]

    # K-means on the embedded coordinates
    from pricebook.statistics.clustering import kmeans
    labels = kmeans(X, n_communities).labels

    communities = [set() for _ in range(n_communities)]
    for i, label in enumerate(labels):
        communities[label].add(i)

    return [c for c in communities if c]


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════


def _quasi_diag(dist: np.ndarray) -> list[int]:
    """Quasi-diagonalise via single-linkage hierarchical clustering leaves.

    Fix T4-RISK23: pre-fix used a greedy nearest-neighbour tour
    starting at index 0 — NOT López de Prado's quasi-diagonalisation.
    Greedy NN produces a path-like ordering that doesn't preserve
    cluster structure; subsequent recursive bisection then splits
    cluster-mates apart, defeating the whole point of HRP.

    Real LdP quasi-diagonalisation: run hierarchical clustering on
    the distance matrix and use the dendrogram-leaves order, so
    adjacent assets in the ordering are cluster-mates.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    n = dist.shape[0]
    if n <= 1:
        return list(range(n))
    # condensed-form distance vector for scipy
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    return list(leaves_list(link))


def _recursive_bisection(cov, weights, order):
    """Recursive bisection for HRP weight allocation."""
    if len(order) <= 1:
        return

    mid = len(order) // 2
    left = order[:mid]
    right = order[mid:]

    # Cluster variance
    var_left = _cluster_var(cov, left)
    var_right = _cluster_var(cov, right)

    # Allocate inversely proportional to variance
    alpha = 1 - var_left / (var_left + var_right) if (var_left + var_right) > 0 else 0.5

    for i in left:
        weights[i] *= alpha
    for i in right:
        weights[i] *= (1 - alpha)

    _recursive_bisection(cov, weights, left)
    _recursive_bisection(cov, weights, right)


def _cluster_var(cov, indices):
    """Variance of an equal-weight cluster."""
    sub_cov = cov[np.ix_(indices, indices)]
    n = len(indices)
    w = np.ones(n) / n
    return float(w @ sub_cov @ w)
