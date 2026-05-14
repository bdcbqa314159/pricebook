"""Clustering and regime detection: K-means, hierarchical, HMM.

    from pricebook.clustering import kmeans, hierarchical_cluster, optimal_k, HMMRegime

References:
    Lloyd (1982). Least Squares Quantization in PCM (K-means).
    Ward (1963). Hierarchical Grouping.
    Baum & Petrie (1966). Statistical Inference for Probabilistic Functions of Markov Chains (HMM).
    Hamilton (1989). A New Approach to the Economic Analysis of Nonstationary Time Series (Regime switching).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ═══════════════════════════════════════════════════════════════
# K-means
# ═══════════════════════════════════════════════════════════════

@dataclass
class KMeansResult:
    """K-means clustering result."""
    centroids: np.ndarray     # (k, n_features)
    labels: np.ndarray        # (n_samples,)
    inertia: float            # sum of squared distances to nearest centroid
    n_iter: int
    k: int

    def to_dict(self) -> dict:
        return {"k": self.k, "inertia": self.inertia, "n_iter": self.n_iter}


def kmeans(
    data: np.ndarray,
    k: int,
    max_iter: int = 100,
    seed: int = 42,
) -> KMeansResult:
    """K-means clustering (Lloyd's algorithm).

    Args:
        data: (n_samples, n_features) or (n_samples,) for 1D.
        k: number of clusters.
        max_iter: maximum iterations.
        seed: random seed for centroid initialisation.
    """
    X = np.atleast_2d(data)
    if X.shape[0] == 1:
        X = X.T
    n, d = X.shape
    rng = np.random.default_rng(seed)

    # Initialise centroids (random selection from data)
    idx = rng.choice(n, size=k, replace=False)
    centroids = X[idx].copy()

    labels = np.zeros(n, dtype=int)
    for iteration in range(max_iter):
        # Assign each point to nearest centroid
        distances = np.zeros((n, k))
        for j in range(k):
            distances[:, j] = np.sum((X - centroids[j]) ** 2, axis=1)
        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(new_labels, labels) and iteration > 0:
            break
        labels = new_labels

        # Update centroids
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)

    inertia = sum(np.sum((X[labels == j] - centroids[j]) ** 2) for j in range(k))

    return KMeansResult(centroids, labels, float(inertia), iteration + 1, k)


def silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    """Average silhouette score (-1 to 1, higher is better)."""
    X = np.atleast_2d(data)
    if X.shape[0] == 1:
        X = X.T
    n = len(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0

    scores = np.zeros(n)
    for i in range(n):
        # a(i) = mean distance to points in same cluster
        same = X[labels == labels[i]]
        a = np.mean(np.sqrt(np.sum((same - X[i]) ** 2, axis=1))) if len(same) > 1 else 0

        # b(i) = min mean distance to points in other clusters
        b = float('inf')
        for c in unique:
            if c == labels[i]:
                continue
            other = X[labels == c]
            b = min(b, np.mean(np.sqrt(np.sum((other - X[i]) ** 2, axis=1))))

        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0

    return float(np.mean(scores))


def optimal_k(
    data: np.ndarray,
    max_k: int = 10,
    method: str = "silhouette",
    seed: int = 42,
) -> int:
    """Find optimal number of clusters.

    Args:
        method: 'silhouette' (best score) or 'elbow' (largest inertia drop).
    """
    X = np.atleast_2d(data)
    if X.shape[0] == 1:
        X = X.T
    max_k = min(max_k, len(X))

    if method == "silhouette":
        best_k, best_score = 2, -1
        for k in range(2, max_k + 1):
            r = kmeans(X, k, seed=seed)
            s = silhouette_score(X, r.labels)
            if s > best_score:
                best_k, best_score = k, s
        return best_k

    elif method == "elbow":
        inertias = []
        for k in range(1, max_k + 1):
            r = kmeans(X, k, seed=seed)
            inertias.append(r.inertia)
        # Find largest relative drop
        drops = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        return int(np.argmax(drops)) + 2  # +2 because drop[i] is from k=i+1 to k=i+2

    raise ValueError(f"unknown method: {method!r}")


# ═══════════════════════════════════════════════════════════════
# Hierarchical clustering
# ═══════════════════════════════════════════════════════════════

@dataclass
class HierarchicalResult:
    """Hierarchical clustering result."""
    linkage: np.ndarray       # (n-1, 4) linkage matrix
    labels: np.ndarray        # cluster labels at optimal cut
    n_clusters: int

    def to_dict(self) -> dict:
        return {"n_clusters": self.n_clusters}


def hierarchical_cluster(
    data: np.ndarray,
    n_clusters: int = 3,
) -> HierarchicalResult:
    """Agglomerative hierarchical clustering (Ward linkage).

    Uses scipy.cluster.hierarchy for the heavy lifting.
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    X = np.atleast_2d(data)
    if X.shape[0] == 1:
        X = X.T

    Z = linkage(X, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1  # 0-indexed

    return HierarchicalResult(Z, labels, n_clusters)


# ═══════════════════════════════════════════════════════════════
# Hidden Markov Model (Gaussian emissions)
# ═══════════════════════════════════════════════════════════════

@dataclass
class HMMResult:
    """HMM fit result."""
    means: np.ndarray          # (n_regimes,) emission means
    vols: np.ndarray           # (n_regimes,) emission stds
    transition_matrix: np.ndarray  # (n_regimes, n_regimes)
    stationary_probs: np.ndarray   # (n_regimes,) long-run probabilities
    log_likelihood: float
    n_iter: int
    labels: np.ndarray         # (T,) most likely regime sequence (Viterbi)
    regime_probs: np.ndarray   # (T, n_regimes) filtered regime probabilities

    def to_dict(self) -> dict:
        return {
            "means": self.means.tolist(),
            "vols": self.vols.tolist(),
            "transition_matrix": self.transition_matrix.tolist(),
            "stationary_probs": self.stationary_probs.tolist(),
            "log_likelihood": self.log_likelihood,
            "n_iter": self.n_iter,
        }


class HMMRegime:
    """Hidden Markov Model for financial regime switching.

    Gaussian emissions: y_t | s_t=k ~ N(mu_k, sigma_k^2)
    Markov transitions: P(s_t=j | s_{t-1}=i) = A[i,j]

    Fitted via Baum-Welch (EM algorithm).

    Args:
        n_regimes: number of hidden states (typically 2: low-vol, high-vol).
    """

    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes

    def fit(
        self,
        returns: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6,
        seed: int = 42,
    ) -> HMMResult:
        """Fit HMM via Baum-Welch (EM).

        Returns HMMResult with estimated parameters and regime labels.
        """
        r = np.asarray(returns, dtype=float)
        T = len(r)
        K = self.n_regimes
        rng = np.random.default_rng(seed)

        # Initialise: sort data, assign regimes by quantile
        sorted_r = np.sort(r)
        quantiles = [sorted_r[int(T * (i + 0.5) / K)] for i in range(K)]
        means = np.array(quantiles)
        vols = np.full(K, np.std(r) / K)
        A = np.full((K, K), 1.0 / K)  # uniform transitions
        np.fill_diagonal(A, 0.9)       # sticky regimes
        A /= A.sum(axis=1, keepdims=True)
        pi = np.ones(K) / K            # initial distribution

        prev_ll = -np.inf

        for iteration in range(max_iter):
            # E-step: forward-backward
            # Emission probabilities
            B = np.zeros((T, K))
            for k in range(K):
                B[:, k] = _norm_pdf_array(r, means[k], vols[k])
            B = np.maximum(B, 1e-300)

            # Forward
            alpha = np.zeros((T, K))
            alpha[0] = pi * B[0]
            scale = np.zeros(T)
            scale[0] = alpha[0].sum()
            alpha[0] /= max(scale[0], 1e-300)

            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ A) * B[t]
                scale[t] = alpha[t].sum()
                alpha[t] /= max(scale[t], 1e-300)

            # Backward
            beta = np.zeros((T, K))
            beta[-1] = 1.0

            for t in range(T - 2, -1, -1):
                beta[t] = A @ (B[t + 1] * beta[t + 1])
                beta[t] /= max(scale[t + 1], 1e-300)

            # Posterior
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True).clip(1e-300)

            # Log-likelihood
            ll = np.sum(np.log(np.maximum(scale, 1e-300)))
            if ll - prev_ll < tol and iteration > 0:
                break
            prev_ll = ll

            # M-step
            for k in range(K):
                wk = gamma[:, k]
                wk_sum = wk.sum()
                if wk_sum < 1e-10:
                    continue
                means[k] = np.dot(wk, r) / wk_sum
                vols[k] = math.sqrt(np.dot(wk, (r - means[k]) ** 2) / wk_sum)
                vols[k] = max(vols[k], 1e-6)

            # Transition matrix
            xi = np.zeros((K, K))
            for t in range(T - 1):
                for i in range(K):
                    for j in range(K):
                        xi[i, j] += alpha[t, i] * A[i, j] * B[t + 1, j] * beta[t + 1, j] / max(scale[t + 1], 1e-300)
            A = xi / xi.sum(axis=1, keepdims=True).clip(1e-10)
            pi = gamma[0]

        # Sort regimes by volatility (low-vol = regime 0)
        order = np.argsort(vols)
        means = means[order]
        vols = vols[order]
        A = A[np.ix_(order, order)]

        # Viterbi (most likely sequence)
        labels = np.argmax(gamma[:, order], axis=1)

        # Stationary distribution
        try:
            eigvals, eigvecs = np.linalg.eig(A.T)
            idx = np.argmin(np.abs(eigvals - 1.0))
            stat = np.abs(eigvecs[:, idx])
            stat /= stat.sum()
        except np.linalg.LinAlgError:
            stat = np.ones(K) / K

        return HMMResult(
            means=means,
            vols=vols,
            transition_matrix=A,
            stationary_probs=stat,
            log_likelihood=float(ll),
            n_iter=iteration + 1,
            labels=labels,
            regime_probs=gamma[:, order],
        )


def _norm_pdf_array(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Vectorised normal PDF."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
