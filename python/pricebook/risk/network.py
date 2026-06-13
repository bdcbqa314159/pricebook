"""Financial network — counterparty importance and systemic risk.

    from pricebook.risk.network import (
        FinancialNetwork, SystemicRiskScore, NetworkResult,
    )

References:
    Eisenberg & Noe (2001). Systemic Risk in Financial Systems. Management Science.
    Battiston et al. (2012). DebtRank. Scientific Reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NetworkResult:
    """Centrality metrics for all nodes."""
    nodes: list[str]
    degree_centrality: dict[str, float]
    betweenness_centrality: dict[str, float]
    eigenvector_centrality: dict[str, float]
    pagerank: dict[str, float]

    def systemic_ranking(self) -> list[dict]:
        """Rank nodes by composite systemic score."""
        scores = {}
        for node in self.nodes:
            scores[node] = (
                self.degree_centrality[node] * 0.2
                + self.betweenness_centrality[node] * 0.3
                + self.eigenvector_centrality[node] * 0.2
                + self.pagerank[node] * 0.3
            )
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [{"node": n, "score": s, "rank": i + 1} for i, (n, s) in enumerate(ranked)]

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "degree_centrality": self.degree_centrality,
            "pagerank": self.pagerank,
            "systemic_ranking": self.systemic_ranking(),
        }


class FinancialNetwork:
    """Weighted directed financial network.

    Nodes = counterparties/institutions. Edges = exposures.

    Args:
        nodes: list of node names.
        adjacency: (N, N) weighted adjacency matrix.
            adj[i][j] = exposure of i to j (i lends to j).
    """

    def __init__(self, nodes: list[str], adjacency: np.ndarray):
        if len(nodes) != adjacency.shape[0]:
            raise ValueError("nodes and adjacency must match")
        self.nodes = nodes
        self.adj = np.asarray(adjacency, dtype=float)
        self.n = len(nodes)

    def degree_centrality(self) -> dict[str, float]:
        """Normalised degree centrality (in + out)."""
        out_deg = (self.adj > 0).sum(axis=1)
        in_deg = (self.adj > 0).sum(axis=0)
        total = (out_deg + in_deg) / max(2 * (self.n - 1), 1)
        return {self.nodes[i]: float(total[i]) for i in range(self.n)}

    def betweenness_centrality(self) -> dict[str, float]:
        """Approximate betweenness centrality via shortest paths."""
        from pricebook.numerical._graph import dijkstra

        # Convert to distance (inverse weight).
        # Fix T4-NET1: pre-fix used ``np.where(self.adj > 0, 1.0 / self.adj,
        # 0.0)`` which evaluates ``1.0 / self.adj`` EAGERLY for every
        # element — including zero entries — emitting RuntimeWarning
        # ("divide by zero").  The where-mask then correctly discards
        # those values, but the warning is real (and the masked NaN /
        # inf could propagate via other operations).  Use ``np.divide``
        # with the ``where`` argument so the division is only performed
        # where the predicate is true.
        dist_matrix = np.zeros_like(self.adj, dtype=float)
        np.divide(1.0, self.adj, out=dist_matrix, where=(self.adj > 0))
        bc = np.zeros(self.n)

        for s in range(self.n):
            dist, pred = dijkstra(dist_matrix, s)
            for t in range(self.n):
                if s == t or np.isinf(dist[t]):
                    continue
                # Trace path and increment betweenness
                v = t
                while pred[v] != -1 and pred[v] != s:
                    bc[pred[v]] += 1
                    v = pred[v]

        # Normalise
        denom = max((self.n - 1) * (self.n - 2), 1)
        bc /= denom
        return {self.nodes[i]: float(bc[i]) for i in range(self.n)}

    def eigenvector_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
        """Eigenvector centrality via power iteration."""
        x = np.ones(self.n) / self.n
        A = self.adj + self.adj.T  # symmetrise for eigenvector
        for _ in range(max_iter):
            x_new = A @ x
            norm = np.linalg.norm(x_new)
            if norm > 0:
                x_new /= norm
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        return {self.nodes[i]: float(x[i]) for i in range(self.n)}

    def pagerank(self, damping: float = 0.85, max_iter: int = 100) -> dict[str, float]:
        """PageRank centrality."""
        A = self.adj.copy()
        out_degree = A.sum(axis=1)
        # Normalise columns (transition matrix)
        M = np.zeros_like(A)
        for i in range(self.n):
            if out_degree[i] > 0:
                M[i, :] = A[i, :] / out_degree[i]
            else:
                M[i, :] = 1.0 / self.n  # dangling node

        pr = np.ones(self.n) / self.n
        for _ in range(max_iter):
            pr_new = (1 - damping) / self.n + damping * (M.T @ pr)
            if np.linalg.norm(pr_new - pr) < 1e-8:
                break
            pr = pr_new

        return {self.nodes[i]: float(pr[i]) for i in range(self.n)}

    def compute_all(self) -> NetworkResult:
        """Compute all centrality metrics."""
        return NetworkResult(
            nodes=self.nodes,
            degree_centrality=self.degree_centrality(),
            betweenness_centrality=self.betweenness_centrality(),
            eigenvector_centrality=self.eigenvector_centrality(),
            pagerank=self.pagerank(),
        )

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "n_nodes": self.n,
            "n_edges": int((self.adj > 0).sum()),
            "total_exposure": float(self.adj.sum()),
        }
