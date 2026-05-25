"""Graph algorithms — pure numpy, no external dependencies.

    from pricebook.numerical._graph import (
        dijkstra, minimum_spanning_tree, max_flow, connected_components,
        ShortestPathResult, MSTResult, MaxFlowResult,
    )

References:
    Cormen et al. (2009). Introduction to Algorithms, 3rd ed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ShortestPathResult:
    """Result of shortest-path computation."""
    distances: np.ndarray
    predecessors: np.ndarray
    source: int

    def path_to(self, target: int) -> list[int]:
        """Reconstruct path from source to target."""
        if np.isinf(self.distances[target]):
            return []
        path = [target]
        current = target
        while self.predecessors[current] != -1:
            current = self.predecessors[current]
            path.append(current)
        return list(reversed(path))

    def to_dict(self) -> dict:
        return {"source": self.source,
                "n_reachable": int(np.sum(~np.isinf(self.distances))),
                "max_distance": float(np.max(self.distances[~np.isinf(self.distances)]))}


@dataclass
class MSTResult:
    """Result of minimum spanning tree computation."""
    edges: list[tuple[int, int, float]]
    total_weight: float

    def to_dict(self) -> dict:
        return {"n_edges": len(self.edges), "total_weight": self.total_weight}


@dataclass
class MaxFlowResult:
    """Result of maximum flow computation."""
    max_flow: float
    residual: np.ndarray

    def to_dict(self) -> dict:
        return {"max_flow": self.max_flow}


def dijkstra(
    adj_matrix: np.ndarray,
    source: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Dijkstra's shortest path from source.

    Args:
        adj_matrix: (N, N) weight matrix. 0 or inf = no edge.
        source: source node index.

    Returns:
        (distances, predecessors) — both (N,) arrays.
    """
    n = adj_matrix.shape[0]
    dist = np.full(n, np.inf)
    pred = np.full(n, -1, dtype=int)
    dist[source] = 0.0
    visited = np.zeros(n, dtype=bool)

    for _ in range(n):
        # Find unvisited node with minimum distance
        unvisited_dist = np.where(visited, np.inf, dist)
        u = int(np.argmin(unvisited_dist))
        if dist[u] == np.inf:
            break
        visited[u] = True

        for v in range(n):
            w = adj_matrix[u, v]
            if w > 0 and not np.isinf(w) and not visited[v]:
                alt = dist[u] + w
                if alt < dist[v]:
                    dist[v] = alt
                    pred[v] = u

    return dist, pred


def dijkstra_full(
    adj_matrix: np.ndarray,
    source: int,
) -> ShortestPathResult:
    """Dijkstra's shortest path returning a ShortestPathResult."""
    dist, pred = dijkstra(adj_matrix, source)
    return ShortestPathResult(dist, pred, source)


def shortest_path(
    adj_matrix: np.ndarray,
    source: int,
    target: int,
) -> list[int]:
    """Reconstruct shortest path from source to target."""
    dist, pred = dijkstra(adj_matrix, source)
    if np.isinf(dist[target]):
        return []
    path = [target]
    current = target
    while pred[current] != -1:
        current = pred[current]
        path.append(current)
    return list(reversed(path))


def minimum_spanning_tree(
    weight_matrix: np.ndarray,
) -> list[tuple[int, int, float]]:
    """Prim's MST algorithm.

    Args:
        weight_matrix: (N, N) symmetric weight matrix. 0 = no edge.

    Returns:
        List of (u, v, weight) edges in the MST.
    """
    n = weight_matrix.shape[0]
    in_mst = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    key[0] = 0.0

    edges = []
    for _ in range(n):
        # Pick minimum key node not in MST
        candidates = np.where(~in_mst, key, np.inf)
        u = int(np.argmin(candidates))
        if key[u] == np.inf:
            break
        in_mst[u] = True
        if parent[u] != -1:
            edges.append((int(parent[u]), u, float(key[u])))

        for v in range(n):
            w = weight_matrix[u, v]
            if w > 0 and not np.isinf(w) and not in_mst[v] and w < key[v]:
                key[v] = w
                parent[v] = u

    return edges


def minimum_spanning_tree_full(
    weight_matrix: np.ndarray,
) -> MSTResult:
    """Prim's MST returning an MSTResult."""
    edges = minimum_spanning_tree(weight_matrix)
    total = sum(e[2] for e in edges)
    return MSTResult(edges, total)


def max_flow(
    capacity: np.ndarray,
    source: int,
    sink: int,
) -> float:
    """Ford-Fulkerson max-flow with BFS (Edmonds-Karp).

    Args:
        capacity: (N, N) capacity matrix.
        source: source node.
        sink: sink node.

    Returns:
        Maximum flow value.
    """
    n = capacity.shape[0]
    residual = capacity.copy().astype(float)
    total_flow = 0.0

    while True:
        # BFS to find augmenting path
        parent = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        visited[source] = True
        queue = [source]

        while queue:
            u = queue.pop(0)
            for v in range(n):
                if not visited[v] and residual[u, v] > 1e-10:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                    if v == sink:
                        break

        if not visited[sink]:
            break  # no more augmenting paths

        # Find bottleneck
        path_flow = np.inf
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u, v])
            v = u

        # Update residual
        v = sink
        while v != source:
            u = parent[v]
            residual[u, v] -= path_flow
            residual[v, u] += path_flow
            v = u

        total_flow += path_flow

    return total_flow


def max_flow_full(
    capacity: np.ndarray,
    source: int,
    sink: int,
) -> MaxFlowResult:
    """Ford-Fulkerson max-flow returning a MaxFlowResult."""
    n = capacity.shape[0]
    residual = capacity.copy().astype(float)
    total_flow = 0.0

    while True:
        parent = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        visited[source] = True
        queue = [source]

        while queue:
            u = queue.pop(0)
            for v in range(n):
                if not visited[v] and residual[u, v] > 1e-10:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                    if v == sink:
                        break

        if not visited[sink]:
            break

        path_flow = np.inf
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u, v])
            v = u

        v = sink
        while v != source:
            u = parent[v]
            residual[u, v] -= path_flow
            residual[v, u] += path_flow
            v = u

        total_flow += path_flow

    return MaxFlowResult(total_flow, residual)


def connected_components(adj_matrix: np.ndarray) -> np.ndarray:
    """Find connected components via BFS.

    Args:
        adj_matrix: (N, N) adjacency matrix (nonzero = edge).

    Returns:
        (N,) array of component labels (0-indexed).
    """
    n = adj_matrix.shape[0]
    labels = np.full(n, -1, dtype=int)
    component = 0

    for start in range(n):
        if labels[start] != -1:
            continue
        # BFS
        queue = [start]
        labels[start] = component
        while queue:
            u = queue.pop(0)
            for v in range(n):
                if (adj_matrix[u, v] > 0 or adj_matrix[v, u] > 0) and labels[v] == -1:
                    labels[v] = component
                    queue.append(v)
        component += 1

    return labels
