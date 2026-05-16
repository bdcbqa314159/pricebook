"""Sparse grids (Smolyak quadrature) for high-dimensional integration.

Breaks the curse of dimensionality: O(N * log(N)^{d-1}) points vs O(N^d)
for full tensor product grids.
"""

from __future__ import annotations

import math
import itertools
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# 1D nested quadrature rules
# ---------------------------------------------------------------------------


def clenshaw_curtis_nodes(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Clenshaw-Curtis quadrature nodes and weights on [-1, 1].

    Nested: level k nodes are a subset of level k+1 nodes.
    Level 0: 1 point (midpoint), Level k >= 1: 2^k + 1 points.
    """
    if level == 0:
        return np.array([0.0]), np.array([2.0])

    n = 2 ** level
    nodes = -np.cos(np.pi * np.arange(n + 1) / n)

    # Vectorised CC weight computation
    k_arr = np.arange(n + 1)
    j_arr = np.arange(1, n // 2 + 1)
    b_arr = np.where(2 * j_arr == n, 1.0, 2.0)
    cos_matrix = np.cos(2.0 * j_arr[None, :] * k_arr[:, None] * np.pi / n)
    denom = 4.0 * j_arr * j_arr - 1.0
    weights = 1.0 - (cos_matrix * b_arr / denom).sum(axis=1)
    weights *= 2.0 / n
    weights[0] *= 0.5
    weights[-1] *= 0.5

    return nodes, weights


def _1d_rule(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Get the 1D quadrature rule at given level (0-indexed)."""
    return clenshaw_curtis_nodes(level)


def _num_points_1d(level: int) -> int:
    if level == 0:
        return 1
    return 2 ** level + 1


# ---------------------------------------------------------------------------
# Smolyak construction
# ---------------------------------------------------------------------------


def smolyak_grid(dim: int, level: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a Smolyak sparse grid on [-1, 1]^d.

    Uses the standard formula:
        A(q, d) = sum_{|i|<=q} (-1)^{q-|i|} * C(d-1, q-|i|) * (Q^{i_1} x ... x Q^{i_d})
    where |i| = i_1 + ... + i_d, i_j >= 1, and Q^k is the 1D rule at level k.

    Args:
        dim: number of dimensions.
        level: approximation level (higher = more accurate).

    Returns:
        (nodes, weights) where nodes is (n_points, dim) and weights is (n_points,).
    """
    q = level + dim  # Smolyak level parameter

    node_dict: dict[tuple, float] = {}

    # Iterate over multi-indices with |i| from dim to q, i_j >= 1
    for total in range(dim, q + 1):
        coeff = (-1) ** (q - total) * math.comb(dim - 1, q - total)
        if coeff == 0:
            continue

        # Generate all compositions of `total` into `dim` parts, each >= 1
        for comp in _compositions(total, dim):
            # Tensor product of 1D rules at levels (comp[j] - 1)
            rules_1d = [_1d_rule(comp[j] - 1) for j in range(dim)]
            node_lists = [r[0] for r in rules_1d]
            weight_lists = [r[1] for r in rules_1d]

            for combo in itertools.product(*[range(len(nl)) for nl in node_lists]):
                node = tuple(round(node_lists[j][combo[j]], 14) for j in range(dim))
                w = coeff
                for j in range(dim):
                    w *= weight_lists[j][combo[j]]
                node_dict[node] = node_dict.get(node, 0.0) + w

    nodes = np.array(list(node_dict.keys()))
    weights = np.array(list(node_dict.values()))

    return nodes, weights


def _compositions(n: int, k: int):
    """Generate all compositions of n into k parts, each >= 1."""
    if k == 1:
        yield (n,)
        return
    for first in range(1, n - k + 2):
        for rest in _compositions(n - first, k - 1):
            yield (first,) + rest


def sparse_grid_count(dim: int, level: int) -> int:
    """Number of points in the Smolyak sparse grid."""
    nodes, _ = smolyak_grid(dim, level)
    return len(nodes)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def sparse_grid_integrate(
    f: Callable,
    dim: int,
    level: int,
    bounds: list[tuple[float, float]] | None = None,
) -> float:
    """Integrate f over a hypercube using Smolyak sparse grids.

    Args:
        f: function taking a d-dimensional array and returning a scalar.
        dim: number of dimensions.
        level: approximation level.
        bounds: list of (low, high) for each dimension. Default [-1,1]^d.

    Returns:
        Approximation of the integral.
    """
    nodes, weights = smolyak_grid(dim, level)

    if bounds is not None:
        # Map from [-1, 1]^d to [a, b]^d
        scale = 1.0
        mapped = np.empty_like(nodes)
        for j in range(dim):
            a, b = bounds[j]
            mapped[:, j] = 0.5 * ((b - a) * nodes[:, j] + (a + b))
            scale *= 0.5 * (b - a)
        nodes = mapped
        weights = weights * scale

    result = 0.0
    for i in range(len(nodes)):
        result += weights[i] * f(nodes[i])

    return result
