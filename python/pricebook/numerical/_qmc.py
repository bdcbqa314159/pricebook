"""Quasi-Monte Carlo sequences and sparse grids.

Low-discrepancy sequences for high-dimensional integration — converge
at O(1/N) vs O(1/√N) for standard MC.

    from pricebook.numerical._qmc import (
        sobol_sequence, halton_sequence, latin_hypercube,
        sparse_grid, SparseGridResult,
    )

References:
    Glasserman (2003). Monte Carlo Methods in Financial Engineering.
    Smolyak (1963). Quadrature and Interpolation Formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sobol_sequence(n_points: int, n_dims: int, seed: int = 0) -> np.ndarray:
    """Sobol quasi-random sequence on [0, 1]^d.

    Uses scipy.stats.qmc if available, otherwise falls back to
    scrambled Halton.

    Args:
        n_points: number of points.
        n_dims: number of dimensions.
        seed: for scrambling.

    Returns:
        (n_points, n_dims) array of quasi-random points in [0, 1]^d.
    """
    try:
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=n_dims, scramble=True, seed=seed)
        # Sobol requires power-of-2 points
        m = int(np.ceil(np.log2(max(n_points, 2))))
        points = sampler.random(2**m)
        return points[:n_points]
    except ImportError:
        return halton_sequence(n_points, n_dims)


def halton_sequence(n_points: int, n_dims: int) -> np.ndarray:
    """Halton quasi-random sequence on [0, 1]^d.

    Uses the first d primes as bases.
    """
    primes = _first_primes(n_dims)
    points = np.zeros((n_points, n_dims))
    for j in range(n_dims):
        points[:, j] = _halton_1d(n_points, primes[j])
    return points


def latin_hypercube(n_points: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """Latin Hypercube Sampling on [0, 1]^d.

    Stratified sampling: each dimension divided into n_points equal strata,
    exactly one sample per stratum per dimension.
    """
    rng = np.random.default_rng(seed)
    points = np.zeros((n_points, n_dims))
    for j in range(n_dims):
        perm = rng.permutation(n_points)
        points[:, j] = (perm + rng.uniform(size=n_points)) / n_points
    return points


# ═══════════════════════════════════════════════════════════════
# Sparse grids (Smolyak)
# ═══════════════════════════════════════════════════════════════


@dataclass
class SparseGridResult:
    """Result of sparse grid quadrature."""
    nodes: np.ndarray            # (N, d) quadrature nodes
    weights: np.ndarray          # (N,) quadrature weights
    n_points: int
    n_dims: int
    level: int

    def integrate(self, f: callable) -> float:
        """Integrate f using the sparse grid."""
        return float(sum(w * f(x) for w, x in zip(self.weights, self.nodes)))

    def to_dict(self) -> dict:
        return {"n_points": self.n_points, "n_dims": self.n_dims, "level": self.level}


def sparse_grid(
    n_dims: int,
    level: int = 3,
    rule: str = "gauss_legendre",
) -> SparseGridResult:
    """Smolyak sparse grid quadrature on [0, 1]^d.

    Combines 1D quadrature rules via the Smolyak formula:
        Q^d_l = Σ_{|k|≤l+d-1} (-1)^{l+d-1-|k|} × C(d-1, l+d-1-|k|) × (Q^1_{k1} ⊗ ... ⊗ Q^1_{kd})

    Reduces O(n^d) points to O(n × log^{d-1}(n)) for smooth integrands.

    Args:
        n_dims: number of dimensions.
        level: accuracy level (higher = more points, more accurate).
        rule: 1D quadrature rule ("gauss_legendre" or "clenshaw_curtis").
    """
    from itertools import product
    from math import comb

    # 1D nested quadrature rules at each level
    def get_1d_rule(lev):
        if rule == "clenshaw_curtis":
            n = max(2 * lev + 1, 1)
            nodes = 0.5 * (1 - np.cos(np.pi * np.arange(n) / max(n - 1, 1)))
            weights = np.ones(n) / n  # simplified
        else:
            from numpy.polynomial.legendre import leggauss
            n = max(lev + 1, 1)
            x, w = leggauss(n)
            nodes = 0.5 * (x + 1)  # map to [0, 1]
            weights = 0.5 * w
        return nodes, weights

    # Smolyak construction
    all_nodes = []
    all_weights = []

    for k_sum in range(max(level, 0), level + n_dims):
        # Generate all multi-indices k with |k| = k_sum, k_i ≥ 0
        sign = (-1) ** (level + n_dims - 1 - k_sum)
        coeff = comb(n_dims - 1, level + n_dims - 1 - k_sum)
        if coeff == 0:
            continue

        # For each valid multi-index
        for k in _multiindex_sum(n_dims, k_sum):
            rules = [get_1d_rule(ki) for ki in k]
            # Tensor product
            node_lists = [r[0] for r in rules]
            weight_lists = [r[1] for r in rules]

            for combo in product(*[range(len(n)) for n in node_lists]):
                node = np.array([node_lists[d][combo[d]] for d in range(n_dims)])
                weight = sign * coeff
                for d in range(n_dims):
                    weight *= weight_lists[d][combo[d]]
                all_nodes.append(node)
                all_weights.append(weight)

    if not all_nodes:
        # Fallback: single center point
        all_nodes = [np.full(n_dims, 0.5)]
        all_weights = [1.0]

    nodes = np.array(all_nodes)
    weights = np.array(all_weights)

    # Merge duplicate nodes
    nodes, weights = _merge_nodes(nodes, weights)

    return SparseGridResult(
        nodes=nodes, weights=weights,
        n_points=len(nodes), n_dims=n_dims, level=level,
    )


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════


def _halton_1d(n: int, base: int) -> np.ndarray:
    """1D Halton sequence with given base."""
    result = np.zeros(n)
    for i in range(n):
        f = 1.0
        r = 0.0
        idx = i + 1
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        result[i] = r
    return result


def _first_primes(n: int) -> list[int]:
    """Return first n primes."""
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes


def _multiindex_sum(d: int, s: int) -> list[tuple]:
    """Generate all d-tuples of non-negative integers summing to s."""
    if d == 1:
        return [(s,)]
    result = []
    for k in range(s + 1):
        for rest in _multiindex_sum(d - 1, s - k):
            result.append((k,) + rest)
    return result


def _merge_nodes(nodes: np.ndarray, weights: np.ndarray, tol: float = 1e-12) -> tuple:
    """Merge duplicate nodes by summing their weights."""
    n = len(nodes)
    if n == 0:
        return nodes, weights

    # Round to merge near-duplicates
    rounded = np.round(nodes / tol) * tol
    unique_map = {}
    for i in range(n):
        key = tuple(rounded[i])
        if key in unique_map:
            unique_map[key][1] += weights[i]
        else:
            unique_map[key] = [nodes[i].copy(), weights[i]]

    merged_nodes = np.array([v[0] for v in unique_map.values()])
    merged_weights = np.array([v[1] for v in unique_map.values()])
    return merged_nodes, merged_weights
