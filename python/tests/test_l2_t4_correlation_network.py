"""Regression for L2 phase-2 audit of `risk.correlation_network`:

`_quasi_diag` was a greedy nearest-neighbour tour starting at index 0,
NOT López de Prado's hierarchical-cluster-leaves order.  Greedy NN
produces a path-like sequence that ignores cluster structure;
subsequent recursive bisection then mixes cluster-mates together,
defeating HRP's whole point.

Fix: use scipy single-linkage clustering + ``leaves_list`` to match
the correct LdP quasi-diagonalisation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.risk.correlation_network import (
    hierarchical_risk_parity, _quasi_diag,
)


class TestQuasiDiagBlockStructure:
    def test_clustered_assets_grouped_together(self):
        """Two well-separated clusters → leaves of cluster A should be
        adjacent in the ordering, separately from cluster B."""
        # Cluster A: assets 0, 1 — distance 0.1 between them.
        # Cluster B: assets 2, 3 — distance 0.1 between them.
        # Between clusters: distance 1.5.
        dist = np.array([
            [0.0, 0.1, 1.5, 1.6],
            [0.1, 0.0, 1.6, 1.5],
            [1.5, 1.6, 0.0, 0.1],
            [1.6, 1.5, 0.1, 0.0],
        ])
        order = _quasi_diag(dist)
        # Cluster A members {0, 1} should be adjacent; same for {2, 3}.
        # Equivalent: in any traversal, the set of *positions* of {0,1} should
        # be consecutive, and so for {2,3}.
        positions = {asset: idx for idx, asset in enumerate(order)}
        a_positions = sorted([positions[0], positions[1]])
        b_positions = sorted([positions[2], positions[3]])
        # Adjacency check: positions differ by exactly 1.
        assert a_positions[1] - a_positions[0] == 1
        assert b_positions[1] - b_positions[0] == 1


class TestHRPCorrelationNetwork:
    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        returns = rng.standard_normal((300, 6))
        result = hierarchical_risk_parity(returns)
        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_weights_non_negative(self):
        rng = np.random.default_rng(42)
        returns = rng.standard_normal((300, 6))
        result = hierarchical_risk_parity(returns)
        assert all(w >= 0 for w in result.weights.values())

    def test_block_structure_preserved_in_weights(self):
        """Two clusters of 3 assets each → cluster_order groups them."""
        rng = np.random.default_rng(7)
        T = 500
        # Block A: 3 assets sharing a common factor.
        common_a = rng.standard_normal((T, 1))
        block_a = np.repeat(common_a, 3, axis=1) + 0.01 * rng.standard_normal((T, 3))
        # Block B: 3 assets with their own factor.
        common_b = rng.standard_normal((T, 1))
        block_b = np.repeat(common_b, 3, axis=1) + 0.01 * rng.standard_normal((T, 3))
        returns = np.hstack([block_a, block_b])

        result = hierarchical_risk_parity(returns, asset_names=[f"a_{i}" for i in range(6)])
        # cluster_order should place assets 0-2 together and 3-5 together
        # (or in some symmetric grouping).
        order = result.cluster_order
        a_positions = sorted([order.index(i) for i in range(3)])
        b_positions = sorted([order.index(i) for i in range(3, 6)])
        # Each block's positions should be consecutive (positions 0,1,2 or 3,4,5).
        assert a_positions[2] - a_positions[0] == 2
        assert b_positions[2] - b_positions[0] == 2


class TestQuasiDiagDegenerate:
    def test_n_one(self):
        dist = np.array([[0.0]])
        order = _quasi_diag(dist)
        assert order == [0]

    def test_n_two(self):
        dist = np.array([[0.0, 1.0], [1.0, 0.0]])
        order = _quasi_diag(dist)
        assert sorted(order) == [0, 1]
