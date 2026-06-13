"""Regression for L2 Wave-2 audit — `risk.network` used the
classic ``np.where`` eager-evaluation pattern, emitting spurious
``RuntimeWarning: divide by zero`` on sparse adjacency matrices.

Pre-fix:

    dist_matrix = np.where(self.adj > 0, 1.0 / self.adj, 0.0)

NumPy evaluates BOTH branches before the where-mask selects, so
``1.0 / 0`` is computed for every zero adjacency entry and emits the
warning.  The where-mask discards the result but the NaN/inf could
propagate via other operations.

Post-fix uses ``np.divide(..., where=...)`` so the division only fires
where the predicate is true — no spurious warning, no NaN/inf in the
output array.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pricebook.risk.network import FinancialNetwork


class TestNetworkBetweennessNoWarning:
    def test_no_divide_by_zero_warning(self):
        nodes = ["A", "B", "C"]
        adj = np.array([[0.0, 1.0, 0.0],
                        [2.0, 0.0, 0.5],
                        [0.0, 0.0, 0.0]])
        n = FinancialNetwork(nodes, adj)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            bc = n.betweenness_centrality()
        assert set(bc.keys()) == {"A", "B", "C"}

    def test_returns_finite_centralities(self):
        nodes = ["A", "B", "C", "D"]
        adj = np.array([
            [0.0, 1.0, 0.0, 2.0],
            [0.5, 0.0, 1.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.5, 0.0, 0.0],
        ])
        n = FinancialNetwork(nodes, adj)
        bc = n.betweenness_centrality()
        for v in bc.values():
            import math as _m
            assert _m.isfinite(v)
