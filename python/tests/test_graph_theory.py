"""Tests for graph theory: algorithms, network, contagion, correlation network."""

import pytest
import numpy as np

from pricebook.numerical._graph import (
    dijkstra, shortest_path, minimum_spanning_tree, max_flow, connected_components,
)
from pricebook.risk.network import FinancialNetwork, NetworkResult
from pricebook.risk.contagion import DefaultCascade, CascadeResult
from pricebook.risk.correlation_network import (
    correlation_to_distance, mst_portfolio, hierarchical_risk_parity,
    MSTResult, HRPResult,
)


# ═══════════════════════════════════════════════════════════════
# 4.3: Graph Algorithms
# ═══════════════════════════════════════════════════════════════

class TestDijkstra:
    def test_simple(self):
        adj = np.array([[0, 1, 4], [0, 0, 2], [0, 0, 0]], dtype=float)
        dist, pred = dijkstra(adj, 0)
        assert dist[0] == 0
        assert dist[1] == 1
        assert dist[2] == 3  # 0→1→2 costs 1+2=3

    def test_shortest_path(self):
        adj = np.array([[0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=float)
        path = shortest_path(adj, 0, 3)
        assert path == [0, 1, 2, 3]

    def test_unreachable(self):
        adj = np.array([[0, 0], [0, 0]], dtype=float)
        path = shortest_path(adj, 0, 1)
        assert path == []


class TestMST:
    def test_triangle(self):
        w = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=float)
        edges = minimum_spanning_tree(w)
        assert len(edges) == 2
        total = sum(e[2] for e in edges)
        assert total == 3  # 1 + 2


class TestMaxFlow:
    def test_simple(self):
        cap = np.array([[0, 10, 10], [0, 0, 10], [0, 0, 0]], dtype=float)
        flow = max_flow(cap, 0, 2)
        assert flow == 20  # two paths: 0→1→2 (10) + 0→2 (10)


class TestConnectedComponents:
    def test_two_components(self):
        adj = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float)
        labels = connected_components(adj)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]


# ═══════════════════════════════════════════════════════════════
# 4.1: Financial Network
# ═══════════════════════════════════════════════════════════════

class TestFinancialNetwork:
    @pytest.fixture
    def network(self):
        nodes = ["BankA", "BankB", "BankC", "BankD"]
        adj = np.array([
            [0, 100, 50, 0],
            [80, 0, 0, 60],
            [0, 30, 0, 40],
            [20, 0, 10, 0],
        ], dtype=float)
        return FinancialNetwork(nodes, adj)

    def test_degree(self, network):
        dc = network.degree_centrality()
        assert all(0 <= v <= 1 for v in dc.values())

    def test_pagerank(self, network):
        pr = network.pagerank()
        assert abs(sum(pr.values()) - 1.0) < 0.01

    def test_eigenvector(self, network):
        ec = network.eigenvector_centrality()
        assert all(v >= 0 for v in ec.values())

    def test_compute_all(self, network):
        result = network.compute_all()
        assert isinstance(result, NetworkResult)
        ranking = result.systemic_ranking()
        assert len(ranking) == 4
        assert ranking[0]["rank"] == 1

    def test_to_dict(self, network):
        d = network.to_dict()
        assert d["n_nodes"] == 4


# ═══════════════════════════════════════════════════════════════
# 4.2: Contagion
# ═══════════════════════════════════════════════════════════════

class TestContagion:
    @pytest.fixture
    def cascade(self):
        nodes = ["A", "B", "C", "D"]
        exposures = np.array([
            [0, 50, 30, 0],
            [40, 0, 0, 20],
            [0, 10, 0, 15],
            [5, 0, 5, 0],
        ], dtype=float)
        buffers = np.array([20, 15, 10, 5], dtype=float)
        return DefaultCascade(nodes, exposures, buffers)

    def test_single_default(self, cascade):
        r = cascade.simulate(["A"])
        assert isinstance(r, CascadeResult)
        assert "A" in r.final_defaults

    def test_cascade_propagates(self, cascade):
        """Default of a large node should cascade."""
        r = cascade.simulate(["A"])
        assert len(r.final_defaults) >= 1

    def test_no_cascade(self):
        """With huge buffers, no cascade."""
        nodes = ["A", "B"]
        exp = np.array([[0, 10], [10, 0]], dtype=float)
        buf = np.array([1000, 1000], dtype=float)
        dc = DefaultCascade(nodes, exp, buf)
        r = dc.simulate(["A"])
        assert len(r.final_defaults) == 1

    def test_contagion_multiplier(self, cascade):
        r = cascade.simulate(["A"])
        assert r.contagion_multiplier() >= 0

    def test_stress_test(self, cascade):
        scenarios = [["A"], ["B"], ["A", "B"]]
        results = cascade.stress_test(scenarios)
        assert len(results) == 3

    def test_to_dict(self, cascade):
        d = cascade.to_dict()
        assert d["n_nodes"] == 4


# ═══════════════════════════════════════════════════════════════
# 4.4: Correlation Network
# ═══════════════════════════════════════════════════════════════

class TestCorrelationNetwork:
    def test_distance_properties(self):
        corr = np.array([[1.0, 0.8, -0.2], [0.8, 1.0, 0.3], [-0.2, 0.3, 1.0]])
        dist = correlation_to_distance(corr)
        assert np.allclose(np.diag(dist), 0)  # self-distance = 0
        assert dist[0, 1] < dist[0, 2]  # high corr → low distance

    def test_mst_portfolio(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, (100, 5))
        result = mst_portfolio(returns, ["A", "B", "C", "D", "E"])
        assert isinstance(result, MSTResult)
        assert len(result.edges) == 4  # MST has N-1 edges

    def test_hrp(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, (200, 4))
        result = hierarchical_risk_parity(returns, ["A", "B", "C", "D"])
        assert isinstance(result, HRPResult)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-8
        assert all(w > 0 for w in result.weights.values())

    def test_hrp_diversified(self):
        """HRP should not concentrate in one asset."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, (200, 5))
        result = hierarchical_risk_parity(returns)
        assert max(result.weights.values()) < 0.5  # no single asset > 50%
