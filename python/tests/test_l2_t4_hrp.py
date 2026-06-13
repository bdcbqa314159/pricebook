"""Regression for L2 phase-2 audit of `risk.hierarchical_risk_parity`:

(a) N=1 crashed.  Pre-fix code path: corr matrix is 1×1, distance
    matrix 1×1 zero, squareform produces empty vector, linkage on
    empty → ValueError.  Now returns trivial weights=[1.0] directly.

(b) ``n_clusters`` reported was the heuristic ``min(N, max(2, N//3))``
    — unrelated to actual cluster structure.  Now computed via
    ``fcluster`` at the median linkage height, so the count
    reflects the dendrogram.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.risk.hierarchical_risk_parity import hrp_portfolio


class TestHRPSingleAsset:
    def test_n_equals_one(self):
        returns = np.random.default_rng(42).standard_normal((100, 1))
        r = hrp_portfolio(returns)
        assert r.n_assets == 1
        assert r.weights.shape == (1,)
        assert r.weights[0] == pytest.approx(1.0, abs=1e-12)
        assert r.n_clusters == 1


class TestHRPClusterCount:
    def test_cluster_count_from_linkage(self):
        """For genuinely-clustered returns, n_clusters reflects structure."""
        rng = np.random.default_rng(11)
        T = 500
        # Three clusters of correlated assets.
        block1 = rng.standard_normal((T, 1)) + 0.1 * rng.standard_normal((T, 3))
        block2 = rng.standard_normal((T, 1)) + 0.1 * rng.standard_normal((T, 3))
        block3 = rng.standard_normal((T, 1)) + 0.1 * rng.standard_normal((T, 4))
        # Repeat the cluster signal so each asset within block is correlated.
        a = block1 + 0.05 * rng.standard_normal((T, 3))
        b = block2 + 0.05 * rng.standard_normal((T, 3))
        c = block3 + 0.05 * rng.standard_normal((T, 4))
        returns = np.hstack([a, b, c])
        r = hrp_portfolio(returns)
        # We expect 2 or more clusters since blocks are distinct.
        assert r.n_clusters >= 2
        # Bound: ≤ N (trivially) and ≤ N - 1 (since the root joins everything).
        assert r.n_clusters <= 10  # N = 10

    def test_uniform_assets_few_clusters(self):
        """Highly correlated assets → few clusters."""
        rng = np.random.default_rng(7)
        T = 300
        common = rng.standard_normal((T, 1))
        # 5 assets, all = common + tiny noise → all in one cluster.
        returns = np.repeat(common, 5, axis=1) + 0.001 * rng.standard_normal((T, 5))
        r = hrp_portfolio(returns)
        # With near-perfect correlations, n_clusters can be 1 or 2.
        assert r.n_clusters <= 3


class TestHRPWeightInvariants:
    """Algorithmic invariants — must hold across the fix."""

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        returns = rng.standard_normal((200, 8))
        r = hrp_portfolio(returns)
        assert r.weights.sum() == pytest.approx(1.0, abs=1e-10)

    def test_weights_non_negative(self):
        rng = np.random.default_rng(42)
        returns = rng.standard_normal((200, 8))
        r = hrp_portfolio(returns)
        assert np.all(r.weights >= 0)
