"""Tests for clustering: kmeans, hierarchical, optimal_k, HMM."""
import pytest
import numpy as np
from pricebook.statistics.clustering import (
    kmeans, silhouette_score, optimal_k, hierarchical_cluster, HMMRegime,
)


class TestKMeans:
    def test_two_clusters(self):
        rng = np.random.default_rng(42)
        data = np.vstack([rng.normal(0, 0.5, (50, 2)), rng.normal(5, 0.5, (50, 2))])
        result = kmeans(data, k=2)
        assert len(result.labels) == 100
        assert len(set(result.labels)) == 2

    def test_single_cluster(self):
        data = np.random.default_rng(42).normal(0, 1, (30, 2))
        result = kmeans(data, k=1)
        assert all(l == 0 for l in result.labels)


class TestSilhouette:
    def test_well_separated(self):
        rng = np.random.default_rng(42)
        data = np.vstack([rng.normal(0, 0.1, (50, 2)), rng.normal(10, 0.1, (50, 2))])
        labels = [0]*50 + [1]*50
        score = silhouette_score(data, labels)
        assert score > 0.8


class TestOptimalK:
    def test_finds_two(self):
        rng = np.random.default_rng(42)
        data = np.vstack([rng.normal(0, 0.3, (50, 2)), rng.normal(5, 0.3, (50, 2))])
        k = optimal_k(data, max_k=5)
        assert k == 2


class TestHierarchical:
    def test_hierarchical(self):
        rng = np.random.default_rng(42)
        data = np.vstack([rng.normal(0, 0.5, (30, 2)), rng.normal(5, 0.5, (30, 2))])
        result = hierarchical_cluster(data, n_clusters=2)
        assert len(result.labels) == 60
        assert len(set(result.labels)) == 2


class TestHMM:
    def test_two_regimes(self):
        rng = np.random.default_rng(42)
        returns = np.concatenate([rng.normal(0, 0.01, 100), rng.normal(0, 0.05, 100)])
        hmm = HMMRegime(n_regimes=2)
        result = hmm.fit(returns)
        assert hasattr(result, "labels")
        assert len(result.labels) == 200
