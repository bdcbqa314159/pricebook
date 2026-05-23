"""Tests for information theory module."""

import pytest
import math
import numpy as np

from pricebook.statistics.information_theory import (
    shannon_entropy, differential_entropy, kl_divergence, js_divergence,
    cross_entropy, wasserstein_distance,
    mutual_information, conditional_mutual_information, information_gain,
    fisher_information_matrix, cramer_rao_bound, parameter_confidence_intervals,
)


class TestEntropy:
    def test_uniform(self):
        """Uniform distribution has max entropy."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        h = shannon_entropy(p)
        assert abs(h - math.log(4)) < 1e-10

    def test_certain(self):
        """Certain outcome has zero entropy."""
        p = np.array([1.0, 0.0, 0.0])
        h = shannon_entropy(p)
        assert abs(h) < 1e-10

    def test_binary(self):
        """Binary entropy at p=0.5."""
        p = np.array([0.5, 0.5])
        h = shannon_entropy(p)
        assert abs(h - math.log(2)) < 1e-10

    def test_differential_kde(self):
        """Normal(0,1) has differential entropy ≈ 0.5*ln(2πe) ≈ 1.42."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 5000)
        h = differential_entropy(x, method="kde")
        expected = 0.5 * math.log(2 * math.pi * math.e)
        assert abs(h - expected) < 0.3  # within 0.3 nats


class TestDivergence:
    def test_kl_identical_zero(self):
        p = np.array([0.3, 0.7])
        assert abs(kl_divergence(p, p)) < 1e-10

    def test_kl_positive(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        assert kl_divergence(p, q) > 0

    def test_kl_asymmetric(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        assert kl_divergence(p, q) != kl_divergence(q, p)

    def test_js_symmetric(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        assert abs(js_divergence(p, q) - js_divergence(q, p)) < 1e-10

    def test_js_bounded(self):
        """JS divergence ≤ ln(2)."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert js_divergence(p, q) <= math.log(2) + 1e-10

    def test_cross_entropy(self):
        p = np.array([0.5, 0.5])
        h = shannon_entropy(p)
        kl = kl_divergence(p, p)
        ce = cross_entropy(p, p)
        assert abs(ce - (h + kl)) < 1e-10

    def test_wasserstein(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        w = wasserstein_distance(p, q)
        assert w > 0


class TestMutualInformation:
    def test_independent(self):
        """Independent variables should have near-zero MI."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(0, 1, 1000)
        mi = mutual_information(x, y)
        assert mi < 0.3  # histogram estimator has positive bias

    def test_dependent(self):
        """Highly dependent variables should have high MI."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = x + rng.normal(0, 0.1, 1000)
        mi = mutual_information(x, y)
        assert mi > 0.5

    def test_information_gain(self):
        """Relevant feature should rank higher."""
        rng = np.random.default_rng(42)
        n = 500
        target = rng.choice([0, 1], n)
        relevant = target + rng.normal(0, 0.3, n)
        irrelevant = rng.normal(0, 1, n)
        features = np.column_stack([relevant, irrelevant])
        ranked = information_gain(features, target, ["relevant", "irrelevant"])
        assert ranked[0]["feature"] == "relevant"


class TestFisherInformation:
    def test_normal_mean(self):
        """FIM for Normal mean with known σ=1: FIM = N/σ² = N."""
        rng = np.random.default_rng(42)
        data = rng.normal(3.0, 1.0, 100)

        def log_lik(params):
            mu = params[0]
            return float(np.sum(-0.5 * (data - mu)**2))

        fim = fisher_information_matrix(log_lik, np.array([3.0]))
        # FIM ≈ N = 100 (for σ=1)
        assert abs(fim[0, 0] - 100) < 10

    def test_crb(self):
        """CRB for Normal mean: Var(μ̂) ≥ 1/N."""
        fim = np.array([[100.0]])
        crb = cramer_rao_bound(fim)
        assert abs(crb[0] - 0.01) < 0.001

    def test_confidence_intervals(self):
        fim = np.array([[100.0]])
        ci = parameter_confidence_intervals(fim, np.array([3.0]))
        assert len(ci) == 1
        assert ci[0]["ci_lower"] < 3.0 < ci[0]["ci_upper"]
        assert ci[0]["std_error"] > 0

    def test_2d(self):
        """2D FIM for Normal(μ, σ²)."""
        rng = np.random.default_rng(42)
        data = rng.normal(2.0, 0.5, 200)

        def log_lik(params):
            mu, sigma = params[0], max(params[1], 1e-6)
            return float(np.sum(-0.5 * np.log(2 * np.pi * sigma**2)
                                - 0.5 * ((data - mu) / sigma)**2))

        fim = fisher_information_matrix(log_lik, np.array([2.0, 0.5]))
        assert fim.shape == (2, 2)
        crb = cramer_rao_bound(fim)
        assert all(c > 0 for c in crb)
