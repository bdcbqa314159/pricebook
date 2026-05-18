"""Tests for regression."""
import pytest
import numpy as np
from pricebook.statistics.regression import (
    ols, ridge, lasso, elastic_net, quantile_regression, robust_regression,
)


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 2))
    y = 3.0 + 2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.1, 100)
    return X, y


class TestOLS:
    def test_recovers_coefficients(self, linear_data):
        X, y = linear_data
        r = ols(X, y)
        assert abs(r.intercept - 3.0) < 0.3
        assert abs(r.coefficients[0] - 2.0) < 0.3
        assert abs(r.coefficients[1] - (-1.5)) < 0.3

    def test_r_squared(self, linear_data):
        X, y = linear_data
        assert ols(X, y).r_squared > 0.95

    def test_residuals(self, linear_data):
        X, y = linear_data
        assert len(ols(X, y).residuals) == 100


class TestRidge:
    def test_shrinkage(self, linear_data):
        X, y = linear_data
        ols_norm = np.sum(ols(X, y).coefficients ** 2)
        ridge_norm = np.sum(ridge(X, y, alpha=10.0).coefficients ** 2)
        assert ridge_norm <= ols_norm + 0.1


class TestLasso:
    def test_sparsity(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 5))
        y = 2.0 * X[:, 0] + rng.normal(0, 0.1, 100)
        r = lasso(X, y, alpha=0.5)
        near_zero = sum(1 for c in r.coefficients if abs(c) < 0.1)
        assert near_zero >= 2


class TestElasticNet:
    def test_runs(self, linear_data):
        X, y = linear_data
        r = elastic_net(X, y, alpha=0.1, l1_ratio=0.5)
        assert hasattr(r, "coefficients")


class TestQuantile:
    def test_median(self, linear_data):
        X, y = linear_data
        r = quantile_regression(X, y, quantile=0.5)
        assert hasattr(r, "coefficients")


class TestRobust:
    def test_outlier_resistance(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 1))
        y = 2.0 * X[:, 0] + rng.normal(0, 0.1, 100)
        y[0] = 100.0
        r = robust_regression(X, y)
        assert abs(r.coefficients[0] - 2.0) < 0.5
