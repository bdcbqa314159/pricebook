"""Tests for linear algebra tools: PCA, eigen, SVD, condition numbers."""

import numpy as np
import pytest

from pricebook.linalg import (
    ConditionReport,
    EigenResult,
    PCAResult,
    SVDResult,
    condition_number,
    eigendecomposition,
    explained_variance,
    pca,
    svd_decomposition,
)


# ---- PCA ----

class TestPCA:
    def _yield_data(self):
        """Synthetic yield curve data: level + slope + noise."""
        np.random.seed(42)
        n = 100
        tenors = 10
        level = np.random.randn(n, 1) * 0.01   # big factor
        slope = np.random.randn(n, 1) * 0.005   # smaller factor
        tenor_weights = np.linspace(1, 1, tenors).reshape(1, -1)
        slope_weights = np.linspace(-1, 1, tenors).reshape(1, -1)
        noise = np.random.randn(n, tenors) * 0.0005
        return level * tenor_weights + slope * slope_weights + noise

    def test_top3_explain_most_variance(self):
        """L3 convergence: top 3 PCs explain >90% of yield curve variance."""
        data = self._yield_data()
        result = pca(data, n_components=3)
        assert result.cumulative_ratio[-1] > 0.90

    def test_components_orthogonal(self):
        data = self._yield_data()
        result = pca(data)
        # Components should be orthonormal
        gram = result.components @ result.components.T
        np.testing.assert_allclose(gram, np.eye(result.n_components), atol=1e-10)

    def test_eigenvalues_non_negative(self):
        data = self._yield_data()
        result = pca(data)
        assert np.all(result.eigenvalues >= 0)

    def test_explained_ratio_sums_to_one(self):
        data = self._yield_data()
        result = pca(data)
        assert result.explained_ratio.sum() == pytest.approx(1.0, abs=1e-10)

    def test_project_and_reconstruct(self):
        data = self._yield_data()
        result = pca(data, n_components=10)
        scores = result.project(data)
        reconstructed = result.reconstruct(scores)
        # Full PCA: perfect reconstruction
        np.testing.assert_allclose(reconstructed, data, atol=1e-10)

    def test_n_components_limits(self):
        data = self._yield_data()
        result = pca(data, n_components=2)
        assert result.n_components == 2
        assert result.components.shape == (2, 10)

    def test_single_sample(self):
        data = [[1.0, 2.0, 3.0]]
        result = pca(data)
        assert result.n_components == 3


# ---- Eigendecomposition ----

class TestEigendecomposition:
    def test_identity(self):
        result = eigendecomposition(np.eye(3))
        np.testing.assert_allclose(result.eigenvalues, [1, 1, 1], atol=1e-10)
        assert result.is_positive_definite

    def test_pd_check(self):
        pd = [[4, 2], [2, 3]]
        result = eigendecomposition(pd)
        assert result.is_positive_definite
        assert np.all(result.eigenvalues > 0)

    def test_not_pd(self):
        not_pd = [[1, 2], [2, 1]]  # eigenvalues: 3, -1
        result = eigendecomposition(not_pd)
        assert not result.is_positive_definite

    def test_sorted_descending(self):
        A = [[5, 1], [1, 2]]
        result = eigendecomposition(A)
        assert result.eigenvalues[0] >= result.eigenvalues[1]

    def test_eigenvectors_orthonormal(self):
        A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
        result = eigendecomposition(A)
        gram = result.eigenvectors.T @ result.eigenvectors
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)


# ---- SVD ----

class TestSVD:
    def test_identity(self):
        result = svd_decomposition(np.eye(3))
        np.testing.assert_allclose(result.singular_values, [1, 1, 1], atol=1e-10)
        assert result.rank == 3

    def test_rank_deficient(self):
        A = [[1, 2], [2, 4]]  # rank 1
        result = svd_decomposition(A)
        assert result.rank == 1

    def test_reconstruction(self):
        A = np.array([[1, 2, 3], [4, 5, 6]])
        result = svd_decomposition(A)
        reconstructed = result.U @ np.diag(result.singular_values) @ result.Vt
        np.testing.assert_allclose(reconstructed, A, atol=1e-10)

    def test_rectangular(self):
        A = np.random.randn(5, 3)
        result = svd_decomposition(A)
        assert result.U.shape == (5, 3)
        assert result.Vt.shape == (3, 3)


# ---- Condition number ----

class TestConditionNumber:
    def test_identity_well_conditioned(self):
        report = condition_number(np.eye(3))
        assert report.condition_number == pytest.approx(1.0)
        assert report.is_well_conditioned
        assert not report.is_ill_conditioned

    def test_ill_conditioned(self):
        A = np.array([[1, 1], [1, 1 + 1e-14]])
        report = condition_number(A)
        assert report.is_ill_conditioned
        assert "unreliable" in report.recommendation

    def test_near_singular_ill_conditioned(self):
        A = [[1, 2], [2, 4]]  # rank 1 — numerically near-singular
        report = condition_number(A)
        assert report.condition_number > 1e12
        assert report.is_ill_conditioned


# ---- Explained variance ----

class TestExplainedVariance:
    def test_threshold_95(self):
        evals = [10, 5, 2, 1, 0.5, 0.1]
        n = explained_variance(evals, threshold=0.95)
        total = sum(evals)
        assert sum(evals[:n]) / total >= 0.95
        assert sum(evals[:n-1]) / total < 0.95 if n > 1 else True

    def test_single_dominant(self):
        evals = [100, 0.01, 0.001]
        assert explained_variance(evals, 0.95) == 1

    def test_all_equal(self):
        evals = [1, 1, 1, 1]
        # Need all 4 to reach 100%
        assert explained_variance(evals, 1.0) == 4

    def test_zero_eigenvalues(self):
        assert explained_variance([0, 0, 0]) == 0
