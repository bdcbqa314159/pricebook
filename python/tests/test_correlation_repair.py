"""Tests for correlation matrix repair."""

import numpy as np
import pytest

from pricebook.correlation_repair import (
    NearestCorrResult,
    correlation_interpolation,
    eigenvalue_floor,
    is_positive_definite,
    is_valid_correlation,
    nearest_correlation_matrix,
)


def _broken_corr():
    """A non-PD 'correlation' matrix (from inconsistent market data)."""
    return np.array([
        [1.0,  0.9,  0.7],
        [0.9,  1.0, -0.4],
        [0.7, -0.4,  1.0],
    ])


def _valid_corr():
    return np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ])


# ---- Checks ----

class TestChecks:
    def test_valid_correlation(self):
        assert is_valid_correlation(_valid_corr())

    def test_pd_identity(self):
        assert is_positive_definite(np.eye(3))

    def test_not_pd(self):
        assert not is_positive_definite(_broken_corr())

    def test_not_symmetric(self):
        A = np.array([[1, 0.5], [0.3, 1]])
        assert not is_valid_correlation(A)

    def test_diagonal_not_one(self):
        A = np.array([[0.9, 0.5], [0.5, 1.0]])
        assert not is_valid_correlation(A)


# ---- Eigenvalue floor ----

class TestEigenvalueFloor:
    def test_repairs_to_pd(self):
        repaired = eigenvalue_floor(_broken_corr())
        assert is_positive_definite(repaired)

    def test_unit_diagonal(self):
        repaired = eigenvalue_floor(_broken_corr())
        np.testing.assert_allclose(np.diag(repaired), 1.0, atol=1e-10)

    def test_symmetric(self):
        repaired = eigenvalue_floor(_broken_corr())
        np.testing.assert_allclose(repaired, repaired.T, atol=1e-10)

    def test_valid_corr_unchanged(self):
        original = _valid_corr()
        repaired = eigenvalue_floor(original, floor=1e-10)
        # Should be very close to original since it's already PD
        np.testing.assert_allclose(repaired, original, atol=1e-6)


# ---- Nearest correlation matrix (Higham) ----

class TestNearestCorrelationMatrix:
    def test_result_is_pd(self):
        """Repaired matrix is positive-definite."""
        result = nearest_correlation_matrix(_broken_corr())
        assert result.is_pd

    def test_unit_diagonal(self):
        result = nearest_correlation_matrix(_broken_corr())
        np.testing.assert_allclose(np.diag(result.matrix), 1.0, atol=1e-10)

    def test_symmetric(self):
        result = nearest_correlation_matrix(_broken_corr())
        np.testing.assert_allclose(result.matrix, result.matrix.T, atol=1e-10)

    def test_closest_in_frobenius(self):
        """Nearest corr has smaller Frobenius distance than eigenvalue floor."""
        broken = _broken_corr()
        higham = nearest_correlation_matrix(broken)
        floor = eigenvalue_floor(broken)
        frob_floor = np.linalg.norm(floor - broken, 'fro')
        assert higham.frobenius_distance <= frob_floor + 1e-8

    def test_valid_input_unchanged(self):
        original = _valid_corr()
        result = nearest_correlation_matrix(original)
        np.testing.assert_allclose(result.matrix, original, atol=1e-8)
        assert result.frobenius_distance < 1e-8

    def test_converges(self):
        result = nearest_correlation_matrix(_broken_corr(), max_iter=200)
        assert result.iterations < 200

    def test_entries_in_range(self):
        result = nearest_correlation_matrix(_broken_corr())
        assert np.all(result.matrix >= -1.0 - 1e-10)
        assert np.all(result.matrix <= 1.0 + 1e-10)

    def test_4x4_broken(self):
        """Larger broken matrix."""
        A = np.array([
            [1.0, 0.9, 0.8, 0.7],
            [0.9, 1.0, 0.9, 0.8],
            [0.8, 0.9, 1.0, 0.9],
            [0.7, 0.8, 0.9, 1.0],
        ])
        # Make it non-PD by tweaking
        A[0, 3] = A[3, 0] = -0.5
        result = nearest_correlation_matrix(A)
        assert result.is_pd


# ---- Interpolation ----

class TestCorrelationInterpolation:
    def test_weight_zero_returns_a(self):
        A = _valid_corr()
        B = np.eye(3)
        result = correlation_interpolation(A, B, 0.0)
        np.testing.assert_allclose(result, A, atol=1e-8)

    def test_weight_one_returns_b(self):
        A = _valid_corr()
        B = np.eye(3)
        result = correlation_interpolation(A, B, 1.0)
        np.testing.assert_allclose(result, B, atol=1e-8)

    def test_mid_interpolation_is_pd(self):
        A = _valid_corr()
        B = np.eye(3)
        result = correlation_interpolation(A, B, 0.5)
        assert is_positive_definite(result)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-10)
