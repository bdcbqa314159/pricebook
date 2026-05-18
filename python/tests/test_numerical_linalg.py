"""Tests for numerical._linalg."""
import pytest, numpy as np
from pricebook.numerical._linalg import expm, qr, cholesky, gmres, sylvester

class TestExpm:
    def test_identity(self):
        result = expm(np.zeros((2,2)))
        assert np.allclose(result, np.eye(2), atol=1e-10)

class TestQR:
    def test_factorisation(self):
        A = np.random.default_rng(42).normal(0, 1, (3, 3))
        result = qr(A)
        assert np.allclose(result.Q @ result.R, A, atol=1e-10)

class TestCholesky:
    def test_positive_definite(self):
        A = np.array([[4, 2], [2, 3]], dtype=float)
        L = cholesky(A)
        assert np.allclose(L @ L.T, A, atol=1e-10)

class TestGMRES:
    def test_callable(self):
        assert callable(gmres)

class TestSylvester:
    def test_solves(self):
        A = np.array([[1, 2], [0, 3]], dtype=float)
        B = np.array([[4, 0], [1, 2]], dtype=float)
        C = np.array([[5, 6], [7, 8]], dtype=float)
        X = sylvester(A, B, C)
        assert np.allclose(A @ X + X @ B, C, atol=1e-8)
