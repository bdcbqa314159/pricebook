"""Tests for sparse matrix operations."""

import pytest
import numpy as np

from pricebook.sparse import (
    SparseMatrix,
    sparse_solve,
    sparse_lu,
    sparse_cg,
    tridiagonal_matrix,
)


class TestSparseMatrix:
    def test_construction(self):
        sm = SparseMatrix(3, 3)
        sm.add(0, 0, 2.0)
        sm.add(1, 1, 3.0)
        sm.add(2, 2, 4.0)
        assert sm.shape == (3, 3)
        assert sm.nnz == 3

    def test_to_dense(self):
        sm = SparseMatrix(2, 2)
        sm.add(0, 0, 1.0)
        sm.add(0, 1, 2.0)
        sm.add(1, 0, 3.0)
        sm.add(1, 1, 4.0)
        expected = np.array([[1, 2], [3, 4]], dtype=float)
        np.testing.assert_array_almost_equal(sm.to_dense(), expected)

    def test_duplicate_sums(self):
        sm = SparseMatrix(2, 2)
        sm.add(0, 0, 1.0)
        sm.add(0, 0, 2.0)
        assert sm.to_dense()[0, 0] == pytest.approx(3.0)

    def test_matmul_vector(self):
        sm = SparseMatrix(2, 2)
        sm.add(0, 0, 2.0)
        sm.add(1, 1, 3.0)
        x = np.array([1.0, 2.0])
        result = sm @ x
        np.testing.assert_array_almost_equal(result, [2.0, 6.0])

    def test_matmul_sparse(self):
        A = SparseMatrix(2, 2)
        A.add(0, 0, 1.0)
        A.add(0, 1, 2.0)
        A.add(1, 0, 3.0)
        A.add(1, 1, 4.0)

        B = SparseMatrix.identity(2)
        C = A @ B
        np.testing.assert_array_almost_equal(C.to_dense(), A.to_dense())

    def test_add(self):
        A = SparseMatrix(2, 2)
        A.add(0, 0, 1.0)
        A.add(1, 1, 2.0)

        B = SparseMatrix(2, 2)
        B.add(0, 0, 3.0)
        B.add(1, 1, 4.0)

        C = A + B
        assert C.to_dense()[0, 0] == pytest.approx(4.0)
        assert C.to_dense()[1, 1] == pytest.approx(6.0)

    def test_transpose(self):
        sm = SparseMatrix(2, 3)
        sm.add(0, 1, 5.0)
        sm.add(1, 2, 7.0)
        t = sm.transpose()
        assert t.shape == (3, 2)
        assert t.to_dense()[1, 0] == pytest.approx(5.0)
        assert t.to_dense()[2, 1] == pytest.approx(7.0)

    def test_identity(self):
        I = SparseMatrix.identity(3)
        np.testing.assert_array_almost_equal(I.to_dense(), np.eye(3))

    def test_add_dense(self):
        sm = SparseMatrix(4, 4)
        block = np.array([[1.0, 2.0], [3.0, 4.0]])
        sm.add_dense(1, 1, block)
        d = sm.to_dense()
        assert d[1, 1] == pytest.approx(1.0)
        assert d[2, 2] == pytest.approx(4.0)


class TestSparseSolvers:
    def _spd_matrix(self, n=5):
        """Create a positive definite sparse matrix."""
        sm = SparseMatrix(n, n)
        for i in range(n):
            sm.add(i, i, 4.0)
            if i > 0:
                sm.add(i, i - 1, -1.0)
            if i < n - 1:
                sm.add(i, i + 1, -1.0)
        return sm

    def test_sparse_solve(self):
        A = self._spd_matrix()
        b = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        x = sparse_solve(A, b)
        np.testing.assert_array_almost_equal(A @ x, b)

    def test_sparse_lu(self):
        A = self._spd_matrix()
        b = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        lu = sparse_lu(A)
        x = lu.solve(b)
        np.testing.assert_array_almost_equal(A @ x, b)

    def test_lu_repeated_solves(self):
        A = self._spd_matrix()
        lu = sparse_lu(A)
        for _ in range(5):
            b = np.random.randn(5)
            x = lu.solve(b)
            np.testing.assert_array_almost_equal(A @ x, b, decimal=10)

    def test_sparse_cg(self):
        A = self._spd_matrix()
        b = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        x = sparse_cg(A, b)
        np.testing.assert_array_almost_equal(A @ x, b, decimal=8)

    def test_solve_matches_dense(self):
        A = self._spd_matrix(10)
        b = np.random.randn(10)
        x_sparse = sparse_solve(A, b)
        x_dense = np.linalg.solve(A.to_dense(), b)
        np.testing.assert_array_almost_equal(x_sparse, x_dense)


class TestTridiagonal:
    def test_tridiagonal(self):
        n = 4
        lower = np.array([-1.0, -1.0, -1.0])
        diag = np.array([2.0, 2.0, 2.0, 2.0])
        upper = np.array([-1.0, -1.0, -1.0])
        T = tridiagonal_matrix(n, lower, diag, upper)
        assert T.shape == (n, n)
        d = T.to_dense()
        assert d[0, 0] == pytest.approx(2.0)
        assert d[0, 1] == pytest.approx(-1.0)
        assert d[1, 0] == pytest.approx(-1.0)
        assert d[3, 2] == pytest.approx(-1.0)

    def test_tridiagonal_solve(self):
        n = 100
        lower = -np.ones(n - 1)
        diag = 4.0 * np.ones(n)
        upper = -np.ones(n - 1)
        T = tridiagonal_matrix(n, lower, diag, upper)
        b = np.ones(n)
        x = sparse_solve(T, b)
        np.testing.assert_array_almost_equal(T @ x, b)

    def test_sparse_vs_dense_performance(self):
        """Sparse solve should work for large systems."""
        n = 500
        lower = -np.ones(n - 1)
        diag = 4.0 * np.ones(n)
        upper = -np.ones(n - 1)
        T = tridiagonal_matrix(n, lower, diag, upper)
        b = np.random.randn(n)
        x = sparse_solve(T, b)
        residual = np.max(np.abs(T @ x - b))
        assert residual < 1e-10
