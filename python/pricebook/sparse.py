"""Sparse matrix operations for PDE assembly and portfolio risk."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg


class SparseMatrix:
    """Thin wrapper around scipy.sparse CSR matrix.

    Constructed from triplets (row, col, value) for easy assembly.
    """

    def __init__(self, rows: int, cols: int):
        self._rows = rows
        self._cols = cols
        self._row_idx: list[int] = []
        self._col_idx: list[int] = []
        self._values: list[float] = []
        self._mat: sparse.csr_matrix | None = None

    def add(self, row: int, col: int, value: float) -> None:
        """Add a value at (row, col). Duplicates are summed."""
        self._row_idx.append(row)
        self._col_idx.append(col)
        self._values.append(value)
        self._mat = None  # invalidate cache

    def add_dense(self, row_start: int, col_start: int, dense: np.ndarray) -> None:
        """Add a dense submatrix at the given offset."""
        nr, nc = dense.shape
        for i in range(nr):
            for j in range(nc):
                if dense[i, j] != 0.0:
                    self.add(row_start + i, col_start + j, dense[i, j])

    @property
    def shape(self) -> tuple[int, int]:
        return (self._rows, self._cols)

    @property
    def nnz(self) -> int:
        return self.to_scipy().nnz

    def to_scipy(self) -> sparse.csr_matrix:
        """Build and cache the CSR matrix."""
        if self._mat is None:
            self._mat = sparse.csr_matrix(
                (self._values, (self._row_idx, self._col_idx)),
                shape=(self._rows, self._cols),
            )
        return self._mat

    def to_dense(self) -> np.ndarray:
        return self.to_scipy().toarray()

    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            return self.to_scipy() @ other
        if isinstance(other, SparseMatrix):
            result = SparseMatrix(self._rows, other._cols)
            result._mat = self.to_scipy() @ other.to_scipy()
            return result
        raise TypeError(f"Cannot multiply with {type(other)}")

    def __add__(self, other: SparseMatrix) -> SparseMatrix:
        result = SparseMatrix(self._rows, self._cols)
        result._mat = self.to_scipy() + other.to_scipy()
        return result

    def transpose(self) -> SparseMatrix:
        result = SparseMatrix(self._cols, self._rows)
        result._mat = self.to_scipy().T.tocsr()
        return result

    @classmethod
    def from_scipy(cls, mat: sparse.spmatrix) -> SparseMatrix:
        sm = cls(mat.shape[0], mat.shape[1])
        sm._mat = sparse.csr_matrix(mat)
        return sm

    @classmethod
    def identity(cls, n: int) -> SparseMatrix:
        sm = cls(n, n)
        sm._mat = sparse.eye(n, format="csr")
        return sm


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def sparse_solve(A: SparseMatrix, b: np.ndarray) -> np.ndarray:
    """Direct sparse solve via LU factorisation."""
    return splinalg.spsolve(A.to_scipy().tocsc(), b)


def sparse_lu(A: SparseMatrix):
    """Return a sparse LU factorisation for repeated solves."""
    return splinalg.splu(A.to_scipy().tocsc())


def sparse_cg(
    A: SparseMatrix,
    b: np.ndarray,
    tol: float = 1e-10,
    maxiter: int = 1000,
) -> np.ndarray:
    """Conjugate gradient solver for SPD systems."""
    x, info = splinalg.cg(A.to_scipy(), b, rtol=tol, maxiter=maxiter)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info})")
    return x


# ---------------------------------------------------------------------------
# Banded utilities
# ---------------------------------------------------------------------------


def tridiagonal_matrix(n: int, lower: np.ndarray, diag: np.ndarray, upper: np.ndarray) -> SparseMatrix:
    """Build a tridiagonal sparse matrix."""
    sm = SparseMatrix(n, n)
    data = np.concatenate([lower, diag, upper])
    rows = np.concatenate([np.arange(1, n), np.arange(n), np.arange(n - 1)])
    cols = np.concatenate([np.arange(n - 1), np.arange(n), np.arange(1, n)])
    sm._mat = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return sm
