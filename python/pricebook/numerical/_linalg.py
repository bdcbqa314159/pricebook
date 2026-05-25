"""Linear algebra: matrix operations, sparse solvers, decompositions.

    from pricebook.numerical import expm, logm, qr, cholesky, gmres, sylvester
    from pricebook.numerical import DecompMethod, IterativeMethod

Wraps scipy.linalg and numpy.linalg behind a clean interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class DecompMethod(Enum):
    """Matrix decomposition methods."""
    QR = "qr"
    CHOLESKY = "cholesky"
    LU = "lu"
    SVD = "svd"


class IterativeMethod(Enum):
    """Iterative linear solver methods."""
    GMRES = "gmres"
    BICGSTAB = "bicgstab"


# ═══════════════════════════════════════════════════════════════
# Matrix decompositions
# ═══════════════════════════════════════════════════════════════

@dataclass
class QRResult:
    Q: np.ndarray
    R: np.ndarray

    def to_dict(self) -> dict:
        return {"Q_shape": list(self.Q.shape), "R_shape": list(self.R.shape)}


@dataclass
class SVDResult:
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray

    def to_dict(self) -> dict:
        return {"U_shape": list(self.U.shape), "rank": int(np.sum(self.S > 1e-10))}


@dataclass
class LUResult:
    P: np.ndarray
    L: np.ndarray
    U: np.ndarray

    def to_dict(self) -> dict:
        return {"shape": list(self.L.shape)}


def qr(A: np.ndarray) -> QRResult:
    """QR factorisation: A = QR where Q is orthogonal, R is upper triangular."""
    Q, R = np.linalg.qr(A)
    return QRResult(Q, R)


def cholesky(A: np.ndarray) -> np.ndarray:
    """Cholesky factorisation: A = LL' for symmetric positive definite A.

    Returns lower triangular L.
    """
    return np.linalg.cholesky(A)


def lu(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LU factorisation: PA = LU.

    Returns (P, L, U) where P is permutation, L lower, U upper.
    """
    from scipy.linalg import lu as _lu
    return _lu(A)


def lu_full(A: np.ndarray) -> LUResult:
    """LU factorisation returning an LUResult."""
    from scipy.linalg import lu as _lu
    P, L, U = _lu(A)
    return LUResult(P, L, U)


def svd(A: np.ndarray) -> SVDResult:
    """Singular Value Decomposition: A = U S V'."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return SVDResult(U, S, Vt)


def decompose(A: np.ndarray, method: DecompMethod | str = DecompMethod.QR):
    """Unified decomposition dispatcher.

    Returns the appropriate result type based on method.
    """
    if isinstance(method, str):
        method = DecompMethod(method.lower())

    if method == DecompMethod.QR:
        return qr(A)
    if method == DecompMethod.CHOLESKY:
        return cholesky(A)
    if method == DecompMethod.LU:
        return lu_full(A)
    if method == DecompMethod.SVD:
        return svd(A)
    raise ValueError(f"unknown method: {method!r}")


# ═══════════════════════════════════════════════════════════════
# Matrix functions
# ═══════════════════════════════════════════════════════════════

def expm(A: np.ndarray) -> np.ndarray:
    """Matrix exponential: exp(A).

    Uses Pade approximation via scipy.linalg.expm.
    Essential for transition matrices, generator matrices.
    """
    from scipy.linalg import expm as _expm
    return _expm(A)


def logm(A: np.ndarray) -> np.ndarray:
    """Matrix logarithm: log(A) such that exp(log(A)) = A.

    Requires A to have no negative real eigenvalues.
    """
    from scipy.linalg import logm as _logm
    return _logm(A)


def sqrtm(A: np.ndarray) -> np.ndarray:
    """Matrix square root: B such that B @ B = A."""
    from scipy.linalg import sqrtm as _sqrtm
    return _sqrtm(A)


# ═══════════════════════════════════════════════════════════════
# Linear system solvers
# ═══════════════════════════════════════════════════════════════

@dataclass
class IterativeSolveResult:
    x: np.ndarray
    iterations: int
    converged: bool
    residual_norm: float
    method: str = ""

    def to_dict(self) -> dict:
        return {"method": self.method, "iterations": self.iterations,
                "converged": self.converged, "residual_norm": self.residual_norm}


def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct solve Ax = b (dense)."""
    return np.linalg.solve(A, b)


def lstsq(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares solve: min ||Ax - b||_2."""
    return np.linalg.lstsq(A, b, rcond=None)[0]


def gmres(
    A,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int = 1000,
) -> IterativeSolveResult:
    """GMRES: Generalised Minimum Residual for non-symmetric systems.

    Works for any square system Ax = b. More general than CG
    (which requires A to be symmetric positive definite).
    """
    from scipy.sparse.linalg import gmres as _gmres
    from scipy.sparse import issparse

    if not issparse(A):
        A_op = np.atleast_2d(A)
    else:
        A_op = A

    x, info = _gmres(A_op, b, x0=x0, rtol=tol, maxiter=maxiter)
    residual = np.linalg.norm(A_op @ x - b) if not issparse(A_op) else np.linalg.norm(A_op.dot(x) - b)

    return IterativeSolveResult(
        x=x,
        iterations=info if info > 0 else 0,
        converged=(info == 0),
        residual_norm=float(residual),
        method="gmres",
    )


def bicgstab(
    A,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int = 1000,
) -> IterativeSolveResult:
    """BiCGSTAB: Bi-Conjugate Gradient Stabilised for non-symmetric systems.

    Often faster than GMRES for certain non-symmetric problems.
    """
    from scipy.sparse.linalg import bicgstab as _bicgstab
    from scipy.sparse import issparse

    if not issparse(A):
        A_op = np.atleast_2d(A)
    else:
        A_op = A

    x, info = _bicgstab(A_op, b, x0=x0, rtol=tol, maxiter=maxiter)
    residual = np.linalg.norm(A_op @ x - b) if not issparse(A_op) else np.linalg.norm(A_op.dot(x) - b)

    return IterativeSolveResult(
        x=x,
        iterations=info if info > 0 else 0,
        converged=(info == 0),
        residual_norm=float(residual),
        method="bicgstab",
    )


def iterative_solve(
    A,
    b: np.ndarray,
    method: IterativeMethod | str = IterativeMethod.GMRES,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int = 1000,
) -> IterativeSolveResult:
    """Unified iterative solver dispatcher."""
    if isinstance(method, str):
        method = IterativeMethod(method.lower())

    if method == IterativeMethod.GMRES:
        return gmres(A, b, x0, tol, maxiter)
    if method == IterativeMethod.BICGSTAB:
        return bicgstab(A, b, x0, tol, maxiter)
    raise ValueError(f"unknown method: {method!r}")


# ═══════════════════════════════════════════════════════════════
# Matrix equations
# ═══════════════════════════════════════════════════════════════

def sylvester(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Solve the Sylvester equation AX + XB = C.

    Arises in control theory, model reduction, Lyapunov equations.
    """
    from scipy.linalg import solve_sylvester
    return solve_sylvester(A, B, C)


def lyapunov(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Solve the continuous Lyapunov equation AX + XA' + Q = 0.

    Equivalently: AX + XA' = -Q.
    """
    return sylvester(A, A.T, -Q)


# ═══════════════════════════════════════════════════════════════
# Condition and properties
# ═══════════════════════════════════════════════════════════════

def cond(A: np.ndarray) -> float:
    """Condition number (2-norm): kappa = sigma_max / sigma_min."""
    return float(np.linalg.cond(A))


def rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """Numerical rank via SVD."""
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > tol))


def is_positive_definite(A: np.ndarray) -> bool:
    """Check if A is symmetric positive definite via Cholesky."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
