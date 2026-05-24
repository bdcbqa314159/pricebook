"""Numerical differentiation — unified framework.

Multiple methods for computing derivatives, gradients, Jacobians, and
Hessians with controllable accuracy.

    from pricebook.numerical._differentiate import (
        derivative, gradient, jacobian, hessian,
        DiffMethod, DiffResult,
    )

    # Simple derivative
    df = derivative(f, x=1.0)

    # Complex-step (machine precision)
    df = derivative(f, x=1.0, method=DiffMethod.COMPLEX_STEP)

    # Richardson-extrapolated
    df = derivative(f, x=1.0, method=DiffMethod.RICHARDSON)

References:
    Squire & Trapp (1998). Using Complex Variables to Estimate Derivatives.
    Richardson (1911). The Approximate Arithmetical Solution by Finite
    Differences of Physical Problems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class DiffMethod(Enum):
    """Differentiation method."""
    FORWARD = "forward"              # f'≈(f(x+h)-f(x))/h, O(h)
    CENTRAL = "central"              # f'≈(f(x+h)-f(x-h))/(2h), O(h²)
    COMPLEX_STEP = "complex_step"    # f'≈Im(f(x+ih))/h, O(h²) with NO truncation error
    RICHARDSON = "richardson"        # Richardson extrapolation on central, O(h⁴+)
    FIVE_POINT = "five_point"        # 5-point stencil, O(h⁴)


@dataclass
class DiffResult:
    """Result of numerical differentiation."""
    value: float | np.ndarray
    error_estimate: float
    method: str
    n_evaluations: int

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_evaluations": self.n_evaluations,
            "error_estimate": self.error_estimate,
        }


# ═══════════════════════════════════════════════════════════════
# Scalar derivatives
# ═══════════════════════════════════════════════════════════════


def derivative(
    f: callable,
    x: float,
    method: DiffMethod = DiffMethod.CENTRAL,
    h: float | None = None,
    order: int = 1,
) -> DiffResult:
    """Compute the derivative of f at x.

    Args:
        f: scalar function f(x) → float.
        x: evaluation point.
        method: differentiation method.
        h: step size (auto-selected if None).
        order: derivative order (1 = first, 2 = second).
    """
    if h is None:
        h = _auto_step(x, method)

    if order == 1:
        return _first_derivative(f, x, h, method)
    elif order == 2:
        return _second_derivative(f, x, h, method)
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")


def _first_derivative(f, x, h, method):
    if method == DiffMethod.FORWARD:
        val = (f(x + h) - f(x)) / h
        return DiffResult(val, abs(h), "forward", 2)

    elif method == DiffMethod.CENTRAL:
        val = (f(x + h) - f(x - h)) / (2 * h)
        return DiffResult(val, h**2, "central", 2)

    elif method == DiffMethod.COMPLEX_STEP:
        # f'(x) = Im(f(x + ih)) / h — NO truncation error, only rounding
        val = f(complex(x, h)).imag / h
        return DiffResult(val, 1e-15, "complex_step", 1)

    elif method == DiffMethod.RICHARDSON:
        # Central differences at h and h/2, extrapolate
        d1 = (f(x + h) - f(x - h)) / (2 * h)
        d2 = (f(x + h/2) - f(x - h/2)) / h
        val = (4 * d2 - d1) / 3  # O(h⁴) error
        err = abs(d2 - d1) / 3
        return DiffResult(val, err, "richardson", 4)

    elif method == DiffMethod.FIVE_POINT:
        val = (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)
        return DiffResult(val, h**4, "five_point", 4)

    raise ValueError(f"Unknown method: {method}")


def _second_derivative(f, x, h, method):
    if method == DiffMethod.CENTRAL or method == DiffMethod.FORWARD:
        val = (f(x + h) - 2 * f(x) + f(x - h)) / h**2
        return DiffResult(val, h**2, f"central_2nd", 3)

    elif method == DiffMethod.FIVE_POINT:
        val = (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h) - f(x - 2*h)) / (12 * h**2)
        return DiffResult(val, h**4, "five_point_2nd", 5)

    elif method == DiffMethod.RICHARDSON:
        d1 = (f(x + h) - 2*f(x) + f(x - h)) / h**2
        h2 = h / 2
        d2 = (f(x + h2) - 2*f(x) + f(x - h2)) / h2**2
        val = (4 * d2 - d1) / 3
        return DiffResult(val, abs(d2 - d1) / 3, "richardson_2nd", 6)

    elif method == DiffMethod.COMPLEX_STEP:
        # Second derivative via complex step: not directly available
        # Fall back to central
        return _second_derivative(f, x, h, DiffMethod.CENTRAL)

    raise ValueError(f"Unknown method for 2nd derivative: {method}")


# ═══════════════════════════════════════════════════════════════
# Vector derivatives: gradient, Jacobian, Hessian
# ═══════════════════════════════════════════════════════════════


def gradient(
    f: callable,
    x: np.ndarray,
    method: DiffMethod = DiffMethod.CENTRAL,
    h: float | None = None,
) -> DiffResult:
    """Gradient ∇f of a scalar function f: ℝⁿ → ℝ.

    Args:
        f: scalar function f(x) → float.
        x: (n,) evaluation point.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if h is None:
        h = _auto_step(np.max(np.abs(x)), method)

    grad = np.zeros(n)
    n_evals = 0

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0

        if method == DiffMethod.CENTRAL:
            grad[i] = (f(x + h * ei) - f(x - h * ei)) / (2 * h)
            n_evals += 2
        elif method == DiffMethod.FORWARD:
            grad[i] = (f(x + h * ei) - f(x)) / h
            n_evals += 1
        elif method == DiffMethod.COMPLEX_STEP:
            x_complex = x.astype(complex)
            x_complex[i] += 1j * h
            grad[i] = f(x_complex).imag / h
            n_evals += 1
        else:
            grad[i] = (f(x + h * ei) - f(x - h * ei)) / (2 * h)
            n_evals += 2

    return DiffResult(grad, h**2 if method != DiffMethod.COMPLEX_STEP else 1e-15,
                       method.value, n_evals)


def jacobian(
    f: callable,
    x: np.ndarray,
    method: DiffMethod = DiffMethod.CENTRAL,
    h: float | None = None,
) -> DiffResult:
    """Jacobian J[i,j] = ∂fᵢ/∂xⱼ of a vector function f: ℝⁿ → ℝᵐ.

    Args:
        f: vector function f(x) → (m,) array.
        x: (n,) evaluation point.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if h is None:
        h = _auto_step(np.max(np.abs(x)), method)

    f0 = np.atleast_1d(f(x))
    m = len(f0)
    J = np.zeros((m, n))
    n_evals = 1

    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0

        if method == DiffMethod.CENTRAL:
            fp = np.atleast_1d(f(x + h * ej))
            fm = np.atleast_1d(f(x - h * ej))
            J[:, j] = (fp - fm) / (2 * h)
            n_evals += 2
        else:
            fp = np.atleast_1d(f(x + h * ej))
            J[:, j] = (fp - f0) / h
            n_evals += 1

    return DiffResult(J, h**2, method.value, n_evals)


def hessian(
    f: callable,
    x: np.ndarray,
    method: DiffMethod = DiffMethod.CENTRAL,
    h: float | None = None,
) -> DiffResult:
    """Hessian H[i,j] = ∂²f/∂xᵢ∂xⱼ of a scalar function f: ℝⁿ → ℝ.

    Args:
        f: scalar function f(x) → float.
        x: (n,) evaluation point.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if h is None:
        h = _auto_step(np.max(np.abs(x)), method)

    H = np.zeros((n, n))
    f0 = f(x)
    n_evals = 1

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0

        # Diagonal: ∂²f/∂xᵢ²
        H[i, i] = (f(x + h * ei) - 2 * f0 + f(x - h * ei)) / h**2
        n_evals += 2

        # Off-diagonal: ∂²f/∂xᵢ∂xⱼ
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = 1.0
            fpp = f(x + h * ei + h * ej)
            fpm = f(x + h * ei - h * ej)
            fmp = f(x - h * ei + h * ej)
            fmm = f(x - h * ei - h * ej)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * h**2)
            H[j, i] = H[i, j]
            n_evals += 4

    return DiffResult(H, h**2, method.value, n_evals)


# ═══════════════════════════════════════════════════════════════
# Step size selection
# ═══════════════════════════════════════════════════════════════


def _auto_step(x_scale: float, method: DiffMethod) -> float:
    """Automatically select step size based on method and scale."""
    eps = np.finfo(float).eps  # ~2.2e-16

    if method == DiffMethod.COMPLEX_STEP:
        return 1e-20  # can be tiny — no truncation error

    elif method == DiffMethod.FORWARD:
        return eps**0.5 * max(abs(x_scale), 1.0)  # O(√ε) optimal for O(h)

    elif method == DiffMethod.CENTRAL:
        return eps**(1/3) * max(abs(x_scale), 1.0)  # O(ε^{1/3}) optimal for O(h²)

    elif method == DiffMethod.FIVE_POINT:
        return eps**0.2 * max(abs(x_scale), 1.0)  # O(ε^{1/5}) for O(h⁴)

    elif method == DiffMethod.RICHARDSON:
        return eps**(1/3) * max(abs(x_scale), 1.0)

    return 1e-8  # safe default
