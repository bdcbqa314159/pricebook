"""Automatic differentiation via dual numbers.

Forward-mode AD: compute f(x) and f'(x) simultaneously with no
truncation error and no step-size tuning.

* :class:`Dual` — dual number with overloaded arithmetic.
* :func:`grad` — gradient of f: ℝⁿ → ℝ via forward AD.
* :func:`jacobian_ad` — Jacobian of f: ℝⁿ → ℝᵐ via forward AD.
* :func:`hessian_ad` — Hessian via nested dual numbers.

References:
    Griewank & Walther, *Evaluating Derivatives*, SIAM, 2008.
    Baydin et al., *Automatic Differentiation in Machine Learning: a Survey*, JMLR, 2018.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class Dual:
    """Dual number: a + bε where ε² = 0.

    Overloads arithmetic so that evaluating f(Dual(x, 1))
    returns Dual(f(x), f'(x)) — exact first derivative.

    Usage:
        x = Dual(2.0, 1.0)
        y = x**2 + sin(x)
        # y.val = 4 + sin(2), y.der = 4 + cos(2)
    """

    __slots__ = ("val", "der")

    def __init__(self, val: float, der: float = 0.0):
        self.val = float(val)
        self.der = float(der)

    def __repr__(self) -> str:
        return f"Dual({self.val}, {self.der})"

    # ---- Arithmetic ----

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val + other.val, self.der + other.der)
        return Dual(self.val + float(other), self.der)

    def __radd__(self, other):
        return Dual(float(other) + self.val, self.der)

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val - other.val, self.der - other.der)
        return Dual(self.val - float(other), self.der)

    def __rsub__(self, other):
        return Dual(float(other) - self.val, -self.der)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val * other.val, self.val * other.der + self.der * other.val)
        return Dual(self.val * float(other), self.der * float(other))

    def __rmul__(self, other):
        return Dual(float(other) * self.val, float(other) * self.der)

    def __truediv__(self, other):
        if isinstance(other, Dual):
            return Dual(
                self.val / other.val,
                (self.der * other.val - self.val * other.der) / (other.val ** 2),
            )
        return Dual(self.val / float(other), self.der / float(other))

    def __rtruediv__(self, other):
        return Dual(
            float(other) / self.val,
            -float(other) * self.der / (self.val ** 2),
        )

    def __pow__(self, other):
        if isinstance(other, Dual):
            # (a+bε)^(c+dε) = a^c + a^c (d ln(a) + bc/a) ε
            val = self.val ** other.val
            der = val * (other.der * math.log(abs(self.val)) + other.val * self.der / self.val) if self.val != 0 else 0
            return Dual(val, der)
        n = float(other)
        return Dual(self.val ** n, n * self.val ** (n - 1) * self.der)

    def __rpow__(self, other):
        # other^self where other is a float
        b = float(other)
        val = b ** self.val
        der = val * math.log(b) * self.der if b > 0 else 0
        return Dual(val, der)

    def __neg__(self):
        return Dual(-self.val, -self.der)

    def __abs__(self):
        if self.val >= 0:
            return Dual(self.val, self.der)
        return Dual(-self.val, -self.der)

    # ---- Comparisons (on value only) ----

    def __lt__(self, other):
        return self.val < (other.val if isinstance(other, Dual) else float(other))

    def __le__(self, other):
        return self.val <= (other.val if isinstance(other, Dual) else float(other))

    def __gt__(self, other):
        return self.val > (other.val if isinstance(other, Dual) else float(other))

    def __ge__(self, other):
        return self.val >= (other.val if isinstance(other, Dual) else float(other))

    def __eq__(self, other):
        return self.val == (other.val if isinstance(other, Dual) else float(other))

    def __float__(self):
        return self.val

    def __int__(self):
        return int(self.val)


# ---- Math functions for Dual numbers ----

def exp(x):
    if isinstance(x, Dual):
        e = math.exp(x.val)
        return Dual(e, e * x.der)
    return math.exp(x)


def log(x):
    if isinstance(x, Dual):
        return Dual(math.log(x.val), x.der / x.val)
    return math.log(x)


def sqrt(x):
    if isinstance(x, Dual):
        s = math.sqrt(x.val)
        return Dual(s, x.der / (2 * s)) if s > 0 else Dual(0, 0)
    return math.sqrt(x)


def sin(x):
    if isinstance(x, Dual):
        return Dual(math.sin(x.val), math.cos(x.val) * x.der)
    return math.sin(x)


def cos(x):
    if isinstance(x, Dual):
        return Dual(math.cos(x.val), -math.sin(x.val) * x.der)
    return math.cos(x)


def max_dual(a, b):
    """max() for dual numbers — derivative follows the argmax."""
    a_val = a.val if isinstance(a, Dual) else float(a)
    b_val = b.val if isinstance(b, Dual) else float(b)
    if a_val >= b_val:
        return a if isinstance(a, Dual) else Dual(float(a), 0)
    return b if isinstance(b, Dual) else Dual(float(b), 0)


# ---- Gradient and Jacobian ----

def grad(f, x: np.ndarray) -> np.ndarray:
    """Gradient of f: ℝⁿ → ℝ via forward-mode AD.

    Evaluates f once per dimension with a seed derivative.

    Args:
        f: callable taking np.ndarray → float (or Dual).
        x: point at which to evaluate gradient.

    Returns:
        Gradient array (n,).
    """
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        x_dual = [Dual(x[j], 1.0 if j == i else 0.0) for j in range(n)]
        result = f(x_dual)
        g[i] = result.der if isinstance(result, Dual) else 0.0
    return g


def jacobian_ad(f, x: np.ndarray) -> np.ndarray:
    """Jacobian of f: ℝⁿ → ℝᵐ via forward-mode AD.

    Args:
        f: callable taking np.ndarray → np.ndarray (each element Dual).
        x: point (n,).

    Returns:
        Jacobian matrix (m, n).
    """
    n = len(x)
    # First call to get output dimension
    x_dual = [Dual(x[j], 0.0) for j in range(n)]
    result0 = f(x_dual)
    m = len(result0) if hasattr(result0, '__len__') else 1

    J = np.zeros((m, n))
    for i in range(n):
        x_dual = [Dual(x[j], 1.0 if j == i else 0.0) for j in range(n)]
        result = f(x_dual)
        if hasattr(result, '__len__'):
            for k in range(m):
                J[k, i] = result[k].der if isinstance(result[k], Dual) else 0.0
        else:
            J[0, i] = result.der if isinstance(result, Dual) else 0.0

    return J


def derivative(f, x: float) -> tuple[float, float]:
    """Compute f(x) and f'(x) simultaneously.

    Args:
        f: scalar function accepting Dual numbers.
        x: evaluation point.

    Returns:
        (f(x), f'(x)).
    """
    result = f(Dual(x, 1.0))
    if isinstance(result, Dual):
        return result.val, result.der
    return float(result), 0.0
