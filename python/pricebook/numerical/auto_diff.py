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
        # Fix T4-AD3: pre-fix this routine produced silent-wrong results
        # at degenerate inputs.  For a base of 0:
        #   - With a Dual exponent: returned `der=0` regardless of what
        #     the exponent was (the singularity was hidden by an
        #     `if self.val != 0 else 0` shortcut).
        #   - With a non-Dual exponent `n`: returned ``n * 0^(n-1) * der``.
        #     For n < 1 this is ``n * 1/0^(1-n) * der`` — actually a
        #     singularity disguised as 0.  For n = 1 the correct
        #     derivative is `der` (well-defined), but the formula returns
        #     `1 * 0^0 * der` which Python evaluates as `der`, so n=1
        #     happens to work by accident.
        # With a NEGATIVE base and a Dual exponent: `math.log(abs(self.val))`
        # was a sleight-of-hand — log of a negative number isn't real, but
        # the abs() made the formula run.  Result was a meaningless float.
        #
        # Post-fix: raise loudly for these singularities.
        if isinstance(other, Dual):
            if self.val == 0:
                raise ValueError(
                    "Dual base 0 with Dual exponent: derivative is singular "
                    "(log(0) is -infinity).  Either ensure self.val != 0 or "
                    "use a separate formula for the zero-base branch."
                )
            if self.val < 0:
                raise ValueError(
                    f"Dual base {self.val} (negative) with Dual exponent: "
                    "log of a negative number is not real; the derivative "
                    "has no meaningful real-valued forward-mode result."
                )
            # (a+bε)^(c+dε) = a^c + a^c (d ln(a) + bc/a) ε
            val = self.val ** other.val
            der = val * (other.der * math.log(self.val)
                         + other.val * self.der / self.val)
            return Dual(val, der)
        n = float(other)
        if self.val == 0 and n < 1:
            raise ValueError(
                f"Dual base 0 with exponent {n} < 1: derivative is singular."
            )
        return Dual(self.val ** n, n * self.val ** (n - 1) * self.der)

    def __rpow__(self, other):
        # other^self where other is a float
        # Fix T4-AD3: pre-fix this returned ``der = 0`` silently when
        # `b <= 0` — log(b) is undefined there, and ``der = b^x · ln(b)``
        # has no real value.  Raise loudly instead.
        b = float(other)
        if b <= 0:
            raise ValueError(
                f"Non-positive base {b} with Dual exponent: derivative is "
                "b^x · ln(b), but ln(b) is undefined for b <= 0."
            )
        val = b ** self.val
        der = val * math.log(b) * self.der
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

    def __hash__(self):
        # Fix T4-AD2: pre-fix Python automatically set __hash__ = None
        # because __eq__ was defined.  That made Dual unhashable, breaking
        # any natural use as a dict key or set member.  But __eq__
        # compares ONLY `val` (intentional, for float compatibility:
        # `Dual(1, 2) == 1.0` is True), so to satisfy the hash/eq contract
        # (a == b ⇒ hash(a) == hash(b)) we must hash on `val` alone.
        # This makes Dual(1, 2) and Dual(1, 99) collide in a hash table,
        # which is consistent with them being == .
        return hash(self.val)

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
    """Square root with forward-mode AD propagation.

    Fix T4-AD3: pre-fix this returned ``Dual(0, 0)`` when ``x.val == 0``,
    silently dropping the genuinely infinite derivative ``1/(2·√x) → ∞``.
    Downstream code computing Greeks of MC paths that touched 0 (e.g.
    Heston variance hitting the zero-boundary in QE-discretisation) would
    silently lose sensitivity to vol-vol there.  Now raises so the
    upstream code can either avoid the zero point or handle the
    singularity explicitly.
    """
    if isinstance(x, Dual):
        if x.val == 0:
            raise ValueError(
                "sqrt(Dual(0, ·)): derivative is singular at x = 0 "
                "(1/(2·√x) → ∞).  Avoid the zero point or use a closed-form "
                "branch that handles it."
            )
        s = math.sqrt(x.val)
        return Dual(s, x.der / (2 * s))
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

_THREADING_ERR = (
    "Forward-mode AD driver requires f to thread Dual numbers through. "
    "f returned {got!r}; expected a Dual (or for jacobian_ad, an iterable of "
    "Duals).  The most common cause is f calling NumPy functions on a "
    "list of Duals (NumPy returns a generic ndarray of objects whose "
    ".der attribute does not propagate), or f using `math.` functions "
    "that strip the Dual wrapper.  Use the Dual-aware operations defined "
    "in pricebook.numerical.auto_diff instead, or wrap external functions "
    "with `lift_to_dual` (which raises if any branch silently drops Duals)."
)


def grad(f, x: np.ndarray) -> np.ndarray:
    """Gradient of f: ℝⁿ → ℝ via forward-mode AD.

    Evaluates f once per dimension with a seed derivative.

    Args:
        f: callable taking np.ndarray → float (or Dual).
        x: point at which to evaluate gradient.

    Returns:
        Gradient array (n,).

    Fix T4-AD1: pre-fix this routine silently returned 0 when ``f`` did not
    thread Duals through (the most common forward-AD bug — easy to introduce
    by using NumPy on a list of Duals).  A user computing "delta" of a
    pricer with a typo or stale code path would see ``grad = [0, 0, …]``
    with no warning and conclude the option had no Greek.  Now we raise
    ``TypeError`` with a diagnostic message so the failure mode is loud.
    """
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        x_dual = [Dual(x[j], 1.0 if j == i else 0.0) for j in range(n)]
        result = f(x_dual)
        if not isinstance(result, Dual):
            raise TypeError(_THREADING_ERR.format(got=type(result).__name__))
        g[i] = result.der
    return g


def jacobian_ad(f, x: np.ndarray) -> np.ndarray:
    """Jacobian of f: ℝⁿ → ℝᵐ via forward-mode AD.

    Args:
        f: callable taking np.ndarray → np.ndarray (each element Dual).
        x: point (n,).

    Returns:
        Jacobian matrix (m, n).

    Fix T4-AD1: pre-fix this routine silently dropped derivative entries
    to 0 when individual output components were not Duals.  Now raises
    ``TypeError`` with a diagnostic when any component fails to be a Dual.
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
                if not isinstance(result[k], Dual):
                    raise TypeError(_THREADING_ERR.format(got=type(result[k]).__name__))
                J[k, i] = result[k].der
        else:
            if not isinstance(result, Dual):
                raise TypeError(_THREADING_ERR.format(got=type(result).__name__))
            J[0, i] = result.der

    return J


def derivative(f, x: float) -> tuple[float, float]:
    """Compute f(x) and f'(x) simultaneously.

    Args:
        f: scalar function accepting Dual numbers.
        x: evaluation point.

    Returns:
        (f(x), f'(x)).

    Fix T4-AD1: pre-fix this returned ``(float(result), 0.0)`` when f did
    not return a Dual, silently producing a zero derivative.  Now raises
    ``TypeError`` so the threading failure is loud.
    """
    result = f(Dual(x, 1.0))
    if not isinstance(result, Dual):
        raise TypeError(_THREADING_ERR.format(got=type(result).__name__))
    return result.val, result.der
