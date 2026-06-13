"""Regression for L2 Wave-2 audit — forward-mode AD drivers silently
returned 0 when the user's function failed to thread Dual numbers through.

Pre-fix:
- `grad(f, x)`: ``g[i] = result.der if isinstance(result, Dual) else 0.0``
- `jacobian_ad(f, x)`: same pattern, component-wise.
- `derivative(f, x)`: returned ``(float(result), 0.0)`` if result wasn't
  a Dual.

This is the **most common bug** in forward-AD code: the user accidentally
strips the Dual wrapper somewhere in `f` (e.g. by calling a NumPy ufunc on
a list of Duals, or by routing through `math.` functions that don't
recognise the wrapper), and the driver silently produces a zero gradient.
A pricer that returns zero delta with no warning is a serious wrong-result
trap — the user may conclude their option has no Greek and act on it.

Post-fix all three drivers raise ``TypeError`` with a diagnostic message
that names the actual returned type and points at the most likely cause.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.numerical.auto_diff import (
    Dual,
    derivative,
    grad,
    jacobian_ad,
)


def _broken_scalar(_xs):
    """f returns a plain float — fails to thread."""
    return 42.0


def _broken_vector(_xs):
    """f returns a plain list — fails to thread for the jacobian path."""
    return [1.0, 2.0]


def _broken_partial(xs):
    """f threads on one component but not the other."""
    return [xs[0] * 2.0, 42.0]  # second component is plain float


def _ok_scalar(xs):
    """f correctly threads — returns a Dual."""
    return xs[0] * xs[0] + xs[1] * xs[1]


def _ok_vector(xs):
    """Each component is a Dual."""
    return [xs[0] * xs[0], xs[1] * xs[1]]


class TestGradRaisesOnThreadingFailure:
    def test_grad_raises_when_f_returns_float(self):
        with pytest.raises(TypeError, match="thread Dual"):
            grad(_broken_scalar, np.array([1.0, 2.0]))

    def test_grad_returns_correct_gradient_when_f_threads(self):
        g = grad(_ok_scalar, np.array([1.0, 2.0]))
        # d/dx (x² + y²) = (2x, 2y) = (2, 4)
        np.testing.assert_allclose(g, [2.0, 4.0])


class TestJacobianAdRaisesOnThreadingFailure:
    def test_jacobian_raises_when_f_returns_floats(self):
        with pytest.raises(TypeError, match="thread Dual"):
            jacobian_ad(_broken_vector, np.array([1.0, 2.0]))

    def test_jacobian_raises_when_one_component_breaks(self):
        with pytest.raises(TypeError, match="thread Dual"):
            jacobian_ad(_broken_partial, np.array([1.0, 2.0]))

    def test_jacobian_correct_when_f_threads(self):
        J = jacobian_ad(_ok_vector, np.array([1.0, 2.0]))
        # ∂(x², y²)/∂(x, y) = diag(2x, 2y) = diag(2, 4)
        expected = np.array([[2.0, 0.0], [0.0, 4.0]])
        np.testing.assert_allclose(J, expected)


class TestDerivativeRaisesOnThreadingFailure:
    def test_derivative_raises_when_f_returns_float(self):
        with pytest.raises(TypeError, match="thread Dual"):
            derivative(lambda _x: 3.14, 1.0)

    def test_derivative_correct_when_f_threads(self):
        val, der = derivative(lambda x: x * x, 3.0)
        assert val == pytest.approx(9.0)
        assert der == pytest.approx(6.0)
