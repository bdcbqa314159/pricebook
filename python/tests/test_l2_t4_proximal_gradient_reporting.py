"""Regression for L2 Wave-2 audit — `proximal_gradient` lied about its
result in two ways.

Pre-fix:
- ``OptimizeResult.fun`` was unconditionally ``0.0``.  A calibration
  consumer reading ``result.fun`` saw "zero" and concluded the solver
  found the optimum, when actually the solver had no idea what the
  objective value was.
- ``OptimizeResult.converged`` was unconditionally ``True``, even when
  the loop exhausted ``maxiter`` without the tolerance check ever firing.
  Downstream code asserting ``result.converged`` got a guaranteed-true
  rubber stamp.

Post-fix:
- `fun` is the true objective at the final iterate if the caller passed
  the optional ``f_obj`` callable; else ``nan`` (clear sentinel that
  the value is unknown, not "zero").
- `converged` reflects whether the within-iteration tolerance check
  actually fired.

The pre-fix interface had no way to discover either correct value
without exhausting iterations or recomputing the objective externally —
both anti-patterns for a calibration loop.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.numerical._optimize import proximal_gradient


def _zero_prox(x, t):
    """Identity proximal operator: prox_{t·0}(x) = x.  Used when g ≡ 0."""
    return x


def _smooth_quad_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of f(x) = 0.5·||x − x*||² with x* = [1, -2]."""
    return x - np.array([1.0, -2.0])


def _smooth_quad_value(x: np.ndarray) -> float:
    """f(x) = 0.5·||x − x*||² with x* = [1, -2]."""
    target = np.array([1.0, -2.0])
    return 0.5 * float(np.sum((x - target) ** 2))


class TestConvergedReflectsTruth:
    def test_converged_true_when_tolerance_fires(self):
        """Plenty of iterations and a small step → converge below tol."""
        r = proximal_gradient(
            grad_f=_smooth_quad_grad,
            prox_g=_zero_prox,
            x0=np.array([5.0, 5.0]),
            step_size=0.5,
            maxiter=1000,
            tol=1e-9,
        )
        assert r.converged is True

    def test_converged_false_when_maxiter_exhausted(self):
        """Tiny step + extremely tight tol + low maxiter → can't converge."""
        r = proximal_gradient(
            grad_f=_smooth_quad_grad,
            prox_g=_zero_prox,
            x0=np.array([5.0, 5.0]),
            step_size=1e-6,
            maxiter=5,
            tol=1e-12,
        )
        assert r.converged is False


class TestFunValue:
    def test_fun_is_nan_when_no_objective_supplied(self):
        """No way to know objective without f_obj — say so clearly."""
        r = proximal_gradient(
            grad_f=_smooth_quad_grad,
            prox_g=_zero_prox,
            x0=np.array([5.0, 5.0]),
            step_size=0.5,
            maxiter=200,
        )
        assert math.isnan(r.fun)

    def test_fun_is_correct_when_objective_supplied(self):
        """With f_obj, `fun` is the true objective at the converged x."""
        r = proximal_gradient(
            grad_f=_smooth_quad_grad,
            prox_g=_zero_prox,
            x0=np.array([5.0, 5.0]),
            step_size=0.5,
            maxiter=2000,
            tol=1e-12,
            f_obj=_smooth_quad_value,
        )
        # Should converge to x* = [1, -2] → f(x*) = 0.
        assert r.fun == pytest.approx(0.0, abs=1e-6)
        # And reaching the optimum should report converged=True.
        assert r.converged is True


class TestSolverFindsOptimum:
    def test_recovers_smooth_quad_optimum(self):
        r = proximal_gradient(
            grad_f=_smooth_quad_grad,
            prox_g=_zero_prox,
            x0=np.array([5.0, 5.0]),
            step_size=0.5,
            maxiter=2000,
            tol=1e-12,
        )
        np.testing.assert_allclose(r.x, [1.0, -2.0], atol=1e-6)
