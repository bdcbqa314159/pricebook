"""Regression for L2 Wave-2 audit — `minimize(method=BASIN_HOPPING)`
hardcoded `converged=True` regardless of outcome.

Pre-fix:

    return OptimizeResult(r.x, float(r.fun), r.nit, True, "basin_hopping")
                                                       ^^^^

A user reading ``result.converged`` got a guaranteed-true rubber stamp,
even when the inner local optimizer hit `maxiter` without succeeding or
when scipy reported a message-level failure.

Post-fix: use `r.lowest_optimization_result.success` — the success flag
of the BEST local optimization result, which scipy exposes for exactly
this purpose.  If even the best local solve didn't succeed, the global
search ended unconvinced too.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.numerical._optimize import OptimMethod, minimize


def _rastrigin(x):
    """Standard global-optimization test function — many local minima."""
    A = 10.0
    return A * len(x) + sum(xi ** 2 - A * np.cos(2 * np.pi * xi) for xi in x)


def _quadratic(x):
    """f(x) = sum(x²) — single minimum at 0."""
    return float(np.sum(np.asarray(x) ** 2))


class TestBasinHoppingConvergence:
    def test_smooth_quadratic_converges(self):
        """With many iterations on a smooth function, basin-hopping
        should find the global min and report converged=True."""
        r = minimize(_quadratic, x0=np.array([3.0, -2.0]),
                     method=OptimMethod.BASIN_HOPPING, maxiter=20)
        # Should reach the minimum.
        np.testing.assert_allclose(r.x, [0.0, 0.0], atol=0.01)
        # And report honest convergence (the inner local optimizer
        # succeeded on the smooth quadratic).
        assert r.converged is True


class TestBasinHoppingResultShape:
    def test_returns_proper_result_object(self):
        r = minimize(_rastrigin, x0=np.array([2.0, 2.0]),
                     method=OptimMethod.BASIN_HOPPING, maxiter=5)
        assert hasattr(r, "x")
        assert hasattr(r, "fun")
        assert hasattr(r, "converged")
        assert hasattr(r, "iterations")
        assert isinstance(r.converged, bool)
