"""Regression for L2 Tier-1 T1.4 — `interior_point` honours equality constraints.

Pre-fix, `interior_point` constructed `constraints = [{"type": "eq", ...}]` but
NEVER passed it to `_minimize`, and used BFGS (unconstrained) for the inner
solve.  So any caller passing `equality_constraints` got them SILENTLY DROPPED:
the optimizer just minimised the (barrier-modified) objective without any
equality enforcement, converging to the unconstrained minimum.

Post-fix: when equality constraints are present, switch to SLSQP and pass the
constraint dicts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.numerical._optimize import interior_point


class TestEqualityHonoured:
    def test_simple_equality_constrained_quadratic(self):
        """min ‖x‖² subject to sum(x) = 3.  Optimum is x = (1, 1, 1)."""
        def obj(x):
            return float(np.dot(x, x))

        def eq(x):
            return float(x.sum() - 3.0)

        x0 = np.array([0.0, 0.0, 0.0])
        result = interior_point(obj, x0, equality_constraints=[eq])

        # The unconstrained min of ‖x‖² is x=0 with f=0.  If equalities were
        # dropped (pre-fix), x → 0 and sum(x) = 0 ≠ 3.
        assert abs(result.x.sum() - 3.0) < 1e-4, (
            f"sum(x) = {result.x.sum():.4f}, expected 3 — equality dropped!"
        )
        # And the solution should be ≈ (1, 1, 1).
        for xi in result.x:
            assert abs(xi - 1.0) < 1e-3, (
                f"x = {result.x}, expected [1,1,1]"
            )

    def test_two_equalities(self):
        """min ‖x‖² in R³ subject to x[0] + x[1] = 1 and x[1] + x[2] = 1.
        Optimum: x = (1/2, 1/2, 1/2) by symmetry of the projection."""
        def obj(x):
            return float(np.dot(x, x))

        def eq1(x):
            return float(x[0] + x[1] - 1.0)

        def eq2(x):
            return float(x[1] + x[2] - 1.0)

        x0 = np.array([0.0, 0.0, 0.0])
        result = interior_point(obj, x0, equality_constraints=[eq1, eq2])

        # Both equalities must hold.
        assert abs(eq1(result.x)) < 1e-4, f"eq1 violated: {eq1(result.x):.4e}"
        assert abs(eq2(result.x)) < 1e-4, f"eq2 violated: {eq2(result.x):.4e}"

    def test_equality_plus_inequality(self):
        """min x² + y² subject to x + y = 2 (equality) AND x >= 0.5 (inequality).
        Optimum: x = y = 1.0 (equality forces x+y=2; midpoint minimises ‖·‖
        and satisfies x >= 0.5)."""
        def obj(z):
            return float(z[0]**2 + z[1]**2)

        def eq(z):
            return float(z[0] + z[1] - 2.0)

        def ineq(z):
            # g(z) <= 0 form: 0.5 - x <= 0
            return float(0.5 - z[0])

        x0 = np.array([0.6, 1.4])  # interior of x >= 0.5
        result = interior_point(
            obj, x0,
            inequality_constraints=[ineq],
            equality_constraints=[eq],
        )
        assert abs(eq(result.x)) < 1e-3, (
            f"Equality x+y=2 violated: x+y={result.x.sum():.4f}"
        )
        assert result.x[0] >= 0.5 - 1e-3, f"Inequality x>=0.5 violated: {result.x[0]}"
        # Optimum at (1, 1) with value 2.
        assert abs(result.fun - 2.0) < 1e-2, (
            f"f(x*) = {result.fun:.4f}, expected 2"
        )


class TestUnchangedBehaviour:
    def test_pure_inequality_path_unchanged(self):
        """No equalities → BFGS path (unchanged from pre-fix).  Sanity: this
        case still works."""
        def obj(x):
            return float((x[0] - 1)**2 + (x[1] - 2)**2)

        def ineq(x):
            # x[0] + x[1] <= 5 (i.e., 5 - x[0] - x[1] >= 0 ⇒ g(x) = x[0]+x[1]-5 <= 0)
            return float(x[0] + x[1] - 5.0)

        x0 = np.array([0.0, 0.0])
        result = interior_point(obj, x0, inequality_constraints=[ineq])

        # Optimum is x=(1, 2), interior to the inequality (1+2=3 < 5).
        assert abs(result.x[0] - 1.0) < 1e-3
        assert abs(result.x[1] - 2.0) < 1e-3
