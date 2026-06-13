"""Regression for L2 Wave-2 audit — `TreeSolver._apply_barrier` silently
no-oped for DOWN_IN / UP_IN barriers.

Pre-fix:

    elif self.barrier_type == BarrierType.DOWN_IN:
        pass  # complex — for now, only knock-out supported
    elif self.barrier_type == BarrierType.UP_IN:
        pass

At the outer ``solve()`` level, knock-in barriers are now computed via
in-out parity (vanilla − knock-out) so ``_apply_barrier`` is never
called with a knock-in type via the normal path.  BUT a caller invoking
``_apply_barrier`` directly (e.g. via a subclass extension or bypassing
solve()) would silently get a vanilla price labelled as a knock-in.

Post-fix the knock-in branches raise ``NotImplementedError`` so the
misuse is loud.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.numerical._trees import BarrierType, TreeSolver


class TestApplyBarrierKnockInRaises:
    def test_down_in_raises_when_called_directly(self):
        solver = TreeSolver(
            barrier_type=BarrierType.DOWN_IN,
            barrier_level=80.0,
        )
        V = np.array([10.0, 20.0, 30.0])
        S = np.array([70.0, 100.0, 130.0])
        with pytest.raises(NotImplementedError, match="knock-in"):
            solver._apply_barrier(V, S)

    def test_up_in_raises_when_called_directly(self):
        solver = TreeSolver(
            barrier_type=BarrierType.UP_IN,
            barrier_level=120.0,
        )
        V = np.array([10.0, 20.0, 30.0])
        S = np.array([70.0, 100.0, 130.0])
        with pytest.raises(NotImplementedError, match="knock-in"):
            solver._apply_barrier(V, S)


class TestKnockOutBarriersStillWork:
    """The OUT branches must remain functional after the IN fix."""

    def test_down_out_zeroes_below_barrier(self):
        solver = TreeSolver(
            barrier_type=BarrierType.DOWN_OUT,
            barrier_level=80.0,
        )
        V = np.array([10.0, 20.0, 30.0])
        S = np.array([70.0, 100.0, 130.0])
        result = solver._apply_barrier(V.copy(), S)
        # S[0]=70 <= 80 → V[0] zeroed
        assert result[0] == 0.0
        assert result[1] == 20.0
        assert result[2] == 30.0

    def test_up_out_zeroes_above_barrier(self):
        solver = TreeSolver(
            barrier_type=BarrierType.UP_OUT,
            barrier_level=120.0,
        )
        V = np.array([10.0, 20.0, 30.0])
        S = np.array([70.0, 100.0, 130.0])
        result = solver._apply_barrier(V.copy(), S)
        # S[2]=130 >= 120 → V[2] zeroed
        assert result[0] == 10.0
        assert result[1] == 20.0
        assert result[2] == 0.0


class TestSolveWithKnockInStillWorks:
    """Calling solve() with a knock-in barrier should still work via
    in-out parity (the outer-level fix from before this slice)."""

    def test_down_in_call_solves_via_parity(self):
        solver = TreeSolver(
            barrier_type=BarrierType.DOWN_IN,
            barrier_level=80.0,
            n_steps=50,
        )
        r = solver.solve(spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0)
        # Must return a finite price (in-out parity at outer level).
        import math as _m
        assert _m.isfinite(r.price)
        assert r.price >= 0.0
