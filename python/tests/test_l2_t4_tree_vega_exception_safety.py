"""Regression for L2 Wave-2 audit — `TreeSolver._compute_vega` left
``self._computing_vega = True`` and ``self.store_tree = False`` if the
inner ``self.solve(...)`` raised.

Pre-fix code:

    self._computing_vega = True
    saved_store = self.store_tree
    self.store_tree = False
    r_up = self.solve(...)                # if this raises, both stay set
    self.store_tree = saved_store
    self._computing_vega = False

If the bumped-vol `solve()` call raised, the solver was left in a
"computing vega" state with `store_tree = False`.  A subsequent unrelated
`solve()` call would then silently produce results without snapshots and
without recursing into vega computation — a wrong-but-finite price.

Post-fix wraps the bump sequence in `try ... finally:` so both attributes
are restored regardless of whether the inner solve fails.  Same shape as
the MCEngine.greek exception-safety fix from v0.946.
"""

from __future__ import annotations

import pytest

from pricebook.numerical._trees import (
    ExerciseType,
    TreeMethod,
    TreeSolver,
)


PARAMS = dict(spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0)


def _raising_payoff(_S):
    raise RuntimeError("synthetic payoff failure")


class TestVegaExceptionSafety:
    def test_state_restored_when_solve_raises(self):
        """Trigger vega-bump failure → check that _computing_vega and
        store_tree are restored."""
        solver = TreeSolver(method=TreeMethod.CRR, n_steps=20,
                            exercise=ExerciseType.EUROPEAN,
                            store_tree=True)
        # Solve once with a payoff that succeeds.  This sets up the base
        # call and then triggers vega via the payoff.  We can't easily
        # make vega's bumped-vol solve raise via the payoff, so we go
        # direct:
        before_computing = solver._computing_vega
        before_store = solver.store_tree
        with pytest.raises(RuntimeError):
            solver._compute_vega(
                spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0,
                payoff_fn=_raising_payoff, is_call=True, div_yield=0.0,
                base_price=10.0,
            )
        # After the exception, both attributes are back to their pre-call
        # values.
        assert solver._computing_vega == before_computing
        assert solver.store_tree == before_store


class TestVegaHappyPath:
    def test_vega_finite_and_state_restored(self):
        """Sanity check: happy-path vega still works AND restores state."""
        solver = TreeSolver(method=TreeMethod.CRR, n_steps=200,
                            exercise=ExerciseType.EUROPEAN)
        # Get base price.
        r = solver.solve(**PARAMS)
        assert r.vega is not None
        # After the solve+vega chain, _computing_vega is False.
        assert solver._computing_vega is False


class TestSubsequentSolveUnaffected:
    def test_solve_after_failed_vega_unchanged(self):
        """Most important: solving after a FAILED vega must give the
        same result as solving before the failed vega."""
        solver = TreeSolver(method=TreeMethod.CRR, n_steps=50,
                            exercise=ExerciseType.EUROPEAN,
                            store_tree=True)
        before = solver.solve(**PARAMS).price

        with pytest.raises(RuntimeError):
            solver._compute_vega(
                spot=100.0, strike=100.0, rate=0.05, vol=0.20, T=1.0,
                payoff_fn=_raising_payoff, is_call=True, div_yield=0.0,
                base_price=before,
            )

        after = solver.solve(**PARAMS).price
        assert after == pytest.approx(before, rel=1e-12), \
            f"solver state corrupted by failed vega: before={before}, after={after}"
