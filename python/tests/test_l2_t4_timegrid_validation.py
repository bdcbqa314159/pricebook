"""Regression for L2 Wave-2 audit — `TimeGrid` constructor accepted any
array, even empty or non-monotonic, silently producing nonsense state.

Pre-fix:

    def __init__(self, times):
        self.times = np.asarray(times, dtype=np.float64)
        self.dt = np.diff(self.times)
        self.n_steps = len(self.dt)
        self.T = float(self.times[-1])       # IndexError on empty

- Empty input → `self.times[-1]` raises ``IndexError`` deep inside the
  constructor with no diagnostic message.
- Length-1 input → `self.dt = []`, `n_steps = 0`, T set to the one time
  point.  The engine would then loop zero times — silent no-op.
- Non-monotonic input → `dt` has negative entries; the engine integrates
  the SDE BACKWARDS in time for those steps, with wrong drift sign.
  Pre-fix this was silent; the user got a finite "price" that was
  computed against time-reversed dynamics.

Post-fix all three failure modes raise ``ValueError`` upfront with a
clear diagnostic.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricebook.models.mc_engine import TimeGrid


class TestEmptyOrSingletonRaises:
    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="empty"):
            TimeGrid(np.array([]))

    def test_singleton_raises(self):
        with pytest.raises(ValueError, match="at least 2 time points"):
            TimeGrid(np.array([0.5]))


class TestNonMonotonicRaises:
    def test_decreasing_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            TimeGrid(np.array([0.0, 1.0, 0.5, 2.0]))

    def test_duplicate_raises(self):
        """Duplicates have dt = 0 which is degenerate (no SDE step)."""
        with pytest.raises(ValueError, match="strictly increasing"):
            TimeGrid(np.array([0.0, 0.5, 0.5, 1.0]))


class TestValidGridWorks:
    def test_uniform_works(self):
        g = TimeGrid.uniform(T=1.0, n_steps=10)
        assert g.n_steps == 10
        assert g.T == pytest.approx(1.0)

    def test_explicit_increasing_works(self):
        g = TimeGrid(np.array([0.0, 0.25, 0.5, 1.0]))
        assert g.n_steps == 3
        assert g.T == pytest.approx(1.0)
