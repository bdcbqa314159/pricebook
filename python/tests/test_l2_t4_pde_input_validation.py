"""Regression for L2 Wave-2 audit — `PDESolver1D` input validation.

Pre-fix four robustness gaps:

1. ``n_time=0`` raised ``ZeroDivisionError`` deep inside ``solve()`` at
   ``dt = T / self.n_time`` with no diagnostic context.

2. ``n_space=0`` (or 1) raised an opaque ``IndexError`` inside the grid
   construction with no diagnostic context.

3. ``T == 0`` produced a finite-but-wrong price (~2.05 for an ATM call
   where the correct intrinsic is 0).  The solver iterated ``n_time``
   times with ``dt = 0``, but the boundary projection and operator
   construction did not cleanly commute with zero time evolution.

4. ``T < 0`` produced a numerical runaway (~1e107) with no exception.

Post-fix:
- ``n_space < 2`` or ``n_time < 1`` raise ``ValueError`` at construction.
- ``T < 0`` raises ``ValueError`` at ``solve()`` entry.
- ``T == 0`` returns the intrinsic value directly without invoking the
  solver — the unique no-arbitrage payoff with no time to evolve.
"""

from __future__ import annotations

import pytest

from pricebook.numerical._pde import PDEMethod, PDESolver1D


class TestConstructorValidation:
    def test_n_time_zero_raises(self):
        with pytest.raises(ValueError, match="n_time must be >= 1"):
            PDESolver1D(n_space=50, n_time=0)

    def test_n_time_negative_raises(self):
        with pytest.raises(ValueError, match="n_time must be >= 1"):
            PDESolver1D(n_space=50, n_time=-1)

    def test_n_space_zero_raises(self):
        with pytest.raises(ValueError, match="n_space must be >= 2"):
            PDESolver1D(n_space=0, n_time=50)

    def test_n_space_one_raises(self):
        """A 1-point grid can't form a finite-difference stencil."""
        with pytest.raises(ValueError, match="n_space must be >= 2"):
            PDESolver1D(n_space=1, n_time=50)


class TestSolveValidation:
    def test_negative_T_raises(self):
        solver = PDESolver1D(n_space=50, n_time=50)
        with pytest.raises(ValueError, match="T must be >= 0"):
            solver.solve(spot=100, strike=100, T=-1.0, vol=0.2, rate=0.05)


class TestZeroExpiryIntrinsic:
    def test_T_zero_call_ITM(self):
        """T=0 ATM call: intrinsic = spot - strike (positive)."""
        solver = PDESolver1D(n_space=50, n_time=50)
        r = solver.solve(spot=120, strike=100, T=0.0, vol=0.2, rate=0.05,
                         is_call=True)
        assert r.price == pytest.approx(20.0)

    def test_T_zero_call_OTM(self):
        solver = PDESolver1D(n_space=50, n_time=50)
        r = solver.solve(spot=80, strike=100, T=0.0, vol=0.2, rate=0.05,
                         is_call=True)
        assert r.price == pytest.approx(0.0)

    def test_T_zero_put_ITM(self):
        solver = PDESolver1D(n_space=50, n_time=50)
        r = solver.solve(spot=80, strike=100, T=0.0, vol=0.2, rate=0.05,
                         is_call=False)
        assert r.price == pytest.approx(20.0)

    def test_T_zero_put_OTM(self):
        solver = PDESolver1D(n_space=50, n_time=50)
        r = solver.solve(spot=120, strike=100, T=0.0, vol=0.2, rate=0.05,
                         is_call=False)
        assert r.price == pytest.approx(0.0)

    def test_T_zero_returns_valid_result_object(self):
        """The PDEResult must be well-formed (values, grid, etc. present)."""
        solver = PDESolver1D(n_space=50, n_time=50)
        r = solver.solve(spot=100, strike=100, T=0.0, vol=0.2, rate=0.05,
                         is_call=True)
        assert r.price == pytest.approx(0.0)
        assert r.values is not None
        assert r.grid is not None
        assert r.method is not None


class TestHealthyPathUnchanged:
    def test_normal_solve_unchanged(self):
        """T > 0 with valid grid: behaviour identical to pre-fix."""
        solver = PDESolver1D(method=PDEMethod.CRANK_NICOLSON,
                             n_space=200, n_time=200)
        r = solver.solve(spot=100, strike=100, T=1.0, vol=0.2, rate=0.05,
                         is_call=True)
        # Black-Scholes ATM call ≈ 10.45.  PDE should be close.
        assert r.price == pytest.approx(10.45, abs=0.5)
