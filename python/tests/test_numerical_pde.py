"""Tests for numerical._pde — redirects to test_pde_solver.py for full coverage."""
import pytest
from pricebook.numerical._pde import PDESolver1D, PDEMethod, solve_bs_pde

class TestImports:
    def test_solver_callable(self):
        assert callable(solve_bs_pde)
    def test_class_exists(self):
        solver = PDESolver1D()
        assert solver.method == PDEMethod.CRANK_NICOLSON
