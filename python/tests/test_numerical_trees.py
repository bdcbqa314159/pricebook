"""Tests for numerical._trees — migrated from old API."""
import pytest
from pricebook.numerical._trees import solve_tree, solve_tree_2d, TreeResult

class TestSolveTree:
    def test_callable(self):
        assert callable(solve_tree)

class TestSolveTree2D:
    def test_runs(self):
        result = solve_tree_2d(S1=100, S2=100, strike=100, rate=0.05,
                                vol1=0.2, vol2=0.25, rho=0.5, n_steps=30, T=1.0)
        assert isinstance(result, TreeResult)
        assert result.price > 0
