"""Tests for numerical._trees."""
import pytest
from pricebook.numerical._trees import tree_greeks, binomial_2d

class TestTreeGreeks:
    def test_callable(self):
        assert callable(tree_greeks)

class TestBinomial2D:
    def test_runs(self):
        result = binomial_2d(S1=100, S2=100, strike=100, rate=0.05,
                              vol1=0.2, vol2=0.25, rho=0.5, n_steps=30, T=1.0)
        assert result.price > 0
