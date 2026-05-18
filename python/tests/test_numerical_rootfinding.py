"""Tests for numerical._rootfinding."""
import pytest, math
from pricebook.numerical._rootfinding import bisection, find_root

class TestBisection:
    def test_sqrt2(self):
        result = bisection(lambda x: x**2 - 2, 1, 2)
        assert abs(result.root - math.sqrt(2)) < 1e-8

class TestFindRoot:
    def test_cubic(self):
        result = find_root(lambda x: x**3 - 1, bracket=(0.5, 2.0))
        assert abs(result.root - 1.0) < 1e-8
