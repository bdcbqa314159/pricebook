"""Tests for numerical._optimize: minimize, linprog, qp."""
import pytest, numpy as np
from pricebook.numerical._optimize import minimize, linprog

class TestMinimize:
    def test_rosenbrock(self):
        f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
        result = minimize(f, [0, 0], method="nelder_mead")
        assert abs(result.x[0] - 1.0) < 0.1
        assert abs(result.x[1] - 1.0) < 0.1

class TestLinprog:
    def test_simple(self):
        # min -x-y s.t. x+y <= 4, x <= 3, y <= 3
        result = linprog(c=[-1, -1], A_ub=[[1, 1], [1, 0], [0, 1]], b_ub=[4, 3, 3])
        assert abs(result.fun + 4.0) < 0.1  # optimal = -4
